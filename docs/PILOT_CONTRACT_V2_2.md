# P-PILOT-CONTRACT-V2.2

The canonical, owner-frozen authority for the pre-GPU pilots that freeze the four training
values the schema-v3 planner requires. This document and its machine-readable twin
`pretrain/pilot_contract_v2.py` are the **single** current interpretation.

    CONTRACT_VERSION      P-PILOT-CONTRACT-V2.2
    STATUS                frozen authority; pilots NOT_AUTHORIZED
    MACHINE_READABLE      pretrain/pilot_contract_v2.py

## Supersession

Supersedes:

    PLAYBOOK.md §11.1, §11.2, §11.3
    the incomplete earlier pilot protocol
    the Pro V2 proposal
    the Fable V2.1 text wherever V2.2 edits it

Does **not** supersede:

    the final model architecture        the tokenizer
    accepted Stage-I / Stage-M data     Stage-A then Stage-B order
    the continuous WSD timeline         production data identities
    the existing Stage-P provenance policy

## Authorization

No pilot is authorized by this document. The final implementation HEAD is bound **later**, by a
separate owner authorization manifest issued after independent review. `authorization_template()`
always serializes `authorization_status="NOT_AUTHORIZED"`, and `require_launch_authorization()`
raises unconditionally, so no tooling in this repository can serialize an authorized launch.

This document deliberately contains no reference to the commit that introduces it.

## 1. Hardware

    load-bearing training pilots     NVIDIA GeForce RTX 4090 24GB class
    current RTX 4000 Ada             training authority NONE
    Stage N                          must run on the exact Stage-O pod instance and base fingerprint
    base fingerprint changes after Stage N   rerun Stage N before Stage O
    runtime estimates                Stage-N measurements on the actual Stage-O pod are authoritative

## 2. Effective batch

    EFFECTIVE_BATCH_TOKENS            262144
    SEQUENCES_PER_OPTIMIZER_UPDATE    128
    SEQ_LEN                           2048
    FROZEN_GRAD_ACCUM                 128 / FROZEN_MICRO_BSZ

Because the sequences per update are frozen at 128, the effective batch is identical for every
Phase-MB candidate; only the microbatch/accumulation split varies.

## 3. Phase MB — microbatch and compile

Ten unconditional probes in this fixed run order. **D-037 and any earlier measurement may not
exclude a candidate.**

    micro_bsz=16 compile=off        micro_bsz=16 compile=on
    micro_bsz=8  compile=off        micro_bsz=8  compile=on
    micro_bsz=4  compile=off        micro_bsz=4  compile=on
    micro_bsz=2  compile=off        micro_bsz=2  compile=on
    micro_bsz=1  compile=off        micro_bsz=1  compile=on

Each candidate: fresh subprocess, fresh model init seed 20260829, fresh AdamW optimizer,
isolated output directory, isolated TorchInductor cache directory, bf16, 40 optimizer updates,
`grad_accum = 128 / micro_bsz`, the same Stage-A train indices in the same order, and no
checkpoint inherited from any other candidate.

    MB_PROBE_PEAK_LR          3e-4
    MB_PROBE_WARMUP_UPDATES   10
    lr(u)                     3e-4 * min(u / 10, 1.0)      for the update about to be applied, u >= 1
    updates 1-10              timing warm-in
    updates 11-40             measured

Timing: synchronize CUDA immediately before and after each timed update; time the complete
optimizer update including gradient accumulation and data loading; record compile wall time
separately; throughput is the **median** tokens/sec over updates 11-40. Reset CUDA peak-memory
statistics at the start of each fresh candidate process.

Eligibility — all must hold:

    completes 40 updates
    no OOM and no uncontrolled exception
    every per-update token-mean loss finite
    every per-update grad norm finite
    AdamW state created for all expected parameter groups
    torch.cuda.max_memory_reserved <= 90% of recorded physical VRAM
    compile=on performs the canonical compile path with no silent fallback

Selection ladder:

    1. fastest eligible median tokens/sec
    2. every candidate within 3% relative of the fastest is tied
    3. among tied: lowest peak reserved VRAM
    4. still tied within 256 MiB: compile=off
    5. still tied: larger micro_bsz

Outputs `FROZEN_MICRO_BSZ`, `FROZEN_GRAD_ACCUM`, `FROZEN_COMPILE`. If no candidate is eligible:
`PHASE_MB_ABORT`.

## 4. Pilot indices

    Stage A universe   0..4882813          Stage B universe   0..1464844
    generator          NumPy 2.2.6, Generator(PCG64(20260829))

One generator, three draws, in exactly this order:

    1. Stage-A eval    4096 without replacement
    2. Stage-A train   131072 without replacement from the REMAINING Stage-A indices
    3. Stage-B eval    4096 without replacement, continuing the same generator

Serialization: Stage-A eval sorted ascending; Stage-B eval sorted ascending; Stage-A train in
**draw order**. Train and eval must be range-valid and disjoint within Stage A.

Per-run train order is a permutation of the fixed Stage-A train set using PCG64(20260829) for
seed-1 and PCG64(20260830) for seed-2.

    seed-1   model init 20260829   train order 20260829
    seed-2   model init 20260830   train order 20260830

All candidates within one seed share identical initialization bytes, block set and block order.

## 5. Optimizer

    FROZEN_OPTIMIZER   adamw          betas (0.9, 0.95)      grad_clip 1.0     fused on CUDA

Every command must pass `--optimizer adamw` explicitly, because the repository default is muon.
Muon is out of scope for this final retrain.

The complete realized configuration is bound from `src/optim.py` and the trainer by
`realized_adamw_config()`: `weight_decay=0.1`, `eps=1e-8`, `lr_ratio=1.0`, `ndim<2` parameters in
a `weight_decay=0.0` group, `tok_emb`/`lm_head`/`.gate.` in the AdamW decay group, remaining 2D
matrices sharing that decay group under `name="adamw"`, tied weights deduplicated by
`named_parameters()`, and `fused=all(p.is_cuda)` with a documented non-fused fallback.

## 6. Phase LR

All Phase-LR runs use `FROZEN_MICRO_BSZ`, `FROZEN_GRAD_ACCUM`, `FROZEN_COMPILE`, AdamW, bf16, an
effective batch of 262144 tokens, and a fresh model, optimizer and scheduler.

    initial seed-1 grid    2e-4  3e-4  4e-4  6e-4
    each run               400 optimizer updates, 50-update linear warmup, constant afterward
    lr(u)                  candidate_peak_lr * min(u / 50, 1.0)

Record token-mean train loss and grad norm at every optimizer update.

Eligibility: completes 400 updates; every loss and grad norm finite; both eval losses finite.

Sustained-divergence guard:

    BASELINE        median token-mean train loss over updates 81-100
    for each complete rolling 20-update window beginning at or after update 101:
        ineligible if WINDOW_MEDIAN > 1.5 * BASELINE

Evaluation at update 400: eval mode, all 4096 Stage-A eval blocks and all 4096 Stage-B eval
blocks, ascending index order, canonical production loss mask, token-mean cross entropy.

    SCORE = (10 * loss_A + 3 * loss_B) / 13

Seed-1 selection: lowest eligible SCORE; candidates within 0.5% relative of the minimum are
tied; ties go to the lower LR.

Seed-2 confirmation: rerun the seed-1 winner and the adjacent **lower** grid candidate, or the
adjacent **higher** one if the winner is the grid minimum.

    FINAL_SCORE = mean(seed-1 SCORE, seed-2 SCORE)     ties within 0.5% go to the lower LR

If one confirmation candidate is ineligible the other eligible one wins; if both are ineligible,
`PHASE_LR_ABORT`. If fewer than two initial seed-1 grid runs are eligible, `PHASE_LR_ABORT`.

Bounded edge expansion, at most once: a final winner of 2e-4 is compared with 1e-4 under both
seeds; a final winner of 6e-4 is compared with 8e-4 under both seeds. No further expansion. If
the edge wins, it is the final result.

Output: `FROZEN_PEAK_LR`.

## 7. Production schedule

    FROZEN_WARMUP_STEPS   500      convention-frozen by V2.2, NOT pilot-derived

Continuous WSD across Stage A then Stage B: no optimizer reset, no scheduler reset, Stage B is
an exact full-state continuation.

The contract distinguishes two different quantities:

    OWNER_DECAY_INTENT_FRACTION_OF_TOTAL   0.10    the intent: final 10% of schedule_total_steps
    PLANNER_DECAY_FRACTION_INPUT           0.10    the literal --decay_fraction the planner takes

They coincide here because `--decay_fraction` is already interpreted against
`schedule_total_steps`; see `DECAY_SEMANTICS_RECOVERY.md` in the segment evidence for the
recovered formulas. Minimum LR is `0.10 * FROZEN_PEAK_LR`, passed as `--min_lr_ratio 0.10`.

**All stage boundaries and total-step integers come only from the canonical planner output. No
hand-computed stage boundary may become authority.**

After the exact plan is generated, launch is allowed only if:

    decay_end_step == stage_b_global_stop_step
    decay_end_step == schedule_total_steps
    decay_start_step matches the final-10%-of-total intent within at most one documented step
    the decay interval lies wholly inside Stage B
    final LR == 0.10 * FROZEN_PEAK_LR

## 8. Pilot budget

    Phase-MB trained-token ceiling      105,000,000   (the ten-probe grid projects 104,857,600)
    Phase-LR run ceiling                8 runs
    global pilot trained-token ceiling  1,000,000,000

Reaching a ceiling without a frozen result is `PILOT_ABORT`. **The ceiling may not be increased
inside this contract.**

## 9. Checkpoint isolation

Pilot candidates always start from fresh initialization. No Phase-MB or Phase-LR checkpoint may
initialize another candidate, Stage N, or Stage O. If temporary recovery checkpoints are
supported they may resume only the exact same candidate, and must bind candidate config, seed,
indices, contract SHA, implementation HEAD and runtime fingerprint; they can never cross a
candidate boundary.

## 10. Base fingerprint and per-run metadata

The base runtime fingerprint records GPU name and total VRAM, driver, CUDA runtime, torch
version and build, Python version and executable, repository branch/HEAD, tracked-worktree
status, the allowed historical untracked-file status, container/template identifier when
available, stable package identity, the contract SHA and the implementation bundle SHA.

**The base fingerprint carries no global compile value**; `compile` is a per-candidate property
recorded in each run_meta, so one fingerprint covers both a compile-on and a compile-off probe.

Each run_meta binds the base fingerprint SHA, phase, candidate identity, micro_bsz, grad_accum,
compile, the optimizer and its full realized config, the LR configuration, model seed, train-order
seed, the pilot-index hashes, the contract SHA and the implementation HEAD.

Pre-launch Git policy: all tracked files clean; exactly one historical untracked exception,
`.codex_r1_manual_context_probe.py`, which must carry its previously recorded bytes and remain
outside execution. Any additional uncontrolled untracked source or config file stops the launch.
