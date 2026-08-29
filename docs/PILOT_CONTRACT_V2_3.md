# P-PILOT-CONTRACT-V2.3

The canonical, owner-frozen authority for the pre-GPU pilots. **Self-sufficient for execution**:
nothing an executor needs is delegated to an untracked file.

    CONTRACT_VERSION      P-PILOT-CONTRACT-V2.3
    STATUS                frozen authority; pilots NOT_AUTHORIZED
    MACHINE_READABLE      pretrain/pilot_contract_v2_3.py
    EXECUTOR              pretrain/pilot_runner_v2_3.py

## Supersession

V2.3 supersedes the **optimizer decision** of P-PILOT-CONTRACT-V2.2 and nothing else. V2.2 froze
AdamW and required an optimizer-family comparison; the owner has since frozen the family
directly. Every non-optimizer V2.2 freeze is retained and restated below, so this document alone
governs execution.

    OWNER_OPTIMIZER_VERDICT               FREEZE_MUON_DIRECTLY
    optimizer-family comparison required  no

Not reopened: model architecture, tokenizer, Stage-I data, Stage-M data, Stage-A→Stage-B order,
the effective-batch decision, the production warmup decision, the continuous WSD family.

`CLAUDE.md`, `DECISIONS.md`, `PLAYBOOK.md` and `RETRAIN_PLAN.md` are gitignored local files and
are **not load-bearing** for pilot execution. No executable decision reads their bytes; a test
walks the AST of both modules to prove it.

## Authorization

No pilot is authorized here. The owner publishes a separate manifest
(`petitgpt-pilot-authorization-v2.3`) binding the exact reviewed HEAD after independent review.
**No tracked code change is required for that transition** — `validate_authorization()` already
consumes that schema, and `validate_execution_artifacts()` is the single gate every training
entry point passes through.

Publishing order for an authorized run:

    1. python pretrain/pilot_runner_v2_3.py write-index-manifest --out <dir>/PILOT_INDICES.json
    2. record that file's SHA-256 as pilot_index_manifest_file_sha256 in the manifest
    3. python pretrain/pilot_runner_v2_3.py run --phase MB \
         --authorization <manifest> --pilot-index-manifest <dir>/PILOT_INDICES.json \
         --output-root <new authorized root>
    4. (FULL_V2_3_PILOT only) the same command with --phase LR — no geometry arguments exist

## 1. Geometry (retained from V2.2)

    EFFECTIVE_BATCH_TOKENS            262144
    SEQUENCES_PER_OPTIMIZER_UPDATE    128
    SEQ_LEN                           2048
    FROZEN_GRAD_ACCUM                 128 / FROZEN_MICRO_BSZ   (must divide 128 exactly)

## 2. Optimizer — Muon

    --optimizer muon        explicit on every command; the repository default is also muon,
                            but the flag is required so the choice is never implicit
    --muon_lr 0.0           the Muon matrix groups reuse the scheduled main --lr
    --muon_momentum 0.95
    --lr                    the single searched axis
    --weight_decay 0.1      --grad_clip 1.0

`build_optimizer` resolves `muon_lr = lr` when `--muon_lr 0.0`, so `ratio = 1.0` and **every**
realized group carries `lr_ratio == 1.0`. There is no separate Muon-LR search dimension.

Realized grouping, read from `src/optim.py` rather than asserted:

| Group | Membership | Update | weight_decay | State |
|---|---|---|---|---|
| `muon_matrices` | 2D params not matched by `ADAM_PARAM_NAME_KEYS` | Muon | 0.1 | `momentum_buffer` |
| `aux_adamw_decay` | names containing `tok_emb`, `lm_head`, `.gate.` | AdamW | 0.1 | `step`, `exp_avg`, `exp_avg_sq` |
| `aux_adamw_no_decay` | `ndim < 2` (norm gains, biases) | AdamW | 0.0 | `step`, `exp_avg`, `exp_avg_sq` |

Auxiliary AdamW uses betas `(0.9, 0.95)` and eps `1e-8`. RMS matching is Moonlight's
`adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))` with 5 Newton–Schulz steps; weight decay is
decoupled in both halves. `Muon.__init__` rejects a non-2D parameter in a `use_muon` group. Both
halves live in one optimizer instance, so the checkpoint schema is unchanged.

`src/optim.py` is the grouping and mechanics authority and was **not** rewritten to match prose.

## 3. Phase MB — microbatch and compile

Ten required probes in this fixed order, each with `--optimizer muon --muon_lr 0.0
--muon_momentum 0.95`, main peak LR 3e-4, bf16, fresh model/optimizer/subprocess/output
directory/Inductor cache, 40 updates, `grad_accum = 128 / micro_bsz`, and the fixed Stage-A train
indices at seed 20260829:

    16/off  16/on  8/off  8/on  4/off  4/on  2/off  2/on  1/off  1/on

    lr(u) = 3e-4 * min(u/10, 1.0);  updates 1-10 warm-in, 11-40 measured

Timing synchronizes CUDA immediately before and after each timed update and measures the
complete end-to-end update — dataloading, accumulation, forward, backward, clipping, step.
Compile wall time is recorded separately. CUDA peak-memory statistics are reset at process start.

Eligibility (all): 40 updates completed; no OOM or uncontrolled exception; every token-mean loss
finite; every logged global grad norm finite; all expected optimizer states instantiated;
realized grouping matches this contract; every group `lr_ratio == 1.0`;
`max_memory_reserved <= 90%` of physical VRAM; `compile=on` used the intended path with no
silent fallback.

Metric: median end-to-end tokens/sec over updates 11–40. Selection: fastest eligible; within 3%
relative is tied; then lowest peak reserved VRAM; still tied within 256 MiB → `compile=off`;
still tied → larger `micro_bsz`. No eligible candidate → `PHASE_MB_ABORT`.

Outputs `FROZEN_MICRO_BSZ`, `FROZEN_GRAD_ACCUM`, `FROZEN_COMPILE`.

## 4. Pilot indices and release binding (retained)

NumPy **exactly 2.2.6**, `Generator(PCG64(20260829))`, one generator, three draws in order:
Stage-A eval 4096; Stage-A train 131072 from the remaining Stage-A indices; Stage-B eval 4096.
Stage-A/Stage-B eval serialize ascending; Stage-A train in draw order.

    seed-1   model init 20260829   train order 20260829
    seed-2   model init 20260830   train order 20260830

**Universes are derived from the accepted releases at runtime**, never from caller-supplied
values: the executor opens the canonical manifest-required Stage-A/B releases and verifies path,
metadata SHA-256, block count and the shard identities the canonical loader checks. The pilot
uses the canonical production loss mask emitted by `PackedBinDataset`, scored through the
trainer's own `masked_weighted_ce_loss` at `eos_weight=1.0`.

## 5. Phase Muon-LR

At `FROZEN_MICRO_BSZ` / `FROZEN_GRAD_ACCUM` / `FROZEN_COMPILE`, Muon, bf16, 262144 tokens per
update, fresh model/optimizer/scheduler. The only search axis is the main scheduled `--lr`.

    initial seed-1 grid   2e-4, 3e-4, 4e-4
    each run              200 updates, 25-update linear warmup, constant afterward
    lr(u)                 candidate_peak_lr * min(u/25, 1.0)
    per run               25,600 Stage-A pilot train blocks, 52,428,800 trained tokens

Recorded every update: token-mean train loss, global grad norm, realized group LRs. Also
recorded as diagnostics only, never as selection thresholds: per-group update and weight RMS for
the Muon and auxiliary AdamW halves.

Eligibility: 200 updates completed; all losses, grad norms and parameters finite; Muon momentum
states present; auxiliary AdamW `exp_avg`/`exp_avg_sq` present; grouping matches; every
`lr_ratio == 1.0`; both eval losses finite; no sustained divergence.

Sustained divergence: `BASELINE = median(updates 41..60)`; every complete 20-update window whose
**final update lies in 80..200** must satisfy `WINDOW_MEDIAN <= 1.5 * BASELINE`.

Evaluation after update 200: eval mode, all 4096 Stage-A and all 4096 Stage-B eval blocks,
ascending index order, canonical mask, token-mean CE.

    SCORE = (10 * loss_A + 3 * loss_B) / 13

## 6. Selection and the complete edge rule

Fewer than two eligible initial seed-1 candidates → `PHASE_MUON_LR_ABORT`. Otherwise the seed-1
winner is the lowest eligible SCORE; within 0.5% relative is tied; ties go to the lower LR.

Seed-2 confirmation reruns the winner and its adjacent **lower** initial-grid candidate — or the
adjacent **higher** one (3e-4) if the winner is the grid minimum 2e-4.
`FINAL_SCORE = mean(seed-1 SCORE, seed-2 SCORE)`; lowest wins; within 0.5% the lower LR wins. If
one seed-2 run is ineligible the other fully eligible candidate wins; if both are ineligible,
`PHASE_MUON_LR_ABORT`.

Bounded edge expansion, **at most once**: confirmed 2e-4 → edge 1e-4; confirmed 4e-4 → edge
6e-4; confirmed 3e-4 → none. An edge candidate runs **both** seeds under the same rules and is
comparison-eligible only if **both** edge runs are individually eligible; otherwise the incumbent
remains. If both are eligible, compare `EDGE_FINAL_SCORE` with the incumbent `FINAL_SCORE`,
lowest wins, within 0.5% the lower LR wins. No second expansion is permitted, and an edge win is
final.

Output: `FROZEN_PEAK_LR`.

## 7. Production schedule and discrete final-LR semantics

    FROZEN_WARMUP_STEPS               500      convention-frozen, NOT pilot-derived
    OWNER_DECAY_INTENT_FRACTION_OF_TOTAL   0.10
    PLANNER_DECAY_FRACTION_INPUT           0.10
    PRODUCTION_MIN_LR_INTENT_RATIO         0.10

Continuous WSD across Stage A then Stage B: no optimizer reset, no scheduler reset, Stage B is
an exact full-state continuation. Expected canonical planner geometry:

    schedule_total_steps  49590      decay updates 4959      decay interval [44631, 49590)

**Discrete final-LR semantics.** The scheduler is configured with endpoint floor intent
`0.10 * FROZEN_PEAK_LR` at the *mathematical* endpoint `schedule_total_steps`. The last update
actually applied is `schedule_total_steps - 1`, so its LR is the exact canonical
one-step-before-end value and is **strictly above** the floor. That is correct and expected:
the last applied update is **not** required to equal the endpoint numerically, and trainer
scheduler mathematics must not be altered to force such equality.

Stage N must later verify exact LR values at the warmup boundary, decay start, the last applied
optimizer update, and the mathematical endpoint. All stage boundaries and total-step integers
come only from the canonical planner output.

## 8. Token budget

    Phase MB ceiling          105,000,000    expected maximum 104,857,600
    Phase Muon-LR ceiling     370,000,000
    global V2.3 hard ceiling  500,000,000

Accounting is **reserve-then-complete**, one documented order per optimizer update. Under an
exclusive `flock`, the ledger reloads from disk, revalidates its complete identity binding,
checks the phase and global ceilings, and persists a **reservation** — all *before* the update
is applied. The reservation moves to `completed` only after the optimizer step returns. A
process that dies between the two leaves the reservation consumed on purpose: budget is never
handed back, so a crash can never produce an uncounted optimizer update. Reaching a ceiling
without a frozen result is `PILOT_ABORT`.

Orchestration verifies that all required Phase-MB candidates are represented with no duplicate
or unknown identity, that the initial LR grid is complete, and that every selected result comes
from an eligible completed run. A caller-supplied partial grid or derived-result JSON cannot
masquerade as complete evidence.

## 9. Runtime gates

    required GPU     exactly "NVIDIA GeForce RTX 4090", VRAM in the 24GB class (22000-26000 MiB),
                     CUDA available, bf16 supported
    NumPy            exactly 2.2.6 (see requirements-pilot-v2_3.txt)

A substring check is explicitly insufficient: an "RTX 4090 Laptop GPU" contains `4090` and is
refused. The RTX 4000 Ada has `TRAINING_AUTHORITY=NONE`; non-training tooling and tests run
normally on it.

## 10. Output and checkpoint isolation

The **authorized output root** is validated before anything is written under it: it may not be,
sit inside, or contain an accepted release, and its parent must already exist. Every candidate
then uses a new candidate-specific directory that must resolve beneath that validated root.

    PILOT_CHECKPOINTING    DISABLED

V2.3 writes and reads **no** pilot training checkpoint. There is no save path, no resume path
and no `--resume` option; `require_checkpointing_disabled()` refuses either action, and
`reject_pilot_checkpoint_as_initialization()` refuses a pilot checkpoint as the initialization
for another candidate, Stage N or Stage O. A candidate that fails is rerun from scratch under a
new authorized root, never resumed.

## 11. Fingerprint separation

The base runtime fingerprint binds GPU identity and VRAM, driver, CUDA runtime, torch
version/build, Python version and executable, NumPy and tokenizers versions, container template,
repository branch/HEAD and worktree status, the contract SHA and the execution bundle SHA. It
carries **no** per-run configuration: `compile`, `micro_bsz`, `grad_accum`, `peak_lr`, `seed` and
`phase` live in each run_meta instead, and the module asserts their absence.

## 12. Validation at execution (R2)

Nothing is trusted because a caller said so. Every real execution root — the parent orchestrator
and each candidate worker independently — calls the same
`validate_execution_artifacts(authorization_path, candidate_spec_path,
pilot_index_manifest_path, output_dir, requested_phase)` and re-derives its authority from
artifact **bytes on disk**:

    authorization manifest      loaded and hashed from disk
    authorized status + scope   from those bytes
    branch / HEAD / worktree    observed via git, including the allowed-probe rule
    contract SHA                recomputed
    execution-bundle SHA        rederived by AST closure walk over the roots
    pilot-index manifest        FILE SHA-256 computed from disk, never read out of the manifest
    pilot indices               regenerated and compared list by list
    accepted Stage A / Stage B  opened through the manifest-required canonical loader
    authorized output root      validated before any write
    runtime fingerprint         RTX 4090 gate + exact NumPy
    token-ledger identity       eight fields, revalidated on every lock-held operation
    requested phase and scope   PHASE_MB_ONLY may never execute Phase LR

There is no in-memory authorization flag, no caller-supplied hash or count, and no context
object that grants anything by existing. The parent hands the child only *paths* — the real
manifest, the real index manifest and the child's own immutable spec — and the child revalidates
all of them. The session identity the child derives must equal the one recorded in its spec.

### Sessions and scope

A session is `sha256(authorization SHA, scope, output root, contract SHA, bundle SHA)`, recorded
in `session.json` at the root. One authorization means one session and one ledger.
`PHASE_MB_ONLY` **terminates** after its Phase-MB report and is never promoted: Phase LR needs a
new `FULL_V2_3_PILOT` authorization, which implies a new session, a new ledger and a new root.

### Result classes

    SUCCESS               0
    CANDIDATE_INELIGIBLE  3    candidate-local; the required grid continues
    PHASE_ABORT           4    the phase cannot continue under the frozen contract
    BINDING_FAILURE       5    an identity binding is broken; never downgraded

Each candidate subprocess writes a structured terminal result
(`petitgpt-pilot-candidate-terminal-result-v2.3`: status, error class, error message, completed
updates, reserved/completed tokens, run_meta SHA) next to its spec, and its stdout and stderr
are preserved verbatim. A missing terminal result, or a status that disagrees with the exit
code, is a `PHASE_ABORT` — never a pass.

### The authoritative Phase-MB report

`PHASE_MB_REPORT.json` is published **once**, with a SHA-256 sidecar, and is the only source of
Phase-LR geometry — the `run` subcommand has no `--micro-bsz` and no `--compile` option. Before
Phase LR starts, the report is re-hashed against its sidecar, revalidated against every session
binding, checked for a complete ten-candidate grid, recomputed candidate by candidate, and its
recorded selection is re-derived and compared.

### Independent recomputation

No selection number is taken on trust:

    Phase MB    median throughput recomputed from the raw per-update wall timings, and the
                measured-update count checked against the frozen 30 (updates 11..40)
    Phase LR    both eval losses recomputed from their raw numerator and weight, then SCORE
                recomputed as (10*loss_A + 3*loss_B)/13

A serialized value that disagrees with its recomputation is a `BINDING_FAILURE`.

### Evidence completeness

The initial LR grid must carry a record for all three grid points; confirmation must carry a
record for both LRs at **both** seeds; a bounded edge expansion must carry a record at both
seeds before it is resolved either way. An ineligible record is still a record — a *missing*
record is an evidence gap and aborts the phase.

### Timing and compile evidence

    torch_compile_wrapper_seconds        the torch.compile() call itself
    first_optimizer_update_wall_seconds  update 1, which materializes lazy compilation
    measured_update_wall_seconds         updates 11..40, the throughput sample

No host/device scalar transfer happens inside a timed region: losses and gradient norms stay as
device tensors and are converted after the loop. `canonical_compile_path` is observed evidence —
realized module type, TorchDynamo graph counters and Inductor artifacts on disk — checked in
both directions, not a Boolean copied from the candidate spec.

### Muon RMS matching

`verify_rms_matching()` runs one real Muon step per deterministic shape case and reconstructs
`p_after = p_before * (1 - lr*wd) - adjusted_lr * NS5(momentum)` with
`adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))`. The shape set includes both non-square
orientations, and each case records the margin by which an unscaled implementation would differ,
so the check cannot pass vacuously. `pilot_runner_v2_3.py rms-matching` prints the evidence
without training.
