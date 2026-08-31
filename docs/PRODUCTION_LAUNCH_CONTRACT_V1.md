# P-PRODUCTION-LAUNCH-CONTRACT-V1

The accepted exact Stage-P run plan fixes **data identity and schedule geometry**. It does
not — and by design cannot — fix the learning rate, optimizer, precision, compile mode,
seeds, evaluation policy or checkpoint policy: those are launch-time bindings. This document
is the adjacent owner-frozen contract that completes the plan **without changing its bytes**.

Machine-readable twin: `pretrain/production_launch_contract_v1.py`.
Tests: `tests/test_production_launch_contract_v1.py`.

**This contract authorizes no training.** Execution additionally requires an external
`authorization_status="AUTHORIZED"` manifest scoped to exactly one stage.

    launch contract SHA-256          f52303d1fa9dbca7df415afad7677c3ee6489d4a65746d30f3a91e7754b2ea94
    trainer execution bundle SHA-256 9639b7cbe2bb95505d052dd151f8f9719020d9b37ca3be944031e436dacb9df7
    authorization status             NOT_AUTHORIZED

## Accepted immutable Stage-P inputs

    exact run plan     runs/p_pilot_acceptance_and_exact_run_plan_v1_2026-08-31/plan/EXACT_RUN_PLAN.json
    exact plan SHA-256 d673089447b4240ad7d5f7fd97dbf5d57567ad68bfffcc708a08f345fd25c117
    plan-generation HEAD          4306f1db60b2c283f504404627e74f921c601800
    Stage-P plan implementation bundle 44d0982c2c0853c035d95528a043ca1cc48d60bdc9f9beb3e47a9d3b148f8f9f
    pilot owner acceptance        runs/p_pilot_acceptance_and_exact_run_plan_v1_2026-08-31/evidence/PILOT_RESULT_OWNER_ACCEPTANCE.json
    acceptance SHA-256            ce5f0366f0f4f276b7ab802006930e3a01c605c023adab6317f0e17755079391

The exact plan is immutable and is **not** by itself a Stage-N/Stage-O training authorization.

## Owner-frozen model

    n_layers 30 | d_model 576 | n_heads 9 | n_kv_heads 3 | d_ff 1536
    seq_len 2048 | vocab_size 32000 | dropout 0.0 | tied embeddings
    GQA, RoPE, RMSNorm, SwiGLU, SDPA, bf16
    parameter_count 124,635,456

    tokenizer        runs/g_production_2026-08-21/release/tokenizer.json
    tokenizer SHA256 d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce

## Owner-frozen training

    optimizer muon | muon_lr 0.0 | muon_momentum 0.95
    peak_lr 0.0006 | min_lr_ratio 0.10
    micro_bsz 8 | grad_accum 16 | effective_batch_tokens 262144 | sequences_per_update 128
    compile true | precision bf16
    warmup_steps 500 | decay_fraction 0.10 | schedule continuous cosine WSD
    weight_decay 0.1 | grad_clip 1.0
    stage order A then B | optimizer reset at A/B false | scheduler reset at A/B false

The complete realized Muon configuration is bound from `src/optim.py`, not asserted from
prose: three groups (`muon_matrices`, `aux_adamw_decay`, `aux_adamw_no_decay`), exact
per-role weight decay (0.1 / 0.1 / 0.0), auxiliary betas (0.9, 0.95), epsilon 1e-8, Nesterov
momentum 0.95, 5 Newton-Schulz steps with coefficients (3.4445, -4.7750, 2.0315),
Moonlight RMS matching at 0.2, and an explicitly present `lr_ratio == 1.0` on every group.
`--muon_lr 0.0` resolves to reuse the scheduled main `--lr`, so there is no separate
Muon-LR dimension.

## Owner-frozen seeds

    model_init_seed       20260831
    stage_a_sampler_seed  20260832
    stage_b_sampler_seed  20260833
    validation_seed       20260834

Pilot seeds (20260829 / 20260830) are **not** production seeds and are rejected.
`model_init_seed` governs Python `random`, NumPy, Torch CPU and Torch CUDA initialization.
Stage A and Stage B never share a mutable sampler seed: the governed path reads only the
per-stage fields. Loader/worker RNG is derived deterministically and recorded
(train +17, validation +17, per-source validation +19).

## Owner-frozen evaluation policy

Explicit milestones only:

    500, 3815, 11445, 22889, 38146, 38147, 43868, 43870, 44631, 49590

    periodic eval_every   DISABLED (0)
    benchmark evaluation  DISABLED (0)
    validation            full canonical release, deterministic no-shuffle order
    validation_seed       20260834
    val_samples / val_samples_per_source  0 (0 means the complete release)
    loss                  canonical global token-mean, eos_weight 1.0
    random subset         forbidden

## Owner-frozen checkpoint policy

    save steps (from the exact plan)
        3815, 11445, 22889, 38146, 38147, 43868, 43870, 44631, 49590
    periodic save_every   DISABLED (0)
    extra checkpoints     FORBIDDEN
    CLI serialization     repeated --save_steps flags derived mechanically from the numeric list

The planner also emits a comma-joined `cli_save_steps` string. Both forms parse under the
trainer's own normalization, but `load_run_plan_binding` applies `int()` per appended
element, so the **governed** launch always uses repeated flags.

## Canonical CWD

    /workspace/petitgpt

The planner and the governed trainer both resolve artifact-relative paths against the process
CWD, so a governed launch must run from the repository root.

## Runtime binding

The contract is GPU-product-agnostic until stage authorization. A Stage-N or Stage-O
authorization must bind the actual selected runtime: GPU UUID, PCI bus/device identity, GPU
name, visible-device count (exactly 1), total VRAM, compute capability, driver, CUDA, PyTorch,
Python, NumPy, trainer HEAD, trainer execution bundle and canonical CWD.

**Stage-N → Stage-O continuity.** Stage N must run on the exact runtime intended for Stage O.
If any bound runtime field changes after Stage N, the Stage-N authorization and result become
insufficient for Stage O and **Stage N must be rerun**. There is no materiality exception, and
none is implemented.

## Authorization

`authorization_template()` is always `NOT_AUTHORIZED`; this repository cannot publish an
AUTHORIZED manifest. Scopes are `STAGE_N` and `STAGE_O`, one stage per authorization.
**Stage O is never authorized because Stage N was authorized** — a scope mismatch is a
refusal, in both directions. No tracked code change is needed for the AUTHORIZED transition:
`validate_authorization` consumes the same schema.

## Enforcement

`require_governed_launch` runs in `main()` immediately after `validate_training_args`
and **before** any model, optimizer, sampler or training dataset is constructed, so a
mismatched CLI value fails before the backend is reached. It validates authorization
status/scope, branch, trainer HEAD and execution bundle, exact-plan SHA, plan-generation HEAD
and bundle, pilot acceptance SHA, model contract, optimizer contract, training geometry,
schedule, all four seeds, evaluation policy, checkpoint policy, canonical CWD and runtime
identity.

Compile fails closed: a governed run aborts if `torch.compile` raises or returns the eager
module, and a run contract may never claim `compile=true` after an eager fallback.

## Run contract and resume

Before the first optimizer update the governed path publishes a normalized run-contract
artifact binding the launch-contract SHA, stage-authorization SHA, exact-plan SHA, acceptance
SHA, trainer HEAD/bundle, the complete frozen model/training values, the full seed tuple,
evaluation and checkpoint policy, runtime fingerprint including GPU UUID/PCI, and the stage
scope and stop boundary. Every checkpoint binds the same digest. Resume rejects any drift in
those fields, and no CLI flag may override a checkpoint-bound governed value.


---

# R1 real-path repair

The contract above is now wired into the **real** trainer, checkpoint and resume paths, not
only into standalone helpers. Three gates run in `main()`:

    Gate A   before any model, optimizer, sampler or dataset exists
    Gate B   after model and optimizer construction, before any forward or update
    Gate C   compile realization, then atomic run-contract publication, before the
             first optimizer update

## Launch-contract artifact authentication

A governed launch supplies `--launch_contract_json` as a **path**. The bytes are read from
disk, parsed as canonical JSON, hashed, and every load-bearing field is compared with the
code authority. A supplied document with an altered `peak_lr`, `authorizes_training`, seed,
cadence or model value fails before construction. A self-declared `launch_contract_sha256`
inside the artifact is never the authority.

## Owner clarifications realized

1. **`num_workers`** is authorization-bound, not freely mutable. It enters the runtime
   fingerprint, the governed run contract and the Stage-N→Stage-O runtime comparison.
2. **Resume** has exactly two modes, both authorization-bound: `FRESH` (no checkpoint path
   or step) and `RESUME_EXACT_CHECKPOINT` (exact path, SHA, expected step, stage and
   governed run-contract digest). No arbitrary CLI override is permitted.
3. **Diagnostic fields** must equal the exact allowed value recorded in the contract. A
   diagnostic classification is not permission to accept arbitrary values; all 13 carry an
   explicit allowed value.
4. **Legacy `--sampler_seed`** is mechanically normalized to the active stage seed and then
   validated, so it can never select a different permutation. This **supersedes** the
   earlier V1 rule that rejected a legacy value equal to a stage seed; that rule would have
   made a correctly normalized governed run unlaunchable.

## Compile: lazy realization, fail closed

`torch.compile` returns a wrapper eagerly and compiles on first call, so a distinct wrapper
proves nothing. Gate C invokes the compiled callable once on a tiny governed-geometry probe
(batch 1, real `seq_len` and vocab, `no_grad`, no optimizer state) and then requires
TorchDynamo to have produced a graph, or Inductor to have left artifacts. An immediate
exception, an identity/eager return, an unrealized wrapper, or a recompile-limit fallback
all abort. Structured evidence is persisted and its SHA-256 bound into the run contract and
every checkpoint.

Compile evidence is a **per-process observation**, so it is deliberately excluded from the
immutable resume identity: a legitimate resume recompiles and produces different counters.
What resume does require is that a governed checkpoint claiming `compile=true` carries
realized evidence.

**R3 correction — Gate C re-derives its verdict.** `require_compile_realized` no longer
trusts the `compile_realized` boolean. It re-derives the verdict from the sub-facts recorded
in the same document — the compiled callable was actually invoked, the realized module is an
`OptimizedModule`, and compilation materialized — refuses a document missing any of them,
and rejects a recorded eager fallback. A document asserting realization while its own
sub-facts deny it is a contradiction and aborts the run.

## Governed run contract, checkpoints and resume

The normalized governed run contract is published **atomically, exactly once**, before the
first optimizer update. The legacy `config.json` snapshot remains for ungoverned/debug
compatibility and is explicitly not the governed publication proof.

Every governed checkpoint binds the full document, its digest, and is marked
`kind = PETITGPT_GOVERNED_V1`, so it is unmistakably distinguishable from a legacy one. An
ungoverned checkpoint cannot resume a governed run.

Resume validates **metadata before any executable state is restored** — before
`model.load_state_dict`, `optimizer.load_state_dict`, scaler restore and RNG restore — so a
mismatched checkpoint can never partially mutate the process.

## Sampler persistence

`build_data_contract` records the **active stage** sampler seed, not the legacy shared
field, alongside both per-stage seeds and the active stage name. The governed run contract
carries the permutation identity, `range_start_position`, `range_stop_position` and cursor.
The A→B transition additionally requires a Stage-A source, the correct saved Stage-A seed, a
complete consumed range, and a cursor exactly at the plan boundary.

**R3 correction — what a permutation identity identifies.** `ResumablePermutationSampler`
derives each epoch's order from `seed` and the epoch index alone; `range_start_position` is
per-invocation bookkeeping for the planned remainder. The identity digest therefore covers
`(stage, sampler_seed, range_stop_position)` only. Keying it on the range start — and
requiring that start to be equal across a resume — would have declared every legitimate
restart a different permutation, because a resuming sampler necessarily begins its range at
the recovered cursor rather than at 0.

What replaces it is stricter, not weaker: same-stage resume requires **exact continuity** —
the resuming sampler's `range_start_position` and cursor must both equal the checkpoint's
committed cursor. Starting one batch early replays data; starting one batch late skips it;
both now fail. Seed, stage, permutation identity and `range_stop_position` must still match
exactly.

**R3 correction — a restart is a new invocation.** `stage_authorization_sha256` and `resume`
are INVOCATION-identity fields, and a crash restart necessarily differs in both: it is
authorized by a new file, and it resumes a checkpoint written under a `FRESH`-mode
authorization. Requiring them to match made a same-stage restart structurally impossible —
an interrupted multi-day Stage A could never have been continued. Same-stage resume
therefore compares `SAME_STAGE_INVOCATION_MATCH_FIELDS` (stage, scope, `out_dir`,
`samples_dir`, active sampler seed, and both absolute stage boundaries) plus the complete
BASE identity.

The compensating control is that the restart authorization must still name the run it
claims to continue: its `resume.governed_run_contract_sha256` must equal the checkpoint's
own recomputed contract digest. A restart cannot point at an unrelated run's checkpoint.

## Selected-device UUID/PCI

The first `nvidia-smi` row is **not** assumed to be the selected device. Torch's selected
logical index is resolved through `CUDA_VISIBLE_DEVICES` in either index or UUID form to its
physical NVML record, and an ambiguous, unresolvable or inconsistent mapping fails.

## Stage-N result and Stage-O chain

A completed Stage N publishes a machine-readable result binding its authorization, contract,
plan, acceptance, trainer identity, governed run-contract digest, final checkpoint identity,
runtime fingerprint, GPU UUID/PCI and `num_workers`.

**R3 correction — the A→B source checks must be populated to run.** The Stage-N result also
records the final sampler permutation identity, range and cursor, and
`derive_stage_o_resume_binding` carries them into the Stage-O binding as `source_*` fields.
Each source check inside `validate_stage_a_to_b_transition` is guarded by `is not None`, so
a binding that omitted them did not fail — it silently skipped the strongest checks the A→B
transition has. Publishing a Stage-N result without the final sampler state is now refused.

A Stage-O authorization must carry the accepted Stage-N chain and is validated by **loading
the accepted Stage-N result from disk** and comparing it with the currently observed
runtime. Because the accepted result is read from bytes rather than taken from the
authorization, changing both the Stage-O authorization and the runtime cannot make them
agree with each other and evade comparison. Any runtime difference requires a new Stage N.
