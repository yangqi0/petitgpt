# P-PRODUCTION-LAUNCH-CONTRACT-V1

The accepted exact Stage-P run plan fixes **data identity and schedule geometry**. It does
not — and by design cannot — fix the learning rate, optimizer, precision, compile mode,
seeds, evaluation policy or checkpoint policy: those are launch-time bindings. This document
is the adjacent owner-frozen contract that completes the plan **without changing its bytes**.

Machine-readable twin: `pretrain/production_launch_contract_v1.py`.
Tests: `tests/test_production_launch_contract_v1.py`.

**This contract authorizes no training.** Execution additionally requires an external
`authorization_status="AUTHORIZED"` manifest scoped to exactly one stage.

    launch contract SHA-256          ae3ff587cdc06e38bc0cd112c1c6ff3b1bb0af77912a08e21138d63fca851c9f
    trainer execution bundle SHA-256 2e57897456bdb3635e6819c1357d1a6eb37adb8cb4e383c7e1097599ecbac417
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
