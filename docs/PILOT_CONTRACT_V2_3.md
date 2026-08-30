# P-PILOT-CONTRACT-V2.3

The canonical, owner-frozen authority for the pre-GPU pilots. **Self-sufficient for execution**:
nothing an executor needs is delegated to an untracked file.

    CONTRACT_VERSION      P-PILOT-CONTRACT-V2.3
    STATUS                frozen authority; pilots NOT_AUTHORIZED
    MACHINE_READABLE      pretrain/pilot_contract_v2_3.py
    EXECUTOR              pretrain/pilot_runner_v2_3.py

## Supersession

V2.3 supersedes two P-PILOT-CONTRACT-V2.2 decisions: the optimizer-family decision and the
permanent training-GPU product choice. The owner froze Muon directly and now defers the exact
NVIDIA CUDA device to a future pilot authorization. Every other V2.2 freeze is retained and
restated below, so this document alone governs execution.

    OWNER_OPTIMIZER_VERDICT               FREEZE_MUON_DIRECTLY
    optimizer-family comparison required  no
    TRAINING_GPU_MODEL                     DEFERRED_UNTIL_OWNER_PILOT_AUTHORIZATION

Not reopened: model architecture, tokenizer, Stage-I data, Stage-M data, Stage-A→Stage-B order,
the effective-batch decision, the production warmup decision, the continuous WSD family.

`CLAUDE.md`, `DECISIONS.md`, `PLAYBOOK.md` and `RETRAIN_PLAN.md` are gitignored local files and
are **not load-bearing** for pilot execution. No executable decision reads their bytes; a test
walks the AST of both modules to prove it.

## Authorization

No pilot is authorized here. The owner publishes a separate manifest
(`petitgpt-pilot-authorization-v2.3`) binding the exact reviewed HEAD and selected training
runtime after independent review. The tracked template remains `NOT_AUTHORIZED` and carries
`training_hardware: null`; it selects no GPU and grants no hardware authority. **No tracked code
change is required for a later owner authorization** — `validate_authorization()` already
consumes the schema. `execute_candidate_from_artifact_paths()` is the sole real candidate
executor; it independently reloads the complete canonical artifact path set and finishes all
validation before defining or invoking model, dataset, forward/backward, or optimizer-step
operations.

A future `AUTHORIZED` manifest must populate `training_hardware` with:

    expected_gpu_device_name                    exact torch.cuda device-name string
    expected_cuda_device_count                  exactly 1 (after visibility selection)
    cuda_required                               true
    bf16_required                               true
    expected_base_runtime_fingerprint_sha256    owner-reviewed observed fingerprint SHA-256

The expected name is authorization data, not a permanent allowlist or a generic string rule.
The fingerprint digest binds the observed name, total VRAM, CUDA capability, NVIDIA driver and
CUDA runtime, torch version/build, Python identity, and the remaining base-runtime fields.

Publishing order for an authorized run:

    0. on the selected runtime, review `pilot_runner_v2_3.py fingerprint` and bind its identity
       and SHA-256 in the owner authorization
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
realized group must carry its own explicit, numeric `lr_ratio == 1.0` field. A missing field is a
verification failure; it is never accepted via a default of `1.0`. There is no separate Muon-LR
search dimension.

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

**Exact realization verification is a precondition for a candidate's first training update.**
`verify_muon_realization()` runs before update 1 and refuses the candidate as a `PHASE_ABORT` if
anything below is not exactly true — a realization the pilot cannot describe is not one it may
measure:

    exactly one Muon group; at most one group per auxiliary role; no unclassifiable group
    weight_decay        muon 0.1     aux decay 0.1     aux no-decay exactly 0.0
    aux AdamW           betas (0.9, 0.95)   eps 1e-8
    Muon                momentum 0.95   nesterov True   ns_steps 5
    every group         lr_ratio field explicitly present and numeric; exactly 1.0;
                        no missing-field default is permitted
    membership          every trainable parameter in exactly one group; no duplicate, no
                        missing parameter, no parameter foreign to the model, and each role's
                        membership equal to the grouping rule's prediction, empty roles included

A group's role is derived from `use_muon` and parameter dimensionality, never from its stored
decay, so a mutated decay surfaces as a wrong value for its role instead of silently
reclassifying the group.

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

Each measured update contributes one **timing record** binding its own update number, its own
trained-token count and its own synchronized wall time:

    update           an integer in 11..40
    trained_tokens   exactly 262144, the frozen per-update geometry
    wall_seconds     positive, finite, synchronized

Exactly one record must exist for **every** update 11..40 — no missing, extra or duplicated
measured update.

Eligibility (all): 40 updates completed; no OOM or uncontrolled exception; every token-mean loss
finite; every logged global grad norm finite; all expected optimizer states instantiated;
realized grouping matches this contract; every group `lr_ratio == 1.0`;
`max_memory_reserved <= 90%` of physical VRAM; `compile=on` used the intended path with no
silent fallback.

Metric — `median(trained_tokens / wall_seconds)` over the thirty records, i.e. the **median of
the per-update rates**. It is explicitly **not** `trained_tokens / median(wall_seconds)`: the two
are different statistics and disagree whenever the sample is not near-constant. The median is
recomputed from the raw records at admission and a disagreeing stored value is a
`BINDING_FAILURE`.

Selection: fastest eligible; within 3% relative is tied; then lowest peak reserved VRAM; still
tied within 256 MiB → `compile=off`; still tied → larger `micro_bsz`. No eligible candidate →
`PHASE_MB_ABORT`.

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
`lr_ratio == 1.0`; both eval losses finite; no sustained divergence; the observed compile path
equals the requested one, in both directions, exactly as Phase MB requires.

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

Stage N must also validate the actual production runtime intended for Stage O. No exact GPU
product is frozen for Stage N/O here. If that production hardware/runtime differs incompatibly
from the runtime whose Phase-MB geometry was frozen, the hardware-dependent MB result may not be
silently reused; continuation requires a new owner decision. Designing or executing that future
migration workflow is outside this amendment.

## 8. Token budget and session semantics

    Phase MB ceiling                       105,000,000   expected maximum 104,857,600
    Phase Muon-LR ceiling                  370,000,000
    FULL_V2_3_PILOT_SESSION_HARD_CEILING   500,000,000
    trained tokens per optimizer update    262,144

**`FULL_V2_3_PILOT` is ONE owner-authorized session**, containing exactly:

    Phase MB  ->  authoritative verified Phase-MB report  ->  Phase Muon-LR

with **one** authorization SHA, **one** session identity and **one** token ledger. The
500,000,000 figure is the hard ceiling for that single authorized session — not a per-attempt
allowance and not a running cross-authorization total. The phase ceilings are unchanged and both
fit inside it.

There is **no automatic second FULL session** and **no automatic cross-authorization retry**.
If a future owner wants to issue another authorization after a failed or abandoned session, that
is a **new owner decision that must explicitly take the prior session's consumed tokens into
account**. This executor never issues one, and deliberately keeps no cross-authorization
aggregate accounting.

`PHASE_MB_ONLY` is a **separate diagnostic authorization scope**. It ends after its Phase-MB
report, cannot authorize Phase LR and can never be promoted to `FULL_V2_3_PILOT`; there is no
automatic MB_ONLY → FULL transition anywhere in the executor.

Accounting is **reserve-then-complete**, one documented order per optimizer update. Under an
exclusive `flock`, the ledger reloads from disk, revalidates its complete identity binding,
checks the phase and global ceilings, and persists a **reservation** — all *before* the update
is applied. The reservation moves to `completed` only after the optimizer step returns. A
process that dies between the two leaves the reservation consumed on purpose: budget is never
handed back, so a crash can never produce an uncounted optimizer update. Reaching a ceiling
without a frozen result is `PILOT_ABORT`.

**Every** read of the ledger — including the `snapshot()` parents and reports consume — takes
the same exclusive lock, reloads the bytes from disk and revalidates identity plus the complete
structural invariant set, so a parent never reports stale in-memory counters after a child
advanced the file:

    reserved >= 0, completed >= 0, completed <= reserved, per bucket and globally
    GLOBAL reserved  == MB reserved  + LR reserved
    GLOBAL completed == MB completed + LR completed
    every token figure is a whole number of 262,144-token updates
    reserved_updates * 262144 == reserved_tokens[GLOBAL];  likewise for completed
    completed_updates <= reserved_updates
    stored effective ceilings equal the ceilings frozen for this authorization and session
    stored session hard ceiling equals FULL_V2_3_PILOT_SESSION_HARD_CEILING
    no stored value exceeds its ceiling

`reserved > completed` is legal and expected — that is exactly what a crash between the two
steps leaves behind. Any invariant failure is a phase-level **ledger-integrity failure**, never
an ineligible candidate.

The R4 ledger also binds one active candidate at a time and appends one immutable receipt when
that candidate terminates. Each receipt binds the complete candidate identity (including
`peak_lr`), its before/after local and aggregate accounting, its run-meta/result hashes, and the
previous receipt hash. The chain head and every receipt hash are revalidated on every locked
reload. A terminal is authoritative only when a fresh locked lookup finds its exact durable
receipt; a later candidate may advance the ledger without invalidating that historical receipt.

Orchestration verifies that all required Phase-MB candidates are represented with no duplicate
or unknown identity, that the initial LR grid is complete, and that every selected result comes
from an eligible completed run. A caller-supplied partial grid or derived-result JSON cannot
masquerade as complete evidence.

## 9. Runtime gates

    TRAINING_GPU_MODEL              DEFERRED_UNTIL_OWNER_PILOT_AUTHORIZATION
    selected CUDA device count      exactly 1
    CUDA available                  required
    bf16 supported                  required
    selected device identity        exact match to the future owner authorization
    base runtime fingerprint        exact match to the future owner authorization
    NumPy                            exactly 2.2.6 (see requirements-pilot-v2_3.txt)

No GPU product name or VRAM class is a permanent V2.3 allow/deny rule. An exact runtime device
name gains authority only when the owner selected that identity and bound the complete observed
fingerprint in an `AUTHORIZED` manifest. The runtime must record positive total VRAM, CUDA
capability, driver/CUDA versions, torch version/build, and Python/runtime identity. A missing or
malformed field, zero or multiple visible CUDA devices, unavailable CUDA, unsupported bf16,
identity mismatch, or fingerprint mismatch refuses execution.

The current machine has no authority to run a real pilot because this contract publishes no
owner `AUTHORIZED` manifest and the owner has not selected it as the pilot runtime. Non-training
tooling and tests may continue on it.

Phase MB remains the hardware calibration on the selected actual GPU. Its frozen ten-candidate
grid, eligibility rules and tie ladder determine `FROZEN_MICRO_BSZ`, `FROZEN_GRAD_ACCUM` and
`FROZEN_COMPILE` from measured throughput and peak VRAM. No GPU model implies eligibility, and
no eligible candidate yields `PHASE_MB_ABORT`.

## 10. Output and checkpoint isolation

The **authorized output root** is validated before anything is written under it: it may not be,
sit inside, or contain an accepted release, and its parent must already exist. Every candidate
then uses a new candidate-specific directory that must resolve beneath that validated root.

Before that validation succeeds, the worker writes nothing beside a candidate spec, phase plan,
authorization, index manifest, accepted release or any other input artifact. Pre-validation
failure is communicated only by process exit status and captured stdout/stderr. After validation,
all candidate artifacts, including the strict `terminal.json` and preserved worker logs, remain
beneath the resolved candidate directory under the authorized root.

    PILOT_CHECKPOINTING    DISABLED

V2.3 writes and reads **no** pilot training checkpoint. There is no save path, no resume path
and no `--resume` option; `require_checkpointing_disabled()` refuses either action, and
`reject_pilot_checkpoint_as_initialization()` refuses a pilot checkpoint as the initialization
for another candidate, Stage N or Stage O. A candidate that fails is rerun from scratch under a
new authorized root, never resumed.

## 11. Fingerprint separation

The base runtime fingerprint records the exact selected device index and `torch.cuda` name,
visible device count, total VRAM in bytes/MiB, CUDA capability, driver and CUDA runtime, torch
version/build, Python version/implementation/executable, NumPy and tokenizers versions, platform
and container template, repository branch/HEAD and worktree status, the contract SHA and the
execution bundle SHA. The owner authorization binds its SHA-256. `SESSION.json` and the Phase-MB
report record both the full fingerprint and its SHA.

It carries **no** per-run configuration: `compile`, `micro_bsz`, `grad_accum`, `peak_lr`, `seed`
and `phase` live in each run_meta instead, and the module asserts their absence. Within a
`FULL_V2_3_PILOT`, the Phase-MB and Phase-Muon-LR runtime fingerprint must remain exactly equal.
Before LR publishes any plan, the MB report fingerprint is self-hashed, compared with its bound
SHA and VRAM, and compared in full with the freshly validated LR runtime. An incompatible change
aborts the session; the executor never migrates the frozen MB geometry automatically.

## 12. Validation at execution (R4)

Nothing is trusted because a caller said so, and **no constructed object is execution
authority**. `validate_execution_artifacts(authorization_path, pilot_index_manifest_path,
output_dir, requested_phase)` re-derives the artifact-bytes layer:

    authorization manifest      loaded and hashed from disk
    authorized status + scope   from those bytes
    branch / HEAD / worktree    observed via git, including the allowed-probe rule
    contract SHA                recomputed
    execution-bundle SHA        rederived by AST closure walk over the roots
    pilot-index manifest        FILE SHA-256 computed from disk, never read out of the manifest
    pilot indices               regenerated and compared list by list
    accepted Stage A / Stage B  frozen manifest SHA plus a pre-construction canonical scan of
                                every declared shard's path, geometry and bytes; gaps, extras,
                                symlinks and hash drift are binding failures
    authorized output root      validated before any write
    runtime fingerprint         owner hardware binding + complete observed identity + exact NumPy
    token-ledger identity       eight fields, revalidated on every lock-held operation
    requested phase and scope   PHASE_MB_ONLY may never execute Phase LR

### The sole real executor takes canonical artifact PATHS only

`execute_candidate_from_artifact_paths()` is the only supported route to model construction, a
forward, a backward or an optimizer update. Its **only** raw inputs are paths:

    authorization_path          session_manifest_path       phase_plan_path
    candidate_spec_path         pilot_index_manifest_path   accepted_stage_a_path
    accepted_stage_b_path       ledger_path                 candidate_output_path

It calls `validate_worker_execution()`, which revalidates every artifact from disk and returns
only non-authorizing decoded data. Only after that call succeeds does the same lexical executor
define and invoke its local model/dataset construction and update-loop operations. Those
training-capable operations have no module-level callable alternative. There is no
`WorkerAuthority`, private-constructor token, caller-created `ExecutionSession`,
`ValidatedContext`, `authorized` Boolean or equivalent mutable object that grants execution;
the orchestrator's `ExecutionSession` is metadata that cannot reach a training backend. Fakes
used by tests are injected as launchers with a different signature and cannot invoke the real
executor.

The authorized root is taken as the parent of the candidate's own output directory and is then
checked against the authorization manifest's `allowed_output_root`, so a worker pointed at a
directory outside the authorized root fails before anything is constructed.

### The immutable artifact chain

    SESSION.json
      -> phase plan
        -> canonical candidate spec
          -> run_meta.json
          -> result.json
          -> durable hash-chained ledger receipt
          -> candidate-local terminal.json
        -> authoritative phase report
          -> next internally derived plan

Every immutable file is published once, atomically, with a SHA-256 sidecar and is re-hashed
whenever it is read. The ledger itself advances atomically under its lock; its immutable receipt
records form the durable bridge from raw candidate evidence to the terminal and authoritative
report. Report JSON is a derived view of that underlying chain, never an independent source of
geometry or eligibility.

`SESSION.json` binds the authorization SHA, the contract SHA, HEAD and branch, the execution
bundle SHA, the pilot-index-manifest FILE SHA, the serialized-index-lists digest, both accepted
release identities, the full base runtime fingerprint and its SHA, the authorized output root,
the ledger identity and relpath, the effective ceilings, the scope and the session ID. The
session ID itself is
`sha256(authorization SHA, scope, output root, contract SHA, bundle SHA)`.

`PHASE_MB_PLAN` binds the SESSION SHA and **exactly the ten frozen candidate specs**, each by
relative path and SHA-256. `PHASE_MB_REPORT` binds the SESSION SHA, the PHASE_MB_PLAN SHA, all
ten validated candidate results and terminal outcomes, the selection trace, measured physical
VRAM, the full base runtime fingerprint and its SHA, and the selected `FROZEN_MICRO_BSZ` /
`FROZEN_GRAD_ACCUM` / `FROZEN_COMPILE`.

For `FULL_V2_3_PILOT`, `PHASE_LR_INITIAL_PLAN` binds the SESSION SHA, the PHASE_MB_REPORT SHA,
the frozen MB geometry and exactly the seed-1 initial LR candidate specs. A confirmation plan
binds the preceding validated LR report and selection SHA plus its internally derived
confirmation specs; an edge plan binds the preceding confirmation report SHA plus its internally
derived edge specs.

**No caller chooses candidate membership after a phase plan is published.** Before validating a
chosen candidate spec, the worker validates the **entire** phase plan as the exact contract-derived
ordered candidate set: no missing, extra or duplicate candidate identity, spec hash, spec path or
output path is permitted, and the aggregate candidate-ID and candidate-spec-SHA lists must match
that order exactly. It opens every canonical `_specs/<candidate_id>.json` file and compares its
bytes with a fresh contract derivation.

Only then does the worker validate the chosen spec's SHA, membership, canonical path and output
directory. A spec that is unlisted, self-declared or inconsistent with any contract-derived field
fails **before model construction**. The worker is an internal subprocess entrypoint
(`internal-worker`); invoking it directly with an arbitrary spec therefore confers no execution
capability at all, and the old public `execute-candidate` command is gone.

That derivation is never taken from the plan's own declarations. For every Phase-LR plan kind the
worker opens the Phase-MB report the plan binds, re-hashes it, recomputes each candidate's
throughput from its raw timing records, re-derives the selection ladder, and requires the plan's
declared geometry to equal the `FROZEN_MICRO_BSZ` / `FROZEN_GRAD_ACCUM` / `FROZEN_COMPILE` it
reproduces. The LR set is then derived the same way:

    INITIAL        the frozen seed-1 grid {2e-4, 3e-4, 4e-4}, at seed-1
    CONFIRMATION   [seed-1 winner, confirmation_neighbor(winner)] at seed-2, where the winner is
                   re-derived from the bound initial report's own recomputed records
    EDGE           [edge_candidate(confirmed LR)] at both seeds, where the confirmed LR is
                   re-derived from the bound confirmation report's own pairs

so a plan that declares a geometry or an LR the bound evidence does not produce is refused before
model construction, and the comparison against the published spec can never be a tautology.

### Completed evidence binds run_meta and the planned candidate

Every admitted candidate result must have a real `run_meta.json` in its own output directory.
The loader opens it, computes its SHA-256 **from disk**, and requires agreement with both the
result artifact, immutable terminal, durable ledger receipt and planned candidate identity
across: phase, candidate ID, candidate-spec SHA, phase-plan SHA, session SHA, seed label,
`micro_bsz`, `grad_accum`, `compile`, the output directory, and—independently of nested
diagnostics—the top-level `peak_lr`. The exact planned LR must therefore agree in the plan
identity, candidate spec, run meta, raw result, terminal, receipt and admission record.

The same chain binds the session ID, authorization SHA, contract SHA, HEAD, execution-bundle SHA,
index-manifest FILE SHA, runtime-fingerprint SHA and ledger identity. A fabricated digest with no
file behind it, an LR-label permutation, or any unknown or mismatched field rejects the candidate
from authoritative selection.

### Result classes

    SUCCESS               0
    CANDIDATE_INELIGIBLE  3    candidate-local; the required grid continues
    PHASE_ABORT           4    the phase cannot continue under the frozen contract
    BINDING_FAILURE       5    an identity binding is broken; never downgraded

Each validated candidate subprocess writes one immutable structured terminal
(`petitgpt-pilot-candidate-terminal-result-v2.3-r4`) at
`<candidate_output>/terminal.json`. It is published only after immutable `run_meta.json`
and `result.json` artifacts and the matching durable ledger receipt exist.

Its exact field set binds status/error data; phase; candidate ID; top-level `peak_lr`;
candidate-spec, phase-plan, session and authorization SHAs; planned, reserved and completed
updates; candidate-local reserved/completed update counts; aggregate reserved/completed token
maps; the full ledger identity; the ledger-receipt SHA; and the run-meta/result SHAs.

JSON types, integer ranges, SHA syntax, map keys and accounting geometry are strict. A fresh
locked receipt lookup must exactly reproduce the terminal's identity and accounting; the fresh
ledger snapshot must be at least as advanced as that receipt, so historical terminals remain
valid after later candidates. A malformed or ledger-inconsistent terminal is a phase-level
integrity failure. A missing terminal or exit/status disagreement is a `PHASE_ABORT`, never a
pass.

The pre-finalization ledger snapshot embedded in each raw result is also redundant evidence: the
admission loader reconstructs its exact active-candidate and aggregate state from the durable
receipt and rejects any mismatch before that result can enter a report.

**Terminal accounting is real progress.** On success it reports exact completed updates and
tokens; after a validated candidate-local exception it reports the actual partial local and
aggregate figures from the durable receipt—never a reconstructed zero. The parent preserves
those fields verbatim in candidate evidence and never guesses them.

### Exception classification

Ordinary candidate-local errors — OOM, non-finite loss or gradient, a compile-candidate failure,
an ordinary runtime error — may become structured candidate-local evidence, and the required
grid continues. Everything below aborts the phase or propagates non-success and is **never**
downgraded to an ineligible candidate:

    malformed or missing terminal artifact     authorization mismatch
    session or phase-plan mismatch             accepted-release mismatch
    implementation mismatch                    ledger-integrity failure
    output-root mismatch                       runtime-binding mismatch
    KeyboardInterrupt                          SystemExit

No `BaseException` handler converts a process-control event into an ordinary outcome:
`KeyboardInterrupt` and `SystemExit` are not `Exception` subclasses and are deliberately not
caught, so the worker leaves no terminal artifact and the parent aborts the phase.

Classification is by **execution stage**, not by exception type. A candidate that has not yet
completed the full artifact validation and entered the validation-owned lexical executor has not
reached model construction and therefore cannot be "candidate-locally ineligible": any failure
while the artifacts are still being revalidated is a
`BINDING_FAILURE`, whatever its type. Every canonical artifact — the authorization manifest, the
pilot-index manifest, `SESSION.json`, every phase plan, every report, every candidate spec,
`run_meta.json`, every result and the token ledger — is decoded through one guarded reader, so an
unreadable or non-object artifact surfaces as a binding failure instead of an unreadable
built-in error that would look like an ordinary ineligible candidate.

### Authoritative reports are reconstructed views

`PHASE_MB_REPORT.json` is published **once**, with a SHA-256 sidecar, and is the only source of
Phase-LR geometry — the `run` subcommand has no `--micro-bsz` and no `--compile` option. Before
Phase LR starts, the report is re-hashed against its sidecar and session binding. The loader then
reopens and hashes the bound immutable Phase-MB plan, every canonical candidate spec, every raw
`result.json`, every `run_meta.json`, every strict terminal and every durable ledger receipt.
It recomputes exact ten-candidate completeness, compile evidence, eligibility, median throughput
and the canonical tie-break selector. The reconstructed candidates, artifact-hash maps, selection
trace and frozen geometry must exactly reproduce the published report.

Every LR report is reconstructed the same way, recursively from its bound Phase-MB report, phase
plan and preceding LR report. The loader reopens the required candidate/seed set and recomputes
raw evaluation losses, SCORE, eligibility, confirmation-neighbour and edge derivation, and the
canonical selectors through the final winner. Only that reconstructed outcome may authorize a
downstream plan. Editing report JSON and its sidecar without matching raw evidence therefore
cannot change a decision.

The MB report's full runtime fingerprint is also self-hashed and required to equal the freshly
validated Phase-LR runtime; its recorded physical VRAM must equal that fingerprint. Any mismatch
aborts before an LR plan or candidate can start.

### Independent recomputation

No selection number is taken on trust:

    Phase MB    the median of the per-update RATES, recomputed from the timing records, with
                exactly one record required for each of updates 11..40
    Phase LR    loss_A = numerator_A / weight_A, loss_B = numerator_B / weight_B, then
                SCORE = (10*loss_A + 3*loss_B)/13

Selection consumes only recomputed values. A stored `eligible` is redundant summary data:
`false` never skips raw-evidence recomputation, `true` never overrides it, and when present it
must exactly equal the recomputed Boolean verdict. Stored metric and selector summaries are also
non-authoritative and must agree with recomputation to within double round-off
(`math.isclose`, relative tolerance 1e-12) — canonical JSON round-trips doubles exactly, so
anything looser would be slack. Any serialized value that disagrees is a `BINDING_FAILURE`.

### Evidence completeness

The initial LR grid must carry a record for all three grid points; confirmation must carry a
record for both LRs at **both** seeds; a bounded edge expansion must carry a record at both
seeds before it is resolved either way. An ineligible record is still a record — a *missing*
record is an evidence gap and aborts the phase.

### Timing and compile evidence

    torch_compile_wrapper_seconds        the torch.compile() call itself
    first_optimizer_update_wall_seconds  update 1, which materializes lazy compilation
    update_timings                       Phase MB: one record per update 11..40, the throughput
                                         sample. Phase LR records every update; only Phase MB's
                                         window is a selection input.

No host/device scalar transfer happens inside a timed region: losses and gradient norms stay as
device tensors and are converted after the loop, and scalar logging happens outside it.

Compile evidence is a **runtime observation of the execution path**, not a Boolean copied from
the candidate spec. The update loop invokes the model through a wrapper that records which
object it called and how many times, so:

    compile=on    the object torch.compile() returned must be the callable that was invoked,
                  the realized module must be the Dynamo wrapper, compilation must have
                  materialized (graph counters or Inductor artifacts on disk), and the
                  candidate must have completed the required updates without compile failure
    compile=off   the uncompiled module must be the callable that was invoked and nothing may
                  have gone through Dynamo

In both directions the invocation count must equal the geometry the contract derives, never a
number measured back off the wrapper being checked:

    Phase MB    updates x grad_accum
    Phase LR    updates x grad_accum, plus one forward per evaluation micro-batch --
                ceil(4096 / micro_bsz) for each of the Stage-A and Stage-B eval sets, since both
                evaluations run through the same observed wrapper

The eligibility/selection loader **re-derives** this verdict from the recorded observation
rather than trusting the stored `canonical_compile_path`, and a candidate whose stored verdict
disagrees with its own recorded observation is a `BINDING_FAILURE`. A candidate that *honestly*
records a silent fallback is merely **ineligible** — the contract's own eligibility rule — and
the required grid continues. Dynamo's counters are used where they are available and are treated
as corroboration, never as a hard dependency on a private API.

### Muon RMS matching — an independent oracle

`verify_rms_matching()` runs one real Muon step per deterministic shape case and compares it
with a closed form that never calls the realization it verifies. The gradient is constructed to
be exactly semi-orthogonal, which collapses the quintic Newton–Schulz iteration to a scalar
recursion that is written out directly:

    beta_0    = sigma / (||g||_F + 1e-7),  ||g||_F = sqrt(min(fan_in, fan_out)) * sigma
    beta_k+1  = a*beta_k + b*beta_k^3 + c*beta_k^5     (a, b, c) = (3.4445, -4.7750, 2.0315)
    expected  = p_before*(1 - lr*wd) - lr*0.2*sqrt(max(fan_in, fan_out)) * beta_5 * g/sigma

Neither half of the expectation comes from `src/optim.py`. The shape set includes both
non-square orientations, and each case records the margin by which an unscaled implementation
would differ — five orders of magnitude above the observed residual — so the check cannot pass
vacuously. `pilot_runner_v2_3.py rms-matching` prints the evidence without training, and
`pilot_runner_v2_3.py session-budget` prints the frozen session semantics.
