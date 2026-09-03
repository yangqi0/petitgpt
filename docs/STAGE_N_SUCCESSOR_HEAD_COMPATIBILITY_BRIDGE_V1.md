# Stage-N successor-head compatibility bridge v1

This document records the incident-specific engineering decisions behind
`N-STAGE-N-TO-STAGE-O-SUCCESSOR-HEAD-COMPATIBILITY-BRIDGE-R2`. It is a design
record, not an authorization. Nothing in this document authorizes N3, Stage O,
or any training operation.

## Incident

The accepted Stage-N authorization schema made `stage_n_completion` optional,
and Gate A accepted the N1 and N2 authorizations without that block. The later
smoke, resume, COMPLETE, and Stage-O publication paths required the block. The
immutable N1/N2 artifacts therefore agree on the true Stage-A endpoint but
cannot pass the old-head publication validator.

The accepted exact Stage-P plan is the boundary authority. It derives Stage-A
stop step `38146`; with `128` sequences per optimizer update it derives sampler
endpoint `4882688`. An authorization's optional `stage_n_completion` is only a
redundant cross-check: absence is accepted when the plan and independently
authenticated execution artifacts agree, while a present mismatch is rejected.

## Challenges encountered

1. **Immutable history versus a late validation defect.** Editing N1/N2
   authorization bytes would change their SHA-256 identities and invalidate the
   governed history. Re-running N1 would be costly and would create different
   execution history; re-running N2 would add no information because its
   zero-update evidence is already sufficient.
2. **Repository identity is part of runtime identity.** The training updates ran
   at historical HEAD `6d80423adc16d4a160a7fe42660020c585b5185d` with trainer
   bundle `bbd49b9d73d3cb2fa18aacb3eee861a901e5a7511ed334b85b37239ab1d50043`.
   A repaired HEAD cannot truthfully claim those updates as its own.
3. **A general cross-HEAD escape hatch would weaken the contract.** A version
   range, allowlist, or materiality heuristic could accidentally authorize a
   later semantic change. The bridge must instead bind one reviewed source,
   one reviewed destination, one byte-comparison manifest, and one exact
   N1/N2/runtime chain.
4. **The successor commit identity is not known while its tracked bytes are
   being authored.** Embedding that commit SHA in the same commit would be
   self-referential. The implementation therefore validates an authorization's
   destination against the repository and bundle observed at execution time;
   the checksummed `NOT_AUTHORIZED` candidate is generated only after the final
   commit and binds that one exact destination.
5. **Zero updates still require a real process transition.** N3 must restore the
   N2 terminal checkpoint under the successor implementation and realize the
   compiled process locally, yet it must not enter a training loop, consume
   data, call backward or `optimizer.step`, advance a schedule/sampler, or alter
   training state.
6. **Stage O must not bypass N3.** Accepting an old checkpoint directly would
   leave the code-identity transition unproved. The successor Stage-O chain must
   authenticate the accepted N3 result and resume only its successor-bound
   terminal checkpoint.
7. **Code root and execution CWD were previously conflated.** The reviewed
   successor is authored in a separate worktree while the immutable historical
   tree must remain at `/workspace/petitgpt`. The old launch module derived
   `CANONICAL_CWD` from its own file root, which cannot represent successor code
   loaded from the repair worktree while preserving the exact historical
   execution/data CWD.
8. **Injectable execution callbacks escape the reviewed bridge bundle.** A
   caller-supplied checkpoint loader, saver, or compile realizer could perform
   the load/realize/save transition using code that is absent from the bridge
   closure. An authorization that binds only the bridge bundle would then not
   bind the code that actually performed N3.
9. **Claim-only validation and time-of-check/time-of-use gaps are unsafe.** A
   structural validator can be useful in tiny tests, but accepting caller-owned
   dictionaries without reopening their exact bytes does not authenticate an
   execution. Likewise, validating once and publishing several files later can
   miss a changed source or strand a partially published result set.
10. **The existing resume evidence schema describes a training resume, not an
    N3 bridge.** Reusing it verbatim would incorrectly require a second ordinary
    Stage-N authorization and could either reject every valid N3 result or tempt
    an overly broad relaxation of normal resume checks.
11. **Namespace-package imports can silently cross the two roots.** With the
    process CWD fixed at the historical tree, a normal
    `pretrain.production_launch_contract_v1` import can resolve historical bytes
    even though the bridge itself was loaded from the successor worktree. A
    previously populated `sys.modules` can preserve the same split-brain state.
    Checking only Git HEAD, a bundle digest, or the bridge module's own path does
    not prove which launch contract and realization dependencies are executing.
12. **Hashing a checkpoint path and then deserializing that path is not one
    authenticated operation.** A path can change between the SHA-256 read and
    `torch.load`, causing reviewed bytes and executed bytes to differ. The real
    checkpoint is also large enough that closing this race by retaining its
    complete byte image has a meaningful peak host-memory cost.
13. **Per-file atomic writes do not make a result set atomic.** Publishing the
    checkpoint and JSON artifacts one at a time can expose a partial N3 result,
    and a check that the destination is absent does not prevent another process
    from creating it immediately before a replacing rename. Crash durability
    also requires an explicit directory-sync boundary, not only flushed files.
14. **Valid COMPLETE bytes are not sufficient location authority.** An exact
    copy of COMPLETE outside the N3 destination retains the same content hash
    and embedded chain. If Stage O validates only that document, the copy can
    detach acceptance from the authorized output topology and the live
    underlying artifacts.
15. **Legacy full-suite tests bind canonical resources through module
    location.** Two unchanged Stage-I test targets derive accepted artifact
    paths from `__file__` and therefore expect their modules to resolve under
    the immutable `/workspace/petitgpt` tree. Loading the entire suite from the
    successor worktree changes those paths and creates location failures
    unrelated to R2 behavior, while loading everything from the historical tree
    would fail to exercise the changed R2 implementation.
16. **A passing artifact alone does not identify the command that produced
    it.** A JUnit document, static-check log, or zero exit code could otherwise
    come from a filtered suite, an unrelated file set, a different repository,
    or selector options injected through the environment. Such evidence would
    look healthy without proving the required R2 regression and closure checks.
17. **A self-consistent hash graph is not semantic authority.** A forged GRC or
    COMPLETE document can recompute every content SHA, update every parent
    reference, and still describe an unauthorized contract. This is especially
    important for governed-digest exclusions: a field outside the semantic
    digest cannot become valid merely because its artifact hash was updated.
18. **Flat runtime fields do not capture device-selection identity.** The N2
    runtime artifact includes a nested `selected_device_resolution` record with
    the logical-to-physical CUDA mapping, NVML identity, failure list, and
    mapping method. Rechecking only UUID, PCI, versions, and worker count would
    allow that nested resolution evidence to drift while the visible summary
    remained unchanged.
19. **Separate JSON hash and parse reads create authority races.** Reading a
    path for SHA-256 and reopening it for JSON permits the validated byte image
    and interpreted document to differ. The same problem occurs when Stage O
    parses COMPLETE, smoke/resume, acceptance, authorization, manifest, GRC,
    runtime, plan, or N2 evidence and later reopens the path for its hash.
20. **Generic compile evidence does not prove bridge-specific zero work.** The
    normal compile verifier authenticates realization, causal diagnostics, and
    fail-closed compiler behavior, but it does not interpret N3's before/after
    state proofs, zero execution counters, or assertions that no loader,
    sampler, scheduler, or training loop was constructed. Those facts require
    their own exact schema and cross-channel consistency checks.

## Decisions and tradeoffs

### Preserve training semantics byte-for-byte

The repair may change the publication/validation contract, add one bridge
module, tests, and this document. The eleven core model, optimizer, schedule,
dataset, sampler, and trainer files remain byte-identical. Across the twelve
historical governed closure files, exactly
`pretrain/production_launch_contract_v1.py` changes.

This is intentionally stricter than a semantic code review. Byte equality is
easy to reproduce and leaves no judgment call about whether a source edit is
"material."

### Derive completion once

All successor publication and validation paths use one pure exact-plan
derivation. This avoids independent constants drifting between smoke, resume,
COMPLETE, bridge, and Stage-O code. The tradeoff is stricter artifact loading:
publication cannot proceed from a claimed step alone; it needs the exact plan
and agreeing governed artifacts.

### Use an incident-scoped N3 authorization

N3 uses scope `STAGE_N_SUCCESSOR_HEAD_COMPATIBILITY_BRIDGE` and begins as
`NOT_AUTHORIZED`. Its future output root, source and destination identities,
semantic-isolation manifest, plan/pilot identities, N1/N2 chain, runtime/GPU,
checkpoint, and zero-work counters are all bound explicitly.

The bridge policy is deliberately not reusable for a second successor HEAD.
Any later repository, bundle, GPU, driver, CUDA, Python, PyTorch, NumPy,
`num_workers`, or canonical-CWD change requires another owner decision and is
not covered by this exception.

### Separate provenance roles

The successor COMPLETE result records three roles independently:

- historical Stage-N training execution HEAD/bundle;
- successor N3 compatibility-bridge HEAD/tool bundle;
- successor Stage-O execution HEAD/governed-trainer bundle.

This adds redundant fields, but the redundancy prevents the standard
successor-bound result/GRC fields from erasing the historical identity of the
actual training updates.

### Require state equivalence, not numerical closeness

Model tensors must be bitwise identical. Optimizer and scaler state, milestone
prefixes, stage/step/seed/cursor, and restored Python/NumPy/Torch CPU/Torch CUDA
RNG streams must be equivalent under canonical zero-update behavior. Only
authorization identity, successor GRC/code identity, process-local compile
evidence, and invocation/output metadata may differ.

This rejects harmless-looking training-state changes as well as harmful ones.
That conservatism is appropriate because N3 performs no learning and therefore
has no legitimate reason to change training state.

### Separate the immutable execution CWD from the successor code root

The future process must actually start in `/workspace/petitgpt`, preserving the
historical runtime and absolute data/artifact topology. The loaded module, Git
HEAD, and closure bundle must come from the exact reviewed successor worktree.
Both roots are observed and recorded independently: process CWD is never inferred
from `__file__`, and successor code identity is never inferred from process CWD.
An explicit absolute entry point or `PYTHONPATH` may select the reviewed code,
but it cannot change either observed value.

This dual-root topology is more operationally explicit than assuming the active
code and data tree are the same directory. It is required here because changing
the historical worktree HEAD would destroy the immutable evidence the bridge is
meant to authenticate. Any different CWD or code root fails the exact candidate.

### Root imports in exact successor files and assert module origins

The bridge loads the launch contract and its realization dependencies from
absolute reviewed successor-worktree paths under private, collision-free module
names. It reuses a canonical or private module only when that module's resolved
`__file__` is the exact expected file. A conflicting historical canonical module
is neither trusted nor overwritten. Before N3 execution, the bridge asserts the
bridge, launch contract, launch root, and dependency origins again. Stage O also
observes its production and bridge module paths and rejects any loaded local
`pretrain` or `src` module outside the exact successor repository root.

This is more rigid than ordinary Python import resolution and requires explicit
file-rooted loading in this incident path. The rigidity is intentional: process
CWD remains the historical root for runtime compatibility, while namespace and
module-cache behavior are not allowed to choose execution authority.

### Isolate canonical-root compatibility to the test harness

For the legacy full-suite regression only, a test-only import finder routes
exactly `pretrain.stage_i_graph_v2` and `pretrain.stage_i_realize_v1` to the
historical tree. Before either route is installed, it hashes the historical and
successor copies and requires byte equality; a missing file, unequal hash, or
already-loaded target fails closed. The finder has no package-prefix or
wildcard route, so `pretrain.production_launch_contract_v1`,
`pretrain.stage_n_successor_head_compatibility_bridge_v1`, and all changed R2
tests continue to resolve from the successor worktree.

This deliberately gives one regression process two code roots, which would
normally be undesirable. The exception is limited to two byte-identical,
location-sensitive legacy modules and exists only in the external test harness;
it is not production bridge behavior and is not part of the governed
implementation bundle. Recording the router bytes and invocation with the
JUnit evidence makes the exception independently auditable.

### Bind regression evidence to exact executions

Evidence publication accepts only fixed interpreter/tool paths, an exact
successor-worktree CWD, explicit environment cleanup, fixed focused-test and
static-check path sets, and an unfiltered default-discovery full-suite command.
It cross-checks the JUnit inventories against every top-level R2 test and the
required legacy CUDA/resume-save anchors, records a distinct direct exit-code
artifact for each command, and binds the reviewed test-router bytes.

The tradeoff is deliberately non-portable command metadata: moving the
worktree, virtual environment, or router requires new evidence rather than a
best-effort replay. That rigidity is confined to the external evidence harness
and prevents a successful but differently scoped command from being presented
as proof for this exact repair.

### Bind all executable bridge behavior

The production entry point uses only canonical in-module load, compile-realize,
and save implementations covered by the bridge closure. Private test seams may
exercise tiny fake checkpoints, but neither authorization validation nor the
public execution API accepts executable callbacks from a caller. The fixed real
checkpoint tensor count is also not caller-selectable.

This makes unit tests slightly more indirect, but it ensures the reviewed tool
bundle covers every piece of code that can mutate or publish the successor
checkpoint.

### Hash and deserialize one immutable checkpoint byte snapshot

For each checkpoint that N3 must interpret, the bridge opens the file once,
reads one complete immutable byte snapshot, computes the expected SHA-256 over
that snapshot, and passes those same bytes to `torch.load` through an in-memory
buffer. It never authenticates one file read and then asks the deserializer to
reopen a mutable pathname. The staged successor checkpoint is likewise reopened
as bytes, hashed, and deserialized from that same snapshot before publication.

The cost is higher peak host memory: serialized checkpoint bytes coexist with
the deserialized checkpoint state, and bridge state-equivalence checks retain
additional copies. That cost was accepted for this one-time zero-update
transition because byte-to-execution identity is stronger than a lower-memory
path-based load. The authoritative COMPLETE check repeats this operation for
the exact N2 source and successor checkpoint, validates both envelopes, and
independently recomputes the fixed 213-tensor/full-state equivalence proof.

### Separate structural checks from authoritative checks

Private helpers may validate synthetic mappings for focused tests. Every public
authorization or Stage-O decision reopens the exact bound paths, rehashes their
bytes, observes the live source/destination/runtime identities, and fails closed
when any live root is unavailable. Immediately before publication the bridge
repeats the immutable-input and identity checks and preflights every destination
as absent. Canonical JSON writes are atomic; a COMPLETE result is published only
after all underlying artifacts validate.

### Stage and validate the whole result before no-replace publication

N3 writes every authorized output into a private sibling staging directory with
exclusive creation and file `fsync`. It verifies the exact staged file set,
reopens the staged COMPLETE document, validates its full graph using an explicit
authorized-to-staged path mapping, and then reobserves authorization, manifest,
N2 evidence, source checkpoint, historical identity, successor identity, and
destination absence. Only that fully validated directory is made visible.

Installation uses Linux `renameat2(RENAME_NOREPLACE)` on the same filesystem,
followed by `fsync` of the destination's parent directory. The no-replace flag
closes the final absence/use race without overwriting a concurrently created
file, symlink, or directory. The portability tradeoff is deliberate: a platform
or filesystem without the required Linux primitive is unsupported for this
incident bridge and fails closed instead of falling back to a replacing or
multi-file publication. A pre-install failure can leave an unpublished staging
directory for inspection, but it leaves the authorized final destination
unpublished.

### Revalidate the authoritative COMPLETE topology at Stage O

The successor Stage-O validator requires its accepted result path to be exactly
the N3 authorization's `destination.complete_result_path`; an identical copy at
any other path is rejected. It then invokes the bridge's public authoritative
COMPLETE validator, which reopens the authorization and the complete governed
artifact graph, checks the authorized runtime/smoke/resume/checkpoint topology
and byte hashes, and performs the live bridge-authorization preflight. Owner
acceptance must bind that same path and SHA-256.

This repeats filesystem reads, hashes, deserialization, state comparison, and
live identity observations already performed during N3 publication, but it
prevents a stale, relocated, partially substituted, or self-consistently forged
result from becoming Stage-O authority. The additional host-memory and latency
cost is accepted at this one-time trust boundary. The Stage-O check remains
read-only: it performs no checkpoint write, bridge execution, data access,
backward pass, optimizer step, scheduler/sampler advance, or training loop.

### Reconstruct the entire successor artifact graph

The authoritative validator authenticates the exact N2 checkpoint snapshot and
rebuilds the only permitted successor GRC from that source, the exact N3
authorization, runtime, semantic manifest, and compile evidence. It then
rebuilds the canonical runtime wrapper, smoke and resume check documents, and
the entire COMPLETE document. Acceptance requires type-strict equality with
those projections, not merely matching embedded hashes or selected fields.
The GRC, smoke evidence, and checkpoint's dynamic compile evidence must all be
the same sealed document. The N3 authorization and semantic-comparison
manifest likewise use exact template and record field sets; unknown fields and
JSON boolean/integer/float aliases fail closed. The optional completion block
may be absent, but when present it contains only the exact derived final step.

This is intentionally less forward-compatible than accepting unknown result or
GRC fields. A schema extension now requires an explicit reviewed builder and
validator change. That cost is preferable to letting an internally consistent
but unauthorized field graph define new launch authority.

### Project runtime from the authenticated N2 fingerprint

The allowed successor runtime is a deep copy of the authenticated historical
N2 runtime mapping with only the reviewed trainer HEAD, trainer bundle, and the
derived runtime self-hash changed. Every other field, including the full nested
`selected_device_resolution`, must remain type-strictly identical. Runtime is
therefore an exact one-time projection, not a comparison of a convenient flat
subset.

This couples N3 to the complete historical runtime schema and makes benign
diagnostic additions incompatible without a new decision. That rigidity is the
intended tradeoff for proving that one specific process/device identity—not a
lookalike summary—continued across the zero-update transition.

### Hash and parse one JSON byte snapshot

At each successor authority boundary, code reads a JSON artifact once, hashes
that byte image, and parses that same image. Cross-call handoffs carry the
captured SHA and, where needed, the exact parsed document; Stage O also passes
its accepted COMPLETE snapshot SHA into the bridge validator. The exact-plan
loader follows the same rule before deriving the completion boundary.

This retains more mappings and repeats some reads when freshness must be proved,
but it eliminates the ambiguity of attributing one path read's digest to a
different path read's semantics. Tests use explicit seams for synthetic
artifacts rather than weakening production snapshot rules.

### Apply bridge-specific compile-evidence semantics

N3 validates the generic governed compile seal and an exact
`bridge_zero_update_observations` block. Both before/after state proofs must show
213 bitwise-identical model tensors, equivalent optimizer/scaler state,
preserved RNG, and absent gradients; every execution counter is the integer
zero; and construction of training-loop, loader, sampler, and scheduler
surfaces is exactly false. The realizer's separately returned counters and live
proof must equal the embedded evidence, preventing contradictory channels from
driving checkpoint publication.

The extra validator and redundant cross-bindings add schema and test burden.
They are retained because a successful compile realization alone cannot prove
that a zero-update compatibility bridge performed no training work.

### Use a dedicated N3 resume-evidence schema

The bridge publishes resume evidence that binds the N3 authorization, successor
GRC and checkpoint, semantic comparison, exact N2 source checkpoint, runtime,
all zero-work counters, and state-equivalence proof. Production result
validation dispatches to this schema only when the surrounding authorization is
the exact N3 schema. Ordinary Stage-N resume evidence keeps its existing strict
validator unchanged.

### Preserve legacy integration intent without weakening successor dispatch

The unfiltered repository suite exposed four older Stage-O integration tests
whose fixtures mechanically copied the live repair branch into otherwise
synthetic, pre-N3 authorizations. Under R2 that branch is the successor
execution authority, so the new fail-closed dispatch correctly rejected those
fixtures before the older tests could reach the resume behavior they were
designed to cover.

The production dispatch was retained: an ordinary legacy chain cannot select
the Stage-O path while the exact successor branch is authoritative. Instead,
the four legacy tests now explicitly select a test-only non-incident policy
branch in both import identities of the launch-contract module. Their original
generic crash-restart and A-to-B assertions remain intact, while the focused R2
suite separately requires the real successor branch to reject that same
pre-N3 shape.

The tradeoff is a small policy-constant monkeypatch in legacy tests. Rebuilding
their fixtures as full accepted N3 graphs would duplicate the successor tests
and obscure their older resume-path purpose; relaxing the production branch
guard would allow an arbitrary pre-N3 checkpoint to bypass the required bridge.
Keeping those concerns separate makes the owner supersession explicit and
keeps the production fail-closed boundary unchanged.

## Lifecycle after this implementation

1. Independent review verifies the successor commit, closure bundles,
   semantic-isolation manifest, tests, and `NOT_AUTHORIZED` candidate.
2. A separate owner action may authorize that exact candidate.
3. Only then may the zero-update N3 bridge run and publish successor-bound
   artifacts for independent acceptance.
4. Only an accepted N3 COMPLETE result may be used to construct a Stage-O
   authorization, and Stage O resumes only the N3 terminal checkpoint.

The implementation segment stops before step 2. It publishes no Stage-N result,
does not execute N3, does not authorize Stage O, and performs no GPU training.
