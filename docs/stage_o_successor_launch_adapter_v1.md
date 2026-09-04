# Stage-O successor launch adapter v1

## Purpose and boundary

This adapter is an adjacent execution tool for the already accepted Stage-O trainer at
commit `7686fd811642dd6246ca3a3c21a4bf43bc28cd3b`. It does not change, copy, or replace
the trainer, launch contract, parser, Gate A, model construction, checkpoint handling, or
training loop. The accepted training identity remains the accepted trainer HEAD and
trainer execution bundle; the adapter records its own HEAD and execution bundle
separately.

The adapter exists only to install one deterministic Python module topology before the
accepted trainer is imported. The canonical name
`pretrain.production_launch_contract_v1`, the bare compatibility name
`production_launch_contract_v1`, and the `pretrain` parent attribute must all reference
the exact same module object loaded once from the accepted successor worktree.

## Safety contract

The adapter validates the accepted successor path, branch, HEAD, governed trainer
closure, launch-contract bytes, and trainer bytes before importing the trainer. It also
validates its own reviewed HEAD, execution bundle, and script bytes against a separate
adapter authorization artifact. Existing module bindings are inspected before mutation.
Bindings with a historical origin, another path, different bytes, incomplete metadata,
or a second object for an accepted source are refused rather than replaced.

Initialization is transactional. If launch-contract or trainer initialization fails, the
adapter restores every relevant `sys.modules` binding, parent-package attribute, and
search-path entry to its pre-attempt value. Repeating setup after success is idempotent.

Preflight imports the real accepted trainer, invokes its real parser, validates the real
successor Stage-N chain, and invokes real Gate A with source authority required for the
`A_TO_B` transition. It never calls model construction or trainer `main()`. Execution is
a separate mode and requires both an owner-authorized adapter artifact and an independently
owner-authorized Stage-O artifact before it delegates exactly once to the accepted
trainer's `main()`. Trainer arguments, stdout, stderr, exceptions, and `SystemExit` are
passed through without retries or reinterpretation.

An adapter authorization can authorize use of the adapter but can never authorize
training. Its `authorizes_training` field is permanently `false`; Stage-O authority comes
only from the separately bound Stage-O authorization consumed by the accepted Gate A.

The machine-readable field contract is recorded in
`docs/stage_o_successor_launch_adapter_authorization_v1.schema.json`.

## Commands

All governed invocations use the historical canonical working directory and the adapter
script by absolute path. The inspection-only closure command is:

```console
/workspace/petitgpt/.venv/bin/python \
  -I -B \
  /workspace/petitgpt_stage_o_successor_launch_adapter_v1/tools/stage_o_successor_launch_adapter_v1.py \
  closure
```

After the implementation is committed, a runtime-bound template can be generated without
trainer import or training:

```console
/workspace/petitgpt/.venv/bin/python \
  -I -B \
  /workspace/petitgpt_stage_o_successor_launch_adapter_v1/tools/stage_o_successor_launch_adapter_v1.py \
  authorization-template \
  --output /absolute/review/path/STAGE_O_ADAPTER_AUTHORIZATION_NOT_AUTHORIZED.json
```

Preflight requires a separately bound Stage-O artifact and passes the exact owner-bound
trainer argument vector after one `--` separator:

```console
/workspace/petitgpt/.venv/bin/python \
  -I -B \
  /workspace/petitgpt_stage_o_successor_launch_adapter_v1/tools/stage_o_successor_launch_adapter_v1.py \
  preflight \
  --adapter-authorization-path /absolute/review/path/ADAPTER_AUTHORIZATION.json \
  --stage-o-authorization-path /absolute/review/path/STAGE_O_AUTHORIZATION.json \
  --output /absolute/review/path/PREFLIGHT_RESULT.json \
  -- <exact accepted trainer arguments>
```

The later execution command has the same boundary and uses `execute` instead of
`preflight`. It is intentionally unusable with the implementation template: execution
requires `authorization_status=AUTHORIZED` and `authorizes_adapter_execution=true` in the
owner-reviewed adapter artifact, while the independently reviewed Stage-O artifact must be
authorized for `STAGE_O`. Both artifacts must bind the exact same adapter identity and
trainer argv. This segment does not create either authorized artifact and does not run the
execution command.

## Challenges encountered

The original direct trainer launch loaded the launch contract under its bare name. When
the successor Stage-O validator later requested the canonical package name, Python either
presented the bridge with a module whose canonical spec name was wrong or executed the
same source a second time. The latter creates distinct class and sentinel families even
though `__file__` and file hashes look identical.

The `pretrain` directory is a namespace package rather than a regular package. With the
required canonical working directory `/workspace/petitgpt`, ordinary namespace discovery
can combine the historical and successor `pretrain` directories. Merely prepending the
successor path is therefore insufficient: a mixed parent-package search path remains a
silent source of historical imports.

The accepted trainer also uses both bare intra-`pretrain` imports and canonical `src.*`
imports during module initialization. A loader that protects only the launch-contract
names can still reuse a preloaded historical dependency. Conversely, eagerly
canonicalizing all bare trainer dependencies would alter their accepted import identities.

Several apparently valid retained states were also unsafe. A retained bridge can keep a
reference to an older launch-contract object after its public bindings disappear; an
accepted `src` package can have a poisoned search path; and a trainer module with correct
path/spec metadata can have a replaced `main` or imported model symbol. File hashes alone
cannot establish that these live Python objects still represent the reviewed executable
topology. Comparing the trainer's imported symbol only with the dependency's current
export is also circular: replacing both `src.model.GPT` and the trainer's `GPT` binding
with the same object would make that weak comparison pass.

Code-object checks alone were likewise insufficient for the required canonical-first
launch reuse case. An exact launch module can retain reviewed functions while its global
`Path`, `hashlib`, or a policy constant is replaced; Gate A resolves those names from
the live module dictionary when authenticating authorization, checkpoint, and Stage-N
bytes. Runtime integrity therefore has to cover the whole launch namespace, not only its
callables.

Runtime baselines also need to preserve negative space and descriptor metadata. If all
bindings for an owned module disappear, an empty candidate scan must not permit its source
to execute again while the old object remains retained. Added globals such as `open` can
shadow builtins without changing any original binding, and an in-place change to a
dataclass `Field` can alter `fields()` and `asdict()` while preserving object identity.

The two authorization files introduced a separate consumption race. The adapter can bind
one Stage-O document while the accepted Gate A later reopens the CLI path. A path alias,
symlink, or atomic replacement could otherwise make the adapter and trainer validate
different documents. The accepted Stage-O schema also does not make its optional
`authorized_by` and `authorized_at` fields load-bearing, while this adapter contract
requires an independent owner decision.

Finally, adapter identity is self-referential if a tracked authorization template tries to
embed the SHA of the commit that contains that template. The reviewed adapter code and the
future owner decision therefore need different artifacts and different lifecycles.

Direct script execution also places the adjacent `tools/` directory on Python's import
search path. A local file such as `tools/json.py`, or a future helper imported from that
directory, could otherwise become executable adapter code without appearing in a closure
derived only from repository-root imports.

That direct-execution exposure exists before the adapter can run its own search-path
cleanup: the interpreter establishes the script directory while starting Python, and the
first filesystem-backed stdlib import could already resolve to a local shadow. Runtime
cleanup alone therefore cannot prove that startup used only reviewed code.

The id-independent launch namespace manifest detects changed executable structure and
policy values, but deliberately normalizes object identities. On repeated setup, a newly
constructed same-structure `LaunchContractError` class or `_Missing` sentinel class
could otherwise satisfy the manifest while splitting the already reviewed object family.

The repository's broad pytest configuration is itself an import-order participant:
`tests/conftest.py` imports `src.*` while pytest is still collecting tests. A rehearsal
run through that global fixture layer therefore presents the adapter with an accepted-path
module that the adapter did not load, and the adapter correctly refuses it before topology
installation. Treating a test harness preload as trusted would weaken the same provenance
rule that protects later execution.

## Tradeoffs and decisions

The adapter creates or reuses an accepted-only `pretrain` namespace and removes historical
worktree entries from the active project import search path before importing accepted
project code. It leaves the accepted trainer's bare `dataset_pretrain` and `sample` names
unchanged, but rejects any preexisting project dependency whose origin is not within the
accepted successor worktree. This preserves trainer semantics while closing the mixed-root
escape route.

An already loaded module is reusable only when its path, current bytes, `__file__`,
`__spec__.origin`, canonical `__spec__.name`, parent package, and object uniqueness are all
valid. In particular, an exact-source object originally executed under the bare spec name
is rejected; rebinding it would hide a structurally different class family. This is a
stricter choice than opportunistic repair, but makes setup deterministic and reviewable.

The launch contract is executed once under its canonical spec and is prebound to both
names during execution so recursive imports cannot create a duplicate. The trainer is
then imported canonically and exposed under its bare compatibility name without executing
it again. Identity checks use object identity for `LaunchContractError`, `_Missing`,
`ObservedForward`, and other cross-boundary symbols; duck typing and class-name matching
are deliberately excluded.

Retained accepted modules are treated as executable state, not just registry entries.
Source modules require coherent names, packages, source loaders, origins, and package
search paths. Every top-level trainer function is compared with the code object compiled
from the exact accepted source without executing that source again, and module-scope
project imports must remain object-identical to their accepted dependency exports.
Additionally, the adapter controls the first load of the bridge and every trainer
dependency, captures their namespace bindings and recursive executable/export tokens, and
publishes those baselines only after the whole topology validates. Repeated setup compares
both dependency and trainer state with that independently retained baseline, so a coherent
two-sided replacement still fails. A stale bridge launch reference or any mutated trainer
surface is refused rather than repaired.

An exact-path trainer, bridge, or dependency that was loaded before this adapter has no
such clean-import provenance and is therefore refused even when its disk bytes and module
metadata look correct. The canonical launch contract is the narrow exception because its
entire cross-boundary function and class family is independently checked against compiled
accepted source. This sacrifices opportunistic reuse of preloaded dependencies in favor
of a deterministic proof that no unreviewed executable object entered the trainer.

For that launch exception, a version-pinned, id-independent runtime-namespace manifest is
derived from the SHA-pinned source contract. AST-derived import bindings must be identical
to their canonical stdlib objects, builtins must use the canonical builtins dictionary,
the namespace name set is exact, and literal/container/`MappingProxyType` policy values
are canonicalized before hashing. Two isolated clean derivations established the reviewed
digest. Pinning the CPython patch version is intentionally strict: a runtime that could
materialize different reviewed objects must receive a new adapter review instead of being
silently normalized.

Owned-module baselines similarly require their recorded bindings and exact namespace name
sets to remain present. They recursively retain function/class/container state, canonical
builtins, and dataclass `Field`/`_DataclassParams` slots. A missing binding, added
global, descriptor mutation, or coherent dependency/trainer replacement is refused and
left untouched; restoring the original state allows the same module objects to be reused
without source re-execution.

The three load-bearing trainer selectors are checked before project import and checked
again after the real parser: a nonempty launch contract, the exact canonical Stage-O
authorization path, and `run_plan_stage=stage_b`. Trainer-side argparse terminators remain
in the delegated argv, but options after a terminator cannot satisfy the early governed
check. This preserves parser semantics while preventing an ungoverned path from reaching
the accepted trainer.

Execution performs the actual accepted Gate A once silently before delegation so both
authorities are validated before `main`. The Gate A naturally executed inside accepted
`main` is not bypassed: its real lower-level call is temporarily guarded so the returned
path, SHA, and document must match the original snapshot before `set_seed` or model
construction. The original function is restored for every return or exception. This costs
one extra read-only chain validation, but retains one normal trainer Gate-A log line,
preserves stdout/stderr, and closes the last document-replacement window at the consumer.
Both authorization paths must be canonical absolute non-symlink paths, and Stage-O owner
name/time fields are independently required for execution.

The adapter execution closure contains only the standalone adapter script. Accepted
trainer code remains an external dependency pinned independently by its accepted HEAD,
governed bundle, and source hashes. Combining the two closures would blur which owner
decision reviewed which executable bytes.

The closure is derived from the exact adapter AST rather than a hand-maintained module
list. Import candidates are resolved against both the repository root and the script's
direct-execution directory; any repository-local module, package prefix, relative import,
or non-stdlib root fails closed. This keeps a one-file closure honest at the cost of
deliberately forbidding adjacent Python helpers and stdlib-shadowing files.

Every governed direct command additionally requires CPython's `-I -B` flags. The first
adapter statement imports only built-in `sys` and refuses before any filesystem-backed
import unless isolated mode and no-bytecode mode are both active. After startup, topology
installation explicitly removes both the adapter repository root and its `tools/`
directory before importing accepted project code. Requiring flags is less convenient for
ad-hoc invocation, but makes interpreter startup part of the exact owner-bound command
derivation instead of trusting cleanup that necessarily runs later.

After the first canonical launch object passes the source-derived runtime manifest, the
adapter records it in the same identity-retaining transactional baseline used for trainer
dependencies. This preserves the required canonical-first reuse case, while every later
setup requires the same module, class, function, sentinel, registry, and parent/bare
binding objects. Publishing this baseline only with a fully valid topology avoids treating
a failed initialization as trusted state; the stricter repeated-setup rule intentionally
refuses structurally identical replacements rather than silently accepting a new family.

The dedicated real Gate-A rehearsal is consequently invoked with pytest's
`--noconftest` collection option. It still uses pytest's built-in `tmp_path` and
`monkeypatch` fixtures, but prevents unrelated repository fixtures from importing
accepted project modules before the adapter. This isolates import ownership without
relaxing adapter validation; a rehearsal accidentally run through the broad conftest
continues to fail closed.

The tracked implementation provides a strict schema and a generator for a
`NOT_AUTHORIZED` template. A checksummed post-commit evidence template binds the final
adapter HEAD, bundle, and script hash without creating a commit-hash cycle. A future owner
may publish a separate authorization decision, but neither the implementation commit nor
its evidence authorizes adapter execution or Stage-O training.

The bounded rehearsal calls the real chain validator and real Gate A even though Gate A
also revalidates the chain. The duplicate validation costs time, but provides separate
structured chain evidence and proves the accepted trainer's own Gate-A path sees the same
one-object topology. It deliberately expects only
`authorization_status_not_authorized`; reaching any model, checkpoint restoration,
compile realization, forward, backward, or optimizer-update boundary is a test failure.

## Monitoring compatibility

Execution does not redirect, capture, buffer, or rewrite trainer output. The existing
detached monitor can therefore continue to observe sampled train loss every 20 steps,
learning rate, throughput, validation losses, checkpoint events, and process exit state.
Grad-norm logging remains unavailable and is not claimed.

## Review boundary

This implementation segment publishes no Stage-O candidate and performs no Stage-O
execution. The generated adapter template remains `NOT_AUTHORIZED`, with
`authorizes_adapter_execution=false` and `authorizes_training=false`. The next action is a
separate Fable 5.1 narrow, read-only validation of the committed adapter and checksummed
evidence.
