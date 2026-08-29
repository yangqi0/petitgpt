#!/usr/bin/env python3
"""P-PILOT-CONTRACT-V2.3 executor: the complete real pilot execution path.

Unlike the V2.2 preflight runner, this module implements the actual training code path --
model construction, canonical packed-dataset access over fixed pilot indices, Muon optimizer and
scheduler construction, the Phase-MB and Phase-Muon-LR update loops with gradient accumulation,
clipping and optimizer steps, CUDA metrics, evaluation, persistent token accounting, result
publication and deterministic selection. It never writes or reads a pilot checkpoint.

It is complete but **unexecuted in the materialization segments**: every entry into the training
path passes through :func:`validate_execution_artifacts`, which requires an external AUTHORIZED
manifest bound to this exact HEAD, contract, execution bundle, pilot-index manifest and accepted
releases, and through the exact RTX 4090 runtime gate. No tracked code change is needed for a
later authorized run: publishing the manifest is sufficient.

R2 -- validation at execution, not trust in context. Every real execution root, the parent
orchestrator and each candidate worker alike, calls the SAME artifact-validation function and
re-derives its authority from artifact bytes on disk. There is no in-memory authorization flag,
no caller-supplied hash or count, and no context object that grants anything by existing. On
top of that: a reserve-then-complete token ledger, an immutable authoritative Phase-MB report
that is the only source of Phase-LR geometry, independent recomputation of every selection
number from raw evidence, a structured subprocess terminal-result protocol, and four explicit
result classes surfaced as process exit codes.

Reuse, never reimplementation: ``src.model.GPT``, ``src.optim.build_optimizer``,
``pretrain.dataset_pretrain.PackedBinDataset`` with the manifest requirement on (which emits
the canonical production loss mask), and the shared canonical primitives in
``src.canonical_loss`` and ``src.canonical_schedule`` -- the same modules the production
trainer imports. The pilot does not import the trainer itself, which keeps the execution
closure minimal and avoids its sibling-style imports.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.pilot_contract_v2_3 import (  # noqa: E402
    ALLOWED_SCOPES,
    BASE_FINGERPRINT_SCHEMA,
    CONTRACT_VERSION,
    EFFECTIVE_BATCH_TOKENS,
    GLOBAL_PILOT_TOKEN_CEILING,
    GRAD_CLIP,
    LR_GRID_SEED1,
    LR_RUN_UPDATES,
    LR_SCORE_WEIGHTS,
    LR_WARMUP_UPDATES,
    MB_MEASURED_FIRST_UPDATE,
    MB_PROBE_UPDATES,
    MUON_LR_ARG,
    MUON_MOMENTUM,
    PHASE_CEILINGS,
    REQUIRED_NUMPY_VERSION,
    RUN_META_SCHEMA,
    SEED_SEMANTICS,
    SEQUENCES_PER_OPTIMIZER_UPDATE,
    TOKEN_LEDGER_SCHEMA,
    WEIGHT_DECAY,
    PilotContractError,
    authorization_template,
    canonical_json_bytes,
    confirmation_neighbor,
    contract_sha256,
    edge_candidate,
    frozen_grad_accum,
    generate_pilot_indices,
    lr_candidate_eligible,
    lr_confirm,
    lr_resolve_edge,
    lr_schedule,
    lr_score,
    lr_select_seed1,
    mb_candidate_grid,
    mb_lr,
    mb_select,
    require,
    require_complete_lr_grid,
    require_complete_mb_grid,
    require_numpy_version,
    require_training_authority,
    train_order,
    validate_authorization,
    verify_optimizer_state,
    verify_realized_grouping,
)

ALLOWED_UNTRACKED = ".codex_r1_manual_context_probe.py"
ALLOWED_UNTRACKED_SHA256 = "4ea1e8ef471138d9d9cf8076a6fd3bd83ce83c131287bd94c84d998b846ca76c"
PROTECTED_PREFIXES = (
    "runs/m_production_v1_2026-08-29/release",
    "runs/i_production_v1_2026-08-25",
    "runs/g_production_2026-08-21/release",
    "runs/g2_production_2026-08-21/release",
    "runs/l1_production_2026-08-20",
)
ACCEPTED_STAGE_A = "runs/m_production_v1_2026-08-29/release/stage_a"
ACCEPTED_STAGE_B = "runs/m_production_v1_2026-08-29/release/stage_b"
ACCEPTED_STAGE_A_META_SHA256 = "334564305f0b5bb058ff4fda9b2f13a9fba01046818ed3600ce2a6ca3cc5c81c"
ACCEPTED_STAGE_B_META_SHA256 = "fe634de7690a1ab56bb7e478b161eb4916724ac648ba09338b55f044fa6e0a2e"
EXECUTION_BUNDLE_SCHEMA = "petitgpt-pilot-execution-implementation-bundle-v2.3"


def repo_root() -> Path:
    return Path(ROOT).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------- execution closure


def execution_closure(root: Path | None = None) -> dict[str, Any]:
    """Derive the complete load-bearing local execution closure by AST walk from the roots."""
    import ast

    base = Path(root) if root is not None else repo_root()
    roots = ["pretrain/pilot_contract_v2_3.py", "pretrain/pilot_runner_v2_3.py"]

    def resolve(mod: str) -> list[str]:
        out, parts = [], mod.split(".")
        for i in range(1, len(parts)):
            q = base.joinpath(*parts[:i], "__init__.py")
            if q.is_file():
                out.append(str(q.relative_to(base)))
        f = base.joinpath(*parts).with_suffix(".py")
        if f.is_file():
            out.append(str(f.relative_to(base)))
        pkg = base.joinpath(*parts, "__init__.py")
        if pkg.is_file():
            out.append(str(pkg.relative_to(base)))
        return out

    seen: set[str] = set()
    graph: dict[str, list[str]] = {}
    external: set[str] = set()
    stack = list(roots)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((base / rel).read_text(encoding="utf-8"), filename=rel)
        deps: set[str] = set()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                hits = resolve(n)
                if hits:
                    deps.update(hits)
                else:
                    external.add(n.split(".")[0])
        graph[rel] = sorted(deps)
        stack.extend(d for d in deps if d not in seen)

    closure = sorted(seen)
    files = {c: file_sha256(base / c) for c in closure}
    digest = hashlib.sha256(
        canonical_json_bytes({
            "schema_version": EXECUTION_BUNDLE_SCHEMA,
            "files": dict(sorted(files.items())),
        })
    ).hexdigest()
    return {
        "bundle_schema_version": EXECUTION_BUNDLE_SCHEMA,
        "roots": roots,
        "derived_closure": closure,
        "derived_closure_count": len(closure),
        "files": files,
        "external_non_repository_modules": sorted(external),
        "local_import_graph": graph,
        "EXECUTION_IMPLEMENTATION_BUNDLE_SHA256": digest,
        "unbound_load_bearing_module_count": 0,
    }


def execution_bundle_sha256() -> str:
    return execution_closure()["EXECUTION_IMPLEMENTATION_BUNDLE_SHA256"]


# --------------------------------------------------------------------- git + paths


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root()), *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def git_policy_status() -> dict[str, Any]:
    tracked = [ln for ln in _git("status", "--porcelain", "-uno").splitlines() if ln.strip()]
    untracked = [
        ln[3:].strip() for ln in _git("status", "--porcelain").splitlines() if ln.startswith("??")
    ]
    probe = repo_root() / ALLOWED_UNTRACKED
    probe_sha = file_sha256(probe) if probe.is_file() else None
    unexpected = [u for u in untracked if u != ALLOWED_UNTRACKED]
    failures = []
    if tracked:
        failures.append("tracked_worktree_not_clean")
    if probe_sha is None:
        failures.append("allowed_historical_untracked_file_missing")
    elif probe_sha != ALLOWED_UNTRACKED_SHA256:
        failures.append("allowed_historical_untracked_file_bytes_changed")
    if unexpected:
        failures.append("uncontrolled_untracked_files_present")
    return {
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_clean": not tracked,
        "allowed_untracked_sha256": probe_sha,
        "allowed_untracked_unchanged": probe_sha == ALLOWED_UNTRACKED_SHA256,
        "unexpected_untracked": unexpected,
        "failures": failures,
        "policy_satisfied": not failures,
    }


# --------------------------------------------------------------------- release binding


def verify_accepted_release(stage: str) -> dict[str, Any]:
    """Open the canonical manifest-required release and derive its identity from disk."""
    from pretrain.dataset_pretrain import PackedBinDataset

    rel = ACCEPTED_STAGE_A if stage == "stage_a" else ACCEPTED_STAGE_B
    expected_meta = (
        ACCEPTED_STAGE_A_META_SHA256 if stage == "stage_a" else ACCEPTED_STAGE_B_META_SHA256
    )
    root = repo_root() / rel
    meta_path = root / "meta.json"
    # Release identity is a binding, not a runtime condition: a mismatch is BINDING_FAILURE.
    if not meta_path.is_file():
        raise BindingFailure(f"accepted release manifest missing: {rel}/meta.json")
    meta_sha = file_sha256(meta_path)
    if meta_sha != expected_meta:
        raise BindingFailure(
            f"{stage} meta SHA-256 mismatch: expected {expected_meta}, got {meta_sha}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset = PackedBinDataset(str(root / "train"), seq_len=2048, require_release_manifest=True)
    declared = meta["shard_files"]["train"]
    return {
        "stage": stage,
        "release_dir": rel,
        "meta_sha256": meta_sha,
        "status": meta.get("status"),
        "shard_count": len(declared),
        "shard_sha256s": [s["sha256"] for s in declared],
        "stored_token_ids": sum(s["token_count"] for s in declared),
        "blocks": len(dataset),
        "dataset": dataset,
    }


def derive_universes() -> dict[str, int]:
    """Block universes come from the accepted releases at runtime, never from the CLI."""
    return {
        "stage_a_blocks": verify_accepted_release("stage_a")["blocks"],
        "stage_b_blocks": verify_accepted_release("stage_b")["blocks"],
    }


# --------------------------------------------------------------------- fingerprint


def base_runtime_fingerprint(*, gpu_required: bool = False) -> dict[str, Any]:
    """Base runtime identity. Carries no per-run configuration (no compile, LR, seed, phase)."""
    import numpy as np
    import tokenizers
    import torch

    gpu: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu.update({
            "name": torch.cuda.get_device_name(0),
            "total_vram_mib": int(props.total_memory // (1024 * 1024)),
            "total_vram_bytes": int(props.total_memory),
            "capability": f"{props.major}.{props.minor}",
            "driver": _nvidia_smi("driver_version"),
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
    if gpu_required:
        require_training_authority(gpu)
    git = git_policy_status()
    fp = {
        "schema_version": BASE_FINGERPRINT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "torch_build": {"cuda": torch.version.cuda, "git_version": torch.version.git_version},
        "numpy_version": np.__version__,
        "required_numpy_version": REQUIRED_NUMPY_VERSION,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "container_template": os.environ.get("RUNPOD_POD_ID") or None,
        "repository": {
            "branch": git["branch"],
            "head": git["head"],
            "tracked_clean": git["tracked_clean"],
            "allowed_untracked_sha256": git["allowed_untracked_sha256"],
            "allowed_untracked_unchanged": git["allowed_untracked_unchanged"],
            "unexpected_untracked": git["unexpected_untracked"],
        },
        "contract_sha256": contract_sha256(),
        "execution_implementation_bundle_sha256": execution_bundle_sha256(),
    }
    for forbidden in ("compile", "micro_bsz", "grad_accum", "peak_lr", "seed", "phase"):
        require(forbidden not in fp, f"per-run field {forbidden!r} must not be in the fingerprint")
    fp["fingerprint_sha256"] = hashlib.sha256(canonical_json_bytes(fp)).hexdigest()
    return fp


def _nvidia_smi(field: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        ).stdout
        return out.strip().splitlines()[0].strip() if out.strip() else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------- result classes

SUCCESS = 0
CANDIDATE_INELIGIBLE = 3
PHASE_ABORT = 4
BINDING_FAILURE = 5

RESULT_CLASSES = MappingProxyType({
    "SUCCESS": SUCCESS,
    "CANDIDATE_INELIGIBLE": CANDIDATE_INELIGIBLE,
    "PHASE_ABORT": PHASE_ABORT,
    "BINDING_FAILURE": BINDING_FAILURE,
})


class BindingFailure(PilotContractError):
    """A phase-level binding failure. Never downgraded to an ineligible candidate."""


class PhaseAbort(PilotContractError):
    """The phase cannot continue under the frozen contract."""


class CandidateFailure(RuntimeError):
    """A candidate-local failure. The parent may continue the required grid."""


# --------------------------------------------------------------------- output root


def validate_authorized_output_root(root: Path) -> Path:
    """Validate the authorized root BEFORE anything is written under it.

    R2: called before the ledger, the session metadata, any candidate directory and any
    evidence file exists. Nothing is created until this returns.
    """
    repo = repo_root()
    resolved = Path(root).resolve()
    for prefix in PROTECTED_PREFIXES:
        protected = (repo / prefix).resolve()
        require(resolved != protected, f"authorized root is an accepted release: {resolved}")
        require(
            protected not in resolved.parents,
            f"authorized root is inside an accepted release: {resolved}",
        )
        require(
            resolved not in protected.parents,
            f"authorized root contains an accepted release: {resolved}",
        )
    require(
        resolved.parent.exists(), f"authorized root parent must already exist: {resolved.parent}"
    )
    return resolved


def require_candidate_output_dir(destination: Path, authorized_root: Path) -> Path:
    """Resolved containment beneath the already-validated authorized root."""
    authorized = Path(authorized_root).resolve()
    resolved = Path(destination).resolve()
    require(
        resolved != authorized and authorized in resolved.parents,
        f"candidate output must resolve beneath the authorized root {authorized}: {resolved}",
    )
    require(not resolved.exists(), f"candidate output directory must not exist: {resolved}")
    return resolved


# --------------------------------------------------------------------- token ledger

LEDGER_FILENAME = "token_ledger.json"
LEDGER_LOCK_FILENAME = "token_ledger.lock"
SESSION_FILENAME = "session.json"


class TokenLedger:
    """Persistent trained-token accounting with a conservative reserve-then-complete protocol.

    R2: a reservation is taken and persisted BEFORE the optimizer update is applied, and moved
    to completed only after it succeeds. A process that dies between the two leaves the
    reservation consumed on purpose: budget is never silently returned, so a crash can never
    result in an uncounted optimizer update.

    Every lock-held operation reloads the state from disk and revalidates the complete identity
    binding, so a ledger belonging to a different authorization, session, scope, contract, HEAD,
    bundle, index manifest or output root is refused rather than reused.
    """

    IDENTITY_FIELDS = (
        "contract_sha256",
        "implementation_head",
        "execution_bundle_sha256",
        "pilot_index_manifest_file_sha256",
        "authorization_sha256",
        "session_id",
        "authorized_output_root",
        "authorized_scope",
    )

    def __init__(
        self, path: Path, identity: Mapping[str, Any], effective_ceilings: Mapping[str, int]
    ):
        missing = [f for f in self.IDENTITY_FIELDS if f not in identity]
        require(not missing, f"ledger identity is incomplete: {missing}")
        self.path = Path(path)
        self.lock_path = self.path.parent / LEDGER_LOCK_FILENAME
        self.identity = {f: identity[f] for f in self.IDENTITY_FIELDS}
        self.effective_ceilings = {k: int(v) for k, v in effective_ceilings.items()}
        if self.path.is_file():
            self.state = self._validated(json.loads(self.path.read_text(encoding="utf-8")))
        else:
            self.state = {
                "schema_version": TOKEN_LEDGER_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "identity": dict(self.identity),
                "effective_ceilings": dict(self.effective_ceilings),
                "reserved_tokens": {"MB": 0, "LR": 0, "GLOBAL": 0},
                "completed_tokens": {"MB": 0, "LR": 0, "GLOBAL": 0},
                "reserved_updates": 0,
                "completed_updates": 0,
            }
            self._write(self.state)

    def _validated(self, state: Mapping[str, Any]) -> dict[str, Any]:
        require(state.get("schema_version") == TOKEN_LEDGER_SCHEMA, "token ledger schema mismatch")
        require(
            state.get("contract_version") == CONTRACT_VERSION,
            "token ledger contract version mismatch",
        )
        stored = state.get("identity") or {}
        mismatched = [k for k, v in self.identity.items() if stored.get(k) != v]
        require(
            not mismatched,
            "token ledger identity does not bind this execution; a different authorization "
            f"or session may not reuse it: {sorted(mismatched)}",
        )
        for bucket in ("reserved_tokens", "completed_tokens"):
            values = state.get(bucket) or {}
            require(
                all(int(values.get(k, 0)) >= 0 for k in ("MB", "LR", "GLOBAL")),
                f"token ledger {bucket} is invalid",
            )
        return dict(state)

    def _write(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            handle.write(canonical_json_bytes(state))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def _lock(self):
        from contextlib import contextmanager
        import fcntl

        @contextmanager
        def guard():
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_path, "w") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return guard()

    def _reload_locked(self) -> None:
        if self.path.is_file():
            self.state = self._validated(json.loads(self.path.read_text(encoding="utf-8")))

    def effective_ceiling(self, key: str) -> int:
        frozen = PHASE_CEILINGS[key] if key in PHASE_CEILINGS else GLOBAL_PILOT_TOKEN_CEILING
        return min(int(frozen), int(self.effective_ceilings[key]))

    def reserve(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> dict[str, Any]:
        """Step one: reserve BEFORE the optimizer update is applied."""
        require(phase in PHASE_CEILINGS, f"unknown phase {phase!r}")
        with self._lock():
            self._reload_locked()
            phase_after = int(self.state["reserved_tokens"].get(phase, 0)) + int(tokens)
            global_after = int(self.state["reserved_tokens"].get("GLOBAL", 0)) + int(tokens)
            breaches = []
            if phase_after > self.effective_ceiling(phase):
                breaches.append(f"phase_{phase}_token_ceiling")
            if global_after > self.effective_ceiling("GLOBAL"):
                breaches.append("global_pilot_token_ceiling")
            require(
                not breaches,
                "PILOT_ABORT: trained-token ceiling would be exceeded: " + ", ".join(breaches),
            )
            self.state["reserved_tokens"][phase] = phase_after
            self.state["reserved_tokens"]["GLOBAL"] = global_after
            self.state["reserved_updates"] = int(self.state["reserved_updates"]) + 1
            self._write(self.state)
            return {
                "phase": phase,
                "reserved_tokens": tokens,
                "phase_reserved_after": phase_after,
                "global_reserved_after": global_after,
            }

    def complete(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> dict[str, Any]:
        """Step two: move an existing reservation to completed, after the update succeeded."""
        require(phase in PHASE_CEILINGS, f"unknown phase {phase!r}")
        with self._lock():
            self._reload_locked()
            completed_after = int(self.state["completed_tokens"].get(phase, 0)) + int(tokens)
            require(
                completed_after <= int(self.state["reserved_tokens"].get(phase, 0)),
                "completed tokens may never exceed reserved tokens; "
                "reserve must precede the optimizer update",
            )
            self.state["completed_tokens"][phase] = completed_after
            self.state["completed_tokens"]["GLOBAL"] = int(
                self.state["completed_tokens"].get("GLOBAL", 0)
            ) + int(tokens)
            self.state["completed_updates"] = int(self.state["completed_updates"]) + 1
            self._write(self.state)
            return {"phase": phase, "completed_tokens_after": completed_after}

    def snapshot(self) -> dict[str, Any]:
        return {
            "reserved_tokens": dict(self.state["reserved_tokens"]),
            "completed_tokens": dict(self.state["completed_tokens"]),
            "reserved_updates": int(self.state["reserved_updates"]),
            "completed_updates": int(self.state["completed_updates"]),
            "identity": dict(self.identity),
        }


def authorized_effective_ceilings(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Effective ceiling = min(frozen contract ceiling, authorized ceiling)."""
    authorized = manifest.get("pilot_trained_token_ceiling")
    require(
        isinstance(authorized, int) and not isinstance(authorized, bool),
        f"authorized token ceiling must be an integer, got {authorized!r}",
    )
    require(authorized > 0, f"authorized token ceiling must be > 0, got {authorized}")
    require(
        authorized <= GLOBAL_PILOT_TOKEN_CEILING,
        f"authorized token ceiling {authorized} exceeds the frozen global ceiling",
    )
    return {
        "MB": min(PHASE_CEILINGS["MB"], authorized),
        "LR": min(PHASE_CEILINGS["LR"], authorized),
        "GLOBAL": min(GLOBAL_PILOT_TOKEN_CEILING, authorized),
    }


# --------------------------------------------------------------------- artifact validation

PILOT_INDEX_MANIFEST_FILENAME = "PILOT_INDICES.json"


def _sha256_file(path: Path) -> str:
    return file_sha256(Path(path))


def validate_execution_artifacts(
    *,
    authorization_path: Path,
    candidate_spec_path: Path | None,
    pilot_index_manifest_path: Path,
    output_dir: Path,
    requested_phase: str,
    gpu_required: bool = True,
) -> dict[str, Any]:
    """THE single gate every path to the model-training backend passes through.

    R2: this validates from canonical artifact BYTES on disk. It trusts no in-memory object,
    no caller-supplied hash, no caller-supplied release count, no ``authorized`` flag and no
    previously constructed context. Every real execution root -- the parent orchestrator and
    each child worker independently -- calls exactly this function.
    """
    require(requested_phase in ("MB", "LR"), f"unknown phase {requested_phase!r}")
    require_numpy_version()

    # --- authorization manifest, from disk ---
    auth_path = Path(authorization_path)
    if not auth_path.is_file():
        raise BindingFailure(
            f"authorization manifest not found at {auth_path}; pilot execution is "
            f"NOT_AUTHORIZED under {CONTRACT_VERSION}"
        )
    auth_bytes = auth_path.read_bytes()
    authorization_sha256 = hashlib.sha256(auth_bytes).hexdigest()
    manifest = json.loads(auth_bytes.decode("utf-8"))

    # --- candidate spec, from disk, immutable (present for a worker, absent for the parent) ---
    spec: dict[str, Any] | None = None
    spec_sha256: str | None = None
    candidate: dict[str, Any] = {}
    if candidate_spec_path is not None:
        spec_path = Path(candidate_spec_path)
        if not spec_path.is_file():
            raise BindingFailure(f"candidate spec not found at {spec_path}")
        spec_bytes = spec_path.read_bytes()
        spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
        spec = json.loads(spec_bytes.decode("utf-8"))
        candidate = dict(spec.get("candidate") or {})
        if candidate.get("phase") != requested_phase:
            raise BindingFailure(
                f"candidate spec phase {candidate.get('phase')!r} != requested {requested_phase!r}"
            )

    # --- pilot index manifest FILE sha, computed from disk (never taken from the manifest) ---
    index_path = Path(pilot_index_manifest_path)
    if not index_path.is_file():
        raise BindingFailure(f"pilot index manifest not found at {index_path}")
    observed_index_file_sha = _sha256_file(index_path)
    index_manifest = json.loads(index_path.read_text(encoding="utf-8"))

    # --- accepted releases, opened and identified from disk ---
    stage_a = verify_accepted_release("stage_a")
    stage_b = verify_accepted_release("stage_b")

    # --- independently regenerate the indices and check the serialized list digests ---
    indices = generate_pilot_indices(
        stage_a_blocks=stage_a["blocks"], stage_b_blocks=stage_b["blocks"]
    )
    for key in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256"):
        if index_manifest.get(key) != indices[key]:
            raise BindingFailure(
                f"pilot index manifest {key} does not match the regenerated indices"
            )
    serialized_index_lists_digest = hashlib.sha256(
        canonical_json_bytes({
            k: indices[k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        })
    ).hexdigest()
    if index_manifest.get("serialized_index_lists_digest") != serialized_index_lists_digest:
        raise BindingFailure(
            "pilot index manifest serialized_index_lists_digest does not match the digest "
            "recomputed from the regenerated index lists"
        )

    git = git_policy_status()
    if not git["policy_satisfied"]:
        raise BindingFailure("pre-launch Git policy not satisfied: " + ", ".join(git["failures"]))

    authorized_root = validate_authorized_output_root(Path(output_dir))
    observed = {
        "branch": git["branch"],
        "head": git["head"],
        "contract_sha256": contract_sha256(),
        "execution_bundle_sha256": execution_bundle_sha256(),
        "serialized_index_lists_digest": serialized_index_lists_digest,
        "pilot_index_manifest_file_sha256": observed_index_file_sha,
        "stage_a_meta_sha256": stage_a["meta_sha256"],
        "stage_b_meta_sha256": stage_b["meta_sha256"],
        "output_root": str(authorized_root),
    }
    verdict = validate_authorization(
        manifest, requested_scope=_scope_for(requested_phase), observed=observed
    )
    if not verdict["authorized"]:
        raise BindingFailure(
            f"pilot execution refused under {CONTRACT_VERSION}: " + ", ".join(verdict["failures"])
        )

    scope = str(manifest["allowed_scope"])
    if scope == "PHASE_MB_ONLY" and requested_phase != "MB":
        raise BindingFailure(
            "scope PHASE_MB_ONLY executes one complete Phase-MB session only and can never be "
            "promoted or reused for Phase LR; a new FULL_V2_3_PILOT authorization is required"
        )

    fingerprint = base_runtime_fingerprint(gpu_required=gpu_required)
    ceilings = authorized_effective_ceilings(manifest)
    session_id = hashlib.sha256(
        canonical_json_bytes({
            "authorization_sha256": authorization_sha256,
            "scope": scope,
            "output_root": str(authorized_root),
            "contract_sha256": observed["contract_sha256"],
            "execution_bundle_sha256": observed["execution_bundle_sha256"],
        })
    ).hexdigest()
    identity = {
        "contract_sha256": observed["contract_sha256"],
        "implementation_head": observed["head"],
        "execution_bundle_sha256": observed["execution_bundle_sha256"],
        "pilot_index_manifest_file_sha256": observed_index_file_sha,
        "authorization_sha256": authorization_sha256,
        "session_id": session_id,
        "authorized_output_root": str(authorized_root),
        "authorized_scope": scope,
    }
    return {
        "manifest": manifest,
        "authorization_sha256": authorization_sha256,
        "authorization_path": str(auth_path),
        "spec": spec,
        "spec_path": (str(candidate_spec_path) if candidate_spec_path is not None else None),
        "spec_sha256": spec_sha256,
        "candidate": candidate,
        "observed": observed,
        "indices": indices,
        "index_manifest": index_manifest,
        "index_manifest_path": str(index_path),
        "serialized_index_lists_digest": serialized_index_lists_digest,
        "pilot_index_manifest_file_sha256": observed_index_file_sha,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "fingerprint": fingerprint,
        "effective_ceilings": ceilings,
        "identity": identity,
        "session_id": session_id,
        "scope": scope,
        "authorized_root": authorized_root,
        "phase": requested_phase,
    }


def _scope_for(phase: str) -> str:
    return "PHASE_MB_ONLY" if phase == "MB" else "FULL_V2_3_PILOT"


class ExecutionSession:
    """Internal convenience over a validated artifact set. NOT an authorization boundary.

    R2: constructing this object grants nothing. Every backend re-derives its authority by
    calling :func:`validate_execution_artifacts` on artifact bytes; this type only carries the
    already-validated result so the orchestrator does not re-read the same files repeatedly.
    """

    __slots__ = ("validated", "ledger", "train_order_by_seed")

    def __init__(self, validated: Mapping[str, Any], ledger: TokenLedger):
        self.validated = dict(validated)
        self.ledger = ledger
        self.train_order_by_seed = {
            label: train_order(validated["indices"]["stage_a_train"], spec["train_order"])
            for label, spec in SEED_SEMANTICS.items()
        }

    @property
    def output_root(self) -> Path:
        return Path(self.validated["authorized_root"])

    @property
    def scope(self) -> str:
        return str(self.validated["scope"])

    @property
    def session_id(self) -> str:
        return str(self.validated["session_id"])


def open_session(
    *,
    authorization_path: Path,
    candidate_spec_path: Path,
    pilot_index_manifest_path: Path,
    output_dir: Path,
    phase: str,
    gpu_required: bool = True,
) -> ExecutionSession:
    """Validate artifacts, then build the ledger AFTER the root has been validated."""
    validated = validate_execution_artifacts(
        authorization_path=authorization_path,
        candidate_spec_path=candidate_spec_path,
        pilot_index_manifest_path=pilot_index_manifest_path,
        output_dir=output_dir,
        requested_phase=phase,
        gpu_required=gpu_required,
    )
    root = validated["authorized_root"]
    root.mkdir(parents=True, exist_ok=True)
    session_path = root / SESSION_FILENAME
    session_doc = {
        "session_id": validated["session_id"],
        "scope": validated["scope"],
        "authorization_sha256": validated["authorization_sha256"],
        "identity": validated["identity"],
    }
    if session_path.is_file():
        existing = json.loads(session_path.read_text(encoding="utf-8"))
        if existing.get("session_id") != validated["session_id"]:
            raise BindingFailure(
                "this output root already belongs to a different session; a new authorization "
                "implies a new session and a new ledger"
            )
    else:
        tmp = session_path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            handle.write(canonical_json_bytes(session_doc))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, session_path)
    ledger = TokenLedger(
        root / LEDGER_FILENAME, validated["identity"], validated["effective_ceilings"]
    )
    return ExecutionSession(validated, ledger)


# --------------------------------------------------------------------- construction


def build_pilot_model(seed: int) -> Any:
    """The frozen 124,635,456-parameter model at a fixed init seed."""
    import torch

    from src.model import GPT, GPTConfig, audit_gpt_parameter_count

    torch.manual_seed(int(seed))
    cfg = GPTConfig()
    model = GPT(cfg)
    audit_gpt_parameter_count(model, cfg)
    return model


def build_pilot_optimizer(model: Any, peak_lr: float) -> Any:
    """Muon, exactly as V2.3 freezes it, through the canonical builder."""
    from src.optim import build_optimizer

    opt = build_optimizer(
        model,
        name="muon",
        lr=float(peak_lr),
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        muon_lr=MUON_LR_ARG,
        muon_momentum=MUON_MOMENTUM,
        verbose=False,
    )
    grouping = verify_realized_grouping(opt, model)
    if not grouping["matches_frozen_realization"]:
        raise PhaseAbort(
            "realized optimizer grouping does not match V2.3: " + ", ".join(grouping["failures"])
        )
    return opt


def apply_scheduled_lr(optimizer: Any, lr: float) -> list[float]:
    """One scalar schedule drives all groups via lr_ratio (all 1.0 under V2.3)."""
    realized = []
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr) * float(pg.get("lr_ratio", 1.0))
        realized.append(pg["lr"])
    return realized


# R2 Part 13: the Moonlight RMS-matching rule the frozen Muon policy depends on. These shapes
# are the real matrix shapes the canonical 30x576 model presents to the Muon groups plus two
# deliberately non-square controls, so the check cannot pass by coincidence on square weights.
RMS_MATCHING_SHAPE_CASES = ((576, 576), (960, 576), (1536, 576), (576, 1536), (64, 4096))
RMS_MATCHING_CONSTANT = 0.2


def expected_rms_matched_lr(lr: float, fan_out: int, fan_in: int) -> float:
    """``adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))`` -- the frozen scaling rule."""
    import math

    return float(lr) * RMS_MATCHING_CONSTANT * math.sqrt(max(int(fan_in), int(fan_out)))


def verify_rms_matching(
    *,
    lr: float = 3e-4,
    shapes: Sequence[tuple[int, int]] = RMS_MATCHING_SHAPE_CASES,
    weight_decay: float = WEIGHT_DECAY,
) -> dict[str, Any]:
    """Prove the realized Muon update applies the RMS-matched LR, per deterministic shape case.

    R2: this does not read the source or trust a comment. For each shape it builds a real
    parameter and a real gradient, runs ONE real Muon step through the canonical optimizer, and
    reconstructs the closed form

        p_after = p_before * (1 - lr * weight_decay) - adjusted_lr * NS5(momentum_buffer)

    with ``adjusted_lr`` taken from :func:`expected_rms_matched_lr`. Agreement to float
    tolerance is the evidence; a realization that ignored ``max(fan_in, fan_out)`` (or used the
    unscaled ``lr``) fails on the non-square cases by orders of magnitude.
    """
    import torch

    from src.optim import Muon, zeropower_via_newtonschulz5

    rng_state = torch.get_rng_state()
    cases = []
    for fan_out, fan_in in shapes:
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(int(fan_out), int(fan_in), dtype=torch.float32))
        before = p.detach().clone()
        p.grad = torch.randn_like(p)
        opt = Muon([
            {
                "params": [p],
                "use_muon": True,
                "lr": float(lr),
                "momentum": MUON_MOMENTUM,
                "nesterov": True,
                "ns_steps": 5,
                "weight_decay": float(weight_decay),
                "lr_ratio": 1.0,
            }
        ])
        buf = torch.zeros_like(p.grad)
        buf.mul_(MUON_MOMENTUM).add_(p.grad)
        direction = zeropower_via_newtonschulz5(p.grad.add(buf, alpha=MUON_MOMENTUM), steps=5)
        adjusted = expected_rms_matched_lr(lr, fan_out, fan_in)
        opt.step()
        predicted = before * (1.0 - float(lr) * float(weight_decay)) - adjusted * direction
        realized_delta = p.detach() - before * (1.0 - float(lr) * float(weight_decay))
        scale = float(direction.abs().max())
        error = float((p.detach() - predicted).abs().max())
        # An implementation using the raw lr instead of the RMS-matched lr would differ by
        # (adjusted - lr) * |direction|; record that margin so the case is not vacuous.
        unscaled_gap = float(abs(adjusted - float(lr)) * scale)
        cases.append({
            "fan_out": int(fan_out),
            "fan_in": int(fan_in),
            "expected_adjusted_lr": adjusted,
            "sqrt_max_fan": adjusted / (float(lr) * RMS_MATCHING_CONSTANT),
            "max_abs_error": error,
            "realized_update_max_abs": float(realized_delta.abs().max()),
            "unscaled_lr_would_differ_by": unscaled_gap,
            "rms_matching_factor": adjusted / float(lr),
            "matches_rms_matching_rule": error <= 1e-5 * max(1.0, adjusted * scale),
            # Discriminating means the RMS-matched update is materially different from the
            # unscaled one AND that difference is far above the float noise floor, so the case
            # cannot pass for an implementation that ignored max(fan_in, fan_out).
            "case_is_discriminating": (
                adjusted / float(lr) >= 2.0 and unscaled_gap > 20.0 * max(error, 1e-12)
            ),
        })
    torch.set_rng_state(rng_state)
    return {
        "rule": "adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))",
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "cases": cases,
        "all_cases_match": all(c["matches_rms_matching_rule"] for c in cases),
        "all_cases_discriminating": all(c["case_is_discriminating"] for c in cases),
    }


class IndexView:
    """Fixed-index view over a canonical PackedBinDataset, consumed sequentially without wrap."""

    def __init__(self, dataset: Any, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        n = len(dataset)
        bad = [i for i in self.indices if not (0 <= i < n)]
        require(not bad, f"pilot index out of range for a {n}-block release: {bad[:4]}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Any:
        position = int(position)
        require(
            0 <= position < len(self.indices),
            f"pilot train order exhausted at position {position}; no replay or wrap is "
            f"permitted (have {len(self.indices)} blocks)",
        )
        return self.dataset[self.indices[position]]


def canonical_loss_components(logits: Any, labels: Any, loss_mask: Any) -> Any:
    """Weighted CE numerator and effective weight, from the SHARED canonical primitive.

    Imports ``src.canonical_loss`` -- the same module the production trainer imports -- rather
    than the whole trainer. Returning the two components (not a normalized value) is what lets
    evaluation accumulate a correct GLOBAL token mean.
    """
    from src.canonical_loss import masked_weighted_ce_components
    from src.special_tokens import EOS_ID

    return masked_weighted_ce_components(
        logits.float(), labels.long(), loss_mask, eos_id=EOS_ID, eos_weight=1.0
    )


def canonical_loss(logits: Any, labels: Any, loss_mask: Any) -> Any:
    """Normalized token-mean CE for a single batch (training path)."""
    numerator, weight = canonical_loss_components(logits, labels, loss_mask)
    return numerator / weight.clamp_min(1.0)


def to_model_ids(stacked: Any, device: str) -> Any:
    """Canonical packed storage surfaces as int16; production feeds torch.long index tensors."""
    import torch

    return stacked.to(device=device, dtype=torch.long)


# --------------------------------------------------------------------- the real loops


def _run_updates(
    *,
    model: Any,
    optimizer: Any,
    view: IndexView,
    micro_bsz: int,
    grad_accum: int,
    updates: int,
    lr_fn: Any,
    ledger: TokenLedger,
    phase: str,
    device: str,
    timed_from: int = 1,
    record_diagnostics: bool = False,
) -> dict[str, Any]:
    """The complete optimizer-update loop: accumulation, clipping, step, timing, ledger.

    R2 Part 5: the ledger reservation is taken BEFORE the update is applied and completed only
    after the optimizer step returns, so a crash mid-update can never leave an applied update
    unaccounted for.

    R2 Part 12: no device-to-host scalar transfer happens inside a timed region. Losses and
    gradient norms are retained as device tensors and converted after the loop, so the recorded
    per-update wall time measures the training step and nothing else.
    """
    import torch

    required_blocks = int(updates) * int(SEQUENCES_PER_OPTIMIZER_UPDATE)
    require(
        len(view) >= required_blocks,
        f"{phase}: need {required_blocks} train blocks without replay, view has {len(view)}",
    )

    model.train()
    loss_tensors: dict[int, Any] = {}
    grad_norm_tensors: dict[int, Any] = {}
    realized_lrs: dict[int, list[float]] = {}
    per_update_seconds: dict[int, float] = {}
    diagnostics: list[dict[str, Any]] = []
    cursor = 0
    for u in range(1, int(updates) + 1):
        lr = lr_fn(u)
        realized = apply_scheduled_lr(optimizer, lr)
        ledger.reserve(phase)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated = None
        for _ in range(int(grad_accum)):
            batch = [view[cursor + k] for k in range(int(micro_bsz))]
            cursor += int(micro_bsz)
            x = to_model_ids(torch.stack([b[0] for b in batch]), device)
            y = to_model_ids(torch.stack([b[1] for b in batch]), device)
            m = torch.stack([b[2] for b in batch]).to(device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = canonical_loss(model(x), y, m) / int(grad_accum)
            loss.backward()
            detached = loss.detach()
            accumulated = detached if accumulated is None else accumulated + detached
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        per_update_seconds[u] = time.perf_counter() - t0
        ledger.complete(phase)
        loss_tensors[u] = accumulated
        grad_norm_tensors[u] = gnorm.detach() if hasattr(gnorm, "detach") else gnorm
        realized_lrs[u] = realized
        if record_diagnostics:
            diagnostics.append(_group_diagnostics(optimizer, u))
    require(
        cursor == required_blocks,
        f"{phase}: consumed {cursor} blocks, contract requires exactly {required_blocks}",
    )

    losses = {u: float(t) for u, t in loss_tensors.items()}
    grad_norms = {u: float(t) for u, t in grad_norm_tensors.items()}
    measured = [per_update_seconds[u] for u in sorted(per_update_seconds) if u >= int(timed_from)]
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "realized_lrs": realized_lrs,
        "per_update_wall_seconds": {str(u): per_update_seconds[u] for u in per_update_seconds},
        "measured_update_wall_seconds": measured,
        "first_optimizer_update_wall_seconds": per_update_seconds.get(1),
        "measured_first_update": int(timed_from),
        "diagnostics": diagnostics,
        "completed_updates": int(updates),
        "blocks_consumed": cursor,
    }


def _group_diagnostics(optimizer: Any, update: int) -> dict[str, Any]:
    """Per-group update/weight RMS. Recorded, never used as a selection threshold."""
    import torch

    out = []
    for gi, g in enumerate(optimizer.param_groups):
        weights = [p for p in g["params"] if p.numel()]
        if not weights:
            continue
        wrms = float(torch.sqrt(sum((p.float() ** 2).mean() for p in weights) / len(weights)))
        key = "momentum_buffer" if g.get("use_muon") else "exp_avg"
        bufs = [optimizer.state[p][key] for p in weights if key in optimizer.state.get(p, {})]
        urms = (
            float(torch.sqrt(sum((b.float() ** 2).mean() for b in bufs) / len(bufs)))
            if bufs
            else None
        )
        out.append({
            "group_index": gi,
            "use_muon": bool(g.get("use_muon")),
            "weight_rms": wrms,
            "update_rms": urms,
            "lr": float(g["lr"]),
        })
    return {"update": update, "groups": out}


def evaluate(model: Any, view: IndexView, *, micro_bsz: int, device: str) -> dict[str, float]:
    """GLOBAL token-mean CE: return the raw numerator and weight, and the single division.

    R2 Part 7: the raw components are what the result artifact carries, so the parent can
    recompute the loss and the SCORE instead of trusting a serialized scalar.
    """
    import torch

    model.eval()
    total_numerator, total_weight = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(view), int(micro_bsz)):
            batch = [view[i] for i in range(start, min(start + int(micro_bsz), len(view)))]
            x = to_model_ids(torch.stack([b[0] for b in batch]), device)
            y = to_model_ids(torch.stack([b[1] for b in batch]), device)
            m = torch.stack([b[2] for b in batch]).to(device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                numerator, weight = canonical_loss_components(model(x), y, m)
            total_numerator += float(numerator)
            total_weight += float(weight)
    model.train()
    return {
        "numerator": total_numerator,
        "weight": total_weight,
        "loss": total_numerator / max(1.0, total_weight),
        "blocks": len(view),
    }


def global_token_mean(components: Sequence[tuple[float, float]]) -> float:
    """Reference for the accumulation rule: sum numerators, sum weights, divide once."""
    n = sum(float(a) for a, _ in components)
    w = sum(float(b) for _, b in components)
    return n / max(1.0, w)


# --------------------------------------------------------------------- compile-path evidence


def compile_path_evidence(model: Any, *, requested: bool, cache_dir: Path) -> dict[str, Any]:
    """Observed evidence that the REQUESTED compile mode was the one actually realized.

    R2 Part 12: not a self-reported Boolean copied from the candidate spec. The evidence is
    the realized module type, TorchDynamo's own graph counters and the Inductor artifacts the
    run left on disk. ``compile=on`` that silently fell back to eager produces zero graphs and
    an empty cache directory and is therefore rejected; ``compile=off`` that somehow ran through
    Dynamo is rejected too.
    """
    import torch

    module_type = f"{type(model).__module__}.{type(model).__qualname__}"
    is_optimized_module = "OptimizedModule" in module_type
    unique_graphs, graph_breaks, counters = 0, 0, {}
    try:
        raw = torch._dynamo.utils.counters
        counters = {k: dict(v) for k, v in raw.items()}
        unique_graphs = int(counters.get("stats", {}).get("unique_graphs", 0))
        graph_breaks = int(sum(int(v) for v in counters.get("graph_break", {}).values()))
    except Exception:  # noqa: BLE001 - counters are diagnostics, never a hard dependency
        counters = {}
    cache = Path(cache_dir)
    artifacts = [p for p in cache.rglob("*") if p.is_file()] if cache.is_dir() else []
    observed_compiled = bool(is_optimized_module and (unique_graphs > 0 or artifacts))
    return {
        "compile_requested": bool(requested),
        "compiled_module_type": module_type,
        "compiled_module_is_optimized_module": is_optimized_module,
        "dynamo_unique_graphs": unique_graphs,
        "dynamo_graph_breaks": graph_breaks,
        "dynamo_counters": counters,
        "inductor_cache_dir": str(cache),
        "inductor_artifact_count": len(artifacts),
        "observed_compiled_execution": observed_compiled,
        # True only when the realized path equals the requested path, in BOTH directions.
        "canonical_compile_path": (
            observed_compiled if requested else (not is_optimized_module and unique_graphs == 0)
        ),
    }


def _reset_dynamo_counters() -> None:
    import torch

    try:
        torch._dynamo.utils.counters.clear()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------- candidate backends


def run_phase_mb_candidate(
    candidate: Mapping[str, Any], session: ExecutionSession
) -> dict[str, Any]:
    """One Phase-MB probe, end to end, inside a validated candidate subprocess."""
    import torch

    require(
        isinstance(session, ExecutionSession),
        "candidate execution requires a validated ExecutionSession",
    )
    validated = session.validated
    require(str(candidate.get("phase")) == "MB", "Phase-MB backend received a non-MB candidate")
    out = require_candidate_output_dir(Path(candidate["output_dir"]), session.output_root)
    out.mkdir(parents=True)
    cache_dir = out / "inductor_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    _reset_dynamo_counters()
    torch.cuda.reset_peak_memory_stats()

    model = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    torch_compile_wrapper_seconds = None
    if candidate["compile"]:
        t0 = time.perf_counter()
        model = torch.compile(model)
        torch_compile_wrapper_seconds = time.perf_counter() - t0
    optimizer = build_pilot_optimizer(model, candidate["peak_lr"])
    view = session.train_view(validated["stage_a"]["dataset"], candidate["seed_label"])

    result = _run_updates(
        model=model,
        optimizer=optimizer,
        view=view,
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=MB_PROBE_UPDATES,
        lr_fn=mb_lr,
        ledger=session.ledger,
        phase="MB",
        device="cuda",
        timed_from=MB_MEASURED_FIRST_UPDATE,
    )
    grouping = verify_realized_grouping(optimizer, model)
    states = verify_optimizer_state(optimizer)
    compile_evidence = compile_path_evidence(
        model, requested=bool(candidate["compile"]), cache_dir=cache_dir
    )
    payload = {
        "phase": "MB",
        "candidate_id": candidate["candidate_id"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": bool(candidate["compile"]),
        "seed_label": candidate["seed_label"],
        "completed_updates": result["completed_updates"],
        "blocks_consumed": result["blocks_consumed"],
        # R2 Part 7: the raw per-update timings the parent recomputes the median from. The
        # serialized median below is a convenience the parent independently re-derives.
        "measured_update_wall_seconds": result["measured_update_wall_seconds"],
        "measured_first_update": result["measured_first_update"],
        "per_update_wall_seconds": result["per_update_wall_seconds"],
        "median_tokens_per_sec": median_tokens_per_sec(result["measured_update_wall_seconds"]),
        # R2 Part 12: three separately named, separately measured quantities.
        "torch_compile_wrapper_seconds": torch_compile_wrapper_seconds,
        "first_optimizer_update_wall_seconds": result["first_optimizer_update_wall_seconds"],
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "oom": False,
        "uncontrolled_exception": False,
        "all_losses_finite": all(map(_finite, result["losses"].values())),
        "all_grad_norms_finite": all(map(_finite, result["grad_norms"].values())),
        "all_optimizer_states_instantiated": states["all_states_instantiated"],
        "grouping_matches_contract": grouping["matches_frozen_realization"],
        "all_lr_ratios_are_one": grouping["all_lr_ratios_are_one"],
        "compile_evidence": compile_evidence,
        "canonical_compile_path": compile_evidence["canonical_compile_path"],
        "rms_matching": verify_rms_matching(lr=float(candidate["peak_lr"])),
    }
    return _publish_result(out, payload, candidate, session)


def run_phase_lr_candidate(
    candidate: Mapping[str, Any], session: ExecutionSession
) -> dict[str, Any]:
    """One Phase-Muon-LR run, end to end, including both global-mean evaluations."""
    import torch

    from pretrain.pilot_contract_v2_3 import sustained_divergence

    require(
        isinstance(session, ExecutionSession),
        "candidate execution requires a validated ExecutionSession",
    )
    validated = session.validated
    require(str(candidate.get("phase")) == "LR", "Phase-LR backend received a non-LR candidate")
    out = require_candidate_output_dir(Path(candidate["output_dir"]), session.output_root)
    out.mkdir(parents=True)
    cache_dir = out / "inductor_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    _reset_dynamo_counters()

    model = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    torch_compile_wrapper_seconds = None
    if candidate["compile"]:
        t0 = time.perf_counter()
        model = torch.compile(model)
        torch_compile_wrapper_seconds = time.perf_counter() - t0
    optimizer = build_pilot_optimizer(model, candidate["peak_lr"])
    view = session.train_view(validated["stage_a"]["dataset"], candidate["seed_label"])
    result = _run_updates(
        model=model,
        optimizer=optimizer,
        view=view,
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=LR_RUN_UPDATES,
        lr_fn=lambda u: lr_schedule(u, candidate["peak_lr"]),
        ledger=session.ledger,
        phase="LR",
        device="cuda",
        record_diagnostics=True,
    )
    eval_a = IndexView(validated["stage_a"]["dataset"], validated["indices"]["stage_a_eval"])
    eval_b = IndexView(validated["stage_b"]["dataset"], validated["indices"]["stage_b_eval"])
    a = evaluate(model, eval_a, micro_bsz=candidate["micro_bsz"], device="cuda")
    b = evaluate(model, eval_b, micro_bsz=candidate["micro_bsz"], device="cuda")
    grouping = verify_realized_grouping(optimizer, model)
    states = verify_optimizer_state(optimizer)
    guard = sustained_divergence(result["losses"])
    compile_evidence = compile_path_evidence(
        model, requested=bool(candidate["compile"]), cache_dir=cache_dir
    )
    payload = {
        "phase": "LR",
        "candidate_id": candidate["candidate_id"],
        "peak_lr": candidate["peak_lr"],
        "seed_label": candidate["seed_label"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": bool(candidate["compile"]),
        "completed_updates": result["completed_updates"],
        "blocks_consumed": result["blocks_consumed"],
        "losses_by_update": {str(u): v for u, v in result["losses"].items()},
        "measured_update_wall_seconds": result["measured_update_wall_seconds"],
        "torch_compile_wrapper_seconds": torch_compile_wrapper_seconds,
        "first_optimizer_update_wall_seconds": result["first_optimizer_update_wall_seconds"],
        "all_losses_finite": all(map(_finite, result["losses"].values())),
        "all_grad_norms_finite": all(map(_finite, result["grad_norms"].values())),
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "muon_momentum_states_present": states["all_states_instantiated"],
        "aux_adamw_states_present": states["all_states_instantiated"],
        "grouping_matches_contract": grouping["matches_frozen_realization"],
        "all_lr_ratios_are_one": grouping["all_lr_ratios_are_one"],
        # R2 Part 7: raw evaluation components. SCORE is recomputed from these by the parent.
        "eval_stage_a_numerator": a["numerator"],
        "eval_stage_a_weight": a["weight"],
        "eval_stage_a_blocks": a["blocks"],
        "eval_stage_b_numerator": b["numerator"],
        "eval_stage_b_weight": b["weight"],
        "eval_stage_b_blocks": b["blocks"],
        "eval_loss_stage_a": a["loss"],
        "eval_loss_stage_b": b["loss"],
        "score": lr_score(a["loss"], b["loss"]),
        "sustained_divergence": guard["diverged"],
        "divergence_detail": guard,
        "compile_evidence": compile_evidence,
        "canonical_compile_path": compile_evidence["canonical_compile_path"],
        "diagnostics": result["diagnostics"][-1] if result["diagnostics"] else None,
        "rms_matching": verify_rms_matching(lr=float(candidate["peak_lr"])),
    }
    return _publish_result(out, payload, candidate, session)


def _finite(v: float) -> bool:
    import math

    return math.isfinite(float(v))


def _publish_result(
    out: Path, payload: Mapping[str, Any], candidate: Mapping[str, Any], session: ExecutionSession
) -> dict[str, Any]:
    """Write run_meta and the result, binding every identity selection will later require."""
    validated = session.validated
    meta = run_meta(candidate=candidate, session=session)
    meta_bytes = canonical_json_bytes(meta)
    (out / "run_meta.json").write_bytes(meta_bytes)
    bound = dict(payload)
    bound.update({
        "eligible": True,
        "run_meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "contract_sha256": validated["observed"]["contract_sha256"],
        "implementation_head": validated["observed"]["head"],
        "execution_bundle_sha256": validated["observed"]["execution_bundle_sha256"],
        "serialized_index_lists_digest": validated["serialized_index_lists_digest"],
        "pilot_index_manifest_file_sha256": validated["pilot_index_manifest_file_sha256"],
        "runtime_fingerprint_sha256": validated["fingerprint"]["fingerprint_sha256"],
        "authorization_sha256": validated["authorization_sha256"],
        "session_id": session.session_id,
        "ledger_identity": dict(session.ledger.identity),
        "ledger_snapshot": session.ledger.snapshot(),
        "output_dir": str(out),
    })
    (out / "result.json").write_bytes(canonical_json_bytes(bound))
    return bound


# --------------------------------------------------------------- independent recomputation


def median_tokens_per_sec(measured_update_wall_seconds: Sequence[float]) -> float:
    """Median throughput derived from the raw per-update timings, never from a stored scalar."""
    import statistics

    values = [float(v) for v in measured_update_wall_seconds]
    require(values, "no measured per-update wall timings were recorded")
    require(all(v > 0.0 for v in values), "a measured per-update wall time was not positive")
    return EFFECTIVE_BATCH_TOKENS / statistics.median(values)


def _agrees(recomputed: float, serialized: Any) -> bool:
    import math

    if not isinstance(serialized, (int, float)) or isinstance(serialized, bool):
        return False
    return math.isclose(float(recomputed), float(serialized), rel_tol=1e-12, abs_tol=0.0)


def verify_recomputed_mb_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """R2 Part 7: recompute Phase-MB throughput; refuse a disagreeing serialized value."""
    recomputed = median_tokens_per_sec(result.get("measured_update_wall_seconds") or [])
    serialized = result.get("median_tokens_per_sec")
    if not _agrees(recomputed, serialized):
        raise BindingFailure(
            f"Phase-MB candidate {result.get('candidate_id')!r}: serialized "
            f"median_tokens_per_sec {serialized!r} disagrees with the value recomputed from the "
            f"raw per-update timings ({recomputed!r})"
        )
    measured_count = len(result["measured_update_wall_seconds"])
    expected_count = MB_PROBE_UPDATES - MB_MEASURED_FIRST_UPDATE + 1
    if measured_count != expected_count:
        raise BindingFailure(
            f"Phase-MB candidate {result.get('candidate_id')!r} recorded {measured_count} "
            f"measured updates; the contract measures exactly {expected_count}"
        )
    return {"median_tokens_per_sec": recomputed, "measured_updates": measured_count}


def verify_recomputed_lr_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """R2 Part 7: recompute both eval losses and SCORE from the raw numerators and weights."""
    cid = result.get("candidate_id")
    components = {}
    for stage in ("a", "b"):
        numerator = result.get(f"eval_stage_{stage}_numerator")
        weight = result.get(f"eval_stage_{stage}_weight")
        if not isinstance(numerator, (int, float)) or isinstance(numerator, bool):
            raise BindingFailure(f"LR candidate {cid!r} has no raw stage-{stage} eval numerator")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or float(weight) <= 0:
            raise BindingFailure(f"LR candidate {cid!r} has no positive stage-{stage} eval weight")
        loss = float(numerator) / max(1.0, float(weight))
        if not _agrees(loss, result.get(f"eval_loss_stage_{stage}")):
            raise BindingFailure(
                f"LR candidate {cid!r}: serialized eval_loss_stage_{stage} "
                f"{result.get(f'eval_loss_stage_{stage}')!r} disagrees with the value recomputed "
                f"from its raw numerator and weight ({loss!r})"
            )
        components[stage] = loss
    score = lr_score(components["a"], components["b"])
    if not _agrees(score, result.get("score")):
        raise BindingFailure(
            f"LR candidate {cid!r}: serialized score {result.get('score')!r} disagrees with the "
            f"value recomputed from the raw evaluation components ({score!r})"
        )
    return {
        "eval_loss_stage_a": components["a"],
        "eval_loss_stage_b": components["b"],
        "score": score,
        "score_weights": list(LR_SCORE_WEIGHTS),
    }


# --------------------------------------------------------------------- orchestrator

MB_REPORT_SCHEMA = "petitgpt-pilot-phase-mb-report-v2.3"
LR_REPORT_SCHEMA = "petitgpt-pilot-phase-lr-report-v2.3"
MB_REPORT_FILENAME = "PHASE_MB_REPORT.json"
LR_REPORT_FILENAME = "PHASE_LR_REPORT.json"

REQUIRED_RESULT_BINDINGS = (
    "phase",
    "candidate_id",
    "seed_label",
    "run_meta_sha256",
    "contract_sha256",
    "implementation_head",
    "execution_bundle_sha256",
    "serialized_index_lists_digest",
    "pilot_index_manifest_file_sha256",
    "runtime_fingerprint_sha256",
    "authorization_sha256",
    "session_id",
    "ledger_identity",
    "output_dir",
    "completed_updates",
)


def _session_bindings(session: ExecutionSession) -> tuple[tuple[str, Any], ...]:
    v = session.validated
    return (
        ("contract_sha256", v["observed"]["contract_sha256"]),
        ("implementation_head", v["observed"]["head"]),
        ("execution_bundle_sha256", v["observed"]["execution_bundle_sha256"]),
        ("serialized_index_lists_digest", v["serialized_index_lists_digest"]),
        ("pilot_index_manifest_file_sha256", v["pilot_index_manifest_file_sha256"]),
        ("runtime_fingerprint_sha256", v["fingerprint"]["fingerprint_sha256"]),
        ("authorization_sha256", v["authorization_sha256"]),
        ("session_id", v["session_id"]),
    )


def load_completed_result(output_dir: Path, session: ExecutionSession) -> dict[str, Any]:
    """Load one candidate's result artifact, verify it binds THIS session, and recompute it."""
    path = Path(output_dir) / "result.json"
    if not path.is_file():
        raise BindingFailure(f"no completed result artifact at {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_RESULT_BINDINGS if f not in result]
    if missing:
        raise BindingFailure(f"result artifact is missing required bindings: {missing}")
    for field, expected in _session_bindings(session):
        if result.get(field) != expected:
            raise BindingFailure(f"result artifact {field} does not bind this execution")
    if result.get("ledger_identity") != dict(session.ledger.identity):
        raise BindingFailure("result artifact ledger identity does not bind this execution")
    if str(result["phase"]) == "MB":
        result.update(verify_recomputed_mb_result(result))
    else:
        result.update(verify_recomputed_lr_result(result))
    return result


def _ineligible_evidence(
    candidate: Mapping[str, Any], session: ExecutionSession, reason: str, detail: str
) -> dict[str, Any]:
    """Structured candidate-local failure evidence; the required grid continues."""
    evidence = {
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": bool(candidate["compile"]),
        "eligible": False,
        "reason": reason,
        "detail": detail,
        "completed_updates": 0,
        "oom": reason == "oom",
        "uncontrolled_exception": reason == "candidate_runtime_exception",
        "all_losses_finite": False,
        "all_grad_norms_finite": False,
        "all_parameters_finite": False,
        "all_optimizer_states_instantiated": False,
        "muon_momentum_states_present": False,
        "aux_adamw_states_present": False,
        "grouping_matches_contract": False,
        "all_lr_ratios_are_one": False,
        "max_memory_reserved_bytes": 0,
        "median_tokens_per_sec": 0.0,
        "measured_update_wall_seconds": [],
        "canonical_compile_path": False,
        "run_meta_sha256": None,
        "output_dir": candidate["output_dir"],
        "ledger_identity": dict(session.ledger.identity),
    }
    if "peak_lr" in candidate:
        evidence["peak_lr"] = float(candidate["peak_lr"])
        evidence["eval_loss_stage_a"] = None
        evidence["eval_loss_stage_b"] = None
        evidence["score"] = None
        evidence["sustained_divergence"] = False
    evidence.update(dict(_session_bindings(session)))
    return evidence


# --------------------------------------------------------------- subprocess terminal protocol

TERMINAL_STATUSES = ("SUCCESS", "CANDIDATE_INELIGIBLE", "PHASE_ABORT", "BINDING_FAILURE")
TERMINAL_RESULT_FIELDS = (
    "schema_version",
    "terminal_status",
    "error_class",
    "error_message",
    "completed_updates",
    "reserved_tokens",
    "completed_tokens",
    "run_meta_sha256",
    "candidate_id",
    "phase",
)
TERMINAL_RESULT_SCHEMA = "petitgpt-pilot-candidate-terminal-result-v2.3"


def terminal_result_path(spec_path: Path) -> Path:
    return Path(spec_path).with_suffix(".terminal.json")


def write_terminal_result(spec_path: Path, payload: Mapping[str, Any]) -> Path:
    path = terminal_result_path(spec_path)
    doc = {"schema_version": TERMINAL_RESULT_SCHEMA, **dict(payload)}
    missing = [f for f in TERMINAL_RESULT_FIELDS if f not in doc]
    require(not missing, f"terminal result is incomplete: {missing}")
    require(
        doc["terminal_status"] in TERMINAL_STATUSES,
        f"unknown terminal status {doc['terminal_status']!r}",
    )
    path.write_bytes(canonical_json_bytes(doc))
    return path


def read_terminal_result(spec_path: Path) -> dict[str, Any]:
    """A missing or malformed terminal result is itself a phase-level failure, never a pass."""
    path = terminal_result_path(spec_path)
    if not path.is_file():
        raise PhaseAbort(
            f"candidate subprocess left no terminal result at {path}; a candidate that cannot "
            f"report its own terminal state may not be treated as merely ineligible"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != TERMINAL_RESULT_SCHEMA:
        raise PhaseAbort(f"terminal result schema mismatch at {path}")
    missing = [f for f in TERMINAL_RESULT_FIELDS if f not in doc]
    if missing:
        raise PhaseAbort(f"terminal result at {path} is missing {missing}")
    if doc["terminal_status"] not in TERMINAL_STATUSES:
        raise PhaseAbort(f"unknown terminal status {doc['terminal_status']!r} at {path}")
    return doc


def _subprocess_backend(candidate: Mapping[str, Any], session: ExecutionSession) -> None:
    """Launch one candidate in a FRESH subprocess that revalidates every artifact itself.

    R2 Part 2: the child is given the REAL authorization manifest path, the real pilot-index
    manifest path and its own immutable spec, and revalidates all of them from disk. The parent
    passes no hashes, no counts and no authority the child could take on trust.

    R2 Part 11: the child's stdout and stderr are preserved verbatim next to the spec, and its
    structured terminal result -- not its exit code alone -- decides how the parent proceeds.
    """
    validated = session.validated
    spec_dir = session.output_root / "_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{candidate['candidate_id']}.json"
    # A stale terminal result from an earlier attempt must never be read as this launch's.
    terminal_result_path(spec_path).unlink(missing_ok=True)
    spec_path.write_bytes(
        canonical_json_bytes({
            "candidate": dict(candidate),
            "authorization_path": validated["authorization_path"],
            "pilot_index_manifest_path": validated["index_manifest_path"],
            "authorized_output_root": str(session.output_root),
            "session_id": session.session_id,
        })
    )
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(Path(candidate["output_dir"]) / "inductor_cache")
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(ROOT) / "pretrain" / "pilot_runner_v2_3.py"),
            "execute-candidate",
            "--spec",
            str(spec_path),
            "--authorization",
            validated["authorization_path"],
            "--pilot-index-manifest",
            validated["index_manifest_path"],
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (spec_dir / f"{candidate['candidate_id']}.stdout").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (spec_dir / f"{candidate['candidate_id']}.stderr").write_text(
        completed.stderr or "", encoding="utf-8"
    )
    terminal = read_terminal_result(spec_path)
    status = terminal["terminal_status"]
    expected_exit = RESULT_CLASSES[status]
    if int(completed.returncode) != expected_exit:
        raise PhaseAbort(
            f"candidate {candidate['candidate_id']} reported terminal status {status} but exited "
            f"{completed.returncode} (expected {expected_exit}); the terminal protocol is broken"
        )
    if status == "BINDING_FAILURE":
        raise BindingFailure(
            f"candidate {candidate['candidate_id']} binding failure: {terminal['error_message']}"
        )
    if status == "PHASE_ABORT":
        raise PhaseAbort(
            f"candidate {candidate['candidate_id']} phase abort: {terminal['error_message']}"
        )
    if status == "CANDIDATE_INELIGIBLE":
        raise CandidateFailure(f"{terminal.get('error_class')}: {terminal.get('error_message')}")


def _classify_candidate_failure(exc: BaseException) -> tuple[str, str]:
    name = type(exc).__name__
    text = str(exc)
    if "OutOfMemory" in name or "out of memory" in text.lower():
        return "oom", f"{name}: {text}"
    if "compile" in text.lower() or "dynamo" in text.lower() or "inductor" in text.lower():
        return "compile_failure", f"{name}: {text}"
    return "candidate_runtime_exception", f"{name}: {text}"


def _launch(
    candidate: Mapping[str, Any], session: ExecutionSession, backend: Any
) -> dict[str, Any]:
    """Launch one candidate. Binding/phase failures propagate; candidate failures become
    structured ineligible evidence so the required grid can still be completed."""
    try:
        backend(candidate, session)
    except (BindingFailure, PhaseAbort):
        raise
    except PilotContractError as exc:
        raise PhaseAbort(f"{candidate['candidate_id']}: {exc}") from exc
    except CandidateFailure as exc:
        reason, detail = _classify_candidate_failure(exc)
        return _ineligible_evidence(candidate, session, reason, detail)
    except BaseException as exc:  # noqa: BLE001 - candidate-local failure
        reason, detail = _classify_candidate_failure(exc)
        return _ineligible_evidence(candidate, session, reason, detail)
    return load_completed_result(Path(candidate["output_dir"]), session)


# --------------------------------------------------------------------- immutable reports


def write_immutable_report(path: Path, payload: Mapping[str, Any]) -> str:
    """R2 Part 6: a phase report is published once and never rewritten."""
    path = Path(path)
    if path.exists():
        raise BindingFailure(
            f"{path.name} already exists at {path}; a phase report is immutable and a rerun "
            f"requires a new authorized output root"
        )
    body = canonical_json_bytes(payload)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = hashlib.sha256(body).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def orchestrate_phase_mb(session: ExecutionSession, *, backend: Any = None) -> dict[str, Any]:
    """The single Phase-MB orchestration path.

    Enumerates exactly the frozen ten candidates, launches each, converts candidate-local
    failures into structured ineligible evidence while continuing the grid, then recomputes
    every eligible candidate's throughput from its raw timings and performs deterministic
    selection on that recomputed evidence alone.
    """
    if session.scope not in ALLOWED_SCOPES:
        raise BindingFailure(f"unknown authorized scope {session.scope!r}")
    backend = backend or _subprocess_backend
    candidates = plan_phase_mb(output_root=session.output_root)
    require(len(candidates) == 10, "the Phase-MB grid must enumerate exactly ten candidates")

    outcomes = [_launch(c, session, backend) for c in candidates]
    require_complete_mb_grid(outcomes)
    vram = int(session.validated["fingerprint"]["gpu"]["total_vram_bytes"])
    selection = mb_select(outcomes, vram)
    report = {
        "schema_version": MB_REPORT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "phase": "MB",
        "candidates": outcomes,
        "selection": selection,
        "ledger": session.ledger.snapshot(),
        "authorized_scope": session.scope,
        # R2 Part 3: PHASE_MB_ONLY completes here. There is no promotion path: Phase LR needs a
        # new FULL_V2_3_PILOT authorization, which implies a new session and a new ledger.
        "session_terminates_after_this_phase": session.scope == "PHASE_MB_ONLY",
        "next_phase_requires_new_authorization": session.scope == "PHASE_MB_ONLY",
        "physical_vram_bytes": vram,
        **dict(_session_bindings(session)),
    }
    report["report_sha256"] = write_immutable_report(
        session.output_root / MB_REPORT_FILENAME, report
    )
    return report


def load_authoritative_mb_report(session: ExecutionSession) -> dict[str, Any]:
    """R2 Part 6: Phase-LR geometry comes ONLY from the verified Phase-MB report.

    The report file is re-hashed against its published sidecar, revalidated against this
    session's bindings, checked for a complete ten-candidate grid, recomputed candidate by
    candidate from raw timings, and its recorded selection is re-derived. Nothing about the
    Phase-LR geometry can be supplied on the command line.
    """
    path = session.output_root / MB_REPORT_FILENAME
    if not path.is_file():
        raise BindingFailure(
            f"Phase-LR requires the authoritative Phase-MB report at {path}; the frozen "
            f"micro_bsz and compile mode are never taken from the command line"
        )
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise BindingFailure(f"Phase-MB report has no published SHA-256 sidecar at {sidecar}")
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    if recorded != digest:
        raise BindingFailure(
            f"Phase-MB report SHA-256 {digest} does not match its published sidecar {recorded}"
        )
    report = json.loads(body.decode("utf-8"))
    if report.get("schema_version") != MB_REPORT_SCHEMA:
        raise BindingFailure("Phase-MB report schema mismatch")
    # Phase LR runs under the SAME authorization, session and ledger as the Phase MB it
    # consumes, so every binding -- authorization and session identity included -- must match.
    for field, expected in _session_bindings(session):
        if report.get(field) != expected:
            raise BindingFailure(
                f"Phase-MB report {field} does not bind this execution; Phase LR consumes only "
                f"a Phase-MB report published by its own authorized session"
            )
    candidates = report.get("candidates") or []
    require_complete_mb_grid(candidates)
    verified = []
    for record in candidates:
        if record.get("eligible"):
            verified.append({**record, **verify_recomputed_mb_result(record)})
        else:
            verified.append(dict(record))
    reselected = mb_select(verified, int(report["physical_vram_bytes"]))
    if reselected != report.get("selection"):
        raise BindingFailure(
            "re-deriving the Phase-MB selection from the recomputed candidate evidence does not "
            "reproduce the published selection"
        )
    if reselected.get("outcome") != "PHASE_MB_FROZEN":
        raise PhaseAbort(
            f"Phase MB did not freeze a geometry ({reselected.get('outcome')}); Phase LR cannot "
            f"start"
        )
    return {
        "report_sha256": digest,
        "selection": reselected,
        "micro_bsz": int(reselected["FROZEN_MICRO_BSZ"]),
        "grad_accum": int(reselected["FROZEN_GRAD_ACCUM"]),
        "compile": bool(reselected["FROZEN_COMPILE"]),
    }


def orchestrate_phase_muon_lr(session: ExecutionSession, *, backend: Any = None) -> dict[str, Any]:
    """The single Phase-Muon-LR orchestration path.

    Derives its batch geometry from the verified Phase-MB report and the initial grid, the
    seed-1 winner, the confirmation pair and the edge candidate ITSELF. Nothing about which
    candidates run, or with what geometry, is taken from caller input.
    """
    if session.scope != "FULL_V2_3_PILOT":
        raise BindingFailure(
            f"scope {session.scope!r} may not execute Phase Muon-LR; PHASE_MB_ONLY terminates "
            f"after its Phase-MB report and is never promoted"
        )
    backend = backend or _subprocess_backend
    frozen = load_authoritative_mb_report(session)

    def run(lrs: Sequence[float], seed_label: str) -> list[dict[str, Any]]:
        planned = plan_phase_lr(
            output_root=session.output_root,
            micro_bsz=frozen["micro_bsz"],
            compile_on=frozen["compile"],
            peak_lrs=lrs,
            seed_label=seed_label,
        )
        return [_launch(c, session, backend) for c in planned]

    def publish(payload: Mapping[str, Any]) -> dict[str, Any]:
        doc = {
            "schema_version": LR_REPORT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "phase": "LR",
            "frozen_geometry": frozen,
            "ledger": session.ledger.snapshot(),
            **dict(_session_bindings(session)),
            **dict(payload),
        }
        doc["report_sha256"] = write_immutable_report(session.output_root / LR_REPORT_FILENAME, doc)
        return doc

    seed1 = run(LR_GRID_SEED1, "seed-1")
    require_complete_lr_grid(seed1)  # R2 Part 8: the initial grid must be complete evidence
    seed1_verdict = lr_select_seed1(seed1)
    if seed1_verdict["outcome"] != "SEED1_WINNER":
        return publish({"seed1": seed1, "outcome": seed1_verdict, "terminal_status": "PHASE_ABORT"})

    winner_lr = float(seed1_verdict["winner_peak_lr"])
    neighbour = confirmation_neighbor(winner_lr)
    seed2 = run([winner_lr, neighbour], "seed-2")
    by_lr_1 = {float(r["peak_lr"]): r for r in seed1}
    by_lr_2 = {float(r["peak_lr"]): r for r in seed2}
    # R2 Part 8: confirmation evidence must be COMPLETE -- both LRs must have a recorded run at
    # both seeds. A missing record is an evidence gap, never an implicit ineligibility.
    missing = [lr for lr in (winner_lr, neighbour) if lr not in by_lr_1 or lr not in by_lr_2]
    if missing:
        raise PhaseAbort(f"confirmation evidence incomplete; no record for peak_lr {missing}")
    pairs = [
        {
            "peak_lr": lr,
            "seed1_score": float(by_lr_1[lr]["score"])
            if by_lr_1[lr].get("eligible")
            else float("inf"),
            "seed1_eligible": bool(by_lr_1[lr].get("eligible"))
            and lr_candidate_eligible(by_lr_1[lr])[0],
            "seed2_score": float(by_lr_2[lr]["score"])
            if by_lr_2[lr].get("eligible")
            else float("inf"),
            "seed2_eligible": bool(by_lr_2[lr].get("eligible"))
            and lr_candidate_eligible(by_lr_2[lr])[0],
        }
        for lr in (winner_lr, neighbour)
    ]
    confirmed = lr_confirm(pairs)
    if confirmed["outcome"] != "CONFIRMED":
        return publish({
            "seed1": seed1,
            "seed2": seed2,
            "confirmation_pairs": pairs,
            "seed1_verdict": seed1_verdict,
            "outcome": confirmed,
            "terminal_status": "PHASE_ABORT",
        })

    incumbent_lr = float(confirmed["confirmed_peak_lr"])
    edge_lr = edge_candidate(incumbent_lr)
    edge_runs: list[dict[str, Any]] = []
    edge_kwargs: dict[str, Any] = {"edge_lr": edge_lr}
    if edge_lr is not None:
        e1 = run([edge_lr], "seed-1")
        e2 = run([edge_lr], "seed-2")
        edge_runs = e1 + e2
        # R2 Part 8: a bounded edge expansion needs a recorded run at BOTH seeds before it can
        # be resolved either way.
        if len(e1) != 1 or len(e2) != 1:
            raise PhaseAbort(
                f"edge evidence incomplete for peak_lr {edge_lr}: "
                f"{len(e1)} seed-1 and {len(e2)} seed-2 records"
            )
        edge_kwargs.update({
            "edge_seed1_eligible": bool(e1[0].get("eligible")) and lr_candidate_eligible(e1[0])[0],
            "edge_seed2_eligible": bool(e2[0].get("eligible")) and lr_candidate_eligible(e2[0])[0],
            "edge_seed1_score": float(e1[0]["score"]) if e1[0].get("eligible") else None,
            "edge_seed2_score": float(e2[0]["score"]) if e2[0].get("eligible") else None,
        })
    final = lr_resolve_edge(
        incumbent_lr=incumbent_lr,
        incumbent_final_score=float(confirmed["final_score"]),
        **edge_kwargs,
    )
    return publish({
        "seed1": seed1,
        "seed2": seed2,
        "confirmation_pairs": pairs,
        "edge_runs": edge_runs,
        "seed1_verdict": seed1_verdict,
        "confirmed": confirmed,
        "final": final,
        "outcome": final,
        "terminal_status": (
            "SUCCESS" if final["outcome"] == "PHASE_MUON_LR_FROZEN" else "PHASE_ABORT"
        ),
    })


# --------------------------------------------------------------------- checkpoint policy


def require_checkpointing_disabled(action: str) -> None:
    """V2.3 freezes PILOT_CHECKPOINTING=DISABLED: the executor neither writes nor reads one."""
    from pretrain.pilot_contract_v2_3 import require_checkpointing_disabled as _refuse

    _refuse(action)


# --------------------------------------------------------------------- planning + meta


def require_new_output_dir(destination: Path) -> Path:
    """Release-containment check for planning-time paths (no authorized root yet)."""
    root = repo_root()
    resolved = Path(destination).resolve()
    for prefix in PROTECTED_PREFIXES:
        protected = (root / prefix).resolve()
        require(
            resolved != protected and protected not in resolved.parents,
            f"refusing to write pilot output inside an accepted release: {resolved}",
        )
    require(not resolved.exists(), f"pilot output directory must not exist: {destination}")
    return resolved


def plan_phase_mb(*, output_root: Path) -> list[dict[str, Any]]:
    return [
        {
            **c,
            "seed_label": "seed-1",
            "train_order_seed": SEED_SEMANTICS["seed-1"]["train_order"],
            "output_dir": str(Path(output_root) / c["candidate_id"]),
            "tokens": c["updates"] * EFFECTIVE_BATCH_TOKENS,
            "blocks_consumed": c["updates"] * SEQUENCES_PER_OPTIMIZER_UPDATE,
            "authorization_status": "NOT_AUTHORIZED",
        }
        for c in mb_candidate_grid()
    ]


def plan_phase_lr(
    *,
    output_root: Path,
    micro_bsz: int,
    compile_on: bool,
    peak_lrs: Sequence[float] = LR_GRID_SEED1,
    seed_label: str = "seed-1",
) -> list[dict[str, Any]]:
    require(seed_label in SEED_SEMANTICS, f"unknown seed label {seed_label!r}")
    seeds = SEED_SEMANTICS[seed_label]
    return [
        {
            "candidate_id": f"lr_{lr:g}_{seed_label}",
            "phase": "LR",
            "seed_label": seed_label,
            "micro_bsz": int(micro_bsz),
            "grad_accum": frozen_grad_accum(micro_bsz),
            "compile": bool(compile_on),
            "peak_lr": float(lr),
            "updates": LR_RUN_UPDATES,
            "warmup_updates": LR_WARMUP_UPDATES,
            "optimizer": "muon",
            "muon_lr": MUON_LR_ARG,
            "muon_momentum": MUON_MOMENTUM,
            "model_init_seed": seeds["model_init"],
            "train_order_seed": seeds["train_order"],
            "output_dir": str(Path(output_root) / f"lr_{lr:g}_{seed_label}"),
            "tokens": LR_RUN_UPDATES * EFFECTIVE_BATCH_TOKENS,
            "blocks_consumed": LR_RUN_UPDATES * SEQUENCES_PER_OPTIMIZER_UPDATE,
            "authorization_status": "NOT_AUTHORIZED",
        }
        for lr in peak_lrs
    ]


def run_meta(*, candidate: Mapping[str, Any], session: ExecutionSession) -> dict[str, Any]:
    v = session.validated
    return {
        "schema_version": RUN_META_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "base_fingerprint_sha256": v["fingerprint"]["fingerprint_sha256"],
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": bool(candidate["compile"]),
        "optimizer": {
            "name": "muon",
            "muon_lr": MUON_LR_ARG,
            "muon_momentum": MUON_MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "lr_configuration": {
            "peak_lr": candidate["peak_lr"],
            "warmup_updates": candidate["warmup_updates"],
            "updates": candidate["updates"],
        },
        "model_seed": candidate["model_init_seed"],
        "train_order_seed": candidate["train_order_seed"],
        "pilot_index_hashes": {
            k: v["indices"][k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        },
        "serialized_index_lists_digest": v["serialized_index_lists_digest"],
        "pilot_index_manifest_file_sha256": v["pilot_index_manifest_file_sha256"],
        "contract_sha256": v["observed"]["contract_sha256"],
        "implementation_head": v["observed"]["head"],
        "execution_implementation_bundle_sha256": v["observed"]["execution_bundle_sha256"],
        "authorized_scope": session.scope,
        "session_id": session.session_id,
        "authorization_sha256": v["authorization_sha256"],
        "authorization_status": "AUTHORIZED_BY_EXTERNAL_MANIFEST",
    }


# --------------------------------------------------------------------- index manifest


def pilot_index_manifest_document() -> dict[str, Any]:
    """Derive the pilot index manifest from the accepted releases. Carries no self-hash.

    Release identity is opened and derived here, so the published manifest names the exact
    Stage-A/Stage-B releases the indices were drawn against. The authorization then binds the
    SHA-256 of this file's bytes, which the executor recomputes from disk at every launch.
    """
    from pretrain.pilot_contract_v2_3 import (
        PILOT_INDEX_SCHEMA,
        PILOT_INDEX_SEED,
        STAGE_A_EVAL_COUNT,
        STAGE_A_TRAIN_COUNT,
        STAGE_B_EVAL_COUNT,
    )

    stage_a = verify_accepted_release("stage_a")
    stage_b = verify_accepted_release("stage_b")
    indices = generate_pilot_indices(
        stage_a_blocks=stage_a["blocks"], stage_b_blocks=stage_b["blocks"]
    )
    keys = ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
    return {
        "schema_version": PILOT_INDEX_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "authorization_status": "NOT_AUTHORIZED",
        "contract_sha256": contract_sha256(),
        "execution_implementation_bundle_sha256": execution_bundle_sha256(),
        "implementation_head": git_policy_status()["head"],
        "numpy_version": require_numpy_version(),
        "generator": f"numpy.random.Generator(PCG64({PILOT_INDEX_SEED}))",
        "seed": PILOT_INDEX_SEED,
        "counts": {
            "stage_a_eval": STAGE_A_EVAL_COUNT,
            "stage_a_train": STAGE_A_TRAIN_COUNT,
            "stage_b_eval": STAGE_B_EVAL_COUNT,
        },
        "stage_a_blocks": stage_a["blocks"],
        "stage_b_blocks": stage_b["blocks"],
        "stage_a_meta_sha256": stage_a["meta_sha256"],
        "stage_b_meta_sha256": stage_b["meta_sha256"],
        **{k: indices[k] for k in keys},
        "serialized_index_lists_digest": hashlib.sha256(
            canonical_json_bytes({k: indices[k] for k in keys})
        ).hexdigest(),
        "universes_derived_from": "the accepted Stage-A/Stage-B releases at runtime",
    }


def write_pilot_index_manifest(path: Path) -> dict[str, Any]:
    """Publish the manifest once; the authorization binds the SHA-256 of these exact bytes."""
    path = Path(path)
    if path.is_dir():
        path = path / PILOT_INDEX_MANIFEST_FILENAME
    require(
        path.name == PILOT_INDEX_MANIFEST_FILENAME,
        f"the pilot index manifest is published as {PILOT_INDEX_MANIFEST_FILENAME}, not "
        f"{path.name!r}; the authorization binds a file with that canonical name",
    )
    require(not path.exists(), f"pilot index manifest already exists: {path}")
    body = canonical_json_bytes(pilot_index_manifest_document())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return {"path": str(path), "pilot_index_manifest_file_sha256": hashlib.sha256(body).hexdigest()}


# --------------------------------------------------------------------- CLI


def _emit(value: Any) -> None:
    sys.stdout.write(canonical_json_bytes(value).decode())


def _cli_execute_candidate(args: argparse.Namespace) -> int:
    """One candidate, in its own process, revalidating every artifact from disk itself."""
    spec_path = Path(args.spec)
    session: ExecutionSession | None = None
    candidate: Mapping[str, Any] = {}
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        candidate = spec["candidate"]
        for key, supplied in (
            ("authorization_path", str(args.authorization)),
            ("pilot_index_manifest_path", str(args.pilot_index_manifest)),
        ):
            if str(spec.get(key)) != supplied:
                raise BindingFailure(
                    f"spec {key} {spec.get(key)!r} does not match the path this process was "
                    f"launched with ({supplied!r})"
                )
        session = open_session(
            authorization_path=Path(args.authorization),
            candidate_spec_path=spec_path,
            pilot_index_manifest_path=Path(args.pilot_index_manifest),
            output_dir=Path(spec["authorized_output_root"]),
            phase=str(candidate["phase"]),
        )
        if spec.get("session_id") != session.session_id:
            raise BindingFailure(
                "the session identity this candidate derived from the artifacts on disk does "
                "not match the identity recorded in its spec"
            )
        backend = run_phase_mb_candidate if candidate["phase"] == "MB" else run_phase_lr_candidate
        result = backend(candidate, session)
        write_terminal_result(
            spec_path,
            {
                "terminal_status": "SUCCESS",
                "error_class": None,
                "error_message": None,
                "completed_updates": int(result["completed_updates"]),
                "reserved_tokens": session.ledger.snapshot()["reserved_tokens"],
                "completed_tokens": session.ledger.snapshot()["completed_tokens"],
                "run_meta_sha256": result["run_meta_sha256"],
                "candidate_id": result["candidate_id"],
                "phase": result["phase"],
            },
        )
        return SUCCESS
    except BaseException as exc:  # noqa: BLE001 - every exit is a structured terminal result
        if isinstance(exc, BindingFailure):
            status = "BINDING_FAILURE"
        elif isinstance(exc, (PhaseAbort, PilotContractError)):
            status = "PHASE_ABORT"
        else:
            status = "CANDIDATE_INELIGIBLE"
        snapshot = session.ledger.snapshot() if session is not None else {}
        write_terminal_result(
            spec_path,
            {
                "terminal_status": status,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "completed_updates": 0,
                "reserved_tokens": snapshot.get("reserved_tokens"),
                "completed_tokens": snapshot.get("completed_tokens"),
                "run_meta_sha256": None,
                "candidate_id": candidate.get("candidate_id"),
                "phase": candidate.get("phase"),
            },
        )
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return RESULT_CLASSES[status]


def _cli_run(args: argparse.Namespace) -> int:
    """The single orchestration entry point. Phase geometry is never a command-line input."""
    try:
        session = open_session(
            authorization_path=Path(args.authorization),
            candidate_spec_path=None,
            pilot_index_manifest_path=Path(args.pilot_index_manifest),
            output_dir=Path(args.output_root),
            phase=args.phase,
        )
        if args.phase == "MB":
            report = orchestrate_phase_mb(session)
            outcome = report["selection"].get("outcome")
            terminal = "SUCCESS" if outcome == "PHASE_MB_FROZEN" else "PHASE_ABORT"
        else:
            report = orchestrate_phase_muon_lr(session)
            terminal = report.get("terminal_status", "PHASE_ABORT")
        _emit({
            "phase": report["phase"],
            "terminal_status": terminal,
            "output_root": str(session.output_root),
            "session_id": session.session_id,
            "report_sha256": report["report_sha256"],
            "session_terminates_after_this_phase": bool(
                report.get(
                    "session_terminates_after_this_phase",
                    args.phase == "LR" or session.scope == "PHASE_MB_ONLY",
                )
            ),
            "ledger": session.ledger.snapshot(),
        })
        return RESULT_CLASSES[terminal]
    except BindingFailure as exc:
        sys.stderr.write(f"BINDING_FAILURE: {exc}\n")
        return BINDING_FAILURE
    except (PhaseAbort, PilotContractError) as exc:
        sys.stderr.write(f"PHASE_ABORT: {exc}\n")
        return PHASE_ABORT
    except BaseException:  # noqa: BLE001 - an unexpected orchestrator fault is a phase abort
        import traceback

        traceback.print_exc()
        sys.stderr.write("PHASE_ABORT: unexpected orchestrator fault (traceback above)\n")
        return PHASE_ABORT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    sub.add_parser("fingerprint")
    sub.add_parser("git-policy")
    sub.add_parser("closure")
    sub.add_parser("authorization-template")
    sub.add_parser("result-classes")
    sub.add_parser("rms-matching", help="CPU-only proof of the Muon RMS-matching rule")
    p = sub.add_parser("plan")
    p.add_argument("--output-root", type=Path, default=Path("runs/PILOT_V2_3_OUTPUT"))
    p.add_argument("--micro-bsz", type=int, default=None)
    i = sub.add_parser("write-index-manifest")
    i.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="the single orchestration entry point")
    r.add_argument("--phase", choices=["MB", "LR"], required=True)
    r.add_argument("--authorization", type=Path, required=True)
    r.add_argument("--pilot-index-manifest", type=Path, required=True)
    r.add_argument("--output-root", type=Path, required=True)
    e = sub.add_parser("execute-candidate", help="internal: one candidate in a fresh subprocess")
    e.add_argument("--spec", type=Path, required=True)
    e.add_argument("--authorization", type=Path, required=True)
    e.add_argument("--pilot-index-manifest", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "contract":
        from pretrain.pilot_contract_v2_3 import contract_document

        _emit(contract_document())
        return SUCCESS
    if args.command == "fingerprint":
        _emit(base_runtime_fingerprint())
        return SUCCESS
    if args.command == "git-policy":
        _emit(git_policy_status())
        return SUCCESS
    if args.command == "closure":
        _emit(execution_closure())
        return SUCCESS
    if args.command == "authorization-template":
        _emit(authorization_template())
        return SUCCESS
    if args.command == "result-classes":
        _emit(dict(RESULT_CLASSES))
        return SUCCESS
    if args.command == "rms-matching":
        _emit(verify_rms_matching())
        return SUCCESS
    if args.command == "write-index-manifest":
        _emit(write_pilot_index_manifest(args.out))
        return SUCCESS
    if args.command == "plan":
        payload: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "authorization_status": "NOT_AUTHORIZED",
            "phase_mb": plan_phase_mb(output_root=args.output_root),
        }
        payload["phase_lr"] = (
            plan_phase_lr(output_root=args.output_root, micro_bsz=args.micro_bsz, compile_on=False)
            if args.micro_bsz is not None
            else "pending FROZEN_MICRO_BSZ from the authoritative Phase-MB report"
        )
        _emit(payload)
        return SUCCESS
    if args.command == "execute-candidate":
        return _cli_execute_candidate(args)
    return _cli_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
