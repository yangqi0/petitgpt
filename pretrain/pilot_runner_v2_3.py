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
releases, and through the owner-selected hardware and observed-runtime binding. No tracked
code change is needed for a later authorized run: publishing the manifest is sufficient.

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
import math
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
    FULL_V2_3_PILOT_SESSION_HARD_CEILING,
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
    MUON_NESTEROV,
    MUON_NS_STEPS,
    NEWTON_SCHULZ_COEFFICIENTS,
    PHASE_CEILINGS,
    REQUIRED_NUMPY_VERSION,
    RMS_MATCHING_CONSTANT,
    RUN_META_SCHEMA,
    SEED_SEMANTICS,
    SEQUENCES_PER_OPTIMIZER_UPDATE,
    SESSION_BUDGET_SEMANTICS,
    TOKEN_LEDGER_SCHEMA,
    TRAINED_TOKENS_PER_UPDATE,
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
    mb_candidate_eligible,
    mb_candidate_grid,
    mb_lr,
    mb_median_update_tokens_per_second,
    mb_select,
    mb_update_tokens_per_second,
    require,
    require_complete_lr_grid,
    require_complete_mb_grid,
    require_exact_mb_timing_records,
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


def base_runtime_fingerprint(
    *,
    gpu_required: bool = False,
    hardware_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Base runtime identity. Carries no per-run configuration (no compile, LR, seed, phase)."""
    import numpy as np
    import tokenizers
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu: dict[str, Any] = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "selected_device_index": None,
    }
    if cuda_available and device_count > 0:
        selected_device_index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(selected_device_index)
        gpu.update({
            "selected_device_index": selected_device_index,
            "name": torch.cuda.get_device_name(selected_device_index),
            "total_vram_mib": int(props.total_memory // (1024 * 1024)),
            "total_vram_bytes": int(props.total_memory),
            "capability": f"{props.major}.{props.minor}",
            "driver": _nvidia_smi("driver_version"),
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
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
        "python_implementation": platform.python_implementation(),
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
    if gpu_required:
        require_training_authority(fp, hardware_binding)
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
SESSION_FILENAME = "SESSION.json"
LEDGER_PHASES = ("MB", "LR")
LEDGER_BUCKETS = ("MB", "LR", "GLOBAL")


class LedgerIntegrityFailure(PhaseAbort):
    """A structural invariant of the persisted ledger does not hold.

    R3 Part 8/10: this is a PHASE-level failure. A ledger whose arithmetic does not close is
    never downgraded to an ordinary ineligible candidate.
    """


class TokenLedger:
    """Persistent trained-token accounting with a conservative reserve-then-complete protocol.

    R2: a reservation is taken and persisted BEFORE the optimizer update is applied, and moved
    to completed only after it succeeds. A process that dies between the two leaves the
    reservation consumed on purpose: budget is never silently returned, so a crash can never
    result in an uncounted optimizer update.

    R3 Part 8: every read of this ledger -- including :meth:`snapshot`, which parents and
    reports consume -- takes the exclusive lock, reloads the bytes from disk and revalidates
    both the complete identity binding and the structural invariants. A parent therefore never
    reports stale in-memory counters after a child process advanced the file, and an arithmetic
    inconsistency is raised as :class:`LedgerIntegrityFailure` rather than reported as fact.
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
        missing_ceilings = [k for k in LEDGER_BUCKETS if k not in effective_ceilings]
        require(not missing_ceilings, f"ledger ceilings are incomplete: {missing_ceilings}")
        self.effective_ceilings = {k: int(effective_ceilings[k]) for k in LEDGER_BUCKETS}
        # POSIX flock() denies a second fd in the SAME process, so a nested acquisition would
        # deadlock a running candidate rather than fail. The depth counter makes the guard
        # reentrant: an inner snapshot() inside reserve()/complete() reuses the held lock.
        self._lock_depth = 0
        if self.path.is_file():
            self.state = self._validated(
                load_json_artifact(self.path, label=f"token ledger at {self.path}")
            )
        else:
            self.state = {
                "schema_version": TOKEN_LEDGER_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "identity": dict(self.identity),
                "effective_ceilings": dict(self.effective_ceilings),
                "session_hard_ceiling": FULL_V2_3_PILOT_SESSION_HARD_CEILING,
                "trained_tokens_per_update": TRAINED_TOKENS_PER_UPDATE,
                "reserved_tokens": {k: 0 for k in LEDGER_BUCKETS},
                "completed_tokens": {k: 0 for k in LEDGER_BUCKETS},
                "reserved_updates": 0,
                "completed_updates": 0,
            }
            self._write(self.state)

    # ---------------------------------------------------------------- validation

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
        self._require_structural_invariants(state)
        return dict(state)

    def _require_structural_invariants(self, state: Mapping[str, Any]) -> None:
        """R3 Part 8: the complete structural invariant set, enforced on EVERY reload."""

        def fail(message: str) -> None:
            raise LedgerIntegrityFailure(
                f"token ledger integrity failure at {self.path}: {message}"
            )

        def integer(value: Any, label: str) -> int:
            if not isinstance(value, int) or isinstance(value, bool):
                fail(f"{label} must be an integer, got {value!r}")
            return int(value)

        if integer(state.get("trained_tokens_per_update"), "trained_tokens_per_update") != (
            TRAINED_TOKENS_PER_UPDATE
        ):
            fail("stored trained-tokens-per-update does not equal the frozen batch geometry")
        if integer(state.get("session_hard_ceiling"), "session_hard_ceiling") != (
            FULL_V2_3_PILOT_SESSION_HARD_CEILING
        ):
            fail("stored session hard ceiling does not equal the frozen FULL-session ceiling")

        stored_ceilings = state.get("effective_ceilings")
        if not isinstance(stored_ceilings, Mapping):
            fail("effective_ceilings is missing or malformed")
        if {
            k: integer(stored_ceilings.get(k), f"effective_ceilings[{k}]") for k in LEDGER_BUCKETS
        } != (self.effective_ceilings):
            fail(
                "stored effective ceilings do not equal the ceilings frozen for this "
                "authorization and session"
            )
        for bucket in LEDGER_BUCKETS:
            frozen = (
                PHASE_CEILINGS[bucket] if bucket in PHASE_CEILINGS else (GLOBAL_PILOT_TOKEN_CEILING)
            )
            if self.effective_ceilings[bucket] > int(frozen):
                fail(f"effective ceiling for {bucket} exceeds the frozen contract ceiling")

        buckets: dict[str, dict[str, int]] = {}
        for name in ("reserved_tokens", "completed_tokens"):
            values = state.get(name)
            if not isinstance(values, Mapping):
                fail(f"{name} is missing or malformed")
            buckets[name] = {k: integer(values.get(k), f"{name}[{k}]") for k in LEDGER_BUCKETS}
            if sorted(values) != sorted(LEDGER_BUCKETS):
                fail(f"{name} must carry exactly the buckets {sorted(LEDGER_BUCKETS)}")
            for bucket, value in buckets[name].items():
                if value < 0:
                    fail(f"{name}[{bucket}] is negative ({value})")
                if value % TRAINED_TOKENS_PER_UPDATE:
                    fail(f"{name}[{bucket}] is not a whole number of optimizer updates")
                if value > self.effective_ceilings[bucket]:
                    fail(f"{name}[{bucket}] exceeds its ceiling {self.effective_ceilings[bucket]}")

        reserved, completed = buckets["reserved_tokens"], buckets["completed_tokens"]
        for bucket in LEDGER_BUCKETS:
            # reserved >= completed is the whole point of reserve-then-complete: a crash between
            # the two legitimately leaves reserved ahead, never behind.
            if completed[bucket] > reserved[bucket]:
                fail(f"completed_tokens[{bucket}] exceeds reserved_tokens[{bucket}]")
        if reserved["GLOBAL"] != sum(reserved[p] for p in LEDGER_PHASES):
            fail("reserved_tokens[GLOBAL] is not the sum of the per-phase reservations")
        if completed["GLOBAL"] != sum(completed[p] for p in LEDGER_PHASES):
            fail("completed_tokens[GLOBAL] is not the sum of the per-phase completions")

        reserved_updates = integer(state.get("reserved_updates"), "reserved_updates")
        completed_updates = integer(state.get("completed_updates"), "completed_updates")
        if reserved_updates < 0 or completed_updates < 0:
            fail("update counters may not be negative")
        if completed_updates > reserved_updates:
            fail("completed_updates exceeds reserved_updates")
        if reserved_updates * TRAINED_TOKENS_PER_UPDATE != reserved["GLOBAL"]:
            fail("reserved_updates disagrees with reserved_tokens[GLOBAL] under the fixed geometry")
        if completed_updates * TRAINED_TOKENS_PER_UPDATE != completed["GLOBAL"]:
            fail(
                "completed_updates disagrees with completed_tokens[GLOBAL] under the fixed geometry"
            )

    # ---------------------------------------------------------------- io

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
            if self._lock_depth:  # already held by this process; flock would deadlock
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_path, "w") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                self._lock_depth = 1
                try:
                    yield
                finally:
                    self._lock_depth = 0
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return guard()

    def _reload_locked(self) -> None:
        if self.path.is_file():
            self.state = self._validated(
                load_json_artifact(self.path, label=f"token ledger at {self.path}")
            )

    def effective_ceiling(self, key: str) -> int:
        frozen = PHASE_CEILINGS[key] if key in PHASE_CEILINGS else GLOBAL_PILOT_TOKEN_CEILING
        return min(int(frozen), int(self.effective_ceilings[key]))

    # ---------------------------------------------------------------- transitions

    def reserve(self, phase: str, tokens: int = TRAINED_TOKENS_PER_UPDATE) -> dict[str, Any]:
        """Step one: reserve BEFORE the optimizer update is applied."""
        require(phase in PHASE_CEILINGS, f"unknown phase {phase!r}")
        require(
            int(tokens) == TRAINED_TOKENS_PER_UPDATE,
            f"one optimizer update trains exactly {TRAINED_TOKENS_PER_UPDATE} tokens under the "
            f"frozen geometry; refusing to reserve {tokens!r}",
        )
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
            self._require_structural_invariants(self.state)
            self._write(self.state)
            return {
                "phase": phase,
                "reserved_tokens": int(tokens),
                "phase_reserved_after": phase_after,
                "global_reserved_after": global_after,
            }

    def complete(self, phase: str, tokens: int = TRAINED_TOKENS_PER_UPDATE) -> dict[str, Any]:
        """Step two: move an existing reservation to completed, after the update succeeded."""
        require(phase in PHASE_CEILINGS, f"unknown phase {phase!r}")
        require(
            int(tokens) == TRAINED_TOKENS_PER_UPDATE,
            f"one optimizer update trains exactly {TRAINED_TOKENS_PER_UPDATE} tokens under the "
            f"frozen geometry; refusing to complete {tokens!r}",
        )
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
            self._require_structural_invariants(self.state)
            self._write(self.state)
            return {"phase": phase, "completed_tokens_after": completed_after}

    # ---------------------------------------------------------------- reporting

    def snapshot(self) -> dict[str, Any]:
        """A FRESH, locked, revalidated view of the persisted ledger.

        R3 Part 8: parents and reports consume this after child processes have advanced the
        file, so it must never answer from stale in-memory counters. It takes the same exclusive
        lock the transitions take, reloads the bytes, revalidates identity and every structural
        invariant, and only then returns.
        """
        with self._lock():
            require(
                self.path.is_file(),
                f"token ledger has disappeared from {self.path}; its accounting cannot be read",
            )
            self.state = self._validated(
                load_json_artifact(self.path, label=f"token ledger at {self.path}")
            )
            state = self.state
            return {
                "reserved_tokens": dict(state["reserved_tokens"]),
                "completed_tokens": dict(state["completed_tokens"]),
                "reserved_updates": int(state["reserved_updates"]),
                "completed_updates": int(state["completed_updates"]),
                "effective_ceilings": dict(state["effective_ceilings"]),
                "session_hard_ceiling": int(state["session_hard_ceiling"]),
                "trained_tokens_per_update": int(state["trained_tokens_per_update"]),
                "identity": dict(self.identity),
                "reloaded_from_disk": True,
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
SESSION_SCHEMA = "petitgpt-pilot-session-v2.3"
PHASE_PLAN_SCHEMA = "petitgpt-pilot-phase-plan-v2.3"
CANDIDATE_SPEC_SCHEMA = "petitgpt-pilot-candidate-spec-v2.3"
SPEC_DIRNAME = "_specs"

PHASE_MB_PLAN_FILENAME = "PHASE_MB_PLAN.json"
PHASE_LR_INITIAL_PLAN_FILENAME = "PHASE_LR_INITIAL_PLAN.json"
PHASE_LR_CONFIRMATION_PLAN_FILENAME = "PHASE_LR_CONFIRMATION_PLAN.json"
PHASE_LR_EDGE_PLAN_FILENAME = "PHASE_LR_EDGE_PLAN.json"

PLAN_KINDS = MappingProxyType({
    "PHASE_MB_PLAN": {"phase": "MB", "filename": PHASE_MB_PLAN_FILENAME},
    "PHASE_LR_INITIAL_PLAN": {"phase": "LR", "filename": PHASE_LR_INITIAL_PLAN_FILENAME},
    "PHASE_LR_CONFIRMATION_PLAN": {
        "phase": "LR",
        "filename": PHASE_LR_CONFIRMATION_PLAN_FILENAME,
    },
    "PHASE_LR_EDGE_PLAN": {"phase": "LR", "filename": PHASE_LR_EDGE_PLAN_FILENAME},
})


def _sha256_file(path: Path) -> str:
    return file_sha256(Path(path))


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def load_json_artifact(source: Path | bytes, *, label: str) -> dict[str, Any]:
    """Decode a canonical artifact, raising BindingFailure on anything unreadable.

    R3 Part 10: an unreadable or non-object artifact is an identity binding that is broken, not
    an ordinary candidate-local problem. Left as a bare ``json.loads`` these would surface as
    ``JSONDecodeError``/``UnicodeDecodeError``/``OSError`` and be classified as a merely
    ineligible candidate, silently dropping it from the required grid.
    """
    try:
        body = Path(source).read_bytes() if not isinstance(source, bytes) else source
    except OSError as exc:
        raise BindingFailure(f"{label} could not be read: {exc}") from exc
    try:
        doc = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise BindingFailure(f"{label} is not readable canonical JSON: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise BindingFailure(f"{label} is not a JSON object")
    return dict(doc)


def validate_execution_artifacts(
    *,
    authorization_path: Path,
    pilot_index_manifest_path: Path,
    output_dir: Path,
    requested_phase: str,
    gpu_required: bool = True,
) -> dict[str, Any]:
    """THE artifact-bytes gate underneath every path to the model-training backend.

    R2: this validates from canonical artifact BYTES on disk. It trusts no in-memory object,
    no caller-supplied hash, no caller-supplied release count, no ``authorized`` flag and no
    previously constructed context.

    R3 Part 1: it no longer accepts a candidate spec. A candidate is authorized by membership in
    an immutable published PHASE PLAN, which :func:`validate_worker_execution` checks on top of
    this layer; a self-declared spec can never become executable on its own.
    """
    require(requested_phase in ("MB", "LR"), f"unknown phase {requested_phase!r}")
    require(isinstance(gpu_required, bool), "gpu_required must be a Boolean")
    require_numpy_version()

    # --- authorization manifest, from disk ---
    auth_path = Path(authorization_path)
    if not auth_path.is_file():
        raise BindingFailure(
            f"authorization manifest not found at {auth_path}; pilot execution is "
            f"NOT_AUTHORIZED under {CONTRACT_VERSION}"
        )
    auth_bytes = auth_path.read_bytes()
    authorization_sha256 = _sha256_bytes(auth_bytes)
    manifest = load_json_artifact(auth_bytes, label=f"authorization manifest at {auth_path}")

    # --- pilot index manifest FILE sha, computed from disk (never taken from the manifest) ---
    index_path = Path(pilot_index_manifest_path)
    if not index_path.is_file():
        raise BindingFailure(f"pilot index manifest not found at {index_path}")
    observed_index_file_sha = _sha256_file(index_path)
    index_manifest = load_json_artifact(index_path, label=f"pilot index manifest at {index_path}")

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
    fingerprint = base_runtime_fingerprint()
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
        "base_runtime_fingerprint": fingerprint,
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


# --------------------------------------------------------------- immutable artifact chain


def write_immutable_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish an artifact exactly once, atomically, with a SHA-256 sidecar.

    R3 Part 3: every link in the session/plan/report chain is single-publication. A rerun needs
    a new authorized output root, never an edit in place.
    """
    path = Path(path)
    if path.exists():
        raise BindingFailure(
            f"{path.name} already exists at {path}; this artifact is immutable and a rerun "
            f"requires a new authorized output root"
        )
    try:
        body = canonical_json_bytes(payload)
    except ValueError as exc:
        # canonical_json_bytes forbids NaN/Infinity: a non-finite value would make the artifact
        # unpublishable, which must surface as a named phase failure, not a raw ValueError.
        raise PhaseAbort(
            f"refusing to publish {path.name}: its payload is not canonically serializable "
            f"({exc}); a non-finite number may never enter the immutable chain"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = _sha256_bytes(body)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def read_immutable_artifact(path: Path, *, schema_version: str) -> tuple[dict[str, Any], str]:
    """Read a published artifact, re-hash its bytes and check the sidecar agrees."""
    path = Path(path)
    if not path.is_file():
        raise BindingFailure(f"required artifact is missing: {path}")
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise BindingFailure(f"required artifact {path} could not be read: {exc}") from exc
    digest = _sha256_bytes(body)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise BindingFailure(f"artifact has no published SHA-256 sidecar at {sidecar}")
    try:
        recorded = sidecar.read_text(encoding="utf-8").split()[0]
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        raise BindingFailure(f"SHA-256 sidecar at {sidecar} is unreadable or empty: {exc}") from exc
    if len(recorded) != 64 or any(c not in "0123456789abcdef" for c in recorded):
        raise BindingFailure(f"SHA-256 sidecar at {sidecar} is not a hex digest: {recorded!r}")
    if recorded != digest:
        raise BindingFailure(
            f"{path.name} SHA-256 {digest} does not match its published sidecar {recorded}"
        )
    doc = load_json_artifact(body, label=str(path))
    if doc.get("schema_version") != schema_version:
        raise BindingFailure(
            f"{path.name} schema {doc.get('schema_version')!r} != expected {schema_version!r}"
        )
    return doc, digest


def session_manifest_document(validated: Mapping[str, Any]) -> dict[str, Any]:
    """SESSION.json: the root of the immutable chain. R3 Part 3."""
    observed = validated["observed"]
    return {
        "schema_version": SESSION_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "session_id": validated["session_id"],
        "authorized_scope": validated["scope"],
        "authorization_sha256": validated["authorization_sha256"],
        "contract_sha256": observed["contract_sha256"],
        "implementation_head": observed["head"],
        "repository_branch": observed["branch"],
        "execution_bundle_sha256": observed["execution_bundle_sha256"],
        "pilot_index_manifest_file_sha256": validated["pilot_index_manifest_file_sha256"],
        "serialized_index_lists_digest": validated["serialized_index_lists_digest"],
        "accepted_releases": {
            "stage_a": {
                "release_dir": ACCEPTED_STAGE_A,
                "meta_sha256": observed["stage_a_meta_sha256"],
                "blocks": int(validated["stage_a"]["blocks"]),
            },
            "stage_b": {
                "release_dir": ACCEPTED_STAGE_B,
                "meta_sha256": observed["stage_b_meta_sha256"],
                "blocks": int(validated["stage_b"]["blocks"]),
            },
        },
        "runtime_fingerprint_sha256": validated["fingerprint"]["fingerprint_sha256"],
        "base_runtime_fingerprint": dict(validated["fingerprint"]),
        "authorized_output_root": str(validated["authorized_root"]),
        "ledger_identity": dict(validated["identity"]),
        "ledger_relpath": LEDGER_FILENAME,
        "effective_ceilings": dict(validated["effective_ceilings"]),
        "session_hard_ceiling": FULL_V2_3_PILOT_SESSION_HARD_CEILING,
        "session_budget_semantics": {
            k: (dict(v) if isinstance(v, Mapping) else v)
            for k, v in SESSION_BUDGET_SEMANTICS.items()
        },
    }


def _session_mismatches(doc: Mapping[str, Any], validated: Mapping[str, Any]) -> list[str]:
    expected = session_manifest_document(validated)
    return sorted(k for k, v in expected.items() if doc.get(k) != v)


def validate_session_manifest(path: Path, validated: Mapping[str, Any]) -> tuple[dict, str]:
    """Re-derive SESSION.json from artifact bytes and require byte-for-byte agreement."""
    doc, digest = read_immutable_artifact(path, schema_version=SESSION_SCHEMA)
    mismatched = _session_mismatches(doc, validated)
    if mismatched:
        raise BindingFailure(
            f"session manifest at {path} does not bind this execution: {mismatched}"
        )
    return doc, digest


def _relpath(path: Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def publish_candidate_spec(
    *,
    root: Path,
    candidate: Mapping[str, Any],
    session_sha256: str,
    session_id: str,
    plan_kind: str,
) -> dict[str, Any]:
    """Serialize ONE candidate as an immutable spec whose SHA the phase plan will list."""
    require(plan_kind in PLAN_KINDS, f"unknown phase-plan kind {plan_kind!r}")
    spec_dir = Path(root) / SPEC_DIRNAME
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{candidate['candidate_id']}.json"
    doc = {
        "schema_version": CANDIDATE_SPEC_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "session_sha256": session_sha256,
        "plan_kind": plan_kind,
        "phase": candidate["phase"],
        "candidate": dict(candidate),
    }
    body = canonical_json_bytes(doc)
    if spec_path.exists():
        raise BindingFailure(
            f"candidate spec already exists at {spec_path}; a candidate identity is published "
            f"once per authorized session"
        )
    tmp = spec_path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, spec_path)
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_spec_relpath": _relpath(spec_path, root),
        "candidate_spec_sha256": _sha256_bytes(body),
        "output_relpath": _relpath(Path(candidate["output_dir"]), root),
        "path": spec_path,
    }


def publish_phase_plan(
    *,
    root: Path,
    plan_kind: str,
    session_sha256: str,
    session_id: str,
    candidates: Sequence[Mapping[str, Any]],
    derived_from: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the authoritative, immutable membership list for one phase step.

    R3 Part 2/3: after this returns, NO caller chooses candidate membership. A candidate spec
    whose SHA-256 is not listed here cannot be executed.
    """
    require(plan_kind in PLAN_KINDS, f"unknown phase-plan kind {plan_kind!r}")
    require(candidates, f"{plan_kind} must enumerate at least one candidate")
    entries = [
        publish_candidate_spec(
            root=root,
            candidate=c,
            session_sha256=session_sha256,
            session_id=session_id,
            plan_kind=plan_kind,
        )
        for c in candidates
    ]
    listed = [{k: v for k, v in e.items() if k != "path"} for e in entries]
    ids = [e["candidate_id"] for e in listed]
    require(len(ids) == len(set(ids)), f"{plan_kind} lists a duplicate candidate identity: {ids}")
    plan = {
        "schema_version": PHASE_PLAN_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "plan_kind": plan_kind,
        "phase": PLAN_KINDS[plan_kind]["phase"],
        "session_id": session_id,
        "session_sha256": session_sha256,
        "derived_from": dict(derived_from),
        "candidates": listed,
        "candidate_ids": ids,
        "candidate_spec_sha256s": [e["candidate_spec_sha256"] for e in listed],
    }
    path = Path(root) / PLAN_KINDS[plan_kind]["filename"]
    plan_sha256 = write_immutable_artifact(path, plan)
    return {
        "plan": plan,
        "plan_path": path,
        "plan_sha256": plan_sha256,
        "specs": {e["candidate_id"]: e for e in entries},
    }


def validate_phase_plan(
    path: Path, *, session_sha256: str, session_id: str, expected_phase: str
) -> tuple[dict[str, Any], str]:
    """Read and bind a published phase plan. Never trusts a caller-supplied plan object."""
    doc, digest = read_immutable_artifact(path, schema_version=PHASE_PLAN_SCHEMA)
    kind = doc.get("plan_kind")
    if kind not in PLAN_KINDS:
        raise BindingFailure(f"unknown phase-plan kind {kind!r} at {path}")
    if Path(path).name != PLAN_KINDS[kind]["filename"]:
        raise BindingFailure(
            f"phase plan {kind} must be published as {PLAN_KINDS[kind]['filename']}, "
            f"found {Path(path).name}"
        )
    if doc.get("phase") != PLAN_KINDS[kind]["phase"] or doc.get("phase") != expected_phase:
        raise BindingFailure(
            f"phase plan at {path} declares phase {doc.get('phase')!r}, expected {expected_phase!r}"
        )
    if doc.get("session_sha256") != session_sha256 or doc.get("session_id") != session_id:
        raise BindingFailure(
            f"phase plan at {path} does not bind this session; a plan is valid only inside the "
            f"session that published it"
        )
    return doc, digest


# ------------------------------------------------------------ session (derived metadata)


class ExecutionSession:
    """DERIVED metadata over an already-validated artifact set. NOT execution authority.

    R3 Part 1: constructing this object grants nothing and it can no longer reach the training
    backend. Only the orchestrator uses it, to avoid re-reading the same artifacts once per
    candidate. Every real training entry re-derives its own authority from artifact PATHS via
    :func:`validate_worker_execution`.
    """

    __slots__ = ("validated", "ledger", "session_path", "session_sha256")

    def __init__(
        self,
        validated: Mapping[str, Any],
        ledger: TokenLedger,
        *,
        session_path: Path | None = None,
        session_sha256: str | None = None,
    ):
        self.validated = dict(validated)
        self.ledger = ledger
        self.session_path = Path(session_path) if session_path is not None else None
        self.session_sha256 = session_sha256

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
    pilot_index_manifest_path: Path,
    output_dir: Path,
    phase: str,
    gpu_required: bool = True,
) -> ExecutionSession:
    """Validate artifacts, publish SESSION.json, then build the ledger under the validated root."""
    validated = validate_execution_artifacts(
        authorization_path=authorization_path,
        pilot_index_manifest_path=pilot_index_manifest_path,
        output_dir=output_dir,
        requested_phase=phase,
        gpu_required=gpu_required,
    )
    root = validated["authorized_root"]
    root.mkdir(parents=True, exist_ok=True)
    session_path = root / SESSION_FILENAME
    if session_path.exists():
        doc, session_sha256 = validate_session_manifest(session_path, validated)
        if doc.get("session_id") != validated["session_id"]:
            raise BindingFailure(
                "this output root already belongs to a different session; a new authorization "
                "implies a new session and a new ledger"
            )
    else:
        session_sha256 = write_immutable_artifact(
            session_path, session_manifest_document(validated)
        )
    ledger = TokenLedger(
        root / LEDGER_FILENAME, validated["identity"], validated["effective_ceilings"]
    )
    return ExecutionSession(
        validated, ledger, session_path=session_path, session_sha256=session_sha256
    )


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


# R3 Part 11: the Moonlight RMS-matching rule the frozen Muon policy depends on, verified
# against an INDEPENDENT closed form. These shapes are the real matrix shapes the canonical
# 30x576 model presents to the Muon group plus deliberately non-square controls, so the check
# cannot pass by coincidence on square weights.
RMS_MATCHING_SHAPE_CASES = ((576, 576), (960, 576), (1536, 576), (576, 1536), (64, 4096))
RMS_MATCHING_RELATIVE_TOLERANCE = 1e-4
# The spectral-normalization guard the quintic iteration divides by, stated independently.
NEWTON_SCHULZ_NORM_EPSILON = 1e-7


def expected_rms_matched_lr(lr: float, fan_out: int, fan_in: int) -> float:
    """``adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))`` -- the frozen scaling rule.

    Written out here as a closed form. It never calls, imports or inspects the realization it
    exists to verify.
    """
    import math

    return float(lr) * RMS_MATCHING_CONSTANT * math.sqrt(max(int(fan_in), int(fan_out)))


def expected_newton_schulz_scalar_gain(
    short_side: int, *, singular_value: float = 1.0, steps: int = MUON_NS_STEPS
) -> float:
    """The closed-form gain the quintic iteration applies to a SEMI-ORTHOGONAL input.

    R3 Part 11: this is the independent half of the oracle. ``zeropower_via_newtonschulz5``
    normalizes its input by the Frobenius norm and then iterates
    ``x <- a*x + (b*(x x^T) + c*(x x^T)^2) @ x``. When the input's short-side Gram matrix is a
    multiple of the identity -- i.e. the input is a scaled semi-orthogonal matrix -- every
    iterate stays proportional to that same matrix, so the whole iteration collapses to the
    SCALAR recursion reproduced here:

        beta_0   = sigma / (||g||_F + 1e-7),   ||g||_F = sqrt(r) * sigma
        beta_k+1 = a*beta_k + b*beta_k**3 + c*beta_k**5

    with ``r`` the short side and ``(a, b, c)`` the frozen quintic coefficients. Reconstructing
    the expected update this way uses no code from ``src.optim``; an implementation that
    silently changed the coefficients, the step count or the normalization would disagree.
    """
    import math

    a, b, c = NEWTON_SCHULZ_COEFFICIENTS
    sigma = float(singular_value)
    beta = sigma / (math.sqrt(int(short_side)) * sigma + NEWTON_SCHULZ_NORM_EPSILON)
    for _ in range(int(steps)):
        beta = a * beta + b * beta**3 + c * beta**5
    return float(beta)


def _semi_orthogonal(fan_out: int, fan_in: int, seed: int) -> Any:
    """A deterministic matrix whose singular values are all exactly 1.0.

    Built with ``torch.linalg.svd`` -- a general library routine, not the code under test -- so
    its short-side Gram matrix is exactly the identity in either orientation.
    """
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    raw = torch.randn(int(fan_out), int(fan_in), generator=generator, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(raw, full_matrices=False)
    return u @ vh


def verify_rms_matching(
    *,
    lr: float = 3e-4,
    shapes: Sequence[tuple[int, int]] = RMS_MATCHING_SHAPE_CASES,
    weight_decay: float = WEIGHT_DECAY,
    momentum: float = MUON_MOMENTUM,
) -> dict[str, Any]:
    """Prove the realized Muon update applies the RMS-matched LR, per deterministic shape case.

    R3 Part 11: the oracle is INDEPENDENT. Neither half of the expected value is obtained by
    calling the implementation being verified:

    * ``expected_adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))`` is written out directly;
    * the orthogonalized direction is reconstructed from the closed-form scalar recursion in
      :func:`expected_newton_schulz_scalar_gain`, valid because the gradient handed to the
      optimizer is constructed to be exactly semi-orthogonal.

    For each shape it builds a real parameter and that gradient, runs ONE real Muon step through
    the canonical optimizer, and reconstructs

        p_after = p_before * (1 - lr * weight_decay) - adjusted_lr * beta_5 * g_used / sigma

    where ``g_used = (1 + momentum) * g`` is the Nesterov-combined gradient of a first step from
    a zero momentum buffer and ``sigma`` is its (constant) singular value. Agreement to float
    tolerance is the evidence; a realization that ignored ``max(fan_in, fan_out)`` -- or used the
    unscaled ``lr`` -- fails by orders of magnitude on every case.
    """
    import torch

    from src.optim import Muon

    rng_state = torch.get_rng_state()
    cases = []
    for index, (fan_out, fan_in) in enumerate(shapes):
        fan_out, fan_in = int(fan_out), int(fan_in)
        short_side = min(fan_out, fan_in)
        gradient_scale = 1.0 + 0.25 * index
        torch.manual_seed(0)
        # float64 parameters: the quintic iteration itself still runs in float32 (its CPU
        # branch), so the residual measures the DIRECTION, not the rounding of the decay term.
        p = torch.nn.Parameter(torch.randn(fan_out, fan_in, dtype=torch.float64))
        before = p.detach().clone()
        p.grad = gradient_scale * _semi_orthogonal(fan_out, fan_in, seed=1000 + index)
        # First step from a zero buffer: buf = g, and Nesterov combines to (1 + momentum) * g.
        g_used = (1.0 + float(momentum)) * p.grad.detach()
        sigma = gradient_scale * (1.0 + float(momentum))
        opt = Muon([
            {
                "params": [p],
                "use_muon": True,
                "lr": float(lr),
                "momentum": float(momentum),
                "nesterov": MUON_NESTEROV,
                "ns_steps": MUON_NS_STEPS,
                "weight_decay": float(weight_decay),
                "lr_ratio": 1.0,
            }
        ])
        adjusted = expected_rms_matched_lr(lr, fan_out, fan_in)
        beta = expected_newton_schulz_scalar_gain(short_side, singular_value=sigma)
        expected_direction = beta * g_used / sigma
        opt.step()
        predicted = before * (1.0 - float(lr) * float(weight_decay)) - adjusted * expected_direction
        realized_delta = p.detach() - before * (1.0 - float(lr) * float(weight_decay))
        error = float((p.detach() - predicted).abs().max())
        update_scale = float((adjusted * expected_direction).abs().max())
        # An implementation using the raw lr instead of the RMS-matched lr would differ by
        # (adjusted - lr) * |direction|; record that margin so the case is not vacuous.
        unscaled_gap = float(abs(adjusted - float(lr)) * float(expected_direction.abs().max()))
        relative_error = error / max(update_scale, 1e-30)
        cases.append({
            "fan_out": fan_out,
            "fan_in": fan_in,
            "short_side": short_side,
            "gradient_scale": gradient_scale,
            "expected_adjusted_lr": adjusted,
            "expected_newton_schulz_gain": beta,
            "sqrt_max_fan": adjusted / (float(lr) * RMS_MATCHING_CONSTANT),
            "max_abs_error": error,
            "relative_error": relative_error,
            "realized_update_max_abs": float(realized_delta.abs().max()),
            "expected_update_max_abs": update_scale,
            "unscaled_lr_would_differ_by": unscaled_gap,
            "rms_matching_factor": adjusted / float(lr),
            "matches_rms_matching_rule": relative_error <= RMS_MATCHING_RELATIVE_TOLERANCE,
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
        "oracle": (
            "independent closed form: the scaling rule is written out directly and the "
            "orthogonalized direction is reconstructed from the quintic scalar recursion on a "
            "semi-orthogonal gradient; no value is obtained from the realization under test"
        ),
        "newton_schulz_coefficients": list(NEWTON_SCHULZ_COEFFICIENTS),
        "newton_schulz_steps": MUON_NS_STEPS,
        "relative_tolerance": RMS_MATCHING_RELATIVE_TOLERANCE,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "momentum": float(momentum),
        "cases": cases,
        "all_cases_match": all(c["matches_rms_matching_rule"] for c in cases),
        "all_cases_discriminating": all(c["case_is_discriminating"] for c in cases),
    }


def verify_muon_realization(model: Any, optimizer: Any, *, peak_lr: float) -> dict[str, Any]:
    """The MANDATORY optimizer-realization precondition for a candidate's first update.

    R3 Part 11: exact grouping verification and the independent RMS-matching oracle must both
    pass before any training update is applied. A failure is a phase-level abort, never an
    ordinary ineligible candidate: the realization the whole pilot measures would be unknown.
    """
    grouping = verify_realized_grouping(optimizer, model)
    rms = verify_rms_matching(lr=float(peak_lr))
    failures = list(grouping["failures"])
    if not rms["all_cases_match"]:
        failures.append("realized Muon update does not apply the RMS-matched learning rate")
    if not rms["all_cases_discriminating"]:
        failures.append("the RMS-matching shape cases would not discriminate an unscaled update")
    if failures:
        raise PhaseAbort(
            "optimizer realization verification FAILED; no training update may be applied: "
            + ", ".join(failures)
        )
    return {
        "grouping": grouping,
        "rms_matching": rms,
        "verified": True,
        "verified_before_first_update": True,
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


class ObservedForward:
    """The exact callable the update loop invokes, wrapped so the choice becomes evidence.

    R3 Part 7: compile evidence is a runtime observation, never a Boolean copied from the
    candidate spec. This records WHICH object the training loop called and how many times, so
    ``compile=on`` proves the object returned by ``torch.compile`` was the one invoked and
    ``compile=off`` proves the uncompiled module was.
    """

    __slots__ = ("target", "compiled_object", "invocations")

    def __init__(self, target: Any, *, compiled_object: Any = None):
        self.target = target
        self.compiled_object = compiled_object
        self.invocations = 0

    @property
    def invoked_compiled_callable(self) -> bool:
        return self.compiled_object is not None and self.target is self.compiled_object

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.invocations += 1
        return self.target(*args, **kwargs)


def _run_updates(
    *,
    module: Any,
    forward: ObservedForward,
    optimizer: Any,
    view: IndexView,
    micro_bsz: int,
    grad_accum: int,
    updates: int,
    lr_fn: Any,
    ledger: TokenLedger,
    phase: str,
    device: str,
    progress: dict[str, Any],
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

    R3 Part 6: each measured update contributes a RECORD binding its own update number, its own
    trained-token count and its own synchronized wall time.

    R3 Part 9: ``progress`` is mutated in place as updates complete, so a candidate that raises
    part-way through still reports the number of updates it actually finished.
    """
    import torch

    required_blocks = int(updates) * int(SEQUENCES_PER_OPTIMIZER_UPDATE)
    require(
        len(view) >= required_blocks,
        f"{phase}: need {required_blocks} train blocks without replay, view has {len(view)}",
    )

    module.train()
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
                loss = canonical_loss(forward(x), y, m) / int(grad_accum)
            loss.backward()
            detached = loss.detach()
            accumulated = detached if accumulated is None else accumulated + detached
        gnorm = torch.nn.utils.clip_grad_norm_(module.parameters(), GRAD_CLIP)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        per_update_seconds[u] = time.perf_counter() - t0
        ledger.complete(phase)
        loss_tensors[u] = accumulated
        grad_norm_tensors[u] = gnorm.detach() if hasattr(gnorm, "detach") else gnorm
        realized_lrs[u] = realized
        progress["completed_updates"] = u
        if u >= int(timed_from):
            progress["update_timings"].append({
                "update": u,
                "trained_tokens": TRAINED_TOKENS_PER_UPDATE,
                "wall_seconds": per_update_seconds[u],
            })
        if record_diagnostics:
            diagnostics.append(_group_diagnostics(optimizer, u))
    require(
        cursor == required_blocks,
        f"{phase}: consumed {cursor} blocks, contract requires exactly {required_blocks}",
    )

    losses = {u: float(t) for u, t in loss_tensors.items()}
    grad_norms = {u: float(t) for u, t in grad_norm_tensors.items()}
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "realized_lrs": realized_lrs,
        "per_update_wall_seconds": {str(u): per_update_seconds[u] for u in per_update_seconds},
        "update_timings": list(progress["update_timings"]),
        "first_optimizer_update_wall_seconds": per_update_seconds.get(1),
        "measured_first_update": int(timed_from),
        "forward_invocations": forward.invocations,
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


def evaluate(
    module: Any, forward: Any, view: IndexView, *, micro_bsz: int, device: str
) -> dict[str, float]:
    """GLOBAL token-mean CE: return the raw numerator and weight, and the single division.

    R2 Part 7 / R3 Part 5: the raw components are what the result artifact carries, so the
    parent can recompute the loss and the SCORE instead of trusting a serialized scalar.
    """
    import torch

    module.eval()
    total_numerator, total_weight = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(view), int(micro_bsz)):
            batch = [view[i] for i in range(start, min(start + int(micro_bsz), len(view)))]
            x = to_model_ids(torch.stack([b[0] for b in batch]), device)
            y = to_model_ids(torch.stack([b[1] for b in batch]), device)
            m = torch.stack([b[2] for b in batch]).to(device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                numerator, weight = canonical_loss_components(forward(x), y, m)
            total_numerator += float(numerator)
            total_weight += float(weight)
    module.train()
    require(
        total_weight > 0.0,
        "evaluation produced no supervised tokens; a global token mean is undefined",
    )
    return {
        "numerator": total_numerator,
        "weight": total_weight,
        "loss": total_numerator / total_weight,
        "blocks": len(view),
    }


def global_token_mean(components: Sequence[tuple[float, float]]) -> float:
    """Reference for the accumulation rule: sum numerators, sum weights, divide once."""
    n = sum(float(a) for a, _ in components)
    w = sum(float(b) for _, b in components)
    require(w > 0.0, "a global token mean needs a positive total effective weight")
    return n / w


# --------------------------------------------------------------- compile-path evidence


def compile_path_evidence(
    module: Any,
    forward: ObservedForward,
    *,
    requested: bool,
    cache_dir: Path,
    expected_forward_invocations: int,
) -> dict[str, Any]:
    """Observed evidence that the REQUESTED compile mode was the one actually realized.

    R3 Part 7: not a self-reported Boolean copied from the candidate spec. The evidence is the
    identity of the callable the update loop actually invoked, how many times it invoked it,
    the realized module type, TorchDynamo's own graph counters and the Inductor artifacts the
    run left on disk.

    ``compile=on`` requires that the object returned by ``torch.compile`` was the invoked
    callable AND that compilation actually materialized (graphs or on-disk artifacts): a silent
    fallback to eager is rejected. ``compile=off`` requires that the uncompiled module was the
    invoked callable and that nothing went through Dynamo.
    """
    import torch

    module_type = f"{type(module).__module__}.{type(module).__qualname__}"
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
    invoked_compiled = bool(forward.invoked_compiled_callable)
    invoked_uncompiled = forward.compiled_object is None and forward.target is module
    invocations_match = int(forward.invocations) == int(expected_forward_invocations)
    compilation_materialized = bool(unique_graphs > 0 or artifacts)
    if requested:
        canonical = bool(
            invoked_compiled
            and is_optimized_module
            and compilation_materialized
            and invocations_match
        )
    else:
        canonical = bool(
            invoked_uncompiled
            and not is_optimized_module
            and unique_graphs == 0
            and invocations_match
        )
    return {
        "compile_requested": bool(requested),
        "realized_module_type": module_type,
        "realized_module_is_optimized_module": is_optimized_module,
        "invoked_callable_type": f"{type(forward.target).__module__}."
        f"{type(forward.target).__qualname__}",
        "invoked_compiled_callable": invoked_compiled,
        "invoked_uncompiled_module": invoked_uncompiled,
        "forward_invocations": int(forward.invocations),
        "expected_forward_invocations": int(expected_forward_invocations),
        "forward_invocations_match_geometry": invocations_match,
        "dynamo_unique_graphs": unique_graphs,
        "dynamo_graph_breaks": graph_breaks,
        "dynamo_counters": counters,
        "compilation_materialized": compilation_materialized,
        "inductor_cache_dir": str(cache),
        "inductor_artifact_count": len(artifacts),
        # True only when the realized path equals the requested path, in BOTH directions.
        "canonical_compile_path": canonical,
    }


def recheck_compile_path_evidence(result: Mapping[str, Any]) -> bool:
    """Re-derive the compile verdict from the recorded observation, at admission time.

    R3 Part 7: the eligibility/selection loader never trusts the stored
    ``canonical_compile_path`` Boolean; it recomputes it from the same recorded observations.
    """
    evidence = result.get("compile_evidence")
    if not isinstance(evidence, Mapping):
        return False
    requested = bool(result.get("compile"))
    if bool(evidence.get("compile_requested")) != requested:
        return False
    if not evidence.get("forward_invocations_match_geometry"):
        return False
    if requested:
        return bool(
            evidence.get("invoked_compiled_callable")
            and evidence.get("realized_module_is_optimized_module")
            and evidence.get("compilation_materialized")
        )
    return bool(
        evidence.get("invoked_uncompiled_module")
        and not evidence.get("realized_module_is_optimized_module")
        and int(evidence.get("dynamo_unique_graphs", -1)) == 0
    )


def _reset_dynamo_counters() -> None:
    import torch

    try:
        torch._dynamo.utils.counters.clear()
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------ the real worker (artifact paths only)

REAL_WORKER_ARTIFACT_INPUTS = (
    "authorization_path",
    "session_manifest_path",
    "phase_plan_path",
    "candidate_spec_path",
    "pilot_index_manifest_path",
    "accepted_stage_a_path",
    "accepted_stage_b_path",
    "ledger_path",
    "candidate_output_path",
)

_WORKER_AUTHORITY_MINT = object()


class WorkerAuthority:
    """Proof that the artifact bytes on disk were revalidated by THIS process.

    R3 Part 1: only :func:`validate_worker_execution` mints one. It cannot be constructed by a
    caller, so no hand-assembled session, context object or ``authorized`` flag can reach model
    construction, a forward, a backward or an optimizer update.
    """

    __slots__ = (
        "validated",
        "candidate",
        "spec",
        "spec_sha256",
        "spec_path",
        "plan",
        "plan_sha256",
        "plan_path",
        "session_doc",
        "session_sha256",
        "session_path",
        "ledger",
        "output_dir",
        "train_order",
        "phase",
    )

    def __init__(self, mint: Any, **fields: Any):
        if mint is not _WORKER_AUTHORITY_MINT:
            raise BindingFailure(
                "WorkerAuthority is minted only by validate_worker_execution() after the "
                "canonical artifact bytes have been revalidated; it cannot be constructed"
            )
        for key in self.__slots__:
            setattr(self, key, fields[key])

    @property
    def session_id(self) -> str:
        return str(self.validated["session_id"])

    @property
    def scope(self) -> str:
        return str(self.validated["scope"])


def verify_mb_report_document(report: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive the frozen Phase-MB geometry from a report's OWN candidate evidence.

    Session-free so both the orchestrator and each candidate worker can use it. It recomputes
    every eligible candidate's throughput from its raw per-update timing records, re-derives the
    selection ladder, and refuses a published selection it cannot reproduce.
    """
    candidates = report.get("candidates") or []
    require_complete_mb_grid(candidates)
    vram = report.get("physical_vram_bytes")
    if not isinstance(vram, int) or isinstance(vram, bool) or vram <= 0:
        raise BindingFailure(f"Phase-MB report records an unusable physical VRAM {vram!r}")
    verified = []
    for record in candidates:
        if record.get("eligible"):
            verified.append({**record, **verify_recomputed_mb_result(record)})
        else:
            verified.append(dict(record))
    reselected = mb_select(verified, vram)
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
        "selection": reselected,
        "micro_bsz": int(reselected["FROZEN_MICRO_BSZ"]),
        "grad_accum": int(reselected["FROZEN_GRAD_ACCUM"]),
        "compile": bool(reselected["FROZEN_COMPILE"]),
    }


def _recomputed_lr_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Recompute every eligible LR record from its own raw evaluation components."""
    out = []
    for record in records or ():
        if record.get("eligible"):
            out.append({**record, **verify_recomputed_lr_result(record)})
        else:
            out.append(dict(record))
    return out


def _bound_report(
    root: Path, derived_from: Mapping[str, Any], keys: tuple[str, str], label: str, schema: str
) -> dict[str, Any]:
    """Open the artifact a plan says it derives from, after re-hashing its exact bytes."""
    relpath, digest = derived_from.get(keys[0]), derived_from.get(keys[1])
    if not isinstance(relpath, str) or not isinstance(digest, str):
        raise BindingFailure(
            f"the plan does not bind the {label} it must be derived from ({keys[0]}/{keys[1]})"
        )
    _require_referenced_artifact(root, relpath, digest, label)
    document, _ = read_immutable_artifact(Path(root) / relpath, schema_version=schema)
    return document


def derive_planned_candidates(
    *, plan: Mapping[str, Any], authorized_root: Path
) -> list[dict[str, Any]]:
    """Re-derive, from the contract and the BOUND evidence, exactly the candidates a plan may hold.

    R3 Part 2/3: the worker never accepts a plan's own declared geometry or LR list as the
    derivation. For Phase MB it regenerates the frozen ten from the contract. For every Phase-LR
    kind it opens the Phase-MB report the plan binds, re-derives ``FROZEN_MICRO_BSZ`` /
    ``FROZEN_GRAD_ACCUM`` / ``FROZEN_COMPILE`` from that report's own candidate evidence, and
    derives the LR set from the contract plus the bound preceding report:

        INITIAL        the frozen seed-1 grid, at seed-1
        CONFIRMATION   [seed-1 winner, confirmation_neighbor(winner)] at seed-2, where the
                       winner is re-derived from the bound initial report's recomputed records
        EDGE           [edge_candidate(confirmed LR)] at both seeds, where the confirmed LR is
                       re-derived from the bound confirmation report's own pairs

    A plan whose declared ``derived_from`` disagrees with any of that is refused, so the
    comparison against the published spec can never be a tautology.
    """
    kind = str(plan["plan_kind"])
    root = Path(authorized_root)
    derived_from = plan.get("derived_from") or {}
    if kind == "PHASE_MB_PLAN":
        return plan_phase_mb(output_root=root)

    # --- the frozen geometry, re-derived from the Phase-MB report this plan binds ---
    mb_report = _bound_report(
        root,
        derived_from,
        ("phase_mb_report_relpath", "phase_mb_report_sha256"),
        "phase_mb_report",
        MB_REPORT_SCHEMA,
    )
    frozen = verify_mb_report_document(mb_report)
    declared = derived_from.get("frozen_geometry")
    expected_geometry = {
        "micro_bsz": frozen["micro_bsz"],
        "grad_accum": frozen["grad_accum"],
        "compile": frozen["compile"],
    }
    if not isinstance(declared, Mapping) or dict(declared) != expected_geometry:
        raise BindingFailure(
            f"{kind} declares geometry {declared!r}, but the Phase-MB report it binds freezes "
            f"{expected_geometry!r}"
        )

    # --- the LR set, re-derived from the contract and the bound preceding evidence ---
    if kind == "PHASE_LR_INITIAL_PLAN":
        expected_lrs = [float(v) for v in LR_GRID_SEED1]
        expected_seeds = ["seed-1"]
    elif kind == "PHASE_LR_CONFIRMATION_PLAN":
        previous = _bound_report(
            root,
            derived_from,
            ("preceding_report_relpath", "preceding_report_sha256"),
            "preceding_lr_report",
            LR_REPORT_SCHEMA,
        )
        seed1 = _recomputed_lr_records(previous.get("seed1") or ())
        verdict = lr_select_seed1(seed1)
        if verdict != previous.get("selection"):
            raise BindingFailure(
                "re-deriving the seed-1 selection from the bound initial report's recomputed "
                "evidence does not reproduce its published selection"
            )
        if verdict["outcome"] != "SEED1_WINNER":
            raise PhaseAbort(
                f"the bound initial report did not name a seed-1 winner ({verdict['outcome']}); "
                f"no confirmation candidate can be derived"
            )
        winner = float(verdict["winner_peak_lr"])
        expected_lrs = [winner, confirmation_neighbor(winner)]
        expected_seeds = ["seed-2"]
    elif kind == "PHASE_LR_EDGE_PLAN":
        previous = _bound_report(
            root,
            derived_from,
            ("preceding_report_relpath", "preceding_report_sha256"),
            "preceding_lr_report",
            LR_REPORT_SCHEMA,
        )
        confirmed = lr_confirm(previous.get("confirmation_pairs") or ())
        if confirmed != previous.get("selection"):
            raise BindingFailure(
                "re-deriving the confirmation outcome from the bound confirmation report's own "
                "pairs does not reproduce its published selection"
            )
        if confirmed["outcome"] != "CONFIRMED":
            raise PhaseAbort(
                f"the bound confirmation report did not confirm an LR ({confirmed['outcome']}); "
                f"no edge candidate can be derived"
            )
        edge = edge_candidate(float(confirmed["confirmed_peak_lr"]))
        if edge is None:
            raise BindingFailure(
                f"the confirmed LR {confirmed['confirmed_peak_lr']} defines no edge expansion; "
                f"an edge plan may not exist"
            )
        expected_lrs = [float(edge)]
        expected_seeds = ["seed-1", "seed-2"]
    else:  # pragma: no cover - PLAN_KINDS is closed and validated upstream
        raise BindingFailure(f"unknown phase-plan kind {kind!r}")

    declared_lrs = derived_from.get("peak_lrs")
    declared_seeds = derived_from.get("seed_labels")
    if [float(v) for v in (declared_lrs or ())] != expected_lrs:
        raise BindingFailure(
            f"{kind} declares peak LRs {declared_lrs!r}, but the contract derives {expected_lrs!r} "
            f"from the evidence it binds"
        )
    if list(declared_seeds or ()) != expected_seeds:
        raise BindingFailure(
            f"{kind} declares seeds {declared_seeds!r}, but the contract derives {expected_seeds!r}"
        )

    out: list[dict[str, Any]] = []
    for seed_label in expected_seeds:
        out.extend(
            plan_phase_lr(
                output_root=root,
                micro_bsz=frozen["micro_bsz"],
                compile_on=frozen["compile"],
                peak_lrs=expected_lrs,
                seed_label=seed_label,
            )
        )
    return out


def _require_referenced_artifact(
    root: Path, relpath: Any, expected_sha256: Any, label: str
) -> None:
    if not isinstance(relpath, str) or not isinstance(expected_sha256, str):
        raise BindingFailure(f"{label} is not recorded as a path plus SHA-256")
    root = Path(root).resolve()
    path = (root / relpath).resolve()
    if root not in path.parents:
        raise BindingFailure(f"{label} resolves outside the authorized root: {path}")
    if not path.is_file():
        raise BindingFailure(f"{label} names a missing artifact: {path}")
    if _sha256_file(path) != expected_sha256:
        raise BindingFailure(f"{label} SHA-256 does not match the bytes at {path}")


def validate_worker_execution(
    *,
    authorization_path: Path,
    session_manifest_path: Path,
    phase_plan_path: Path,
    candidate_spec_path: Path,
    pilot_index_manifest_path: Path,
    accepted_stage_a_path: Path,
    accepted_stage_b_path: Path,
    ledger_path: Path,
    candidate_output_path: Path,
    gpu_required: bool = True,
) -> WorkerAuthority:
    """THE single gate to the real model-training backend. Canonical artifact PATHS only.

    R3 Parts 1-4: it accepts no session object, no validated context, no ``authorized`` flag and
    no caller-supplied hash. It revalidates the authorization manifest, the pilot-index
    manifest, both accepted releases, the runtime, the authorized root, the immutable session
    manifest, the immutable phase plan, this candidate's immutable spec, the spec's membership
    in that plan, every candidate field against a fresh contract derivation, and the ledger --
    all from bytes on disk -- before minting the authority that the training entry requires.
    """
    spec_path = Path(candidate_spec_path)
    if not spec_path.is_file():
        raise BindingFailure(f"candidate spec not found at {spec_path}")
    spec_bytes = spec_path.read_bytes()
    spec_sha256 = _sha256_bytes(spec_bytes)
    spec = load_json_artifact(spec_bytes, label=f"candidate spec at {spec_path}")
    if spec.get("schema_version") != CANDIDATE_SPEC_SCHEMA:
        raise BindingFailure(f"candidate spec schema mismatch at {spec_path}")
    if spec.get("contract_version") != CONTRACT_VERSION:
        raise BindingFailure(f"candidate spec contract version mismatch at {spec_path}")
    candidate = dict(spec.get("candidate") or {})
    phase = str(spec.get("phase") or candidate.get("phase") or "")
    if phase not in ("MB", "LR") or candidate.get("phase") != phase:
        raise BindingFailure(f"candidate spec declares an unusable phase {phase!r}")

    # The authorized root is the parent of this candidate's own output directory; the
    # authorization manifest's allowed_output_root is what decides whether that root is legal.
    output_dir = Path(candidate_output_path).resolve()
    authorized_root = output_dir.parent

    validated = validate_execution_artifacts(
        authorization_path=authorization_path,
        pilot_index_manifest_path=pilot_index_manifest_path,
        output_dir=authorized_root,
        requested_phase=phase,
        gpu_required=gpu_required,
    )
    root = validated["authorized_root"]

    repo = repo_root()
    for supplied, relative, stage in (
        (accepted_stage_a_path, ACCEPTED_STAGE_A, "stage_a"),
        (accepted_stage_b_path, ACCEPTED_STAGE_B, "stage_b"),
    ):
        if Path(supplied).resolve() != (repo / relative).resolve():
            raise BindingFailure(
                f"accepted {stage} release path {supplied} is not the frozen release {relative}"
            )

    if Path(session_manifest_path).resolve() != (root / SESSION_FILENAME).resolve():
        raise BindingFailure(
            f"the session manifest must be {root / SESSION_FILENAME}, got {session_manifest_path}"
        )
    session_doc, session_sha256 = validate_session_manifest(Path(session_manifest_path), validated)
    if spec.get("session_id") != validated["session_id"] or (
        spec.get("session_sha256") != session_sha256
    ):
        raise BindingFailure(
            "the candidate spec does not bind the session manifest this process revalidated"
        )

    plan_path = Path(phase_plan_path).resolve()
    if plan_path.parent != root:
        raise BindingFailure(f"the phase plan must live directly under {root}, got {plan_path}")
    plan, plan_sha256 = validate_phase_plan(
        plan_path,
        session_sha256=session_sha256,
        session_id=validated["session_id"],
        expected_phase=phase,
    )
    if spec.get("plan_kind") != plan.get("plan_kind"):
        raise BindingFailure(
            f"candidate spec plan_kind {spec.get('plan_kind')!r} != the plan it was launched "
            f"against ({plan.get('plan_kind')!r})"
        )

    entries = [e for e in (plan.get("candidates") or []) if isinstance(e, Mapping)]
    matched = [e for e in entries if e.get("candidate_spec_sha256") == spec_sha256]
    if not matched:
        raise BindingFailure(
            f"candidate spec {spec_path} (sha256 {spec_sha256}) is not listed in "
            f"{plan['plan_kind']}; a self-declared candidate is never executable"
        )
    if len(matched) != 1:
        raise BindingFailure(f"{plan['plan_kind']} lists the same candidate spec more than once")
    entry = matched[0]
    if entry.get("candidate_id") != candidate.get("candidate_id"):
        raise BindingFailure("the planned candidate identity does not match the spec it points at")
    if (root / str(entry.get("candidate_spec_relpath"))).resolve() != spec_path.resolve():
        raise BindingFailure("the candidate spec is not at the path the phase plan lists for it")
    if (root / str(entry.get("output_relpath"))).resolve() != output_dir:
        raise BindingFailure(
            f"candidate output {output_dir} is not the directory the phase plan assigns to "
            f"{candidate.get('candidate_id')!r}"
        )

    # Every candidate field is re-derived from the contract and from the evidence the plan binds
    # (which derive_planned_candidates re-hashes and re-verifies), never echoed back from the plan.
    expected = {
        c["candidate_id"]: c for c in derive_planned_candidates(plan=plan, authorized_root=root)
    }
    if candidate.get("candidate_id") not in expected:
        raise BindingFailure(
            f"candidate {candidate.get('candidate_id')!r} is not one the contract derives for "
            f"{plan['plan_kind']}"
        )
    if candidate != expected[candidate["candidate_id"]]:
        differing = sorted(
            k
            for k in set(candidate) | set(expected[candidate["candidate_id"]])
            if candidate.get(k) != expected[candidate["candidate_id"]].get(k)
        )
        raise BindingFailure(
            f"candidate {candidate['candidate_id']!r} fields differ from the contract "
            f"derivation: {differing}"
        )

    if Path(ledger_path).resolve() != (root / LEDGER_FILENAME).resolve():
        raise BindingFailure(
            f"the token ledger must be {root / LEDGER_FILENAME}, got {ledger_path}"
        )
    if not Path(ledger_path).is_file():
        raise BindingFailure(
            f"no token ledger at {ledger_path}; a candidate never creates the session's "
            f"accounting file, it joins the one its authorized session opened"
        )
    # Nothing is written and no accounting is opened until the output directory is proven to be
    # the one this plan assigns, and to not already exist.
    require_candidate_output_dir(output_dir, root)
    ledger = TokenLedger(Path(ledger_path), validated["identity"], validated["effective_ceilings"])
    order = train_order(
        validated["indices"]["stage_a_train"],
        SEED_SEMANTICS[candidate["seed_label"]]["train_order"],
    )
    return WorkerAuthority(
        _WORKER_AUTHORITY_MINT,
        validated=validated,
        candidate=candidate,
        spec=spec,
        spec_sha256=spec_sha256,
        spec_path=spec_path,
        plan=plan,
        plan_sha256=plan_sha256,
        plan_path=plan_path,
        session_doc=session_doc,
        session_sha256=session_sha256,
        session_path=Path(session_manifest_path),
        ledger=ledger,
        output_dir=output_dir,
        train_order=order,
        phase=phase,
    )


def _require_worker_authority(worker: Any) -> WorkerAuthority:
    if not isinstance(worker, WorkerAuthority):
        raise BindingFailure(
            "the real training backend accepts only a WorkerAuthority minted by "
            "validate_worker_execution() from canonical artifact paths"
        )
    return worker


def _train_phase_mb(worker: Any, progress: dict[str, Any]) -> dict[str, Any]:
    """One Phase-MB probe, end to end, inside a revalidated candidate process."""
    import torch

    worker = _require_worker_authority(worker)
    candidate = worker.candidate
    require(str(candidate.get("phase")) == "MB", "Phase-MB backend received a non-MB candidate")
    out = worker.output_dir
    out.mkdir(parents=True)
    cache_dir = out / "inductor_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    _reset_dynamo_counters()
    torch.cuda.reset_peak_memory_stats()

    module = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    torch_compile_wrapper_seconds = None
    compiled_object = None
    if candidate["compile"]:
        t0 = time.perf_counter()
        compiled_object = torch.compile(module)
        torch_compile_wrapper_seconds = time.perf_counter() - t0
        module = compiled_object
    forward = ObservedForward(module, compiled_object=compiled_object)
    optimizer = build_pilot_optimizer(module, candidate["peak_lr"])
    # R3 Part 11: exact optimizer-realization verification is a PRECONDITION for update 1.
    realization = verify_muon_realization(module, optimizer, peak_lr=float(candidate["peak_lr"]))
    view = IndexView(worker.validated["stage_a"]["dataset"], worker.train_order)

    result = _run_updates(
        module=module,
        forward=forward,
        optimizer=optimizer,
        view=view,
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=MB_PROBE_UPDATES,
        lr_fn=mb_lr,
        ledger=worker.ledger,
        phase="MB",
        device="cuda",
        progress=progress,
        timed_from=MB_MEASURED_FIRST_UPDATE,
    )
    grouping = verify_realized_grouping(optimizer, module)
    states = verify_optimizer_state(optimizer)
    compile_evidence = compile_path_evidence(
        module,
        forward,
        requested=bool(candidate["compile"]),
        cache_dir=cache_dir,
        expected_forward_invocations=MB_PROBE_UPDATES * int(candidate["grad_accum"]),
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
        # R3 Part 6: the per-update timing records the parent recomputes the median from. The
        # serialized median below is a convenience the parent independently re-derives.
        "update_timings": result["update_timings"],
        "measured_first_update": result["measured_first_update"],
        "per_update_wall_seconds": result["per_update_wall_seconds"],
        "median_update_tokens_per_second": mb_median_update_tokens_per_second(
            result["update_timings"]
        ),
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
        "optimizer_realization_verified": realization["verified"],
        "rms_matching": realization["rms_matching"],
    }
    return _publish_result(payload, worker)


def _train_phase_lr(worker: Any, progress: dict[str, Any]) -> dict[str, Any]:
    """One Phase-Muon-LR run, end to end, including both global-mean evaluations."""
    import torch

    from pretrain.pilot_contract_v2_3 import sustained_divergence

    worker = _require_worker_authority(worker)
    candidate = worker.candidate
    validated = worker.validated
    require(str(candidate.get("phase")) == "LR", "Phase-LR backend received a non-LR candidate")
    out = worker.output_dir
    out.mkdir(parents=True)
    cache_dir = out / "inductor_cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    _reset_dynamo_counters()

    module = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    torch_compile_wrapper_seconds = None
    compiled_object = None
    if candidate["compile"]:
        t0 = time.perf_counter()
        compiled_object = torch.compile(module)
        torch_compile_wrapper_seconds = time.perf_counter() - t0
        module = compiled_object
    forward = ObservedForward(module, compiled_object=compiled_object)
    optimizer = build_pilot_optimizer(module, candidate["peak_lr"])
    realization = verify_muon_realization(module, optimizer, peak_lr=float(candidate["peak_lr"]))
    view = IndexView(validated["stage_a"]["dataset"], worker.train_order)
    result = _run_updates(
        module=module,
        forward=forward,
        optimizer=optimizer,
        view=view,
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=LR_RUN_UPDATES,
        lr_fn=lambda u: lr_schedule(u, candidate["peak_lr"]),
        ledger=worker.ledger,
        phase="LR",
        device="cuda",
        progress=progress,
        record_diagnostics=True,
    )
    eval_a = IndexView(validated["stage_a"]["dataset"], validated["indices"]["stage_a_eval"])
    eval_b = IndexView(validated["stage_b"]["dataset"], validated["indices"]["stage_b_eval"])
    a = evaluate(module, forward, eval_a, micro_bsz=candidate["micro_bsz"], device="cuda")
    b = evaluate(module, forward, eval_b, micro_bsz=candidate["micro_bsz"], device="cuda")
    grouping = verify_realized_grouping(optimizer, module)
    states = verify_optimizer_state(optimizer)
    guard = sustained_divergence(result["losses"])
    # R3 Part 7: the expectation is DERIVED from the frozen geometry and the fixed evaluation
    # block counts -- never measured back off the wrapper it is meant to check.
    micro = int(candidate["micro_bsz"])
    expected_eval_forwards = -(-len(eval_a) // micro) + -(-len(eval_b) // micro)
    compile_evidence = compile_path_evidence(
        module,
        forward,
        requested=bool(candidate["compile"]),
        cache_dir=cache_dir,
        expected_forward_invocations=(
            LR_RUN_UPDATES * int(candidate["grad_accum"]) + expected_eval_forwards
        ),
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
        "update_timings": result["update_timings"],
        "torch_compile_wrapper_seconds": torch_compile_wrapper_seconds,
        "first_optimizer_update_wall_seconds": result["first_optimizer_update_wall_seconds"],
        "all_losses_finite": all(map(_finite, result["losses"].values())),
        "all_grad_norms_finite": all(map(_finite, result["grad_norms"].values())),
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in module.parameters()),
        "muon_momentum_states_present": states["all_states_instantiated"],
        "aux_adamw_states_present": states["all_states_instantiated"],
        "grouping_matches_contract": grouping["matches_frozen_realization"],
        "all_lr_ratios_are_one": grouping["all_lr_ratios_are_one"],
        # R3 Part 5: raw evaluation components. SCORE is recomputed from these by the parent.
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
        "optimizer_realization_verified": realization["verified"],
        "diagnostics": result["diagnostics"][-1] if result["diagnostics"] else None,
        "rms_matching": realization["rms_matching"],
    }
    return _publish_result(payload, worker)


REAL_TRAINING_ENTRYPOINTS = (_train_phase_mb, _train_phase_lr)


def execute_validated_candidate(worker: Any, progress: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the real training backend for an already-minted worker authority."""
    worker = _require_worker_authority(worker)
    backend = _train_phase_mb if worker.phase == "MB" else _train_phase_lr
    return backend(worker, progress)


def _finite(v: float) -> bool:
    import math

    return math.isfinite(float(v))


def _publish_result(payload: Mapping[str, Any], worker: WorkerAuthority) -> dict[str, Any]:
    """Write run_meta and the result, binding every identity selection will later require."""
    worker = _require_worker_authority(worker)
    validated = worker.validated
    out = worker.output_dir
    meta = run_meta(worker)
    meta_bytes = canonical_json_bytes(meta)
    (out / "run_meta.json").write_bytes(meta_bytes)
    bound = dict(payload)
    bound.update({
        "eligible": True,
        "run_meta_sha256": _sha256_bytes(meta_bytes),
        "candidate_spec_sha256": worker.spec_sha256,
        "phase_plan_sha256": worker.plan_sha256,
        "phase_plan_kind": worker.plan["plan_kind"],
        "session_sha256": worker.session_sha256,
        "contract_sha256": validated["observed"]["contract_sha256"],
        "implementation_head": validated["observed"]["head"],
        "execution_bundle_sha256": validated["observed"]["execution_bundle_sha256"],
        "serialized_index_lists_digest": validated["serialized_index_lists_digest"],
        "pilot_index_manifest_file_sha256": validated["pilot_index_manifest_file_sha256"],
        "runtime_fingerprint_sha256": validated["fingerprint"]["fingerprint_sha256"],
        "authorization_sha256": validated["authorization_sha256"],
        "session_id": worker.session_id,
        "ledger_identity": dict(worker.ledger.identity),
        "ledger_snapshot": worker.ledger.snapshot(),
        "output_dir": str(out),
    })
    (out / "result.json").write_bytes(canonical_json_bytes(bound))
    return bound


# --------------------------------------------------------------- independent recomputation

# Serialized doubles round-trip exactly through canonical JSON, so a recomputed value and the
# value the producer serialized must agree to within double round-off, not to a loose band.
SERIALIZATION_RELATIVE_TOLERANCE = 1e-12


def _agrees(recomputed: float, serialized: Any) -> bool:
    import math

    if not isinstance(serialized, (int, float)) or isinstance(serialized, bool):
        return False
    return math.isclose(
        float(recomputed),
        float(serialized),
        rel_tol=SERIALIZATION_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    )


def verify_recomputed_mb_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """R3 Part 6: recompute Phase-MB throughput from the raw per-update timing records.

    The median is taken over the per-update RATES for updates 11..40, one record each. A stored
    median that disagrees -- including one computed as tokens / median(seconds) -- is refused.
    """
    cid = result.get("candidate_id")
    try:
        records = require_exact_mb_timing_records(result.get("update_timings") or ())
    except PilotContractError as exc:
        raise BindingFailure(
            f"Phase-MB candidate {cid!r} timing evidence is unusable: {exc}"
        ) from exc
    rates = [mb_update_tokens_per_second(r) for r in records]
    recomputed = mb_median_update_tokens_per_second(records)
    serialized = result.get("median_update_tokens_per_second")
    if not _agrees(recomputed, serialized):
        raise BindingFailure(
            f"Phase-MB candidate {cid!r}: serialized median_update_tokens_per_second "
            f"{serialized!r} disagrees with the median of the per-update rates recomputed from "
            f"the raw timing records ({recomputed!r})"
        )
    # R3 Part 7: the verdict is re-derived from the recorded observation. A candidate that
    # honestly recorded a silent fallback is INELIGIBLE (the contract's own rule); only a
    # stored verdict that disagrees with its own evidence is a binding failure.
    observed_compile_path = recheck_compile_path_evidence(result)
    if bool(result.get("canonical_compile_path")) != observed_compile_path:
        raise BindingFailure(
            f"Phase-MB candidate {cid!r}: stored canonical_compile_path "
            f"{result.get('canonical_compile_path')!r} does not support the verdict recomputed "
            f"from its recorded compile-path observation ({observed_compile_path!r})"
        )
    return {
        "median_update_tokens_per_second": recomputed,
        "measured_updates": len(records),
        "measured_update_ids": [r["update"] for r in records],
        "update_tokens_per_second": rates,
        "canonical_compile_path": observed_compile_path,
    }


def verify_recomputed_lr_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """R3 Part 5: recompute both eval losses and SCORE from the raw numerators and weights."""
    cid = result.get("candidate_id")
    components = {}
    for stage in ("a", "b"):
        numerator = result.get(f"eval_stage_{stage}_numerator")
        weight = result.get(f"eval_stage_{stage}_weight")
        if (
            not isinstance(numerator, (int, float))
            or isinstance(numerator, bool)
            or not math.isfinite(float(numerator))
        ):
            raise BindingFailure(f"LR candidate {cid!r} has no raw stage-{stage} eval numerator")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise BindingFailure(f"LR candidate {cid!r} has no positive stage-{stage} eval weight")
        loss = float(numerator) / float(weight)
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
    observed_compile_path = recheck_compile_path_evidence(result)
    if bool(result.get("canonical_compile_path")) != observed_compile_path:
        raise BindingFailure(
            f"LR candidate {cid!r}: stored canonical_compile_path "
            f"{result.get('canonical_compile_path')!r} does not support the verdict recomputed "
            f"from its recorded compile-path observation ({observed_compile_path!r})"
        )
    return {
        "eval_loss_stage_a": components["a"],
        "eval_loss_stage_b": components["b"],
        "score": score,
        "score_weights": list(LR_SCORE_WEIGHTS),
        "score_formula": "(10*loss_A + 3*loss_B)/13",
        "canonical_compile_path": observed_compile_path,
    }


# --------------------------------------------------------------------- orchestrator

MB_REPORT_SCHEMA = "petitgpt-pilot-phase-mb-report-v2.3"
LR_REPORT_SCHEMA = "petitgpt-pilot-phase-lr-report-v2.3"
MB_REPORT_FILENAME = "PHASE_MB_REPORT.json"
LR_REPORT_FILENAME = "PHASE_LR_REPORT.json"
LR_INITIAL_REPORT_FILENAME = "PHASE_LR_INITIAL_REPORT.json"
LR_CONFIRMATION_REPORT_FILENAME = "PHASE_LR_CONFIRMATION_REPORT.json"
LR_EDGE_REPORT_FILENAME = "PHASE_LR_EDGE_REPORT.json"

REQUIRED_RESULT_BINDINGS = (
    "phase",
    "candidate_id",
    "seed_label",
    "run_meta_sha256",
    "candidate_spec_sha256",
    "phase_plan_sha256",
    "session_sha256",
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

# R3 Part 4: the fields run_meta.json must agree on with BOTH the result artifact and the
# planned candidate identity before the candidate may inform an authoritative selection.
RUN_META_RESULT_BINDINGS = (
    "phase",
    "candidate_id",
    "candidate_spec_sha256",
    "phase_plan_sha256",
    "session_sha256",
    "micro_bsz",
    "grad_accum",
    "compile",
    "seed_label",
    "output_dir",
    "session_id",
    "authorization_sha256",
    "contract_sha256",
    "implementation_head",
    "execution_bundle_sha256",
    "pilot_index_manifest_file_sha256",
    "runtime_fingerprint_sha256",
)
RUN_META_CANDIDATE_BINDINGS = (
    "phase",
    "candidate_id",
    "micro_bsz",
    "grad_accum",
    "compile",
    "seed_label",
    "model_seed",
    "train_order_seed",
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


def load_completed_result(
    session: ExecutionSession, *, planned: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Load one candidate's evidence and bind it to its run_meta file and its PLANNED identity.

    ``plan`` is the published-plan bundle returned by :func:`publish_phase_plan`: the plan
    document, its path, its SHA-256 and the specs it lists.

    R3 Part 4: ``run_meta.json`` is opened from the candidate's own output directory, hashed
    from disk, and required to agree with both the result artifact and the candidate the phase
    plan authorized. Unknown or mismatched metadata rejects the candidate from authoritative
    selection.
    """
    document = plan["plan"]
    output_dir = Path(planned["output_dir"])
    path = output_dir / "result.json"
    if not path.is_file():
        raise BindingFailure(f"no completed result artifact at {path}")
    result = load_json_artifact(path, label=f"candidate result at {path}")
    missing = [f for f in REQUIRED_RESULT_BINDINGS if f not in result]
    if missing:
        raise BindingFailure(f"result artifact is missing required bindings: {missing}")
    for field, expected in _session_bindings(session):
        if result.get(field) != expected:
            raise BindingFailure(f"result artifact {field} does not bind this execution")
    if result.get("ledger_identity") != dict(session.ledger.identity):
        raise BindingFailure("result artifact ledger identity does not bind this execution")

    # --- the planned identity the phase plan authorized ---
    entry = next(
        (
            e
            for e in (document.get("candidates") or [])
            if e.get("candidate_id") == planned["candidate_id"]
        ),
        None,
    )
    if entry is None:
        raise BindingFailure(
            f"candidate {planned['candidate_id']!r} is not listed in {document.get('plan_kind')!r}"
        )
    planned_identity = {
        "phase": planned["phase"],
        "candidate_id": planned["candidate_id"],
        "candidate_spec_sha256": entry["candidate_spec_sha256"],
        "phase_plan_sha256": plan["plan_sha256"],
        "session_sha256": session.session_sha256,
        "micro_bsz": planned["micro_bsz"],
        "grad_accum": planned["grad_accum"],
        "compile": bool(planned["compile"]),
        "seed_label": planned["seed_label"],
        "output_dir": str(Path(planned["output_dir"]).resolve()),
        "session_id": session.session_id,
    }
    for field, expected in planned_identity.items():
        actual = result.get(field)
        if field == "output_dir":
            actual = str(Path(str(actual)).resolve()) if actual is not None else None
        if actual != expected:
            raise BindingFailure(
                f"result artifact {field} {actual!r} does not match the planned candidate "
                f"identity {expected!r}"
            )

    # --- the real run_meta.json file, hashed from disk ---
    meta_path = output_dir / "run_meta.json"
    if not meta_path.is_file():
        raise BindingFailure(
            f"candidate {planned['candidate_id']!r} has no run_meta.json at {meta_path}; a "
            f"result without real run metadata may not inform an authoritative selection"
        )
    meta_bytes = meta_path.read_bytes()
    meta_sha256 = _sha256_bytes(meta_bytes)
    if result.get("run_meta_sha256") != meta_sha256:
        raise BindingFailure(
            f"candidate {planned['candidate_id']!r}: result run_meta_sha256 "
            f"{result.get('run_meta_sha256')!r} does not match the SHA-256 of the bytes at "
            f"{meta_path} ({meta_sha256})"
        )
    meta = load_json_artifact(meta_bytes, label=f"run metadata at {meta_path}")
    if meta.get("schema_version") != RUN_META_SCHEMA:
        raise BindingFailure(f"run_meta.json schema mismatch at {meta_path}")
    if meta.get("contract_version") != CONTRACT_VERSION:
        raise BindingFailure(f"run_meta.json contract version mismatch at {meta_path}")
    for field in RUN_META_RESULT_BINDINGS:
        if field not in meta:
            raise BindingFailure(f"run_meta.json at {meta_path} is missing {field!r}")
        expected = result.get(field)
        actual = meta.get(field)
        if field == "output_dir":
            expected = str(Path(str(expected)).resolve())
            actual = str(Path(str(actual)).resolve()) if actual is not None else None
        if actual != expected:
            raise BindingFailure(
                f"run_meta.json {field} {actual!r} disagrees with the result artifact {expected!r}"
            )
    planned_meta = {
        "phase": planned["phase"],
        "candidate_id": planned["candidate_id"],
        "micro_bsz": planned["micro_bsz"],
        "grad_accum": planned["grad_accum"],
        "compile": bool(planned["compile"]),
        "seed_label": planned["seed_label"],
        "model_seed": planned["model_init_seed"],
        "train_order_seed": planned["train_order_seed"],
    }
    for field in RUN_META_CANDIDATE_BINDINGS:
        if meta.get(field) != planned_meta[field]:
            raise BindingFailure(
                f"run_meta.json {field} {meta.get(field)!r} does not match the planned candidate "
                f"identity {planned_meta[field]!r}"
            )
    lr_configuration = meta.get("lr_configuration")
    if not isinstance(lr_configuration, Mapping):
        raise BindingFailure(f"run_meta.json at {meta_path} records no LR configuration")
    expected_lr = {
        "peak_lr": planned["peak_lr"],
        "warmup_updates": planned["warmup_updates"],
        "updates": planned["updates"],
    }
    if {k: lr_configuration.get(k) for k in expected_lr} != expected_lr:
        raise BindingFailure(
            f"run_meta.json LR configuration {dict(lr_configuration)!r} does not match the "
            f"planned candidate configuration {expected_lr!r}"
        )
    if meta.get("ledger_identity") != dict(session.ledger.identity):
        raise BindingFailure("run_meta.json ledger identity does not bind this execution")

    if str(result["phase"]) == "MB":
        result.update(verify_recomputed_mb_result(result))
    else:
        result.update(verify_recomputed_lr_result(result))
    result["run_meta_verified_from_disk"] = True
    result["run_meta_path"] = str(meta_path)
    return result


def _ineligible_evidence(
    candidate: Mapping[str, Any],
    session: ExecutionSession,
    reason: str,
    detail: str,
    terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured candidate-local failure evidence; the required grid continues.

    R3 Part 9: the candidate's own terminal artifact supplies the real completed-update count
    and the ledger token state the parent records. The parent never reconstructs them.
    """
    reported = dict(terminal or {})
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
        # The child's own number when it reported one; a marked zero when there is no terminal
        # artifact at all, so an invented value can never be mistaken for a measurement.
        "completed_updates": int(reported.get("completed_updates") or 0),
        "terminal_evidence_present": terminal is not None,
        "terminal_reserved_tokens": reported.get("reserved_tokens"),
        "terminal_completed_tokens": reported.get("completed_tokens"),
        "terminal_status": reported.get("terminal_status"),
        "terminal_error_class": reported.get("error_class"),
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
        "median_update_tokens_per_second": 0.0,
        "update_timings": [],
        "canonical_compile_path": False,
        "run_meta_sha256": reported.get("run_meta_sha256"),
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
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PhaseAbort(f"terminal result at {path} is not readable JSON: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise PhaseAbort(f"terminal result at {path} is not an object")
    if doc.get("schema_version") != TERMINAL_RESULT_SCHEMA:
        raise PhaseAbort(f"terminal result schema mismatch at {path}")
    missing = [f for f in TERMINAL_RESULT_FIELDS if f not in doc]
    if missing:
        raise PhaseAbort(f"terminal result at {path} is missing {missing}")
    if doc["terminal_status"] not in TERMINAL_STATUSES:
        raise PhaseAbort(f"unknown terminal status {doc['terminal_status']!r} at {path}")
    completed = doc.get("completed_updates")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise PhaseAbort(
            f"terminal result at {path} reports an unusable completed_updates {completed!r}"
        )
    return dict(doc)


class CandidateTerminalFailure(CandidateFailure):
    """A candidate-local failure carrying the child's own structured terminal artifact."""

    def __init__(self, terminal: Mapping[str, Any]):
        super().__init__(f"{terminal.get('error_class')}: {terminal.get('error_message')}")
        self.terminal = dict(terminal)


def _worker_argv(
    *,
    spec_path: Path,
    plan: Mapping[str, Any],
    session: ExecutionSession,
    candidate: Mapping[str, Any],
) -> list[str]:
    repo = repo_root()
    root = session.output_root
    return [
        sys.executable,
        str(Path(ROOT) / "pretrain" / "pilot_runner_v2_3.py"),
        "internal-worker",
        "--authorization",
        session.validated["authorization_path"],
        "--session-manifest",
        str(session.session_path),
        "--phase-plan",
        str(plan["plan_path"]),
        "--candidate-spec",
        str(spec_path),
        "--pilot-index-manifest",
        session.validated["index_manifest_path"],
        "--accepted-stage-a",
        str(repo / ACCEPTED_STAGE_A),
        "--accepted-stage-b",
        str(repo / ACCEPTED_STAGE_B),
        "--ledger",
        str(root / LEDGER_FILENAME),
        "--candidate-output",
        str(candidate["output_dir"]),
    ]


def _subprocess_launcher(
    candidate: Mapping[str, Any], session: ExecutionSession, plan: Mapping[str, Any]
) -> None:
    """Launch one candidate in a FRESH subprocess that revalidates every artifact itself.

    R3 Parts 1/2: the child receives ONLY canonical artifact paths -- the authorization
    manifest, the session manifest, the immutable phase plan, its own published spec, the
    pilot-index manifest, both accepted releases, the ledger and its own output directory. It
    revalidates all of them from disk. The parent passes no hashes, no counts and no authority
    the child could take on trust, and the child gains nothing from being invoked directly.
    """
    spec_path = plan["specs"][candidate["candidate_id"]]["path"]
    # A stale terminal result from an earlier attempt must never be read as this launch's.
    terminal_result_path(spec_path).unlink(missing_ok=True)
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(Path(candidate["output_dir"]) / "inductor_cache")
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        _worker_argv(spec_path=spec_path, plan=plan, session=session, candidate=candidate),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    # R3 Part 9: the child's own terminal artifact is read FIRST. A fault while preserving the
    # logs must never discard the candidate's real progress accounting.
    terminal = read_terminal_result(spec_path)
    spec_dir = Path(spec_path).parent
    (spec_dir / f"{candidate['candidate_id']}.stdout").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (spec_dir / f"{candidate['candidate_id']}.stderr").write_text(
        completed.stderr or "", encoding="utf-8"
    )
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
        raise CandidateTerminalFailure(terminal)


def _classify_candidate_failure(exc: Exception) -> tuple[str, str]:
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, CandidateTerminalFailure):
        name = str(exc.terminal.get("error_class") or name)
        text = str(exc.terminal.get("error_message") or text)
    lowered = text.lower()
    if "OutOfMemory" in name or "out of memory" in lowered:
        return "oom", f"{name}: {text}"
    if "compile" in lowered or "dynamo" in lowered or "inductor" in lowered:
        return "compile_failure", f"{name}: {text}"
    if "not finite" in lowered or "non-finite" in lowered or "nonfinite" in lowered:
        return "nonfinite", f"{name}: {text}"
    return "candidate_runtime_exception", f"{name}: {text}"


def _launch(
    candidate: Mapping[str, Any],
    session: ExecutionSession,
    plan: Mapping[str, Any],
    launcher: Any,
) -> dict[str, Any]:
    """Launch one candidate.

    R3 Part 10: binding failures, phase aborts and every process-control event propagate. Only
    an ordinary candidate-local ``Exception`` becomes structured ineligible evidence, so the
    required grid can still be completed. ``KeyboardInterrupt`` and ``SystemExit`` are not
    ``Exception`` subclasses and are therefore never downgraded here.
    """
    try:
        launcher(candidate, session, plan)
    except (BindingFailure, PhaseAbort):
        raise
    except CandidateTerminalFailure as exc:
        reason, detail = _classify_candidate_failure(exc)
        return _ineligible_evidence(candidate, session, reason, detail, terminal=exc.terminal)
    except PilotContractError as exc:
        raise PhaseAbort(f"{candidate['candidate_id']}: {exc}") from exc
    except CandidateFailure as exc:
        reason, detail = _classify_candidate_failure(exc)
        return _ineligible_evidence(candidate, session, reason, detail)
    except Exception as exc:  # noqa: BLE001 - candidate-local failure, never process control
        reason, detail = _classify_candidate_failure(exc)
        return _ineligible_evidence(candidate, session, reason, detail)
    return load_completed_result(session, planned=candidate, plan=plan)


# --------------------------------------------------------------------- immutable reports


def write_immutable_report(path: Path, payload: Mapping[str, Any]) -> str:
    """R2 Part 6: a phase report is published once and never rewritten."""
    return write_immutable_artifact(path, payload)


def _require_test_launcher(launcher: Any) -> Any:
    """A non-default launcher is a fake/test hook, never a route into the real backend."""
    if launcher is None:
        return _subprocess_launcher
    if launcher in REAL_TRAINING_ENTRYPOINTS or launcher is execute_validated_candidate:
        raise BindingFailure(
            "the real training backend is reachable only from validate_worker_execution(); it "
            "may not be injected as an orchestrator launcher"
        )
    return launcher


def orchestrate_phase_mb(session: ExecutionSession, *, launcher: Any = None) -> dict[str, Any]:
    """The single Phase-MB orchestration path.

    Publishes the immutable PHASE_MB_PLAN listing exactly the frozen ten candidate specs and
    their SHA-256s, launches each, converts candidate-local failures into structured ineligible
    evidence while continuing the grid, then recomputes every eligible candidate's throughput
    from its raw per-update timing records and performs deterministic selection on that
    recomputed evidence alone.
    """
    if session.scope not in ALLOWED_SCOPES:
        raise BindingFailure(f"unknown authorized scope {session.scope!r}")
    launcher = _require_test_launcher(launcher)
    candidates = plan_phase_mb(output_root=session.output_root)
    require(len(candidates) == 10, "the Phase-MB grid must enumerate exactly ten candidates")
    require_complete_mb_grid(candidates)
    plan = publish_phase_plan(
        root=session.output_root,
        plan_kind="PHASE_MB_PLAN",
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        candidates=candidates,
        derived_from={"source": "the frozen P-PILOT-CONTRACT-V2.3 Phase-MB grid"},
    )

    outcomes = [_launch(c, session, plan, launcher) for c in candidates]
    require_complete_mb_grid(outcomes)
    vram = int(session.validated["fingerprint"]["gpu"]["total_vram_bytes"])
    selection = mb_select(outcomes, vram)
    report = {
        "schema_version": MB_REPORT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "phase": "MB",
        "session_sha256": session.session_sha256,
        "phase_plan_kind": "PHASE_MB_PLAN",
        "phase_plan_sha256": plan["plan_sha256"],
        "candidate_spec_sha256s": list(plan["plan"]["candidate_spec_sha256s"]),
        "candidates": outcomes,
        "selection": selection,
        "selection_trace": {
            "physical_vram_bytes": vram,
            "throughput_statistic": "median of the per-update rates over updates 11..40",
            "eligibility": {
                str(r["candidate_id"]): list(mb_candidate_eligible(r, vram)[1]) for r in outcomes
            },
            "median_update_tokens_per_second": {
                str(r["candidate_id"]): r.get("median_update_tokens_per_second") for r in outcomes
            },
        },
        "FROZEN_MICRO_BSZ": selection.get("FROZEN_MICRO_BSZ"),
        "FROZEN_GRAD_ACCUM": selection.get("FROZEN_GRAD_ACCUM"),
        "FROZEN_COMPILE": selection.get("FROZEN_COMPILE"),
        "ledger": session.ledger.snapshot(),
        "authorized_scope": session.scope,
        # R3 Part 3/12: PHASE_MB_ONLY completes here. There is no promotion path: Phase LR needs
        # a new FULL_V2_3_PILOT authorization, which implies a new session and a new ledger.
        "session_terminates_after_this_phase": session.scope == "PHASE_MB_ONLY",
        "next_phase_requires_new_authorization": session.scope == "PHASE_MB_ONLY",
        "physical_vram_bytes": vram,
        "base_runtime_fingerprint": dict(session.validated["fingerprint"]),
        **dict(_session_bindings(session)),
    }
    report["report_sha256"] = write_immutable_report(
        session.output_root / MB_REPORT_FILENAME, report
    )
    return report


def load_authoritative_mb_report(session: ExecutionSession) -> dict[str, Any]:
    """R3 Part 3: Phase-LR geometry comes ONLY from the verified Phase-MB report.

    The report file is re-hashed against its published sidecar, revalidated against this
    session's bindings and its own phase plan, checked for a complete ten-candidate grid,
    recomputed candidate by candidate from raw timing records, and its recorded selection is
    re-derived. Nothing about the Phase-LR geometry can be supplied on the command line.
    """
    path = session.output_root / MB_REPORT_FILENAME
    if not path.is_file():
        raise BindingFailure(
            f"Phase-LR requires the authoritative Phase-MB report at {path}; the frozen "
            f"micro_bsz and compile mode are never taken from the command line"
        )
    report, digest = read_immutable_artifact(path, schema_version=MB_REPORT_SCHEMA)
    reported_fingerprint = report.get("base_runtime_fingerprint")
    current_fingerprint = session.validated["fingerprint"]
    if not isinstance(reported_fingerprint, Mapping):
        raise BindingFailure("Phase-MB report has no bound base runtime fingerprint")
    fingerprint_payload = dict(reported_fingerprint)
    recorded_fingerprint_sha256 = fingerprint_payload.pop("fingerprint_sha256", None)
    recomputed_fingerprint_sha256 = hashlib.sha256(
        canonical_json_bytes(fingerprint_payload)
    ).hexdigest()
    if recorded_fingerprint_sha256 != recomputed_fingerprint_sha256:
        raise BindingFailure("Phase-MB report base runtime fingerprint self-hash is invalid")
    if report.get("runtime_fingerprint_sha256") != recorded_fingerprint_sha256:
        raise BindingFailure("Phase-MB report runtime fingerprint SHA disagrees with its payload")
    if reported_fingerprint != current_fingerprint:
        raise BindingFailure(
            "Phase-MB report runtime fingerprint is incompatible with the current Phase-LR "
            "runtime; aborting this FULL session without migrating frozen MB geometry"
        )
    reported_gpu = reported_fingerprint.get("gpu")
    if not isinstance(reported_gpu, Mapping):
        raise BindingFailure("Phase-MB report runtime fingerprint has malformed GPU identity")
    if report.get("physical_vram_bytes") != reported_gpu.get("total_vram_bytes"):
        raise BindingFailure("Phase-MB report physical VRAM disagrees with its runtime fingerprint")
    # Phase LR runs under the SAME authorization, session and ledger as the Phase MB it
    # consumes, so every binding -- authorization and session identity included -- must match.
    for field, expected in _session_bindings(session):
        if report.get(field) != expected:
            raise BindingFailure(
                f"Phase-MB report {field} does not bind this execution; Phase LR consumes only "
                f"a Phase-MB report published by its own authorized session"
            )
    if report.get("session_sha256") != session.session_sha256:
        raise BindingFailure("Phase-MB report does not bind this session manifest")
    plan_path = session.output_root / PHASE_MB_PLAN_FILENAME
    plan, plan_sha256 = validate_phase_plan(
        plan_path,
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        expected_phase="MB",
    )
    if report.get("phase_plan_sha256") != plan_sha256:
        raise BindingFailure("Phase-MB report does not bind the phase plan it was produced under")
    frozen = verify_mb_report_document(report)
    return {
        "report_relpath": MB_REPORT_FILENAME,
        "report_sha256": digest,
        "phase_plan_sha256": plan_sha256,
        "plan_kind": plan["plan_kind"],
        **frozen,
    }


def orchestrate_phase_muon_lr(session: ExecutionSession, *, launcher: Any = None) -> dict[str, Any]:
    """The single Phase-Muon-LR orchestration path.

    R3 Parts 2/3: each step publishes an immutable plan bound to the validated evidence it was
    derived from -- the initial plan to the verified Phase-MB report, the confirmation plan to
    the published initial LR report, the edge plan to the published confirmation report -- and
    an immutable report of its own. Nothing about which candidates run, or with what geometry,
    is taken from caller input.
    """
    if session.scope != "FULL_V2_3_PILOT":
        raise BindingFailure(
            f"scope {session.scope!r} may not execute Phase Muon-LR; PHASE_MB_ONLY terminates "
            f"after its Phase-MB report and is never promoted"
        )
    launcher = _require_test_launcher(launcher)
    frozen = load_authoritative_mb_report(session)
    root = session.output_root

    def step(
        *,
        plan_kind: str,
        peak_lrs: Sequence[float],
        seed_labels: Sequence[str],
        derived_from: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for seed_label in seed_labels:
            candidates.extend(
                plan_phase_lr(
                    output_root=root,
                    micro_bsz=frozen["micro_bsz"],
                    compile_on=frozen["compile"],
                    peak_lrs=peak_lrs,
                    seed_label=seed_label,
                )
            )
        plan = publish_phase_plan(
            root=root,
            plan_kind=plan_kind,
            session_sha256=session.session_sha256,
            session_id=session.session_id,
            candidates=candidates,
            derived_from={
                "frozen_geometry": {
                    "micro_bsz": frozen["micro_bsz"],
                    "grad_accum": frozen["grad_accum"],
                    "compile": frozen["compile"],
                },
                "peak_lrs": [float(v) for v in peak_lrs],
                "seed_labels": list(seed_labels),
                **dict(derived_from),
            },
        )
        return [_launch(c, session, plan, launcher) for c in candidates], plan

    def publish(filename: str, schema: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        doc = {
            "schema_version": schema,
            "contract_version": CONTRACT_VERSION,
            "phase": "LR",
            "session_sha256": session.session_sha256,
            "frozen_geometry": frozen,
            "ledger": session.ledger.snapshot(),
            **dict(_session_bindings(session)),
            **dict(payload),
        }
        doc["report_sha256"] = write_immutable_report(root / filename, doc)
        return doc

    mb_derivation = {
        "phase_mb_report_relpath": frozen["report_relpath"],
        "phase_mb_report_sha256": frozen["report_sha256"],
        "phase_mb_plan_sha256": frozen["phase_plan_sha256"],
    }
    seed1, initial_plan = step(
        plan_kind="PHASE_LR_INITIAL_PLAN",
        peak_lrs=list(LR_GRID_SEED1),
        seed_labels=["seed-1"],
        derived_from=mb_derivation,
    )
    require_complete_lr_grid(seed1)  # R2 Part 8: the initial grid must be complete evidence
    seed1_verdict = lr_select_seed1(seed1)
    initial_report = publish(
        LR_INITIAL_REPORT_FILENAME,
        LR_REPORT_SCHEMA,
        {
            "step": "INITIAL",
            "phase_plan_kind": "PHASE_LR_INITIAL_PLAN",
            "phase_plan_sha256": initial_plan["plan_sha256"],
            "candidate_spec_sha256s": list(initial_plan["plan"]["candidate_spec_sha256s"]),
            "seed1": seed1,
            "selection": seed1_verdict,
        },
    )
    if seed1_verdict["outcome"] != "SEED1_WINNER":
        return publish(
            LR_REPORT_FILENAME,
            LR_REPORT_SCHEMA,
            {
                "step": "FINAL",
                "initial_report_sha256": initial_report["report_sha256"],
                "seed1": seed1,
                "outcome": seed1_verdict,
                "terminal_status": "PHASE_ABORT",
            },
        )

    winner_lr = float(seed1_verdict["winner_peak_lr"])
    neighbour = confirmation_neighbor(winner_lr)
    confirm_lrs = [winner_lr, neighbour]
    seed2, confirm_plan = step(
        plan_kind="PHASE_LR_CONFIRMATION_PLAN",
        peak_lrs=confirm_lrs,
        seed_labels=["seed-2"],
        derived_from={
            **mb_derivation,
            "preceding_report_relpath": LR_INITIAL_REPORT_FILENAME,
            "preceding_report_sha256": initial_report["report_sha256"],
            "preceding_selection": dict(seed1_verdict),
        },
    )
    by_lr_1 = {float(r["peak_lr"]): r for r in seed1}
    by_lr_2 = {float(r["peak_lr"]): r for r in seed2}
    # R2 Part 8: confirmation evidence must be COMPLETE -- both LRs must have a recorded run at
    # both seeds. A missing record is an evidence gap, never an implicit ineligibility.
    missing = [lr for lr in confirm_lrs if lr not in by_lr_1 or lr not in by_lr_2]
    if missing:
        raise PhaseAbort(f"confirmation evidence incomplete; no record for peak_lr {missing}")
    # R3: an ineligible run records a null score, never a non-finite one -- these pairs are
    # serialized into the immutable confirmation and final reports, and canonical JSON forbids
    # NaN/Infinity, so an inf here would make the report unpublishable.
    pairs = [
        {
            "peak_lr": lr,
            "seed1_score": float(by_lr_1[lr]["score"]) if by_lr_1[lr].get("eligible") else None,
            "seed1_eligible": bool(by_lr_1[lr].get("eligible"))
            and lr_candidate_eligible(by_lr_1[lr])[0],
            "seed2_score": float(by_lr_2[lr]["score"]) if by_lr_2[lr].get("eligible") else None,
            "seed2_eligible": bool(by_lr_2[lr].get("eligible"))
            and lr_candidate_eligible(by_lr_2[lr])[0],
        }
        for lr in confirm_lrs
    ]
    confirmed = lr_confirm(pairs)
    confirmation_report = publish(
        LR_CONFIRMATION_REPORT_FILENAME,
        LR_REPORT_SCHEMA,
        {
            "step": "CONFIRMATION",
            "phase_plan_kind": "PHASE_LR_CONFIRMATION_PLAN",
            "phase_plan_sha256": confirm_plan["plan_sha256"],
            "candidate_spec_sha256s": list(confirm_plan["plan"]["candidate_spec_sha256s"]),
            "initial_report_sha256": initial_report["report_sha256"],
            "seed2": seed2,
            "confirmation_pairs": pairs,
            "selection": confirmed,
        },
    )
    if confirmed["outcome"] != "CONFIRMED":
        return publish(
            LR_REPORT_FILENAME,
            LR_REPORT_SCHEMA,
            {
                "step": "FINAL",
                "initial_report_sha256": initial_report["report_sha256"],
                "confirmation_report_sha256": confirmation_report["report_sha256"],
                "seed1": seed1,
                "seed2": seed2,
                "confirmation_pairs": pairs,
                "seed1_verdict": seed1_verdict,
                "outcome": confirmed,
                "terminal_status": "PHASE_ABORT",
            },
        )

    incumbent_lr = float(confirmed["confirmed_peak_lr"])
    edge_lr = edge_candidate(incumbent_lr)
    edge_runs: list[dict[str, Any]] = []
    edge_report_sha256 = None
    edge_kwargs: dict[str, Any] = {"edge_lr": edge_lr}
    if edge_lr is not None:
        edge_runs, edge_plan = step(
            plan_kind="PHASE_LR_EDGE_PLAN",
            peak_lrs=[edge_lr],
            seed_labels=["seed-1", "seed-2"],
            derived_from={
                **mb_derivation,
                "preceding_report_relpath": LR_CONFIRMATION_REPORT_FILENAME,
                "preceding_report_sha256": confirmation_report["report_sha256"],
                "preceding_selection": dict(confirmed),
            },
        )
        e1 = [r for r in edge_runs if r["seed_label"] == "seed-1"]
        e2 = [r for r in edge_runs if r["seed_label"] == "seed-2"]
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
        edge_report_sha256 = publish(
            LR_EDGE_REPORT_FILENAME,
            LR_REPORT_SCHEMA,
            {
                "step": "EDGE",
                "phase_plan_kind": "PHASE_LR_EDGE_PLAN",
                "phase_plan_sha256": edge_plan["plan_sha256"],
                "candidate_spec_sha256s": list(edge_plan["plan"]["candidate_spec_sha256s"]),
                "confirmation_report_sha256": confirmation_report["report_sha256"],
                "edge_runs": edge_runs,
                "selection": {k: v for k, v in edge_kwargs.items()},
            },
        )["report_sha256"]
    final = lr_resolve_edge(
        incumbent_lr=incumbent_lr,
        incumbent_final_score=float(confirmed["final_score"]),
        **edge_kwargs,
    )
    return publish(
        LR_REPORT_FILENAME,
        LR_REPORT_SCHEMA,
        {
            "step": "FINAL",
            "initial_report_sha256": initial_report["report_sha256"],
            "confirmation_report_sha256": confirmation_report["report_sha256"],
            "edge_report_sha256": edge_report_sha256,
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
        },
    )


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


def run_meta(worker: WorkerAuthority) -> dict[str, Any]:
    """The candidate's real run metadata, written to disk next to its result.

    R3 Part 4: every field here is one the admission loader recomputes or cross-checks against
    the result artifact and the planned candidate identity, so a result whose metadata does not
    describe the candidate the phase plan authorized is rejected from selection.
    """
    worker = _require_worker_authority(worker)
    v = worker.validated
    candidate = worker.candidate
    return {
        "schema_version": RUN_META_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "base_fingerprint_sha256": v["fingerprint"]["fingerprint_sha256"],
        "runtime_fingerprint_sha256": v["fingerprint"]["fingerprint_sha256"],
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": bool(candidate["compile"]),
        "optimizer": {
            "name": "muon",
            "muon_lr": MUON_LR_ARG,
            "muon_momentum": MUON_MOMENTUM,
            "momentum": MUON_MOMENTUM,
            "nesterov": MUON_NESTEROV,
            "ns_steps": MUON_NS_STEPS,
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
        "output_dir": str(worker.output_dir),
        "candidate_spec_sha256": worker.spec_sha256,
        "phase_plan_kind": worker.plan["plan_kind"],
        "phase_plan_sha256": worker.plan_sha256,
        "session_sha256": worker.session_sha256,
        "pilot_index_hashes": {
            k: v["indices"][k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        },
        "serialized_index_lists_digest": v["serialized_index_lists_digest"],
        "pilot_index_manifest_file_sha256": v["pilot_index_manifest_file_sha256"],
        "contract_sha256": v["observed"]["contract_sha256"],
        "implementation_head": v["observed"]["head"],
        "execution_bundle_sha256": v["observed"]["execution_bundle_sha256"],
        "execution_implementation_bundle_sha256": v["observed"]["execution_bundle_sha256"],
        "accepted_stage_a_meta_sha256": v["observed"]["stage_a_meta_sha256"],
        "accepted_stage_b_meta_sha256": v["observed"]["stage_b_meta_sha256"],
        "authorized_scope": worker.scope,
        "session_id": worker.session_id,
        "authorization_sha256": v["authorization_sha256"],
        "ledger_identity": dict(worker.ledger.identity),
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


def _cli_internal_worker(args: argparse.Namespace) -> int:
    """One candidate, in its own process, revalidating every canonical artifact from disk.

    R3 Part 1/2: the ONLY inputs are artifact paths. There is no way to hand this process a
    constructed session, a validation object or an ``authorized`` flag, and a candidate spec
    that the immutable phase plan does not list is refused before model construction.

    R3 Part 9: ``progress`` is advanced by the training loop itself, so a candidate that raises
    after partial progress reports the updates it really completed and the ledger state read
    back under the lock -- never a reconstructed zero.

    R3 Part 10: ``KeyboardInterrupt`` and ``SystemExit`` are deliberately NOT caught. A
    process-control event leaves no terminal artifact, which the parent treats as a phase-level
    failure rather than as an ordinary ineligible candidate.
    """
    spec_path = Path(args.candidate_spec)
    worker: WorkerAuthority | None = None
    candidate: Mapping[str, Any] = {}
    progress: dict[str, Any] = {"completed_updates": 0, "update_timings": []}
    # R3 Part 10: a candidate that never reached model construction cannot be "candidate-locally
    # ineligible". Anything that goes wrong while the artifacts are still being revalidated is an
    # identity/binding failure by definition of WHERE it happened, regardless of its type.
    stage = "VALIDATION"
    try:
        worker = validate_worker_execution(
            authorization_path=Path(args.authorization),
            session_manifest_path=Path(args.session_manifest),
            phase_plan_path=Path(args.phase_plan),
            candidate_spec_path=spec_path,
            pilot_index_manifest_path=Path(args.pilot_index_manifest),
            accepted_stage_a_path=Path(args.accepted_stage_a),
            accepted_stage_b_path=Path(args.accepted_stage_b),
            ledger_path=Path(args.ledger),
            candidate_output_path=Path(args.candidate_output),
        )
        candidate = worker.candidate
        stage = "EXECUTION"
        result = execute_validated_candidate(worker, progress)
        snapshot = worker.ledger.snapshot()
        write_terminal_result(
            spec_path,
            {
                "terminal_status": "SUCCESS",
                "error_class": None,
                "error_message": None,
                "completed_updates": int(result["completed_updates"]),
                "reserved_tokens": snapshot["reserved_tokens"],
                "completed_tokens": snapshot["completed_tokens"],
                "run_meta_sha256": result["run_meta_sha256"],
                "candidate_id": result["candidate_id"],
                "phase": result["phase"],
            },
        )
        return SUCCESS
    except Exception as exc:  # noqa: BLE001 - every ordinary exit is a structured terminal result
        if isinstance(exc, BindingFailure):
            status = "BINDING_FAILURE"
        elif isinstance(exc, (PhaseAbort, PilotContractError)):
            status = "PHASE_ABORT"
        elif stage == "VALIDATION":
            # An unreadable artifact, a missing release, an OS fault -- whatever it is, it broke
            # before this process had any authority, so it is never merely an ineligible run.
            status = "BINDING_FAILURE"
        else:
            status = "CANDIDATE_INELIGIBLE"
        # R3 Part 9: a fresh, locked ledger snapshot -- not a stale in-memory counter.
        snapshot: dict[str, Any] = {}
        if worker is not None:
            try:
                snapshot = worker.ledger.snapshot()
            except Exception:  # noqa: BLE001 - a broken ledger must not mask the real failure
                snapshot = {}
        write_terminal_result(
            spec_path,
            {
                "terminal_status": status,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "completed_updates": int(progress.get("completed_updates") or 0),
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
            "session_sha256": session.session_sha256,
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
    except Exception:  # noqa: BLE001 - an unexpected orchestrator fault is a phase abort
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
    sub.add_parser("session-budget", help="the frozen FULL/MB_ONLY session budget semantics")
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
    # R3 Part 2: the internal worker takes canonical artifact PATHS and nothing else. Invoking
    # it directly with a candidate spec the published phase plan does not list confers no
    # execution capability whatsoever.
    e = sub.add_parser(
        "internal-worker", help="internal: one planned candidate, artifact paths only"
    )
    e.add_argument("--authorization", type=Path, required=True)
    e.add_argument("--session-manifest", type=Path, required=True)
    e.add_argument("--phase-plan", type=Path, required=True)
    e.add_argument("--candidate-spec", type=Path, required=True)
    e.add_argument("--pilot-index-manifest", type=Path, required=True)
    e.add_argument("--accepted-stage-a", type=Path, required=True)
    e.add_argument("--accepted-stage-b", type=Path, required=True)
    e.add_argument("--ledger", type=Path, required=True)
    e.add_argument("--candidate-output", type=Path, required=True)

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
    if args.command == "session-budget":
        _emit({
            "contract_version": CONTRACT_VERSION,
            "FULL_V2_3_PILOT_SESSION_HARD_CEILING": FULL_V2_3_PILOT_SESSION_HARD_CEILING,
            "phase_ceilings": dict(PHASE_CEILINGS),
            "semantics": {
                k: (dict(v) if isinstance(v, Mapping) else v)
                for k, v in SESSION_BUDGET_SEMANTICS.items()
            },
        })
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
    if args.command == "internal-worker":
        return _cli_internal_worker(args)
    return _cli_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
