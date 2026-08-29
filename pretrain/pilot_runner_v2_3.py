#!/usr/bin/env python3
"""P-PILOT-CONTRACT-V2.3 executor: the complete real pilot execution path.

Unlike the V2.2 preflight runner, this module implements the actual training code path --
model construction, canonical packed-dataset access over fixed pilot indices, Muon optimizer and
scheduler construction, the Phase-MB and Phase-Muon-LR update loops with gradient accumulation,
clipping and optimizer steps, CUDA metrics, evaluation, persistent token accounting, checkpoint
save/resume, result publication and deterministic selection.

It is complete but **unexecuted in the materialization segment**: every entry into the training
path passes through :func:`authorize_execution`, which requires an external AUTHORIZED manifest
bound to this exact HEAD, contract, execution bundle, pilot indices and accepted releases, and
through the exact RTX 4090 runtime gate. No tracked code change is needed for a later authorized
run: publishing the manifest is sufficient.

Reuse, never reimplementation: ``src.model.GPT``, ``src.optim.build_optimizer``,
``pretrain.dataset_pretrain.PackedBinDataset`` with the manifest requirement on (which emits
the canonical production loss mask), and the shared canonical primitives in
``src.canonical_loss`` and ``src.canonical_schedule`` -- the same modules the production
trainer imports. R1: the pilot no longer imports the trainer itself, which keeps the execution
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
    lr_select_seed1,
    mb_candidate_grid,
    mb_lr,
    mb_select,
    require,
    require_complete_lr_grid,
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
    require(meta_path.is_file(), f"accepted release manifest missing: {rel}/meta.json")
    meta_sha = file_sha256(meta_path)
    require(
        meta_sha == expected_meta,
        f"{stage} meta SHA-256 mismatch: expected {expected_meta}, got {meta_sha}",
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


# --------------------------------------------------------------------- token ledger

LEDGER_FILENAME = "token_ledger.json"
LEDGER_LOCK_FILENAME = "token_ledger.lock"


class TokenLedger:
    """Persistent trained-token accounting, bound to the validated execution identity.

    R1: the ledger is constructed by the orchestrator from a validated context, binds that
    identity, validates any pre-existing state before reuse, and serializes concurrent
    subprocess updates with a Linux ``fcntl.flock`` advisory lock.

    Atomic protocol, one documented order per optimizer update:

        1. acquire the exclusive lock
        2. reload the on-disk state and validate its identity binding
        3. verify the next update fits BOTH the phase and the global effective ceiling
        4. commit the update (state written and fsynced, then atomically renamed) BEFORE the
           lock is released

    Commit-on-completion is the whole protocol: there is no separate reservation that a crashed
    candidate could leak, and a failed candidate can neither double-count nor give back updates
    that already ran.
    """

    def __init__(
        self, path: Path, identity: Mapping[str, Any], effective_ceilings: Mapping[str, int]
    ):
        self.path = Path(path)
        self.lock_path = self.path.parent / LEDGER_LOCK_FILENAME
        self.identity = dict(identity)
        self.effective_ceilings = {k: int(v) for k, v in effective_ceilings.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self.state = self._validated(json.loads(self.path.read_text(encoding="utf-8")))
        else:
            self.state = {
                "schema_version": TOKEN_LEDGER_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "identity": dict(self.identity),
                "effective_ceilings": dict(self.effective_ceilings),
                "phase_tokens": {"MB": 0, "LR": 0},
                "global_tokens": 0,
                "updates": 0,
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
            f"token ledger identity does not bind this execution: {sorted(mismatched)}",
        )
        require(int(state.get("global_tokens", -1)) >= 0, "token ledger global count invalid")
        return dict(state)

    def _write(self, state: Mapping[str, Any]) -> None:
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
            with open(self.lock_path, "w") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return guard()

    def effective_ceiling(self, phase: str) -> int:
        """Never larger than the frozen contract ceiling or the authorized ceiling."""
        return min(int(PHASE_CEILINGS[phase]), int(self.effective_ceilings[phase]))

    def check(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> dict[str, Any]:
        phase_after = int(self.state["phase_tokens"].get(phase, 0)) + int(tokens)
        global_after = int(self.state["global_tokens"]) + int(tokens)
        breaches = []
        if phase_after > self.effective_ceiling(phase):
            breaches.append(f"phase_{phase}_token_ceiling")
        if global_after > min(GLOBAL_PILOT_TOKEN_CEILING, int(self.effective_ceilings["GLOBAL"])):
            breaches.append("global_pilot_token_ceiling")
        return {
            "phase": phase,
            "phase_tokens_after": phase_after,
            "global_tokens_after": global_after,
            "breaches": breaches,
            "may_execute": not breaches,
            "outcome": "PILOT_ABORT" if breaches else "WITHIN_BUDGET",
        }

    def commit_update(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> dict[str, Any]:
        """Steps 1-4 of the atomic protocol, for one completed optimizer update."""
        with self._lock():
            if self.path.is_file():
                self.state = self._validated(json.loads(self.path.read_text(encoding="utf-8")))
            verdict = self.check(phase, tokens)
            require(
                verdict["may_execute"],
                "PILOT_ABORT: trained-token ceiling would be exceeded: "
                + ", ".join(verdict["breaches"]),
            )
            self.state["phase_tokens"][phase] = verdict["phase_tokens_after"]
            self.state["global_tokens"] = verdict["global_tokens_after"]
            self.state["updates"] = int(self.state["updates"]) + 1
            self._write(self.state)
            return verdict


def authorized_effective_ceilings(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Effective ceilings = min(frozen contract ceiling, authorized ceiling)."""
    authorized = manifest.get("pilot_trained_token_ceiling")
    require(
        isinstance(authorized, int) and not isinstance(authorized, bool),
        f"authorized token ceiling must be an integer, got {authorized!r}",
    )
    require(authorized > 0, f"authorized token ceiling must be > 0, got {authorized}")
    require(
        authorized <= GLOBAL_PILOT_TOKEN_CEILING,
        f"authorized token ceiling {authorized} exceeds the frozen global ceiling "
        f"{GLOBAL_PILOT_TOKEN_CEILING}",
    )
    return {
        "MB": min(PHASE_CEILINGS["MB"], authorized),
        "LR": min(PHASE_CEILINGS["LR"], authorized),
        "GLOBAL": min(GLOBAL_PILOT_TOKEN_CEILING, authorized),
    }


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
    require(
        grouping["matches_frozen_realization"],
        "realized optimizer grouping does not match V2.3: " + ", ".join(grouping["failures"]),
    )
    return opt


def apply_scheduled_lr(optimizer: Any, lr: float) -> list[float]:
    """One scalar schedule drives all groups via lr_ratio (all 1.0 under V2.3)."""
    realized = []
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr) * float(pg.get("lr_ratio", 1.0))
        realized.append(pg["lr"])
    return realized


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

    R1: imports ``src.canonical_loss`` -- the same module the production trainer imports --
    rather than the whole trainer, which keeps the execution closure minimal and avoids the
    trainer's sibling-style imports. Returning the two components (not a normalized value) is
    what lets evaluation accumulate a correct GLOBAL token mean.
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


# --------------------------------------------------------------------- validated context


class ValidatedExecutionContext:
    """The ONLY object a candidate function accepts.

    R1: constructed exclusively by :func:`build_validated_context` after every binding has been
    validated. There is no caller-supplied ``authorized`` boolean and no partial-dict path: a
    candidate cannot be launched from anything but an instance of this class.
    """

    __slots__ = (
        "scope",
        "phase",
        "manifest_sha256",
        "observed",
        "indices",
        "fingerprint",
        "stage_a",
        "stage_b",
        "ledger",
        "output_root",
        "train_order_by_seed",
        "effective_ceilings",
    )

    def __init__(self, **kwargs: Any):
        for key in self.__slots__:
            setattr(self, key, kwargs[key])

    def require_phase_allowed(self, phase: str) -> None:
        require(phase in ("MB", "LR"), f"unknown phase {phase!r}")
        if self.scope == "PHASE_MB_ONLY":
            require(
                phase == "MB",
                f"scope PHASE_MB_ONLY may not execute phase {phase}; a FULL_V2_3_PILOT "
                "authorization is required",
            )

    def require_binds_candidate(self, candidate: Mapping[str, Any]) -> None:
        """At candidate entry, re-verify the context still binds this exact request."""
        self.require_phase_allowed(str(candidate["phase"]))
        require(
            candidate["seed_label"] in SEED_SEMANTICS,
            f"unknown seed label {candidate.get('seed_label')!r}",
        )
        require(str(candidate["candidate_id"]).strip() != "", "candidate identity is empty")
        require_candidate_output_dir(Path(candidate["output_dir"]), self.output_root)
        require(self.ledger is not None, "context carries no bound token ledger")
        require(
            self.ledger.identity.get("authorization_sha256") == self.manifest_sha256,
            "ledger identity does not bind this authorization",
        )
        require(
            self.fingerprint.get("fingerprint_sha256"), "context carries no runtime fingerprint"
        )

    def train_view(self, stage_a_dataset: Any, seed_label: str) -> IndexView:
        return IndexView(stage_a_dataset, self.train_order_by_seed[seed_label])


def build_validated_context(
    *,
    manifest_path: Path | None,
    requested_scope: str,
    output_root: Path,
    phase: str,
    gpu_required: bool = True,
) -> ValidatedExecutionContext:
    """Validate EVERYTHING, then and only then produce an execution context."""
    require(requested_scope in ALLOWED_SCOPES, f"unknown scope {requested_scope!r}")
    require_numpy_version()

    git = git_policy_status()
    require(
        git["policy_satisfied"],
        "pre-launch Git policy not satisfied: " + ", ".join(git["failures"]),
    )

    stage_a = verify_accepted_release("stage_a")
    stage_b = verify_accepted_release("stage_b")
    indices = generate_pilot_indices(
        stage_a_blocks=stage_a["blocks"], stage_b_blocks=stage_b["blocks"]
    )
    serialized_index_lists_digest = hashlib.sha256(
        canonical_json_bytes({
            k: indices[k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        })
    ).hexdigest()

    resolved_root = Path(output_root).resolve()
    manifest = None
    manifest_sha256 = None
    if manifest_path is not None and Path(manifest_path).is_file():
        raw = Path(manifest_path).read_bytes()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw.decode("utf-8"))
    index_manifest_file_sha256 = (manifest or {}).get("pilot_index_manifest_file_sha256")

    observed = {
        "branch": git["branch"],
        "head": git["head"],
        "contract_sha256": contract_sha256(),
        "execution_bundle_sha256": execution_bundle_sha256(),
        "serialized_index_lists_digest": serialized_index_lists_digest,
        "pilot_index_manifest_file_sha256": index_manifest_file_sha256,
        "stage_a_meta_sha256": stage_a["meta_sha256"],
        "stage_b_meta_sha256": stage_b["meta_sha256"],
        "output_root": str(resolved_root),
    }
    verdict = validate_authorization(manifest, requested_scope=requested_scope, observed=observed)
    require(
        verdict["authorized"],
        f"pilot execution refused under {CONTRACT_VERSION}: " + ", ".join(verdict["failures"]),
    )

    scope = str(manifest["allowed_scope"])
    if scope == "PHASE_MB_ONLY":
        require(phase == "MB", "scope PHASE_MB_ONLY may not execute phase " + str(phase))

    fingerprint = base_runtime_fingerprint(gpu_required=gpu_required)
    ceilings = authorized_effective_ceilings(manifest)
    identity = {
        "contract_sha256": observed["contract_sha256"],
        "head": observed["head"],
        "execution_bundle_sha256": observed["execution_bundle_sha256"],
        "serialized_index_lists_digest": serialized_index_lists_digest,
        "authorization_sha256": manifest_sha256,
        "authorized_output_root": str(resolved_root),
        "authorized_scope": scope,
    }
    ledger = TokenLedger(resolved_root / LEDGER_FILENAME, identity, ceilings)

    return ValidatedExecutionContext(
        scope=scope,
        phase=phase,
        manifest_sha256=manifest_sha256,
        observed=observed,
        indices=indices,
        fingerprint=fingerprint,
        stage_a=stage_a,
        stage_b=stage_b,
        ledger=ledger,
        output_root=resolved_root,
        effective_ceilings=ceilings,
        train_order_by_seed={
            label: train_order(indices["stage_a_train"], spec["train_order"])
            for label, spec in SEED_SEMANTICS.items()
        },
    )


def require_candidate_output_dir(destination: Path, authorized_root: Path) -> Path:
    """Resolved-path containment beneath the authorized root; new; never an accepted release."""
    root = repo_root()
    authorized = Path(authorized_root).resolve()
    resolved = Path(destination).resolve()
    require(
        resolved != authorized and authorized in resolved.parents,
        f"candidate output must resolve beneath the authorized root {authorized}: {resolved}",
    )
    for prefix in PROTECTED_PREFIXES:
        protected = (root / prefix).resolve()
        require(
            resolved != protected and protected not in resolved.parents,
            f"refusing to write pilot output inside an accepted release: {resolved}",
        )
    require(not resolved.exists(), f"candidate output directory must not exist: {resolved}")
    return resolved


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


# --------------------------------------------------------------------- checkpoint policy


def require_checkpointing_disabled(action: str) -> None:
    """V2.3 freezes PILOT_CHECKPOINTING=DISABLED: the executor neither writes nor reads one."""
    from pretrain.pilot_contract_v2_3 import require_checkpointing_disabled as _refuse

    _refuse(action)


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
    """The complete optimizer-update loop: accumulation, clipping, step, metrics, ledger."""
    import torch

    required_blocks = int(updates) * int(SEQUENCES_PER_OPTIMIZER_UPDATE)
    require(
        len(view) >= required_blocks,
        f"{phase}: need {required_blocks} train blocks without replay, view has {len(view)}",
    )

    model.train()
    losses: dict[int, float] = {}
    grad_norms: dict[int, float] = {}
    realized_lrs: dict[int, list[float]] = {}
    step_seconds: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    cursor = 0
    for u in range(1, int(updates) + 1):
        lr = lr_fn(u)
        realized = apply_scheduled_lr(optimizer, lr)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(int(grad_accum)):
            batch = [view[cursor + k] for k in range(int(micro_bsz))]
            cursor += int(micro_bsz)
            x = to_model_ids(torch.stack([b[0] for b in batch]), device)
            y = to_model_ids(torch.stack([b[1] for b in batch]), device)
            m = torch.stack([b[2] for b in batch]).to(device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = canonical_loss(model(x), y, m) / int(grad_accum)
            loss.backward()
            total += float(loss.detach()) * int(grad_accum)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ledger.commit_update(phase)
        losses[u] = total / int(grad_accum)
        grad_norms[u] = gnorm
        realized_lrs[u] = realized
        if u >= timed_from:
            step_seconds.append(elapsed)
        if record_diagnostics:
            diagnostics.append(_group_diagnostics(optimizer, u))
    require(
        cursor == required_blocks,
        f"{phase}: consumed {cursor} blocks, contract requires exactly {required_blocks}",
    )
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "realized_lrs": realized_lrs,
        "step_seconds": step_seconds,
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


def evaluate(model: Any, view: IndexView, *, micro_bsz: int, device: str) -> float:
    """GLOBAL token-mean CE: accumulate numerator and weight, divide once at the end."""
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
    return total_numerator / max(1.0, total_weight)


def global_token_mean(components: Sequence[tuple[float, float]]) -> float:
    """Reference for the accumulation rule: sum numerators, sum weights, divide once."""
    n = sum(float(a) for a, _ in components)
    w = sum(float(b) for _, b in components)
    return n / max(1.0, w)


# --------------------------------------------------------------------- candidate backends


def run_phase_mb_candidate(
    candidate: Mapping[str, Any], context: ValidatedExecutionContext
) -> dict[str, Any]:
    """One Phase-MB probe, end to end, inside a candidate subprocess."""
    import statistics

    import torch

    require(
        isinstance(context, ValidatedExecutionContext),
        "candidate execution requires a ValidatedExecutionContext, not a plain mapping",
    )
    context.require_binds_candidate({**candidate, "phase": "MB"})
    out = require_candidate_output_dir(Path(candidate["output_dir"]), context.output_root)
    out.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor_cache")
    torch.cuda.reset_peak_memory_stats()

    model = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    compile_wrapper_seconds = None
    if candidate["compile"]:
        t0 = time.perf_counter()
        model = torch.compile(model)
        compile_wrapper_seconds = time.perf_counter() - t0
    optimizer = build_pilot_optimizer(model, candidate["peak_lr"])
    view = context.train_view(context.stage_a["dataset"], candidate["seed_label"])

    materialization_start = time.perf_counter()
    result = _run_updates(
        model=model,
        optimizer=optimizer,
        view=view,
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=MB_PROBE_UPDATES,
        lr_fn=mb_lr,
        ledger=context.ledger,
        phase="MB",
        device="cuda",
        timed_from=MB_MEASURED_FIRST_UPDATE,
    )
    first_update_seconds = result["step_seconds"][0] if result["step_seconds"] else None
    first_synchronized_update_seconds = result.get("first_update_seconds", first_update_seconds)

    median_tps = EFFECTIVE_BATCH_TOKENS / statistics.median(result["step_seconds"])
    grouping = verify_realized_grouping(optimizer, model)
    states = verify_optimizer_state(optimizer)
    payload = {
        "phase": "MB",
        "candidate_id": candidate["candidate_id"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "seed_label": candidate["seed_label"],
        "completed_updates": result["completed_updates"],
        "blocks_consumed": result["blocks_consumed"],
        "median_tokens_per_sec": median_tps,
        # R1: lazy compilation materializes through model execution, so the near-zero wrapper
        # call is recorded separately and is NOT presented as compile time.
        "compile_wrapper_seconds": compile_wrapper_seconds,
        "first_synchronized_update_seconds": first_synchronized_update_seconds,
        "compile_materialization_wall_seconds": (
            (time.perf_counter() - materialization_start) if candidate["compile"] else None
        ),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "oom": False,
        "uncontrolled_exception": False,
        "all_losses_finite": all(map(_finite, result["losses"].values())),
        "all_grad_norms_finite": all(map(_finite, result["grad_norms"].values())),
        "all_optimizer_states_instantiated": states["all_states_instantiated"],
        "grouping_matches_contract": grouping["matches_frozen_realization"],
        "all_lr_ratios_are_one": grouping["all_lr_ratios_are_one"],
        "canonical_compile_path": bool(candidate["compile"]),
    }
    _publish_result(out, payload, candidate, context)
    return payload


def run_phase_lr_candidate(
    candidate: Mapping[str, Any], context: ValidatedExecutionContext
) -> dict[str, Any]:
    """One Phase-Muon-LR run, end to end, including both global-mean evaluations."""
    import torch

    from pretrain.pilot_contract_v2_3 import lr_score, sustained_divergence

    require(
        isinstance(context, ValidatedExecutionContext),
        "candidate execution requires a ValidatedExecutionContext, not a plain mapping",
    )
    context.require_binds_candidate({**candidate, "phase": "LR"})
    out = require_candidate_output_dir(Path(candidate["output_dir"]), context.output_root)
    out.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor_cache")

    model = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    if candidate["compile"]:
        model = torch.compile(model)
    optimizer = build_pilot_optimizer(model, candidate["peak_lr"])
    view = context.train_view(context.stage_a["dataset"], candidate["seed_label"])
    result = _run_updates(
        model=model,
        optimizer=optimizer,
        view=view,
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=LR_RUN_UPDATES,
        lr_fn=lambda u: lr_schedule(u, candidate["peak_lr"]),
        ledger=context.ledger,
        phase="LR",
        device="cuda",
        record_diagnostics=True,
    )
    eval_a = IndexView(context.stage_a["dataset"], context.indices["stage_a_eval"])
    eval_b = IndexView(context.stage_b["dataset"], context.indices["stage_b_eval"])
    loss_a = evaluate(model, eval_a, micro_bsz=candidate["micro_bsz"], device="cuda")
    loss_b = evaluate(model, eval_b, micro_bsz=candidate["micro_bsz"], device="cuda")
    grouping = verify_realized_grouping(optimizer, model)
    states = verify_optimizer_state(optimizer)
    guard = sustained_divergence(result["losses"])
    payload = {
        "phase": "LR",
        "candidate_id": candidate["candidate_id"],
        "peak_lr": candidate["peak_lr"],
        "seed_label": candidate["seed_label"],
        "completed_updates": result["completed_updates"],
        "blocks_consumed": result["blocks_consumed"],
        "all_losses_finite": all(map(_finite, result["losses"].values())),
        "all_grad_norms_finite": all(map(_finite, result["grad_norms"].values())),
        "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
        "muon_momentum_states_present": states["all_states_instantiated"],
        "aux_adamw_states_present": states["all_states_instantiated"],
        "grouping_matches_contract": grouping["matches_frozen_realization"],
        "all_lr_ratios_are_one": grouping["all_lr_ratios_are_one"],
        "eval_loss_stage_a": loss_a,
        "eval_loss_stage_b": loss_b,
        "score": lr_score(loss_a, loss_b),
        "sustained_divergence": guard["diverged"],
        "divergence_detail": guard,
        "diagnostics": result["diagnostics"][-1] if result["diagnostics"] else None,
    }
    _publish_result(out, payload, candidate, context)
    return payload


def _finite(v: float) -> bool:
    import math

    return math.isfinite(float(v))


def _publish_result(
    out: Path,
    payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    context: ValidatedExecutionContext,
) -> dict[str, Any]:
    """Write run_meta and the result, binding every identity selection will later require."""
    meta = run_meta(candidate=candidate, context=context)
    meta_bytes = canonical_json_bytes(meta)
    (out / "run_meta.json").write_bytes(meta_bytes)
    bound = dict(payload)
    bound.update({
        "run_meta_sha256": hashlib.sha256(meta_bytes).hexdigest(),
        "contract_sha256": context.observed["contract_sha256"],
        "implementation_head": context.observed["head"],
        "execution_bundle_sha256": context.observed["execution_bundle_sha256"],
        "serialized_index_lists_digest": context.observed["serialized_index_lists_digest"],
        "runtime_fingerprint_sha256": context.fingerprint["fingerprint_sha256"],
        "authorization_sha256": context.manifest_sha256,
        "ledger_identity": dict(context.ledger.identity),
        "output_dir": str(out),
    })
    (out / "result.json").write_bytes(canonical_json_bytes(bound))
    return bound


# --------------------------------------------------------------------- orchestrator

REQUIRED_RESULT_BINDINGS = (
    "phase",
    "candidate_id",
    "seed_label",
    "run_meta_sha256",
    "contract_sha256",
    "implementation_head",
    "execution_bundle_sha256",
    "serialized_index_lists_digest",
    "runtime_fingerprint_sha256",
    "authorization_sha256",
    "ledger_identity",
    "output_dir",
    "completed_updates",
)


def load_completed_result(output_dir: Path, context: ValidatedExecutionContext) -> dict[str, Any]:
    """Load one candidate's result artifact and verify it binds THIS validated execution."""
    path = Path(output_dir) / "result.json"
    require(path.is_file(), f"no completed result artifact at {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_RESULT_BINDINGS if f not in result]
    require(not missing, f"result artifact is missing required bindings: {missing}")
    for field, expected in (
        ("contract_sha256", context.observed["contract_sha256"]),
        ("implementation_head", context.observed["head"]),
        ("execution_bundle_sha256", context.observed["execution_bundle_sha256"]),
        ("serialized_index_lists_digest", context.observed["serialized_index_lists_digest"]),
        ("runtime_fingerprint_sha256", context.fingerprint["fingerprint_sha256"]),
        ("authorization_sha256", context.manifest_sha256),
    ):
        require(
            result.get(field) == expected, f"result artifact {field} does not bind this execution"
        )
    require(
        result.get("ledger_identity") == dict(context.ledger.identity),
        "result artifact ledger identity does not bind this execution",
    )
    return result


def orchestrate_phase_mb(
    context: ValidatedExecutionContext, *, backend: Any = None
) -> dict[str, Any]:
    """The single Phase-MB orchestration path.

    Enumerates exactly the frozen ten candidates, launches each, converts candidate-local
    failures into structured ineligible evidence while continuing the grid, then loads the
    completed artifacts and performs deterministic selection on that evidence alone.
    """
    context.require_phase_allowed("MB")
    backend = backend or _subprocess_backend
    candidates = plan_phase_mb(output_root=context.output_root)
    require(len(candidates) == 10, "the Phase-MB grid must enumerate exactly ten candidates")

    outcomes: list[dict[str, Any]] = []
    for candidate in candidates:
        # Phase-level bindings are re-verified before every launch and ABORT if broken.
        context.require_binds_candidate(candidate)
        try:
            backend(candidate, context)
        except PilotContractError:
            raise  # phase-level failure: abort, never downgrade
        except BaseException as exc:  # noqa: BLE001 - candidate-local failure
            outcomes.append(_ineligible_evidence(candidate, context, exc))
            continue
        outcomes.append(load_completed_result(Path(candidate["output_dir"]), context))

    vram = int(context.fingerprint["gpu"]["total_vram_bytes"])
    selection = mb_select(outcomes, vram)
    report = {
        "phase": "MB",
        "candidates": outcomes,
        "selection": selection,
        "ledger": dict(context.ledger.state),
    }
    (context.output_root / "PHASE_MB_REPORT.json").write_bytes(canonical_json_bytes(report))
    return report


def _ineligible_evidence(
    candidate: Mapping[str, Any], context: ValidatedExecutionContext, exc: BaseException
) -> dict[str, Any]:
    """Structured candidate-local failure evidence; the grid continues."""
    reason = (
        "oom"
        if exc.__class__.__name__ == "OutOfMemoryError"
        else "compile_failure"
        if "compile" in str(exc).lower()
        else "candidate_runtime_exception"
    )
    return {
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "eligible": False,
        "reason": reason,
        "exception": f"{type(exc).__name__}: {exc}",
        "completed_updates": 0,
        "oom": reason == "oom",
        "uncontrolled_exception": reason == "candidate_runtime_exception",
        "all_losses_finite": False,
        "all_grad_norms_finite": False,
        "all_optimizer_states_instantiated": False,
        "grouping_matches_contract": False,
        "all_lr_ratios_are_one": False,
        "max_memory_reserved_bytes": 0,
        "median_tokens_per_sec": 0.0,
        "canonical_compile_path": False,
        "run_meta_sha256": None,
        "contract_sha256": context.observed["contract_sha256"],
        "implementation_head": context.observed["head"],
        "execution_bundle_sha256": context.observed["execution_bundle_sha256"],
        "serialized_index_lists_digest": context.observed["serialized_index_lists_digest"],
        "runtime_fingerprint_sha256": context.fingerprint["fingerprint_sha256"],
        "authorization_sha256": context.manifest_sha256,
        "ledger_identity": dict(context.ledger.identity),
        "output_dir": candidate["output_dir"],
    }


def orchestrate_phase_muon_lr(
    context: ValidatedExecutionContext, *, micro_bsz: int, compile_on: bool, backend: Any = None
) -> dict[str, Any]:
    """The single Phase-Muon-LR orchestration path.

    Derives the initial grid, the seed-1 winner, the confirmation pair and the edge candidate
    ITSELF. Nothing about which candidates run is taken from caller input.
    """
    context.require_phase_allowed("LR")
    backend = backend or _subprocess_backend

    def run(lrs: Sequence[float], seed_label: str) -> list[dict[str, Any]]:
        out = []
        for candidate in plan_phase_lr(
            output_root=context.output_root,
            micro_bsz=micro_bsz,
            compile_on=compile_on,
            peak_lrs=lrs,
            seed_label=seed_label,
        ):
            context.require_binds_candidate(candidate)
            try:
                backend(candidate, context)
            except PilotContractError:
                raise
            except BaseException as exc:  # noqa: BLE001
                out.append({
                    **_ineligible_evidence(candidate, context, exc),
                    "peak_lr": candidate["peak_lr"],
                })
                continue
            out.append(load_completed_result(Path(candidate["output_dir"]), context))
        return out

    seed1 = run(LR_GRID_SEED1, "seed-1")  # authoritative initial grid, internal
    require_complete_lr_grid(seed1)
    seed1_verdict = lr_select_seed1(seed1)
    if seed1_verdict["outcome"] != "SEED1_WINNER":
        report = {"phase": "LR", "seed1": seed1, "outcome": seed1_verdict}
        (context.output_root / "PHASE_LR_REPORT.json").write_bytes(canonical_json_bytes(report))
        return report

    winner_lr = float(seed1_verdict["winner_peak_lr"])
    neighbour = confirmation_neighbor(winner_lr)  # derived internally
    seed2 = run([winner_lr, neighbour], "seed-2")
    by_lr_1 = {float(r["peak_lr"]): r for r in seed1}
    by_lr_2 = {float(r["peak_lr"]): r for r in seed2}
    pairs = [
        {
            "peak_lr": lr,
            "seed1_score": float(by_lr_1[lr]["score"]),
            "seed1_eligible": lr_candidate_eligible(by_lr_1[lr])[0],
            "seed2_score": float(by_lr_2[lr].get("score", float("inf"))),
            "seed2_eligible": lr_candidate_eligible(by_lr_2[lr])[0] if lr in by_lr_2 else False,
        }
        for lr in (winner_lr, neighbour)
        if lr in by_lr_1
    ]
    confirmed = lr_confirm(pairs)
    if confirmed["outcome"] != "CONFIRMED":
        report = {"phase": "LR", "seed1": seed1, "seed2": seed2, "outcome": confirmed}
        (context.output_root / "PHASE_LR_REPORT.json").write_bytes(canonical_json_bytes(report))
        return report

    incumbent_lr = float(confirmed["confirmed_peak_lr"])
    edge_lr = edge_candidate(incumbent_lr)  # derived internally
    edge_runs: list[dict[str, Any]] = []
    edge_kwargs: dict[str, Any] = {"edge_lr": edge_lr}
    if edge_lr is not None:
        e1 = run([edge_lr], "seed-1")
        e2 = run([edge_lr], "seed-2")
        edge_runs = e1 + e2
        edge_kwargs.update({
            "edge_seed1_eligible": lr_candidate_eligible(e1[0])[0] if e1 else False,
            "edge_seed2_eligible": lr_candidate_eligible(e2[0])[0] if e2 else False,
            "edge_seed1_score": float(e1[0].get("score", float("inf"))) if e1 else None,
            "edge_seed2_score": float(e2[0].get("score", float("inf"))) if e2 else None,
        })
    final = lr_resolve_edge(
        incumbent_lr=incumbent_lr,
        incumbent_final_score=float(confirmed["final_score"]),
        **edge_kwargs,
    )
    report = {
        "phase": "LR",
        "seed1": seed1,
        "seed2": seed2,
        "edge_runs": edge_runs,
        "seed1_verdict": seed1_verdict,
        "confirmed": confirmed,
        "final": final,
        "ledger": dict(context.ledger.state),
    }
    (context.output_root / "PHASE_LR_REPORT.json").write_bytes(canonical_json_bytes(report))
    return report


def _subprocess_backend(candidate: Mapping[str, Any], context: ValidatedExecutionContext) -> None:
    """Launch one candidate in a FRESH subprocess with an immutable serialized run spec."""
    spec_dir = context.output_root / "_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{candidate['candidate_id']}.json"
    spec_path.write_bytes(
        canonical_json_bytes({
            "candidate": dict(candidate),
            "bound_identity": dict(context.ledger.identity),
            "authorized_output_root": str(context.output_root),
            "runtime_fingerprint_sha256": context.fingerprint["fingerprint_sha256"],
            "contract_sha256": context.observed["contract_sha256"],
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
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    (Path(candidate["output_dir"]).parent / f"{candidate['candidate_id']}.stdout").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"candidate subprocess exited {completed.returncode}: "
            f"{(completed.stderr or '').strip()[:400]}"
        )


# --------------------------------------------------------------------- planning + meta


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


def run_meta(*, candidate: Mapping[str, Any], context: ValidatedExecutionContext) -> dict[str, Any]:
    return {
        "schema_version": RUN_META_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "base_fingerprint_sha256": context.fingerprint["fingerprint_sha256"],
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
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
            k: context.indices[k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        },
        "serialized_index_lists_digest": context.observed["serialized_index_lists_digest"],
        "contract_sha256": context.observed["contract_sha256"],
        "implementation_head": context.observed["head"],
        "execution_implementation_bundle_sha256": context.observed["execution_bundle_sha256"],
        "authorized_scope": context.scope,
        "authorization_status": "AUTHORIZED_BY_EXTERNAL_MANIFEST",
    }


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    sub.add_parser("fingerprint")
    sub.add_parser("git-policy")
    sub.add_parser("closure")
    sub.add_parser("authorization-template")
    p = sub.add_parser("plan")
    p.add_argument("--output-root", type=Path, default=Path("runs/PILOT_V2_3_OUTPUT"))
    p.add_argument("--micro-bsz", type=int, default=None)
    r = sub.add_parser("run", help="the single orchestration entry point")
    r.add_argument("--phase", choices=["MB", "LR"], required=True)
    r.add_argument("--authorization", type=Path, default=None)
    r.add_argument("--scope", choices=list(ALLOWED_SCOPES), default="PHASE_MB_ONLY")
    r.add_argument("--output-root", type=Path, required=True)
    r.add_argument("--micro-bsz", type=int, default=None)
    r.add_argument("--compile", action="store_true")
    e = sub.add_parser("execute-candidate", help="internal: one candidate in a fresh subprocess")
    e.add_argument("--spec", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "contract":
        from pretrain.pilot_contract_v2_3 import contract_document

        sys.stdout.write(canonical_json_bytes(contract_document()).decode())
        return 0
    if args.command == "fingerprint":
        sys.stdout.write(canonical_json_bytes(base_runtime_fingerprint()).decode())
        return 0
    if args.command == "git-policy":
        sys.stdout.write(canonical_json_bytes(git_policy_status()).decode())
        return 0
    if args.command == "closure":
        sys.stdout.write(canonical_json_bytes(execution_closure()).decode())
        return 0
    if args.command == "authorization-template":
        sys.stdout.write(canonical_json_bytes(authorization_template()).decode())
        return 0
    if args.command == "plan":
        payload: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "authorization_status": "NOT_AUTHORIZED",
            "phase_mb": plan_phase_mb(output_root=args.output_root),
        }
        payload["phase_lr"] = (
            plan_phase_lr(output_root=args.output_root, micro_bsz=args.micro_bsz, compile_on=False)
            if args.micro_bsz is not None
            else "pending FROZEN_MICRO_BSZ from Phase MB"
        )
        sys.stdout.write(canonical_json_bytes(payload).decode())
        return 0
    if args.command == "execute-candidate":
        try:
            spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
            candidate = spec["candidate"]
            context = build_validated_context(
                manifest_path=None,
                requested_scope="PHASE_MB_ONLY",
                output_root=Path(spec["authorized_output_root"]),
                phase=candidate["phase"],
            )
            backend = (
                run_phase_mb_candidate if candidate["phase"] == "MB" else run_phase_lr_candidate
            )
            backend(candidate, context)
        except PilotContractError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        return 0
    if args.command == "run":
        try:
            context = build_validated_context(
                manifest_path=args.authorization,
                requested_scope=args.scope,
                output_root=args.output_root,
                phase=args.phase,
            )
            if args.phase == "MB":
                report = orchestrate_phase_mb(context)
            else:
                require(
                    args.micro_bsz is not None,
                    "--micro-bsz is required for phase LR and comes from the Phase-MB result",
                )
                report = orchestrate_phase_muon_lr(
                    context, micro_bsz=args.micro_bsz, compile_on=bool(args.compile)
                )
            sys.stdout.write(
                canonical_json_bytes({
                    "phase": report["phase"],
                    "output_root": str(args.output_root),
                }).decode()
            )
        except PilotContractError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
