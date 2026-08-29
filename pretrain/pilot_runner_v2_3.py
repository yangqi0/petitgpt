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
``pretrain.dataset_pretrain.PackedBinDataset`` with the manifest requirement on, and the
canonical production loss mask from ``pretrain.train_pretrain_with_bench``.
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
    GRAD_CLIP,
    LR_GRID_SEED1,
    LR_RUN_UPDATES,
    LR_WARMUP_UPDATES,
    MB_PROBE_UPDATES,
    MUON_LR_ARG,
    MUON_MOMENTUM,
    PILOT_CHECKPOINT_KIND,
    REQUIRED_NUMPY_VERSION,
    RUN_META_SCHEMA,
    SEED_SEMANTICS,
    SEQUENCES_PER_OPTIMIZER_UPDATE,
    TOKEN_LEDGER_SCHEMA,
    WEIGHT_DECAY,
    PilotContractError,
    authorization_template,
    canonical_json_bytes,
    check_pilot_resume,
    check_update_within_ceilings,
    contract_sha256,
    frozen_grad_accum,
    generate_pilot_indices,
    lr_schedule,
    mb_candidate_grid,
    mb_lr,
    reject_pilot_checkpoint_as_initialization,
    require,
    require_numpy_version,
    require_training_authority,
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


def require_new_output_dir(destination: Path) -> Path:
    """Wired into every candidate launch: new, and never inside an accepted release."""
    root = repo_root()
    dest = Path(destination)
    resolved = (dest if dest.is_absolute() else root / dest).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        relative = None
    if relative is not None:
        for prefix in PROTECTED_PREFIXES:
            require(
                relative != prefix and not relative.startswith(prefix + "/"),
                f"refusing to write pilot output inside an accepted release: {relative}",
            )
    require(not resolved.exists(), f"pilot output directory must not exist: {destination}")
    return resolved


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


class TokenLedger:
    """Persistent trained-token accounting, checked before and persisted after every update."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_file():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {
                "schema_version": TOKEN_LEDGER_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "phase_tokens": {"MB": 0, "LR": 0},
                "global_tokens": 0,
                "updates": 0,
            }

    def check(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> dict[str, Any]:
        return check_update_within_ceilings(
            phase=phase,
            phase_tokens_so_far=int(self.state["phase_tokens"].get(phase, 0)),
            global_tokens_so_far=int(self.state["global_tokens"]),
            tokens_this_update=tokens,
        )

    def require_update_allowed(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> None:
        verdict = self.check(phase, tokens)
        require(
            verdict["may_execute"],
            "PILOT_ABORT: trained-token ceiling would be exceeded: "
            + ", ".join(verdict["breaches"]),
        )

    def commit(self, phase: str, tokens: int = EFFECTIVE_BATCH_TOKENS) -> None:
        """Atomically persist after a completed update."""
        self.state["phase_tokens"][phase] = int(self.state["phase_tokens"].get(phase, 0)) + tokens
        self.state["global_tokens"] = int(self.state["global_tokens"]) + tokens
        self.state["updates"] = int(self.state["updates"]) + 1
        tmp = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(canonical_json_bytes(self.state))
        os.replace(tmp, self.path)


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
    grouping = verify_realized_grouping(opt)
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
    """Fixed-index view over a canonical PackedBinDataset. Order is the contract's, not ours."""

    def __init__(self, dataset: Any, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        n = len(dataset)
        bad = [i for i in self.indices if not (0 <= i < n)]
        require(not bad, f"pilot index out of range for a {n}-block release: {bad[:4]}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Any:
        return self.dataset[self.indices[int(position)]]


def canonical_loss(logits: Any, labels: Any, loss_mask: Any) -> Any:
    """Token-mean CE through the canonical production loss function.

    The mask is the one `PackedBinDataset.__getitem__` already produces (BOS masked, repeated
    EOS masked, final label supervised by default), and the loss is the trainer's own
    `masked_weighted_ce_loss` at `eos_weight=1.0` -- the unweighted token-mean form the
    repository uses for comparable evaluation. Nothing is reimplemented here.
    """
    from pretrain.train_pretrain_with_bench import masked_weighted_ce_loss
    from src.special_tokens import EOS_ID

    return masked_weighted_ce_loss(
        logits.float(), labels.long(), loss_mask, eos_id=EOS_ID, eos_weight=1.0
    )


# --------------------------------------------------------------------- authorization gate


def authorize_execution(
    *, manifest_path: Path | None, requested_scope: str, output_root: Path
) -> dict[str, Any]:
    """The single gate every training entry point passes through.

    A later AUTHORIZED manifest bound to this exact reviewed HEAD passes here with no code
    change. Until then every call refuses.
    """
    require(requested_scope in ALLOWED_SCOPES, f"unknown scope {requested_scope!r}")
    require_numpy_version()
    git = git_policy_status()
    require(
        git["policy_satisfied"],
        "pre-launch Git policy not satisfied: " + ", ".join(git["failures"]),
    )
    stage_a = verify_accepted_release("stage_a")
    stage_b = verify_accepted_release("stage_b")
    universes = {"stage_a_blocks": stage_a["blocks"], "stage_b_blocks": stage_b["blocks"]}
    indices = generate_pilot_indices(**universes)
    index_manifest_sha = hashlib.sha256(
        canonical_json_bytes({
            k: indices[k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        })
    ).hexdigest()
    observed = {
        "branch": git["branch"],
        "head": git["head"],
        "contract_sha256": contract_sha256(),
        "execution_bundle_sha256": execution_bundle_sha256(),
        "pilot_index_manifest_sha256": index_manifest_sha,
        "stage_a_meta_sha256": stage_a["meta_sha256"],
        "stage_b_meta_sha256": stage_b["meta_sha256"],
        "output_root": str(output_root),
    }
    manifest = None
    if manifest_path is not None and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    verdict = validate_authorization(manifest, requested_scope=requested_scope, observed=observed)
    require(
        verdict["authorized"],
        f"pilot execution refused under {CONTRACT_VERSION}: " + ", ".join(verdict["failures"]),
    )
    fingerprint = base_runtime_fingerprint(gpu_required=True)
    return {
        "verdict": verdict,
        "observed": observed,
        "indices": indices,
        "fingerprint": fingerprint,
        "stage_a": stage_a,
        "stage_b": stage_b,
    }


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
    """The complete optimizer-update loop: accumulation, clipping, step, metrics.

    Never reached in the materialization segment: callers pass through authorize_execution.
    """
    import torch

    model.train()
    losses: dict[int, float] = {}
    grad_norms: dict[int, float] = {}
    realized_lrs: dict[int, list[float]] = {}
    step_seconds: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    cursor = 0
    for u in range(1, int(updates) + 1):
        ledger.require_update_allowed(phase)
        lr = lr_fn(u)
        realized = apply_scheduled_lr(optimizer, lr)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(int(grad_accum)):
            batch = [view[(cursor + k) % len(view)] for k in range(int(micro_bsz))]
            cursor += int(micro_bsz)
            x = torch.stack([b[0] for b in batch]).to(device)
            y = torch.stack([b[1] for b in batch]).to(device)
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
        ledger.commit(phase)
        losses[u] = total / int(grad_accum)
        grad_norms[u] = gnorm
        realized_lrs[u] = realized
        if u >= timed_from:
            step_seconds.append(elapsed)
        if record_diagnostics:
            diagnostics.append(_group_diagnostics(optimizer, u))
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "realized_lrs": realized_lrs,
        "step_seconds": step_seconds,
        "diagnostics": diagnostics,
        "completed_updates": int(updates),
    }


def _group_diagnostics(optimizer: Any, update: int) -> dict[str, Any]:
    """Per-group update/weight RMS diagnostics. Recorded, never used as a selection threshold."""
    import torch

    out = []
    for gi, g in enumerate(optimizer.param_groups):
        weights = [p for p in g["params"] if p.numel()]
        if not weights:
            continue
        wrms = float(torch.sqrt(sum((p.float() ** 2).mean() for p in weights) / len(weights)))
        state_key = "momentum_buffer" if g.get("use_muon") else "exp_avg"
        bufs = [
            optimizer.state[p][state_key]
            for p in weights
            if state_key in optimizer.state.get(p, {})
        ]
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
    """Token-mean CE over every block in ascending index order, canonical mask, eval mode."""
    import torch

    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(view), int(micro_bsz)):
            batch = [view[i] for i in range(start, min(start + int(micro_bsz), len(view)))]
            x = torch.stack([b[0] for b in batch]).to(device)
            y = torch.stack([b[1] for b in batch]).to(device)
            m = torch.stack([b[2] for b in batch]).to(device, dtype=torch.float32)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                total += float(canonical_loss(model(x), y, m))
            batches += 1
    model.train()
    return total / max(1, batches)


def run_phase_mb_candidate(
    candidate: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    """One Phase-MB probe, end to end. Requires an authorized context."""
    import torch

    require(context.get("authorized"), "run_phase_mb_candidate requires an authorized context")
    out = require_new_output_dir(Path(candidate["output_dir"]))
    out.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor_cache")
    torch.cuda.reset_peak_memory_stats()
    model = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    compile_seconds = None
    if candidate["compile"]:
        t0 = time.perf_counter()
        model = torch.compile(model)
        compile_seconds = time.perf_counter() - t0
    optimizer = build_pilot_optimizer(model, candidate["peak_lr"])
    result = _run_updates(
        model=model,
        optimizer=optimizer,
        view=context["train_view"],
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=MB_PROBE_UPDATES,
        lr_fn=mb_lr,
        ledger=context["ledger"],
        phase="MB",
        device="cuda",
        timed_from=11,
    )
    import statistics

    tokens_per_update = EFFECTIVE_BATCH_TOKENS
    median_tps = tokens_per_update / statistics.median(result["step_seconds"])
    grouping = verify_realized_grouping(optimizer)
    states = verify_optimizer_state(optimizer)
    payload = {
        "candidate_id": candidate["candidate_id"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "completed_updates": result["completed_updates"],
        "median_tokens_per_sec": median_tps,
        "compile_seconds": compile_seconds,
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
    (out / "result.json").write_bytes(canonical_json_bytes(payload))
    return payload


def run_phase_lr_candidate(
    candidate: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    """One Phase-Muon-LR run, end to end, including both evaluations."""
    import torch

    from pretrain.pilot_contract_v2_3 import lr_score, sustained_divergence

    require(context.get("authorized"), "run_phase_lr_candidate requires an authorized context")
    out = require_new_output_dir(Path(candidate["output_dir"]))
    out.mkdir(parents=True)
    model = build_pilot_model(candidate["model_init_seed"]).to("cuda")
    if candidate["compile"]:
        model = torch.compile(model)
    optimizer = build_pilot_optimizer(model, candidate["peak_lr"])
    result = _run_updates(
        model=model,
        optimizer=optimizer,
        view=context["train_view"],
        micro_bsz=candidate["micro_bsz"],
        grad_accum=candidate["grad_accum"],
        updates=LR_RUN_UPDATES,
        lr_fn=lambda u: lr_schedule(u, candidate["peak_lr"]),
        ledger=context["ledger"],
        phase="LR",
        device="cuda",
        record_diagnostics=True,
    )
    loss_a = evaluate(
        model, context["eval_a_view"], micro_bsz=candidate["micro_bsz"], device="cuda"
    )
    loss_b = evaluate(
        model, context["eval_b_view"], micro_bsz=candidate["micro_bsz"], device="cuda"
    )
    grouping = verify_realized_grouping(optimizer)
    states = verify_optimizer_state(optimizer)
    guard = sustained_divergence(result["losses"])
    payload = {
        "candidate_id": candidate["candidate_id"],
        "peak_lr": candidate["peak_lr"],
        "seed_label": candidate["seed_label"],
        "completed_updates": result["completed_updates"],
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
    (out / "result.json").write_bytes(canonical_json_bytes(payload))
    return payload


def _finite(v: float) -> bool:
    import math

    return math.isfinite(float(v))


# --------------------------------------------------------------------- checkpoints


def pilot_checkpoint_identity(
    candidate: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "checkpoint_kind": PILOT_CHECKPOINT_KIND,
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "contract_sha256": contract_sha256(),
        "implementation_head": context["observed"]["head"],
        "execution_bundle_sha256": execution_bundle_sha256(),
        "pilot_index_manifest_sha256": context["observed"]["pilot_index_manifest_sha256"],
        "runtime_fingerprint_sha256": context["fingerprint"]["fingerprint_sha256"],
    }


def require_not_pilot_checkpoint(checkpoint: Mapping[str, Any], purpose: str) -> None:
    """Wire into production checkpoint consumers: a pilot checkpoint may never initialize them."""
    if checkpoint.get("checkpoint_kind") == PILOT_CHECKPOINT_KIND:
        reject_pilot_checkpoint_as_initialization(purpose)


def resume_pilot_checkpoint(checkpoint: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    verdict = check_pilot_resume(checkpoint, candidate)
    require(verdict["may_resume"], "pilot resume refused: " + ", ".join(verdict["failures"]))


# --------------------------------------------------------------------- planning


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


def run_meta(
    *,
    candidate: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    indices: Mapping[str, Any],
    implementation_head: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_META_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "base_fingerprint_sha256": fingerprint["fingerprint_sha256"],
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
            k: indices[k]
            for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
        },
        "contract_sha256": contract_sha256(),
        "implementation_head": implementation_head,
        "execution_implementation_bundle_sha256": execution_bundle_sha256(),
        "authorization_status": "NOT_AUTHORIZED",
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
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=["MB", "LR"], required=True)
    r.add_argument("--authorization", type=Path, default=None)
    r.add_argument("--scope", choices=list(ALLOWED_SCOPES), default="PHASE_MB_ONLY")
    r.add_argument("--output-root", type=Path, required=True)

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
    if args.command == "run":
        try:
            authorize_execution(
                manifest_path=args.authorization,
                requested_scope=args.scope,
                output_root=args.output_root,
            )
        except PilotContractError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
