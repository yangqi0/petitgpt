#!/usr/bin/env python3
"""P-PILOT-CONTRACT-V2.2 pilot runner: planning, fingerprinting and publication.

Everything here is pre-GPU. The module plans Phase-MB and Phase-LR candidates, generates and
publishes the deterministic pilot indices, builds the base runtime fingerprint and per-run
metadata, enforces the pre-launch Git policy and the output-path safety rules, and can emit an
execution proposal. It never trains: :func:`execute_candidate` refuses unconditionally, because
:func:`pretrain.pilot_contract_v2.require_launch_authorization` raises by construction.

Reuse, not reimplementation: the model comes from ``src.model``, the optimizer from
``src.optim``, packed data from ``pretrain.dataset_pretrain`` with the manifest requirement on,
and the loss mask from the canonical production helper.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.pilot_contract_v2 import (  # noqa: E402
    BASE_FINGERPRINT_SCHEMA,
    CONTRACT_VERSION,
    EFFECTIVE_BATCH_TOKENS,
    LR_GRID_SEED1,
    LR_RUN_UPDATES,
    RUN_META_SCHEMA,
    SEED_SEMANTICS,
    SEQUENCES_PER_OPTIMIZER_UPDATE,
    PilotContractError,
    canonical_json_bytes,
    contract_sha256,
    frozen_grad_accum,
    generate_pilot_indices,
    gpu_has_training_authority,
    mb_candidate_grid,
    realized_adamw_config,
    require,
    require_launch_authorization,
)

ALLOWED_UNTRACKED = ".codex_r1_manual_context_probe.py"
ALLOWED_UNTRACKED_SHA256 = "4ea1e8ef471138d9d9cf8076a6fd3bd83ce83c131287bd94c84d998b846ca76c"

# Accepted upstream releases. Pilot output may never be written inside any of these.
PROTECTED_PREFIXES = (
    "runs/m_production_v1_2026-08-29/release",
    "runs/i_production_v1_2026-08-25",
    "runs/g_production_2026-08-21/release",
    "runs/g2_production_2026-08-21/release",
    "runs/l1_production_2026-08-20",
)

PILOT_IMPLEMENTATION_BUNDLE_SCHEMA = "petitgpt-pilot-implementation-bundle-v1"
PILOT_IMPLEMENTATION_FILES = (
    "pretrain/pilot_contract_v2.py",
    "pretrain/pilot_runner_v2.py",
)


def repo_root() -> Path:
    """The repository root, independent of the caller's working directory."""
    return Path(ROOT).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------- git policy


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root()), *args],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def git_policy_status() -> dict[str, Any]:
    """Pre-launch Git policy: tracked clean, exactly one allowed historical untracked file."""
    tracked = [line for line in _git("status", "--porcelain", "-uno").splitlines() if line.strip()]
    untracked = [
        line[3:].strip()
        for line in _git("status", "--porcelain").splitlines()
        if line.startswith("??")
    ]
    probe = repo_root() / ALLOWED_UNTRACKED
    probe_sha = file_sha256(probe) if probe.is_file() else None
    unexpected = [u for u in untracked if u != ALLOWED_UNTRACKED]
    failures: list[str] = []
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
        "tracked_dirty_entries": tracked,
        "allowed_untracked": ALLOWED_UNTRACKED,
        "allowed_untracked_sha256": probe_sha,
        "allowed_untracked_unchanged": probe_sha == ALLOWED_UNTRACKED_SHA256,
        "unexpected_untracked": unexpected,
        "failures": failures,
        "policy_satisfied": not failures,
    }


def require_git_policy() -> dict[str, Any]:
    status = git_policy_status()
    require(
        status["policy_satisfied"],
        "pre-launch Git policy not satisfied: " + ", ".join(status["failures"]),
    )
    return status


# --------------------------------------------------------------------- output safety


def require_new_output_dir(destination: Path) -> Path:
    """Every pilot output directory must be new and outside every accepted release."""
    root = repo_root()
    dest = Path(destination)
    resolved = (dest if dest.is_absolute() else root / dest).resolve()
    # The accepted-release guard runs FIRST: a path inside a production release is refused for
    # that reason, whether or not it happens to exist.
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


# --------------------------------------------------------------------- fingerprint


def base_runtime_fingerprint(*, gpu_required: bool = False) -> dict[str, Any]:
    """The base runtime fingerprint. Deliberately carries NO global compile value.

    ``compile`` is a per-candidate property recorded in each run_meta, never a property of the
    machine, so the same fingerprint covers a compile-on and a compile-off probe.
    """
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
            "has_training_authority": gpu_has_training_authority(torch.cuda.get_device_name(0)),
        })
    if gpu_required:
        require(gpu.get("cuda_available"), "no CUDA device available")
        require(
            gpu.get("has_training_authority"),
            f"GPU {gpu.get('name')!r} has no training authority under {CONTRACT_VERSION}; "
            "load-bearing pilots require an RTX 4090 24GB-class device",
        )

    git = git_policy_status()
    fingerprint = {
        "schema_version": BASE_FINGERPRINT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "torch_build": {"cuda": torch.version.cuda, "git_version": torch.version.git_version},
        "numpy_version": np.__version__,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "container_template": os.environ.get("RUNPOD_POD_ID")
        or os.environ.get("CONTAINER_TEMPLATE")
        or None,
        "repository": {
            "branch": git["branch"],
            "head": git["head"],
            "tracked_clean": git["tracked_clean"],
            "allowed_untracked": ALLOWED_UNTRACKED,
            "allowed_untracked_sha256": git["allowed_untracked_sha256"],
            "allowed_untracked_unchanged": git["allowed_untracked_unchanged"],
            "unexpected_untracked": git["unexpected_untracked"],
        },
        "contract_sha256": contract_sha256(),
        "implementation_bundle_sha256": implementation_bundle()[1],
    }
    require(
        "compile" not in fingerprint,
        "the base fingerprint must not carry a global compile value",
    )
    fingerprint["fingerprint_sha256"] = hashlib.sha256(
        canonical_json_bytes(fingerprint)
    ).hexdigest()
    return fingerprint


def _nvidia_smi(field: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        ).stdout.strip()
        return out.splitlines()[0].strip() if out else None
    except Exception:  # noqa: BLE001 - absence of nvidia-smi is not an error here
        return None


def implementation_bundle(root: Path | None = None) -> tuple[dict[str, str], str]:
    """Exact byte binding of the pilot implementation files."""
    base = Path(root) if root is not None else repo_root()
    files = {}
    for relative in PILOT_IMPLEMENTATION_FILES:
        path = base / relative
        require(path.is_file(), f"pilot implementation member missing: {relative}")
        files[relative] = file_sha256(path)
    digest = hashlib.sha256(
        canonical_json_bytes({
            "schema_version": PILOT_IMPLEMENTATION_BUNDLE_SCHEMA,
            "files": dict(sorted(files.items())),
        })
    ).hexdigest()
    return files, digest


# --------------------------------------------------------------------- indices


def publish_pilot_indices(
    *,
    stage_a_blocks: int,
    stage_b_blocks: int,
    stage_a_meta_sha256: str,
    stage_b_meta_sha256: str,
    destination: Path,
    implementation_head: str,
) -> dict[str, Any]:
    """Generate and publish the deterministic pilot indices, bound to the accepted releases."""
    out = require_new_output_dir(destination)
    indices = generate_pilot_indices(stage_a_blocks, stage_b_blocks)
    manifest = {
        **{
            k: v
            for k, v in indices.items()
            if k not in ("stage_a_eval", "stage_a_train", "stage_b_eval")
        },
        "stage_a_release_meta_sha256": stage_a_meta_sha256,
        "stage_b_release_meta_sha256": stage_b_meta_sha256,
        "implementation_head": implementation_head,
        "implementation_bundle_sha256": implementation_bundle()[1],
        "contract_sha256": contract_sha256(),
        "authorization_status": "NOT_AUTHORIZED",
    }
    out.mkdir(parents=True)
    for name in ("stage_a_eval", "stage_a_train", "stage_b_eval"):
        payload = ("\n".join(str(v) for v in indices[name]) + "\n").encode("utf-8")
        (out / f"{name}.txt").write_bytes(payload)
    (out / "PILOT_INDICES.json").write_bytes(canonical_json_bytes(manifest))
    manifest["manifest_sha256"] = hashlib.sha256(
        (out / "PILOT_INDICES.json").read_bytes()
    ).hexdigest()
    return {"directory": str(out), "manifest": manifest, "indices": indices}


def blocks_consumed(updates: int) -> int:
    """Blocks a run consumes: updates x 128 sequences per optimizer update."""
    return int(updates) * SEQUENCES_PER_OPTIMIZER_UPDATE


# --------------------------------------------------------------------- candidate planning


def plan_phase_mb(*, output_root: Path) -> list[dict[str, Any]]:
    """The ten Phase-MB candidates, each with an isolated output and Inductor cache."""
    plans = []
    for candidate in mb_candidate_grid():
        cid = candidate["candidate_id"]
        plans.append({
            **candidate,
            "seed_label": "seed-1",
            "train_order_seed": SEED_SEMANTICS["seed-1"]["train_order"],
            "output_dir": str(Path(output_root) / cid),
            "inductor_cache_dir": str(Path(output_root) / cid / "inductor_cache"),
            "tokens": candidate["updates"] * EFFECTIVE_BATCH_TOKENS,
            "blocks_consumed": blocks_consumed(candidate["updates"]),
            "fresh_model": True,
            "fresh_optimizer": True,
            "inherits_checkpoint": False,
            "authorization_status": "NOT_AUTHORIZED",
        })
    return plans


def plan_phase_lr(
    *,
    output_root: Path,
    micro_bsz: int,
    compile_on: bool,
    peak_lrs: Sequence[float] = LR_GRID_SEED1,
    seed_label: str = "seed-1",
) -> list[dict[str, Any]]:
    """Phase-LR candidates at the frozen geometry."""
    require(seed_label in SEED_SEMANTICS, f"unknown seed label {seed_label!r}")
    seeds = SEED_SEMANTICS[seed_label]
    plans = []
    for peak_lr in peak_lrs:
        cid = f"lr_{peak_lr:g}_{seed_label}"
        plans.append({
            "candidate_id": cid,
            "phase": "LR",
            "seed_label": seed_label,
            "micro_bsz": int(micro_bsz),
            "grad_accum": frozen_grad_accum(micro_bsz),
            "compile": bool(compile_on),
            "peak_lr": float(peak_lr),
            "updates": LR_RUN_UPDATES,
            "warmup_updates": 50,
            "model_init_seed": seeds["model_init"],
            "train_order_seed": seeds["train_order"],
            "output_dir": str(Path(output_root) / cid),
            "inductor_cache_dir": str(Path(output_root) / cid / "inductor_cache"),
            "tokens": LR_RUN_UPDATES * EFFECTIVE_BATCH_TOKENS,
            "blocks_consumed": blocks_consumed(LR_RUN_UPDATES),
            "fresh_model": True,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "inherits_checkpoint": False,
            "authorization_status": "NOT_AUTHORIZED",
        })
    return plans


def run_meta(
    *,
    candidate: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    implementation_head: str,
) -> dict[str, Any]:
    """Per-run metadata binding every field the contract requires."""
    return {
        "schema_version": RUN_META_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "base_fingerprint_sha256": fingerprint["fingerprint_sha256"],
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "optimizer": realized_adamw_config(),
        "lr_configuration": {
            "peak_lr": candidate["peak_lr"],
            "warmup_updates": candidate["warmup_updates"],
            "updates": candidate["updates"],
        },
        "model_seed": candidate["model_init_seed"],
        "train_order_seed": candidate["train_order_seed"],
        "pilot_index_hashes": {
            "stage_a_eval_sha256": index_manifest["stage_a_eval_sha256"],
            "stage_a_train_sha256": index_manifest["stage_a_train_sha256"],
            "stage_b_eval_sha256": index_manifest["stage_b_eval_sha256"],
        },
        "contract_sha256": contract_sha256(),
        "implementation_head": implementation_head,
        "implementation_bundle_sha256": implementation_bundle()[1],
        "authorization_status": "NOT_AUTHORIZED",
    }


# --------------------------------------------------------------------- execution refusal


def execute_candidate(*_args: Any, **_kwargs: Any) -> None:
    """Refused unconditionally in this materialization segment."""
    require_launch_authorization()


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fingerprint", help="print the base runtime fingerprint")
    sub.add_parser("contract", help="print the machine-readable V2.2 contract")
    sub.add_parser("git-policy", help="check the pre-launch Git policy")
    p = sub.add_parser("plan", help="print the Phase-MB and Phase-LR candidate plans (dry run)")
    p.add_argument("--output-root", type=Path, default=Path("runs/PILOT_OUTPUT_ROOT"))
    p.add_argument(
        "--micro-bsz",
        type=int,
        default=None,
        help="Phase-LR geometry; omitted until Phase MB freezes it",
    )
    idx = sub.add_parser("indices", help="generate and publish the deterministic pilot indices")
    idx.add_argument("--stage-a-blocks", type=int, required=True)
    idx.add_argument("--stage-b-blocks", type=int, required=True)
    idx.add_argument("--stage-a-meta-sha256", required=True)
    idx.add_argument("--stage-b-meta-sha256", required=True)
    idx.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="execute a pilot candidate (always refused here)")
    run.add_argument("--candidate", required=True)

    args = parser.parse_args(argv)
    if args.command == "contract":
        from pretrain.pilot_contract_v2 import contract_document

        sys.stdout.write(canonical_json_bytes(contract_document()).decode("utf-8"))
        return 0
    if args.command == "fingerprint":
        sys.stdout.write(canonical_json_bytes(base_runtime_fingerprint()).decode("utf-8"))
        return 0
    if args.command == "git-policy":
        sys.stdout.write(canonical_json_bytes(git_policy_status()).decode("utf-8"))
        return 0
    if args.command == "plan":
        payload: dict[str, Any] = {
            "contract_version": CONTRACT_VERSION,
            "authorization_status": "NOT_AUTHORIZED",
            "phase_mb": plan_phase_mb(output_root=args.output_root),
        }
        if args.micro_bsz is not None:
            payload["phase_lr"] = plan_phase_lr(
                output_root=args.output_root, micro_bsz=args.micro_bsz, compile_on=False
            )
        else:
            payload["phase_lr"] = "pending FROZEN_MICRO_BSZ from Phase MB"
        sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))
        return 0
    if args.command == "indices":
        head = git_policy_status()["head"]
        result = publish_pilot_indices(
            stage_a_blocks=args.stage_a_blocks,
            stage_b_blocks=args.stage_b_blocks,
            stage_a_meta_sha256=args.stage_a_meta_sha256,
            stage_b_meta_sha256=args.stage_b_meta_sha256,
            destination=args.out,
            implementation_head=head,
        )
        sys.stdout.write(canonical_json_bytes(result["manifest"]).decode("utf-8"))
        return 0
    if args.command == "run":
        try:
            execute_candidate(candidate=args.candidate)
        except PilotContractError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
