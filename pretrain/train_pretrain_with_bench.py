# pretrain/train_pretrain.py
# Robust GPT pretraining for petitgpt (causal LM)
# - PackedBinDataset: returns (input_ids_u16, labels_u16, loss_mask_f32/u8)
# - token-level loss mask (MiniMind-style): weighted reduction by sum(mask)
# - optional EOS down-weight warmup to avoid early EOS collapse
# - gradient accumulation, bf16/fp16 autocast, grad clip, checkpoints, eval, samples
# - debug: dataset stats, label shift sanity, future-leak checks

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Make imports work no matter where the script is launched from.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pretrain import (  # noqa: E402
    FixedSubsetSampler,
    PackedBinDataset,
    ResumablePermutationSampler,
)
from sample import generate_default_samples  # noqa: E402

from pretrain.run_plan_contract import (  # noqa: E402
    load_run_plan_binding,
    resolve_run_plan_sample_budget,
    synchronize_validated_run_plan_binding,
    validate_run_plan_dataset,
    validate_run_plan_resume_transition,
    validate_run_plan_validation_dataset,
)
from src.canonical_loss import (  # noqa: E402
    masked_weighted_ce_components,
    masked_weighted_ce_loss,
)
from src.canonical_schedule import lr_schedule  # noqa: E402
from src.model import (  # noqa: E402
    GPT,
    GPTConfig,
    audit_gpt_parameter_count,
)
from src.optim import build_optimizer  # noqa: E402
from src.special_tokens import (  # noqa: E402
    CANONICAL_VOCAB_SIZE,
    assert_tokenizer_contract,
)
from src.tracking import Tracker  # noqa: E402

# -----------------------------------------------------------------------------
# Performance toggles
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def maybe_ppl(loss_value: float, eos_weight: float) -> float | None:
    # only when eos_weight==1, treat it as a more standard ppl
    if abs(float(eos_weight) - 1.0) > 1e-12:
        return None
    try:
        # avoid extreme values causing overflow
        if loss_value > 20:
            return None
        return float(math.exp(loss_value))
    except Exception:
        return None


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.exists():
        return str(path)
    alt = PROJECT_ROOT / p
    if alt.exists():
        return str(alt)
    return str(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_bench_eval_v5(
    *,
    bench_path: str,
    bench_script: str,
    out_json: str,
    ckpt_path: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_new_tokens: int,
    min_new_tokens: int,
    ban_first_steps: int,
) -> None:
    """Run eval_bench_v5.py as a subprocess, writing results to out_json."""
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    script_path = _resolve_path(bench_script)
    cmd = [
        sys.executable,
        script_path,
        "--bench",
        bench_path,
        "--out",
        out_json,
        "--ckpt",
        ckpt_path,
        "--tokenizer_path",
        tokenizer_path,
        "--greedy",
        "--max_seq_len",
        str(int(max_seq_len)),
        "--max_new_tokens",
        str(int(max_new_tokens)),
        "--min_new_tokens",
        str(int(min_new_tokens)),
        "--avoid_first_whitespace",
        "--ban_first_steps",
        str(int(ban_first_steps)),
        "--repetition_penalty",
        "1.0",
        "--no_repeat_ngram_size",
        "0",
        "--max_repeat_token",
        "0",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        if out.strip():
            print(out.strip().splitlines()[-1], flush=True)
    except subprocess.CalledProcessError as e:
        print("[bench_eval] FAILED", flush=True)
        print(e.output, flush=True)


def build_data_contract(
    train_dir: Path,
    dataset: PackedBinDataset,
    args: argparse.Namespace,
) -> dict:
    """Cheap, reproducible stage-data fingerprint without rereading multi-GB shards."""
    manifest = []
    for shard in dataset.shards:
        stat = shard.stat()
        manifest.append({
            "name": shard.name,
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    meta_hashes = {}
    for meta_label, meta_path in (
        ("train", train_dir / "meta.json"),
        ("parent", train_dir.parent / "meta.json"),
    ):
        if meta_path.is_file():
            meta_hashes[meta_label] = _sha256_file(meta_path)
    fingerprint_payload = {
        "manifest": manifest,
        "meta_sha256": meta_hashes,
        "dtype": str(getattr(dataset, "_dtype", "unknown")),
        "total_raw_tokens": int(dataset.total_raw_tokens),
        "usable_transitions": int(dataset.usable_transitions),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "train_dir": str(train_dir.resolve()),
        "fingerprint": fingerprint,
        "dataset_length": int(len(dataset)),
        "total_raw_tokens": int(dataset.total_raw_tokens),
        "usable_transitions": int(dataset.usable_transitions),
        "sampling_mode": str(dataset.sampling_mode),
        "sampler_seed": int(getattr(args, "sampler_seed", getattr(args, "seed", 1234))),
        "data_stage_start_step": int(args.data_stage_start_step),
        "samples_per_optimizer_step": int(args.micro_bsz) * int(args.grad_accum),
    }


def validate_data_resume_state(
    *,
    saved_data_contract: dict | None,
    current_data_contract: dict,
    saved_sampler_state: dict | None,
    current_sampler: ResumablePermutationSampler,
    global_step: int,
    data_stage_start_step: int,
    strict: bool,
) -> None:
    """Restore sampler state only for a proven same-stage exact continuation."""
    at_stage_boundary = int(global_step) == int(data_stage_start_step)
    same_stage = (
        isinstance(saved_data_contract, dict)
        and saved_data_contract.get("fingerprint") == current_data_contract.get("fingerprint")
        and saved_data_contract.get("data_stage_start_step")
        == current_data_contract.get("data_stage_start_step")
    )
    if at_stage_boundary and not same_stage:
        boundary_issues = []
        if not isinstance(saved_data_contract, dict):
            boundary_issues.append("checkpoint has no previous-stage data_contract")
        if not isinstance(saved_sampler_state, dict):
            boundary_issues.append("checkpoint has no previous-stage data_sampler state")
        else:
            if int(saved_sampler_state.get("version", 0)) != 2:
                boundary_issues.append("previous-stage sampler state is not schema version 2")
            try:
                range_start = int(saved_sampler_state["range_start_position"])
                committed = int(saved_sampler_state["committed_position"])
                end_position = int(saved_sampler_state["end_position"])
            except (KeyError, TypeError, ValueError):
                boundary_issues.append("previous-stage sampler positions are missing or invalid")
            else:
                if committed != end_position:
                    boundary_issues.append(
                        "previous-stage sampler is incomplete: "
                        f"committed={committed}, planned_end={end_position}"
                    )
                if isinstance(saved_data_contract, dict):
                    try:
                        previous_start = int(saved_data_contract["data_stage_start_step"])
                        samples_per_step = int(saved_data_contract["samples_per_optimizer_step"])
                        expected_committed = (
                            range_start + (int(global_step) - previous_start) * samples_per_step
                        )
                    except (KeyError, TypeError, ValueError):
                        boundary_issues.append(
                            "previous-stage data contract cannot prove committed exposure"
                        )
                    else:
                        if committed != expected_committed:
                            boundary_issues.append(
                                "previous-stage committed position does not match checkpoint step: "
                                f"committed={committed}, expected={expected_committed}"
                            )

        if boundary_issues:
            detail = "\n  - ".join(boundary_issues)
            if strict:
                raise RuntimeError(f"[resume] unproven previous-stage completion:\n  - {detail}")
            warnings.warn(
                f"[resume] unproven previous-stage completion; resetting new-stage "
                f"sampler only because strict resume is disabled:\n  - {detail}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            print(
                "[resume] verified previous-stage sampler fully committed; "
                "new-stage sampler starts at 0"
            )
        return

    mismatches = []
    if not isinstance(saved_data_contract, dict):
        mismatches.append("checkpoint has no data_contract")
    else:
        for key in (
            "fingerprint",
            "dataset_length",
            "sampling_mode",
            "sampler_seed",
            "data_stage_start_step",
            "samples_per_optimizer_step",
        ):
            if saved_data_contract.get(key) != current_data_contract.get(key):
                mismatches.append(
                    f"data_contract.{key}: checkpoint={saved_data_contract.get(key)!r}, "
                    f"current={current_data_contract.get(key)!r}"
                )

    expected_sampler = current_sampler.state_dict()
    if not isinstance(saved_sampler_state, dict):
        mismatches.append("checkpoint has no data_sampler state")
    else:
        for key in ("data_length", "seed", "committed_position", "end_position"):
            if saved_sampler_state.get(key) != expected_sampler.get(key):
                mismatches.append(
                    f"data_sampler.{key}: checkpoint={saved_sampler_state.get(key)!r}, "
                    f"current={expected_sampler.get(key)!r}"
                )

    if mismatches:
        detail = "\n  - ".join(mismatches)
        if strict:
            raise RuntimeError(f"[resume] same-stage data/sampler contract mismatch:\n  - {detail}")
        warnings.warn(
            f"[resume] same-stage data/sampler mismatch; using step-derived position:\n  - {detail}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    current_sampler.load_state_dict(saved_sampler_state)
    print(
        f"[resume] restored verified sampler position={current_sampler.position:,} "
        f"remaining={len(current_sampler):,}"
    )


def _resolve_resume_path(resume_path: str, out_dir: Path, resume_step: int = -1) -> Path | None:
    """Resolve a checkpoint path.

    Accepts:
      - a direct .pt file path
      - a directory containing latest.pt and/or step_XXXXXX.pt
      - a relative path (resolved against PROJECT_ROOT)
    """
    if not resume_path:
        return None

    p = Path(_resolve_path(resume_path))

    # If user passed an output directory, select a checkpoint inside it.
    if p.is_dir():
        if resume_step is not None and resume_step >= 0:
            cand = p / f"step_{resume_step:06d}.pt"
            if cand.exists():
                return cand
            # Sometimes users keep checkpoints under out_dir/ckpt or similar; try a shallow search.
            cand2 = next(p.glob(f"**/step_{resume_step:06d}.pt"), None)
            if cand2 is not None and cand2.exists():
                return cand2
            return None

        latest = p / "latest.pt"
        if latest.exists():
            return latest

        # Fallback to the largest step_*.pt
        steps = sorted(p.glob("step_*.pt"))
        if steps:
            return steps[-1]
        return None

    # If it's a file, use it.
    if p.is_file():
        return p

    # If not found, try interpreting it relative to out_dir.
    p2 = out_dir / resume_path
    if p2.is_file():
        return p2

    return None


def set_seed(seed: int) -> None:
    import random

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_schedule_branch(
    saved_schedule: dict,
    current_schedule: dict,
    *,
    checkpoint_step: int,
    base_lr: float,
) -> None:
    """Allow changed future LR only when every already-used LR is identical."""
    required = {
        "name",
        "warmup_steps",
        "schedule_total_steps",
        "decay_start_step",
        "decay_end_step",
        "min_lr_ratio",
    }
    if not required.issubset(saved_schedule) or not required.issubset(current_schedule):
        raise RuntimeError("[resume] schedule branch requires complete v1 schedule metadata")

    def value(step: int, spec: dict) -> float:
        return lr_schedule(
            step,
            int(spec["warmup_steps"]),
            float(base_lr),
            schedule=str(spec["name"]),
            schedule_total_steps=int(spec["schedule_total_steps"]),
            decay_start_step=int(spec["decay_start_step"]),
            decay_end_step=int(spec["decay_end_step"]),
            min_lr_ratio=float(spec["min_lr_ratio"]),
        )

    for step in range(max(0, int(checkpoint_step))):
        saved_lr = value(step, saved_schedule)
        current_lr = value(step, current_schedule)
        if not math.isclose(saved_lr, current_lr, rel_tol=1e-12, abs_tol=1e-18):
            raise RuntimeError(
                "[resume] rejected schedule branch: LR histories diverge before the "
                f"checkpoint at step {step} ({saved_lr:.17g} != {current_lr:.17g})"
            )


def normalize_save_steps(
    raw_values: object,
    *,
    schedule_total_steps: int,
) -> list[int]:
    """Parse repeatable/comma-separated absolute checkpoint milestones."""
    if raw_values is None:
        values: list[object] = []
    elif isinstance(raw_values, (str, int)):
        values = [raw_values]
    else:
        try:
            values = list(raw_values)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError("--save_steps must be integers separated by commas") from e

    parsed: list[int] = []
    for value in values:
        if isinstance(value, int):
            parsed.append(int(value))
            continue
        text = str(value).strip()
        if not text:
            raise ValueError("--save_steps entries must not be empty")
        fields = text.split(",")
        if any(not field.strip() for field in fields):
            raise ValueError("--save_steps contains an empty comma-separated entry")
        try:
            parsed.extend(int(field.strip()) for field in fields)
        except ValueError as e:
            raise ValueError("--save_steps must contain only integer steps") from e

    if any(step <= 0 for step in parsed):
        raise ValueError("--save_steps must contain positive absolute optimizer steps")
    if parsed != sorted(parsed) or len(parsed) != len(set(parsed)):
        raise ValueError("--save_steps must be strictly increasing and unique")
    if parsed and parsed[-1] > int(schedule_total_steps):
        raise ValueError(
            "--save_steps cannot exceed --schedule_total_steps "
            f"({parsed[-1]} > {int(schedule_total_steps)})"
        )
    return parsed


def should_save_checkpoint(
    global_step: int,
    *,
    save_every: int,
    save_steps: list[int] | tuple[int, ...] | set[int],
) -> bool:
    """Return true once when a periodic or explicit absolute milestone is hit."""
    step = int(global_step)
    return step % int(save_every) == 0 or step in save_steps


def should_retain_step_checkpoint(
    global_step: int,
    *,
    save_steps: list[int] | tuple[int, ...] | set[int],
    invocation_final_step: int,
) -> bool:
    """Retain named checkpoints only for explicit milestones and invocation final."""
    step = int(global_step)
    return step == int(invocation_final_step) or step in save_steps


def validate_training_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or discontinuous production schedules up front."""
    if int(args.max_steps) <= 0:
        raise ValueError("--max_steps must be a positive absolute global stop step")
    if int(args.schedule_total_steps) <= 0:
        args.schedule_total_steps = int(args.max_steps)
    if int(args.schedule_total_steps) < int(args.max_steps):
        raise ValueError("--schedule_total_steps must be >= the absolute --max_steps stop")
    args.save_steps = normalize_save_steps(
        getattr(args, "save_steps", []),
        schedule_total_steps=int(args.schedule_total_steps),
    )
    if int(args.warmup_steps) < 0:
        raise ValueError("--warmup_steps must be >= 0")
    if int(args.warmup_steps) >= int(args.schedule_total_steps):
        raise ValueError("--warmup_steps must be smaller than --schedule_total_steps")
    if not (0.0 <= float(args.min_lr_ratio) <= 1.0):
        raise ValueError("--min_lr_ratio must be in [0, 1]")
    if int(args.data_stage_start_step) < 0:
        raise ValueError("--data_stage_start_step must be >= 0")
    if int(args.data_stage_start_step) >= int(args.max_steps):
        raise ValueError("--data_stage_start_step must be smaller than --max_steps")
    if int(args.micro_bsz) <= 0 or int(args.grad_accum) <= 0:
        raise ValueError("--micro_bsz and --grad_accum must be positive")

    model_seed = int(getattr(args, "seed", 1234))
    sampler_seed = int(getattr(args, "sampler_seed", 1234))
    val_seed = int(getattr(args, "val_seed", 1234))
    if model_seed < 0 or sampler_seed < 0 or val_seed < 0:
        raise ValueError("--seed, --sampler_seed, and --val_seed must be non-negative")
    val_samples = int(getattr(args, "val_samples", 200))
    val_samples_per_source = int(getattr(args, "val_samples_per_source", 80))
    if val_samples < 0 or val_samples_per_source < 0:
        raise ValueError("--val_samples and --val_samples_per_source must be non-negative")
    if int(getattr(args, "vocab_size", CANONICAL_VOCAB_SIZE)) != CANONICAL_VOCAB_SIZE:
        raise ValueError(f"--vocab_size must be exactly {CANONICAL_VOCAB_SIZE}")
    if not bool(getattr(args, "add_bos_to_prompts", True)):
        raise ValueError("--no-add_bos_to_prompts violates the canonical BOS prompt contract")

    if args.lr_schedule == "wsd":
        if int(args.decay_start_step) < int(args.warmup_steps):
            raise ValueError("WSD requires --decay_start_step >= --warmup_steps")
        if int(args.decay_end_step) <= int(args.decay_start_step):
            raise ValueError("WSD requires --decay_end_step > --decay_start_step")
        if int(args.decay_end_step) > int(args.schedule_total_steps):
            raise ValueError("--decay_end_step must be <= --schedule_total_steps")
    elif int(args.decay_start_step) >= 0 or int(args.decay_end_step) >= 0:
        raise ValueError("--decay_start_step/--decay_end_step require --lr_schedule wsd")

    if bool(args.mask_last_label_in_loss) and bool(args.no_mask_last_label_in_loss):
        raise ValueError("conflicting final-label mask flags")
    if bool(args.no_mask_last_label_in_loss):
        warnings.warn(
            "--no_mask_last_label_in_loss is deprecated and now a no-op because "
            "valid final labels are supervised by default; remove the flag.",
            FutureWarning,
            stacklevel=2,
        )
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("--num_workers must be >= 0")
    if int(args.eos_weight_warmup_steps) < 0 or float(args.eos_weight) < 0.0:
        raise ValueError("EOS weight and warmup must be non-negative")
    for interval_name in ("log_every", "eval_every", "save_every", "debug_every"):
        if int(getattr(args, interval_name)) <= 0:
            raise ValueError(f"--{interval_name} must be positive")
    if int(args.resume_step) < -1:
        raise ValueError("--resume_step must be -1 or a non-negative checkpoint step")
    if bool(args.mask_last_label_in_loss):
        warnings.warn(
            "--mask_last_label_in_loss is a legacy ablation that discards a valid target; "
            "do not use it for canonical training.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.resume_path and not bool(args.resume_full):
        if int(args.data_stage_start_step) > 0:
            raise ValueError("Stage B/nonzero --data_stage_start_step requires --resume_full")
        if not bool(args.allow_weights_only_resume):
            raise ValueError(
                "--resume_path requires --resume_full by default; use "
                "--allow_weights_only_resume only for an intentional migration"
            )
    if bool(args.allow_weights_only_resume) and bool(args.resume_full):
        raise ValueError("--allow_weights_only_resume conflicts with --resume_full")
    if bool(args.allow_weights_only_resume) and not args.resume_path:
        raise ValueError("--allow_weights_only_resume requires --resume_path")
    if bool(args.allow_schedule_branch) and not bool(args.resume_full):
        raise ValueError("--allow_schedule_branch requires a full-state resume")
    if bool(getattr(args, "allow_data_branch", False)) and not bool(args.resume_full):
        raise ValueError("--allow_data_branch requires a full-state resume")
    if bool(getattr(args, "allow_data_branch", False)) and bool(args.allow_schedule_branch):
        raise ValueError("--allow_data_branch and --allow_schedule_branch cannot be combined")
    if bool(args.resume_full) and not args.resume_path:
        raise ValueError("--resume_full requires --resume_path")


def resolve_validation_sample_count(dataset_size: int, requested: int) -> int:
    """Resolve a fixed validation-block budget; zero means the full dataset."""
    dataset_size = int(dataset_size)
    requested = int(requested)
    if dataset_size < 0 or requested < 0:
        raise ValueError("validation dataset size and requested count must be non-negative")
    return dataset_size if requested == 0 else min(dataset_size, requested)


def build_run_contract(
    args: argparse.Namespace,
    model_config: dict,
    tokenizer_sha256: str,
    parameter_count: dict | None = None,
    run_plan_binding: dict | None = None,
) -> dict:
    """Immutable state required for an exact full-state continuation."""
    return {
        "schema_version": 3,
        "model_config": dict(model_config),
        "parameter_count": dict(parameter_count) if parameter_count is not None else None,
        "optimizer": {
            "name": str(args.optimizer),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "muon_lr": float(args.muon_lr),
            "muon_momentum": float(args.muon_momentum),
            "grad_clip": float(args.grad_clip),
        },
        "schedule": {
            "name": str(args.lr_schedule),
            "warmup_steps": int(args.warmup_steps),
            "schedule_total_steps": int(args.schedule_total_steps),
            "decay_start_step": int(args.decay_start_step),
            "decay_end_step": int(args.decay_end_step),
            "min_lr_ratio": float(args.min_lr_ratio),
        },
        "precision": str(args.precision),
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "compile": bool(args.compile),
        },
        "rng_consumers": {
            "eval_every": int(args.eval_every),
            "add_bos_to_prompts": bool(args.add_bos_to_prompts),
            "sample_temperature": float(args.sample_temperature),
            "sample_top_p": float(args.sample_top_p),
            "sample_top_k": int(args.sample_top_k),
            "sample_max_new_tokens": int(args.sample_max_new_tokens),
            "sample_min_new_tokens": int(args.sample_min_new_tokens),
        },
        "checkpointing": {
            "save_every": int(args.save_every),
            "save_steps": [int(step) for step in args.save_steps],
            "retention_policy": {
                "periodic": "atomic_latest_only",
                "explicit_save_step": "atomic_named_step_then_latest",
                "invocation_final": "atomic_named_step_then_latest",
            },
        },
        "seq_len": int(args.seq_len),
        "micro_bsz": int(args.micro_bsz),
        "grad_accum": int(args.grad_accum),
        "bos_id": int(args.bos_id),
        "eos_id": int(args.eos_id),
        "mask_bos_in_loss": not bool(args.no_mask_bos_in_loss),
        "mask_last_label_in_loss": bool(args.mask_last_label_in_loss),
        "mask_repeated_eos_in_loss": True,
        "eos_weight": float(args.eos_weight),
        "eos_weight_warmup_steps": int(args.eos_weight_warmup_steps),
        "sampling_mode": "deterministic",
        "model_seed": int(getattr(args, "seed", 1234)),
        "sampler_seed": int(getattr(args, "sampler_seed", 1234)),
        "val_seed": int(getattr(args, "val_seed", 1234)),
        "validation_selection": {
            "combined_samples": int(getattr(args, "val_samples", 200)),
            "samples_per_source": int(getattr(args, "val_samples_per_source", 80)),
        },
        "tokenizer_sha256": str(tokenizer_sha256),
        "run_plan": dict(run_plan_binding) if run_plan_binding is not None else None,
    }


def validate_resume_contract(
    checkpoint: dict,
    current: dict,
    *,
    strict: bool,
    checkpoint_step: int,
    allow_schedule_branch: bool,
    allow_data_branch: bool = False,
) -> None:
    saved = checkpoint.get("run_contract")
    if allow_data_branch and allow_schedule_branch:
        raise RuntimeError("[resume] data and schedule branches cannot be combined in one handoff")
    if saved is None:
        msg = "checkpoint has no run_contract (legacy checkpoint)"
        if strict:
            raise RuntimeError(
                f"[resume] {msg}; use --no-strict_resume_contract only for an intentional migration"
            )
        warnings.warn(
            f"[resume] {msg}; exact continuation is not verified", RuntimeWarning, stacklevel=2
        )
        return

    mismatches = []
    for key, expected in current.items():
        actual = saved.get(key)
        if key == "run_plan" and actual != expected:
            try:
                validate_run_plan_resume_transition(
                    actual,
                    expected,
                    checkpoint_step=int(checkpoint_step),
                    allow_data_branch=bool(allow_data_branch),
                )
            except RuntimeError as exc:
                mismatches.append(f"run_plan: {exc}")
            continue
        if key == "schedule" and actual != expected and allow_schedule_branch:
            saved_base_lr = float((saved.get("optimizer") or {}).get("lr"))
            current_base_lr = float((current.get("optimizer") or {}).get("lr"))
            if not math.isclose(saved_base_lr, current_base_lr, rel_tol=0.0, abs_tol=0.0):
                raise RuntimeError("[resume] schedule branch cannot change the optimizer base LR")
            validate_schedule_branch(
                actual or {},
                expected,
                checkpoint_step=int(checkpoint_step),
                base_lr=current_base_lr,
            )
            print("[resume] accepted scoped schedule branch: all pre-checkpoint LR values match")
            continue
        if actual != expected:
            mismatches.append(f"{key}: checkpoint={actual!r}, current={expected!r}")
    if mismatches:
        detail = "\n  - ".join(mismatches)
        if strict:
            raise RuntimeError(f"[resume] run-contract mismatch:\n  - {detail}")
        warnings.warn(
            f"[resume] run-contract mismatch (legacy override enabled):\n  - {detail}",
            RuntimeWarning,
            stacklevel=2,
        )


def empty_position_stats() -> dict[str, int]:
    return {
        "serialized_positions": 0,
        "supervised_positions": 0,
        "masked_positions": 0,
        "masked_bos_positions": 0,
        "masked_repeated_eos_positions": 0,
        "masked_final_label_positions": 0,
        "masked_unattributed_positions": 0,
    }


def update_position_stats(
    totals: dict[str, int],
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    bos_id: int,
    eos_id: int,
    mask_bos: bool,
    mask_repeated_eos: bool,
    mask_final: bool,
) -> None:
    """Accumulate exact positions and disjoint configured mask causes on CPU."""
    y = labels.long()
    m = loss_mask.float()
    total = int(m.numel())
    supervised = int((m > 0).sum().item())

    active = torch.ones_like(y, dtype=torch.bool)
    bos_count = 0
    repeated_count = 0
    final_count = 0
    if mask_bos:
        cause = active & (y == int(bos_id))
        bos_count = int(cause.sum().item())
        active &= ~cause
    if mask_repeated_eos and y.shape[-1] > 1:
        repeated = torch.zeros_like(active)
        repeated[..., 1:] = (y[..., 1:] == int(eos_id)) & (y[..., :-1] == int(eos_id))
        cause = active & repeated
        repeated_count = int(cause.sum().item())
        active &= ~cause
    if mask_final and y.shape[-1] > 0:
        final = torch.zeros_like(active)
        final[..., -1] = True
        cause = active & final
        final_count = int(cause.sum().item())

    attributed = bos_count + repeated_count + final_count
    masked = total - supervised
    totals["serialized_positions"] += total
    totals["supervised_positions"] += supervised
    totals["masked_positions"] += masked
    totals["masked_bos_positions"] += bos_count
    totals["masked_repeated_eos_positions"] += repeated_count
    totals["masked_final_label_positions"] += final_count
    totals["masked_unattributed_positions"] += max(0, masked - attributed)


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


@torch.no_grad()
def masked_ce_128_debug(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tt: int = 128,
) -> float:
    """Same as training loss (mask-weighted), but only first tt tokens."""
    B, T, V = logits.shape
    t = min(T, int(tt))
    flat_logits = logits[:, :t, :].reshape(-1, V).float()
    y = labels[:, :t].reshape(-1)
    m = loss_mask[:, :t].reshape(-1).float()
    per = F.cross_entropy(flat_logits, y, reduction="none")
    return float((per * m).sum().item() / m.sum().clamp_min(1.0).item())


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def capture_rng_state() -> dict:
    """Capture every RNG stream that can affect an exact full-state resume."""
    import random

    state: dict = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np
    except Exception:  # pragma: no cover - NumPy is an optional runtime dependency
        state["numpy"] = None
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: dict | None) -> None:
    """Restore all RNG streams; CUDA full resumes fail if CUDA state is absent."""
    import random

    if not isinstance(state, dict):
        raise RuntimeError("[resume] --resume_full requires rng_state in checkpoint")
    if state.get("python") is None or state.get("torch_cpu") is None:
        raise RuntimeError("[resume] checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "[resume] checkpoint has NumPy RNG state but NumPy is unavailable"
            ) from e
        np.random.set_state(numpy_state)

    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise RuntimeError("[resume] CUDA full resume requires saved CUDA RNG states")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "[resume] CUDA RNG device-count mismatch: "
                f"checkpoint={len(cuda_state)}, current={torch.cuda.device_count()}"
            )
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as e:
            raise RuntimeError(f"[resume] CUDA RNG state is incompatible: {e}") from e


def _atomic_torch_save(obj: dict, final_path: Path) -> None:
    """
    Atomically save a torch checkpoint:
      1) write to final_path.with_suffix(final_path.suffix + ".tmp")
      2) flush + fsync
      3) os.replace(tmp, final)
    This prevents half-written latest.pt when the filesystem is flaky.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    # best effort: remove stale tmp
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    # Use a real file handle so we can fsync.
    # torch.save can accept a file-like object.
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace on POSIX
    os.replace(tmp_path, final_path)


def save_ckpt(
    out_dir: Path,
    global_step: int,
    local_step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    model_config: dict,
    train_args: dict,
    run_contract: dict,
    position_stats: dict[str, int],
    sampler_state: dict,
    data_contract: dict,
    retain_step: bool = True,
) -> None:
    """Atomically update latest and optionally retain a named full-state checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": model_to_save.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "local_step": int(local_step),
        "config": model_config,
        "train_args": train_args,
        "run_contract": run_contract,
        "data_contract": data_contract,
        "rng_state": capture_rng_state(),
        "position_stats": {k: int(v) for k, v in position_stats.items()},
        "data_sampler": sampler_state,
        "checkpoint_retention": {"retain_step": bool(retain_step)},
        "saved_at_unix": int(time.time()),
    }

    latest_path = out_dir / "latest.pt"
    step_path = out_dir / f"step_{global_step:06d}.pt"
    write_paths = [step_path, latest_path] if retain_step else [latest_path]
    try:
        for path in write_paths:
            _atomic_torch_save(ckpt, path)
    except Exception as e:
        for path in write_paths:
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        raise RuntimeError(f"[ckpt] save failed: {e}") from e


def load_ckpt(
    resume_path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    resume_full: bool,
    current_run_contract: dict,
    strict_resume_contract: bool,
    allow_schedule_branch: bool,
    allow_data_branch: bool = False,
) -> tuple[int, int, dict[str, int], dict]:
    ckpt = torch.load(resume_path, map_location="cpu")
    global_step = int(ckpt.get("global_step", 0))
    if resume_full:
        saved_optimizer = ((ckpt.get("run_contract") or {}).get("optimizer") or {}).get("name")
        current_optimizer = (current_run_contract.get("optimizer") or {}).get("name")
        if saved_optimizer is not None and saved_optimizer != current_optimizer:
            raise RuntimeError(
                "[resume] --resume_full cannot migrate optimizer implementations: "
                f"checkpoint={saved_optimizer!r}, current={current_optimizer!r}"
            )
    validate_resume_contract(
        ckpt,
        current_run_contract,
        strict=bool(strict_resume_contract),
        checkpoint_step=global_step,
        allow_data_branch=bool(allow_data_branch),
        allow_schedule_branch=bool(allow_schedule_branch),
    )
    if "model" not in ckpt:
        raise RuntimeError(f"[resume] checkpoint has no model state: {resume_path}")
    state = ckpt["model"]

    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod.") :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    local_step = int(ckpt.get("local_step", 0))
    if resume_full:
        if ckpt.get("optim") is None:
            raise RuntimeError("[resume] --resume_full requires optimizer state in checkpoint")
        try:
            optim.load_state_dict(ckpt["optim"])
        except Exception as e:
            raise RuntimeError(f"[resume] optimizer state is incompatible: {e}") from e

        if scaler is not None:
            if ckpt.get("scaler") is None:
                raise RuntimeError("[resume] fp16 --resume_full requires scaler state")
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                raise RuntimeError(f"[resume] scaler state is incompatible: {e}") from e

    if resume_full:
        restore_rng_state(ckpt.get("rng_state"))

    position_stats = empty_position_stats()
    resume_metadata: dict = {}
    saved_stats = ckpt.get("position_stats", {})
    if isinstance(saved_stats, dict):
        for key in position_stats:
            if key in saved_stats:
                position_stats[key] = int(saved_stats[key])
    resume_metadata = {
        "data_sampler": ckpt.get("data_sampler"),
        "data_contract": ckpt.get("data_contract"),
        "train_args": ckpt.get("train_args"),
        "rng_state": ckpt.get("rng_state"),
    }
    return global_step, local_step, position_stats, resume_metadata


# -----------------------------------------------------------------------------
# Debug sanity checks (causality + label shift)
# -----------------------------------------------------------------------------
@torch.no_grad()
def causal_leak_check(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    *,
    vocab_size: int,
    check_pos: int = 128,
    delta_pos: int = 8,
) -> float:
    """
    Perturb ONE token at position (check_pos + delta_pos) and measure how much
    the logits on prefix [0:check_pos] change. For a strictly causal model,
    this should be ~0.
    """
    model.eval()
    x = input_ids.to(device, non_blocking=True)
    logits1 = model(x).float()

    x2 = x.clone()
    p = min(x2.shape[1] - 1, int(check_pos + delta_pos))
    x2[:, p] = (x2[:, p] + 123) % int(vocab_size)
    logits2 = model(x2).float()

    diff = (logits1[:, :check_pos, :] - logits2[:, :check_pos, :]).abs().max().item()
    print(f"[dbg] local_future_leak_check max_abs_diff={diff:.6f} (expect ~0)")

    model.train()
    return float(diff)


@torch.no_grad()
def label_shift_sanity(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    """
    Check labels are next-token targets:
        labels[t] == input_ids[t+1]  (for supervised positions).
    """
    m = loss_mask[:, :-1] > 0
    if m.sum().item() == 0:
        acc = 0.0
    else:
        ok = ((labels[:, :-1] == input_ids[:, 1:]) & m).float().sum().item()
        tot = m.float().sum().item()
        acc = ok / max(1.0, tot)
    print(f"[dbg] label_shift_sanity next-token match over supervised: {acc:.6f}")
    return float(acc)


# -----------------------------------------------------------------------------
# Eval + dataset stats
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    precision: str,
    *,
    eos_id: int,
    eos_weight: float,
    max_batches: int | None = None,
) -> float:
    """Evaluate exact global target-weighted CE over the selected blocks."""
    was_training = model.training
    model.eval()
    autocast_dtype = _autocast_dtype(precision)
    loss_numerator = 0.0
    target_weight = 0.0

    try:
        for batch_index, batch in enumerate(dl):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            if len(batch) == 2:
                input_u16, labels_u16 = batch
                loss_mask = torch.ones_like(labels_u16, dtype=torch.float32)
            else:
                input_u16, labels_u16, loss_mask = batch

            input_ids = input_u16.to(device, dtype=torch.long, non_blocking=True)
            labels = labels_u16.to(device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask.to(device, dtype=torch.float32, non_blocking=True)

            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    logits = model(input_ids)
                    numerator, weight = masked_weighted_ce_components(
                        logits,
                        labels,
                        loss_mask,
                        eos_id=eos_id,
                        eos_weight=eos_weight,
                    )
            else:
                logits = model(input_ids)
                numerator, weight = masked_weighted_ce_components(
                    logits,
                    labels,
                    loss_mask,
                    eos_id=eos_id,
                    eos_weight=eos_weight,
                )

            loss_numerator += float(numerator.detach().double().item())
            target_weight += float(weight.detach().double().item())
    finally:
        model.train(was_training)

    return float(loss_numerator / max(1.0, target_weight))


def _estimate_dataset_stats(ds: PackedBinDataset, seq_len: int, *, name: str) -> None:
    """Print exact virtual-stream exposure statistics plus sampled mask density."""
    stats = ds.stats()
    print(f"[*] {name} dataset stats:")
    print(f"    - sampling mode: {stats['sampling_mode']}")
    print(f"    - shards: {stats['n_shards']}")
    print(f"    - serialized tokens in .bin: {stats['total_raw_tokens']:,}")
    print(f"    - possible next-token transitions: {stats['total_transitions']:,}")
    print(f"    - window size (T+1): {stats['window_size']:,}")
    print(f"    - block stride (T): {stats['block_stride']:,}")
    print(f"    - deterministic blocks: {stats['n_blocks']:,}")
    print(f"    - usable transitions per full traversal: {stats['usable_transitions']:,}")
    print(f"    - dropped tail transitions: {stats['tail_transitions']:,}")
    print(f"    - covered cross-shard transitions: {stats['covered_cross_shard_transitions']:,}")

    # sample a few batches to estimate mask density and EOS fraction
    try:
        # sample 32 blocks from dataset (deterministic indices)
        n_samp = min(32, int(len(ds)))
        mask_sum = 0.0
        eos_sum = 0.0
        tot = 0.0
        for i in range(n_samp):
            batch = ds[i]
            if len(batch) == 2:
                _, labels_u16 = batch
                loss_mask = torch.ones_like(labels_u16, dtype=torch.float32)
            else:
                _, labels_u16, loss_mask = batch

            m = loss_mask.float()
            y = labels_u16.long()

            mask_sum += float(m.sum().item())
            tot += float(m.numel())
            # try to get eos_id from dataset if present
            eos_id = int(getattr(ds, "eos_id", 3))
            eos_sum += float(((y == eos_id) & (m > 0)).float().sum().item())

        avg_sup = mask_sum / max(1.0, float(n_samp))
        eos_frac = eos_sum / max(1.0, mask_sum)
        print(f"    - avg supervised tokens per block (sampled): {avg_sup:.1f} / {seq_len}")
        print(f"    - avg EOS fraction over supervised tokens (sampled): {eos_frac:.4f}")
    except Exception as e:
        print(f"    - (could not sample mask/EOS stats: {e})")


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)

    # Output
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    # Model
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--d_model", type=int, default=576)
    ap.add_argument("--n_heads", type=int, default=9)
    ap.add_argument("--n_kv_heads", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Special tokens
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    # Loss shaping
    ap.add_argument("--no_mask_bos_in_loss", action="store_true")
    ap.add_argument(
        "--mask_last_label_in_loss",
        action="store_true",
        help="Legacy opt-in only. Canonical training supervises every valid final label.",
    )
    ap.add_argument("--no_mask_last_label_in_loss", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--eos_weight", type=float, default=1.0)
    ap.add_argument("--eos_weight_warmup_steps", type=int, default=0)
    # Train
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument(
        "--optimizer",
        choices=["muon", "adamw"],
        default="muon",
        help="muon: Muon on hidden matrices + AdamW on embeddings/norms (default). adamw: AdamW everywhere.",
    )
    ap.add_argument(
        "--muon_lr",
        type=float,
        default=0.0,
        help="LR for Muon matrix groups (<=0: reuse --lr; Muon update RMS is matched to AdamW's).",
    )
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument(
        "--lr_schedule",
        choices=["cosine", "constant", "wsd"],
        default="cosine",
        help="Absolute-step schedule: warmup+cosine, warmup+constant, or warmup/stable/decay.",
    )
    ap.add_argument(
        "--schedule_total_steps",
        type=int,
        default=0,
        help="LR schedule horizon in global steps (0: --max_steps). Independent of stage stop.",
    )
    ap.add_argument("--decay_start_step", type=int, default=-1, help="WSD cosine-decay start.")
    ap.add_argument("--decay_end_step", type=int, default=-1, help="WSD cosine-decay end.")
    ap.add_argument("--min_lr_ratio", type=float, default=0.1, help="Decay floor / peak LR.")
    ap.add_argument(
        "--max_steps",
        type=int,
        default=100000,
        help="Absolute global optimizer step at which this invocation stops.",
    )
    ap.add_argument(
        "--data_stage_start_step",
        type=int,
        default=0,
        help="Absolute global step where the current --train_dir stage begins.",
    )
    ap.add_argument(
        "--run_plan_json",
        default="",
        help=(
            "Frozen schema-v3 plan_pretrain_run.py output. Required by the strict "
            "production contract and hashed into checkpoints/run metadata."
        ),
    )
    ap.add_argument(
        "--run_plan_stage",
        choices=["stage_a", "stage_b"],
        default=None,
        help="Stage entry in --run_plan_json bound to this trainer invocation.",
    )
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Model initialization and training-RNG seed; does not select data blocks.",
    )
    ap.add_argument(
        "--sampler_seed",
        type=int,
        default=1234,
        help="Frozen training permutation seed, independent of --seed.",
    )
    ap.add_argument(
        "--val_seed",
        type=int,
        default=1234,
        help="Frozen validation-subset seed, independent of --seed and batch size.",
    )
    ap.add_argument(
        "--val_samples",
        type=int,
        default=200,
        help="Combined-validation blocks per evaluation; 0 evaluates the full stream.",
    )
    ap.add_argument(
        "--val_samples_per_source",
        type=int,
        default=80,
        help="Blocks per source validation; 0 evaluates each full source stream.",
    )

    # Logging / eval / save
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Periodically refresh atomic latest.pt only; named steps are explicit/final.",
    )
    ap.add_argument(
        "--save_steps",
        action="append",
        default=[],
        metavar="STEP[,STEP...]",
        help=(
            "Repeatable absolute optimizer-step milestones. Comma-separated values are "
            "accepted; the complete list must be strictly increasing, unique, and no "
            "greater than --schedule_total_steps. Milestones outside this invocation's "
            "stage range are intentionally ignored."
        ),
    )
    ap.add_argument("--debug_every", type=int, default=500)

    # Optional: periodic instruction-style bench eval (runs pretrain/eval_bench_v5.py)
    ap.add_argument(
        "--bench_eval_path",
        default="",
        help="Path to bench jsonl for periodic eval (disabled if empty).",
    )
    ap.add_argument(
        "--bench_eval_every",
        type=int,
        default=0,
        help="Run bench eval every N steps (0 disables). Prefer a multiple of --save_every.",
    )
    ap.add_argument(
        "--bench_eval_script",
        default="pretrain/eval_bench_v5.py",
        help="Path to eval script (default: pretrain/eval_bench_v5.py).",
    )
    ap.add_argument(
        "--bench_eval_out_dir",
        default="",
        help="Where to write bench results (default: <out_dir>/bench_eval).",
    )
    ap.add_argument("--bench_eval_max_seq_len", type=int, default=1024)
    ap.add_argument("--bench_eval_max_new_tokens", type=int, default=192)
    ap.add_argument("--bench_eval_min_new_tokens", type=int, default=1)
    ap.add_argument("--bench_eval_ban_first_steps", type=int, default=4)

    # Sampling during training
    ap.add_argument(
        "--add_bos_to_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Canonical default ON; periodic base-model prompts begin with BOS.",
    )
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens", type=int, default=256)
    ap.add_argument("--sample_min_new_tokens", type=int, default=0)

    # Resume
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_full", action="store_true")
    ap.add_argument(
        "--allow_weights_only_resume",
        action="store_true",
        help="Explicit legacy/migration override; forbidden for nonzero data-stage starts.",
    )
    ap.add_argument(
        "--allow_data_branch",
        action="store_true",
        help=(
            "At the exact Stage-A endpoint, allow a separately frozen control plan "
            "whose only differences are the validated Stage-B data cohort/release."
        ),
    )
    ap.add_argument(
        "--allow_schedule_branch",
        action="store_true",
        help="Allow only a future LR branch whose pre-checkpoint LR history is identical.",
    )
    ap.add_argument(
        "--resume_step",
        type=int,
        default=-1,
        help="For a checkpoint directory, require step_XXXXXX.pt; -1 selects latest.",
    )
    ap.set_defaults(strict_resume_contract=True)
    ap.add_argument("--strict_resume_contract", dest="strict_resume_contract", action="store_true")
    ap.add_argument(
        "--no_strict_resume_contract",
        "--no-strict_resume_contract",
        dest="strict_resume_contract",
        action="store_false",
        help="Allow an intentional legacy/inexact resume after warning.",
    )
    # torch.compile
    ap.add_argument("--compile", action="store_true")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    validate_training_args(args)
    set_seed(args.seed)

    train_dir = Path(_resolve_path(args.train_dir))
    val_dir = Path(_resolve_path(args.val_dir))
    out_dir = Path(_resolve_path(args.out_dir))
    samples_dir = Path(_resolve_path(args.samples_dir))
    tok_path = Path(_resolve_path(args.tokenizer_path))

    resolved: Path | None = None
    if args.resume_path:
        resolved = _resolve_resume_path(
            args.resume_path,
            out_dir=out_dir,
            resume_step=int(args.resume_step),
        )
        if resolved is None:
            raise FileNotFoundError(
                f"[resume] could not resolve resume_path={args.resume_path!r} "
                f"(resume_step={args.resume_step}); refusing to start from scratch"
            )

    assert_tokenizer_contract(tok_path)
    tokenizer_sha256 = _sha256_file(tok_path)
    run_plan_binding = load_run_plan_binding(
        args,
        train_dir=train_dir,
        val_dir=val_dir,
        tokenizer_sha256=tokenizer_sha256,
    )
    if run_plan_binding is None:
        warnings.warn(
            "training is not bound to a frozen run plan because strict resume contract "
            "validation was explicitly disabled",
            RuntimeWarning,
            stacklevel=2,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects a CUDA GPU")

    tracker = Tracker(out_dir)

    cfg = GPTConfig(
        vocab_size=CANONICAL_VOCAB_SIZE,
        n_layers=int(args.layers),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_kv_heads=int(args.n_kv_heads),
        d_ff=int(args.d_ff),
        max_seq_len=int(args.seq_len),
        dropout=float(args.dropout),
        tie_embeddings=True,
    )
    model = GPT(cfg)
    parameter_count = audit_gpt_parameter_count(model, cfg)
    run_contract = build_run_contract(
        args,
        asdict(cfg),
        tokenizer_sha256,
        parameter_count=parameter_count,
        run_plan_binding=run_plan_binding,
    )
    model = model.to(device)

    use_fp16 = args.precision == "fp16"
    ac_dtype = _autocast_dtype(args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    optim = build_optimizer(
        model,
        name=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
        muon_lr=float(args.muon_lr),
        muon_momentum=float(args.muon_momentum),
    )

    common_dataset_args = {
        "seq_len": int(args.seq_len),
        "bos_id": int(args.bos_id),
        "eos_id": int(args.eos_id),
        "mask_bos_in_loss": not bool(args.no_mask_bos_in_loss),
        "mask_last_label_in_loss": bool(args.mask_last_label_in_loss),
        "require_release_manifest": True,
    }
    train_ds = PackedBinDataset(
        str(train_dir),
        sampling_mode="deterministic",
        **common_dataset_args,
    )
    if run_plan_binding is not None:
        validate_run_plan_dataset(run_plan_binding, train_ds)
    val_ds = PackedBinDataset(str(val_dir), sampling_mode="deterministic", **common_dataset_args)
    data_contract = build_data_contract(train_dir, train_ds, args)
    if run_plan_binding is not None:
        validate_run_plan_validation_dataset(run_plan_binding, val_ds)
    _estimate_dataset_stats(train_ds, int(args.seq_len), name="train")
    _estimate_dataset_stats(val_ds, int(args.seq_len), name="val")
    print(
        f"[*] model params: {int(parameter_count['actual_total']):,} "
        f"(audit={parameter_count['status']}, "
        f"canonical={parameter_count['canonical_parameterization']})"
    )

    requested_val_samples = int(args.val_samples)
    val_samples = resolve_validation_sample_count(len(val_ds), requested_val_samples)
    val_sampler = FixedSubsetSampler(
        val_ds,
        num_samples=val_samples,
        seed=int(args.val_seed),
    )
    print(
        f"[val] frozen combined subset: {val_samples:,}/{len(val_ds):,} blocks "
        f"(seed={args.val_seed})"
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(args.micro_bsz),
        sampler=val_sampler,
        shuffle=False,
        generator=torch.Generator(device="cpu").manual_seed(int(args.val_seed) + 17),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
    )

    domain_val: list[tuple[str, DataLoader]] = []
    requested_val_samples_per_source = int(args.val_samples_per_source)
    val_by_src_root = val_dir.parent / "val_by_source"
    if val_by_src_root.is_dir():
        for sub in sorted(p for p in val_by_src_root.iterdir() if p.is_dir()):
            try:
                dom_ds = PackedBinDataset(
                    str(sub), sampling_mode="deterministic", **common_dataset_args
                )
            except (RuntimeError, FileNotFoundError) as e:
                print(f"[val_by_source] skip {sub.name}: {e}")
                continue
            dom_sampler = FixedSubsetSampler(
                dom_ds,
                num_samples=resolve_validation_sample_count(
                    len(dom_ds), requested_val_samples_per_source
                ),
                seed=int(args.val_seed) + 1,
            )
            domain_val.append((
                sub.name,
                DataLoader(
                    dom_ds,
                    batch_size=int(args.micro_bsz),
                    sampler=dom_sampler,
                    shuffle=False,
                    generator=torch.Generator(device="cpu").manual_seed(int(args.val_seed) + 19),
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                ),
            ))
        if domain_val:
            print(f"[val_by_source] domain val loss tracked for: {[n for n, _ in domain_val]}")

    global_step = 0

    position_stats = empty_position_stats()
    resume_metadata: dict = {}
    if resolved is not None:
        global_step, _, position_stats, resume_metadata = load_ckpt(
            resume_path=resolved,
            model=model,
            optim=optim,
            scaler=scaler if use_fp16 else None,
            resume_full=bool(args.resume_full),
            current_run_contract=run_contract,
            strict_resume_contract=bool(args.strict_resume_contract),
            allow_schedule_branch=bool(args.allow_schedule_branch),
            allow_data_branch=bool(args.allow_data_branch),
        )
        synchronize_validated_run_plan_binding(run_contract, run_plan_binding)
        print(f"[resume] loaded {resolved} (global_step={global_step})")

    if global_step < int(args.data_stage_start_step):
        raise RuntimeError(
            f"checkpoint global_step={global_step} precedes --data_stage_start_step="
            f"{args.data_stage_start_step}"
        )
    if global_step > int(args.max_steps):
        raise RuntimeError(
            f"checkpoint global_step={global_step} is beyond absolute stop --max_steps={args.max_steps}"
        )
    if resolved is None and int(args.data_stage_start_step) != 0:
        raise RuntimeError("a nonzero --data_stage_start_step requires a resume checkpoint")

    local_step = global_step - int(args.data_stage_start_step)
    samples_per_step = int(args.micro_bsz) * int(args.grad_accum)
    stage_sample_position = local_step * samples_per_step
    step_derived_stage_samples = (
        int(args.max_steps) - int(args.data_stage_start_step)
    ) * samples_per_step
    planned_stage_samples, remaining_samples = resolve_run_plan_sample_budget(
        run_plan_binding,
        stage_sample_position=stage_sample_position,
        step_derived_stage_samples=step_derived_stage_samples,
    )
    train_sampler = ResumablePermutationSampler(
        train_ds,
        seed=int(args.sampler_seed),
        start_position=stage_sample_position,
        num_samples=remaining_samples,
    )
    if train_sampler.end_position != planned_stage_samples:
        raise RuntimeError("deterministic sampler end is not bound to the frozen run plan")
    if resolved is not None:
        validate_data_resume_state(
            saved_data_contract=resume_metadata.get("data_contract"),
            current_data_contract=data_contract,
            saved_sampler_state=resume_metadata.get("data_sampler"),
            current_sampler=train_sampler,
            global_step=global_step,
            data_stage_start_step=int(args.data_stage_start_step),
            strict=bool(args.strict_resume_contract),
        )
    train_dl = DataLoader(
        train_ds,
        batch_size=int(args.micro_bsz),
        sampler=train_sampler,
        shuffle=False,
        generator=torch.Generator(device="cpu").manual_seed(int(args.sampler_seed) + 17),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    print(
        f"[*] deterministic sampler: stage_position={stage_sample_position:,} "
        f"remaining_samples={remaining_samples:,} epoch={train_sampler.epoch} "
        f"epoch_offset={train_sampler.epoch_offset:,}"
    )

    tracker.log_run_start(
        {
            **vars(args),
            "model_cfg": asdict(cfg),
            "parameter_count": parameter_count,
            "run_plan_binding": run_plan_binding,
        },
        str(tok_path),
    )

    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] torch.compile failed: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        **vars(args),
        "model_cfg": asdict(cfg),
        "run_contract": run_contract,
        "data_contract": data_contract,
        "train_dataset": train_ds.stats(),
        "val_dataset": val_ds.stats(),
        "samples_per_optimizer_step": samples_per_step,
        "stage_sample_position_at_start": stage_sample_position,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )

    best_val_path = out_dir / "best_val.json"
    best_val = float("inf")
    best_step = -1
    if best_val_path.is_file():
        try:
            saved_best = json.loads(best_val_path.read_text(encoding="utf-8"))
            best_val = float(saved_best["best_val_loss"])
            best_step = int(saved_best["best_step"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            warnings.warn(f"could not restore best_val.json: {e}", RuntimeWarning, stacklevel=2)

    if resolved is not None and bool(args.resume_full):
        # Reset once more after setup/compile so setup cannot perturb exact continuation.
        restore_rng_state(resume_metadata.get("rng_state"))

    model.train()
    data_iter = iter(train_dl)
    t_window = time.time()
    window_serialized_positions = 0
    window_supervised_positions = 0
    last_checkpoint_step: int | None = None
    last_checkpoint_retained = False

    while global_step < int(args.max_steps):
        lr = lr_schedule(
            global_step,
            int(args.warmup_steps),
            float(args.lr),
            schedule=str(args.lr_schedule),
            schedule_total_steps=int(args.schedule_total_steps),
            decay_start_step=int(args.decay_start_step),
            decay_end_step=int(args.decay_end_step),
            min_lr_ratio=float(args.min_lr_ratio),
        )
        for pg in optim.param_groups:
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)
        optim.zero_grad(set_to_none=True)

        cur_eos_weight = float(args.eos_weight)
        if int(args.eos_weight_warmup_steps) > 0 and global_step >= int(
            args.eos_weight_warmup_steps
        ):
            cur_eos_weight = 1.0
        accum_loss_raw = 0.0

        for micro in range(int(args.grad_accum)):
            try:
                batch = next(data_iter)
            except StopIteration as e:
                raise RuntimeError(
                    "deterministic training sampler exhausted before --max_steps; "
                    "the sample-position contract is inconsistent"
                ) from e

            if len(batch) == 2:
                input_cpu, labels_cpu = batch
                loss_mask_cpu = torch.ones_like(labels_cpu, dtype=torch.float32)
            else:
                input_cpu, labels_cpu, loss_mask_cpu = batch

            update_position_stats(
                position_stats,
                labels_cpu,
                loss_mask_cpu,
                bos_id=int(args.bos_id),
                eos_id=int(args.eos_id),
                mask_bos=not bool(args.no_mask_bos_in_loss),
                mask_repeated_eos=bool(getattr(train_ds, "mask_repeated_eos_in_loss", False)),
                mask_final=bool(args.mask_last_label_in_loss),
            )
            window_serialized_positions += int(labels_cpu.numel())
            window_supervised_positions += int((loss_mask_cpu > 0).sum().item())

            input_ids = input_cpu.to(device, dtype=torch.long, non_blocking=True)
            labels = labels_cpu.to(device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask_cpu.to(device, dtype=torch.float32, non_blocking=True)

            if ac_dtype is not None:
                with torch.autocast("cuda", dtype=ac_dtype):
                    logits = model(input_ids)
                    loss_raw = masked_weighted_ce_loss(
                        logits,
                        labels,
                        loss_mask,
                        eos_id=int(args.eos_id),
                        eos_weight=float(cur_eos_weight),
                    )
                    loss = loss_raw / float(args.grad_accum)
            else:
                logits = model(input_ids)
                loss_raw = masked_weighted_ce_loss(
                    logits,
                    labels,
                    loss_mask,
                    eos_id=int(args.eos_id),
                    eos_weight=float(cur_eos_weight),
                )
                loss = loss_raw / float(args.grad_accum)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss_raw += float(loss_raw.detach().item())

            if global_step % int(args.debug_every) == 0 and micro == 0:
                lm = float(logits.float().mean().item())
                ls = float(logits.float().std().item())
                mce = masked_ce_128_debug(logits, labels, loss_mask, tt=128)
                m = loss_mask
                eos_frac = float(
                    (((labels == int(args.eos_id)) & (m > 0)).float().sum().item())
                    / m.sum().clamp_min(1.0).item()
                )
                pred = logits.argmax(dim=-1)
                hit = ((pred == labels) & (m > 0)).sum().item()
                tot = (m > 0).sum().item()
                top1 = float(hit / max(1.0, tot))
                print(
                    f"[dbg] step={global_step} logits_mean={lm:.4f} logits_std={ls:.4f} masked_ce_128={mce:.6f}"
                )
                print(
                    f"[dbg] mask_mean={float(m.mean().item()):.6f} mask_sum={float(m.sum().item()):.1f}"
                )
                print(
                    f"[dbg] labels min/max: {int(labels.min().item())} {int(labels.max().item())}"
                )
                print(f"[dbg] eos_frac_supervised: {eos_frac:.6f}")
                print(f"[dbg] masked_top1_acc: {top1:.6f}")
                ce_nomask = F.cross_entropy(
                    logits[:, :128, :].reshape(-1, logits.shape[-1]).float(),
                    labels[:, :128].reshape(-1),
                    reduction="mean",
                )
                print(f"[dbg] ce_nomask_128: {float(ce_nomask.item()):.6f}")
                causal_leak_check(
                    model, input_ids, device, vocab_size=cfg.vocab_size, check_pos=128, delta_pos=8
                )
                label_shift_sanity(input_ids, labels, loss_mask)

        if float(args.grad_clip) > 0:
            if use_fp16:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        if use_fp16:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        global_step += 1
        local_step = global_step - int(args.data_stage_start_step)
        train_sampler.commit(samples_per_step)

        if global_step % int(args.log_every) == 0:
            dt = time.time() - t_window
            serialized_pos_s = window_serialized_positions / max(dt, 1e-6)
            supervised_pos_s = window_supervised_positions / max(dt, 1e-6)
            mean_loss_raw = accum_loss_raw / max(1, int(args.grad_accum))
            print(
                f"[train] step={global_step} loss={mean_loss_raw:.4f} "
                f"(eos_w={cur_eos_weight:g}) lr={lr:.2e} "
                f"serialized_pos/s={serialized_pos_s:.0f} supervised_pos/s={supervised_pos_s:.0f}"
            )
            tracker.log(
                "train",
                global_step,
                loss=float(mean_loss_raw),
                lr=float(lr),
                serialized_positions_per_second=float(serialized_pos_s),
                supervised_positions_per_second=float(supervised_pos_s),
                cumulative_serialized_positions=int(position_stats["serialized_positions"]),
                cumulative_supervised_positions=int(position_stats["supervised_positions"]),
                cumulative_masked_positions=int(position_stats["masked_positions"]),
                cumulative_masked_bos_positions=int(position_stats["masked_bos_positions"]),
                cumulative_masked_repeated_eos_positions=int(
                    position_stats["masked_repeated_eos_positions"]
                ),
                cumulative_masked_final_label_positions=int(
                    position_stats["masked_final_label_positions"]
                ),
                cumulative_masked_unattributed_positions=int(
                    position_stats["masked_unattributed_positions"]
                ),
                data_sampler_committed_position=int(train_sampler.position),
                eos_weight=float(cur_eos_weight),
            )
            t_window = time.time()
            window_serialized_positions = 0
            window_supervised_positions = 0

        if global_step % int(args.eval_every) == 0:
            val_loss = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                precision=args.precision,
                eos_id=int(args.eos_id),
                eos_weight=1.0,
            )
            print(f"[eval] step={global_step} val_loss={val_loss:.4f}")
            val_ppl = maybe_ppl(val_loss, 1.0)
            if val_loss < best_val:
                best_val = float(val_loss)
                best_step = int(global_step)
                best_val_path.write_text(
                    json.dumps(
                        {"best_step": best_step, "best_val_loss": best_val, "time": time.time()},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            dom_metrics: dict[str, float] = {}
            for dom_name, dom_dl in domain_val:
                dom_loss = evaluate(
                    model=model,
                    dl=dom_dl,
                    device=device,
                    precision=args.precision,
                    eos_id=int(args.eos_id),
                    eos_weight=1.0,
                )
                dom_metrics[f"val_loss_{dom_name}"] = float(dom_loss)
                print(f"[eval] step={global_step} val_loss[{dom_name}]={dom_loss:.4f}")
            tracker.log(
                "val",
                global_step,
                val_loss=float(val_loss),
                val_ppl=float(val_ppl) if val_ppl is not None else None,
                eos_weight=float(cur_eos_weight),
                lr=float(lr),
                **dom_metrics,
            )
            tracker.render()

            samples_dir.mkdir(parents=True, exist_ok=True)
            out_path = samples_dir / f"step_{global_step:06d}.txt"
            try:
                generate_default_samples(
                    model=model,
                    tokenizer_path=str(tok_path),
                    device=device,
                    max_seq_len=int(args.seq_len),
                    precision=args.precision,
                    out_path=out_path,
                    temperature=float(args.sample_temperature),
                    top_p=float(args.sample_top_p),
                    top_k=int(args.sample_top_k),
                    max_new_tokens=int(args.sample_max_new_tokens),
                    min_new_tokens=int(args.sample_min_new_tokens),
                    eos_id=int(args.eos_id),
                    add_bos=bool(args.add_bos_to_prompts),
                    bos_id=int(args.bos_id),
                    greedy=False,
                    debug=True,
                )
                print(f"[sample] wrote {out_path}")
            except Exception as e:
                print(f"[sample] failed: {e}")

        if should_save_checkpoint(
            global_step,
            save_every=int(args.save_every),
            save_steps=args.save_steps,
        ):
            retain_step = should_retain_step_checkpoint(
                global_step,
                save_steps=args.save_steps,
                invocation_final_step=int(args.max_steps),
            )
            save_ckpt(
                out_dir=out_dir,
                global_step=global_step,
                local_step=local_step,
                model=model,
                optim=optim,
                scaler=scaler if use_fp16 else None,
                model_config=asdict(cfg),
                train_args=vars(args),
                run_contract=run_contract,
                position_stats=position_stats,
                sampler_state=train_sampler.state_dict(),
                data_contract=data_contract,
                retain_step=retain_step,
            )
            last_checkpoint_step = global_step
            last_checkpoint_retained = retain_step
            retained = f" + step_{global_step:06d}.pt" if retain_step else ""
            print(f"[ckpt] saved latest.pt{retained} to {out_dir}")
            if (
                args.bench_eval_path
                and int(args.bench_eval_every) > 0
                and global_step % int(args.bench_eval_every) == 0
            ):
                bench_dir = (
                    Path(args.bench_eval_out_dir)
                    if args.bench_eval_out_dir
                    else out_dir / "bench_eval"
                )
                bench_dir.mkdir(parents=True, exist_ok=True)
                ckpt_step_path = out_dir / f"step_{global_step:06d}.pt"
                ckpt_for_bench = (
                    str(ckpt_step_path) if ckpt_step_path.exists() else str(out_dir / "latest.pt")
                )
                out_json = str(bench_dir / f"step_{global_step:06d}.json")
                print(
                    f"[bench_eval] step={global_step} ckpt={Path(ckpt_for_bench).name}", flush=True
                )
                run_bench_eval_v5(
                    bench_path=str(args.bench_eval_path),
                    bench_script=str(args.bench_eval_script),
                    out_json=out_json,
                    ckpt_path=ckpt_for_bench,
                    tokenizer_path=str(tok_path),
                    max_seq_len=int(args.bench_eval_max_seq_len),
                    max_new_tokens=int(args.bench_eval_max_new_tokens),
                    min_new_tokens=int(args.bench_eval_min_new_tokens),
                    ban_first_steps=int(args.bench_eval_ban_first_steps),
                )
                try:
                    bench_result = json.loads(Path(out_json).read_text(encoding="utf-8"))
                    tracker.log(
                        "bench",
                        global_step,
                        acc_arithmetic=float(bench_result.get("acc_arithmetic", 0.0)),
                        acc_syllogism=float(bench_result.get("acc_syllogism", 0.0)),
                        acc_code=float(bench_result.get("acc_code", 0.0)),
                        bench_json=out_json,
                    )
                    tracker.render()
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    print(f"[bench_eval] failed to summarize metrics: {e}", flush=True)

    if last_checkpoint_step != global_step or not last_checkpoint_retained:
        save_ckpt(
            out_dir=out_dir,
            global_step=global_step,
            local_step=local_step,
            model=model,
            optim=optim,
            scaler=scaler if use_fp16 else None,
            model_config=asdict(cfg),
            train_args=vars(args),
            run_contract=run_contract,
            position_stats=position_stats,
            sampler_state=train_sampler.state_dict(),
            data_contract=data_contract,
            retain_step=True,
        )
    tracker.render()
    print(f"[done] saved final checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
