"""Exact-resume primitives shared by SFT, DPO, and GRPO."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
import math
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler

EXACT_RESUME_SCHEMA_VERSION = 1
SAMPLER_NAME = "deterministic_epoch_shuffle_v1"
_RUNTIME_ONLY_ARGS = frozenset({"out_dir", "resume"})


def validate_training_controls(
    args: Any,
    *,
    positive_fields: tuple[str, ...] = (),
    nonnegative_fields: tuple[str, ...] = (),
) -> None:
    """Validate shared optimizer schedule, batch, worker, and cadence controls."""

    def require_number(name: str, *, positive: bool) -> float:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"--{name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"--{name} must be finite")
        if positive and numeric <= 0:
            raise ValueError(f"--{name} must be positive")
        if not positive and numeric < 0:
            raise ValueError(f"--{name} must be non-negative")
        return numeric

    max_steps = require_number("max_steps", positive=True)
    require_number("lr", positive=True)
    warmup_steps = require_number("warmup_steps", positive=False)
    if warmup_steps > max_steps:
        raise ValueError("--warmup_steps must not exceed --max_steps")
    for name in positive_fields:
        require_number(name, positive=True)
    for name in nonnegative_fields:
        require_number(name, positive=False)


def file_identity(path: str | Path) -> dict[str, Any]:
    """Return a content identity suitable for binding an exact resume."""
    resolved = Path(path)
    digest = hashlib.sha256()
    size = 0
    with open(resolved, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _runtime_identity() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def build_resume_contract_base(
    *,
    stage: str,
    args: Mapping[str, Any],
    input_paths: Mapping[str, str | Path],
    dataset_size: int,
    batch_size: int,
    batches_per_step: int,
    seed: int,
) -> dict[str, Any]:
    """Bind training semantics, input bytes, and deterministic sampler geometry."""
    if not stage:
        raise ValueError("resume stage must be non-empty")
    if dataset_size <= 0:
        raise ValueError("exact resume requires a non-empty training dataset")
    if batch_size <= 0 or batches_per_step <= 0:
        raise ValueError("batch_size and batches_per_step must be positive")
    batches_per_epoch = dataset_size // batch_size
    if batches_per_epoch <= 0:
        raise ValueError(
            "drop_last training has zero batches: dataset_size must be >= batch_size"
        )

    bound_args = {
        key: _canonical_value(value)
        for key, value in sorted(args.items())
        if key not in _RUNTIME_ONLY_ARGS
    }
    identities = {
        name: file_identity(path)
        for name, path in sorted(input_paths.items())
        if str(path)
    }
    return {
        "schema_version": EXACT_RESUME_SCHEMA_VERSION,
        "stage": stage,
        "runtime": _runtime_identity(),
        "args": bound_args,
        "inputs": identities,
        "data_order": {
            "sampler": SAMPLER_NAME,
            "dataset_size": int(dataset_size),
            "batch_size": int(batch_size),
            "drop_last": True,
            "batches_per_epoch": int(batches_per_epoch),
            "batches_per_step": int(batches_per_step),
            "seed": int(seed),
        },
    }


def resume_contract_for_step(
    base_contract: Mapping[str, Any], step: int
) -> dict[str, Any]:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    data_order = dict(base_contract["data_order"])
    data_order["consumed_batches"] = step * int(data_order["batches_per_step"])
    return {
        **dict(base_contract),
        "data_order": data_order,
    }


def _first_contract_difference(
    saved: Any, expected: Any, path: str = "resume_contract"
) -> str | None:
    if isinstance(saved, Mapping) and isinstance(expected, Mapping):
        saved_keys = set(saved)
        expected_keys = set(expected)
        if saved_keys != expected_keys:
            return (
                f"{path} keys differ: saved_only={sorted(saved_keys - expected_keys)}, "
                f"current_only={sorted(expected_keys - saved_keys)}"
            )
        for key in sorted(saved_keys):
            difference = _first_contract_difference(
                saved[key], expected[key], f"{path}.{key}"
            )
            if difference:
                return difference
        return None
    if saved != expected:
        return f"{path} differs: saved={saved!r}, current={expected!r}"
    return None


def require_resume_step(
    checkpoint: Mapping[str, Any],
    *,
    stage: str,
    weights_only_hint: str,
) -> int:
    """Reject legacy/partial checkpoints before treating them as resumable."""
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("exact resume checkpoint has an invalid or missing step")
    if checkpoint.get("kind") != stage:
        raise RuntimeError(
            f"exact resume requires a {stage!r} checkpoint; "
            f"got kind={checkpoint.get('kind')!r}"
        )
    contract = checkpoint.get("resume_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError(
            "checkpoint lacks the exact-resume contract; start a new weights-only "
            f"run with {weights_only_hint} instead of --resume"
        )
    if contract.get("schema_version") != EXACT_RESUME_SCHEMA_VERSION:
        raise RuntimeError("unsupported exact-resume contract schema")
    if "optimizer" not in checkpoint or checkpoint["optimizer"] is None:
        raise RuntimeError(
            "checkpoint lacks optimizer state; start a new weights-only "
            f"run with {weights_only_hint} instead of --resume"
        )
    if "scaler" not in checkpoint:
        raise RuntimeError("checkpoint lacks scaler state declaration")
    if not isinstance(checkpoint.get("rng_state"), Mapping):
        raise RuntimeError("checkpoint lacks exact Python/NumPy/Torch RNG state")
    if not isinstance(checkpoint.get("loop_state"), Mapping):
        raise RuntimeError("checkpoint lacks exact post-training loop state")
    if stage in {"sft", "distill"} and not isinstance(checkpoint.get("aux_rng_state"), Mapping):
        raise RuntimeError("checkpoint lacks exact auxiliary sampling RNG state")
    return step


def validate_resume_contract(
    checkpoint: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    *,
    weights_only_hint: str,
) -> None:
    difference = _first_contract_difference(
        checkpoint.get("resume_contract"), expected_contract
    )
    if difference:
        raise RuntimeError(
            f"exact resume contract mismatch: {difference}. To branch or load only "
            f"weights, start a new run with {weights_only_hint}."
        )


def restore_training_state(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_fp16: bool,
) -> None:
    """Strictly restore model/optimizer/scaler; never fall back to fresh state."""
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("exact resume checkpoint lacks model state")
    if any(str(key).startswith("_orig_mod.") for key in state):
        state = {
            str(key)[len("_orig_mod.") :]: value
            for key, value in state.items()
        }
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(f"exact resume model state is incompatible: {exc}") from exc

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as exc:
        raise RuntimeError(
            f"exact resume optimizer state is incompatible: {exc}"
        ) from exc

    saved_scaler = checkpoint["scaler"]
    if use_fp16:
        if not isinstance(saved_scaler, Mapping):
            raise RuntimeError("FP16 exact resume requires saved GradScaler state")
        try:
            scaler.load_state_dict(saved_scaler)
        except Exception as exc:
            raise RuntimeError(
                f"exact resume scaler state is incompatible: {exc}"
            ) from exc
    elif saved_scaler is not None:
        raise RuntimeError(
            "checkpoint contains FP16 scaler state but current run is not FP16"
        )


def capture_rng_state() -> dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "schema_version": 1,
        "python": random.getstate(),
        "numpy": {
            "bit_generator": bit_generator,
            "keys": keys.tolist(),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "cuda_device_count": torch.cuda.device_count() if cuda_states is not None else 0,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore all process RNGs, rejecting CPU/CUDA topology changes."""
    if state.get("schema_version") != 1:
        raise RuntimeError("unsupported RNG-state schema")
    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, Mapping):
        raise RuntimeError("exact resume lacks NumPy RNG state")
    torch_cpu = state.get("torch_cpu")
    if not isinstance(torch_cpu, torch.Tensor):
        raise RuntimeError("exact resume lacks Torch CPU RNG state")

    saved_cuda = state.get("torch_cuda")
    saved_cuda_count = state.get("cuda_device_count")
    if isinstance(saved_cuda_count, bool) or not isinstance(saved_cuda_count, int):
        raise RuntimeError("exact resume has an invalid CUDA device count")
    current_cuda = torch.cuda.is_available()
    if (saved_cuda is None) != (not current_cuda):
        raise RuntimeError(
            "exact resume requires the same CPU/CUDA execution topology"
        )
    if saved_cuda is not None:
        if not isinstance(saved_cuda, list):
            raise RuntimeError("saved Torch CUDA RNG state must be a list")
        if saved_cuda_count != len(saved_cuda):
            raise RuntimeError("saved CUDA RNG state/device count is inconsistent")
        if saved_cuda_count != torch.cuda.device_count():
            raise RuntimeError(
                "exact resume requires the same number of CUDA devices"
            )
    elif saved_cuda_count != 0:
        raise RuntimeError("CPU RNG state must declare zero CUDA devices")

    try:
        random.setstate(state["python"])
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(torch_cpu.cpu())
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)
    except Exception as exc:
        raise RuntimeError(f"exact resume RNG state is incompatible: {exc}") from exc


class DeterministicEpochBatchSampler(Sampler[list[int]]):
    """Epoch-shuffled batches whose cursor is reconstructable from batch count."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int,
        start_batch: int = 0,
        drop_last: bool = True,
    ):
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        if start_batch < 0:
            raise ValueError("start_batch must be non-negative")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.batches_per_epoch = (
            self.dataset_size // self.batch_size
            if self.drop_last
            else (self.dataset_size + self.batch_size - 1) // self.batch_size
        )
        if self.batches_per_epoch <= 0:
            raise ValueError("sampler has zero batches per epoch")
        self.epoch, self.batch_offset = divmod(
            int(start_batch), self.batches_per_epoch
        )

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        offset = self.batch_offset
        generator = torch.Generator()
        mixed_seed = (
            self.seed + epoch * 0x4F1BBCDCBFA54001
        ) & 0x7FFF_FFFF_FFFF_FFFF
        generator.manual_seed(mixed_seed)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        for batch_index in range(offset, self.batches_per_epoch):
            start = batch_index * self.batch_size
            batch = order[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
        self.epoch = epoch + 1
        self.batch_offset = 0

    def __len__(self) -> int:
        return self.batches_per_epoch - self.batch_offset


def make_loader_generator(seed: int, stream: int) -> torch.Generator:
    """Keep DataLoader worker seeding off the model's global Torch RNG."""
    generator = torch.Generator()
    generator.manual_seed(
        (int(seed) + int(stream) * 0x6A09E667F3BCC909)
        & 0x7FFF_FFFF_FFFF_FFFF
    )
    return generator
