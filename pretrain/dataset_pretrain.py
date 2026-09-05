"""Packed pretraining data with reproducible, no-replacement traversal.

The production path is deliberately split into two concerns:

* :class:`PackedBinDataset` maps a stable block index to a fixed token window.
* :class:`ResumablePermutationSampler` controls the order in which those block
  indices are visited.

Keeping randomness out of ``Dataset.__getitem__`` is important. A shuffled
DataLoader over random windows is sampling *with* replacement, so one nominal
epoch does not mean that every training transition was seen. The sampler below
instead permutes every block exactly once before starting a new pass and can
reconstruct its order from an absolute, committed sample position.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sized
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS


def _shard_dtype(shard_dir: str) -> np.dtype:
    """Read the shard dtype from the build's meta.json (default uint16)."""
    d = Path(shard_dir)
    for meta_path in (d.parent / "meta.json", d.parent.parent / "meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dtype") == "uint32":
                return np.dtype(np.uint32)
            return np.dtype(np.uint16)
        except (OSError, json.JSONDecodeError):
            continue
    return np.dtype(np.uint16)


def list_shards(dir_path: str) -> list[Path]:
    p = Path(dir_path)
    files = sorted(x for x in p.glob("*.bin") if x.is_file())
    if not files:
        raise FileNotFoundError(f"No .bin shards found in: {dir_path}")
    return files


_CANONICAL_VOCAB_SIZE = 32_000


def _read_release_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read shard release manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"shard release manifest must be a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_int(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"release field {field} must be an integer, got {value!r}")
    result = int(value)
    if result < 0 or (positive and result <= 0):
        relation = "positive" if positive else "non-negative"
        raise RuntimeError(f"release field {field} must be {relation}, got {result}")
    return result


def _validated_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"release field {field} must be a lowercase SHA-256")
    return value


def _release_location(
    shard_dir: str | Path,
) -> tuple[Path, Path, str, str | None, str]:
    directory = Path(shard_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"shard directory does not exist: {directory}")

    if directory.name in {"train", "val"}:
        root = directory.parent
        split = directory.name
        source_name = None
    elif directory.parent.name == "val_by_source":
        root = directory.parent.parent
        split = "val_by_source"
        source_name = directory.name
    else:
        raise RuntimeError(
            "manifest-required shard paths must be <release>/train, <release>/val, "
            f"or <release>/val_by_source/<name>; got {directory}"
        )

    failed_markers = [
        path for path in (root / "meta.failed.json", root / "manifest.failed.json")
        if path.exists()
    ]
    if failed_markers:
        raise RuntimeError(
            "refusing a failed/ambiguous shard release; failure marker(s): "
            + ", ".join(str(path) for path in failed_markers)
        )

    candidates = [
        path for path in (root / "meta.json", root / "manifest.json") if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no meta.json or manifest.json release manifest found for {directory}"
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"ambiguous shard release at {root}: found {[path.name for path in candidates]}"
        )
    manifest_path = candidates[0]
    release_kind = "regular" if manifest_path.name == "meta.json" else "reference"
    return directory, manifest_path, split, source_name, release_kind


def _validate_canonical_release_contract(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    release_kind: str,
) -> None:
    expected_schema_version = 3 if release_kind == "regular" else 2
    if manifest.get("schema_version") != expected_schema_version:
        raise RuntimeError(
            f"unsupported {release_kind} release schema_version in {manifest_path}: "
            f"got {manifest.get('schema_version')!r}, expected {expected_schema_version}"
        )
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"shard release status must be 'complete' in {manifest_path}, "
            f"got {manifest.get('status')!r}"
        )

    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError(f"release has no canonical contract object: {manifest_path}")

    exact_contract = {
        "mode": "canonical",
        "canonical": True,
        "legacy_allow_noncanonical_contract": False,
        "expected_special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "expected_vocab_size": _CANONICAL_VOCAB_SIZE,
        "actual_vocab_size": _CANONICAL_VOCAB_SIZE,
        "add_bos": True,
        "add_eos": True,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "doc_sep": "",
    }
    mismatches = [
        f"{key}={contract.get(key)!r}, expected {expected!r}"
        for key, expected in exact_contract.items()
        if contract.get(key) != expected
    ]
    if contract.get("issues") != []:
        mismatches.append(f"issues={contract.get('issues')!r}, expected []")
    if mismatches:
        raise RuntimeError(
            f"non-canonical shard release contract in {manifest_path}: "
            + "; ".join(mismatches)
        )
    if manifest.get("vocab_size") != _CANONICAL_VOCAB_SIZE:
        raise RuntimeError(
            f"release vocab_size must be {_CANONICAL_VOCAB_SIZE}, "
            f"got {manifest.get('vocab_size')!r}"
        )

    if release_kind == "regular":
        legacy = manifest.get("legacy_flags")
        if not isinstance(legacy, dict):
            raise RuntimeError(f"regular release has no legacy_flags object: {manifest_path}")
        required_false = ("allow_noncanonical_contract", "replay_on_exhaustion")
        bad_legacy = [
            f"{name}={legacy.get(name)!r}"
            for name in required_false
            if legacy.get(name) is not False
        ]
        if bad_legacy:
            raise RuntimeError(
                f"legacy/debug shard release is forbidden in production: {', '.join(bad_legacy)}"
            )
        if manifest.get("source_exhaustion_policy") != "fail_fast":
            raise RuntimeError(
                "production shard release must use source_exhaustion_policy='fail_fast'"
            )
    else:
        if manifest.get("kind") != "petitgpt_cross_stage_reference_validation":
            raise RuntimeError(
                f"unsupported reference release kind: {manifest.get('kind')!r}"
            )
        if manifest.get("immutable") is not True:
            raise RuntimeError("reference validation release must declare immutable=true")


def _regular_reference_exclusion_sha256s(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[str, ...]:
    exclusion = manifest.get("reference_validation_exclusion")
    if not isinstance(exclusion, dict) or exclusion.get("enabled") is not True:
        raise RuntimeError(
            f"production release must enable reference validation exclusion: {manifest_path}"
        )
    manifest_count = _manifest_int(
        exclusion.get("manifest_count"),
        field="reference_validation_exclusion.manifest_count",
        positive=True,
    )
    _manifest_int(
        exclusion.get("union_hash_count"),
        field="reference_validation_exclusion.union_hash_count",
        positive=True,
    )
    entries = exclusion.get("manifests")
    if not isinstance(entries, list) or len(entries) != manifest_count:
        raise RuntimeError(
            "reference validation exclusion manifest_count does not match manifests"
        )

    sha256s: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("enabled") is not True:
            raise RuntimeError(
                f"reference exclusion manifest {index} is missing or not enabled"
            )
        value = entry.get("manifest_sha256")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise RuntimeError(
                f"reference exclusion manifest {index} has invalid manifest_sha256"
            )
        _manifest_int(
            entry.get("hash_count"),
            field=f"reference_validation_exclusion.manifests.{index}.hash_count",
            positive=True,
        )
        sha256s.append(value)

    if len(set(sha256s)) != len(sha256s):
        raise RuntimeError("reference exclusion manifests contain duplicate SHA-256 values")
    return tuple(sorted(sha256s))


def _regular_release_expectations(
    manifest: Mapping[str, Any],
    *,
    split: str,
    source_name: str | None,
) -> tuple[int, int, int]:
    if split == "train":
        expected_shards = _manifest_int(
            manifest.get("train_shards"), field="train_shards", positive=True
        )
        expected_tokens = _manifest_int(
            manifest.get("train_tokens"), field="train_tokens", positive=True
        )
        shard_tokens = _manifest_int(
            manifest.get("shard_tokens"), field="shard_tokens", positive=True
        )
        accounting = (manifest.get("accounting") or {}).get("train") or {}
        if accounting.get("emitted_shard_tokens") != expected_tokens:
            raise RuntimeError("train token count disagrees with accounting.train")
        return expected_shards, expected_tokens, shard_tokens

    if split == "val":
        expected_shards = _manifest_int(
            manifest.get("val_shards"), field="val_shards", positive=True
        )
        expected_tokens = _manifest_int(
            manifest.get("val_tokens"), field="val_tokens", positive=True
        )
        shard_tokens = _manifest_int(
            manifest.get("val_shard_tokens"), field="val_shard_tokens", positive=True
        )
        accounting = (manifest.get("accounting") or {}).get("val") or {}
        if accounting.get("emitted_shard_tokens") != expected_tokens:
            raise RuntimeError("val token count disagrees with accounting.val")
        return expected_shards, expected_tokens, shard_tokens

    assert source_name is not None
    entry = (manifest.get("val_by_source") or {}).get(source_name)
    if not isinstance(entry, dict):
        raise RuntimeError(f"release has no val_by_source entry for {source_name!r}")
    expected_shards = _manifest_int(
        entry.get("shards"), field=f"val_by_source.{source_name}.shards", positive=True
    )
    expected_tokens = _manifest_int(
        entry.get("tokens"), field=f"val_by_source.{source_name}.tokens", positive=True
    )
    shard_tokens = _manifest_int(
        manifest.get("val_shard_tokens"), field="val_shard_tokens", positive=True
    )
    return expected_shards, expected_tokens, shard_tokens


def _reference_release_expectations(
    manifest: Mapping[str, Any],
    *,
    split: str,
    source_name: str | None,
) -> tuple[int, int, int]:
    shard_tokens = _manifest_int(
        (manifest.get("packing") or {}).get("shard_tokens"),
        field="packing.shard_tokens",
        positive=True,
    )
    if split == "val":
        if (manifest.get("outputs") or {}).get("combined_val") != "val":
            raise RuntimeError("reference manifest combined_val output is not 'val'")
        accounting = manifest.get("accounting") or {}
        expected_shards = _manifest_int(
            accounting.get("combined_shards"),
            field="accounting.combined_shards",
            positive=True,
        )
        expected_tokens = _manifest_int(
            accounting.get("emitted_shard_tokens"),
            field="accounting.emitted_shard_tokens",
            positive=True,
        )
        if accounting.get("serialized_tokens") != expected_tokens:
            raise RuntimeError("reference combined token accounting is inconsistent")
        return expected_shards, expected_tokens, shard_tokens

    if split == "train":
        raise RuntimeError("reference validation releases do not contain a train split")

    assert source_name is not None
    if (manifest.get("outputs") or {}).get("val_by_source") != "val_by_source":
        raise RuntimeError("reference manifest val_by_source output is invalid")
    entry = (manifest.get("sources") or {}).get(source_name)
    if not isinstance(entry, dict):
        raise RuntimeError(f"reference release has no source entry for {source_name!r}")
    realized = entry.get("realized") or {}
    expected_shards = _manifest_int(
        realized.get("shards"), field=f"sources.{source_name}.realized.shards", positive=True
    )
    expected_tokens = _manifest_int(
        realized.get("serialized_tokens"),
        field=f"sources.{source_name}.realized.serialized_tokens",
        positive=True,
    )
    content = _manifest_int(
        realized.get("content_tokens"),
        field=f"sources.{source_name}.realized.content_tokens",
    )
    boundaries = _manifest_int(
        realized.get("boundary_tokens"),
        field=f"sources.{source_name}.realized.boundary_tokens",
    )
    if content + boundaries != expected_tokens:
        raise RuntimeError(f"reference source accounting is inconsistent for {source_name!r}")
    return expected_shards, expected_tokens, shard_tokens


def _release_shard_records(
    manifest: Mapping[str, Any],
    *,
    release_kind: str,
    split: str,
    source_name: str | None,
) -> list[Mapping[str, Any]]:
    """Return the ordered byte-integrity records for one release split."""
    shard_files = manifest.get("shard_files")
    if not isinstance(shard_files, dict):
        raise RuntimeError("release has no shard_files integrity object")
    expected_keys = (
        {"hash_algorithm", "train", "val", "val_by_source"}
        if release_kind == "regular"
        else {"hash_algorithm", "val", "val_by_source"}
    )
    if set(shard_files) != expected_keys:
        raise RuntimeError(
            "release shard_files keys mismatch: "
            f"actual={sorted(shard_files)}, expected={sorted(expected_keys)}"
        )
    if shard_files.get("hash_algorithm") != "sha256":
        raise RuntimeError("release shard_files.hash_algorithm must be 'sha256'")

    val_by_source = shard_files.get("val_by_source")
    if not isinstance(val_by_source, dict):
        raise RuntimeError("release shard_files.val_by_source must be an object")
    declared_sources = (
        set((manifest.get("val_by_source") or {}).keys())
        if release_kind == "regular"
        else set((manifest.get("sources") or {}).keys())
    )
    if set(val_by_source) != declared_sources:
        raise RuntimeError(
            "release shard_files.val_by_source keys disagree with declared sources"
        )

    raw_records = (
        val_by_source.get(source_name)
        if split == "val_by_source"
        else shard_files.get(split)
    )
    if not isinstance(raw_records, list):
        raise RuntimeError(f"release shard_files.{split} must be a list")
    return raw_records


def _stable_shard_sha256(path: Path) -> str:
    """Hash one shard and reject mutation during the integrity scan."""
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    identity_before = (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_ino,
        before.st_dev,
    )
    identity_after = (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_ino,
        after.st_dev,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"shard changed while hashing: {path}")
    return digest


def validate_shard_release(shard_dir: str | Path) -> dict[str, Any]:
    """Validate a canonical, completely published packed-shard split.

    Production callers use this before opening any memmap. The check rejects
    missing/failed/legacy manifests, ambiguous split paths, gaps or extras in the
    contiguous shard sequence, every count/dtype/byte-geometry mismatch, and any
    shard whose bytes no longer match its ordered SHA-256 manifest record.
    """

    directory, manifest_path, split, source_name, release_kind = _release_location(
        shard_dir
    )
    manifest = _read_release_json(manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)
    _validate_canonical_release_contract(
        manifest, manifest_path=manifest_path, release_kind=release_kind
    )
    tokenizer_sha256 = _validated_sha256(
        manifest.get("tokenizer_sha256"), field="tokenizer_sha256"
    )

    dtype_name = manifest.get("dtype")
    if dtype_name != "uint16":
        raise RuntimeError(
            f"canonical 32k shard release dtype must be 'uint16', got {dtype_name!r}"
        )
    dtype = np.dtype(np.uint16)

    if release_kind == "regular":
        reference_exclusion_manifest_sha256s = (
            _regular_reference_exclusion_sha256s(
                manifest, manifest_path=manifest_path
            )
        )
        expected_shards, expected_tokens, shard_tokens = _regular_release_expectations(
            manifest, split=split, source_name=source_name
        )
    else:
        reference_exclusion_manifest_sha256s = ()
        expected_shards, expected_tokens, shard_tokens = _reference_release_expectations(
            manifest, split=split, source_name=source_name
        )

    count_from_tokens = (expected_tokens + shard_tokens - 1) // shard_tokens
    if count_from_tokens != expected_shards:
        raise RuntimeError(
            "manifest shard count/token geometry is inconsistent: "
            f"shards={expected_shards}, tokens={expected_tokens}, "
            f"shard_tokens={shard_tokens}"
        )

    expected_names = [f"shard_{index:05d}.bin" for index in range(expected_shards)]
    shard_records = _release_shard_records(
        manifest,
        release_kind=release_kind,
        split=split,
        source_name=source_name,
    )
    if len(shard_records) != expected_shards:
        raise RuntimeError(
            "shard integrity record count mismatch: "
            f"actual={len(shard_records)}, expected={expected_shards}"
        )
    actual_bin_names = sorted(
        path.name for path in directory.iterdir() if path.is_file() and path.suffix == ".bin"
    )
    if actual_bin_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_bin_names))
        extra = sorted(set(actual_bin_names) - set(expected_names))
        raise RuntimeError(
            f"shard filename/count mismatch in {directory}: missing={missing}, extra={extra}"
        )

    unexpected_artifacts = sorted(
        path.name
        for path in directory.iterdir()
        if path.name.startswith("shard_") and path.name not in expected_names
    )
    if unexpected_artifacts:
        raise RuntimeError(
            f"unexpected shard artifacts in {directory}: {unexpected_artifacts}"
        )

    expected_lengths = [shard_tokens] * (expected_shards - 1)
    expected_lengths.append(expected_tokens - shard_tokens * (expected_shards - 1))
    shard_paths: list[Path] = []
    validated_records: list[dict[str, Any]] = []
    observed_tokens = 0
    record_keys = {"path", "size_bytes", "token_count", "sha256"}
    for index, (name, expected_length) in enumerate(
        zip(expected_names, expected_lengths, strict=True)
    ):
        raw_record = shard_records[index]
        if not isinstance(raw_record, dict) or set(raw_record) != record_keys:
            raise RuntimeError(
                f"shard integrity record {index} must contain exactly "
                f"{sorted(record_keys)}"
            )
        expected_relative_path = (
            Path("val_by_source") / str(source_name) / name
            if split == "val_by_source"
            else Path(split) / name
        ).as_posix()
        if raw_record.get("path") != expected_relative_path:
            raise RuntimeError(
                f"shard integrity record order/path mismatch at index {index}: "
                f"actual={raw_record.get('path')!r}, expected={expected_relative_path!r}"
            )
        record_tokens = _manifest_int(
            raw_record.get("token_count"),
            field=f"shard_files.{split}.{index}.token_count",
        )
        record_bytes = _manifest_int(
            raw_record.get("size_bytes"),
            field=f"shard_files.{split}.{index}.size_bytes",
        )
        record_sha256 = _validated_sha256(
            raw_record.get("sha256"),
            field=f"shard_files.{split}.{index}.sha256",
        )
        expected_bytes = expected_length * dtype.itemsize
        if record_tokens != expected_length or record_bytes != expected_bytes:
            raise RuntimeError(
                f"shard integrity geometry mismatch for {name}: "
                f"tokens={record_tokens}, bytes={record_bytes}, "
                f"expected_tokens={expected_length}, expected_bytes={expected_bytes}"
            )
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"shard must be a regular non-symlink file: {path}")
        byte_size = int(path.stat().st_size)
        if byte_size % dtype.itemsize:
            raise RuntimeError(
                f"shard byte size is not divisible by {dtype.name} itemsize: {path}"
            )
        token_count = byte_size // dtype.itemsize
        if token_count != expected_length:
            raise RuntimeError(
                f"shard token length mismatch for {path}: "
                f"actual={token_count}, expected={expected_length}"
            )
        actual_sha256 = _stable_shard_sha256(path)
        if actual_sha256 != record_sha256:
            raise RuntimeError(
                f"shard SHA-256 mismatch for {path}: "
                f"actual={actual_sha256}, expected={record_sha256}"
            )
        observed_tokens += token_count
        shard_paths.append(path)
        validated_records.append(dict(raw_record))

    if observed_tokens != expected_tokens:
        raise RuntimeError(
            f"release token count mismatch: actual={observed_tokens}, expected={expected_tokens}"
        )

    return {
        "release_kind": release_kind,
        "release_root": str(manifest_path.parent.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "split": split,
        "source_name": source_name,
        "dtype": dtype.name,
        "expected_shards": expected_shards,
        "expected_tokens": expected_tokens,
        "shard_tokens": shard_tokens,
        "reference_exclusion_manifest_sha256s": reference_exclusion_manifest_sha256s,
        "shard_file_records": tuple(validated_records),
        "shard_paths": tuple(shard_paths),
    }


def _mixed_epoch_seed(seed: int, epoch: int) -> int:
    """Mix ``seed`` and ``epoch`` into a stable torch Generator seed."""
    mask = (1 << 64) - 1
    x = ((int(seed) & mask) + ((int(epoch) + 1) * 0x9E3779B97F4A7C15)) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    x ^= x >> 31
    # Keep the high bit clear for portability across torch versions.
    return int(x & ((1 << 63) - 1))


class PackedBinDataset(Dataset):
    """A virtual packed token stream exposed as fixed next-token windows.

    In the default ``deterministic`` mode, all shard files are treated as one
    virtual concatenated stream. Item ``i`` starts at ``i * seq_len`` and reads
    ``seq_len + 1`` tokens, so adjacent items overlap by exactly one token.
    Consequently every usable next-token transition appears exactly once,
    including transitions across physical shard boundaries. Only the final
    global tail shorter than ``seq_len`` transitions is omitted.

    ``sampling_mode="random"`` preserves random-window sampling for explicit
    debugging experiments. It samples with replacement and may reject
    EOS-heavy candidates; it must not be used for token-accounted production
    runs.

    Returned token tensors use signed storage because PyTorch 2.2 has no
    uint16 tensor dtype. Canonical 32k-vocabulary uint16 IDs are exposed as a
    zero-copy int16 view (all IDs are <= 31,999); an unexpected ID above
    32,767 safely falls back to int32. uint32 shards widen to int64. Trainers
    convert these storage tensors to long before embedding lookup.
    """

    VALID_SAMPLING_MODES = frozenset({"deterministic", "random"})

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        bos_id: int = 2,
        eos_id: int = 3,
        mask_bos_in_loss: bool = True,
        mask_last_label_in_loss: bool = False,
        *,
        sampling_mode: str = "deterministic",
        mask_repeated_eos_in_loss: bool = True,
        max_eos_frac: float = 0.20,
        resample_tries: int = 8,
        require_release_manifest: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.window_size = self.seq_len + 1
        self.block_stride = self.seq_len
        # Compatibility with existing diagnostics which called T+1 `block`.
        self.block = self.window_size

        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.mask_bos_in_loss = bool(mask_bos_in_loss)
        self.mask_last_label_in_loss = bool(mask_last_label_in_loss)

        sampling_mode = str(sampling_mode).strip().lower()
        if sampling_mode not in self.VALID_SAMPLING_MODES:
            choices = ", ".join(sorted(self.VALID_SAMPLING_MODES))
            raise ValueError(f"sampling_mode must be one of {{{choices}}}, got {sampling_mode!r}")
        self.sampling_mode = sampling_mode

        self.mask_repeated_eos_in_loss = bool(mask_repeated_eos_in_loss)
        self.max_eos_frac = float(max_eos_frac)
        if not 0.0 <= self.max_eos_frac <= 1.0:
            raise ValueError(f"max_eos_frac must be in [0, 1], got {max_eos_frac}")
        self.resample_tries = int(resample_tries)
        if self.resample_tries < 1:
            raise ValueError(f"resample_tries must be >= 1, got {resample_tries}")

        self.require_release_manifest = bool(require_release_manifest)
        self.release_validation: dict[str, Any] | None = None
        if self.require_release_manifest:
            release = validate_shard_release(shard_dir)
            self.shards = list(release["shard_paths"])
            self._dtype = np.dtype(release["dtype"])
            self.release_validation = {
                key: value
                for key, value in release.items()
                if key != "shard_paths"
            }
        else:
            self.shards = list_shards(shard_dir)
            self._dtype = _shard_dtype(shard_dir)
        self._mms: list[np.memmap] = []
        self.shard_lens: list[int] = []
        self.shard_token_offsets: list[int] = [0]

        itemsize = int(self._dtype.itemsize)
        for shard in self.shards:
            byte_size = int(shard.stat().st_size)
            if byte_size == 0:
                raise RuntimeError(f"Empty shard is not valid packed data: {shard}")
            if byte_size % itemsize:
                raise RuntimeError(
                    "Shard byte size is not divisible by dtype itemsize: "
                    f"path={shard}, bytes={byte_size}, dtype={self._dtype}"
                )
            mm = np.memmap(shard, dtype=self._dtype, mode="r")
            n_tokens = int(mm.shape[0])
            self._mms.append(mm)
            self.shard_lens.append(n_tokens)
            self.shard_token_offsets.append(self.shard_token_offsets[-1] + n_tokens)

        self.total_raw_tokens = int(self.shard_token_offsets[-1])
        self.total_transitions = max(0, self.total_raw_tokens - 1)
        self.n_blocks = self.total_transitions // self.block_stride
        self.usable_transitions = self.n_blocks * self.block_stride
        self.tail_transitions = self.total_transitions - self.usable_transitions
        # These are target tokens after the final covered transition, not an
        # extra T+1 anchor token.
        self.tail_tokens = self.tail_transitions

        boundary_offsets = self.shard_token_offsets[1:-1]
        self.cross_shard_transitions = len(boundary_offsets)
        self.covered_cross_shard_transitions = sum(
            int(boundary <= self.usable_transitions) for boundary in boundary_offsets
        )
        self.tail_cross_shard_transitions = (
            self.cross_shard_transitions - self.covered_cross_shard_transitions
        )

        self.shard_stats: list[dict[str, Any]] = [
            {
                "path": str(path),
                "tokens": int(length),
                "global_start": int(self.shard_token_offsets[i]),
                "global_end_exclusive": int(self.shard_token_offsets[i + 1]),
            }
            for i, (path, length) in enumerate(
                zip(self.shards, self.shard_lens, strict=True)
            )
        ]

        if self.n_blocks == 0:
            raise RuntimeError(
                f"No full windows found. seq_len={seq_len}, "
                f"tokens={self.total_raw_tokens}, dir={shard_dir}"
            )

        # Dataset objects are copied into DataLoader worker processes. This
        # state is worker-local and used only by explicit random mode.
        self._rng_inst: np.random.Generator | None = None

    @property
    def _lens(self) -> list[int]:
        """Backward-compatible, non-duplicated view of ``shard_lens``."""
        return self.shard_lens

    def stats(self) -> dict[str, Any]:
        """Return exact packing/exposure statistics for run manifests."""
        return {
            "sampling_mode": self.sampling_mode,
            "require_release_manifest": self.require_release_manifest,
            "release_validation": self.release_validation,
            "shard_storage_dtype": self._dtype.name,
            "torch_common_storage_dtype": "int16" if self._dtype == np.uint16 else "int64",
            "torch_high_uint16_id_fallback_dtype": "int32",
            "n_shards": len(self.shards),
            "shard_lens": list(self.shard_lens),
            "total_raw_tokens": self.total_raw_tokens,
            "total_transitions": self.total_transitions,
            "seq_len": self.seq_len,
            "window_size": self.window_size,
            "block_stride": self.block_stride,
            "n_blocks": self.n_blocks,
            "usable_transitions": self.usable_transitions,
            "tail_transitions": self.tail_transitions,
            "cross_shard_transitions": self.cross_shard_transitions,
            "covered_cross_shard_transitions": self.covered_cross_shard_transitions,
            "tail_cross_shard_transitions": self.tail_cross_shard_transitions,
            "mask_last_label_in_loss": self.mask_last_label_in_loss,
        }

    def __len__(self) -> int:
        return self.n_blocks

    def _validate_index(self, i: int) -> int:
        i = int(i)
        if i < 0:
            i += self.n_blocks
        if i < 0 or i >= self.n_blocks:
            raise IndexError(f"Packed block index {i} out of range for {self.n_blocks} blocks")
        return i

    def _shard_for_global_token(self, global_token_i: int) -> int:
        if global_token_i < 0 or global_token_i >= self.total_raw_tokens:
            raise IndexError(
                f"Global token index {global_token_i} out of range "
                f"for {self.total_raw_tokens} tokens"
            )
        return bisect_right(self.shard_token_offsets, int(global_token_i)) - 1

    def _slice_global(self, start: int, length: int) -> torch.Tensor:
        """Copy a virtual-stream slice, including across shard boundaries."""
        start = int(start)
        length = int(length)
        end = start + length
        if start < 0 or length < 0 or end > self.total_raw_tokens:
            raise IndexError(
                f"Invalid virtual token slice [{start}:{end}] "
                f"for {self.total_raw_tokens} tokens"
            )

        toks = np.empty(length, dtype=self._dtype)
        global_pos = start
        written = 0
        while written < length:
            shard_i = self._shard_for_global_token(global_pos)
            local_start = global_pos - self.shard_token_offsets[shard_i]
            available = self.shard_lens[shard_i] - local_start
            take = min(length - written, available)
            toks[written : written + take] = self._mms[shard_i][
                local_start : local_start + take
            ]
            global_pos += take
            written += take

        if toks.dtype == np.uint16:
            if int(toks.max()) <= np.iinfo(np.int16).max:
                # Canonical 32k IDs fit int16; keep worker IPC at 2 B/token.
                return torch.from_numpy(toks.view(np.int16))
            # Preserve unexpected-but-valid uint16 IDs instead of wrapping.
            return torch.from_numpy(toks.astype(np.int32))

        # torch 2.2 has no uint32 tensor dtype; widen without changing IDs.
        return torch.from_numpy(toks.astype(np.int64))

    def _rng(self) -> np.random.Generator:
        """Worker-safe stateful RNG used only for explicit random mode."""
        if self._rng_inst is not None:
            return self._rng_inst

        info = torch.utils.data.get_worker_info()
        if info is None:
            seed = int(torch.initial_seed() % (2**32))
        else:
            seed = int(info.seed % (2**32))
        self._rng_inst = np.random.default_rng(seed)
        return self._rng_inst

    def _deterministic_block(self, i: int) -> torch.Tensor:
        return self._slice_global(i * self.block_stride, self.window_size)

    def _random_block(self) -> torch.Tensor:
        rng = self._rng()
        max_start = self.total_raw_tokens - self.window_size
        last_candidate: torch.Tensor | None = None
        for _ in range(self.resample_tries):
            start = int(rng.integers(0, max_start + 1))
            candidate = self._slice_global(start, self.window_size)
            last_candidate = candidate
            labels = candidate[1:]
            eos_frac = float((labels == self.eos_id).float().mean().item())
            if eos_frac <= self.max_eos_frac:
                break

        assert last_candidate is not None
        return last_candidate

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self._validate_index(i)
        if self.sampling_mode == "deterministic":
            toks = self._deterministic_block(i)
        else:
            toks = self._random_block()

        input_ids = toks[:-1].contiguous()
        labels = toks[1:].contiguous()
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        if self.mask_bos_in_loss:
            loss_mask *= (labels != self.bos_id).float()

        if self.mask_repeated_eos_in_loss and labels.numel() > 1:
            repeated_eos = (labels[1:] == self.eos_id) & (labels[:-1] == self.eos_id)
            loss_mask[1:] *= (~repeated_eos).float()

        if self.mask_last_label_in_loss and loss_mask.numel() > 0:
            loss_mask[-1] = 0.0

        return input_ids, labels, loss_mask


class ResumablePermutationSampler(Sampler[int]):
    """Reconstructible no-replacement permutations from a global position.

    A global position is ``epoch * N + offset``. Each epoch has a deterministic
    permutation derived from ``seed`` and ``epoch``. Iteration may span epoch
    boundaries, but no index is replayed before the current permutation ends.

    ``committed_position`` advances only via :meth:`commit`: DataLoader workers
    prefetch indices, so issued indices are not necessarily consumed indices.
    The finite ``end_position`` is stored in checkpoints, so loading sampler
    state emits only the unconsumed suffix of the original sample budget.

    If ``num_samples`` is omitted, the planned range ends at the current epoch
    boundary. Production trainers should pass an exact batch-divisible sample
    budget for the remainder of the stage.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int = 0,
        start_position: int = 0,
        num_samples: int | None = None,
    ) -> None:
        self.data_source = data_source
        self.data_length = int(len(data_source))
        if self.data_length <= 0:
            raise ValueError("ResumablePermutationSampler requires a non-empty dataset")
        sampling_mode = getattr(data_source, "sampling_mode", "deterministic")
        if sampling_mode != "deterministic":
            raise ValueError(
                "ResumablePermutationSampler requires deterministic dataset "
                f"indexing, got sampling_mode={sampling_mode!r}"
            )

        self.seed = int(seed)
        start_position = self._validate_position(start_position)
        if num_samples is not None and int(num_samples) < 0:
            raise ValueError(f"num_samples must be >= 0 or None, got {num_samples}")
        if num_samples is None:
            requested = self.data_length - (start_position % self.data_length)
        else:
            requested = int(num_samples)
        self.range_start_position = start_position
        self.committed_position = start_position
        self.end_position = start_position + requested

    @staticmethod
    def _validate_position(position: int) -> int:
        position = int(position)
        if position < 0:
            raise ValueError(f"start/committed position must be >= 0, got {position}")
        return position

    @property
    def position(self) -> int:
        return self.committed_position

    @property
    def epoch(self) -> int:
        return self.committed_position // self.data_length

    @property
    def epoch_offset(self) -> int:
        return self.committed_position % self.data_length

    @property
    def num_samples(self) -> int:
        """Number of planned samples that have not yet been committed."""
        return self.end_position - self.committed_position

    @property
    def configured_num_samples(self) -> int:
        return self.end_position - self.range_start_position

    def set_position(self, position: int, *, num_samples: int | None = None) -> None:
        """Set committed position, optionally replacing the planned range."""
        position = self._validate_position(position)
        if num_samples is not None:
            if int(num_samples) < 0:
                raise ValueError(f"num_samples must be >= 0, got {num_samples}")
            self.range_start_position = position
            self.end_position = position + int(num_samples)
        elif position > self.end_position:
            raise ValueError(
                f"Position {position} exceeds planned end position {self.end_position}"
            )
        self.committed_position = position

    def commit(self, num_samples: int) -> None:
        """Advance checkpoint-safe state after samples are actually consumed."""
        num_samples = int(num_samples)
        if num_samples < 0:
            raise ValueError(f"Cannot commit a negative number of samples: {num_samples}")
        new_position = self.committed_position + num_samples
        if new_position > self.end_position:
            raise ValueError(
                f"Commit would exceed planned end position: {new_position} > {self.end_position}"
            )
        self.committed_position = new_position

    def state_dict(self) -> dict[str, int]:
        return {
            "version": 2,
            "data_length": self.data_length,
            "seed": self.seed,
            "range_start_position": self.range_start_position,
            "committed_position": self.committed_position,
            "end_position": self.end_position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        version = int(state.get("version", 0))
        if version != 2:
            raise ValueError(f"Unsupported sampler state version: {version}")
        saved_length = int(state["data_length"])
        if saved_length != self.data_length:
            raise ValueError(
                "Cannot resume sampler with a different dataset length: "
                f"checkpoint={saved_length}, current={self.data_length}"
            )

        range_start = self._validate_position(int(state["range_start_position"]))
        committed = self._validate_position(int(state["committed_position"]))
        end = self._validate_position(int(state["end_position"]))
        if not range_start <= committed <= end:
            raise ValueError(
                "Invalid sampler state positions: expected "
                f"range_start <= committed <= end, got {range_start}, {committed}, {end}"
            )
        self.seed = int(state["seed"])
        self.range_start_position = range_start
        self.committed_position = committed
        self.end_position = end

    def _permutation(self, epoch: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_mixed_epoch_seed(self.seed, epoch))
        # ~48 MiB for six million blocks, with no Python-int list copy.
        return torch.randperm(self.data_length, generator=generator, dtype=torch.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        global_position = self.committed_position
        remaining = len(self)
        while remaining > 0:
            epoch, offset = divmod(global_position, self.data_length)
            permutation = self._permutation(epoch)
            take = min(remaining, self.data_length - offset)
            # `.tolist()` would create millions of heavyweight Python ints.
            for sample_i in permutation[offset : offset + take].numpy():
                yield int(sample_i)
            global_position += take
            remaining -= take
            # Release this tensor before constructing the next epoch's
            # permutation, keeping peak sampler memory bounded.
            del permutation


class FixedSubsetSampler(Sampler[int]):
    """A repeatable validation order spread across the virtual token stream.

    For a subset, the stream is divided into equal non-overlapping strata and
    one seed-fixed index is selected from each. This avoids head-of-shard bias
    and reconstructs exactly the same order for every evaluation.
    """

    def __init__(
        self,
        data_source: Sized,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.data_length = int(len(data_source))
        if self.data_length <= 0:
            raise ValueError("FixedSubsetSampler requires a non-empty dataset")
        sampling_mode = getattr(data_source, "sampling_mode", "deterministic")
        if sampling_mode != "deterministic":
            raise ValueError(
                "FixedSubsetSampler requires deterministic dataset indexing, "
                f"got sampling_mode={sampling_mode!r}"
            )
        if num_samples is None:
            num_samples = self.data_length
        self.num_samples = int(num_samples)
        if not 0 <= self.num_samples <= self.data_length:
            raise ValueError(
                f"num_samples must be in [0, {self.data_length}], got {self.num_samples}"
            )
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.num_samples == self.data_length:
            yield from range(self.data_length)
            return
        if self.num_samples == 0:
            return

        rng = np.random.default_rng(self.seed)
        n = self.data_length
        k = self.num_samples
        for stratum_i in range(k):
            start = (stratum_i * n) // k
            end = ((stratum_i + 1) * n) // k
            yield int(rng.integers(start, end))
