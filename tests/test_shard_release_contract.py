from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pretrain.dataset_pretrain import PackedBinDataset, validate_shard_release
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS


def _canonical_contract() -> dict[str, Any]:
    return {
        "mode": "canonical",
        "canonical": True,
        "legacy_allow_noncanonical_contract": False,
        "issues": [],
        "expected_special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "expected_vocab_size": 32_000,
        "actual_vocab_size": 32_000,
        "add_bos": True,
        "add_eos": True,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "doc_sep": "",
    }


def _write_shards(
    directory: Path,
    lengths: tuple[int, ...] = (4, 3),
    *,
    dtype: np.dtype[Any] | None = None,
) -> None:
    if dtype is None:
        dtype = np.dtype(np.uint16)
    directory.mkdir(parents=True)
    start = 0
    for index, length in enumerate(lengths):
        np.arange(start, start + length, dtype=dtype).tofile(directory / f"shard_{index:05d}.bin")
        start += length


def _shard_records(root: Path, directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.bin")):
        size_bytes = path.stat().st_size
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": size_bytes,
            "token_count": size_bytes // np.dtype(np.uint16).itemsize,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records


def _write_regular_release(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "regular"
    _write_shards(root / "train")
    _write_shards(root / "val")
    _write_shards(root / "val_by_source" / "wiki")
    meta = {
        "schema_version": 3,
        "status": "complete",
        "dtype": "uint16",
        "vocab_size": 32_000,
        "tokenizer_sha256": "d" * 64,
        "contract": _canonical_contract(),
        "legacy_flags": {
            "allow_noncanonical_contract": False,
            "replay_on_exhaustion": False,
        },
        "reference_validation_exclusion": {
            "enabled": True,
            "manifest_count": 1,
            "union_hash_count": 2,
            "manifests": [
                {
                    "enabled": True,
                    "manifest_sha256": "a" * 64,
                    "hash_count": 2,
                }
            ],
        },
        "source_exhaustion_policy": "fail_fast",
        "shard_tokens": 4,
        "val_shard_tokens": 4,
        "train_shards": 2,
        "train_tokens": 7,
        "val_shards": 2,
        "val_tokens": 7,
        "accounting": {
            "train": {"emitted_shard_tokens": 7},
            "val": {"emitted_shard_tokens": 7},
        },
        "val_by_source": {
            "wiki": {"tokens": 7, "shards": 2},
        },
        "shard_files": {
            "hash_algorithm": "sha256",
            "train": _shard_records(root, root / "train"),
            "val": _shard_records(root, root / "val"),
            "val_by_source": {
                "wiki": _shard_records(root, root / "val_by_source" / "wiki"),
            },
        },
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root, meta


def _write_reference_release(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "reference"
    _write_shards(root / "val")
    _write_shards(root / "val_by_source" / "wiki")
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "kind": "petitgpt_cross_stage_reference_validation",
        "immutable": True,
        "dtype": "uint16",
        "vocab_size": 32_000,
        "tokenizer_sha256": "d" * 64,
        "contract": _canonical_contract(),
        "packing": {"shard_tokens": 4},
        "accounting": {
            "serialized_tokens": 7,
            "emitted_shard_tokens": 7,
            "combined_shards": 2,
        },
        "outputs": {
            "combined_val": "val",
            "val_by_source": "val_by_source",
        },
        "sources": {
            "wiki": {
                "realized": {
                    "content_tokens": 5,
                    "boundary_tokens": 2,
                    "serialized_tokens": 7,
                    "shards": 2,
                }
            }
        },
        "shard_files": {
            "hash_algorithm": "sha256",
            "val": _shard_records(root, root / "val"),
            "val_by_source": {
                "wiki": _shard_records(root, root / "val_by_source" / "wiki"),
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def _rewrite_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_regular_release_happy_path_and_dataset_summary(tmp_path: Path) -> None:
    root, _ = _write_regular_release(tmp_path)

    release = validate_shard_release(root / "train")
    assert release["release_kind"] == "regular"
    assert release["expected_shards"] == 2
    assert release["expected_tokens"] == 7
    assert (
        release["manifest_sha256"] == hashlib.sha256((root / "meta.json").read_bytes()).hexdigest()
    )
    assert release["tokenizer_sha256"] == "d" * 64
    assert release["reference_exclusion_manifest_sha256s"] == ("a" * 64,)
    assert [path.name for path in release["shard_paths"]] == [
        "shard_00000.bin",
        "shard_00001.bin",
    ]
    assert [item["path"] for item in release["shard_file_records"]] == [
        "train/shard_00000.bin",
        "train/shard_00001.bin",
    ]

    dataset = PackedBinDataset(
        str(root / "train"),
        seq_len=2,
        require_release_manifest=True,
    )
    stats = dataset.stats()
    assert stats["n_blocks"] == 3
    assert stats["require_release_manifest"] is True
    assert stats["release_validation"]["expected_tokens"] == 7
    assert len(stats["release_validation"]["manifest_sha256"]) == 64
    assert stats["release_validation"]["reference_exclusion_manifest_sha256s"] == ("a" * 64,)
    assert "shard_paths" not in stats["release_validation"]


def test_reference_release_happy_for_combined_and_per_source(
    tmp_path: Path,
) -> None:
    root, _ = _write_reference_release(tmp_path)

    combined = validate_shard_release(root / "val")
    source = validate_shard_release(root / "val_by_source" / "wiki")

    assert combined["release_kind"] == source["release_kind"] == "reference"
    assert combined["split"] == "val"
    assert source["split"] == "val_by_source"
    assert source["source_name"] == "wiki"
    assert combined["expected_tokens"] == source["expected_tokens"] == 7
    assert len(combined["manifest_sha256"]) == 64
    assert combined["reference_exclusion_manifest_sha256s"] == ()


def test_missing_release_manifest_is_rejected(tmp_path: Path) -> None:
    shard_dir = tmp_path / "missing" / "train"
    _write_shards(shard_dir)

    with pytest.raises(FileNotFoundError, match="no meta.json or manifest.json"):
        validate_shard_release(shard_dir)


@pytest.mark.parametrize("release_kind", ["regular", "reference"])
def test_pre_digest_release_schema_is_rejected(
    tmp_path: Path,
    release_kind: str,
) -> None:
    if release_kind == "regular":
        root, manifest = _write_regular_release(tmp_path)
        path = root / "meta.json"
        shard_dir = root / "train"
        manifest["schema_version"] = 2
    else:
        root, manifest = _write_reference_release(tmp_path)
        path = root / "manifest.json"
        shard_dir = root / "val"
        manifest["schema_version"] = 1
    _rewrite_json(path, manifest)

    with pytest.raises(RuntimeError, match="unsupported .* schema_version"):
        validate_shard_release(shard_dir)


@pytest.mark.parametrize("failure_mode", ["status", "marker"])
def test_failed_release_is_rejected(tmp_path: Path, failure_mode: str) -> None:
    root, meta = _write_regular_release(tmp_path)
    if failure_mode == "status":
        meta["status"] = "failed"
        _rewrite_json(root / "meta.json", meta)
    else:
        _rewrite_json(root / "meta.failed.json", {"status": "failed"})

    with pytest.raises(RuntimeError, match="failed|status"):
        validate_shard_release(root / "train")


def test_legacy_release_is_rejected(tmp_path: Path) -> None:
    root, meta = _write_regular_release(tmp_path)
    meta["legacy_flags"]["replay_on_exhaustion"] = True
    meta["source_exhaustion_policy"] = "legacy_replay"
    _rewrite_json(root / "meta.json", meta)

    with pytest.raises(RuntimeError, match="legacy/debug"):
        validate_shard_release(root / "train")


@pytest.mark.parametrize(
    ("failure_mode", "message"),
    [
        ("disabled", "must enable"),
        ("no_manifests", "must be positive"),
        ("missing_sha", "invalid manifest_sha256"),
    ],
)
def test_missing_or_invalid_reference_exclusion_is_rejected(
    tmp_path: Path,
    failure_mode: str,
    message: str,
) -> None:
    root, meta = _write_regular_release(tmp_path)
    exclusion = meta["reference_validation_exclusion"]
    if failure_mode == "disabled":
        exclusion["enabled"] = False
    elif failure_mode == "no_manifests":
        exclusion["manifest_count"] = 0
        exclusion["union_hash_count"] = 0
        exclusion["manifests"] = []
    else:
        exclusion["manifests"][0].pop("manifest_sha256")
    _rewrite_json(root / "meta.json", meta)

    with pytest.raises(RuntimeError, match=message):
        validate_shard_release(root / "train")


@pytest.mark.parametrize("layout", ["gap", "extra"])
def test_shard_gap_or_extra_is_rejected(tmp_path: Path, layout: str) -> None:
    root, _ = _write_regular_release(tmp_path)
    train = root / "train"
    if layout == "gap":
        (train / "shard_00001.bin").rename(train / "shard_00002.bin")
    else:
        np.asarray([9], dtype=np.uint16).tofile(train / "shard_00002.bin")

    with pytest.raises(RuntimeError, match="filename/count mismatch"):
        validate_shard_release(train)


def test_manifest_count_mismatch_is_rejected(tmp_path: Path) -> None:
    root, meta = _write_regular_release(tmp_path)
    meta["train_shards"] = 3
    meta["train_tokens"] = 12
    meta["accounting"]["train"]["emitted_shard_tokens"] = 12
    _rewrite_json(root / "meta.json", meta)

    with pytest.raises(RuntimeError, match="record count mismatch"):
        validate_shard_release(root / "train")


@pytest.mark.parametrize("corruption", ["odd_byte", "token_length"])
def test_shard_byte_or_token_geometry_mismatch_is_rejected(
    tmp_path: Path,
    corruption: str,
) -> None:
    root, _ = _write_regular_release(tmp_path)
    final_shard = root / "train" / "shard_00001.bin"
    if corruption == "odd_byte":
        with final_shard.open("ab") as handle:
            handle.write(b"\x00")
        message = "not divisible"
    else:
        np.asarray([1, 2], dtype=np.uint16).tofile(final_shard)
        message = "token length mismatch"

    with pytest.raises(RuntimeError, match=message):
        validate_shard_release(root / "train")


def test_same_length_shard_content_tampering_is_rejected(tmp_path: Path) -> None:
    root, _ = _write_regular_release(tmp_path)
    shard = root / "train" / "shard_00000.bin"
    values = np.fromfile(shard, dtype=np.uint16)
    values[1] ^= np.uint16(1)
    values.tofile(shard)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_shard_release(root / "train")


@pytest.mark.parametrize("corruption", ["missing", "extra", "reordered"])
def test_shard_integrity_record_cardinality_and_order_are_enforced(
    tmp_path: Path,
    corruption: str,
) -> None:
    root, meta = _write_regular_release(tmp_path)
    records = meta["shard_files"]["train"]
    if corruption == "missing":
        records.pop()
        message = "record count mismatch"
    elif corruption == "extra":
        records.append(dict(records[-1]))
        message = "record count mismatch"
    else:
        records.reverse()
        message = "order/path mismatch"
    _rewrite_json(root / "meta.json", meta)

    with pytest.raises(RuntimeError, match=message):
        validate_shard_release(root / "train")


def test_manifest_dtype_mismatch_is_rejected(tmp_path: Path) -> None:
    root, meta = _write_regular_release(tmp_path)
    meta["dtype"] = "uint32"
    _rewrite_json(root / "meta.json", meta)

    with pytest.raises(RuntimeError, match="dtype must be 'uint16'"):
        validate_shard_release(root / "train")
