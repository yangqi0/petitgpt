from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from pretrain import (
    sanity_check_pretrain_jsonl as jsonl_check,
    sanity_check_pretrain_shards as shard_check,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS


def _save_tokenizer(tokenizer: Tokenizer, path: Path) -> Path:
    tokenizer.save(str(path))
    return path


def _canonical_contract() -> dict:
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


def _write_release(tmp_path: Path, tokenizer_path: Path) -> Path:
    root = tmp_path / "release"
    train = root / "train"
    val = root / "val"
    train.mkdir(parents=True)
    val.mkdir()
    np.asarray([BOS_ID, 7, 8, EOS_ID, BOS_ID], dtype=np.uint16).tofile(
        train / "shard_00000.bin"
    )
    np.asarray([9, 10, EOS_ID, BOS_ID, EOS_ID], dtype=np.uint16).tofile(
        train / "shard_00001.bin"
    )
    train_records = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "token_count": path.stat().st_size // np.dtype(np.uint16).itemsize,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(train.glob("*.bin"))
    ]
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    meta = {
        "schema_version": 3,
        "status": "complete",
        "dtype": "uint16",
        "vocab_size": 32_000,
        "tokenizer_sha256": tokenizer_sha256,
        "contract": _canonical_contract(),
        "legacy_flags": {
            "allow_noncanonical_contract": False,
            "replay_on_exhaustion": False,
        },
        "source_exhaustion_policy": "fail_fast",
        "reference_validation_exclusion": {
            "enabled": True,
            "manifest_count": 1,
            "union_hash_count": 1,
            "manifests": [
                {
                    "enabled": True,
                    "manifest_sha256": "a" * 64,
                    "hash_count": 1,
                }
            ],
        },
        "shard_tokens": 5,
        "val_shard_tokens": 5,
        "train_shards": 2,
        "train_tokens": 10,
        "val_shards": 0,
        "val_tokens": 0,
        "train_content_tokens": 8,
        "val_content_tokens": 0,
        "accounting": {
            "train": {
                "content_tokens": 8,
                "boundary_tokens": 2,
                "serialized_tokens": 10,
                "emitted_shard_tokens": 10,
            },
            "val": {
                "content_tokens": 0,
                "boundary_tokens": 0,
                "serialized_tokens": 0,
                "emitted_shard_tokens": 0,
            },
        },
        "per_source": {
            "alpha": {
                "weight": 1.0,
                "realized": {
                    "train": {"serialized_tokens": 6},
                    "val": {"serialized_tokens": 0},
                },
            },
            "beta": {
                "weight": 1.0,
                "realized": {
                    "train": {"serialized_tokens": 4},
                    "val": {"serialized_tokens": 0},
                },
            },
        },
        "shard_files": {
            "hash_algorithm": "sha256",
            "train": train_records,
            "val": [],
            "val_by_source": {},
        },
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def _write_jsonl(path: Path, rows: list[object]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row) + "\n")
    return path


def test_jsonl_max_lines_reservoir_and_atomic_report(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [{"text": f"row-{index}"} for index in range(20)],
    )
    report_path = tmp_path / "reports" / "jsonl.json"

    first = jsonl_check.main(
        [
            "--jsonl",
            str(source),
            "--max_lines",
            "1",
            "--sample",
            "1",
            "--seed",
            "17",
            "--out_json",
            str(report_path),
        ]
    )
    assert first["seen_lines"] == 1
    assert first["valid"] == 1
    assert first["sample_line_numbers"] == [1]
    assert json.loads(report_path.read_text(encoding="utf-8")) == first
    assert not list(report_path.parent.glob(".*.tmp"))

    args = [
        "--jsonl",
        str(source),
        "--max_lines",
        "0",
        "--sample",
        "4",
        "--seed",
        "17",
    ]
    full_a = jsonl_check.main(args)
    full_b = jsonl_check.main(args)
    assert full_a["sample_line_numbers"] == full_b["sample_line_numbers"]
    assert any(line_number > 4 for line_number in full_a["sample_line_numbers"])


def test_jsonl_counts_bad_records_and_literal_special_roundtrip(
    tmp_path: Path, production_chat_tok: Tokenizer
) -> None:
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    source = _write_jsonl(
        tmp_path / "candidate.jsonl",
        [
            "{bad json",
            {"other": "missing"},
            {"text": 123},
            {"text": "literal <|user|> text\n    kept"},
        ],
    )
    report = jsonl_check.main(
        [
            "--jsonl",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--sample",
            "10",
            "--roundtrip_samples",
            "10",
        ]
    )
    assert report["bad_json"] == 1
    assert report["missing_field"] == 1
    assert report["non_str"] == 1
    assert report["valid"] == 1
    assert report["roundtrip_checked"] == 1
    assert report["roundtrip_ok"] == 1
    assert report["roundtrip_ok_frac"] == 1.0


def test_shard_checker_uses_global_blocks_and_serialized_units(
    tmp_path: Path, production_chat_tok: Tokenizer
) -> None:
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    release = _write_release(tmp_path, tokenizer_path)
    report_path = tmp_path / "reports" / "shards.json"
    args = [
        "--out_dir",
        str(release),
        "--tokenizer_path",
        str(tokenizer_path),
        "--split",
        "both",
        "--seq_len",
        "4",
        "--sample_shards",
        "1",
        "--blocks_per_shard",
        "2",
        "--seed",
        "9",
        "--out_json",
        str(report_path),
    ]
    first = shard_check.main(args)
    second = shard_check.main(args)
    assert first == second
    assert json.loads(report_path.read_text(encoding="utf-8")) == first
    assert first["summary"]["meta_train_serialized_tokens"] == 10
    assert first["summary"]["meta_train_content_tokens"] == 8
    sources = {row["source"]: row for row in first["summary"]["sources"]}
    assert sources["alpha"]["train_serialized_fraction"] == pytest.approx(0.6)
    assert sources["beta"]["train_serialized_fraction"] == pytest.approx(0.4)

    train_report, val_report = first["split_reports"]
    assert train_report["sampled_global_block_ids"] == [0, 1]
    assert train_report["blocks_sampled"] == 2
    assert train_report["cross_shard_blocks_sampled"] == 1
    assert train_report["token_ids_checked"] == 10
    assert val_report["blocks_sampled"] == 0
    assert val_report["n_shards"] == 0
    assert not list(report_path.parent.glob(".*.tmp"))


def test_shard_checker_rejects_tokenizer_and_contract_drift(
    tmp_path: Path, production_chat_tok: Tokenizer
) -> None:
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    release = _write_release(tmp_path, tokenizer_path)
    meta_path = release / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tokenizer_sha256"] = "b" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(RuntimeError, match="tokenizer SHA-256"):
        shard_check.main(
            [
                "--out_dir",
                str(release),
                "--tokenizer_path",
                str(tokenizer_path),
                "--split",
                "train",
                "--seq_len",
                "4",
            ]
        )

    meta["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    meta["dtype"] = "uint32"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dtype"):
        shard_check.main(
            [
                "--out_dir",
                str(release),
                "--tokenizer_path",
                str(tokenizer_path),
                "--split",
                "train",
                "--seq_len",
                "4",
            ]
        )
