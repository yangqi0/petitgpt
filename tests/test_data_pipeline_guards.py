from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

import pretrain.build_pretrain_shards as shard_builder
from pretrain.build_pretrain_shards import (
    SourceExhaustedError,
    is_validation_holdout,
    validate_build_contract,
    write_shards,
)
from pretrain.data_preparation_tools.extract_smollm_python_edu import (
    decode_blob_bytes,
    extract_dataset_records,
    fetch_blob_cached,
)
from src.special_tokens import BOS_ID, EOS_ID


def _save_tokenizer(chat_tok: Tokenizer, path: Path) -> Path:
    chat_tok.save(str(path))
    return path


def _write_jsonl(path: Path, texts: list[str]) -> Path:
    with open(path, "w", encoding="utf-8") as fout:
        for text in texts:
            fout.write(json.dumps({"text": text}) + "\n")
    return path


def _build_kwargs(
    *,
    source: Path,
    out_dir: Path,
    tokenizer_path: Path,
    target_train_tokens: int,
    legacy_replay_on_exhaustion: bool = False,
) -> dict:
    return {
        "sources": [(source, 1.0)],
        "out_dir": out_dir,
        "tokenizer_path": str(tokenizer_path),
        "shard_tokens": 4,
        "val_shard_tokens": 4,
        "val_ratio": 0.0,
        "min_val_tokens_per_source": 0,
        "seed": 7,
        "validation_hash_seed": 99,
        "add_bos": True,
        "add_eos": True,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "target_train_tokens": target_train_tokens,
        "precheck_max_lines": 100,
        "doc_sep": "",
        "first_token_topk": 10,
        "strip_leading_noise": False,
        "normalize_quotes": False,
        "underscores_policy": "keep",
        "min_chars": 0,
        "min_ascii_ratio": 0.0,
        "legacy_allow_noncanonical_contract": False,
        "legacy_replay_on_exhaustion": legacy_replay_on_exhaustion,
    }


def test_canonical_contract_rejects_overrides_without_legacy_flag(
    tmp_path: Path,
    chat_tok: Tokenizer,
    production_chat_tok: Tokenizer,
):
    tiny_tokenizer_path = _save_tokenizer(chat_tok, tmp_path / "tiny_tokenizer.json")
    with pytest.raises(ValueError, match=r"runtime vocab IDs.*vocab_size=32000"):
        validate_build_contract(
            tokenizer_path=str(tiny_tokenizer_path),
            add_bos=True,
            add_eos=True,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            doc_sep="",
        )
    tiny_legacy = validate_build_contract(
        tokenizer_path=str(tiny_tokenizer_path),
        add_bos=True,
        add_eos=True,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        doc_sep="",
        legacy_allow_noncanonical_contract=True,
    )
    assert tiny_legacy["canonical"] is False
    assert tiny_legacy["actual_vocab_size"] != 32_000

    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    canonical = {
        "tokenizer_path": str(tokenizer_path),
        "add_bos": True,
        "add_eos": True,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "doc_sep": "",
    }
    result = validate_build_contract(**canonical)
    assert result["canonical"] is True
    assert result["mode"] == "canonical"

    for override in (
        {"add_bos": False},
        {"add_eos": False},
        {"bos_id": 123},
        {"eos_id": 123},
        {"doc_sep": "\n\n"},
    ):
        with pytest.raises(ValueError, match="non-canonical pretraining build refused"):
            validate_build_contract(**{**canonical, **override})

    legacy = validate_build_contract(
        **{**canonical, "doc_sep": "\n\n"},
        legacy_allow_noncanonical_contract=True,
    )
    assert legacy["canonical"] is False
    assert legacy["mode"] == "legacy_noncanonical"
    assert legacy["legacy_allow_noncanonical_contract"] is True
    assert legacy["issues"]


def test_validation_holdout_is_stable_and_order_independent():
    texts = [f"document-{index}" for index in range(256)]
    forward = {
        text: is_validation_holdout(
            text,
            train_target_tokens=3,
            val_target_tokens=1,
            seed=2025,
        )
        for text in texts
    }
    reverse = {
        text: is_validation_holdout(
            text,
            train_target_tokens=3,
            val_target_tokens=1,
            seed=2025,
        )
        for text in reversed(texts)
    }
    assert forward == reverse
    assert any(forward.values())
    assert not all(forward.values())
    assert forward == {
        text: is_validation_holdout(
            text,
            train_target_tokens=3,
            val_target_tokens=1,
            seed=2025,
        )
        for text in texts
    }


def test_source_exhaustion_fails_fast_and_writes_failure_manifest(
    tmp_path: Path, production_chat_tok: Tokenizer
):
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    source = _write_jsonl(tmp_path / "source.jsonl", ["x"])
    out_dir = tmp_path / "failed"

    with pytest.raises(SourceExhaustedError, match="source exhausted before quota"):
        write_shards(
            **_build_kwargs(
                source=source,
                out_dir=out_dir,
                tokenizer_path=tokenizer_path,
                target_train_tokens=10,
            )
        )

    assert not out_dir.exists()
    failures = list(tmp_path.glob("failed.failed-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    source_meta = failure["per_source"][str(source)]
    assert failure["status"] == "failed"
    assert failure["source_exhaustion_policy"] == "fail_fast"
    assert failure["production_directory_published"] is False
    assert source_meta["exhaustion"]["exhausted_before_quota"] is True
    assert source_meta["exhaustion"]["remaining_train_serialized_tokens"] > 0
    assert not list(tmp_path.glob(".failed.building-*"))


@pytest.mark.parametrize("with_stale_file", [False, True])
def test_existing_output_directory_is_always_refused_without_mutation(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    with_stale_file: bool,
):
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    source = _write_jsonl(tmp_path / "source.jsonl", ["safe document"])
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    stale = out_dir / "stale.bin"
    if with_stale_file:
        stale.write_bytes(b"do-not-touch")

    with pytest.raises(FileExistsError, match="existing shard output path"):
        write_shards(
            **_build_kwargs(
                source=source,
                out_dir=out_dir,
                tokenizer_path=tokenizer_path,
                target_train_tokens=3,
            )
        )

    assert out_dir.is_dir()
    if with_stale_file:
        assert stale.read_bytes() == b"do-not-touch"
    else:
        assert not stale.exists()
    assert not list(tmp_path.glob(".existing.building-*"))
    assert not list(tmp_path.glob("existing.failed-*.json"))


def test_legacy_replay_is_explicit_and_accounting_includes_boundaries(
    tmp_path: Path, production_chat_tok: Tokenizer
):
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    source = _write_jsonl(tmp_path / "source.jsonl", ["x"])
    out_dir = tmp_path / "replay"
    meta = write_shards(
        **_build_kwargs(
            source=source,
            out_dir=out_dir,
            tokenizer_path=tokenizer_path,
            target_train_tokens=10,
            legacy_replay_on_exhaustion=True,
        )
    )

    train = meta["accounting"]["train"]
    source_meta = meta["per_source"][str(source)]
    assert meta["source_exhaustion_policy"] == "legacy_replay"
    assert meta["legacy_flags"]["replay_on_exhaustion"] is True
    assert source_meta["exhaustion"]["replay_count"] > 0
    assert train["content_tokens"] + train["boundary_tokens"] == train["serialized_tokens"]
    assert train["boundary_tokens"] == train["documents"] * 2
    assert train["separator_tokens"] == 0
    assert train["serialized_tokens"] == meta["train_serialized_tokens"]
    assert train["emitted_shard_tokens"] == meta["train_tokens"]
    assert source_meta["targets"]["train_serialized_tokens"] == 10
    assert source_meta["realized"]["train"]["serialized_tokens"] >= 10
    fingerprint = meta["source_fingerprints"][str(source)]
    assert fingerprint["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert fingerprint["size"] == source.stat().st_size
    assert meta["tokenizer_sha256"] == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert (out_dir / "meta.json").is_file()
    assert (out_dir / "train").is_dir()
    assert not list(tmp_path.glob(".replay.building-*"))

    emitted = np.concatenate([
        np.fromfile(path, dtype=np.uint16) for path in sorted((out_dir / "train").glob("*.bin"))
    ])
    assert int(np.count_nonzero(emitted == BOS_ID)) == train["documents"]
    assert int(np.count_nonzero(emitted == EOS_ID)) == train["documents"]


def test_source_snapshot_change_aborts_atomic_publication(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    monkeypatch,
):
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    source = _write_jsonl(tmp_path / "source.jsonl", ["safe document"])
    out_dir = tmp_path / "mutated"
    original_fingerprint = shard_builder.file_fingerprint
    source_calls = 0

    def changed_on_post_scan(path: Path) -> dict:
        nonlocal source_calls
        result = original_fingerprint(path)
        if path.resolve() == source.resolve():
            source_calls += 1
            if source_calls == 2:
                result["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(shard_builder, "file_fingerprint", changed_on_post_scan)
    with pytest.raises(RuntimeError, match="changed between the pre-scan and post-scan"):
        write_shards(
            **_build_kwargs(
                source=source,
                out_dir=out_dir,
                tokenizer_path=tokenizer_path,
                target_train_tokens=3,
            )
        )

    assert source_calls == 2
    assert not out_dir.exists()
    failures = list(tmp_path.glob("mutated.failed-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["production_directory_published"] is False
    assert (
        failure["source_fingerprints"][str(source)]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_python_blob_decoder_is_strict_utf8():
    with pytest.raises(UnicodeDecodeError, match="blob_id=bad-blob"):
        decode_blob_bytes(b"valid\xffinvalid", blob_id="bad-blob")


def test_python_blob_cache_is_atomic_and_reused(tmp_path: Path):
    calls: list[str] = []

    def fetcher(blob_id: str) -> bytes:
        calls.append(blob_id)
        return f"payload:{blob_id}".encode()

    cache_dir = tmp_path / "cache"
    first, first_hit = fetch_blob_cached("blob-a", fetcher, cache_dir)
    second, second_hit = fetch_blob_cached("blob-a", fetcher, cache_dir)
    assert first == second == b"payload:blob-a"
    assert first_hit is False
    assert second_hit is True
    assert calls == ["blob-a"]
    assert not list(cache_dir.rglob("*.tmp"))


def test_python_extractor_resumes_by_truncating_uncheckpointed_output(
    tmp_path: Path,
):
    records = [
        {"blob_id": "a", "int_score": 4, "score": 1.0},
        {"blob_id": "b", "int_score": 4, "score": 1.0},
        {"blob_id": "c", "int_score": 4, "score": 1.0},
    ]
    payloads = {
        "a": b"  alpha\n",
        "b": b"beta",
        "c": b"gamma",
    }
    output = tmp_path / "python.jsonl"
    cache = tmp_path / "cache"
    config = {"dataset_fingerprint": "fixture-v1"}

    def interrupted_fetch(blob_id: str) -> bytes:
        if blob_id == "b":
            raise KeyboardInterrupt
        return payloads[blob_id]

    with pytest.raises(KeyboardInterrupt):
        extract_dataset_records(
            records,
            fetcher=interrupted_fetch,
            output_path=output,
            cache_dir=cache,
            config=config,
            min_int_score=4,
            min_chars=1,
            max_chars=100,
            checkpoint_every=2,
        )

    partial = Path(str(output) + ".partial")
    state = Path(str(output) + ".state.json")
    assert not output.exists()
    assert partial.exists() and partial.stat().st_size > 0
    assert json.loads(state.read_text(encoding="utf-8"))["output_bytes"] == 0

    metadata = extract_dataset_records(
        records,
        fetcher=lambda blob_id: payloads[blob_id],
        output_path=output,
        cache_dir=cache,
        config=config,
        min_int_score=4,
        min_chars=1,
        max_chars=100,
        checkpoint_every=2,
    )

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [line["meta"]["blob_id"] for line in lines] == ["a", "b", "c"]
    assert lines[0]["text"] == "  alpha\n"
    assert metadata["stats"]["total_seen"] == 3
    assert metadata["stats"]["kept"] == 3
    assert metadata["stats"]["cache_hits"] == 1
    assert not partial.exists()
    assert not state.exists()
    assert Path(str(output) + ".meta.json").exists()
