from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from tokenizers import Tokenizer

from pretrain.build_pretrain_shards import (
    CLEANED_TEXT_HASH_ALGORITHM,
    EXCLUSION_MANIFEST_KIND,
    cleaned_text_sha256,
    load_exclusion_hash_manifest,
    write_shards,
)
from pretrain.build_reference_val_shards import (
    EXCLUSION_MANIFEST_NAME,
    MANIFEST_NAME,
    RESERVE_MANIFEST_NAME,
    ReferenceSourceExhaustedError,
    finalize_reference_validation,
    parse_reference_sources,
    reserve_reference_candidates,
)
from pretrain.dataset_pretrain import validate_shard_release
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS
from tokenizer.tokenizer_training import train_tokenizer


def _save_tokenizer(chat_tok: Tokenizer, path: Path) -> Path:
    chat_tok.save(str(path))
    return path


def _write_jsonl(path: Path, texts: list[str]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for value in texts:
            f.write(json.dumps({"text": value}, ensure_ascii=False) + "\n")
    return path


def _read_hashes(path: Path) -> frozenset[str]:
    hashes, _ = load_exclusion_hash_manifest(path)
    return hashes


def _all_bin_tokens(directory: Path) -> np.ndarray:
    arrays = [np.fromfile(path, dtype=np.uint16) for path in sorted(directory.glob("*.bin"))]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.uint16)


def _decoded_documents(tok: Tokenizer, ids: np.ndarray) -> list[str]:
    documents: list[str] = []
    current: list[int] | None = None
    for value in ids.tolist():
        if value == BOS_ID:
            assert current is None
            current = []
        elif value == EOS_ID:
            assert current is not None
            documents.append(tok.decode(current, skip_special_tokens=False))
            current = None
        else:
            assert current is not None
            current.append(int(value))
    assert current is None
    return documents


def _train_kwargs(
    source: Path,
    out_dir: Path,
    tokenizer_path: Path,
    exclusions: list[Path],
) -> dict:
    return {
        "sources": [(source, 1.0)],
        "out_dir": out_dir,
        "tokenizer_path": str(tokenizer_path),
        "shard_tokens": 16,
        "val_shard_tokens": 16,
        "val_ratio": 0.0,
        "min_val_tokens_per_source": 0,
        "seed": 7,
        "validation_hash_seed": 99,
        "add_bos": True,
        "add_eos": True,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "target_train_tokens": 30,
        "precheck_max_lines": 100,
        "doc_sep": "",
        "first_token_topk": 10,
        "strip_leading_noise": False,
        "normalize_quotes": False,
        "underscores_policy": "keep",
        "min_chars": 0,
        "min_ascii_ratio": 0.0,
        "exclude_hash_manifests": exclusions,
    }


def test_two_phase_selection_is_input_order_independent(
    tmp_path: Path, production_chat_tok: Tokenizer
):
    texts = [f"stable reference document number {index}." for index in range(40)]
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    forward = _write_jsonl(tmp_path / "forward.jsonl", texts)
    reverse = _write_jsonl(tmp_path / "reverse.jsonl", list(reversed(texts)))

    outputs: list[tuple[frozenset[str], np.ndarray]] = []
    for label, source in (("a", forward), ("b", reverse)):
        reserve_dir = tmp_path / f"reserve_{label}"
        reserve_reference_candidates(
            sources=parse_reference_sources([f"{source}:20"]),
            out_dir=reserve_dir,
            seed=2025,
            reserve_bytes_per_target_token=10.0,
        )
        final_dir = tmp_path / f"final_{label}"
        manifest = finalize_reference_validation(
            reserve_manifest_path=reserve_dir / RESERVE_MANIFEST_NAME,
            out_dir=final_dir,
            tokenizer_path=str(tokenizer_path),
            shard_tokens=17,
        )
        assert manifest["schema_version"] == 2
        assert manifest["selection"]["restricted_to_pre_tokenizer_reserve"] is True
        provenance = manifest["reserve_provenance"]
        assert (
            provenance["reserve_manifest_size_bytes"]
            == (reserve_dir / RESERVE_MANIFEST_NAME).stat().st_size
        )
        assert (
            provenance["reserve_exclusion"]["manifest_size_bytes"]
            == (reserve_dir / EXCLUSION_MANIFEST_NAME).stat().st_size
        )
        release = validate_shard_release(final_dir / "val")
        assert release["manifest_schema_version"] == 2
        assert release["shard_file_records"]
        outputs.append((
            _read_hashes(final_dir / EXCLUSION_MANIFEST_NAME),
            _all_bin_tokens(final_dir / "val"),
        ))

    assert outputs[0][0] == outputs[1][0]
    assert np.array_equal(outputs[0][1], outputs[1][1])


def test_seven_sources_exact_contract_and_literal_specials(
    tmp_path: Path, production_chat_tok: Tokenizer
):
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    source_args: list[str] = []
    for index in range(7):
        source = _write_jsonl(
            tmp_path / f"domain_{index}.jsonl",
            [f"domain {index}: literal [EOS] and <|user|> are ordinary text"],
        )
        source_args.append(f"{source}:3")

    reserve_dir = tmp_path / "reserve"
    reserve_reference_candidates(
        sources=parse_reference_sources(source_args),
        out_dir=reserve_dir,
        seed=11,
        reserve_bytes_per_target_token=1.0,
    )
    final_dir = tmp_path / "final"
    manifest = finalize_reference_validation(
        reserve_manifest_path=reserve_dir / RESERVE_MANIFEST_NAME,
        out_dir=final_dir,
        tokenizer_path=str(tokenizer_path),
        shard_tokens=19,
    )

    assert len(manifest["sources"]) == 7
    assert len(list((final_dir / "val_by_source").iterdir())) == 7
    assert manifest["contract"]["expected_special_token_ids"] == SPECIAL_TOKEN_IDS
    emitted = _all_bin_tokens(final_dir / "val")
    assert int(np.count_nonzero(emitted == BOS_ID)) == 7
    assert int(np.count_nonzero(emitted == EOS_ID)) == 7
    for role_id in (4, 5, 6):
        assert int(np.count_nonzero(emitted == role_id)) == 0
    assert manifest["accounting"]["boundary_tokens"] == 14
    assert not list(final_dir.rglob("*.tmp"))


def test_reserve_and_finalize_fail_fast_on_insufficient_source(
    tmp_path: Path, production_chat_tok: Tokenizer
):
    source = _write_jsonl(tmp_path / "tiny.jsonl", ["x"])
    with pytest.raises(ReferenceSourceExhaustedError, match="byte quota"):
        reserve_reference_candidates(
            sources=parse_reference_sources([f"{source}:100"]),
            out_dir=tmp_path / "reserve_too_large",
            seed=1,
            reserve_bytes_per_target_token=2.0,
        )
    assert not (tmp_path / "reserve_too_large").exists()

    reserve_dir = tmp_path / "reserve_small"
    reserve_reference_candidates(
        sources=parse_reference_sources([f"{source}:100"]),
        out_dir=reserve_dir,
        seed=1,
        reserve_bytes_per_target_token=0.01,
    )
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    with pytest.raises(ReferenceSourceExhaustedError, match="exhausted before quota"):
        finalize_reference_validation(
            reserve_manifest_path=reserve_dir / RESERVE_MANIFEST_NAME,
            out_dir=tmp_path / "final_too_large",
            tokenizer_path=str(tokenizer_path),
            shard_tokens=10,
        )
    assert not (tmp_path / "final_too_large").exists()


def test_finalizer_rejects_noncanonical_tokenizer(tmp_path: Path, production_chat_tok: Tokenizer):
    source = _write_jsonl(tmp_path / "source.jsonl", ["enough reference text"])
    reserve_dir = tmp_path / "reserve"
    reserve_reference_candidates(
        sources=parse_reference_sources([f"{source}:3"]),
        out_dir=reserve_dir,
        seed=2,
        reserve_bytes_per_target_token=1.0,
    )
    good = _save_tokenizer(production_chat_tok, tmp_path / "good.json")
    payload = json.loads(good.read_text(encoding="utf-8"))
    role = next(item for item in payload["added_tokens"] if item["content"] == "<|user|>")
    role["id"] = 99
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-canonical pretraining build refused"):
        finalize_reference_validation(
            reserve_manifest_path=reserve_dir / RESERVE_MANIFEST_NAME,
            out_dir=tmp_path / "final",
            tokenizer_path=str(bad),
            shard_tokens=10,
        )
    assert not (tmp_path / "final").exists()


def test_repeatable_exclusions_never_leak_into_train_or_ordinary_val(
    tmp_path: Path, production_chat_tok: Tokenizer
):
    tokenizer_path = _save_tokenizer(production_chat_tok, tmp_path / "tokenizer.json")
    original = [f"reserved candidate {index}" for index in range(20)]
    reference_source = _write_jsonl(tmp_path / "reference.jsonl", original)
    reserve_dir = tmp_path / "reserve"
    reserve_reference_candidates(
        sources=parse_reference_sources([f"{reference_source}:5"]),
        out_dir=reserve_dir,
        seed=9,
        reserve_bytes_per_target_token=2.0,
    )
    reserve_exclusion = reserve_dir / EXCLUSION_MANIFEST_NAME
    reserved_hashes = _read_hashes(reserve_exclusion)
    reserved_texts = [value for value in original if cleaned_text_sha256(value) in reserved_hashes]

    second_text = "independently excluded document"
    cleaning = json.loads(reserve_exclusion.read_text(encoding="utf-8"))["cleaning"]
    second_manifest = tmp_path / "second_exclusion.json"
    second_manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
            "cleaning": cleaning,
            "hash_count": 1,
            "hashes": [cleaned_text_sha256(second_text)],
        }),
        encoding="utf-8",
    )
    fresh = [f"fresh train document {index}" for index in range(100)]
    train_source = _write_jsonl(tmp_path / "train.jsonl", reserved_texts + [second_text] + fresh)
    out_dir = tmp_path / "train_shards"
    kwargs = _train_kwargs(
        train_source,
        out_dir,
        tokenizer_path,
        [reserve_exclusion, second_manifest],
    )
    kwargs["val_ratio"] = 0.2
    meta = write_shards(**kwargs)

    release = validate_shard_release(out_dir / "train")
    assert meta["schema_version"] == 3
    assert release["manifest_schema_version"] == 3
    assert release["shard_file_records"]

    exclusion_meta = meta["reference_validation_exclusion"]
    assert exclusion_meta["manifest_count"] == 2
    assert all(item["matched_documents"] >= 1 for item in exclusion_meta["manifests"])
    documents = _decoded_documents(production_chat_tok, _all_bin_tokens(out_dir / "train"))
    excluded_union = reserved_hashes | {cleaned_text_sha256(second_text)}
    assert documents
    assert not {cleaned_text_sha256(value) for value in documents} & excluded_union
    val_documents = _decoded_documents(production_chat_tok, _all_bin_tokens(out_dir / "val"))
    assert val_documents
    assert not {cleaned_text_sha256(value) for value in val_documents} & excluded_union


def test_tokenizer_training_excludes_reserve_and_records_release_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    held_out = "this exact document is reserved"
    kept = [f"tokenizer training document {index} with enough variation" for index in range(20)]
    data = _write_jsonl(tmp_path / "tokenizer_data.jsonl", [held_out, *kept])
    cleaning = {
        "strip_leading_noise": False,
        "normalize_quotes": False,
        "underscores_policy": "keep",
        "min_chars": 0,
        "min_ascii_ratio": 0.0,
    }
    exclusion = tmp_path / "exclusion.json"
    exclusion.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
            "cleaning": cleaning,
            "hash_count": 1,
            "hashes": [cleaned_text_sha256(held_out)],
        }),
        encoding="utf-8",
    )
    out_dir = tmp_path / "tokenizer_release"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--data",
            str(data),
            "--out_dir",
            str(out_dir),
            "--vocab_size",
            "300",
            "--min_freq",
            "1",
            "--legacy_allow_noncanonical_contract",
            "--exclude_hash_manifest",
            str(exclusion),
        ],
    )
    train_tokenizer.main()

    release = json.loads((out_dir / "tokenizer_release_manifest.json").read_text(encoding="utf-8"))
    audit = release["reference_reserve_exclusion"]
    assert audit["considered_samples"] == 21
    assert audit["excluded_samples"] == 1
    assert audit["yielded_samples"] == 20
    assert audit["manifests"][0]["matched_samples"] == 1
    assert audit["manifests"][0]["manifest_sha256"]
    assert audit["manifests"][0]["manifest_size_bytes"] == exclusion.stat().st_size
    assert release["special_token_ids"] == SPECIAL_TOKEN_IDS
    assert (out_dir / "tokenizer.json").is_file()
    assert (out_dir / MANIFEST_NAME).exists() is False
    tokenizer_config = json.loads((out_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "chat_template" not in tokenizer_config
    assert not hasattr(train_tokenizer, "HF_CHAT_TEMPLATE")


def test_tokenizer_training_refuses_noncanonical_release_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = _write_jsonl(tmp_path / "data.jsonl", ["enough text for the parser"])
    out_dir = tmp_path / "bad_release"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--data",
            str(data),
            "--out_dir",
            str(out_dir),
            "--vocab_size",
            "300",
        ],
    )

    with pytest.raises(SystemExit, match="noncanonical tokenizer release refused"):
        train_tokenizer.main()
    assert not out_dir.exists()


def test_tokenizer_training_refuses_existing_output_without_touching_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = _write_jsonl(tmp_path / "data.jsonl", ["enough text for the parser"])
    out_dir = tmp_path / "existing_release"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--data",
            str(data),
            "--out_dir",
            str(out_dir),
            "--vocab_size",
            "300",
            "--legacy_allow_noncanonical_contract",
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to replace existing"):
        train_tokenizer.main()
    assert sentinel.read_text(encoding="utf-8") == "user data"
