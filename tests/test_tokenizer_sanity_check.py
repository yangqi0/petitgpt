from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tokenizer.tokenizer_training import sanity_check_tokenizer as checker


def _save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    tokenizer.save(str(path))


def test_default_run_uses_full_contract_and_writes_auditable_atomic_report(
    production_chat_tok: Tokenizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    corpus_path = tmp_path / "domain.jsonl"
    report_path = tmp_path / "release" / "tokenizer_report.json"
    _save_tokenizer(production_chat_tok, tokenizer_path)
    corpus_path.write_text(
        "".join(
            json.dumps({"text": f"sample {index} with unicode café"}) + "\n" for index in range(12)
        ),
        encoding="utf-8",
    )

    real_assert = checker.assert_tokenizer_contract
    checked_paths: list[Path] = []

    def record_contract_check(path):
        checked_paths.append(Path(path))
        real_assert(path)

    monkeypatch.setattr(checker, "assert_tokenizer_contract", record_contract_check)
    report = checker.main([
        "--tokenizer",
        str(tokenizer_path),
        "--jsonl",
        str(corpus_path),
        "--n_samples",
        "4",
        "--seed",
        "17",
        "--report_json",
        str(report_path),
    ])

    assert checked_paths == [tokenizer_path]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["tokenizer_sha256"] == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert report["production_ready"] is True
    assert report["sampling"]["seed"] == 17
    assert report["contract"]["production_contract_ok"] is True
    assert report["contract"]["requirements"]["vocab_size"] == 32_000
    assert report["contract"]["observed"]["vocab_size"] == 32_000
    assert report["contract"]["observed"]["normalizer"] is None
    assert report["contract"]["observed"]["post_processor"] is None
    assert report["contract"]["observed"]["pre_tokenizer"]["type"] == "ByteLevel"
    assert report["contract"]["observed"]["pre_tokenizer"]["add_prefix_space"] is False
    assert report["contract"]["observed"]["decoder"]["type"] == "ByteLevel"
    per_file = report["compression"][str(corpus_path)]
    assert per_file["n_samples"] == 4
    assert isinstance(per_file["sample_seed"], int)
    assert len(per_file["sample_sha256"]) == 64
    assert not list(report_path.parent.glob(f".{report_path.name}.*.tmp"))


def test_legacy_bypass_is_explicitly_non_production_and_reported(
    chat_tok: Tokenizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    tokenizer_path = tmp_path / "tiny.json"
    report_path = tmp_path / "legacy.json"
    _save_tokenizer(chat_tok, tokenizer_path)

    def production_check_must_not_run(_path):
        raise AssertionError("legacy bypass unexpectedly called the production check")

    monkeypatch.setattr(checker, "assert_tokenizer_contract", production_check_must_not_run)
    report = checker.main([
        "--tokenizer",
        str(tokenizer_path),
        "--no-strict_special_ids",
        "--report_json",
        str(report_path),
    ])

    assert "LEGACY ONLY" in capsys.readouterr().err
    assert report["production_ready"] is False
    assert report["strict_special_ids_ok"] is False
    assert report["contract"]["legacy_contract_bypass"] is True
    assert report["contract"]["production_contract_ok"] is False
    assert report["legacy_warning"] == checker.LEGACY_CONTRACT_WARNING
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_default_mode_rejects_noncanonical_vocab(chat_tok: Tokenizer, tmp_path: Path):
    tokenizer_path = tmp_path / "tiny.json"
    _save_tokenizer(chat_tok, tokenizer_path)

    with pytest.raises(ValueError, match="vocab_size=32000"):
        checker.main(["--tokenizer", str(tokenizer_path)])


def test_raw_text_encode_guard_and_per_file_sampling_are_deterministic(
    chat_tok: Tokenizer,
    tmp_path: Path,
):
    tokenizer_path = tmp_path / "tiny.json"
    _save_tokenizer(chat_tok, tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.encode_special_tokens = False

    with pytest.raises(RuntimeError, match="encode_special_tokens=True"):
        checker.strict_roundtrip(tokenizer, "literal [EOS]")

    texts = [f"document {index}" for index in range(30)]
    first, first_seed = checker._sample_texts(
        texts,
        limit=8,
        base_seed=91,
        path=str(tmp_path / "source.jsonl"),
    )
    second, second_seed = checker._sample_texts(
        texts,
        limit=8,
        base_seed=91,
        path=str(tmp_path / "source.jsonl"),
    )
    _, different_seed = checker._sample_texts(
        texts,
        limit=8,
        base_seed=92,
        path=str(tmp_path / "source.jsonl"),
    )

    assert first == second
    assert first_seed == second_seed
    assert different_seed != first_seed
    assert len(first) == len(set(first)) == 8


def test_atomic_report_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(checker.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        checker._atomic_write_json(report_path, {"new": True})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob(f".{report_path.name}.*.tmp"))
