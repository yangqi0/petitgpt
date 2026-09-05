"""Tests for the scalable, deterministic upstream document selector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from tokenizers import Tokenizer

from pretrain import select_pretrain_documents as selector
from src.special_tokens import SPECIAL_TOKEN_IDS


def _write_tokenizer(path: Path, tokenizer: Tokenizer) -> Path:
    path.write_text(tokenizer.to_str(), encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    with open(path, "w", encoding="utf-8") as output:
        for record in records:
            json.dump(record, output, ensure_ascii=False, sort_keys=True)
            output.write("\n")
    return path


def _write_spec(
    path: Path,
    sources: list[dict[str, object]],
    *,
    seed: int = 20250814,
    cleaning: dict[str, object] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "seed": seed,
        "text_field": "text",
        "cleaning": cleaning or {},
        "sources": sources,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_reference_exclusion(
    path: Path,
    texts: list[str] | None = None,
    *,
    cleaning: dict[str, object] | None = None,
) -> Path:
    cleaning_spec = selector.CleaningSpec(**(cleaning or {}))
    hashes: list[str] = []
    for text in texts or ["reference-only document outside all candidate pools"]:
        cleaned, disposition = selector._clean_for_selection(text, cleaning_spec)
        assert disposition == "eligible"
        assert cleaned is not None
        hashes.append(hashlib.sha256(cleaned.encode("utf-8")).hexdigest())
    hashes = sorted(set(hashes))
    payload = {
        "schema_version": 1,
        "kind": selector.EXCLUSION_MANIFEST_KIND,
        "hash_algorithm": selector.CLEANED_TEXT_HASH_ALGORITHM,
        "cleaning": cleaning_spec.as_dict(),
        "hash_count": len(hashes),
        "hashes": hashes,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def reference_exclusion(tmp_path: Path) -> Path:
    return _write_reference_exclusion(tmp_path / "reference_exclusion.json")


def _source(
    *,
    stage: str,
    source_id: str,
    path: Path,
    target: int = 1,
) -> dict[str, object]:
    return {
        "stage": stage,
        "source_id": source_id,
        "path": str(path),
        "target_serialized_tokens": target,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _serialized_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids) + 2


def test_seven_sources_stage_priority_and_exact_hash_audit(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    reference_text = "immutable reference validation document"
    first_reference_exclusion = _write_reference_exclusion(
        tmp_path / "reference_one.json",
        [reference_text],
    )
    second_reference_exclusion = _write_reference_exclusion(
        tmp_path / "reference_two.json",
        [reference_text, "another reserved reference document"],
    )
    shared = "the exact shared premium document"
    source_records = {
        "b0_premium": [{"text": reference_text}, {"text": shared}],
        "b1_other": [{"text": "premium source two"}],
        "a0_foundation": [{"text": shared}, {"text": "foundation replacement"}],
        "a1_python": [{"text": "def selected_function():\n    return 7"}],
        "control_general": [{"text": shared}, {"text": "general control replacement"}],
        "control_python": [{"text": "literal <|assistant|> and [EOS] stay ordinary text"}],
        "control_stop": [{"text": "a natural ending diagnostic"}],
    }
    stages = {
        "b0_premium": "stage_b",
        "b1_other": "stage_b",
        "a0_foundation": "stage_a",
        "a1_python": "stage_a",
        "control_general": "control",
        "control_python": "control",
        "control_stop": "control",
    }
    inputs: dict[str, Path] = {}
    for source_id, records in source_records.items():
        inputs[source_id] = _write_jsonl(tmp_path / f"{source_id}.jsonl", records)

    # Deliberately scramble the spec: fixed stage priority and source IDs govern selection.
    spec_sources = [
        _source(stage=stages[source_id], source_id=source_id, path=inputs[source_id])
        for source_id in reversed(list(source_records))
    ]
    spec_path = _write_spec(tmp_path / "selection_spec.json", spec_sources)
    out_dir = tmp_path / "selected_registry"
    manifest = selector.build_selection_registry(
        spec_path=spec_path,
        tokenizer_path=tokenizer_path,
        out_dir=out_dir,
        exclude_hash_manifests=(first_reference_exclusion, second_reference_exclusion),
        sqlite_cache_mb=1,
        commit_every=1,
    )

    expected_order = [
        "b0_premium",
        "b1_other",
        "a0_foundation",
        "a1_python",
        "control_general",
        "control_python",
        "control_stop",
    ]
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "complete"
    assert manifest["priority_order"] == ["stage_b", "stage_a", "control"]
    assert manifest["source_order"] == expected_order
    assert manifest["near_deduplication"]["required_upstream"] is True
    assert manifest["bounded_memory"]["python_identity_set"] is False
    assert manifest["database"]["selected_document_rows"] == 7
    assert manifest["database"]["uncommitted_candidate_rows"] == 0
    assert (out_dir / "manifest.json").is_file()
    database_path = out_dir / "selection_registry.sqlite"
    assert _sha256(database_path) == manifest["database"]["sha256"]
    assert database_path.stat().st_size == manifest["database"]["size_bytes"]
    assert tokenizer_path.stat().st_size == manifest["tokenizer"]["size_bytes"]
    assert spec_path.stat().st_size == manifest["spec"]["size_bytes"]

    all_metadata: list[dict[str, object]] = []
    selected_text: dict[str, str] = {}
    for source_entry in manifest["sources"]:
        source_id = source_entry["source_id"]
        assert source_entry["input_sha256"] == _sha256(inputs[source_id])
        selected_path = out_dir / source_entry["output"]["relative_path"]
        records = _read_jsonl(selected_path)
        assert len(records) == 1
        selected_text[source_id] = str(records[0]["text"])
        metadata = records[0][selector.SELECTION_METADATA_FIELD]
        assert metadata["source_id"] == source_id
        assert metadata["stage"] == stages[source_id]
        assert metadata["boundary_tokens"] == 2
        assert metadata["serialized_tokens"] == metadata["content_tokens"] + 2
        all_metadata.append(metadata)
        assert _sha256(selected_path) == source_entry["output"]["sha256"]

    assert selected_text["b0_premium"] == shared
    assert selected_text["a0_foundation"] == "foundation replacement"
    assert selected_text["control_general"] == "general control replacement"
    by_source = {entry["source_id"]: entry for entry in manifest["sources"]}
    assert by_source["a0_foundation"]["exact_hash_exclusions_by_owner"] == {"b0_premium": 1}
    assert by_source["control_general"]["exact_hash_exclusions_by_owner"] == {"b0_premium": 1}
    assert by_source["b0_premium"]["scan"]["excluded_reference_validation"] == 1
    reference_summary = manifest["reference_validation_exclusion"]
    assert reference_summary["required"] is True
    assert reference_summary["manifest_count"] == 2
    assert reference_summary["union_hash_count"] == 2
    assert reference_summary["cross_manifest_duplicate_memberships"] == 1
    assert reference_summary["selected_reference_intersection"] == 0
    assert reference_summary["intersection_zero"] is True
    assert [entry["sha256"] for entry in reference_summary["manifests"]] == [
        _sha256(first_reference_exclusion),
        _sha256(second_reference_exclusion),
    ]
    assert [entry["size_bytes"] for entry in reference_summary["manifests"]] == [
        first_reference_exclusion.stat().st_size,
        second_reference_exclusion.stat().st_size,
    ]
    assert all(
        entry["matched_per_source"] == {"b0_premium": 1} for entry in reference_summary["manifests"]
    )

    for field in ("raw_sha256", "cleaned_sha256", "canonical_fingerprint"):
        values = [str(metadata[field]) for metadata in all_metadata]
        assert len(values) == len(set(values))

    audit_path = out_dir / "audit_exact_intersections.json"
    audit = json.loads(audit_path.read_text())
    assert _sha256(audit_path) == manifest["audit"]["sha256"]
    assert audit_path.stat().st_size == manifest["audit"]["size_bytes"]
    assert audit["status"] == "passed"
    assert audit["all_exact_intersections_zero"] is True
    assert audit["reference_validation"]["selected_reference_intersection"] == 0
    assert audit["reference_validation"]["intersection_zero"] is True
    assert len(audit["pairwise_sources"]) == 21
    assert all(
        set(pair["intersection_counts"].values()) == {0} for pair in audit["pairwise_sources"]
    )

    with sqlite3.connect(out_dir / "selection_registry.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0] == 7
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(DISTINCT canonical_fingerprint) FROM selections"
            ).fetchone()[0]
            == 7
        )

    assert not list(tmp_path.glob(".selected_registry.selecting-*"))


def test_selection_is_input_order_independent_and_duplicate_choice_is_stable(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    records = [
        {"text": "alpha one", "record": 1},
        {"text": "“same canonical text”", "record": 2},
        {"text": '"same canonical text"', "record": 3},
        {"text": "bravo two", "record": 4},
        {"text": "charlie three", "record": 5},
        {"text": "delta four", "record": 6},
    ]
    input_one = _write_jsonl(tmp_path / "ordered.jsonl", records)
    input_two = _write_jsonl(tmp_path / "reversed.jsonl", list(reversed(records)))
    maximum_document = max(
        _serialized_tokens(production_chat_tok, str(record["text"])) for record in records
    )
    target = maximum_document + 1
    cleaning = {"normalize_quotes": True}
    reference_exclusion = _write_reference_exclusion(
        tmp_path / "reference_exclusion.json",
        cleaning=cleaning,
    )
    spec_one = _write_spec(
        tmp_path / "spec_one.json",
        [_source(stage="stage_a", source_id="stable_source", path=input_one, target=target)],
        cleaning=cleaning,
    )
    spec_two = _write_spec(
        tmp_path / "spec_two.json",
        [_source(stage="stage_a", source_id="stable_source", path=input_two, target=target)],
        cleaning=cleaning,
    )

    out_one = tmp_path / "out_one"
    out_two = tmp_path / "out_two"
    first = selector.build_selection_registry(
        spec_path=spec_one,
        tokenizer_path=tokenizer_path,
        out_dir=out_one,
        exclude_hash_manifests=(reference_exclusion,),
        sqlite_cache_mb=1,
        commit_every=2,
    )
    second = selector.build_selection_registry(
        spec_path=spec_two,
        tokenizer_path=tokenizer_path,
        out_dir=out_two,
        exclude_hash_manifests=(reference_exclusion,),
        sqlite_cache_mb=1,
        commit_every=2,
    )

    relative = Path("selected/stage_a/stable_source.jsonl")
    assert (out_one / relative).read_bytes() == (out_two / relative).read_bytes()
    assert first["sources"][0]["selected"] == second["sources"][0]["selected"]
    assert first["sources"][0]["output"]["sha256"] == second["sources"][0]["output"]["sha256"]
    # The exact input artifact is still recorded, so reordered source files differ audibly.
    assert first["sources"][0]["input_sha256"] != second["sources"][0]["input_sha256"]


def test_whole_document_quota_has_explicit_boundary_accounting(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    reference_exclusion: Path,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    texts = [
        "short alpha",
        "medium beta with more words",
        "literal [BOS] <|user|> [EOS] strings are content",
        "def f(value):\n    return value * value",
    ]
    input_path = _write_jsonl(
        tmp_path / "quota.jsonl",
        [{"text": text, "license": "test"} for text in texts],
    )
    target = max(_serialized_tokens(production_chat_tok, text) for text in texts) + 1
    spec_path = _write_spec(
        tmp_path / "quota_spec.json",
        [_source(stage="stage_b", source_id="quota_source", path=input_path, target=target)],
    )
    out_dir = tmp_path / "quota_out"
    manifest = selector.build_selection_registry(
        spec_path=spec_path,
        tokenizer_path=tokenizer_path,
        out_dir=out_dir,
        exclude_hash_manifests=(reference_exclusion,),
        sqlite_cache_mb=1,
    )

    selected = manifest["sources"][0]["selected"]
    records = _read_jsonl(out_dir / "selected/stage_b/quota_source.jsonl")
    metadata = [record[selector.SELECTION_METADATA_FIELD] for record in records]
    assert selected["serialized_tokens"] >= target
    assert selected["documents"] == len(records)
    assert selected["boundary_tokens"] == 2 * len(records)
    assert selected["serialized_tokens"] == (
        selected["content_tokens"] + selected["boundary_tokens"]
    )
    assert selected["serialized_tokens"] == sum(item["serialized_tokens"] for item in metadata)
    assert selected["serialized_token_overshoot"] < max(
        item["serialized_tokens"] for item in metadata
    )
    assert all(item["boundary_tokens"] == 2 for item in metadata)
    assert manifest["tokenizer"]["special_token_ids"] == SPECIAL_TOKEN_IDS


def test_source_exhaustion_is_atomic_and_leaves_only_failure_manifest(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    reference_exclusion: Path,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    text = "the only available document"
    input_path = _write_jsonl(tmp_path / "small.jsonl", [{"text": text}])
    target = _serialized_tokens(production_chat_tok, text) + 1
    spec_path = _write_spec(
        tmp_path / "small_spec.json",
        [_source(stage="stage_b", source_id="too_small", path=input_path, target=target)],
    )
    out_dir = tmp_path / "failed_selection"

    with pytest.raises(selector.SourceExhaustedError, match="exhausted"):
        selector.build_selection_registry(
            spec_path=spec_path,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            exclude_hash_manifests=(reference_exclusion,),
            sqlite_cache_mb=1,
        )

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".failed_selection.selecting-*"))
    failures = list(tmp_path.glob("failed_selection.failed-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["status"] == "failed"
    assert failure["error_type"] == "SourceExhaustedError"
    assert failure["current_source_id"] == "too_small"
    assert failure["production_directory_published"] is False


def test_canonical_hash_collision_fails_loudly_and_does_not_publish(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    reference_exclusion: Path,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    input_path = _write_jsonl(
        tmp_path / "collision.jsonl",
        [{"text": "first distinct document"}, {"text": "second distinct document"}],
    )
    spec_path = _write_spec(
        tmp_path / "collision_spec.json",
        [
            _source(
                stage="stage_b",
                source_id="collision_source",
                path=input_path,
                target=1_000_000,
            )
        ],
    )
    monkeypatch.setattr(
        selector,
        "canonical_document_fingerprint",
        lambda _text: "0" * 64,
    )
    out_dir = tmp_path / "collision_out"

    with pytest.raises(selector.HashCollisionError, match="canonical fingerprint collision"):
        selector.build_selection_registry(
            spec_path=spec_path,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            exclude_hash_manifests=(reference_exclusion,),
            sqlite_cache_mb=1,
        )
    assert not out_dir.exists()
    failure = json.loads(next(tmp_path.glob("collision_out.failed-*.json")).read_text())
    assert failure["error_type"] == "HashCollisionError"


def test_spec_rejects_source_id_and_resolved_path_collisions(tmp_path: Path) -> None:
    first = _write_jsonl(tmp_path / "first.jsonl", [{"text": "first"}])
    second = _write_jsonl(tmp_path / "second.jsonl", [{"text": "second"}])

    duplicate_id = _write_spec(
        tmp_path / "duplicate_id.json",
        [
            _source(stage="stage_b", source_id="same", path=first),
            _source(stage="stage_a", source_id="same", path=second),
        ],
    )
    with pytest.raises(ValueError, match="duplicate source_id"):
        selector.load_selection_spec(duplicate_id)

    duplicate_path = _write_spec(
        tmp_path / "duplicate_path.json",
        [
            _source(stage="stage_b", source_id="first", path=first),
            _source(stage="control", source_id="second", path=first),
        ],
    )
    with pytest.raises(ValueError, match="duplicate resolved source path"):
        selector.load_selection_spec(duplicate_path)


def test_noncanonical_tokenizer_is_rejected_before_staging(
    tmp_path: Path,
    chat_tok: Tokenizer,
    reference_exclusion: Path,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tiny_tokenizer.json", chat_tok)
    input_path = _write_jsonl(tmp_path / "input.jsonl", [{"text": "some text"}])
    spec_path = _write_spec(
        tmp_path / "spec.json",
        [_source(stage="stage_b", source_id="source", path=input_path)],
    )
    out_dir = tmp_path / "should_not_exist"

    with pytest.raises(ValueError, match="runtime vocab IDs must be exactly"):
        selector.build_selection_registry(
            spec_path=spec_path,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            exclude_hash_manifests=(reference_exclusion,),
        )
    assert not out_dir.exists()
    assert not list(tmp_path.glob(".should_not_exist.selecting-*"))


def test_reference_exclusion_manifest_is_mandatory(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    input_path = _write_jsonl(tmp_path / "input.jsonl", [{"text": "some text"}])
    spec_path = _write_spec(
        tmp_path / "spec.json",
        [_source(stage="stage_b", source_id="source", path=input_path)],
    )
    out_dir = tmp_path / "missing_reference_contract"

    with pytest.raises(ValueError, match="at least one --exclude_hash_manifest is required"):
        selector.build_selection_registry(
            spec_path=spec_path,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            exclude_hash_manifests=(),
        )
    assert not out_dir.exists()
    assert not list(tmp_path.glob(".missing_reference_contract.selecting-*"))


def test_existing_output_is_never_mixed_or_overwritten(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    reference_exclusion: Path,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    input_path = _write_jsonl(tmp_path / "input.jsonl", [{"text": "some text"}])
    spec_path = _write_spec(
        tmp_path / "spec.json",
        [_source(stage="stage_b", source_id="source", path=input_path)],
    )
    out_dir = tmp_path / "existing"
    out_dir.mkdir()
    sentinel = out_dir / "sentinel.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        selector.build_selection_registry(
            spec_path=spec_path,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            exclude_hash_manifests=(reference_exclusion,),
        )
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert sorted(path.name for path in out_dir.iterdir()) == ["sentinel.txt"]


def test_keyboard_interrupt_cleans_staging_and_is_reraised(
    tmp_path: Path,
    production_chat_tok: Tokenizer,
    monkeypatch: pytest.MonkeyPatch,
    reference_exclusion: Path,
) -> None:
    tokenizer_path = _write_tokenizer(tmp_path / "tokenizer.json", production_chat_tok)
    input_path = _write_jsonl(tmp_path / "input.jsonl", [{"text": "some text"}])
    spec_path = _write_spec(
        tmp_path / "spec.json",
        [_source(stage="stage_b", source_id="source", path=input_path)],
    )

    def interrupt(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(selector, "_select_source", interrupt)
    out_dir = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        selector.build_selection_registry(
            spec_path=spec_path,
            tokenizer_path=tokenizer_path,
            out_dir=out_dir,
            exclude_hash_manifests=(reference_exclusion,),
            sqlite_cache_mb=1,
        )

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".interrupted.selecting-*"))
    failure = json.loads(next(tmp_path.glob("interrupted.failed-*.json")).read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    assert failure["production_directory_published"] is False
