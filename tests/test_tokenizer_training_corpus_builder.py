"""Focused contracts for the Stage-F tokenizer-training corpus builder."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tokenizer.data_preparation.build_tokenizer_training_corpus import (
    EXCLUSION_ALGORITHM,
    EXCLUSION_KIND,
    EXPECTED_L1_CLEANING,
    OUTPUT_FIELDS,
    RANK_PERSON,
    RANK_SEED_ASCII,
    SELECTION_RECORD,
    BucketSpec,
    BuildContract,
    ReleaseSpec,
    build_corpus,
    f_payload,
    l1_identity_digest,
    rank_digest,
    sha256_file,
)


def _write_jsonl(path: Path, texts: list[str]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for text in texts:
            handle.write(
                json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return path


def _write_index(path: Path, rows: list[int]) -> Path:
    path.write_bytes(b"".join(struct.pack("<I", row) for row in rows))
    return path


def _write_l1_manifest(path: Path, texts: list[str] = ()) -> Path:
    hashes = sorted({l1_identity_digest(text).hex() for text in texts})
    payload = {
        "schema_version": 1,
        "kind": EXCLUSION_KIND,
        "hash_algorithm": EXCLUSION_ALGORITHM,
        "membership_basis": "cleaned document text encoded as UTF-8",
        "cleaning": EXPECTED_L1_CLEANING,
        "hash_count": len(hashes),
        "hashes": hashes,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _release(
    tmp_path: Path,
    release_id: str,
    bucket_name: str,
    texts: list[str],
    *,
    excluded_rows: list[int] | None = None,
) -> ReleaseSpec:
    source = _write_jsonl(tmp_path / f"{release_id}.jsonl", texts)
    excluded_rows = sorted(excluded_rows or [])
    index = _write_index(tmp_path / f"{release_id}.excluded.u32.raw", excluded_rows)
    return ReleaseSpec(
        canonical_release_id=release_id,
        bucket_name=bucket_name,
        source_path=source,
        documents_sha256=sha256_file(source),
        documents_file_bytes=source.stat().st_size,
        physical_rows=len(texts),
        eligible_rows=len(texts) - len(excluded_rows),
        excluded_rows=len(excluded_rows),
        excluded_rows_path=index,
        excluded_rows_sha256=sha256_file(index),
    )


def _contract(
    tmp_path: Path,
    buckets: tuple[BucketSpec, ...],
    releases: tuple[ReleaseSpec, ...],
    *,
    l1_texts: list[str] | None = None,
) -> BuildContract:
    exclusion = _write_l1_manifest(tmp_path / "l1_exclusions.json", l1_texts or [])
    return BuildContract(
        buckets=buckets,
        releases=releases,
        l1_exclusion_manifest_path=exclusion,
        l1_exclusion_manifest_sha256=sha256_file(exclusion),
        l1_exclusion_hash_count=len({l1_identity_digest(text) for text in l1_texts or []}),
    )


def _one_bucket(
    tmp_path: Path,
    texts: list[str],
    target_bytes: int,
    *,
    excluded_rows: list[int] | None = None,
    l1_texts: list[str] | None = None,
    bucket_name: str = "FineWeb",
) -> tuple[BuildContract, ReleaseSpec]:
    release = _release(
        tmp_path,
        "release_a",
        bucket_name,
        texts,
        excluded_rows=excluded_rows,
    )
    bucket = BucketSpec(bucket_name, "bucket", target_bytes, (release.canonical_release_id,))
    return _contract(tmp_path, (bucket,), (release,), l1_texts=l1_texts), release


def _read_rows(release_dir: Path, slug: str = "bucket") -> list[dict]:
    with open(release_dir / f"{slug}.jsonl", encoding="utf-8", newline="") as handle:
        return [json.loads(line) for line in handle]


def _selection_records(path: Path) -> list[tuple[bytes, bytes, int, int, int]]:
    raw = path.read_bytes()
    assert len(raw) % SELECTION_RECORD.size == 0
    return list(struct.iter_unpack(SELECTION_RECORD.format, raw))


def test_rank_known_answer_pins_person_and_exact_message_bytes():
    payload_sha = bytes(range(32))
    message = b"20250814\0structured/tutorial\0" + payload_sha
    expected = hashlib.blake2b(message, digest_size=16, person=b"PetitGPT-F-v1").digest()

    assert RANK_SEED_ASCII == b"20250814"
    assert RANK_PERSON == b"PetitGPT-F-v1"
    assert expected.hex() == "39c7d933db49852c69ab115f556cc400"
    assert rank_digest("structured/tutorial", payload_sha) == expected
    with pytest.raises(ValueError, match="exactly 32"):
        rank_digest("structured/tutorial", payload_sha[:-1])


def test_raw_whitespace_and_unicode_round_trip_without_f_transform(tmp_path: Path):
    texts = [
        "\n\r  Cafe\u0301 and ‘quotes’\t\n\r",
        "  汉字\u00a0stay byte-exact  ",
        "line one\n\nline three\r",
    ]
    total = sum(len(text.encode("utf-8")) for text in texts)
    contract, _ = _one_bucket(tmp_path, texts, total)
    out = tmp_path / "release"
    manifest = build_corpus(contract, out, workers=1, make_immutable=False)

    rows = _read_rows(out)
    by_index = {row["physical_row_index"]: row for row in rows}
    assert [by_index[index]["text"] for index in range(len(texts))] == texts
    for index, text in enumerate(texts):
        digest, size = f_payload(text)
        assert by_index[index]["cleaned_text_sha256"] == digest.hex()
        assert len(by_index[index]["text"].encode("utf-8")) == size
    assert manifest["contract"]["text_contract"]["transform"] == "identity"
    assert manifest["contract"]["text_contract"]["strip"] is False
    assert manifest["selection"]["total_realized_cleaned_utf8_bytes"] == total


def test_d2_d3_and_legacy_l1_filter_before_rank_then_replace(tmp_path: Path):
    texts = [f"\r\nfilter candidate {index} café\n" for index in range(80)]
    ranked = sorted(
        range(len(texts)),
        key=lambda index: (
            rank_digest("FineWeb", f_payload(texts[index])[0]),
            f_payload(texts[index])[0],
            "release_a",
            index,
        ),
    )
    d2_victim, l1_victim, expected_replacement = ranked[:3]
    contract, _ = _one_bucket(
        tmp_path,
        texts,
        1,
        excluded_rows=[d2_victim],
        l1_texts=[texts[l1_victim]],
    )

    out = tmp_path / "release"
    manifest = build_corpus(contract, out, workers=1, make_immutable=False)
    rows = _read_rows(out)

    assert (
        l1_identity_digest(texts[l1_victim])
        == hashlib.sha256(texts[l1_victim].strip("\n\r").encode("utf-8")).digest()
    )
    assert f_payload(texts[l1_victim])[0] != l1_identity_digest(texts[l1_victim])
    assert [row["physical_row_index"] for row in rows] == [expected_replacement]
    measured = manifest["input_bindings"]["releases"][0]["measured"]
    assert measured["d2_d3_eligible_rows"] == len(texts) - 1
    assert measured["l1_match_occurrences"] == 1
    assert manifest["selection"]["l1_exclusion_intersection"] == 0


def test_duplicate_occurrences_use_release_and_row_tiebreak_and_ignore_input_order(
    tmp_path: Path,
):
    text = "same tokenizer-visible occurrence"
    text_bytes = len(text.encode("utf-8"))
    release_a = _release(tmp_path, "a_release", "DCLM", [text, text])
    release_z = _release(tmp_path, "z_release", "DCLM", [text])
    bucket = BucketSpec("DCLM", "dclm", text_bytes + 1, ("a_release", "z_release"))
    l1 = _write_l1_manifest(tmp_path / "l1_exclusions.json")

    def make_contract(releases: tuple[ReleaseSpec, ...]) -> BuildContract:
        return BuildContract(
            buckets=(bucket,),
            releases=releases,
            l1_exclusion_manifest_path=l1,
            l1_exclusion_manifest_sha256=sha256_file(l1),
            l1_exclusion_hash_count=0,
        )

    out_reverse = tmp_path / "reverse"
    reverse_manifest = build_corpus(
        make_contract((release_z, release_a)),
        out_reverse,
        workers=1,
        make_immutable=False,
    )
    out_forward = tmp_path / "forward"
    forward_manifest = build_corpus(
        make_contract((release_a, release_z)),
        out_forward,
        workers=1,
        make_immutable=False,
    )

    expected_occurrences = [("a_release", 0), ("a_release", 1)]
    assert [
        (row["canonical_release_id"], row["physical_row_index"])
        for row in _read_rows(out_reverse, "dclm")
    ] == expected_occurrences
    assert (out_reverse / "dclm.jsonl").read_bytes() == (out_forward / "dclm.jsonl").read_bytes()
    assert (out_reverse / "indices/dclm.selection.idx").read_bytes() == (
        out_forward / "indices/dclm.selection.idx"
    ).read_bytes()
    assert reverse_manifest["run_fingerprint_sha256"] == forward_manifest["run_fingerprint_sha256"]
    assert reverse_manifest["selection"]["buckets"][0]["selected_documents"] == 2


def test_whole_document_prefix_has_only_final_document_overshoot(tmp_path: Path):
    texts = ["a" * 7, "b" * 11, "c" * 17, "d" * 23]
    order = sorted(
        range(len(texts)),
        key=lambda index: (
            rank_digest("Wikipedia", f_payload(texts[index])[0]),
            f_payload(texts[index])[0],
            "release_a",
            index,
        ),
    )
    first_bytes = len(texts[order[0]].encode("utf-8"))
    second_bytes = len(texts[order[1]].encode("utf-8"))
    target = first_bytes + 1
    contract, _ = _one_bucket(tmp_path, texts, target, bucket_name="Wikipedia")
    out = tmp_path / "release"
    manifest = build_corpus(contract, out, workers=1, make_immutable=False)
    bucket = manifest["selection"]["buckets"][0]

    assert [row["physical_row_index"] for row in _read_rows(out)] == order[:2]
    assert bucket["realized_cleaned_utf8_bytes"] == first_bytes + second_bytes
    assert bucket["overshoot_bytes"] == second_bytes - 1
    assert (
        bucket["realized_cleaned_utf8_bytes"] - bucket["last_document_cleaned_utf8_bytes"] < target
    )
    assert bucket["last_selected"]["physical_row_index"] == order[1]
    assert bucket["first_unselected"]["physical_row_index"] == order[2]


def test_under_capacity_fails_without_publishing_or_leaving_staging(tmp_path: Path):
    texts = ["short", "also short"]
    target = sum(len(text.encode("utf-8")) for text in texts) + 1
    contract, _ = _one_bucket(tmp_path, texts, target)
    out = tmp_path / "release"

    with pytest.raises(RuntimeError, match="capacity below target"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert not out.exists()
    assert not (tmp_path / ".release.staging").exists()


def test_missing_source_fails_closed_before_staging(tmp_path: Path):
    contract, release = _one_bucket(tmp_path, ["document"], 1)
    release.source_path.unlink()
    out = tmp_path / "release"

    with pytest.raises(FileNotFoundError, match="source is missing"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert not out.exists()
    assert not (tmp_path / ".release.staging").exists()


def test_same_size_source_mutation_fails_closed_and_cleans_staging(tmp_path: Path):
    contract, release = _one_bucket(tmp_path, ["alpha document", "beta document"], 1)
    raw = release.source_path.read_bytes()
    assert b"alpha" in raw
    release.source_path.write_bytes(raw.replace(b"alpha", b"omega", 1))
    assert release.source_path.stat().st_size == release.documents_file_bytes
    out = tmp_path / "release"

    with pytest.raises(RuntimeError, match="source SHA-256 mismatch"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert not out.exists()
    assert not (tmp_path / ".release.staging").exists()


def test_mutated_d2_d3_index_fails_closed_before_staging(tmp_path: Path):
    contract, release = _one_bucket(tmp_path, ["zero", "one", "two"], 1, excluded_rows=[0])
    release.excluded_rows_path.write_bytes(struct.pack("<I", 1))
    out = tmp_path / "release"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert not out.exists()
    assert not (tmp_path / ".release.staging").exists()


def test_mutated_l1_exclusion_manifest_fails_closed_before_staging(tmp_path: Path):
    contract, _ = _one_bucket(tmp_path, ["document"], 1)
    contract.l1_exclusion_manifest_path.write_text("{}", encoding="utf-8")
    out = tmp_path / "release"

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert not out.exists()
    assert not (tmp_path / ".release.staging").exists()


def test_existing_final_and_stale_staging_are_refused(tmp_path: Path):
    contract, _ = _one_bucket(tmp_path, ["document"], 1)
    out = tmp_path / "release"
    out.mkdir()
    marker = out / "marker"
    marker.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable release exists"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert marker.read_text(encoding="utf-8") == "do not overwrite"

    out.rmdir() if not any(out.iterdir()) else marker.unlink()
    if out.exists():
        out.rmdir()
    staging = tmp_path / ".release.staging"
    staging.mkdir()
    (staging / "evidence").write_text("preserve for inspection", encoding="utf-8")
    with pytest.raises(FileExistsError, match="stale staging requires inspection"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert (staging / "evidence").read_text(encoding="utf-8") == "preserve for inspection"


def test_malformed_input_failure_is_atomic_and_cleans_owned_staging(tmp_path: Path):
    contract, release = _one_bucket(tmp_path, ["valid", "also valid"], 1)
    malformed = b'{"text":"valid"}\nnot-json-but-same-size!!\n'
    release.source_path.write_bytes(malformed)
    mutated = replace(
        release,
        documents_sha256=sha256_file(release.source_path),
        documents_file_bytes=release.source_path.stat().st_size,
        physical_rows=2,
        eligible_rows=2,
    )
    contract = replace(contract, releases=(mutated,))
    out = tmp_path / "release"

    with pytest.raises(ValueError, match="invalid JSON"):
        build_corpus(contract, out, workers=1, make_immutable=False)
    assert not out.exists()
    assert not (tmp_path / ".release.staging").exists()


def test_five_outputs_schema_hashes_and_occurrence_fingerprints(tmp_path: Path):
    bucket_defs = [
        ("FineWeb", "fineweb"),
        ("DCLM", "dclm"),
        ("Wikipedia", "wikipedia"),
        ("Python", "python"),
        ("structured/tutorial", "structured_tutorial"),
    ]
    releases = tuple(
        _release(tmp_path, f"release_{index}", name, [f"payload {index} Ω"])
        for index, (name, _) in enumerate(bucket_defs)
    )
    buckets = tuple(
        BucketSpec(name, slug, 1, (releases[index].canonical_release_id,))
        for index, (name, slug) in enumerate(bucket_defs)
    )
    contract = _contract(tmp_path, buckets, releases)
    out = tmp_path / "release"
    manifest = build_corpus(contract, out, workers=1, make_immutable=False)

    assert manifest["status"] == "COMPLETE_SELF_VERIFIED"
    assert manifest["verification"]["status"] == "PASS"
    assert manifest["verification"]["output_schema_and_field_order"] == list(OUTPUT_FIELDS)
    assert len(manifest["selection"]["buckets"]) == 5

    aggregate = hashlib.sha256(b"PetitGPT-F-selected-occurrence-sets-v1\0")
    for bucket_spec, result in zip(buckets, manifest["selection"]["buckets"], strict=True):
        output_path = out / result["output"]["path"]
        index_path = out / result["selection_index"]["path"]
        row = _read_rows(out, bucket_spec.slug)[0]
        assert list(row) == list(OUTPUT_FIELDS)
        assert row["canonical_source"] == bucket_spec.canonical_name
        assert row["cleaned_text_sha256"] == hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        assert sha256_file(output_path) == result["output"]["sha256"]
        assert sha256_file(index_path) == result["selection_index"]["sha256"]

        records = _selection_records(index_path)
        sequence = hashlib.sha256(b"PetitGPT-F-selected-occurrence-sequence-v1\0")
        sequence.update(index_path.read_bytes())
        assert sequence.hexdigest() == result["selected_occurrence_sequence_sha256"]

        canonical = b"".join(
            SELECTION_RECORD.pack(*record)
            for record in sorted(records, key=lambda item: (item[2], item[3]))
        )
        name = bucket_spec.canonical_name.encode("utf-8")
        per_bucket = hashlib.sha256()
        per_bucket.update(b"PetitGPT-F-selected-occurrence-set-v1\0")
        per_bucket.update(len(name).to_bytes(2, "big"))
        per_bucket.update(name)
        per_bucket.update(len(records).to_bytes(8, "big"))
        per_bucket.update(canonical)
        assert per_bucket.hexdigest() == result["selected_occurrence_set_sha256"]

        aggregate.update(len(name).to_bytes(2, "big"))
        aggregate.update(name)
        aggregate.update(len(canonical).to_bytes(8, "big"))
        aggregate.update(canonical)

    assert aggregate.hexdigest() == manifest["selection"]["selected_occurrence_set_sha256"]
    assert sha256_file(out / "manifest.json")
    checksum_lines = (out / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert len(checksum_lines) == 10


def test_aux_acceleration_is_identical_to_full_parse_for_exact_payload(tmp_path: Path):
    texts = [
        "\r\nleading and trailing café\n\r",
        "plain Unicode 汉字 and e\u0301",
        "trailing carriage return\r",
        "\nleading line feed",
        "middle\nnewline stays untouched",
    ]
    release = _release(tmp_path, "release_a", "FineWeb", texts)

    # Match the frozen non-Python row layout used by the aux boundary detector.
    with open(release.source_path, "w", encoding="utf-8", newline="") as handle:
        for text_value in texts:
            row = {"text": text_value, "text_bytes": len(text_value.encode("utf-8"))}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    release = replace(
        release,
        documents_sha256=sha256_file(release.source_path),
        documents_file_bytes=release.source_path.stat().st_size,
    )
    target = sum(len(text_value.encode("utf-8")) for text_value in texts)
    bucket = BucketSpec("FineWeb", "fineweb", target, ("release_a",))
    full_contract = _contract(tmp_path, (bucket,), (release,))

    aux_dtype = np.dtype([("nb", "<i8"), ("sha", "V32")], align=False)
    aux_values = np.empty(len(texts), dtype=aux_dtype)
    for index, text_value in enumerate(texts):
        digest, text_bytes = f_payload(text_value)
        aux_values["nb"][index] = text_bytes
        aux_values["sha"][index] = np.void(digest)
    aux_path = tmp_path / "release_a.aux.npy"
    np.save(aux_path, aux_values, allow_pickle=False)

    loaded_aux = np.load(aux_path, allow_pickle=False)
    assert loaded_aux.dtype.names == ("nb", "sha")
    assert loaded_aux.dtype.itemsize == 40
    for index, text_value in enumerate(texts):
        digest, text_bytes = f_payload(text_value)
        raw_record = loaded_aux[index].tobytes()
        assert int.from_bytes(raw_record[:8], "little", signed=True) == text_bytes
        assert len(raw_record[8:]) == 32
        assert raw_record[8:] == digest

    accelerated_release = replace(
        release,
        aux_path=aux_path,
        aux_sha256=sha256_file(aux_path),
    )
    accelerated_contract = replace(full_contract, releases=(accelerated_release,))
    full_out = tmp_path / "full_parse"
    accelerated_out = tmp_path / "aux_parse"
    full_manifest = build_corpus(full_contract, full_out, workers=1, make_immutable=False)
    accelerated_manifest = build_corpus(
        accelerated_contract,
        accelerated_out,
        workers=1,
        make_immutable=False,
    )

    assert (full_out / "fineweb.jsonl").read_bytes() == (
        accelerated_out / "fineweb.jsonl"
    ).read_bytes()
    assert (full_out / "indices/fineweb.selection.idx").read_bytes() == (
        accelerated_out / "indices/fineweb.selection.idx"
    ).read_bytes()
    assert (full_out / "SHA256SUMS").read_bytes() == (accelerated_out / "SHA256SUMS").read_bytes()
    full_bucket = full_manifest["selection"]["buckets"][0]
    accelerated_bucket = accelerated_manifest["selection"]["buckets"][0]
    for key in (
        "output",
        "selection_index",
        "selected_occurrence_sequence_sha256",
        "selected_occurrence_set_sha256",
        "last_selected",
        "first_unselected",
    ):
        assert accelerated_bucket[key] == full_bucket[key]
    assert (
        accelerated_manifest["selection"]["selected_occurrence_set_sha256"]
        == (full_manifest["selection"]["selected_occurrence_set_sha256"])
    )
    measured = accelerated_manifest["input_bindings"]["releases"][0]["measured"]
    assert measured["boundary_reparsed_rows"] == 3
    assert {
        row["physical_row_index"]: row["text"] for row in _read_rows(accelerated_out, "fineweb")
    } == {index: text_value for index, text_value in enumerate(texts)}
