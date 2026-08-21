"""Focused contracts for canonical tokenizer-training provenance and publication.

These pin the OD-G1/OD-G2 behaviour: a canonical release must prove which frozen corpus
produced it, and it must fail closed — before any training work — on a mutated manifest, a
mutated data file, or a training file set that is not exactly the bound corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from pretrain.build_pretrain_shards import (
    CLEANED_TEXT_HASH_ALGORITHM,
    EXCLUSION_MANIFEST_KIND,
)
from tokenizer.tokenizer_training.train_tokenizer import (
    CORPUS_MANIFEST_SCHEMA,
    ROUNDTRIP_FIXTURES,
    load_corpus_release_manifest,
    main,
    verify_corpus_binding,
    write_sha256sums,
)

BUCKETS = (("Alpha", "alpha.jsonl"), ("Beta", "beta.jsonl"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def _corpus(root: Path, *, rows: dict[str, list[str]] | None = None) -> Path:
    """Build a frozen-release-shaped corpus directory with a manifest that binds it."""
    release = root / "release"
    release.mkdir(parents=True)
    payload = rows or {
        "alpha.jsonl": [f"alpha document {index} with words\n" * 3 for index in range(40)],
        "beta.jsonl": [f"def beta_{index}(x):\n    return x + {index}\n" for index in range(40)],
    }
    buckets = []
    for name, filename in BUCKETS:
        target = release / filename
        _write_jsonl(target, payload[filename])
        buckets.append({
            "canonical_name": name,
            "output": {
                "path": filename,
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
                "rows": len(payload[filename]),
            },
        })
    manifest = {
        "schema_version": CORPUS_MANIFEST_SCHEMA,
        "status": "COMPLETE_SELF_VERIFIED",
        "immutable_publication": True,
        "run_fingerprint_sha256": "aa" * 32,
        "selection": {
            "buckets": buckets,
            "selected_occurrence_set_sha256": "bb" * 32,
            "total_selected_documents": sum(len(v) for v in payload.values()),
            "total_realized_cleaned_utf8_bytes": 1234,
        },
    }
    (release / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return release / "manifest.json"


def _exclusions(path: Path) -> Path:
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
            "membership_basis": "cleaned document text encoded as UTF-8",
            "cleaning": {
                "strip_leading_noise": False,
                "normalize_quotes": False,
                "underscores_policy": "keep",
                "min_chars": 0,
                "min_ascii_ratio": 0.0,
            },
            "hash_count": 0,
            "hashes": [],
        }),
        encoding="utf-8",
    )
    return path


def _data_paths(manifest_path: Path) -> list[str]:
    return [str(manifest_path.parent / filename) for _, filename in BUCKETS]


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_tokenizer.py", *argv])
    main()


def _base_argv(manifest_path: Path, out_dir: Path, exclusions: Path, **over: object) -> list[str]:
    argv = [
        "--data",
        *_data_paths(manifest_path),
        "--fields",
        "text",
        "--vocab_size",
        str(over.get("vocab_size", 300)),
        "--exclude_hash_manifest",
        str(exclusions),
        "--corpus_release_manifest",
        str(manifest_path),
        "--corpus_release_manifest_sha256",
        str(over.get("sha", _sha256(manifest_path))),
        "--out_dir",
        str(out_dir),
        "--full_corpus_validation",
        "--legacy_allow_noncanonical_contract",
    ]
    return argv


# --------------------------------------------------------------------------- binding


def test_manifest_sha_pin_mismatch_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_corpus_release_manifest(manifest_path, "cc" * 32)


def test_unexpected_manifest_schema_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "some-other-corpus-v9"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected corpus manifest schema"):
        load_corpus_release_manifest(manifest_path, _sha256(manifest_path))


def test_non_immutable_manifest_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["immutable_publication"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not marked immutable"):
        load_corpus_release_manifest(manifest_path, _sha256(manifest_path))


def test_missing_fingerprints_fail_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["selection"]["selected_occurrence_set_sha256"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selected-set fingerprint"):
        load_corpus_release_manifest(manifest_path, _sha256(manifest_path))


def test_binding_records_every_bucket_identity(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    binding = load_corpus_release_manifest(manifest_path, _sha256(manifest_path))
    records = verify_corpus_binding(binding, _data_paths(manifest_path))
    assert {item["canonical_bucket"] for item in records} == {"Alpha", "Beta"}
    for record in records:
        assert record["verified_sha256"] == record["sha256"]
        assert record["expected_occurrences"] == 40
        assert record["size_bytes"] == Path(record["path"]).stat().st_size


def test_missing_member_file_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    binding = load_corpus_release_manifest(manifest_path, _sha256(manifest_path))
    with pytest.raises(ValueError, match="does not cover the bound corpus release"):
        verify_corpus_binding(binding, _data_paths(manifest_path)[:1])


def test_extra_sixth_unbound_file_is_not_silently_accepted(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    stray = tmp_path / "stray.jsonl"
    _write_jsonl(stray, ["not part of the frozen corpus\n"])
    binding = load_corpus_release_manifest(manifest_path, _sha256(manifest_path))
    with pytest.raises(ValueError, match="not a member of the bound corpus release"):
        verify_corpus_binding(binding, [*_data_paths(manifest_path), str(stray)])


def test_duplicate_input_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    binding = load_corpus_release_manifest(manifest_path, _sha256(manifest_path))
    paths = _data_paths(manifest_path)
    with pytest.raises(ValueError, match="supplied twice"):
        verify_corpus_binding(binding, [*paths, paths[0]])


def test_same_size_content_mutation_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    binding = load_corpus_release_manifest(manifest_path, _sha256(manifest_path))
    target = manifest_path.parent / "alpha.jsonl"
    raw = bytearray(target.read_bytes())
    index = raw.find(b"alpha document 0")
    raw[index : index + 5] = b"AlPhA"
    target.write_bytes(bytes(raw))
    assert target.stat().st_size == binding["outputs"][str(target.resolve())]["size_bytes"]
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_corpus_binding(binding, _data_paths(manifest_path))


def test_missing_input_file_fails_closed(tmp_path: Path):
    manifest_path = _corpus(tmp_path)
    binding = load_corpus_release_manifest(manifest_path, _sha256(manifest_path))
    (manifest_path.parent / "beta.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        verify_corpus_binding(binding, _data_paths(manifest_path))


# --------------------------------------------------------------------------- CLI gates


def test_canonical_release_requires_corpus_manifest_and_full_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    with pytest.raises(SystemExit) as excinfo:
        _run(
            [
                "--data",
                *_data_paths(manifest_path),
                "--vocab_size",
                "32000",
                "--exclude_hash_manifest",
                str(exclusions),
                "--out_dir",
                str(tmp_path / "out"),
            ],
            monkeypatch,
        )
    message = str(excinfo.value)
    assert "canonical corpus release manifest missing" in message
    assert "full-corpus streaming validation disabled" in message
    assert not (tmp_path / "out").exists()


def test_corpus_manifest_without_sha_pin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    argv = _base_argv(manifest_path, tmp_path / "out", exclusions)
    argv.remove("--corpus_release_manifest_sha256")
    argv.remove(_sha256(manifest_path))
    with pytest.raises(SystemExit, match="requires --corpus_release_manifest_sha256"):
        _run(argv, monkeypatch)
    assert not (tmp_path / "out").exists()


def test_binding_is_verified_before_training_and_leaves_no_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    out_dir = tmp_path / "out"
    target = manifest_path.parent / "alpha.jsonl"
    target.write_bytes(target.read_bytes() + b'{"text":"late addition"}\n')
    with pytest.raises(RuntimeError, match="size mismatch"):
        _run(_base_argv(manifest_path, out_dir, exclusions), monkeypatch)
    assert not out_dir.exists()
    assert not list(out_dir.parent.glob(".out.building-*"))


def test_occurrence_count_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = _corpus(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selection"]["buckets"][0]["output"]["rows"] = 39
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    exclusions = _exclusions(tmp_path / "excl.json")
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="corpus manifest declares 39"):
        _run(_base_argv(manifest_path, out_dir, exclusions), monkeypatch)
    assert not out_dir.exists()


# --------------------------------------------------------------------------- publication


def test_publication_emits_bound_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    out_dir = tmp_path / "out"
    _run(_base_argv(manifest_path, out_dir, exclusions), monkeypatch)

    for name in (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "manifest.json",
        "environment.json",
        "validation.json",
        "SHA256SUMS",
    ):
        assert (out_dir / name).is_file(), name

    # Exactly one manifest identity: the compatibility copy is byte-identical.
    assert (out_dir / "manifest.json").read_bytes() == (
        out_dir / "tokenizer_release_manifest.json"
    ).read_bytes()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    binding = manifest["corpus_binding"]
    assert binding["verified_before_training"] is True
    assert binding["manifest_sha256"] == _sha256(manifest_path)
    assert binding["run_fingerprint_sha256"] == "aa" * 32
    assert binding["selected_occurrence_set_sha256"] == "bb" * 32
    assert {item["canonical_bucket"] for item in binding["files"]} == {"Alpha", "Beta"}
    assert manifest["training"]["consumed_occurrences"] == 80
    for entry in manifest["training"]["per_file"]:
        assert entry["yielded_samples"] == entry["expected_occurrences"]

    assert manifest["tokenizer_sha256"] == _sha256(out_dir / "tokenizer.json")
    assert manifest["artifacts"]["canonical_runtime_artifact"] == "tokenizer.json"
    assert manifest["artifacts"]["authoritative_manifest"] == "tokenizer_release_manifest.json"
    assert manifest["artifacts"]["byte_identical_manifest_copy"] == "manifest.json"

    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["fixtures"]["count"] == len(ROUNDTRIP_FIXTURES)
    assert validation["fixtures"]["failures"] == 0
    stream = validation["full_corpus_stream"]
    assert stream["status"] == "PASS"
    assert stream["totals"]["occurrences"] == 80
    assert stream["totals"]["roundtrip_failures"] == 0
    assert stream["totals"]["unk_occurrences"] == 0
    assert stream["totals"]["special_id_occurrences"] == 0

    environment = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["tokenizers_version"]
    assert environment["python_version"]
    assert environment["trainer_sha256"]
    assert environment["input_file_order"] == _data_paths(manifest_path)


def test_sha256sums_covers_payload_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    out_dir = tmp_path / "out"
    _run(_base_argv(manifest_path, out_dir, exclusions), monkeypatch)

    listed = {}
    for line in (out_dir / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    # Manifest copies and the checksum file itself are hashed at closeout instead.
    assert "manifest.json" not in listed
    assert "tokenizer_release_manifest.json" not in listed
    assert "SHA256SUMS" not in listed
    assert {"tokenizer.json", "vocab.json", "merges.txt", "environment.json"} <= set(listed)
    for name, digest in listed.items():
        assert _sha256(out_dir / name) == digest, name


def test_write_sha256sums_skips_excluded_names(tmp_path: Path):
    root = tmp_path / "staging"
    (root / "nested").mkdir(parents=True)
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    (root / "nested" / "deep.txt").write_text("deep", encoding="utf-8")
    (root / "skip.json").write_text("skip", encoding="utf-8")
    write_sha256sums(root, exclude={"skip.json", "SHA256SUMS"})
    names = [
        line.split("  ", 1)[1]
        for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    ]
    assert sorted(names) == ["keep.txt", "nested/deep.txt"]


def test_under_capacity_vocabulary_is_never_padded_or_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A corpus too small to fill the target must hard-stop, not silently ship a short vocab."""
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit, match=r"vocab_size \d+ != 4000"):
        _run(_base_argv(manifest_path, out_dir, exclusions, vocab_size=4000), monkeypatch)
    assert not out_dir.exists()
    assert not list(out_dir.parent.glob(".out.building-*"))


def test_existing_output_directory_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = _corpus(tmp_path)
    exclusions = _exclusions(tmp_path / "excl.json")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(FileExistsError):
        _run(_base_argv(manifest_path, out_dir, exclusions), monkeypatch)
