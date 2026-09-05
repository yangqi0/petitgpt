"""Contract tests for the L1 reserve eligibility seam.

`build_reference_val_shards.py` predates D2 near-duplicate dedup and D3 benchmark
decontamination, and the candidate releases are never physically rewritten, so a release file still
contains rows those stages logically removed. The reserve path must therefore filter rows by an
explicitly bound eligibility manifest BEFORE cleaning, hashing and stable ranking.

These tests pin the seam itself: filter-before-rank ordering, fail-closed identity binding, and the
guarantee that the fix changed none of the already-frozen selection semantics.
"""

from __future__ import annotations

import array
import hashlib
import json
from pathlib import Path

import pytest

from pretrain.build_reference_val_shards import (
    ELIGIBILITY_MANIFEST_KIND,
    EXCLUSION_MANIFEST_NAME,
    RESERVE_MANIFEST_NAME,
    load_reserve_eligibility,
    parse_reference_sources,
    reserve_reference_candidates,
)

SEED = 20250814


def _write_jsonl(path: Path, texts: list[str]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for value in texts:
            handle.write(json.dumps({"text": value}, ensure_ascii=False) + "\n")
    return path


def _write_eligibility(
    directory: Path,
    entries: dict[Path, list[int]],
    *,
    total_rows: dict[Path, int],
    corrupt_hash: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for index, (source_path, rows) in enumerate(entries.items()):
        raw = array.array("I", rows).tobytes()
        name = f"excluded_{index}.u32.raw"
        (directory / name).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if corrupt_hash:
            digest = "0" * 64
        files[str(source_path.resolve())] = {
            "release_key": source_path.stem,
            "total_rows": total_rows[source_path],
            "excluded_rows": len(rows),
            "eligible_rows": total_rows[source_path] - len(rows),
            "excluded_row_indices_file": name,
            "excluded_row_indices_sha256": digest,
        }
    manifest = {
        "schema_version": 1,
        "kind": ELIGIBILITY_MANIFEST_KIND,
        "provenance": {"test": True},
        "files": files,
    }
    path = directory / "eligibility_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _reserved_hashes(reserve_dir: Path) -> set[str]:
    payload = json.loads((reserve_dir / RESERVE_MANIFEST_NAME).read_text(encoding="utf-8"))
    return {
        document["cleaned_text_sha256"]
        for source in payload["sources"].values()
        for document in source["reserved_documents"]
    }


def _build(
    tmp_path: Path,
    label: str,
    source_spec: str,
    *,
    eligibility: Path | None = None,
    target_tokens: int = 4,
    bytes_per_token: float = 10.0,
) -> tuple[Path, dict]:
    excluded, provenance = (None, None)
    if eligibility is not None:
        excluded, provenance = load_reserve_eligibility(eligibility)
    out_dir = tmp_path / f"reserve_{label}"
    manifest = reserve_reference_candidates(
        sources=parse_reference_sources([f"{source_spec}:{target_tokens}"]),
        out_dir=out_dir,
        seed=SEED,
        reserve_bytes_per_target_token=bytes_per_token,
        excluded_rows_by_path=excluded,
        eligibility_provenance=provenance,
    )
    return out_dir, manifest


# ---------------------------------------------------------------- A / B: filter before ranking
@pytest.mark.parametrize("stage", ["d2", "d3"])
def test_ineligible_row_is_filtered_before_ranking_and_replaced(tmp_path: Path, stage: str):
    """A dropped document that would otherwise rank into the reserve must be absent, and the next
    eligible document must take its place -- proving the filter runs BEFORE ranking rather than
    trimming the result afterwards."""
    texts = [f"reference document {index} with stable body text." for index in range(60)]
    source = _write_jsonl(tmp_path / "source.jsonl", texts)

    unfiltered_dir, _ = _build(tmp_path, "unfiltered", str(source))
    baseline = _reserved_hashes(unfiltered_dir)
    assert baseline

    # Pick a row that the unfiltered build actually selected.
    from pretrain.build_pretrain_shards import cleaned_text_sha256

    selected_rows = [
        index for index, value in enumerate(texts) if cleaned_text_sha256(value) in baseline
    ]
    assert selected_rows, "expected at least one selected row to drop"
    victim = selected_rows[0]

    eligibility = _write_eligibility(
        tmp_path / f"elig_{stage}",
        {source: [victim]},
        total_rows={source: len(texts)},
    )
    filtered_dir, manifest = _build(
        tmp_path, f"filtered_{stage}", str(source), eligibility=eligibility
    )
    filtered = _reserved_hashes(filtered_dir)

    assert cleaned_text_sha256(texts[victim]) not in filtered
    # capacity is still met, so a different eligible document replaced it
    assert filtered - baseline, "an eligible replacement should have entered the reserve"
    assert manifest["eligibility"]["applied"] is True
    assert manifest["eligibility"]["excluded_rows_total"] == 1
    scan = next(iter(manifest["sources"].values()))["scan"]
    assert scan["excluded_ineligible_rows"] == 1
    # the excluded row never reached the eligibility/cleaning counters
    assert scan["eligible_documents"] == len(texts) - 1


# ---------------------------------------------------------------- C: physical order independence
def test_reserve_membership_is_independent_of_physical_row_order(tmp_path: Path):
    texts = [f"order independent reference body {index}." for index in range(50)]
    forward = _write_jsonl(tmp_path / "forward.jsonl", texts)
    reverse = _write_jsonl(tmp_path / "reverse.jsonl", list(reversed(texts)))

    forward_elig = _write_eligibility(
        tmp_path / "elig_fwd", {forward: [0, 1]}, total_rows={forward: len(texts)}
    )
    # same two documents, at their mirrored physical positions
    reverse_elig = _write_eligibility(
        tmp_path / "elig_rev",
        {reverse: sorted([len(texts) - 1, len(texts) - 2])},
        total_rows={reverse: len(texts)},
    )
    a, _ = _build(tmp_path, "fwd", str(forward), eligibility=forward_elig)
    b, _ = _build(tmp_path, "rev", str(reverse), eligibility=reverse_elig)
    assert _reserved_hashes(a) == _reserved_hashes(b)


# ---------------------------------------------------------------- D: mutated exclusion artifact
def test_mutated_eligibility_index_fails_closed(tmp_path: Path):
    texts = [f"binding document {index}." for index in range(20)]
    source = _write_jsonl(tmp_path / "source.jsonl", texts)
    eligibility = _write_eligibility(
        tmp_path / "elig",
        {source: [3]},
        total_rows={source: len(texts)},
        corrupt_hash=True,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_reserve_eligibility(eligibility)


def test_eligibility_index_out_of_range_fails_closed(tmp_path: Path):
    texts = [f"range document {index}." for index in range(10)]
    source = _write_jsonl(tmp_path / "source.jsonl", texts)
    eligibility = _write_eligibility(
        tmp_path / "elig", {source: [99]}, total_rows={source: len(texts)}
    )
    with pytest.raises(ValueError, match="out of range"):
        load_reserve_eligibility(eligibility)


def test_eligibility_manifest_kind_is_checked(tmp_path: Path):
    path = tmp_path / "not_eligibility.json"
    path.write_text(json.dumps({"kind": "something_else", "files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a reference-reserve eligibility manifest"):
        load_reserve_eligibility(path)


# ---------------------------------------------------------------- E: no silent unfiltered fallback
def test_production_cli_refuses_to_run_without_an_eligibility_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    from pretrain import build_reference_val_shards as builder

    source = _write_jsonl(tmp_path / "source.jsonl", [f"doc {i}." for i in range(20)])
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_reference_val_shards.py",
            "reserve",
            "--source",
            f"{source}:4",
            "--out_dir",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        builder.main()
    assert "--eligibility_manifest is required" in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_escape_hatch_and_manifest_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pretrain import build_reference_val_shards as builder

    source = _write_jsonl(tmp_path / "source.jsonl", [f"doc {i}." for i in range(20)])
    eligibility = _write_eligibility(tmp_path / "elig", {source: [0]}, total_rows={source: 20})
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_reference_val_shards.py",
            "reserve",
            "--source",
            f"{source}:4",
            "--out_dir",
            str(tmp_path / "out"),
            "--eligibility_manifest",
            str(eligibility),
            "--allow_unfiltered_sources",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        builder.main()
    assert "mutually exclusive" in str(excinfo.value)


def test_eligibility_manifest_must_cover_every_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from pretrain import build_reference_val_shards as builder

    covered = _write_jsonl(tmp_path / "covered.jsonl", [f"a {i}." for i in range(20)])
    uncovered = _write_jsonl(tmp_path / "uncovered.jsonl", [f"b {i}." for i in range(20)])
    eligibility = _write_eligibility(tmp_path / "elig", {covered: [0]}, total_rows={covered: 20})
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_reference_val_shards.py",
            "reserve",
            "--source",
            f"{covered}:4",
            "--source",
            f"{uncovered}:4",
            "--out_dir",
            str(tmp_path / "out"),
            "--eligibility_manifest",
            str(eligibility),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        builder.main()
    assert "does not cover every --source release" in str(excinfo.value)


# ---------------------------------------------------------------- F: ineligible rows are inert
def test_adding_an_ineligible_row_does_not_perturb_eligible_membership(tmp_path: Path):
    texts = [f"inert document {index}." for index in range(40)]
    clean = _write_jsonl(tmp_path / "clean.jsonl", texts)
    clean_elig = _write_eligibility(
        tmp_path / "elig_clean", {clean: []}, total_rows={clean: len(texts)}
    )
    base_dir, _ = _build(tmp_path, "clean", str(clean), eligibility=clean_elig)

    # same corpus plus one extra row that is then declared ineligible
    with_extra = _write_jsonl(
        tmp_path / "with_extra.jsonl", [*texts, "an ineligible intruder document."]
    )
    extra_elig = _write_eligibility(
        tmp_path / "elig_extra",
        {with_extra: [len(texts)]},
        total_rows={with_extra: len(texts) + 1},
    )
    extra_dir, _ = _build(tmp_path, "extra", str(with_extra), eligibility=extra_elig)
    assert _reserved_hashes(base_dir) == _reserved_hashes(extra_dir)


# ---------------------------------------------------------------- G: frozen semantics unchanged
def test_frozen_selection_semantics_are_unchanged_by_the_fix(tmp_path: Path):
    """With an empty exclusion set the reserve must be byte-identical to the unfiltered build."""
    texts = [f"frozen semantics document {index}." for index in range(45)]
    source = _write_jsonl(tmp_path / "source.jsonl", texts)

    unfiltered_dir, unfiltered = _build(tmp_path, "plain", str(source))
    elig = _write_eligibility(tmp_path / "elig", {source: []}, total_rows={source: len(texts)})
    filtered_dir, filtered = _build(tmp_path, "empty_filter", str(source), eligibility=elig)

    assert _reserved_hashes(unfiltered_dir) == _reserved_hashes(filtered_dir)
    assert unfiltered["selection"] == filtered["selection"]
    assert unfiltered["cleaning"] == filtered["cleaning"]
    assert unfiltered["selection"]["seed"] == SEED
    assert unfiltered["selection"]["algorithm"] == "blake2b-128-seed-plus-cleaned-sha256-v1"
    assert (unfiltered_dir / EXCLUSION_MANIFEST_NAME).read_bytes() == (
        filtered_dir / EXCLUSION_MANIFEST_NAME
    ).read_bytes()
    assert unfiltered["eligibility"]["applied"] is False
    assert filtered["eligibility"]["applied"] is True


# ---------------------------------------------------------------- multi-release family (union)
def test_multi_release_family_is_ranked_as_one_union_without_an_internal_quota(tmp_path: Path):
    """The structured family spans two releases; it must be ranked as ONE pool, so membership is
    whatever the content hash produces and no per-release split is implied."""
    left = _write_jsonl(tmp_path / "left.jsonl", [f"left body {i}." for i in range(30)])
    right = _write_jsonl(tmp_path / "right.jsonl", [f"right body {i}." for i in range(30)])
    combined = _write_jsonl(
        tmp_path / "combined.jsonl",
        [f"left body {i}." for i in range(30)] + [f"right body {i}." for i in range(30)],
    )
    elig_pair = _write_eligibility(
        tmp_path / "elig_pair",
        {left: [], right: []},
        total_rows={left: 30, right: 30},
    )
    elig_combined = _write_eligibility(
        tmp_path / "elig_comb", {combined: []}, total_rows={combined: 60}
    )
    pair_dir, pair_manifest = _build(
        tmp_path, "pair", f"{left},{right}", eligibility=elig_pair, target_tokens=6
    )
    comb_dir, _ = _build(
        tmp_path, "comb", str(combined), eligibility=elig_combined, target_tokens=6
    )

    # one union ranking: identical to ranking the concatenation
    assert _reserved_hashes(pair_dir) == _reserved_hashes(comb_dir)
    entry = next(iter(pair_manifest["sources"].values()))
    assert entry["release_count"] == 2
    assert len(entry["resolved_paths"]) == 2
    assert len(entry["source_fingerprints"]) == 2
