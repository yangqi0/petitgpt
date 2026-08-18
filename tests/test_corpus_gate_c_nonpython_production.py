"""Focused contract tests for the Non-Python Gate C production candidate builder.

No network, no tokenizer, no GPU, no real corpus.  These tests only cover what distinguishes
*production* semantics from the accepted bounded C0 diagnostic: scope honesty, complete traversal,
frozen filter reuse, resume, publication, and honest resource accounting.  The C0 verifier suites
already cover the shared transport and filter contracts and are not re-litigated here.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain import (  # noqa: E402
    corpus_gate_c as gc,
    corpus_gate_c_nonpython_production as prod,
    corpus_gate_c_parquet as gp,
)

RUNS_ROOT = PROJECT_ROOT / "runs"
SOURCE = "finewiki_en"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _prose(index: int, sentences: int = 16) -> str:
    return " ".join(
        f"Observation {index}-{position} recorded a value of {index * 31 + position * 7} "
        f"during trial {position}."
        for position in range(sentences)
    )


def finewiki_rows(start: int, count: int) -> list[dict[str, object]]:
    return [
        {
            "bytes_html": 4096,
            "date_modified": "2024-01-01",
            "has_math": False,
            "id": f"finewiki-{index}",
            "in_language": "en",
            "infoboxes": "",
            "page_id": 1000 + index,
            "text": f"Topic {index}. {_prose(index)}",
            "title": f"Title {index}",
            "url": f"https://en.wikipedia.org/wiki/Topic_{index}",
            "version": 1,
            "wikidata_id": f"Q{index}",
            "wikiname": "enwiki",
            "wikitext": "== Heading ==",
        }
        for index in range(start, start + count)
    ]


def write_parquet(path: Path, rows: list[dict[str, object]], row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


class LocalStore(gp.ObjectStore):
    """Serves pinned objects from a local directory; no network."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.opened: list[str] = []

    def list_objects(self, binding: gp.ParquetBinding) -> tuple[gp.RemoteObject, ...]:
        base = self.root / binding.path_prefix.rstrip("/")
        objects = []
        for item in sorted(base.glob("*.parquet")):
            payload = item.read_bytes()
            objects.append(
                gp.RemoteObject(
                    path=f"{binding.path_prefix}{item.name}",
                    size=len(payload),
                    oid=hashlib.sha256(payload).hexdigest(),
                    etag=hashlib.sha1(payload, usedforsecurity=False).hexdigest(),
                )
            )
        if not objects:
            raise gp.GateCParquetError("no local fixtures")
        return tuple(objects)

    @contextmanager
    def open(self, binding: gp.ParquetBinding, path: str):
        self.opened.append(path)
        handle = open(self.root / path, "rb")
        reader = gp._CountingReader(handle)
        try:
            yield reader
        finally:
            handle.close()


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    prefix = gp.PARQUET_BINDINGS[SOURCE].path_prefix
    write_parquet(root / prefix / "000.parquet", finewiki_rows(0, 300), 100)
    write_parquet(root / prefix / "001.parquet", finewiki_rows(300, 300), 100)
    write_parquet(root / prefix / "002.parquet", finewiki_rows(600, 300), 100)
    return root


@pytest.fixture
def run_dir(tmp_path_factory) -> Path:
    root = RUNS_ROOT / "_pytest_gate_c_nonpython_production"
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(tmp_path_factory.mktemp("case", numbered=True)).name
    target.mkdir(parents=True, exist_ok=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def make_config(run_dir: Path, **overrides) -> prod.ProductionConfig:
    base = {
        "source_key": SOURCE,
        "output_dir": run_dir / "release",
        "work_dir": run_dir / "work",
        "checkpoint_every": 7,
        "batch_rows": 64,
    }
    base.update(overrides)
    return prod.ProductionConfig(**base)


def build(run_dir: Path, store_root: Path, **overrides) -> dict:
    return prod.build_production_candidates(
        make_config(run_dir, **overrides), store=LocalStore(store_root)
    )


def released(run_dir: Path) -> list[dict]:
    path = run_dir / "release" / prod.DOCUMENTS_NAME
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def manifest_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "release" / prod.MANIFEST_NAME).read_text())


def _restamp_checksums(run_dir: Path) -> None:
    """Re-sign MANIFEST.sha256 so a tampering test exercises the contract, not the checksums."""
    release = run_dir / "release"
    lines = []
    for entry in (release / prod.CHECKSUMS_NAME).read_text().splitlines():
        _, name = entry.split("  ", 1)
        digest = hashlib.sha256((release / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (release / prod.CHECKSUMS_NAME).write_text("\n".join(lines) + "\n")


def _rewrite_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "release" / prod.MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _restamp_checksums(run_dir)


def _rewrite_transport(run_dir: Path, *, drop: tuple = (), add: dict | None = None) -> None:
    path = run_dir / "release" / prod.TRANSPORT_MANIFEST_NAME
    transport = json.loads(path.read_text())
    for key in drop:
        transport.pop(key, None)
    transport.update(add or {})
    transport.pop("manifest_sha256", None)
    transport["manifest_sha256"] = prod.scoped_transport_manifest_sha256(transport)
    path.write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((run_dir / "release" / prod.MANIFEST_NAME).read_text())
    manifest["transport_manifest_sha256"] = transport["manifest_sha256"]
    _rewrite_manifest(run_dir, manifest)


# --------------------------------------------------------------------------------------
# Frozen contract reuse — production must not fork the filters
# --------------------------------------------------------------------------------------


def test_production_reuses_the_frozen_gate_c_filter():
    assert prod.evaluate_row is gc.evaluate_row
    assert prod.SOURCES is gc.SOURCES
    for key in gp.PARQUET_BINDINGS:
        contract = prod.filter_contract(key)
        spec = gc.SOURCES[key]
        assert contract["body_path"] == spec.body_path
        assert contract["min_bytes"] == spec.min_bytes
        assert contract["max_bytes"] == spec.max_bytes
        assert contract["evaluator"] == "pretrain.corpus_gate_c.evaluate_row"
        assert len(prod.filter_contract_sha256(key)) == 64


def test_all_seven_frozen_sources_are_buildable():
    assert set(gp.PARQUET_BINDINGS) == {
        "fineweb_edu_dedup",
        "dclm_edu",
        "finewiki_en",
        "pes2o",
        "stackexchange",
        "cosmopedia_v2",
        "finephrase_tutorial",
    }


def test_production_release_identity_differs_from_c0():
    assert prod.RELEASE_KIND == "nonpython_gate_c_production_candidate"
    assert prod.RELEASE_KIND != "c0_diagnostic"
    assert prod.PRODUCTION_TOOL_SCHEMA_VERSION != gp.TOOL_SCHEMA_VERSION


def test_production_does_not_import_the_c0_sampler(run_dir, store_root):
    """A production population must not inherit the diagnostic sampler's claim."""
    build(run_dir, store_root)
    manifest = manifest_of(run_dir)
    assert manifest["sampler"] is None
    assert "sampler_version" not in manifest
    assert manifest["candidate_scope"]["row_level_sampling"] is False
    # The frozen C0 sampler constant is untouched.
    assert gp.SAMPLER_VERSION == "pps-rowgroup-fixed-head-cluster-v2"


def test_c0_ceilings_are_not_applied_to_production():
    fields = prod.ProductionConfig.__dataclass_fields__
    for banned in ("target_documents", "max_scanned", "units", "rows_per_unit", "seed"):
        assert banned not in fields
    assert gp.MAX_ACCEPTED_DOCUMENTS == 8192
    assert gp.MAX_SCANNED_RECORDS == 50_000


def test_production_exceeds_the_c0_document_ceiling(run_dir, tmp_path):
    root = tmp_path / "hub"
    prefix = gp.PARQUET_BINDINGS[SOURCE].path_prefix
    write_parquet(root / prefix / "000.parquet", finewiki_rows(0, 9000), 1000)
    result = build(run_dir, root)
    assert result["accepted"] > gp.MAX_ACCEPTED_DOCUMENTS
    assert result["scanned"] > gp.MAX_ACCEPTED_DOCUMENTS


# --------------------------------------------------------------------------------------
# Scope honesty
# --------------------------------------------------------------------------------------


def test_full_source_scope_claims_full_traversal(run_dir, store_root):
    result = build(run_dir, store_root)
    assert result["full_source_traversal"] is True
    assert result["files_in_scope"] == 3
    assert result["pinned_file_count"] == 3
    assert result["scanned"] == 900
    manifest = manifest_of(run_dir)
    scope = manifest["candidate_scope"]
    assert scope["mode"] == "full_source"
    assert scope["representativeness_claim"] == "complete population of the pinned source"
    assert manifest["all_scope_files_complete"] is True


def test_file_prefix_scope_refuses_a_full_source_claim(run_dir, store_root):
    result = build(run_dir, store_root, scope_files=2)
    assert result["full_source_traversal"] is False
    assert result["files_in_scope"] == 2
    assert result["pinned_file_count"] == 3
    assert result["scanned"] == 600
    manifest = manifest_of(run_dir)
    scope = manifest["candidate_scope"]
    assert scope["mode"] == "file_prefix"
    assert "NOT the complete source population" in scope["representativeness_claim"]
    assert "NOT a row-level representative sample" in scope["representativeness_claim"]
    assert manifest["scope_byte_share_of_pinned_source"] < 1.0


def test_bounded_scope_is_still_complete_within_its_scope(run_dir, store_root):
    build(run_dir, store_root, scope_files=2)
    documents = released(run_dir)
    files = {document["file_path"] for document in documents}
    assert len(files) == 2
    manifest = manifest_of(run_dir)
    assert manifest["all_scope_files_complete"] is True
    assert all(entry["complete"] for entry in manifest["per_file"].values())


def test_scope_cannot_exceed_the_pinned_file_count(run_dir, store_root):
    result = build(run_dir, store_root, scope_files=99)
    assert result["files_in_scope"] == 3
    assert result["full_source_traversal"] is True


def test_full_source_scope_rejects_a_short_file_count():
    with pytest.raises(prod.GateCProductionError, match="must cover every pinned file"):
        prod.CandidateScope(
            mode="full_source", file_count=2, total_file_count=3, file_list_sha256="0" * 64
        )


# --------------------------------------------------------------------------------------
# Complete traversal
# --------------------------------------------------------------------------------------


def test_every_row_of_every_in_scope_file_is_evaluated(run_dir, store_root):
    result = build(run_dir, store_root)
    manifest = manifest_of(run_dir)
    assert result["scanned"] == sum(entry["scanned"] for entry in manifest["per_file"].values())
    for entry in manifest["per_file"].values():
        assert entry["scanned"] == entry["pinned_rows"]
        assert entry["accepted"] + entry["rejected"] == entry["scanned"]


def test_row_group_topology_drift_fails_closed(run_dir, store_root):
    prod.build_production_candidates(
        make_config(run_dir, stop_after_documents=5), store=LocalStore(store_root)
    )
    transport_path = run_dir / "work" / prod.TRANSPORT_MANIFEST_NAME
    transport = json.loads(transport_path.read_text())
    transport["files"][0]["row_group_rows"][0] += 1
    transport["manifest_sha256"] = prod.scoped_transport_manifest_sha256(transport)
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True))
    with pytest.raises(prod.GateCProductionError, match="row count drifted|fingerprint mismatch"):
        build(run_dir, store_root)


def test_schema_drift_fails_closed(run_dir, store_root):
    prod.build_production_candidates(
        make_config(run_dir, stop_after_documents=5), store=LocalStore(store_root)
    )
    transport_path = run_dir / "work" / prod.TRANSPORT_MANIFEST_NAME
    transport = json.loads(transport_path.read_text())
    transport["schema_hash"] = "0" * 64
    transport["manifest_sha256"] = prod.scoped_transport_manifest_sha256(transport)
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True))
    with pytest.raises(prod.GateCProductionError, match="schema drift|fingerprint mismatch"):
        build(run_dir, store_root)


def test_transport_manifest_checksum_corruption_fails_closed(run_dir, store_root):
    prod.build_production_candidates(
        make_config(run_dir, stop_after_documents=5), store=LocalStore(store_root)
    )
    transport_path = run_dir / "work" / prod.TRANSPORT_MANIFEST_NAME
    transport = json.loads(transport_path.read_text())
    transport["scope_rows"] = 1
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True))
    with pytest.raises(prod.GateCProductionError, match="transport manifest checksum mismatch"):
        build(run_dir, store_root)


# --------------------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------------------


def test_duplicate_text_across_files_is_rejected_once(run_dir, tmp_path):
    root = tmp_path / "hub"
    prefix = gp.PARQUET_BINDINGS[SOURCE].path_prefix
    shared = finewiki_rows(0, 50)
    other = [dict(row, id=f"other-{i}", page_id=90_000 + i) for i, row in enumerate(shared)]
    write_parquet(root / prefix / "000.parquet", shared, 25)
    write_parquet(root / prefix / "001.parquet", other, 25)
    result = build(run_dir, root)
    assert result["accepted"] == 50
    assert result["rejections"]["duplicate_text_sha256"] == 50


# --------------------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------------------


def test_resume_reproduces_the_uninterrupted_release(run_dir, store_root, tmp_path_factory):
    reference = RUNS_ROOT / "_pytest_gate_c_nonpython_production" / "reference"
    shutil.rmtree(reference, ignore_errors=True)
    reference.mkdir(parents=True, exist_ok=True)
    try:
        single = prod.build_production_candidates(
            make_config(reference), store=LocalStore(store_root)
        )
        partial = build(run_dir, store_root, stop_after_documents=137)
        assert partial["published"] is False
        assert partial["accepted"] == 137

        final = build(run_dir, store_root)
        assert final["published"] is True
        assert final["resumed"] is True
        assert final["resume_count"] == 1
        assert final["accepted"] == single["accepted"]
        assert final["documents_sha256"] == single["documents_sha256"]
        ids = [document["source_record_id"] for document in released(run_dir)]
        assert len(set(ids)) == len(ids)
    finally:
        shutil.rmtree(reference, ignore_errors=True)


def test_resume_rebuilds_dedup_from_the_committed_prefix(run_dir, tmp_path):
    root = tmp_path / "hub"
    prefix = gp.PARQUET_BINDINGS[SOURCE].path_prefix
    shared = finewiki_rows(0, 10)
    other = [dict(row, id=f"other-{i}", page_id=90_000 + i) for i, row in enumerate(shared)]
    write_parquet(root / prefix / "000.parquet", shared, 5)
    write_parquet(root / prefix / "001.parquet", other, 5)
    build(run_dir, root, stop_after_documents=10)
    final = build(run_dir, root)
    assert final["published"] is True
    assert final["accepted"] == 10
    assert final["rejections"]["duplicate_text_sha256"] == 10


def test_uncommitted_suffix_is_truncated_on_resume(run_dir, store_root):
    build(run_dir, store_root, stop_after_documents=50, checkpoint_every=10_000)
    staged = prod._staging_path(run_dir / "release") / prod.DOCUMENTS_NAME
    staged.write_bytes(staged.read_bytes() + b'{"partially":"written"}\n')
    final = build(run_dir, store_root)
    assert final["published"] is True
    assert all("partially" not in json.dumps(document) for document in released(run_dir))


def test_checkpoint_checksum_corruption_refuses_resume(run_dir, store_root):
    build(run_dir, store_root, stop_after_documents=20)
    checkpoint = run_dir / "work" / prod.CHECKPOINT_NAME
    state = json.loads(checkpoint.read_text())
    state["counters"]["accepted"] = 999
    checkpoint.write_text(json.dumps(state))
    with pytest.raises(prod.GateCProductionError, match="checkpoint checksum mismatch"):
        build(run_dir, store_root)


def test_scope_change_refuses_resume(run_dir, store_root):
    build(run_dir, store_root, scope_files=2, stop_after_documents=20)
    # A different scope is a different population: the persisted transport manifest pins it,
    # so the mismatch must fail closed rather than silently reuse the earlier scope.
    with pytest.raises(prod.GateCProductionError, match="requested scope does not match"):
        build(run_dir, store_root, scope_files=3)


# --------------------------------------------------------------------------------------
# Partial traversal never publishes
# --------------------------------------------------------------------------------------


def test_stop_after_documents_publishes_nothing(run_dir, store_root):
    result = build(run_dir, store_root, stop_after_documents=25)
    assert result["published"] is False
    assert not (run_dir / "release").exists()


def test_time_capped_run_publishes_nothing(run_dir, store_root):
    ticks = iter([0.0] + [1000.0] * 100_000)
    result = prod.build_production_candidates(
        make_config(run_dir, max_wall_seconds=1.0),
        store=LocalStore(store_root),
        clock=lambda: next(ticks),
    )
    assert result["published"] is False
    assert result["stop_reason"] == "time_cap"
    assert not (run_dir / "release").exists()


# --------------------------------------------------------------------------------------
# Publication and verification
# --------------------------------------------------------------------------------------


def test_publication_is_atomic_and_leaves_no_staging(run_dir, store_root):
    build(run_dir, store_root)
    release = run_dir / "release"
    assert sorted(path.name for path in release.iterdir()) == [
        prod.CHECKSUMS_NAME,
        prod.DOCUMENTS_NAME,
        prod.MANIFEST_NAME,
        prod.TRANSPORT_MANIFEST_NAME,
    ]
    assert not prod._staging_path(release).exists()


def test_publication_refuses_to_overwrite(run_dir, store_root):
    build(run_dir, store_root)
    with pytest.raises(prod.GateCProductionError, match="refusing to overwrite published output"):
        build(run_dir, store_root)


def test_release_declares_candidate_not_final_corpus(run_dir, store_root):
    build(run_dir, store_root)
    manifest = manifest_of(run_dir)
    assert manifest["release_kind"] == "nonpython_gate_c_production_candidate"
    assert manifest["promotion_eligible"] is False
    assert manifest["proxy_token_label"] == "NOT CANONICAL TOKEN COUNT"
    assert manifest["gate_c_scope"]["cross_source_near_dedup"] is False
    assert manifest["gate_c_scope"]["benchmark_decontamination"] is False
    assert manifest["gate_c_scope"]["reference_reserve_exclusion"] is False
    assert manifest["gate_c_scope"]["final_source_quota_applied"] is False
    assert manifest["gate_c_scope"]["stage_a_stage_b_split_performed"] is False
    assert all(value is False for value in manifest["hard_stops"].values())


def test_verify_accepts_a_freshly_published_release(run_dir, store_root):
    result = build(run_dir, store_root)
    verified = prod.verify_release(run_dir / "release")
    assert verified["accepted"] == result["accepted"]
    assert verified["documents_sha256"] == result["documents_sha256"]
    assert verified["distinct_record_ids"] == verified["accepted"]
    assert verified["distinct_text_hashes"] == verified["accepted"]
    assert verified["full_source_traversal"] is True


def test_verify_detects_tampered_text(run_dir, store_root):
    build(run_dir, store_root)
    documents = run_dir / "release" / prod.DOCUMENTS_NAME
    lines = documents.read_bytes().split(b"\n")
    record = json.loads(lines[0])
    record["text"] = record["text"] + " tampered"
    lines[0] = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
    documents.write_bytes(b"\n".join(lines))
    with pytest.raises(prod.GateCProductionError):
        prod.verify_release(run_dir / "release")


def test_verify_rejects_a_manifest_claiming_a_later_stage(run_dir, store_root):
    build(run_dir, store_root)
    path = run_dir / "release" / prod.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["gate_c_scope"]["cross_source_near_dedup"] = True
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(prod.GateCProductionError, match="must be false"):
        prod.verify_release(run_dir / "release")


def test_verify_rejects_a_sampler_claim(run_dir, store_root):
    build(run_dir, store_root)
    path = run_dir / "release" / prod.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["sampler"] = "pps-rowgroup-fixed-head-cluster-v2"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(prod.GateCProductionError, match="must not carry a sampler claim"):
        prod.verify_release(run_dir / "release")


# --------------------------------------------------------------------------------------
# Object-integrity manifest honesty
# --------------------------------------------------------------------------------------


def test_transport_manifest_does_not_deny_full_object_download(run_dir, store_root):
    """The superseded claim "no pinned object is downloaded in full" must be gone.

    Production traversal materializes, size-checks and SHA-256 hashes every in-scope object
    before evaluating a row, so a manifest asserting the opposite is a false provenance claim.
    """
    build(run_dir, store_root)
    transport = json.loads((run_dir / "release" / prod.TRANSPORT_MANIFEST_NAME).read_text())
    assert "local_full_file_sha256_computed" not in transport
    assert "downloaded in full" not in transport["identity_note"]
    assert transport["preflight_full_file_sha256_computed"] is False
    contract = transport["production_traversal_object_integrity"]
    assert contract["objects_materialized_completely_before_row_evaluation"] is True
    assert contract["expected_size_verified"] is True
    assert contract["local_full_file_sha256_computed"] is True


def test_release_states_every_in_scope_object_was_integrity_verified(run_dir, store_root):
    build(run_dir, store_root)
    manifest = manifest_of(run_dir)
    claim = manifest["object_integrity"]
    assert claim["local_full_file_sha256_computed"] is True
    assert claim["all_scope_files_integrity_verified"] is True
    assert claim["scope_files_integrity_verified"] == manifest["files_in_scope"] == 3
    assert all(entry["integrity_verified"] is True for entry in manifest["per_file"].values())
    assert manifest["resources"]["measured"]["object_integrity_verifications"] >= 3


def test_oid_comparability_is_reported_not_assumed(run_dir, store_root):
    """The LFS-oid comparison only happens when the pinned oid is itself a SHA-256."""
    build(run_dir, store_root)
    transport = json.loads((run_dir / "release" / prod.TRANSPORT_MANIFEST_NAME).read_text())
    contract = transport["production_traversal_object_integrity"]
    # The local fixture store publishes real SHA-256 oids, so the comparison does happen.
    assert contract["pinned_oids_are_sha256"] is True
    assert contract["local_sha256_compared_with_pinned_oid"] is True
    assert manifest_of(run_dir)["object_integrity"]["local_sha256_compared_with_pinned_oid"] is True


def test_verify_rejects_the_pre_fix_contradictory_transport_metadata(run_dir, store_root):
    """A release cannot claim no full download while its traversal hashed every object."""
    build(run_dir, store_root)
    _rewrite_transport(
        run_dir,
        drop=("preflight_full_file_sha256_computed", "production_traversal_object_integrity"),
        add={
            "local_full_file_sha256_computed": False,
            "identity_note": (
                "identity is the Hub LFS OID plus blob etag at the pinned commit; no pinned "
                "object is downloaded in full, so no local full-file SHA-256 is claimed"
            ),
        },
    )
    with pytest.raises(prod.GateCProductionError, match="superseded top-level"):
        prod.verify_release(run_dir / "release")


def test_verify_rejects_a_downgraded_production_integrity_contract(run_dir, store_root):
    build(run_dir, store_root)
    _rewrite_transport(
        run_dir,
        add={
            "production_traversal_object_integrity": {
                "objects_materialized_completely_before_row_evaluation": False,
                "expected_size_verified": True,
                "local_full_file_sha256_computed": False,
                "pinned_oids_are_sha256": True,
                "local_sha256_compared_with_pinned_oid": True,
                "note": "weakened",
            }
        },
    )
    with pytest.raises(prod.GateCProductionError, match="production object integrity"):
        prod.verify_release(run_dir / "release")


def test_verify_rejects_an_unverified_in_scope_object(run_dir, store_root):
    build(run_dir, store_root)
    path = run_dir / "release" / prod.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    first = sorted(manifest["per_file"])[0]
    manifest["per_file"][first]["integrity_verified"] = False
    _rewrite_manifest(run_dir, manifest)
    with pytest.raises(prod.GateCProductionError, match="integrity-verify every in-scope object"):
        prod.verify_release(run_dir / "release")


def test_verify_rejects_an_object_integrity_claim_the_counters_do_not_support(run_dir, store_root):
    build(run_dir, store_root)
    manifest = json.loads((run_dir / "release" / prod.MANIFEST_NAME).read_text())
    manifest["resources"]["measured"]["object_integrity_verifications"] = 1
    _rewrite_manifest(run_dir, manifest)
    with pytest.raises(prod.GateCProductionError, match="below one per in-scope object"):
        prod.verify_release(run_dir / "release")


def test_corrected_release_carries_the_post_fix_spec_version(run_dir, store_root):
    build(run_dir, store_root)
    assert manifest_of(run_dir)["spec_version"].endswith("2026-08-18")


# --------------------------------------------------------------------------------------
# Honest resource accounting
# --------------------------------------------------------------------------------------


def test_unmeasured_network_request_count_is_null_never_zero(run_dir, store_root):
    build(run_dir, store_root)
    resources = manifest_of(run_dir)["resources"]
    assert resources["network_request_count"] is None
    assert resources["network_request_count_measured"] is False
    assert "network_request_count" not in resources["measured"]
    assert "network_request_count" in resources["unmeasured_fields"]


def test_measured_resource_counters_are_exact(run_dir, store_root):
    build(run_dir, store_root)
    measured = manifest_of(run_dir)["resources"]["measured"]
    assert set(measured) == set(prod.ResourceAccounting.MEASURED_FIELDS)
    assert all(isinstance(value, int) for value in measured.values())
    assert measured["parquet_files_opened"] == 3
    assert measured["parquet_footers_read"] == 3
    assert measured["row_groups_read"] == 9
    assert measured["objects_downloaded"] == 3
    assert measured["object_cache_reuses"] == 0
    assert measured["object_integrity_verifications"] == 3
    assert measured["downloaded_bytes"] > 0
    assert measured["resume_reread_row_groups"] == 0


def test_resume_reread_row_groups_is_measured(run_dir, store_root):
    build(run_dir, store_root, stop_after_documents=137)
    result = build(run_dir, store_root)
    assert result["resources"]["measured"]["resume_reread_row_groups"] >= 1


# --------------------------------------------------------------------------------------
# Diagnostics and CLI
# --------------------------------------------------------------------------------------


def test_diagnostics_describe_the_population(run_dir, store_root):
    build(run_dir, store_root)
    diagnostics = prod.diagnose_release(run_dir / "release")
    assert diagnostics["documents"] == 900
    assert diagnostics["proxy_token_label"] == "NOT CANONICAL TOKEN COUNT"
    assert diagnostics["canonical_token_counting_performed"] is False
    assert diagnostics["length_bytes"]["n"] == 900
    assert diagnostics["file_distribution"]["files_covered"] == 3
    assert diagnostics["release"]["full_source_traversal"] is True


def test_cli_reports_expected_failures_as_json_without_a_traceback(run_dir, capsys):
    code = prod.main([
        "build",
        "--source",
        SOURCE,
        "--output-dir",
        str(run_dir / "release"),
        "--work-dir",
        str(run_dir / "work"),
        "--batch-rows",
        "0",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "batch_rows" in json.loads(captured.err)["error"]


def test_cli_verify_and_diagnose_round_trip(run_dir, store_root, capsys):
    build(run_dir, store_root)
    capsys.readouterr()
    assert prod.main(["verify", "--output-dir", str(run_dir / "release")]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["release_kind"] == "nonpython_gate_c_production_candidate"

    out = run_dir / "diagnostics.json"
    assert prod.main(["diagnose", "--output-dir", str(run_dir / "release"), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["canonical_token_counting_performed"] is False
    assert prod.main(["diagnose", "--output-dir", str(run_dir / "release"), "--out", str(out)]) == 2
    assert "refusing to overwrite" in json.loads(capsys.readouterr().err)["error"]


def test_output_paths_must_be_git_ignored(tmp_path, store_root):
    config = make_config(PROJECT_ROOT / "src")
    with pytest.raises(gc.GateCError, match="not Git-ignored"):
        prod.build_production_candidates(config, store=LocalStore(store_root))


def test_pinned_object_is_verified_against_its_oid(run_dir, store_root):
    """No row may be evaluated from an object whose bytes do not match the pinned identity."""

    class TamperingStore(LocalStore):
        @contextmanager
        def open(self, binding, path):
            import io

            payload = bytearray((self.root / path).read_bytes())
            # Corrupt mid-file so the Parquet footer stays readable and the identity check,
            # not the footer parse, is what rejects the object.
            middle = len(payload) // 2
            payload[middle : middle + 64] = b"\x00" * 64
            handle = io.BytesIO(bytes(payload))
            reader = gp._CountingReader(handle)
            try:
                yield reader
            finally:
                handle.close()

    with pytest.raises(prod.GateCProductionError, match="SHA-256 does not match|size mismatch"):
        prod.build_production_candidates(make_config(run_dir), store=TamperingStore(store_root))


def test_object_cache_is_released_after_each_file(run_dir, store_root):
    build(run_dir, store_root)
    cache = run_dir / "work" / "objects"
    assert list(cache.glob("*.parquet")) == []


def test_object_cache_is_reused_when_kept(run_dir, store_root):
    build(run_dir, store_root, stop_after_documents=50, keep_object_cache=True)
    result = build(run_dir, store_root, keep_object_cache=True)
    assert result["resources"]["measured"]["object_cache_reuses"] >= 1
