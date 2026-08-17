"""Pure synthetic contract tests for the full Python Gate C candidate builder.

No network, no tokenizer, no GPU, no real corpus.  The frozen record shape and the frozen
mechanical filter are imported from the accepted C0 test fixtures so the two suites cannot drift
apart about what an upstream record looks like.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_corpus_gate_c_python import (  # noqa: E402
    SHARD_COUNT,
    _python_source,
    _repetitive_python,
    _sized_python,
    make_record,
    shard_bytes,
)

from pretrain import (  # noqa: E402
    corpus_gate_c_python as gcp,
    corpus_gate_c_python_full as full,
)

RUNS_ROOT = PROJECT_ROOT / "runs"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _sized_unique_python(total_bytes: int, index: int) -> str:
    """Valid Python of an exact byte length whose content is unique per ``index``."""
    head = f"x = '{index:08d}"
    tail = "'\n"
    filler = total_bytes - len(head) - len(tail)
    assert filler >= 0
    return head + ("a" * filler) + tail


def write_shards(root: Path, per_shard: list[list[object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for shard_index, records in enumerate(per_shard):
        (root / f"stack-edu-{shard_index:04d}.json.gz").write_bytes(shard_bytes(records))


def uniform_shards(root: Path, records_per_shard: int = 4) -> None:
    write_shards(
        root,
        [
            [
                make_record(shard * 1000 + offset, score=2.0 + 0.25 * offset)
                for offset in range(records_per_shard)
            ]
            for shard in range(SHARD_COUNT)
        ],
    )


def make_full_source(root: Path, **overrides) -> gcp.PythonSourceSpec:
    store = full.LocalShardSource(root)
    paths = sorted(path.name for path in root.glob("*.json.gz"))
    shards = tuple(
        gcp.ShardSpec(
            path=item.path, size=item.size, lfs_sha256=item.lfs_sha256, blob_id=item.blob_id
        )
        for item in (store._describe(path) for path in paths)
    )
    base = {
        "key": "common_pile_stackv2_edu_python",
        "dataset": "common-pile/stackv2_edu_filtered",
        "dataset_config": "default",
        "split": "train",
        "revision": gcp.PYTHON_SOURCE.revision,
        "body_path": "text",
        "language": "Python",
        "license": "per-record metadata.license",
        "shards": shards,
        "revision_shard_count": len(paths),
        "required_schema": gcp.PYTHON_REQUIRED_SCHEMA,
        "frozen_leaf_set": gcp.PYTHON_FROZEN_LEAF_SET,
        "nullable_paths": gcp.PYTHON_NULLABLE_PATHS,
        "natural_id_path": "id",
        "score_path": "score",
    }
    base.update(overrides)
    return gcp.PythonSourceSpec(**base)


def make_config(run_dir: Path, source: gcp.PythonSourceSpec, **overrides) -> full.FullBuildConfig:
    base = {
        "source": source,
        "output_dir": run_dir / "release",
        "work_dir": run_dir / "work",
        "cache_dir": run_dir / "cache",
        "max_wall_seconds": 300.0,
        "checkpoint_every": 3,
    }
    base.update(overrides)
    return full.FullBuildConfig(**base)


@pytest.fixture
def run_dir(tmp_path_factory) -> Path:
    """A Git-ignored working directory: Gate C refuses to write anywhere else."""
    root = RUNS_ROOT / "_pytest_gate_c_python_full"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tmp_path_factory.mktemp("case", numbered=True))
    target = root / path.name
    target.mkdir(parents=True, exist_ok=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def build(run_dir: Path, source: gcp.PythonSourceSpec, root: Path, **overrides) -> dict:
    return full.build_full_candidates(
        make_config(run_dir, source, **overrides), store=full.LocalShardSource(root)
    )


def released(run_dir: Path) -> list[dict]:
    path = run_dir / "release" / full.DOCUMENTS_NAME
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def manifest_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "release" / full.MANIFEST_NAME).read_text())


# --------------------------------------------------------------------------------------
# Frozen contract reuse
# --------------------------------------------------------------------------------------


def test_full_builder_reuses_the_frozen_c0_source_binding():
    assert full.PYTHON_SOURCE is gcp.PYTHON_SOURCE
    assert full.PYTHON_SOURCE.revision == "c354dbe88469a1153e97c6a63ac50591849654de"
    assert len(full.PYTHON_SOURCE.shards) == SHARD_COUNT
    assert [shard.path for shard in full.PYTHON_SOURCE.shards] == [
        f"stack-edu-{number:04d}.json.gz" for number in range(73, 85)
    ]


def test_full_builder_reuses_the_frozen_mechanical_filter():
    assert full.evaluate_record is gcp.evaluate_record
    assert full.FILTER_CONTRACT["min_bytes"] == gcp.PYTHON_MIN_BYTES == 200
    assert full.FILTER_CONTRACT["max_bytes"] == gcp.PYTHON_MAX_BYTES == 8192
    assert full.FILTER_CONTRACT["max_comment_blank_fraction"] == 0.70
    assert full.FILTER_CONTRACT["filter_order"] == list(gcp.FILTER_ORDER)
    assert len(full.FILTER_CONTRACT_SHA256) == 64


def test_full_release_identity_is_distinct_from_the_c0_diagnostic():
    assert full.RELEASE_KIND == "python_gate_c_full_candidate"
    assert full.RELEASE_KIND != "c0_diagnostic"
    assert full.FULL_TOOL_SCHEMA_VERSION != gcp.TOOL_SCHEMA_VERSION


def test_c0_ceilings_are_not_applied_to_the_full_build():
    config_fields = full.FullBuildConfig.__dataclass_fields__
    for banned in ("target_documents", "max_scanned", "max_shard_records", "stride"):
        assert banned not in config_fields
    assert not hasattr(full, "MAX_ACCEPTED_DOCUMENTS")
    assert not hasattr(full, "MAX_SHARD_COMPRESSED_BYTES")
    # The accepted C0 module keeps its own ceilings untouched.
    assert gcp.MAX_ACCEPTED_DOCUMENTS == 60_000
    assert gcp.MAX_SHARD_COMPRESSED_BYTES == 256 * 1024 * 1024


def test_full_build_exceeds_the_c0_document_ceiling(run_dir):
    """A population larger than the whole C0 ceiling must build without any cap intervening."""
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [make_record(shard * 100_000 + offset) for offset in range(60)]
            for shard in range(SHARD_COUNT)
        ],
    )
    result = build(run_dir, make_full_source(root), root)
    assert result["published"] is True
    assert result["accepted"] == SHARD_COUNT * 60
    assert manifest_of(run_dir)["caps"]["document_cap"] is None


# --------------------------------------------------------------------------------------
# Complete-object transport
# --------------------------------------------------------------------------------------


def test_every_physical_record_of_every_shard_is_traversed(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [make_record(shard * 100 + offset) for offset in range(9)]
            for shard in range(SHARD_COUNT)
        ],
    )
    result = build(run_dir, make_full_source(root), root)
    assert result["physical_records_seen"] == SHARD_COUNT * 9
    assert result["evaluated"] == SHARD_COUNT * 9
    assert result["shards_complete"] == SHARD_COUNT
    manifest = manifest_of(run_dir)
    assert manifest["traversal"]["all_shards_complete"] is True
    assert manifest["traversal"]["sampling"] is False
    assert manifest["traversal"]["stride"] == 1
    assert all(entry["complete"] for entry in manifest["per_shard"].values())
    assert all(entry["physical_records"] == 9 for entry in manifest["per_shard"].values())


def test_truncated_shard_object_fails_closed(run_dir, tmp_path):
    payload = shard_bytes([make_record(index) for index in range(20)])
    target = tmp_path / "truncated.json.gz"
    target.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(full.GateCPythonFullError, match="ended before its terminator"):
        list(full.iter_full_shard_records(target))


def test_unterminated_final_record_fails_closed(tmp_path):
    payload = shard_bytes([make_record(0), make_record(1)], terminate=False)
    target = tmp_path / "unterminated.json.gz"
    target.write_bytes(payload)
    with pytest.raises(full.GateCPythonFullError, match="unterminated JSONL record"):
        list(full.iter_full_shard_records(target))


def test_empty_object_fails_closed(tmp_path):
    target = tmp_path / "empty.json.gz"
    target.write_bytes(b"")
    with pytest.raises(full.GateCPythonFullError, match="object is empty"):
        list(full.iter_full_shard_records(target))


def test_corrupt_gzip_fails_closed(tmp_path):
    payload = bytearray(shard_bytes([make_record(index) for index in range(8)]))
    payload[40:80] = b"\x00" * 40
    target = tmp_path / "corrupt.json.gz"
    target.write_bytes(bytes(payload))
    with pytest.raises(full.GateCPythonFullError, match="corrupt|terminator"):
        list(full.iter_full_shard_records(target))


def test_concatenated_gzip_members_are_fully_decoded(tmp_path):
    """A multi-member object must not silently lose its trailing members."""
    first = gzip.compress(
        b"".join(json.dumps(make_record(i)).encode() + b"\n" for i in range(3)), mtime=0
    )
    second = gzip.compress(
        b"".join(json.dumps(make_record(100 + i)).encode() + b"\n" for i in range(4)), mtime=0
    )
    target = tmp_path / "multi.json.gz"
    target.write_bytes(first + second)
    indices = [index for index, _, _, _ in full.iter_full_shard_records(target) if index >= 0]
    assert indices == list(range(7))


def test_resume_offset_skips_exactly_the_committed_prefix(tmp_path):
    target = tmp_path / "shard.json.gz"
    target.write_bytes(shard_bytes([make_record(index) for index in range(10)]))
    tail = [
        index
        for index, _, _, _ in full.iter_full_shard_records(target, start_record_index=6)
        if index >= 0
    ]
    assert tail == [6, 7, 8, 9]


def test_fetched_object_failing_its_pinned_digest_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)

    class TamperingSource(full.LocalShardSource):
        def fetch(self, source, shard, destination):  # noqa: A002 - mirrors the base signature
            payload = bytearray((self.root / shard.path).read_bytes())
            payload[-1] ^= 0xFF  # same length, different bytes: only the digest can catch this
            destination.write_bytes(bytes(payload))
            return destination.stat().st_size

    with pytest.raises(full.GateCPythonFullError, match="does not match the frozen LFS object"):
        full.build_full_candidates(make_config(run_dir, source), store=TamperingSource(root))


def test_shard_identity_drift_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    (root / "stack-edu-0004.json.gz").write_bytes(shard_bytes([make_record(4242)]))
    # Shard-scope drift is enforced by the reused frozen C0 helper, so it raises its error type.
    with pytest.raises(gcp.GateCPythonError, match="drifted"):
        build(run_dir, source, root)


def test_revision_file_count_drift_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root, revision_shard_count=95)
    with pytest.raises(gcp.GateCPythonError, match="shard count drifted"):
        build(run_dir, source, root)


def test_shard_scope_must_be_twelve_shards(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    with pytest.raises(full.GateCPythonFullError, match="exactly the 12 pinned shards"):
        build(run_dir, make_full_source(root, shards=source.shards[:-1]), root)


def test_valid_cache_entry_is_reused_instead_of_refetched(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    cache = run_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for shard in source.shards:
        shutil.copyfile(root / shard.path, cache / shard.path)
    result = build(run_dir, source, root, keep_shard_cache=True)
    measured = result["resources"]["measured"]
    assert measured["shard_cache_reuse_count"] == SHARD_COUNT
    assert measured["shard_download_count"] == 0


def test_corrupt_cache_entry_is_discarded_and_refetched(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    cache = run_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / source.shards[0].path).write_bytes(b"not the pinned object")
    result = build(run_dir, source, root)
    measured = result["resources"]["measured"]
    assert measured["shard_download_count"] == SHARD_COUNT
    assert measured["shard_cache_reuse_count"] == 0


def test_shard_cache_is_released_after_each_shard_by_default(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    assert list((run_dir / "cache").glob("*.json.gz")) == []


# --------------------------------------------------------------------------------------
# Filter semantics carried over unchanged
# --------------------------------------------------------------------------------------


def test_frozen_filter_rejections_are_counted_at_full_scale(run_dir):
    root = run_dir / "corpus"
    rows = [
        make_record(1),
        make_record(2, language="Java"),
        make_record(3, is_generated=True),
        make_record(4, is_vendor=True),
        make_record(5, text=_sized_python(199)),
        make_record(6, text=_sized_python(8193)),
        make_record(7, text=_sized_python(220) + "print 'legacy'\n"),
        make_record(8, text=_repetitive_python()),
        make_record(
            9,
            text="\n".join(
                f"# explanatory note number {i} concerning value {i * 7}" for i in range(8)
            )
            + "\nz = 1\ny = 2\n",
        ),
        make_record(10, text="x = 1\n�\n"),
    ]
    write_shards(root, [rows] + [[] for _ in range(SHARD_COUNT - 1)])
    result = build(run_dir, make_full_source(root), root)
    assert result["accepted"] == 1
    rejections = result["rejections"]
    assert rejections["language_not_python"] == 1
    assert rejections["generated"] == 1
    assert rejections["vendor"] == 1
    assert rejections["size_band_short"] == 1
    assert rejections["size_band_long"] == 1
    assert rejections["ast_parse"] == 1
    assert rejections["repetition"] == 1
    assert rejections["comment_blank_fraction"] == 1
    assert rejections["strict_utf8"] == 1


def test_transport_level_rejections_are_controlled(run_dir):
    root = run_dir / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    payload = (
        b"{not json\n"
        + b"\xff\xfe not utf-8\n"
        + b"[1, 2, 3]\n"
        + json.dumps(make_record(1)).encode()
        + b"\n"
    )
    (root / "stack-edu-0000.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    for shard in range(1, SHARD_COUNT):
        (root / f"stack-edu-{shard:04d}.json.gz").write_bytes(
            shard_bytes([make_record(100 + shard)])
        )
    result = build(run_dir, make_full_source(root), root)
    assert result["rejections"]["malformed_json"] == 1
    assert result["rejections"]["line_not_utf8"] == 1
    assert result["rejections"]["record_not_object"] == 1
    assert result["accepted"] == SHARD_COUNT


def test_schema_drift_fails_closed(run_dir):
    root = run_dir / "corpus"
    record = make_record(1)
    record["surprise"] = "new upstream column"
    write_shards(root, [[record]] + [[make_record(100 + i)] for i in range(1, SHARD_COUNT)])
    with pytest.raises(gcp.GateCPythonError, match="added=\\['surprise'\\]"):
        build(run_dir, make_full_source(root), root)


def test_accepted_text_is_verbatim_with_no_gate_framing(run_dir):
    root = run_dir / "corpus"
    body = _python_source(3)
    write_shards(root, [[make_record(3, text=body)]] + [[] for _ in range(SHARD_COUNT - 1)])
    build(run_dir, make_full_source(root), root)
    document = released(run_dir)[0]
    assert document["text"] == body
    assert "[BOS]" not in document["text"] and "[EOS]" not in document["text"]
    assert document["metadata"]["metadata.repo_name"] == "octocat/example"
    assert document["metadata"]["metadata.license"] == "MIT"
    assert document["metadata"]["score"] == 3.0
    assert document["metadata"]["int_score"] == 3


# --------------------------------------------------------------------------------------
# Cross-shard dedup
# --------------------------------------------------------------------------------------


def test_cross_shard_text_dedup(run_dir):
    root = run_dir / "corpus"
    shared = _python_source(7)
    write_shards(
        root,
        [
            [make_record(shard, text=shared, record_id=f"{shard:040x}")]
            for shard in range(SHARD_COUNT)
        ],
    )
    result = build(run_dir, make_full_source(root), root)
    assert result["accepted"] == 1
    assert result["rejections"]["duplicate_text_sha256"] == SHARD_COUNT - 1


def test_cross_shard_record_id_dedup(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [make_record(shard * 7 + 1, text=_python_source(shard * 7 + 1), record_id="d" * 40)]
            for shard in range(SHARD_COUNT)
        ],
    )
    result = build(run_dir, make_full_source(root), root)
    assert result["accepted"] == 1
    assert result["rejections"]["duplicate_source_record_id"] == SHARD_COUNT - 1


# --------------------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------------------


def test_resume_reproduces_the_uninterrupted_release(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=5)
    source = make_full_source(root)

    reference = run_dir / "reference"
    reference.mkdir()
    single = full.build_full_candidates(
        make_config(reference, source), store=full.LocalShardSource(root)
    )
    assert single["published"] is True

    partial = build(run_dir, source, root, stop_after_documents=17)
    assert partial["published"] is False
    assert partial["stop_reason"] == "stop_after_documents"

    final = build(run_dir, source, root)
    assert final["published"] is True
    assert final["resumed"] is True
    assert final["resume_count"] == 1
    assert final["accepted"] == single["accepted"]
    assert final["documents_sha256"] == single["documents_sha256"]

    ids = [document["source_record_id"] for document in released(run_dir)]
    assert len(set(ids)) == len(ids)


def test_resume_rebuilds_dedup_state_from_the_committed_prefix(run_dir):
    """The seen-id/seen-hash sets are not checkpointed; a resume must still deduplicate."""
    root = run_dir / "corpus"
    shared = _python_source(11)
    rows = [make_record(1, text=shared, record_id="a" * 40)]
    later = [make_record(2, text=shared, record_id="b" * 40)]
    write_shards(root, [rows] + [[] for _ in range(SHARD_COUNT - 2)] + [later])
    source = make_full_source(root)

    partial = build(run_dir, source, root, stop_after_documents=1)
    assert partial["accepted"] == 1
    final = build(run_dir, source, root)
    assert final["published"] is True
    assert final["accepted"] == 1
    assert final["rejections"]["duplicate_text_sha256"] == 1


def test_uncommitted_suffix_is_truncated_on_resume(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=5)
    source = make_full_source(root)
    build(run_dir, source, root, stop_after_documents=9, checkpoint_every=1000)

    staged = full._staging_path(run_dir / "release") / full.DOCUMENTS_NAME
    staged.write_bytes(staged.read_bytes() + b'{"partially":"written"}\n')

    final = build(run_dir, source, root)
    assert final["published"] is True
    assert all("partially" not in json.dumps(document) for document in released(run_dir))


def test_checkpoint_checksum_corruption_refuses_resume(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=5)
    source = make_full_source(root)
    build(run_dir, source, root, stop_after_documents=6)
    checkpoint = run_dir / "work" / full.CHECKPOINT_NAME
    state = json.loads(checkpoint.read_text())
    state["counters"]["accepted"] = 999
    checkpoint.write_text(json.dumps(state))
    with pytest.raises(full.GateCPythonFullError, match="checkpoint checksum mismatch"):
        build(run_dir, source, root)


def test_fingerprint_mismatch_refuses_resume(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=5)
    source = make_full_source(root)
    build(run_dir, source, root, stop_after_documents=6)
    other = make_full_source(root, key="different_source_key")
    with pytest.raises(full.GateCPythonFullError, match="run fingerprint mismatch"):
        build(run_dir, other, root)


def test_filter_contract_change_refuses_resume(run_dir, monkeypatch):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=5)
    source = make_full_source(root)
    build(run_dir, source, root, stop_after_documents=6)
    monkeypatch.setattr(full, "FILTER_CONTRACT_SHA256", "0" * 64)
    with pytest.raises(full.GateCPythonFullError, match="filter contract mismatch"):
        build(run_dir, source, root)


def test_fingerprint_ignores_operational_cadence(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    base = full.run_fingerprint(make_config(run_dir, source))
    assert base == full.run_fingerprint(make_config(run_dir, source, checkpoint_every=99))
    assert base == full.run_fingerprint(make_config(run_dir, source, max_wall_seconds=10.0))
    assert base != full.run_fingerprint(make_config(run_dir, make_full_source(root, key="other")))


# --------------------------------------------------------------------------------------
# Partial traversal is never published
# --------------------------------------------------------------------------------------


def test_time_capped_run_publishes_nothing(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=6)
    ticks = iter([0.0] + [1000.0] * 100_000)
    result = full.build_full_candidates(
        make_config(run_dir, make_full_source(root), max_wall_seconds=1.0),
        store=full.LocalShardSource(root),
        clock=lambda: next(ticks),
    )
    assert result["published"] is False
    assert result["stop_reason"] == "time_cap"
    assert result["incomplete_shards"]
    assert not (run_dir / "release").exists()


def test_stop_after_documents_publishes_nothing(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=6)
    result = build(run_dir, make_full_source(root), root, stop_after_documents=5)
    assert result["published"] is False
    assert result["shards_complete"] < SHARD_COUNT
    assert not (run_dir / "release").exists()


# --------------------------------------------------------------------------------------
# Publication and release identity
# --------------------------------------------------------------------------------------


def test_publication_is_atomic_and_leaves_no_staging(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    release = run_dir / "release"
    assert sorted(path.name for path in release.iterdir()) == [
        full.CHECKSUMS_NAME,
        full.DOCUMENTS_NAME,
        full.MANIFEST_NAME,
    ]
    assert not full._staging_path(release).exists()


def test_publication_refuses_to_overwrite(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    build(run_dir, source, root)
    with pytest.raises(full.GateCPythonFullError, match="refusing to overwrite published output"):
        build(run_dir, source, root)


def test_release_declares_candidate_not_final_corpus(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    manifest = manifest_of(run_dir)
    assert manifest["release_kind"] == "python_gate_c_full_candidate"
    assert manifest["release_kind"] != "c0_diagnostic"
    assert manifest["full_source_traversal"] is True
    assert manifest["final_quota_selection_performed"] is False
    assert manifest["canonical_token_counting_performed"] is False
    assert manifest["stage_a_stage_b_final_split_performed"] is False
    assert manifest["promotion_to_final_training_corpus"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["gate_c_scope"]["tokenizer_counting"] is False
    assert manifest["gate_c_scope"]["document_truncation"] is False
    assert manifest["gate_c_scope"]["cross_source_near_dedup"] is False
    assert manifest["gate_c_scope"]["benchmark_decontamination"] is False
    assert all(value is False for value in manifest["hard_stops"].values())


def test_manifest_labels_proxy_tokens_as_non_canonical(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    result = build(run_dir, make_full_source(root), root)
    manifest = manifest_of(run_dir)
    assert manifest["proxy_tokens_4_bytes_per_token"] == result["accepted_text_bytes"] / 4.0
    assert "NOT a canonical token count" in manifest["proxy_token_note"]
    assert manifest["canonical_token_counting_performed"] is False


# --------------------------------------------------------------------------------------
# Honest resource accounting
# --------------------------------------------------------------------------------------


def test_unmeasured_network_request_count_is_null_never_zero(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    resources = manifest_of(run_dir)["resources"]
    assert resources["network_request_count"] is None
    assert resources["network_request_count_measured"] is False
    assert resources["unmeasured"]["network_request_count"] is None
    assert "network_request_count" in resources["unmeasured_fields"]
    assert "network_request_count" not in resources["measured"]
    assert "network_request_count" not in resources["measured_fields"]


def test_measured_resource_counters_are_exact(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    source = make_full_source(root)
    build(run_dir, source, root)
    resources = manifest_of(run_dir)["resources"]["measured"]
    total_compressed = sum(shard.size for shard in source.shards)
    assert resources["shard_download_count"] == SHARD_COUNT
    assert resources["shard_open_count"] == SHARD_COUNT
    assert resources["shard_integrity_verification_count"] == SHARD_COUNT
    assert resources["downloaded_compressed_bytes"] == total_compressed
    assert resources["integrity_hashed_compressed_bytes"] == total_compressed
    assert resources["compressed_source_bytes"] == total_compressed
    assert resources["resume_reread_compressed_bytes"] == 0
    manifest = manifest_of(run_dir)
    per_shard_decompressed = sum(
        entry["decompressed_bytes"] for entry in manifest["per_shard"].values()
    )
    assert resources["decompressed_bytes"] == per_shard_decompressed


def test_resume_reread_bytes_are_measured_not_hidden(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=8)
    source = make_full_source(root)
    build(run_dir, source, root, stop_after_documents=3)
    result = build(run_dir, source, root)
    measured = result["resources"]["measured"]
    assert measured["resume_reread_compressed_bytes"] > 0
    assert measured["resume_reread_decompressed_bytes"] > 0


def test_measured_fields_are_all_integers(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    measured = manifest_of(run_dir)["resources"]["measured"]
    assert set(measured) == set(full.ResourceAccounting.MEASURED_FIELDS)
    assert all(isinstance(value, int) for value in measured.values())


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def test_verify_accepts_a_freshly_published_release(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    result = build(run_dir, make_full_source(root), root)
    verified = full.verify_release(run_dir / "release")
    assert verified["accepted"] == result["accepted"]
    assert verified["documents_sha256"] == result["documents_sha256"]
    assert verified["shards_complete"] == SHARD_COUNT
    assert verified["distinct_record_ids"] == verified["accepted"]
    assert verified["distinct_text_hashes"] == verified["accepted"]


def test_verify_detects_tampered_text(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    documents = run_dir / "release" / full.DOCUMENTS_NAME
    lines = documents.read_bytes().split(b"\n")
    record = json.loads(lines[0])
    record["text"] = record["text"] + "# tampered\n"
    lines[0] = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
    documents.write_bytes(b"\n".join(lines))
    with pytest.raises(full.GateCPythonFullError):
        full.verify_release(run_dir / "release")


def test_verify_rejects_a_manifest_claiming_final_selection(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    path = run_dir / "release" / full.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["final_quota_selection_performed"] = True
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(full.GateCPythonFullError, match="must be false"):
        full.verify_release(run_dir / "release")


def test_verify_rejects_an_incomplete_shard_claim(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    path = run_dir / "release" / full.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    first = sorted(manifest["per_shard"])[0]
    manifest["per_shard"][first]["complete"] = False
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(full.GateCPythonFullError, match="marked incomplete"):
        full.verify_release(run_dir / "release")


def test_verify_rejects_a_c0_release(run_dir):
    """The full verifier must not accept a bounded C0 diagnostic release."""
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    path = run_dir / "release" / full.MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["release_kind"] = "c0_diagnostic"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(full.GateCPythonFullError, match="release_kind must be"):
        full.verify_release(run_dir / "release")


# --------------------------------------------------------------------------------------
# Full-population diagnostics and the provisional boundary
# --------------------------------------------------------------------------------------


def test_provisional_cutoff_is_exact_on_a_controlled_population(run_dir):
    """12 equal-sized documents with distinct scores: the 5/12 boundary is the 5th-highest."""
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [
                make_record(
                    shard,
                    text=_sized_unique_python(240, shard),
                    score=2.0 + 0.25 * shard,
                    record_id=f"{shard:040x}",
                )
            ]
            for shard in range(SHARD_COUNT)
        ],
    )
    build(run_dir, make_full_source(root), root)
    diagnostics = full.diagnose_release(run_dir / "release")
    cutoff = diagnostics["provisional_stage_b_cutoff"]
    assert diagnostics["documents"] == SHARD_COUNT
    assert diagnostics["accepted_text_bytes"] == SHARD_COUNT * 240
    assert cutoff["included_documents"] == 5
    assert cutoff["included_bytes"] == 5 * 240
    assert cutoff["cutoff_score"] == pytest.approx(2.0 + 0.25 * 7)
    assert cutoff["included_byte_share"] == pytest.approx(5 / 12)
    assert cutoff["provisional_byte_weighted_only"] is True
    assert cutoff["canonical_token_split_performed"] is False
    assert cutoff["final_stage_a_stage_b_split_performed"] is False


def test_provisional_cutoff_resolves_ties_by_ascending_sha256(run_dir):
    """All scores equal: the boundary is decided purely by the frozen tie-break."""
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [
                make_record(
                    shard,
                    text=_sized_unique_python(240, shard),
                    score=3.0,
                    record_id=f"{shard:040x}",
                )
            ]
            for shard in range(SHARD_COUNT)
        ],
    )
    build(run_dir, make_full_source(root), root)
    diagnostics = full.diagnose_release(run_dir / "release")
    cutoff = diagnostics["provisional_stage_b_cutoff"]
    assert cutoff["documents_strictly_above_cutoff"] == 0
    assert cutoff["documents_admitted_by_tie_break"] == 5
    assert cutoff["included_documents"] == 5
    assert cutoff["cutoff_score"] == 3.0


def test_diagnostics_report_the_full_population(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=5)
    build(run_dir, make_full_source(root), root)
    diagnostics = full.diagnose_release(run_dir / "release")
    assert diagnostics["documents"] == SHARD_COUNT * 5
    assert diagnostics["proxy_token_label"] == "NOT CANONICAL TOKEN COUNT"
    assert diagnostics["canonical_token_counting_performed"] is False
    assert diagnostics["proxy_tokens_4_bytes_per_token"] == (
        diagnostics["accepted_text_bytes"] / 4.0
    )
    assert diagnostics["length_bytes"]["n"] == SHARD_COUNT * 5
    assert diagnostics["score"]["n"] == SHARD_COUNT * 5
    assert diagnostics["shard_distribution"]["shards_covered"] == SHARD_COUNT
    assert diagnostics["provenance_availability"]["repo_name_present_rate"] == 1.0
    assert diagnostics["provenance_availability"]["license_present_rate"] == 1.0
    assert diagnostics["ast_diagnostics"]["has_function"] == SHARD_COUNT * 5
    assert diagnostics["exact_dedup"]["duplicate_text_sha256_rejects"] == 0


def test_score_quantiles_match_a_direct_computation(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [
                make_record(shard * 10 + offset, score=2.0 + 0.0625 * (shard * 10 + offset) % 3.0)
                for offset in range(5)
            ]
            for shard in range(SHARD_COUNT)
        ],
    )
    build(run_dir, make_full_source(root), root)
    diagnostics = full.diagnose_release(run_dir / "release")
    scores = sorted(document["metadata"]["score"] for document in released(run_dir))
    reported = diagnostics["score"]
    assert reported["min"] == scores[0]
    assert reported["max"] == scores[-1]
    assert reported["median"] == scores[min(len(scores) - 1, int(0.5 * len(scores)))]
    assert reported["mean"] == pytest.approx(sum(scores) / len(scores))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_reports_expected_failures_as_json_without_a_traceback(run_dir, capsys):
    code = full.main([
        "build",
        "--output-dir",
        str(run_dir / "release"),
        "--work-dir",
        str(run_dir / "work"),
        "--cache-dir",
        str(run_dir / "cache"),
        "--checkpoint-every",
        "0",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "checkpoint_every" in json.loads(captured.err)["error"]


def test_cli_verify_and_diagnose_round_trip(run_dir, capsys):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_full_source(root), root)
    capsys.readouterr()

    assert full.main(["verify", "--output-dir", str(run_dir / "release")]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["release_kind"] == "python_gate_c_full_candidate"

    out = run_dir / "diagnostics.json"
    assert full.main(["diagnose", "--output-dir", str(run_dir / "release"), "--out", str(out)]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["canonical_token_counting_performed"] is False
    assert reported["provisional_byte_weighted_only"] is True

    assert full.main(["diagnose", "--output-dir", str(run_dir / "release"), "--out", str(out)]) == 2
    assert "refusing to overwrite" in json.loads(capsys.readouterr().err)["error"]


def test_cli_verify_on_a_missing_release_is_a_controlled_error(run_dir, capsys):
    code = full.main(["verify", "--output-dir", str(run_dir / "absent")])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "error" in json.loads(captured.err)


def test_output_paths_must_be_git_ignored(tmp_path):
    root = tmp_path / "corpus"
    uniform_shards(root)
    source = make_full_source(root)
    config = make_config(PROJECT_ROOT / "src", source)
    with pytest.raises(full.GateCError, match="not Git-ignored"):
        full.build_full_candidates(config, store=full.LocalShardSource(root))
