"""Pure synthetic contract tests for the bounded Python Gate C C0 diagnostic builder.

No network, no tokenizer, no GPU, no real corpus.  Every shard here is a locally generated gzip
JSON-lines file, so the tests exercise the fail-closed contracts rather than upstream data.
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

from pretrain import corpus_gate_c_python as gcp  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "runs"
SHARD_COUNT = 12


# --------------------------------------------------------------------------------------
# Synthetic Python bodies
# --------------------------------------------------------------------------------------


def _python_source(index: int, statements: int = 12) -> str:
    """Valid Python 3 whose lines, words and 8-grams are all varied enough to pass the

    coarse repetition gate.
    """
    lines = [f"def compute_{index}(value):"]
    lines.extend(
        f"    value = value * {index * 7 + step + 3} + {step * 11 + 1}"
        for step in range(statements)
    )
    lines.append("    return value")
    return "\n".join(lines) + "\n"


def _sized_python(total_bytes: int) -> str:
    """Valid Python whose strict UTF-8 length is exactly ``total_bytes``."""
    head = "x = '"
    tail = "'\n"
    filler = total_bytes - len(head) - len(tail)
    assert filler >= 0
    return head + ("a" * filler) + tail


def _commented_python(comment_lines: int, code_lines: int) -> str:
    lines = [f"# note {i} covering the computation of {i * 13 + 5}" for i in range(comment_lines)]
    body = ["def f(a):", "    return a + 1", "b = f(2)", "c = f(3)", "d = f(4)", "e = f(5)"]
    lines.extend(body[:code_lines])
    return "\n".join(lines) + "\n"


def _repetitive_python(lines: int = 20) -> str:
    return "\n".join(["value = value + 1"] * lines) + "\n"


# --------------------------------------------------------------------------------------
# Synthetic records and shards
# --------------------------------------------------------------------------------------


def make_record(
    index: int,
    *,
    text: str | None = None,
    record_id: str | None = None,
    language: str = "Python",
    is_generated: bool = False,
    is_vendor: bool = False,
    score: float = 3.0,
    int_score: int = 3,
    repo_name: str | None = "octocat/example",
    license_name: str | None = "MIT",
) -> dict[str, object]:
    body = _python_source(index) if text is None else text
    identity = record_id if record_id is not None else f"{index:040x}"
    return {
        "added": "2024-11-18T18:05:43.375397+00:00",
        "created": "2023-06-27T19:37:42",
        "id": identity,
        "int_score": int_score,
        "metadata": {
            "blob_id": identity,
            "branch_name": "refs/heads/main",
            "committer_date": "2023-06-27T19:37:42",
            "content_id": f"{index:040x}",
            "detected_licenses": [license_name] if license_name else [],
            "directory_id": f"{index + 1:040x}",
            "extension": "py",
            "filename": f"module_{index}.py",
            "fork_events_count": index % 5,
            "gha_created_at": None,
            "gha_event_created_at": None,
            "gha_language": None,
            "gha_license_id": None,
            "github_id": None,
            "is_generated": is_generated,
            "is_vendor": is_vendor,
            "language": language,
            "length_bytes": len(body.encode("utf-8")),
            "license": license_name,
            "license_type": "permissive",
            "path": f"/src/module_{index}.py",
            "provenance": f"stack-edu-0055.json.gz:{index}",
            "repo_name": repo_name,
            "revision_date": "2023-06-27T19:37:42",
            "revision_id": f"{index + 2:040x}",
            "snapshot_id": f"{index + 3:040x}",
            "src_encoding": "UTF-8",
            "star_events_count": index % 7,
            "url": f"https://example.invalid/module_{index}.py",
            "visit_date": "2023-07-08T22:11:55.202669",
        },
        "score": score,
        "source": "stackv2",
        "text": body,
    }


def shard_bytes(records: list[object], *, terminate: bool = True) -> bytes:
    payload = b"".join(
        json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n" for record in records
    )
    if not terminate and payload.endswith(b"\n"):
        payload = payload[:-1]
    return gzip.compress(payload, mtime=0)


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


def make_source(root: Path, **overrides) -> gcp.PythonSourceSpec:
    """A frozen spec whose shard identities are taken from the fixture files on disk."""
    store = gcp.LocalShardStore(root)
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


def make_config(run_dir: Path, source: gcp.PythonSourceSpec, **overrides) -> gcp.BuildConfig:
    base = {
        "source": source,
        "output_dir": run_dir / "release",
        "work_dir": run_dir / "work",
        "target_documents": 200,
        "max_scanned": 2000,
        "max_shard_records": 200,
        "max_shard_compressed_bytes": 1 << 20,
        "max_response_bytes": 64 * 1024 * 1024,
        "max_wall_seconds": 120.0,
        "stride": 1,
        "seed": 20260817,
        "checkpoint_every": 4,
    }
    base.update(overrides)
    return gcp.BuildConfig(**base)


@pytest.fixture
def run_dir(tmp_path_factory) -> Path:
    """A Git-ignored working directory: Gate C refuses to write anywhere else."""
    root = RUNS_ROOT / "_pytest_gate_c_python"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tmp_path_factory.mktemp("case", numbered=True))
    target = root / path.name
    target.mkdir(parents=True, exist_ok=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def build(run_dir: Path, source: gcp.PythonSourceSpec, root: Path, **overrides) -> dict:
    return gcp.build_candidates(
        make_config(run_dir, source, **overrides), store=gcp.LocalShardStore(root)
    )


def read_documents(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def released(run_dir: Path) -> list[dict]:
    return read_documents(run_dir / "release" / gcp.DOCUMENTS_NAME)


def manifest_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "release" / gcp.MANIFEST_NAME).read_text())


# --------------------------------------------------------------------------------------
# Frozen source binding and shard scope
# --------------------------------------------------------------------------------------


def test_pinned_revision_is_the_exact_frozen_commit():
    assert gcp.PYTHON_SOURCE.revision == "c354dbe88469a1153e97c6a63ac50591849654de"
    assert gcp.PYTHON_SOURCE.dataset == "common-pile/stackv2_edu_filtered"
    assert gcp.PYTHON_SOURCE.dataset_config == "default"
    assert gcp.PYTHON_SOURCE.split == "train"
    assert gcp.PYTHON_SOURCE.revision_shard_count == 95


def test_python_shard_scope_is_exactly_the_twelve_pinned_shards():
    shards = gcp.PYTHON_SOURCE.shards
    assert len(shards) == SHARD_COUNT
    assert [shard.path for shard in shards] == [
        f"stack-edu-{number:04d}.json.gz" for number in range(73, 85)
    ]
    assert len({shard.lfs_sha256 for shard in shards}) == SHARD_COUNT
    assert all(len(shard.lfs_sha256) == 64 for shard in shards)
    assert sum(shard.size for shard in shards) == 11_716_959_758


def test_c0_ceilings_cannot_cover_the_pinned_python_corpus():
    """The structural guarantee that a C0 run can never become a full Python build."""
    reachable = gcp.MAX_SHARD_COMPRESSED_BYTES * SHARD_COUNT
    assert reachable < sum(shard.size for shard in gcp.PYTHON_SOURCE.shards)


def test_shard_scope_drift_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_source(root)
    live = gcp.LocalShardStore(root).list_shards(source)

    gcp.assert_shard_scope(source, live, revision_file_count=SHARD_COUNT)

    with pytest.raises(gcp.GateCPythonError, match="shard count drifted"):
        gcp.assert_shard_scope(source, live, revision_file_count=SHARD_COUNT + 1)

    drifted = source.shards[:-1] + (
        gcp.ShardSpec(
            path=source.shards[-1].path,
            size=source.shards[-1].size + 1,
            lfs_sha256=source.shards[-1].lfs_sha256,
            blob_id=source.shards[-1].blob_id,
        ),
    )
    with pytest.raises(gcp.GateCPythonError, match="size drifted"):
        gcp.assert_shard_scope(
            make_source(root, shards=drifted), live, revision_file_count=SHARD_COUNT
        )

    hashed = source.shards[:-1] + (
        gcp.ShardSpec(
            path=source.shards[-1].path,
            size=source.shards[-1].size,
            lfs_sha256="0" * 64,
            blob_id=source.shards[-1].blob_id,
        ),
    )
    with pytest.raises(gcp.GateCPythonError, match="LFS SHA-256 drifted"):
        gcp.assert_shard_scope(
            make_source(root, shards=hashed), live, revision_file_count=SHARD_COUNT
        )

    blobbed = source.shards[:-1] + (
        gcp.ShardSpec(
            path=source.shards[-1].path,
            size=source.shards[-1].size,
            lfs_sha256=source.shards[-1].lfs_sha256,
            blob_id="0" * 40,
        ),
    )
    with pytest.raises(gcp.GateCPythonError, match="blob id drifted"):
        gcp.assert_shard_scope(
            make_source(root, shards=blobbed), live, revision_file_count=SHARD_COUNT
        )


def test_shard_content_drift_after_pinning_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_source(root)
    (root / "stack-edu-0003.json.gz").write_bytes(shard_bytes([make_record(999)]))
    with pytest.raises(gcp.GateCPythonError, match="drifted"):
        build(run_dir, source, root)


def test_shard_scope_must_be_twelve_shards(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_source(root)
    short = make_source(root, shards=source.shards[:-1])
    with pytest.raises(gcp.GateCPythonError, match="exactly the 12 pinned shards"):
        build(run_dir, short, root)


def test_missing_shard_file_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_source(root)
    (root / "stack-edu-0005.json.gz").unlink()
    with pytest.raises(gcp.GateCPythonError, match="missing from the local store"):
        build(run_dir, source, root)


# --------------------------------------------------------------------------------------
# Transport: gzip / JSONL framing
# --------------------------------------------------------------------------------------


def test_gzip_jsonl_framing_yields_every_complete_record():
    records = [make_record(index) for index in range(5)]
    payload = shard_bytes(records)
    seen = [
        (index, line)
        for index, line, _, _ in gcp.iter_shard_records([payload], expect_complete=True)
        if index >= 0
    ]
    assert [index for index, _ in seen] == [0, 1, 2, 3, 4]
    assert json.loads(seen[2][1])["id"] == records[2]["id"]


def test_record_truncated_by_the_prefix_cap_is_never_yielded():
    payload = shard_bytes([make_record(index) for index in range(20)])
    prefix = payload[: len(payload) // 2]
    complete = [index for index, _, _, _ in gcp.iter_shard_records([prefix]) if index >= 0]
    assert complete == list(range(len(complete)))
    assert len(complete) < 20
    # Every yielded line is a complete, parseable record.
    for _, line, _, _ in gcp.iter_shard_records([prefix]):
        if line:
            json.loads(line)


def test_unterminated_final_record_fails_closed_on_a_complete_read():
    payload = shard_bytes([make_record(0), make_record(1)], terminate=False)
    with pytest.raises(gcp.GateCPythonError, match="unterminated JSONL record"):
        list(gcp.iter_shard_records([payload], expect_complete=True))


def test_corrupt_gzip_stream_fails_closed():
    payload = bytearray(shard_bytes([make_record(index) for index in range(8)]))
    payload[40:80] = b"\x00" * 40
    with pytest.raises(gcp.GateCPythonError, match="gzip stream is corrupt"):
        list(gcp.iter_shard_records([bytes(payload)]))


def test_resume_offset_skips_exactly_the_committed_prefix():
    payload = shard_bytes([make_record(index) for index in range(10)])
    tail = [index for index, _, _, _ in gcp.iter_shard_records([payload], start_record_index=6)]
    assert [index for index in tail if index >= 0] == [6, 7, 8, 9]


# --------------------------------------------------------------------------------------
# Frozen mechanical filter
# --------------------------------------------------------------------------------------


def evaluate(record: dict) -> gcp.Decision:
    return gcp.evaluate_record(record, gcp.PYTHON_SOURCE)


def test_ordinary_python_record_is_accepted():
    decision = evaluate(make_record(1))
    assert decision.accepted
    assert decision.reason is None
    assert "has_function" in decision.diagnostics


def test_language_other_than_python_is_rejected():
    assert evaluate(make_record(1, language="Java")).reason == "language_not_python"


def test_generated_and_vendor_records_are_rejected():
    assert evaluate(make_record(1, is_generated=True)).reason == "generated"
    assert evaluate(make_record(1, is_vendor=True)).reason == "vendor"


def test_strict_utf8_replacement_character_is_rejected():
    assert evaluate(make_record(1, text="x = 1\n�\n")).reason == "strict_utf8"


@pytest.mark.parametrize(
    ("size", "reason"),
    [(199, "size_band_short"), (200, None), (8192, None), (8193, "size_band_long")],
)
def test_byte_band_boundaries_are_inclusive(size, reason):
    decision = evaluate(make_record(1, text=_sized_python(size)))
    assert decision.reason == reason
    assert decision.accepted is (reason is None)


def test_python2_print_fails_ast_parse():
    text = _sized_python(220) + "print 'legacy output'\n"
    assert evaluate(make_record(1, text=text)).reason == "ast_parse"


def test_pathological_repetition_is_rejected():
    assert evaluate(make_record(1, text=_repetitive_python())).reason == "repetition"


@pytest.mark.parametrize(
    ("comments", "code", "reason"),
    [(7, 3, None), (8, 2, "comment_blank_fraction")],
)
def test_comment_blank_fraction_boundary(comments, code, reason):
    text = _commented_python(comments, code)
    assert gcp.comment_blank_fraction(text) == pytest.approx((comments) / (comments + code))
    assert evaluate(make_record(1, text=text)).reason == reason


def test_comment_blank_fraction_matches_the_frozen_gate_e_definition():
    assert gcp.comment_blank_fraction("# a\n\ncode = 1\nmore = 2\n") == 0.5
    assert gcp.comment_blank_fraction("") == 0.0
    assert gcp.comment_blank_fraction("    # indented comment\ncode = 1\n") == 0.5


def test_missing_required_field_is_a_controlled_reject():
    record = make_record(1)
    del record["metadata"]["repo_name"]
    decision = evaluate(record)
    assert decision.reason == "missing_field"
    assert decision.detail == "metadata.repo_name"


def test_null_required_field_is_a_controlled_reject():
    record = make_record(1, license_name=None)
    record["metadata"]["license"] = None
    decision = evaluate(record)
    assert decision.reason == "null_field"
    assert decision.detail == "metadata.license"


def test_wrong_required_field_type_is_a_controlled_reject():
    record = make_record(1)
    record["score"] = "3.0"
    decision = evaluate(record)
    assert decision.reason == "field_type"
    assert decision.detail == "score"


def test_boolean_is_not_accepted_as_a_number():
    record = make_record(1)
    record["int_score"] = True
    assert evaluate(record).reason == "field_type"


def test_nullable_metadata_leaves_do_not_block_acceptance():
    record = make_record(1)
    for path in gcp.PYTHON_NULLABLE_PATHS:
        assert record["metadata"][path.split(".", 1)[1]] is None
    assert evaluate(record).accepted


def test_filter_order_is_frozen_and_published():
    assert gcp.FILTER_ORDER[0] == "record_shape"
    assert gcp.FILTER_ORDER[-2:] == ("duplicate_source_record_id", "duplicate_text_sha256")
    assert "comment_blank_fraction" in gcp.FILTER_ORDER


# --------------------------------------------------------------------------------------
# Schema drift
# --------------------------------------------------------------------------------------


def test_added_leaf_fails_closed(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    record = make_record(1)
    record["surprise"] = "new upstream column"
    write_shards(root, [[record]] + [[make_record(i)] for i in range(1, SHARD_COUNT)])
    with pytest.raises(gcp.GateCPythonError, match="added=\\['surprise'\\]"):
        build(run_dir, make_source(root), root)


def test_missing_leaf_fails_closed(run_dir):
    root = run_dir / "corpus"
    record = make_record(1)
    del record["metadata"]["url"]
    write_shards(root, [[record]] + [[make_record(i)] for i in range(1, SHARD_COUNT)])
    with pytest.raises(gcp.GateCPythonError, match="missing=\\['metadata.url'\\]"):
        build(run_dir, make_source(root), root)


def test_malformed_json_line_is_a_controlled_reject_not_a_traceback(run_dir):
    root = run_dir / "corpus"
    good = make_record(1)
    payload = b"{not json\n" + json.dumps(good, ensure_ascii=False).encode() + b"\n"
    root.mkdir(parents=True, exist_ok=True)
    (root / "stack-edu-0000.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    for shard in range(1, SHARD_COUNT):
        (root / f"stack-edu-{shard:04d}.json.gz").write_bytes(
            shard_bytes([make_record(100 + shard)])
        )
    result = build(run_dir, make_source(root), root)
    assert result["rejections"]["malformed_json"] == 1
    assert result["accepted"] == SHARD_COUNT


def test_non_utf8_line_is_a_controlled_reject(run_dir):
    root = run_dir / "corpus"
    payload = b"\xff\xfe not utf-8\n" + json.dumps(make_record(1)).encode() + b"\n"
    root.mkdir(parents=True, exist_ok=True)
    (root / "stack-edu-0000.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    for shard in range(1, SHARD_COUNT):
        (root / f"stack-edu-{shard:04d}.json.gz").write_bytes(
            shard_bytes([make_record(100 + shard)])
        )
    result = build(run_dir, make_source(root), root)
    assert result["rejections"]["line_not_utf8"] == 1


def test_json_array_line_is_a_controlled_reject(run_dir):
    root = run_dir / "corpus"
    payload = b"[1, 2, 3]\n" + json.dumps(make_record(1)).encode() + b"\n"
    root.mkdir(parents=True, exist_ok=True)
    (root / "stack-edu-0000.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    for shard in range(1, SHARD_COUNT):
        (root / f"stack-edu-{shard:04d}.json.gz").write_bytes(
            shard_bytes([make_record(100 + shard)])
        )
    result = build(run_dir, make_source(root), root)
    assert result["rejections"]["record_not_object"] == 1


# --------------------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------------------


def test_duplicate_text_is_rejected_once(run_dir):
    root = run_dir / "corpus"
    shared = _python_source(42)
    records = [
        make_record(1, text=shared, record_id="a" * 40),
        make_record(2, text=shared, record_id="b" * 40),
    ]
    write_shards(root, [records] + [[] for _ in range(SHARD_COUNT - 1)])
    result = build(run_dir, make_source(root), root)
    assert result["accepted"] == 1
    assert result["rejections"]["duplicate_text_sha256"] == 1


def test_duplicate_record_id_is_rejected_once(run_dir):
    root = run_dir / "corpus"
    records = [
        make_record(1, record_id="c" * 40),
        make_record(2, record_id="c" * 40),
    ]
    write_shards(root, [records] + [[] for _ in range(SHARD_COUNT - 1)])
    result = build(run_dir, make_source(root), root)
    assert result["accepted"] == 1
    assert result["rejections"]["duplicate_source_record_id"] == 1


def test_dedup_holds_across_shards(run_dir):
    root = run_dir / "corpus"
    shared = _python_source(7)
    per_shard = [
        [make_record(shard, text=shared, record_id=f"{shard:040x}")] for shard in range(SHARD_COUNT)
    ]
    write_shards(root, per_shard)
    result = build(run_dir, make_source(root), root)
    assert result["accepted"] == 1
    assert result["rejections"]["duplicate_text_sha256"] == SHARD_COUNT - 1


# --------------------------------------------------------------------------------------
# Traversal, caps and coverage
# --------------------------------------------------------------------------------------


def test_all_twelve_shards_are_covered(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=3)
    result = build(run_dir, make_source(root), root)
    assert result["shards_covered"] == SHARD_COUNT
    assert result["stop_reason"] == "all_shard_windows_completed"
    shards = {document["shard_path"] for document in released(run_dir)}
    assert len(shards) == SHARD_COUNT


def test_stride_evaluates_only_the_strided_record_indices(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [make_record(shard * 100 + offset) for offset in range(6)]
            for shard in range(SHARD_COUNT)
        ],
    )
    result = build(run_dir, make_source(root), root, stride=3)
    indices = {document["record_index"] for document in released(run_dir)}
    assert indices == {0, 3}
    assert result["scanned"] == SHARD_COUNT * 2
    manifest = manifest_of(run_dir)
    assert manifest["accounting"]["stride_skipped"] == SHARD_COUNT * 4
    assert manifest["traversal"]["stride"] == 3
    assert manifest["traversal"]["row_level_representative_sampler"] is False


def test_per_shard_record_cap_stops_each_shard(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=8)
    result = build(run_dir, make_source(root), root, max_shard_records=2)
    assert result["scanned"] == SHARD_COUNT * 2
    manifest = manifest_of(run_dir)
    assert {entry["stop_reason"] for entry in manifest["per_shard"].values()} == {
        "shard_record_cap"
    }


def test_target_documents_cap_stops_the_run(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=8)
    result = build(run_dir, make_source(root), root, target_documents=5)
    assert result["accepted"] == 5
    assert result["stop_reason"] == "target_reached"


def test_scan_cap_stops_the_run(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=8)
    result = build(
        run_dir, make_source(root), root, max_scanned=6, target_documents=6, max_shard_records=6
    )
    assert result["stop_reason"] in {"scan_cap", "target_reached"}
    assert result["scanned"] <= 6


def test_byte_cap_stops_the_run(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=8)
    result = build(run_dir, make_source(root), root, max_response_bytes=1)
    assert result["stop_reason"] == "byte_cap"


def test_time_cap_stops_the_run(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=8)
    ticks = iter([0.0] + [1000.0] * 10_000)
    result = gcp.build_candidates(
        make_config(run_dir, make_source(root), max_wall_seconds=1.0),
        store=gcp.LocalShardStore(root),
        clock=lambda: next(ticks),
    )
    assert result["stop_reason"] == "time_cap"


def test_shard_compressed_prefix_cap_bounds_each_shard(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [make_record(shard * 100 + offset) for offset in range(40)]
            for shard in range(SHARD_COUNT)
        ],
    )
    result = build(run_dir, make_source(root), root, max_shard_compressed_bytes=200)
    manifest = manifest_of(run_dir)
    assert result["accepted"] < SHARD_COUNT * 40
    assert all(entry["compressed_bytes"] <= 200 for entry in manifest["per_shard"].values())


# --------------------------------------------------------------------------------------
# Checkpoint and resume
# --------------------------------------------------------------------------------------


def test_resume_produces_exactly_the_uninterrupted_release(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    source = make_source(root)

    reference_dir = run_dir / "reference"
    reference_dir.mkdir()
    single = gcp.build_candidates(
        make_config(reference_dir, source), store=gcp.LocalShardStore(root)
    )

    partial = build(run_dir, source, root, stop_after_documents=7)
    assert partial["published"] is False
    assert partial["accepted"] == 7

    final = build(run_dir, source, root)
    assert final["published"] is True
    assert final["resumed"] is True
    assert final["resume_count"] == 1

    reference_ids = [
        document["source_record_id"]
        for document in read_documents(reference_dir / "release" / gcp.DOCUMENTS_NAME)
    ]
    resumed_ids = [document["source_record_id"] for document in released(run_dir)]
    assert resumed_ids == reference_ids
    assert len(set(resumed_ids)) == len(resumed_ids)
    assert final["accepted"] == single["accepted"]
    assert final["documents_sha256"] == single["documents_sha256"]


def test_uncommitted_suffix_is_truncated_on_resume(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    source = make_source(root)
    build(run_dir, source, root, stop_after_documents=6, checkpoint_every=4)

    documents = run_dir / "work" / gcp.DOCUMENTS_NAME
    committed = json.loads((run_dir / "work" / gcp.CHECKPOINT_NAME).read_text())["documents_bytes"]
    documents.write_bytes(documents.read_bytes() + b'{"partially":"written"}\n')

    final = build(run_dir, source, root)
    assert final["published"] is True
    assert all("partially" not in json.dumps(document) for document in released(run_dir))
    assert committed <= final["accepted"] * 10_000


def test_checkpoint_checksum_corruption_refuses_resume(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    source = make_source(root)
    build(run_dir, source, root, stop_after_documents=5)

    checkpoint = run_dir / "work" / gcp.CHECKPOINT_NAME
    state = json.loads(checkpoint.read_text())
    state["counters"]["accepted"] = 999
    checkpoint.write_text(json.dumps(state))

    with pytest.raises(gcp.GateCPythonError, match="checkpoint checksum mismatch"):
        build(run_dir, source, root)


def test_checkpoint_fingerprint_mismatch_refuses_resume(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    source = make_source(root)
    build(run_dir, source, root, stop_after_documents=5)
    with pytest.raises(gcp.GateCPythonError, match="run fingerprint mismatch"):
        build(run_dir, source, root, stride=2)


def test_checkpoint_records_a_shard_and_record_cursor(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    build(run_dir, make_source(root), root, stop_after_documents=6)
    state = json.loads((run_dir / "work" / gcp.CHECKPOINT_NAME).read_text())
    assert state["next_shard_index"] == 1
    assert state["next_record_index"] == 2
    assert state["completed"] is False


def test_fingerprint_binds_the_semantic_inputs(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_source(root)
    base = gcp.run_fingerprint(make_config(run_dir, source))
    assert base == gcp.run_fingerprint(make_config(run_dir, source, checkpoint_every=8))
    assert base != gcp.run_fingerprint(make_config(run_dir, source, stride=2))
    assert base != gcp.run_fingerprint(make_config(run_dir, source, max_shard_records=1))


# --------------------------------------------------------------------------------------
# Publication, verification and immutability
# --------------------------------------------------------------------------------------


def test_release_is_published_atomically_with_no_staging_left_behind(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_source(root), root)
    release = run_dir / "release"
    assert sorted(path.name for path in release.iterdir()) == [
        gcp.CHECKSUMS_NAME,
        gcp.DOCUMENTS_NAME,
        gcp.MANIFEST_NAME,
    ]
    assert not gcp._staging_path(release).exists()


def test_publication_refuses_to_overwrite_an_existing_release(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    source = make_source(root)
    build(run_dir, source, root)
    with pytest.raises(gcp.GateCPythonError, match="refusing to overwrite published output"):
        build(run_dir, source, root)


def test_release_declares_its_diagnostic_status(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_source(root), root)
    manifest = manifest_of(run_dir)
    assert manifest["release_kind"] == "c0_diagnostic"
    assert manifest["promotion_eligible"] is False
    assert manifest["production_candidate_quota_authorized"] is False
    assert manifest["full_python_candidate_build_authorized"] is False
    assert manifest["provisional_byte_weighted_only"] is True
    assert manifest["canonical_token_split_performed"] is False
    assert manifest["gate_c_scope"]["tokenizer_counting"] is False
    assert manifest["gate_c_scope"]["bos_eos_inserted"] is False
    assert manifest["gate_c_scope"]["document_truncation"] is False
    assert manifest["gate_c_scope"]["stage_a_stage_b_split_performed"] is False
    assert all(value is False for value in manifest["hard_stops"].values())


def test_verify_accepts_a_freshly_published_release(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    result = build(run_dir, make_source(root), root)
    verified = gcp.verify_release(run_dir / "release")
    assert verified["accepted"] == result["accepted"]
    assert verified["documents_sha256"] == result["documents_sha256"]
    assert verified["shards_covered"] == SHARD_COUNT


def test_verify_detects_tampered_document_text(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_source(root), root)
    documents = run_dir / "release" / gcp.DOCUMENTS_NAME
    lines = documents.read_bytes().split(b"\n")
    record = json.loads(lines[0])
    record["text"] = record["text"] + "# tampered\n"
    lines[0] = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    documents.write_bytes(b"\n".join(lines))
    with pytest.raises(gcp.GateCPythonError):
        gcp.verify_release(run_dir / "release")


def test_verify_rejects_a_manifest_claiming_promotion(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_source(root), root)
    manifest_path = run_dir / "release" / gcp.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["promotion_eligible"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(gcp.GateCPythonError, match="never be promotion eligible"):
        gcp.verify_release(run_dir / "release")


def test_documents_retain_score_repo_and_license_provenance(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [
                make_record(
                    shard,
                    score=2.5 + shard * 0.1,
                    repo_name=f"org/repo{shard}",
                    license_name="Apache-2.0",
                )
            ]
            for shard in range(SHARD_COUNT)
        ],
    )
    build(run_dir, make_source(root), root)
    document = released(run_dir)[0]
    metadata = document["metadata"]
    assert metadata["score"] == pytest.approx(2.5)
    assert metadata["metadata.repo_name"] == "org/repo0"
    assert metadata["metadata.license"] == "Apache-2.0"
    assert metadata["metadata.detected_licenses"] == ["Apache-2.0"]
    assert metadata["metadata.blob_id"] == document["natural_id"]
    assert metadata["metadata.provenance"].startswith("stack-edu-")
    assert document["provenance"]["revision"] == gcp.PYTHON_SOURCE.revision
    assert document["provenance"]["shard"] == "stack-edu-0000.json.gz"


def test_document_text_is_verbatim_with_no_gate_framing(run_dir):
    root = run_dir / "corpus"
    body = _python_source(3)
    write_shards(root, [[make_record(3, text=body)]] + [[] for _ in range(SHARD_COUNT - 1)])
    build(run_dir, make_source(root), root)
    document = released(run_dir)[0]
    assert document["text"] == body
    assert "[BOS]" not in document["text"] and "[EOS]" not in document["text"]
    assert document["text_bytes"] == len(body.encode("utf-8"))


# --------------------------------------------------------------------------------------
# Score diagnostics
# --------------------------------------------------------------------------------------


def test_provisional_byte_weighted_cutoff_reaches_the_stage_b_mass(run_dir):
    root = run_dir / "corpus"
    write_shards(
        root,
        [
            [
                make_record(shard * 10 + offset, score=2.0 + 0.5 * (shard * 10 + offset) % 4)
                for offset in range(4)
            ]
            for shard in range(SHARD_COUNT)
        ],
    )
    build(run_dir, make_source(root), root)
    diagnostics = gcp.diagnose_release(run_dir / "release")
    cutoff = diagnostics["provisional_stage_b_cutoff"]
    assert cutoff["provisional_byte_weighted_only"] is True
    assert cutoff["canonical_token_split_performed"] is False
    assert cutoff["stage_b_mass_share"] == pytest.approx(5 / 12)
    assert cutoff["included_byte_share"] >= 5 / 12
    assert cutoff["included_documents"] >= 1
    assert cutoff["cutoff_score"] is not None
    assert cutoff["bytes_strictly_above_cutoff"] <= cutoff["included_bytes"]


def test_diagnostics_report_distributions_and_provenance_rates(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root, records_per_shard=4)
    build(run_dir, make_source(root), root)
    diagnostics = gcp.diagnose_release(run_dir / "release")
    assert diagnostics["documents"] == SHARD_COUNT * 4
    assert diagnostics["length_bytes"]["n"] == SHARD_COUNT * 4
    assert diagnostics["provenance_availability"]["repo_name_present_rate"] == 1.0
    assert diagnostics["provenance_availability"]["license_present_rate"] == 1.0
    assert diagnostics["shard_coverage"]["shards_covered"] == SHARD_COUNT
    assert diagnostics["ast_diagnostics"]["has_function"] == SHARD_COUNT * 4
    assert diagnostics["release"]["release_kind"] == "c0_diagnostic"
    assert diagnostics["score"]["n"] == SHARD_COUNT * 4


def test_diagnose_binds_the_release_checksums(run_dir):
    root = run_dir / "corpus"
    uniform_shards(root)
    result = build(run_dir, make_source(root), root)
    diagnostics = gcp.diagnose_release(run_dir / "release")
    assert diagnostics["release"]["documents_sha256"] == result["documents_sha256"]
    assert diagnostics["release"]["manifest_sha256"] == result["manifest_sha256"]


# --------------------------------------------------------------------------------------
# Configuration guards and CLI
# --------------------------------------------------------------------------------------


def test_output_and_work_paths_must_be_git_ignored(tmp_path):
    root = tmp_path / "corpus"
    uniform_shards(root)
    source = make_source(root)
    config = make_config(PROJECT_ROOT / "src", source)
    # Raised by the shared Gate C helper, so this is the base class, not the Python subclass.
    with pytest.raises(gcp.GateCError, match="not Git-ignored"):
        gcp.build_candidates(config, store=gcp.LocalShardStore(root))


@pytest.mark.parametrize(
    "overrides",
    [
        {"stride": 0},
        {"stride": gcp.MAX_STRIDE + 1},
        {"target_documents": 0},
        {"target_documents": gcp.MAX_ACCEPTED_DOCUMENTS + 1},
        {"max_scanned": 1, "target_documents": 2},
        {"max_shard_compressed_bytes": gcp.MAX_SHARD_COMPRESSED_BYTES + 1},
        {"max_response_bytes": 0},
        {"max_wall_seconds": 0.0},
        {"max_wall_seconds": float(gcp.MAX_WALL_SECONDS + 1)},
        {"seed": -1},
        {"checkpoint_every": 0},
        {"max_shard_records": 0},
        {"stop_after_documents": 0},
    ],
)
def test_out_of_range_caps_fail_closed(run_dir, overrides):
    root = run_dir / "corpus"
    uniform_shards(root)
    with pytest.raises(gcp.GateCPythonError):
        build(run_dir, make_source(root), root, **overrides)


def test_cli_reports_expected_failures_as_json_without_a_traceback(run_dir, capsys):
    code = gcp.main([
        "build",
        "--output-dir",
        str(run_dir / "release"),
        "--work-dir",
        str(run_dir / "work"),
        "--target-documents",
        "10",
        "--max-scanned",
        "100",
        "--max-shard-records",
        "10",
        "--max-shard-compressed-bytes",
        "1024",
        "--max-response-bytes",
        "1024",
        "--max-wall-seconds",
        "10",
        "--stride",
        "0",
        "--seed",
        "1",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert json.loads(captured.err)["error"].startswith("stride must be in")


def test_cli_verify_and_diagnose_round_trip(run_dir, capsys):
    root = run_dir / "corpus"
    uniform_shards(root)
    build(run_dir, make_source(root), root)
    capsys.readouterr()

    assert gcp.main(["verify", "--output-dir", str(run_dir / "release")]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["release_kind"] == "c0_diagnostic"
    assert verified["promotion_eligible"] is False

    out = run_dir / gcp.DIAGNOSTICS_NAME
    assert gcp.main(["diagnose", "--output-dir", str(run_dir / "release"), "--out", str(out)]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["canonical_token_split_performed"] is False
    assert out.exists()

    assert gcp.main(["diagnose", "--output-dir", str(run_dir / "release"), "--out", str(out)]) == 2
    assert "refusing to overwrite" in json.loads(capsys.readouterr().err)["error"]


def test_cli_verify_on_a_missing_release_is_a_controlled_error(run_dir, capsys):
    code = gcp.main(["verify", "--output-dir", str(run_dir / "absent")])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "error" in json.loads(captured.err)
