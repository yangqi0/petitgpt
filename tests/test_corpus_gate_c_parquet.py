"""Pure synthetic contract tests for the Gate C v2 Parquet transport and sampler.

No network, no tokenizer, no GPU, no real corpus.  Parquet fixtures are written to a temporary
directory and served through a local object store, so every contract below is exercised against
real Parquet bytes without touching the Hub.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
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
    corpus_gate_c_parquet as gp,
)

RUNS_ROOT = PROJECT_ROOT / "runs"


# --------------------------------------------------------------------------------------
# Fixtures: real Parquet bytes served by a local object store
# --------------------------------------------------------------------------------------


def _prose(index: int, sentences: int = 14) -> str:
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
            "wikitext": "== MUST NOT BE EMITTED ==",
        }
        for index in range(start, start + count)
    ]


def cosmopedia_rows(start: int, count: int) -> list[dict[str, object]]:
    return [
        {
            "audience": "middle_school",
            "format": "textbook",
            "prompt": "PROMPT TEXT THAT MUST NEVER BE EMITTED",
            "seed_data": "fineweb",
            "text": f" Chapter {index}: the topic explained.\n\n{_prose(index)}",
            "token_length": 512,
        }
        for index in range(start, start + count)
    ]


def finephrase_rows(start: int, count: int) -> list[dict[str, object]]:
    return [
        {
            "dataset": "fineweb-edu",
            "dump": "CC-MAIN-2024-10",
            "file_path": "s3://bucket/file.parquet",
            "id": f"fp-{index}",
            "int_score": 3,
            "language": "en",
            "language_score": 0.98,
            "rollout_results": [
                {
                    "finish_reason": "stop",
                    "text": f"Making bread, part {index}.\n\n{_prose(index, 20)}",
                    "usage": {
                        "completion_tokens": 400,
                        "prompt_tokens": 800,
                        "prompt_tokens_details": None,
                        "total_tokens": 1200,
                    },
                }
            ],
            "score": 3.1,
            "text": f"ORIGINAL FINEWEB SOURCE {index} MUST NEVER BE EMITTED. {_prose(index + 900, 20)}",
            "token_count": 900,
            "url": f"https://example.org/article/{index}",
        }
        for index in range(start, start + count)
    ]


def write_parquet(path: Path, rows: list[dict[str, object]], row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


class LocalStore(gp.ObjectStore):
    """Serves pinned objects from a local directory; counts nothing over the network."""

    def __init__(self, root: Path, *, corrupt: set[str] | None = None) -> None:
        self.root = root
        self.corrupt = corrupt or set()
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
                    etag=hashlib.sha1(payload).hexdigest(),  # noqa: S324 - fixture identity only
                )
            )
        if not objects:
            raise gp.GateCParquetError("no local fixtures")
        return tuple(objects)

    @contextmanager
    def open(self, binding: gp.ParquetBinding, path: str):
        self.opened.append(path)
        local = self.root / path
        if path in self.corrupt:
            data = bytearray(local.read_bytes())
            data[-64:] = b"\x00" * 64
            import io

            handle = io.BytesIO(bytes(data))
        else:
            handle = open(local, "rb")
        reader = gp._CountingReader(handle)
        try:
            yield reader
        finally:
            handle.close()


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    # Three files of deliberately different sizes, so equal-per-shard sampling is detectable.
    write_parquet(root / binding.path_prefix / "000.parquet", finewiki_rows(0, 400), 100)
    write_parquet(root / binding.path_prefix / "001.parquet", finewiki_rows(400, 800), 100)
    write_parquet(root / binding.path_prefix / "002.parquet", finewiki_rows(1200, 1600), 100)
    return root


@pytest.fixture
def work_root(tmp_path_factory) -> Path:
    root = RUNS_ROOT / "_pytest_gate_c_parquet"
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(tmp_path_factory.mktemp("case", numbered=True)).name
    target.mkdir(parents=True, exist_ok=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def make_manifest(store: LocalStore, source_key: str = "finewiki_en") -> dict:
    return gp.build_transport_manifest(source_key, store, footer_policy="complete")


def make_config(work_root: Path, **overrides) -> gp.BuildV2Config:
    base = {
        "source": gc.SOURCES["finewiki_en"],
        "output_dir": work_root / "release",
        "work_dir": work_root / "work",
        "target_documents": 40,
        "max_scanned": 4000,
        "max_transfer_bytes": 512 * 1024 * 1024,
        "max_wall_seconds": 300.0,
        "seed": 20260817,
        "units": 8,
        "rows_per_unit": 10,
        "checkpoint_every": 2,
    }
    base.update(overrides)
    return gp.BuildV2Config(**base)


def read_documents(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# --------------------------------------------------------------------------------------
# Bindings and scope
# --------------------------------------------------------------------------------------


def test_no_python_source_is_bound():
    blob = json.dumps({
        k: [b.repo_id, b.path_prefix] for k, b in gp.PARQUET_BINDINGS.items()
    }).lower()
    for forbidden in (
        "python-edu",
        "stack-edu",
        "stackv2",
        "the-stack",
        "common-pile",
        "software-heritage",
        "starcoder",
    ):
        assert forbidden not in blob
    assert not any("python" in key for key in gp.PARQUET_BINDINGS)
    assert sorted(gp.PARQUET_BINDINGS) == sorted(gc.SOURCES)


def test_tool_schema_version_is_v2_and_distinct_from_v1():
    assert gp.TOOL_SCHEMA_VERSION == "petitgpt-corpus-gate-c-v2"
    assert gp.TOOL_SCHEMA_VERSION != gc.TOOL_SCHEMA_VERSION


def test_parquet_mode_never_touches_the_dataset_server(monkeypatch, store_root, work_root):
    def explode(*args, **kwargs):
        raise AssertionError("the Parquet path must not call the Dataset Server")

    monkeypatch.setattr(gc, "_dataset_server_request", explode)
    monkeypatch.setattr(gp, "HubObjectStore", explode)
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    result = gp.build_c0v2(make_config(work_root), manifest, plan, store)
    assert result["published"] is True


# --------------------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------------------


def test_same_manifest_and_seed_give_an_identical_plan(store_root):
    manifest = make_manifest(LocalStore(store_root))
    first = gp.build_selection_plan(manifest, seed=7, units=12, rows_per_unit=10)
    second = gp.build_selection_plan(manifest, seed=7, units=12, rows_per_unit=10)
    assert first["selection_plan_sha256"] == second["selection_plan_sha256"]
    assert first["units"] == second["units"]


def test_different_seed_gives_a_different_traversal_plan(store_root):
    manifest = make_manifest(LocalStore(store_root))
    first = gp.build_selection_plan(manifest, seed=7, units=12, rows_per_unit=10)
    second = gp.build_selection_plan(manifest, seed=8, units=12, rows_per_unit=10)
    assert first["selection_plan_sha256"] != second["selection_plan_sha256"]
    units_a = [(u["file_path"], u["row_group"]) for u in first["units"]]
    units_b = [(u["file_path"], u["row_group"]) for u in second["units"]]
    assert units_a != units_b


def test_seed_changes_the_actual_output_not_only_the_fingerprint(store_root, work_root):
    """The v1 defect was a seed that only changed the fingerprint.  Pin the real behaviour."""
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    results = []
    for seed in (11, 12):
        plan = gp.build_selection_plan(manifest, seed=seed, units=8, rows_per_unit=10)
        config = make_config(
            work_root,
            seed=seed,
            output_dir=work_root / f"release_{seed}",
            work_dir=work_root / f"work_{seed}",
        )
        results.append(gp.build_c0v2(config, manifest, plan, store))
    assert results[0]["documents_sha256"] != results[1]["documents_sha256"]

    docs_a = read_documents(Path(results[0]["output_dir"]) / gp.DOCUMENTS_NAME)
    docs_b = read_documents(Path(results[1]["output_dir"]) / gp.DOCUMENTS_NAME)
    rows_a = {d["provenance"]["source_global_row_index"] for d in docs_a}
    rows_b = {d["provenance"]["source_global_row_index"] for d in docs_b}
    assert rows_a != rows_b


def test_units_are_drawn_without_replacement(store_root):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=3, units=28, rows_per_unit=10)
    units = [(u["file_path"], u["row_group"]) for u in plan["units"]]
    assert len(units) == len(set(units))
    assert plan["distinct_row_groups"] == len(units)


def test_unequal_files_are_not_sampled_equally(store_root):
    """The 400/800/1600-row fixture must not yield a 1:1:1 unit split."""
    manifest = make_manifest(LocalStore(store_root))
    rows_by_file = {item["path"]: item["rows"] for item in manifest["files"]}
    assert sorted(rows_by_file.values()) == [400, 800, 1600]
    counts: dict[str, int] = {path: 0 for path in rows_by_file}
    for seed in range(40):
        plan = gp.build_selection_plan(manifest, seed=seed, units=12, rows_per_unit=10)
        for unit in plan["units"]:
            counts[unit["file_path"]] += 1
    biggest = max(counts, key=lambda p: rows_by_file[p])
    smallest = min(counts, key=lambda p: rows_by_file[p])
    # 4x the rows must draw materially more units than an equal-per-shard scheme would.
    assert counts[biggest] > counts[smallest] * 1.5


def test_unequal_single_row_group_files_are_weighted_not_uniform(tmp_path):
    """File-weighting test.

    Both fixtures below hold exactly ONE row group, so this exercises the FILE layer only. The
    row-group layer is covered separately by
    ``test_single_file_unequal_row_groups_are_weighted_by_row_count``.
    """
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    root = tmp_path / "hub"
    write_parquet(root / binding.path_prefix / "000.parquet", finewiki_rows(0, 900), 900)
    write_parquet(root / binding.path_prefix / "001.parquet", finewiki_rows(900, 100), 100)
    manifest = make_manifest(LocalStore(root))
    picks: dict[int, int] = {}
    for seed in range(60):
        plan = gp.build_selection_plan(manifest, seed=seed, units=1, rows_per_unit=5)
        unit = plan["units"][0]
        picks[unit["row_group_rows"]] = picks.get(unit["row_group_rows"], 0) + 1
    assert picks.get(900, 0) > picks.get(100, 0)


def test_feistel_is_a_permutation():
    key = b"k" * 32
    for domain in (1, 2, 7, 64, 100):
        seen = {gp.feistel_permutation(i, domain, key) for i in range(domain)}
        assert seen == set(range(domain))


# --------------------------------------------------------------------------------------
# Manifest and plan integrity
# --------------------------------------------------------------------------------------


def test_canonical_file_order_makes_listing_order_irrelevant(store_root):
    manifest = make_manifest(LocalStore(store_root))
    shuffled = dict(manifest)
    shuffled["files"] = list(reversed(manifest["files"]))
    shuffled.pop("manifest_sha256")
    shuffled["manifest_sha256"] = gp.transport_manifest_sha256(shuffled)
    # Canonical JSON hashing is order-sensitive by design, so a reordered manifest is a different
    # manifest and must not silently reuse the original plan.
    plan = gp.build_selection_plan(manifest, seed=5, units=6, rows_per_unit=10)
    other = gp.build_selection_plan(shuffled, seed=5, units=6, rows_per_unit=10)
    assert plan["transport_manifest_sha256"] != other["transport_manifest_sha256"]
    # The global row index of a given file is still computed from the recorded order only.
    first_path = manifest["files"][0]["path"]
    assert gp.source_global_row_index(manifest, first_path, 0) == 0


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda m: m["files"][0].__setitem__("path", "data/enwiki/zzz.parquet"), id="path"
        ),
        pytest.param(lambda m: m["files"][0].__setitem__("oid", "0" * 64), id="oid"),
        pytest.param(lambda m: m["files"][0].__setitem__("rows", 999_999), id="row_count"),
        pytest.param(
            lambda m: m["files"][0].__setitem__("row_group_rows", [1, 2, 3]), id="row_groups"
        ),
        pytest.param(lambda m: m.__setitem__("total_rows", 1), id="total_rows"),
    ],
)
def test_transport_manifest_drift_fails_closed(store_root, mutate):
    manifest = make_manifest(LocalStore(store_root))
    mutate(manifest)
    with pytest.raises(gp.GateCParquetError, match="transport manifest checksum mismatch"):
        gp.verify_transport_manifest(manifest)


def test_selection_plan_checksum_tampering_is_detected(store_root):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=5, units=6, rows_per_unit=10)
    plan["units"][0]["rows_to_read"] = 999
    with pytest.raises(gp.GateCParquetError, match="selection plan checksum mismatch"):
        gp.verify_selection_plan(plan)


def test_plan_bound_to_a_different_manifest_is_rejected(store_root, tmp_path, work_root):
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=6, rows_per_unit=10)

    other_root = tmp_path / "hub2"
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    write_parquet(other_root / binding.path_prefix / "000.parquet", finewiki_rows(0, 300), 100)
    other_manifest = make_manifest(LocalStore(other_root))
    with pytest.raises(gp.GateCParquetError, match="transport_manifest_sha256"):
        gp.build_c0v2(make_config(work_root), other_manifest, plan, store)


def test_incomplete_inventory_refuses_to_guess_a_global_row_index(store_root):
    store = LocalStore(store_root)
    manifest = gp.build_transport_manifest(
        "finewiki_en", store, footer_policy="selected_files_only"
    )
    assert manifest["production_row_group_manifest_complete"] is False
    assert manifest["total_rows"] is None
    assert gp.source_global_row_index(manifest, manifest["files"][1]["path"], 5) is None
    assert manifest["files"][0]["rows"] is None


# --------------------------------------------------------------------------------------
# Transport failures
# --------------------------------------------------------------------------------------


def test_truncated_parquet_fails_closed(store_root, work_root):
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    corrupt = LocalStore(store_root, corrupt={plan["units"][0]["file_path"]})
    with pytest.raises(Exception):  # noqa: B017 - pyarrow raises its own footer error
        gp.build_c0v2(make_config(work_root), manifest, plan, corrupt)


def test_parquet_schema_drift_fails_closed(store_root, work_root, tmp_path):
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    drifted_root = tmp_path / "drift"
    shutil.copytree(store_root, drifted_root)
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    target = drifted_root / plan["units"][0]["file_path"]
    rows = pq.read_table(target).to_pylist()
    for row in rows:
        row["page_id"] = str(row["page_id"])  # int64 -> string
    write_parquet(target, rows, 100)
    del binding
    with pytest.raises(gp.GateCParquetError, match="Parquet schema drift"):
        gp.build_c0v2(make_config(work_root), manifest, plan, LocalStore(drifted_root))


# --------------------------------------------------------------------------------------
# Checkpoint, interruption, resume
# --------------------------------------------------------------------------------------


def _prepared(store_root, work_root, **overrides):
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    config = make_config(work_root, **overrides)
    plan = gp.build_selection_plan(
        manifest, seed=config.seed, units=config.units, rows_per_unit=config.rows_per_unit
    )
    return store, manifest, plan, config


def test_v1_checkpoint_is_rejected_outright(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    (config.work_dir / gp.CHECKPOINT_NAME).write_text(
        json.dumps({"tool_schema_version": gc.TOOL_SCHEMA_VERSION, "next_row_index": 10})
    )
    with pytest.raises(gp.GateCParquetError, match="no automatic migration"):
        gp.build_c0v2(config, manifest, plan, store)


def test_interruption_before_a_unit_boundary_keeps_committed_work(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root, target_documents=60)
    with pytest.raises(gp.PlannedInterruption):
        gp.build_c0v2(
            config,
            manifest,
            plan,
            store,
            unit_hook=lambda index: (
                (_ for _ in ()).throw(gp.PlannedInterruption("stop")) if index == 4 else None
            ),
        )
    state = json.loads((config.work_dir / gp.CHECKPOINT_NAME).read_text())
    assert state["unit_cursor"] == 4
    assert state["unit_row_cursor"] == 0
    result = gp.build_c0v2(config, manifest, plan, store)
    assert result["published"] is True
    assert result["resume_count"] == 1


def test_interruption_inside_a_unit_resumes_at_the_row_cursor(store_root, work_root):
    store, manifest, plan, config = _prepared(
        store_root, work_root, target_documents=60, stop_after_documents=25
    )
    first = gp.build_c0v2(config, manifest, plan, store)
    assert first["published"] is False
    state = json.loads((config.work_dir / gp.CHECKPOINT_NAME).read_text())
    assert state["unit_row_cursor"] > 0
    prefix = (config.work_dir / gp.DOCUMENTS_NAME).read_bytes()

    full = make_config(work_root, target_documents=60)
    second = gp.build_c0v2(full, manifest, plan, store)
    published = (Path(second["output_dir"]) / gp.DOCUMENTS_NAME).read_bytes()
    assert published.startswith(prefix)
    assert second["accepted"] == 60


def test_resume_never_duplicates_committed_records(store_root, work_root):
    store, manifest, plan, config = _prepared(
        store_root, work_root, target_documents=60, stop_after_documents=20
    )
    gp.build_c0v2(config, manifest, plan, store)
    result = gp.build_c0v2(make_config(work_root, target_documents=60), manifest, plan, store)
    docs = read_documents(Path(result["output_dir"]) / gp.DOCUMENTS_NAME)
    assert len({d["source_record_id"] for d in docs}) == len(docs)
    assert len({d["text_sha256"] for d in docs}) == len(docs)
    rows = [d["provenance"]["source_global_row_index"] for d in docs]
    assert len(set(rows)) == len(rows)


def test_manifest_drift_between_runs_refuses_to_resume(store_root, work_root):
    store, manifest, plan, config = _prepared(
        store_root, work_root, target_documents=60, stop_after_documents=20
    )
    gp.build_c0v2(config, manifest, plan, store)
    drifted = json.loads(json.dumps(manifest))
    drifted["files"][0]["oid"] = "1" * 64
    drifted.pop("manifest_sha256")
    drifted["manifest_sha256"] = gp.transport_manifest_sha256(drifted)
    drifted_plan = gp.build_selection_plan(
        drifted, seed=config.seed, units=config.units, rows_per_unit=config.rows_per_unit
    )
    with pytest.raises(gp.GateCParquetError, match="drifted|fingerprint mismatch"):
        gp.build_c0v2(make_config(work_root, target_documents=60), drifted, drifted_plan, store)


def test_corrupted_v2_checkpoint_is_a_hard_error(store_root, work_root):
    store, manifest, plan, config = _prepared(
        store_root, work_root, target_documents=60, stop_after_documents=20
    )
    gp.build_c0v2(config, manifest, plan, store)
    path = config.work_dir / gp.CHECKPOINT_NAME
    state = json.loads(path.read_text())
    state["counters"]["accepted"] = 999
    path.write_text(json.dumps(state))
    with pytest.raises(gp.GateCParquetError, match="checksum mismatch"):
        gp.build_c0v2(make_config(work_root, target_documents=60), manifest, plan, store)


# --------------------------------------------------------------------------------------
# Release semantics and provenance
# --------------------------------------------------------------------------------------


def test_release_is_always_a_non_promotable_c0_diagnostic(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    assert result["stop_reason"] == "target_reached"
    assert result["release_kind"] == "c0_diagnostic"
    assert result["promotion_eligible"] is False
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    assert published["promotion_eligible"] is False
    assert published["hard_stops"]["production_candidate_quota_authorized"] is False
    assert gp.verify_release_v2(Path(result["output_dir"]))["promotion_eligible"] is False


def test_cap_stop_is_published_but_never_marked_complete(store_root, work_root):
    """A byte-capped run publishes, but must never look like a completed one."""
    store, manifest, plan, config = _prepared(
        store_root, work_root, target_documents=200, max_transfer_bytes=1
    )
    result = gp.build_c0v2(config, manifest, plan, store)
    assert result["stop_reason"] == "byte_cap"
    assert result["accepted"] < config.target_documents
    assert result["promotion_eligible"] is False
    assert result["release_kind"] == "c0_diagnostic"
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    assert published["stop_reason"] == "byte_cap"
    assert published["promotion_eligible"] is False
    assert gp.verify_release_v2(Path(result["output_dir"]))["promotion_eligible"] is False


def test_time_cap_stop_is_also_non_promotable(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root, target_documents=200)
    ticks = iter(range(0, 100_000))
    config = make_config(work_root, target_documents=200, max_wall_seconds=2.0)
    result = gp.build_c0v2(config, manifest, plan, store, clock=lambda: float(next(ticks)))
    assert result["stop_reason"] == "time_cap"
    assert result["promotion_eligible"] is False


def test_v1_manifest_also_declares_non_promotability():
    """The same hazard existed in v1: a cap-stopped release looked complete."""
    import inspect

    source = inspect.getsource(gc._make_manifest)
    assert '"release_kind": "c0_diagnostic"' in source
    assert '"promotion_eligible": False' in source


def test_provenance_records_the_parquet_transport(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    docs = read_documents(Path(result["output_dir"]) / gp.DOCUMENTS_NAME)
    assert docs
    for doc in docs:
        provenance = doc["provenance"]
        assert provenance["transport"] == "huggingface_hub_parquet"
        assert provenance["transport"] != "huggingface_dataset_server_rows"
        for key in (
            "parquet_file",
            "row_group",
            "row_in_group",
            "row_in_file",
            "source_global_row_index",
            "parquet_file_oid",
            "transport_manifest_sha256",
            "selection_plan_sha256",
        ):
            assert key in provenance
        assert not doc["text"].startswith("[BOS]")
        assert "MUST NOT BE EMITTED" not in doc["text"]


def test_published_release_carries_plan_and_transport_manifest(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    output = Path(result["output_dir"])
    assert sorted(item.name for item in output.iterdir()) == [
        gp.CHECKSUMS_NAME,
        gp.DOCUMENTS_NAME,
        gp.MANIFEST_NAME,
        gp.SELECTION_PLAN_NAME,
        gp.TRANSPORT_MANIFEST_NAME,
    ]
    assert not (output.parent / f".{output.name}.partial").exists()
    verified = gp.verify_release_v2(output)
    assert verified["accepted"] == result["accepted"]


def test_refuses_to_overwrite_a_published_release(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    gp.build_c0v2(config, manifest, plan, store)
    with pytest.raises(gp.GateCParquetError, match="refusing to overwrite"):
        gp.build_c0v2(make_config(work_root, work_dir=work_root / "work2"), manifest, plan, store)


def test_coverage_spans_multiple_files_and_row_groups(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    accepted = published["accepted_document_coverage"]
    transport = published["transport_read_coverage"]
    assert accepted["accepted_file_count"] >= 2
    assert accepted["accepted_row_group_count"] >= 4
    assert transport["files_opened"] >= accepted["accepted_file_count"]
    assert transport["row_groups_opened"] >= accepted["accepted_row_group_count"]
    histogram = accepted["global_row_decile_histogram"]
    assert histogram is not None and len(histogram) >= 3
    # The deprecated block must state which coverage it mirrors.
    assert published["coverage"]["deprecated"] is True
    assert published["coverage"]["files_touched"] == accepted["accepted_file_count"]


# --------------------------------------------------------------------------------------
# Source-specific identity and field discipline
# --------------------------------------------------------------------------------------


def test_cosmopedia_global_row_identity_is_stable_and_manifest_derived(tmp_path, work_root):
    binding = gp.PARQUET_BINDINGS["cosmopedia_v2"]
    root = tmp_path / "hub"
    write_parquet(root / binding.path_prefix / "a.parquet", cosmopedia_rows(0, 300), 100)
    write_parquet(root / binding.path_prefix / "b.parquet", cosmopedia_rows(300, 300), 100)
    store = LocalStore(root)
    manifest = make_manifest(store, "cosmopedia_v2")
    assert manifest["production_row_group_manifest_complete"] is True
    assert gc.SOURCES["cosmopedia_v2"].natural_id_path is None
    plan = gp.build_selection_plan(manifest, seed=20260817, units=6, rows_per_unit=10)

    outputs = []
    for run in range(2):
        config = gp.BuildV2Config(
            source=gc.SOURCES["cosmopedia_v2"],
            output_dir=work_root / f"release{run}",
            work_dir=work_root / f"work{run}",
            target_documents=30,
            max_scanned=600,
            max_transfer_bytes=512 * 1024 * 1024,
            max_wall_seconds=300.0,
            seed=20260817,
            units=6,
            rows_per_unit=10,
        )
        outputs.append(gp.build_c0v2(config, manifest, plan, store))
    assert outputs[0]["documents_sha256"] == outputs[1]["documents_sha256"]
    docs = read_documents(Path(outputs[0]["output_dir"]) / gp.DOCUMENTS_NAME)
    for doc in docs:
        assert doc["natural_id"] == f"grow:{doc['provenance']['source_global_row_index']}"
        assert "PROMPT TEXT" not in doc["text"]
        assert not doc["text"].startswith(" ")
    # Global index equals preceding-file rows plus row-in-file, from the manifest alone.
    second_file = manifest["files"][1]["path"]
    assert gp.source_global_row_index(manifest, second_file, 0) == manifest["files"][0]["rows"]


def test_finephrase_emits_the_rollout_body_not_the_top_level_text(tmp_path, work_root):
    binding = gp.PARQUET_BINDINGS["finephrase_tutorial"]
    root = tmp_path / "hub"
    write_parquet(root / binding.path_prefix / "a.parquet", finephrase_rows(0, 200), 50)
    write_parquet(root / binding.path_prefix / "b.parquet", finephrase_rows(200, 200), 50)
    store = LocalStore(root)
    manifest = make_manifest(store, "finephrase_tutorial")
    plan = gp.build_selection_plan(manifest, seed=20260817, units=6, rows_per_unit=10)
    config = gp.BuildV2Config(
        source=gc.SOURCES["finephrase_tutorial"],
        output_dir=work_root / "release",
        work_dir=work_root / "work",
        target_documents=20,
        max_scanned=400,
        max_transfer_bytes=512 * 1024 * 1024,
        max_wall_seconds=300.0,
        seed=20260817,
        units=6,
        rows_per_unit=10,
    )
    result = gp.build_c0v2(config, manifest, plan, store)
    docs = read_documents(Path(result["output_dir"]) / gp.DOCUMENTS_NAME)
    assert docs
    for doc in docs:
        assert "ORIGINAL FINEWEB SOURCE" not in doc["text"]
        assert doc["text"].startswith("Making bread, part ")
    assert len({doc["provenance"]["parquet_file"] for doc in docs}) >= 2


def test_c0_format_quota_and_forum_policy_are_recorded_not_hidden(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    config = make_config(work_root, forum_deny=("meta_example_com",), forum_cap=5)
    result = gp.build_c0v2(config, manifest, plan, store)
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    assert published["c0_policy"]["forum_deny"] == ["meta_example_com"]
    assert published["c0_policy"]["forum_cap"] == 5


def test_filters_are_reused_from_the_frozen_v1_module():
    """No source filter may be reimplemented here."""
    import inspect

    source = inspect.getsource(gp)
    assert "evaluate_row" in source
    for forbidden in (
        "_filter_finewiki",
        "_filter_pes2o",
        "_filter_stackexchange",
        "_filter_cosmopedia",
        "_filter_finephrase",
        "_pathological_repetition",
    ):
        assert f"def {forbidden}" not in source


# --------------------------------------------------------------------------------------
# Option-B sampler contract (Correction A)
# --------------------------------------------------------------------------------------


def test_sampler_version_is_the_option_b_v2_version():
    assert gp.SAMPLER_VERSION == "pps-rowgroup-fixed-head-cluster-v2"


def test_no_false_representative_claims_in_the_module():
    """The frozen Option-B wording forbids representative/equal-inclusion language."""
    import inspect

    text = inspect.getsource(gp).lower()
    for banned in (
        "representative traversal",
        "row-weighted representative",
        "equally likely",
        "the two cancel",
        "deterministic representative",
        "seeded representative",
    ):
        assert banned not in text, banned
    # "representative sampler" may appear only in a negated form.
    for index in range(len(text)):
        position = text.find("representative sampler", index)
        if position < 0:
            break
        window = text[max(0, position - 40) : position]
        assert "not a" in window or "not_a" in window or "false" in window, window


def test_selection_plan_records_the_honest_contract(store_root):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=5, units=6, rows_per_unit=10)
    assert plan["file_weighting"] == "exact_row_counts"
    assert plan["file_weighting_exact_for_rows"] is True
    assert plan["forced_distinct_file_first_pass"] is True
    assert plan["row_group_weighting"] == "exact_row_count_within_resolved_file"
    assert plan["within_row_group_selection"] == "contiguous_head_slice"
    assert plan["within_row_group_start"] == 0
    assert plan["within_row_group_seed_sensitive"] is False
    assert plan["without_replacement_scope"] == "selected_file_row_group_unit"
    assert plan["row_level_nonzero_inclusion_for_all_rows"] is False
    assert plan["row_level_equal_inclusion_proven"] is False
    assert plan["global_exact_row_weighting"] is False
    assert plan["representative_sampler"] is False
    assert plan["allowed_claim"] == gp.ALLOWED_SAMPLER_CLAIM
    assert "not a row-level representative sampler" in plan["allowed_claim"]


def test_selected_files_only_plan_calls_its_weighting_a_proxy(store_root):
    store = LocalStore(store_root)
    manifest = gp.build_transport_manifest(
        "finewiki_en", store, footer_policy="selected_files_only"
    )
    plan = gp.build_selection_plan(manifest, seed=5, units=3, rows_per_unit=10, store=store)
    assert plan["file_weighting"] == "hub_file_byte_size_proxy"
    assert plan["file_weighting_exact_for_rows"] is False


def test_release_manifest_declares_non_representative(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    sampler = published["sampler"]
    assert sampler["representative_sampler"] is False
    assert sampler["row_level_equal_inclusion_proven"] is False
    assert sampler["global_exact_row_weighting"] is False
    assert sampler["allowed_claim"] == gp.ALLOWED_SAMPLER_CLAIM
    assert gp.verify_release_v2(Path(result["output_dir"]))["representative_sampler"] is False


def test_only_the_head_of_each_selected_row_group_is_read(store_root, work_root):
    """Rows deeper than rows_per_unit have inclusion probability exactly zero."""
    store, manifest, plan, config = _prepared(store_root, work_root, rows_per_unit=7, units=8)
    result = gp.build_c0v2(config, manifest, plan, store)
    docs = read_documents(Path(result["output_dir"]) / gp.DOCUMENTS_NAME)
    assert docs
    assert max(d["provenance"]["row_in_group"] for d in docs) < 7
    assert all(u["row_group_rows"] > 7 for u in plan["units"])


def test_single_file_unequal_row_groups_are_weighted_by_row_count(tmp_path):
    """A real unequal-row-group fixture: ONE file whose row groups differ in size.

    The pre-existing unequal test used two single-row-group files, which only exercised file
    weighting; this one isolates the row-group layer.
    """
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    root = tmp_path / "hub"
    path = root / binding.path_prefix / "000.parquet"
    rows = finewiki_rows(0, 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, pa.Table.from_pylist(rows).schema)
    writer.write_table(pa.Table.from_pylist(rows[:900]))  # row group 0: 900 rows
    writer.write_table(pa.Table.from_pylist(rows[900:]))  # row group 1: 100 rows
    writer.close()
    manifest = make_manifest(LocalStore(root))
    assert manifest["files"][0]["row_group_rows"] == [900, 100]
    picks = {900: 0, 100: 0}
    for seed in range(60):
        plan = gp.build_selection_plan(manifest, seed=seed, units=1, rows_per_unit=5)
        picks[plan["units"][0]["row_group_rows"]] += 1
    assert picks[900] > picks[100] * 2


# --------------------------------------------------------------------------------------
# Coverage contract (Correction B)
# --------------------------------------------------------------------------------------


def test_transport_coverage_accumulates_across_a_resume(store_root, work_root):
    store, manifest, plan, config = _prepared(
        store_root,
        work_root,
        target_documents=60,
        rows_per_unit=6,
        units=10,
        stop_after_documents=18,
    )
    first = gp.build_c0v2(config, manifest, plan, store)
    assert first["published"] is False
    state = json.loads((config.work_dir / gp.CHECKPOINT_NAME).read_text())
    first_files = set(state["transport_files_opened"])
    first_groups = set(state["transport_row_groups_opened"])
    assert len(first_groups) >= 2

    full = make_config(work_root, target_documents=60, rows_per_unit=6, units=10)
    second = gp.build_c0v2(full, manifest, plan, store)
    published = json.loads((Path(second["output_dir"]) / gp.MANIFEST_NAME).read_text())
    transport = published["transport_read_coverage"]
    assert first_files <= set(transport["files_opened_list"])
    assert first_groups <= set(transport["row_groups_opened_list"])
    assert transport["row_groups_opened"] > len(first_groups)

    accepted = published["accepted_document_coverage"]
    rebuilt = gp.reconstruct_accepted_document_coverage(
        Path(second["output_dir"]) / gp.DOCUMENTS_NAME, manifest["total_rows"]
    )
    assert accepted["accepted_files"] == rebuilt["accepted_files"]
    assert accepted["accepted_row_groups"] == rebuilt["accepted_row_groups"]
    assert accepted["accepted_documents"] == second["accepted"]


def test_transport_coverage_can_exceed_accepted_coverage(tmp_path, work_root):
    """A unit whose rows are all rejected is opened but contributes no accepted document."""
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    root = tmp_path / "hub"
    good = finewiki_rows(0, 100)
    bad = [dict(r, text="too short") for r in finewiki_rows(100, 100)]
    write_parquet(root / binding.path_prefix / "000.parquet", good, 100)
    write_parquet(root / binding.path_prefix / "001.parquet", bad, 100)
    store = LocalStore(root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=2, rows_per_unit=20)
    config = make_config(work_root, target_documents=20, units=2, rows_per_unit=20)
    result = gp.build_c0v2(config, manifest, plan, store)
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    transport = published["transport_read_coverage"]
    accepted = published["accepted_document_coverage"]
    assert transport["files_opened"] == 2
    assert accepted["accepted_file_count"] == 1
    assert transport["row_groups_opened"] > accepted["accepted_row_group_count"]
    assert published["accounting"]["rejections"]["below_min_bytes"] == 20


# --------------------------------------------------------------------------------------
# Per-file schema pinning (Correction C)
# --------------------------------------------------------------------------------------


def test_selected_files_only_pins_schema_per_file(tmp_path, work_root):
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    root = tmp_path / "hub"
    write_parquet(root / binding.path_prefix / "000.parquet", finewiki_rows(0, 200), 100)
    write_parquet(root / binding.path_prefix / "001.parquet", finewiki_rows(200, 200), 100)
    store = LocalStore(root)
    manifest = gp.build_transport_manifest(
        "finewiki_en", store, footer_policy="selected_files_only"
    )
    assert all(item["schema_hash"] is None for item in manifest["files"])
    plan = gp.build_selection_plan(manifest, seed=20260817, units=2, rows_per_unit=10, store=store)
    topology = plan["resolved_file_topology"]
    assert topology and all(v["schema_hash"] for v in topology.values())
    assert all(v["row_group_rows"] for v in topology.values())

    drifted = tmp_path / "drift"
    shutil.copytree(root, drifted)
    target = drifted / plan["units"][0]["file_path"]
    rows = pq.read_table(target).to_pylist()
    for row in rows:
        row["page_id"] = str(row["page_id"])
    write_parquet(target, rows, 100)
    config = make_config(work_root, target_documents=10, units=2, rows_per_unit=10)
    with pytest.raises(gp.GateCParquetError, match="Parquet schema drift"):
        gp.build_c0v2(config, manifest, plan, LocalStore(drifted))


# --------------------------------------------------------------------------------------
# Persist before read (Correction D)
# --------------------------------------------------------------------------------------


def test_inputs_are_persisted_and_verified_before_the_first_body_read(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    seen = {}

    class Instrumented(LocalStore):
        def open(self, binding, path):
            work = config.work_dir
            for name in (gp.TRANSPORT_MANIFEST_NAME, gp.SELECTION_PLAN_NAME):
                target = work / name
                sidecar = work / f"{name}.sha256"
                assert target.exists(), f"{name} must exist before the first body open"
                assert sidecar.exists(), f"{name}.sha256 must exist before the first body open"
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                assert sidecar.read_text().split("  ")[0] == digest
                seen[name] = digest
            return super().open(binding, path)

    result = gp.build_c0v2(config, manifest, plan, Instrumented(store.root))
    assert set(seen) == {gp.TRANSPORT_MANIFEST_NAME, gp.SELECTION_PLAN_NAME}
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    assert published["persisted_inputs_before_first_body_read"] == seen


def test_persisted_inputs_that_differ_fail_closed(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    (config.work_dir / gp.SELECTION_PLAN_NAME).write_text('{"tampered": true}')
    with pytest.raises(gp.GateCParquetError, match="refusing to overwrite"):
        gp.build_c0v2(config, manifest, plan, store)


# --------------------------------------------------------------------------------------
# JSONL framing (Correction E)
# --------------------------------------------------------------------------------------


def test_unicode_line_separators_do_not_split_a_record(tmp_path, work_root):
    binding = gp.PARQUET_BINDINGS["finewiki_en"]
    root = tmp_path / "hub"
    rows = finewiki_rows(0, 100)
    marker = "alpha\u2028beta\u2029gamma\u0085delta"
    rows[0]["text"] = f"Topic X. {marker} {_prose(7)}"
    write_parquet(root / binding.path_prefix / "000.parquet", rows, 100)
    store = LocalStore(root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=1, rows_per_unit=20)
    config = make_config(work_root, target_documents=20, units=1, rows_per_unit=20)
    result = gp.build_c0v2(config, manifest, plan, store)
    raw = (Path(result["output_dir"]) / gp.DOCUMENTS_NAME).read_bytes()

    byte_lines = [line for line in raw.split(b"\n") if line]
    str_lines = [line for line in raw.decode().splitlines() if line]
    assert len(byte_lines) == len(str_lines) == 20
    assert b"\xe2\x80\xa8" not in raw and b"\xe2\x80\xa9" not in raw
    assert b"\xc2\x85" not in raw

    docs = [json.loads(line) for line in byte_lines]
    hit = [d for d in docs if marker in d["text"]]
    assert len(hit) == 1
    # Body characters survive verbatim, and the digests stay based on the original text.
    assert "\u2028" in hit[0]["text"] and "\u2029" in hit[0]["text"] and "\u0085" in hit[0]["text"]
    assert hashlib.sha256(hit[0]["text"].encode()).hexdigest() == hit[0]["text_sha256"]
    assert len(hit[0]["text"].encode()) == hit[0]["text_bytes"]
    assert gp.verify_release_v2(Path(result["output_dir"]))["jsonl_byte_lines_equal_str_lines"]


# --------------------------------------------------------------------------------------
# Resource accounting (Correction F)
# --------------------------------------------------------------------------------------


def test_resource_accounting_is_honest_about_wire_bytes(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    published = json.loads((Path(result["output_dir"]) / gp.MANIFEST_NAME).read_text())
    res = published["resource_accounting"]
    assert res["wire_bytes"] is None
    assert res["wire_bytes_measured"] is False
    assert res["network_byte_cap_status"] == "partially_verified"
    assert res["cap_enforcement"] == "pre_unit_soft_cap_may_overshoot_one_unit"
    assert res["gpu_api_called"] is False
    assert res["body_reader_exposed_bytes"] > 0
    assert res["cumulative_body_reader_exposed_bytes"] >= res["body_reader_exposed_bytes"]
    assert "current_invocation_wall_seconds" in published
    assert "cumulative_build_wall_seconds" in published


def test_cumulative_resource_accounting_survives_a_resume(store_root, work_root):
    store, manifest, plan, config = _prepared(
        store_root,
        work_root,
        target_documents=60,
        rows_per_unit=6,
        units=10,
        stop_after_documents=18,
    )
    first = gp.build_c0v2(config, manifest, plan, store)
    first_bytes = first["cumulative_body_reader_exposed_bytes"]
    assert first_bytes > 0
    full = make_config(work_root, target_documents=60, rows_per_unit=6, units=10)
    second = gp.build_c0v2(full, manifest, plan, store)
    published = json.loads((Path(second["output_dir"]) / gp.MANIFEST_NAME).read_text())
    res = published["resource_accounting"]
    assert res["cumulative_body_reader_exposed_bytes"] > first_bytes
    assert res["cumulative_body_reader_exposed_bytes"] > res["body_reader_exposed_bytes"]
    assert (
        published["cumulative_build_wall_seconds"] >= published["current_invocation_wall_seconds"]
    )


# --------------------------------------------------------------------------------------
# Upstream claim scoping (Correction G)
# --------------------------------------------------------------------------------------


def test_export_notes_do_not_claim_upstream_population_equivalence():
    for binding in gp.PARQUET_BINDINGS.values():
        note = binding.export_note.lower()
        assert "full row count" not in note
        assert "equals the pinned" not in note
        assert "outside this correction task" in note


def test_transport_manifest_marks_upstream_audit_as_not_performed(store_root):
    manifest = make_manifest(LocalStore(store_root))
    assert manifest["upstream_population_equivalence"] == "not_audited_in_this_task"


# --------------------------------------------------------------------------------------
# Strict selection-plan validation: a self-consistent checksum must not buy a bad contract
# --------------------------------------------------------------------------------------


class CountingStore(LocalStore):
    """LocalStore that records how many times a body was opened."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.open_count = 0

    @contextmanager
    def open(self, binding, path):
        self.open_count += 1
        with super().open(binding, path) as reader:
            yield reader


def mutate_plan(plan: dict, mutator) -> dict:
    """Deep-copy a valid plan, mutate it, and re-checksum so it is self-consistent."""
    forged = copy.deepcopy(plan)
    forged.pop("selection_plan_sha256", None)
    mutator(forged)
    forged["selection_plan_sha256"] = gp.selection_plan_sha256(forged)
    assert gp.verify_selection_plan(forged) == forged["selection_plan_sha256"]
    return forged


def assert_plan_rejected_without_touching_anything(
    store_root: Path, work_root: Path, manifest: dict, forged: dict, match: str
) -> None:
    store = CountingStore(store_root)
    config = make_config(work_root)
    with pytest.raises(gp.GateCParquetError, match=match):
        gp.build_c0v2(config, manifest, forged, store)
    assert store.open_count == 0, "an invalid plan must open zero bodies"
    work = Path(config.work_dir)
    for name in (
        gp.SELECTION_PLAN_NAME,
        f"{gp.SELECTION_PLAN_NAME}.sha256",
        gp.TRANSPORT_MANIFEST_NAME,
        f"{gp.TRANSPORT_MANIFEST_NAME}.sha256",
        gp.CHECKPOINT_NAME,
        gp.DOCUMENTS_NAME,
    ):
        assert not (work / name).exists(), f"invalid plan must not persist {name}"
    assert not Path(config.output_dir).exists()


def test_stale_sampler_version_is_rejected(store_root, work_root):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(
        plan, lambda p: p.__setitem__("sampler_version", "pps-rowgroup-fixed-cluster-v1")
    )
    assert_plan_rejected_without_touching_anything(
        store_root, work_root, manifest, forged, "sampler_version"
    )


def test_stale_tool_schema_version_is_rejected(store_root, work_root):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(
        plan, lambda p: p.__setitem__("tool_schema_version", "petitgpt-corpus-gate-c-v1")
    )
    assert_plan_rejected_without_touching_anything(
        store_root, work_root, manifest, forged, "tool_schema_version"
    )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(
            lambda p: p.__setitem__("representative_sampler", True),
            "representative_sampler",
            id="representative_sampler_true",
        ),
        pytest.param(
            lambda p: p.__setitem__(
                "allowed_claim", gp.ALLOWED_SAMPLER_CLAIM.replace("not a row-level", "a row-level")
            ),
            "allowed_claim",
            id="near_miss_allowed_claim",
        ),
        pytest.param(
            lambda p: p.__setitem__("within_row_group_start", 5),
            "within_row_group_start",
            id="nonzero_within_group_start",
        ),
        pytest.param(
            lambda p: p.__setitem__("global_exact_row_weighting", True),
            "global_exact_row_weighting",
            id="global_exact_row_weighting_true",
        ),
        pytest.param(
            lambda p: p.__setitem__("file_weighting", "hub_file_byte_size_proxy"),
            "file_weighting",
            id="weighting_contradicts_complete_topology",
        ),
        pytest.param(
            lambda p: p.__setitem__("file_weighting_exact_for_rows", False),
            "file_weighting_exact_for_rows",
            id="exactness_contradicts_topology",
        ),
        pytest.param(
            lambda p: p.__setitem__("within_row_group_seed_sensitive", True),
            "within_row_group_seed_sensitive",
            id="claims_seed_sensitive_offset",
        ),
        pytest.param(
            lambda p: p.__setitem__("row_level_equal_inclusion_proven", True),
            "row_level_equal_inclusion_proven",
            id="claims_equal_inclusion",
        ),
        pytest.param(
            lambda p: p.pop("without_replacement_scope"),
            "without_replacement_scope",
            id="missing_frozen_field",
        ),
        pytest.param(
            lambda p: p.__setitem__("forced_distinct_file_first_pass", 1),
            "forced_distinct_file_first_pass",
            id="truthy_int_instead_of_bool",
        ),
    ],
)
def test_frozen_semantic_contradictions_are_rejected(store_root, work_root, mutator, match):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(plan, mutator)
    assert_plan_rejected_without_touching_anything(store_root, work_root, manifest, forged, match)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(
            lambda p: p["resolved_file_topology"].pop(p["units"][0]["file_path"]),
            "does not cover exactly the selected files",
            id="missing_topology_entry",
        ),
        pytest.param(
            lambda p: p["resolved_file_topology"][p["units"][0]["file_path"]].__setitem__(
                "schema_hash", None
            ),
            "schema_hash must be a 64-hex",
            id="null_schema_hash",
        ),
        pytest.param(
            lambda p: p["resolved_file_topology"][p["units"][0]["file_path"]].__setitem__(
                "schema_hash", "not-a-sha"
            ),
            "schema_hash must be a 64-hex",
            id="malformed_schema_hash",
        ),
        pytest.param(
            lambda p: p["resolved_file_topology"][p["units"][0]["file_path"]].__setitem__(
                "row_group_count", 999
            ),
            "row_group_count",
            id="row_group_count_mismatch",
        ),
        pytest.param(
            lambda p: p["units"][0].__setitem__("row_group_rows", 7),
            "rows_to_read",
            id="unit_row_group_rows_contradicts_topology",
        ),
        pytest.param(
            lambda p: p["resolved_file_topology"][p["units"][0]["file_path"]].__setitem__(
                "rows", 12345
            ),
            "!= sum of row groups",
            id="rows_not_sum_of_row_groups",
        ),
    ],
)
def test_broken_exact_file_topology_is_rejected(store_root, work_root, mutator, match):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(plan, mutator)
    assert_plan_rejected_without_touching_anything(store_root, work_root, manifest, forged, match)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(lambda p: p.__setitem__("dataset", "someone/else"), "dataset", id="dataset"),
        pytest.param(
            lambda p: p.__setitem__("data_revision", "0" * 40),
            "data_revision",
            id="data_revision",
        ),
        pytest.param(
            lambda p: p.__setitem__("parquet_revision", "1" * 40),
            "parquet_revision",
            id="parquet_revision",
        ),
        pytest.param(
            lambda p: p.__setitem__("transport_manifest_sha256", "2" * 64),
            "transport_manifest_sha256",
            id="transport_manifest_binding",
        ),
        pytest.param(
            lambda p: p["units"][0].__setitem__("file_oid", "3" * 64),
            "file_oid",
            id="file_oid",
        ),
        pytest.param(lambda p: p.__setitem__("split", "validation"), "split", id="split"),
        pytest.param(
            lambda p: p.__setitem__("production_row_group_manifest_complete", False),
            "production_row_group_manifest_complete|file_weighting",
            id="topology_completeness_flag",
        ),
    ],
)
def test_binding_contradictions_are_rejected(store_root, work_root, mutator, match):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(plan, mutator)
    assert_plan_rejected_without_touching_anything(store_root, work_root, manifest, forged, match)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(lambda p: p.__setitem__("seed", 999), "seed", id="seed"),
        pytest.param(
            lambda p: p.__setitem__("requested_units", 999), "requested_units", id="units"
        ),
        pytest.param(
            lambda p: p.__setitem__("rows_per_unit", 999), "rows_per_unit", id="rows_per_unit"
        ),
    ],
)
def test_config_binding_contradictions_are_rejected(store_root, work_root, mutator, match):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(plan, mutator)
    assert_plan_rejected_without_touching_anything(store_root, work_root, manifest, forged, match)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(
            lambda p: p.__setitem__("planned_units", 999), "planned_units", id="planned_units"
        ),
        pytest.param(
            lambda p: p.__setitem__("distinct_files", 999), "distinct_files", id="distinct_files"
        ),
        pytest.param(
            lambda p: p.__setitem__("planned_rows", 999), "planned_rows", id="planned_rows"
        ),
        pytest.param(
            lambda p: p["units"].__setitem__(1, copy.deepcopy(p["units"][0])),
            "distinct_files|distinct_row_groups|order|repeats unit",
            id="duplicate_unit",
        ),
    ],
)
def test_derived_field_contradictions_are_rejected(store_root, work_root, mutator, match):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=8, rows_per_unit=10)
    forged = mutate_plan(plan, mutator)
    assert_plan_rejected_without_touching_anything(store_root, work_root, manifest, forged, match)


def test_selected_files_only_plan_still_validates(store_root, work_root):
    """Positive regression: the selected-files-only contract must keep passing."""
    store = LocalStore(store_root)
    manifest = gp.build_transport_manifest(
        "finewiki_en", store, footer_policy="selected_files_only"
    )
    plan = gp.build_selection_plan(manifest, seed=20260817, units=3, rows_per_unit=10, store=store)
    assert gp.validate_selection_plan(plan, manifest) == plan["selection_plan_sha256"]
    config = gp.BuildV2Config(
        source=gc.SOURCES["finewiki_en"],
        output_dir=work_root / "release_sfo",
        work_dir=work_root / "work_sfo",
        target_documents=20,
        max_scanned=400,
        max_transfer_bytes=512 * 1024 * 1024,
        max_wall_seconds=300.0,
        seed=20260817,
        units=3,
        rows_per_unit=10,
    )
    result = gp.build_c0v2(config, manifest, plan, store)
    assert result["published"] is True
    assert gp.verify_release_v2(Path(result["output_dir"]))["accepted"] == result["accepted"]


def test_validator_is_shared_by_build_and_release_verification(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    output = Path(result["output_dir"])
    assert gp.verify_release_v2(output)["accepted"] == result["accepted"]

    # Forge the PUBLISHED plan so it stays checksum-consistent but breaks the frozen contract.
    published_plan = json.loads((output / gp.SELECTION_PLAN_NAME).read_text())
    forged = mutate_plan(published_plan, lambda p: p.__setitem__("representative_sampler", True))
    (output / gp.SELECTION_PLAN_NAME).write_text(json.dumps(forged, indent=2, sort_keys=True))
    with pytest.raises(gp.GateCParquetError, match="representative_sampler"):
        gp.verify_release_v2(output)


def test_expected_schema_hash_helper_refuses_a_missing_pin(store_root):
    manifest = make_manifest(LocalStore(store_root))
    plan = gp.build_selection_plan(manifest, seed=20260817, units=4, rows_per_unit=10)
    path = plan["units"][0]["file_path"]
    assert (
        gp.expected_schema_hash_for(plan, path)
        == plan["resolved_file_topology"][path]["schema_hash"]
    )
    with pytest.raises(gp.GateCParquetError, match="no validated schema hash is pinned"):
        gp.expected_schema_hash_for(plan, "data/enwiki/does-not-exist.parquet")


# --------------------------------------------------------------------------------------
# Release-manifest contract: a checksum-consistent forgery must still be rejected
# --------------------------------------------------------------------------------------


def forge_release_manifest(output: Path, mutator) -> None:
    """Rewrite manifest.json in production format and re-sync MANIFEST.sha256.

    Every file checksum stays self-consistent, so a rejection can only come from the semantic
    contract and never from a stale checksum.
    """
    manifest = json.loads((output / gp.MANIFEST_NAME).read_text())
    mutator(manifest)
    (output / gp.MANIFEST_NAME).write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    lines = []
    for entry in (output / gp.CHECKSUMS_NAME).read_text().splitlines():
        if not entry:
            continue
        _, name = entry.split("  ", 1)
        lines.append(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n")
    (output / gp.CHECKSUMS_NAME).write_text("".join(lines))


def _built_release(store_root, work_root):
    store, manifest, plan, config = _prepared(store_root, work_root)
    result = gp.build_c0v2(config, manifest, plan, store)
    return Path(result["output_dir"]), result


def test_forged_release_helper_keeps_checksums_consistent(store_root, work_root):
    """Guard the guard: the helper must not smuggle in a stale-checksum rejection."""
    output, _ = _built_release(store_root, work_root)
    forge_release_manifest(output, lambda m: m["sampler"].__setitem__("seed", m["sampler"]["seed"]))
    for entry in (output / gp.CHECKSUMS_NAME).read_text().splitlines():
        digest, name = entry.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert gp.verify_release_v2(output)["release_manifest_validated"] is True


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(
            lambda m: m["sampler"].__setitem__("representative_sampler", True),
            "representative_sampler",
            id="representative_sampler_true",
        ),
        pytest.param(
            lambda m: m["sampler"].__setitem__(
                "allowed_claim", gp.ALLOWED_SAMPLER_CLAIM.replace("not a row-level", "a row-level")
            ),
            "allowed_claim",
            id="near_miss_allowed_claim",
        ),
        pytest.param(
            lambda m: m["sampler"].__setitem__("version", "pps-rowgroup-fixed-cluster-v1"),
            "version",
            id="stale_sampler_version",
        ),
        pytest.param(
            lambda m: m["hard_stops"].__setitem__("production_candidate_quota_authorized", True),
            "hard_stops",
            id="quota_authorized_true",
        ),
        pytest.param(
            lambda m: m["gate_c_scope"].__setitem__("tokenizer_counting", True),
            "gate_c_scope",
            id="tokenizer_counting_true",
        ),
        pytest.param(
            lambda m: m["accepted_document_coverage"].__setitem__("accepted_file_count", 999),
            "accepted_file_count",
            id="forged_accepted_file_count",
        ),
        pytest.param(
            lambda m: m["accepted_document_coverage"].__setitem__(
                "accepted_file_histogram", {"bogus.parquet": 1}
            ),
            "accepted_file_histogram",
            id="forged_accepted_file_histogram",
        ),
        pytest.param(
            lambda m: m["transport_read_coverage"].__setitem__("files_opened", 99),
            "files_opened",
            id="files_opened_count_mismatch",
        ),
        pytest.param(
            lambda m: m["transport_read_coverage"]["row_groups_opened_list"].append(
                "zzz.parquet#999"
            ),
            "row_groups_opened_list|row_groups_opened",
            id="unplanned_row_group",
        ),
        pytest.param(
            lambda m: m["resource_accounting"].__setitem__("wire_bytes_measured", True),
            "wire_bytes_measured",
            id="wire_bytes_measured_true",
        ),
        pytest.param(
            lambda m: m["resource_accounting"].__setitem__("gpu_api_called", True),
            "gpu_api_called",
            id="gpu_api_called_true",
        ),
        pytest.param(
            lambda m: m["persisted_inputs_before_first_body_read"].__setitem__(
                gp.SELECTION_PLAN_NAME, "4" * 64
            ),
            "persisted_inputs_before_first_body_read",
            id="forged_persisted_digest",
        ),
        pytest.param(
            lambda m: m["accounting"].__setitem__("accepted_text_bytes", 12345),
            "accepted_text_bytes",
            id="forged_accepted_text_bytes",
        ),
        pytest.param(
            lambda m: m["coverage"].__setitem__("files_touched", 42),
            "files_touched",
            id="deprecated_mirror_mismatch",
        ),
        pytest.param(
            lambda m: m["source"].__setitem__("license", "proprietary"),
            "license",
            id="forged_source_license",
        ),
        pytest.param(
            lambda m: m.__setitem__("promotion_eligible", True),
            "promotion_eligible",
            id="promotion_eligible_true",
        ),
        pytest.param(
            lambda m: m["hard_stops"].__setitem__("extra_contradictory_flag", True),
            "key set is wrong",
            id="extra_hard_stop_key",
        ),
    ],
)
def test_forged_release_manifest_is_rejected(store_root, work_root, mutator, match):
    output, _ = _built_release(store_root, work_root)
    forge_release_manifest(output, mutator)
    with pytest.raises(gp.GateCParquetError, match=match):
        gp.verify_release_v2(output)


def test_legal_release_manifest_bytes_are_unchanged_by_the_validator(store_root, work_root):
    """The validator must not rewrite or reshape a legal release."""
    output, result = _built_release(store_root, work_root)
    before = (output / gp.MANIFEST_NAME).read_bytes()
    gp.verify_release_v2(output)
    assert (output / gp.MANIFEST_NAME).read_bytes() == before
    assert result["documents_sha256"] == gp.verify_release_v2(output)["documents_sha256"]


# --------------------------------------------------------------------------------------
# Malformed structures are normalised to GateCParquetError, never a bare KeyError/TypeError
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(lambda p: [], "must be a JSON object", id="plan_top_level_list"),
        pytest.param(
            lambda p: {k: v for k, v in p.items() if k != "units"},
            "missing the required field 'units'",
            id="plan_missing_units",
        ),
        pytest.param(lambda p: {**p, "units": None}, "must be a list", id="plan_units_null"),
        pytest.param(lambda p: {**p, "units": []}, "must not be empty", id="plan_units_empty"),
        pytest.param(
            lambda p: {**p, "units": [{k: v for k, v in p["units"][0].items() if k != "order"}]},
            "malformed|missing",
            id="unit_missing_order",
        ),
        pytest.param(
            lambda p: {**p, "units": ["not-a-mapping"]},
            "must be a JSON object",
            id="unit_not_a_mapping",
        ),
        pytest.param(
            lambda p: {**p, "requested_units": "many"},
            "requested_units",
            id="requested_units_wrong_type",
        ),
        pytest.param(
            lambda p: {**p, "resolved_file_topology": []},
            "must be a JSON object",
            id="topology_not_a_mapping",
        ),
    ],
)
def test_malformed_plan_raises_gatec_error(store_root, mutator, match):
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=6, rows_per_unit=10)
    forged = mutator(copy.deepcopy(plan))
    if isinstance(forged, dict):
        # Re-checksum so the failure isolates the STRUCTURAL check, not a stale checksum.
        forged.pop("selection_plan_sha256", None)
        forged["selection_plan_sha256"] = gp.selection_plan_sha256(forged)
    with pytest.raises(gp.GateCParquetError, match=match):
        gp.validate_selection_plan(forged, manifest)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(lambda m: [], "must be a JSON object", id="manifest_top_level_list"),
        pytest.param(
            lambda m: {k: v for k, v in m.items() if k != "files"},
            "missing the required field 'files'",
            id="manifest_missing_files",
        ),
        pytest.param(
            lambda m: {**m, "files": "nope"}, "must be a list", id="manifest_files_not_a_list"
        ),
        pytest.param(
            lambda m: {**m, "files": ["nope"]},
            "must be a JSON object",
            id="manifest_file_entry_not_a_mapping",
        ),
    ],
)
def test_malformed_transport_manifest_raises_gatec_error(store_root, mutator, match):
    store = LocalStore(store_root)
    manifest = make_manifest(store)
    plan = gp.build_selection_plan(manifest, seed=20260817, units=6, rows_per_unit=10)
    forged = mutator(copy.deepcopy(manifest))
    forged_plan = copy.deepcopy(plan)
    if isinstance(forged, dict):
        # Re-bind and re-checksum both sides so the failure isolates the STRUCTURAL check.
        forged.pop("manifest_sha256", None)
        forged["manifest_sha256"] = gp.transport_manifest_sha256(forged)
        forged_plan["transport_manifest_sha256"] = forged["manifest_sha256"]
        forged_plan.pop("selection_plan_sha256", None)
        forged_plan["selection_plan_sha256"] = gp.selection_plan_sha256(forged_plan)
    with pytest.raises(gp.GateCParquetError, match=match):
        gp.validate_selection_plan(forged_plan, forged)


def test_malformed_release_manifest_raises_gatec_error(store_root, work_root):
    output, _ = _built_release(store_root, work_root)

    def rewrite(payload: bytes) -> None:
        (output / gp.MANIFEST_NAME).write_bytes(payload)
        lines = []
        for entry in (output / gp.CHECKSUMS_NAME).read_text().splitlines():
            if entry:
                _, name = entry.split("  ", 1)
                lines.append(
                    f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
                )
        (output / gp.CHECKSUMS_NAME).write_text("".join(lines))

    rewrite(b"[]\n")
    with pytest.raises(gp.GateCParquetError, match="must be a JSON object"):
        gp.verify_release_v2(output)

    rewrite(b"{not json\n")
    with pytest.raises(gp.GateCParquetError, match="not strict UTF-8 JSON"):
        gp.verify_release_v2(output)


def test_release_manifest_missing_block_raises_gatec_error(store_root, work_root):
    output, _ = _built_release(store_root, work_root)
    forge_release_manifest(output, lambda m: m.pop("sampler"))
    with pytest.raises(gp.GateCParquetError, match="missing the required field 'sampler'"):
        gp.verify_release_v2(output)


def test_release_manifest_wrong_block_type_raises_gatec_error(store_root, work_root):
    output, _ = _built_release(store_root, work_root)
    forge_release_manifest(output, lambda m: m.__setitem__("accepted_document_coverage", []))
    with pytest.raises(gp.GateCParquetError, match="must be a JSON object"):
        gp.verify_release_v2(output)


def test_missing_release_file_raises_gatec_error(store_root, work_root):
    output, _ = _built_release(store_root, work_root)
    (output / gp.SELECTION_PLAN_NAME).unlink()
    with pytest.raises(gp.GateCParquetError, match="missing selection_plan.json"):
        gp.verify_release_v2(output)


def test_cli_verify_returns_controlled_rc_for_a_malformed_release(store_root, work_root, capsys):
    """The CLI must surface a controlled rc=2, not a traceback."""
    output, _ = _built_release(store_root, work_root)
    forge_release_manifest(
        output, lambda m: m["sampler"].__setitem__("representative_sampler", True)
    )
    rc = gp.main(["verify", "--output-dir", str(output)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "representative_sampler" in captured.err
    assert "Traceback" not in captured.err and "Traceback" not in captured.out


# --------------------------------------------------------------------------------------
# Transport-manifest contract: identity forgeries must fail even with every checksum re-synced
# --------------------------------------------------------------------------------------


def forge_transport_manifest(output: Path, mutator) -> None:
    """Mutate the published transport manifest and re-sync EVERY downstream binding.

    After this helper runs, the transport self-checksum, the selection-plan binding and checksum,
    the release-manifest bindings, the persisted-input file-byte digests and MANIFEST.sha256 are
    all internally consistent, so a rejection can only come from the semantic contract.
    """
    dump = {"indent": 2, "sort_keys": True}

    transport = json.loads((output / gp.TRANSPORT_MANIFEST_NAME).read_text())
    mutator(transport)
    transport.pop("manifest_sha256", None)
    transport["manifest_sha256"] = gp.transport_manifest_sha256(transport)
    (output / gp.TRANSPORT_MANIFEST_NAME).write_bytes(
        json.dumps(transport, **dump).encode() + b"\n"
    )

    plan = json.loads((output / gp.SELECTION_PLAN_NAME).read_text())
    plan["transport_manifest_sha256"] = transport["manifest_sha256"]
    for field in (
        "dataset",
        "config",
        "split",
        "data_revision",
        "parquet_revision",
        "production_row_group_manifest_complete",
    ):
        if field in transport:
            plan[field] = transport[field]
    plan.pop("selection_plan_sha256", None)
    plan["selection_plan_sha256"] = gp.selection_plan_sha256(plan)
    (output / gp.SELECTION_PLAN_NAME).write_bytes(json.dumps(plan, **dump).encode() + b"\n")

    manifest = json.loads((output / gp.MANIFEST_NAME).read_text())
    manifest["transport_manifest_sha256"] = transport["manifest_sha256"]
    manifest["selection_plan_sha256"] = plan["selection_plan_sha256"]
    if "production_row_group_manifest_complete" in transport:
        manifest["production_row_group_manifest_complete"] = transport[
            "production_row_group_manifest_complete"
        ]
    manifest["persisted_inputs_before_first_body_read"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in (gp.TRANSPORT_MANIFEST_NAME, gp.SELECTION_PLAN_NAME)
    }
    (output / gp.MANIFEST_NAME).write_bytes(json.dumps(manifest, **dump).encode() + b"\n")

    lines = []
    for entry in (output / gp.CHECKSUMS_NAME).read_text().splitlines():
        if entry:
            _, name = entry.split("  ", 1)
            lines.append(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n")
    (output / gp.CHECKSUMS_NAME).write_text("".join(lines))


def test_transport_forgery_helper_keeps_every_checksum_consistent(store_root, work_root):
    """Guard on the guard: a no-op transport mutation must leave the release verifiable."""
    output, _ = _built_release(store_root, work_root)
    forge_transport_manifest(output, lambda t: t.__setitem__("split", t["split"]))
    for entry in (output / gp.CHECKSUMS_NAME).read_text().splitlines():
        digest, name = entry.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert gp.verify_release_v2(output)["release_manifest_validated"] is True


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(
            lambda t: t.__setitem__("parquet_repo_id", "evil/example"),
            "parquet_repo_id",
            id="forged_parquet_repo_id",
        ),
        pytest.param(
            lambda t: t.__setitem__("transport", "other_transport"),
            "transport",
            id="forged_transport",
        ),
        pytest.param(
            lambda t: t.__setitem__("path_prefix", "evil/"), "path_prefix", id="forged_path_prefix"
        ),
        pytest.param(
            lambda t: t.__setitem__("tool_schema_version", "stale-tool-schema"),
            "tool_schema_version",
            id="stale_tool_schema_version",
        ),
        pytest.param(
            lambda t: t.__setitem__("spec_version", "stale-spec"),
            "spec_version",
            id="stale_spec_version",
        ),
        pytest.param(
            lambda t: t.__setitem__("parquet_is_export_branch", not t["parquet_is_export_branch"]),
            "parquet_is_export_branch",
            id="negated_export_branch_flag",
        ),
        pytest.param(
            lambda t: t.__setitem__("file_manifest_complete", False),
            "file_manifest_complete",
            id="file_manifest_incomplete",
        ),
        pytest.param(
            lambda t: t.__setitem__("data_revision", "0" * 40),
            "data_revision",
            id="forged_data_revision",
        ),
        pytest.param(
            lambda t: t.__setitem__("parquet_revision", "1" * 40),
            "parquet_revision",
            id="forged_parquet_revision",
        ),
        pytest.param(
            lambda t: t.__setitem__("source_key", "pes2o"),
            "source_key|dataset",
            id="forged_source_key",
        ),
    ],
)
def test_forged_transport_identity_is_rejected(store_root, work_root, mutator, match):
    output, _ = _built_release(store_root, work_root)
    forge_transport_manifest(output, mutator)
    with pytest.raises(gp.GateCParquetError, match=match):
        gp.verify_release_v2(output)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        pytest.param(
            lambda t: t.pop("production_row_group_manifest_complete"),
            "missing the required field 'production_row_group_manifest_complete'",
            id="missing_completeness_field",
        ),
        pytest.param(
            lambda t: t.__setitem__("production_row_group_manifest_complete", 1),
            "must be a bool",
            id="completeness_is_int_one",
        ),
        pytest.param(
            lambda t: t.__setitem__("production_row_group_manifest_complete", "true"),
            "must be a bool",
            id="completeness_is_string",
        ),
        pytest.param(
            lambda t: t.__setitem__("production_row_group_manifest_complete", False),
            "contradicts",
            id="complete_policy_but_flag_false",
        ),
        pytest.param(
            lambda t: t.__setitem__("footer_policy", "selected_files_only"),
            "contradicts",
            id="selected_files_only_policy_but_flag_true",
        ),
        pytest.param(
            lambda t: t.__setitem__("footer_policy", "sometimes"),
            "footer_policy must be one of",
            id="unknown_footer_policy",
        ),
        pytest.param(
            lambda t: t.pop("files"), "missing the required field 'files'", id="missing_files"
        ),
        pytest.param(lambda t: t.__setitem__("files", []), "must not be empty", id="empty_files"),
        pytest.param(
            lambda t: t["files"].append(dict(t["files"][0])),
            "duplicate path",
            id="duplicate_file_path",
        ),
    ],
)
def test_transport_completeness_and_shape_failures(store_root, work_root, mutator, match):
    output, _ = _built_release(store_root, work_root)
    forge_transport_manifest(output, mutator)
    with pytest.raises(gp.GateCParquetError, match=match):
        gp.verify_release_v2(output)


def test_transport_validator_is_the_single_authority():
    """verify_transport_manifest must delegate, so build/plan/release share one contract."""
    import inspect

    source = inspect.getsource(gp.verify_transport_manifest)
    assert "validate_transport_manifest_v2" in source


def test_forged_transport_is_rejected_before_any_body_or_footer_read(store_root, tmp_path):
    """A checksum-consistent transport forgery must be refused with zero object opens."""
    store = CountingStore(store_root)
    manifest = make_manifest(store)
    forged = json.loads(json.dumps(manifest))
    forged["parquet_repo_id"] = "evil/example"
    forged.pop("manifest_sha256")
    forged["manifest_sha256"] = gp.transport_manifest_sha256(forged)
    store.open_count = 0
    with pytest.raises(gp.GateCParquetError, match="parquet_repo_id"):
        gp.build_selection_plan(forged, seed=20260817, units=4, rows_per_unit=10, store=store)
    assert store.open_count == 0


def test_legal_selected_files_only_plan_still_reads_footers(store_root):
    """The zero-open rule must not break the legal footer reads a partial inventory needs."""
    store = CountingStore(store_root)
    manifest = gp.build_transport_manifest(
        "finewiki_en", store, footer_policy="selected_files_only"
    )
    store.open_count = 0
    plan = gp.build_selection_plan(manifest, seed=20260817, units=2, rows_per_unit=10, store=store)
    assert store.open_count > 0
    assert plan["file_weighting"] == "hub_file_byte_size_proxy"


def test_legal_transport_manifest_bytes_are_unchanged_by_validation(store_root, work_root):
    output, _ = _built_release(store_root, work_root)
    before = (output / gp.TRANSPORT_MANIFEST_NAME).read_bytes()
    gp.verify_release_v2(output)
    assert (output / gp.TRANSPORT_MANIFEST_NAME).read_bytes() == before


# --------------------------------------------------------------------------------------
# documents.jsonl strict UTF-8 boundary
# --------------------------------------------------------------------------------------


def _inject_invalid_utf8(output: Path) -> None:
    raw = bytearray((output / gp.DOCUMENTS_NAME).read_bytes())
    position = raw.find(b"Observation")
    assert position > 0
    raw[position + 3] = 0xFF
    (output / gp.DOCUMENTS_NAME).write_bytes(bytes(raw))
    manifest = json.loads((output / gp.MANIFEST_NAME).read_text())
    manifest["documents_sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
    manifest["documents_bytes"] = len(raw)
    (output / gp.MANIFEST_NAME).write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    lines = []
    for entry in (output / gp.CHECKSUMS_NAME).read_text().splitlines():
        if entry:
            _, name = entry.split("  ", 1)
            lines.append(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n")
    (output / gp.CHECKSUMS_NAME).write_text("".join(lines))


def test_invalid_utf8_documents_raise_gatec_error(store_root, work_root):
    output, _ = _built_release(store_root, work_root)
    _inject_invalid_utf8(output)
    with pytest.raises(gp.GateCParquetError, match="documents.jsonl|strict UTF-8"):
        gp.verify_release_v2(output)
    try:
        gp.verify_release_v2(output)
    except gp.GateCParquetError as exc:
        assert not isinstance(exc, UnicodeDecodeError)


def test_invalid_utf8_documents_cli_returns_controlled_rc(store_root, work_root, capsys):
    output, _ = _built_release(store_root, work_root)
    _inject_invalid_utf8(output)
    rc = gp.main(["verify", "--output-dir", str(output)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "documents.jsonl" in captured.err or "strict UTF-8" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
