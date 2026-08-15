from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse

import pytest

import pretrain.collect_python_p1 as p1
from pretrain.collect_python_p1 import (
    MAX_METADATA_BYTES,
    METADATA_ROWS,
    P1_COLLECTION_POLICY,
    P1_RESOURCE_POLICY,
    SELECTED_BLOBS,
    BlobFetchIssue,
    CollectionError,
    P1Config,
    collect_python_p1,
    select_p1_rows,
)
from pretrain.inspect_python_sources import HttpJsonResponse
from pretrain.python_source_adapters import AdapterError, get_adapter

REVISION = "0123456789abcdef0123456789abcdef01234567"


class FakeRowsServer:
    def __init__(self, rows: list[dict], *, revision: str = REVISION) -> None:
        self.rows = rows
        self.revision = revision
        self.calls: list[str] = []
        self.features = [
            {"name": name, "type": {"dtype": "string", "_type": "Value"}} for name in rows[0]
        ]

    def __call__(self, url: str, timeout_seconds: float) -> HttpJsonResponse:
        assert timeout_seconds == 30.0
        parsed = urlparse(url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        self.calls.append(endpoint)
        if endpoint == "size":
            payload = {
                "size": {
                    "splits": [
                        {
                            "config": "python-edu",
                            "split": "train",
                            "num_rows": len(self.rows),
                        }
                    ]
                }
            }
        elif endpoint == "rows":
            offset = int(query["offset"][0])
            length = int(query["length"][0])
            payload = {
                "features": self.features,
                "rows": [
                    {
                        "row_idx": index,
                        "row": dict(self.rows[index]),
                        "truncated_cells": [],
                    }
                    for index in range(offset, offset + length)
                ],
            }
        else:
            raise AssertionError(f"unexpected endpoint {endpoint}")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return HttpJsonResponse(
            payload=payload,
            headers={"x-revision": self.revision},
            body_sha256=hashlib.sha256(body).hexdigest(),
            http_status=200,
            body_bytes=len(body),
            request_latency_seconds=0.001,
        )


def _corpus(size: int = 620) -> tuple[list[dict], dict[str, bytes]]:
    rows: list[dict] = []
    blobs: dict[str, bytes] = {}
    for index in range(size):
        blob_id = f"blob-{index:04d}"
        if index % 19 == 0:
            raw = b"\xffnot utf8\n"
        elif index % 17 == 0:
            raw = b"def broken(:\n    pass\n"
        elif index % 13 == 0:
            raw = b"# auto-generated; do not edit\ndef generated():\n    return 1\n"
        else:
            raw = (
                f'"""module {index}"""\n# comment\ndef f_{index}(x):\n    return x + {index}\n'
            ).encode()
        path = f"vendor/pkg/file_{index}.py" if index % 11 == 0 else f"src/file_{index}.py"
        rows.append({
            "blob_id": blob_id,
            "repo_name": "one-repository",
            "path": path,
            "length_bytes": len(raw),
            "score": -100.0 if index % 2 else 0.0,
            "int_score": -7,
        })
        blobs[blob_id] = raw
    return rows, blobs


def _config(tmp_path: Path, **overrides) -> P1Config:
    policy_path = overrides.pop("policy_path", tmp_path / "policy.json")
    if not policy_path.exists():
        policy = {
            "schema_version": 1,
            "kind": "petitgpt_python_p1_matched_source_policy",
            "status": "FROZEN_BEFORE_COLLECTION",
            "decision_scope": "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL",
            "analysis": {},
            "privacy": {
                "cache_and_reports_git_policy": "DO_NOT_COMMIT",
                "report_source_characters_exposed": 0,
            },
            "collection": P1_COLLECTION_POLICY,
            "resource_budget_per_arm": P1_RESOURCE_POLICY,
            "arms": [
                {
                    "name": "smollm_python_edu_primary",
                    "dataset": "HuggingFaceTB/smollm-corpus",
                    "dataset_config": "python-edu",
                    "split": "train",
                    "expected_revision": REVISION,
                },
                {
                    "name": "stack_edu_python_comparator",
                    "dataset": "HuggingFaceTB/stack-edu",
                    "dataset_config": "Python",
                    "split": "train",
                    "expected_revision": "f" * 40,
                },
            ],
        }
        policy_path.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
    expected_policy_sha256 = overrides.pop(
        "expected_policy_sha256", hashlib.sha256(policy_path.read_bytes()).hexdigest()
    )
    values = {
        "expected_revision": REVISION,
        "adapter": "smollm_python_edu",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "dataset_config": "python-edu",
        "split": "train",
        "expected_policy_sha256": expected_policy_sha256,
        "output_dir": tmp_path / "evidence",
        "cache_dir": tmp_path / "private-cache",
        "policy_path": policy_path,
        "workers": 4,
        "enforce_ignored_paths": False,
    }
    values.update(overrides)
    return P1Config(**values)


def _injected_cache_origin(config: P1Config) -> dict:
    def unused_transport(_url: str, _timeout: float) -> HttpJsonResponse:
        raise AssertionError("cache-origin helper transport must not run")

    def unused_fetcher(_blob_id: str) -> bytes:
        raise AssertionError("cache-origin helper fetcher must not run")

    backend = p1._backend_provenance(
        config,
        transport=unused_transport,
        blob_fetcher=unused_fetcher,
    )
    return p1.cache_origin_contract(backend)


def _read_addressed_json(path: Path) -> dict:
    digest = path.name.split(".sha256-", 1)[1].removesuffix(".json")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    return json.loads(data)


def _write_addressed_json(directory: Path, stem: str, value: dict) -> Path:
    data = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True).encode()
        + b"\n"
    )
    digest = hashlib.sha256(data).hexdigest()
    path = directory / f"{stem}.sha256-{digest}.json"
    path.write_bytes(data)
    return path


def test_adapters_preserve_available_optionals_and_never_invent_absent_values():
    smollm = get_adapter("smollm_python_edu")
    primary_fields = {
        "blob_id",
        "repo_name",
        "path",
        "length_bytes",
        "score",
        "int_score",
    }
    for missing in sorted(smollm.required_fields):
        with pytest.raises(AdapterError, match="missing required upstream fields"):
            smollm.resolve_schema(primary_fields - {missing})

    field_map = smollm.resolve_schema(primary_fields)
    primary_row = {
        "row_idx": 3,
        "blob_id": "abc",
        "repo_name": "org/project",
        "path": "src/module.py",
        "length_bytes": 91,
        "score": 3.75,
        "int_score": 3,
    }
    normalized = smollm.normalize(primary_row, field_map=field_map)
    assert normalized == primary_row
    for absent in ("license", "language", "src_encoding", "detected_licenses", "license_type"):
        assert absent not in normalized

    stack = get_adapter("stack_edu_python")
    stack_fields = {
        "blob_id",
        "language",
        "repo_name",
        "path",
        "src_encoding",
        "length_bytes",
        "score",
        "int_score",
        "detected_licenses",
        "license_type",
    }
    for missing in sorted(stack.required_fields):
        with pytest.raises(AdapterError, match="missing required upstream fields"):
            stack.resolve_schema(stack_fields - {missing})

    stack_map = stack.resolve_schema(stack_fields)
    stack_source_row = {
        "row_idx": 4,
        "blob_id": "def",
        "language": "Python",
        "repo_name": "org/stack-project",
        "path": "package/module.py",
        "src_encoding": "utf-8",
        "length_bytes": 113,
        "score": 4.5,
        "int_score": 4,
        "detected_licenses": ["MIT", "Apache-2.0"],
        "license_type": "permissive",
    }
    stack_row = stack.normalize(stack_source_row, field_map=stack_map)
    assert stack_row == stack_source_row
    assert stack_row["detected_licenses"] is not stack_source_row["detected_licenses"]
    assert "license" not in stack_row

    with pytest.raises(AdapterError, match="invalid detected_licenses"):
        stack.normalize(
            {
                **stack_source_row,
                "row_idx": 5,
                "blob_id": "ghi",
                "detected_licenses": "MIT",
            },
            field_map=stack_map,
        )

    with pytest.raises(AdapterError, match="ambiguous"):
        stack.resolve_schema(stack_fields | {"content_id"})


def test_source_errors_expose_only_categories_and_numeric_positions():
    outcome, decode_quality = p1._source_quality(
        "private.py",
        b"\xffDO_NOT_EXPOSE_SOURCE_TEXT",
    )
    assert outcome == "decode_failed"
    assert decode_quality == {
        "decode_error": {
            "category": "UnicodeDecodeError",
            "start": 0,
            "end": 1,
        },
        "raw_bytes": 26,
    }

    sensitive = "∆ = 1\nDO_NOT_EXPOSE_SYNTAX_SOURCE\n".encode()
    outcome, syntax_quality = p1._source_quality("private.py", sensitive)
    assert outcome == "decoded"
    assert syntax_quality["ast_parse_ok"] is False
    assert syntax_quality["ast_error"]["category"] == "SyntaxError"
    serialized = json.dumps(syntax_quality, ensure_ascii=False)
    assert "∆" not in serialized
    assert "DO_NOT_EXPOSE_SYNTAX_SOURCE" not in serialized


def test_syntax_error_positions_accept_only_python_minus_one_offset_sentinel():
    evidence = {
        "category": "IndentationError",
        "line": 2,
        "offset": 4,
        "end_line": 2,
        "end_offset": 7,
    }
    p1._validate_source_error_evidence(evidence, syntax=True)

    for field in ("offset", "end_offset"):
        sentinel = {**evidence, field: -1}
        p1._validate_source_error_evidence(sentinel, syntax=True)

    p1._validate_source_error_evidence(
        {
            "category": "ValueError",
            "line": None,
            "offset": None,
            "end_line": None,
            "end_offset": None,
        },
        syntax=True,
    )

    for field in ("line", "end_line"):
        for invalid in (0, -1):
            with pytest.raises(CollectionError, match="below 1"):
                p1._validate_source_error_evidence(
                    {**evidence, field: invalid},
                    syntax=True,
                )

    for field in ("offset", "end_offset"):
        with pytest.raises(CollectionError, match="must be -1 or a positive integer"):
            p1._validate_source_error_evidence(
                {**evidence, field: 0},
                syntax=True,
            )
        with pytest.raises(CollectionError, match="below -1"):
            p1._validate_source_error_evidence(
                {**evidence, field: -2},
                syntax=True,
            )
        with pytest.raises(CollectionError, match="must be an integer"):
            p1._validate_source_error_evidence(
                {**evidence, field: -1.0},
                syntax=True,
            )


def test_latency_evidence_is_integer_and_accumulation_order_independent():
    seconds = [1.0, *([1e-9] * 10_000)]
    forward_float = 0.0
    for value in seconds:
        forward_float += value
    reverse_float = 0.0
    for value in reversed(seconds):
        reverse_float += value
    builtin_float = sum(seconds)
    assert len({builtin_float, forward_float, reverse_float}) > 1

    nanoseconds = [
        p1._latency_nanoseconds(value, label=f"regression {ordinal}")
        for ordinal, value in enumerate(seconds)
    ]
    expected = 1_000_010_000
    assert (
        p1._sum_latency_nanoseconds(
            nanoseconds,
            label="online accounting",
        )
        == expected
    )
    assert (
        p1._sum_latency_nanoseconds(
            list(reversed(nanoseconds)),
            label="replay accounting",
        )
        == expected
    )
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in nanoseconds)


def test_selection_uses_only_max_length_distinct_id_and_stable_rank():
    rows = [
        {
            "row_idx": index,
            "blob_id": f"blob-{index}",
            "length_bytes": 0,
            "repo_name": "same-repository",
            "score": -999,
            "int_score": -999,
        }
        for index in range(350)
    ]
    rows.append({
        "row_idx": 351,
        "blob_id": "too-large",
        "length_bytes": MAX_METADATA_BYTES + 1,
        "score": 999,
    })
    rows.append({**rows[0], "row_idx": 352})
    selected_a, stats_a = select_p1_rows(rows)
    selected_b, stats_b = select_p1_rows(list(reversed(rows)))
    assert [row["blob_id"] for row in selected_a] == [row["blob_id"] for row in selected_b]
    assert len(selected_a) == SELECTED_BLOBS
    assert len({row["blob_id"] for row in selected_a}) == SELECTED_BLOBS
    assert {row["repo_name"] for row in selected_a} == {"same-repository"}
    assert all(row["score"] == -999 for row in selected_a)
    assert stats_a == stats_b
    assert stats_a["metadata_length_above_100000"] == 1
    assert stats_a["duplicate_blob_id"] == 1


def test_online_collection_is_exact_content_addressed_and_offline_replayable(
    tmp_path: Path,
):
    rows, blobs = _corpus()
    server = FakeRowsServer(rows)
    outputs = collect_python_p1(
        _config(tmp_path),
        transport=server,
        blob_fetcher=blobs.__getitem__,
        sleeper=lambda _seconds: None,
    )

    assert server.calls.count("size") == 1
    assert server.calls.count("rows") == 50
    assert len(server.calls) == 51
    manifest_path = Path(outputs["manifest"])
    report_path = Path(outputs["report"])
    manifest = _read_addressed_json(manifest_path)
    report = _read_addressed_json(report_path)
    policy_sha256 = hashlib.sha256((tmp_path / "policy.json").read_bytes()).hexdigest()
    assert manifest["policy_binding"]["sha256"] == policy_sha256
    assert report["policy_binding"]["sha256"] == policy_sha256
    assert manifest["policy_binding"]["expected_sha256"] == policy_sha256
    assert manifest["backend_provenance"]["production"] is False
    assert manifest["backend_provenance"]["test_only"] is True
    assert manifest["contract"]["timing_evidence"] == {
        "unit": "integer_nanoseconds",
        "quantization": "binary64_exact_ratio_round_half_even_per_measurement_v1",
        "accumulation": "exact_integer_sum",
    }
    assert manifest["backend_provenance"]["mode"] == "test_only_injected"
    assert manifest["backend_provenance"]["hf"]["transport"] == "injected_callable"
    assert manifest["backend_provenance"]["swh"]["fetcher"] == "injected_callable"
    assert manifest["backend_provenance"]["swh"]["bucket"] is None
    assert manifest["cache_origin_contract"] == p1.cache_origin_contract(
        manifest["backend_provenance"]
    )
    assert manifest["cache_origin_contract"]["production"] is False
    assert report["backend_provenance"] == manifest["backend_provenance"]
    assert report["cache_origin_contract"] == manifest["cache_origin_contract"]
    assert report["backend_accounting"] == manifest["backend_accounting"]
    assert manifest["backend_accounting"]["hf"]["logical_calls"] == 51
    assert manifest["backend_accounting"]["hf"]["attempts"] == 51
    assert manifest["backend_accounting"]["hf"]["retries"] == 0
    assert manifest["backend_accounting"]["hf"]["total_latency_nanoseconds"] == 51_000_000
    assert all(
        isinstance(attempt["request_latency_nanoseconds"], int)
        for attempt in manifest["hf_evidence"]["attempts"]
    )
    assert all(
        "request_latency_seconds" not in attempt
        and "backoff_before_next_attempt_seconds" not in attempt
        and (
            attempt["backoff_before_next_attempt_nanoseconds"] is None
            or isinstance(attempt["backoff_before_next_attempt_nanoseconds"], int)
        )
        for attempt in manifest["hf_evidence"]["attempts"]
    )
    assert manifest["backend_accounting"]["swh"]["attempts"] == 300
    assert manifest["backend_accounting"]["swh"]["retries"] == 0
    assert manifest["backend_accounting"]["swh"]["cache_origin_verified"] is True
    assert isinstance(
        manifest["backend_accounting"]["swh"]["total_latency_nanoseconds"],
        int,
    )
    assert isinstance(report["resources"]["elapsed_nanoseconds"], int)
    assert all(
        isinstance(entry["fetch_latency_nanoseconds"], int)
        and all(
            isinstance(latency, int) for latency in entry["fetch_attempt_latencies_nanoseconds"]
        )
        and "fetch_latency_seconds" not in entry
        and "fetch_attempt_latencies_seconds" not in entry
        for entry in manifest["selected_blobs"]
    )

    assert len(manifest["metadata_rows"]) == METADATA_ROWS
    assert len(manifest["sampling"]["windows"]) == 50
    assert {window["length"] for window in manifest["sampling"]["windows"]} == {10}
    assert len(manifest["selected_blobs"]) == SELECTED_BLOBS
    assert len({entry["metadata"]["blob_id"] for entry in manifest["selected_blobs"]}) == 300
    assert all(
        entry["cache_origin_verified"] is False
        for entry in manifest["selected_blobs"]
        if entry["fetch_outcome"] == "success"
    )
    first_success = next(
        entry for entry in manifest["selected_blobs"] if entry["fetch_outcome"] == "success"
    )
    first_index, _ = p1._cache_paths(
        tmp_path / "private-cache",
        first_success["metadata"]["blob_id"],
    )
    index = json.loads(first_index.read_text(encoding="utf-8"))
    assert index["schema_version"] == p1.CACHE_INDEX_SCHEMA_VERSION == 2
    assert index["origin"] == manifest["cache_origin_contract"]
    assert report["content"]["fetch_success"] == 300
    assert report["content"]["strict_utf8_failed"] > 0
    assert report["content"]["ast_parse_failed"] > 0
    assert report["content"]["vendor_path_heuristic"] > 0
    assert report["content"]["generated_file_heuristic"] > 0
    serialized = manifest_path.read_bytes() + report_path.read_bytes()
    assert b"def f_" not in serialized
    assert b"auto-generated; do not edit" not in serialized
    assert not [path for path in (tmp_path / "private-cache").iterdir() if path.is_dir()]

    replay_config = _config(
        tmp_path,
        output_dir=tmp_path / "replay-evidence",
        replay_manifest=manifest_path,
    )

    def forbidden_transport(_url: str, _timeout: float) -> HttpJsonResponse:
        raise AssertionError("offline replay attempted network access")

    replay = collect_python_p1(
        replay_config,
        transport=forbidden_transport,
        blob_fetcher=lambda _blob: (_ for _ in ()).throw(
            AssertionError("offline replay attempted blob fetch")
        ),
    )
    replay_report = _read_addressed_json(Path(replay["report"]))
    assert replay_report["network_access"] is False
    assert replay_report["verified_blobs"] == 300
    assert replay_report["cache_origin_verified"] is True
    assert replay_report["cache_origin_contract"] == manifest["cache_origin_contract"]
    assert isinstance(replay_report["elapsed_nanoseconds"], int)

    mutation_dir = tmp_path / "mutated-manifests"
    mutation_dir.mkdir()
    mutations: list[tuple[str, dict, str]] = []

    unsafe_path = copy.deepcopy(manifest)
    unsafe_path["selected_blobs"][0]["cache_object"] = "../escape.raw"
    mutations.append(("unsafe-cache-path", unsafe_path, "unsafe or noncanonical cache_object"))

    bad_contract = copy.deepcopy(manifest)
    bad_contract["contract"]["seed"] += 1
    mutations.append(("contract", bad_contract, "contract drifted"))

    bad_input = copy.deepcopy(manifest)
    bad_input["input"]["dataset"] = "unbound/dataset"
    mutations.append(("input", bad_input, "arguments do not match"))

    bad_cache_origin = copy.deepcopy(manifest)
    bad_cache_origin["cache_origin_contract"]["production"] = True
    mutations.append(("cache-origin", bad_cache_origin, "cache origin contract drifted"))

    bad_cache_hit_evidence = copy.deepcopy(manifest)
    bad_cache_hit_evidence["selected_blobs"][0]["cache_origin_verified"] = True
    mutations.append((
        "cache-origin-evidence",
        bad_cache_hit_evidence,
        "cache-origin verification evidence drifted",
    ))

    bad_swh_latency = copy.deepcopy(manifest)
    bad_swh_latency["selected_blobs"][0]["fetch_latency_nanoseconds"] += 1
    mutations.append(("swh-latency", bad_swh_latency, "SWH total latency drifted"))

    bad_hf_latency = copy.deepcopy(manifest)
    bad_hf_latency["hf_evidence"]["responses"][0]["request_latency_nanoseconds"] = 0.5
    mutations.append(("hf-latency", bad_hf_latency, "must be an integer"))

    bad_metadata_count = copy.deepcopy(manifest)
    bad_metadata_count["metadata_rows"].pop()
    mutations.append(("metadata-count", bad_metadata_count, "exactly 500 metadata"))

    bad_window_count = copy.deepcopy(manifest)
    bad_window_count["sampling"]["windows"].pop()
    mutations.append(("window-count", bad_window_count, "deterministic windows drifted"))

    bad_selected_count = copy.deepcopy(manifest)
    bad_selected_count["selected_blobs"].pop()
    mutations.append(("selected-count", bad_selected_count, "exactly 300 selected"))

    bad_rank = copy.deepcopy(manifest)
    bad_rank["selected_blobs"][0]["selection_rank_sha256"] = "0" * 64
    mutations.append(("stable-rank", bad_rank, "stable selection rank drifted"))

    bad_distinct = copy.deepcopy(manifest)
    bad_distinct["selected_blobs"][1]["metadata"] = copy.deepcopy(
        bad_distinct["selected_blobs"][0]["metadata"]
    )
    bad_distinct["selected_blobs"][1]["selection_rank_sha256"] = bad_distinct["selected_blobs"][0][
        "selection_rank_sha256"
    ]
    mutations.append(("distinct-selection", bad_distinct, "stable selection metadata drifted"))

    bad_hf_revision = copy.deepcopy(manifest)
    bad_hf_revision["hf_evidence"]["responses"][0]["x_revision"] = "f" * 40
    mutations.append(("hf-revision", bad_hf_revision, "HF success evidence drifted"))

    for ordinal, (name, mutated_manifest, message) in enumerate(mutations):
        mutated_path = _write_addressed_json(mutation_dir, name, mutated_manifest)
        failed_output = tmp_path / f"mutated-replay-{ordinal}"
        with pytest.raises(CollectionError, match=message):
            collect_python_p1(
                _config(
                    tmp_path,
                    output_dir=failed_output,
                    replay_manifest=mutated_path,
                ),
                transport=forbidden_transport,
                blob_fetcher=lambda _blob: (_ for _ in ()).throw(
                    AssertionError("invalid replay attempted blob fetch")
                ),
            )
        assert not list(failed_output.glob("*.json"))

    first = manifest["selected_blobs"][0]
    index_path, _ = p1._cache_paths(
        replay_config.cache_dir,
        first["metadata"]["blob_id"],
    )
    original_index = index_path.read_bytes()
    tampered_index = json.loads(original_index)
    production_backend = p1._backend_provenance(
        replay_config,
        transport=p1.requests_json_transport,
        blob_fetcher=None,
    )
    tampered_index["origin"] = p1.cache_origin_contract(production_backend)
    index_path.write_text(json.dumps(tampered_index), encoding="utf-8")
    with pytest.raises(CollectionError, match="cache origin mismatch"):
        collect_python_p1(
            _config(
                tmp_path,
                output_dir=tmp_path / "cache-origin-tamper-evidence",
                replay_manifest=manifest_path,
            ),
            transport=forbidden_transport,
        )
    assert not list((tmp_path / "cache-origin-tamper-evidence").glob("*.json"))
    index_path.write_bytes(original_index)

    raw_path = replay_config.cache_dir / first["cache_object"]
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
    with pytest.raises(CollectionError, match="SHA tamper"):
        collect_python_p1(
            _config(
                tmp_path,
                output_dir=tmp_path / "tamper-evidence",
                replay_manifest=manifest_path,
            ),
            transport=forbidden_transport,
        )
    assert not list((tmp_path / "tamper-evidence").glob("*.json"))


def test_revision_or_internal_failure_publishes_no_manifest(tmp_path: Path):
    rows, blobs = _corpus()
    bad_revision = FakeRowsServer(rows, revision="f" * 40)
    with pytest.raises(RuntimeError, match="revision drift"):
        collect_python_p1(
            _config(tmp_path, output_dir=tmp_path / "revision-failure"),
            transport=bad_revision,
            blob_fetcher=blobs.__getitem__,
            sleeper=lambda _seconds: None,
        )
    assert not list((tmp_path / "revision-failure").glob("*.json"))

    server = FakeRowsServer(rows)

    def internal_failure(_blob_id: str) -> bytes:
        raise RuntimeError("synthetic implementation bug")

    with pytest.raises(CollectionError, match="internal blob fetcher failure"):
        collect_python_p1(
            _config(tmp_path, output_dir=tmp_path / "internal-failure"),
            transport=server,
            blob_fetcher=internal_failure,
            sleeper=lambda _seconds: None,
        )
    assert not list((tmp_path / "internal-failure").glob("*.json"))


def test_data_failures_are_counted_without_retry_or_backfill(tmp_path: Path):
    rows, blobs = _corpus()
    server = FakeRowsServer(rows)
    calls: Counter[str] = Counter()

    def mixed_fetch(blob_id: str) -> bytes:
        calls[blob_id] += 1
        index = int(blob_id.rsplit("-", 1)[1])
        if index % 7 == 0:
            raise BlobFetchIssue("not_found", "synthetic NoSuchKey", transient=False)
        if index % 7 == 1:
            return blobs[blob_id] + b"x"
        if index % 7 == 2:
            return b"x" * (MAX_METADATA_BYTES + 1)
        return blobs[blob_id]

    outputs = collect_python_p1(
        _config(tmp_path, output_dir=tmp_path / "data-failure-evidence"),
        transport=server,
        blob_fetcher=mixed_fetch,
        sleeper=lambda _seconds: None,
    )
    report = _read_addressed_json(Path(outputs["report"]))
    manifest = _read_addressed_json(Path(outputs["manifest"]))
    assert sum(calls.values()) == SELECTED_BLOBS
    assert max(calls.values()) == 1
    assert report["content"]["selected_attempts"] == SELECTED_BLOBS
    assert report["content"]["fetch_failed"] > 0
    assert report["content"]["fidelity_length_mismatch"] > 0
    assert report["content"]["failure_counts_by_cause"]["not_found"] > 0
    assert report["content"]["failure_counts_by_cause"]["decompressed_size_above_100000"] > 0
    assert len(manifest["selected_blobs"]) == SELECTED_BLOBS
    assert manifest["selection"]["selected"] == SELECTED_BLOBS


def test_concurrent_distinct_blob_ids_can_publish_identical_raw_content(tmp_path: Path):
    rows, _ = _corpus()
    shared_raw = b'"""same object"""\n# comment\nvalue = 1\n'
    for row in rows:
        row["length_bytes"] = len(shared_raw)
    server = FakeRowsServer(rows)
    barrier = threading.Barrier(4)
    counter_lock = threading.Lock()
    started = 0

    def identical_fetch(_blob_id: str) -> bytes:
        nonlocal started
        with counter_lock:
            started += 1
            ordinal = started
        if ordinal <= 4:
            barrier.wait(timeout=5)
        return shared_raw

    outputs = collect_python_p1(
        _config(tmp_path, output_dir=tmp_path / "duplicate-raw-evidence"),
        transport=server,
        blob_fetcher=identical_fetch,
        sleeper=lambda _seconds: None,
    )
    manifest = _read_addressed_json(Path(outputs["manifest"]))
    assert len({entry["raw_sha256"] for entry in manifest["selected_blobs"]}) == 1
    cache_files = list((tmp_path / "private-cache").iterdir())
    assert len([path for path in cache_files if path.name.startswith("raw-sha256-")]) == 1
    assert len([path for path in cache_files if path.name.startswith("index-")]) == 300


def test_injected_warm_cache_is_rejected_by_production_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rows, blobs = _corpus()
    warm_outputs = collect_python_p1(
        _config(tmp_path, output_dir=tmp_path / "injected-warm-evidence"),
        transport=FakeRowsServer(rows),
        blob_fetcher=blobs.__getitem__,
        sleeper=lambda _seconds: None,
    )
    warm_manifest = _read_addressed_json(Path(warm_outputs["manifest"]))
    assert warm_manifest["cache_origin_contract"]["production"] is False

    production_transport = FakeRowsServer(rows)
    monkeypatch.setattr(p1, "requests_json_transport", production_transport)
    fetch_calls = 0

    def forbidden_production_fetch(_blob_id: str) -> bytes:
        nonlocal fetch_calls
        fetch_calls += 1
        raise AssertionError("origin mismatch must fail before production fetch")

    monkeypatch.setattr(
        p1,
        "make_bounded_swh_fetcher",
        lambda: forbidden_production_fetch,
    )
    production_output = tmp_path / "production-origin-failure"
    with pytest.raises(CollectionError, match="cache origin mismatch"):
        collect_python_p1(
            _config(tmp_path, output_dir=production_output),
            transport=production_transport,
            blob_fetcher=None,
            sleeper=lambda _seconds: None,
        )
    assert fetch_calls == 0
    assert not list(production_output.glob("*.json"))


def test_production_cache_can_be_reused_only_with_verified_production_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rows, blobs = _corpus()
    production_transport = FakeRowsServer(rows)
    monkeypatch.setattr(p1, "requests_json_transport", production_transport)
    fetch_calls: Counter[str] = Counter()
    factory_calls = 0

    def production_fetch(blob_id: str) -> bytes:
        fetch_calls[blob_id] += 1
        return blobs[blob_id]

    def production_fetcher_factory():
        nonlocal factory_calls
        factory_calls += 1
        return production_fetch

    monkeypatch.setattr(p1, "make_bounded_swh_fetcher", production_fetcher_factory)
    warm_outputs = collect_python_p1(
        _config(tmp_path, output_dir=tmp_path / "production-warm-evidence"),
        transport=production_transport,
        blob_fetcher=None,
        sleeper=lambda _seconds: None,
    )
    warm_manifest = _read_addressed_json(Path(warm_outputs["manifest"]))
    assert warm_manifest["backend_provenance"]["production"] is True
    assert warm_manifest["cache_origin_contract"]["production"] is True
    assert sum(fetch_calls.values()) == SELECTED_BLOBS
    assert warm_manifest["backend_accounting"]["swh"]["cache_hits"] == 0
    assert warm_manifest["backend_accounting"]["swh"]["cache_origin_verified"] is True

    hit_outputs = collect_python_p1(
        _config(tmp_path, output_dir=tmp_path / "production-hit-evidence"),
        transport=production_transport,
        blob_fetcher=None,
        sleeper=lambda _seconds: None,
    )
    hit_manifest = _read_addressed_json(Path(hit_outputs["manifest"]))
    hit_report = _read_addressed_json(Path(hit_outputs["report"]))
    assert factory_calls == 2
    assert sum(fetch_calls.values()) == SELECTED_BLOBS
    assert hit_manifest["backend_provenance"]["production"] is True
    assert hit_manifest["cache_origin_contract"] == warm_manifest["cache_origin_contract"]
    assert hit_manifest["backend_accounting"]["swh"]["network_objects"] == 0
    assert hit_manifest["backend_accounting"]["swh"]["attempts"] == 0
    assert hit_manifest["backend_accounting"]["swh"]["cache_hits"] == SELECTED_BLOBS
    assert hit_manifest["backend_accounting"]["swh"]["cache_origin_verified"] is True
    assert hit_report["backend_accounting"] == hit_manifest["backend_accounting"]
    assert all(
        entry["fetch_attempts"] == 0 and entry["cache_origin_verified"] is True
        for entry in hit_manifest["selected_blobs"]
    )


def test_legacy_cache_index_is_never_accepted_for_production(tmp_path: Path):
    config = _config(tmp_path)
    config.cache_dir.mkdir(parents=True)
    config.output_dir.mkdir(parents=True)
    raw = b"value = 1\n"
    injected_origin = _injected_cache_origin(config)
    p1._store_cache_entry(
        config,
        "legacy-blob",
        raw,
        origin=injected_origin,
    )
    index_path, _ = p1._cache_paths(config.cache_dir, "legacy-blob")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = 1
    index.pop("origin")
    index_path.write_text(json.dumps(index), encoding="utf-8")

    production_backend = p1._backend_provenance(
        config,
        transport=p1.requests_json_transport,
        blob_fetcher=None,
    )
    with pytest.raises(CollectionError, match="legacy or unsupported cache index"):
        p1._load_cache_entry(
            config.cache_dir,
            "legacy-blob",
            expected_origin=p1.cache_origin_contract(production_backend),
        )


def test_policy_storage_and_real_cli_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows, blobs = _corpus()
    config = _config(tmp_path, output_dir=tmp_path / "bad-policy-evidence")
    policy = json.loads(config.policy_path.read_text(encoding="utf-8"))
    policy["status"] = "MUTATED_AFTER_FREEZE"
    mutated = tmp_path / "mutated-policy.json"
    mutated.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(CollectionError, match="policy status mismatch"):
        collect_python_p1(
            _config(
                tmp_path,
                output_dir=tmp_path / "bad-policy-evidence",
                policy_path=mutated,
            ),
            transport=FakeRowsServer(rows),
            blob_fetcher=blobs.__getitem__,
        )
    assert not list((tmp_path / "bad-policy-evidence").glob("*.json"))

    with pytest.raises(ValueError, match="exact lowercase 64-hex"):
        collect_python_p1(
            _config(
                tmp_path,
                output_dir=tmp_path / "invalid-policy-sha",
                expected_policy_sha256="A" * 64,
            ),
            transport=FakeRowsServer(rows),
            blob_fetcher=blobs.__getitem__,
        )
    with pytest.raises(CollectionError, match="does not match --expected_policy_sha256"):
        collect_python_p1(
            _config(
                tmp_path,
                output_dir=tmp_path / "wrong-policy-sha",
                expected_policy_sha256="0" * 64,
            ),
            transport=FakeRowsServer(rows),
            blob_fetcher=blobs.__getitem__,
        )
    assert not list((tmp_path / "wrong-policy-sha").glob("*.json"))

    production_backend = p1._backend_provenance(
        config,
        transport=p1.requests_json_transport,
        blob_fetcher=None,
    )
    assert production_backend["production"] is True
    assert production_backend["test_only"] is False
    assert production_backend["mode"] == "production"
    assert production_backend["hf"] == {
        "api_root": p1.DEFAULT_HF_API_ROOT,
        "api_root_is_canonical": True,
        "transport": "requests_json_transport",
        "transport_mode": "production",
    }
    assert production_backend["swh"] == {
        "bucket": "softwareheritage",
        "key_template": "content/{blob_id}",
        "region": "us-west-2",
        "auth": "anonymous_unsigned",
        "fetcher": "boto3_unsigned_s3",
        "fetcher_mode": "production",
    }

    high_usage = {
        "logical_file_bytes": 1,
        "apparent_bytes_including_directories": 1,
        "allocated_bytes": p1.MAX_CACHE_OR_OUTPUT_BYTES + 1,
        "file_count": 1,
        "directory_count": 1,
    }
    monkeypatch.setattr(p1, "_tree_usage", lambda _path: dict(high_usage))
    with pytest.raises(CollectionError, match="allocated_bytes"):
        p1._check_storage_bounds(config)

    command = [sys.executable, str(Path(p1.__file__)), "--help"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert "--policy_json" in completed.stdout
    assert "--expected_policy_sha256" in completed.stdout
    assert "--replay_manifest" in completed.stdout


def test_cache_write_preflight_and_postcheck_roll_back_new_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _config(tmp_path)
    config.cache_dir.mkdir(parents=True)
    config.output_dir.mkdir(parents=True)
    empty_usage = {
        "logical_file_bytes": 0,
        "apparent_bytes_including_directories": 0,
        "allocated_bytes": 0,
        "file_count": 0,
        "directory_count": 1,
    }
    near_limit = {
        **empty_usage,
        "allocated_bytes": p1.MAX_CACHE_OR_OUTPUT_BYTES - 1,
    }
    monkeypatch.setattr(p1, "_allocation_unit", lambda _path: 4096)
    monkeypatch.setattr(
        p1,
        "_check_storage_bounds",
        lambda _config: {"cache": dict(near_limit), "output": dict(empty_usage)},
    )
    with pytest.raises(CollectionError, match="preflight allocated_bytes"):
        p1._store_cache_entry(
            config,
            "preflight-blob",
            b"value = 1\n",
            origin=_injected_cache_origin(config),
        )
    assert not list(config.cache_dir.iterdir())

    calls = 0

    def fail_postcheck(_config: P1Config) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"cache": dict(empty_usage), "output": dict(empty_usage)}
        raise CollectionError("physical quota postcheck exceeded")

    monkeypatch.setattr(p1, "_check_storage_bounds", fail_postcheck)
    with pytest.raises(CollectionError, match="physical quota postcheck"):
        p1._store_cache_entry(
            config,
            "postcheck-blob",
            b"value = 2\n",
            origin=_injected_cache_origin(config),
        )
    assert calls == 2
    assert not list(config.cache_dir.iterdir())
