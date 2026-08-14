from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from pretrain.inspect_python_sources import (
    HttpJsonResponse,
    InspectionConfig,
    deterministic_windows,
    inspect_python_source,
    select_content_rows,
    validate_revision,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


class FakeDatasetServer:
    def __init__(
        self,
        rows: list[dict],
        *,
        revision: str = REVISION,
        row_revision: str | None = None,
        row_outcomes: list[int | BaseException] | None = None,
    ) -> None:
        self.rows = rows
        self.revision = revision
        self.row_revision = row_revision or revision
        self.row_outcomes = list(row_outcomes or [])
        self.calls: list[tuple[str, dict[str, list[str]]]] = []
        self.features = [
            {"name": name, "type": {"dtype": "string", "_type": "Value"}} for name in rows[0]
        ]

    def __call__(self, url: str, timeout_seconds: float) -> HttpJsonResponse:
        assert timeout_seconds > 0
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        self.calls.append((endpoint, query))
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
            revision = self.revision
        elif endpoint == "rows":
            outcome = self.row_outcomes.pop(0) if self.row_outcomes else 200
            if isinstance(outcome, BaseException):
                raise outcome
            offset = int(query["offset"][0])
            length = int(query["length"][0])
            if outcome == 200:
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
                revision = self.row_revision
            else:
                payload = {"error": f"synthetic HTTP {outcome}"}
                revision = "f" * 40
            http_status = outcome
        else:
            raise AssertionError(f"unexpected endpoint: {endpoint}")
        if endpoint == "size":
            http_status = 200
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return HttpJsonResponse(
            payload=payload,
            headers={"X-Revision": revision},
            body_sha256=hashlib.sha256(encoded).hexdigest(),
            http_status=http_status,
            body_bytes=len(encoded),
            request_latency_seconds=0.01,
        )


def _source_rows(blobs: dict[str, bytes]) -> list[dict]:
    paths = [
        "src/healthy.py",
        "src/bad_encoding.py",
        "src/syntax_error.py",
        "vendor/generated.py",
        "src/fetch_failure.py",
        "src/repeated.py",
    ]
    repos = ["repo-a", "repo-b", "repo-c", "repo-d", "repo-e", "repo-f"]
    rows = []
    for index, blob_id in enumerate(blobs):
        rows.append({
            "blob_id": blob_id,
            "repo_name": repos[index],
            "path": paths[index],
            "length_bytes": len(blobs[blob_id]),
            "score": 4.5,
            "int_score": 4,
        })
    return rows


def _config(tmp_path: Path, **overrides) -> InspectionConfig:
    values = {
        "expected_revision": REVISION,
        "report_path": tmp_path / "report.json",
        "examples_path": tmp_path / "report.examples.jsonl",
        "cache_dir": tmp_path / "blob_cache",
        "metadata_rows": 6,
        "window_size": 2,
        "content_blobs": 6,
        "max_per_repo": 6,
        "max_examples": 4,
        "min_bytes": 0,
    }
    values.update(overrides)
    return InspectionConfig(**values)


def test_revision_must_be_an_exact_lowercase_commit_hash():
    assert validate_revision(REVISION) == REVISION
    for invalid in ("main", "a" * 39, "A" * 40, "g" * 40, f"{REVISION}extra"):
        with pytest.raises(ValueError, match="40-hex"):
            validate_revision(invalid)


def test_deterministic_windows_are_exact_stratified_and_non_overlapping():
    first = deterministic_windows(
        total_rows=1_003,
        sample_rows=23,
        window_size=5,
        seed=19,
    )
    second = deterministic_windows(
        total_rows=1_003,
        sample_rows=23,
        window_size=5,
        seed=19,
    )
    assert first == second
    assert sum(window.length for window in first) == 23
    assert [window.length for window in first] == [5, 5, 5, 5, 3]
    assert all(
        left.offset + left.length <= right.offset
        for left, right in zip(first, first[1:], strict=False)
    )
    for index, window in enumerate(first):
        lower = 1_003 * index // len(first)
        upper = 1_003 * (index + 1) // len(first)
        assert lower <= window.offset
        assert window.offset + window.length <= upper


def test_content_selection_is_order_independent_deduplicated_and_repo_capped():
    rows = [
        {
            "row_idx": index,
            "blob_id": f"blob-{index}",
            "repo_name": "dominant" if index < 7 else f"repo-{index}",
            "path": f"p{index}.py",
            "length_bytes": 500,
            "score": 4.5,
            "int_score": 4,
        }
        for index in range(10)
    ]
    rows[0]["optional_metadata"] = "z"
    rows.append(dict(rows[0], optional_metadata="a"))
    forward, stats = select_content_rows(
        rows,
        count=5,
        max_per_repo=2,
        min_int_score=4,
        min_bytes=0,
        max_bytes=1_000,
        seed=7,
    )
    backward, _ = select_content_rows(
        list(reversed(rows)),
        count=5,
        max_per_repo=2,
        min_int_score=4,
        min_bytes=0,
        max_bytes=1_000,
        seed=7,
    )
    assert [row["blob_id"] for row in forward] == [row["blob_id"] for row in backward]
    assert len({row["blob_id"] for row in forward}) == len(forward) == 5
    assert max(Counter(row["repo_name"] for row in forward).values()) <= 2
    assert stats["duplicate_blob_id"] == 1


def test_bounded_inspection_records_quality_metrics_and_atomic_outputs(tmp_path: Path):
    blobs = {
        "blob-0": b'"""module docs"""\n# useful comment\ndef add(a, b):\n    return a + b\n',
        "blob-1": b"\xff\xfeinvalid utf8",
        "blob-2": b"def broken(:\n    pass\n",
        "blob-3": b"# AUTO-GENERATED - DO NOT EDIT\nvalue = 1\n",
        "blob-4": b"this fetch is replaced by an exception",
        "blob-5": ("value = 1\n" * 25).encode(),
    }
    rows = _source_rows(blobs)
    server = FakeDatasetServer(rows)
    fetched: list[str] = []

    def fetcher(blob_id: str) -> bytes:
        fetched.append(blob_id)
        if blob_id == "blob-4":
            raise OSError("synthetic fetch failure")
        return blobs[blob_id]

    ticks = iter(index / 100 for index in range(100))
    config = _config(tmp_path)
    report = inspect_python_source(
        config,
        transport=server,
        blob_fetcher=fetcher,
        clock=lambda: next(ticks),
    )

    assert len(fetched) == 6
    assert [endpoint for endpoint, _ in server.calls] == ["size", "rows", "rows", "rows"]
    assert report["dataset"]["total_rows"] == 6
    assert report["decision_scope"] == "NOT_SOURCE_APPROVAL"
    assert report["connectivity_gate"]["status"] == "PASS"
    assert report["connectivity_gate"]["scope"] == "CONNECTIVITY_ONLY_NOT_SOURCE_APPROVAL"
    assert report["sampling"]["sampled_metadata_rows"] == 6
    assert len(report["sampled_row_identifiers"]) == 6
    assert all(
        set(identifier) == {"row_idx", "blob_id", "repo_name", "path"}
        for identifier in report["sampled_row_identifiers"]
    )
    assert report["metadata_schema"]["features"] == server.features
    assert report["metadata_metrics"]["missing_required_fields"] == []
    optional = report["metadata_metrics"]["explicitly_checked_optional_fields"]
    assert optional["license"] == {"numerator": 0, "denominator": 6, "value": 0.0}
    assert optional["src_encoding"] == {"numerator": 0, "denominator": 6, "value": 0.0}
    assert report["content_selection"]["selected"] == 6
    content = report["content_metrics"]
    assert content["attempted"] == 6
    assert content["fetch_failed"] == 1
    assert content["strict_utf8_failed"] == 1
    assert content["strict_utf8_success"] == 4
    assert content["ast_parse_success"] == 3
    assert content["ast_parse_failed"] == 1
    assert content["vendor_path_heuristic"] == 1
    assert content["generated_file_heuristic"] == 1
    assert content["repetition_heuristic"] == 1
    assert content["has_docstring"] == 1
    assert content["has_comment"] >= 2
    assert content["metadata_length_mismatches"] == 0
    assert content["utf8_reencode_matches_raw"] == 4
    assert content["utf8_reencode_mismatches_raw"] == 0
    assert all(
        len(record["analysis"]["raw_sha256"]) == 64
        for record in report["content_records"]
        if record["outcome"] == "decoded"
    )
    assert report["limits"]["full_dataset_downloaded"] is False
    assert report["limits"]["token_yield_estimate_supported"] is False
    api_evidence = report["api_evidence"]
    successful = api_evidence["verified_successful_responses"]
    assert all(item["x_revision"] == REVISION for item in successful)
    assert all(item["revision_verified"] is True for item in successful)
    assert all(item["http_status"] == 200 for item in successful)
    assert all(item["body_bytes"] > 0 for item in successful)
    assert all(item["request_latency_seconds"] == 0.01 for item in successful)
    assert api_evidence["retry_count"] == 0
    assert content["rates"]["strict_utf8_success"] == {
        "numerator": 4,
        "denominator": 5,
        "value": 0.8,
    }

    on_disk = json.loads(config.report_path.read_text(encoding="utf-8"))
    actual_examples_path = Path(report["outputs"]["examples_path"])
    examples_bytes = actual_examples_path.read_bytes()
    examples = [json.loads(line) for line in examples_bytes.splitlines()]
    assert on_disk == report
    assert actual_examples_path != config.examples_path
    assert report["outputs"]["examples_content_handling"] == (
        "PRIVATE_FULLY_CHARACTER_REDACTED_SOURCE_SHAPES"
    )
    assert len(examples) == config.max_examples
    assert all(
        example["content_handling"] == "private_fully_character_redacted_source_shape"
        for example in examples
    )
    assert all(
        set(example["excerpt"]).issubset({"█", " ", "\t", "\r", "\n"}) for example in examples
    )
    assert all(example["redaction"]["source_characters_exposed"] == 0 for example in examples)
    assert all(len(example["excerpt"]) <= config.excerpt_chars for example in examples)
    assert report["outputs"]["examples_sha256"] == hashlib.sha256(examples_bytes).hexdigest()
    assert not list(tmp_path.glob(".*.tmp"))


def test_revision_drift_preserves_existing_outputs_even_with_overwrite(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    config = _config(tmp_path, overwrite=True)
    config.report_path.write_text("old report\n", encoding="utf-8")
    config.examples_path.write_text("old examples\n", encoding="utf-8")
    server = FakeDatasetServer(rows, row_revision="f" * 40)

    with pytest.raises(RuntimeError, match="revision drift"):
        inspect_python_source(
            config,
            transport=server,
            blob_fetcher=lambda blob_id: blobs[blob_id],
        )

    assert config.report_path.read_text(encoding="utf-8") == "old report\n"
    assert config.examples_path.read_text(encoding="utf-8") == "old examples\n"


def test_existing_outputs_require_explicit_overwrite_before_any_api_call(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    config = _config(tmp_path)
    config.report_path.write_text("keep me", encoding="utf-8")
    server = FakeDatasetServer(rows)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        inspect_python_source(
            config,
            transport=server,
            blob_fetcher=lambda blob_id: blobs[blob_id],
        )

    assert server.calls == []
    assert config.report_path.read_text(encoding="utf-8") == "keep me"


def test_non_finite_score_is_rejected_before_json_serialization():
    row = {
        "row_idx": 0,
        "blob_id": "blob",
        "repo_name": "repo",
        "path": "source.py",
        "length_bytes": 500,
        "score": float("nan"),
        "int_score": 4,
    }
    selected, stats = select_content_rows(
        [row],
        count=1,
        max_per_repo=1,
        min_int_score=4,
        min_bytes=0,
        max_bytes=1_000,
        seed=1,
    )
    assert selected == []
    assert stats["non_finite_score"] == 1


def test_upstream_row_idx_cannot_overwrite_verified_wrapper_index(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    rows[0]["row_idx"] = 999
    config = _config(tmp_path)
    server = FakeDatasetServer(rows)

    with pytest.raises(RuntimeError, match="reserved row_idx"):
        inspect_python_source(
            config,
            transport=server,
            blob_fetcher=lambda blob_id: blobs[blob_id],
        )

    assert not config.report_path.exists()


def test_non_finite_timeout_and_clock_fail_before_report_publication(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    with pytest.raises(ValueError, match="timeout_seconds"):
        inspect_python_source(
            _config(tmp_path, timeout_seconds=float("nan")),
            transport=FakeDatasetServer(rows),
            blob_fetcher=lambda blob_id: blobs[blob_id],
        )

    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite"):
        inspect_python_source(
            config,
            transport=FakeDatasetServer(rows),
            blob_fetcher=lambda blob_id: blobs[blob_id],
            clock=lambda: float("nan"),
        )
    assert not config.report_path.exists()


def test_transient_network_and_http_failures_retry_then_report_attempts(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    server = FakeDatasetServer(
        rows,
        row_outcomes=[
            TimeoutError("synthetic timeout"),
            URLError("synthetic connection reset"),
            502,
            200,
        ],
    )
    sleeps: list[float] = []
    report = inspect_python_source(
        _config(
            tmp_path,
            http_max_attempts=4,
            http_backoff_seconds=0.1,
            http_max_backoff_seconds=0.25,
        ),
        transport=server,
        blob_fetcher=lambda blob_id: blobs[blob_id],
        sleeper=sleeps.append,
    )

    evidence = report["api_evidence"]
    assert sleeps == [0.1, 0.2, 0.25]
    assert evidence["request_count"] == 4
    assert evidence["attempt_count"] == 7
    assert evidence["retry_count"] == 3
    attempts = evidence["all_request_attempts"]
    failures = [attempt for attempt in attempts if attempt["outcome"] != "success"]
    assert [attempt["outcome"] for attempt in failures] == [
        "transient_network_exception",
        "transient_network_exception",
        "transient_http_status",
    ]
    assert all(attempt["transient"] is True for attempt in failures)
    assert all(attempt["revision_verified"] is False for attempt in failures)
    http_failure = failures[-1]
    assert http_failure["http_status"] == 502
    assert http_failure["x_revision"] == "f" * 40
    assert report["inspection_config"]["datasets_server"]["retry_policy"] == {
        "max_attempts_per_request": 4,
        "backoff_strategy": "deterministic_exponential_capped_v1",
        "initial_backoff_seconds": 0.1,
        "maximum_backoff_seconds": 0.25,
        "retryable_http_statuses": [429, 500, 502, 503, 504],
        "retryable_exception_types": ["URLError", "TimeoutError"],
        "retry_after_header_honored": False,
        "revision_or_schema_error_retryable": False,
    }


def test_transient_http_exhaustion_does_not_publish_outputs(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    server = FakeDatasetServer(rows, row_outcomes=[503, 503])
    sleeps: list[float] = []
    config = _config(
        tmp_path,
        http_max_attempts=2,
        http_backoff_seconds=0.0,
        http_max_backoff_seconds=0.0,
    )
    with pytest.raises(RuntimeError, match="HTTP 503 exhausted 2 attempts"):
        inspect_python_source(
            config,
            transport=server,
            blob_fetcher=lambda blob_id: blobs[blob_id],
            sleeper=sleeps.append,
        )

    assert [endpoint for endpoint, _ in server.calls] == ["size", "rows", "rows"]
    assert sleeps == [0.0]
    assert not config.report_path.exists()
    assert not list(tmp_path.glob("*.jsonl"))


def test_nonretryable_http_error_is_attempted_once(tmp_path: Path):
    blobs = {f"blob-{index}": f"value = {index}\n".encode() for index in range(6)}
    rows = _source_rows(blobs)
    server = FakeDatasetServer(rows, row_outcomes=[404, 200, 200])
    sleeps: list[float] = []
    config = _config(tmp_path, http_max_attempts=4)
    with pytest.raises(RuntimeError, match="nonretryable HTTP 404"):
        inspect_python_source(
            config,
            transport=server,
            blob_fetcher=lambda blob_id: blobs[blob_id],
            sleeper=sleeps.append,
        )

    assert [endpoint for endpoint, _ in server.calls] == ["size", "rows"]
    assert sleeps == []
    assert not config.report_path.exists()
