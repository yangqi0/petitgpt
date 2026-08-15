#!/usr/bin/env python3

"""Collect the fixed P1 Python inspection sample without exposing source text.

The online path makes exactly one dataset-size request and fifty ten-row
Dataset Viewer requests, selects exactly 300 distinct Software Heritage blobs,
and publishes content-addressed evidence only after all selected objects have
been resolved.  The replay path performs no network access and verifies every
cached object against the immutable manifest.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import tokenize
from typing import Any
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.inspect_python_sources import (  # noqa: E402
    DatasetServerClient,
    HttpJsonResponse,
    _canonical_json_bytes,
    _publish_content_addressed,
    _write_bytes_atomic,
    deterministic_windows,
    validate_revision,
)
from pretrain.python_source_adapters import get_adapter  # noqa: E402

METADATA_ROWS = 500
WINDOW_COUNT = 50
WINDOW_SIZE = 10
SELECTED_BLOBS = 300
SEED = 20_250_814
MAX_METADATA_BYTES = 100_000
MAX_DECOMPRESSED_BYTES = 100_000
MAX_TOTAL_RAW_BYTES = 32 * 1024 * 1024
MAX_CACHE_OR_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_WORKERS = 4
HTTP_TIMEOUT_SECONDS = 30.0
MAX_ATTEMPTS_PER_CALL = 4
MAX_TOTAL_RETRIES = 50
MAX_WALL_SECONDS = 30 * 60
LATENCY_NANOSECONDS_PER_SECOND = 1_000_000_000
MAX_SINGLE_LATENCY_NANOSECONDS = MAX_WALL_SECONDS * LATENCY_NANOSECONDS_PER_SECOND
EXPECTED_HF_LOGICAL_CALLS = 51
_MANIFEST_KIND = "petitgpt_python_p1_collection"
_MANIFEST_VERSION = 1
CACHE_INDEX_SCHEMA_VERSION = 2
CACHE_ORIGIN_SCHEMA_VERSION = 1
CACHE_ORIGIN_KIND = "petitgpt_python_p1_swh_cache_origin"
_CACHE_PUBLICATION_LOCK = threading.Lock()
POLICY_KIND = "petitgpt_python_p1_matched_source_policy"
POLICY_STATUS = "FROZEN_BEFORE_COLLECTION"
DECISION_SCOPE = "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL"
_ADDRESS_RE = re.compile(r"\.sha256-([0-9a-f]{64})\.json$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_HF_API_ROOT = "https://datasets-server.huggingface.co"
SWH_BUCKET = "softwareheritage"
SWH_KEY_TEMPLATE = "content/{blob_id}"
SWH_REGION = "us-west-2"
SWH_AUTH = "anonymous_unsigned"

FETCH_FAILURE_CATEGORIES = frozenset({
    "access_denied",
    "decompressed_size_above_100000",
    "gzip_decode_failed",
    "not_found",
    "s3_client_error",
    "s3_transient_retries_exhausted",
    "transport_error_retries_exhausted",
})

POLICY_ARM_NAMES = {
    "smollm_python_edu": "smollm_python_edu_primary",
    "stack_edu_python": "stack_edu_python_comparator",
}
P1_COLLECTION_POLICY = {
    "content_blobs_per_arm": SELECTED_BLOBS,
    "content_selection": {
        "distinct_identity": "blob_id",
        "eligible_maximum_metadata_length_bytes": MAX_METADATA_BYTES,
        "maximum_per_repository": None,
        "minimum_int_score": None,
        "minimum_metadata_length_bytes": None,
        "ordering": "sha256(seed, blob_id) ascending",
        "oversize_rows_remain_in_metadata_accounting": True,
    },
    "metadata_rows_per_arm": METADATA_ROWS,
    "metadata_sampling": {
        "strategy": "seeded_contiguous_window_within_equal_row_index_strata_v1",
        "window_count": WINDOW_COUNT,
        "window_size": WINDOW_SIZE,
    },
    "seed": SEED,
}
P1_RESOURCE_POLICY = {
    "datasets_server_logical_calls": EXPECTED_HF_LOGICAL_CALLS,
    "decompressed_object_maximum_bytes": MAX_DECOMPRESSED_BYTES,
    "gpu_enabled": False,
    "http_max_attempts_per_call": MAX_ATTEMPTS_PER_CALL,
    "http_timeout_seconds": int(HTTP_TIMEOUT_SECONDS),
    "maximum_cache_and_output_bytes": MAX_CACHE_OR_OUTPUT_BYTES,
    "maximum_concurrent_blob_fetches": MAX_WORKERS,
    "maximum_total_raw_blob_bytes": MAX_TOTAL_RAW_BYTES,
    "maximum_total_retries": MAX_TOTAL_RETRIES,
    "maximum_wall_seconds": MAX_WALL_SECONDS,
    "swh_blob_selections": SELECTED_BLOBS,
}

Transport = Callable[[str, float], HttpJsonResponse]
BlobFetcher = Callable[[str], bytes]


class CollectionError(RuntimeError):
    """A fail-closed P1 contract violation."""


class BlobFetchIssue(RuntimeError):
    """A stable, reportable data-access failure category."""

    def __init__(self, category: str, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.category = category
        self.transient = transient


def _latency_nanoseconds(value: Any, *, label: str) -> int:
    """Quantize one measured duration before it enters immutable evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionError(f"{label} latency must be a finite nonnegative number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise CollectionError(f"{label} latency must be a finite nonnegative number")
    numerator, denominator = seconds.as_integer_ratio()
    quotient, remainder = divmod(
        numerator * LATENCY_NANOSECONDS_PER_SECOND,
        denominator,
    )
    doubled_remainder = remainder * 2
    if doubled_remainder > denominator or (doubled_remainder == denominator and quotient % 2 == 1):
        quotient += 1
    nanoseconds = quotient
    if nanoseconds > MAX_SINGLE_LATENCY_NANOSECONDS:
        raise CollectionError(f"{label} latency exceeds the P1 wall-clock bound")
    return nanoseconds


def _sum_latency_nanoseconds(values: Sequence[Any], *, label: str) -> int:
    """Validate and exactly accumulate already-quantized duration evidence."""
    total = 0
    for ordinal, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > MAX_SINGLE_LATENCY_NANOSECONDS
        ):
            raise CollectionError(f"{label} latency {ordinal} is not canonical nanoseconds")
        total += value
    return total


@dataclass(frozen=True)
class P1Config:
    expected_revision: str
    adapter: str
    dataset: str
    dataset_config: str
    split: str
    output_dir: Path
    cache_dir: Path
    policy_path: Path
    expected_policy_sha256: str
    api_root: str = DEFAULT_HF_API_ROOT
    workers: int = MAX_WORKERS
    replay_manifest: Path | None = None
    enforce_ignored_paths: bool = True


class NetworkBudget:
    """Thread-safe retry and wall-clock budget shared by HF and SWH calls."""

    def __init__(self, *, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.started = float(clock())
        self.retries = 0
        self.retries_by_backend: Counter[str] = Counter()
        self._lock = threading.Lock()

    def check(self) -> None:
        elapsed = float(self.clock()) - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise CollectionError("collection clock is invalid or moved backwards")
        if elapsed > MAX_WALL_SECONDS:
            raise CollectionError(f"P1 wall-clock budget exceeded: {elapsed:.3f}s")

    def begin_network_call(self) -> None:
        """Reserve one full per-call timeout inside the hard wall budget."""
        self.check()
        elapsed = float(self.clock()) - self.started
        if elapsed + HTTP_TIMEOUT_SECONDS > MAX_WALL_SECONDS:
            raise CollectionError("insufficient wall-clock budget for another bounded network call")

    def retry(self, *, backend: str, operation: str) -> None:
        if backend not in {"hf", "swh"}:
            raise AssertionError(f"unknown retry backend {backend!r}")
        with self._lock:
            if self.retries >= MAX_TOTAL_RETRIES:
                raise CollectionError(f"global retry budget exhausted before retrying {operation}")
            self.retries += 1
            self.retries_by_backend[backend] += 1
        self.check()

    @property
    def elapsed_nanoseconds(self) -> int:
        self.check()
        return _latency_nanoseconds(
            float(self.clock()) - self.started,
            label="collection elapsed",
        )


class BudgetedTransport:
    def __init__(self, base: Transport, budget: NetworkBudget) -> None:
        self.base = base
        self.budget = budget
        self._calls: Counter[str] = Counter()

    def __call__(self, url: str, timeout_seconds: float) -> HttpJsonResponse:
        self.budget.begin_network_call()
        if self._calls[url]:
            self.budget.retry(backend="hf", operation=f"HF {url.split('?', 1)[0]}")
        self._calls[url] += 1
        return self.base(url, timeout_seconds)


def requests_json_transport(url: str, timeout_seconds: float) -> HttpJsonResponse:
    """Use requests while translating network failures into P0 retry types."""
    import requests

    started = time.perf_counter()
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "PetitGPT-python-P1/1"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise URLError(str(exc)) from exc
    body = response.content
    try:
        payload = response.json() if response.status_code == 200 else {}
    except ValueError as exc:
        raise CollectionError("datasets-server returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise CollectionError("datasets-server returned non-object JSON")
    return HttpJsonResponse(
        payload=payload,
        headers={key.lower(): value for key, value in response.headers.items()},
        body_sha256=hashlib.sha256(body).hexdigest(),
        http_status=int(response.status_code),
        body_bytes=len(body),
        request_latency_seconds=time.perf_counter() - started,
    )


def _manifest_contract() -> dict[str, Any]:
    return {
        "metadata_rows": METADATA_ROWS,
        "windows": WINDOW_COUNT,
        "rows_per_window": WINDOW_SIZE,
        "selected_distinct_blobs": SELECTED_BLOBS,
        "seed": SEED,
        "selection_gates": ["length_bytes<=100000", "distinct_blob_id", "stable_hash_rank"],
        "explicitly_not_selection_gates": ["score", "minimum_size", "repository_cap"],
        "no_backfill_after_content_selection": True,
        "source_code_exposed_in_manifest": False,
        "timing_evidence": {
            "unit": "integer_nanoseconds",
            "quantization": "binary64_exact_ratio_round_half_even_per_measurement_v1",
            "accumulation": "exact_integer_sum",
        },
    }


def _expected_input(config: P1Config) -> dict[str, str]:
    return {
        "adapter": config.adapter,
        "dataset": config.dataset,
        "dataset_config": config.dataset_config,
        "split": config.split,
        "expected_revision": config.expected_revision,
    }


def _backend_provenance(
    config: P1Config,
    *,
    transport: Transport,
    blob_fetcher: BlobFetcher | None,
) -> dict[str, Any]:
    api_root = config.api_root.rstrip("/")
    hf_production = transport is requests_json_transport
    swh_production = blob_fetcher is None
    canonical_api_root = api_root == DEFAULT_HF_API_ROOT
    test_only = not hf_production or not swh_production
    production = canonical_api_root and hf_production and swh_production
    if test_only:
        mode = "test_only_injected"
    elif production:
        mode = "production"
    else:
        mode = "noncanonical_api_root"
    return {
        "schema_version": 1,
        "production": production,
        "test_only": test_only,
        "mode": mode,
        "hf": {
            "api_root": api_root,
            "api_root_is_canonical": canonical_api_root,
            "transport": "requests_json_transport" if hf_production else "injected_callable",
            "transport_mode": "production" if hf_production else "injected_test_only",
        },
        "swh": {
            "bucket": SWH_BUCKET if swh_production else None,
            "key_template": SWH_KEY_TEMPLATE if swh_production else None,
            "region": SWH_REGION if swh_production else None,
            "auth": SWH_AUTH if swh_production else None,
            "fetcher": "boto3_unsigned_s3" if swh_production else "injected_callable",
            "fetcher_mode": "production" if swh_production else "injected_test_only",
        },
    }


def cache_origin_contract(backend_provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact SWH origin allowed to populate or satisfy this cache."""
    swh = backend_provenance.get("swh")
    if not isinstance(swh, dict):
        raise CollectionError("backend provenance has no SWH origin")
    fetcher = swh.get("fetcher")
    if fetcher == "boto3_unsigned_s3":
        expected_swh = {
            "bucket": SWH_BUCKET,
            "key_template": SWH_KEY_TEMPLATE,
            "region": SWH_REGION,
            "auth": SWH_AUTH,
            "fetcher": "boto3_unsigned_s3",
            "fetcher_mode": "production",
        }
        production = True
    elif fetcher == "injected_callable":
        expected_swh = {
            "bucket": None,
            "key_template": None,
            "region": None,
            "auth": None,
            "fetcher": "injected_callable",
            "fetcher_mode": "injected_test_only",
        }
        production = False
    else:
        raise CollectionError("unknown SWH cache origin")
    if swh != expected_swh:
        raise CollectionError("SWH backend provenance cannot define a cache origin")
    return {
        "schema_version": CACHE_ORIGIN_SCHEMA_VERSION,
        "kind": CACHE_ORIGIN_KIND,
        "production": production,
        **expected_swh,
    }


def _validate_cache_origin_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollectionError("cache origin contract is missing or malformed")
    production_backend = {
        "swh": {
            "bucket": SWH_BUCKET,
            "key_template": SWH_KEY_TEMPLATE,
            "region": SWH_REGION,
            "auth": SWH_AUTH,
            "fetcher": "boto3_unsigned_s3",
            "fetcher_mode": "production",
        }
    }
    injected_backend = {
        "swh": {
            "bucket": None,
            "key_template": None,
            "region": None,
            "auth": None,
            "fetcher": "injected_callable",
            "fetcher_mode": "injected_test_only",
        }
    }
    allowed = (
        cache_origin_contract(production_backend),
        cache_origin_contract(injected_backend),
    )
    if value not in allowed:
        raise CollectionError("unknown or noncanonical cache origin contract")
    return dict(value)


def _policy_arm_tuple(arm: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    try:
        values = (
            str(arm["name"]),
            str(arm["dataset"]),
            str(arm["dataset_config"]),
            str(arm["split"]),
            str(arm["expected_revision"]),
        )
    except KeyError as exc:
        raise CollectionError(f"P1 policy arm is missing {exc.args[0]!r}") from exc
    validate_revision(values[-1])
    return values


def load_policy_binding(config: P1Config) -> dict[str, Any]:
    try:
        raw = config.policy_path.read_bytes()
        policy = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"cannot read P1 policy JSON: {config.policy_path}") from exc
    actual_policy_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_policy_sha256 != config.expected_policy_sha256:
        raise CollectionError("P1 policy SHA-256 does not match --expected_policy_sha256")
    if not isinstance(policy, dict):
        raise CollectionError("P1 policy must be a JSON object")
    expected_header = {
        "schema_version": 1,
        "kind": POLICY_KIND,
        "status": POLICY_STATUS,
        "decision_scope": DECISION_SCOPE,
    }
    for field, expected in expected_header.items():
        if policy.get(field) != expected:
            raise CollectionError(
                f"P1 policy {field} mismatch: expected {expected!r}, got {policy.get(field)!r}"
            )
    if policy.get("collection") != P1_COLLECTION_POLICY:
        raise CollectionError("P1 policy collection block does not match collector constants")
    if policy.get("resource_budget_per_arm") != P1_RESOURCE_POLICY:
        raise CollectionError("P1 policy resource block does not match collector constants")
    privacy = policy.get("privacy")
    if not isinstance(privacy, dict):
        raise CollectionError("P1 policy has no privacy block")
    if privacy.get("cache_and_reports_git_policy") != "DO_NOT_COMMIT":
        raise CollectionError("P1 policy must keep cache and reports private")
    if privacy.get("report_source_characters_exposed") != 0:
        raise CollectionError("P1 policy must expose zero source characters")
    if not isinstance(policy.get("analysis"), dict):
        raise CollectionError("P1 policy has no frozen analysis block")
    raw_arms = policy.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) != 2:
        raise CollectionError("P1 policy must contain exactly two source arms")
    if not all(isinstance(arm, dict) for arm in raw_arms):
        raise CollectionError("P1 policy arms must be JSON objects")
    arm_tuples = [_policy_arm_tuple(arm) for arm in raw_arms]
    names = {arm[0] for arm in arm_tuples}
    if names != set(POLICY_ARM_NAMES.values()):
        raise CollectionError("P1 policy must bind exactly the frozen primary/comparator arms")
    if len(set(arm_tuples)) != 2:
        raise CollectionError("P1 policy contains duplicate source arms")
    current = (
        POLICY_ARM_NAMES[config.adapter],
        config.dataset,
        config.dataset_config,
        config.split,
        config.expected_revision,
    )
    if current not in arm_tuples:
        raise CollectionError(f"current source arm is not bound by policy: {current!r}")
    return {
        "path": str(config.policy_path),
        "sha256": actual_policy_sha256,
        "expected_sha256": config.expected_policy_sha256,
        "schema_version": 1,
        "kind": POLICY_KIND,
        "status": POLICY_STATUS,
        "decision_scope": DECISION_SCOPE,
        "arm_tuple": list(current),
    }


def validate_p1_config(config: P1Config) -> None:
    validate_revision(config.expected_revision)
    get_adapter(config.adapter)
    if _SHA256_RE.fullmatch(config.expected_policy_sha256) is None:
        raise ValueError("expected_policy_sha256 must be an exact lowercase 64-hex SHA-256")
    if not config.dataset or not config.dataset_config or not config.split:
        raise ValueError("dataset, dataset_config, and split must be non-empty")
    if not 1 <= config.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")
    if config.output_dir.resolve() == config.cache_dir.resolve():
        raise ValueError("output_dir and private cache_dir must differ")
    if config.enforce_ignored_paths:
        for label, path in (
            ("raw cache", config.cache_dir),
            ("metadata output", config.output_dir),
        ):
            try:
                relative = path.resolve().relative_to(PROJECT_ROOT)
            except ValueError as exc:
                raise CollectionError(f"{label} must be inside the Git worktree") from exc
            result = subprocess.run(
                ["git", "check-ignore", "-q", "--", str(relative)],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if result.returncode != 0:
                raise CollectionError(f"{label} is not Git-ignored: {path}")


def _tree_usage(path: Path) -> dict[str, int]:
    usage = {
        "logical_file_bytes": 0,
        "apparent_bytes_including_directories": 0,
        "allocated_bytes": 0,
        "file_count": 0,
        "directory_count": 0,
    }
    if not path.exists():
        return usage
    for item in (path, *path.rglob("*")):
        if item.is_symlink():
            raise CollectionError(f"symlink is forbidden in bounded P1 storage: {item}")
        stat = item.stat()
        usage["apparent_bytes_including_directories"] += int(stat.st_size)
        usage["allocated_bytes"] += int(getattr(stat, "st_blocks", 0)) * 512
        if item.is_file():
            usage["file_count"] += 1
            usage["logical_file_bytes"] += int(stat.st_size)
        elif item.is_dir():
            usage["directory_count"] += 1
    return usage


def _check_storage_bounds(config: P1Config) -> dict[str, dict[str, int]]:
    usages = {
        label: _tree_usage(path)
        for label, path in (("cache", config.cache_dir), ("output", config.output_dir))
    }
    for label, usage in usages.items():
        for metric in (
            "logical_file_bytes",
            "apparent_bytes_including_directories",
            "allocated_bytes",
        ):
            if usage[metric] > MAX_CACHE_OR_OUTPUT_BYTES:
                raise CollectionError(
                    f"P1 {label} {metric} exceeds {MAX_CACHE_OR_OUTPUT_BYTES}: {usage[metric]}"
                )
    for metric in (
        "logical_file_bytes",
        "apparent_bytes_including_directories",
        "allocated_bytes",
    ):
        combined = sum(usage[metric] for usage in usages.values())
        if combined > MAX_CACHE_OR_OUTPUT_BYTES:
            raise CollectionError(
                f"combined P1 cache/output {metric} exceeds {MAX_CACHE_OR_OUTPUT_BYTES}: {combined}"
            )
    return usages


def _allocation_unit(path: Path) -> int:
    probe = path if path.exists() else path.parent
    try:
        statvfs = os.statvfs(probe)
    except OSError as exc:
        raise CollectionError(f"cannot determine allocation unit for {probe}") from exc
    return max(512, int(statvfs.f_frsize), int(statvfs.f_bsize))


def _preflight_cache_write(
    config: P1Config,
    payloads: Sequence[tuple[Path, bytes]],
) -> dict[str, int]:
    usages = _check_storage_bounds(config)
    new_payloads = [(path, data) for path, data in payloads if not path.exists()]
    unit = _allocation_unit(config.cache_dir)
    estimate = {
        "logical_file_bytes": sum(len(data) for _, data in new_payloads),
        "apparent_bytes_including_directories": sum(len(data) for _, data in new_payloads)
        + unit * len(new_payloads),
        "allocated_bytes": sum(
            math.ceil(max(1, len(data)) / unit) * unit for _, data in new_payloads
        )
        + unit * len(new_payloads),
    }
    for metric, addition in estimate.items():
        if usages["cache"][metric] + addition > MAX_CACHE_OR_OUTPUT_BYTES:
            raise CollectionError(f"P1 cache preflight {metric} would exceed storage budget")
        combined = usages["cache"][metric] + usages["output"][metric] + addition
        if combined > MAX_CACHE_OR_OUTPUT_BYTES:
            raise CollectionError(
                f"combined P1 cache/output preflight {metric} would exceed storage budget"
            )
    return estimate


def _raw_has_cache_reference(cache_dir: Path, raw_sha256: str) -> bool:
    for index_path in cache_dir.glob("index-*.json"):
        try:
            value = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if not isinstance(value, dict):
            return True
        if value.get("raw_sha256") == raw_sha256:
            return True
    return False


def _rollback_new_cache_files(
    cache_dir: Path,
    *,
    index_path: Path,
    index_created: bool,
    raw_path: Path,
    raw_created: bool,
    raw_sha256: str,
) -> None:
    if index_created:
        index_path.unlink(missing_ok=True)
    if raw_created and not _raw_has_cache_reference(cache_dir, raw_sha256):
        raw_path.unlink(missing_ok=True)


def _stable_selection_rank(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}\0{row['blob_id']}".encode()).hexdigest()


def select_p1_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply only the frozen max-length, distinct-ID, stable-rank contract."""
    rejected: Counter[str] = Counter()
    canonical_by_blob: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        blob_id = str(row["blob_id"])
        if int(row["length_bytes"]) > MAX_METADATA_BYTES:
            rejected["metadata_length_above_100000"] += 1
            continue
        current = canonical_by_blob.get(blob_id)
        if current is None or int(row["row_idx"]) < int(current["row_idx"]):
            if current is not None:
                rejected["duplicate_blob_id"] += 1
            canonical_by_blob[blob_id] = row
        else:
            rejected["duplicate_blob_id"] += 1

    ordered = sorted(
        canonical_by_blob.values(),
        key=lambda row: (_stable_selection_rank(row), str(row["blob_id"])),
    )
    if len(ordered) < SELECTED_BLOBS:
        raise CollectionError(
            f"P1 selection underfilled: need {SELECTED_BLOBS} distinct <=100KB blobs, "
            f"found {len(ordered)}"
        )
    selected = ordered[:SELECTED_BLOBS]
    rejected["not_selected_by_stable_rank"] += len(ordered) - len(selected)
    return selected, {
        "input_rows": len(rows),
        "eligible_distinct_blob_ids": len(ordered),
        "selected": len(selected),
        **{name: int(value) for name, value in sorted(rejected.items())},
    }


def _cache_paths(
    cache_dir: Path,
    blob_id: str,
    raw_sha256: str | None = None,
) -> tuple[Path, Path | None]:
    key = hashlib.sha256(blob_id.encode()).hexdigest()
    index = cache_dir / f"index-{key}.json"
    raw_path = None
    if raw_sha256 is not None:
        raw_path = cache_dir / f"raw-sha256-{raw_sha256}.raw"
    return index, raw_path


def _verify_raw(path: Path, *, expected_sha256: str, expected_bytes: int) -> bytes:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise CollectionError(f"missing cached raw object: {path}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise CollectionError(
            f"cached raw SHA tamper/corruption at {path}: expected {expected_sha256}, got {actual}"
        )
    if len(raw) != expected_bytes:
        raise CollectionError(
            f"cached raw size drift at {path}: expected {expected_bytes}, got {len(raw)}"
        )
    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise CollectionError(f"cached raw object exceeds {MAX_DECOMPRESSED_BYTES} bytes")
    return raw


def _load_cache_entry(
    cache_dir: Path,
    blob_id: str,
    *,
    expected_origin: Mapping[str, Any],
) -> tuple[bytes, str, str] | None:
    canonical_origin = _validate_cache_origin_contract(expected_origin)
    index_path, _ = _cache_paths(cache_dir, blob_id)
    if not index_path.exists():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"invalid cache index: {index_path}") from exc
    if not isinstance(index, dict):
        raise CollectionError(f"invalid cache index: {index_path}")
    if index.get("schema_version") != CACHE_INDEX_SCHEMA_VERSION:
        raise CollectionError(
            f"legacy or unsupported cache index at {index_path}; refusing unbound cache reuse"
        )
    expected_fields = {
        "schema_version",
        "blob_id",
        "raw_sha256",
        "raw_bytes",
        "raw_path",
        "origin",
    }
    if set(index) != expected_fields:
        raise CollectionError(f"invalid cache index fields at {index_path}")
    if index.get("blob_id") != blob_id:
        raise CollectionError(f"cache index blob ID mismatch at {index_path}")
    recorded_origin = _validate_cache_origin_contract(index.get("origin"))
    if recorded_origin != canonical_origin:
        raise CollectionError(
            f"cache origin mismatch at {index_path}; refusing cross-origin cache reuse"
        )
    raw_sha256 = index.get("raw_sha256")
    raw_bytes = index.get("raw_bytes")
    raw_path_value = index.get("raw_path")
    if not isinstance(raw_sha256, str) or _SHA256_RE.fullmatch(raw_sha256) is None:
        raise CollectionError(f"invalid raw digest in cache index: {index_path}")
    if (
        isinstance(raw_bytes, bool)
        or not isinstance(raw_bytes, int)
        or raw_bytes < 0
        or raw_bytes > MAX_DECOMPRESSED_BYTES
    ):
        raise CollectionError(f"invalid raw size in cache index: {index_path}")
    _, raw_path = _cache_paths(cache_dir, blob_id, raw_sha256)
    assert raw_path is not None
    expected_raw_path = str(raw_path.relative_to(cache_dir))
    if raw_path_value != expected_raw_path:
        raise CollectionError(f"noncanonical raw path in cache index: {index_path}")
    raw = _verify_raw(
        raw_path,
        expected_sha256=raw_sha256,
        expected_bytes=raw_bytes,
    )
    return raw, raw_sha256, expected_raw_path


def _store_cache_entry(
    config: P1Config,
    blob_id: str,
    raw: bytes,
    *,
    origin: Mapping[str, Any],
) -> tuple[str, str]:
    cache_dir = config.cache_dir
    canonical_origin = _validate_cache_origin_contract(origin)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    index_path, raw_path = _cache_paths(cache_dir, blob_id, raw_sha256)
    assert raw_path is not None
    index_bytes = (
        _canonical_json_bytes({
            "schema_version": CACHE_INDEX_SCHEMA_VERSION,
            "blob_id": blob_id,
            "raw_sha256": raw_sha256,
            "raw_bytes": len(raw),
            "raw_path": str(raw_path.relative_to(cache_dir)),
            "origin": canonical_origin,
        })
        + b"\n"
    )
    with _CACHE_PUBLICATION_LOCK:
        raw_exists = raw_path.exists()
        index_exists = index_path.exists()
        if raw_exists:
            _verify_raw(
                raw_path,
                expected_sha256=raw_sha256,
                expected_bytes=len(raw),
            )
        if index_exists and index_path.read_bytes() != index_bytes:
            raise CollectionError(f"cache index collision/tamper at {index_path}")
        _preflight_cache_write(config, ((raw_path, raw), (index_path, index_bytes)))
        raw_created = False
        index_created = False
        try:
            if not raw_exists:
                _write_bytes_atomic(raw_path, raw)
                raw_created = True
            if not index_exists:
                _write_bytes_atomic(index_path, index_bytes)
                index_created = True
            _check_storage_bounds(config)
        except Exception:
            _rollback_new_cache_files(
                cache_dir,
                index_path=index_path,
                index_created=index_created,
                raw_path=raw_path,
                raw_created=raw_created,
                raw_sha256=raw_sha256,
            )
            raise
    return raw_sha256, str(raw_path.relative_to(cache_dir))


def make_bounded_swh_fetcher() -> BlobFetcher:
    try:
        import boto3
        import botocore
    except ImportError as exc:
        raise RuntimeError("boto3/botocore are required for online SWH collection") from exc

    client = boto3.client(
        "s3",
        region_name=SWH_REGION,
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED,
            connect_timeout=HTTP_TIMEOUT_SECONDS,
            read_timeout=HTTP_TIMEOUT_SECONDS,
            retries={"total_max_attempts": 1, "mode": "standard"},
        ),
    )
    transient_exceptions = (
        botocore.exceptions.ConnectTimeoutError,
        botocore.exceptions.ReadTimeoutError,
        botocore.exceptions.EndpointConnectionError,
        botocore.exceptions.ConnectionClosedError,
    )
    transient_codes = {
        "InternalError",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
    }
    terminal_codes = {
        "AccessDenied": "access_denied",
        "InvalidAccessKeyId": "access_denied",
        "NoSuchKey": "not_found",
        "NotFound": "not_found",
    }

    def fetch(blob_id: str) -> bytes:
        try:
            response = client.get_object(
                Bucket=SWH_BUCKET,
                Key=SWH_KEY_TEMPLATE.format(blob_id=blob_id),
            )
        except botocore.exceptions.ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", "Unknown"))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code in transient_codes or status == 429 or status >= 500:
                raise BlobFetchIssue(
                    "s3_transient",
                    f"S3 transient {code} HTTP {status}",
                    transient=True,
                ) from exc
            raise BlobFetchIssue(
                terminal_codes.get(code, "s3_client_error"),
                f"S3 terminal {code} HTTP {status}",
                transient=False,
            ) from exc
        except transient_exceptions as exc:
            raise BlobFetchIssue(
                "transport_error",
                type(exc).__name__,
                transient=True,
            ) from exc

        body = response["Body"]
        try:
            with gzip.GzipFile(fileobj=body) as compressed:
                raw = compressed.read(MAX_DECOMPRESSED_BYTES + 1)
        except transient_exceptions as exc:
            raise BlobFetchIssue(
                "transport_error",
                type(exc).__name__,
                transient=True,
            ) from exc
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            raise BlobFetchIssue(
                "gzip_decode_failed",
                type(exc).__name__,
                transient=False,
            ) from exc
        finally:
            body.close()
        if len(raw) > MAX_DECOMPRESSED_BYTES:
            raise BlobFetchIssue(
                "decompressed_size_above_100000",
                f"object exceeds {MAX_DECOMPRESSED_BYTES} decompressed bytes",
                transient=False,
            )
        return raw

    return fetch


def _source_quality(path: str, raw: bytes) -> tuple[str, dict[str, Any]]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return "decode_failed", {
            "decode_error": {
                "category": "UnicodeDecodeError",
                "start": int(exc.start),
                "end": int(exc.end),
            },
            "raw_bytes": len(raw),
        }
    try:
        tree = ast.parse(text, filename=path or "<blob>")
        ast_ok = True
        has_docstring = any(
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node, clean=False) is not None
            for node in ast.walk(tree)
        )
    except (SyntaxError, ValueError) as exc:
        ast_ok = False
        has_docstring = False
        if isinstance(exc, TabError):
            category = "TabError"
        elif isinstance(exc, IndentationError):
            category = "IndentationError"
        elif isinstance(exc, SyntaxError):
            category = "SyntaxError"
        else:
            category = "ValueError"
        ast_error = {
            "category": category,
            "line": int(exc.lineno) if isinstance(exc, SyntaxError) and exc.lineno else None,
            "offset": int(exc.offset) if isinstance(exc, SyntaxError) and exc.offset else None,
            "end_line": int(exc.end_lineno)
            if isinstance(exc, SyntaxError) and exc.end_lineno
            else None,
            "end_offset": int(exc.end_offset)
            if isinstance(exc, SyntaxError) and exc.end_offset
            else None,
        }
    try:
        has_comment = any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
        )
    except (IndentationError, tokenize.TokenError):
        has_comment = False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(lines)
    repeated = sum(value - 1 for value in counts.values() if value > 1)
    repetition_ratio = repeated / len(lines) if lines else 0.0
    lowered_path = path.lower().replace("\\", "/")
    parts = {part for part in lowered_path.split("/") if part}
    head = "\n".join(text.splitlines()[:30]).lower()
    vendor = bool(
        parts.intersection({"vendor", "vendored", "third_party", "thirdparty", "site-packages"})
    )
    generated = any(
        marker in head
        for marker in (
            "@generated",
            "auto-generated",
            "autogenerated",
            "do not edit",
            "generated by",
        )
    )
    result: dict[str, Any] = {
        "raw_bytes": len(raw),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "ast_parse_ok": ast_ok,
        "has_docstring": has_docstring,
        "has_comment": has_comment,
        "vendor_path_heuristic": vendor,
        "generated_file_heuristic": generated,
        "repeated_line_ratio": repetition_ratio,
        "repetition_heuristic": len(lines) >= 20
        and (repetition_ratio >= 0.30 or max(counts.values(), default=0) >= 5),
    }
    if not ast_ok:
        result["ast_error"] = ast_error
    return "decoded", result


def _blob_issue(exc: BaseException) -> BlobFetchIssue | None:
    if isinstance(exc, BlobFetchIssue):
        return exc
    if isinstance(exc, KeyError):
        return BlobFetchIssue("not_found", type(exc).__name__, transient=False)
    if isinstance(exc, PermissionError):
        return BlobFetchIssue("access_denied", type(exc).__name__, transient=False)
    if isinstance(exc, (TimeoutError, URLError, ConnectionError, OSError)):
        return BlobFetchIssue("transport_error", type(exc).__name__, transient=True)
    return None


def _fetch_failure_entry(
    row: Mapping[str, Any],
    *,
    issue: BlobFetchIssue,
    attempts: int,
    attempt_latencies_nanoseconds: Sequence[int],
) -> dict[str, Any]:
    category = issue.category
    if issue.transient and attempts == MAX_ATTEMPTS_PER_CALL:
        category = f"{category}_retries_exhausted"
    return {
        "metadata": dict(row),
        "selection_rank_sha256": _stable_selection_rank(row),
        "fetch_outcome": "failed",
        "error_category": category,
        "fetch_attempts": attempts,
        "fetch_attempt_latencies_nanoseconds": list(attempt_latencies_nanoseconds),
        "fetch_latency_nanoseconds": _sum_latency_nanoseconds(
            attempt_latencies_nanoseconds,
            label="SWH fetch failure",
        ),
        "cache_hit": False,
        "fidelity_outcome": "not_available",
        "decode_outcome": "not_attempted",
    }


def _fetch_selected(
    row: Mapping[str, Any],
    *,
    config: P1Config,
    fetcher: BlobFetcher,
    budget: NetworkBudget,
    expected_origin: Mapping[str, Any],
) -> dict[str, Any]:
    budget.check()
    blob_id = str(row["blob_id"])
    cached = _load_cache_entry(
        config.cache_dir,
        blob_id,
        expected_origin=expected_origin,
    )
    cache_hit = cached is not None
    attempts = 0
    attempt_latencies_nanoseconds: list[int] = []
    if cached is None:
        raw: bytes | None = None
        for attempt in range(1, MAX_ATTEMPTS_PER_CALL + 1):
            attempts = attempt
            budget.begin_network_call()
            started = float(budget.clock())
            try:
                raw = fetcher(blob_id)
            except Exception as exc:
                latency_nanoseconds = _latency_nanoseconds(
                    float(budget.clock()) - started,
                    label="SWH failed fetch",
                )
                attempt_latencies_nanoseconds.append(latency_nanoseconds)
                issue = _blob_issue(exc)
                if issue is None:
                    raise CollectionError(
                        f"internal blob fetcher failure for {blob_id}: {type(exc).__name__}"
                    ) from exc
                if not issue.transient or attempt == MAX_ATTEMPTS_PER_CALL:
                    return _fetch_failure_entry(
                        row,
                        issue=issue,
                        attempts=attempt,
                        attempt_latencies_nanoseconds=attempt_latencies_nanoseconds,
                    )
                budget.retry(backend="swh", operation=f"SWH blob {blob_id}")
            else:
                latency_nanoseconds = _latency_nanoseconds(
                    float(budget.clock()) - started,
                    label="SWH successful fetch",
                )
                attempt_latencies_nanoseconds.append(latency_nanoseconds)
                break
        if raw is None:
            raise AssertionError("bounded blob loop exited without result")
        if not isinstance(raw, bytes):
            raise CollectionError("blob fetcher returned a non-bytes object")
        if len(raw) > MAX_DECOMPRESSED_BYTES:
            issue = BlobFetchIssue(
                "decompressed_size_above_100000",
                f"object exceeds {MAX_DECOMPRESSED_BYTES} decompressed bytes",
                transient=False,
            )
            return _fetch_failure_entry(
                row,
                issue=issue,
                attempts=attempts,
                attempt_latencies_nanoseconds=attempt_latencies_nanoseconds,
            )
        raw_sha256, relative_path = _store_cache_entry(
            config,
            blob_id,
            raw,
            origin=expected_origin,
        )
    else:
        raw, raw_sha256, relative_path = cached

    base = {
        "metadata": dict(row),
        "selection_rank_sha256": _stable_selection_rank(row),
        "fetch_outcome": "success",
        "error_category": None,
        "fetch_attempts": attempts,
        "fetch_attempt_latencies_nanoseconds": list(attempt_latencies_nanoseconds),
        "fetch_latency_nanoseconds": _sum_latency_nanoseconds(
            attempt_latencies_nanoseconds,
            label="SWH fetch success",
        ),
        "raw_sha256": raw_sha256,
        "raw_bytes": len(raw),
        "cache_object": relative_path,
        "cache_hit": cache_hit,
        "cache_origin_verified": cache_hit,
    }
    if len(raw) != int(row["length_bytes"]):
        return {
            **base,
            "fidelity_outcome": "metadata_length_mismatch",
            "decode_outcome": "not_attempted",
        }
    outcome, quality = _source_quality(str(row.get("path", "")), raw)
    return {
        **base,
        "fidelity_outcome": "length_matches_metadata",
        "decode_outcome": outcome,
        "quality": quality,
    }


def _metadata_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def distribution(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": len(values),
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    repos = Counter(str(row["repo_name"]) for row in rows if "repo_name" in row)
    paths = Counter(str(row["path"]) for row in rows if "path" in row)
    fields = sorted({field for row in rows for field in row})
    return {
        "rows": len(rows),
        "normalized_fields": fields,
        "field_presence": {field: sum(field in row for row in rows) for field in fields},
        "length_bytes": distribution([float(row["length_bytes"]) for row in rows]),
        "score": distribution([float(row["score"]) for row in rows if "score" in row]),
        "int_score_counts": dict(
            sorted(Counter(str(row["int_score"]) for row in rows if "int_score" in row).items())
        ),
        "unique_repositories": len(repos),
        "unique_paths": len(paths),
        "top_repositories": repos.most_common(10),
        "top_paths": paths.most_common(10),
        "provenance_presence": {
            field: sum(field in row for row in rows)
            for field in (
                "blob_id",
                "repo_name",
                "path",
                "license",
                "detected_licenses",
                "license_type",
                "src_encoding",
            )
        },
    }


def _content_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fetched = [entry for entry in entries if entry["fetch_outcome"] == "success"]
    faithful = [
        entry for entry in fetched if entry["fidelity_outcome"] == "length_matches_metadata"
    ]
    decoded = [entry for entry in faithful if entry["decode_outcome"] == "decoded"]
    quality = [entry["quality"] for entry in decoded]
    reasons = Counter(
        str(entry["error_category"]) for entry in entries if entry.get("error_category") is not None
    )
    reasons.update(
        str(entry["fidelity_outcome"])
        for entry in fetched
        if entry["fidelity_outcome"] != "length_matches_metadata"
    )
    reasons.update(
        str(entry["decode_outcome"]) for entry in faithful if entry["decode_outcome"] != "decoded"
    )
    return {
        "selected_attempts": len(entries),
        "fetch_success": len(fetched),
        "fetch_failed": len(entries) - len(fetched),
        "fidelity_length_matches": len(faithful),
        "fidelity_length_mismatch": len(fetched) - len(faithful),
        "strict_utf8_success": len(decoded),
        "strict_utf8_failed": len(faithful) - len(decoded),
        "downstream_not_attempted": len(entries) - len(faithful),
        "failure_counts_by_cause": dict(sorted(reasons.items())),
        "ast_parse_success": sum(item["ast_parse_ok"] for item in quality),
        "ast_parse_failed": sum(not item["ast_parse_ok"] for item in quality),
        "vendor_path_heuristic": sum(item["vendor_path_heuristic"] for item in quality),
        "generated_file_heuristic": sum(item["generated_file_heuristic"] for item in quality),
        "repetition_heuristic": sum(item["repetition_heuristic"] for item in quality),
        "has_docstring": sum(item["has_docstring"] for item in quality),
        "has_comment": sum(item["has_comment"] for item in quality),
        "cache_hits": sum(entry.get("cache_hit") is True for entry in entries),
        "cache_misses": sum(entry.get("cache_hit") is False for entry in entries),
        "total_raw_bytes": sum(int(entry.get("raw_bytes", 0)) for entry in entries),
    }


def _normalized_hf_evidence(client: DatasetServerClient) -> dict[str, Any]:
    def normalize(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        if (
            "request_latency_seconds" not in record
            or "request_latency_nanoseconds" in record
            or "backoff_before_next_attempt_seconds" not in record
            or "backoff_before_next_attempt_nanoseconds" in record
        ):
            raise CollectionError(f"{label} has noncanonical latency evidence")
        normalized = dict(record)
        seconds = normalized.pop("request_latency_seconds")
        normalized["request_latency_nanoseconds"] = _latency_nanoseconds(
            seconds,
            label=label,
        )
        backoff_seconds = normalized.pop("backoff_before_next_attempt_seconds")
        normalized["backoff_before_next_attempt_nanoseconds"] = (
            None
            if backoff_seconds is None
            else _latency_nanoseconds(backoff_seconds, label=f"{label} backoff")
        )
        return normalized

    return {
        "logical_calls": client._request_ordinal,
        "responses": [
            normalize(response, label=f"HF response {ordinal}")
            for ordinal, response in enumerate(client.responses, start=1)
        ],
        "attempts": [
            normalize(attempt, label=f"HF attempt {ordinal}")
            for ordinal, attempt in enumerate(client.attempts, start=1)
        ],
    }


def _backend_accounting(
    hf_evidence: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    budget: NetworkBudget,
) -> dict[str, Any]:
    hf_logical_calls = int(hf_evidence["logical_calls"])
    hf_attempt_evidence = hf_evidence["attempts"]
    hf_response_evidence = hf_evidence["responses"]
    hf_attempts = len(hf_attempt_evidence)
    hf_retries = hf_attempts - hf_logical_calls
    if hf_retries < 0:
        raise CollectionError("HF attempt accounting underflow")
    swh_attempts = sum(int(entry["fetch_attempts"]) for entry in entries)
    swh_retries = sum(max(0, int(entry["fetch_attempts"]) - 1) for entry in entries)
    if hf_retries != int(budget.retries_by_backend["hf"]):
        raise CollectionError("HF retry accounting drift")
    if swh_retries != int(budget.retries_by_backend["swh"]):
        raise CollectionError("SWH retry accounting drift")
    if hf_retries + swh_retries != budget.retries:
        raise CollectionError("global retry accounting drift")
    return {
        "hf": {
            "logical_calls": hf_logical_calls,
            "attempts": hf_attempts,
            "retries": hf_retries,
            "total_latency_nanoseconds": _sum_latency_nanoseconds(
                [attempt["request_latency_nanoseconds"] for attempt in hf_attempt_evidence],
                label="HF total",
            ),
            "final_outcomes": dict(
                sorted(
                    Counter(str(response["outcome"]) for response in hf_response_evidence).items()
                )
            ),
        },
        "swh": {
            "selected_objects": len(entries),
            "network_objects": sum(int(entry["fetch_attempts"]) > 0 for entry in entries),
            "cache_hits": sum(entry.get("cache_hit") is True for entry in entries),
            "cache_origin_verified": all(
                entry.get("cache_origin_verified") is True
                for entry in entries
                if entry.get("cache_hit") is True
            ),
            "attempts": swh_attempts,
            "retries": swh_retries,
            "total_latency_nanoseconds": _sum_latency_nanoseconds(
                [entry["fetch_latency_nanoseconds"] for entry in entries],
                label="SWH total",
            ),
            "final_error_counts": dict(
                sorted(
                    Counter(
                        str(entry["error_category"])
                        for entry in entries
                        if entry.get("error_category") is not None
                    ).items()
                )
            ),
        },
        "total_retries": budget.retries,
    }


def _content_addressed_json(output_dir: Path, stem: str, value: Mapping[str, Any]) -> Path:
    data = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    digest = hashlib.sha256(data).hexdigest()
    path = output_dir / f"{stem}.sha256-{digest}.json"
    _publish_content_addressed(path, data)
    return path


def _online_collect(
    config: P1Config,
    *,
    policy_binding: Mapping[str, Any],
    transport: Transport,
    blob_fetcher: BlobFetcher | None,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> tuple[dict[str, Any], dict[str, Any], NetworkBudget]:
    budget = NetworkBudget(clock=clock)
    backend_provenance = _backend_provenance(
        config,
        transport=transport,
        blob_fetcher=blob_fetcher,
    )
    cache_origin = cache_origin_contract(backend_provenance)
    client = DatasetServerClient(
        api_root=config.api_root,
        expected_revision=config.expected_revision,
        timeout_seconds=HTTP_TIMEOUT_SECONDS,
        transport=BudgetedTransport(transport, budget),
        max_attempts=MAX_ATTEMPTS_PER_CALL,
        backoff_seconds=0.5,
        max_backoff_seconds=4.0,
        sleeper=sleeper,
    )
    total_rows = client.total_rows(
        dataset=config.dataset,
        config=config.dataset_config,
        split=config.split,
    )
    windows = deterministic_windows(
        total_rows=total_rows,
        sample_rows=METADATA_ROWS,
        window_size=WINDOW_SIZE,
        seed=SEED,
    )
    if len(windows) != WINDOW_COUNT or any(window.length != WINDOW_SIZE for window in windows):
        raise AssertionError("fixed P1 window contract drifted")

    raw_rows: list[dict[str, Any]] = []
    features: Any = None
    raw_fields: set[str] | None = None
    for window in windows:
        batch, batch_features = client.rows(
            dataset=config.dataset,
            config=config.dataset_config,
            split=config.split,
            offset=window.offset,
            length=window.length,
        )
        fields = {field for field in batch[0] if field != "row_idx"}
        if raw_fields is None:
            raw_fields = fields
            features = batch_features
        elif fields != raw_fields or batch_features != features:
            raise CollectionError("upstream schema drifted between P1 windows")
        if any({field for field in row if field != "row_idx"} != raw_fields for row in batch):
            raise CollectionError("upstream schema drifted within a P1 window")
        raw_rows.extend(batch)
    if client._request_ordinal != EXPECTED_HF_LOGICAL_CALLS:
        raise CollectionError(
            f"expected {EXPECTED_HF_LOGICAL_CALLS} HF logical calls, got {client._request_ordinal}"
        )

    adapter = get_adapter(config.adapter)
    assert raw_fields is not None
    field_map = adapter.resolve_schema(raw_fields)
    rows = [adapter.normalize(row, field_map=field_map) for row in raw_rows]
    selected, selection = select_p1_rows(rows)
    if blob_fetcher is None:
        blob_fetcher = make_bounded_swh_fetcher()

    def worker(row: Mapping[str, Any]) -> dict[str, Any]:
        assert blob_fetcher is not None
        return _fetch_selected(
            row,
            config=config,
            fetcher=blob_fetcher,
            budget=budget,
            expected_origin=cache_origin,
        )

    with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="python-p1") as pool:
        entries = list(pool.map(worker, selected))
    if len(entries) != SELECTED_BLOBS:
        raise CollectionError("content collection underfilled; backfill is forbidden")
    total_raw = sum(int(entry.get("raw_bytes", 0)) for entry in entries)
    if total_raw > MAX_TOTAL_RAW_BYTES:
        raise CollectionError(
            f"selected raw total exceeds {MAX_TOTAL_RAW_BYTES} bytes: {total_raw}"
        )
    budget.check()
    storage_before_publish = _check_storage_bounds(config)
    immutable_entries = [
        {key: value for key, value in entry.items() if key != "cache_hit"} for entry in entries
    ]
    hf_evidence = _normalized_hf_evidence(client)
    backend_accounting = _backend_accounting(hf_evidence, entries, budget)
    schema_evidence = {
        "upstream_features": features,
        "upstream_fields": sorted(raw_fields),
        "normalized_field_map": field_map,
    }
    schema_evidence["fingerprint_sha256"] = hashlib.sha256(
        _canonical_json_bytes(schema_evidence)
    ).hexdigest()

    manifest = {
        "schema_version": _MANIFEST_VERSION,
        "kind": _MANIFEST_KIND,
        "decision_scope": DECISION_SCOPE,
        "policy_binding": dict(policy_binding),
        "contract": _manifest_contract(),
        "input": _expected_input(config),
        "schema": schema_evidence,
        "sampling": {
            "total_rows": total_rows,
            "windows": [asdict(window) for window in windows],
        },
        "metadata_rows": rows,
        "selection": selection,
        "selected_blobs": immutable_entries,
        "backend_provenance": backend_provenance,
        "cache_origin_contract": cache_origin,
        "backend_accounting": backend_accounting,
        "hf_evidence": hf_evidence,
    }
    report = {
        "schema_version": 1,
        "kind": "petitgpt_python_p1_report",
        "status": "complete",
        "source_code_exposed": False,
        "decision_scope": DECISION_SCOPE,
        "policy_binding": dict(policy_binding),
        "input": manifest["input"],
        "contract": manifest["contract"],
        "backend_provenance": backend_provenance,
        "cache_origin_contract": cache_origin,
        "backend_accounting": backend_accounting,
        "metadata": _metadata_summary(rows),
        "selection": selection,
        "content": _content_summary(entries),
        "resources": {
            "workers": config.workers,
            "hf_logical_calls": client._request_ordinal,
            "swh_selected_objects": len(entries),
            "total_retries": budget.retries,
            "elapsed_nanoseconds": budget.elapsed_nanoseconds,
            "storage_before_publish": storage_before_publish,
        },
    }
    return manifest, report, budget


def _verify_addressed_manifest(path: Path) -> dict[str, Any]:
    match = _ADDRESS_RE.search(path.name)
    if match is None:
        raise CollectionError("replay manifest filename is not content-addressed")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != match.group(1):
        raise CollectionError(
            f"manifest SHA tamper/corruption: filename={match.group(1)}, actual={actual}"
        )
    try:
        manifest = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CollectionError("replay manifest is invalid JSON") from exc
    if manifest.get("kind") != _MANIFEST_KIND or manifest.get("schema_version") != 1:
        raise CollectionError("unsupported replay manifest kind/version")
    return manifest


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CollectionError(f"replay manifest {label} must be an integer")
    if minimum is not None and value < minimum:
        raise CollectionError(f"replay manifest {label} is below {minimum}")
    if maximum is not None and value > maximum:
        raise CollectionError(f"replay manifest {label} exceeds {maximum}")
    return value


def _safe_replay_cache_path(config: P1Config, entry: Mapping[str, Any]) -> Path:
    raw_sha256 = entry.get("raw_sha256")
    cache_object = entry.get("cache_object")
    if not isinstance(raw_sha256, str) or _SHA256_RE.fullmatch(raw_sha256) is None:
        raise CollectionError("malformed raw SHA-256 in replay manifest")
    if not isinstance(cache_object, str):
        raise CollectionError("malformed cache_object in replay manifest")
    candidate = Path(cache_object)
    expected_name = f"raw-sha256-{raw_sha256}.raw"
    if (
        candidate.is_absolute()
        or cache_object in {"", ".", ".."}
        or "/" in cache_object
        or "\\" in cache_object
        or any(part == ".." for part in candidate.parts)
        or candidate.name != expected_name
    ):
        raise CollectionError("unsafe or noncanonical cache_object in replay manifest")
    return config.cache_dir / expected_name


def _validate_backend_provenance_evidence(
    value: Any,
    *,
    config: P1Config,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollectionError("replay manifest has no backend provenance")
    hf = value.get("hf")
    swh = value.get("swh")
    if not isinstance(hf, dict) or not isinstance(swh, dict):
        raise CollectionError("replay manifest backend provenance is malformed")
    hf_transport = hf.get("transport")
    swh_fetcher = swh.get("fetcher")
    if hf_transport not in {"requests_json_transport", "injected_callable"}:
        raise CollectionError("unknown HF transport provenance")
    if swh_fetcher not in {"boto3_unsigned_s3", "injected_callable"}:
        raise CollectionError("unknown SWH fetcher provenance")
    hf_production = hf_transport == "requests_json_transport"
    swh_production = swh_fetcher == "boto3_unsigned_s3"
    api_root = config.api_root.rstrip("/")
    canonical_api_root = api_root == DEFAULT_HF_API_ROOT
    test_only = not hf_production or not swh_production
    production = canonical_api_root and hf_production and swh_production
    mode = (
        "test_only_injected"
        if test_only
        else "production"
        if production
        else "noncanonical_api_root"
    )
    expected = {
        "schema_version": 1,
        "production": production,
        "test_only": test_only,
        "mode": mode,
        "hf": {
            "api_root": api_root,
            "api_root_is_canonical": canonical_api_root,
            "transport": hf_transport,
            "transport_mode": "production" if hf_production else "injected_test_only",
        },
        "swh": {
            "bucket": SWH_BUCKET if swh_production else None,
            "key_template": SWH_KEY_TEMPLATE if swh_production else None,
            "region": SWH_REGION if swh_production else None,
            "auth": SWH_AUTH if swh_production else None,
            "fetcher": swh_fetcher,
            "fetcher_mode": "production" if swh_production else "injected_test_only",
        },
    }
    if value != expected:
        raise CollectionError("replay manifest backend provenance is inconsistent")
    return expected


def _validate_schema_evidence(value: Any, *, config: P1Config) -> None:
    if not isinstance(value, dict) or set(value) != {
        "upstream_features",
        "upstream_fields",
        "normalized_field_map",
        "fingerprint_sha256",
    }:
        raise CollectionError("replay manifest schema evidence is malformed")
    fields = value["upstream_fields"]
    field_map = value["normalized_field_map"]
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
        or fields != sorted(set(fields))
        or "row_idx" in fields
        or not isinstance(field_map, dict)
    ):
        raise CollectionError("replay manifest upstream schema is malformed")
    try:
        expected_map = get_adapter(config.adapter).resolve_schema(set(fields))
    except Exception as exc:
        raise CollectionError("replay manifest schema violates adapter requirements") from exc
    if field_map != expected_map:
        raise CollectionError("replay manifest normalized field map drifted")
    fingerprint = value["fingerprint_sha256"]
    base = {
        "upstream_features": value["upstream_features"],
        "upstream_fields": fields,
        "normalized_field_map": field_map,
    }
    expected_fingerprint = hashlib.sha256(_canonical_json_bytes(base)).hexdigest()
    if fingerprint != expected_fingerprint:
        raise CollectionError("replay manifest schema fingerprint mismatch")


def _expected_hf_calls(
    config: P1Config,
    windows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = [
        ("size", {"dataset": config.dataset}),
    ]
    calls.extend(
        (
            "rows",
            {
                "dataset": config.dataset,
                "config": config.dataset_config,
                "split": config.split,
                "offset": window["offset"],
                "length": window["length"],
            },
        )
        for window in windows
    )
    return calls


def _validate_hf_evidence(
    value: Any,
    *,
    config: P1Config,
    windows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"logical_calls", "responses", "attempts"}:
        raise CollectionError("replay manifest HF evidence is malformed")
    if (
        _strict_int(value["logical_calls"], "HF logical_calls", minimum=0)
        != EXPECTED_HF_LOGICAL_CALLS
    ):
        raise CollectionError("replay manifest HF logical call count drifted")
    responses = value["responses"]
    attempts = value["attempts"]
    expected_calls = _expected_hf_calls(config, windows)
    if not isinstance(responses, list) or len(responses) != EXPECTED_HF_LOGICAL_CALLS:
        raise CollectionError("replay manifest must contain 51 HF responses")
    if (
        not isinstance(attempts, list)
        or len(attempts) < EXPECTED_HF_LOGICAL_CALLS
        or len(attempts) > EXPECTED_HF_LOGICAL_CALLS * MAX_ATTEMPTS_PER_CALL
    ):
        raise CollectionError("replay manifest HF attempt count is invalid")
    evidence_fields = {
        "request_ordinal",
        "endpoint",
        "parameters",
        "attempt",
        "max_attempts",
        "outcome",
        "transient",
        "retry_scheduled",
        "backoff_before_next_attempt_nanoseconds",
        "http_status",
        "body_bytes",
        "body_sha256",
        "request_latency_nanoseconds",
        "x_revision",
        "revision_verified",
        "exception_type",
        "error",
    }
    cursor = 0
    attempt_latencies_nanoseconds: list[int] = []
    for ordinal, (response, (endpoint, parameters)) in enumerate(
        zip(responses, expected_calls, strict=True),
        start=1,
    ):
        if not isinstance(response, dict) or set(response) != evidence_fields:
            raise CollectionError("replay manifest HF response evidence is malformed")
        final_attempt = _strict_int(
            response["attempt"],
            f"HF response {ordinal} attempt",
            minimum=1,
            maximum=MAX_ATTEMPTS_PER_CALL,
        )
        if (
            response["request_ordinal"] != ordinal
            or response["endpoint"] != endpoint
            or response["parameters"] != parameters
            or response["max_attempts"] != MAX_ATTEMPTS_PER_CALL
            or response["outcome"] != "success"
            or response["transient"] is not False
            or response["retry_scheduled"] is not False
            or response["backoff_before_next_attempt_nanoseconds"] is not None
            or response["http_status"] != 200
            or response["x_revision"] != config.expected_revision
            or response["revision_verified"] is not True
            or response["exception_type"] is not None
            or response["error"] is not None
        ):
            raise CollectionError("replay manifest HF success evidence drifted")
        _strict_int(response["body_bytes"], f"HF response {ordinal} bytes", minimum=0)
        if (
            not isinstance(response["body_sha256"], str)
            or _SHA256_RE.fullmatch(response["body_sha256"]) is None
        ):
            raise CollectionError("replay manifest HF response digest is invalid")
        _strict_int(
            response["request_latency_nanoseconds"],
            f"HF response {ordinal} latency nanoseconds",
            minimum=0,
            maximum=MAX_SINGLE_LATENCY_NANOSECONDS,
        )
        group: list[dict[str, Any]] = []
        while (
            cursor < len(attempts)
            and isinstance(attempts[cursor], dict)
            and attempts[cursor].get("request_ordinal") == ordinal
        ):
            group.append(attempts[cursor])
            cursor += 1
        if len(group) != final_attempt or group[-1] != response:
            raise CollectionError("replay manifest HF attempt sequence drifted")
        for attempt_number, attempt in enumerate(group, start=1):
            if set(attempt) != evidence_fields:
                raise CollectionError("replay manifest HF attempt evidence is malformed")
            if (
                attempt["request_ordinal"] != ordinal
                or attempt["endpoint"] != endpoint
                or attempt["parameters"] != parameters
                or attempt["attempt"] != attempt_number
                or attempt["max_attempts"] != MAX_ATTEMPTS_PER_CALL
            ):
                raise CollectionError("replay manifest HF attempt identity drifted")
            attempt_latency = _strict_int(
                attempt["request_latency_nanoseconds"],
                f"HF attempt {ordinal}.{attempt_number} latency nanoseconds",
                minimum=0,
                maximum=MAX_SINGLE_LATENCY_NANOSECONDS,
            )
            attempt_latencies_nanoseconds.append(attempt_latency)
            retry_expected = attempt_number < final_attempt
            expected_backoff = (
                _latency_nanoseconds(
                    min(0.5 * (2 ** (attempt_number - 1)), 4.0),
                    label="expected HF retry backoff",
                )
                if retry_expected
                else None
            )
            if (
                attempt["backoff_before_next_attempt_nanoseconds"] != expected_backoff
                or retry_expected
                and (
                    attempt["transient"] is not True
                    or attempt["retry_scheduled"] is not True
                    or attempt["outcome"]
                    not in {"transient_http_status", "transient_network_exception"}
                )
            ):
                raise CollectionError("replay manifest HF retry evidence is invalid")
    if cursor != len(attempts):
        raise CollectionError("replay manifest contains orphan HF attempts")
    return {
        "logical_calls": EXPECTED_HF_LOGICAL_CALLS,
        "attempts": len(attempts),
        "retries": len(attempts) - EXPECTED_HF_LOGICAL_CALLS,
        "total_latency_nanoseconds": _sum_latency_nanoseconds(
            attempt_latencies_nanoseconds,
            label="replay HF total",
        ),
        "final_outcomes": {"success": EXPECTED_HF_LOGICAL_CALLS},
    }


def _validate_normalized_metadata(row: Any, *, config: P1Config) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise CollectionError("replay manifest metadata row is not an object")
    adapter = get_adapter(config.adapter)
    allowed = {"row_idx", *adapter.aliases}
    if set(row).difference(allowed):
        raise CollectionError("replay manifest metadata contains an unnormalized field")
    field_map = {field: field for field in row if field != "row_idx"}
    try:
        normalized = adapter.normalize(row, field_map=field_map)
    except Exception as exc:
        raise CollectionError("replay manifest metadata violates adapter types") from exc
    if normalized != row:
        raise CollectionError("replay manifest metadata is not canonically normalized")
    return row


def _validate_source_error_evidence(value: Any, *, syntax: bool) -> None:
    if not isinstance(value, dict):
        raise CollectionError("replay manifest source error evidence is malformed")
    if syntax:
        if set(value) != {"category", "line", "offset", "end_line", "end_offset"}:
            raise CollectionError("replay manifest syntax evidence has unexpected fields")
        if value["category"] not in {"SyntaxError", "IndentationError", "TabError", "ValueError"}:
            raise CollectionError("replay manifest syntax category is invalid")
        for field in ("line", "end_line"):
            if value[field] is not None:
                _strict_int(value[field], f"syntax {field}", minimum=1)
        for field in ("offset", "end_offset"):
            if value[field] is None:
                continue
            offset = _strict_int(value[field], f"syntax {field}", minimum=-1)
            if offset == 0:
                raise CollectionError(
                    f"replay manifest syntax {field} must be -1 or a positive integer"
                )
    else:
        if set(value) != {"category", "start", "end"}:
            raise CollectionError("replay manifest decode evidence has unexpected fields")
        if value["category"] != "UnicodeDecodeError":
            raise CollectionError("replay manifest decode category is invalid")
        start = _strict_int(value["start"], "decode start", minimum=0)
        end = _strict_int(value["end"], "decode end", minimum=0)
        if end < start:
            raise CollectionError("replay manifest decode positions are reversed")


def _validate_quality_evidence(value: Any, *, decode_outcome: str, raw_bytes: int) -> None:
    if not isinstance(value, dict):
        raise CollectionError("replay manifest quality evidence is malformed")
    if decode_outcome == "decode_failed":
        if set(value) != {"decode_error", "raw_bytes"} or value["raw_bytes"] != raw_bytes:
            raise CollectionError("replay manifest decode-failure evidence drifted")
        _validate_source_error_evidence(value["decode_error"], syntax=False)
        return
    base_fields = {
        "raw_bytes",
        "characters",
        "lines",
        "ast_parse_ok",
        "has_docstring",
        "has_comment",
        "vendor_path_heuristic",
        "generated_file_heuristic",
        "repeated_line_ratio",
        "repetition_heuristic",
    }
    if not base_fields.issubset(value) or set(value).difference(base_fields | {"ast_error"}):
        raise CollectionError("replay manifest decoded quality fields drifted")
    if value["raw_bytes"] != raw_bytes:
        raise CollectionError("replay manifest decoded raw byte count drifted")
    for field in (
        "ast_parse_ok",
        "has_docstring",
        "has_comment",
        "vendor_path_heuristic",
        "generated_file_heuristic",
        "repetition_heuristic",
    ):
        if not isinstance(value[field], bool):
            raise CollectionError(f"replay manifest quality {field} must be boolean")
    for field in ("characters", "lines"):
        _strict_int(value[field], f"quality {field}", minimum=0)
    ratio = value["repeated_line_ratio"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or not 0 <= float(ratio) <= 1
    ):
        raise CollectionError("replay manifest repetition ratio is invalid")
    if value["ast_parse_ok"]:
        if "ast_error" in value:
            raise CollectionError("successful AST evidence contains an error")
    else:
        if "ast_error" not in value:
            raise CollectionError("failed AST evidence has no error category")
        _validate_source_error_evidence(value["ast_error"], syntax=True)


def _validate_selected_entry(
    entry: Any,
    *,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise CollectionError("replay manifest selected entry is not an object")
    common_fields = {
        "metadata",
        "selection_rank_sha256",
        "fetch_outcome",
        "error_category",
        "fetch_attempts",
        "fetch_attempt_latencies_nanoseconds",
        "fetch_latency_nanoseconds",
        "fidelity_outcome",
        "decode_outcome",
    }
    if entry.get("metadata") != expected_metadata:
        raise CollectionError("replay manifest stable selection metadata drifted")
    expected_rank = _stable_selection_rank(expected_metadata)
    if entry.get("selection_rank_sha256") != expected_rank:
        raise CollectionError("replay manifest stable selection rank drifted")
    attempts = _strict_int(
        entry.get("fetch_attempts"),
        "SWH fetch_attempts",
        minimum=0,
        maximum=MAX_ATTEMPTS_PER_CALL,
    )
    latencies = entry.get("fetch_attempt_latencies_nanoseconds")
    if not isinstance(latencies, list) or len(latencies) != attempts:
        raise CollectionError("replay manifest SWH attempt latency count drifted")
    expected_total_latency = _sum_latency_nanoseconds(
        latencies,
        label="replay SWH attempt",
    )
    total_latency = _strict_int(
        entry.get("fetch_latency_nanoseconds"),
        "SWH fetch latency nanoseconds",
        minimum=0,
        maximum=MAX_ATTEMPTS_PER_CALL * MAX_SINGLE_LATENCY_NANOSECONDS,
    )
    if total_latency != expected_total_latency:
        raise CollectionError("replay manifest SWH total latency drifted")
    if entry.get("fetch_outcome") == "failed":
        if set(entry) != common_fields:
            raise CollectionError("replay manifest failed-fetch entry fields drifted")
        if attempts < 1 or entry.get("error_category") not in FETCH_FAILURE_CATEGORIES:
            raise CollectionError("replay manifest failed-fetch category is invalid")
        if (
            entry.get("fidelity_outcome") != "not_available"
            or entry.get("decode_outcome") != "not_attempted"
        ):
            raise CollectionError("replay manifest failed-fetch funnel drifted")
        return entry
    if entry.get("fetch_outcome") != "success":
        raise CollectionError("replay manifest fetch outcome is invalid")
    success_fields = common_fields | {
        "raw_sha256",
        "raw_bytes",
        "cache_object",
        "cache_origin_verified",
    }
    if entry.get("fidelity_outcome") == "length_matches_metadata":
        success_fields.add("quality")
    if set(entry) != success_fields or entry.get("error_category") is not None:
        raise CollectionError("replay manifest successful-fetch entry fields drifted")
    if entry.get("cache_origin_verified") is not (attempts == 0):
        raise CollectionError("replay manifest cache-origin verification evidence drifted")
    raw_sha256 = entry.get("raw_sha256")
    if not isinstance(raw_sha256, str) or _SHA256_RE.fullmatch(raw_sha256) is None:
        raise CollectionError("replay manifest raw digest is invalid")
    raw_bytes = _strict_int(
        entry.get("raw_bytes"),
        "raw_bytes",
        minimum=0,
        maximum=MAX_DECOMPRESSED_BYTES,
    )
    fidelity = entry.get("fidelity_outcome")
    if fidelity == "metadata_length_mismatch":
        if entry.get("decode_outcome") != "not_attempted":
            raise CollectionError("replay manifest mismatch entry must not be decoded")
    elif fidelity == "length_matches_metadata":
        outcome = entry.get("decode_outcome")
        if outcome not in {"decoded", "decode_failed"}:
            raise CollectionError("replay manifest decode outcome is invalid")
        _validate_quality_evidence(entry["quality"], decode_outcome=outcome, raw_bytes=raw_bytes)
    else:
        raise CollectionError("replay manifest fidelity outcome is invalid")
    return entry


def _validate_replay_manifest(
    manifest: Mapping[str, Any],
    *,
    config: P1Config,
    policy_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if manifest.get("decision_scope") != DECISION_SCOPE:
        raise CollectionError("replay manifest decision scope drifted")
    recorded_policy = manifest.get("policy_binding")
    if not isinstance(recorded_policy, dict):
        raise CollectionError("replay manifest has no policy binding")
    for field in (
        "sha256",
        "expected_sha256",
        "schema_version",
        "kind",
        "status",
        "decision_scope",
        "arm_tuple",
    ):
        if recorded_policy.get(field) != policy_binding.get(field):
            raise CollectionError(f"offline replay policy {field} mismatch")
    if manifest.get("contract") != _manifest_contract():
        raise CollectionError("offline replay contract drifted")
    if manifest.get("input") != _expected_input(config):
        raise CollectionError("offline replay arguments do not match immutable manifest input")
    backend_provenance = _validate_backend_provenance_evidence(
        manifest.get("backend_provenance"),
        config=config,
    )
    expected_cache_origin = cache_origin_contract(backend_provenance)
    if manifest.get("cache_origin_contract") != expected_cache_origin:
        raise CollectionError("replay manifest cache origin contract drifted")
    _validate_schema_evidence(manifest.get("schema"), config=config)
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"total_rows", "windows"}:
        raise CollectionError("replay manifest sampling evidence is malformed")
    total_rows = _strict_int(sampling["total_rows"], "sampling total_rows", minimum=METADATA_ROWS)
    try:
        expected_windows = [
            asdict(window)
            for window in deterministic_windows(
                total_rows=total_rows,
                sample_rows=METADATA_ROWS,
                window_size=WINDOW_SIZE,
                seed=SEED,
            )
        ]
    except ValueError as exc:
        raise CollectionError("replay manifest sampling total cannot satisfy contract") from exc
    windows = sampling["windows"]
    if windows != expected_windows or len(expected_windows) != WINDOW_COUNT:
        raise CollectionError("replay manifest deterministic windows drifted")
    metadata_rows = manifest.get("metadata_rows")
    if not isinstance(metadata_rows, list) or len(metadata_rows) != METADATA_ROWS:
        raise CollectionError("replay manifest must contain exactly 500 metadata rows")
    expected_indices = [
        index
        for window in expected_windows
        for index in range(window["offset"], window["offset"] + window["length"])
    ]
    validated_rows: list[dict[str, Any]] = []
    for expected_index, row in zip(expected_indices, metadata_rows, strict=True):
        normalized = _validate_normalized_metadata(row, config=config)
        if normalized["row_idx"] != expected_index:
            raise CollectionError("replay manifest metadata row index drifted from windows")
        validated_rows.append(normalized)
    try:
        selected_rows, expected_selection = select_p1_rows(validated_rows)
    except Exception as exc:
        raise CollectionError("replay manifest stable selection cannot be reproduced") from exc
    if manifest.get("selection") != expected_selection:
        raise CollectionError("replay manifest selection accounting drifted")
    entries = manifest.get("selected_blobs")
    if not isinstance(entries, list) or len(entries) != SELECTED_BLOBS:
        raise CollectionError("replay manifest does not contain exactly 300 selected blobs")
    validated_entries = [
        _validate_selected_entry(entry, expected_metadata=selected)
        for entry, selected in zip(entries, selected_rows, strict=True)
    ]
    blob_ids = [entry["metadata"]["blob_id"] for entry in validated_entries]
    if len(set(blob_ids)) != SELECTED_BLOBS:
        raise CollectionError("replay manifest selected blob IDs are not distinct")
    hf_accounting = _validate_hf_evidence(
        manifest.get("hf_evidence"),
        config=config,
        windows=expected_windows,
    )
    swh_attempts = sum(entry["fetch_attempts"] for entry in validated_entries)
    swh_retries = sum(max(0, entry["fetch_attempts"] - 1) for entry in validated_entries)
    swh_accounting = {
        "selected_objects": SELECTED_BLOBS,
        "network_objects": sum(entry["fetch_attempts"] > 0 for entry in validated_entries),
        "cache_hits": sum(
            entry["fetch_outcome"] == "success" and entry["fetch_attempts"] == 0
            for entry in validated_entries
        ),
        "cache_origin_verified": all(
            entry["cache_origin_verified"] is True
            for entry in validated_entries
            if entry["fetch_outcome"] == "success" and entry["fetch_attempts"] == 0
        ),
        "attempts": swh_attempts,
        "retries": swh_retries,
        "total_latency_nanoseconds": _sum_latency_nanoseconds(
            [entry["fetch_latency_nanoseconds"] for entry in validated_entries],
            label="replay SWH total",
        ),
        "final_error_counts": dict(
            sorted(
                Counter(
                    entry["error_category"]
                    for entry in validated_entries
                    if entry["error_category"] is not None
                ).items()
            )
        ),
    }
    expected_accounting = {
        "hf": hf_accounting,
        "swh": swh_accounting,
        "total_retries": hf_accounting["retries"] + swh_retries,
    }
    if expected_accounting["total_retries"] > MAX_TOTAL_RETRIES:
        raise CollectionError("replay manifest exceeds the global retry budget")
    if manifest.get("backend_accounting") != expected_accounting:
        raise CollectionError("replay manifest backend accounting drifted")
    if sum(int(entry.get("raw_bytes", 0)) for entry in validated_entries) > MAX_TOTAL_RAW_BYTES:
        raise CollectionError("replay manifest exceeds the total raw-byte budget")
    return validated_entries


def _offline_replay(
    config: P1Config,
    *,
    policy_binding: Mapping[str, Any],
    clock: Callable[[], float],
) -> dict[str, Any]:
    assert config.replay_manifest is not None
    budget = NetworkBudget(clock=clock)
    manifest = _verify_addressed_manifest(config.replay_manifest)
    entries = _validate_replay_manifest(manifest, config=config, policy_binding=policy_binding)
    cache_origin = _validate_cache_origin_contract(manifest.get("cache_origin_contract"))
    replayed: list[dict[str, Any]] = []
    verified_blobs = 0
    for entry in entries:
        metadata = entry["metadata"]
        if entry.get("fetch_outcome") != "success":
            replayed.append({**entry, "cache_hit": False})
            budget.check()
            continue
        raw_path = _safe_replay_cache_path(config, entry)
        cached = _load_cache_entry(
            config.cache_dir,
            str(metadata["blob_id"]),
            expected_origin=cache_origin,
        )
        if cached is None:
            raise CollectionError(f"missing cache index for replay blob {metadata['blob_id']}")
        raw, cached_sha256, cached_object = cached
        if (
            cached_sha256 != entry["raw_sha256"]
            or cached_object != entry["cache_object"]
            or config.cache_dir / cached_object != raw_path
        ):
            raise CollectionError(
                f"cache index/manifest mismatch for replay blob {metadata['blob_id']}"
            )
        verified_blobs += 1
        if entry.get("fidelity_outcome") == "metadata_length_mismatch":
            if len(raw) == int(metadata["length_bytes"]):
                raise CollectionError(
                    f"offline fidelity replay drift for blob {metadata['blob_id']}"
                )
        elif entry.get("fidelity_outcome") == "length_matches_metadata":
            if len(raw) != int(metadata["length_bytes"]):
                raise CollectionError(
                    f"offline fidelity replay drift for blob {metadata['blob_id']}"
                )
            outcome, quality = _source_quality(str(metadata.get("path", "")), raw)
            if outcome != entry.get("decode_outcome") or quality != entry.get("quality"):
                raise CollectionError(
                    f"offline quality replay drift for blob {metadata['blob_id']}"
                )
        else:
            raise CollectionError("unknown fidelity outcome in replay manifest")
        replayed.append({**entry, "cache_hit": True})
        budget.check()
    return {
        "schema_version": 1,
        "kind": "petitgpt_python_p1_replay_report",
        "status": "complete",
        "network_access": False,
        "source_code_exposed": False,
        "decision_scope": DECISION_SCOPE,
        "policy_binding": dict(policy_binding),
        "collection_backend_provenance": manifest["backend_provenance"],
        "cache_origin_contract": cache_origin,
        "cache_origin_verified": verified_blobs
        == sum(entry.get("fetch_outcome") == "success" for entry in replayed),
        "collection_backend_accounting": manifest["backend_accounting"],
        "input_manifest": str(config.replay_manifest),
        "input_manifest_sha256": hashlib.sha256(config.replay_manifest.read_bytes()).hexdigest(),
        "selected_records": len(replayed),
        "verified_blobs": verified_blobs,
        "recorded_fetch_failures": len(replayed) - verified_blobs,
        "content": _content_summary(replayed),
        "elapsed_nanoseconds": budget.elapsed_nanoseconds,
    }


def collect_python_p1(
    config: P1Config,
    *,
    transport: Transport = requests_json_transport,
    blob_fetcher: BlobFetcher | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    validate_p1_config(config)
    policy_binding = load_policy_binding(config)
    _check_storage_bounds(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if config.replay_manifest is not None:
        report = _offline_replay(
            config,
            policy_binding=policy_binding,
            clock=clock,
        )
        report_path = _content_addressed_json(config.output_dir, "replay-report", report)
        _check_storage_bounds(config)
        return {"mode": "offline_replay", "report": str(report_path)}

    manifest, report, _ = _online_collect(
        config,
        policy_binding=policy_binding,
        transport=transport,
        blob_fetcher=blob_fetcher,
        clock=clock,
        sleeper=sleeper,
    )
    # Immutable manifest is durable before the summary that points to it.
    manifest_path = _content_addressed_json(config.output_dir, "collection-manifest", manifest)
    report["manifest"] = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    report_path = _content_addressed_json(config.output_dir, "collection-report", report)
    _check_storage_bounds(config)
    return {
        "mode": "online",
        "manifest": str(manifest_path),
        "report": str(report_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect or offline-replay the fixed 500-row/300-blob Python P1 sample."
    )
    parser.add_argument("--expected_revision", required=True)
    parser.add_argument("--expected_policy_sha256", required=True)
    parser.add_argument(
        "--adapter",
        required=True,
        choices=("smollm_python_edu", "stack_edu_python"),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--policy_json", type=Path, required=True)
    parser.add_argument("--api_root", default=DEFAULT_HF_API_ROOT)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--replay_manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = collect_python_p1(
        P1Config(
            expected_revision=args.expected_revision,
            adapter=args.adapter,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            policy_path=args.policy_json,
            expected_policy_sha256=args.expected_policy_sha256,
            api_root=args.api_root,
            workers=args.workers,
            replay_manifest=args.replay_manifest,
        )
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
