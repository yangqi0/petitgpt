#!/usr/bin/env python3

"""Analyze a frozen Python P1 collection entirely offline.

The analyzer verifies the content-addressed collection manifest, its private
cache objects, and a separately frozen policy before computing aggregate
quality and pre-tokenizer yield evidence.  Reports contain no source text,
source lines, identifiers, literals, repository names, or paths.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.collect_python_p1 import (  # noqa: E402
    CollectionError,
    _load_cache_entry,
    _manifest_contract as _collector_manifest_contract,
    cache_origin_contract,
)
from pretrain.python_quality import (  # noqa: E402
    BINARY_CONTROL_RATIO,
    CHARS_PER_TOKEN_SENSITIVITY,
    MAX_IDENTICAL_LINE_COUNT,
    MAX_REPEATED_LINE_RATIO,
    MAX_SOURCE_BYTES,
    MIN_REPETITION_LINES,
    MIN_SOURCE_BYTES,
    MINIFIED_MAX_NONEMPTY_LINES,
    MINIFIED_MIN_MAX_LINE_LENGTH,
    YIELD_TARGETS,
    analyze_python_source,
    estimate_pretokenizer_yield,
    summarize_hard_gate_funnel,
    wilson_interval,
)

COLLECTION_KIND = "petitgpt_python_p1_collection"
ANALYSIS_KIND = "petitgpt_python_p1_analysis"
POLICY_KIND = "petitgpt_python_p1_matched_source_policy"
DECISION_SCOPE = "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL"
SCHEMA_VERSION = 1
METADATA_ROWS = 500
SELECTED_BLOBS = 300
WINDOWS = 50
ROWS_PER_WINDOW = 10
COLLECTION_SEED = 20_250_814
SELECTION_MAX_BYTES = 100_000
BOOTSTRAP_RESAMPLES = 2_000
MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM = 100
FETCH_FAILURE_CATEGORIES = frozenset({
    "access_denied",
    "decompressed_size_above_100000",
    "gzip_decode_failed",
    "not_found",
    "s3_client_error",
    "s3_transient_retries_exhausted",
    "transport_error_retries_exhausted",
})
FULL_GATE_ORDER = (
    "fetch_success",
    "fidelity_length_matches",
    "strict_utf8",
    "raw_size_200_to_100000_bytes",
    "nonempty_nonwhitespace",
    "not_binary_like",
    "python3_ast_parse",
    "not_strong_generated",
    "not_vendor",
    "not_strong_repetition",
)
ARM_ADAPTERS = {
    "primary": "smollm_python_edu",
    "stack_comparison": "stack_edu_python",
}
ARM_POLICY_NAMES = {
    "primary": "smollm_python_edu_primary",
    "stack_comparison": "stack_edu_python_comparator",
}
_COLLECTION_NAME_RE = re.compile(r"^collection-manifest\.sha256-([0-9a-f]{64})\.json$")
_ANALYSIS_NAME_RE = re.compile(r"^python-p1-analysis\.sha256-([0-9a-f]{64})\.json$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

PRODUCTION_BACKEND_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "production": True,
    "test_only": False,
    "mode": "production",
    "hf": {
        "api_root": "https://datasets-server.huggingface.co",
        "api_root_is_canonical": True,
        "transport": "requests_json_transport",
        "transport_mode": "production",
    },
    "swh": {
        "bucket": "softwareheritage",
        "key_template": "content/{blob_id}",
        "region": "us-west-2",
        "auth": "anonymous_unsigned",
        "fetcher": "boto3_unsigned_s3",
        "fetcher_mode": "production",
    },
}
PRODUCTION_CACHE_ORIGIN_CONTRACT = cache_origin_contract(PRODUCTION_BACKEND_CONTRACT)


class AnalysisError(RuntimeError):
    """A frozen offline-analysis contract was violated."""


@dataclass(frozen=True)
class AnalysisConfig:
    collection_manifest: Path
    policy_path: Path
    policy_sha256: str
    cache_dir: Path
    output_dir: Path
    expected_arm: str
    enforce_ignored_paths: bool = True


def p1_policy_template() -> dict[str, Any]:
    """Return the exact frozen matched-source P1 policy content."""
    return {
        "analysis": {
            "comparison": {
                "common_score_slice": {
                    "minimum_int_score": 4,
                    "minimum_common_score_content_documents_per_arm": (
                        MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM
                    ),
                    "purpose": "matched comparison only; score never affects collection sampling",
                },
                "inconclusive_absolute_rate_difference_below": 0.1,
                "require_matching_analysis_policy_sha256": True,
                "require_matching_content_sample_size": True,
                "require_matching_metadata_sample_size": True,
                "require_matching_seed": True,
            },
            "duplicate_identities": [
                "blob_id",
                "raw_sha256",
                "ast_structural_sha256_when_parseable",
            ],
            "hard_gate_funnel_order": list(FULL_GATE_ORDER),
            "path_categories_are_signals_not_automatic_rejects": [
                "test",
                "config",
                "lock",
                "notebook",
                "minified",
                "boilerplate",
            ],
            "pretokenizer_yield": {
                "canonical_token_count_available": False,
                "chars_per_token_sensitivity": list(CHARS_PER_TOKEN_SENSITIVITY),
                "target_token_scenarios": list(YIELD_TARGETS),
            },
            "quality_signals": {
                "binary_like": {
                    "disallowed_c0_control_ratio_greater_than": BINARY_CONTROL_RATIO,
                    "nul_byte_is_binary": True,
                },
                "minified": {
                    "maximum_nonempty_lines": MINIFIED_MAX_NONEMPTY_LINES,
                    "minimum_max_line_length": MINIFIED_MIN_MAX_LINE_LENGTH,
                },
                "repetition": {
                    "minimum_nonempty_lines": MIN_REPETITION_LINES,
                    "or_max_identical_line_count_at_least": MAX_IDENTICAL_LINE_COUNT,
                    "repeated_line_excess_ratio_at_least": MAX_REPEATED_LINE_RATIO,
                },
                "size": {
                    "maximum_raw_bytes": MAX_SOURCE_BYTES,
                    "minimum_raw_bytes": MIN_SOURCE_BYTES,
                },
            },
            "report_confidence": {
                "confidence_level": 0.95,
                "method": "deterministic bootstrap for yield plus Wilson intervals for rates",
            },
        },
        "arms": [
            {
                "dataset": "HuggingFaceTB/smollm-corpus",
                "dataset_config": "python-edu",
                "expected_revision": "3ba9d605774198c5868892d7a8deda78031a781f",
                "name": "smollm_python_edu_primary",
                "split": "train",
            },
            {
                "dataset": "HuggingFaceTB/stack-edu",
                "dataset_config": "Python",
                "expected_revision": "eeec5caac5cc3758a18f1d3ba4416837a9ba814c",
                "name": "stack_edu_python_comparator",
                "split": "train",
            },
        ],
        "collection": {
            "content_blobs_per_arm": SELECTED_BLOBS,
            "content_selection": {
                "distinct_identity": "blob_id",
                "eligible_maximum_metadata_length_bytes": SELECTION_MAX_BYTES,
                "maximum_per_repository": None,
                "minimum_int_score": None,
                "minimum_metadata_length_bytes": None,
                "ordering": "sha256(seed, blob_id) ascending",
                "oversize_rows_remain_in_metadata_accounting": True,
            },
            "metadata_rows_per_arm": METADATA_ROWS,
            "metadata_sampling": {
                "strategy": "seeded_contiguous_window_within_equal_row_index_strata_v1",
                "window_count": WINDOWS,
                "window_size": ROWS_PER_WINDOW,
            },
            "seed": COLLECTION_SEED,
        },
        "decision_scope": "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL",
        "kind": POLICY_KIND,
        "privacy": {
            "cache_and_reports_git_policy": "DO_NOT_COMMIT",
            "report_source_characters_exposed": 0,
            "stable_manual_review_identifiers_per_outcome_per_arm": 12,
        },
        "resource_budget_per_arm": {
            "datasets_server_logical_calls": 51,
            "decompressed_object_maximum_bytes": 100_000,
            "gpu_enabled": False,
            "http_max_attempts_per_call": 4,
            "http_timeout_seconds": 30,
            "maximum_cache_and_output_bytes": 134_217_728,
            "maximum_concurrent_blob_fetches": 4,
            "maximum_total_raw_blob_bytes": 33_554_432,
            "maximum_total_retries": 50,
            "maximum_wall_seconds": 1_800,
            "swh_blob_selections": 300,
        },
        "schema_version": 1,
        "status": "FROZEN_BEFORE_COLLECTION",
    }


def _strict_json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant!r}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AnalysisError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be a JSON object")
    return value


def _read_regular_file(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"cannot read {label}: {path}") from exc


def _load_policy(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise AnalysisError("policy_sha256 must be an exact lowercase SHA-256")
    data = _read_regular_file(path, label="policy")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise AnalysisError(f"policy SHA mismatch: expected {expected_sha256}, actual {actual}")
    policy = _strict_json(data, label="policy")
    if policy != p1_policy_template():
        raise AnalysisError("policy content does not match the frozen P1 v1 contract")
    return policy, actual


def _load_collection(path: Path) -> tuple[dict[str, Any], str]:
    match = _COLLECTION_NAME_RE.fullmatch(path.name)
    if match is None:
        raise AnalysisError("collection manifest filename is not content-addressed")
    data = _read_regular_file(path, label="collection manifest")
    actual = hashlib.sha256(data).hexdigest()
    if actual != match.group(1):
        raise AnalysisError(
            f"collection manifest SHA mismatch: filename={match.group(1)}, actual={actual}"
        )
    manifest = _strict_json(data, label="collection manifest")
    if manifest.get("kind") != COLLECTION_KIND or manifest.get("schema_version") != 1:
        raise AnalysisError("unsupported collection manifest kind/version")
    return manifest, actual


def _publish_addressed_json(output_dir: Path, stem: str, value: Mapping[str, Any]) -> Path:
    data = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(data).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.sha256-{digest}.json"
    if path.exists():
        if path.read_bytes() != data:
            raise AnalysisError(f"content-address collision at {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_analysis_report(path: Path) -> tuple[dict[str, Any], str]:
    """Load and re-hash one content-addressed analyzer report."""
    match = _ANALYSIS_NAME_RE.fullmatch(path.name)
    if match is None:
        raise AnalysisError("analysis report filename is not content-addressed")
    data = _read_regular_file(path, label="analysis report")
    actual = hashlib.sha256(data).hexdigest()
    if actual != match.group(1):
        raise AnalysisError(
            f"analysis report SHA mismatch: filename={match.group(1)}, actual={actual}"
        )
    report = _strict_json(data, label="analysis report")
    if report.get("kind") != ANALYSIS_KIND or report.get("schema_version") != 1:
        raise AnalysisError("unsupported analysis report kind/version")
    return report, actual


def _require_git_ignored(path: Path, *, label: str) -> None:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise AnalysisError(f"{label} must be inside the Git worktree: {path}") from exc
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise AnalysisError(f"{label} is not Git-ignored: {path}")


def _validate_private_paths(config: AnalysisConfig) -> None:
    if config.expected_arm not in ARM_ADAPTERS:
        raise AnalysisError(f"unknown expected arm {config.expected_arm!r}")
    if config.cache_dir.resolve() == config.output_dir.resolve():
        raise AnalysisError("private cache and analysis output directory must differ")
    if not config.cache_dir.is_dir():
        raise AnalysisError(f"private cache directory does not exist: {config.cache_dir}")
    if config.enforce_ignored_paths:
        _require_git_ignored(config.cache_dir, label="private cache")
        _require_git_ignored(config.output_dir, label="analysis output directory")


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}")
    return value


def _score(row: Mapping[str, Any]) -> float | None:
    value = row.get("int_score", row.get("score"))
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("metadata score is not numeric") from exc
    if not math.isfinite(result):
        raise AnalysisError("metadata score is non-finite")
    return result


def _validate_entry_latency_evidence(
    entry: Mapping[str, Any], *, ordinal: int, fetch_attempts: int
) -> None:
    if any(
        field in entry for field in ("fetch_attempt_latencies_seconds", "fetch_latency_seconds")
    ):
        raise AnalysisError(f"selected blob {ordinal} contains legacy seconds timing evidence")
    latencies = entry.get("fetch_attempt_latencies_nanoseconds")
    if not isinstance(latencies, list) or len(latencies) != fetch_attempts:
        raise AnalysisError(
            f"selected blob {ordinal} has malformed integer-nanosecond attempt timing evidence"
        )
    validated = [
        _require_int(
            latency,
            label=f"selected[{ordinal}].fetch_attempt_latencies_nanoseconds",
            minimum=0,
        )
        for latency in latencies
    ]
    total = _require_int(
        entry.get("fetch_latency_nanoseconds"),
        label=f"selected[{ordinal}].fetch_latency_nanoseconds",
        minimum=0,
    )
    if total != sum(validated):
        raise AnalysisError(f"selected blob {ordinal} timing evidence does not sum exactly")


def _deterministic_windows(*, total_rows: int) -> list[dict[str, int]]:
    if total_rows < METADATA_ROWS:
        raise AnalysisError("sampling total cannot satisfy the fixed 500-row contract")
    window_count = math.ceil(METADATA_ROWS / ROWS_PER_WINDOW)
    lengths = [ROWS_PER_WINDOW] * window_count
    lengths[-1] = METADATA_ROWS - ROWS_PER_WINDOW * (window_count - 1)
    windows: list[dict[str, int]] = []
    for stratum, length in enumerate(lengths):
        lower = total_rows * stratum // window_count
        upper = total_rows * (stratum + 1) // window_count
        if upper - lower < length:
            raise AnalysisError("a deterministic sampling stratum is too small")
        available_offsets = upper - lower - length + 1
        rank = hashlib.sha256(f"{COLLECTION_SEED}\0{stratum}".encode()).digest()
        jitter = int.from_bytes(rank[:8], "big") % available_offsets
        windows.append({
            "stratum": stratum,
            "offset": lower + jitter,
            "length": length,
        })
    return windows


def _validate_collection_contract(
    manifest: Mapping[str, Any],
    *,
    expected_arm: str,
    policy: Mapping[str, Any],
    policy_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if manifest.get("decision_scope") != DECISION_SCOPE:
        raise AnalysisError("collection manifest decision scope drifted")
    if manifest.get("backend_provenance") != PRODUCTION_BACKEND_CONTRACT:
        raise AnalysisError(
            "collection manifest must come from the exact production backend; "
            "test fixtures, injected callables, and noncanonical API roots are refused"
        )
    if manifest.get("cache_origin_contract") != PRODUCTION_CACHE_ORIGIN_CONTRACT:
        raise AnalysisError(
            "collection manifest must bind cached objects to the exact production SWH origin"
        )
    backend_accounting = manifest.get("backend_accounting")
    if (
        not isinstance(backend_accounting, Mapping)
        or not isinstance(backend_accounting.get("hf"), Mapping)
        or not isinstance(backend_accounting.get("swh"), Mapping)
        or backend_accounting["swh"].get("cache_origin_verified") is not True
    ):
        raise AnalysisError("collection backend accounting lacks verified cache-origin evidence")
    for backend in ("hf", "swh"):
        accounting = backend_accounting[backend]
        if "total_latency_seconds" in accounting:
            raise AnalysisError("collection backend accounting contains legacy seconds timing")
        _require_int(
            accounting.get("total_latency_nanoseconds"),
            label=f"backend_accounting.{backend}.total_latency_nanoseconds",
            minimum=0,
        )
    if manifest.get("contract") != _collector_manifest_contract():
        raise AnalysisError("collection manifest contract drifted from fixed P1")
    collection_policy = policy["collection"]
    if (
        collection_policy["metadata_rows_per_arm"] != METADATA_ROWS
        or collection_policy["content_blobs_per_arm"] != SELECTED_BLOBS
        or collection_policy["seed"] != COLLECTION_SEED
        or collection_policy["content_selection"]["eligible_maximum_metadata_length_bytes"]
        != SELECTION_MAX_BYTES
    ):
        raise AnalysisError("policy and collection sample contracts disagree")
    policy_arms = {arm["name"]: arm for arm in policy["arms"] if isinstance(arm, dict)}
    policy_arm = policy_arms.get(ARM_POLICY_NAMES[expected_arm])
    if policy_arm is None:
        raise AnalysisError(f"frozen policy lacks expected arm {expected_arm!r}")
    expected_input = {
        "adapter": ARM_ADAPTERS[expected_arm],
        "dataset": policy_arm["dataset"],
        "dataset_config": policy_arm["dataset_config"],
        "split": policy_arm["split"],
        "expected_revision": policy_arm["expected_revision"],
    }
    input_spec = manifest.get("input")
    if input_spec != expected_input:
        raise AnalysisError(f"collection input does not match frozen {expected_arm!r} policy arm")
    binding = manifest.get("policy_binding")
    expected_binding = {
        "sha256": policy_sha256,
        "expected_sha256": policy_sha256,
        "schema_version": 1,
        "kind": POLICY_KIND,
        "status": "FROZEN_BEFORE_COLLECTION",
        "decision_scope": DECISION_SCOPE,
        "arm_tuple": [
            ARM_POLICY_NAMES[expected_arm],
            policy_arm["dataset"],
            policy_arm["dataset_config"],
            policy_arm["split"],
            policy_arm["expected_revision"],
        ],
    }
    if not isinstance(binding, dict) or set(binding) != {"path", *expected_binding}:
        raise AnalysisError("collection manifest has malformed policy binding")
    if not isinstance(binding.get("path"), str) or not binding["path"]:
        raise AnalysisError("collection manifest policy binding lacks its source path")
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        raise AnalysisError(
            "collection manifest policy binding does not match frozen policy SHA/arm"
        )

    metadata = manifest.get("metadata_rows")
    selected = manifest.get("selected_blobs")
    if not isinstance(metadata, list) or len(metadata) != METADATA_ROWS:
        raise AnalysisError(f"collection must contain exactly {METADATA_ROWS} metadata rows")
    if not isinstance(selected, list) or len(selected) != SELECTED_BLOBS:
        raise AnalysisError(f"collection must contain exactly {SELECTED_BLOBS} selected blobs")
    if not all(isinstance(row, dict) for row in metadata):
        raise AnalysisError("collection metadata rows must be objects")
    if not all(isinstance(entry, dict) for entry in selected):
        raise AnalysisError("collection selected blob entries must be objects")

    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict) or set(sampling) != {"total_rows", "windows"}:
        raise AnalysisError("collection sampling must be an object")
    total_rows = _require_int(
        sampling.get("total_rows"), label="sampling.total_rows", minimum=METADATA_ROWS
    )
    windows = sampling.get("windows")
    expected_windows = _deterministic_windows(total_rows=total_rows)
    if windows != expected_windows or len(expected_windows) != WINDOWS:
        raise AnalysisError("collection deterministic sampling windows drifted")
    expected_row_indices = [
        row_idx
        for window in expected_windows
        for row_idx in range(window["offset"], window["offset"] + window["length"])
    ]
    observed_row_indices = [
        _require_int(row.get("row_idx"), label="metadata.row_idx", minimum=0) for row in metadata
    ]
    if observed_row_indices != expected_row_indices:
        raise AnalysisError("collection metadata row order drifted from deterministic windows")
    return list(metadata), list(selected), total_rows


def _canonical_eligible_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:
    seen_row_indices: set[int] = set()
    canonical: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        row_idx = _require_int(row.get("row_idx"), label="metadata.row_idx", minimum=0)
        if row_idx in seen_row_indices:
            raise AnalysisError(f"duplicate metadata row_idx {row_idx}")
        seen_row_indices.add(row_idx)
        blob_id = row.get("blob_id")
        if not isinstance(blob_id, str) or not blob_id:
            raise AnalysisError(f"metadata row {row_idx} has invalid blob_id")
        length = _require_int(
            row.get("length_bytes"), label=f"metadata[{row_idx}].length_bytes", minimum=0
        )
        _score(row)
        if length > SELECTION_MAX_BYTES:
            continue
        previous = canonical.get(blob_id)
        if previous is None or row_idx < int(previous["row_idx"]):
            canonical[blob_id] = row
    eligible = sorted(
        canonical.values(),
        key=lambda row: (
            hashlib.sha256(f"{COLLECTION_SEED}\0{row['blob_id']}".encode()).hexdigest(),
            str(row["blob_id"]),
        ),
    )
    return eligible, {int(row["row_idx"]) for row in eligible}


def _cache_object_path(cache_dir: Path, relative: Any, raw_sha256: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AnalysisError("selected cache_object must be a non-empty relative path")
    expected_name = f"raw-sha256-{raw_sha256}.raw"
    if relative != expected_name:
        raise AnalysisError("selected cache_object is not the canonical raw SHA filename")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AnalysisError("selected cache_object escapes the private cache")
    path = cache_dir / relative_path
    try:
        path.resolve().relative_to(cache_dir.resolve())
    except ValueError as exc:
        raise AnalysisError("selected cache_object resolves outside the private cache") from exc
    if path.name != expected_name:
        raise AssertionError("validated cache object name drifted")
    return path


def _verify_and_analyze_selected(
    selected: Sequence[Mapping[str, Any]],
    *,
    eligible: Sequence[Mapping[str, Any]],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    if len(eligible) < SELECTED_BLOBS:
        raise AnalysisError("collection metadata cannot support 300 selected blobs")
    expected = list(eligible[:SELECTED_BLOBS])
    processed: list[dict[str, Any]] = []
    seen_blobs: set[str] = set()
    for ordinal, (entry, expected_row) in enumerate(zip(selected, expected, strict=True)):
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict) or metadata != expected_row:
            raise AnalysisError(
                f"selected blob {ordinal} does not match frozen stable-rank selection"
            )
        blob_id = str(metadata["blob_id"])
        if blob_id in seen_blobs:
            raise AnalysisError(f"duplicate selected blob_id at ordinal {ordinal}")
        seen_blobs.add(blob_id)
        expected_rank = hashlib.sha256(f"{COLLECTION_SEED}\0{blob_id}".encode()).hexdigest()
        if entry.get("selection_rank_sha256") != expected_rank:
            raise AnalysisError(f"selection rank drift for selected blob {ordinal}")
        fetch_outcome = entry.get("fetch_outcome")
        if fetch_outcome == "failed":
            if (
                entry.get("fidelity_outcome") != "not_available"
                or entry.get("decode_outcome") != "not_attempted"
            ):
                raise AnalysisError(f"failed fetch outcome drift for selected blob {ordinal}")
            if any(
                key in entry
                for key in (
                    "raw_sha256",
                    "raw_bytes",
                    "cache_object",
                    "cache_origin_verified",
                    "quality",
                )
            ):
                raise AnalysisError(
                    f"failed fetch exposes raw-object fields at selected blob {ordinal}"
                )
            category = entry.get("error_category")
            if category not in FETCH_FAILURE_CATEGORIES:
                raise AnalysisError(
                    f"failed fetch has a non-allowlisted error category at selected blob {ordinal}"
                )
            fetch_attempts = _require_int(
                entry.get("fetch_attempts"),
                label=f"selected[{ordinal}].fetch_attempts",
                minimum=1,
            )
            _validate_entry_latency_evidence(
                entry,
                ordinal=ordinal,
                fetch_attempts=fetch_attempts,
            )
            processed.append({
                "metadata": metadata,
                "fetch_success": False,
                "fetch_attempts": fetch_attempts,
                "cache_origin_verified": None,
                "fidelity_length_matches": None,
                "failure_category": category,
                "raw_sha256": None,
                "raw_bytes": 0,
                "funnel_bytes": int(metadata["length_bytes"]),
                "strict_utf8": None,
                "analysis": None,
            })
            continue
        if fetch_outcome != "success":
            raise AnalysisError(f"unknown fetch outcome at selected blob {ordinal}")
        fetch_attempts = _require_int(
            entry.get("fetch_attempts"),
            label=f"selected[{ordinal}].fetch_attempts",
            minimum=0,
        )
        _validate_entry_latency_evidence(
            entry,
            ordinal=ordinal,
            fetch_attempts=fetch_attempts,
        )
        cache_origin_verified = entry.get("cache_origin_verified")
        if cache_origin_verified is not (fetch_attempts == 0):
            raise AnalysisError(
                f"cache-origin verification evidence drift for selected blob {ordinal}"
            )
        raw_sha256 = entry.get("raw_sha256")
        if not isinstance(raw_sha256, str) or not _SHA256_RE.fullmatch(raw_sha256):
            raise AnalysisError(f"invalid raw SHA for selected blob {ordinal}")
        raw_bytes = _require_int(
            entry.get("raw_bytes"), label=f"selected[{ordinal}].raw_bytes", minimum=0
        )
        if raw_bytes > SELECTION_MAX_BYTES:
            raise AnalysisError(f"selected blob {ordinal} violates raw-object size contract")
        raw_path = _cache_object_path(cache_dir, entry.get("cache_object"), raw_sha256)
        try:
            indexed = _load_cache_entry(
                cache_dir,
                blob_id,
                expected_origin=PRODUCTION_CACHE_ORIGIN_CONTRACT,
            )
        except CollectionError as exc:
            raise AnalysisError(
                f"production cache index verification failed at selected blob {ordinal}"
            ) from exc
        if indexed is None:
            raise AnalysisError(f"missing production cache index at selected blob {ordinal}")
        indexed_raw, indexed_sha256, indexed_object = indexed
        if indexed_sha256 != raw_sha256 or indexed_object != entry.get("cache_object"):
            raise AnalysisError(f"cache index disagrees with manifest at selected blob {ordinal}")
        raw = _read_regular_file(raw_path, label=f"private raw object {ordinal}")
        if raw != indexed_raw:
            raise AnalysisError(f"cache index raw object drift at selected blob {ordinal}")
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != raw_sha256:
            raise AnalysisError(
                f"private raw object SHA tamper at ordinal {ordinal}: "
                f"expected {raw_sha256}, actual {actual_sha}"
            )
        if len(raw) != raw_bytes:
            raise AnalysisError(
                f"private raw object size drift at ordinal {ordinal}: "
                f"expected {raw_bytes}, actual {len(raw)}"
            )
        fidelity_outcome = (
            "length_matches_metadata"
            if raw_bytes == int(metadata["length_bytes"])
            else "metadata_length_mismatch"
        )
        if entry.get("fidelity_outcome") != fidelity_outcome:
            raise AnalysisError(f"recorded fidelity outcome drift for selected blob {ordinal}")
        if fidelity_outcome == "metadata_length_mismatch":
            decode_outcome = "not_attempted"
            strict_utf8 = None
            analysis = None
        else:
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                decode_outcome = "decode_failed"
                strict_utf8 = False
                analysis = None
            else:
                decode_outcome = "decoded"
                strict_utf8 = True
                analysis = analyze_python_source(raw, text, path=str(metadata.get("path", "")))
                if analysis["content_identity"]["raw_sha256"] != raw_sha256:
                    raise AssertionError("quality analyzer raw identity drifted")
        if entry.get("decode_outcome") != decode_outcome:
            raise AnalysisError(f"recorded strict UTF-8 outcome drift for selected blob {ordinal}")
        processed.append({
            "metadata": metadata,
            "fetch_success": True,
            "fetch_attempts": fetch_attempts,
            "cache_origin_verified": cache_origin_verified,
            "fidelity_length_matches": fidelity_outcome == "length_matches_metadata",
            "failure_category": None,
            "raw_sha256": raw_sha256,
            "raw_bytes": raw_bytes,
            "funnel_bytes": raw_bytes,
            "strict_utf8": strict_utf8,
            "analysis": analysis,
        })
    return processed


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, str) and item.strip() for item in value)
    return value is not None


def _presence_summary(
    rows: Sequence[Mapping[str, Any]], *, normalized_fields: set[str]
) -> dict[str, Any]:
    fields = (
        "blob_id",
        "repo_name",
        "path",
        "license",
        "detected_licenses",
        "license_type",
        "src_encoding",
    )
    count = len(rows)
    result: dict[str, Any] = {}
    for field in fields:
        present = sum(field in row for row in rows)
        nonempty = sum(field in row and _nonempty(row[field]) for row in rows)
        result[field] = {
            "schema_available": field in normalized_fields,
            "classification": (
                "ABSENT_UPSTREAM_NORMALIZED_SCHEMA"
                if field not in normalized_fields
                else "PRESENT_UPSTREAM_NORMALIZED_SCHEMA"
            ),
            "present_rows": present,
            "nonempty_rows": nonempty,
            "empty_or_missing_rows": count - nonempty,
            "nonempty_fraction_wilson_95": wilson_interval(nonempty, count),
        }
    license_any = sum(
        _nonempty(row.get("license")) or _nonempty(row.get("detected_licenses")) for row in rows
    )
    encoding_any = sum(_nonempty(row.get("src_encoding")) for row in rows)
    encoding_utf8 = sum(
        isinstance(row.get("src_encoding"), str)
        and row["src_encoding"].casefold().replace("_", "-") in {"utf-8", "utf8"}
        for row in rows
    )
    core = sum(
        all(_nonempty(row.get(field)) for field in ("blob_id", "repo_name", "path")) for row in rows
    )
    result["derived"] = {
        "license_or_detected_nonempty": {
            "rows": license_any,
            "fraction_wilson_95": wilson_interval(license_any, count),
        },
        "encoding_declared_nonempty": {
            "rows": encoding_any,
            "fraction_wilson_95": wilson_interval(encoding_any, count),
        },
        "encoding_declared_utf8_like": {
            "rows": encoding_utf8,
            "fraction_wilson_95": wilson_interval(encoding_utf8, count),
        },
        "core_blob_repo_path_complete": {
            "rows": core,
            "fraction_wilson_95": wilson_interval(core, count),
        },
    }
    return result


def _duplicate_summary(values: Sequence[str]) -> dict[str, Any]:
    counts = Counter(values)
    groups = [count for count in counts.values() if count > 1]
    return {
        "considered_documents": len(values),
        "unique_values": len(counts),
        "duplicate_groups": len(groups),
        "documents_in_duplicate_groups": sum(groups),
        "duplicate_excess_documents": sum(count - 1 for count in groups),
        "maximum_group_size": max(groups, default=1 if values else 0),
        "duplicate_group_size_counts": dict(
            sorted(Counter(str(count) for count in groups).items())
        ),
        "identifiers_or_hashes_exposed": False,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _concentration(rows: Sequence[Mapping[str, Any]], *, field: str) -> dict[str, Any]:
    document_counts: Counter[str] = Counter()
    byte_counts: Counter[str] = Counter()
    missing_documents = 0
    missing_bytes = 0
    for row in rows:
        value = row.get(field)
        raw_bytes = int(row["_raw_bytes"])
        if not isinstance(value, str) or not value:
            missing_documents += 1
            missing_bytes += raw_bytes
            continue
        document_counts[value] += 1
        byte_counts[value] += raw_bytes
    present_documents = sum(document_counts.values())
    present_bytes = sum(byte_counts.values())

    def top(counter: Counter[str], denominator: int, limit: int) -> dict[str, Any]:
        value = sum(count for _, count in counter.most_common(limit))
        return {"count": value, "fraction": _ratio(value, denominator)}

    document_hhi = (
        sum((count / present_documents) ** 2 for count in document_counts.values())
        if present_documents
        else None
    )
    byte_hhi = (
        sum((count / present_bytes) ** 2 for count in byte_counts.values())
        if present_bytes
        else None
    )
    return {
        "documents": len(rows),
        "raw_bytes": sum(int(row["_raw_bytes"]) for row in rows),
        "present_documents": present_documents,
        "present_raw_bytes": present_bytes,
        "missing_documents": missing_documents,
        "missing_raw_bytes": missing_bytes,
        "unique_values": len(document_counts),
        "top_1_document_share": top(document_counts, present_documents, 1),
        "top_5_document_share": top(document_counts, present_documents, 5),
        "top_10_document_share": top(document_counts, present_documents, 10),
        "top_1_byte_share": top(byte_counts, present_bytes, 1),
        "top_5_byte_share": top(byte_counts, present_bytes, 5),
        "top_10_byte_share": top(byte_counts, present_bytes, 10),
        "document_hhi": document_hhi,
        "byte_hhi": byte_hhi,
        "values_exposed": False,
    }


def _full_funnel(processed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[tuple[int, dict[str, bool | None]]] = []
    for item in processed:
        analysis = item["analysis"]
        gates: dict[str, bool | None] = {
            "fetch_success": bool(item["fetch_success"]),
            "fidelity_length_matches": item["fidelity_length_matches"],
            "strict_utf8": item["strict_utf8"],
            "raw_size_200_to_100000_bytes": (
                None
                if analysis is None
                else bool(analysis["hard_gates"]["size_200_to_100000_bytes"])
            ),
            "nonempty_nonwhitespace": (
                None if analysis is None else bool(analysis["hard_gates"]["nonempty_nonwhitespace"])
            ),
            "not_binary_like": (
                None if analysis is None else bool(analysis["hard_gates"]["not_binary_like"])
            ),
            "python3_ast_parse": (
                None if analysis is None else bool(analysis["hard_gates"]["python3_ast_parse"])
            ),
            "not_strong_generated": (
                None if analysis is None else bool(analysis["hard_gates"]["not_generated"])
            ),
            "not_vendor": (
                None if analysis is None else bool(analysis["hard_gates"]["not_vendor"])
            ),
            "not_strong_repetition": (
                None
                if analysis is None
                else bool(analysis["hard_gates"]["not_pathological_repetition"])
            ),
        }
        records.append((int(item["funnel_bytes"]), gates))
    total_documents = len(records)
    total_bytes = sum(raw_bytes for raw_bytes, _ in records)
    independent: dict[str, Any] = {}
    for gate in FULL_GATE_ORDER:
        evaluated = [(size, gates) for size, gates in records if gates[gate] is not None]
        passed = [(size, gates) for size, gates in evaluated if gates[gate] is True]
        evaluated_documents = len(evaluated)
        evaluated_bytes = sum(size for size, _ in evaluated)
        passed_documents = len(passed)
        passed_bytes = sum(size for size, _ in passed)
        independent[gate] = {
            "evaluated_documents": evaluated_documents,
            "evaluated_raw_bytes": evaluated_bytes,
            "not_evaluable_documents": total_documents - evaluated_documents,
            "not_evaluable_raw_bytes": total_bytes - evaluated_bytes,
            "passed_documents": passed_documents,
            "passed_raw_bytes": passed_bytes,
            "failed_documents": evaluated_documents - passed_documents,
            "failed_raw_bytes": evaluated_bytes - passed_bytes,
            "pass_rate_evaluated": _ratio(passed_documents, evaluated_documents),
        }
    survivors = records
    ordered: list[dict[str, Any]] = []
    for gate in FULL_GATE_ORDER:
        if any(gates[gate] is None for _, gates in survivors):
            raise AssertionError(f"survivor cannot be unevaluable at ordered gate {gate}")
        passed = [(size, gates) for size, gates in survivors if gates[gate] is True]
        input_documents = len(survivors)
        input_bytes = sum(size for size, _ in survivors)
        passed_documents = len(passed)
        passed_bytes = sum(size for size, _ in passed)
        ordered.append({
            "gate": gate,
            "input_documents": input_documents,
            "input_raw_bytes": input_bytes,
            "rejected_at_gate_documents": input_documents - passed_documents,
            "rejected_at_gate_raw_bytes": input_bytes - passed_bytes,
            "passed_documents": passed_documents,
            "passed_raw_bytes": passed_bytes,
        })
        survivors = passed
    retained_documents = len(survivors)
    retained_bytes = sum(size for size, _ in survivors)
    return {
        "gate_order": list(FULL_GATE_ORDER),
        "considered_documents": total_documents,
        "considered_raw_bytes": total_bytes,
        "byte_accounting": (
            "verified raw bytes after fetch; metadata length proxy for fetch failures"
        ),
        "independent_gate_results": independent,
        "ordered_gate_funnel": ordered,
        "retained_documents": retained_documents,
        "retained_raw_bytes": retained_bytes,
        "retained_fraction_wilson_95": wilson_interval(retained_documents, total_documents),
    }


def _distribution(values: Sequence[int | float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": sum(numeric) / len(numeric),
        "max": max(numeric),
    }


def _quality_summary(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path_names = ("vendor", "generated", "test", "config", "lock", "notebook", "minified")
    proxy_names = (
        "no_imports",
        "absolute_imports_are_stdlib_only",
        "has_nonstdlib_or_local_absolute_import",
        "has_relative_import",
        "has_wildcard_import",
    )
    boilerplate_names = (
        "only_imports_assignments_and_docstring",
        "only_declarations_imports_assignments_and_docstring",
        "has_main_guard",
        "comment_or_docstring_dominant_at_50_percent",
    )
    construct_counts: Counter[str] = Counter()
    construct_groups: Counter[str] = Counter()
    for analysis in analyses:
        construct_counts.update(analysis["ast_constructs"]["counts"])
        construct_groups.update(analysis["ast_constructs"]["groups"])
    return {
        "decoded_documents": len(analyses),
        "path_signal_documents": {
            name: sum(bool(analysis["path_signals"][name]) for analysis in analyses)
            for name in path_names
        },
        "generated_signal_documents": sum(
            bool(analysis["generated"]["heuristic_flag"]) for analysis in analyses
        ),
        "repetition_signal_documents": sum(
            bool(analysis["repetition"]["heuristic_flag"]) for analysis in analyses
        ),
        "ast_construct_counts": dict(sorted(construct_counts.items())),
        "ast_construct_group_counts": dict(sorted(construct_groups.items())),
        "import_totals": {
            name: sum(int(analysis["imports"][name]) for analysis in analyses)
            for name in (
                "statements",
                "imported_names",
                "absolute_stdlib_names",
                "absolute_nonstdlib_or_local_names",
                "relative_names",
                "wildcard_names",
            )
        },
        "import_proxy_documents": {
            name: sum(bool(analysis["imports"]["proxies"][name]) for analysis in analyses)
            for name in proxy_names
        },
        "comments": {
            "documents_with_comments": sum(
                int(analysis["comments"]["count"]) > 0 for analysis in analyses
            ),
            "total_comments": sum(int(analysis["comments"]["count"]) for analysis in analyses),
            "total_characters": sum(
                int(analysis["comments"]["characters"]) for analysis in analyses
            ),
            "character_share_distribution": _distribution([
                analysis["comments"]["character_share"]
                for analysis in analyses
                if analysis["comments"]["character_share"] is not None
            ]),
        },
        "docstrings": {
            "documents_with_docstrings": sum(
                int(analysis["docstrings"]["count"]) > 0 for analysis in analyses
            ),
            "total_docstrings": sum(int(analysis["docstrings"]["count"]) for analysis in analyses),
            "total_characters": sum(
                int(analysis["docstrings"]["characters"]) for analysis in analyses
            ),
            "character_share_distribution": _distribution([
                analysis["docstrings"]["character_share"]
                for analysis in analyses
                if analysis["docstrings"]["character_share"] is not None
            ]),
        },
        "boilerplate_descriptor_documents": {
            name: sum(bool(analysis["boilerplate_descriptors"][name]) for analysis in analyses)
            for name in boilerplate_names
        },
        "raw_bytes": _distribution([analysis["size"]["raw_bytes"] for analysis in analyses]),
        "text_characters": _distribution([
            analysis["size"]["text_characters"] for analysis in analyses
        ]),
        "line_count": _distribution([analysis["size"]["line_count"] for analysis in analyses]),
        "contains_source_text": False,
    }


def _yield_observation(item: Mapping[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    if analysis is None:
        return {
            "size": {"raw_bytes": int(item["funnel_bytes"]), "text_characters": 0},
            "passes_all_hard_gates": False,
        }
    return {
        "size": {
            "raw_bytes": int(analysis["size"]["raw_bytes"]),
            "text_characters": int(analysis["size"]["text_characters"]),
        },
        "passes_all_hard_gates": bool(analysis["passes_all_hard_gates"]),
    }


def _scaled_uncertainty_envelope(
    metric: Mapping[str, Any], *, lower_scale: float, upper_scale: float
) -> dict[str, Any]:
    return {
        "estimate": float(metric["estimate"]),
        "heuristic_lower_uncertainty_envelope": float(metric["lower_95"]) * lower_scale,
        "heuristic_upper_uncertainty_envelope": float(metric["upper_95"]) * upper_scale,
    }


def _two_stage_yield(
    *,
    metadata_rows: Sequence[Mapping[str, Any]],
    eligible_row_indices: set[int],
    processed: Sequence[Mapping[str, Any]],
    total_rows: int,
    threshold: int | None,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    def in_slice(row: Mapping[str, Any]) -> bool:
        value = _score(row)
        return threshold is None or (value is not None and value >= threshold)

    eligible_flags = [
        int(int(row["row_idx"]) in eligible_row_indices and in_slice(row)) for row in metadata_rows
    ]
    eligible_count = sum(eligible_flags)
    stage1 = wilson_interval(eligible_count, len(metadata_rows))
    selected_slice = [item for item in processed if in_slice(item["metadata"])]
    missing_scores = sum(_score(row) is None for row in metadata_rows)
    if not selected_slice or eligible_count == 0:
        return {
            "available": False,
            "score_threshold_gte": threshold,
            "reason": "no eligible metadata or selected content rows in score slice",
            "metadata_missing_score_rows": missing_scores,
        }
    population_point = max(1, round(total_rows * eligible_count / len(metadata_rows)))
    pure = estimate_pretokenizer_yield(
        [_yield_observation(item) for item in selected_slice],
        population_documents=population_point,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    lower_population = total_rows * float(stage1["lower"])
    upper_population = total_rows * float(stage1["upper"])
    lower_scale = lower_population / population_point
    upper_scale = upper_population / population_point

    projection = {
        "retained_documents": _scaled_uncertainty_envelope(
            pure["projected_retained_documents"],
            lower_scale=lower_scale,
            upper_scale=upper_scale,
        ),
        "retained_raw_bytes": _scaled_uncertainty_envelope(
            pure["projected_retained_raw_bytes"],
            lower_scale=lower_scale,
            upper_scale=upper_scale,
        ),
        "retained_characters": _scaled_uncertainty_envelope(
            pure["projected_retained_characters"],
            lower_scale=lower_scale,
            upper_scale=upper_scale,
        ),
    }
    sensitivity: list[dict[str, Any]] = []
    for scenario in pure["serialized_token_sensitivity"]:
        sensitivity.append({
            "assumed_characters_per_content_token": scenario[
                "assumed_characters_per_content_token"
            ],
            "includes_bos_and_eos_per_retained_document": True,
            "projected_serialized_token_equivalent": _scaled_uncertainty_envelope(
                scenario["projected_serialized_token_equivalent"],
                lower_scale=lower_scale,
                upper_scale=upper_scale,
            ),
        })
    break_even: list[dict[str, Any]] = []
    docs = projection["retained_documents"]
    characters = projection["retained_characters"]

    def break_even_value(chars: float, documents: float, target: int) -> float | None:
        denominator = target - 2 * documents
        return None if denominator <= 0 else chars / denominator

    for target in YIELD_TARGETS:
        break_even.append({
            "target_serialized_tokens": target,
            "estimate": break_even_value(
                float(characters["estimate"]), float(docs["estimate"]), target
            ),
            "heuristic_lower_uncertainty_envelope": break_even_value(
                float(characters["heuristic_lower_uncertainty_envelope"]),
                float(docs["heuristic_lower_uncertainty_envelope"]),
                target,
            ),
            "heuristic_upper_uncertainty_envelope": break_even_value(
                float(characters["heuristic_upper_uncertainty_envelope"]),
                float(docs["heuristic_upper_uncertainty_envelope"]),
                target,
            ),
            "meaning": (
                "maximum assumed characters per content token that still reaches the target, "
                "including two boundary tokens per retained document"
            ),
        })
    return {
        "available": True,
        "score_threshold_gte": threshold,
        "scope": "PRE_TOKENIZER_TWO_STAGE_SENSITIVITY_NOT_A_CANONICAL_TOKEN_COUNT",
        "canonical_tokenizer_used": False,
        "final_token_quota_supported": False,
        "metadata_stage": {
            "sampled_rows": len(metadata_rows),
            "eligible_unique_within_max_size_rows": eligible_count,
            "eligible_fraction_wilson_95": stage1,
            "interval_assumption": "IID_BINOMIAL_APPROXIMATION_ON_CLUSTER_SAMPLED_ROWS",
            "sampling_design": "50_CONTIGUOUS_WINDOWS_OF_10_ROWS",
            "metadata_missing_score_rows": missing_scores,
            "population_documents": total_rows,
            "projected_eligible_documents_point": population_point,
        },
        "content_stage_conditional_on_eligible": {
            "sampled_documents": len(selected_slice),
            "retained_documents": pure["sample"]["retained_documents"],
            "retained_fraction_wilson_95": pure["sample"]["retained_fraction_wilson_95"],
        },
        "combined_uncertainty_envelope_method": (
            "metadata Wilson endpoints multiplied by conditional SHA-256 bootstrap endpoints"
        ),
        "combined_uncertainty_envelope_calibration": (
            "HEURISTIC_NOT_A_CALIBRATED_JOINT_95_PERCENT_CONFIDENCE_INTERVAL"
        ),
        "bootstrap": pure["bootstrap"],
        "projected": projection,
        "serialized_token_sensitivity": sensitivity,
        "break_even_characters_per_content_token": break_even,
        "limitations": [
            "The 500-row metadata stage estimates the <=100KB distinct-blob eligible fraction.",
            "Wilson treats 500 rows as iid; contiguous-window cluster correlation can make it optimistic.",
            "The 300-content stage estimates quality conditionally within that eligible pool.",
            "Combined projection envelopes are heuristic products of separate endpoints, not "
            "calibrated joint 95% confidence intervals.",
            "Within-sample duplicate rates do not prove corpus-wide exact-dedup yield.",
            "Near-deduplication, decontamination, repository caps, and selection can reduce yield.",
            "No canonical tokenizer was used; token-equivalent scenarios are not measured tokens.",
        ],
    }


def analyze_python_p1(config: AnalysisConfig) -> Path:
    """Verify and analyze one P1 arm without any network-capable dependency."""
    _validate_private_paths(config)
    policy, policy_sha256 = _load_policy(config.policy_path, config.policy_sha256)
    manifest, manifest_sha256 = _load_collection(config.collection_manifest)
    metadata, selected, total_rows = _validate_collection_contract(
        manifest,
        expected_arm=config.expected_arm,
        policy=policy,
        policy_sha256=policy_sha256,
    )
    eligible, eligible_row_indices = _canonical_eligible_rows(metadata)
    processed = _verify_and_analyze_selected(
        selected,
        eligible=eligible,
        cache_dir=config.cache_dir,
    )
    decoded_analyses = [item["analysis"] for item in processed if item["analysis"] is not None]
    assert all(isinstance(analysis, dict) for analysis in decoded_analyses)
    schema = manifest.get("schema")
    if not isinstance(schema, dict) or not isinstance(schema.get("normalized_field_map"), dict):
        raise AnalysisError("collection schema lacks normalized_field_map")
    normalized_fields = set(schema["normalized_field_map"])
    selected_metadata = [item["metadata"] for item in processed]
    metadata_concentration_rows = [
        {**row, "_raw_bytes": int(row["length_bytes"])} for row in metadata
    ]
    selected_concentration_rows = [
        {**item["metadata"], "_raw_bytes": int(item["funnel_bytes"])} for item in processed
    ]
    retained_concentration_rows = [
        {**item["metadata"], "_raw_bytes": int(item["raw_bytes"])}
        for item in processed
        if item["analysis"] is not None and item["analysis"]["passes_all_hard_gates"]
    ]
    ast_fingerprints = [
        str(item["analysis"]["content_identity"]["ast_canonical_fingerprint"]["sha256"])
        for item in processed
        if item["analysis"] is not None
        and item["analysis"]["content_identity"]["ast_canonical_fingerprint"]["available"]
    ]
    full_funnel = _full_funnel(processed)
    fetch_success = sum(bool(item["fetch_success"]) for item in processed)
    fidelity_matches = sum(item["fidelity_length_matches"] is True for item in processed)
    fidelity_mismatches = sum(item["fidelity_length_matches"] is False for item in processed)
    strict_evaluated = sum(item["strict_utf8"] is not None for item in processed)
    strict_success = sum(item["strict_utf8"] is True for item in processed)
    fetch_failure_categories = Counter(
        str(item["failure_category"]) for item in processed if not item["fetch_success"]
    )
    cache_hits = sum(
        item["fetch_success"] and int(item["fetch_attempts"]) == 0 for item in processed
    )
    yields = {
        "all_scores": _two_stage_yield(
            metadata_rows=metadata,
            eligible_row_indices=eligible_row_indices,
            processed=processed,
            total_rows=total_rows,
            threshold=None,
            bootstrap_seed=COLLECTION_SEED,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )
    }
    for threshold in (3, 4):
        yields[f"score_gte_{threshold}"] = _two_stage_yield(
            metadata_rows=metadata,
            eligible_row_indices=eligible_row_indices,
            processed=processed,
            total_rows=total_rows,
            threshold=int(threshold),
            bootstrap_seed=COLLECTION_SEED + int(threshold),
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "status": "complete",
        "decision_scope": "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL",
        "automatic_source_approval": False,
        "network_access": False,
        "source_code_exposed": False,
        "arm": {
            "role": config.expected_arm,
            "adapter": ARM_ADAPTERS[config.expected_arm],
        },
        "inputs": {
            "collection_manifest": {
                "filename": config.collection_manifest.name,
                "sha256": manifest_sha256,
            },
            "collection_backend_provenance": dict(manifest["backend_provenance"]),
            "collection_cache_origin_contract": dict(manifest["cache_origin_contract"]),
            "policy": {
                "filename": config.policy_path.name,
                "sha256": policy_sha256,
                "expected_sha256": config.policy_sha256,
            },
            "private_cache": {
                "objects_verified": fetch_success,
                "cache_indexes_verified": fetch_success,
                "path_exposed": False,
                "every_raw_sha_and_size_verified": True,
                "production_origin_contract_verified": True,
                "cache_hit_origin_evidence_verified": True,
                "cache_hits": cache_hits,
            },
        },
        "matched_contract": {
            "seed": COLLECTION_SEED,
            "metadata_rows": len(metadata),
            "selected_blobs": len(processed),
            "total_population_rows": total_rows,
            "policy_sha256": policy_sha256,
            "minimum_common_score_content_documents_per_arm": (
                MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM
            ),
        },
        "dataset": dict(manifest["input"]),
        "collection_outcomes": {
            "selected_attempts": len(processed),
            "fetch_success": fetch_success,
            "fetch_failed": len(processed) - fetch_success,
            "fetch_success_fraction_wilson_95": wilson_interval(fetch_success, len(processed)),
            "fetch_failure_counts_by_category": dict(sorted(fetch_failure_categories.items())),
            "fidelity_length_matches": fidelity_matches,
            "fidelity_length_mismatch": fidelity_mismatches,
            "fidelity_not_available": len(processed) - fetch_success,
            "strict_utf8_success": strict_success,
            "strict_utf8_failed": strict_evaluated - strict_success,
            "strict_utf8_not_evaluable": len(processed) - strict_evaluated,
            "strict_utf8_fraction_wilson_95": wilson_interval(strict_success, len(processed)),
            "strict_utf8_denominator": "ALL_SELECTED_ATTEMPTS_END_TO_END",
            "strict_utf8_conditional_on_fetch_and_fidelity": {
                "numerator": strict_success,
                "denominator": strict_evaluated,
                "fraction_wilson_95": (
                    wilson_interval(strict_success, strict_evaluated) if strict_evaluated else None
                ),
            },
        },
        "provenance_presence": {
            "metadata_sample": _presence_summary(metadata, normalized_fields=normalized_fields),
            "selected_content_sample": _presence_summary(
                selected_metadata, normalized_fields=normalized_fields
            ),
            "values_exposed": False,
        },
        "duplicates": {
            "metadata_blob_id": _duplicate_summary([str(row["blob_id"]) for row in metadata]),
            "selected_raw_sha256": _duplicate_summary([
                str(item["raw_sha256"]) for item in processed if item["raw_sha256"] is not None
            ]),
            "decoded_ast_canonical_fingerprint": _duplicate_summary(ast_fingerprints),
        },
        "concentration": {
            "metadata_repository_rows_and_bytes": _concentration(
                metadata_concentration_rows, field="repo_name"
            ),
            "metadata_path_rows_and_bytes": _concentration(
                metadata_concentration_rows, field="path"
            ),
            "selected_repository_rows_and_bytes": _concentration(
                selected_concentration_rows, field="repo_name"
            ),
            "selected_path_rows_and_bytes": _concentration(
                selected_concentration_rows, field="path"
            ),
            "retained_repository_rows_and_bytes": _concentration(
                retained_concentration_rows, field="repo_name"
            ),
            "retained_path_rows_and_bytes": _concentration(
                retained_concentration_rows, field="path"
            ),
            "values_exposed": False,
        },
        "full_hard_gate_funnel": full_funnel,
        "decoded_local_quality_funnel": summarize_hard_gate_funnel(decoded_analyses),
        "decoded_quality_metrics": _quality_summary(decoded_analyses),
        "score_slices_and_pretokenizer_yield": yields,
        "privacy": {
            "source_text_fields": 0,
            "source_lines": 0,
            "repository_or_path_values": 0,
            "raw_or_ast_hash_values_exposed": 0,
        },
        "limitations": [
            "This report supports source comparison, not automatic source approval.",
            "PROVENANCE_FIELD_COVERAGE_ONLY_NOT_LICENSE_CLEARANCE: automated license-field "
            "presence does not establish legal permission, and missing fields do not establish "
            "that no license exists; removal and terms gates remain independent and still block "
            "production until satisfied.",
            "P1 cannot establish final-tokenizer yield or post-dedup/decontamination supply.",
        ],
    }
    return _publish_addressed_json(config.output_dir, "python-p1-analysis", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a frozen Python P1 collection offline.")
    parser.add_argument("--collection_manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected_policy_sha256", dest="policy_sha256", required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_arm", choices=tuple(ARM_ADAPTERS), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = analyze_python_p1(
        AnalysisConfig(
            collection_manifest=args.collection_manifest,
            policy_path=args.policy,
            policy_sha256=args.policy_sha256,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            expected_arm=args.expected_arm,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
