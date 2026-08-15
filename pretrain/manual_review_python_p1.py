#!/usr/bin/env python3

"""Blinded, fail-closed manual review for the frozen Python P1 artifacts.

The tool has three explicit phases: ``prepare`` publishes an opaque queue only
after complete offline revalidation, ``review`` displays sanitized source only
on the RunPod controlling TTY and seals human labels, and ``unblind`` publishes
aggregate confusion counts only after the attestation is durable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import termios
from typing import Any, NoReturn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pretrain.analyze_python_p1 as analyzer  # noqa: E402
from pretrain.analyze_python_p1 import AnalysisConfig  # noqa: E402
import pretrain.collect_python_p1 as collector  # noqa: E402
import pretrain.compare_python_p1 as comparator  # noqa: E402
from pretrain.compare_python_p1 import ComparisonConfig  # noqa: E402
from pretrain.manual_review_io import (  # noqa: E402
    ManualReviewIOError,
    SessionLock,
    canonical_json_bytes,
    create_session_marker,
    disable_core_dumps,
    open_directory_nofollow,
    open_regular_file_nofollow,
    publish_content_addressed_json,
    read_regular_file_at_nofollow,
    read_regular_file_nofollow,
    sanitize_terminal_text,
)

FROZEN_SPEC_SHA256 = "700f6f4e99652c3afa633456f788211f04dedaf8e284b89c59471f1dedee0b7c"
QUEUE_KIND = "petitgpt_python_p1_blinded_manual_review_queue"
ATTESTATION_KIND = "petitgpt_python_p1_blinded_manual_review_attestation"
RESULT_KIND = "petitgpt_python_p1_blinded_manual_review_result"
DECISION_SCOPE = "BLINDED_MANUAL_QUALITY_SPOT_CHECK_NOT_SOURCE_OR_LICENSE_APPROVAL"
QUEUE_MARKER = "QUEUE_READY.json"
ATTESTATION_MARKER = "ATTESTATION_SEALED.json"
RESULT_MARKER = "RESULT_COMMITTED.json"
LOCK_NAME = ".manual-review-v2.lock"
ALLOWED_LABELS = ("MANUAL_KEEP", "MANUAL_REJECT", "UNREVIEWABLE")
AUTOMATIC_OUTCOMES = ("keep", "reject")
PRESENTATION_DOMAIN = b"petitgpt-python-p1-blinded-manual-review-v2-presentation"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_ADDRESSED_RE = re.compile(r"^[a-z0-9-]+\.sha256-([0-9a-f]{64})\.json$")
_INDEX_RE = re.compile(r"index-[0-9a-f]{64}\.json")
_RAW_RE = re.compile(r"raw-sha256-[0-9a-f]{64}\.raw")
_MAX_JSON_BYTES = 64 * 1024 * 1024


class ManualReviewError(RuntimeError):
    """The blinded manual-review contract was violated."""


class ReviewInterrupted(ManualReviewError):
    """The reviewer interrupted the controlling-TTY session."""


@dataclass(frozen=True)
class ArmInputs:
    role: str
    manifest: Path
    collection_report: Path
    replay_report: Path
    analysis_report: Path
    cache_dir: Path


@dataclass(frozen=True)
class ManualReviewConfig:
    spec_path: Path
    policy_path: Path
    comparison_report: Path
    primary: ArmInputs
    stack_comparison: ArmInputs
    session_dir: Path
    expected_generator_commit: str
    enforce_environment: bool = True
    enforce_frozen_spec: bool = True


@dataclass(frozen=True, repr=False)
class ReviewItem:
    review_id: str
    arm: str
    automatic_outcome: str
    selection_rank_sha256: str
    presentation_sha256: str
    raw: bytes = field(repr=False)


@dataclass(frozen=True, repr=False)
class VerifiedInputs:
    spec: dict[str, Any]
    spec_sha256: str
    items: tuple[ReviewItem, ...]
    queue: dict[str, Any]
    sensitive_values: frozenset[str]


@dataclass(frozen=True)
class BlindReviewItem:
    review_id: str
    raw: bytes = field(repr=False)


def _fatal_network(*args: Any, **kwargs: Any) -> NoReturn:
    del args, kwargs
    raise ManualReviewError("network access is forbidden during manual review")


def install_zero_network_guards() -> None:
    """Deny common Python network entry points for the rest of the process."""
    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket.") or event in {"os.system", "pty.spawn"}:
            raise ManualReviewError("network or shell execution is forbidden during review")
        if event == "subprocess.Popen":
            executable = os.path.basename(os.fspath(args[0])) if args else ""
            if executable != "git":
                raise ManualReviewError("non-Git subprocess execution is forbidden during review")

    sys.addaudithook(audit_hook)
    socket.socket = _fatal_network  # type: ignore[assignment]
    socket.create_connection = _fatal_network  # type: ignore[assignment]
    socket.getaddrinfo = _fatal_network  # type: ignore[assignment]
    collector.requests_json_transport = _fatal_network  # type: ignore[assignment]
    collector.make_bounded_swh_fetcher = _fatal_network  # type: ignore[assignment]
    inspect_module = sys.modules.get("pretrain.inspect_python_sources")
    if inspect_module is not None:
        inspect_module.urlopen = _fatal_network
    requests_module = sys.modules.get("requests")
    if requests_module is not None:
        requests_module.get = _fatal_network
        requests_module.request = _fatal_network
    boto3_module = sys.modules.get("boto3")
    if boto3_module is not None:
        boto3_module.client = _fatal_network
        boto3_module.resource = _fatal_network


def _inside_project(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        absolute.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ManualReviewError(f"{label} must be inside the PetitGPT worktree") from exc
    return absolute


def _strict_json(data: bytes, *, label: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate object key")
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=object_hook,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManualReviewError(f"{label} is not strict duplicate-free JSON") from exc
    if not isinstance(value, dict):
        raise ManualReviewError(f"{label} must be a JSON object")
    return value


def _secure_bytes(path: Path, *, label: str, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
    absolute = _inside_project(path, label=label)
    try:
        return read_regular_file_nofollow(absolute, max_bytes=max_bytes)
    except ManualReviewIOError as exc:
        raise ManualReviewError(f"cannot securely read {label}") from exc


def _load_sha_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    require_addressed_filename: bool = True,
) -> tuple[dict[str, Any], bytes]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ManualReviewError(f"{label} expected SHA-256 is malformed")
    data = _secure_bytes(path, label=label)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ManualReviewError(f"{label} SHA-256 mismatch")
    if require_addressed_filename:
        match = _ADDRESSED_RE.fullmatch(path.name)
        if match is None or match.group(1) != actual:
            raise ManualReviewError(f"{label} filename is not bound to its bytes")
    return _strict_json(data, label=label), data


def _run_git(arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ManualReviewError("local Git preflight failed")
    return result.stdout


def _require_ignored(path: Path) -> None:
    absolute = _inside_project(path, label="private path")
    relative = absolute.relative_to(PROJECT_ROOT)
    result = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "check-ignore",
            "-q",
            "--",
            os.fspath(relative),
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ManualReviewError("a private input/output path is not Git-ignored")


def _environment_preflight(config: ManualReviewConfig) -> None:
    if not config.enforce_environment:
        return
    if _COMMIT_RE.fullmatch(config.expected_generator_commit) is None:
        raise ManualReviewError("expected generator commit is malformed")
    head = _run_git(["rev-parse", "HEAD"]).decode("ascii", errors="strict").strip()
    if head != config.expected_generator_commit:
        raise ManualReviewError("generator Git commit differs from the frozen invocation")
    if _run_git(["status", "--porcelain=v1", "--untracked-files=all"]):
        raise ManualReviewError("generator worktree is not clean")
    _run_git([
        "ls-files",
        "--error-unmatch",
        "pretrain/manual_review_python_p1.py",
        "pretrain/manual_review_io.py",
    ])
    for arm in (config.primary, config.stack_comparison):
        for path in (
            arm.manifest,
            arm.collection_report,
            arm.replay_report,
            arm.analysis_report,
            arm.cache_dir,
        ):
            _require_ignored(path)
    for path in (
        config.spec_path,
        config.policy_path,
        config.comparison_report,
        config.session_dir,
    ):
        _require_ignored(path)


def _validate_spec(spec: Mapping[str, Any], *, spec_sha256: str, frozen: bool) -> None:
    if frozen and spec_sha256 != FROZEN_SPEC_SHA256:
        raise ManualReviewError("manual-review spec SHA is not the frozen v2 SHA")
    if (
        spec.get("schema_version") != 2
        or spec.get("kind") != "petitgpt_python_p1_blinded_manual_review_policy"
        or spec.get("status") != "FROZEN_BEFORE_INDIVIDUAL_REVIEW"
        or spec.get("decision_scope") != DECISION_SCOPE
    ):
        raise ManualReviewError("manual-review spec header drifted")
    sampling = spec.get("sampling")
    outcomes = spec.get("outcomes")
    attestation = spec.get("manual_attestation")
    validation = spec.get("validation")
    if not all(isinstance(value, Mapping) for value in (sampling, outcomes, attestation, validation)):
        raise ManualReviewError("manual-review spec blocks are malformed")
    assert isinstance(sampling, Mapping)
    assert isinstance(outcomes, Mapping)
    assert isinstance(attestation, Mapping)
    assert isinstance(validation, Mapping)
    if (
        sampling.get("presentation_domain_ascii") != PRESENTATION_DOMAIN.decode("ascii")
        or sampling.get("presentation_separator_hex") != "00"
        or sampling.get("selected_records") != 48
        or outcomes.get("reviewable_records_per_outcome_per_arm") != 12
        or outcomes.get("gate_order") != list(analyzer.FULL_GATE_ORDER)
        or attestation.get("allowed_labels") != list(ALLOWED_LABELS)
        or attestation.get("all_48_labels_required") is not True
        or validation.get("network_access") is not False
        or validation.get("expected_cache_indexes_per_arm") != 300
        or validation.get("expected_raw_objects_per_arm") != 300
    ):
        raise ManualReviewError("manual-review spec executable contract drifted")


def _index_name(blob_id: str) -> str:
    return f"index-{hashlib.sha256(blob_id.encode('utf-8')).hexdigest()}.json"


def _snapshot_cache(
    cache_dir: Path,
    *,
    manifest: Mapping[str, Any],
    expected_origin: Mapping[str, Any],
) -> dict[str, tuple[bytes, str, str]]:
    entries = manifest.get("selected_blobs")
    if not isinstance(entries, list) or len(entries) != 300:
        raise ManualReviewError("manifest selected-blob population drifted")
    expected_indexes: set[str] = set()
    expected_raw: set[str] = set()
    expected_by_blob: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("fetch_outcome") != "success":
            raise ManualReviewError("frozen manual-review cache must contain all 300 objects")
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("blob_id"), str):
            raise ManualReviewError("manifest cache identity is malformed")
        blob_id = metadata["blob_id"]
        if blob_id in expected_by_blob:
            raise ManualReviewError("manifest cache identity collision")
        raw_sha = entry.get("raw_sha256")
        raw_name = entry.get("cache_object")
        if (
            not isinstance(raw_sha, str)
            or _SHA256_RE.fullmatch(raw_sha) is None
            or raw_name != f"raw-sha256-{raw_sha}.raw"
        ):
            raise ManualReviewError("manifest raw-object identity is malformed")
        expected_by_blob[blob_id] = entry
        expected_indexes.add(_index_name(blob_id))
        expected_raw.add(raw_name)
    if len(expected_indexes) != 300 or len(expected_raw) != 300:
        raise ManualReviewError("cache filename population or identity collided")

    absolute = _inside_project(cache_dir, label="private cache")
    snapshots: dict[str, tuple[bytes, str, str]] = {}
    try:
        with open_directory_nofollow(absolute) as directory_fd:
            directory_before = os.fstat(directory_fd)
            observed = set(os.listdir(directory_fd))
            if observed != expected_indexes | expected_raw:
                raise ManualReviewError("private cache file population drifted")
            if any(
                (_INDEX_RE.fullmatch(name) is None and _RAW_RE.fullmatch(name) is None)
                for name in observed
            ):
                raise ManualReviewError("private cache contains a noncanonical filename")
            for blob_id, manifest_entry in expected_by_blob.items():
                index_data = read_regular_file_at_nofollow(
                    directory_fd,
                    _index_name(blob_id),
                    max_bytes=1024 * 1024,
                )
                index = _strict_json(index_data, label="private cache index")
                expected_fields = {
                    "schema_version",
                    "blob_id",
                    "raw_sha256",
                    "raw_bytes",
                    "raw_path",
                    "origin",
                }
                if (
                    set(index) != expected_fields
                    or index.get("schema_version") != collector.CACHE_INDEX_SCHEMA_VERSION
                    or index.get("blob_id") != blob_id
                    or index.get("origin") != expected_origin
                    or index.get("raw_sha256") != manifest_entry.get("raw_sha256")
                    or index.get("raw_path") != manifest_entry.get("cache_object")
                    or index.get("raw_bytes") != manifest_entry.get("raw_bytes")
                ):
                    raise ManualReviewError("private cache index disagrees with manifest")
                raw_name = str(index["raw_path"])
                raw = read_regular_file_at_nofollow(
                    directory_fd,
                    raw_name,
                    max_bytes=collector.MAX_DECOMPRESSED_BYTES,
                )
                raw_sha = hashlib.sha256(raw).hexdigest()
                if raw_sha != index["raw_sha256"] or len(raw) != index["raw_bytes"]:
                    raise ManualReviewError("private raw object digest or size drifted")
                snapshots[blob_id] = (raw, raw_sha, raw_name)
            directory_after = os.fstat(directory_fd)
            if set(os.listdir(directory_fd)) != observed or (
                directory_before.st_dev,
                directory_before.st_ino,
                directory_before.st_mtime_ns,
                directory_before.st_ctime_ns,
            ) != (
                directory_after.st_dev,
                directory_after.st_ino,
                directory_after.st_mtime_ns,
                directory_after.st_ctime_ns,
            ):
                raise ManualReviewError("private cache directory changed during snapshot")
    except ManualReviewIOError as exc:
        raise ManualReviewError("private cache no-follow snapshot failed") from exc
    if len(snapshots) != 300:
        raise ManualReviewError("private cache snapshot underfilled")
    return snapshots


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManualReviewError(f"{label} must be a nonnegative integer")
    return value


def _validate_runtime_storage(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ManualReviewError("collection storage evidence is malformed")
    for usage in value.values():
        if not isinstance(usage, Mapping) or not usage:
            raise ManualReviewError("collection storage usage evidence is malformed")
        for amount in usage.values():
            _nonnegative_int(amount, label="collection storage amount")


def _validate_collection_report(
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
) -> None:
    entries = manifest.get("selected_blobs")
    metadata = manifest.get("metadata_rows")
    if not isinstance(entries, list) or not isinstance(metadata, list):
        raise ManualReviewError("collection manifest populations are malformed")
    replayed_entries = [
        {
            **entry,
            "cache_hit": (
                entry.get("fetch_outcome") == "success" and entry.get("fetch_attempts") == 0
            ),
        }
        for entry in entries
        if isinstance(entry, Mapping)
    ]
    if len(replayed_entries) != len(entries):
        raise ManualReviewError("collection manifest selected entry is malformed")
    resources = report.get("resources")
    if not isinstance(resources, Mapping) or set(resources) != {
        "workers",
        "hf_logical_calls",
        "swh_selected_objects",
        "total_retries",
        "elapsed_nanoseconds",
        "storage_before_publish",
    }:
        raise ManualReviewError("collection runtime resource evidence is malformed")
    _nonnegative_int(resources.get("elapsed_nanoseconds"), label="collection elapsed time")
    _validate_runtime_storage(resources.get("storage_before_publish"))
    expected_resources = {
        **resources,
        "workers": collector.MAX_WORKERS,
        "hf_logical_calls": collector.EXPECTED_HF_LOGICAL_CALLS,
        "swh_selected_objects": 300,
        "total_retries": manifest["backend_accounting"]["total_retries"],
    }
    manifest_binding = report.get("manifest")
    if (
        not isinstance(manifest_binding, Mapping)
        or manifest_binding.get("sha256") != manifest_sha256
        or not isinstance(manifest_binding.get("path"), str)
        or Path(manifest_binding["path"]).name != manifest_path.name
    ):
        raise ManualReviewError("collection report manifest binding drifted")
    expected = {
        "schema_version": 1,
        "kind": "petitgpt_python_p1_report",
        "status": "complete",
        "source_code_exposed": False,
        "decision_scope": collector.DECISION_SCOPE,
        "policy_binding": manifest["policy_binding"],
        "input": manifest["input"],
        "contract": manifest["contract"],
        "backend_provenance": manifest["backend_provenance"],
        "cache_origin_contract": manifest["cache_origin_contract"],
        "backend_accounting": manifest["backend_accounting"],
        "metadata": collector._metadata_summary(metadata),
        "selection": manifest["selection"],
        "content": collector._content_summary(replayed_entries),
        "resources": expected_resources,
        "manifest": dict(manifest_binding),
    }
    if report != expected:
        raise ManualReviewError("collection report does not recompute from its manifest")


def _p1_config(
    arm: ArmInputs,
    *,
    manifest: Mapping[str, Any],
    policy_path: Path,
    policy_sha256: str,
) -> collector.P1Config:
    input_spec = manifest.get("input")
    if not isinstance(input_spec, Mapping):
        raise ManualReviewError("collection manifest input is malformed")
    try:
        return collector.P1Config(
            expected_revision=str(input_spec["expected_revision"]),
            adapter=str(input_spec["adapter"]),
            dataset=str(input_spec["dataset"]),
            dataset_config=str(input_spec["dataset_config"]),
            split=str(input_spec["split"]),
            output_dir=arm.replay_report.parent,
            cache_dir=arm.cache_dir,
            policy_path=policy_path,
            expected_policy_sha256=policy_sha256,
            workers=collector.MAX_WORKERS,
            replay_manifest=arm.manifest,
            enforce_ignored_paths=False,
        )
    except KeyError as exc:
        raise ManualReviewError("collection manifest input is incomplete") from exc


def _validate_replay_report(
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    p1_config: collector.P1Config,
    cache: Mapping[str, tuple[bytes, str, str]],
) -> list[dict[str, Any]]:
    try:
        entries = collector._validate_replay_manifest(
            manifest,
            config=p1_config,
            policy_binding=manifest["policy_binding"],
        )
    except (collector.CollectionError, KeyError, TypeError, ValueError) as exc:
        raise ManualReviewError("collector replay semantic validation failed") from exc
    replayed: list[dict[str, Any]] = []
    verified = 0
    for entry in entries:
        metadata = entry["metadata"]
        blob_id = metadata["blob_id"]
        if entry.get("fetch_outcome") != "success":
            replayed.append({**entry, "cache_hit": False})
            continue
        snapshot = cache.get(blob_id)
        if snapshot is None:
            raise ManualReviewError("verified cache lacks a replay object")
        raw, raw_sha, raw_name = snapshot
        if raw_sha != entry.get("raw_sha256") or raw_name != entry.get("cache_object"):
            raise ManualReviewError("verified cache identity disagrees with replay manifest")
        fidelity = entry.get("fidelity_outcome")
        if fidelity == "metadata_length_mismatch":
            if len(raw) == metadata["length_bytes"]:
                raise ManualReviewError("recomputed replay fidelity disagrees with manifest")
        elif fidelity == "length_matches_metadata":
            if len(raw) != metadata["length_bytes"]:
                raise ManualReviewError("recomputed replay fidelity disagrees with manifest")
            outcome, quality = collector._source_quality(str(metadata.get("path", "")), raw)
            if outcome != entry.get("decode_outcome") or quality != entry.get("quality"):
                raise ManualReviewError("recomputed replay quality disagrees with manifest")
        else:
            raise ManualReviewError("replay manifest has an unknown fidelity outcome")
        verified += 1
        replayed.append({**entry, "cache_hit": True})
    elapsed = _nonnegative_int(report.get("elapsed_nanoseconds"), label="replay elapsed time")
    recorded_path = report.get("input_manifest")
    if not isinstance(recorded_path, str) or Path(recorded_path).name != manifest_path.name:
        raise ManualReviewError("replay report manifest path binding drifted")
    expected = {
        "schema_version": 1,
        "kind": "petitgpt_python_p1_replay_report",
        "status": "complete",
        "network_access": False,
        "source_code_exposed": False,
        "decision_scope": collector.DECISION_SCOPE,
        "policy_binding": manifest["policy_binding"],
        "collection_backend_provenance": manifest["backend_provenance"],
        "cache_origin_contract": manifest["cache_origin_contract"],
        "cache_origin_verified": verified
        == sum(entry.get("fetch_outcome") == "success" for entry in replayed),
        "collection_backend_accounting": manifest["backend_accounting"],
        "input_manifest": recorded_path,
        "input_manifest_sha256": manifest_sha256,
        "selected_records": len(replayed),
        "verified_blobs": verified,
        "recorded_fetch_failures": len(replayed) - verified,
        "content": collector._content_summary(replayed),
        "elapsed_nanoseconds": elapsed,
    }
    if report != expected:
        raise ManualReviewError("replay report does not recompute from immutable cache bytes")
    return entries


def _record_gates(record: Mapping[str, Any]) -> dict[str, bool]:
    analysis = record.get("analysis")
    hard = analysis.get("hard_gates") if isinstance(analysis, Mapping) else {}
    if not isinstance(hard, Mapping):
        hard = {}
    gates = {
        "fetch_success": record.get("fetch_success") is True,
        "fidelity_length_matches": record.get("fidelity_length_matches") is True,
        "strict_utf8": record.get("strict_utf8") is True,
        "raw_size_200_to_100000_bytes": hard.get("size_200_to_100000_bytes") is True,
        "nonempty_nonwhitespace": hard.get("nonempty_nonwhitespace") is True,
        "not_binary_like": hard.get("not_binary_like") is True,
        "python3_ast_parse": hard.get("python3_ast_parse") is True,
        "not_strong_generated": hard.get("not_generated") is True,
        "not_vendor": hard.get("not_vendor") is True,
        "not_strong_repetition": hard.get("not_pathological_repetition") is True,
    }
    if list(gates) != list(analyzer.FULL_GATE_ORDER):
        raise AssertionError("manual-review gate mapping drifted")
    return gates


def _automatic_outcome(record: Mapping[str, Any]) -> tuple[str, bool]:
    gates = _record_gates(record)
    outcome = "keep" if all(gates.values()) else "reject"
    reviewable = outcome == "reject" and all(
        gates[name] for name in analyzer.FULL_GATE_ORDER[:3]
    )
    return outcome, reviewable


def _sensitive_values(manifests: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    values: set[str] = set()
    for manifest in manifests:
        for row in manifest.get("metadata_rows", []):
            if not isinstance(row, Mapping):
                continue
            for field_name in ("blob_id", "repo_name", "path"):
                value = row.get(field_name)
                if isinstance(value, str) and len(value) >= 8:
                    values.add(value)
        for entry in manifest.get("selected_blobs", []):
            if not isinstance(entry, Mapping):
                continue
            for field_name in ("raw_sha256", "cache_object", "selection_rank_sha256"):
                value = entry.get(field_name)
                if isinstance(value, str) and len(value) >= 8:
                    values.add(value)
    return frozenset(values)


def _assert_no_private_values(data: bytes, sensitive_values: frozenset[str]) -> None:
    for value in sensitive_values:
        if value.encode("utf-8") in data:
            raise ManualReviewError("persistent manual-review output failed privacy scan")


def _verify_arm(
    arm: ArmInputs,
    *,
    expected: Mapping[str, Any],
    policy: Mapping[str, Any],
    policy_sha256: str,
    policy_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[bytes, str, str]], dict[str, Any], str]:
    manifest_sha = str(expected.get("collection_manifest_sha256"))
    collection_sha = str(expected.get("collection_report_sha256"))
    replay_sha = str(expected.get("replay_report_sha256"))
    analysis_sha = str(expected.get("analysis_sha256"))
    manifest, _ = _load_sha_bound_json(
        arm.manifest,
        expected_sha256=manifest_sha,
        label=f"{arm.role} manifest",
    )
    collection_report, _ = _load_sha_bound_json(
        arm.collection_report,
        expected_sha256=collection_sha,
        label=f"{arm.role} collection report",
    )
    replay_report, _ = _load_sha_bound_json(
        arm.replay_report,
        expected_sha256=replay_sha,
        label=f"{arm.role} replay report",
    )
    analysis_report, analysis_bytes = _load_sha_bound_json(
        arm.analysis_report,
        expected_sha256=analysis_sha,
        label=f"{arm.role} analysis report",
    )
    cache_origin = manifest.get("cache_origin_contract")
    if not isinstance(cache_origin, Mapping):
        raise ManualReviewError("collection manifest cache origin is malformed")
    cache = _snapshot_cache(arm.cache_dir, manifest=manifest, expected_origin=cache_origin)
    _validate_collection_report(
        collection_report,
        manifest=manifest,
        manifest_path=arm.manifest,
        manifest_sha256=manifest_sha,
    )
    p1_config = _p1_config(
        arm,
        manifest=manifest,
        policy_path=policy_path,
        policy_sha256=policy_sha256,
    )
    _validate_replay_report(
        replay_report,
        manifest=manifest,
        manifest_path=arm.manifest,
        manifest_sha256=manifest_sha,
        p1_config=p1_config,
        cache=cache,
    )
    records: list[dict[str, Any]] = []
    try:
        rebuilt_analysis = analyzer.build_python_p1_analysis(
            AnalysisConfig(
                collection_manifest=arm.manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha256,
                cache_dir=arm.cache_dir,
                output_dir=arm.analysis_report.parent,
                expected_arm=arm.role,
                enforce_ignored_paths=False,
            ),
            verified_cache_entries=cache,
            verified_policy=(policy, policy_sha256),
            verified_manifest=(manifest, manifest_sha),
            verified_records_out=records,
        )
    except (analyzer.AnalysisError, KeyError, TypeError, ValueError) as exc:
        raise ManualReviewError("offline analyzer recomputation failed") from exc
    if rebuilt_analysis != analysis_report or canonical_json_bytes(rebuilt_analysis) != analysis_bytes:
        raise ManualReviewError("immutable analysis report does not match recomputed bytes")
    if len(records) != 300:
        raise ManualReviewError("offline analyzer record population drifted")
    expected_population = expected.get("expected_population")
    if not isinstance(expected_population, Mapping):
        raise ManualReviewError("spec expected population is malformed")
    outcomes = [_automatic_outcome(record) for record in records]
    counts = Counter(outcome for outcome, _ in outcomes)
    reviewable_reject = sum(reviewable for _, reviewable in outcomes)
    observed_population = {
        "total": len(records),
        "keep": counts["keep"],
        "reject": counts["reject"],
        "reviewable_reject": reviewable_reject,
        "nonreviewable_reject": counts["reject"] - reviewable_reject,
    }
    if dict(expected_population) != observed_population:
        raise ManualReviewError("recomputed automatic outcome population disagrees with spec")
    return manifest, records, cache, analysis_report, analysis_sha


def _select_items(
    *,
    arm_records: Mapping[str, list[dict[str, Any]]],
    arm_caches: Mapping[str, Mapping[str, tuple[bytes, str, str]]],
) -> tuple[ReviewItem, ...]:
    selected: list[tuple[str, str, str, bytes]] = []
    all_ranks: set[str] = set()
    for arm in ("primary", "stack_comparison"):
        frames: dict[str, list[tuple[str, dict[str, Any]]]] = {
            "keep": [],
            "reject": [],
        }
        records = arm_records[arm]
        for record in records:
            outcome, reviewable = _automatic_outcome(record)
            metadata = record.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ManualReviewError("verified analyzer record metadata is malformed")
            blob_id = metadata.get("blob_id")
            if not isinstance(blob_id, str):
                raise ManualReviewError("verified analyzer record identity is malformed")
            rank = hashlib.sha256(f"{analyzer.COLLECTION_SEED}\0{blob_id}".encode()).hexdigest()
            if rank in all_ranks:
                raise ManualReviewError("selection-rank collision across P1 arms")
            all_ranks.add(rank)
            if outcome == "keep" or reviewable:
                frames[outcome].append((rank, record))
        for outcome in AUTOMATIC_OUTCOMES:
            ordered = sorted(frames[outcome], key=lambda item: item[0])
            if len(ordered) < 12:
                raise ManualReviewError("manual-review frame underfilled; backfill is forbidden")
            for rank, record in ordered[:12]:
                metadata = record["metadata"]
                raw_snapshot = arm_caches[arm].get(metadata["blob_id"])
                if raw_snapshot is None:
                    raise ManualReviewError("selected review source lacks a verified cache snapshot")
                raw, raw_sha, _ = raw_snapshot
                if (
                    record.get("raw_sha256") != raw_sha
                    or record.get("raw_bytes") != len(raw)
                    or hashlib.sha256(raw).hexdigest() != raw_sha
                ):
                    raise ManualReviewError("selected review source failed final in-memory rehash")
                try:
                    raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise ManualReviewError("selected review source is not strict UTF-8") from exc
                selected.append((arm, outcome, rank, raw))
    if len(selected) != 48:
        raise ManualReviewError("manual-review deterministic selection did not produce 48 items")
    presentation_rows: list[tuple[str, str, str, bytes, str]] = []
    presentation_hashes: set[str] = set()
    for arm, outcome, rank, raw in selected:
        digest = hashlib.sha256(PRESENTATION_DOMAIN + b"\0" + rank.encode("ascii")).hexdigest()
        if digest in presentation_hashes:
            raise ManualReviewError("manual-review presentation hash collision")
        presentation_hashes.add(digest)
        presentation_rows.append((digest, arm, outcome, raw, rank))
    presentation_rows.sort(key=lambda item: item[0])
    return tuple(
        ReviewItem(
            review_id=f"mrv2-{ordinal:04d}",
            arm=arm,
            automatic_outcome=outcome,
            selection_rank_sha256=rank,
            presentation_sha256=digest,
            raw=raw,
        )
        for ordinal, (digest, arm, outcome, raw, rank) in enumerate(
            presentation_rows,
            start=1,
        )
    )


def verify_inputs(config: ManualReviewConfig) -> VerifiedInputs:
    """Revalidate every immutable artifact and cache object without publishing."""
    if (
        config.primary.role != "primary"
        or config.stack_comparison.role != "stack_comparison"
        or config.primary.role == config.stack_comparison.role
    ):
        raise ManualReviewError("manual-review config must contain one exact arm of each role")
    _environment_preflight(config)
    spec_data = _secure_bytes(config.spec_path, label="manual-review spec", max_bytes=1024 * 1024)
    spec_sha = hashlib.sha256(spec_data).hexdigest()
    spec = _strict_json(spec_data, label="manual-review spec")
    _validate_spec(spec, spec_sha256=spec_sha, frozen=config.enforce_frozen_spec)
    inputs = spec.get("inputs")
    outputs = spec.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ManualReviewError("manual-review spec input/output blocks are malformed")
    expected_namespace = outputs.get("exact_output_namespace")
    if not isinstance(expected_namespace, str):
        raise ManualReviewError("manual-review output namespace is malformed")
    expected_session = _inside_project(PROJECT_ROOT / expected_namespace, label="session output")
    if _inside_project(config.session_dir, label="session output") != expected_session:
        raise ManualReviewError("session output path differs from the frozen namespace")
    policy_sha = inputs.get("frozen_p1_policy_sha256")
    if not isinstance(policy_sha, str):
        raise ManualReviewError("manual-review spec policy SHA is malformed")
    policy, policy_bytes = _load_sha_bound_json(
        config.policy_path,
        expected_sha256=policy_sha,
        label="frozen P1 policy",
        require_addressed_filename=False,
    )
    if policy != analyzer.p1_policy_template() or hashlib.sha256(policy_bytes).hexdigest() != policy_sha:
        raise ManualReviewError("frozen P1 policy content drifted")
    arm_specs = {
        "primary": inputs.get("primary"),
        "stack_comparison": inputs.get("stack_comparison"),
    }
    if not all(isinstance(value, Mapping) for value in arm_specs.values()):
        raise ManualReviewError("manual-review arm input bindings are malformed")
    verified: dict[str, tuple[Any, ...]] = {}
    for arm in (config.primary, config.stack_comparison):
        expected = arm_specs[arm.role]
        assert isinstance(expected, Mapping)
        verified[arm.role] = _verify_arm(
            arm,
            expected=expected,
            policy=policy,
            policy_sha256=policy_sha,
            policy_path=config.policy_path,
        )
    primary_manifest, primary_records, primary_cache, primary_analysis, primary_sha = verified[
        "primary"
    ]
    stack_manifest, stack_records, stack_cache, stack_analysis, stack_sha = verified[
        "stack_comparison"
    ]
    comparison_sha = inputs.get("comparison_sha256")
    if not isinstance(comparison_sha, str):
        raise ManualReviewError("manual-review comparison SHA is malformed")
    comparison_report, comparison_bytes = _load_sha_bound_json(
        config.comparison_report,
        expected_sha256=comparison_sha,
        label="matched comparison report",
    )
    try:
        rebuilt_comparison = comparator.build_python_p1_comparison(
            ComparisonConfig(
                primary_report=config.primary.analysis_report,
                stack_report=config.stack_comparison.analysis_report,
                policy_sha256=policy_sha,
                output_dir=config.comparison_report.parent,
                enforce_ignored_output=False,
            ),
            verified_reports=((primary_analysis, primary_sha), (stack_analysis, stack_sha)),
        )
    except (comparator.ComparisonError, KeyError, TypeError, ValueError) as exc:
        raise ManualReviewError("matched comparison recomputation failed") from exc
    if (
        rebuilt_comparison != comparison_report
        or canonical_json_bytes(rebuilt_comparison) != comparison_bytes
    ):
        raise ManualReviewError("matched comparison artifact differs from recomputed bytes")
    matched_contract = comparison_report.get("matched_contract")
    interpretation = comparison_report.get("interpretation")
    if (
        comparison_report.get("automatic_source_approval") is not False
        or comparison_report.get("source_selection_result") is not None
        or not isinstance(matched_contract, Mapping)
        or matched_contract.get("common_score_slice_sufficient") is not False
        or not isinstance(interpretation, Mapping)
        or interpretation.get("manual_review_required") is not True
    ):
        raise ManualReviewError("matched comparison is not the frozen inconclusive decision")
    items = _select_items(
        arm_records={"primary": primary_records, "stack_comparison": stack_records},
        arm_caches={"primary": primary_cache, "stack_comparison": stack_cache},
    )
    attestation_spec = spec["manual_attestation"]
    queue = {
        "schema_version": 2,
        "kind": QUEUE_KIND,
        "status": "READY_FOR_BLINDED_REVIEW",
        "decision_scope": DECISION_SCOPE,
        "review_session_id": attestation_spec["review_session_id"],
        "spec_sha256": spec_sha,
        "record_count": 48,
        "records": [{"review_id": item.review_id} for item in items],
    }
    sensitive = _sensitive_values((primary_manifest, stack_manifest))
    _assert_no_private_values(canonical_json_bytes(queue), sensitive)
    return VerifiedInputs(
        spec=spec,
        spec_sha256=spec_sha,
        items=items,
        queue=queue,
        sensitive_values=sensitive,
    )


def _mode(path: Path) -> int:
    try:
        with open_regular_file_nofollow(path) as descriptor:
            return stat.S_IMODE(os.fstat(descriptor).st_mode)
    except ManualReviewIOError as exc:
        raise ManualReviewError("manual-review output is not a regular no-follow file") from exc


def _session_entries(session_dir: Path) -> set[str]:
    absolute = _inside_project(session_dir, label="manual-review session")
    try:
        with open_directory_nofollow(absolute) as descriptor:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise ManualReviewError("manual-review session directory mode is not 0700")
            return set(os.listdir(descriptor))
    except ManualReviewIOError as exc:
        raise ManualReviewError("manual-review session directory is unsafe") from exc


def _create_session_directory(session_dir: Path) -> None:
    absolute = _inside_project(session_dir, label="manual-review session")
    parent = absolute.parent
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        with open_directory_nofollow(parent) as parent_fd:
            try:
                os.mkdir(absolute.name, 0o700, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise ManualReviewError("manual-review session namespace already exists") from exc
            session_fd = os.open(absolute.name, flags, dir_fd=parent_fd)
            try:
                os.fchmod(session_fd, 0o700)
                metadata = os.fstat(session_fd)
                if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
                    raise ManualReviewError("new manual-review session is not a private directory")
                os.fsync(session_fd)
                os.fsync(parent_fd)
            finally:
                os.close(session_fd)
    except ManualReviewIOError as exc:
        raise ManualReviewError("cannot securely create manual-review session") from exc


def _unlink_at_session(session_dir: Path, name: str) -> None:
    try:
        with open_directory_nofollow(session_dir) as descriptor:
            os.unlink(name, dir_fd=descriptor)
            os.fsync(descriptor)
    except (FileNotFoundError, ManualReviewIOError):
        return


def _cleanup_new_session(session_dir: Path) -> None:
    absolute = _inside_project(session_dir, label="manual-review session")
    try:
        with open_directory_nofollow(absolute) as descriptor:
            names = set(os.listdir(descriptor))
            if any("/" in name or "\\" in name for name in names):
                return
            for name in names:
                os.unlink(name, dir_fd=descriptor)
            os.fsync(descriptor)
        with open_directory_nofollow(absolute.parent) as parent_fd:
            os.rmdir(absolute.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    except (OSError, ManualReviewIOError):
        return


def _artifact_identity(path: Path) -> dict[str, str]:
    data = _secure_bytes(path, label="manual-review output")
    digest = hashlib.sha256(data).hexdigest()
    match = _ADDRESSED_RE.fullmatch(path.name)
    if match is None or match.group(1) != digest or _mode(path) != 0o600:
        raise ManualReviewError("manual-review output identity or mode drifted")
    return {"filename": path.name, "sha256": digest}


def _publish_with_marker(
    session_dir: Path,
    *,
    stem: str,
    value: Mapping[str, Any],
    marker_name: str,
    marker_kind: str,
) -> Path:
    artifact: Path | None = None
    marker_created = False
    try:
        artifact = publish_content_addressed_json(session_dir, stem=stem, value=value)
        identity = _artifact_identity(artifact)
        marker = {
            "schema_version": 1,
            "kind": marker_kind,
            "status": "COMMITTED",
            "artifact": identity,
        }
        marker_path = create_session_marker(
            session_dir,
            name=marker_name,
            value=marker,
        )
        marker_created = True
        if _mode(marker_path) != 0o600:
            raise ManualReviewError("manual-review commit marker mode drifted")
        observed = _strict_json(
            _secure_bytes(marker_path, label="manual-review commit marker", max_bytes=64 * 1024),
            label="manual-review commit marker",
        )
        if observed != marker or _artifact_identity(artifact) != identity:
            raise ManualReviewError("manual-review commit marker readback drifted")
        return artifact
    except (ManualReviewError, ManualReviewIOError, OSError) as exc:
        if marker_created:
            _unlink_at_session(session_dir, marker_name)
        if artifact is not None:
            _unlink_at_session(session_dir, artifact.name)
        raise ManualReviewError("atomic manual-review publication failed") from exc


def _load_marker_artifact(
    session_dir: Path,
    *,
    marker_name: str,
    marker_kind: str,
    artifact_kind: str,
) -> tuple[dict[str, Any], Path, str]:
    marker_path = session_dir / marker_name
    marker = _strict_json(
        _secure_bytes(marker_path, label="manual-review commit marker", max_bytes=64 * 1024),
        label="manual-review commit marker",
    )
    if (
        marker.get("schema_version") != 1
        or marker.get("kind") != marker_kind
        or marker.get("status") != "COMMITTED"
        or not isinstance(marker.get("artifact"), Mapping)
        or _mode(marker_path) != 0o600
    ):
        raise ManualReviewError("manual-review commit marker is malformed")
    identity = marker["artifact"]
    filename = identity.get("filename")
    digest = identity.get("sha256")
    if not isinstance(filename, str) or not isinstance(digest, str):
        raise ManualReviewError("manual-review commit marker identity is malformed")
    artifact_path = session_dir / filename
    artifact, data = _load_sha_bound_json(
        artifact_path,
        expected_sha256=digest,
        label="committed manual-review artifact",
    )
    if artifact.get("kind") != artifact_kind or _mode(artifact_path) != 0o600:
        raise ManualReviewError("committed manual-review artifact kind or mode drifted")
    if canonical_json_bytes(artifact) != data:
        raise ManualReviewError("committed manual-review artifact is not canonical JSON")
    return artifact, artifact_path, digest


def _queue_identity(verified: VerifiedInputs) -> dict[str, str]:
    data = canonical_json_bytes(verified.queue)
    digest = hashlib.sha256(data).hexdigest()
    stem = verified.spec["outputs"]["queue_stem"]
    return {"filename": f"{stem}.sha256-{digest}.json", "sha256": digest}


def _validate_queue(session_dir: Path, verified: VerifiedInputs) -> Path:
    expected = _queue_identity(verified)
    marker, _, _ = _load_marker_artifact(
        session_dir,
        marker_name=QUEUE_MARKER,
        marker_kind="petitgpt_python_p1_manual_review_queue_commit",
        artifact_kind=QUEUE_KIND,
    )
    del marker
    queue_path = session_dir / expected["filename"]
    queue, data = _load_sha_bound_json(
        queue_path,
        expected_sha256=expected["sha256"],
        label="blinded review queue",
    )
    if queue != verified.queue or canonical_json_bytes(queue) != data or _mode(queue_path) != 0o600:
        raise ManualReviewError("blinded review queue differs from deterministic reconstruction")
    _assert_no_private_values(data, verified.sensitive_values)
    return queue_path


def prepare_queue(config: ManualReviewConfig) -> Path:
    """Fully verify inputs, then publish a source-free opaque queue exactly once."""
    verified = verify_inputs(config)
    _create_session_directory(config.session_dir)
    try:
        queue_path = _publish_with_marker(
            config.session_dir,
            stem=verified.spec["outputs"]["queue_stem"],
            value=verified.queue,
            marker_name=QUEUE_MARKER,
            marker_kind="petitgpt_python_p1_manual_review_queue_commit",
        )
        expected_entries = {queue_path.name, QUEUE_MARKER}
        if _session_entries(config.session_dir) != expected_entries:
            raise ManualReviewError("new manual-review session contains unexpected files")
        _validate_queue(config.session_dir, verified)
        return queue_path
    except Exception:
        _cleanup_new_session(config.session_dir)
        raise


def _write_tty(descriptor: int, text: str) -> None:
    data = text.encode("utf-8", errors="strict")
    view = memoryview(data)
    written = 0
    try:
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise ManualReviewError("controlling TTY write made no progress")
            written += count
    except OSError as exc:
        raise ManualReviewError("controlling TTY write failed") from exc


def _read_tty_line(descriptor: int, *, maximum_bytes: int = 256) -> str:
    data = bytearray()
    try:
        while len(data) <= maximum_bytes:
            chunk = os.read(descriptor, 1)
            if not chunk:
                raise ManualReviewError("controlling TTY closed during review")
            if chunk == b"\n":
                return data.decode("ascii", errors="strict").strip()
            data.extend(chunk)
    except (OSError, UnicodeError) as exc:
        raise ManualReviewError("controlling TTY input is invalid") from exc
    raise ManualReviewError("controlling TTY input exceeded its fixed bound")


def _terminal_units(text: str, *, columns: int) -> list[str]:
    width = max(20, columns)
    units: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line[:-1] if line.endswith("\n") else line
        ending = "\n" if line.endswith("\n") else ""
        if not body:
            units.append(ending or "\n")
            continue
        while body:
            units.append(body[:width])
            body = body[width:]
        if ending:
            units[-1] += ending
    return units or ["\n"]


def _page_source(descriptor: int, *, review_id: str, source: str) -> None:
    if "\x1b" in source or "\x00" in source or "\r" in source or "\b" in source:
        raise ManualReviewError("terminal sanitizer emitted a forbidden control character")
    try:
        size = os.get_terminal_size(descriptor)
        rows, columns = max(8, size.lines), max(20, size.columns)
    except OSError:
        rows, columns = 24, 80
    _write_tty(descriptor, f"\nReview ID: {review_id}\n--- source begins ---\n")
    units = _terminal_units(source, columns=columns)
    page_rows = max(1, rows - 6)
    for offset in range(0, len(units), page_rows):
        _write_tty(descriptor, "".join(units[offset : offset + page_rows]))
        if offset + page_rows < len(units):
            _write_tty(descriptor, "\n[Press Enter for the next source page] ")
            if _read_tty_line(descriptor) != "":
                raise ManualReviewError("page prompt accepts Enter only")
    _write_tty(descriptor, "\n--- source ends ---\n")


def _tty_labels(items: Sequence[BlindReviewItem]) -> list[str]:
    if not all(os.isatty(fd) for fd in (0, 1, 2)):
        raise ManualReviewError("stdin/stdout/stderr must all be the controlling TTY")
    flags = os.O_RDWR | os.O_NOCTTY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open("/dev/tty", flags)
    except OSError as exc:
        raise ManualReviewError("cannot open the RunPod controlling TTY") from exc
    old_handlers: dict[int, Any] = {}
    original_attributes: list[Any] | None = None

    def interrupted(signum: int, frame: Any) -> NoReturn:
        del signum, frame
        raise ReviewInterrupted("manual review was interrupted before sealing")

    try:
        tty_stat = os.fstat(descriptor)
        if not os.isatty(descriptor) or any(os.fstat(fd).st_rdev != tty_stat.st_rdev for fd in (0, 1, 2)):
            raise ManualReviewError("standard streams do not match /dev/tty")
        original_attributes = termios.tcgetattr(descriptor)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        _write_tty(
            descriptor,
            "\nBlinded Python P1 quality review on RunPod.\n"
            "MANUAL_KEEP = independently suitable reusable Python source.\n"
            "MANUAL_REJECT = independently unsuitable for the frozen quality gates.\n"
            "UNREVIEWABLE = the displayed source cannot be judged.\n"
            "No arm, provenance, automatic outcome, gate, score, or identifier is shown.\n"
            "Type exactly RECORDING_DISABLED to confirm terminal recording is off: ",
        )
        if _read_tty_line(descriptor) != "RECORDING_DISABLED":
            raise ManualReviewError("terminal-recording acknowledgement was not exact")
        labels: list[str] = []
        for item in items:
            try:
                decoded = item.raw.decode("utf-8", errors="strict")
                source = sanitize_terminal_text(decoded)
            except (UnicodeError, ManualReviewIOError) as exc:
                raise ManualReviewError("selected source could not be safely rendered") from exc
            _page_source(descriptor, review_id=item.review_id, source=source)
            _write_tty(descriptor, "Label [K=MANUAL_KEEP, R=MANUAL_REJECT, U=UNREVIEWABLE]: ")
            response = _read_tty_line(descriptor).upper()
            label = {"K": "MANUAL_KEEP", "R": "MANUAL_REJECT", "U": "UNREVIEWABLE"}.get(
                response
            )
            if label is None:
                raise ManualReviewError("invalid label; the in-memory review was not sealed")
            labels.append(label)
        return labels
    finally:
        if original_attributes is not None:
            try:
                termios.tcsetattr(descriptor, termios.TCSANOW, original_attributes)
            except termios.error:
                pass
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        os.close(descriptor)


LabelProvider = Callable[[str, str], str]


def review_and_seal(
    config: ManualReviewConfig,
    *,
    label_provider: LabelProvider | None = None,
) -> Path:
    """Run the blinded review and publish one sealed, source-free attestation."""
    if label_provider is not None and config.enforce_environment:
        raise ManualReviewError("injected label providers are test-only")
    verified = verify_inputs(config)
    with SessionLock(config.session_dir, LOCK_NAME):
        queue_path = _validate_queue(config.session_dir, verified)
        entries = _session_entries(config.session_dir)
        expected = {queue_path.name, QUEUE_MARKER, LOCK_NAME}
        if entries != expected:
            raise ManualReviewError("manual-review session is not pristine before sealing")
        if label_provider is None:
            blind_items = tuple(
                BlindReviewItem(review_id=item.review_id, raw=item.raw)
                for item in verified.items
            )
            labels = _tty_labels(blind_items)
        else:
            labels = []
            for item in verified.items:
                source = sanitize_terminal_text(item.raw.decode("utf-8", errors="strict"))
                label = label_provider(item.review_id, source)
                if label not in ALLOWED_LABELS:
                    raise ManualReviewError("test label provider returned an invalid label")
                labels.append(label)
        if len(labels) != 48:
            raise ManualReviewError("all 48 labels are required before sealing")
        records = [
            {"review_id": item.review_id, "label": label}
            for item, label in zip(verified.items, labels, strict=True)
        ]
        if len({record["review_id"] for record in records}) != 48:
            raise ManualReviewError("attestation review IDs are not unique")
        attestation = {
            "schema_version": 2,
            "kind": ATTESTATION_KIND,
            "status": "SEALED_BEFORE_UNBLINDING",
            "decision_scope": DECISION_SCOPE,
            "review_session_id": verified.spec["manual_attestation"]["review_session_id"],
            "spec_sha256": verified.spec_sha256,
            "queue": _artifact_identity(queue_path),
            "label_count": 48,
            "allowed_labels": list(ALLOWED_LABELS),
            "records": records,
            "reviewer_identity_persisted": False,
            "free_text_fields": 0,
            "source_characters_persisted": 0,
            "automatic_truth_exposed_before_sealing": False,
        }
        attestation_data = canonical_json_bytes(attestation)
        _assert_no_private_values(attestation_data, verified.sensitive_values)
        attestation_path = _publish_with_marker(
            config.session_dir,
            stem=verified.spec["outputs"]["attestation_stem"],
            value=attestation,
            marker_name=ATTESTATION_MARKER,
            marker_kind="petitgpt_python_p1_manual_review_attestation_commit",
        )
        expected_after_seal = {
            queue_path.name,
            QUEUE_MARKER,
            attestation_path.name,
            ATTESTATION_MARKER,
            LOCK_NAME,
        }
        if _session_entries(config.session_dir) != expected_after_seal:
            raise ManualReviewError("manual-review session changed while sealing")
        return attestation_path


def _validate_attestation(
    attestation: Mapping[str, Any],
    *,
    verified: VerifiedInputs,
    queue_path: Path,
) -> dict[str, str]:
    expected_keys = {
        "schema_version",
        "kind",
        "status",
        "decision_scope",
        "review_session_id",
        "spec_sha256",
        "queue",
        "label_count",
        "allowed_labels",
        "records",
        "reviewer_identity_persisted",
        "free_text_fields",
        "source_characters_persisted",
        "automatic_truth_exposed_before_sealing",
    }
    if (
        set(attestation) != expected_keys
        or attestation.get("schema_version") != 2
        or attestation.get("kind") != ATTESTATION_KIND
        or attestation.get("status") != "SEALED_BEFORE_UNBLINDING"
        or attestation.get("decision_scope") != DECISION_SCOPE
        or attestation.get("review_session_id")
        != verified.spec["manual_attestation"]["review_session_id"]
        or attestation.get("spec_sha256") != verified.spec_sha256
        or attestation.get("queue") != _artifact_identity(queue_path)
        or attestation.get("label_count") != 48
        or attestation.get("allowed_labels") != list(ALLOWED_LABELS)
        or attestation.get("reviewer_identity_persisted") is not False
        or attestation.get("free_text_fields") != 0
        or attestation.get("source_characters_persisted") != 0
        or attestation.get("automatic_truth_exposed_before_sealing") is not False
    ):
        raise ManualReviewError("sealed attestation contract drifted")
    records = attestation.get("records")
    if not isinstance(records, list) or len(records) != 48:
        raise ManualReviewError("sealed attestation does not contain 48 records")
    labels: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"review_id", "label"}:
            raise ManualReviewError("sealed attestation record schema drifted")
        review_id = record.get("review_id")
        label = record.get("label")
        if (
            not isinstance(review_id, str)
            or review_id in labels
            or label not in ALLOWED_LABELS
        ):
            raise ManualReviewError("sealed attestation label identity drifted")
        labels[review_id] = str(label)
    if set(labels) != {item.review_id for item in verified.items}:
        raise ManualReviewError("sealed attestation IDs differ from the blinded queue")
    return labels


def unblind_and_publish(config: ManualReviewConfig) -> Path:
    """Unblind only a sealed attestation and publish aggregate confusion counts."""
    verified = verify_inputs(config)
    with SessionLock(config.session_dir, LOCK_NAME):
        queue_path = _validate_queue(config.session_dir, verified)
        attestation, attestation_path, attestation_sha = _load_marker_artifact(
            config.session_dir,
            marker_name=ATTESTATION_MARKER,
            marker_kind="petitgpt_python_p1_manual_review_attestation_commit",
            artifact_kind=ATTESTATION_KIND,
        )
        expected_before_result = {
            queue_path.name,
            QUEUE_MARKER,
            attestation_path.name,
            ATTESTATION_MARKER,
            LOCK_NAME,
        }
        if _session_entries(config.session_dir) != expected_before_result:
            raise ManualReviewError("manual-review session is not pristine before unblinding")
        labels = _validate_attestation(attestation, verified=verified, queue_path=queue_path)
        confusion: dict[str, dict[str, dict[str, int]]] = {
            arm: {
                outcome: {label: 0 for label in ALLOWED_LABELS}
                for outcome in AUTOMATIC_OUTCOMES
            }
            for arm in ("primary", "stack_comparison")
        }
        mismatches = 0
        unreviewable = 0
        for item in verified.items:
            label = labels[item.review_id]
            confusion[item.arm][item.automatic_outcome][label] += 1
            expected_label = (
                "MANUAL_KEEP" if item.automatic_outcome == "keep" else "MANUAL_REJECT"
            )
            mismatches += label != expected_label
            unreviewable += label == "UNREVIEWABLE"
        passed = mismatches == 0 and unreviewable == 0
        result = {
            "schema_version": 2,
            "kind": RESULT_KIND,
            "status": (
                "CLASSIFICATION_SPOT_CHECK_MATCHED"
                if passed
                else "BLOCKED_MANUAL_REVIEW_EXCEPTION"
            ),
            "decision_scope": DECISION_SCOPE,
            "review_session_id": verified.spec["manual_attestation"]["review_session_id"],
            "spec_sha256": verified.spec_sha256,
            "sealed_attestation_sha256": attestation_sha,
            "aggregate_confusion_counts": confusion,
            "reviewed_records": 48,
            "manual_label_mismatches": mismatches,
            "unreviewable_records": unreviewable,
            "manual_gate_passed": passed,
            "individual_truth_rows_persisted": 0,
            "automatic_source_approval": False,
            "source_selection_result": None,
            "license_clearance": False,
            "token_quota_approval": False,
            "bulk_candidate_construction_approval": False,
            "p1b_authorized": False,
            "matched_comparison_remains_inconclusive": True,
        }
        result_data = canonical_json_bytes(result)
        _assert_no_private_values(result_data, verified.sensitive_values)
        result_path = _publish_with_marker(
            config.session_dir,
            stem=verified.spec["outputs"]["result_stem"],
            value=result,
            marker_name=RESULT_MARKER,
            marker_kind="petitgpt_python_p1_manual_review_result_commit",
        )
        if _session_entries(config.session_dir) != expected_before_result | {
            result_path.name,
            RESULT_MARKER,
        }:
            raise ManualReviewError("manual-review session changed while publishing result")
        return result_path


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", dest="spec_path", type=Path, required=True)
    parser.add_argument("--policy", dest="policy_path", type=Path, required=True)
    parser.add_argument("--comparison_report", type=Path, required=True)
    parser.add_argument("--session_dir", type=Path, required=True)
    parser.add_argument("--expected_generator_commit", required=True)
    for prefix in ("primary", "stack"):
        parser.add_argument(f"--{prefix}_manifest", type=Path, required=True)
        parser.add_argument(f"--{prefix}_collection_report", type=Path, required=True)
        parser.add_argument(f"--{prefix}_replay_report", type=Path, required=True)
        parser.add_argument(f"--{prefix}_analysis_report", type=Path, required=True)
        parser.add_argument(f"--{prefix}_cache_dir", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed blinded manual review for frozen Python P1 artifacts."
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("prepare", "review", "unblind"):
        phase_parser = subparsers.add_parser(phase)
        _add_common_arguments(phase_parser)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> ManualReviewConfig:
    return ManualReviewConfig(
        spec_path=args.spec_path,
        policy_path=args.policy_path,
        comparison_report=args.comparison_report,
        session_dir=args.session_dir,
        expected_generator_commit=args.expected_generator_commit,
        primary=ArmInputs(
            role="primary",
            manifest=args.primary_manifest,
            collection_report=args.primary_collection_report,
            replay_report=args.primary_replay_report,
            analysis_report=args.primary_analysis_report,
            cache_dir=args.primary_cache_dir,
        ),
        stack_comparison=ArmInputs(
            role="stack_comparison",
            manifest=args.stack_manifest,
            collection_report=args.stack_collection_report,
            replay_report=args.stack_replay_report,
            analysis_report=args.stack_analysis_report,
            cache_dir=args.stack_cache_dir,
        ),
    )


def main() -> None:
    args = _parse_args()
    try:
        disable_core_dumps()
        install_zero_network_guards()
        config = _config_from_args(args)
        if args.phase == "prepare":
            path = prepare_queue(config)
        elif args.phase == "review":
            path = review_and_seal(config)
        elif args.phase == "unblind":
            path = unblind_and_publish(config)
        else:  # pragma: no cover - argparse enforces the choices.
            raise AssertionError("unknown manual-review phase")
    except Exception:
        os.write(2, b"manual review stopped fail-closed; no output was overwritten\n")
        raise SystemExit(2) from None
    os.write(1, (os.fspath(path) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
