"""Plan exact deterministic Stage A/B pretraining step boundaries.

The planner derives its counts from :class:`PackedBinDataset` itself, so the
launch configuration and the production data loader cannot silently disagree
about virtual-stream tails or block counts. One exposure is a no-replacement
pass; larger integer exposure counts are explicit named replay experiments.
Each expanded stage is floored once to a complete optimizer step, and every
dropped block is reported.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.dataset_pretrain import (  # noqa: E402
    PackedBinDataset,
    validate_shard_release,
)
from pretrain.stage_m_contract_v1 import (  # noqa: E402
    exclusion_authority,
    require_identical_exclusion_authorities,
)
from pretrain.stage_p_native_provenance_v1 import (  # noqa: E402
    LEGACY_CHAIN_KIND,
    NATIVE_CHAIN_KIND,
    PROVENANCE_CHAIN_KINDS,
    assert_single_branch,
    post_merge_data_branch_identity_sha256,
    validate_native_chain,
)
from src.special_tokens import (  # noqa: E402
    CANONICAL_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    assert_tokenizer_contract,
)

SCHEMA_VERSION = 3
_NOMINAL_CHECKPOINT_TOKEN_TARGETS = (
    ("nominal_1b_tokens", 1_000_000_000),
    ("nominal_3b_tokens", 3_000_000_000),
    ("nominal_6b_tokens", 6_000_000_000),
    ("nominal_10b_tokens", 10_000_000_000),
    ("nominal_11_5b_tokens", 11_500_000_000),
    ("nominal_13b_tokens", 13_000_000_000),
)
_REFERENCE_RELEASE_KIND = "petitgpt_cross_stage_reference_validation"
_TOKENIZER_RELEASE_KIND = "petitgpt_tokenizer_release"
_SELECTION_MANIFEST_KIND = "petitgpt_pretrain_document_selection"
_REFERENCE_RESERVE_KIND = "petitgpt_reference_validation_reserve"
_EXCLUSION_MANIFEST_KIND = "petitgpt_reference_validation_exclusions"
_CLEANED_TEXT_HASH_ALGORITHM = "sha256-cleaned-text-utf8-v1"
_RAW_TEXT_HASH_ALGORITHM = "sha256-raw-text-utf8-v1"
_CANONICAL_FINGERPRINT_ALGORITHM = "sha256-domain-separated-cleaned-text-v1"
_SELECTION_IDENTITY_ALGORITHMS = {
    "raw_sha256": _RAW_TEXT_HASH_ALGORITHM,
    "cleaned_sha256": _CLEANED_TEXT_HASH_ALGORITHM,
    "canonical_fingerprint": _CANONICAL_FINGERPRINT_ALGORITHM,
}
_EVIDENCE_READ_CHUNK_BYTES = 1024 * 1024
STAGE_B_SELECTION_STAGES = ("stage_b", "control")


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} must be a JSON object")
    return value


def _require_manifest_int(
    value: Any,
    *,
    field: str,
    positive: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be an integer, got {value!r}")
    if value < 0 or (positive and value <= 0):
        relation = "positive" if positive else "non-negative"
        raise RuntimeError(f"{field} must be {relation}, got {value}")
    return int(value)


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"{field} must be a lowercase SHA-256")
    return value


def _stat_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _read_regular_evidence(
    path: str | Path,
    *,
    label: str,
    collect_bytes: bool = True,
) -> tuple[Path, bytes, str, int]:
    """Read one stable regular file without following a final-component symlink."""
    candidate = Path(path).expanduser()
    try:
        before = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {candidate}")

    try:
        resolved_before = candidate.resolve(strict=True)
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(f"cannot open stable {label} {candidate}: {exc}") from exc

    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise RuntimeError(f"{label} changed to a non-regular file while opening: {candidate}")
        if (opened_before.st_dev, opened_before.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{label} changed while opening: {candidate}")

        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _EVIDENCE_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            if collect_bytes:
                chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as exc:
        raise RuntimeError(f"cannot read {label} {candidate}: {exc}") from exc
    finally:
        os.close(descriptor)

    try:
        after = candidate.stat(follow_symlinks=False)
        resolved_after = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{label} disappeared while being read: {candidate}") from exc
    signatures = {
        _stat_signature(before),
        _stat_signature(opened_before),
        _stat_signature(opened_after),
        _stat_signature(after),
    }
    if len(signatures) != 1 or resolved_after != resolved_before:
        raise RuntimeError(f"{label} changed while being read: {candidate}")
    raw = b"".join(chunks)
    if collect_bytes and len(raw) != opened_after.st_size:
        raise RuntimeError(f"{label} size changed while being read: {candidate}")
    return resolved_after, raw, digest.hexdigest(), int(opened_after.st_size)


def _verify_file_evidence(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[Path, str, int]:
    expected_sha256 = _require_sha256(expected_sha256, field=f"{label}.sha256")
    expected_size_bytes = _require_manifest_int(
        expected_size_bytes,
        field=f"{label}.size_bytes",
        positive=True,
    )
    resolved, _raw, actual_sha256, actual_size_bytes = _read_regular_evidence(
        path,
        label=label,
        collect_bytes=False,
    )
    if actual_size_bytes != expected_size_bytes:
        raise RuntimeError(
            f"{label} size disagrees with its manifest: "
            f"expected={expected_size_bytes}, actual={actual_size_bytes}"
        )
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 disagrees with its manifest")
    return resolved, actual_sha256, actual_size_bytes


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_artifact(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[Path, dict[str, Any], str, int]:
    resolved, raw, actual_sha256, actual_size_bytes = _read_regular_evidence(
        path,
        label=label,
    )
    if expected_sha256 is not None:
        expected_sha256 = _require_sha256(expected_sha256, field=f"{label}.sha256")
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{label} SHA-256 disagrees with its manifest")
    if expected_size_bytes is not None:
        expected_size_bytes = _require_manifest_int(
            expected_size_bytes,
            field=f"{label}.size_bytes",
            positive=True,
        )
        if actual_size_bytes != expected_size_bytes:
            raise RuntimeError(
                f"{label} size disagrees with its manifest: "
                f"expected={expected_size_bytes}, actual={actual_size_bytes}"
            )
    try:
        payload = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {resolved}")
    return resolved, payload, actual_sha256, actual_size_bytes


def _require_manifest_header(
    payload: Mapping[str, Any],
    *,
    label: str,
    schema_version: int,
    kind: str,
) -> None:
    if payload.get("schema_version") != schema_version:
        raise RuntimeError(
            f"{label} schema_version must be {schema_version}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("kind") != kind:
        raise RuntimeError(f"{label} kind must be {kind!r}, got {payload.get('kind')!r}")
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} status must be 'complete', got {payload.get('status')!r}")


def _resolve_relative_evidence_path(
    root: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise RuntimeError(f"{label} escapes its artifact root")
    candidate = root / relative
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"{label} escapes its artifact root")
    return candidate


def _require_evidence_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty path")
    return Path(value).expanduser()


def _validate_exclusion_evidence(
    entry: Mapping[str, Any],
    *,
    label: str,
    collection_cleaning: Mapping[str, Any],
    path_field: str,
    sha_field: str,
    size_field: str,
    resolved_field: str | None,
) -> tuple[dict[str, Any], frozenset[str]]:
    evidence_path = _require_evidence_path(entry.get(path_field), label=f"{label}.{path_field}")
    expected_sha256 = _require_sha256(entry.get(sha_field), field=f"{label}.{sha_field}")
    expected_size_bytes = _require_manifest_int(
        entry.get(size_field),
        field=f"{label}.{size_field}",
        positive=True,
    )
    resolved, payload, actual_sha256, actual_size_bytes = _read_json_artifact(
        evidence_path,
        label=label,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    if resolved_field is not None:
        recorded_resolved_value = entry.get(resolved_field)
        if not isinstance(recorded_resolved_value, str) or not recorded_resolved_value:
            raise RuntimeError(f"{label}.{resolved_field} must be a non-empty absolute path")
        recorded_resolved = Path(recorded_resolved_value).expanduser()
        if not recorded_resolved.is_absolute() or recorded_resolved != resolved:
            raise RuntimeError(f"{label}.{resolved_field} disagrees with the evidence path")

    if payload.get("schema_version") != 1:
        raise RuntimeError(f"{label} schema_version must be 1")
    if payload.get("kind") != _EXCLUSION_MANIFEST_KIND:
        raise RuntimeError(f"{label} has an unsupported kind")
    if payload.get("hash_algorithm") != _CLEANED_TEXT_HASH_ALGORITHM:
        raise RuntimeError(f"{label} has an unsupported hash algorithm")
    if payload.get("cleaning") != collection_cleaning:
        raise RuntimeError(f"{label} cleaning contract disagrees with its parent manifest")
    if entry.get("kind") != payload.get("kind"):
        raise RuntimeError(f"{label}.kind disagrees with the evidence file")
    if entry.get("hash_algorithm") != payload.get("hash_algorithm"):
        raise RuntimeError(f"{label}.hash_algorithm disagrees with the evidence file")
    if entry.get("cleaning") is not None and entry.get("cleaning") != payload.get("cleaning"):
        raise RuntimeError(f"{label}.cleaning disagrees with the evidence file")

    declared_hash_count = _require_manifest_int(
        entry.get("hash_count"),
        field=f"{label}.hash_count",
        positive=True,
    )
    payload_hash_count = _require_manifest_int(
        payload.get("hash_count"),
        field=f"{label} payload.hash_count",
        positive=True,
    )
    raw_hashes = payload.get("hashes")
    if not isinstance(raw_hashes, list) or len(raw_hashes) != payload_hash_count:
        raise RuntimeError(f"{label} payload.hash_count does not match hashes")
    hashes = [
        _require_sha256(value, field=f"{label} payload.hashes[{index}]")
        for index, value in enumerate(raw_hashes)
    ]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError(f"{label} contains duplicate cleaned-text hashes")
    if declared_hash_count != payload_hash_count:
        raise RuntimeError(f"{label}.hash_count disagrees with the evidence file")

    return (
        {
            "path": str(resolved),
            "sha256": actual_sha256,
            "size_bytes": actual_size_bytes,
            "hash_count": payload_hash_count,
        },
        frozenset(hashes),
    )


def _validate_exclusion_collection(
    value: Any,
    *,
    label: str,
    activation_field: str,
    path_field: str,
    sha_field: str,
    size_field: str,
    resolved_field: str | None,
    require_top_level_kind: bool,
    require_entry_enabled: bool,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    collection = _require_mapping(value, field=label)
    if collection.get(activation_field) is not True:
        raise RuntimeError(f"{label}.{activation_field} must be true")
    if require_top_level_kind and collection.get("kind") != _EXCLUSION_MANIFEST_KIND:
        raise RuntimeError(
            f"{label}.kind must be {_EXCLUSION_MANIFEST_KIND!r}, got {collection.get('kind')!r}"
        )
    if collection.get("hash_algorithm") != _CLEANED_TEXT_HASH_ALGORITHM:
        raise RuntimeError(f"{label}.hash_algorithm must be {_CLEANED_TEXT_HASH_ALGORITHM!r}")
    cleaning = _require_mapping(collection.get("cleaning"), field=f"{label}.cleaning")
    manifest_count = _require_manifest_int(
        collection.get("manifest_count"),
        field=f"{label}.manifest_count",
        positive=True,
    )
    union_hash_count = _require_manifest_int(
        collection.get("union_hash_count"),
        field=f"{label}.union_hash_count",
        positive=True,
    )
    entries = collection.get("manifests")
    if not isinstance(entries, list) or len(entries) != manifest_count:
        raise RuntimeError(f"{label}.manifest_count does not match manifests")

    sha256s: list[str] = []
    evidence_records: list[dict[str, Any]] = []
    union_hashes: set[str] = set()
    total_memberships = 0
    for index, raw_entry in enumerate(entries):
        entry_label = f"{label}.manifests[{index}]"
        entry = _require_mapping(raw_entry, field=entry_label)
        if require_entry_enabled and entry.get("enabled") is not True:
            raise RuntimeError(f"{entry_label}.enabled must be true")
        if entry.get("kind") != _EXCLUSION_MANIFEST_KIND:
            raise RuntimeError(f"{entry_label}.kind must be {_EXCLUSION_MANIFEST_KIND!r}")
        if entry.get("hash_algorithm") != _CLEANED_TEXT_HASH_ALGORITHM:
            raise RuntimeError(
                f"{entry_label}.hash_algorithm must be {_CLEANED_TEXT_HASH_ALGORITHM!r}"
            )
        evidence, hashes = _validate_exclusion_evidence(
            entry,
            label=entry_label,
            collection_cleaning=cleaning,
            path_field=path_field,
            sha_field=sha_field,
            size_field=size_field,
            resolved_field=resolved_field,
        )
        sha256s.append(evidence["sha256"])
        evidence_records.append(evidence)
        union_hashes.update(hashes)
        total_memberships += len(hashes)

    if len(set(sha256s)) != len(sha256s):
        raise RuntimeError(f"{label} contains duplicate exclusion manifest SHA-256 values")
    if len(union_hashes) != union_hash_count:
        raise RuntimeError(
            f"{label}.union_hash_count disagrees with evidence files: "
            f"expected={union_hash_count}, actual={len(union_hashes)}"
        )
    duplicate_memberships = collection.get("cross_manifest_duplicate_memberships")
    if duplicate_memberships is not None and _require_manifest_int(
        duplicate_memberships,
        field=f"{label}.cross_manifest_duplicate_memberships",
    ) != total_memberships - len(union_hashes):
        raise RuntimeError(
            f"{label}.cross_manifest_duplicate_memberships disagrees with evidence files"
        )
    evidence_records.sort(key=lambda item: item["sha256"])
    return tuple(sorted(sha256s)), evidence_records


def _selection_outputs_by_stage(
    selection_manifest_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    raw_sources = selection.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError("selection manifest.sources must be a non-empty list")

    selection_root = selection_manifest_path.parent.resolve()
    outputs: dict[str, dict[str, dict[str, Any]]] = {
        "stage_a": {},
        "stage_b": {},
        "control": {},
    }
    source_ids: set[str] = set()
    input_paths: set[str] = set()
    output_paths: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        label = f"selection manifest.sources[{index}]"
        source = _require_mapping(raw_source, field=label)
        stage = source.get("stage")
        if stage not in {"stage_a", "stage_b", "control"}:
            raise RuntimeError(f"{label}.stage is invalid: {stage!r}")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError(f"{label}.source_id must be a non-empty string")
        if source_id in source_ids:
            raise RuntimeError(f"duplicate selection source_id: {source_id!r}")
        source_ids.add(source_id)

        input_candidate = _require_evidence_path(
            source.get("input_path"),
            label=f"{label}.input_path",
        )
        input_resolved, input_sha256, input_size_bytes = _verify_file_evidence(
            input_candidate,
            label=f"{label} input",
            expected_sha256=source.get("input_sha256"),
            expected_size_bytes=source.get("input_size_bytes"),
        )
        input_key = str(input_resolved)
        if input_key in input_paths:
            raise RuntimeError(f"duplicate selection input path: {input_key}")
        input_paths.add(input_key)

        output = _require_mapping(source.get("output"), field=f"{label}.output")
        output_candidate = _resolve_relative_evidence_path(
            selection_root,
            output.get("relative_path"),
            label=f"{label}.output.relative_path",
        )
        output_resolved, output_sha256, output_size_bytes = _verify_file_evidence(
            output_candidate,
            label=f"{label} output",
            expected_sha256=output.get("sha256"),
            expected_size_bytes=output.get("size_bytes"),
        )
        documents = _require_manifest_int(
            output.get("documents"),
            field=f"{label}.output.documents",
            positive=True,
        )
        output_key = str(output_resolved)
        if output_key in output_paths:
            raise RuntimeError(f"duplicate selection output path: {output_key}")
        output_paths.add(output_key)
        outputs[stage][output_key] = {
            "source_id": source_id,
            "path": output_key,
            "sha256": output_sha256,
            "size_bytes": output_size_bytes,
            "documents": documents,
            "input_path": input_key,
            "input_sha256": input_sha256,
            "input_size_bytes": input_size_bytes,
        }

    if not outputs["stage_a"]:
        raise RuntimeError("selection manifest has no stage_a outputs")
    return outputs


def _validate_stage_source_bindings(
    *,
    stage: str,
    release: Mapping[str, Any],
    expected_outputs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_path, manifest, manifest_sha256, _manifest_size_bytes = _read_json_artifact(
        release["manifest_path"],
        label=f"{stage} shard release manifest",
    )
    if manifest_sha256 != release["manifest_sha256"]:
        raise RuntimeError(f"{stage} shard release manifest changed during validation")

    raw_fingerprints = manifest.get("source_fingerprints")
    if not isinstance(raw_fingerprints, dict) or not raw_fingerprints:
        raise RuntimeError(f"{stage} shard release has no source SHA fingerprints")
    observed: dict[str, dict[str, Any]] = {}
    for source_name, raw_fingerprint in raw_fingerprints.items():
        label = f"{stage} source_fingerprints[{source_name!r}]"
        if not isinstance(source_name, str) or not source_name:
            raise RuntimeError(f"{label} has an invalid source key")
        fingerprint = _require_mapping(raw_fingerprint, field=label)
        if fingerprint.get("path") != source_name:
            raise RuntimeError(f"{label}.path does not match its source key")
        resolved_value = fingerprint.get("resolved")
        if not isinstance(resolved_value, str) or not resolved_value:
            raise RuntimeError(f"{label}.resolved must be non-empty")
        resolved_path = str(Path(resolved_value).expanduser().resolve())
        if str(Path(source_name).expanduser().resolve()) != resolved_path:
            raise RuntimeError(f"{label} path and resolved path disagree")
        if resolved_path in observed:
            raise RuntimeError(f"{stage} shard release contains duplicate source paths")
        observed[resolved_path] = {
            "sha256": _require_sha256(fingerprint.get("sha256"), field=f"{label}.sha256"),
            "size_bytes": _require_manifest_int(
                fingerprint.get("size"), field=f"{label}.size", positive=True
            ),
        }

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError(f"{stage} shard release.sources must be non-empty")
    declared_paths: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        label = f"{stage} shard release.sources[{index}]"
        source = _require_mapping(raw_source, field=label)
        source_path = source.get("path")
        if not isinstance(source_path, str) or not source_path:
            raise RuntimeError(f"{label}.path must be non-empty")
        declared_paths.add(str(Path(source_path).expanduser().resolve()))
        weight = source.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise RuntimeError(f"{label}.weight must be positive")
    if declared_paths != set(observed):
        raise RuntimeError(f"{stage} source list and source fingerprints disagree")

    expected_paths = set(expected_outputs)
    if set(observed) != expected_paths:
        missing = sorted(expected_paths - set(observed))
        extra = sorted(set(observed) - expected_paths)
        raise RuntimeError(
            f"{stage} shard sources do not exactly match selection outputs: "
            f"missing={missing}, extra={extra}"
        )

    result: list[dict[str, Any]] = []
    for path in sorted(expected_paths):
        expected = expected_outputs[path]
        actual = observed[path]
        if actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"{stage} source SHA-256 disagrees with selection output: {path}")
        if actual["size_bytes"] != expected["size_bytes"]:
            raise RuntimeError(f"{stage} source size disagrees with selection output: {path}")
        result.append(dict(expected))

    return result


def _validate_reference_reserve_evidence(
    reserve: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    reserve_path = _require_evidence_path(
        reserve.get("reserve_manifest_path"),
        label="reference reserve provenance.reserve_manifest_path",
    )
    reserve_sha256 = _require_sha256(
        reserve.get("reserve_manifest_sha256"),
        field="reference reserve provenance.reserve_manifest_sha256",
    )
    reserve_size_bytes = _require_manifest_int(
        reserve.get("reserve_manifest_size_bytes"),
        field="reference reserve provenance.reserve_manifest_size_bytes",
        positive=True,
    )
    reserve_resolved, reserve_payload, actual_sha256, actual_size_bytes = _read_json_artifact(
        reserve_path,
        label="reference reserve manifest",
        expected_sha256=reserve_sha256,
        expected_size_bytes=reserve_size_bytes,
    )
    _require_manifest_header(
        reserve_payload,
        label="reference reserve manifest",
        schema_version=1,
        kind=_REFERENCE_RESERVE_KIND,
    )
    if (
        reserve_payload.get("immutable") is not True
        or reserve_payload.get("tokenizer_independent") is not True
    ):
        raise RuntimeError("reference reserve manifest is not immutable and tokenizer-independent")

    reserve_cleaning = _require_mapping(
        reserve.get("cleaning"),
        field="reference reserve provenance.cleaning",
    )
    if reserve_payload.get("cleaning") != reserve_cleaning:
        raise RuntimeError("reference reserve cleaning contract disagrees with its manifest")
    outputs = _require_mapping(
        reserve_payload.get("outputs"),
        field="reference reserve manifest.outputs",
    )
    exclusion_name = outputs.get("exclusion_hash_manifest")
    if (
        not isinstance(exclusion_name, str)
        or not exclusion_name
        or Path(exclusion_name).name != exclusion_name
    ):
        raise RuntimeError("reference reserve exclusion output must be a safe sibling filename")

    raw_sources = reserve_payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise RuntimeError("reference reserve manifest.sources must be a non-empty object")
    reserved_hashes: set[str] = set()
    for source_name, raw_source in raw_sources.items():
        label = f"reference reserve manifest.sources[{source_name!r}]"
        if not isinstance(source_name, str) or not source_name:
            raise RuntimeError(f"{label} has an invalid source key")
        source = _require_mapping(raw_source, field=label)
        raw_documents = source.get("reserved_documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise RuntimeError(f"{label}.reserved_documents must be non-empty")
        source_hashes: list[str] = []
        for index, raw_document in enumerate(raw_documents):
            document = _require_mapping(
                raw_document,
                field=f"{label}.reserved_documents[{index}]",
            )
            source_hashes.append(
                _require_sha256(
                    document.get("cleaned_text_sha256"),
                    field=f"{label}.reserved_documents[{index}].cleaned_text_sha256",
                )
            )
        if len(set(source_hashes)) != len(source_hashes):
            raise RuntimeError(f"{label} contains duplicate reserved hashes")
        reserved_hashes.update(source_hashes)
    unique_reserved_hashes = _require_manifest_int(
        reserve_payload.get("unique_reserved_hashes"),
        field="reference reserve manifest.unique_reserved_hashes",
        positive=True,
    )
    if len(reserved_hashes) != unique_reserved_hashes:
        raise RuntimeError(
            "reference reserve unique_reserved_hashes disagrees with reserved_documents"
        )

    reserve_exclusion = _require_mapping(
        reserve.get("reserve_exclusion"),
        field="reference validation release manifest.reserve_provenance.reserve_exclusion",
    )
    if reserve_exclusion.get("enabled") is not True:
        raise RuntimeError("reference reserve exclusion must be enabled")
    exclusion_record, exclusion_hashes = _validate_exclusion_evidence(
        reserve_exclusion,
        label="reference reserve exclusion",
        collection_cleaning=reserve_cleaning,
        path_field="manifest_path",
        sha_field="manifest_sha256",
        size_field="manifest_size_bytes",
        resolved_field="manifest_resolved",
    )
    expected_exclusion_path = (reserve_resolved.parent / exclusion_name).resolve(strict=True)
    if Path(exclusion_record["path"]) != expected_exclusion_path:
        raise RuntimeError(
            "reference reserve exclusion evidence is not the sibling declared by reserve outputs"
        )
    if exclusion_hashes != reserved_hashes:
        raise RuntimeError(
            "reference reserve documents and exclusion hash evidence do not contain the same hashes"
        )

    return (
        {
            "path": str(reserve_resolved),
            "sha256": actual_sha256,
            "size_bytes": actual_size_bytes,
        },
        exclusion_record,
        (exclusion_record["sha256"],),
    )


def _validate_selection_supporting_evidence(
    selection: Mapping[str, Any],
    *,
    tokenizer_json_path: Path,
) -> dict[str, dict[str, Any]]:
    tokenizer = _require_mapping(
        selection.get("tokenizer"),
        field="selection manifest.tokenizer",
    )
    tokenizer_path = _require_evidence_path(
        tokenizer.get("path"),
        label="selection manifest.tokenizer.path",
    )
    tokenizer_resolved, tokenizer_sha256, tokenizer_size_bytes = _verify_file_evidence(
        tokenizer_path,
        label="selection tokenizer",
        expected_sha256=tokenizer.get("sha256"),
        expected_size_bytes=tokenizer.get("size_bytes"),
    )
    if tokenizer_resolved != tokenizer_json_path.resolve(strict=True):
        raise RuntimeError(
            "selection tokenizer path does not identify the tokenizer release tokenizer.json"
        )

    spec = _require_mapping(selection.get("spec"), field="selection manifest.spec")
    spec_path = _require_evidence_path(spec.get("path"), label="selection manifest.spec.path")
    spec_resolved, spec_payload, spec_sha256, spec_size_bytes = _read_json_artifact(
        spec_path,
        label="selection spec",
        expected_sha256=spec.get("sha256"),
        expected_size_bytes=spec.get("size_bytes"),
    )
    spec_schema_version = _require_manifest_int(
        spec.get("schema_version"),
        field="selection manifest.spec.schema_version",
        positive=True,
    )
    if spec_payload.get("schema_version") != spec_schema_version:
        raise RuntimeError("selection spec schema_version disagrees with the evidence file")

    return {
        "tokenizer": {
            "path": str(tokenizer_resolved),
            "sha256": tokenizer_sha256,
            "size_bytes": tokenizer_size_bytes,
        },
        "spec": {
            "path": str(spec_resolved),
            "sha256": spec_sha256,
            "size_bytes": spec_size_bytes,
        },
    }


def _validate_selection_audit_evidence(
    selection_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _require_mapping(selection.get("audit"), field="selection manifest.audit")
    audit_path = _resolve_relative_evidence_path(
        selection_path.parent,
        audit.get("relative_path"),
        label="selection manifest.audit.relative_path",
    )
    resolved, payload, actual_sha256, actual_size_bytes = _read_json_artifact(
        audit_path,
        label="selection exact-intersection audit",
        expected_sha256=audit.get("sha256"),
        expected_size_bytes=audit.get("size_bytes"),
    )
    if payload.get("schema_version") != 2:
        raise RuntimeError("selection exact-intersection audit schema_version must be 2")
    if payload.get("status") != "passed":
        raise RuntimeError("selection exact-intersection audit status is not passed")
    if payload.get("identity_algorithms") != _SELECTION_IDENTITY_ALGORITHMS:
        raise RuntimeError("selection exact-intersection audit identity algorithms are unsupported")

    raw_sources = selection.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError("selection manifest.sources must be a non-empty list")
    source_stages: dict[str, str] = {}
    for index, raw_source in enumerate(raw_sources):
        source = _require_mapping(raw_source, field=f"selection manifest.sources[{index}]")
        source_id = source.get("source_id")
        stage = source.get("stage")
        if not isinstance(source_id, str) or not source_id or not isinstance(stage, str):
            raise RuntimeError("selection manifest source identity is invalid")
        if source_id in source_stages:
            raise RuntimeError(f"duplicate selection source_id: {source_id!r}")
        source_stages[source_id] = stage

    pairwise = payload.get("pairwise_sources")
    if not isinstance(pairwise, list):
        raise RuntimeError("selection exact-intersection audit pairwise_sources must be a list")
    observed_pairs: set[frozenset[str]] = set()
    for index, raw_pair in enumerate(pairwise):
        label = f"selection exact-intersection audit.pairwise_sources[{index}]"
        pair = _require_mapping(raw_pair, field=label)
        left = pair.get("left_source_id")
        right = pair.get("right_source_id")
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left == right
            or left not in source_stages
            or right not in source_stages
        ):
            raise RuntimeError(f"{label} has invalid source identities")
        if (
            pair.get("left_stage") != source_stages[left]
            or pair.get("right_stage") != source_stages[right]
        ):
            raise RuntimeError(f"{label} stages disagree with the selection manifest")
        pair_key = frozenset((left, right))
        if pair_key in observed_pairs:
            raise RuntimeError(f"{label} duplicates a source pair")
        observed_pairs.add(pair_key)
        counts = _require_mapping(
            pair.get("intersection_counts"), field=f"{label}.intersection_counts"
        )
        if set(counts) != set(_SELECTION_IDENTITY_ALGORITHMS):
            raise RuntimeError(f"{label}.intersection_counts has unexpected identity fields")
        for identity_name, count in counts.items():
            if (
                _require_manifest_int(
                    count,
                    field=f"{label}.intersection_counts.{identity_name}",
                )
                != 0
            ):
                raise RuntimeError(f"{label} reports a non-zero exact intersection")

    expected_pairs = {
        frozenset((left, right))
        for left_index, left in enumerate(source_stages)
        for right in tuple(source_stages)[left_index + 1 :]
    }
    if observed_pairs != expected_pairs:
        raise RuntimeError("selection exact-intersection audit does not cover every source pair")

    reference_audit = _require_mapping(
        payload.get("reference_validation"),
        field="selection exact-intersection audit.reference_validation",
    )
    selected_reference_intersection = _require_manifest_int(
        reference_audit.get("selected_reference_intersection"),
        field=(
            "selection exact-intersection audit.reference_validation."
            "selected_reference_intersection"
        ),
    )
    if (
        selected_reference_intersection != 0
        or reference_audit.get("intersection_zero") is not True
        or payload.get("all_exact_intersections_zero") is not True
    ):
        raise RuntimeError("selection exact-intersection audit evidence did not pass")
    if (
        audit.get("all_exact_intersections_zero") is not True
        or audit.get("selected_reference_intersection") != selected_reference_intersection
    ):
        raise RuntimeError("selection audit summary disagrees with its evidence file")

    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "size_bytes": actual_size_bytes,
        "selected_reference_intersection": selected_reference_intersection,
    }


def _selection_sources_for_database(
    selection: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_sources = selection.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError("selection manifest.sources must be a non-empty list")
    expected: dict[str, dict[str, Any]] = {}
    for index, raw_source in enumerate(raw_sources):
        label = f"selection manifest.sources[{index}]"
        source = _require_mapping(raw_source, field=label)
        source_id = source.get("source_id")
        stage = source.get("stage")
        if not isinstance(source_id, str) or not source_id or not isinstance(stage, str):
            raise RuntimeError(f"{label} has an invalid source identity")
        if source_id in expected:
            raise RuntimeError(f"duplicate selection source_id: {source_id!r}")
        output = _require_mapping(source.get("output"), field=f"{label}.output")
        expected[source_id] = {
            "stage": stage,
            "relative_path": output.get("relative_path"),
            "sha256": _require_sha256(
                output.get("sha256"),
                field=f"{label}.output.sha256",
            ),
            "size_bytes": _require_manifest_int(
                output.get("size_bytes"),
                field=f"{label}.output.size_bytes",
                positive=True,
            ),
            "documents": _require_manifest_int(
                output.get("documents"),
                field=f"{label}.output.documents",
                positive=True,
            ),
        }
    return expected


def _validate_selection_database_evidence(
    selection_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    database = _require_mapping(
        selection.get("database"),
        field="selection manifest.database",
    )
    database_path = _resolve_relative_evidence_path(
        selection_path.parent,
        database.get("relative_path"),
        label="selection manifest.database.relative_path",
    )
    expected_sha256 = _require_sha256(
        database.get("sha256"),
        field="selection manifest.database.sha256",
    )
    expected_size_bytes = _require_manifest_int(
        database.get("size_bytes"),
        field="selection manifest.database.size_bytes",
        positive=True,
    )
    resolved, actual_sha256, actual_size_bytes = _verify_file_evidence(
        database_path,
        label="selection registry database",
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    expected_sources = _selection_sources_for_database(selection)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        integrity = [str(row[0]) for row in integrity_rows]
        selected_rows = int(connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0])
        candidate_rows = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        reference_intersection = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM selections AS selected
                JOIN reference_exclusion_hashes AS reference
                  ON selected.cleaned_sha256 = reference.cleaned_sha256
                """
            ).fetchone()[0]
        )
        for column in _SELECTION_IDENTITY_ALGORITHMS:
            duplicate_groups = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                        SELECT {column} FROM selections
                        GROUP BY {column}
                        HAVING COUNT(DISTINCT source_id) > 1
                    )
                    """  # noqa: S608 - column comes from a fixed local mapping
                ).fetchone()[0]
            )
            if duplicate_groups:
                raise RuntimeError(
                    f"selection registry contains cross-source {column} intersections"
                )

        source_rows = connection.execute(
            """
            SELECT source_id, stage, output_relative_path, output_sha256,
                   output_size_bytes
            FROM sources ORDER BY source_id
            """
        ).fetchall()
        actual_sources = {
            str(row["source_id"]): {
                "stage": str(row["stage"]),
                "relative_path": str(row["output_relative_path"]),
                "sha256": str(row["output_sha256"]),
                "size_bytes": int(row["output_size_bytes"]),
            }
            for row in source_rows
        }
        selected_per_source = {
            str(row["source_id"]): int(row["documents"])
            for row in connection.execute(
                """
                SELECT source_id, COUNT(*) AS documents
                FROM selections GROUP BY source_id ORDER BY source_id
                """
            )
        }
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot validate selection registry database {resolved}: {exc}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()

    if integrity != ["ok"]:
        raise RuntimeError(f"selection registry PRAGMA integrity_check failed: {integrity!r}")
    if database.get("integrity_check") != "ok":
        raise RuntimeError("selection registry summary integrity_check is not ok")
    if candidate_rows != 0 or database.get("uncommitted_candidate_rows") != candidate_rows:
        raise RuntimeError("selection registry contains uncommitted candidate rows")
    if reference_intersection != 0:
        raise RuntimeError("selection registry intersects reference exclusion hashes")
    if selected_rows != _require_manifest_int(
        database.get("selected_document_rows"),
        field="selection manifest.database.selected_document_rows",
        positive=True,
    ):
        raise RuntimeError("selection registry selected row count disagrees with its manifest")
    if selected_rows != sum(source["documents"] for source in expected_sources.values()):
        raise RuntimeError("selection registry row count disagrees with selected output accounting")
    if set(actual_sources) != set(expected_sources):
        raise RuntimeError("selection registry sources disagree with the selection manifest")
    for source_id, expected in expected_sources.items():
        actual = actual_sources[source_id]
        if actual != {key: expected[key] for key in actual}:
            raise RuntimeError(
                f"selection registry source evidence disagrees for source_id={source_id!r}"
            )
        if selected_per_source.get(source_id, 0) != expected["documents"]:
            raise RuntimeError(
                f"selection registry selected document count disagrees for source_id={source_id!r}"
            )

    post_resolved, post_sha256, post_size_bytes = _verify_file_evidence(
        database_path,
        label="selection registry database after semantic validation",
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    if (
        post_resolved != resolved
        or post_sha256 != actual_sha256
        or post_size_bytes != actual_size_bytes
    ):
        raise RuntimeError("selection registry database changed during semantic validation")
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "size_bytes": actual_size_bytes,
        "selected_document_rows": selected_rows,
    }


def native_provenance_repo_root() -> Path:
    """The installation root the native provenance chain resolves accepted authorities from.

    A named seam so bounded tests can point the chain at a synthetic accepted G/G2/L1 set
    without the production call site taking a caller-supplied root.
    """
    return Path(__file__).resolve().parent.parent


def _derived_exclusion_authority(
    entry: Any, *, label: str, path_field: str, sha_field: str, count_field: str
) -> dict[str, Any]:
    """Read one accepted authority's OWN exclusion identity and count.

    These values were already proved against the artifact's real bytes by
    `_validate_exclusion_collection`, which derives the union hash set by reading the files.
    Returning them lets the native branch compare counts that each authority derived for
    itself, instead of the candidate's count standing in for everyone's.
    """
    record = _require_mapping(entry, field=f"{label} exclusion entry")
    return {
        "artifact_path": str(record.get(path_field)),
        "artifact_sha256": _require_sha256(record.get(sha_field), field=f"{label}.{sha_field}"),
        "derived_count": _require_manifest_int(
            record.get(count_field), field=f"{label}.{count_field}", positive=True
        ),
    }


def _validate_reference_release(
    reference_val_dir: str | Path,
) -> tuple[dict[str, Any], tuple[str, ...], Mapping[str, Any], dict[str, Any]]:
    """Validate the frozen reference-validation release and return its provenance block.

    R3-B: also returns the exclusion authority THIS release independently derives from its own
    accepted evidence -- the artifact it names, that artifact's digest, and the union count
    `_validate_exclusion_collection` proved against the artifact's actual bytes.

    Extracted verbatim from ``_validate_full_provenance`` so the legacy selector-v1 branch and
    the accepted-Stage-I-native branch validate the G2 reference release through exactly the
    same code. Legacy behaviour is unchanged: the caller below still runs it in the same order
    with the same inputs.
    """
    reference_release = validate_shard_release(reference_val_dir)
    if reference_release["release_kind"] != "reference" or reference_release["split"] != "val":
        raise RuntimeError("--reference_val_dir must identify the combined reference val split")
    (
        reference_path,
        reference,
        reference_manifest_sha256,
        reference_manifest_size_bytes,
    ) = _read_json_artifact(
        reference_release["manifest_path"],
        label="reference validation release manifest",
    )
    if reference_manifest_sha256 != reference_release["manifest_sha256"]:
        raise RuntimeError("reference validation manifest changed during validation")
    _require_manifest_header(
        reference,
        label="reference validation release manifest",
        schema_version=2,
        kind=_REFERENCE_RELEASE_KIND,
    )
    reference_selection = _require_mapping(
        reference.get("selection"),
        field="reference validation release manifest.selection",
    )
    if reference_selection.get("restricted_to_pre_tokenizer_reserve") is not True:
        raise RuntimeError(
            "reference validation release was not restricted to the pre-tokenizer reserve"
        )
    reserve = _require_mapping(
        reference.get("reserve_provenance"),
        field="reference validation release manifest.reserve_provenance",
    )
    (
        reserve_evidence,
        reserve_exclusion_evidence,
        reference_exclusion_sha256s,
    ) = _validate_reference_reserve_evidence(reserve)
    return (
        {
            "release_root": reference_release["release_root"],
            "manifest_path": str(reference_path),
            "manifest_sha256": reference_manifest_sha256,
            "manifest_size_bytes": reference_manifest_size_bytes,
            "reserve_manifest_path": reserve_evidence["path"],
            "reserve_manifest_sha256": reserve_evidence["sha256"],
            "reserve_manifest_size_bytes": reserve_evidence["size_bytes"],
            "reserve_exclusion_manifest_sha256s": list(reference_exclusion_sha256s),
            "reserve_exclusion_evidence": reserve_exclusion_evidence,
        },
        reference_exclusion_sha256s,
        reference,
        _derived_exclusion_authority(
            reserve.get("reserve_exclusion"),
            label="accepted G2 reference release",
            path_field="manifest_path",
            sha_field="manifest_sha256",
            count_field="hash_count",
        ),
    )


def _validate_tokenizer_release(
    tokenizer_release_manifest: str | Path,
) -> tuple[dict[str, Any], tuple[str, ...], str, Path, str, dict[str, Any]]:
    """Validate the canonical tokenizer release and return its provenance block.

    Extracted verbatim from ``_validate_full_provenance`` for the same reason as
    :func:`_validate_reference_release`.
    """
    (
        tokenizer_release_path,
        tokenizer_release,
        tokenizer_manifest_sha256,
        tokenizer_manifest_size_bytes,
    ) = _read_json_artifact(
        tokenizer_release_manifest,
        label="tokenizer release manifest",
    )
    _require_manifest_header(
        tokenizer_release,
        label="tokenizer release manifest",
        schema_version=2,
        kind=_TOKENIZER_RELEASE_KIND,
    )
    tokenizer_contract = _require_mapping(
        tokenizer_release.get("contract"),
        field="tokenizer release manifest.contract",
    )
    if (
        tokenizer_contract.get("canonical") is not True
        or tokenizer_contract.get("issues") != []
        or tokenizer_contract.get("legacy_allow_noncanonical_contract") is not False
    ):
        raise RuntimeError("tokenizer release manifest declares a non-canonical contract")
    if tokenizer_release.get("publication") != "sibling_staging_then_atomic_rename":
        raise RuntimeError("tokenizer release manifest has a non-atomic publication mode")
    if tokenizer_release.get("vocab_size") != CANONICAL_VOCAB_SIZE:
        raise RuntimeError(f"tokenizer release vocab_size must be {CANONICAL_VOCAB_SIZE}")
    if tokenizer_release.get("special_token_ids") != dict(SPECIAL_TOKEN_IDS):
        raise RuntimeError("tokenizer release special-token IDs are not canonical")
    tokenizer_training = _require_mapping(
        tokenizer_release.get("training"),
        field="tokenizer release manifest.training",
    )
    if (
        tokenizer_training.get("vocab_size_target") != CANONICAL_VOCAB_SIZE
        or tokenizer_training.get("add_prefix_space") is not False
        or tokenizer_training.get("post_processor_enabled") is not False
    ):
        raise RuntimeError("tokenizer release training settings are not canonical")
    (
        tokenizer_exclusion_sha256s,
        tokenizer_exclusion_evidence,
    ) = _validate_exclusion_collection(
        tokenizer_release.get("reference_reserve_exclusion"),
        label="tokenizer release reference_reserve_exclusion",
        activation_field="enabled",
        path_field="manifest_path",
        sha_field="manifest_sha256",
        size_field="manifest_size_bytes",
        resolved_field="manifest_resolved",
        require_top_level_kind=False,
        require_entry_enabled=True,
    )
    tokenizer_release_sha256 = _require_sha256(
        tokenizer_release.get("tokenizer_sha256"),
        field="tokenizer release manifest.tokenizer_sha256",
    )
    tokenizer_json_path = tokenizer_release_path.parent / "tokenizer.json"
    (
        tokenizer_json_resolved,
        _tokenizer_raw,
        actual_tokenizer_sha256,
        actual_tokenizer_size_bytes,
    ) = _read_regular_evidence(
        tokenizer_json_path,
        label="tokenizer release tokenizer.json",
        collect_bytes=False,
    )
    if actual_tokenizer_sha256 != tokenizer_release_sha256:
        raise RuntimeError("tokenizer.json SHA-256 does not match the tokenizer release manifest")
    try:
        assert_tokenizer_contract(tokenizer_json_resolved)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"tokenizer.json is not canonical: {exc}") from exc
    _verify_file_evidence(
        tokenizer_json_path,
        label="tokenizer release tokenizer.json after contract validation",
        expected_sha256=actual_tokenizer_sha256,
        expected_size_bytes=actual_tokenizer_size_bytes,
    )
    return (
        {
            "manifest_path": str(tokenizer_release_path),
            "manifest_sha256": tokenizer_manifest_sha256,
            "manifest_size_bytes": tokenizer_manifest_size_bytes,
            "tokenizer_json_path": str(tokenizer_json_resolved),
            "tokenizer_sha256": actual_tokenizer_sha256,
            "tokenizer_size_bytes": actual_tokenizer_size_bytes,
            "reserve_exclusion_manifest_sha256s": list(tokenizer_exclusion_sha256s),
            "reserve_exclusion_evidence": tokenizer_exclusion_evidence,
        },
        tokenizer_exclusion_sha256s,
        actual_tokenizer_sha256,
        tokenizer_json_resolved,
        tokenizer_release_sha256,
        _derived_exclusion_authority(
            (
                (tokenizer_release.get("reference_reserve_exclusion") or {}).get("manifests")
                or [None]
            )[0],
            label="accepted G tokenizer release",
            path_field="manifest_path",
            sha_field="manifest_sha256",
            count_field="hash_count",
        ),
    )


def _validate_full_provenance(
    *,
    reference_val_dir: str | Path,
    tokenizer_release_manifest: str | Path,
    selection_manifest: str | Path,
    stage_b_selection_stage: str,
    expected_exclusion_sha256s: tuple[str, ...],
    stage_a_release: Mapping[str, Any],
    stage_b_release: Mapping[str, Any],
    expected_tokenizer_sha256: str,
) -> dict[str, Any]:
    (
        reference_block,
        reference_exclusion_sha256s,
        reference,
        _reference_exclusion_authority,
    ) = _validate_reference_release(reference_val_dir)

    (
        tokenizer_block,
        tokenizer_exclusion_sha256s,
        actual_tokenizer_sha256,
        tokenizer_json_resolved,
        tokenizer_release_sha256,
        _tokenizer_exclusion_authority,
    ) = _validate_tokenizer_release(tokenizer_release_manifest)

    (
        selection_path,
        selection,
        selection_manifest_sha256,
        selection_manifest_size_bytes,
    ) = _read_json_artifact(
        selection_manifest,
        label="selection manifest",
    )
    _require_manifest_header(
        selection,
        label="selection manifest",
        schema_version=2,
        kind=_SELECTION_MANIFEST_KIND,
    )
    publication = _require_mapping(
        selection.get("publication"),
        field="selection manifest.publication",
    )
    if publication.get("mode") != "sibling_staging_then_atomic_rename":
        raise RuntimeError("selection manifest has a non-atomic publication mode")
    selection_exclusion = _require_mapping(
        selection.get("reference_validation_exclusion"),
        field="selection manifest.reference_validation_exclusion",
    )
    if (
        selection_exclusion.get("intersection_zero") is not True
        or selection_exclusion.get("selected_reference_intersection") != 0
    ):
        raise RuntimeError("selection manifest intersects the reference reserve")
    (
        selection_exclusion_sha256s,
        selection_exclusion_evidence,
    ) = _validate_exclusion_collection(
        selection_exclusion,
        label="selection manifest.reference_validation_exclusion",
        activation_field="required",
        path_field="path",
        sha_field="sha256",
        size_field="size_bytes",
        resolved_field=None,
        require_top_level_kind=True,
        require_entry_enabled=False,
    )
    selection_tokenizer = _require_mapping(
        selection.get("tokenizer"),
        field="selection manifest.tokenizer",
    )
    if (
        selection_tokenizer.get("vocab_size") != CANONICAL_VOCAB_SIZE
        or selection_tokenizer.get("special_token_ids") != dict(SPECIAL_TOKEN_IDS)
        or selection_tokenizer.get("automatic_bos_eos") is not False
        or selection_tokenizer.get("literal_special_tokens_encoded_as_text") is not True
    ):
        raise RuntimeError("selection manifest tokenizer contract is not canonical")
    selection_tokenizer_sha256 = _require_sha256(
        selection_tokenizer.get("sha256"),
        field="selection manifest.tokenizer.sha256",
    )
    selection_supporting_evidence = _validate_selection_supporting_evidence(
        selection,
        tokenizer_json_path=tokenizer_json_resolved,
    )
    selection_audit_evidence = _validate_selection_audit_evidence(
        selection_path,
        selection,
    )
    selection_database_evidence = _validate_selection_database_evidence(
        selection_path,
        selection,
    )
    selection_outputs = _selection_outputs_by_stage(selection_path, selection)
    if not selection_outputs[stage_b_selection_stage]:
        raise RuntimeError(
            "selection manifest has no outputs for requested Stage-B cohort "
            f"{stage_b_selection_stage!r}"
        )
    source_bindings = {
        "validated": True,
        "stage_b_selection_stage": stage_b_selection_stage,
        "stage_a": _validate_stage_source_bindings(
            stage="stage_a",
            release=stage_a_release,
            expected_outputs=selection_outputs["stage_a"],
        ),
        "stage_b": _validate_stage_source_bindings(
            stage="stage_b",
            release=stage_b_release,
            expected_outputs=selection_outputs[stage_b_selection_stage],
        ),
    }

    exclusion_sets = {
        "Stage A/B shards": expected_exclusion_sha256s,
        "reference validation": reference_exclusion_sha256s,
        "tokenizer release": tokenizer_exclusion_sha256s,
        "selection": selection_exclusion_sha256s,
    }
    differing_exclusions = {
        name: values
        for name, values in exclusion_sets.items()
        if values != expected_exclusion_sha256s
    }
    if differing_exclusions:
        raise RuntimeError(
            "reference reserve exclusion manifest SHA sets disagree: "
            + ", ".join(f"{name}={values}" for name, values in differing_exclusions.items())
        )

    reference_tokenizer_sha256 = _require_sha256(
        reference.get("tokenizer_sha256"),
        field="reference validation release manifest.tokenizer_sha256",
    )
    tokenizer_sha_values = {
        "Stage A/B shards": expected_tokenizer_sha256,
        "reference validation": reference_tokenizer_sha256,
        "tokenizer release": tokenizer_release_sha256,
        "tokenizer.json": actual_tokenizer_sha256,
        "selection": selection_tokenizer_sha256,
        "selection tokenizer evidence": selection_supporting_evidence["tokenizer"]["sha256"],
    }
    differing_tokenizers = {
        name: value
        for name, value in tokenizer_sha_values.items()
        if value != expected_tokenizer_sha256
    }
    if differing_tokenizers:
        raise RuntimeError(
            "tokenizer SHA-256 values disagree: "
            + ", ".join(f"{name}={value}" for name, value in differing_tokenizers.items())
        )

    return {
        "full_chain_validated": True,
        "source_bindings": source_bindings,
        "reference_validation": reference_block,
        "tokenizer_release": tokenizer_block,
        "selection": {
            "stage_b_selection_stage": stage_b_selection_stage,
            "manifest_path": str(selection_path),
            "manifest_sha256": selection_manifest_sha256,
            "manifest_size_bytes": selection_manifest_size_bytes,
            "tokenizer_sha256": selection_tokenizer_sha256,
            "reserve_exclusion_manifest_sha256s": list(selection_exclusion_sha256s),
            "reserve_exclusion_evidence": selection_exclusion_evidence,
            "supporting_evidence": selection_supporting_evidence,
            "audit_evidence": selection_audit_evidence,
            "database_evidence": selection_database_evidence,
        },
    }


def build_checkpoint_milestones(
    *,
    stage_specs: Sequence[Mapping[str, Any]],
    sequences_per_optimizer_step: int,
    consumed_transitions_per_optimizer_step: int,
    decay_start_step: int,
    nominal_token_targets: Sequence[tuple[str, int]] = (_NOMINAL_CHECKPOINT_TOKEN_TARGETS),
) -> dict[str, Any]:
    """Build deduplicated post-update checkpoint milestones on the global timeline.

    Nominal token milestones use the first completed optimizer step at or after
    the target. A target beyond the actual run is honestly clamped to the final
    step and carries a negative delta. Exposure endpoints use the first step
    that has consumed the complete permutation epoch, so any unavoidable
    optimizer-batch overshoot is explicit.
    """
    sequences_per_optimizer_step = _positive_int(
        "sequences_per_optimizer_step", sequences_per_optimizer_step
    )
    consumed_transitions_per_optimizer_step = _positive_int(
        "consumed_transitions_per_optimizer_step",
        consumed_transitions_per_optimizer_step,
    )

    normalized_stages: dict[str, dict[str, int]] = {}
    for index, raw in enumerate(stage_specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"stage_specs[{index}] must be a mapping")
        name = str(raw.get("name", ""))
        if name not in {"stage_a", "stage_b"} or name in normalized_stages:
            raise ValueError("stage_specs must contain exactly one stage_a and stage_b")
        start_step = int(raw.get("start_step", -1))
        planned_steps = _positive_int(
            f"{name}.planned_optimizer_steps",
            int(raw.get("planned_optimizer_steps", 0)),
        )
        unique_blocks = _positive_int(f"{name}.unique_blocks", int(raw.get("unique_blocks", 0)))
        completed_exposures = int(raw.get("completed_full_exposures", -1))
        if start_step < 0 or completed_exposures < 0:
            raise ValueError(f"{name} milestone inputs must be non-negative")
        normalized_stages[name] = {
            "start_step": start_step,
            "planned_optimizer_steps": planned_steps,
            "unique_blocks": unique_blocks,
            "completed_full_exposures": completed_exposures,
        }

    if set(normalized_stages) != {"stage_a", "stage_b"}:
        raise ValueError("stage_specs must contain exactly one stage_a and stage_b")
    stage_a = normalized_stages["stage_a"]
    stage_b = normalized_stages["stage_b"]
    stage_a_stop = stage_a["start_step"] + stage_a["planned_optimizer_steps"]
    if stage_a["start_step"] != 0 or stage_b["start_step"] != stage_a_stop:
        raise ValueError("stage milestone specs must form one contiguous A->B timeline")
    total_steps = stage_b["start_step"] + stage_b["planned_optimizer_steps"]
    decay_start_step = int(decay_start_step)
    if not 0 < decay_start_step <= total_steps:
        raise ValueError("decay_start_step must lie inside the absolute run horizon")

    reasons_by_step: dict[int, set[str]] = {}

    def add_reason(step: int, reason: str) -> None:
        step = int(step)
        if not 0 < step <= total_steps:
            raise RuntimeError(f"checkpoint milestone {reason!r} has out-of-range step {step}")
        reasons_by_step.setdefault(step, set()).add(reason)

    nominal_records = []
    seen_nominal_labels: set[str] = set()
    previous_target = 0
    for label, raw_target in nominal_token_targets:
        label = str(label)
        target = _positive_int(f"nominal target {label!r}", raw_target)
        if not label or label in seen_nominal_labels or target <= previous_target:
            raise ValueError(
                "nominal checkpoint targets require unique labels and strictly "
                "increasing positive token counts"
            )
        seen_nominal_labels.add(label)
        previous_target = target
        unclamped_step = (
            target + consumed_transitions_per_optimizer_step - 1
        ) // consumed_transitions_per_optimizer_step
        step = min(total_steps, max(1, unclamped_step))
        actual = step * consumed_transitions_per_optimizer_step
        add_reason(step, label)
        nominal_records.append({
            "reason": label,
            "nominal_cumulative_tokens": target,
            "absolute_step": step,
            "actual_cumulative_consumed_transitions": actual,
            "delta_consumed_transitions": actual - target,
            "horizon_clamped": unclamped_step > total_steps,
        })

    stage_b_midpoint = stage_b["start_step"] + (stage_b["planned_optimizer_steps"] + 1) // 2
    add_reason(stage_a_stop, "stage_a_end")
    add_reason(stage_b_midpoint, "stage_b_midpoint")
    add_reason(decay_start_step, "wsd_decay_start")
    add_reason(total_steps, "stage_b_end")

    exposure_records = []
    for name in ("stage_a", "stage_b"):
        stage = normalized_stages[name]
        stage_stop = stage["start_step"] + stage["planned_optimizer_steps"]
        for exposure_index in range(1, stage["completed_full_exposures"] + 1):
            boundary_blocks = exposure_index * stage["unique_blocks"]
            local_step = (
                boundary_blocks + sequences_per_optimizer_step - 1
            ) // sequences_per_optimizer_step
            absolute_step = stage["start_step"] + local_step
            if absolute_step > stage_stop:
                raise RuntimeError(
                    f"{name} completed exposure {exposure_index} lies after stage stop"
                )
            actual_stage_blocks = local_step * sequences_per_optimizer_step
            reason = f"{name}_exposure_{exposure_index}_end"
            add_reason(absolute_step, reason)
            exposure_records.append({
                "reason": reason,
                "stage": name,
                "exposure_index": exposure_index,
                "boundary_consumed_blocks": boundary_blocks,
                "absolute_step": absolute_step,
                "actual_stage_consumed_blocks": actual_stage_blocks,
                "optimizer_batch_overshoot_blocks": (actual_stage_blocks - boundary_blocks),
            })

    absolute_steps = sorted(reasons_by_step)
    entries = [
        {
            "absolute_step": step,
            "actual_cumulative_consumed_transitions": (
                step * consumed_transitions_per_optimizer_step
            ),
            "reasons": sorted(reasons_by_step[step]),
        }
        for step in absolute_steps
    ]
    return {
        "schema_version": 1,
        "coordinate": "post_update_absolute_global_optimizer_step",
        "accounting_unit": "serialized_target_positions_consumed",
        "nominal_rounding": ("first_step_at_or_after_target_clamped_to_final_horizon"),
        "consumed_transitions_per_optimizer_step": (consumed_transitions_per_optimizer_step),
        "absolute_steps": absolute_steps,
        "cli_save_steps": ",".join(str(step) for step in absolute_steps),
        "entries": entries,
        "nominal_cumulative_token_targets": nominal_records,
        "exposure_epoch_endpoints": exposure_records,
    }


def _stage_plan(
    name: str,
    dataset: PackedBinDataset,
    source_dir: Path,
    sequences_per_step: int,
    exposures: int,
) -> dict[str, Any]:
    stats = dataset.stats()
    exposures = _positive_int(f"{name}_exposures", exposures)
    unique_blocks = int(stats["n_blocks"])
    candidate_exposure_blocks = unique_blocks * exposures
    # Floor once after exposure expansion. This preserves partial blocks between
    # full passes instead of discarding a batch-alignment tail every exposure.
    steps = candidate_exposure_blocks // sequences_per_step
    if steps <= 0:
        raise ValueError(
            f"{name} has {candidate_exposure_blocks} candidate exposure blocks, "
            f"fewer than one optimizer step ({sequences_per_step} sequences)"
        )

    consumed_blocks = steps * sequences_per_step
    dropped_blocks = candidate_exposure_blocks - consumed_blocks
    unique_blocks_consumed = min(unique_blocks, consumed_blocks)
    consumed_replay_blocks = consumed_blocks - unique_blocks_consumed
    planned_replay_blocks = candidate_exposure_blocks - unique_blocks
    seq_len = int(stats["seq_len"])
    consumed_transitions = consumed_blocks * seq_len
    alignment_dropped_transitions = dropped_blocks * seq_len
    global_tail_transitions = int(stats["tail_transitions"])
    global_tail_transition_opportunities = global_tail_transitions * exposures
    total_transition_opportunities = int(stats["total_transitions"]) * exposures
    usable_transition_opportunities = int(stats["usable_transitions"]) * exposures
    unconsumed_transitions = total_transition_opportunities - consumed_transitions

    if (
        unconsumed_transitions
        != alignment_dropped_transitions + global_tail_transition_opportunities
    ):
        raise RuntimeError(f"{name} deterministic exposure accounting is inconsistent")

    return {
        "source_dir": str(source_dir.resolve()),
        "planned_optimizer_steps": steps,
        "requested_exposures": exposures,
        "full_blocks": unique_blocks,
        "unique_blocks": unique_blocks,
        "candidate_exposure_blocks": candidate_exposure_blocks,
        "planned_replay_blocks": planned_replay_blocks,
        "consumed_blocks": consumed_blocks,
        "consumed_exposure_blocks": consumed_blocks,
        "unique_blocks_consumed": unique_blocks_consumed,
        "consumed_replay_blocks": consumed_replay_blocks,
        "completed_full_exposures": consumed_blocks // unique_blocks,
        "partial_exposure_blocks": consumed_blocks % unique_blocks,
        "realized_mean_exposures_per_unique_block": consumed_blocks / unique_blocks,
        "dropped_batch_alignment_blocks": dropped_blocks,
        "consumed_serialized_target_positions": consumed_transitions,
        "dropped_batch_alignment_transitions": alignment_dropped_transitions,
        "global_tail_transitions": global_tail_transitions,
        "global_tail_transition_opportunities": global_tail_transition_opportunities,
        "total_transition_opportunities": total_transition_opportunities,
        "unconsumed_transitions_total": unconsumed_transitions,
        "block_coverage_fraction": consumed_blocks / candidate_exposure_blocks,
        "unique_block_coverage_fraction": unique_blocks_consumed / unique_blocks,
        "exposure_block_coverage_fraction": consumed_blocks / candidate_exposure_blocks,
        "usable_transition_coverage_fraction": (
            consumed_transitions / usable_transition_opportunities
        ),
        "total_transition_coverage_fraction": (
            consumed_transitions / total_transition_opportunities
        ),
        "dataset": stats,
    }


def build_run_plan(
    *,
    stage_a_dir: str | Path,
    stage_b_dir: str | Path,
    seq_len: int,
    micro_bsz: int,
    grad_accum: int,
    warmup_steps: int,
    decay_fraction: float,
    stage_a_exposures: int = 1,
    stage_b_exposures: int = 1,
    stage_b_selection_stage: str = "stage_b",
    reference_val_dir: str | Path | None = None,
    tokenizer_release_manifest: str | Path | None = None,
    selection_manifest: str | Path | None = None,
    provenance_chain_kind: str = LEGACY_CHAIN_KIND,
    accepted_stage_i_dir: str | Path | None = None,
    candidate_m_plan: str | Path | None = None,
    expected_candidate_m_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an exact explicit-exposure Stage A/B plan and WSD candidate.

    ``provenance_chain_kind`` selects which provenance branch validates the releases.
    ``legacy_selector_v1`` is the historical D-026 chain and is unchanged.
    ``accepted_stage_i_native_v1`` is the branch for accepted H/I-derived releases
    (DECISIONS D-146); it validates the accepted Stage-I publication and the derived Stage-M
    releases directly. The two branches are never mixed and never silently fall back to
    each other.
    """
    if provenance_chain_kind not in PROVENANCE_CHAIN_KINDS:
        raise ValueError(
            f"provenance_chain_kind must be one of {list(PROVENANCE_CHAIN_KINDS)}, "
            f"got {provenance_chain_kind!r}"
        )
    _native_inputs = (accepted_stage_i_dir, candidate_m_plan, expected_candidate_m_plan_sha256)
    if provenance_chain_kind == NATIVE_CHAIN_KIND:
        # R1-B. The reference-validation release (G2) and the tokenizer release (G) are
        # H/I-native artefacts and are *required* on this branch: the native chain must supply
        # everything strict reference/tokenizer validation needs without ever creating a legacy
        # selection manifest. Only the selector-v1 selection manifest is forbidden here.
        if selection_manifest is not None:
            raise ValueError(
                "native provenance must not be given a legacy selector-v1 selection_manifest"
            )
        if not all(value is not None for value in _native_inputs):
            raise ValueError(
                "accepted_stage_i_dir, candidate_m_plan and expected_candidate_m_plan_sha256 "
                "must be supplied together for the native provenance chain"
            )
        missing_shared = [
            name
            for name, value in (
                ("reference_val_dir", reference_val_dir),
                ("tokenizer_release_manifest", tokenizer_release_manifest),
            )
            if value is None
        ]
        if missing_shared:
            raise ValueError(
                "native provenance requires the frozen reference-validation and tokenizer "
                f"release authorities: missing {missing_shared}"
            )
    elif any(value is not None for value in _native_inputs):
        raise ValueError("legacy provenance must not be given accepted Stage-I native inputs")
    seq_len = _positive_int("seq_len", seq_len)
    micro_bsz = _positive_int("micro_bsz", micro_bsz)
    grad_accum = _positive_int("grad_accum", grad_accum)
    stage_a_exposures = _positive_int("stage_a_exposures", stage_a_exposures)
    stage_b_exposures = _positive_int("stage_b_exposures", stage_b_exposures)
    stage_b_selection_stage = str(stage_b_selection_stage)
    if stage_b_selection_stage not in STAGE_B_SELECTION_STAGES:
        raise ValueError(
            "stage_b_selection_stage must be one of "
            f"{STAGE_B_SELECTION_STAGES}, got {stage_b_selection_stage!r}"
        )
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    decay_fraction = float(decay_fraction)
    if not math.isfinite(decay_fraction) or not 0.0 < decay_fraction <= 1.0:
        raise ValueError(f"decay_fraction must be finite and in (0, 1], got {decay_fraction}")

    stage_a_path = Path(stage_a_dir).expanduser().resolve()
    stage_b_path = Path(stage_b_dir).expanduser().resolve()
    if stage_a_path == stage_b_path:
        raise ValueError("stage_a_dir and stage_b_dir must be different directories")

    sequences_per_step = micro_bsz * grad_accum
    stage_a_ds = PackedBinDataset(
        str(stage_a_path),
        seq_len=seq_len,
        sampling_mode="deterministic",
        require_release_manifest=True,
    )
    stage_b_ds = PackedBinDataset(
        str(stage_b_path),
        seq_len=seq_len,
        sampling_mode="deterministic",
        require_release_manifest=True,
    )
    stage_a = _stage_plan(
        "stage_a",
        stage_a_ds,
        stage_a_path,
        sequences_per_step,
        stage_a_exposures,
    )
    stage_b = _stage_plan(
        "stage_b",
        stage_b_ds,
        stage_b_path,
        sequences_per_step,
        stage_b_exposures,
    )
    stage_a_release = stage_a["dataset"].get("release_validation")
    stage_b_release = stage_b["dataset"].get("release_validation")
    if not isinstance(stage_a_release, dict) or not isinstance(stage_b_release, dict):
        raise RuntimeError("planner requires validated Stage A and Stage B release metadata")
    stage_a_exclusion_sha256s = tuple(
        stage_a_release.get("reference_exclusion_manifest_sha256s") or ()
    )
    stage_b_exclusion_sha256s = tuple(
        stage_b_release.get("reference_exclusion_manifest_sha256s") or ()
    )
    if stage_a_exclusion_sha256s != stage_b_exclusion_sha256s:
        raise RuntimeError(
            "Stage A and Stage B use different reference exclusion manifest SHA sets: "
            f"A={stage_a_exclusion_sha256s}, B={stage_b_exclusion_sha256s}"
        )
    stage_a_tokenizer_sha256 = str(stage_a_release["tokenizer_sha256"])
    stage_b_tokenizer_sha256 = str(stage_b_release["tokenizer_sha256"])
    if stage_a_tokenizer_sha256 != stage_b_tokenizer_sha256:
        raise RuntimeError(
            "Stage A and Stage B use different tokenizer SHA-256 values: "
            f"A={stage_a_tokenizer_sha256}, B={stage_b_tokenizer_sha256}"
        )
    release_provenance = {
        "stage_b_selection_stage": stage_b_selection_stage,
        "shared_reference_exclusion_manifest_sha256s": list(stage_a_exclusion_sha256s),
        "shared_tokenizer_sha256": stage_a_tokenizer_sha256,
        "stage_a": {
            "manifest_path": stage_a_release["manifest_path"],
            "manifest_sha256": stage_a_release["manifest_sha256"],
        },
        "stage_b": {
            "manifest_path": stage_b_release["manifest_path"],
            "manifest_sha256": stage_b_release["manifest_sha256"],
        },
    }
    provenance_inputs = (
        reference_val_dir,
        tokenizer_release_manifest,
        selection_manifest,
    )
    # R1-B: the all-or-nothing triple is a *legacy* requirement. The native branch has its own
    # required-input rule (accepted-I + candidate-M plan + digest, plus the G/G2 authorities,
    # and never a selection manifest), checked at the top of this function.
    if provenance_chain_kind != NATIVE_CHAIN_KIND and (
        any(value is not None for value in provenance_inputs)
        and not all(value is not None for value in provenance_inputs)
    ):
        raise ValueError(
            "reference_val_dir, tokenizer_release_manifest, and selection_manifest "
            "must be supplied together"
        )
    release_provenance["provenance_chain_kind"] = provenance_chain_kind
    if provenance_chain_kind == NATIVE_CHAIN_KIND:
        # The native branch derives full_chain_validated itself, and only after every link of
        # accepted-I -> candidate-M plan -> both Stage-M releases has been proved from bytes.
        assert_single_branch(release_provenance, chain_kind=NATIVE_CHAIN_KIND)
        assert accepted_stage_i_dir is not None
        assert candidate_m_plan is not None
        assert expected_candidate_m_plan_sha256 is not None
        native = validate_native_chain(
            repo_root=native_provenance_repo_root(),
            accepted_stage_i_dir=Path(accepted_stage_i_dir),
            candidate_m_plan=Path(candidate_m_plan),
            expected_candidate_m_plan_sha256=str(expected_candidate_m_plan_sha256),
            stage_releases={"stage_a": stage_a_path.parent, "stage_b": stage_b_path.parent},
        )
        if native.get("full_chain_validated") is not True:
            raise RuntimeError("native provenance chain did not validate")

        # R1-B. The same frozen validators the legacy branch uses, so the native branch gets
        # the full G2 reserve provenance and G tokenizer-release authority without a selection
        # manifest. These are the blocks strict reference/tokenizer validation consumes later.
        assert reference_val_dir is not None
        assert tokenizer_release_manifest is not None
        (
            native_reference_block,
            native_reference_exclusion_sha256s,
            native_reference_manifest,
            native_reference_exclusion_authority,
        ) = _validate_reference_release(reference_val_dir)
        (
            native_tokenizer_block,
            native_tokenizer_exclusion_sha256s,
            native_tokenizer_sha256,
            _native_tokenizer_json,
            _native_tokenizer_declared_sha256,
            native_tokenizer_exclusion_authority,
        ) = _validate_tokenizer_release(tokenizer_release_manifest)

        differing = {
            name: values
            for name, values in (
                ("Stage A/B shards", stage_a_exclusion_sha256s),
                ("reference validation", native_reference_exclusion_sha256s),
                ("tokenizer release", native_tokenizer_exclusion_sha256s),
            )
            if values != stage_a_exclusion_sha256s
        }
        if differing:
            raise RuntimeError(
                "native chain reference reserve exclusion manifest SHA sets disagree: "
                + ", ".join(f"{name}={values}" for name, values in differing.items())
            )
        native_reference_tokenizer_sha256 = _require_sha256(
            native_reference_manifest.get("tokenizer_sha256"),
            field="reference validation release manifest.tokenizer_sha256",
        )
        tokenizer_disagreements = {
            name: value
            for name, value in (
                ("Stage A/B shards", stage_a_tokenizer_sha256),
                ("reference validation", native_reference_tokenizer_sha256),
                ("tokenizer release", native_tokenizer_sha256),
                ("candidate M plan", str(native["shared_tokenizer_sha256"])),
            )
            if value != stage_a_tokenizer_sha256
        }
        if tokenizer_disagreements:
            raise RuntimeError(
                "native chain tokenizer SHA-256 values disagree: "
                + ", ".join(f"{name}={value}" for name, value in tokenizer_disagreements.items())
            )

        # R2-A: the exclusion authority agreement now spans the candidate-M plan and both
        # Stage-M releases (proved inside validate_native_chain) plus G and G2 here, so
        # native_shared_authority_validated is the result of a real comparison rather than a
        # constant.
        # R3-B: each accepted authority contributes the count IT derived from ITS OWN
        # evidence. Previously the candidate's count was passed in as G's and G2's, so an
        # inconsistent underlying count could not be seen.
        agreed_exclusion = require_identical_exclusion_authorities([
            dict(
                native["shared_exclusion_authority"],
                participant="candidate_m_and_releases",
                artifact_sha256=native["shared_exclusion_authority"]["artifact_sha256"],
                derived_count=native["shared_exclusion_authority"]["derived_count"],
                artifact_path=native["shared_exclusion_authority"]["artifact_paths"][0],
            ),
            exclusion_authority(
                participant="g2_reference_release_validator",
                artifact_path=native_reference_exclusion_authority["artifact_path"],
                artifact_sha256=native_reference_exclusion_authority["artifact_sha256"],
                derived_count=native_reference_exclusion_authority["derived_count"],
            ),
            exclusion_authority(
                participant="g_tokenizer_release_validator",
                artifact_path=native_tokenizer_exclusion_authority["artifact_path"],
                artifact_sha256=native_tokenizer_exclusion_authority["artifact_sha256"],
                derived_count=native_tokenizer_exclusion_authority["derived_count"],
            ),
        ])

        release_provenance.update(native)
        release_provenance["reference_validation"] = native_reference_block
        release_provenance["tokenizer_release"] = native_tokenizer_block
        release_provenance["shared_exclusion_authority"] = agreed_exclusion
        release_provenance["native_shared_authority_validated"] = True
        # R2-E / section 13: the post-merge identity is computed only once every load-bearing
        # native data field has been assembled, and is serialized so the training-facing
        # contract can independently recompute and compare it.
        release_provenance["native_post_merge_data_branch_identity_sha256"] = (
            post_merge_data_branch_identity_sha256(release_provenance)
        )
        full_chain_validated = True
    else:
        full_chain_validated = all(value is not None for value in provenance_inputs)
        if full_chain_validated:
            assert reference_val_dir is not None
            assert tokenizer_release_manifest is not None
            assert selection_manifest is not None
            release_provenance.update(
                _validate_full_provenance(
                    reference_val_dir=reference_val_dir,
                    tokenizer_release_manifest=tokenizer_release_manifest,
                    selection_manifest=selection_manifest,
                    stage_b_selection_stage=stage_b_selection_stage,
                    expected_exclusion_sha256s=stage_a_exclusion_sha256s,
                    stage_a_release=stage_a_release,
                    stage_b_release=stage_b_release,
                    expected_tokenizer_sha256=stage_a_tokenizer_sha256,
                )
            )
        else:
            release_provenance["full_chain_validated"] = False

    stage_a_steps = int(stage_a["planned_optimizer_steps"])
    stage_b_steps = int(stage_b["planned_optimizer_steps"])
    total_steps = stage_a_steps + stage_b_steps
    if warmup_steps >= stage_a_steps:
        raise ValueError(
            "warmup_steps must finish inside Stage A: "
            f"warmup={warmup_steps}, stage_a_stop={stage_a_steps}"
        )

    decay_steps = max(1, math.ceil(total_steps * decay_fraction))
    decay_start = total_steps - decay_steps
    if decay_start < stage_a_steps:
        raise ValueError(
            "decay_fraction places WSD decay before Stage B; choose at most "
            f"{stage_b_steps / total_steps:.12g} for this corpus/batch plan"
        )

    target_positions_per_step = sequences_per_step * seq_len
    consumed_blocks = int(stage_a["consumed_blocks"]) + int(stage_b["consumed_blocks"])
    unique_blocks = int(stage_a["unique_blocks"]) + int(stage_b["unique_blocks"])
    candidate_exposure_blocks = int(stage_a["candidate_exposure_blocks"]) + int(
        stage_b["candidate_exposure_blocks"]
    )
    unique_blocks_consumed = int(stage_a["unique_blocks_consumed"]) + int(
        stage_b["unique_blocks_consumed"]
    )
    consumed_replay_blocks = int(stage_a["consumed_replay_blocks"]) + int(
        stage_b["consumed_replay_blocks"]
    )
    planned_replay_blocks = int(stage_a["planned_replay_blocks"]) + int(
        stage_b["planned_replay_blocks"]
    )
    consumed_transitions = int(stage_a["consumed_serialized_target_positions"]) + int(
        stage_b["consumed_serialized_target_positions"]
    )
    total_transition_opportunities = int(stage_a["total_transition_opportunities"]) + int(
        stage_b["total_transition_opportunities"]
    )
    dropped_alignment_blocks = int(stage_a["dropped_batch_alignment_blocks"]) + int(
        stage_b["dropped_batch_alignment_blocks"]
    )
    unconsumed_transitions = int(stage_a["unconsumed_transitions_total"]) + int(
        stage_b["unconsumed_transitions_total"]
    )
    explicit_replay = stage_a_exposures > 1 or stage_b_exposures > 1
    expected_consumed_transitions = total_steps * target_positions_per_step
    if consumed_transitions != expected_consumed_transitions:
        raise RuntimeError(
            "planned steps and consumed-transition accounting disagree: "
            f"{consumed_transitions} != {expected_consumed_transitions}"
        )
    checkpoint_milestones = build_checkpoint_milestones(
        stage_specs=(
            {
                "name": "stage_a",
                "start_step": 0,
                "planned_optimizer_steps": stage_a_steps,
                "unique_blocks": int(stage_a["unique_blocks"]),
                "completed_full_exposures": int(stage_a["completed_full_exposures"]),
            },
            {
                "name": "stage_b",
                "start_step": stage_a_steps,
                "planned_optimizer_steps": stage_b_steps,
                "unique_blocks": int(stage_b["unique_blocks"]),
                "completed_full_exposures": int(stage_b["completed_full_exposures"]),
            },
        ),
        sequences_per_optimizer_step=sequences_per_step,
        consumed_transitions_per_optimizer_step=target_positions_per_step,
        decay_start_step=decay_start,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "plan_type": (
            "deterministic_explicit_multi_exposure_stage_a_b"
            if explicit_replay
            else "deterministic_no_replacement_stage_a_b"
        ),
        "invariants": {
            "sampling_mode": "deterministic",
            "replacement": False,
            "implicit_replay": False,
            "explicit_replay": explicit_replay,
            "stage_rounding": "floor_once_after_exposure_expansion_per_stage",
            "full_production_provenance_chain": full_chain_validated,
        },
        "release_provenance": release_provenance,
        "inputs": {
            "stage_a_dir": str(stage_a_path),
            "stage_b_dir": str(stage_b_path),
            "seq_len": seq_len,
            "micro_bsz": micro_bsz,
            "grad_accum": grad_accum,
            "warmup_steps": warmup_steps,
            "decay_fraction": decay_fraction,
            "stage_a_exposures": stage_a_exposures,
            "stage_b_exposures": stage_b_exposures,
            "stage_b_selection_stage": stage_b_selection_stage,
            "reference_val_dir": (
                str(Path(reference_val_dir).expanduser().resolve())
                if reference_val_dir is not None
                else None
            ),
            "tokenizer_release_manifest": (
                str(Path(tokenizer_release_manifest).expanduser().resolve())
                if tokenizer_release_manifest is not None
                else None
            ),
            "selection_manifest": (
                str(Path(selection_manifest).expanduser().resolve())
                if selection_manifest is not None
                else None
            ),
        },
        "batch": {
            "sequences_per_optimizer_step": sequences_per_step,
            "serialized_target_positions_per_optimizer_step": target_positions_per_step,
        },
        "boundaries": {
            "stage_a_start_step": 0,
            "stage_a_stop_step": stage_a_steps,
            "stage_b_start_step": stage_a_steps,
            "stage_b_global_stop_step": total_steps,
            "schedule_total_steps": total_steps,
        },
        "wsd_candidate": {
            "lr_schedule": "wsd",
            "warmup_steps": warmup_steps,
            "stable_start_step": warmup_steps,
            "decay_fraction_requested": decay_fraction,
            "decay_fraction_realized": decay_steps / total_steps,
            "decay_steps": decay_steps,
            "decay_start_step": decay_start,
            "decay_end_step": total_steps,
        },
        "checkpoint_milestones": checkpoint_milestones,
        "stages": {"stage_a": stage_a, "stage_b": stage_b},
        "totals": {
            "full_blocks": unique_blocks,
            "unique_blocks": unique_blocks,
            "candidate_exposure_blocks": candidate_exposure_blocks,
            "planned_replay_blocks": planned_replay_blocks,
            "consumed_blocks": consumed_blocks,
            "consumed_exposure_blocks": consumed_blocks,
            "unique_blocks_consumed": unique_blocks_consumed,
            "consumed_replay_blocks": consumed_replay_blocks,
            "realized_mean_exposures_per_unique_block": (consumed_blocks / unique_blocks),
            "dropped_batch_alignment_blocks": dropped_alignment_blocks,
            "dropped_batch_alignment_transitions": int(
                stage_a["dropped_batch_alignment_transitions"]
            )
            + int(stage_b["dropped_batch_alignment_transitions"]),
            "global_tail_transitions": int(stage_a["global_tail_transitions"])
            + int(stage_b["global_tail_transitions"]),
            "global_tail_transition_opportunities": int(
                stage_a["global_tail_transition_opportunities"]
            )
            + int(stage_b["global_tail_transition_opportunities"]),
            "total_transition_opportunities": total_transition_opportunities,
            "consumed_serialized_target_positions": consumed_transitions,
            "unconsumed_transitions_total": unconsumed_transitions,
            "block_coverage_fraction": consumed_blocks / candidate_exposure_blocks,
            "unique_block_coverage_fraction": unique_blocks_consumed / unique_blocks,
            "exposure_block_coverage_fraction": (consumed_blocks / candidate_exposure_blocks),
            "usable_transition_coverage_fraction": (consumed_blocks / candidate_exposure_blocks),
            "total_transition_coverage_fraction": (
                consumed_transitions / total_transition_opportunities
            ),
        },
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Durably replace ``path`` without exposing a partial JSON manifest."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan exact deterministic Stage A/B exposures and a WSD schedule."
    )
    parser.add_argument("--stage_a_dir", required=True)
    parser.add_argument("--stage_b_dir", required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--micro_bsz", type=int, required=True)
    parser.add_argument("--grad_accum", type=int, required=True)
    parser.add_argument("--warmup_steps", type=int, required=True)
    parser.add_argument("--decay_fraction", type=float, required=True)
    parser.add_argument("--stage_a_exposures", type=int, default=1)
    parser.add_argument("--stage_b_exposures", type=int, default=1)
    parser.add_argument(
        "--stage_b_selection_stage",
        choices=STAGE_B_SELECTION_STAGES,
        default="stage_b",
    )
    # Legacy selector-v1 provenance inputs. Still mandatory for the legacy branch: `main`
    # rejects a legacy invocation that omits any of them, exactly as `required=True` did.
    parser.add_argument("--reference_val_dir")
    parser.add_argument("--tokenizer_release_manifest")
    parser.add_argument("--selection_manifest")
    # Accepted-Stage-I native provenance inputs (DECISIONS D-146).
    parser.add_argument(
        "--provenance_chain_kind",
        choices=list(PROVENANCE_CHAIN_KINDS),
        default=LEGACY_CHAIN_KIND,
    )
    parser.add_argument("--accepted_stage_i_dir")
    parser.add_argument("--candidate_m_plan")
    parser.add_argument("--expected_candidate_m_plan_sha256")
    parser.add_argument("--out_json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    legacy_required = (
        "reference_val_dir",
        "tokenizer_release_manifest",
        "selection_manifest",
    )
    if args.provenance_chain_kind == LEGACY_CHAIN_KIND:
        missing = [name for name in legacy_required if getattr(args, name) is None]
        if missing:
            parser.error(
                "the following arguments are required for "
                f"--provenance_chain_kind {LEGACY_CHAIN_KIND}: "
                + ", ".join(f"--{name}" for name in missing)
            )
    try:
        plan = build_run_plan(
            stage_a_dir=args.stage_a_dir,
            stage_b_dir=args.stage_b_dir,
            seq_len=args.seq_len,
            micro_bsz=args.micro_bsz,
            grad_accum=args.grad_accum,
            warmup_steps=args.warmup_steps,
            decay_fraction=args.decay_fraction,
            stage_a_exposures=args.stage_a_exposures,
            stage_b_exposures=args.stage_b_exposures,
            stage_b_selection_stage=args.stage_b_selection_stage,
            reference_val_dir=args.reference_val_dir,
            tokenizer_release_manifest=args.tokenizer_release_manifest,
            selection_manifest=args.selection_manifest,
            provenance_chain_kind=args.provenance_chain_kind,
            accepted_stage_i_dir=args.accepted_stage_i_dir,
            candidate_m_plan=args.candidate_m_plan,
            expected_candidate_m_plan_sha256=args.expected_candidate_m_plan_sha256,
        )
        output = write_json_atomic(args.out_json, plan)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    boundaries = plan["boundaries"]
    wsd = plan["wsd_candidate"]
    print(
        "planned "
        f"A_stop={boundaries['stage_a_stop_step']} "
        f"B_stop={boundaries['stage_b_global_stop_step']} "
        f"decay={wsd['decay_start_step']}:{wsd['decay_end_step']} "
        f"save_steps={plan['checkpoint_milestones']['cli_save_steps']} "
        f"json={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
