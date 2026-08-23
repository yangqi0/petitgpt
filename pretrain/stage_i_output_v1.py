#!/usr/bin/env python3
"""Stage-I realization output contract v1: record schema, sharding, publication and consumer.

Stage I publishes a *selected-document realization*, not packed training shards. The output is
strict UTF-8 canonical JSON Lines so that every downstream stage re-derives its own token stream
from bytes it can hash, rather than trusting an opaque binary produced here.

Three properties this module exists to hold:

* the record schema is closed. An unknown, missing or mistyped field is rejected rather than
  ignored, so a future writer cannot quietly add a field that a consumer silently drops.
* publication is atomic and COMPLETE is written last. A failed or interrupted run leaves a staging
  directory, never a discoverable result, so "the directory exists" and "the run finished" cannot
  disagree.
* sharding is a declared policy, not an accident of the machine that ran it. The rule is a fixed
  record count per shard over a fully determined physical order, so the same selection produces
  byte-identical shards on any host regardless of RAM, CPU count or scheduling.

Physical order is deliberately different from selection rank order: records are laid out by
``(stage priority, source_id, input_binding_id, stable_input_record_ordinal)`` so the output is
readable and diffable against the frozen inputs, while ``selection_ordinal_within_node`` retains
the rank order so the frozen selection sequence and its fingerprint remain reconstructible.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from pretrain.stage_i_audit_v1 import (
    DEFAULT_READ_WINDOW_BYTES,
    DEFAULT_SORT_CHUNK_LINES,
    RealizationAudit,
    ShardReader,
    audit_realization,
)
from pretrain.stage_i_graph_v2 import (
    STAGE_PRIORITY,
    canonical_json_bytes,
    read_authoritative_bytes,
    strict_json_object,
)

RECORD_SCHEMA = "petitgpt-stage-i-document-v1"
MANIFEST_SCHEMA = "petitgpt-stage-i-manifest-v1"
COMPLETE_MARKER = "COMPLETE"
MANIFEST_FILENAME = "manifest.json"
DOCUMENTS_DIRNAME = "documents"

# Declared, versioned, host-independent sharding. Whole records only; the shard boundary is a
# record count over the fully determined physical order, so it cannot depend on memory, worker
# count or arrival timing. Recorded in the candidate I plan and re-verified by the consumer.
SHARD_POLICY_VERSION = "petitgpt-stage-i-shard-policy-v1"
RECORDS_PER_SHARD = 50_000

RECORD_FIELDS = (
    "schema_version",
    "stage",
    "source_id",
    "input_binding_id",
    "stable_input_record_ordinal",
    "input_record_sha256",
    "raw_sha256",
    "cleaned_text_sha256",
    "canonical_fingerprint",
    "selection_ordinal_within_node",
    "content_token_count",
    "serialized_token_count",
    "training_text",
)
_RECORD_FIELD_SET = frozenset(RECORD_FIELDS)

_HEX64 = frozenset("0123456789abcdef")


class OutputError(RuntimeError):
    """Fail-closed Stage-I output condition."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OutputError(message)


def _hex64(value: Any, where: str) -> str:
    _require(
        type(value) is str and len(value) == 64 and set(value) <= _HEX64,
        f"{where} must be 64 lowercase hex characters",
    )
    return value


def _exact_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    _require(type(value) is int, f"{where} must be an exact integer")
    if minimum is not None:
        _require(value >= minimum, f"{where} must be >= {minimum}")
    return value


def _exact_str(value: Any, where: str) -> str:
    _require(type(value) is str, f"{where} must be a string")
    return value


# --------------------------------------------------------------------- record contract


def build_record(
    *,
    stage: str,
    source_id: str,
    input_binding_id: str,
    stable_input_record_ordinal: int,
    input_record_sha256: str,
    raw_sha256: str,
    cleaned_text_sha256: str,
    canonical_fingerprint: str,
    selection_ordinal_within_node: int,
    content_token_count: int,
    serialized_token_count: int,
    training_text: str,
) -> dict[str, Any]:
    record = {
        "schema_version": RECORD_SCHEMA,
        "stage": stage,
        "source_id": source_id,
        "input_binding_id": input_binding_id,
        "stable_input_record_ordinal": stable_input_record_ordinal,
        "input_record_sha256": input_record_sha256,
        "raw_sha256": raw_sha256,
        "cleaned_text_sha256": cleaned_text_sha256,
        "canonical_fingerprint": canonical_fingerprint,
        "selection_ordinal_within_node": selection_ordinal_within_node,
        "content_token_count": content_token_count,
        "serialized_token_count": serialized_token_count,
        "training_text": training_text,
    }
    validate_record(record)
    return record


def validate_record(record: Any) -> dict[str, Any]:
    """Closed-schema validation. Extra, missing and mistyped fields are all rejected."""
    _require(type(record) is dict, "record must be a JSON object")
    present = frozenset(record)
    unknown = sorted(present - _RECORD_FIELD_SET)
    missing = sorted(_RECORD_FIELD_SET - present)
    _require(not unknown, f"record carries unknown field(s): {unknown}")
    _require(not missing, f"record is missing field(s): {missing}")

    _require(
        _exact_str(record["schema_version"], "record.schema_version") == RECORD_SCHEMA,
        "record carries the wrong schema version",
    )
    stage = _exact_str(record["stage"], "record.stage")
    _require(stage in STAGE_PRIORITY, f"record.stage {stage!r} is not a known stage")
    _require(bool(_exact_str(record["source_id"], "record.source_id")), "record.source_id is empty")
    _require(
        bool(_exact_str(record["input_binding_id"], "record.input_binding_id")),
        "record.input_binding_id is empty",
    )
    _exact_int(
        record["stable_input_record_ordinal"], "record.stable_input_record_ordinal", minimum=0
    )
    _hex64(record["input_record_sha256"], "record.input_record_sha256")
    _hex64(record["raw_sha256"], "record.raw_sha256")
    _hex64(record["cleaned_text_sha256"], "record.cleaned_text_sha256")
    _hex64(record["canonical_fingerprint"], "record.canonical_fingerprint")
    _exact_int(
        record["selection_ordinal_within_node"], "record.selection_ordinal_within_node", minimum=0
    )
    content = _exact_int(record["content_token_count"], "record.content_token_count", minimum=0)
    serialized = _exact_int(
        record["serialized_token_count"], "record.serialized_token_count", minimum=1
    )
    _require(
        serialized == content + 2,
        "record.serialized_token_count must be content_token_count plus the two frozen "
        "BOS/EOS boundary tokens",
    )
    text = _exact_str(record["training_text"], "record.training_text")
    _require(bool(text), "record.training_text must not be empty")
    return record


def record_sort_key(record: Mapping[str, Any]) -> tuple[int, str, str, int]:
    """The frozen physical layout order, distinct from selection rank order."""
    return (
        STAGE_PRIORITY[record["stage"]],
        record["source_id"],
        record["input_binding_id"],
        record["stable_input_record_ordinal"],
    )


def shard_name(index: int) -> str:
    return f"documents-{index:05d}.jsonl"


def plan_shards(record_count: int) -> int:
    """How many shards a given record count produces under the declared policy."""
    _exact_int(record_count, "record_count", minimum=0)
    if record_count == 0:
        return 0
    return (record_count + RECORDS_PER_SHARD - 1) // RECORDS_PER_SHARD


# --------------------------------------------------------------------- manifest contract

_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "record_schema_version",
    "shard_policy_version",
    "records_per_shard",
    "shards",
    "totals",
    "nodes",
    "ownership_matrix",
    "bindings",
    "environment",
    "h_binding",
    "stage_i_run",
})
_SHARD_FIELDS = frozenset({"name", "records", "bytes", "sha256"})
_TOTALS_FIELDS = frozenset({
    "records",
    "content_tokens",
    "serialized_tokens",
    "unique_cleaned_identities",
    "shards",
})
_NODE_FIELDS = frozenset({
    "source_id",
    "stage",
    "target_serialized_tokens",
    "branch",
    "selection_mode",
    "selected_identities",
    "selected_serialized_tokens",
    "selection_fingerprint",
    "crossing_identity",
    "actual_overshoot_tokens",
})
_MARKER_FIELDS = frozenset({
    "marker",
    "manifest_sha256",
    "record_schema_version",
    "manifest_schema_version",
    "h_run_identity",
    "stage_i_run_identity",
})

# Closed, versioned field list for the published Stage-I run identity. Deliberately explicit: an
# open-ended dictionary digest would change meaning whenever a key was added, and would let two
# runs differing in an unlisted field claim the same identity.
_STAGE_I_RUN_FIELDS = frozenset({
    "run_identity",
    "candidate_i_plan_sha256",
    "implementation_commit",
    "implementation_bundle_sha256",
    "plan_schema_version",
    "output_schema_version",
    "manifest_schema_version",
    "shard_policy_version",
    "records_per_shard",
    "h_run_identity",
    "h_complete_sha256",
    "h_census_sha256",
    "h_predictions_sha256",
    "owner_graph_sha256",
})
STAGE_I_RUN_IDENTITY_SCHEMA = "petitgpt-stage-i-run-identity-v1"


def recompute_stage_i_run_identity(stage_i_run: Mapping[str, Any]) -> str:
    """Recompute the published run identity from its own explicit fields.

    Defined here as well as in the driver so the consumer can verify the identity without
    importing the producer: a run that claims an identity its own fields do not generate is
    rejected, so a producer cannot certify some other internally-consistent run as equivalent to
    the externally authorized one.
    """
    payload = {
        "schema_version": STAGE_I_RUN_IDENTITY_SCHEMA,
        "candidate_plan_sha256": stage_i_run["candidate_i_plan_sha256"],
        "implementation_commit": stage_i_run["implementation_commit"],
        "implementation_bundle_sha256": stage_i_run["implementation_bundle_sha256"],
        "plan_schema_version": stage_i_run["plan_schema_version"],
        "output_schema_version": stage_i_run["output_schema_version"],
        "manifest_schema_version": stage_i_run["manifest_schema_version"],
        "shard_policy_version": stage_i_run["shard_policy_version"],
        "records_per_shard": stage_i_run["records_per_shard"],
        "h_run_identity": stage_i_run["h_run_identity"],
        "h_complete_sha256": stage_i_run["h_complete_sha256"],
        "h_census_sha256": stage_i_run["h_census_sha256"],
        "h_predictions_sha256": stage_i_run["h_predictions_sha256"],
        "owner_graph_sha256": stage_i_run["owner_graph_sha256"],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _closed(obj: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    _require(type(obj) is dict, f"{where} must be a JSON object")
    present = frozenset(obj)
    unknown = sorted(present - fields)
    missing = sorted(fields - present)
    _require(not unknown, f"{where} carries unknown field(s): {unknown}")
    _require(not missing, f"{where} is missing field(s): {missing}")
    return obj


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Everything about the manifest that must hold before COMPLETE may be written."""
    obj = _closed(manifest, _MANIFEST_FIELDS, "manifest")
    _require(
        _exact_str(obj["schema_version"], "manifest.schema_version") == MANIFEST_SCHEMA,
        "manifest carries the wrong schema version",
    )
    _require(
        _exact_str(obj["record_schema_version"], "manifest.record_schema_version") == RECORD_SCHEMA,
        "manifest record schema mismatch",
    )
    _require(
        _exact_str(obj["shard_policy_version"], "manifest.shard_policy_version")
        == SHARD_POLICY_VERSION,
        "manifest shard policy mismatch",
    )
    _require(
        _exact_int(obj["records_per_shard"], "manifest.records_per_shard", minimum=1)
        == RECORDS_PER_SHARD,
        "manifest records_per_shard disagrees with the declared policy",
    )

    shards = obj["shards"]
    _require(type(shards) is list, "manifest.shards must be a list")
    total_records = 0
    for index, shard in enumerate(shards):
        entry = _closed(shard, _SHARD_FIELDS, f"manifest.shards[{index}]")
        _require(
            _exact_str(entry["name"], f"manifest.shards[{index}].name") == shard_name(index),
            f"manifest.shards[{index}] is not in canonical shard order",
        )
        records = _exact_int(entry["records"], f"manifest.shards[{index}].records", minimum=1)
        _exact_int(entry["bytes"], f"manifest.shards[{index}].bytes", minimum=1)
        _hex64(entry["sha256"], f"manifest.shards[{index}].sha256")
        _require(
            records <= RECORDS_PER_SHARD,
            f"manifest.shards[{index}] exceeds the declared records_per_shard",
        )
        if index < len(shards) - 1:
            _require(
                records == RECORDS_PER_SHARD,
                f"manifest.shards[{index}] is short but is not the final shard",
            )
        total_records += records

    totals = _closed(obj["totals"], _TOTALS_FIELDS, "manifest.totals")
    for key in _TOTALS_FIELDS:
        _exact_int(totals[key], f"manifest.totals.{key}", minimum=0)
    _require(
        totals["records"] == total_records,
        "manifest.totals.records disagrees with the per-shard counts",
    )
    _require(
        totals["shards"] == len(shards), "manifest.totals.shards disagrees with the shard list"
    )
    _require(
        totals["unique_cleaned_identities"] == totals["records"],
        "manifest.totals: every published record must be a distinct cleaned identity",
    )

    nodes = obj["nodes"]
    _require(type(nodes) is list and nodes, "manifest.nodes must be a non-empty list")
    node_records = 0
    node_tokens = 0
    seen: set[str] = set()
    previous_key: tuple[int, str] | None = None
    for index, node in enumerate(nodes):
        entry = _closed(node, _NODE_FIELDS, f"manifest.nodes[{index}]")
        source_id = _exact_str(entry["source_id"], f"manifest.nodes[{index}].source_id")
        _require(source_id not in seen, f"manifest.nodes: duplicate source_id {source_id!r}")
        seen.add(source_id)
        stage = _exact_str(entry["stage"], f"manifest.nodes[{index}].stage")
        _require(stage in STAGE_PRIORITY, f"manifest.nodes[{index}].stage is not a known stage")
        key = (STAGE_PRIORITY[stage], source_id)
        _require(
            previous_key is None or key > previous_key,
            "manifest.nodes must follow the frozen ascending (stage_priority, source_id) order",
        )
        previous_key = key
        _exact_int(
            entry["target_serialized_tokens"],
            f"manifest.nodes[{index}].target_serialized_tokens",
            minimum=1,
        )
        _require(
            _exact_str(entry["branch"], f"manifest.nodes[{index}].branch")
            in {"ORDINARY", "PRIMARY_GE4", "FALLBACK_RANKED_GE3"},
            f"manifest.nodes[{index}].branch is invalid",
        )
        _require(
            _exact_str(entry["selection_mode"], f"manifest.nodes[{index}].selection_mode")
            in {"SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC"},
            f"manifest.nodes[{index}].selection_mode is invalid",
        )
        node_records += _exact_int(
            entry["selected_identities"], f"manifest.nodes[{index}].selected_identities", minimum=0
        )
        node_tokens += _exact_int(
            entry["selected_serialized_tokens"],
            f"manifest.nodes[{index}].selected_serialized_tokens",
            minimum=0,
        )
        _hex64(entry["selection_fingerprint"], f"manifest.nodes[{index}].selection_fingerprint")
        _exact_int(
            entry["actual_overshoot_tokens"],
            f"manifest.nodes[{index}].actual_overshoot_tokens",
            minimum=0,
        )
        crossing = entry["crossing_identity"]
        if crossing is not None:
            _hex64(crossing, f"manifest.nodes[{index}].crossing_identity")
    _require(
        node_records == totals["records"],
        "manifest.totals.records disagrees with the per-node selected identity counts",
    )
    _require(
        node_tokens == totals["serialized_tokens"],
        "manifest.totals.serialized_tokens disagrees with the per-node selected token counts",
    )

    ownership = obj["ownership_matrix"]
    _require(type(ownership) is dict, "manifest.ownership_matrix must be a JSON object")
    for consumer, owners in ownership.items():
        _exact_str(consumer, "manifest.ownership_matrix key")
        _require(
            type(owners) is dict and owners,
            f"manifest.ownership_matrix[{consumer!r}] must be a non-empty object",
        )
        for owner, count in owners.items():
            _exact_str(owner, "manifest.ownership_matrix owner key")
            _exact_int(count, f"manifest.ownership_matrix[{consumer!r}][{owner!r}]", minimum=1)

    bindings = obj["bindings"]
    _require(type(bindings) is dict and bindings, "manifest.bindings must be a non-empty object")
    for binding_id, digest in bindings.items():
        _exact_str(binding_id, "manifest.bindings key")
        _hex64(digest, f"manifest.bindings[{binding_id!r}]")

    environment = obj["environment"]
    _require(type(environment) is dict, "manifest.environment must be a JSON object")
    for key in ("python_version", "tokenizers_version", "python_executable"):
        _require(key in environment, f"manifest.environment is missing {key!r}")
        _exact_str(environment[key], f"manifest.environment.{key}")

    h_binding = obj["h_binding"]
    _require(type(h_binding) is dict, "manifest.h_binding must be a JSON object")
    for key in (
        "h_run_identity",
        "h_census_sha256",
        "h_predictions_sha256",
        "h_complete_sha256",
        "h_candidate_plan_sha256",
        "h_implementation_bundle_sha256",
        "owner_graph_sha256",
    ):
        _require(key in h_binding, f"manifest.h_binding is missing {key!r}")
        _hex64(h_binding[key], f"manifest.h_binding.{key}")

    stage_i_run = _closed(obj["stage_i_run"], _STAGE_I_RUN_FIELDS, "manifest.stage_i_run")
    for key in (
        "run_identity",
        "candidate_i_plan_sha256",
        "implementation_bundle_sha256",
        "h_run_identity",
        "h_complete_sha256",
        "h_census_sha256",
        "h_predictions_sha256",
        "owner_graph_sha256",
    ):
        _hex64(stage_i_run[key], f"manifest.stage_i_run.{key}")
    _exact_str(stage_i_run["implementation_commit"], "manifest.stage_i_run.implementation_commit")
    _require(
        _exact_str(
            stage_i_run["output_schema_version"], "manifest.stage_i_run.output_schema_version"
        )
        == RECORD_SCHEMA
        and _exact_str(
            stage_i_run["manifest_schema_version"], "manifest.stage_i_run.manifest_schema_version"
        )
        == MANIFEST_SCHEMA
        and _exact_str(
            stage_i_run["shard_policy_version"], "manifest.stage_i_run.shard_policy_version"
        )
        == SHARD_POLICY_VERSION,
        "manifest.stage_i_run schema/policy versions disagree with this implementation",
    )
    _exact_str(stage_i_run["plan_schema_version"], "manifest.stage_i_run.plan_schema_version")
    _require(
        _exact_int(stage_i_run["records_per_shard"], "manifest.stage_i_run.records_per_shard")
        == RECORDS_PER_SHARD,
        "manifest.stage_i_run.records_per_shard disagrees with the declared policy",
    )
    _require(
        recompute_stage_i_run_identity(stage_i_run) == stage_i_run["run_identity"],
        "manifest.stage_i_run.run_identity is not generated by its own bound fields",
    )
    _require(
        obj["h_binding"]["h_run_identity"] == stage_i_run["h_run_identity"]
        and obj["h_binding"]["h_complete_sha256"] == stage_i_run["h_complete_sha256"]
        and obj["h_binding"]["h_census_sha256"] == stage_i_run["h_census_sha256"]
        and obj["h_binding"]["h_predictions_sha256"] == stage_i_run["h_predictions_sha256"],
        "manifest.h_binding disagrees with manifest.stage_i_run",
    )

    encoded = canonical_json_bytes(obj)
    _require(
        canonical_json_bytes(strict_json_object(encoded)) == encoded,
        "manifest canonical serialisation is not a fixed point",
    )
    return obj


# --------------------------------------------------------------------- publication


def _complete_marker(manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    return {
        "marker": COMPLETE_MARKER,
        "manifest_sha256": manifest_sha256,
        "record_schema_version": RECORD_SCHEMA,
        "manifest_schema_version": MANIFEST_SCHEMA,
        "h_run_identity": manifest["h_binding"]["h_run_identity"],
        "stage_i_run_identity": manifest["stage_i_run"]["run_identity"],
    }


def _validate_complete_marker(
    marker: Any, manifest: Mapping[str, Any], manifest_sha256: str
) -> None:
    obj = _closed(marker, _MARKER_FIELDS, "COMPLETE marker")
    _require(
        _exact_str(obj["marker"], "COMPLETE marker.marker") == COMPLETE_MARKER,
        "COMPLETE marker: wrong marker literal",
    )
    _hex64(obj["manifest_sha256"], "COMPLETE marker.manifest_sha256")
    _hex64(obj["h_run_identity"], "COMPLETE marker.h_run_identity")
    _hex64(obj["stage_i_run_identity"], "COMPLETE marker.stage_i_run_identity")
    _require(
        obj == _complete_marker(manifest, manifest_sha256),
        "COMPLETE marker and manifest disagree",
    )


def _write_durable(path: Path, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_shards(staging: Path, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Write the declared shards from records already in canonical physical order.

    The caller owns the ordering contract; this only enforces that what it receives is already
    sorted, because silently re-sorting here would hide an upstream ordering defect behind a
    correct-looking output.

    Global cleaned-identity uniqueness is deliberately NOT checked here any more. It used to be an
    in-memory set, which is ~13.8M entries at production scale; the streaming realization audit
    now establishes it from the staged bytes in bounded memory, which is both cheaper and a
    stronger statement -- it holds for the files that will actually be published, not merely for
    the objects that passed through this function.
    """
    documents_dir = staging / DOCUMENTS_DIRNAME
    documents_dir.mkdir(parents=True, exist_ok=False)

    shards: list[dict[str, Any]] = []
    buffer: list[bytes] = []
    count = 0
    previous_key: tuple[int, str, str, int] | None = None

    def flush() -> None:
        if not buffer:
            return
        index = len(shards)
        payload = b"".join(buffer)
        path = documents_dir / shard_name(index)
        _write_durable(path, payload)
        shards.append({
            "name": shard_name(index),
            "records": count,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    for record in records:
        validate_record(record)
        key = record_sort_key(record)
        _require(
            previous_key is None or key > previous_key,
            "records reached the writer out of canonical physical order",
        )
        previous_key = key
        buffer.append(canonical_json_bytes(record))
        count += 1
        if count == RECORDS_PER_SHARD:
            flush()
            buffer = []
            count = 0
    flush()
    _fsync_dir(documents_dir)
    return shards


def reconcile_manifest_with_audit(manifest: Mapping[str, Any], audit: RealizationAudit) -> None:
    """The manifest must restate the physical realization exactly, field for field.

    This is the repair for the reviewed defect where COMPLETE could be published with manifest
    token totals that disagreed with the records on disk. Every number below is compared against
    one the audit derived by streaming the actual bytes, so a caller-supplied total that is merely
    internally consistent is no longer sufficient.
    """
    totals = manifest["totals"]
    for field_name, actual in (
        ("records", audit.records),
        ("content_tokens", audit.content_tokens),
        ("serialized_tokens", audit.serialized_tokens),
        ("shards", audit.shards),
        ("unique_cleaned_identities", audit.unique_cleaned_identities),
    ):
        _require(
            totals[field_name] == actual,
            f"manifest.totals.{field_name} is {totals[field_name]} but the staged records "
            f"actually contain {actual}",
        )

    declared = manifest["shards"]
    _require(
        len(declared) == len(audit.per_shard),
        f"manifest declares {len(declared)} shards but {len(audit.per_shard)} were audited",
    )
    for index, (claimed, actual) in enumerate(zip(declared, audit.per_shard, strict=True)):
        for field_name in ("name", "records", "bytes", "sha256"):
            _require(
                claimed[field_name] == actual[field_name],
                f"manifest.shards[{index}].{field_name} is {claimed[field_name]!r} but the "
                f"physical shard has {actual[field_name]!r}",
            )

    audited = {node.source_id: node for node in audit.nodes}
    for entry in manifest["nodes"]:
        source_id = entry["source_id"]
        if entry["selected_identities"] == 0:
            _require(
                source_id not in audited,
                f"manifest node {source_id} claims no selected identities but records exist",
            )
            continue
        _require(source_id in audited, f"manifest node {source_id} has no physical records")
        node = audited[source_id]
        _require(
            entry["stage"] == node.stage,
            f"manifest node {source_id} stage disagrees with its physical records",
        )
        _require(
            entry["selected_identities"] == node.records,
            f"manifest node {source_id} claims {entry['selected_identities']} identities but "
            f"{node.records} records were audited",
        )
        _require(
            entry["selected_serialized_tokens"] == node.serialized_tokens,
            f"manifest node {source_id} claims {entry['selected_serialized_tokens']} serialized "
            f"tokens but the records sum to {node.serialized_tokens}",
        )
        _require(
            entry["selection_fingerprint"] == node.selection_fingerprint,
            f"manifest node {source_id} fingerprint disagrees with the fingerprint reconstructed "
            "from its physical records in selection order",
        )
    unknown = sorted(set(audited) - {e["source_id"] for e in manifest["nodes"]})
    _require(not unknown, f"physical records exist for undeclared node(s): {unknown}")


def audit_staged_realization(
    root: Path,
    shard_names: Sequence[str],
    work_dir: Path,
    *,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
    sort_chunk_lines: int = DEFAULT_SORT_CHUNK_LINES,
) -> RealizationAudit:
    """Run the shared streaming audit over a staged or published documents directory."""
    return audit_realization(
        root / DOCUMENTS_DIRNAME,
        shard_names,
        work_dir,
        validate_record=validate_record,
        parse_record=lambda line: strict_json_object(line, where=str(root)),
        read_window_bytes=read_window_bytes,
        sort_chunk_lines=sort_chunk_lines,
    )


def publish_atomic(
    out_dir: Path,
    run_name: str,
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
    sort_chunk_lines: int = DEFAULT_SORT_CHUNK_LINES,
) -> Path:
    """Stage, audit the physical bytes, reconcile, mark COMPLETE last, then rename atomically.

    Ordering is the whole point and it is now:

        write staging data -> finalize shards -> audit the actual staged realization ->
        reconcile the manifest against that audit -> validate the manifest ->
        write the manifest -> write COMPLETE -> fsync -> atomic rename

    Nothing is certified from an in-memory object. The audit re-reads every staged byte in bounded
    memory and the manifest must agree with it exactly, so a realization whose totals do not
    describe its own records cannot become discoverable. Any failure removes both the staging tree
    and the audit scratch space and leaves prior state untouched.
    """
    final = out_dir / run_name
    _require(
        not final.exists(), f"realization directory already exists, refusing to overwrite: {final}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(out_dir.glob(f".{run_name}.staging-*")) + sorted(
        out_dir.glob(f".{run_name}.audit-*")
    )
    _require(not stale, f"stale staging/audit directories must be removed first: {stale}")

    staging = Path(tempfile.mkdtemp(prefix=f".{run_name}.staging-", dir=str(out_dir)))
    # Audit scratch lives OUTSIDE the staging tree so it can never be renamed into the published
    # result, and it is removed on every exit path.
    audit_dir = Path(tempfile.mkdtemp(prefix=f".{run_name}.audit-", dir=str(out_dir)))
    try:
        shards = write_shards(staging, records)

        # The shard list and count are only knowable after the records have been written, so the
        # publisher owns them. Copy `totals` rather than mutating it: the caller's manifest is its
        # own object and must not be edited in place.
        complete = dict(manifest)
        complete["shards"] = shards
        complete["totals"] = dict(manifest["totals"])
        complete["totals"]["shards"] = len(shards)

        audit = audit_staged_realization(
            staging,
            [entry["name"] for entry in shards],
            audit_dir,
            read_window_bytes=read_window_bytes,
            sort_chunk_lines=sort_chunk_lines,
        )
        reconcile_manifest_with_audit(complete, audit)
        validate_manifest(complete)

        manifest_bytes = canonical_json_bytes(complete)
        _write_durable(staging / MANIFEST_FILENAME, manifest_bytes)
        written, manifest_digest = read_authoritative_bytes(
            staging / MANIFEST_FILENAME, max_bytes=1 << 31
        )
        _require(
            written == manifest_bytes, "staged manifest bytes do not match what was serialised"
        )

        marker = canonical_json_bytes(_complete_marker(complete, manifest_digest))
        _write_durable(staging / COMPLETE_MARKER, marker)
        _fsync_dir(staging)
        os.replace(staging, final)
        _fsync_dir(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(audit_dir, ignore_errors=True)
    return final


# --------------------------------------------------------------------- strict consumer


def load_published_realization(
    final: Path,
    *,
    work_dir: Path | None = None,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
    sort_chunk_lines: int = DEFAULT_SORT_CHUNK_LINES,
) -> dict[str, Any]:
    """Refuse anything that is not a complete, self-consistent, physically verified realization.

    The reviewed defect was that a realization could be loaded and fully iterated without any
    physical invariant being enforced. This now runs the SAME streaming audit the publisher ran,
    over the published bytes, and requires the manifest to restate it exactly. Because publisher
    and consumer share one implementation, their two interpretations cannot drift apart.

    Everything is derived in bounded memory: records stream a line at a time, and the global
    questions (identity uniqueness, per-node ordinal contiguity, per-node fingerprints) go through
    a deterministic disk-backed sort. Nothing here scales with the size of the realization, so a
    56 GB production result validates on the same footprint as a three-record fixture.

    Scratch space is created under ``work_dir`` (a system temp directory by default), is never
    written inside the published result, and is removed on every exit path.
    """
    _require(final.is_dir(), f"{final}: not a published realization directory")
    marker_path = final / COMPLETE_MARKER
    manifest_path = final / MANIFEST_FILENAME
    _require(
        marker_path.is_file(),
        f"{final}: no COMPLETE marker; this is not publishable Stage-I output",
    )
    payload, manifest_digest = read_authoritative_bytes(manifest_path, max_bytes=1 << 31)
    marker_bytes, _ = read_authoritative_bytes(marker_path, max_bytes=1 << 20)
    manifest = strict_json_object(payload, where=f"{final}/{MANIFEST_FILENAME}")
    marker = strict_json_object(marker_bytes, where=f"{final}/{COMPLETE_MARKER}")
    _require(
        canonical_json_bytes(manifest) == payload, f"{final}: manifest bytes are not canonical"
    )
    _require(
        canonical_json_bytes(marker) == marker_bytes,
        f"{final}: COMPLETE marker bytes are not canonical",
    )
    validate_manifest(manifest)
    _validate_complete_marker(marker, manifest, manifest_digest)

    documents_dir = final / DOCUMENTS_DIRNAME
    _require(documents_dir.is_dir(), f"{final}: no documents directory")
    present = sorted(p.name for p in documents_dir.iterdir())
    expected = [entry["name"] for entry in manifest["shards"]]
    _require(
        present == sorted(expected),
        f"{final}: documents directory contents disagree with the manifest shard list",
    )

    owned_scratch = work_dir is None
    scratch = (
        Path(tempfile.mkdtemp(prefix="stage-i-consumer-audit-")) if owned_scratch else work_dir
    )
    try:
        audit = audit_staged_realization(
            final,
            expected,
            scratch,
            read_window_bytes=read_window_bytes,
            sort_chunk_lines=sort_chunk_lines,
        )
        reconcile_manifest_with_audit(manifest, audit)
    finally:
        if owned_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    return manifest


def iter_records(
    final: Path,
    manifest: Mapping[str, Any],
    *,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
) -> Iterator[dict[str, Any]]:
    """Stream every published record in canonical physical order, validating as it goes.

    Bounded: one line at a time, with each shard's digest recomputed from the bytes consumed and
    compared with the manifest at the end of that shard. This does not re-run the full audit --
    ``load_published_realization`` already did -- it is the ordinary reading path.
    """
    documents_dir = final / DOCUMENTS_DIRNAME
    for entry in manifest["shards"]:
        reader = ShardReader(documents_dir / entry["name"], read_window_bytes=read_window_bytes)
        seen = 0
        for line in reader:
            record = validate_record(strict_json_object(line, where=f"{final}/{entry['name']}"))
            _require(
                canonical_json_bytes(record) == line + b"\n",
                f"{final}: shard {entry['name']} record {seen} is not canonical",
            )
            seen += 1
            yield record
        _require(
            reader.sha256 == entry["sha256"],
            f"{final}: shard {entry['name']} SHA-256 disagrees with the manifest",
        )
        _require(
            reader.bytes_read == entry["bytes"] and seen == entry["records"],
            f"{final}: shard {entry['name']} size/record count disagrees with the manifest",
        )


__all__ = [
    "COMPLETE_MARKER",
    "DOCUMENTS_DIRNAME",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "RECORDS_PER_SHARD",
    "RECORD_FIELDS",
    "RECORD_SCHEMA",
    "SHARD_POLICY_VERSION",
    "OutputError",
    "build_record",
    "iter_records",
    "audit_staged_realization",
    "load_published_realization",
    "recompute_stage_i_run_identity",
    "reconcile_manifest_with_audit",
    "plan_shards",
    "publish_atomic",
    "record_sort_key",
    "shard_name",
    "validate_manifest",
    "validate_record",
    "write_shards",
]
