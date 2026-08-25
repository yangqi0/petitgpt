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
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from pretrain.stage_i_audit_v1 import (
    DEFAULT_READ_WINDOW_BYTES,
    DEFAULT_SORT_CHUNK_LINES,
    SELECTION_SEQUENCE_SCHEMA,
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

# R4-D: the frozen Stage-H/I execution environment. Defined here rather than in the driver so the
# published manifest's environment block can be closed against exact values without the output
# contract importing the module that consumes it. The driver re-exports these names.
FROZEN_PYTHON_EXECUTABLE = "/workspace/petitgpt/.venv/bin/python"
FROZEN_PYTHON_VERSION = "3.10.12"
FROZEN_TOKENIZERS_VERSION = "0.22.2"
ENVIRONMENT_FIELDS = ("python_executable", "python_version", "tokenizers_version")
FROZEN_ENVIRONMENT: Mapping[str, str] = MappingProxyType({
    "python_executable": FROZEN_PYTHON_EXECUTABLE,
    "python_version": FROZEN_PYTHON_VERSION,
    "tokenizers_version": FROZEN_TOKENIZERS_VERSION,
})
_ENVIRONMENT_FIELD_SET = frozenset(ENVIRONMENT_FIELDS)
ENVIRONMENT_SCHEMA = "petitgpt-stage-i-environment-v1"
BINDING_DOCUMENT_DIGESTS_SCHEMA = "petitgpt-stage-i-binding-document-digests-v1"

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
    "node_binding_projection",
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
    "selection_sequence_commitment",
    "crossing_identity",
    "actual_overshoot_tokens",
    "input_binding_ids",
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
# Closed, versioned field list for the published Stage-I run identity. Deliberately explicit: an
# open-ended dictionary digest would change meaning whenever a key was added, and would let two
# runs differing in an unlisted field claim the same identity.
#
# R3-A: the v3 identity additionally binds the authorization-time canonical state digest (Layer 1)
# and the post-Pass-1 expected-result identity (Layer 2). A published realization can therefore no
# longer be resealed into describing a different selection while keeping its name: the name is
# downstream of an expectation that was frozen to disk before the first output byte existed.
_STAGE_I_RUN_FIELDS = frozenset({
    "run_identity",
    "candidate_i_plan_sha256",
    "authorized_state_sha256",
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
    "node_binding_projection_sha256",
    "environment_sha256",
    "binding_document_digests_sha256",
    "post_pass1_result_identity_schema",
    "post_pass1_result_identity_sha256",
    "selection_sequence_commitment_version",
    "selection_sequence_commitment_map_sha256",
})
# R4-D: v4 adds the trusted environment digest and the trusted binding-document-digest projection
# to the published identity, so a manifest cannot report false canonical provenance under a name
# the authorized run generated.
STAGE_I_RUN_IDENTITY_SCHEMA = "petitgpt-stage-i-run-identity-v4"

# The Layer-1 anchors every layer below restates. Held in one place so the post-Pass-1 result, the
# published run identity and the manifest cannot drift into three slightly different opinions of
# what the authorized run was.
_AUTHORIZATION_FIELDS = frozenset({
    "candidate_i_plan_sha256",
    "authorized_state_sha256",
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
    "node_binding_projection_sha256",
    "environment_sha256",
    "binding_document_digests_sha256",
})
_AUTHORIZATION_HEX_FIELDS = (
    "candidate_i_plan_sha256",
    "authorized_state_sha256",
    "implementation_bundle_sha256",
    "h_run_identity",
    "h_complete_sha256",
    "h_census_sha256",
    "h_predictions_sha256",
    "owner_graph_sha256",
    "node_binding_projection_sha256",
    "environment_sha256",
    "binding_document_digests_sha256",
)


def node_binding_projection_sha256(projection: Mapping[str, Sequence[str]]) -> str:
    """Digest over the closed node -> authorized-binding projection.

    Derived from the owner-authorized plan, never from whichever bindings happen to appear in the
    output, and folded into the run identity so a manifest cannot quietly widen it.
    """
    payload = {
        "schema_version": NODE_BINDING_PROJECTION_SCHEMA,
        "projection": {k: sorted(v) for k, v in sorted(projection.items())},
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


NODE_BINDING_PROJECTION_SCHEMA = "petitgpt-stage-i-node-binding-projection-v1"
SELECTION_SEQUENCE_MAP_SCHEMA = "petitgpt-stage-i-selection-sequence-map-v1"
# R4-D: v2 adds the trusted environment projection and the trusted binding-document-digest
# projection. The serialized semantics changed, so the version changed with them.
PASS1_RESULT_SCHEMA = "petitgpt-stage-i-pass1-result-v2"


def environment_sha256(environment: Mapping[str, str]) -> str:
    """Closed, versioned identity of the execution environment a run was authorized under.

    R4-D: the manifest recorded an environment block that nothing checked, so a published result
    could claim a different interpreter, Python or tokenizers build than the one that produced it.
    This digest is derived from canonical Layer-1 state, carried by the post-Pass-1 result and
    folded into the published run identity.
    """
    payload = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "environment": {key: environment[key] for key in ENVIRONMENT_FIELDS},
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def binding_document_digests_sha256(digests: Mapping[str, str]) -> str:
    """Closed, versioned identity of the input-binding -> document-digest projection.

    R4-D: the manifest's ``bindings`` map was an unchecked observation, so a release digest could
    be replaced with zeros and still publish. This commits to the exact map the authorized owner
    graph binds.
    """
    payload = {
        "schema_version": BINDING_DOCUMENT_DIGESTS_SCHEMA,
        "digests": dict(sorted(digests.items())),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def selection_sequence_commitment_map_sha256(commitments: Mapping[str, str]) -> str:
    """Closed canonical identity of the per-node ordinal -> identity sequence commitments.

    R2 committed to each node's sequence but published the expectation only inside the manifest,
    so a full reseal could restate it. This digest is folded into the published run identity via
    the post-Pass-1 result, which is frozen before materialization, so the map cannot be restated
    after the fact without changing the name of the run.
    """
    payload = {
        "schema_version": SELECTION_SEQUENCE_MAP_SCHEMA,
        "commitment_version": SELECTION_SEQUENCE_SCHEMA,
        "commitments": dict(sorted(commitments.items())),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _run_identity_payload(
    authorization: Mapping[str, Any],
    *,
    post_pass1_result_identity_schema: str,
    post_pass1_result_identity_sha256: str,
    selection_sequence_commitment_version: str,
    selection_sequence_commitment_map_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": STAGE_I_RUN_IDENTITY_SCHEMA,
        "candidate_plan_sha256": authorization["candidate_i_plan_sha256"],
        "authorized_state_sha256": authorization["authorized_state_sha256"],
        "implementation_commit": authorization["implementation_commit"],
        "implementation_bundle_sha256": authorization["implementation_bundle_sha256"],
        "plan_schema_version": authorization["plan_schema_version"],
        "output_schema_version": authorization["output_schema_version"],
        "manifest_schema_version": authorization["manifest_schema_version"],
        "shard_policy_version": authorization["shard_policy_version"],
        "records_per_shard": authorization["records_per_shard"],
        "h_run_identity": authorization["h_run_identity"],
        "h_complete_sha256": authorization["h_complete_sha256"],
        "h_census_sha256": authorization["h_census_sha256"],
        "h_predictions_sha256": authorization["h_predictions_sha256"],
        "owner_graph_sha256": authorization["owner_graph_sha256"],
        "node_binding_projection_sha256": authorization["node_binding_projection_sha256"],
        "environment_sha256": authorization["environment_sha256"],
        "binding_document_digests_sha256": authorization["binding_document_digests_sha256"],
        "post_pass1_result_identity_schema": post_pass1_result_identity_schema,
        "post_pass1_result_identity_sha256": post_pass1_result_identity_sha256,
        "selection_sequence_commitment_version": selection_sequence_commitment_version,
        "selection_sequence_commitment_map_sha256": selection_sequence_commitment_map_sha256,
    }


def stage_i_published_run_identity(
    authorization: Mapping[str, Any],
    *,
    post_pass1_result_identity_sha256: str,
    selection_sequence_commitment_map_sha256: str,
    post_pass1_result_identity_schema: str = PASS1_RESULT_SCHEMA,
    selection_sequence_commitment_version: str = SELECTION_SEQUENCE_SCHEMA,
) -> str:
    """The published run's name: Layer-1 authorization plus the Layer-2 expected result.

    Ordering matters and is not circular: the post-Pass-1 result binds the Layer-1 anchors, this
    identity binds the post-Pass-1 result, and the manifest binds this identity. Nothing upstream
    ever quotes anything downstream of it.
    """
    payload = _run_identity_payload(
        authorization,
        post_pass1_result_identity_schema=post_pass1_result_identity_schema,
        post_pass1_result_identity_sha256=post_pass1_result_identity_sha256,
        selection_sequence_commitment_version=selection_sequence_commitment_version,
        selection_sequence_commitment_map_sha256=selection_sequence_commitment_map_sha256,
    )
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def recompute_stage_i_run_identity(stage_i_run: Mapping[str, Any]) -> str:
    """Recompute the published run identity from its own explicit fields.

    Defined here as well as in the driver so the consumer can verify the identity without
    importing the producer: a run that claims an identity its own fields do not generate is
    rejected. This is an internal-consistency check only -- it proves the manifest did not
    contradict itself, never that it describes the authorized run. That second question is
    answered by comparing against the trusted post-Pass-1 expected result.
    """
    return stage_i_published_run_identity(
        stage_i_run,
        post_pass1_result_identity_schema=stage_i_run["post_pass1_result_identity_schema"],
        post_pass1_result_identity_sha256=stage_i_run["post_pass1_result_identity_sha256"],
        selection_sequence_commitment_version=stage_i_run["selection_sequence_commitment_version"],
        selection_sequence_commitment_map_sha256=stage_i_run[
            "selection_sequence_commitment_map_sha256"
        ],
    )


# ------------------------------------------------- Layer 2: the post-Pass-1 expected result

_PASS1_TOP_FIELDS = frozenset({
    "schema_version",
    "authorization",
    "environment",
    "binding_document_digests",
    "selection_sequence_commitment_version",
    "selection_sequence_commitments",
    "selection_sequence_commitment_map_sha256",
    "node_binding_projection",
    "authorized_input_binding_ids",
    "nodes",
    "totals",
    "ownership_matrix",
    "h_i_gate",
    "h_binding",
})
_PASS1_NODE_FIELDS = frozenset({
    "source_id",
    "stage",
    "target_serialized_tokens",
    "branch",
    "selection_mode",
    "selected_identities",
    "selected_serialized_tokens",
    "selection_fingerprint",
    "selection_sequence_commitment",
    "crossing_identity",
    "actual_overshoot_tokens",
    "input_binding_ids",
})
_PASS1_TOTALS_FIELDS = frozenset({
    "records",
    "content_tokens",
    "serialized_tokens",
    "unique_cleaned_identities",
})
_PASS1_GATE_FIELDS = frozenset({
    "ALL_H_I_BRANCHES_MATCH",
    "ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH",
    "ALL_H_I_FINGERPRINTS_MATCH",
    "ALL_H_I_CROSSING_IDENTITIES_MATCH",
    "ALL_H_I_OVERSHOOTS_MATCH",
    "OWNERSHIP_MATRIX_MATCH",
    "ALL_NODES_MATCH",
})
_H_BINDING_FIELDS = frozenset({
    "h_run_identity",
    "h_census_sha256",
    "h_predictions_sha256",
    "h_complete_sha256",
    "h_candidate_plan_sha256",
    "h_implementation_bundle_sha256",
    "owner_graph_sha256",
})


@dataclass(frozen=True)
class TrustedExpectedNode:
    """What Pass 1 committed one node to be, before any of it existed on disk."""

    source_id: str
    stage: str
    target_serialized_tokens: int
    branch: str
    selection_mode: str
    selected_identities: int
    selected_serialized_tokens: int
    selection_fingerprint: str
    selection_sequence_commitment: str
    crossing_identity: str | None
    actual_overshoot_tokens: int
    input_binding_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrustedExpectedResult:
    """Layer 2. The expectation a physical realization must prove itself equal to.

    This is deliberately a separate object from the manifest and is never constructed from one.
    The publisher receives it from the driver, which built it from the Pass-1 selection ledger and
    froze it to disk before materialization; the strict consumer receives it from its caller, who
    supplies the frozen artifact and its owner-held digest. Layer 3 may prove equality to this. It
    may not define it.
    """

    result_identity_schema: str
    result_identity_sha256: str
    authorization: Mapping[str, Any]
    stage_i_run_identity: str
    selection_sequence_commitment_version: str
    selection_sequence_commitments: Mapping[str, str]
    selection_sequence_commitment_map_sha256: str
    node_binding_projection: Mapping[str, tuple[str, ...]]
    node_binding_projection_sha256: str
    authorized_input_binding_ids: tuple[str, ...]
    environment: Mapping[str, str]
    environment_sha256: str
    binding_document_digests: Mapping[str, str]
    binding_document_digests_sha256: str
    nodes: tuple[TrustedExpectedNode, ...]
    totals: Mapping[str, int]
    ownership_matrix: Mapping[str, Mapping[str, int]]
    h_binding: Mapping[str, str]

    def node(self, source_id: str) -> TrustedExpectedNode:
        for entry in self.nodes:
            if entry.source_id == source_id:
                return entry
        raise OutputError(f"trusted expected result has no node {source_id!r}")


def validate_pass1_result(payload: Any) -> dict[str, Any]:
    """Closed-schema validation of the post-Pass-1 expected result, at every level."""
    obj = _closed(payload, _PASS1_TOP_FIELDS, "pass-1 result")
    _require(
        _exact_str(obj["schema_version"], "pass-1 result.schema_version") == PASS1_RESULT_SCHEMA,
        f"pass-1 result carries the wrong schema version; expected {PASS1_RESULT_SCHEMA}",
    )
    authorization = _closed(obj["authorization"], _AUTHORIZATION_FIELDS, "pass-1 authorization")
    for key in _AUTHORIZATION_HEX_FIELDS:
        _hex64(authorization[key], f"pass-1 authorization.{key}")
    _exact_str(authorization["implementation_commit"], "pass-1 authorization.implementation_commit")
    for key in (
        "plan_schema_version",
        "output_schema_version",
        "manifest_schema_version",
        "shard_policy_version",
    ):
        _require(
            bool(_exact_str(authorization[key], f"pass-1 authorization.{key}")),
            f"pass-1 authorization.{key} must be a non-empty string",
        )
    _require(
        authorization["output_schema_version"] == RECORD_SCHEMA
        and authorization["manifest_schema_version"] == MANIFEST_SCHEMA
        and authorization["shard_policy_version"] == SHARD_POLICY_VERSION,
        "pass-1 authorization schema/policy versions disagree with this implementation",
    )
    _require(
        _exact_int(authorization["records_per_shard"], "pass-1 authorization.records_per_shard")
        == RECORDS_PER_SHARD,
        "pass-1 authorization.records_per_shard disagrees with the declared policy",
    )

    _require(
        _exact_str(
            obj["selection_sequence_commitment_version"],
            "pass-1 result.selection_sequence_commitment_version",
        )
        == SELECTION_SEQUENCE_SCHEMA,
        "pass-1 result was written against a different selection-sequence commitment version",
    )

    # R4-D: the trusted environment. Closed on shape AND on the frozen values, so a Layer-2
    # artifact cannot legitimise a run under a different interpreter or tokenizers build.
    environment = _closed(obj["environment"], _ENVIRONMENT_FIELD_SET, "pass-1 result.environment")
    for key in ENVIRONMENT_FIELDS:
        _require(
            _exact_str(environment[key], f"pass-1 result.environment.{key}")
            == FROZEN_ENVIRONMENT[key],
            f"pass-1 result.environment.{key} is {environment[key]!r} but the frozen Stage-I "
            f"environment requires {FROZEN_ENVIRONMENT[key]!r}",
        )
    _require(
        environment_sha256(environment) == authorization["environment_sha256"],
        "pass-1 result.environment does not generate its bound environment digest",
    )

    # R4-D: the trusted input-binding -> document-digest projection.
    binding_digests = obj["binding_document_digests"]
    _require(
        type(binding_digests) is dict and binding_digests,
        "pass-1 result.binding_document_digests must be a non-empty object",
    )
    for binding_id, digest in sorted(binding_digests.items()):
        _exact_str(binding_id, "pass-1 result.binding_document_digests key")
        _hex64(digest, f"pass-1 result.binding_document_digests[{binding_id!r}]")
    _require(
        binding_document_digests_sha256(binding_digests)
        == authorization["binding_document_digests_sha256"],
        "pass-1 result.binding_document_digests does not generate its bound projection digest",
    )

    projection = obj["node_binding_projection"]
    _require(
        type(projection) is dict and projection,
        "pass-1 result.node_binding_projection must be a non-empty object",
    )
    for source_id, allowed in sorted(projection.items()):
        _exact_str(source_id, "pass-1 result.node_binding_projection key")
        _require(
            type(allowed) is list and allowed and allowed == sorted(set(allowed)),
            f"pass-1 result.node_binding_projection[{source_id!r}] must be a sorted unique "
            "non-empty list",
        )
        for binding_id in allowed:
            _exact_str(binding_id, f"pass-1 result.node_binding_projection[{source_id!r}][]")
    _require(
        node_binding_projection_sha256(projection)
        == authorization["node_binding_projection_sha256"],
        "pass-1 result.node_binding_projection does not generate its bound projection digest",
    )

    authorized_bindings = obj["authorized_input_binding_ids"]
    _require(
        type(authorized_bindings) is list
        and authorized_bindings
        and authorized_bindings == sorted(set(authorized_bindings)),
        "pass-1 result.authorized_input_binding_ids must be a sorted unique non-empty list",
    )
    for binding_id in authorized_bindings:
        _exact_str(binding_id, "pass-1 result.authorized_input_binding_ids[]")
    unknown = sorted(
        {b for allowed in projection.values() for b in allowed} - set(authorized_bindings)
    )
    _require(
        not unknown,
        f"pass-1 result projects binding(s) outside the plan-authorized global set: {unknown}",
    )
    _require(
        set(binding_digests) == set(authorized_bindings),
        "pass-1 result.binding_document_digests must cover exactly the plan-authorized global "
        f"input-binding set; digests={sorted(binding_digests)} "
        f"authorized={sorted(authorized_bindings)}",
    )

    commitments = obj["selection_sequence_commitments"]
    _require(
        type(commitments) is dict and commitments,
        "pass-1 result.selection_sequence_commitments must be a non-empty object",
    )
    for source_id, value in sorted(commitments.items()):
        _exact_str(source_id, "pass-1 result.selection_sequence_commitments key")
        _hex64(value, f"pass-1 result.selection_sequence_commitments[{source_id!r}]")
    _require(
        selection_sequence_commitment_map_sha256(commitments)
        == _hex64(
            obj["selection_sequence_commitment_map_sha256"],
            "pass-1 result.selection_sequence_commitment_map_sha256",
        ),
        "pass-1 result.selection_sequence_commitment_map_sha256 does not describe its own map",
    )

    nodes = obj["nodes"]
    _require(type(nodes) is list and nodes, "pass-1 result.nodes must be a non-empty list")
    seen: set[str] = set()
    previous_key: tuple[int, str] | None = None
    node_records = 0
    node_tokens = 0
    for index, node in enumerate(nodes):
        entry = _closed(node, _PASS1_NODE_FIELDS, f"pass-1 result.nodes[{index}]")
        source_id = _exact_str(entry["source_id"], f"pass-1 result.nodes[{index}].source_id")
        _require(source_id not in seen, f"pass-1 result.nodes: duplicate source_id {source_id!r}")
        seen.add(source_id)
        stage = _exact_str(entry["stage"], f"pass-1 result.nodes[{index}].stage")
        _require(stage in STAGE_PRIORITY, f"pass-1 result.nodes[{index}].stage is not a stage")
        key = (STAGE_PRIORITY[stage], source_id)
        _require(
            previous_key is None or key > previous_key,
            "pass-1 result.nodes must follow the frozen ascending (stage_priority, source_id) "
            "order",
        )
        previous_key = key
        _exact_int(
            entry["target_serialized_tokens"],
            f"pass-1 result.nodes[{index}].target_serialized_tokens",
            minimum=1,
        )
        _require(
            _exact_str(entry["branch"], f"pass-1 result.nodes[{index}].branch")
            in {"ORDINARY", "PRIMARY_GE4", "FALLBACK_RANKED_GE3"},
            f"pass-1 result.nodes[{index}].branch is invalid",
        )
        _require(
            _exact_str(entry["selection_mode"], f"pass-1 result.nodes[{index}].selection_mode")
            in {"SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC"},
            f"pass-1 result.nodes[{index}].selection_mode is invalid",
        )
        node_records += _exact_int(
            entry["selected_identities"],
            f"pass-1 result.nodes[{index}].selected_identities",
            minimum=0,
        )
        node_tokens += _exact_int(
            entry["selected_serialized_tokens"],
            f"pass-1 result.nodes[{index}].selected_serialized_tokens",
            minimum=0,
        )
        _hex64(
            entry["selection_fingerprint"], f"pass-1 result.nodes[{index}].selection_fingerprint"
        )
        _hex64(
            entry["selection_sequence_commitment"],
            f"pass-1 result.nodes[{index}].selection_sequence_commitment",
        )
        _require(
            commitments.get(source_id) == entry["selection_sequence_commitment"],
            f"pass-1 result.nodes[{index}] sequence commitment disagrees with the committed map",
        )
        _exact_int(
            entry["actual_overshoot_tokens"],
            f"pass-1 result.nodes[{index}].actual_overshoot_tokens",
            minimum=0,
        )
        crossing = entry["crossing_identity"]
        if crossing is not None:
            _hex64(crossing, f"pass-1 result.nodes[{index}].crossing_identity")
        declared = entry["input_binding_ids"]
        _require(
            type(declared) is list and declared and declared == sorted(set(declared)),
            f"pass-1 result.nodes[{index}].input_binding_ids must be sorted, unique, non-empty",
        )
        allowed = projection.get(source_id)
        _require(
            allowed is not None,
            f"pass-1 result.nodes[{index}] is outside the authorized node/binding projection",
        )
        _require(
            set(declared) <= set(allowed),
            f"pass-1 result.nodes[{index}] draws from a binding outside its authorized projection",
        )
    _require(
        set(commitments) == seen,
        "pass-1 result.selection_sequence_commitments must cover exactly the declared nodes",
    )
    _require(
        set(projection) == seen,
        "pass-1 result.node_binding_projection must cover exactly the declared nodes",
    )

    totals = _closed(obj["totals"], _PASS1_TOTALS_FIELDS, "pass-1 result.totals")
    for key in sorted(_PASS1_TOTALS_FIELDS):
        _exact_int(totals[key], f"pass-1 result.totals.{key}", minimum=0)
    _require(
        totals["records"] == node_records,
        "pass-1 result.totals.records disagrees with the per-node selected identity counts",
    )
    _require(
        totals["serialized_tokens"] == node_tokens,
        "pass-1 result.totals.serialized_tokens disagrees with the per-node token counts",
    )
    _require(
        totals["unique_cleaned_identities"] == totals["records"],
        "pass-1 result.totals: every selected identity must be distinct",
    )

    gate = _closed(obj["h_i_gate"], _PASS1_GATE_FIELDS, "pass-1 result.h_i_gate")
    for key in sorted(_PASS1_GATE_FIELDS):
        _require(
            gate[key] is True,
            f"pass-1 result.h_i_gate.{key} is not True; this result never passed the H/I gate",
        )

    h_binding = _closed(obj["h_binding"], _H_BINDING_FIELDS, "pass-1 result.h_binding")
    for key in sorted(_H_BINDING_FIELDS):
        _hex64(h_binding[key], f"pass-1 result.h_binding.{key}")
    _require(
        h_binding["h_run_identity"] == authorization["h_run_identity"]
        and h_binding["h_census_sha256"] == authorization["h_census_sha256"]
        and h_binding["h_predictions_sha256"] == authorization["h_predictions_sha256"]
        and h_binding["h_complete_sha256"] == authorization["h_complete_sha256"]
        and h_binding["owner_graph_sha256"] == authorization["owner_graph_sha256"],
        "pass-1 result.h_binding disagrees with its own authorization block",
    )

    ownership = obj["ownership_matrix"]
    _require(type(ownership) is dict, "pass-1 result.ownership_matrix must be a JSON object")
    for consumer, owners in ownership.items():
        _exact_str(consumer, "pass-1 result.ownership_matrix key")
        _require(
            type(owners) is dict and owners,
            f"pass-1 result.ownership_matrix[{consumer!r}] must be a non-empty object",
        )
        for owner, count in owners.items():
            _exact_str(owner, "pass-1 result.ownership_matrix owner key")
            _exact_int(count, f"pass-1 result.ownership_matrix[{consumer!r}][{owner!r}]", minimum=1)

    encoded = canonical_json_bytes(obj)
    _require(
        canonical_json_bytes(strict_json_object(encoded)) == encoded,
        "pass-1 result canonical serialisation is not a fixed point",
    )
    return obj


def trusted_expected_result(
    pass1: Mapping[str, Any], *, expected_sha256: str
) -> TrustedExpectedResult:
    """Turn a validated post-Pass-1 artifact plus its owner-held digest into the Layer-2 truth.

    ``expected_sha256`` is supplied from outside the artifact. Reading the file and trusting the
    digest it happens to hash to would make the artifact self-certifying, which is the same defect
    one layer up: an expectation must be pinned by something that did not travel with it.
    """
    obj = validate_pass1_result(pass1)
    _hex64(expected_sha256, "trusted expected result digest")
    actual = hashlib.sha256(canonical_json_bytes(obj)).hexdigest()
    _require(
        actual == expected_sha256,
        f"post-Pass-1 result digest {actual} is not the supplied {expected_sha256}",
    )
    authorization = MappingProxyType(dict(obj["authorization"]))
    identity = stage_i_published_run_identity(
        authorization,
        post_pass1_result_identity_sha256=actual,
        selection_sequence_commitment_map_sha256=obj["selection_sequence_commitment_map_sha256"],
        post_pass1_result_identity_schema=obj["schema_version"],
        selection_sequence_commitment_version=obj["selection_sequence_commitment_version"],
    )
    return TrustedExpectedResult(
        result_identity_schema=obj["schema_version"],
        result_identity_sha256=actual,
        authorization=authorization,
        stage_i_run_identity=identity,
        selection_sequence_commitment_version=obj["selection_sequence_commitment_version"],
        selection_sequence_commitments=MappingProxyType(
            dict(sorted(obj["selection_sequence_commitments"].items()))
        ),
        selection_sequence_commitment_map_sha256=obj["selection_sequence_commitment_map_sha256"],
        node_binding_projection=MappingProxyType({
            source_id: tuple(allowed)
            for source_id, allowed in sorted(obj["node_binding_projection"].items())
        }),
        node_binding_projection_sha256=authorization["node_binding_projection_sha256"],
        authorized_input_binding_ids=tuple(obj["authorized_input_binding_ids"]),
        environment=MappingProxyType({key: obj["environment"][key] for key in ENVIRONMENT_FIELDS}),
        environment_sha256=authorization["environment_sha256"],
        binding_document_digests=MappingProxyType(
            dict(sorted(obj["binding_document_digests"].items()))
        ),
        binding_document_digests_sha256=authorization["binding_document_digests_sha256"],
        nodes=tuple(
            TrustedExpectedNode(
                source_id=entry["source_id"],
                stage=entry["stage"],
                target_serialized_tokens=entry["target_serialized_tokens"],
                branch=entry["branch"],
                selection_mode=entry["selection_mode"],
                selected_identities=entry["selected_identities"],
                selected_serialized_tokens=entry["selected_serialized_tokens"],
                selection_fingerprint=entry["selection_fingerprint"],
                selection_sequence_commitment=entry["selection_sequence_commitment"],
                crossing_identity=entry["crossing_identity"],
                actual_overshoot_tokens=entry["actual_overshoot_tokens"],
                input_binding_ids=tuple(entry["input_binding_ids"]),
            )
            for entry in obj["nodes"]
        ),
        totals=MappingProxyType(dict(obj["totals"])),
        ownership_matrix=MappingProxyType({
            consumer: MappingProxyType(dict(owners))
            for consumer, owners in sorted(obj["ownership_matrix"].items())
        }),
        h_binding=MappingProxyType(dict(obj["h_binding"])),
    )


def load_trusted_expected_result(path: Path, *, expected_sha256: str) -> TrustedExpectedResult:
    """Load the frozen post-Pass-1 artifact from outside the published realization.

    The strict consumer takes its expectation from here, never from the manifest it is checking.
    """
    payload, digest = read_authoritative_bytes(path, max_bytes=1 << 30)
    _require(
        digest == expected_sha256,
        f"{path}: post-Pass-1 result SHA-256 {digest} is not the supplied {expected_sha256}",
    )
    obj = strict_json_object(payload, where=str(path))
    _require(
        canonical_json_bytes(obj) == payload,
        f"{path}: post-Pass-1 result bytes are not canonical",
    )
    return trusted_expected_result(obj, expected_sha256=expected_sha256)


def _closed(obj: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    _require(type(obj) is dict, f"{where} must be a JSON object")
    present = frozenset(obj)
    unknown = sorted(present - fields)
    missing = sorted(fields - present)
    _require(not unknown, f"{where} carries unknown field(s): {unknown}")
    _require(not missing, f"{where} is missing field(s): {missing}")
    return obj


def require_manifest_shape(manifest: Any) -> dict[str, Any]:
    """Close the manifest's top-level shape before anything indexes into it.

    R4-F: reconciliation reached ``manifest["stage_i_run"]`` directly, so a manifest with the
    whole block absent raised a bare ``KeyError`` instead of a controlled fail-closed refusal.
    Every entry point that touches manifest fields goes through here first.

    Container types are pinned here as well, so every later index or iteration is safe no matter
    which check happens to run first -- a wrong-typed block fails on being wrong-typed rather than
    on whatever downstream expression happened to trip over it.
    """
    obj = _closed(manifest, _MANIFEST_FIELDS, "manifest")
    for field_name, kind in (
        ("shards", list),
        ("nodes", list),
        ("totals", dict),
        ("ownership_matrix", dict),
        ("bindings", dict),
        ("environment", dict),
        ("h_binding", dict),
        ("stage_i_run", dict),
        ("node_binding_projection", dict),
    ):
        _require(
            type(obj[field_name]) is kind,
            f"manifest.{field_name} must be a JSON {'array' if kind is list else 'object'}",
        )
    return obj


# R6-C: exactly the nested manifest fields that are indexed before the full semantic validator
# runs, with the type each of those reads needs in order to execute safely. Presence and scalar
# type only -- nothing here compares a value with anything, recomputes an identity or asks a
# Layer-2 question. Those all stay after the physical audit, which is the whole point of the
# physical-first ordering.
_EARLY_TOTALS_FIELDS = (
    "records",
    "content_tokens",
    "serialized_tokens",
    "shards",
    "unique_cleaned_identities",
)
_EARLY_NODE_INT_FIELDS = (
    "target_serialized_tokens",
    "selected_identities",
    "selected_serialized_tokens",
    "actual_overshoot_tokens",
)
_EARLY_NODE_STR_FIELDS = (
    "source_id",
    "stage",
    "branch",
    "selection_mode",
    "selection_fingerprint",
    "selection_sequence_commitment",
)
_EARLY_STAGE_I_RUN_STR_FIELDS = (
    "run_identity",
    "environment_sha256",
    "binding_document_digests_sha256",
    "node_binding_projection_sha256",
)
_EARLY_H_BINDING_STR_FIELDS = ("h_run_identity",)


def _early_present(obj: Mapping[str, Any], key: str, where: str) -> Any:
    _require(key in obj, f"{where} is missing field(s): ['{key}']")
    return obj[key]


def require_physical_projection_shape(manifest: Mapping[str, Any]) -> None:
    """Presence and scalar type of every nested field read before full manifest validation.

    ``require_manifest_shape`` closes the manifest's top level and pins the nine container types,
    so ``manifest[X]`` is total. It says nothing about ``manifest[X][Y]``, and R5-C moved the full
    semantic validator to AFTER the physical audit -- so the marker check, the physical
    reconciliation and the Layer-2 comparisons became the first code to index those nested fields.
    A manifest missing ``stage_i_run.run_identity`` or ``totals.serialized_tokens`` therefore
    escaped as a raw ``KeyError`` instead of a controlled Stage-I refusal: still fail-closed, but
    outside the validation contract.

    This closes that gap without weakening the ordering. It establishes only that the fields those
    reads consume exist and are the right kind of scalar; every question about what they MEAN --
    the run identity's own derivation, the Layer-2 equalities, the projection equality -- still
    runs after the physical facts have been reconciled, so a realization whose totals do not
    describe its own records still fails on that.
    """
    totals = manifest["totals"]
    for field_name in _EARLY_TOTALS_FIELDS:
        _exact_int(
            _early_present(totals, field_name, "manifest.totals"),
            f"manifest.totals.{field_name}",
        )

    for index, entry in enumerate(manifest["nodes"]):
        where = f"manifest.nodes[{index}]"
        _require(type(entry) is dict, f"{where} must be a JSON object")
        for field_name in _EARLY_NODE_STR_FIELDS:
            _exact_str(_early_present(entry, field_name, where), f"{where}.{field_name}")
        for field_name in _EARLY_NODE_INT_FIELDS:
            _exact_int(_early_present(entry, field_name, where), f"{where}.{field_name}")
        crossing = _early_present(entry, "crossing_identity", where)
        _require(
            crossing is None or type(crossing) is str,
            f"{where}.crossing_identity must be a string or null",
        )
        bindings = _early_present(entry, "input_binding_ids", where)
        _require(type(bindings) is list, f"{where}.input_binding_ids must be a JSON array")
        for position, binding_id in enumerate(bindings):
            _exact_str(binding_id, f"{where}.input_binding_ids[{position}]")

    for field_name in _EARLY_STAGE_I_RUN_STR_FIELDS:
        _exact_str(
            _early_present(manifest["stage_i_run"], field_name, "manifest.stage_i_run"),
            f"manifest.stage_i_run.{field_name}",
        )
    for field_name in _EARLY_H_BINDING_STR_FIELDS:
        _exact_str(
            _early_present(manifest["h_binding"], field_name, "manifest.h_binding"),
            f"manifest.h_binding.{field_name}",
        )

    # `tuple(value)` over a scalar raises TypeError, and a str would splat into characters.
    for source_id, allowed in manifest["node_binding_projection"].items():
        where = f"manifest.node_binding_projection[{source_id!r}]"
        _require(type(allowed) is list, f"{where} must be a JSON array")
        for position, binding_id in enumerate(allowed):
            _exact_str(binding_id, f"{where}[{position}]")

    for binding_id, digest in manifest["bindings"].items():
        _exact_str(digest, f"manifest.bindings[{binding_id!r}]")


def validate_shard_list(manifest: Mapping[str, Any]) -> int:
    """The shard list's structure, in isolation. Returns the declared record total.

    R5-C: the strict consumer needs exactly this much of the manifest -- and no more -- to locate
    and read the physical shards. Splitting it out lets the consumer establish where the data is,
    audit it, and reconcile the physical facts BEFORE any high-level identity question is asked,
    so a realization whose totals do not describe its own records fails on that rather than on
    whichever downstream check happened to run first.
    """
    shards = manifest["shards"]
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
    return total_records


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Everything about the manifest that must hold before COMPLETE may be written."""
    obj = require_manifest_shape(manifest)
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

    total_records = validate_shard_list(obj)
    shards = obj["shards"]

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
        _hex64(
            entry["selection_sequence_commitment"],
            f"manifest.nodes[{index}].selection_sequence_commitment",
        )
        declared = entry["input_binding_ids"]
        _require(
            type(declared) is list and declared,
            f"manifest.nodes[{index}].input_binding_ids must be a non-empty list",
        )
        for binding_id in declared:
            _exact_str(binding_id, f"manifest.nodes[{index}].input_binding_ids[]")
        _require(
            declared == sorted(set(declared)),
            f"manifest.nodes[{index}].input_binding_ids must be sorted and unique",
        )
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

    # R4-D: closed on all three axes. R3 required the three keys to be present and stringly
    # typed and checked nothing else, so an unknown key rode along and a false interpreter,
    # Python version or tokenizers build published unchallenged.
    environment = _closed(obj["environment"], _ENVIRONMENT_FIELD_SET, "manifest.environment")
    for key in ENVIRONMENT_FIELDS:
        _require(
            _exact_str(environment[key], f"manifest.environment.{key}") == FROZEN_ENVIRONMENT[key],
            f"manifest.environment.{key} is {environment[key]!r} but the frozen Stage-I "
            f"environment requires {FROZEN_ENVIRONMENT[key]!r}",
        )

    h_binding = _closed(obj["h_binding"], _H_BINDING_FIELDS, "manifest.h_binding")
    for key in sorted(_H_BINDING_FIELDS):
        _hex64(h_binding[key], f"manifest.h_binding.{key}")

    projection = obj["node_binding_projection"]
    _require(
        type(projection) is dict and projection,
        "manifest.node_binding_projection must be a non-empty object",
    )
    declared_nodes = {entry["source_id"] for entry in nodes}
    for source_id, bindings in projection.items():
        _exact_str(source_id, "manifest.node_binding_projection key")
        _require(
            type(bindings) is list and bindings,
            f"manifest.node_binding_projection[{source_id!r}] must be a non-empty list",
        )
        for binding_id in bindings:
            _exact_str(binding_id, f"manifest.node_binding_projection[{source_id!r}][]")
        _require(
            bindings == sorted(set(bindings)),
            f"manifest.node_binding_projection[{source_id!r}] must be sorted and unique",
        )
    _require(
        set(projection) == declared_nodes,
        "manifest.node_binding_projection must cover exactly the declared nodes",
    )
    for entry in nodes:
        allowed = projection[entry["source_id"]]
        _require(
            set(entry["input_binding_ids"]) <= set(allowed),
            f"manifest node {entry['source_id']} draws from a binding outside its authorized "
            "projection",
        )

    stage_i_run = _closed(obj["stage_i_run"], _STAGE_I_RUN_FIELDS, "manifest.stage_i_run")
    for key in ("run_identity", "post_pass1_result_identity_sha256", *_AUTHORIZATION_HEX_FIELDS):
        _hex64(stage_i_run[key], f"manifest.stage_i_run.{key}")
    _require(
        _exact_str(
            stage_i_run["post_pass1_result_identity_schema"],
            "manifest.stage_i_run.post_pass1_result_identity_schema",
        )
        == PASS1_RESULT_SCHEMA,
        "manifest.stage_i_run names a post-Pass-1 result schema this implementation cannot read",
    )
    _require(
        _exact_str(
            stage_i_run["selection_sequence_commitment_version"],
            "manifest.stage_i_run.selection_sequence_commitment_version",
        )
        == SELECTION_SEQUENCE_SCHEMA,
        "manifest.stage_i_run names a different selection-sequence commitment version",
    )
    _hex64(
        stage_i_run["selection_sequence_commitment_map_sha256"],
        "manifest.stage_i_run.selection_sequence_commitment_map_sha256",
    )
    _require(
        selection_sequence_commitment_map_sha256({
            entry["source_id"]: entry["selection_sequence_commitment"] for entry in nodes
        })
        == stage_i_run["selection_sequence_commitment_map_sha256"],
        "manifest.stage_i_run.selection_sequence_commitment_map_sha256 does not describe the "
        "per-node commitments the manifest itself declares",
    )
    _require(
        node_binding_projection_sha256(projection) == stage_i_run["node_binding_projection_sha256"],
        "manifest.stage_i_run.node_binding_projection_sha256 does not describe "
        "manifest.node_binding_projection",
    )
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


def reconcile_manifest_with_audit(
    manifest: Mapping[str, Any],
    audit: RealizationAudit,
    expected: TrustedExpectedResult,
) -> None:
    """Three-way equality: the bytes on disk == the manifest == the trusted expectation.

    R2 compared the manifest against the audit only. Both are Layer 3, so a fully resealed
    publication -- records rewritten, the manifest's own expected sequence and projection restated
    to match, every digest and total recomputed -- was internally consistent and passed. The
    missing party is ``expected``: the post-Pass-1 result, built from the selection ledger and
    frozen to disk before a single output byte existed.

    R4-E fixes the ORDER. Physical facts are reconciled first, so a result whose totals do not
    describe its own records fails on that -- the concrete, diagnosable defect -- rather than on
    whichever higher-level identity check happened to run earlier. Only once the realization has
    been proved to describe itself are the Layer-2 questions asked: is this the authorized run,
    under the authorized environment, over the authorized inputs, with the committed selection?
    """
    obj = require_manifest_shape(manifest)
    require_physical_projection_shape(obj)

    # --- Layer 3 against the bytes it claims to describe --------------------------------
    _reconcile_physical(obj, audit)

    # --- Layer 3 against the expectation that predates it -------------------------------
    _reconcile_run_identity(obj, expected)
    _reconcile_projection(obj, expected)
    _reconcile_environment(obj, expected)
    _reconcile_binding_digests(obj, expected)
    _reconcile_expected_result(obj, audit, expected)


def _reconcile_physical(manifest: Mapping[str, Any], audit: RealizationAudit) -> None:
    """Everything the manifest claims about itself, against the records actually staged."""
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
            "from its physical records",
        )
        _require(
            list(node.input_binding_ids) == sorted(set(entry["input_binding_ids"])),
            f"manifest node {source_id} declares input bindings "
            f"{sorted(set(entry['input_binding_ids']))} but its physical records use "
            f"{list(node.input_binding_ids)}",
        )
    unknown = sorted(set(audited) - {e["source_id"] for e in manifest["nodes"]})
    _require(not unknown, f"physical records exist for undeclared node(s): {unknown}")


def _reconcile_expected_result(
    manifest: Mapping[str, Any], audit: RealizationAudit, expected: TrustedExpectedResult
) -> None:
    """The realization must be the one Pass 1 committed to, node for node and token for token."""
    totals = manifest["totals"]
    for field_name in (
        "records",
        "content_tokens",
        "serialized_tokens",
        "unique_cleaned_identities",
    ):
        _require(
            totals[field_name] == expected.totals[field_name],
            f"manifest.totals.{field_name} is {totals[field_name]} but the trusted post-Pass-1 "
            f"result committed to {expected.totals[field_name]}",
        )

    _require(
        dict(manifest["ownership_matrix"])
        == {consumer: dict(owners) for consumer, owners in expected.ownership_matrix.items()},
        "manifest.ownership_matrix disagrees with the trusted post-Pass-1 result",
    )

    audited = {node.source_id: node for node in audit.nodes}
    _require(
        {entry["source_id"] for entry in manifest["nodes"]}
        == {node.source_id for node in expected.nodes},
        "manifest declares a different node set than the trusted post-Pass-1 result",
    )
    for entry in manifest["nodes"]:
        source_id = entry["source_id"]
        commitment = expected.node(source_id)
        for field_name, trusted in (
            ("stage", commitment.stage),
            ("target_serialized_tokens", commitment.target_serialized_tokens),
            ("branch", commitment.branch),
            ("selection_mode", commitment.selection_mode),
            ("selected_identities", commitment.selected_identities),
            ("selected_serialized_tokens", commitment.selected_serialized_tokens),
            ("selection_fingerprint", commitment.selection_fingerprint),
            ("selection_sequence_commitment", commitment.selection_sequence_commitment),
            ("crossing_identity", commitment.crossing_identity),
            ("actual_overshoot_tokens", commitment.actual_overshoot_tokens),
        ):
            _require(
                entry[field_name] == trusted,
                f"manifest node {source_id}.{field_name} is {entry[field_name]!r} but the trusted "
                f"post-Pass-1 result committed to {trusted!r}",
            )
        _require(
            tuple(entry["input_binding_ids"]) == commitment.input_binding_ids,
            f"manifest node {source_id} declares input bindings {entry['input_binding_ids']} but "
            f"the trusted post-Pass-1 result committed to {list(commitment.input_binding_ids)}",
        )
        if source_id not in audited:
            continue
        # R3-A: the order-sensitive commitment. The expected value is the one frozen before
        # materialization; the actual comes from an external sort over the published ordinals. A
        # reseal can restate the manifest, but it cannot restate an artifact that already exists
        # outside the realization and whose digest names the run.
        _require(
            commitment.selection_sequence_commitment
            == audited[source_id].selection_sequence_commitment,
            f"manifest node {source_id} selection-sequence commitment disagrees with the sequence "
            "reconstructed from its physical records; the ordinal-to-identity mapping differs "
            "from the trusted post-Pass-1 expectation",
        )


def _reconcile_environment(manifest: Mapping[str, Any], expected: TrustedExpectedResult) -> None:
    """R4-D: the environment a run reports must be the one it was authorized under.

    The manifest's own environment block is an observation. The expectation comes from canonical
    Layer-1 state via the frozen post-Pass-1 result, so a resealed publication cannot claim a
    different interpreter, Python build or tokenizers version -- the two things it would have to
    edit are on opposite sides of the freeze.
    """
    declared = _closed(manifest["environment"], _ENVIRONMENT_FIELD_SET, "manifest.environment")
    for key in ENVIRONMENT_FIELDS:
        _require(
            declared[key] == expected.environment[key],
            f"manifest.environment.{key} is {declared[key]!r} but the trusted post-Pass-1 result "
            f"binds {expected.environment[key]!r}",
        )
    _require(
        environment_sha256(declared) == expected.environment_sha256,
        "manifest.environment does not generate the trusted environment digest",
    )
    _require(
        manifest["stage_i_run"]["environment_sha256"] == expected.environment_sha256,
        "manifest.stage_i_run.environment_sha256 disagrees with the trusted post-Pass-1 result",
    )


def _reconcile_binding_digests(
    manifest: Mapping[str, Any], expected: TrustedExpectedResult
) -> None:
    """R4-D: the input-binding release digests a run reports must be the authorized ones."""
    declared = manifest["bindings"]
    trusted = dict(expected.binding_document_digests)
    _require(
        dict(declared) == trusted,
        "manifest.bindings disagrees with the trusted input-binding document digests; "
        f"manifest={dict(sorted(declared.items()))} trusted={dict(sorted(trusted.items()))}",
    )
    _require(
        binding_document_digests_sha256(declared) == expected.binding_document_digests_sha256,
        "manifest.bindings does not generate the trusted binding-document-digest projection",
    )
    _require(
        manifest["stage_i_run"]["binding_document_digests_sha256"]
        == expected.binding_document_digests_sha256,
        "manifest.stage_i_run.binding_document_digests_sha256 disagrees with the trusted "
        "post-Pass-1 result",
    )


def _reconcile_run_identity(manifest: Mapping[str, Any], expected: TrustedExpectedResult) -> None:
    """The published run must be the authorized one, named by the frozen Layer-2 expectation."""
    # The manifest's top-level shape was closed by `require_manifest_shape`; this closes the run
    # block itself, so a missing or mistyped field here is a controlled refusal, never a KeyError.
    stage_i_run = _closed(manifest["stage_i_run"], _STAGE_I_RUN_FIELDS, "manifest.stage_i_run")
    _require(
        stage_i_run["post_pass1_result_identity_schema"] == expected.result_identity_schema,
        "manifest names a different post-Pass-1 result schema than the trusted expectation",
    )
    _require(
        stage_i_run["post_pass1_result_identity_sha256"] == expected.result_identity_sha256,
        "manifest binds post-Pass-1 result "
        f"{stage_i_run['post_pass1_result_identity_sha256']} but the trusted expectation supplied "
        f"out of band is {expected.result_identity_sha256}",
    )
    _require(
        stage_i_run["selection_sequence_commitment_version"]
        == expected.selection_sequence_commitment_version,
        "manifest names a different selection-sequence commitment version than the trusted "
        "expectation",
    )
    _require(
        stage_i_run["selection_sequence_commitment_map_sha256"]
        == expected.selection_sequence_commitment_map_sha256,
        "manifest's selection-sequence commitment map digest disagrees with the trusted "
        "post-Pass-1 expectation",
    )
    for key in sorted(_AUTHORIZATION_FIELDS):
        _require(
            stage_i_run[key] == expected.authorization[key],
            f"manifest.stage_i_run.{key} is {stage_i_run[key]!r} but the trusted post-Pass-1 "
            f"result binds {expected.authorization[key]!r}",
        )
    _require(
        stage_i_run["run_identity"] == expected.stage_i_run_identity,
        f"manifest claims run identity {stage_i_run['run_identity']} but the trusted expectation "
        f"generates {expected.stage_i_run_identity}",
    )
    _require(
        dict(manifest["h_binding"]) == dict(expected.h_binding),
        "manifest.h_binding disagrees with the trusted post-Pass-1 result",
    )


def _reconcile_projection(manifest: Mapping[str, Any], expected: TrustedExpectedResult) -> None:
    """R3-B: node -> allowed-binding authority comes from Layer 1/2, never from the manifest.

    Four things must agree, and the manifest is only one of them: the trusted projection carried
    by the expectation, the projection the manifest records as an observation, the plan-authorized
    global binding set, and (through the audit, which is handed the trusted projection) every
    physical record's membership.
    """
    declared = {
        source_id: tuple(bindings)
        for source_id, bindings in manifest["node_binding_projection"].items()
    }
    trusted = dict(expected.node_binding_projection)
    _require(
        declared == trusted,
        "manifest.node_binding_projection disagrees with the trusted node/binding authority; "
        f"manifest={ {k: list(v) for k, v in sorted(declared.items())} } "
        f"trusted={ {k: list(v) for k, v in sorted(trusted.items())} }",
    )
    _require(
        manifest["stage_i_run"]["node_binding_projection_sha256"]
        == expected.node_binding_projection_sha256,
        "manifest's node/binding projection digest disagrees with the trusted authority",
    )
    authorized = set(expected.authorized_input_binding_ids)
    projected = {binding for allowed in trusted.values() for binding in allowed}
    outside = sorted(projected - authorized)
    _require(
        not outside,
        f"trusted projection names binding(s) outside the plan-authorized global set: {outside}",
    )
    undeclared = sorted(set(manifest["bindings"]) - authorized)
    _require(
        not undeclared,
        f"manifest declares input binding(s) the plan never authorized: {undeclared}",
    )
    missing = sorted(projected - set(manifest["bindings"]))
    _require(
        not missing,
        f"manifest omits the release digest for authorized binding(s) in use: {missing}",
    )


def audit_staged_realization(
    root: Path,
    shard_names: Sequence[str],
    work_dir: Path,
    node_binding_projection: Mapping[str, Sequence[str]],
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
        node_binding_projection=node_binding_projection,
        read_window_bytes=read_window_bytes,
        sort_chunk_lines=sort_chunk_lines,
    )


def publish_atomic(
    out_dir: Path,
    run_name: str,
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    expected: TrustedExpectedResult,
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

    ``expected`` is the trusted post-Pass-1 result, passed in separately from the manifest and
    never derived from it. It supplies the node/binding authority the audit checks records
    against, and the per-node sequence commitments the reconciliation requires -- so neither the
    manifest nor the records can supply their own expected values.
    """
    _require(
        isinstance(expected, TrustedExpectedResult),
        "publication requires a trusted post-Pass-1 expected result, not a manifest-derived value",
    )
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

        # R3-B: the audit's expectation is the trusted projection, not the one the manifest
        # carries. Handing a document its own expected value back is exactly the defect this
        # repair exists to close.
        audit = audit_staged_realization(
            staging,
            [entry["name"] for entry in shards],
            audit_dir,
            expected.node_binding_projection,
            read_window_bytes=read_window_bytes,
            sort_chunk_lines=sort_chunk_lines,
        )
        reconcile_manifest_with_audit(complete, audit, expected)
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
    expected: TrustedExpectedResult,
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

    ``expected`` is mandatory and must come from outside the realization -- the frozen post-Pass-1
    artifact plus the digest its owner holds, or registry-owned authorized execution state. The
    reviewed defect was that the consumer discovered its own expectation by reading the manifest,
    the COMPLETE marker and the published run identity, all three of which belong to the thing
    being checked.
    """
    _require(
        isinstance(expected, TrustedExpectedResult),
        "the strict consumer requires a trusted post-Pass-1 expected result supplied from outside "
        "the published realization; a manifest-derived expectation is not an expectation",
    )
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
    # R5-C: only enough structure to locate and read the physical data. Full manifest validation
    # -- which recomputes the published run identity -- runs after the physical facts have been
    # reconciled, so a false total is reported as a false total.
    # R6-C: and enough nested structure that every read between here and there is safe. Presence
    # and scalar type only: no value is compared, no identity recomputed, no Layer-2 question
    # asked before the physical audit. It sits after the shard list so that shard-location
    # problems keep surfacing first, exactly as they did before, and before the COMPLETE marker
    # check, which is the first code to index a nested manifest field.
    require_manifest_shape(manifest)
    validate_shard_list(manifest)
    require_physical_projection_shape(manifest)
    _validate_complete_marker(marker, manifest, manifest_digest)

    documents_dir = final / DOCUMENTS_DIRNAME
    _require(documents_dir.is_dir(), f"{final}: no documents directory")
    present = sorted(p.name for p in documents_dir.iterdir())
    shard_names = [entry["name"] for entry in manifest["shards"]]
    _require(
        present == sorted(shard_names),
        f"{final}: documents directory contents disagree with the manifest shard list",
    )

    owned_scratch = work_dir is None
    scratch = (
        Path(tempfile.mkdtemp(prefix="stage-i-consumer-audit-")) if owned_scratch else work_dir
    )
    try:
        audit = audit_staged_realization(
            final,
            shard_names,
            scratch,
            expected.node_binding_projection,
            read_window_bytes=read_window_bytes,
            sort_chunk_lines=sort_chunk_lines,
        )
        # Physical reconciliation first (it is physical-first internally too), then the Layer-2
        # comparisons, and only then the manifest's own full validation and run identity.
        reconcile_manifest_with_audit(manifest, audit, expected)
    finally:
        if owned_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    validate_manifest(manifest)
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
    "PASS1_RESULT_SCHEMA",
    "RECORDS_PER_SHARD",
    "RECORD_FIELDS",
    "RECORD_SCHEMA",
    "SELECTION_SEQUENCE_MAP_SCHEMA",
    "SHARD_POLICY_VERSION",
    "STAGE_I_RUN_IDENTITY_SCHEMA",
    "OutputError",
    "TrustedExpectedNode",
    "TrustedExpectedResult",
    "build_record",
    "iter_records",
    "audit_staged_realization",
    "load_published_realization",
    "load_trusted_expected_result",
    "NODE_BINDING_PROJECTION_SCHEMA",
    "node_binding_projection_sha256",
    "recompute_stage_i_run_identity",
    "reconcile_manifest_with_audit",
    "require_manifest_shape",
    "require_physical_projection_shape",
    "environment_sha256",
    "binding_document_digests_sha256",
    "ENVIRONMENT_FIELDS",
    "ENVIRONMENT_SCHEMA",
    "BINDING_DOCUMENT_DIGESTS_SCHEMA",
    "FROZEN_ENVIRONMENT",
    "FROZEN_PYTHON_EXECUTABLE",
    "FROZEN_PYTHON_VERSION",
    "FROZEN_TOKENIZERS_VERSION",
    "plan_shards",
    "publish_atomic",
    "record_sort_key",
    "selection_sequence_commitment_map_sha256",
    "shard_name",
    "stage_i_published_run_identity",
    "trusted_expected_result",
    "validate_manifest",
    "validate_pass1_result",
    "validate_shard_list",
    "validate_record",
    "write_shards",
]
