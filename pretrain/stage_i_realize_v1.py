#!/usr/bin/env python3
"""Stage-I authoritative realization driver v1.

Stage H predicted; Stage I decides and materializes. This module orchestrates that in two
authoritative passes over the frozen releases and refuses to write a single document byte until
its independently derived selection has been proved equal to the accepted Stage-H prediction on
every dimension the frozen contract names.

Pass 1 reads every bound release once, derives canonical candidates, collapses duplicates, applies
ownership, chooses branches, ranks and cuts, and produces a complete selection ledger. Pass 2
re-reads the same bound inputs, locates exactly the selected physical representatives, recomputes
their hashes and token counts from the bytes actually being written, and publishes atomically. No
third full-corpus pass is required, and there is no resume: a run either publishes a complete
result or publishes nothing.

Dependency boundary, which is load-bearing (see DECISIONS D-142 on the shared-core common-mode
risk):

* the selection itself comes from ``pretrain.stage_i_select_v1``, which was written independently
  and has no import path to Stage H at all;
* from ``pretrain.h_census_v2`` this module imports exactly one name, ``load_published_run``. That
  is the reviewed strict *consumer* used to load and validate the accepted H result, not a
  selection function. Importing it is required by the segment contract; importing anything that
  decides a selection would defeat the comparison;
* everything else reused is byte-level infrastructure -- canonical JSON, SHA-256, file identity,
  text cleaning, tokenizer accounting, frozen graph and authority loading. Forking those would
  fork the frozen corpus contract rather than demonstrate independence.

``tests/test_stage_i_realize_v1.py`` asserts this boundary structurally over the parsed AST.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import weakref

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import (  # noqa: E402
    clean_text,
    cleaned_text_sha256,
    encode_with_accounting,
    load_tokenizer,
)

# The reviewed strict consumer for the accepted Stage-H result. This is the ONLY name taken from
# the Stage-H implementation, and it validates rather than selects.
from pretrain.h_census_v2 import load_published_run  # noqa: E402
from pretrain.select_pretrain_documents import (  # noqa: E402
    SELECTION_METADATA_FIELD,
    _canonical_json_bytes,
)
from pretrain.stage_i_audit_v1 import (  # noqa: E402
    DEFAULT_READ_WINDOW_BYTES,
    DEFAULT_SORT_CHUNK_LINES,
    SELECTION_SEQUENCE_SCHEMA,
    AuditError,
    selection_sequence_commitment,
    stream_lines,
)
from pretrain.stage_i_graph_v2 import (  # noqa: E402
    STAGE_PRIORITY,
    GraphError,
    InputBinding,
    SourceGraph,
    StrictJSONError,
    canonical_json_bytes,
    load_source_graph,
    open_authoritative,
    read_authoritative_bytes,
    strict_json_object,
    verify_binding_inputs,
)
from pretrain.stage_i_output_v1 import (  # noqa: E402
    MANIFEST_SCHEMA,
    RECORD_SCHEMA,
    RECORDS_PER_SHARD,
    SHARD_POLICY_VERSION,
    OutputError,
    build_record,
    node_binding_projection_sha256,
    publish_atomic,
)
from pretrain.stage_i_select_v1 import (  # noqa: E402
    PHYSICAL_LOCATOR_RULE,
    REPRESENTATIVE_RULE,
    Candidate,
    NodeSelection,
    SelectionError,
    canonical_document_fingerprint_v1,
    ownership_matrix_v1,
    read_score_v1,
    realize_selection,
    score_to_bits_v1,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS  # noqa: E402

PLAN_SCHEMA = "petitgpt-i-candidate-plan-v2"

# Closed plan schema. R1 validated only the fields it happened to read, so a rehashed plan could
# drop `hq_policy`, drop `selection_rules`, or carry unknown keys and still authorize. Every level
# below is exact: unknown rejected, missing rejected, types checked.
_PLAN_TOP_FIELDS = frozenset({
    "schema_version",
    "authorization_status",
    "authorization_note",
    "realization_label",
    "resume_supported",
    "implementation_commit",
    "implementation_files",
    "implementation_bundle_sha256",
    "output_schema_version",
    "manifest_schema_version",
    "shard_policy",
    "selection_rules",
    "accepted_h",
    "graph_path",
    "graph_sha256",
    "seed",
    "authorities",
    "bound_authorities",
    "input_bindings",
    "node_order",
    "nodes",
    "node_binding_projection",
    "environment_contract",
})
_PLAN_SHARD_POLICY_FIELDS = frozenset({"version", "records_per_shard", "rule"})
_PLAN_SELECTION_RULES_FIELDS = frozenset({"representative_rule", "physical_locator_rule"})
_PLAN_ACCEPTED_H_FIELDS = frozenset({
    "run_dir",
    "run_identity",
    "census_sha256",
    "predictions_sha256",
    "complete_sha256",
    "candidate_plan_sha256",
    "implementation_bundle_sha256",
})
_PLAN_AUTHORITY_FIELDS = frozenset({"path", "sha256"})
_PLAN_BINDING_FIELDS = frozenset({
    "documents_sha256",
    "documents_size_bytes",
    "eligibility_index_sha256",
    "release_manifest_sha256",
    "total_physical_rows",
    "expected_eligible_rows",
})
_PLAN_NODE_FIELDS = frozenset({
    "source_id",
    "stage",
    "stage_priority",
    "target_serialized_tokens",
    "selection_mode",
    "input_binding_ids",
})
_PLAN_ENVIRONMENT_FIELDS = frozenset({
    "python_executable",
    "python_version",
    "tokenizers_version",
    "observed_at_generation",
})

# The exact authority set this Stage-I schema requires. Closed on purpose: R1 verified whichever
# entries were present, so removing one was invisible.
REQUIRED_AUTHORITIES = frozenset({
    "d2_d3_eligibility_manifest",
    "g2_release_manifest",
    "g_release_manifest",
    "hq_policy",
    "reference_exclusion",
    "selector_v1",
    "stage_e_allocation",
    "tokenizer",
})
BUNDLE_SCHEMA = "petitgpt-i-implementation-bundle-v1"
REALIZATION_LABEL = "AUTHORITATIVE_STAGE_I_REALIZATION"
RESUME_SUPPORTED = False

# Every module whose bytes can change a Stage-I result. The audit module is here because it
# decides whether a realization may be published at all and what the consumer will accept; leaving
# it out would mean a change to that logic did not invalidate an authorized plan.
IMPLEMENTATION_BUNDLE_FILES = (
    "pretrain/stage_i_audit_v1.py",
    "pretrain/stage_i_output_v1.py",
    "pretrain/stage_i_realize_v1.py",
    "pretrain/stage_i_select_v1.py",
)

REQUIRED_PYTHON_EXECUTABLE = "/workspace/petitgpt/.venv/bin/python"
REQUIRED_PYTHON_VERSION = "3.10.12"
REQUIRED_TOKENIZERS_VERSION = "0.22.2"

ACCEPTED_H_RUN_DIR = "runs/h_production_v2_2026-08-23"
ACCEPTED_H_RUN_IDENTITY = "63f5ef84ab56c6da7f76ecfb9e9196a3e98a791d607224c83a9c183c32be111a"
ACCEPTED_H_PREDICTIONS_SHA256 = "fff205494b1379eaf0e77a5d58591c085af2764712a024ed25c3767c10382f87"
# The exact accepted Stage-H COMPLETE bytes. Binding the digest rather than the marker's name
# is what makes "this Stage-I run descends from that Stage-H result" a checkable claim; the
# reviewed plan recorded only the literal string "COMPLETE", which any directory could satisfy.
ACCEPTED_H_COMPLETE_SHA256 = "b4d340afde8db55830115d4e7ba21757215122518567be4dad352c5adb28881f"

STAGE_I_RUN_IDENTITY_SCHEMA = "petitgpt-stage-i-run-identity-v2"
CANONICAL_STATE_SCHEMA = "petitgpt-stage-i-canonical-state-v1"

# The six dimensions the frozen H/I contract requires, plus every other field Stage H publishes.
# Comparing the full projection rather than only the six is free and strictly stronger.
H_I_REQUIRED_DIMENSIONS = (
    "branch",
    "selected_serialized_tokens",
    "selection_fingerprint",
    "crossing_identity",
    "actual_overshoot_tokens",
)


class RealizationError(RuntimeError):
    """Fail-closed Stage-I realization condition."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RealizationError(message)


def file_sha256(path: Path) -> str:
    _, digest = read_authoritative_bytes(path, max_bytes=1 << 31)
    return digest


def implementation_files(repo_root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for relative in IMPLEMENTATION_BUNDLE_FILES:
        path = repo_root / relative
        _require(path.is_file(), f"implementation bundle member missing: {relative}")
        digests[relative] = file_sha256(path)
    return digests


def implementation_bundle_sha256(files: Mapping[str, str]) -> str:
    """One digest over the exact member list and their digests."""
    payload = {"schema_version": BUNDLE_SCHEMA, "files": dict(sorted(files.items()))}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


# --------------------------------------------------------------------- environment contract


@dataclass(frozen=True)
class Environment:
    python_executable: str
    python_version: str
    tokenizers_version: str

    def as_canonical(self) -> dict[str, str]:
        return {
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "tokenizers_version": self.tokenizers_version,
        }


def current_environment() -> Environment:
    import tokenizers

    return Environment(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        tokenizers_version=tokenizers.__version__,
    )


def verify_environment(environment: Environment, *, require_executable: bool = True) -> None:
    """Stop before any corpus processing if the interpreter is not the frozen one.

    The tokenizers version is the load-bearing check: serialized token counts are a function of
    the tokenizer library, so running Stage I under a different build would silently produce token
    totals that cannot equal Stage H's and would only surface as a comparison failure after a
    multi-hour scan. Failing here costs seconds instead.
    """
    _require(
        environment.tokenizers_version == REQUIRED_TOKENIZERS_VERSION,
        f"tokenizers {environment.tokenizers_version} is not the frozen "
        f"{REQUIRED_TOKENIZERS_VERSION}; serialized token counts would not be comparable with "
        "the accepted Stage-H result",
    )
    _require(
        environment.python_version == REQUIRED_PYTHON_VERSION,
        f"python {environment.python_version} is not the frozen {REQUIRED_PYTHON_VERSION}",
    )
    if require_executable:
        _require(
            environment.python_executable == REQUIRED_PYTHON_EXECUTABLE,
            f"python executable {environment.python_executable} is not the frozen "
            f"{REQUIRED_PYTHON_EXECUTABLE}",
        )


# --------------------------------------------------------------------- accepted H binding


_PREDICTION_TO_CENSUS = {
    "predicted_branch": "branch",
    "predicted_selection_mode": "selection_mode",
    "predicted_selected_identity_fingerprint": "selection_fingerprint",
    "predicted_selected_serialized_tokens": "selected_serialized_tokens",
    "predicted_selected_identities": "selected_identities",
    "predicted_crossing_identity": "crossing_identity",
    "predicted_crossing_document_serialized_tokens": "crossing_document_serialized_tokens",
    "predicted_overshoot_tokens": "actual_overshoot_tokens",
    "predicted_residual_identities": "residual_identities",
    "predicted_residual_serialized_tokens": "residual_serialized_tokens",
    "predicted_post_exclusion_candidate_identities": "post_exclusion_candidate_identities",
    "predicted_post_exclusion_candidate_serialized_tokens": (
        "post_exclusion_candidate_serialized_tokens"
    ),
    "predicted_pre_exclusion_unique_identities": "pre_exclusion_unique_identities",
    "predicted_g2_excluded_identities": "g2_excluded_identities",
    "predicted_prior_commit_excluded_identities": "prior_commit_excluded_identities",
    "predicted_exclusions_by_owner": "exclusions_by_owner",
    "predicted_boundary_evidence": "boundary_evidence",
    "predicted_feasible": "feasible",
    "stage": "stage",
    "target_serialized_tokens": "target_serialized_tokens",
}


@dataclass(frozen=True)
class AcceptedH:
    """The accepted Stage-H result, loaded through the reviewed consumer and cross-checked.

    Both artefacts are bound, not just the convenient one: ``census`` is the canonical authority
    and ``predictions`` is only a projection of it. The projection is verified field-by-field
    against the census before it is allowed to drive any comparison, so a corrupted or
    hand-edited predictions file cannot move a Stage-I decision.
    """

    run_dir: Path
    census: Mapping[str, Any]
    predictions: Mapping[str, Any]
    census_sha256: str
    predictions_sha256: str
    complete_sha256: str
    run_identity: str

    def node(self, source_id: str) -> Mapping[str, Any]:
        for node in self.census["nodes"]:
            if node["source_id"] == source_id:
                return node
        raise RealizationError(f"accepted H census has no node {source_id!r}")

    @property
    def h_ownership_matrix(self) -> Mapping[str, Any]:
        return self.census["ownership_matrix"]


def verify_h_evidence_manifest(run_dir: Path) -> dict[str, str]:
    """Re-verify every file listed in the accepted H run's own SHA256SUMS."""
    manifest_path = run_dir / "SHA256SUMS"
    _require(manifest_path.is_file(), f"{manifest_path}: accepted H evidence manifest is missing")
    payload, _ = read_authoritative_bytes(manifest_path, max_bytes=1 << 24)
    entries: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        _require(
            len(digest) == 64 and relative,
            f"{manifest_path}: malformed manifest line {line!r}",
        )
        entries[relative] = digest
    _require(bool(entries), f"{manifest_path}: evidence manifest is empty")
    for relative, expected in sorted(entries.items()):
        path = run_dir / relative
        _require(path.is_file(), f"accepted H evidence file missing: {relative}")
        actual = file_sha256(path)
        _require(
            actual == expected,
            f"accepted H evidence file {relative} changed: {actual} != {expected}",
        )
    return entries


def load_accepted_h(
    run_dir: Path,
    *,
    expected_run_identity: str = ACCEPTED_H_RUN_IDENTITY,
    expected_predictions_sha256: str = ACCEPTED_H_PREDICTIONS_SHA256,
    expected_complete_sha256: str = ACCEPTED_H_COMPLETE_SHA256,
    expected_census_sha256: str | None = None,
) -> AcceptedH:
    """Bind the accepted Stage-H result: census first, projection second, agreement last."""
    verify_h_evidence_manifest(run_dir)

    census_root = run_dir / "census"
    _require(census_root.is_dir(), f"{census_root}: accepted H census directory is missing")
    published = sorted(p for p in census_root.iterdir() if p.is_dir() and p.name.startswith("run-"))
    _require(
        len(published) == 1,
        f"{census_root}: expected exactly one published H run, found {len(published)}",
    )
    census = load_published_run(published[0], expected_run_identity=expected_run_identity)
    census_sha256 = file_sha256(published[0] / "census.json")
    if expected_census_sha256 is not None:
        _require(
            census_sha256 == expected_census_sha256,
            f"accepted H census SHA-256 {census_sha256} is not the bound {expected_census_sha256}",
        )

    # Bind the exact COMPLETE bytes, not merely the marker's name. The reviewed plan recorded the
    # literal string "COMPLETE", which says nothing about which run finished: any directory with a
    # file of that name would have satisfied it.
    complete_sha256 = file_sha256(published[0] / "COMPLETE")
    _require(
        complete_sha256 == expected_complete_sha256,
        f"accepted H COMPLETE SHA-256 {complete_sha256} is not the bound "
        f"{expected_complete_sha256}",
    )

    predictions_path = run_dir / "evidence" / "H_PREDICTIONS.json"
    payload, predictions_sha256 = read_authoritative_bytes(predictions_path, max_bytes=1 << 28)
    _require(
        predictions_sha256 == expected_predictions_sha256,
        f"H_PREDICTIONS.json SHA-256 {predictions_sha256} is not the accepted "
        f"{expected_predictions_sha256}",
    )
    predictions = strict_json_object(payload, where=str(predictions_path))

    _require(
        predictions.get("census_sha256") == census_sha256,
        "H_PREDICTIONS.json does not describe the accepted canonical census",
    )
    _require(
        predictions.get("run_identity") == expected_run_identity,
        "H_PREDICTIONS.json carries a different H run identity",
    )

    census_nodes = {node["source_id"]: node for node in census["nodes"]}
    projected = predictions.get("predictions")
    _require(type(projected) is dict, "H_PREDICTIONS.json has no predictions object")
    _require(
        set(projected) == set(census_nodes),
        "H_PREDICTIONS.json does not cover exactly the canonical census nodes",
    )
    for source_id, entry in projected.items():
        canonical = census_nodes[source_id]
        for projected_key, census_key in _PREDICTION_TO_CENSUS.items():
            _require(
                projected_key in entry,
                f"H_PREDICTIONS.json[{source_id}] is missing {projected_key!r}",
            )
            _require(
                entry[projected_key] == canonical[census_key],
                f"H_PREDICTIONS.json[{source_id}].{projected_key} disagrees with the canonical "
                f"census field {census_key!r}",
            )
    _require(
        predictions.get("predicted_ownership_matrix") == census["ownership_matrix"],
        "H_PREDICTIONS.json ownership matrix disagrees with the canonical census",
    )
    _require(
        predictions.get("predicted_totals") == census["totals"],
        "H_PREDICTIONS.json totals disagree with the canonical census",
    )
    _require(
        predictions.get("node_execution_order") == [n["source_id"] for n in census["nodes"]],
        "H_PREDICTIONS.json execution order disagrees with the canonical census",
    )

    return AcceptedH(
        run_dir=run_dir,
        census=census,
        predictions=predictions,
        census_sha256=census_sha256,
        predictions_sha256=predictions_sha256,
        complete_sha256=complete_sha256,
        run_identity=expected_run_identity,
    )


# --------------------------------------------------------------------- H/I comparison


def compare_node(selection: NodeSelection, h_node: Mapping[str, Any]) -> dict[str, Any]:
    """One node's full H/I comparison, most-important dimensions named explicitly."""
    realized = selection.comparable()
    dimensions: dict[str, Any] = {}
    for key in H_I_REQUIRED_DIMENSIONS:
        dimensions[key] = {
            "h": h_node[key],
            "i": realized[key],
            "match": h_node[key] == realized[key],
        }
    ownership_match = realized["exclusions_by_owner"] == dict(
        sorted(h_node["exclusions_by_owner"].items())
    )
    dimensions["exclusions_by_owner"] = {
        "h": dict(sorted(h_node["exclusions_by_owner"].items())),
        "i": realized["exclusions_by_owner"],
        "match": ownership_match,
    }
    extra_mismatches = sorted(
        key
        for key in realized
        if key in h_node and realized[key] != h_node[key] and key not in dimensions
    )
    return {
        "source_id": selection.source_id,
        "stage": selection.stage,
        "target_serialized_tokens": selection.target_serialized_tokens,
        "dimensions": dimensions,
        "additional_field_mismatches": extra_mismatches,
        "node_match": all(d["match"] for d in dimensions.values()) and not extra_mismatches,
    }


def compare_with_h(selections: list[NodeSelection], accepted: AcceptedH) -> dict[str, Any]:
    """Mechanical comparison for all nodes. Produces a verdict; it does not decide what to do."""
    h_nodes = {node["source_id"]: node for node in accepted.census["nodes"]}
    _require(
        [s.source_id for s in selections] == [n["source_id"] for n in accepted.census["nodes"]],
        "Stage-I node execution order disagrees with the accepted Stage-H order",
    )
    nodes = [compare_node(selection, h_nodes[selection.source_id]) for selection in selections]
    realized_matrix = ownership_matrix_v1(selections)
    matrix_match = realized_matrix == dict(accepted.h_ownership_matrix)

    def all_match(dimension: str) -> bool:
        return all(node["dimensions"][dimension]["match"] for node in nodes)

    return {
        "nodes": nodes,
        "ownership_matrix": {
            "h": dict(accepted.h_ownership_matrix),
            "i": realized_matrix,
            "match": matrix_match,
        },
        "ALL_H_I_BRANCHES_MATCH": all_match("branch"),
        "ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH": all_match("selected_serialized_tokens"),
        "ALL_H_I_FINGERPRINTS_MATCH": all_match("selection_fingerprint"),
        "ALL_H_I_CROSSING_IDENTITIES_MATCH": all_match("crossing_identity"),
        "ALL_H_I_OVERSHOOTS_MATCH": all_match("actual_overshoot_tokens"),
        "OWNERSHIP_MATRIX_MATCH": matrix_match,
        "ALL_NODES_MATCH": all(node["node_match"] for node in nodes) and matrix_match,
    }


def require_h_i_equality(comparison: Mapping[str, Any]) -> None:
    """The gate. Nothing may be materialized unless every required equality holds."""
    failed = [
        key
        for key in (
            "ALL_H_I_BRANCHES_MATCH",
            "ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH",
            "ALL_H_I_FINGERPRINTS_MATCH",
            "ALL_H_I_CROSSING_IDENTITIES_MATCH",
            "ALL_H_I_OVERSHOOTS_MATCH",
            "OWNERSHIP_MATRIX_MATCH",
        )
        if not comparison[key]
    ]
    if failed or not comparison["ALL_NODES_MATCH"]:
        offenders = sorted(
            node["source_id"] for node in comparison["nodes"] if not node["node_match"]
        )
        raise RealizationError(
            "Stage-I realization disagrees with the accepted Stage-H prediction; refusing to "
            f"materialize. failed conditions={failed or ['ALL_NODES_MATCH']} nodes={offenders}"
        )


# --------------------------------------------------------------------- pass 1: derivation

REFERENCE_EXCLUSION_SCOPE = "entire_pre_tokenizer_reserved_pool"


def load_reference_exclusion(path: Path, *, expected_sha256: str | None = None) -> set[str]:
    """Load the whole-reserve exclusion identity set from one verified descriptor."""
    payload, digest = read_authoritative_bytes(path, max_bytes=1 << 31)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"reference exclusion manifest SHA mismatch at {path}")
    manifest = strict_json_object(payload, where="reference exclusion manifest")
    _require(
        manifest.get("exclusion_scope") == REFERENCE_EXCLUSION_SCOPE,
        "reference exclusion manifest is not the whole-reserve scope",
    )
    hashes = manifest.get("hashes")
    _require(isinstance(hashes, list), "reference exclusion manifest has no hash list")
    identities = set(hashes)
    count = manifest.get("hash_count")
    _require(
        not isinstance(count, bool) and isinstance(count, int),
        "reference exclusion hash_count must be an exact integer",
    )
    _require(
        len(identities) == count and len(hashes) == count,
        "reference exclusion hash_count disagrees with the hash list",
    )
    return identities


def _decode_row(raw: bytes, *, where: str) -> dict[str, Any]:
    try:
        return strict_json_object(raw, where=where)
    except StrictJSONError as exc:
        raise RealizationError(str(exc)) from exc


def _document_payload(
    row: Mapping[str, Any], binding: InputBinding, *, where: str
) -> tuple[str, str] | None:
    """Return ``(raw_text, cleaned_text)`` for one eligible row, or None if it drops out."""
    _require(
        SELECTION_METADATA_FIELD not in row,
        f"{where}: reserved field {SELECTION_METADATA_FIELD!r} present",
    )
    text = row.get(binding.text_field)
    _require(text is not None, f"{where}: missing {binding.text_field!r}")
    _require(isinstance(text, str), f"{where}: {binding.text_field!r} is not a string")
    if not text:
        return None
    cleaning = binding.cleaning_contract
    cleaned = clean_text(
        text,
        strip_leading_noise=cleaning["strip_leading_noise"],
        normalize_quotes=cleaning["normalize_quotes"],
        underscores_policy=cleaning["underscores_policy"],
        min_chars=cleaning["min_chars"],
        min_ascii_ratio=cleaning["min_ascii_ratio"],
    )
    if cleaned is None:
        return None
    return text, cleaned


def _encode_document(tokenizer: Any, cleaned: str, *, where: str) -> tuple[int, int]:
    """Canonical ``[BOS] document [EOS]`` accounting for one cleaned document."""
    ids, content, boundary = encode_with_accounting(
        tokenizer, cleaned, add_bos=True, add_eos=True, bos_id=BOS_ID, eos_id=EOS_ID
    )
    _require(
        bool(ids) and ids[0] == BOS_ID and ids[-1] == EOS_ID and boundary == 2,
        f"{where}: framing violation",
    )
    leaked = SPECIAL_IDS_INTERIOR.intersection(ids[1:-1])
    _require(not leaked, f"{where}: text encoded to special ids {sorted(leaked)}")
    return content, content + boundary


SPECIAL_IDS_INTERIOR = frozenset(SPECIAL_TOKEN_IDS.values())


def scan_binding_candidates(
    binding: InputBinding, tokenizer_path: str, excluded_rows: Any
) -> tuple[list[Candidate], dict[str, int]]:
    """Pass 1 over one frozen release: stream it once and emit eligible candidate metadata.

    The corpus is opened exactly once and its digest is computed from the same descriptor the rows
    are parsed from, so there is no window in which the bytes validated and the bytes consumed
    could differ. The physical row index is the stable immutable record ordinal: the eligibility
    index is already expressed in those indices and ``documents.jsonl`` is pinned by SHA-256 and
    exact size, so no new locator semantics are invented here.
    """
    excluded = {int(row) for row in excluded_rows.tolist()}
    _require(
        len(excluded) == binding.excluded_rows,
        f"{binding.input_binding_id}: eligibility rows are not unique",
    )
    tokenizer = load_tokenizer(tokenizer_path)
    accessor = binding.schema_accessor
    candidates: list[Candidate] = []
    counters = {
        "physical_rows": 0,
        "d2_d3_excluded_rows": 0,
        "empty_or_non_string_text": 0,
        "empty_after_cleaning": 0,
        "eligible_rows": 0,
    }
    digest = hashlib.sha256()

    with open_authoritative(binding.documents_path, buffering=8 * 1024 * 1024) as (handle, _id):
        for row_index, raw in enumerate(handle):
            digest.update(raw)
            counters["physical_rows"] += 1
            if row_index in excluded:
                counters["d2_d3_excluded_rows"] += 1
                continue
            where = f"{binding.input_binding_id}:{row_index}"
            row = _decode_row(raw, where=where)
            payload = _document_payload(row, binding, where=where)
            if payload is None:
                if row.get(binding.text_field):
                    counters["empty_after_cleaning"] += 1
                else:
                    counters["empty_or_non_string_text"] += 1
                continue
            text, cleaned = payload
            content, serialized = _encode_document(tokenizer, cleaned, where=where)
            raw_continuous = read_score_v1(row, accessor, "continuous_score")
            try:
                record_digest = hashlib.sha256(_canonical_json_bytes(row)).hexdigest()
            except ValueError as exc:
                raise RealizationError(f"{where}: {exc}") from exc
            candidates.append(
                Candidate(
                    input_binding_id=binding.input_binding_id,
                    stable_input_record_ordinal=row_index,
                    raw_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    input_record_sha256=record_digest,
                    cleaned_sha256=cleaned_text_sha256(cleaned),
                    canonical_fingerprint=canonical_document_fingerprint_v1(cleaned),
                    content_token_count=content,
                    serialized_token_count=serialized,
                    int_score=read_score_v1(row, accessor, "integer_score"),
                    score_bits=(
                        None if raw_continuous is None else score_to_bits_v1(raw_continuous)
                    ),
                )
            )
            counters["eligible_rows"] += 1

    _require(
        counters["physical_rows"] == binding.total_physical_rows,
        f"{binding.input_binding_id}: physical row count mismatch",
    )
    _require(
        counters["d2_d3_excluded_rows"] == binding.excluded_rows,
        f"{binding.input_binding_id}: eligibility exclusions not fully consumed",
    )
    _require(
        digest.hexdigest() == binding.documents_sha256,
        f"{binding.input_binding_id}: documents SHA-256 mismatch",
    )
    return candidates, counters


# --------------------------------------------------------------------- pass 2: materialization


@dataclass(frozen=True)
class SelectedTarget:
    """What pass 2 must find at one physical locator, and what it must prove about it."""

    stage: str
    source_id: str
    selection_ordinal_within_node: int
    cleaned_sha256: str
    raw_sha256: str
    input_record_sha256: str
    canonical_fingerprint: str
    content_token_count: int
    serialized_token_count: int


def build_materialization_targets(
    selections: list[NodeSelection],
) -> dict[str, dict[int, SelectedTarget]]:
    """Invert the selection ledger into ``binding -> ordinal -> target``.

    A locator may appear at most once across the whole realization: ownership is global over
    cleaned identities, so two nodes can never both commit the same document, and a shared binding
    must therefore never produce duplicate A/B copies of one row.
    """
    targets: dict[str, dict[int, SelectedTarget]] = {}
    claimed: dict[str, str] = {}
    for selection in selections:
        for document in selection.selected:
            owner = claimed.get(document.cleaned_sha256)
            _require(
                owner is None,
                f"identity {document.cleaned_sha256} is claimed by both {owner!r} and "
                f"{selection.source_id!r}; global ownership is violated",
            )
            claimed[document.cleaned_sha256] = selection.source_id
            per_binding = targets.setdefault(document.input_binding_id, {})
            _require(
                document.stable_input_record_ordinal not in per_binding,
                f"{document.input_binding_id}:{document.stable_input_record_ordinal} would be "
                "materialized twice",
            )
            per_binding[document.stable_input_record_ordinal] = SelectedTarget(
                stage=selection.stage,
                source_id=selection.source_id,
                selection_ordinal_within_node=document.selection_ordinal_within_node,
                cleaned_sha256=document.cleaned_sha256,
                raw_sha256=document.raw_sha256,
                input_record_sha256=document.input_record_sha256,
                canonical_fingerprint=document.canonical_fingerprint,
                content_token_count=document.content_token_count,
                serialized_token_count=document.serialized_token_count,
            )
    return targets


def materialize_binding(
    binding: InputBinding,
    tokenizer_path: str,
    wanted: Mapping[int, SelectedTarget],
) -> Iterator[dict[str, Any]]:
    """Re-read one bound release and yield the selected documents, recomputing everything.

    This is a generator on purpose. At production scale a single binding contributes millions of
    selected documents and tens of gigabytes of training text; accumulating them would put the
    whole selected corpus in memory for no benefit. Yielding lets the caller spool each record as
    it appears, so peak memory stays independent of how much of a binding was selected.

    Nothing is trusted from pass 1: the raw digest, the record digest, the cleaned identity, the
    canonical fingerprint and both token counts are all recomputed from the bytes actually being
    written, then compared with what selection committed to. A disagreement means the frozen input
    moved underneath the run, and the realization stops rather than publishing text that does not
    match the identity it claims.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    found: set[int] = set()
    digest = hashlib.sha256()

    with open_authoritative(binding.documents_path, buffering=8 * 1024 * 1024) as (handle, _id):
        for row_index, raw in enumerate(handle):
            digest.update(raw)
            target = wanted.get(row_index)
            if target is None:
                continue
            where = f"{binding.input_binding_id}:{row_index}"
            row = _decode_row(raw, where=where)
            payload = _document_payload(row, binding, where=where)
            _require(payload is not None, f"{where}: selected row is no longer eligible")
            text, cleaned = payload
            content, serialized = _encode_document(tokenizer, cleaned, where=where)

            recomputed = {
                "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "input_record_sha256": hashlib.sha256(_canonical_json_bytes(row)).hexdigest(),
                "cleaned_sha256": cleaned_text_sha256(cleaned),
                "canonical_fingerprint": canonical_document_fingerprint_v1(cleaned),
                "content_token_count": content,
                "serialized_token_count": serialized,
            }
            for key, value in recomputed.items():
                _require(
                    value == getattr(target, key),
                    f"{where}: recomputed {key} disagrees with the committed selection "
                    f"({value!r} != {getattr(target, key)!r})",
                )

            found.add(row_index)
            yield build_record(
                stage=target.stage,
                source_id=target.source_id,
                input_binding_id=binding.input_binding_id,
                stable_input_record_ordinal=row_index,
                input_record_sha256=recomputed["input_record_sha256"],
                raw_sha256=recomputed["raw_sha256"],
                cleaned_text_sha256=recomputed["cleaned_sha256"],
                canonical_fingerprint=recomputed["canonical_fingerprint"],
                selection_ordinal_within_node=target.selection_ordinal_within_node,
                content_token_count=content,
                serialized_token_count=serialized,
                training_text=cleaned,
            )

    missing = sorted(set(wanted) - found)
    _require(not missing, f"{binding.input_binding_id}: selected rows not found: {missing[:8]}")
    _require(
        digest.hexdigest() == binding.documents_sha256,
        f"{binding.input_binding_id}: documents SHA-256 changed between pass 1 and pass 2",
    )


# --------------------------------------------------------------------- orchestration


def derive_selection(
    graph: SourceGraph, tokenizer_path: str, reference_exclusion: set[str]
) -> tuple[list[NodeSelection], dict[str, dict[str, int]]]:
    """Pass 1 end to end: scan every bound release once, then decide."""
    candidates_by_binding: dict[str, list[Candidate]] = {}
    counters: dict[str, dict[str, int]] = {}
    for binding_id in sorted(graph.bindings):
        binding = graph.bindings[binding_id]
        candidates, stats = scan_binding_candidates(
            binding, tokenizer_path, graph.validated_eligibility_rows(binding_id)
        )
        candidates_by_binding[binding_id] = candidates
        counters[binding_id] = stats
    return realize_selection(graph, candidates_by_binding, reference_exclusion), counters


def _spool_key(stage: str, source_id: str, binding_id: str) -> tuple[int, str, str]:
    return (STAGE_PRIORITY[stage], source_id, binding_id)


def iter_records_in_physical_order(
    graph: SourceGraph,
    tokenizer_path: str,
    selections: list[NodeSelection],
    work_dir: Path,
    *,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
) -> Iterator[dict[str, Any]]:
    """Pass 2 end to end, emitting records in the frozen physical layout order.

    Each bound release is read exactly once. Because one binding can feed several nodes, records
    are spooled per ``(stage_priority, source_id, input_binding_id)`` group as they are produced,
    and the groups are then streamed back in canonical order. That keeps the total to one full
    corpus pass while still emitting a globally ordered stream.

    Both halves are bounded. Writing holds one record; replay reads each spool through a fixed
    window and holds one framed line plus a partial tail. The reviewed defect was here: replay
    used to pull an entire ``(stage, source, binding)`` spool into memory with a single read
    capped at 16 GiB, which is not an implementation for a source the size of FineWeb Stage A --
    that one spool alone is tens of gigabytes. Memory now depends on the window, not on the spool.
    """
    targets = build_materialization_targets(selections)
    work_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[tuple[int, str, str], Any] = {}
    paths: dict[tuple[int, str, str], Path] = {}
    try:
        for binding_id in sorted(graph.bindings):
            wanted = targets.get(binding_id, {})
            if not wanted:
                continue
            binding = graph.bindings[binding_id]
            for record in materialize_binding(binding, tokenizer_path, wanted):
                key = _spool_key(record["stage"], record["source_id"], binding_id)
                handle = handles.get(key)
                if handle is None:
                    path = work_dir / f"spool-{key[0]}-{key[1]}-{key[2]}.jsonl"
                    paths[key] = path
                    handle = open(path, "wb")
                    handles[key] = handle
                handle.write(canonical_json_bytes(record))
        for handle in handles.values():
            handle.flush()
            handle.close()
        handles.clear()

        # Streaming replay. There is deliberately no whole-spool read here; `stream_lines`
        # advances through the file in fixed windows and never materialises it.
        for key in sorted(paths):
            path = paths[key]
            for line in stream_lines(path, read_window_bytes=read_window_bytes):
                yield strict_json_object(line, where=str(path))
    finally:
        for handle in handles.values():
            handle.close()
        for path in paths.values():
            path.unlink(missing_ok=True)


def build_manifest(
    selections: list[NodeSelection],
    context: AuthorizedIContext,
) -> dict[str, Any]:
    """The manifest minus its shard list, which only the publisher can fill in.

    The graph is taken from the authorized context rather than passed in separately. An earlier
    signature accepted both, which meant a caller could describe a realization with one graph while
    the run identity had been computed from another -- the same substitution class this repair
    exists to close.

    The totals here are what the selection ledger believes. They are NOT taken on trust: the
    publisher audits the staged records and refuses to publish unless every number below equals
    one derived from the bytes on disk.
    """
    _require_authorized(context, "manifest construction")
    accepted = context.accepted
    graph = context.graph
    total_records = sum(s.selected_identities for s in selections)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "shards": [],
        "totals": {
            "records": total_records,
            "content_tokens": sum(d.content_token_count for s in selections for d in s.selected),
            "serialized_tokens": sum(s.selected_serialized_tokens for s in selections),
            "unique_cleaned_identities": total_records,
            # Filled in by the publisher, which is the only thing that knows how many shards the
            # declared policy actually produced; the audit then requires the two to agree.
            "shards": 0,
        },
        "nodes": [
            {
                "source_id": s.source_id,
                "stage": s.stage,
                "target_serialized_tokens": s.target_serialized_tokens,
                "branch": s.branch,
                "selection_mode": s.selection_mode,
                "selected_identities": s.selected_identities,
                "selected_serialized_tokens": s.selected_serialized_tokens,
                "selection_fingerprint": s.selection_fingerprint,
                # R2-A: the expected order-sensitive commitment, computed from the Pass-1
                # selection sequence -- before anything is materialized, and not from the physical
                # records the audit will later check against it.
                "selection_sequence_commitment": selection_sequence_commitment(
                    source_id=s.source_id,
                    stage=s.stage,
                    pairs=((d.selection_ordinal_within_node, d.cleaned_sha256) for d in s.selected),
                ),
                "input_binding_ids": sorted({d.input_binding_id for d in s.selected}),
                "crossing_identity": s.crossing_identity,
                "actual_overshoot_tokens": s.actual_overshoot_tokens,
            }
            for s in selections
        ],
        "node_binding_projection": {
            source_id: list(allowed)
            for source_id, allowed in sorted(context.node_binding_projection.items())
        },
        "ownership_matrix": ownership_matrix_v1(selections),
        "bindings": {
            binding_id: graph.bindings[binding_id].documents_sha256
            for binding_id in sorted(graph.bindings)
        },
        "environment": context.environment.as_canonical(),
        "stage_i_run": {
            "run_identity": context.run_identity,
            "candidate_i_plan_sha256": context.plan_sha256,
            "implementation_commit": context.implementation_commit,
            "implementation_bundle_sha256": context.bundle_sha256,
            "plan_schema_version": PLAN_SCHEMA,
            "output_schema_version": RECORD_SCHEMA,
            "manifest_schema_version": MANIFEST_SCHEMA,
            "shard_policy_version": SHARD_POLICY_VERSION,
            "records_per_shard": RECORDS_PER_SHARD,
            "h_run_identity": accepted.run_identity,
            "h_complete_sha256": accepted.complete_sha256,
            "h_census_sha256": accepted.census_sha256,
            "h_predictions_sha256": accepted.predictions_sha256,
            "owner_graph_sha256": graph.graph_sha256,
            "node_binding_projection_sha256": node_binding_projection_sha256(
                context.node_binding_projection
            ),
        },
        "h_binding": {
            "h_run_identity": accepted.run_identity,
            "h_census_sha256": accepted.census_sha256,
            "h_predictions_sha256": accepted.predictions_sha256,
            "h_complete_sha256": accepted.complete_sha256,
            "h_candidate_plan_sha256": accepted.census["authorization"]["candidate_plan_sha256"],
            "h_implementation_bundle_sha256": accepted.census["authorization"][
                "implementation_bundle_sha256"
            ],
            "owner_graph_sha256": accepted.census["graph_sha256"],
        },
    }


# --------------------------------------------------------------------- candidate I plan


def _git_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RealizationError(f"cannot determine repository HEAD: {exc}") from exc
    return result.stdout.strip()


def generate_candidate_plan(
    *,
    repo_root: Path,
    graph_path: Path,
    graph: SourceGraph,
    accepted: AcceptedH,
    environment: Environment,
    implementation_commit: str | None = None,
) -> dict[str, Any]:
    """A candidate Stage-I plan that cannot authorise itself.

    Authorization is deliberately absent from the bytes: the owner authorises later by supplying
    this plan's exact SHA-256 externally, so the plan can be regenerated, diffed and reviewed
    without ever becoming self-authorising. Every field is derived here, so the sealed
    owner-facing plan is byte-reproducible from this generator with nothing added by hand.
    """
    files = implementation_files(repo_root)
    authorities = {
        name: {
            "path": str(_authority_relative(repo_root, path)),
            "sha256": digest,
        }
        for name, (path, digest) in _authority_paths(repo_root, graph, accepted).items()
    }
    return {
        "schema_version": PLAN_SCHEMA,
        "authorization_status": "NOT_AUTHORIZED",
        "authorization_note": (
            "This plan carries no owner authorization and cannot create one. Stage-I production "
            "must be invoked with --expected-plan-sha256 supplied externally by the owner, and "
            "that authorization applies only to these exact plan bytes, this graph, these "
            "authorities, this accepted Stage-H result and this implementation bundle."
        ),
        "realization_label": REALIZATION_LABEL,
        "resume_supported": RESUME_SUPPORTED,
        "implementation_commit": implementation_commit or _git_head(repo_root),
        "implementation_files": dict(sorted(files.items())),
        "implementation_bundle_sha256": implementation_bundle_sha256(files),
        "output_schema_version": RECORD_SCHEMA,
        "manifest_schema_version": MANIFEST_SCHEMA,
        "shard_policy": {
            "version": SHARD_POLICY_VERSION,
            "records_per_shard": RECORDS_PER_SHARD,
            "rule": (
                "whole records only; fixed record count per shard over the canonical physical "
                "order (stage_priority, source_id, input_binding_id, stable_input_record_ordinal)"
            ),
        },
        "selection_rules": {
            "representative_rule": REPRESENTATIVE_RULE,
            "physical_locator_rule": PHYSICAL_LOCATOR_RULE,
        },
        "accepted_h": {
            "run_dir": str(_authority_relative(repo_root, accepted.run_dir)),
            "run_identity": accepted.run_identity,
            "census_sha256": accepted.census_sha256,
            "predictions_sha256": accepted.predictions_sha256,
            # The exact COMPLETE bytes, not the marker's name. Recording only the literal
            # "COMPLETE" said nothing about which Stage-H run finished.
            "complete_sha256": accepted.complete_sha256,
            "candidate_plan_sha256": accepted.census["authorization"]["candidate_plan_sha256"],
            "implementation_bundle_sha256": accepted.census["authorization"][
                "implementation_bundle_sha256"
            ],
        },
        "graph_path": str(_authority_relative(repo_root, graph_path)),
        "graph_sha256": graph.graph_sha256,
        "seed": graph.seed,
        "authorities": dict(sorted(authorities.items())),
        "bound_authorities": dict(sorted(graph.bound_authorities.items())),
        "input_bindings": {
            binding_id: {
                "documents_sha256": graph.bindings[binding_id].documents_sha256,
                "documents_size_bytes": graph.bindings[binding_id].documents_size_bytes,
                "eligibility_index_sha256": graph.bindings[binding_id].eligibility_index_sha256,
                "release_manifest_sha256": graph.bindings[binding_id].release_manifest_sha256,
                "total_physical_rows": graph.bindings[binding_id].total_physical_rows,
                "expected_eligible_rows": graph.bindings[binding_id].expected_eligible_rows,
            }
            for binding_id in sorted(graph.bindings)
        },
        "node_order": [node.source_id for node in graph.nodes],
        # R2-B: the closed node -> authorized-binding projection, taken from the frozen owner
        # graph. Bound here so the audit has an authority to check records against, rather than
        # inferring what is allowed from whatever the output happens to contain.
        "node_binding_projection": {
            node.source_id: sorted(set(node.input_binding_ids)) for node in graph.nodes
        },
        "nodes": [
            {
                "source_id": node.source_id,
                "stage": node.stage,
                "stage_priority": node.stage_priority,
                "target_serialized_tokens": node.target_serialized_tokens,
                "selection_mode": node.selection_mode,
                "input_binding_ids": list(node.input_binding_ids),
            }
            for node in graph.nodes
        ],
        "environment_contract": {
            "python_executable": REQUIRED_PYTHON_EXECUTABLE,
            "python_version": REQUIRED_PYTHON_VERSION,
            "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
            "observed_at_generation": environment.as_canonical(),
        },
    }


def _authority_relative(repo_root: Path, path: Path) -> Path:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve())
    except ValueError:
        return Path(path)


def _authority_paths(
    repo_root: Path, graph: SourceGraph, accepted: AcceptedH
) -> dict[str, tuple[Path, str]]:
    """Resolve every authority the accepted Stage-H census bound, and re-hash it from disk."""
    plan_authorities = accepted.census["authorization"]["authority_sha256"]
    h_plan = accepted.predictions.get("bound_authorities", {})
    _require(
        dict(h_plan) == dict(graph.bound_authorities),
        "accepted H bound authorities disagree with the frozen graph",
    )
    resolved: dict[str, tuple[Path, str]] = {}
    for name, expected in sorted(plan_authorities.items()):
        path = _AUTHORITY_PATHS[name]
        absolute = repo_root / path
        _require(absolute.is_file(), f"authority {name} missing at {path}")
        digest = file_sha256(absolute)
        _require(
            digest == expected,
            f"authority {name} changed: {digest} != accepted {expected}",
        )
        resolved[name] = (absolute, digest)
    return resolved


# The accepted Stage-H census names its authorities by key; their frozen locations come from the
# accepted H candidate plan and are re-hashed against the census values above.
_AUTHORITY_PATHS = {
    "d2_d3_eligibility_manifest": "runs/l1_production_2026-08-20/eligibility/eligibility_manifest.json",
    "g2_release_manifest": "runs/g2_production_2026-08-21/release/manifest.json",
    "g_release_manifest": "runs/g_production_2026-08-21/release/manifest.json",
    "hq_policy": "runs/h_production_2026-08-21/policy/stage_b_hq_policy_v1.json",
    "reference_exclusion": "runs/g2_production_2026-08-21/release/exclusion_hash_manifest.json",
    "selector_v1": "pretrain/select_pretrain_documents.py",
    "stage_e_allocation": "runs/stage_e_2026-08-20/allocation_contract.json",
    "tokenizer": "runs/g_production_2026-08-21/release/tokenizer.json",
}


# --------------------------------------------------------------------- authoritative run


def stage_i_run_identity(
    *,
    node_binding_projection_sha256: str,
    candidate_plan_sha256: str,
    implementation_commit: str,
    implementation_bundle_sha256: str,
    plan_schema_version: str,
    output_schema_version: str,
    manifest_schema_version: str,
    shard_policy_version: str,
    records_per_shard: int,
    h_run_identity: str,
    h_complete_sha256: str,
    h_census_sha256: str,
    h_predictions_sha256: str,
    owner_graph_sha256: str,
) -> str:
    """The published run's name, over an explicit, versioned, closed field list.

    Written out field by field rather than hashing an open-ended dictionary: a dict digest would
    silently change meaning the day someone adds a key, and would let two runs that differ in an
    unlisted field claim the same identity. A change to any field below is a different run, always.
    """
    payload = {
        "schema_version": STAGE_I_RUN_IDENTITY_SCHEMA,
        "candidate_plan_sha256": candidate_plan_sha256,
        "implementation_commit": implementation_commit,
        "implementation_bundle_sha256": implementation_bundle_sha256,
        "plan_schema_version": plan_schema_version,
        "output_schema_version": output_schema_version,
        "manifest_schema_version": manifest_schema_version,
        "shard_policy_version": shard_policy_version,
        "records_per_shard": records_per_shard,
        "h_run_identity": h_run_identity,
        "h_complete_sha256": h_complete_sha256,
        "h_census_sha256": h_census_sha256,
        "h_predictions_sha256": h_predictions_sha256,
        "owner_graph_sha256": owner_graph_sha256,
        "node_binding_projection_sha256": node_binding_projection_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


# Exact-instance authority registry. R1 stored a sentinel in a context field, which meant
# `copy.copy` carried it and anything that could import the sentinel could stamp a lookalike.
# Authority now lives here, keyed by object identity, alongside the canonical state digest that
# instance was authorized for -- so there is no field on the object that granting authority
# depends on, and no importable constant that confers it.
_AUTHORIZED: dict[int, tuple[weakref.ref, str]] = {}


def _register_authorized(context: AuthorizedIContext, state_digest: str) -> None:
    key = id(context)

    def _drop(_ref: Any, key: int = key) -> None:
        _AUTHORIZED.pop(key, None)

    _AUTHORIZED[key] = (weakref.ref(context, _drop), state_digest)


@dataclass(frozen=True)
class AuthorizedIContext:
    """The single capability that can run and publish an authoritative Stage-I realization.

    Authority is the *exact instance* returned by a successful ``authorize_plan``. It is recorded
    in a module-private registry keyed by object identity, so a manually built lookalike, a
    restamped object or a copy is not this instance and cannot act as one. Copying is refused
    outright rather than silently producing a powerless twin.

    The canonical state this instance was authorized for is digested at authorization time and the
    digest is held in the registry, not on the object. ``revalidate`` recomputes that digest from
    the object's current fields and from a fresh derivation off disk, so swapping ``graph`` or
    ``accepted`` in memory, or changing a bound artifact on disk, both fail.
    """

    repo_root: Path
    plan_path: Path
    plan_sha256: str
    plan: Mapping[str, Any]
    graph: SourceGraph
    graph_path: Path
    accepted: AcceptedH
    environment: Environment
    bundle_sha256: str
    bundle_files: Mapping[str, str]
    implementation_commit: str
    tokenizer_path: Path
    reference_exclusion_path: Path
    node_binding_projection: Mapping[str, tuple[str, ...]]
    run_identity: str

    def __copy__(self) -> None:
        raise RealizationError(
            "an AuthorizedIContext must not be copied; authority is the exact authorized instance"
        )

    def __deepcopy__(self, memo: Any) -> None:
        raise RealizationError(
            "an AuthorizedIContext must not be copied; authority is the exact authorized instance"
        )

    def __reduce__(self) -> None:
        raise RealizationError("an AuthorizedIContext must not be pickled or reconstructed")

    @property
    def run_name(self) -> str:
        return f"run-{self.run_identity[:32]}"

    def revalidate(self) -> None:
        """Prove the authorized state is still exactly what was authorized, in memory and on disk.

        Three separate things are checked, because R1 only did the third and Codex walked through
        the gap: the instance is the registered authority; the object's *current* fields still
        digest to the state it was authorized for (catching an in-memory graph or H substitution);
        and a fresh derivation from the plan bytes and bound artifacts digests to the same value
        (catching a change on disk).
        """
        expected = _require_authorized(self, "revalidation")
        _require(
            _canonical_state_digest(self) == expected,
            "authorized Stage-I state was substituted after authorization: the context's current "
            "graph/H/authority projection no longer matches what was authorized",
        )
        fresh = _derive_canonical_state(self.repo_root, self.plan_path, self.plan_sha256)
        _require(
            fresh == expected,
            "the plan bytes or a bound artifact changed after authorization",
        )


def _canonical_state_digest(context: AuthorizedIContext) -> str:
    """Digest over every load-bearing projection an authorized run depends on.

    Written out explicitly rather than hashing the object, so adding a field is a deliberate act
    and a substituted nested object cannot hide behind an unhashed attribute.
    """
    graph = context.graph
    accepted = context.accepted
    payload = {
        "schema_version": CANONICAL_STATE_SCHEMA,
        "plan_sha256": context.plan_sha256,
        "implementation_commit": context.implementation_commit,
        "implementation_bundle_sha256": context.bundle_sha256,
        "implementation_files": dict(sorted(context.bundle_files.items())),
        "graph_sha256": graph.graph_sha256,
        "graph_seed": graph.seed,
        "graph_bound_authorities": dict(sorted(graph.bound_authorities.items())),
        "graph_nodes": [
            {
                "source_id": node.source_id,
                "stage": node.stage,
                "stage_priority": node.stage_priority,
                "target_serialized_tokens": node.target_serialized_tokens,
                "selection_mode": node.selection_mode,
                "input_binding_ids": list(node.input_binding_ids),
            }
            for node in graph.nodes
        ],
        "graph_bindings": {
            binding_id: {
                "documents_sha256": graph.bindings[binding_id].documents_sha256,
                "documents_size_bytes": graph.bindings[binding_id].documents_size_bytes,
                "eligibility_index_sha256": graph.bindings[binding_id].eligibility_index_sha256,
                "release_manifest_sha256": graph.bindings[binding_id].release_manifest_sha256,
                "total_physical_rows": graph.bindings[binding_id].total_physical_rows,
                "expected_eligible_rows": graph.bindings[binding_id].expected_eligible_rows,
            }
            for binding_id in sorted(graph.bindings)
        },
        "h_run_identity": accepted.run_identity,
        "h_census_sha256": accepted.census_sha256,
        "h_predictions_sha256": accepted.predictions_sha256,
        "h_complete_sha256": accepted.complete_sha256,
        "h_candidate_plan_sha256": accepted.census["authorization"]["candidate_plan_sha256"],
        "h_bundle_sha256": accepted.census["authorization"]["implementation_bundle_sha256"],
        "h_graph_sha256": accepted.census["graph_sha256"],
        "authorities": {
            name: entry["sha256"] for name, entry in sorted(context.plan["authorities"].items())
        },
        "node_binding_projection": {
            source_id: list(bindings)
            for source_id, bindings in sorted(context.node_binding_projection.items())
        },
        "plan_schema_version": PLAN_SCHEMA,
        "output_schema_version": RECORD_SCHEMA,
        "manifest_schema_version": MANIFEST_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "environment": context.environment.as_canonical(),
        "run_identity": context.run_identity,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_authorized(context: Any, where: str) -> str:
    """Return the registered state digest, or refuse. Identity is the capability."""
    if not isinstance(context, AuthorizedIContext):
        raise RealizationError(f"{where} requires an authorized Stage-I context")
    entry = _AUTHORIZED.get(id(context))
    if entry is None or entry[0]() is not context:
        raise RealizationError(
            f"{where} requires the exact Stage-I context instance returned by authorize_plan; a "
            "manually constructed, restamped, copied or replaced context is not an authorization"
        )
    return entry[1]


def _closed_plan(obj: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    _require(type(obj) is dict, f"candidate plan: {where} must be a JSON object")
    present = frozenset(obj)
    unknown = sorted(present - fields)
    missing = sorted(fields - present)
    _require(not unknown, f"candidate plan: {where} carries unknown field(s): {unknown}")
    _require(not missing, f"candidate plan: {where} is missing field(s): {missing}")
    return obj


def validate_plan_schema(plan: Any) -> dict[str, Any]:
    """Closed-schema validation of the candidate plan, at every level."""
    obj = _closed_plan(plan, _PLAN_TOP_FIELDS, "top level")
    _require(
        obj["schema_version"] == PLAN_SCHEMA,
        f"candidate plan schema {obj['schema_version']!r} is not {PLAN_SCHEMA}",
    )
    _require(
        obj["authorization_status"] == "NOT_AUTHORIZED",
        "candidate plan bytes must not carry an owner authorization",
    )
    _require(obj["resume_supported"] is False, "candidate plan must not enable resume")
    _require(
        obj["realization_label"] == REALIZATION_LABEL,
        "candidate plan carries the wrong realization label",
    )
    for key in ("authorization_note", "graph_path", "implementation_commit"):
        _require(type(obj[key]) is str and obj[key], f"candidate plan: {key} must be a string")
    _require(type(obj["seed"]) is int, "candidate plan: seed must be an exact integer")
    for key in ("graph_sha256", "implementation_bundle_sha256"):
        _require(
            type(obj[key]) is str and len(obj[key]) == 64,
            f"candidate plan: {key} must be a 64-hex digest",
        )

    _closed_plan(obj["shard_policy"], _PLAN_SHARD_POLICY_FIELDS, "shard_policy")
    _closed_plan(obj["selection_rules"], _PLAN_SELECTION_RULES_FIELDS, "selection_rules")
    accepted = _closed_plan(obj["accepted_h"], _PLAN_ACCEPTED_H_FIELDS, "accepted_h")
    for key in (
        "run_identity",
        "census_sha256",
        "predictions_sha256",
        "complete_sha256",
        "candidate_plan_sha256",
        "implementation_bundle_sha256",
    ):
        _require(
            type(accepted[key]) is str and len(accepted[key]) == 64,
            f"candidate plan: accepted_h.{key} must be a 64-hex digest",
        )

    authorities = obj["authorities"]
    _require(type(authorities) is dict, "candidate plan: authorities must be an object")
    _require(
        frozenset(authorities) == REQUIRED_AUTHORITIES,
        "candidate plan authority set is not the exact required set; "
        f"missing={sorted(REQUIRED_AUTHORITIES - frozenset(authorities))} "
        f"unexpected={sorted(frozenset(authorities) - REQUIRED_AUTHORITIES)}",
    )
    for name, entry in sorted(authorities.items()):
        spec = _closed_plan(entry, _PLAN_AUTHORITY_FIELDS, f"authorities.{name}")
        _require(
            type(spec["path"]) is str and spec["path"],
            f"candidate plan: authorities.{name}.path must be a string",
        )
        _require(
            type(spec["sha256"]) is str and len(spec["sha256"]) == 64,
            f"candidate plan: authorities.{name}.sha256 must be a 64-hex digest",
        )

    bindings = obj["input_bindings"]
    _require(
        type(bindings) is dict and bindings, "candidate plan: input_bindings must be an object"
    )
    for binding_id, entry in sorted(bindings.items()):
        _closed_plan(entry, _PLAN_BINDING_FIELDS, f"input_bindings.{binding_id}")

    nodes = obj["nodes"]
    _require(type(nodes) is list and nodes, "candidate plan: nodes must be a non-empty list")
    for index, entry in enumerate(nodes):
        _closed_plan(entry, _PLAN_NODE_FIELDS, f"nodes[{index}]")

    projection = obj["node_binding_projection"]
    _require(
        type(projection) is dict and projection,
        "candidate plan: node_binding_projection must be a non-empty object",
    )
    for source_id, allowed in sorted(projection.items()):
        _require(
            type(allowed) is list and allowed and allowed == sorted(set(allowed)),
            f"candidate plan: node_binding_projection[{source_id!r}] must be a sorted unique "
            "non-empty list",
        )
    _require(
        set(projection) == {entry["source_id"] for entry in nodes},
        "candidate plan: node_binding_projection must cover exactly the declared nodes",
    )
    for entry in nodes:
        _require(
            projection[entry["source_id"]] == sorted(set(entry["input_binding_ids"])),
            f"candidate plan: node_binding_projection[{entry['source_id']!r}] disagrees with that "
            "node's declared input_binding_ids",
        )

    _closed_plan(obj["environment_contract"], _PLAN_ENVIRONMENT_FIELDS, "environment_contract")
    _require(
        type(obj["node_order"]) is list and obj["node_order"],
        "candidate plan: node_order must be a non-empty list",
    )
    _require(
        type(obj["implementation_files"]) is dict and obj["implementation_files"],
        "candidate plan: implementation_files must be a non-empty object",
    )
    _require(
        type(obj["bound_authorities"]) is dict and obj["bound_authorities"],
        "candidate plan: bound_authorities must be a non-empty object",
    )
    return obj


def _resolve_repo_path(repo_root: Path, relative: str) -> Path:
    """Resolve a plan-declared path strictly inside the repository."""
    candidate = (repo_root / relative).resolve()
    root = repo_root.resolve()
    _require(
        candidate == root or root in candidate.parents,
        f"plan path {relative!r} escapes the repository root",
    )
    return candidate


def authorize_plan(
    plan_path: Path,
    expected_plan_sha256: str,
    repo_root: Path,
    *,
    require_executable: bool = True,
) -> AuthorizedIContext:
    """Authorization is a capability, not a comparison.

    The owner supplies the expected digest out of band; it is checked against the plan bytes, and
    those same bytes then drive every load. A plan cannot authorise itself, an authorization for
    one plan cannot be reused for another, and nothing load-bearing is taken from a CLI argument.
    """
    payload, digest = read_authoritative_bytes(plan_path, max_bytes=1 << 28)
    _require(
        digest == expected_plan_sha256,
        f"candidate plan SHA-256 {digest} is not the owner-supplied {expected_plan_sha256}",
    )
    plan = validate_plan_schema(strict_json_object(payload, where=str(plan_path)))

    # --- this implementation ------------------------------------------------------------
    files = implementation_files(repo_root)
    bundle = implementation_bundle_sha256(files)
    _require(
        plan.get("implementation_bundle_sha256") == bundle,
        "candidate plan was generated against a different Stage-I implementation bundle",
    )
    _require(
        dict(plan.get("implementation_files", {})) == dict(sorted(files.items())),
        "candidate plan implementation file digests disagree with this checkout",
    )
    runtime_commit = _git_head(repo_root)
    _require(
        plan.get("implementation_commit") == runtime_commit,
        f"runtime repository HEAD {runtime_commit} is not the plan's implementation commit "
        f"{plan.get('implementation_commit')}",
    )
    _require(
        plan.get("output_schema_version") == RECORD_SCHEMA
        and plan.get("manifest_schema_version") == MANIFEST_SCHEMA,
        "candidate plan schema versions disagree with this implementation",
    )
    shard_policy = plan.get("shard_policy") or {}
    _require(
        shard_policy.get("version") == SHARD_POLICY_VERSION
        and shard_policy.get("records_per_shard") == RECORDS_PER_SHARD,
        "candidate plan shard policy disagrees with this implementation",
    )

    # --- environment --------------------------------------------------------------------
    contract = plan.get("environment_contract") or {}
    environment = current_environment()
    verify_environment(environment, require_executable=require_executable)
    _require(
        contract.get("python_version") == REQUIRED_PYTHON_VERSION
        and contract.get("tokenizers_version") == REQUIRED_TOKENIZERS_VERSION
        and contract.get("python_executable") == REQUIRED_PYTHON_EXECUTABLE,
        "candidate plan environment contract disagrees with this implementation",
    )

    # --- the graph named by the PLAN, not by a CLI argument -----------------------------
    graph_path = _resolve_repo_path(repo_root, plan["graph_path"])
    graph = load_source_graph(graph_path, verify_hashes=True)
    _require(
        graph.graph_sha256 == plan["graph_sha256"],
        f"owner graph SHA-256 {graph.graph_sha256} is not the plan's {plan['graph_sha256']}",
    )
    _require(graph.seed == plan["seed"], "owner graph seed disagrees with the plan")
    _require(
        dict(graph.bound_authorities) == dict(plan["bound_authorities"]),
        "owner graph bound authorities disagree with the plan",
    )

    # --- every authority, re-hashed from disk -------------------------------------------
    for name, entry in sorted(plan["authorities"].items()):
        path = _resolve_repo_path(repo_root, entry["path"])
        _require(path.is_file(), f"plan authority {name} is missing at {entry['path']}")
        actual = file_sha256(path)
        _require(
            actual == entry["sha256"],
            f"plan authority {name} is {actual} on disk but the plan binds {entry['sha256']}",
        )

    # --- every input binding, against the graph and against disk ------------------------
    _require(
        set(plan["input_bindings"]) == set(graph.bindings),
        "plan input bindings do not cover exactly the graph's bindings",
    )
    for binding_id, bound in sorted(plan["input_bindings"].items()):
        binding = graph.bindings[binding_id]
        for field_name, actual in (
            ("documents_sha256", binding.documents_sha256),
            ("documents_size_bytes", binding.documents_size_bytes),
            ("eligibility_index_sha256", binding.eligibility_index_sha256),
            ("release_manifest_sha256", binding.release_manifest_sha256),
            ("total_physical_rows", binding.total_physical_rows),
            ("expected_eligible_rows", binding.expected_eligible_rows),
        ):
            _require(
                bound[field_name] == actual,
                f"plan binding {binding_id}.{field_name} disagrees with the frozen graph",
            )
        verify_binding_inputs(binding)

    # --- every node, target and the execution order -------------------------------------
    _require(
        plan["node_order"] == [n.source_id for n in graph.nodes],
        "plan node order disagrees with the frozen graph execution order",
    )
    _require(
        len(plan["nodes"]) == len(graph.nodes),
        "plan node count disagrees with the frozen graph",
    )
    for bound, node in zip(plan["nodes"], graph.nodes, strict=True):
        _require(
            bound["source_id"] == node.source_id
            and bound["stage"] == node.stage
            and bound["stage_priority"] == node.stage_priority
            and bound["target_serialized_tokens"] == node.target_serialized_tokens
            and bound["selection_mode"] == node.selection_mode
            and list(bound["input_binding_ids"]) == list(node.input_binding_ids),
            f"plan node {bound['source_id']} disagrees with the frozen graph",
        )

    # --- the accepted Stage-H result named by the PLAN ----------------------------------
    accepted_spec = plan["accepted_h"]
    h_run_dir = _resolve_repo_path(repo_root, accepted_spec["run_dir"])
    accepted = load_accepted_h(
        h_run_dir,
        expected_run_identity=accepted_spec["run_identity"],
        expected_predictions_sha256=accepted_spec["predictions_sha256"],
        expected_complete_sha256=accepted_spec["complete_sha256"],
        expected_census_sha256=accepted_spec["census_sha256"],
    )
    _require(
        accepted.census["authorization"]["candidate_plan_sha256"]
        == accepted_spec["candidate_plan_sha256"],
        "accepted Stage-H candidate plan SHA disagrees with the Stage-I plan",
    )
    _require(
        accepted.census["authorization"]["implementation_bundle_sha256"]
        == accepted_spec["implementation_bundle_sha256"],
        "accepted Stage-H implementation bundle disagrees with the Stage-I plan",
    )
    _require(
        accepted.census["graph_sha256"] == plan["graph_sha256"],
        "accepted Stage-H ran against a different owner graph than the Stage-I plan binds",
    )

    projection = {
        source_id: tuple(sorted(set(allowed)))
        for source_id, allowed in plan["node_binding_projection"].items()
    }
    _require(
        {source_id: list(allowed) for source_id, allowed in sorted(projection.items())}
        == {node.source_id: sorted(set(node.input_binding_ids)) for node in graph.nodes},
        "plan node/binding projection disagrees with the frozen owner graph",
    )

    identity = stage_i_run_identity(
        candidate_plan_sha256=digest,
        implementation_commit=runtime_commit,
        implementation_bundle_sha256=bundle,
        plan_schema_version=PLAN_SCHEMA,
        output_schema_version=RECORD_SCHEMA,
        manifest_schema_version=MANIFEST_SCHEMA,
        shard_policy_version=SHARD_POLICY_VERSION,
        records_per_shard=RECORDS_PER_SHARD,
        h_run_identity=accepted.run_identity,
        h_complete_sha256=accepted.complete_sha256,
        h_census_sha256=accepted.census_sha256,
        h_predictions_sha256=accepted.predictions_sha256,
        owner_graph_sha256=graph.graph_sha256,
        node_binding_projection_sha256=node_binding_projection_sha256(projection),
    )

    context = AuthorizedIContext(
        repo_root=repo_root,
        plan_path=plan_path,
        plan_sha256=digest,
        plan=plan,
        graph=graph,
        graph_path=graph_path,
        accepted=accepted,
        environment=environment,
        bundle_sha256=bundle,
        bundle_files=dict(sorted(files.items())),
        implementation_commit=runtime_commit,
        tokenizer_path=_resolve_repo_path(repo_root, plan["authorities"]["tokenizer"]["path"]),
        reference_exclusion_path=_resolve_repo_path(
            repo_root, plan["authorities"]["reference_exclusion"]["path"]
        ),
        node_binding_projection=projection,
        run_identity=identity,
    )
    _register_authorized(context, _canonical_state_digest(context))
    return context


def _derive_canonical_state(repo_root: Path, plan_path: Path, expected_sha256: str) -> str:
    """Re-authorize from the plan bytes and digest the result, without minting a new authority.

    ``revalidate`` compares this against the digest recorded at authorization, so a change to the
    plan or to any bound artifact on disk is detected. It deliberately reuses the full
    authorization path rather than a cheaper subset: a partial re-check is exactly how R1's
    revalidate missed graph and H substitution.
    """
    fresh = authorize_plan(plan_path, expected_sha256, repo_root, require_executable=False)
    digest = _canonical_state_digest(fresh)
    _AUTHORIZED.pop(id(fresh), None)
    return digest


def realize_and_publish(
    context: AuthorizedIContext,
    *,
    out_dir: Path,
    work_dir: Path,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
    sort_chunk_lines: int = DEFAULT_SORT_CHUNK_LINES,
) -> tuple[Path, dict[str, Any]]:
    """The authoritative Stage-I run: derive, prove equal to H, then and only then materialize.

    Every load-bearing input comes from ``context``, which only ``authorize_plan`` can produce. No
    graph, accepted-H run, authority, binding, node set or shard policy can be substituted by a
    caller: the CLI supplies a plan digest and an output directory and nothing else that decides
    what gets built.

    The ordering is the contract. The environment, the plan and every input are validated before a
    byte of corpus is read; the whole selection is derived and proved equal to the accepted Stage-H
    prediction before a byte of output is written; the staged bytes are then audited and reconciled
    before COMPLETE exists. A disagreement at any gate raises with nothing published.
    """
    _require_authorized(context, "realization")
    context.revalidate()

    graph = context.graph
    accepted = context.accepted
    tokenizer_path = str(context.tokenizer_path)
    reference_exclusion = load_reference_exclusion(
        context.reference_exclusion_path,
        expected_sha256=graph.bound_authorities["g2_exclusion_manifest_sha256"],
    )

    selections, counters = derive_selection(graph, tokenizer_path, reference_exclusion)
    comparison = compare_with_h(selections, accepted)
    require_h_i_equality(comparison)

    manifest = build_manifest(selections, context)
    records = iter_records_in_physical_order(
        graph, tokenizer_path, selections, work_dir, read_window_bytes=read_window_bytes
    )
    # Revalidate one last time: everything the manifest attests to has now been derived, and
    # nothing has been written, so a late change to the plan or the implementation still costs
    # nothing to reject.
    context.revalidate()
    final = publish_atomic(
        out_dir,
        context.run_name,
        manifest,
        records,
        read_window_bytes=read_window_bytes,
        sort_chunk_lines=sort_chunk_lines,
    )

    return final, {
        "run_identity": context.run_identity,
        "comparison": comparison,
        "binding_counters": counters,
        "selections": [s.comparable() for s in selections],
    }


# --------------------------------------------------------------------- CLI


def _write_durable(path: Path, payload: bytes) -> None:
    import os

    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Checked before argparse so a resume attempt fails on its own terms rather than as a generic
    # usage error, in every spelling. Stage I v1 either completes or publishes nothing.
    for token in raw_argv:
        if token == "--resume" or token.startswith("--resume=") or token.startswith("--resume-"):
            raise RealizationError("resume is not supported")

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="generate an unauthorized candidate Stage-I plan")
    plan_parser.add_argument("--graph", type=Path, required=True)
    plan_parser.add_argument("--h-run-dir", type=Path, default=Path(ACCEPTED_H_RUN_DIR))
    plan_parser.add_argument("--out", type=Path, required=True)
    plan_parser.add_argument("--repo-root", type=Path, default=Path(ROOT))
    plan_parser.add_argument("--implementation-commit", type=str, default=None)

    env_parser = sub.add_parser("verify-environment", help="check the frozen interpreter contract")
    env_parser.add_argument("--require-executable", action="store_true")

    # The authoritative inputs are the plan, its owner-supplied digest, and where to write. The
    # graph, the accepted Stage-H run, every authority, every binding, the node set, the shard
    # policy and the run name all come from the authorized plan; there is deliberately no flag
    # that could substitute any of them.
    run_parser = sub.add_parser("run", help="run the authorized Stage-I realization and publish it")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--expected-plan-sha256", type=str, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--repo-root", type=Path, default=Path(ROOT))
    run_parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="scratch space for spools and audit sorts; defaults to a temp dir under --out-dir",
    )

    args = parser.parse_args(raw_argv)

    if args.command == "verify-environment":
        environment = current_environment()
        verify_environment(environment, require_executable=args.require_executable)
        print(canonical_json_bytes(environment.as_canonical()).decode("utf-8"), end="")
        return 0

    if args.command == "run":
        repo_root = args.repo_root.resolve()
        context = authorize_plan(args.plan, args.expected_plan_sha256, repo_root)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        owned = args.work_dir is None
        work_dir = (
            Path(tempfile.mkdtemp(prefix=f".{context.run_name}.work-", dir=str(args.out_dir)))
            if owned
            else args.work_dir
        )
        try:
            final, _summary = realize_and_publish(context, out_dir=args.out_dir, work_dir=work_dir)
        finally:
            if owned:
                shutil.rmtree(work_dir, ignore_errors=True)
        print(f"published {final}")
        return 0

    repo_root = args.repo_root.resolve()
    graph = load_source_graph(args.graph, verify_hashes=True)
    accepted = load_accepted_h(args.h_run_dir)
    plan = generate_candidate_plan(
        repo_root=repo_root,
        graph_path=args.graph,
        graph=graph,
        accepted=accepted,
        environment=current_environment(),
        implementation_commit=args.implementation_commit or _git_head(repo_root),
    )
    payload = canonical_json_bytes(plan)
    _require(
        not args.out.exists(), f"candidate plan already exists, refusing to overwrite: {args.out}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_durable(args.out, payload)
    print(f"{hashlib.sha256(payload).hexdigest()}  {args.out}")
    return 0


__all__ = [
    "ACCEPTED_H_COMPLETE_SHA256",
    "ACCEPTED_H_PREDICTIONS_SHA256",
    "ACCEPTED_H_RUN_DIR",
    "ACCEPTED_H_RUN_IDENTITY",
    "BUNDLE_SCHEMA",
    "H_I_REQUIRED_DIMENSIONS",
    "IMPLEMENTATION_BUNDLE_FILES",
    "PLAN_SCHEMA",
    "REQUIRED_PYTHON_EXECUTABLE",
    "REQUIRED_PYTHON_VERSION",
    "REQUIRED_TOKENIZERS_VERSION",
    "AcceptedH",
    "AuthorizedIContext",
    "authorize_plan",
    "Environment",
    "RealizationError",
    "SelectedTarget",
    "build_manifest",
    "build_materialization_targets",
    "compare_node",
    "compare_with_h",
    "current_environment",
    "derive_selection",
    "generate_candidate_plan",
    "implementation_bundle_sha256",
    "implementation_files",
    "iter_records_in_physical_order",
    "load_accepted_h",
    "load_reference_exclusion",
    "materialize_binding",
    "realize_and_publish",
    "SELECTION_SEQUENCE_SCHEMA",
    "stage_i_run_identity",
    "validate_plan_schema",
    "require_h_i_equality",
    "scan_binding_candidates",
    "verify_environment",
    "verify_h_evidence_manifest",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RealizationError,
        SelectionError,
        OutputError,
        AuditError,
        GraphError,
        StrictJSONError,
    ) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
