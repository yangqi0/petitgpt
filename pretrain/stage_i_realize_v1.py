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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
from types import MappingProxyType
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
    BOUND_AUTHORITY_KEYS,
    BRANCH_DEPENDENT,
    SELECTION_MODES,
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
    NODE_BINDING_PROJECTION_SCHEMA,
    PASS1_RESULT_SCHEMA,
    RECORD_SCHEMA,
    RECORDS_PER_SHARD,
    SELECTION_SEQUENCE_MAP_SCHEMA,
    SHARD_POLICY_VERSION,
    STAGE_I_RUN_IDENTITY_SCHEMA,
    OutputError,
    TrustedExpectedResult,
    build_record,
    load_published_realization,
    load_trusted_expected_result,
    node_binding_projection_sha256,
    publish_atomic,
    selection_sequence_commitment_map_sha256,
    trusted_expected_result,
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

PLAN_SCHEMA = "petitgpt-i-candidate-plan-v3"

# Closed plan schema. R1 validated only the fields it happened to read, so a rehashed plan could
# drop `hq_policy`, drop `selection_rules`, or carry unknown keys and still authorize. R2 closed
# the key sets but not the values inside them, so a structured frozen rule could be replaced by a
# bare integer, an unknown field could hide one level down inside `observed_at_generation`, and an
# equivalent-but-noncanonical authority path resolved to the right bytes and authorized.
#
# v3 closes the values as well as the keys: exact scalar types at every level (`type(x) is str`,
# never `isinstance`, so bool/int and str subclasses cannot blur a literal), exact frozen literals
# for the selection and shard rules, closed nested objects including `observed_at_generation`, and
# an exact canonical name -> path -> digest mapping for the required authority set.
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
# R3-E: `observed_at_generation` was an open object in v2, so an unknown field one level down
# authorized. It is exactly the three environment fields and nothing else.
_PLAN_OBSERVED_FIELDS = frozenset({
    "python_executable",
    "python_version",
    "tokenizers_version",
})

# The exact frozen shard rule text, promoted to a constant so the plan validator can require the
# literal rather than merely require the key to be present.
SHARD_POLICY_RULE = (
    "whole records only; fixed record count per shard over the canonical physical "
    "order (stage_priority, source_id, input_binding_id, stable_input_record_ordinal)"
)

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

# R3-D: one closed canonical projection of everything an authorized run depends on. v1 named
# fields reactively as gaps were found; v2 is generated from the exact authorized plan bytes and
# the exact plan-bound artifacts, and covers every field consumed by Pass-1 selection, the H/I
# gate, locator/binding resolution, materialization, manifest construction and publication.
CANONICAL_STATE_SCHEMA = "petitgpt-stage-i-canonical-state-v2"

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


_H_I_GATE_KEYS = (
    "ALL_H_I_BRANCHES_MATCH",
    "ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH",
    "ALL_H_I_FINGERPRINTS_MATCH",
    "ALL_H_I_CROSSING_IDENTITIES_MATCH",
    "ALL_H_I_OVERSHOOTS_MATCH",
    "OWNERSHIP_MATRIX_MATCH",
    "ALL_NODES_MATCH",
)


def _node_projection(selection: NodeSelection) -> dict[str, Any]:
    """One node's committed outcome, in the exact shape both Layer 2 and Layer 3 restate."""
    return {
        "source_id": selection.source_id,
        "stage": selection.stage,
        "target_serialized_tokens": selection.target_serialized_tokens,
        "branch": selection.branch,
        "selection_mode": selection.selection_mode,
        "selected_identities": selection.selected_identities,
        "selected_serialized_tokens": selection.selected_serialized_tokens,
        "selection_fingerprint": selection.selection_fingerprint,
        "selection_sequence_commitment": selection_sequence_commitment(
            source_id=selection.source_id,
            stage=selection.stage,
            pairs=((d.selection_ordinal_within_node, d.cleaned_sha256) for d in selection.selected),
        ),
        "crossing_identity": selection.crossing_identity,
        "actual_overshoot_tokens": selection.actual_overshoot_tokens,
        "input_binding_ids": sorted({d.input_binding_id for d in selection.selected}),
    }


def _h_binding_block(state: CanonicalAuthorizedState) -> dict[str, Any]:
    accepted = state.accepted
    return {
        "h_run_identity": accepted.run_identity,
        "h_census_sha256": accepted.census_sha256,
        "h_predictions_sha256": accepted.predictions_sha256,
        "h_complete_sha256": accepted.complete_sha256,
        "h_candidate_plan_sha256": accepted.census["authorization"]["candidate_plan_sha256"],
        "h_implementation_bundle_sha256": accepted.census["authorization"][
            "implementation_bundle_sha256"
        ],
        "owner_graph_sha256": accepted.census["graph_sha256"],
    }


def build_pass1_result(
    selections: list[NodeSelection],
    state: CanonicalAuthorizedState,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """R3-A: Layer 2. The expected result, stated after Pass 1 and before anything is written.

    This is the artifact the whole repair turns on. It is produced from the Pass-1 selection
    ledger and the H/I gate verdict, it carries the Layer-1 authorization anchors including the
    complete canonical authorized-state digest, and it is frozen to disk -- outside the
    realization -- before the first output byte exists. Its digest then names the published run.

    A fully resealed physical result can restate the manifest, the shards and the COMPLETE marker
    all it likes. It cannot restate this, because this predates it and lives outside it; and it
    cannot keep the run's name while disagreeing with it, because the name is derived from it.
    """
    nodes = [_node_projection(selection) for selection in selections]
    commitments = {entry["source_id"]: entry["selection_sequence_commitment"] for entry in nodes}
    return {
        "schema_version": PASS1_RESULT_SCHEMA,
        "authorization": authorization_block(state),
        "selection_sequence_commitment_version": SELECTION_SEQUENCE_SCHEMA,
        "selection_sequence_commitments": commitments,
        "selection_sequence_commitment_map_sha256": selection_sequence_commitment_map_sha256(
            commitments
        ),
        # Trusted node -> allowed-binding authority, copied from the authorized Layer-1 state.
        # The manifest will record its own copy, but only as an observation to be checked.
        "node_binding_projection": {
            source_id: list(allowed)
            for source_id, allowed in sorted(state.node_binding_projection.items())
        },
        "authorized_input_binding_ids": list(state.authorized_input_binding_ids),
        "nodes": nodes,
        "totals": {
            "records": sum(s.selected_identities for s in selections),
            "content_tokens": sum(d.content_token_count for s in selections for d in s.selected),
            "serialized_tokens": sum(s.selected_serialized_tokens for s in selections),
            "unique_cleaned_identities": sum(s.selected_identities for s in selections),
        },
        "ownership_matrix": ownership_matrix_v1(selections),
        "h_i_gate": {key: bool(comparison[key]) for key in _H_I_GATE_KEYS},
        "h_binding": _h_binding_block(state),
    }


def build_manifest(
    selections: list[NodeSelection],
    state: CanonicalAuthorizedState,
    expected: TrustedExpectedResult,
) -> dict[str, Any]:
    """The manifest minus its shard list, which only the publisher can fill in.

    Everything load-bearing comes from the freshly re-derived authorized state, never from an
    object a caller could have substituted. The per-node numbers here are recomputed from the
    selection ledger rather than copied out of ``expected``, so the publisher's three-way
    reconciliation compares three independently produced answers instead of two copies and a
    derivation.

    The totals are what the selection ledger believes. They are NOT taken on trust: the publisher
    audits the staged records and refuses to publish unless every number below equals one derived
    from the bytes on disk AND one committed to before those bytes existed.
    """
    graph = state.graph
    nodes = [_node_projection(selection) for selection in selections]
    total_records = sum(s.selected_identities for s in selections)
    authorization = authorization_block(state)
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
        "nodes": nodes,
        "node_binding_projection": {
            source_id: list(allowed)
            for source_id, allowed in sorted(state.node_binding_projection.items())
        },
        "ownership_matrix": ownership_matrix_v1(selections),
        "bindings": {
            binding_id: graph.bindings[binding_id].documents_sha256
            for binding_id in sorted(graph.bindings)
        },
        "environment": state.environment.as_canonical(),
        "stage_i_run": {
            **authorization,
            "run_identity": expected.stage_i_run_identity,
            "post_pass1_result_identity_schema": expected.result_identity_schema,
            "post_pass1_result_identity_sha256": expected.result_identity_sha256,
            "selection_sequence_commitment_version": (
                expected.selection_sequence_commitment_version
            ),
            "selection_sequence_commitment_map_sha256": (
                expected.selection_sequence_commitment_map_sha256
            ),
        },
        "h_binding": _h_binding_block(state),
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
            "rule": SHARD_POLICY_RULE,
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


def authorization_block(state: CanonicalAuthorizedState) -> dict[str, Any]:
    """The Layer-1 anchors, restated identically by the post-Pass-1 result and the manifest.

    Written out field by field rather than hashing an open-ended dictionary: a dict digest would
    silently change meaning the day someone adds a key, and would let two runs that differ in an
    unlisted field claim the same identity.
    """
    return {
        "candidate_i_plan_sha256": state.plan_sha256,
        "authorized_state_sha256": state.state_sha256,
        "implementation_commit": state.implementation_commit,
        "implementation_bundle_sha256": state.bundle_sha256,
        "plan_schema_version": PLAN_SCHEMA,
        "output_schema_version": RECORD_SCHEMA,
        "manifest_schema_version": MANIFEST_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "h_run_identity": state.accepted.run_identity,
        "h_complete_sha256": state.accepted.complete_sha256,
        "h_census_sha256": state.accepted.census_sha256,
        "h_predictions_sha256": state.accepted.predictions_sha256,
        "owner_graph_sha256": state.graph.graph_sha256,
        "node_binding_projection_sha256": node_binding_projection_sha256(
            state.node_binding_projection
        ),
    }


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze parsed JSON: dicts become read-only, lists become tuples.

    R2 kept the graph and the accepted-H census as ordinary nested dicts hanging off the context,
    so ``census["nodes"][0]["selected_serialized_tokens"] += 1`` was an ordinary assignment. There
    is no nested mutation surface left to reach for.
    """
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if type(value) is MappingProxyType:
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if type(value) in (list, tuple):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _frozen_graph(graph: SourceGraph) -> SourceGraph:
    """The same graph with its remaining mutable containers replaced by read-only ones."""
    import dataclasses

    return dataclasses.replace(
        graph,
        bindings=MappingProxyType(dict(graph.bindings)),
        bound_authorities=MappingProxyType(dict(graph.bound_authorities)),
        raw=_deep_freeze(dict(graph.raw)),
        eligibility_rows=(
            None
            if graph.eligibility_rows is None
            else MappingProxyType(dict(graph.eligibility_rows))
        ),
        binding_identities=(
            None
            if graph.binding_identities is None
            else MappingProxyType(dict(graph.binding_identities))
        ),
    )


def _frozen_accepted(accepted: AcceptedH) -> AcceptedH:
    import dataclasses

    return dataclasses.replace(
        accepted,
        census=_deep_freeze(dict(accepted.census)),
        predictions=_deep_freeze(dict(accepted.predictions)),
    )


@dataclass(frozen=True)
class CanonicalAuthorizedState:
    """Everything an authorized Stage-I run is allowed to depend on, derived fresh from disk.

    This replaces the mutable ``context.graph`` / ``context.accepted`` / ``context.plan``
    attributes R2 exposed. Runtime code never reads load-bearing truth off a long-lived object a
    caller can reach: it asks the authorization registry to re-derive this state from the exact
    authorization-time plan bytes, proves the re-derivation digests to what was authorized, and
    then uses the freshly derived, deeply immutable values.
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
    h_run_dir: Path
    tokenizer_path: Path
    reference_exclusion_path: Path
    authority_paths: Mapping[str, str]
    authority_sha256: Mapping[str, str]
    node_binding_projection: Mapping[str, tuple[str, ...]]
    authorized_input_binding_ids: tuple[str, ...]
    canonical_payload: Mapping[str, Any]
    state_sha256: str


def _authorized_state_payload(
    *,
    plan_sha256: str,
    plan: Mapping[str, Any],
    graph: SourceGraph,
    graph_path: str,
    accepted: AcceptedH,
    h_run_dir: str,
    environment: Environment,
    bundle_sha256: str,
    bundle_files: Mapping[str, str],
    implementation_commit: str,
    authority_paths: Mapping[str, str],
    authority_sha256: Mapping[str, str],
    tokenizer_path: str,
    reference_exclusion_path: str,
    node_binding_projection: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """The ONE closed canonical projection. Complete by construction, not by accretion.

    R2's projection named fields one at a time as reviewers found gaps, so a nested graph branch
    threshold, an H census value, an H prediction and every path string were all invisible to it.
    This projection is generated from the exact authorized plan bytes plus the exact plan-bound
    artifacts and includes, in full:

    * the plan -- committed by its digest, which covers every byte of it, plus the derived
      projections the runtime actually consumes;
    * the owner graph -- its digest, its seed, its complete raw document, and the complete derived
      node and binding structures including branch policy, thresholds, predicates, ranking
      semantics, cleaning contracts, schema accessors and every resource path;
    * the accepted Stage-H state -- run identity, the three bound digests, the complete canonical
      census and predictions documents, and the H/I gate values named explicitly;
    * every canonical path string, so an equivalent-but-different location is a different state;
    * the environment, the implementation bundle, and every schema/policy literal this
      implementation will stamp into the published result.

    A change to any of them is a different authorized state, and the runtime refuses to proceed.
    """
    return {
        "schema_version": CANONICAL_STATE_SCHEMA,
        # --- plan -------------------------------------------------------------------------
        "plan": {
            "sha256": plan_sha256,
            "schema_version": PLAN_SCHEMA,
            "canonical_sha256": hashlib.sha256(canonical_json_bytes(dict(plan))).hexdigest(),
            "authorization_status": plan["authorization_status"],
            "realization_label": plan["realization_label"],
            "resume_supported": plan["resume_supported"],
            "seed": plan["seed"],
            "graph_path": plan["graph_path"],
            "graph_sha256": plan["graph_sha256"],
            "node_order": list(plan["node_order"]),
            "shard_policy": dict(plan["shard_policy"]),
            "selection_rules": dict(plan["selection_rules"]),
            "environment_contract": {
                "python_executable": plan["environment_contract"]["python_executable"],
                "python_version": plan["environment_contract"]["python_version"],
                "tokenizers_version": plan["environment_contract"]["tokenizers_version"],
                "observed_at_generation": dict(
                    plan["environment_contract"]["observed_at_generation"]
                ),
            },
            "accepted_h": dict(plan["accepted_h"]),
            "authorities": {
                name: dict(entry) for name, entry in sorted(plan["authorities"].items())
            },
            "bound_authorities": dict(sorted(plan["bound_authorities"].items())),
            "input_bindings": {
                binding_id: dict(entry)
                for binding_id, entry in sorted(plan["input_bindings"].items())
            },
            "nodes": [dict(entry) for entry in plan["nodes"]],
            "node_binding_projection": {
                source_id: list(allowed)
                for source_id, allowed in sorted(plan["node_binding_projection"].items())
            },
            "implementation_commit": plan["implementation_commit"],
            "implementation_bundle_sha256": plan["implementation_bundle_sha256"],
            "implementation_files": dict(sorted(plan["implementation_files"].items())),
        },
        # --- owner graph, complete ---------------------------------------------------------
        "graph": {
            "path": graph_path,
            "sha256": graph.graph_sha256,
            "seed": graph.seed,
            "raw_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(_thaw(graph.raw))
            ).hexdigest(),
            "raw": _thaw(graph.raw),
            "bound_authorities": dict(sorted(graph.bound_authorities.items())),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "source_id": node.source_id,
                    "stage": node.stage,
                    "stage_priority": node.stage_priority,
                    "target_serialized_tokens": node.target_serialized_tokens,
                    "input_binding_ids": list(node.input_binding_ids),
                    "selection_mode": node.selection_mode,
                    "candidate_predicate": _thaw(node.candidate_predicate),
                    "branch_primary": _thaw(node.branch_primary),
                    "branch_fallback": _thaw(node.branch_fallback),
                }
                for node in graph.nodes
            ],
            "bindings": {
                binding_id: {
                    "input_binding_id": binding.input_binding_id,
                    "release_key": binding.release_key,
                    "documents_path": str(binding.documents_path),
                    "documents_sha256": binding.documents_sha256,
                    "documents_size_bytes": binding.documents_size_bytes,
                    "release_manifest_path": str(binding.release_manifest_path),
                    "release_manifest_sha256": binding.release_manifest_sha256,
                    "eligibility_index_path": str(binding.eligibility_index_path),
                    "eligibility_index_sha256": binding.eligibility_index_sha256,
                    "total_physical_rows": binding.total_physical_rows,
                    "excluded_rows": binding.excluded_rows,
                    "expected_eligible_rows": binding.expected_eligible_rows,
                    "schema_accessor": _thaw(binding.schema_accessor),
                    "text_field": binding.text_field,
                    "cleaning_contract": _thaw(binding.cleaning_contract),
                }
                for binding_id, binding in sorted(graph.bindings.items())
            },
        },
        # --- accepted Stage-H state, complete ----------------------------------------------
        "accepted_h": {
            "run_dir": h_run_dir,
            "run_identity": accepted.run_identity,
            "census_sha256": accepted.census_sha256,
            "predictions_sha256": accepted.predictions_sha256,
            "complete_sha256": accepted.complete_sha256,
            "census_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(_thaw(accepted.census))
            ).hexdigest(),
            "predictions_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(_thaw(accepted.predictions))
            ).hexdigest(),
            "census": _thaw(accepted.census),
            "predictions": _thaw(accepted.predictions),
            "gate_projection": [
                {
                    "source_id": node["source_id"],
                    "stage": node["stage"],
                    "target_serialized_tokens": node["target_serialized_tokens"],
                    "branch": node["branch"],
                    "selection_mode": node["selection_mode"],
                    "selected_identities": node["selected_identities"],
                    "selected_serialized_tokens": node["selected_serialized_tokens"],
                    "selection_fingerprint": node["selection_fingerprint"],
                    "crossing_identity": node["crossing_identity"],
                    "crossing_document_serialized_tokens": node[
                        "crossing_document_serialized_tokens"
                    ],
                    "actual_overshoot_tokens": node["actual_overshoot_tokens"],
                    "residual_identities": node["residual_identities"],
                    "residual_serialized_tokens": node["residual_serialized_tokens"],
                    "exclusions_by_owner": _thaw(node["exclusions_by_owner"]),
                    "boundary_evidence": _thaw(node["boundary_evidence"]),
                    "feasible": node["feasible"],
                }
                for node in accepted.census["nodes"]
            ],
            "ownership_matrix": _thaw(accepted.census["ownership_matrix"]),
            "totals": _thaw(accepted.census["totals"]),
            "graph_sha256": accepted.census["graph_sha256"],
            "candidate_plan_sha256": accepted.census["authorization"]["candidate_plan_sha256"],
            "implementation_bundle_sha256": accepted.census["authorization"][
                "implementation_bundle_sha256"
            ],
        },
        # --- canonical paths and references ------------------------------------------------
        "paths": {
            "tokenizer": tokenizer_path,
            "reference_exclusion": reference_exclusion_path,
            "graph": graph_path,
            "accepted_h_run_dir": h_run_dir,
            "authorities": dict(sorted(authority_paths.items())),
            "input_documents": {
                binding_id: str(binding.documents_path)
                for binding_id, binding in sorted(graph.bindings.items())
            },
            "input_release_manifests": {
                binding_id: str(binding.release_manifest_path)
                for binding_id, binding in sorted(graph.bindings.items())
            },
            "input_eligibility_indexes": {
                binding_id: str(binding.eligibility_index_path)
                for binding_id, binding in sorted(graph.bindings.items())
            },
        },
        "authority_sha256": dict(sorted(authority_sha256.items())),
        # --- this implementation and the literals it will stamp ----------------------------
        "implementation": {
            "commit": implementation_commit,
            "bundle_sha256": bundle_sha256,
            "files": dict(sorted(bundle_files.items())),
        },
        "environment": environment.as_canonical(),
        "contract_literals": {
            "plan_schema_version": PLAN_SCHEMA,
            "output_schema_version": RECORD_SCHEMA,
            "manifest_schema_version": MANIFEST_SCHEMA,
            "shard_policy_version": SHARD_POLICY_VERSION,
            "shard_policy_rule": SHARD_POLICY_RULE,
            "records_per_shard": RECORDS_PER_SHARD,
            "run_identity_schema": STAGE_I_RUN_IDENTITY_SCHEMA,
            "pass1_result_schema": PASS1_RESULT_SCHEMA,
            "sequence_commitment_schema": SELECTION_SEQUENCE_SCHEMA,
            "sequence_commitment_map_schema": SELECTION_SEQUENCE_MAP_SCHEMA,
            "node_binding_projection_schema": NODE_BINDING_PROJECTION_SCHEMA,
            "representative_rule": REPRESENTATIVE_RULE,
            "physical_locator_rule": PHYSICAL_LOCATOR_RULE,
            "realization_label": REALIZATION_LABEL,
            "resume_supported": RESUME_SUPPORTED,
            "required_authorities": sorted(REQUIRED_AUTHORITIES),
            "h_i_required_dimensions": list(H_I_REQUIRED_DIMENSIONS),
            "reference_exclusion_scope": REFERENCE_EXCLUSION_SCOPE,
            "required_python_executable": REQUIRED_PYTHON_EXECUTABLE,
            "required_python_version": REQUIRED_PYTHON_VERSION,
            "required_tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
            "accepted_h_run_dir": ACCEPTED_H_RUN_DIR,
            "accepted_h_run_identity": ACCEPTED_H_RUN_IDENTITY,
            "accepted_h_predictions_sha256": ACCEPTED_H_PREDICTIONS_SHA256,
            "accepted_h_complete_sha256": ACCEPTED_H_COMPLETE_SHA256,
        },
        "node_binding_projection": {
            source_id: list(allowed)
            for source_id, allowed in sorted(node_binding_projection.items())
        },
    }


def _thaw(value: Any) -> Any:
    """Plain-JSON view of a frozen structure, for canonical serialisation only."""
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if type(value) in (list, tuple):
        return [_thaw(child) for child in value]
    return value


# ------------------------------------------------------------------ exact plan value helpers


def _plan_str(obj: Mapping[str, Any], key: str, where: str, *, nonempty: bool = True) -> str:
    value = obj[key]
    _require(type(value) is str, f"candidate plan: {where}.{key} must be an exact string")
    _require(not nonempty or value, f"candidate plan: {where}.{key} must not be empty")
    return value


def _plan_int(obj: Mapping[str, Any], key: str, where: str, *, minimum: int | None = None) -> int:
    value = obj[key]
    _require(type(value) is int, f"candidate plan: {where}.{key} must be an exact integer")
    if minimum is not None:
        _require(value >= minimum, f"candidate plan: {where}.{key} must be >= {minimum}")
    return value


def _plan_hex64(obj: Mapping[str, Any], key: str, where: str) -> str:
    value = obj[key]
    _require(
        type(value) is str and len(value) == 64 and set(value) <= _HEX64_ALPHABET,
        f"candidate plan: {where}.{key} must be 64 lowercase hex characters",
    )
    return value


def _plan_str_list(
    obj: Mapping[str, Any], key: str, where: str, *, sorted_unique: bool = False
) -> list[str]:
    value = obj[key]
    _require(
        type(value) is list and value,
        f"candidate plan: {where}.{key} must be a non-empty list",
    )
    for item in value:
        _require(
            type(item) is str and item,
            f"candidate plan: {where}.{key}[] must contain exact non-empty strings",
        )
    _require(
        len(set(value)) == len(value), f"candidate plan: {where}.{key} must not repeat an entry"
    )
    if sorted_unique:
        _require(value == sorted(value), f"candidate plan: {where}.{key} must be sorted")
    return value


_HEX64_ALPHABET = frozenset("0123456789abcdef")


def _canonical_relative_path(value: Any, where: str) -> str:
    """A repository-relative path in exactly one spelling.

    ``runs/x/y.json`` authorises; ``./runs/x/y.json``, ``runs/x/../x/y.json``, ``/abs/runs/x/y``
    and ``runs/x//y.json`` all name the same bytes and none of them is this string. R2 resolved
    the path and compared the digest, so every one of those spellings authorised.
    """
    _require(type(value) is str and value, f"candidate plan: {where} must be a non-empty string")
    _require("\\" not in value, f"candidate plan: {where} must use forward slashes")
    pure = PurePosixPath(value)
    _require(not pure.is_absolute(), f"candidate plan: {where} must be repository-relative")
    _require(
        all(part not in ("", ".", "..") for part in pure.parts),
        f"candidate plan: {where} must not contain '.', '..' or empty path segments",
    )
    _require(
        str(pure) == value,
        f"candidate plan: {where} is not in canonical form; expected {str(pure)!r}",
    )
    return value


def _closed_plan(obj: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    _require(type(obj) is dict, f"candidate plan: {where} must be a JSON object")
    present = frozenset(obj)
    unknown = sorted(present - fields)
    missing = sorted(fields - present)
    _require(not unknown, f"candidate plan: {where} carries unknown field(s): {unknown}")
    _require(not missing, f"candidate plan: {where} is missing field(s): {missing}")
    return obj


def validate_plan_schema(plan: Any) -> dict[str, Any]:
    """Closed-schema validation of the candidate plan: exact keys AND exact values, at every level."""
    obj = _closed_plan(plan, _PLAN_TOP_FIELDS, "top level")
    _require(
        _plan_str(obj, "schema_version", "top level") == PLAN_SCHEMA,
        f"candidate plan schema {obj['schema_version']!r} is not {PLAN_SCHEMA}",
    )
    _require(
        _plan_str(obj, "authorization_status", "top level") == "NOT_AUTHORIZED",
        "candidate plan bytes must not carry an owner authorization",
    )
    _require(
        obj["resume_supported"] is False,
        "candidate plan must not enable resume",
    )
    _require(
        _plan_str(obj, "realization_label", "top level") == REALIZATION_LABEL,
        "candidate plan carries the wrong realization label",
    )
    _plan_str(obj, "authorization_note", "top level")
    _plan_str(obj, "implementation_commit", "top level")
    _canonical_relative_path(obj["graph_path"], "graph_path")
    _plan_int(obj, "seed", "top level", minimum=0)
    _plan_hex64(obj, "graph_sha256", "top level")
    _plan_hex64(obj, "implementation_bundle_sha256", "top level")
    _require(
        _plan_str(obj, "output_schema_version", "top level") == RECORD_SCHEMA
        and _plan_str(obj, "manifest_schema_version", "top level") == MANIFEST_SCHEMA,
        "candidate plan schema versions disagree with this implementation",
    )

    # --- frozen structured rules, as literals not as "some JSON value" -----------------
    shard_policy = _closed_plan(obj["shard_policy"], _PLAN_SHARD_POLICY_FIELDS, "shard_policy")
    _require(
        _plan_str(shard_policy, "version", "shard_policy") == SHARD_POLICY_VERSION,
        "candidate plan shard policy version disagrees with this implementation",
    )
    _require(
        _plan_int(shard_policy, "records_per_shard", "shard_policy", minimum=1)
        == RECORDS_PER_SHARD,
        "candidate plan shard policy record count disagrees with this implementation",
    )
    _require(
        _plan_str(shard_policy, "rule", "shard_policy") == SHARD_POLICY_RULE,
        "candidate plan shard rule is not the frozen shard rule literal",
    )
    rules = _closed_plan(obj["selection_rules"], _PLAN_SELECTION_RULES_FIELDS, "selection_rules")
    _require(
        _plan_str(rules, "representative_rule", "selection_rules") == REPRESENTATIVE_RULE,
        "candidate plan representative rule is not the frozen selector-v1 literal",
    )
    _require(
        _plan_str(rules, "physical_locator_rule", "selection_rules") == PHYSICAL_LOCATOR_RULE,
        "candidate plan physical locator rule is not the frozen literal",
    )

    # --- accepted Stage-H ---------------------------------------------------------------
    accepted = _closed_plan(obj["accepted_h"], _PLAN_ACCEPTED_H_FIELDS, "accepted_h")
    for key in (
        "run_identity",
        "census_sha256",
        "predictions_sha256",
        "complete_sha256",
        "candidate_plan_sha256",
        "implementation_bundle_sha256",
    ):
        _plan_hex64(accepted, key, "accepted_h")
    _canonical_relative_path(accepted["run_dir"], "accepted_h.run_dir")
    _require(
        accepted["run_dir"] == ACCEPTED_H_RUN_DIR,
        f"candidate plan accepted_h.run_dir {accepted['run_dir']!r} is not the frozen accepted "
        f"Stage-H run directory {ACCEPTED_H_RUN_DIR!r}",
    )
    _require(
        accepted["complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256,
        "candidate plan does not bind the exact accepted Stage-H COMPLETE bytes",
    )

    # --- authorities: exact name -> canonical path -> digest ----------------------------
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
        _canonical_relative_path(spec["path"], f"authorities.{name}.path")
        _require(
            spec["path"] == _AUTHORITY_PATHS[name],
            f"candidate plan: authorities.{name}.path is {spec['path']!r} but the frozen "
            f"canonical location is {_AUTHORITY_PATHS[name]!r}; an equivalent spelling of the "
            "same bytes is not the canonical mapping and does not authorize",
        )
        _plan_hex64(spec, "sha256", f"authorities.{name}")

    bound = obj["bound_authorities"]
    _require(type(bound) is dict, "candidate plan: bound_authorities must be an object")
    _require(
        frozenset(bound) == BOUND_AUTHORITY_KEYS,
        "candidate plan bound_authorities is not the exact frozen graph authority key set",
    )
    for key in sorted(bound):
        _plan_hex64(bound, key, "bound_authorities")

    # --- input bindings -----------------------------------------------------------------
    bindings = obj["input_bindings"]
    _require(
        type(bindings) is dict and bindings, "candidate plan: input_bindings must be an object"
    )
    for binding_id, entry in sorted(bindings.items()):
        _require(
            type(binding_id) is str and binding_id,
            "candidate plan: input_bindings keys must be exact non-empty strings",
        )
        spec = _closed_plan(entry, _PLAN_BINDING_FIELDS, f"input_bindings.{binding_id}")
        for key in ("documents_sha256", "eligibility_index_sha256", "release_manifest_sha256"):
            _plan_hex64(spec, key, f"input_bindings.{binding_id}")
        for key in ("documents_size_bytes", "total_physical_rows", "expected_eligible_rows"):
            _plan_int(spec, key, f"input_bindings.{binding_id}", minimum=0)

    # --- nodes and execution order ------------------------------------------------------
    nodes = obj["nodes"]
    _require(type(nodes) is list and nodes, "candidate plan: nodes must be a non-empty list")
    allowed_modes = frozenset(SELECTION_MODES) | {BRANCH_DEPENDENT}
    for index, entry in enumerate(nodes):
        where = f"nodes[{index}]"
        spec = _closed_plan(entry, _PLAN_NODE_FIELDS, where)
        _plan_str(spec, "source_id", where)
        stage = _plan_str(spec, "stage", where)
        _require(stage in STAGE_PRIORITY, f"candidate plan: {where}.stage is not a known stage")
        _require(
            _plan_int(spec, "stage_priority", where, minimum=0) == STAGE_PRIORITY[stage],
            f"candidate plan: {where}.stage_priority disagrees with its stage",
        )
        _plan_int(spec, "target_serialized_tokens", where, minimum=1)
        _require(
            _plan_str(spec, "selection_mode", where) in allowed_modes,
            f"candidate plan: {where}.selection_mode is not a declared selection mode",
        )
        _plan_str_list(spec, "input_binding_ids", where)
        unknown = sorted(set(spec["input_binding_ids"]) - set(bindings))
        _require(
            not unknown,
            f"candidate plan: {where} names input binding(s) the plan never declares: {unknown}",
        )
    _plan_str_list(obj, "node_order", "top level")

    # --- node -> binding projection ------------------------------------------------------
    projection = obj["node_binding_projection"]
    _require(
        type(projection) is dict and projection,
        "candidate plan: node_binding_projection must be a non-empty object",
    )
    for source_id, allowed in sorted(projection.items()):
        _require(
            type(source_id) is str and source_id,
            "candidate plan: node_binding_projection keys must be exact non-empty strings",
        )
        _plan_str_list(projection, source_id, "node_binding_projection", sorted_unique=True)
        outside = sorted(set(allowed) - set(bindings))
        _require(
            not outside,
            f"candidate plan: node_binding_projection[{source_id!r}] names binding(s) outside "
            f"the plan-authorized global input-binding set: {outside}",
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

    # --- environment contract, closed all the way down -----------------------------------
    contract = _closed_plan(
        obj["environment_contract"], _PLAN_ENVIRONMENT_FIELDS, "environment_contract"
    )
    _require(
        _plan_str(contract, "python_executable", "environment_contract")
        == REQUIRED_PYTHON_EXECUTABLE
        and _plan_str(contract, "python_version", "environment_contract") == REQUIRED_PYTHON_VERSION
        and _plan_str(contract, "tokenizers_version", "environment_contract")
        == REQUIRED_TOKENIZERS_VERSION,
        "candidate plan environment contract disagrees with this implementation",
    )
    observed = _closed_plan(
        contract["observed_at_generation"],
        _PLAN_OBSERVED_FIELDS,
        "environment_contract.observed_at_generation",
    )
    for key in sorted(_PLAN_OBSERVED_FIELDS):
        _plan_str(observed, key, "environment_contract.observed_at_generation")

    # --- this implementation --------------------------------------------------------------
    files = obj["implementation_files"]
    _require(
        type(files) is dict and files,
        "candidate plan: implementation_files must be a non-empty object",
    )
    _require(
        frozenset(files) == frozenset(IMPLEMENTATION_BUNDLE_FILES),
        "candidate plan implementation_files is not the exact Stage-I bundle member list",
    )
    for member in sorted(files):
        _plan_hex64(files, member, "implementation_files")
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


# ------------------------------------------------------------------ derivation of the state


def _derive_authorized_state(
    plan_path: Path,
    expected_plan_sha256: str,
    repo_root: Path,
    *,
    require_executable: bool,
) -> CanonicalAuthorizedState:
    """Read the plan bytes, prove every bound artifact, and build the complete canonical state.

    This function mints nothing. It is the single derivation used both to authorize a plan and to
    re-derive that same state immediately before every load-bearing use, so the two can never be
    partial re-checks of each other -- which is exactly how R1's and R2's revalidation missed
    graph, H and path substitution.
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
        plan["implementation_bundle_sha256"] == bundle,
        "candidate plan was generated against a different Stage-I implementation bundle",
    )
    _require(
        dict(plan["implementation_files"]) == dict(sorted(files.items())),
        "candidate plan implementation file digests disagree with this checkout",
    )
    runtime_commit = _git_head(repo_root)
    _require(
        plan["implementation_commit"] == runtime_commit,
        f"runtime repository HEAD {runtime_commit} is not the plan's implementation commit "
        f"{plan['implementation_commit']}",
    )

    # --- environment --------------------------------------------------------------------
    environment = current_environment()
    verify_environment(environment, require_executable=require_executable)

    # --- the graph named by the PLAN, not by a CLI argument -----------------------------
    graph_path = _resolve_repo_path(repo_root, plan["graph_path"])
    graph = load_source_graph(
        graph_path, verify_hashes=True, expected_graph_sha256=plan["graph_sha256"]
    )
    _require(graph.seed == plan["seed"], "owner graph seed disagrees with the plan")
    _require(
        dict(graph.bound_authorities) == dict(plan["bound_authorities"]),
        "owner graph bound authorities disagree with the plan",
    )

    # --- every authority, re-hashed from disk at its exact canonical location ------------
    authority_paths: dict[str, str] = {}
    authority_sha256: dict[str, str] = {}
    for name, entry in sorted(plan["authorities"].items()):
        path = _resolve_repo_path(repo_root, entry["path"])
        _require(path.is_file(), f"plan authority {name} is missing at {entry['path']}")
        actual = file_sha256(path)
        _require(
            actual == entry["sha256"],
            f"plan authority {name} is {actual} on disk but the plan binds {entry['sha256']}",
        )
        authority_paths[name] = entry["path"]
        authority_sha256[name] = actual

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
        # The 151.6 GB corpus is NOT re-hashed here; its identity is proved on the descriptor the
        # authoritative scan consumes. What is re-proved is the frozen metadata identity: size,
        # release manifest bytes and eligibility index bytes.
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

    # --- R3-B: the trusted node -> allowed-binding authority ----------------------------
    projection = {
        source_id: tuple(sorted(set(allowed)))
        for source_id, allowed in plan["node_binding_projection"].items()
    }
    _require(
        {source_id: list(allowed) for source_id, allowed in sorted(projection.items())}
        == {node.source_id: sorted(set(node.input_binding_ids)) for node in graph.nodes},
        "plan node/binding projection disagrees with the frozen owner graph",
    )
    authorized_bindings = tuple(sorted(graph.bindings))
    outside = sorted(
        {binding for allowed in projection.values() for binding in allowed}
        - set(authorized_bindings)
    )
    _require(
        not outside,
        f"node/binding projection names binding(s) outside the plan-authorized global "
        f"input-binding set: {outside}",
    )

    frozen_graph = _frozen_graph(graph)
    frozen_accepted = _frozen_accepted(accepted)
    tokenizer_path = plan["authorities"]["tokenizer"]["path"]
    reference_path = plan["authorities"]["reference_exclusion"]["path"]
    payload_projection = _authorized_state_payload(
        plan_sha256=digest,
        plan=plan,
        graph=frozen_graph,
        graph_path=plan["graph_path"],
        accepted=frozen_accepted,
        h_run_dir=accepted_spec["run_dir"],
        environment=environment,
        bundle_sha256=bundle,
        bundle_files=dict(sorted(files.items())),
        implementation_commit=runtime_commit,
        authority_paths=authority_paths,
        authority_sha256=authority_sha256,
        tokenizer_path=tokenizer_path,
        reference_exclusion_path=reference_path,
        node_binding_projection=projection,
    )
    state_sha256 = hashlib.sha256(canonical_json_bytes(payload_projection)).hexdigest()

    return CanonicalAuthorizedState(
        repo_root=repo_root,
        plan_path=plan_path,
        plan_sha256=digest,
        plan=_deep_freeze(plan),
        graph=frozen_graph,
        graph_path=graph_path,
        accepted=frozen_accepted,
        environment=environment,
        bundle_sha256=bundle,
        bundle_files=MappingProxyType(dict(sorted(files.items()))),
        implementation_commit=runtime_commit,
        h_run_dir=h_run_dir,
        tokenizer_path=_resolve_repo_path(repo_root, tokenizer_path),
        reference_exclusion_path=_resolve_repo_path(repo_root, reference_path),
        authority_paths=MappingProxyType(dict(sorted(authority_paths.items()))),
        authority_sha256=MappingProxyType(dict(sorted(authority_sha256.items()))),
        node_binding_projection=MappingProxyType(dict(sorted(projection.items()))),
        authorized_input_binding_ids=authorized_bindings,
        canonical_payload=_deep_freeze(payload_projection),
        state_sha256=state_sha256,
    )


# ------------------------------------------------------------------ the authorization subsystem


def _build_authorization_subsystem():
    """Own the authority registry inside a closure. There is no way in from module scope.

    R2 kept the registry and its minting helper as module attributes, so
    ``_register_authorized(lookalike, _canonical_state_digest(lookalike))`` promoted a
    hand-built object to full authority -- Codex demonstrated exactly that. Renaming those
    symbols would not have fixed it. The registry, the record type and the only operation that
    can write to it now live here, and nothing that escapes can add an entry.

    Exported: ``authorize_plan`` (the sole mint), ``require_authorized`` (a read-only membership
    test) and ``resolve_authorized_state`` (fresh re-derivation + equality). Deliberately NOT
    exported: any way to register, restamp, or hand-construct an authorized instance.
    """
    registry: dict[int, tuple[weakref.ref, CanonicalAuthorizedState]] = {}

    class AuthorizedIContext:
        """The single capability that can run and publish an authoritative Stage-I realization.

        A pure capability handle: it has no ``__dict__`` and no slots for data, so there is
        nothing on it to read, copy, restamp or substitute. Authority is the *exact instance*
        recorded in the closure-owned registry when ``authorize_plan`` succeeded, and the
        authorized state lives in that registry record rather than on the object.

        Every load-bearing value the runtime needs comes from ``revalidate()``, which re-derives
        the whole state from the authorization-time plan bytes and proves it still digests to
        what was authorized -- so there is no long-lived, caller-reachable graph, census or path
        to swap in the first place.
        """

        __slots__ = ("__weakref__",)

        def __init__(self) -> None:
            raise RealizationError(
                "an AuthorizedIContext cannot be constructed; authority is minted only by a "
                "successful authorize_plan against exact owner-supplied plan bytes"
            )

        def __copy__(self) -> None:
            raise RealizationError(
                "an AuthorizedIContext must not be copied; authority is the exact authorized "
                "instance"
            )

        def __deepcopy__(self, memo: Any) -> None:
            raise RealizationError(
                "an AuthorizedIContext must not be copied; authority is the exact authorized "
                "instance"
            )

        def __reduce__(self) -> None:
            raise RealizationError("an AuthorizedIContext must not be pickled or reconstructed")

        def __repr__(self) -> str:
            entry = registry.get(id(self))
            if entry is None or entry[0]() is not self:
                return "<AuthorizedIContext unauthorized>"
            return f"<AuthorizedIContext state={entry[1].state_sha256[:16]}>"

        # -- read-only views. None of these confer authority; they report it. ------------
        @property
        def plan_path(self) -> Path:
            return require_authorized(self, "plan path").plan_path

        @property
        def plan_sha256(self) -> str:
            return require_authorized(self, "plan digest").plan_sha256

        @property
        def repo_root(self) -> Path:
            return require_authorized(self, "repository root").repo_root

        @property
        def authorized_state_sha256(self) -> str:
            return require_authorized(self, "authorized state digest").state_sha256

        def revalidate(self) -> CanonicalAuthorizedState:
            """Re-derive the authorized state from disk and prove it is what was authorized.

            Returns the freshly derived state. Callers must use the returned value rather than
            anything they were holding: that is what makes substitution structurally impossible
            instead of merely detectable.
            """
            return resolve_authorized_state(self, "revalidation")

    def require_authorized(context: Any, where: str) -> CanonicalAuthorizedState:
        """Return the registered state, or refuse. Exact object identity is the capability."""
        if not isinstance(context, AuthorizedIContext):
            raise RealizationError(f"{where} requires an authorized Stage-I context")
        entry = registry.get(id(context))
        if entry is None or entry[0]() is not context:
            raise RealizationError(
                f"{where} requires the exact Stage-I context instance returned by authorize_plan; "
                "a manually constructed, restamped, copied or replaced context is not an "
                "authorization"
            )
        return entry[1]

    def resolve_authorized_state(context: Any, where: str) -> CanonicalAuthorizedState:
        """Fresh derivation from the authorization-time plan bytes, proved equal to authorized."""
        authorized = require_authorized(context, where)
        fresh = _derive_authorized_state(
            authorized.plan_path,
            authorized.plan_sha256,
            authorized.repo_root,
            require_executable=False,
        )
        _require(
            fresh.state_sha256 == authorized.state_sha256,
            "the authorized Stage-I state no longer re-derives to what was authorized: the plan "
            "bytes, the owner graph, the accepted Stage-H result, a bound authority or a bound "
            "path changed after authorization",
        )
        return fresh

    def authorize_plan(
        plan_path: Path,
        expected_plan_sha256: str,
        repo_root: Path,
        *,
        require_executable: bool = True,
    ) -> AuthorizedIContext:
        """Authorization is a capability, not a comparison.

        The owner supplies the expected digest out of band; it is checked against the plan bytes,
        and those same bytes then drive every load. A plan cannot authorise itself, an
        authorization for one plan cannot be reused for another, and nothing load-bearing is taken
        from a CLI argument. This is the only operation in the program that can add an entry to
        the authority registry.
        """
        state = _derive_authorized_state(
            plan_path, expected_plan_sha256, repo_root, require_executable=require_executable
        )
        context = object.__new__(AuthorizedIContext)
        key = id(context)

        def _drop(_ref: Any, key: int = key) -> None:
            registry.pop(key, None)

        registry[key] = (weakref.ref(context, _drop), state)
        return context

    return AuthorizedIContext, authorize_plan, require_authorized, resolve_authorized_state


(
    AuthorizedIContext,
    authorize_plan,
    _require_authorized,
    resolve_authorized_state,
) = _build_authorization_subsystem()


def revalidate_authorized(context: Any) -> CanonicalAuthorizedState:
    """Module-level spelling of ``context.revalidate()``, for callers holding only the handle."""
    return resolve_authorized_state(context, "revalidation")


PASS1_RESULT_PREFIX = "pass1_result"


def _freeze_pass1_result(out_dir: Path, pass1: Mapping[str, Any], explicit: Path | None) -> Path:
    """Write the post-Pass-1 expected result durably, outside the realization, before Pass 2.

    Placed as a sibling of the run directory rather than inside it: an expectation that lives in
    the tree it is meant to constrain is not an expectation. Refuses to overwrite, so the frozen
    digest for a given expectation can never be quietly replaced by a later one.
    """
    payload = canonical_json_bytes(pass1)
    digest = hashlib.sha256(payload).hexdigest()
    path = explicit if explicit is not None else out_dir / f"{PASS1_RESULT_PREFIX}-{digest}.json"
    _require(
        not path.exists(),
        f"post-Pass-1 expected result already exists, refusing to overwrite: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_durable(path, payload)
    _fsync_directory(path.parent)
    written, written_digest = read_authoritative_bytes(path, max_bytes=1 << 30)
    _require(
        written == payload and written_digest == digest,
        f"{path}: the frozen post-Pass-1 result does not read back as it was written",
    )
    return path


def _fsync_directory(path: Path) -> None:
    import os

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def realize_and_publish(
    context: AuthorizedIContext,
    *,
    out_dir: Path,
    work_dir: Path,
    pass1_result_path: Path | None = None,
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
    sort_chunk_lines: int = DEFAULT_SORT_CHUNK_LINES,
) -> tuple[Path, dict[str, Any]]:
    """The authoritative Stage-I run: derive, prove equal to H, commit the expectation, then build.

    Three authority layers, in strict order, each one only ever reading upward:

    1. the externally authorized candidate plan and the complete canonical authorized state it
       determines -- re-derived from disk here, not read off an object;
    2. the post-Pass-1 expected result, produced from the independent Pass-1 selection and the
       H/I gate verdict and frozen to disk *before* materialization;
    3. the physical realization -- shards, manifest, COMPLETE -- which must prove itself equal to
       layer 2 and may never define it.

    Every load-bearing input comes from ``context``, which only ``authorize_plan`` can produce,
    and is re-derived through it. No graph, accepted-H run, authority, binding, node set or shard
    policy can be substituted by a caller: the CLI supplies a plan digest and an output directory
    and nothing else that decides what gets built.
    """
    state = resolve_authorized_state(context, "realization")

    tokenizer_path = str(state.tokenizer_path)
    reference_exclusion = load_reference_exclusion(
        state.reference_exclusion_path,
        expected_sha256=state.graph.bound_authorities["g2_exclusion_manifest_sha256"],
    )

    selections, counters = derive_selection(state.graph, tokenizer_path, reference_exclusion)
    comparison = compare_with_h(selections, state.accepted)
    require_h_i_equality(comparison)

    # --- layer 2 is created here, and its digest is fixed before a single record exists ----
    pass1 = build_pass1_result(selections, state, comparison)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = _freeze_pass1_result(out_dir, pass1, pass1_result_path)
    pass1_sha256 = hashlib.sha256(canonical_json_bytes(pass1)).hexdigest()
    expected = trusted_expected_result(pass1, expected_sha256=pass1_sha256)
    run_name = f"run-{expected.stage_i_run_identity[:32]}"

    manifest = build_manifest(selections, state, expected)
    records = iter_records_in_physical_order(
        state.graph, tokenizer_path, selections, work_dir, read_window_bytes=read_window_bytes
    )
    # Revalidate one last time: everything the manifest attests to has now been derived, and
    # nothing has been written, so a late change to the plan or the implementation still costs
    # nothing to reject.
    final_state = resolve_authorized_state(context, "publication")
    _require(
        final_state.state_sha256 == state.state_sha256,
        "the authorized Stage-I state changed between selection and publication",
    )
    final = publish_atomic(
        out_dir,
        run_name,
        manifest,
        records,
        expected=expected,
        read_window_bytes=read_window_bytes,
        sort_chunk_lines=sort_chunk_lines,
    )

    return final, {
        "run_identity": expected.stage_i_run_identity,
        "authorized_state_sha256": state.state_sha256,
        "post_pass1_result_identity_schema": expected.result_identity_schema,
        "post_pass1_result_identity_sha256": expected.result_identity_sha256,
        "post_pass1_result_path": str(frozen_path),
        "selection_sequence_commitment_version": expected.selection_sequence_commitment_version,
        "selection_sequence_commitment_map_sha256": (
            expected.selection_sequence_commitment_map_sha256
        ),
        "node_binding_projection_sha256": expected.node_binding_projection_sha256,
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
    run_parser.add_argument(
        "--pass1-result-out",
        type=Path,
        default=None,
        help=(
            "where to freeze the post-Pass-1 expected result before materialization; defaults to "
            "a digest-named sibling of the run directory. Must not already exist."
        ),
    )

    # The strict consumer, exposed as its own command precisely because its expectation must come
    # from outside the thing it is checking: the frozen post-Pass-1 artifact and the digest the
    # owner holds for it. There is deliberately no flag that lets the realization supply its own.
    verify_parser = sub.add_parser(
        "verify", help="strictly verify a published realization against a trusted expected result"
    )
    verify_parser.add_argument("--realization", type=Path, required=True)
    verify_parser.add_argument("--pass1-result", type=Path, required=True)
    verify_parser.add_argument("--expected-pass1-result-sha256", type=str, required=True)
    verify_parser.add_argument("--work-dir", type=Path, default=None)

    args = parser.parse_args(raw_argv)

    if args.command == "verify-environment":
        environment = current_environment()
        verify_environment(environment, require_executable=args.require_executable)
        print(canonical_json_bytes(environment.as_canonical()).decode("utf-8"), end="")
        return 0

    if args.command == "verify":
        expected = load_trusted_expected_result(
            args.pass1_result, expected_sha256=args.expected_pass1_result_sha256
        )
        manifest = load_published_realization(
            args.realization, expected=expected, work_dir=args.work_dir
        )
        print(
            canonical_json_bytes({
                "verified": str(args.realization),
                "stage_i_run_identity": manifest["stage_i_run"]["run_identity"],
                "post_pass1_result_identity_sha256": expected.result_identity_sha256,
                "records": manifest["totals"]["records"],
                "serialized_tokens": manifest["totals"]["serialized_tokens"],
            }).decode("utf-8"),
            end="",
        )
        return 0

    if args.command == "run":
        repo_root = args.repo_root.resolve()
        context = authorize_plan(args.plan, args.expected_plan_sha256, repo_root)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        owned = args.work_dir is None
        # The run's name is not knowable until Pass 1 has committed its expected result, so the
        # scratch directory is named after the authorized state instead of the published run.
        work_dir = (
            Path(
                tempfile.mkdtemp(
                    prefix=f".stage-i-{context.authorized_state_sha256[:16]}.work-",
                    dir=str(args.out_dir),
                )
            )
            if owned
            else args.work_dir
        )
        try:
            final, summary = realize_and_publish(
                context,
                out_dir=args.out_dir,
                work_dir=work_dir,
                pass1_result_path=args.pass1_result_out,
            )
        finally:
            if owned:
                shutil.rmtree(work_dir, ignore_errors=True)
        print(f"published {final}")
        print(f"post_pass1_result {summary['post_pass1_result_path']}")
        print(f"post_pass1_result_sha256 {summary['post_pass1_result_identity_sha256']}")
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
    "PASS1_RESULT_SCHEMA",
    "PLAN_SCHEMA",
    "REQUIRED_PYTHON_EXECUTABLE",
    "REQUIRED_PYTHON_VERSION",
    "REQUIRED_TOKENIZERS_VERSION",
    "SHARD_POLICY_RULE",
    "AcceptedH",
    "AuthorizedIContext",
    "CanonicalAuthorizedState",
    "authorization_block",
    "authorize_plan",
    "Environment",
    "RealizationError",
    "SelectedTarget",
    "build_manifest",
    "build_materialization_targets",
    "build_pass1_result",
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
    "resolve_authorized_state",
    "revalidate_authorized",
    "SELECTION_SEQUENCE_SCHEMA",
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
