#!/usr/bin/env python3
"""Strict parser and validator for the owner-frozen Stage-I source graph (schema v2).

This module owns three things and nothing else: parsing the frozen graph, validating the
immutable input-binding registry, and validating eligibility row-index sets. It never selects
documents and never reads document bytes.

It is deliberately separate from ``pretrain/select_pretrain_documents.py``, which remains frozen
and byte-identical as prior evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

GRAPH_SCHEMA_VERSION = "petitgpt-stage-i-source-graph-v1"
SELECTION_MODES = ("SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC")
BRANCH_DEPENDENT = "BRANCH_DEPENDENT"
STAGE_PRIORITY = {"stage_b": 0, "stage_a": 1}
PREDICATE_KINDS = ("ALL_ELIGIBLE", "INTEGER_SCORE_AT_LEAST")


class GraphError(RuntimeError):
    """Any fail-closed graph, binding or index validation failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical hash-bearing serialisation: UTF-8, sorted keys, fixed separators, one newline."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphError(message)


def _check_keys(value: dict[str, Any], allowed: set[str], *, where: str) -> None:
    _require(isinstance(value, dict), f"{where}: expected an object")
    unknown = sorted(set(value) - allowed)
    _require(not unknown, f"{where}: unknown keys {unknown}")


def derive_seed(domain_utf8: str) -> int:
    """The frozen seed derivation. No search, no alternatives, no data conditioning."""
    digest = hashlib.sha256(domain_utf8.encode("utf-8")).digest()
    return int.from_bytes(digest[0:8], "big") & 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class InputBinding:
    input_binding_id: str
    release_key: str
    documents_path: Path
    documents_sha256: str
    documents_size_bytes: int
    release_manifest_path: Path
    release_manifest_sha256: str
    eligibility_index_path: Path
    eligibility_index_sha256: str
    total_physical_rows: int
    excluded_rows: int
    expected_eligible_rows: int
    schema_accessor: dict[str, Any]
    text_field: str
    cleaning_contract: dict[str, Any]


@dataclass(frozen=True)
class Node:
    node_id: str
    source_id: str
    stage: str
    stage_priority: int
    target_serialized_tokens: int
    input_binding_ids: tuple[str, ...]
    selection_mode: str
    candidate_predicate: dict[str, Any]
    branch_primary: dict[str, Any] | None
    branch_fallback: dict[str, Any] | None

    @property
    def priority_key(self) -> tuple[int, str]:
        return STAGE_PRIORITY[self.stage], self.source_id


@dataclass(frozen=True)
class SourceGraph:
    seed: int
    nodes: tuple[Node, ...]
    bindings: dict[str, InputBinding]
    bound_authorities: dict[str, str]
    raw: dict[str, Any]
    graph_sha256: str


def _parse_predicate(raw: Any, *, where: str) -> dict[str, Any]:
    _check_keys(raw, {"kind", "field", "value", "note"}, where=where)
    kind = raw.get("kind")
    _require(kind in PREDICATE_KINDS, f"{where}: predicate kind must be one of {PREDICATE_KINDS}")
    if kind == "INTEGER_SCORE_AT_LEAST":
        _require(
            isinstance(raw.get("field"), str) and raw["field"], f"{where}: predicate needs field"
        )
        value = raw.get("value")
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{where}: predicate value must be int",
        )
    return dict(raw)


def _parse_branch(raw: Any, *, where: str, expect: str) -> dict[str, Any]:
    _check_keys(
        raw,
        {"branch", "selection_mode", "candidate_predicate", "rank_order", "continuous_score_field"},
        where=where,
    )
    _require(raw.get("branch") == expect, f"{where}: branch must be {expect}")
    mode = raw.get("selection_mode")
    _require(mode in SELECTION_MODES, f"{where}: selection_mode must be one of {SELECTION_MODES}")
    _parse_predicate(raw.get("candidate_predicate"), where=f"{where}.candidate_predicate")
    if mode == "EXACT_SCORE_DESC_SHA_ASC":
        field = raw.get("continuous_score_field")
        _require(
            isinstance(field, str) and field, f"{where}: fallback needs continuous_score_field"
        )
    return dict(raw)


def _validate_binding(raw: dict[str, Any], key: str, *, verify_hashes: bool) -> InputBinding:
    where = f"input_bindings[{key!r}]"
    _check_keys(
        raw,
        {
            "input_binding_id",
            "release_key",
            "documents_path",
            "documents_sha256",
            "documents_size_bytes",
            "release_manifest_path",
            "release_manifest_sha256",
            "eligibility_index_path",
            "eligibility_index_sha256",
            "eligibility_index_element_width_bytes",
            "eligibility_index_dtype",
            "total_physical_rows",
            "excluded_rows",
            "expected_eligible_rows",
            "schema_accessor",
            "text_field",
            "cleaning_contract",
        },
        where=where,
    )
    _require(
        raw["input_binding_id"] == key, f"{where}: input_binding_id must equal its registry key"
    )
    _require(raw["eligibility_index_element_width_bytes"] == 4, f"{where}: element width must be 4")
    _require(raw["eligibility_index_dtype"] == "<u4", f"{where}: dtype must be <u4")
    total, excluded, eligible = (
        int(raw["total_physical_rows"]),
        int(raw["excluded_rows"]),
        int(raw["expected_eligible_rows"]),
    )
    _require(
        total - excluded == eligible,
        f"{where}: total_rows - excluded_rows != expected_eligible_rows",
    )
    binding = InputBinding(
        input_binding_id=key,
        release_key=raw["release_key"],
        documents_path=Path(raw["documents_path"]),
        documents_sha256=raw["documents_sha256"],
        documents_size_bytes=int(raw["documents_size_bytes"]),
        release_manifest_path=Path(raw["release_manifest_path"]),
        release_manifest_sha256=raw["release_manifest_sha256"],
        eligibility_index_path=Path(raw["eligibility_index_path"]),
        eligibility_index_sha256=raw["eligibility_index_sha256"],
        total_physical_rows=total,
        excluded_rows=excluded,
        expected_eligible_rows=eligible,
        schema_accessor=dict(raw["schema_accessor"]),
        text_field=raw["text_field"],
        cleaning_contract=dict(raw["cleaning_contract"]),
    )
    if verify_hashes:
        verify_binding_inputs(binding)
    return binding


def verify_binding_inputs(binding: InputBinding) -> None:
    """Fail closed unless the bound documents, manifest and index are byte-for-byte as frozen."""
    where = binding.input_binding_id
    _require(binding.documents_path.is_file(), f"{where}: documents file missing")
    actual_size = binding.documents_path.stat().st_size
    _require(actual_size == binding.documents_size_bytes, f"{where}: documents size changed")
    _require(
        sha256_file(binding.documents_path) == binding.documents_sha256,
        f"{where}: documents SHA mismatch",
    )
    _require(binding.release_manifest_path.is_file(), f"{where}: release manifest missing")
    _require(
        sha256_file(binding.release_manifest_path) == binding.release_manifest_sha256,
        f"{where}: release manifest SHA mismatch",
    )
    validate_eligibility_index(binding)


def validate_eligibility_index(binding: InputBinding) -> np.ndarray:
    """Validate an eligibility row-index set by content, not merely by length."""
    where = binding.input_binding_id
    path = binding.eligibility_index_path
    _require(path.is_file(), f"{where}: eligibility index missing")
    size = path.stat().st_size
    _require(size % 4 == 0, f"{where}: eligibility index length {size} is not a multiple of 4")
    _require(
        sha256_file(path) == binding.eligibility_index_sha256,
        f"{where}: eligibility index SHA mismatch",
    )
    rows = np.fromfile(path, dtype="<u4")
    _require(
        len(rows) == binding.excluded_rows, f"{where}: eligibility index element count mismatch"
    )
    if len(rows):
        _require(
            bool(np.all(rows[1:] > rows[:-1])),
            f"{where}: eligibility index is not strictly increasing",
        )
        _require(
            int(rows[-1]) < binding.total_physical_rows,
            f"{where}: eligibility index row out of range",
        )
    return rows


def load_source_graph(path: Path, *, verify_hashes: bool = True) -> SourceGraph:
    """Load and strictly validate the owner-frozen graph."""
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    _check_keys(
        raw,
        {
            "schema_version",
            "policy_status",
            "authority",
            "date",
            "note",
            "bound_authorities",
            "selection_seed",
            "stage_priority",
            "execution_order_rule",
            "selection_modes_closed_enum",
            "control_namespace",
            "stage_a_population_rule",
            "ownership_rule",
            "structured_tutorial",
            "h_boundary",
            "resume_supported",
            "input_bindings",
            "nodes",
        },
        where="source graph",
    )
    _require(
        raw.get("schema_version") == GRAPH_SCHEMA_VERSION, "source graph: wrong schema_version"
    )
    _require(
        raw.get("policy_status") == "OWNER_FROZEN",
        "source graph: policy_status must be OWNER_FROZEN",
    )
    _require(raw.get("resume_supported") is False, "source graph: resume must be disabled")

    seed_block = raw["selection_seed"]
    _check_keys(
        seed_block,
        {"domain_utf8", "domain_sha256", "derivation", "seed", "seed_hex"},
        where="selection_seed",
    )
    seed = seed_block["seed"]
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        "selection_seed.seed must be an exact integer",
    )
    derived = derive_seed(seed_block["domain_utf8"])
    _require(
        derived == seed,
        f"selection_seed: derivation mismatch, derived {derived} != declared {seed}",
    )
    _require(
        hashlib.sha256(seed_block["domain_utf8"].encode("utf-8")).hexdigest()
        == seed_block["domain_sha256"],
        "selection_seed: domain_sha256 mismatch",
    )

    bindings: dict[str, InputBinding] = {}
    seen_paths: dict[str, str] = {}
    for key in sorted(raw["input_bindings"]):
        binding = _validate_binding(raw["input_bindings"][key], key, verify_hashes=verify_hashes)
        resolved = (
            str(binding.documents_path.resolve())
            if binding.documents_path.exists()
            else str(binding.documents_path)
        )
        prior = seen_paths.get(resolved)
        _require(
            prior is None,
            f"input_bindings: resolved documents path {resolved!r} registered under both {prior!r} and {key!r}",
        )
        seen_paths[resolved] = key
        bindings[key] = binding

    nodes: list[Node] = []
    seen_ids: set[str] = set()
    for index, raw_node in enumerate(raw["nodes"]):
        where = f"nodes[{index}]"
        _check_keys(
            raw_node,
            {
                "node_id",
                "source_id",
                "stage",
                "stage_priority",
                "target_serialized_tokens",
                "input_binding_ids",
                "selection_mode",
                "candidate_predicate",
                "branch_primary",
                "branch_fallback",
            },
            where=where,
        )
        source_id = raw_node["source_id"]
        _require(raw_node["node_id"] == source_id, f"{where}: node_id must equal source_id")
        _require(source_id not in seen_ids, f"{where}: duplicate source_id {source_id!r}")
        seen_ids.add(source_id)
        stage = raw_node["stage"]
        _require(stage in STAGE_PRIORITY, f"{where}: stage must be one of {sorted(STAGE_PRIORITY)}")
        _require(
            raw_node["stage_priority"] == STAGE_PRIORITY[stage],
            f"{where}: stage_priority disagrees with stage",
        )
        target = raw_node["target_serialized_tokens"]
        _require(
            isinstance(target, int) and not isinstance(target, bool) and target > 0,
            f"{where}: bad target",
        )
        ids = raw_node["input_binding_ids"]
        _require(
            isinstance(ids, list) and ids, f"{where}: input_binding_ids must be a non-empty list"
        )
        _require(len(set(ids)) == len(ids), f"{where}: duplicate input_binding_id in node")
        for ib in ids:
            _require(ib in bindings, f"{where}: unknown input_binding_id {ib!r}")
        mode = raw_node["selection_mode"]
        _require(
            mode in SELECTION_MODES or mode == BRANCH_DEPENDENT,
            f"{where}: bad selection_mode {mode!r}",
        )
        primary = raw_node.get("branch_primary")
        fallback = raw_node.get("branch_fallback")
        if mode == BRANCH_DEPENDENT:
            _require(
                primary is not None and fallback is not None,
                f"{where}: branch-dependent node needs both branches",
            )
            primary = _parse_branch(primary, where=f"{where}.branch_primary", expect="PRIMARY_GE4")
            fallback = _parse_branch(
                fallback, where=f"{where}.branch_fallback", expect="FALLBACK_RANKED_GE3"
            )
        else:
            _require(
                primary is None and fallback is None,
                f"{where}: non-branch node must not declare branches",
            )
        nodes.append(
            Node(
                node_id=raw_node["node_id"],
                source_id=source_id,
                stage=stage,
                stage_priority=STAGE_PRIORITY[stage],
                target_serialized_tokens=target,
                input_binding_ids=tuple(ids),
                selection_mode=mode,
                candidate_predicate=_parse_predicate(
                    raw_node["candidate_predicate"], where=f"{where}.candidate_predicate"
                ),
                branch_primary=primary,
                branch_fallback=fallback,
            )
        )
    _require(nodes, "source graph: no nodes")
    ordered = sorted(nodes, key=lambda n: n.priority_key)
    _require(
        [n.source_id for n in ordered] == [n.source_id for n in nodes],
        "source graph: nodes are not stored in ascending (stage_priority, source_id) order",
    )
    return SourceGraph(
        seed=seed,
        nodes=tuple(ordered),
        bindings=bindings,
        bound_authorities=dict(raw["bound_authorities"]),
        raw=raw,
        graph_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
