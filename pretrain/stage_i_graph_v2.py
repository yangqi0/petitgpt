#!/usr/bin/env python3
"""Strict parser and validator for the owner-frozen Stage-I source graph (schema v2).

This module owns three things and nothing else: parsing the frozen graph, validating the
immutable input-binding registry, and validating eligibility row-index sets. It never selects
documents and never reads document bytes.

It is deliberately separate from ``pretrain/select_pretrain_documents.py``, which remains frozen
and byte-identical as prior evidence.

Two properties are load-bearing here and are enforced rather than assumed:

* every authoritative JSON document is parsed with a closed schema by a loader that rejects
  duplicate keys at any depth, non-finite constants and silent numeric coercion, so an owner
  policy cannot drift through a second copy of a key or a numeric string;
* every authoritative input is opened once and validated on the descriptor it is consumed from,
  so the bytes that were hashed are the bytes that were used. Nothing here hashes a path, closes
  it and reopens it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np

GRAPH_SCHEMA_VERSION = "petitgpt-stage-i-source-graph-v1"
SELECTION_MODES = ("SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC")
BRANCH_DEPENDENT = "BRANCH_DEPENDENT"
STAGE_PRIORITY = {"stage_b": 0, "stage_a": 1}
PREDICATE_KINDS = ("ALL_ELIGIBLE", "INTEGER_SCORE_AT_LEAST")
H_CANONICAL_OUTPUT_LABEL = "NON_AUTHORITATIVE_FEASIBILITY_REPLAY"
UNDERSCORES_POLICIES = ("keep", "space", "remove")
JSON_SCORE_TYPES = ("int", "float")
BOUND_AUTHORITY_KEYS = frozenset({
    "d2_d3_eligibility_manifest_sha256",
    "g2_exclusion_manifest_sha256",
    "g2_manifest_sha256",
    "g_manifest_sha256",
    "hq_policy_sha256",
    "selector_v1_sha256_preserved",
    "stage_e_allocation_sha256",
    "tokenizer_sha256",
})
ELIGIBILITY_INDEX_DTYPE = "<u4"
ELIGIBILITY_INDEX_WIDTH = 4
_HEX64 = frozenset("0123456789abcdef")


class GraphError(RuntimeError):
    """Any fail-closed graph, binding or index validation failure."""


# --------------------------------------------------------------------------- strict JSON


class StrictJSONError(GraphError):
    """A JSON document violated the closed parsing contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) > 1:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise StrictJSONError(f"duplicate JSON key {key!r}")
            seen.add(key)
    return dict(pairs)


def _reject_constant(value: str) -> Any:
    raise StrictJSONError(f"non-finite JSON constant {value!r}")


def strict_json_loads(text: str | bytes, *, where: str = "JSON") -> Any:
    """Parse JSON with duplicate keys and NaN/Infinity constants rejected at any depth."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StrictJSONError(f"{where}: not valid UTF-8: {exc}") from exc
    try:
        return json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except StrictJSONError as exc:
        raise StrictJSONError(f"{where}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"{where}: malformed JSON: {exc}") from exc


def strict_json_object(text: str | bytes, *, where: str = "JSON") -> dict[str, Any]:
    value = strict_json_loads(text, where=where)
    if not isinstance(value, dict):
        raise StrictJSONError(f"{where}: expected a JSON object, got {type(value).__name__}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical hash-bearing serialisation: UTF-8, sorted keys, fixed separators, one newline."""
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


# --------------------------------------------------------------------------- typed accessors


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphError(message)


def _check_keys(
    value: Any, *, required: set[str], optional: set[str] = frozenset(), where: str
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where}: expected an object")
    unknown = sorted(set(value) - required - set(optional))
    _require(not unknown, f"{where}: unknown keys {unknown}")
    missing = sorted(required - set(value))
    _require(not missing, f"{where}: missing required keys {missing}")
    return value


def _req_str(obj: dict[str, Any], key: str, where: str, *, nonempty: bool = True) -> str:
    value = obj[key]
    _require(isinstance(value, str), f"{where}.{key}: must be a JSON string")
    _require(not nonempty or value != "", f"{where}.{key}: must not be empty")
    return value


def _req_bool(obj: dict[str, Any], key: str, where: str) -> bool:
    value = obj[key]
    _require(isinstance(value, bool), f"{where}.{key}: must be a JSON boolean")
    return value


def _req_int(obj: dict[str, Any], key: str, where: str, *, minimum: int | None = None) -> int:
    value = obj[key]
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{where}.{key}: must be an exact JSON integer",
    )
    if minimum is not None:
        _require(value >= minimum, f"{where}.{key}: must be >= {minimum}")
    return value


def _req_number(
    obj: dict[str, Any], key: str, where: str, *, minimum: float, maximum: float
) -> float:
    value = obj[key]
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{where}.{key}: must be a JSON number",
    )
    number = float(value)
    _require(number == number, f"{where}.{key}: must not be NaN")
    _require(minimum <= number <= maximum, f"{where}.{key}: must be within [{minimum}, {maximum}]")
    return number


def _req_hex64(obj: dict[str, Any], key: str, where: str) -> str:
    value = _req_str(obj, key, where)
    _require(
        len(value) == 64 and set(value) <= _HEX64,
        f"{where}.{key}: must be a lowercase 64-character SHA-256 hex digest",
    )
    return value


def _req_obj(obj: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = obj[key]
    _require(isinstance(value, dict), f"{where}.{key}: must be a JSON object")
    return value


def _req_list(obj: dict[str, Any], key: str, where: str, *, nonempty: bool = True) -> list[Any]:
    value = obj[key]
    _require(isinstance(value, list), f"{where}.{key}: must be a JSON array")
    _require(not nonempty or value, f"{where}.{key}: must not be empty")
    return value


def derive_seed(domain_utf8: str) -> int:
    """The frozen seed derivation. No search, no alternatives, no data conditioning."""
    digest = hashlib.sha256(domain_utf8.encode("utf-8")).digest()
    return int.from_bytes(digest[0:8], "big") & 0x7FFFFFFFFFFFFFFF


# --------------------------------------------------------------------------- authoritative I/O


@dataclass(frozen=True)
class FileIdentity:
    """The identity of the opened object, not of the path that named it."""

    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int

    def as_dict(self) -> dict[str, int]:
        return {
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "st_size": self.st_size,
            "st_mtime_ns": self.st_mtime_ns,
        }


@contextlib.contextmanager
def open_authoritative(path: Path, *, buffering: int = -1):
    """Open one regular file for authoritative reading and prove the object never changed.

    Yields ``(stream, identity)``. Callers must read exclusively from ``stream``; reopening the
    path would reintroduce exactly the hash-then-reread race this exists to remove. Symlinks are
    refused at the final component, non-regular files are refused, and the descriptor is
    ``fstat``-ed before and after so a replacement under the same path cannot pass unnoticed.
    """
    # O_NONBLOCK matters for the refusal itself: opening a FIFO for reading blocks until a writer
    # appears, so without it a non-regular "input" would hang rather than fail closed. Regular
    # files ignore it.
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GraphError(f"{path}: cannot open authoritatively: {exc.strerror}") from exc
    stream = os.fdopen(descriptor, "rb", buffering=buffering)
    try:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise GraphError(f"{path}: not a regular file")
        if hasattr(os, "O_NONBLOCK"):
            current = fcntl.fcntl(stream.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(stream.fileno(), fcntl.F_SETFL, current & ~os.O_NONBLOCK)
        identity = FileIdentity(
            st_dev=before.st_dev,
            st_ino=before.st_ino,
            st_size=before.st_size,
            st_mtime_ns=before.st_mtime_ns,
        )
        yield stream, identity
        after = os.fstat(stream.fileno())
        if (after.st_dev, after.st_ino, after.st_size) != (
            identity.st_dev,
            identity.st_ino,
            identity.st_size,
        ):
            raise GraphError(f"{path}: opened object changed while it was being read")
    finally:
        stream.close()


def sha256_stream(stream: Any, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_bytes), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a bounded authority file from a single verified descriptor."""
    with open_authoritative(path) as (stream, _identity):
        return sha256_stream(stream)


def read_authoritative_bytes(
    path: Path, *, max_bytes: int = 512 * 1024 * 1024
) -> tuple[bytes, str]:
    """Read a bounded file once and return ``(bytes, sha256)`` computed from those same bytes."""
    with open_authoritative(path) as (stream, identity):
        if identity.st_size > max_bytes:
            raise GraphError(f"{path}: {identity.st_size} bytes exceeds the {max_bytes}-byte cap")
        payload = stream.read()
        if len(payload) != identity.st_size:
            raise GraphError(f"{path}: short read, file changed size while being read")
    return payload, hashlib.sha256(payload).hexdigest()


def load_authoritative_json(
    path: Path, *, expected_sha256: str | None = None, where: str | None = None
) -> tuple[dict[str, Any], str, bytes]:
    """Read, hash and strictly parse one JSON authority from a single descriptor.

    The parsed object comes from the same in-memory bytes that produced the digest, so an
    approved hash can never certify bytes other than the ones that were interpreted.
    """
    label = where or str(path)
    payload, digest = read_authoritative_bytes(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise GraphError(f"{label}: SHA-256 mismatch, expected {expected_sha256}, got {digest}")
    return strict_json_object(payload, where=label), digest, payload


# --------------------------------------------------------------------------- graph model


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
    graph_bytes: bytes = b""
    eligibility_rows: dict[str, np.ndarray] | None = None
    binding_identities: dict[str, FileIdentity] | None = None

    def validated_eligibility_rows(self, binding_id: str) -> np.ndarray:
        """The exact row set validated on the descriptor it was hashed from."""
        if self.eligibility_rows is None or binding_id not in self.eligibility_rows:
            raise GraphError(
                f"{binding_id}: eligibility rows were never validated; load the graph with "
                "verify_hashes=True before scanning"
            )
        return self.eligibility_rows[binding_id]


# --------------------------------------------------------------------------- schema fragments


def _parse_predicate(raw: Any, *, where: str) -> dict[str, Any]:
    _check_keys(raw, required={"kind"}, optional={"field", "value", "note"}, where=where)
    kind = _req_str(raw, "kind", where)
    _require(kind in PREDICATE_KINDS, f"{where}: predicate kind must be one of {PREDICATE_KINDS}")
    if kind == "INTEGER_SCORE_AT_LEAST":
        _require("field" in raw, f"{where}: integer-score predicate needs field")
        _req_str(raw, "field", where)
        _require("value" in raw, f"{where}: integer-score predicate needs value")
        _req_int(raw, "value", where)
    else:
        _require("value" not in raw, f"{where}: {kind} predicate must not carry a value")
    if "note" in raw:
        _req_str(raw, "note", where, nonempty=False)
    return dict(raw)


def _parse_branch(raw: Any, *, where: str, expect: str) -> dict[str, Any]:
    _check_keys(
        raw,
        required={"branch", "selection_mode", "candidate_predicate"},
        optional={"rank_order", "continuous_score_field"},
        where=where,
    )
    _require(_req_str(raw, "branch", where) == expect, f"{where}: branch must be {expect}")
    mode = _req_str(raw, "selection_mode", where)
    _require(mode in SELECTION_MODES, f"{where}: selection_mode must be one of {SELECTION_MODES}")
    _parse_predicate(raw["candidate_predicate"], where=f"{where}.candidate_predicate")
    if mode == "EXACT_SCORE_DESC_SHA_ASC":
        _require("continuous_score_field" in raw, f"{where}: needs continuous_score_field")
        _req_str(raw, "continuous_score_field", where)
        _require("rank_order" in raw, f"{where}: needs an explicit rank_order")
        order = _req_list(raw, "rank_order", where)
        _require(
            all(isinstance(item, str) for item in order), f"{where}.rank_order: must be strings"
        )
    else:
        _require(
            "continuous_score_field" not in raw,
            f"{where}: {mode} must not declare continuous_score_field",
        )
    return dict(raw)


def _parse_cleaning_contract(raw: Any, *, where: str) -> dict[str, Any]:
    """Closed cleaning contract with exact JSON types; nothing here is coerced later."""
    _check_keys(
        raw,
        required={
            "strip_leading_noise",
            "normalize_quotes",
            "underscores_policy",
            "min_chars",
            "min_ascii_ratio",
        },
        where=where,
    )
    policy = _req_str(raw, "underscores_policy", where)
    _require(
        policy in UNDERSCORES_POLICIES,
        f"{where}.underscores_policy: must be one of {UNDERSCORES_POLICIES}",
    )
    return {
        "strip_leading_noise": _req_bool(raw, "strip_leading_noise", where),
        "normalize_quotes": _req_bool(raw, "normalize_quotes", where),
        "underscores_policy": policy,
        "min_chars": _req_int(raw, "min_chars", where, minimum=0),
        "min_ascii_ratio": _req_number(raw, "min_ascii_ratio", where, minimum=0.0, maximum=1.0),
    }


def _parse_score_spec(raw: Any, *, where: str) -> dict[str, Any]:
    _check_keys(raw, required={"container", "key", "json_type"}, optional={"note"}, where=where)
    json_type = _req_str(raw, "json_type", where)
    _require(json_type in JSON_SCORE_TYPES, f"{where}.json_type: must be one of {JSON_SCORE_TYPES}")
    spec = {
        "container": _req_str(raw, "container", where),
        "key": _req_str(raw, "key", where),
        "json_type": json_type,
    }
    if "note" in raw:
        spec["note"] = _req_str(raw, "note", where, nonempty=False)
    return spec


def _parse_schema_accessor(raw: Any, *, where: str) -> dict[str, Any]:
    _check_keys(
        raw,
        required={"accessor_id"},
        optional={"integer_score", "continuous_score", "note"},
        where=where,
    )
    accessor: dict[str, Any] = {"accessor_id": _req_str(raw, "accessor_id", where)}
    for which in ("integer_score", "continuous_score"):
        if which in raw:
            accessor[which] = _parse_score_spec(raw[which], where=f"{where}.{which}")
    if "note" in raw:
        accessor["note"] = _req_str(raw, "note", where, nonempty=False)
    return accessor


# --------------------------------------------------------------------------- bindings


def _validate_binding(raw: Any, key: str) -> InputBinding:
    where = f"input_bindings[{key!r}]"
    _check_keys(
        raw,
        required={
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
        _req_str(raw, "input_binding_id", where) == key,
        f"{where}: input_binding_id must equal its registry key",
    )
    _require(
        _req_int(raw, "eligibility_index_element_width_bytes", where) == ELIGIBILITY_INDEX_WIDTH,
        f"{where}: element width must be {ELIGIBILITY_INDEX_WIDTH}",
    )
    _require(
        _req_str(raw, "eligibility_index_dtype", where) == ELIGIBILITY_INDEX_DTYPE,
        f"{where}: dtype must be {ELIGIBILITY_INDEX_DTYPE}",
    )
    total = _req_int(raw, "total_physical_rows", where, minimum=0)
    excluded = _req_int(raw, "excluded_rows", where, minimum=0)
    eligible = _req_int(raw, "expected_eligible_rows", where, minimum=0)
    _require(
        total - excluded == eligible,
        f"{where}: total_rows - excluded_rows != expected_eligible_rows",
    )
    binding = InputBinding(
        input_binding_id=key,
        release_key=_req_str(raw, "release_key", where),
        documents_path=Path(_req_str(raw, "documents_path", where)),
        documents_sha256=_req_hex64(raw, "documents_sha256", where),
        documents_size_bytes=_req_int(raw, "documents_size_bytes", where, minimum=0),
        release_manifest_path=Path(_req_str(raw, "release_manifest_path", where)),
        release_manifest_sha256=_req_hex64(raw, "release_manifest_sha256", where),
        eligibility_index_path=Path(_req_str(raw, "eligibility_index_path", where)),
        eligibility_index_sha256=_req_hex64(raw, "eligibility_index_sha256", where),
        total_physical_rows=total,
        excluded_rows=excluded,
        expected_eligible_rows=eligible,
        schema_accessor=_parse_schema_accessor(
            raw["schema_accessor"], where=f"{where}.schema_accessor"
        ),
        text_field=_req_str(raw, "text_field", where),
        cleaning_contract=_parse_cleaning_contract(
            raw["cleaning_contract"], where=f"{where}.cleaning_contract"
        ),
    )
    return binding


def verify_binding_inputs(
    binding: InputBinding,
) -> tuple[np.ndarray, FileIdentity]:
    """Fail closed unless the bound manifest and eligibility index are byte-for-byte as frozen.

    The document corpus is NOT re-hashed here. Its 151.6 GB of bytes are hashed exactly once, by
    the authoritative scan, on the same descriptor that consumes them, and compared against
    ``documents_sha256`` there. Hashing it here as well would both double the I/O and reintroduce
    a hash-then-reopen window between validation and use. What is checked here is the identity of
    the opened object: a regular, non-symlink file of exactly the frozen size.
    """
    where = binding.input_binding_id
    with open_authoritative(binding.documents_path) as (_stream, identity):
        _require(
            identity.st_size == binding.documents_size_bytes,
            f"{where}: documents size {identity.st_size} != frozen {binding.documents_size_bytes}",
        )
    load_authoritative_json(
        binding.release_manifest_path,
        expected_sha256=binding.release_manifest_sha256,
        where=f"{where}: release manifest",
    )
    rows = validate_eligibility_index(binding)
    return rows, identity


def validate_eligibility_index(binding: InputBinding) -> np.ndarray:
    """Validate an eligibility row-index set by content, on the bytes it is decoded from."""
    where = binding.input_binding_id
    payload, digest = read_authoritative_bytes(binding.eligibility_index_path)
    _require(digest == binding.eligibility_index_sha256, f"{where}: eligibility index SHA mismatch")
    _require(
        len(payload) % ELIGIBILITY_INDEX_WIDTH == 0,
        f"{where}: eligibility index length {len(payload)} is not a multiple of "
        f"{ELIGIBILITY_INDEX_WIDTH}",
    )
    rows = np.frombuffer(payload, dtype=ELIGIBILITY_INDEX_DTYPE)
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


# --------------------------------------------------------------------------- graph loading


def _validate_graph_constants(raw: dict[str, Any]) -> None:
    """Mechanically enforce the frozen graph policy constants, not merely their presence."""
    _require(
        _req_str(raw, "schema_version", "source graph") == GRAPH_SCHEMA_VERSION,
        "source graph: wrong schema_version",
    )
    _require(
        _req_str(raw, "policy_status", "source graph") == "OWNER_FROZEN",
        "source graph: policy_status must be OWNER_FROZEN",
    )
    _require(
        _req_bool(raw, "resume_supported", "source graph") is False,
        "source graph: resume must be disabled",
    )
    for key in (
        "authority",
        "date",
        "note",
        "execution_order_rule",
        "ownership_rule",
        "stage_a_population_rule",
    ):
        _req_str(raw, key, "source graph")

    priority = _req_obj(raw, "stage_priority", "source graph")
    _require(
        priority == STAGE_PRIORITY,
        f"source graph: stage_priority must be exactly {STAGE_PRIORITY}",
    )
    modes = _req_list(raw, "selection_modes_closed_enum", "source graph")
    _require(
        modes == list(SELECTION_MODES),
        f"source graph: selection_modes_closed_enum must be exactly {list(SELECTION_MODES)}",
    )

    boundary = _req_obj(raw, "h_boundary", "source graph")
    _check_keys(
        boundary,
        required={"h_canonical_output_label", "h_publishes_physical_views"},
        where="h_boundary",
    )
    _require(
        _req_str(boundary, "h_canonical_output_label", "h_boundary") == H_CANONICAL_OUTPUT_LABEL,
        f"h_boundary: label must be {H_CANONICAL_OUTPUT_LABEL}",
    )
    _require(
        _req_bool(boundary, "h_publishes_physical_views", "h_boundary") is False,
        "h_boundary: H must not publish physical candidate views",
    )

    namespace = _req_obj(raw, "control_namespace", "source graph")
    _check_keys(namespace, required={"exists", "note"}, where="control_namespace")
    _require(
        _req_bool(namespace, "exists", "control_namespace") is False,
        "control_namespace: graph v1 assigns no control semantics",
    )
    _req_str(namespace, "note", "control_namespace")

    authorities = _req_obj(raw, "bound_authorities", "source graph")
    _check_keys(authorities, required=set(BOUND_AUTHORITY_KEYS), where="bound_authorities")
    for key in sorted(authorities):
        _req_hex64(authorities, key, "bound_authorities")


def _validate_structured_tutorial(
    raw: dict[str, Any], nodes: list[Node], bindings: dict[str, Any]
) -> None:
    """The structured_tutorial union must remain exactly one logical node, with no sub-targets."""
    block = _req_obj(raw, "structured_tutorial", "source graph")
    _check_keys(
        block,
        required={
            "input_binding_ids",
            "input_binding_order_source",
            "realization",
            "selection_mode",
            "sub_targets",
        },
        where="structured_tutorial",
    )
    _require(
        _req_str(block, "realization", "structured_tutorial") == "SINGLE_LOGICAL_NODE",
        "structured_tutorial: realization must be SINGLE_LOGICAL_NODE",
    )
    _require(
        block["sub_targets"] is None,
        "structured_tutorial: sub_targets must be null; a union node has one target",
    )
    _req_str(block, "input_binding_order_source", "structured_tutorial")
    mode = _req_str(block, "selection_mode", "structured_tutorial")
    _require(mode in SELECTION_MODES, "structured_tutorial: selection_mode outside the closed enum")
    ids = _req_list(block, "input_binding_ids", "structured_tutorial")
    _require(
        all(isinstance(item, str) for item in ids),
        "structured_tutorial.input_binding_ids: must be strings",
    )
    _require(len(set(ids)) == len(ids), "structured_tutorial: duplicate input_binding_id")
    for binding_id in ids:
        _require(
            binding_id in bindings,
            f"structured_tutorial: unknown input_binding_id {binding_id!r}",
        )
    owning = [node for node in nodes if set(node.input_binding_ids) & set(ids)]
    _require(
        len(owning) <= 1,
        "structured_tutorial: SINGLE_LOGICAL_NODE realization forbids splitting its bindings "
        f"across nodes, found {[node.source_id for node in owning]}",
    )
    if owning:
        _require(
            list(owning[0].input_binding_ids) == list(ids),
            "structured_tutorial: the owning node must consume exactly the declared bindings in "
            "the declared order",
        )
        _require(
            owning[0].selection_mode == mode,
            "structured_tutorial: the owning node's selection_mode disagrees with the policy block",
        )


def load_source_graph(
    path: Path,
    *,
    verify_hashes: bool = True,
    expected_graph_sha256: str | None = None,
) -> SourceGraph:
    """Load and strictly validate the owner-frozen graph.

    ``expected_graph_sha256`` binds the load to specific owner bytes. Production authorisation
    always supplies it; leaving it unset is a development and fixture affordance only.
    """
    raw_bytes, graph_sha256 = read_authoritative_bytes(path)
    if expected_graph_sha256 is not None and graph_sha256 != expected_graph_sha256:
        raise GraphError(
            f"source graph: SHA-256 mismatch, expected {expected_graph_sha256}, got {graph_sha256}"
        )
    raw = strict_json_object(raw_bytes, where="source graph")
    _check_keys(
        raw,
        required={
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
    _validate_graph_constants(raw)

    seed_block = _req_obj(raw, "selection_seed", "source graph")
    _check_keys(
        seed_block,
        required={"domain_utf8", "domain_sha256", "derivation", "seed", "seed_hex"},
        where="selection_seed",
    )
    seed = _req_int(seed_block, "seed", "selection_seed", minimum=0)
    domain = _req_str(seed_block, "domain_utf8", "selection_seed")
    _req_str(seed_block, "derivation", "selection_seed")
    derived = derive_seed(domain)
    _require(
        derived == seed,
        f"selection_seed: derivation mismatch, derived {derived} != declared {seed}",
    )
    _require(
        hashlib.sha256(domain.encode("utf-8")).hexdigest()
        == _req_hex64(seed_block, "domain_sha256", "selection_seed"),
        "selection_seed: domain_sha256 mismatch",
    )
    _require(
        _req_str(seed_block, "seed_hex", "selection_seed") == hex(seed),
        "selection_seed: seed_hex disagrees with seed",
    )

    raw_bindings = _req_obj(raw, "input_bindings", "source graph")
    _require(raw_bindings, "source graph: input_bindings must not be empty")
    bindings: dict[str, InputBinding] = {}
    eligibility_rows: dict[str, np.ndarray] = {}
    identities: dict[str, FileIdentity] = {}
    seen_paths: dict[str, str] = {}
    seen_objects: dict[tuple[int, int], str] = {}
    for key in sorted(raw_bindings):
        binding = _validate_binding(raw_bindings[key], key)
        resolved = (
            str(binding.documents_path.resolve())
            if binding.documents_path.exists()
            else str(binding.documents_path)
        )
        prior = seen_paths.get(resolved)
        _require(
            prior is None,
            f"input_bindings: resolved documents path {resolved!r} registered under both "
            f"{prior!r} and {key!r}",
        )
        seen_paths[resolved] = key
        if verify_hashes:
            rows, identity = verify_binding_inputs(binding)
            object_key = (identity.st_dev, identity.st_ino)
            prior_object = seen_objects.get(object_key)
            _require(
                prior_object is None,
                f"input_bindings: bindings {prior_object!r} and {key!r} name the same underlying "
                f"file object (device {identity.st_dev}, inode {identity.st_ino}); a hard link is "
                "not a distinct release",
            )
            seen_objects[object_key] = key
            eligibility_rows[key] = rows
            identities[key] = identity
        bindings[key] = binding

    raw_nodes = _req_list(raw, "nodes", "source graph")
    nodes: list[Node] = []
    seen_ids: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        where = f"nodes[{index}]"
        _check_keys(
            raw_node,
            required={
                "node_id",
                "source_id",
                "stage",
                "stage_priority",
                "target_serialized_tokens",
                "input_binding_ids",
                "selection_mode",
                "candidate_predicate",
            },
            optional={"branch_primary", "branch_fallback"},
            where=where,
        )
        source_id = _req_str(raw_node, "source_id", where)
        _require(
            _req_str(raw_node, "node_id", where) == source_id,
            f"{where}: node_id must equal source_id",
        )
        _require(source_id not in seen_ids, f"{where}: duplicate source_id {source_id!r}")
        seen_ids.add(source_id)
        stage = _req_str(raw_node, "stage", where)
        _require(stage in STAGE_PRIORITY, f"{where}: stage must be one of {sorted(STAGE_PRIORITY)}")
        _require(
            _req_int(raw_node, "stage_priority", where) == STAGE_PRIORITY[stage],
            f"{where}: stage_priority disagrees with stage",
        )
        target = _req_int(raw_node, "target_serialized_tokens", where, minimum=1)
        ids = _req_list(raw_node, "input_binding_ids", where)
        _require(
            all(isinstance(item, str) for item in ids),
            f"{where}.input_binding_ids: must be strings",
        )
        _require(len(set(ids)) == len(ids), f"{where}: duplicate input_binding_id in node")
        for binding_id in ids:
            _require(binding_id in bindings, f"{where}: unknown input_binding_id {binding_id!r}")
        mode = _req_str(raw_node, "selection_mode", where)
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
                node_id=source_id,
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
    ordered = sorted(nodes, key=lambda node: node.priority_key)
    _require(
        [node.source_id for node in ordered] == [node.source_id for node in nodes],
        "source graph: nodes are not stored in ascending (stage_priority, source_id) order",
    )
    _validate_structured_tutorial(raw, nodes, bindings)

    return SourceGraph(
        seed=seed,
        nodes=tuple(ordered),
        bindings=bindings,
        bound_authorities=dict(raw["bound_authorities"]),
        raw=raw,
        graph_sha256=graph_sha256,
        graph_bytes=raw_bytes,
        eligibility_rows=eligibility_rows if verify_hashes else None,
        binding_identities=identities if verify_hashes else None,
    )


__all__ = [
    "BOUND_AUTHORITY_KEYS",
    "BRANCH_DEPENDENT",
    "FileIdentity",
    "GRAPH_SCHEMA_VERSION",
    "GraphError",
    "H_CANONICAL_OUTPUT_LABEL",
    "InputBinding",
    "Node",
    "SELECTION_MODES",
    "STAGE_PRIORITY",
    "SourceGraph",
    "StrictJSONError",
    "canonical_json_bytes",
    "derive_seed",
    "load_authoritative_json",
    "load_source_graph",
    "open_authoritative",
    "read_authoritative_bytes",
    "sha256_file",
    "sha256_stream",
    "strict_json_loads",
    "strict_json_object",
    "validate_eligibility_index",
    "verify_binding_inputs",
]
