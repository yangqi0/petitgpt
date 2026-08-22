#!/usr/bin/env python3
"""Pure committed-prefix selection core shared by Stage-H feasibility replay and Stage-I realization.

The point of sharing one core is that Stage H's prediction and Stage I's authoritative run are the
same function of the same inputs, so their fingerprints must agree exactly. Stage H feeds it
metadata sidecar records; Stage I feeds it records derived from the same frozen release bytes.

Four properties this module exists to get right, each one a rejected defect in earlier tooling:

* ownership transfers only on actual commit, never on mere occurrence;
* the whole-document prefix crossing is exact, so overshoot is measured rather than bounded;
* continuous-score ordering is over exact IEEE-754 binary64 values, never a quantised bucket;
* duplicate identities collapse to the representative selector v1 would have kept, which is the
  minimum ``(raw_sha256, input_record_sha256)`` of the duplicate group and contains no positional
  component at all. Because that key ignores file, ordinal, row index and arrival order, the
  minimum over a union of ordered input bindings is the same object v1 would have retained had
  the union been one physical file in any concatenation order.

Every value that can move a branch, a mass or a fingerprint is validated at construction and
again at the replay boundary, so a caller reaching the pure API directly cannot inject a NaN, an
infinity or a mistyped score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import hashlib
import math
import struct
from typing import Any

from pretrain.stage_i_graph_v2 import BRANCH_DEPENDENT, GraphError, Node, SourceGraph

SELECTION_DOMAIN = b"PetitGPT-pretrain-selection-v1\0"
FINGERPRINT_DOMAIN = b"PetitGPT-stage-i-selection-fingerprint-v1\0"
NODE_RESULT_SCHEMA = "petitgpt-stage-i-node-result-v2"
REPRESENTATIVE_RULE = "selector-v1 min(raw_sha256, input_record_sha256)"
NODE_RESULT_FIELDS = frozenset({
    "canonical_schema_version",
    "source_id",
    "stage",
    "target_serialized_tokens",
    "branch",
    "selection_mode",
    "pre_exclusion_unique_identities",
    "g2_excluded_identities",
    "prior_commit_excluded_identities",
    "exclusions_by_owner",
    "post_exclusion_candidate_identities",
    "post_exclusion_candidate_serialized_tokens",
    "selected_identities",
    "selected_serialized_tokens",
    "crossing_identity",
    "crossing_document_serialized_tokens",
    "actual_overshoot_tokens",
    "residual_identities",
    "residual_serialized_tokens",
    "selection_fingerprint",
    "feasible",
    "boundary_evidence",
})
BOUNDARY_EVIDENCE_FIELDS = (
    "representative_rule",
    "crossing_selection_rank",
    "crossing_score_bits_hex",
    "crossing_score_hex",
    "next_unselected_identity",
    "next_unselected_serialized_tokens",
    "next_unselected_selection_rank",
    "next_unselected_score_bits_hex",
    "next_unselected_score_hex",
)
_HEX64 = frozenset("0123456789abcdef")
_UINT64_LIMIT = 1 << 64


class ReplayError(RuntimeError):
    """Fail-closed replay condition."""


def selection_rank(*, seed: int, stage: str, source_id: str, canonical_fingerprint: str) -> str:
    """Byte-identical to the frozen selector v1 rank, so SEEDED_HASH mode reproduces it exactly."""
    digest = hashlib.sha256()
    digest.update(SELECTION_DOMAIN)
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(stage.encode("ascii"))
    digest.update(b"\0")
    digest.update(source_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(canonical_fingerprint))
    return digest.hexdigest()


def score_to_bits(value: Any) -> int:
    """Lossless binary64 representation. Rejects bool, non-numeric, NaN and infinities."""
    if isinstance(value, bool):
        raise ReplayError("continuous score must not be a boolean")
    if not isinstance(value, (int, float)):
        raise ReplayError(f"continuous score must be numeric, got {type(value).__name__}")
    as_float = float(value)
    if math.isnan(as_float):
        raise ReplayError("continuous score must not be NaN")
    if math.isinf(as_float):
        raise ReplayError("continuous score must be finite")
    return int.from_bytes(struct.pack(">d", as_float), "big")


def bits_to_score(bits: Any) -> float:
    """Decode binary64 bits, refusing anything that is not an exact finite 64-bit pattern."""
    if isinstance(bits, bool):
        raise ReplayError("score bits must not be a boolean")
    if not isinstance(bits, int):
        raise ReplayError(f"score bits must be an integer, got {type(bits).__name__}")
    if not 0 <= bits < _UINT64_LIMIT:
        raise ReplayError("score bits must be an unsigned 64-bit value")
    value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
    if math.isnan(value):
        raise ReplayError("score bits decode to NaN")
    if math.isinf(value):
        raise ReplayError("score bits decode to an infinity")
    return value


def _require_hex64(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX64:
        raise ReplayError(f"{what} must be a lowercase 64-character SHA-256 hex digest")
    return value


def representative_key(*, raw_sha256: str, input_record_sha256: str) -> bytes:
    """Pack selector v1's representative key into 64 comparable bytes.

    Lexicographic order over the packed bytes is identical to lexicographic order over the
    ``(raw_sha256, input_record_sha256)`` hex pair, because lowercase hex is an order-preserving
    encoding of the underlying digest bytes. Packing costs one object per record instead of two
    64-character strings, which matters at thirty million records.
    """
    return bytes.fromhex(_require_hex64(raw_sha256, "raw_sha256")) + bytes.fromhex(
        _require_hex64(input_record_sha256, "input_record_sha256")
    )


def representative_key_of(raw_text: str, canonical_record_sha256: str) -> bytes:
    """Convenience wrapper for callers holding the raw text rather than its digest."""
    return representative_key(
        raw_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        input_record_sha256=canonical_record_sha256,
    )


@dataclass(frozen=True)
class CandidateRecord:
    """One eligible logical document, as metadata only. Never carries document text.

    ``representative_key`` is selector v1's duplicate representative key, packed. It is required:
    a record that cannot say which duplicate v1 would have kept cannot be deduplicated correctly,
    and defaulting it would silently reintroduce a positional rule.
    """

    input_binding_id: str
    row_index: int
    cleaned_sha256: str
    canonical_fingerprint: str
    serialized_tokens: int
    representative_key: bytes
    int_score: int | None = None
    score_bits: int | None = None

    def __post_init__(self) -> None:
        validate_candidate_record(self)

    @property
    def raw_sha256(self) -> str:
        return self.representative_key[:32].hex()

    @property
    def input_record_sha256(self) -> str:
        return self.representative_key[32:].hex()


def validate_candidate_record(record: CandidateRecord) -> None:
    """Every invariant a record must satisfy, checked at construction and at replay entry.

    Re-checking at the boundary is deliberate: a frozen dataclass can still be mutated through
    ``object.__setattr__``, so construction-time validation alone is not a guarantee about the
    object that actually reaches selection.
    """
    if not isinstance(record.input_binding_id, str) or not record.input_binding_id:
        raise ReplayError("input_binding_id must be a non-empty string")
    if isinstance(record.row_index, bool) or not isinstance(record.row_index, int):
        raise ReplayError("row_index must be an exact integer")
    if record.row_index < 0:
        raise ReplayError("row_index must not be negative")
    _require_hex64(record.cleaned_sha256, "cleaned_sha256")
    _require_hex64(record.canonical_fingerprint, "canonical_fingerprint")
    if isinstance(record.serialized_tokens, bool) or not isinstance(record.serialized_tokens, int):
        raise ReplayError("serialized_tokens must be an exact integer")
    if record.serialized_tokens <= 0:
        raise ReplayError("serialized_tokens must be positive")
    if not isinstance(record.representative_key, bytes) or len(record.representative_key) != 64:
        raise ReplayError(
            "representative_key must be the 64 packed bytes of selector v1's "
            "(raw_sha256, input_record_sha256)"
        )
    if record.int_score is not None:
        if isinstance(record.int_score, bool) or not isinstance(record.int_score, int):
            raise ReplayError("int_score must be an exact integer or None")
    if record.score_bits is not None:
        bits_to_score(record.score_bits)


@dataclass(frozen=True)
class NodeResult:
    source_id: str
    stage: str
    target_serialized_tokens: int
    branch: str
    selection_mode: str
    pre_exclusion_unique_identities: int
    g2_excluded_identities: int
    prior_commit_excluded_identities: int
    exclusions_by_owner: dict[str, int]
    post_exclusion_candidate_identities: int
    post_exclusion_candidate_serialized_tokens: int
    selected_identities: int
    selected_serialized_tokens: int
    crossing_identity: str | None
    crossing_document_serialized_tokens: int | None
    actual_overshoot_tokens: int
    residual_identities: int
    residual_serialized_tokens: int
    selection_fingerprint: str
    feasible: bool
    boundary_evidence: dict[str, Any] = field(default_factory=dict)

    #: Bumped whenever the canonical projection below gains, loses or renames a field.
    CANONICAL_SCHEMA_VERSION = NODE_RESULT_SCHEMA

    def as_canonical(self) -> dict[str, Any]:
        """Explicit, versioned, telemetry-free projection.

        The field list is written out rather than derived from ``self.__dict__`` so that a future
        cache, timing or debugging attribute cannot silently enter hash-bearing output. Adding a
        field to the contract requires editing this list and the schema version together.
        """
        return {
            "canonical_schema_version": NODE_RESULT_SCHEMA,
            "source_id": self.source_id,
            "stage": self.stage,
            "target_serialized_tokens": self.target_serialized_tokens,
            "branch": self.branch,
            "selection_mode": self.selection_mode,
            "pre_exclusion_unique_identities": self.pre_exclusion_unique_identities,
            "g2_excluded_identities": self.g2_excluded_identities,
            "prior_commit_excluded_identities": self.prior_commit_excluded_identities,
            "exclusions_by_owner": dict(sorted(self.exclusions_by_owner.items())),
            "post_exclusion_candidate_identities": self.post_exclusion_candidate_identities,
            "post_exclusion_candidate_serialized_tokens": (
                self.post_exclusion_candidate_serialized_tokens
            ),
            "selected_identities": self.selected_identities,
            "selected_serialized_tokens": self.selected_serialized_tokens,
            "crossing_identity": self.crossing_identity,
            "crossing_document_serialized_tokens": self.crossing_document_serialized_tokens,
            "actual_overshoot_tokens": self.actual_overshoot_tokens,
            "residual_identities": self.residual_identities,
            "residual_serialized_tokens": self.residual_serialized_tokens,
            "selection_fingerprint": self.selection_fingerprint,
            "feasible": self.feasible,
            "boundary_evidence": _canonical_boundary_evidence(self.boundary_evidence),
        }


def _canonical_boundary_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Explicit projection of the rank-boundary evidence; unknown keys never reach output."""
    return {name: evidence.get(name) for name in BOUNDARY_EVIDENCE_FIELDS}


def _representatives(records: Iterable[CandidateRecord]) -> dict[str, CandidateRecord]:
    """Collapse duplicates exactly as selector v1 does, independent of iteration order."""
    chosen: dict[str, CandidateRecord] = {}
    for record in records:
        prior = chosen.get(record.cleaned_sha256)
        if prior is None:
            chosen[record.cleaned_sha256] = record
            continue
        if record.canonical_fingerprint != prior.canonical_fingerprint:
            raise ReplayError(
                f"identity collision: {record.cleaned_sha256} maps to two canonical fingerprints"
            )
        if record.representative_key < prior.representative_key:
            chosen[record.cleaned_sha256] = record
    return chosen


def _predicate_matches(record: CandidateRecord, predicate: dict[str, Any]) -> bool:
    kind = predicate["kind"]
    if kind == "ALL_ELIGIBLE":
        return True
    if kind == "INTEGER_SCORE_AT_LEAST":
        if record.int_score is None:
            raise ReplayError(
                "integer-score predicate requires an integer score on every candidate"
            )
        threshold = predicate["value"]
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise ReplayError("integer-score predicate value must be an exact integer")
        return record.int_score >= threshold
    raise ReplayError(f"unsupported predicate kind {kind!r}")


def _order_key(record: CandidateRecord, *, mode: str, seed: int, node: Node):
    if mode == "SEEDED_HASH":
        rank = selection_rank(
            seed=seed,
            stage=node.stage,
            source_id=node.source_id,
            canonical_fingerprint=record.canonical_fingerprint,
        )
        return (rank, record.canonical_fingerprint)
    if mode == "EXACT_SCORE_DESC_SHA_ASC":
        if record.score_bits is None:
            raise ReplayError("score-ranked mode requires a continuous score on every candidate")
        # Descending by exact binary64 VALUE, then cleaned SHA ascending. The frozen rank_order is
        # numeric, so +0.0 and -0.0 compare equal here and the SHA breaks that tie, exactly as it
        # breaks a tie between two identical bit patterns. Sorting raw bit patterns instead would
        # order -0.0 after every positive value and invert the whole negative range.
        return (-bits_to_score(record.score_bits), record.cleaned_sha256)
    raise ReplayError(f"unsupported selection mode {mode!r}")


def _available(
    universe: dict[str, CandidateRecord],
    predicate: dict[str, Any],
    reference_exclusion: set[str],
    committed_owner: dict[str, str],
) -> tuple[list[CandidateRecord], int, int, dict[str, int]]:
    """Apply the node predicate, then the G2 and prior-commit exclusions, in that order.

    Ownership is consulted only through ``committed_owner``, which contains an identity solely
    because some earlier node actually selected and committed it.
    """
    matched = [r for r in universe.values() if _predicate_matches(r, predicate)]
    g2 = sum(1 for r in matched if r.cleaned_sha256 in reference_exclusion)
    by_owner: dict[str, int] = {}
    available: list[CandidateRecord] = []
    for record in matched:
        if record.cleaned_sha256 in reference_exclusion:
            continue
        owner = committed_owner.get(record.cleaned_sha256)
        if owner is not None:
            by_owner[owner] = by_owner.get(owner, 0) + 1
            continue
        available.append(record)
    return available, len(matched), g2, by_owner


def _fingerprint(identities: list[str]) -> str:
    digest = hashlib.sha256(FINGERPRINT_DOMAIN)
    digest.update(len(identities).to_bytes(8, "big"))
    for value in sorted(identities):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _select(
    candidates: list[CandidateRecord], *, node: Node, mode: str, seed: int, target: int
) -> tuple[list[CandidateRecord], CandidateRecord | None, CandidateRecord | None]:
    """Whole-document prefix whose serialized mass first reaches or exceeds the target.

    Also returns the first record that was ranked but not taken, which is the other half of the
    rank boundary and the only way an auditor can check the cut without rescanning the corpus.
    """
    ordered = sorted(candidates, key=lambda r: _order_key(r, mode=mode, seed=seed, node=node))
    picked: list[CandidateRecord] = []
    crossing: CandidateRecord | None = None
    mass = 0
    for record in ordered:
        if mass >= target:
            break
        picked.append(record)
        mass += record.serialized_tokens
        if mass >= target:
            crossing = record
    next_unselected = ordered[len(picked)] if len(ordered) > len(picked) else None
    return picked, crossing, next_unselected


def _boundary_evidence(
    *,
    mode: str,
    seed: int,
    node: Node,
    crossing: CandidateRecord | None,
    next_unselected: CandidateRecord | None,
) -> dict[str, Any]:
    """Compact, exact, lossless evidence for both sides of the rank cut."""

    def rank_of(record: CandidateRecord | None) -> str | None:
        if record is None or mode != "SEEDED_HASH":
            return None
        return selection_rank(
            seed=seed,
            stage=node.stage,
            source_id=node.source_id,
            canonical_fingerprint=record.canonical_fingerprint,
        )

    def bits_hex(record: CandidateRecord | None) -> str | None:
        if record is None or record.score_bits is None:
            return None
        return f"{record.score_bits:016x}"

    def float_hex(record: CandidateRecord | None) -> str | None:
        if record is None or record.score_bits is None:
            return None
        return bits_to_score(record.score_bits).hex()

    return {
        "representative_rule": REPRESENTATIVE_RULE,
        "crossing_selection_rank": rank_of(crossing),
        "crossing_score_bits_hex": bits_hex(crossing),
        "crossing_score_hex": float_hex(crossing),
        "next_unselected_identity": next_unselected.cleaned_sha256 if next_unselected else None,
        "next_unselected_serialized_tokens": (
            next_unselected.serialized_tokens if next_unselected else None
        ),
        "next_unselected_selection_rank": rank_of(next_unselected),
        "next_unselected_score_bits_hex": bits_hex(next_unselected),
        "next_unselected_score_hex": float_hex(next_unselected),
    }


def replay(
    graph: SourceGraph,
    records_by_binding: dict[str, list[CandidateRecord]],
    reference_exclusion: set[str],
) -> list[NodeResult]:
    """Replay the frozen graph, committing only what each node actually selects."""
    consumed = {b for node in graph.nodes for b in node.input_binding_ids}
    for binding_id in sorted(consumed):
        if binding_id not in records_by_binding:
            raise ReplayError(f"missing candidate records for input binding {binding_id!r}")
    for binding_id, records in records_by_binding.items():
        if binding_id not in graph.bindings:
            raise ReplayError(f"records supplied for unknown input binding {binding_id!r}")
        if not isinstance(records, list):
            raise ReplayError(f"candidate records for {binding_id!r} must be a list")
        for record in records:
            if not isinstance(record, CandidateRecord):
                raise ReplayError(f"{binding_id!r}: candidates must be CandidateRecord instances")
            validate_candidate_record(record)
            if record.input_binding_id != binding_id:
                raise ReplayError(
                    f"{binding_id!r}: record claims input binding {record.input_binding_id!r}"
                )
    if not isinstance(reference_exclusion, (set, frozenset)):
        raise ReplayError("reference_exclusion must be a set of cleaned identities")

    committed_owner: dict[str, str] = {}
    results: list[NodeResult] = []

    for node in graph.nodes:
        # The representative key carries no positional component, so pooling the bindings of a
        # union node needs no per-node copy and no ordinal: iterating them in order is enough.
        pooled: list[CandidateRecord] = []
        for binding_id in node.input_binding_ids:
            pooled.extend(records_by_binding[binding_id])
        universe = _representatives(pooled)
        del pooled

        if node.selection_mode == BRANCH_DEPENDENT:
            primary = node.branch_primary
            available, matched_n, g2_n, by_owner = _available(
                universe, primary["candidate_predicate"], reference_exclusion, committed_owner
            )
            capacity = sum(r.serialized_tokens for r in available)
            if capacity >= node.target_serialized_tokens:
                branch, mode = "PRIMARY_GE4", primary["selection_mode"]
            else:
                fallback = node.branch_fallback
                branch, mode = "FALLBACK_RANKED_GE3", fallback["selection_mode"]
                available, matched_n, g2_n, by_owner = _available(
                    universe, fallback["candidate_predicate"], reference_exclusion, committed_owner
                )
        else:
            branch, mode = "ORDINARY", node.selection_mode
            available, matched_n, g2_n, by_owner = _available(
                universe, node.candidate_predicate, reference_exclusion, committed_owner
            )

        capacity = sum(r.serialized_tokens for r in available)
        picked, crossing, next_unselected = _select(
            available, node=node, mode=mode, seed=graph.seed, target=node.target_serialized_tokens
        )
        selected_mass = sum(r.serialized_tokens for r in picked)
        feasible = selected_mass >= node.target_serialized_tokens
        for record in picked:
            committed_owner[record.cleaned_sha256] = node.source_id

        results.append(
            NodeResult(
                source_id=node.source_id,
                stage=node.stage,
                target_serialized_tokens=node.target_serialized_tokens,
                branch=branch,
                selection_mode=mode,
                pre_exclusion_unique_identities=matched_n,
                g2_excluded_identities=g2_n,
                prior_commit_excluded_identities=sum(by_owner.values()),
                exclusions_by_owner=dict(sorted(by_owner.items())),
                post_exclusion_candidate_identities=len(available),
                post_exclusion_candidate_serialized_tokens=capacity,
                selected_identities=len(picked),
                selected_serialized_tokens=selected_mass,
                crossing_identity=crossing.cleaned_sha256 if crossing else None,
                crossing_document_serialized_tokens=crossing.serialized_tokens
                if crossing
                else None,
                actual_overshoot_tokens=(
                    selected_mass - node.target_serialized_tokens if feasible else 0
                ),
                residual_identities=len(available) - len(picked),
                residual_serialized_tokens=capacity - selected_mass,
                selection_fingerprint=_fingerprint([r.cleaned_sha256 for r in picked]),
                feasible=feasible,
                boundary_evidence=_boundary_evidence(
                    mode=mode,
                    seed=graph.seed,
                    node=node,
                    crossing=crossing,
                    next_unselected=next_unselected,
                ),
            )
        )
    return results


def ownership_matrix(results: list[NodeResult]) -> dict[str, dict[str, int]]:
    return {r.source_id: dict(r.exclusions_by_owner) for r in results if r.exclusions_by_owner}


__all__ = [
    "BOUNDARY_EVIDENCE_FIELDS",
    "CandidateRecord",
    "GraphError",
    "NODE_RESULT_SCHEMA",
    "NODE_RESULT_FIELDS",
    "NodeResult",
    "REPRESENTATIVE_RULE",
    "ReplayError",
    "bits_to_score",
    "ownership_matrix",
    "replay",
    "representative_key",
    "representative_key_of",
    "score_to_bits",
    "selection_rank",
    "validate_candidate_record",
]
