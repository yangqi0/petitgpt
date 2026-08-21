#!/usr/bin/env python3
"""Pure committed-prefix selection core shared by Stage-H feasibility replay and Stage-I realization.

The point of sharing one core is that Stage H's prediction and Stage I's authoritative run are the
same function of the same inputs, so their fingerprints must agree exactly. Stage H feeds it
metadata sidecar records; Stage I feeds it records derived from the same frozen release bytes.

Three properties this module exists to get right, each one a rejected defect in the v1 tooling:

* ownership transfers only on actual commit, never on mere occurrence;
* the whole-document prefix crossing is exact, so overshoot is measured rather than bounded;
* continuous-score ordering is over exact IEEE-754 binary64 values, never a quantised bucket.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import struct
from typing import Any

from pretrain.stage_i_graph_v2 import BRANCH_DEPENDENT, GraphError, Node, SourceGraph

SELECTION_DOMAIN = b"PetitGPT-pretrain-selection-v1\0"


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
    if as_float != as_float:
        raise ReplayError("continuous score must not be NaN")
    if as_float in (float("inf"), float("-inf")):
        raise ReplayError("continuous score must be finite")
    return int.from_bytes(struct.pack(">d", as_float), "big")


def bits_to_score(bits: int) -> float:
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


@dataclass(frozen=True)
class CandidateRecord:
    """One eligible logical document, as metadata only. Never carries document text."""

    input_binding_id: str
    binding_ordinal: int
    row_index: int
    cleaned_sha256: str
    canonical_fingerprint: str
    serialized_tokens: int
    int_score: int | None = None
    score_bits: int | None = None

    @property
    def representative_key(self) -> tuple[int, int]:
        return self.binding_ordinal, self.row_index


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

    def as_canonical(self) -> dict[str, Any]:
        """Canonical, telemetry-free projection suitable for hashing."""
        return {k: v for k, v in self.__dict__.items()}


def _representatives(records: Iterable[CandidateRecord]) -> dict[str, CandidateRecord]:
    """One deterministic representative per cleaned identity, independent of iteration order."""
    chosen: dict[str, CandidateRecord] = {}
    for record in records:
        prior = chosen.get(record.cleaned_sha256)
        if prior is None or record.representative_key < prior.representative_key:
            chosen[record.cleaned_sha256] = record
        elif record.canonical_fingerprint != prior.canonical_fingerprint:
            raise ReplayError(
                f"identity collision: {record.cleaned_sha256} maps to two canonical fingerprints"
            )
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
        return record.int_score >= int(predicate["value"])
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
        # Descending by exact binary64 value; ties (identical bits) broken by cleaned SHA ascending.
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
    digest = hashlib.sha256(b"PetitGPT-stage-i-selection-fingerprint-v1\0")
    digest.update(len(identities).to_bytes(8, "big"))
    for value in sorted(identities):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _select(
    candidates: list[CandidateRecord], *, node: Node, mode: str, seed: int, target: int
) -> tuple[list[CandidateRecord], CandidateRecord | None]:
    """Whole-document prefix whose serialized mass first reaches or exceeds the target."""
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
    return picked, crossing


def replay(
    graph: SourceGraph,
    records_by_binding: dict[str, list[CandidateRecord]],
    reference_exclusion: set[str],
) -> list[NodeResult]:
    """Replay the frozen graph, committing only what each node actually selects."""
    for binding_id in graph.bindings:
        if binding_id not in records_by_binding:
            raise ReplayError(f"missing candidate records for input binding {binding_id!r}")

    committed_owner: dict[str, str] = {}
    results: list[NodeResult] = []

    for node in graph.nodes:
        pooled: list[CandidateRecord] = []
        for ordinal, binding_id in enumerate(node.input_binding_ids):
            for record in records_by_binding[binding_id]:
                pooled.append(
                    CandidateRecord(
                        input_binding_id=record.input_binding_id,
                        binding_ordinal=ordinal,
                        row_index=record.row_index,
                        cleaned_sha256=record.cleaned_sha256,
                        canonical_fingerprint=record.canonical_fingerprint,
                        serialized_tokens=record.serialized_tokens,
                        int_score=record.int_score,
                        score_bits=record.score_bits,
                    )
                )
        universe = _representatives(pooled)

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
        picked, crossing = _select(
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
            )
        )
    return results


def ownership_matrix(results: list[NodeResult]) -> dict[str, dict[str, int]]:
    return {r.source_id: dict(r.exclusions_by_owner) for r in results if r.exclusions_by_owner}


__all__ = [
    "CandidateRecord",
    "GraphError",
    "NodeResult",
    "ReplayError",
    "bits_to_score",
    "ownership_matrix",
    "replay",
    "score_to_bits",
    "selection_rank",
]
