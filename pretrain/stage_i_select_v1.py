#!/usr/bin/env python3
"""Independent Stage-I selection core v1.

This module is the *independently implemented* side of the Stage-H / Stage-I equality contract.
Stage H predicts with ``pretrain/stage_i_replay_v2.py``; Stage I decides with this file. The two
were written separately and share no selection code, so an H==I agreement is evidence about the
semantics rather than a tautology about one shared implementation (DECISIONS D-142 records the
confirmed common-mode risk that motivates this split).

Independence is structural, not a promise: this module imports nothing from
``pretrain.stage_i_replay_v2`` and nothing from ``pretrain.h_census_v2``. It has no import path to
either, so it cannot consult H's answer even by accident. ``tests/test_stage_i_realize_v1.py``
asserts that structurally over the parsed AST.

What is re-derived here, from the frozen graph and policy rather than from H's code: the canonical
document fingerprint, the duplicate representative key and its multi-binding extension, the score
accessor, exact binary64 score ordering, seeded-hash ordering, the whole-document prefix cut,
committed-only ownership, the ownership matrix, crossing evidence and the selection fingerprint.

What is deliberately shared, because forking it would fork the frozen corpus contract rather than
prove anything: canonical JSON serialisation, SHA-256 helpers, text cleaning and tokenizer
accounting. Those are byte-level infrastructure that the frozen releases, the G2 exclusion set and
selector v1 are all already defined in terms of; a second implementation of them would create a
divergence risk with no independence benefit. ``tests/test_stage_i_realize_v1.py`` pins the
re-derived identity primitives against frozen selector v1 differentially.

The physical locator rule lives here too. Representative choice is positional-free by contract, so
when several physical occurrences carry the identical winning ``(raw_sha256, input_record_sha256)``
tuple the selection result is already decided; the locator only says which byte range to copy. It
is the lexicographically minimal ``(input_binding_id, stable_input_record_ordinal)``, and every
occurrence of the winning tuple must agree on text, identity, token counts and selection metadata
or the run stops before publication.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any

# Re-derived independently of stage_i_replay_v2. Differentially pinned against frozen selector v1
# in tests/test_stage_i_realize_v1.py; these byte strings are the frozen wire contract, not a
# borrowed constant.
CANONICAL_DOCUMENT_DOMAIN = b"PetitGPT-canonical-document-v1\0"
SELECTION_RANK_DOMAIN = b"PetitGPT-pretrain-selection-v1\0"
SELECTION_FINGERPRINT_DOMAIN = b"PetitGPT-stage-i-selection-fingerprint-v1\0"

REPRESENTATIVE_RULE = "selector-v1 min(raw_sha256, input_record_sha256)"
PHYSICAL_LOCATOR_RULE = (
    "min(input_binding_id, stable_input_record_ordinal) among winning-tuple occurrences"
)
BRANCH_DEPENDENT = "BRANCH_DEPENDENT"
SELECTION_MODES = ("SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC")
BRANCHES = ("ORDINARY", "PRIMARY_GE4", "FALLBACK_RANKED_GE3")

_HEX64 = frozenset("0123456789abcdef")


class SelectionError(RuntimeError):
    """Fail-closed Stage-I selection condition."""


def _hex64(value: Any, where: str) -> str:
    if type(value) is not str or len(value) != 64 or not set(value) <= _HEX64:
        raise SelectionError(f"{where} must be 64 lowercase hex characters")
    return value


def _exact_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(f"{where} must be an exact integer")
    return value


# --------------------------------------------------------------------- identity primitives


def canonical_document_fingerprint_v1(cleaned_text: str) -> str:
    """Length-prefixed, domain-separated fingerprint of the cleaned document bytes.

    Length prefixing is what makes the domain separation actually separate: without it a document
    whose bytes happen to continue another's would collide under a bare concatenation.
    """
    if not isinstance(cleaned_text, str):
        raise SelectionError("cleaned text must be a string")
    payload = cleaned_text.encode("utf-8", errors="strict")
    digest = hashlib.sha256()
    digest.update(CANONICAL_DOCUMENT_DOMAIN)
    digest.update(len(payload).to_bytes(8, "big", signed=False))
    digest.update(payload)
    return digest.hexdigest()


def selection_rank_v1(*, seed: int, stage: str, source_id: str, canonical_fingerprint: str) -> str:
    """The frozen seeded rank. Domain, seed, stage and source all separate the keyspace."""
    _exact_int(seed, "seed")
    digest = hashlib.sha256()
    digest.update(SELECTION_RANK_DOMAIN)
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(stage.encode("ascii"))
    digest.update(b"\0")
    digest.update(source_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(_hex64(canonical_fingerprint, "canonical_fingerprint")))
    return digest.hexdigest()


def representative_key_v1(*, raw_sha256: str, input_record_sha256: str) -> bytes:
    """Selector v1's duplicate representative key, packed to 64 comparable bytes.

    Lowercase hex is order preserving over the underlying digest bytes, so comparing the packed
    form is the same total order as comparing the ``(raw_sha256, input_record_sha256)`` hex pair.
    No file, ordinal, row index or arrival order enters this key by construction.
    """
    return bytes.fromhex(_hex64(raw_sha256, "raw_sha256")) + bytes.fromhex(
        _hex64(input_record_sha256, "input_record_sha256")
    )


def selection_fingerprint_v1(identities: Iterable[str]) -> str:
    """Order-independent fingerprint of a committed identity set.

    Sorting before hashing is deliberate: the fingerprint must identify the *set* that was
    committed, so that a change in traversal order is not mistaken for a change in outcome.
    """
    values = sorted(identities)
    digest = hashlib.sha256(SELECTION_FINGERPRINT_DOMAIN)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        digest.update(bytes.fromhex(_hex64(value, "identity")))
    return digest.hexdigest()


# --------------------------------------------------------------------- exact score handling


def score_to_bits_v1(value: Any) -> int:
    """Lossless binary64 bits. Rejects bool, non-numeric, NaN and infinities.

    Scores decide a ranked cut, so a non-finite value is a corpus defect rather than an ordering
    question: there is no defensible position for NaN in a total order.
    """
    if isinstance(value, bool):
        raise SelectionError("continuous score must not be a boolean")
    if not isinstance(value, (int, float)):
        raise SelectionError(f"continuous score must be numeric, got {type(value).__name__}")
    as_float = float(value)
    if math.isnan(as_float):
        raise SelectionError("continuous score must not be NaN")
    if math.isinf(as_float):
        raise SelectionError("continuous score must not be infinite")
    return int.from_bytes(struct.pack(">d", as_float), "big", signed=False)


def bits_to_score_v1(bits: Any) -> float:
    bits = _exact_int(bits, "score bits")
    if bits < 0 or bits >= (1 << 64):
        raise SelectionError("score bits must fit in an unsigned 64-bit word")
    value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
    if math.isnan(value) or math.isinf(value):
        raise SelectionError("score bits decode to a non-finite value")
    return value


def read_score_v1(row: Mapping[str, Any], accessor: Mapping[str, Any], which: str) -> Any:
    """Read a score strictly through the release-pinned accessor.

    The FineWeb release stores the LITERAL key ``"metadata.int_score"`` inside its ``metadata``
    object, so generic dotted-path splitting would silently miss it. The accessor names one
    container and one exact key and nothing else is attempted. The declared JSON type is enforced
    exactly: an integer where a float is declared is a schema disagreement, not a convenience.
    """
    spec = accessor.get(which)
    if spec is None:
        return None
    container = spec["container"]
    key = spec["key"]
    holder = row.get(container)
    if not isinstance(holder, dict):
        raise SelectionError(f"accessor container {container!r} missing or not an object")
    if key not in holder:
        raise SelectionError(f"accessor key {key!r} missing from {container!r}")
    value = holder[key]
    expected = spec["json_type"]
    if isinstance(value, bool):
        raise SelectionError(f"{key!r} must be a JSON {expected}, got boolean")
    if expected == "int":
        if not isinstance(value, int):
            raise SelectionError(f"{key!r} must be a JSON integer, got {type(value).__name__}")
    elif expected == "float":
        if not isinstance(value, float):
            raise SelectionError(
                f"{key!r} must be a JSON float, got {type(value).__name__}; integer and "
                "continuous scores are distinct and are never coerced between"
            )
    else:
        raise SelectionError(f"accessor declares an unsupported json_type {expected!r}")
    return value


# --------------------------------------------------------------------- candidate records


@dataclass(frozen=True)
class Candidate:
    """One eligible physical occurrence of a logical document.

    This carries a physical locator (``input_binding_id`` + ``stable_input_record_ordinal``)
    because Stage I must eventually copy bytes, and it carries the positional-free representative
    key because selection must not see the locator. Keeping both on one object, with the rule that
    only the key may reach an ordering decision, is what lets the same scan feed both passes.
    """

    input_binding_id: str
    stable_input_record_ordinal: int
    raw_sha256: str
    input_record_sha256: str
    cleaned_sha256: str
    canonical_fingerprint: str
    content_token_count: int
    serialized_token_count: int
    int_score: int | None = None
    score_bits: int | None = None

    def __post_init__(self) -> None:
        validate_candidate(self)

    @property
    def representative_key(self) -> bytes:
        return representative_key_v1(
            raw_sha256=self.raw_sha256, input_record_sha256=self.input_record_sha256
        )

    @property
    def locator(self) -> tuple[str, int]:
        return (self.input_binding_id, self.stable_input_record_ordinal)


def validate_candidate(candidate: Candidate) -> None:
    """Every invariant re-checked at the boundary as well as at construction.

    A frozen dataclass is still mutable through ``object.__setattr__``, so construction-time
    validation alone says nothing about the object that actually reaches selection.
    """
    if not isinstance(candidate.input_binding_id, str) or not candidate.input_binding_id:
        raise SelectionError("input_binding_id must be a non-empty string")
    ordinal = _exact_int(candidate.stable_input_record_ordinal, "stable_input_record_ordinal")
    if ordinal < 0:
        raise SelectionError("stable_input_record_ordinal must not be negative")
    _hex64(candidate.raw_sha256, "raw_sha256")
    _hex64(candidate.input_record_sha256, "input_record_sha256")
    _hex64(candidate.cleaned_sha256, "cleaned_sha256")
    _hex64(candidate.canonical_fingerprint, "canonical_fingerprint")
    content = _exact_int(candidate.content_token_count, "content_token_count")
    serialized = _exact_int(candidate.serialized_token_count, "serialized_token_count")
    if content < 0:
        raise SelectionError("content_token_count must not be negative")
    if serialized <= 0:
        raise SelectionError("serialized_token_count must be positive")
    if serialized != content + 2:
        raise SelectionError(
            "serialized_token_count must be content_token_count plus the two frozen "
            "BOS/EOS boundary tokens"
        )
    if candidate.int_score is not None:
        _exact_int(candidate.int_score, "int_score")
    if candidate.score_bits is not None:
        bits_to_score_v1(candidate.score_bits)


# --------------------------------------------------------------------- representative collapse


@dataclass(frozen=True)
class Representative:
    """The single logical document a duplicate group collapses to, plus where to read it."""

    candidate: Candidate
    locator: tuple[str, int]
    occurrences: int

    @property
    def cleaned_sha256(self) -> str:
        return self.candidate.cleaned_sha256

    @property
    def serialized_token_count(self) -> int:
        return self.candidate.serialized_token_count


# Fields on which every occurrence of the identical winning tuple must agree. A disagreement means
# two different documents are claiming one identity, which would make the realized bytes depend on
# which occurrence happened to be copied.
_WINNER_AGREEMENT_FIELDS = (
    "cleaned_sha256",
    "canonical_fingerprint",
    "content_token_count",
    "serialized_token_count",
    "int_score",
    "score_bits",
)


def choose_representatives(candidates: Iterable[Candidate]) -> dict[str, Representative]:
    """Collapse duplicates the way selector v1 would, independent of traversal order.

    Two separate decisions live here and must not be confused:

    * *which document wins* is the minimum ``(raw_sha256, input_record_sha256)`` of the group. That
      key contains no positional component, so the winner is identical whatever order the bindings
      are pooled in and whatever order rows arrive in.
    * *which physical copy to read* is the lexicographically minimal
      ``(input_binding_id, stable_input_record_ordinal)`` **among the occurrences carrying the
      winning tuple**. This is provenance only. It is computed after the winner is fixed and can
      never move a branch, a rank, an owner, a fingerprint, a crossing or an overshoot.

    Any occurrence of the winning tuple that disagrees about text identity, token accounting or
    selection metadata stops the run: the realization would otherwise depend on which duplicate was
    copied, which is exactly the non-determinism this rule exists to remove.
    """
    best: dict[str, Candidate] = {}
    winners: dict[str, list[Candidate]] = {}

    for candidate in candidates:
        validate_candidate(candidate)
        identity = candidate.cleaned_sha256
        prior = best.get(identity)
        if prior is None:
            best[identity] = candidate
            winners[identity] = [candidate]
            continue
        if candidate.canonical_fingerprint != prior.canonical_fingerprint:
            raise SelectionError(
                f"identity collision: {identity} maps to two canonical fingerprints"
            )
        key = candidate.representative_key
        prior_key = prior.representative_key
        if key < prior_key:
            best[identity] = candidate
            winners[identity] = [candidate]
        elif key == prior_key:
            winners[identity].append(candidate)

    representatives: dict[str, Representative] = {}
    for identity, winner in best.items():
        group = winners[identity]
        for other in group:
            for field_name in _WINNER_AGREEMENT_FIELDS:
                if getattr(other, field_name) != getattr(winner, field_name):
                    raise SelectionError(
                        f"identity {identity}: occurrences carrying the identical winning "
                        f"representative tuple disagree on {field_name!r}; refusing to publish a "
                        "realization whose bytes would depend on which duplicate was copied"
                    )
        locator = min(candidate.locator for candidate in group)
        representatives[identity] = Representative(
            candidate=winner, locator=locator, occurrences=len(group)
        )
    return representatives


# --------------------------------------------------------------------- node selection


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


@dataclass(frozen=True)
class SelectedDocument:
    """One committed document, with both its selection rank position and its physical locator."""

    cleaned_sha256: str
    selection_ordinal_within_node: int
    input_binding_id: str
    stable_input_record_ordinal: int
    raw_sha256: str
    input_record_sha256: str
    canonical_fingerprint: str
    content_token_count: int
    serialized_token_count: int


@dataclass(frozen=True)
class NodeSelection:
    """The complete outcome for one graph node, comparable field-by-field against Stage H."""

    source_id: str
    stage: str
    target_serialized_tokens: int
    branch: str
    selection_mode: str
    pre_exclusion_unique_identities: int
    g2_excluded_identities: int
    prior_commit_excluded_identities: int
    exclusions_by_owner: Mapping[str, int]
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
    boundary_evidence: Mapping[str, Any]
    selected: tuple[SelectedDocument, ...]

    def comparable(self) -> dict[str, Any]:
        """Exactly the projection Stage H publishes, for mechanical H/I comparison.

        ``selected`` is deliberately excluded: it is Stage-I materialization detail that Stage H
        never emits. The identity *set* it represents is already covered, losslessly, by
        ``selection_fingerprint``.
        """
        return {
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
            "boundary_evidence": {
                k: self.boundary_evidence.get(k) for k in BOUNDARY_EVIDENCE_FIELDS
            },
        }


def predicate_matches(candidate: Candidate, predicate: Mapping[str, Any]) -> bool:
    kind = predicate["kind"]
    if kind == "ALL_ELIGIBLE":
        return True
    if kind == "INTEGER_SCORE_AT_LEAST":
        if candidate.int_score is None:
            raise SelectionError(
                "integer-score predicate requires an integer score on every candidate"
            )
        threshold = predicate["value"]
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise SelectionError("integer-score predicate value must be an exact integer")
        return candidate.int_score >= threshold
    raise SelectionError(f"unsupported predicate kind {kind!r}")


def _order_key(representative: Representative, *, mode: str, seed: int, stage: str, source_id: str):
    candidate = representative.candidate
    if mode == "SEEDED_HASH":
        rank = selection_rank_v1(
            seed=seed,
            stage=stage,
            source_id=source_id,
            canonical_fingerprint=candidate.canonical_fingerprint,
        )
        return (rank, candidate.canonical_fingerprint)
    if mode == "EXACT_SCORE_DESC_SHA_ASC":
        if candidate.score_bits is None:
            raise SelectionError("score-ranked mode requires a continuous score on every candidate")
        # Descending by exact binary64 VALUE, then cleaned SHA ascending. Ordering the raw bit
        # patterns instead would place -0.0 after every positive value and invert the entire
        # negative range; negating the decoded value keeps the frozen numeric intent, and the SHA
        # breaks both the +0.0/-0.0 tie and the identical-bits tie identically.
        return (-bits_to_score_v1(candidate.score_bits), candidate.cleaned_sha256)
    raise SelectionError(f"unsupported selection mode {mode!r}")


def _available(
    universe: Mapping[str, Representative],
    predicate: Mapping[str, Any],
    reference_exclusion: frozenset[str] | set[str],
    committed_owner: Mapping[str, str],
) -> tuple[list[Representative], int, int, dict[str, int]]:
    """Apply the node predicate, then the G2 reserve exclusion, then prior-commit ownership.

    Ownership is consulted only through ``committed_owner``, which holds an identity solely because
    some earlier node actually selected and committed it. Merely *seeing* an identity leaves no
    ownership state, so an unselected occurrence never suppresses a later node.
    """
    matched = [r for r in universe.values() if predicate_matches(r.candidate, predicate)]
    g2_count = 0
    by_owner: dict[str, int] = {}
    available: list[Representative] = []
    for representative in matched:
        identity = representative.cleaned_sha256
        if identity in reference_exclusion:
            g2_count += 1
            continue
        owner = committed_owner.get(identity)
        if owner is not None:
            by_owner[owner] = by_owner.get(owner, 0) + 1
            continue
        available.append(representative)
    return available, len(matched), g2_count, by_owner


def _select_prefix(
    ordered: list[Representative], *, target: int
) -> tuple[list[Representative], Representative | None, Representative | None]:
    """Minimal whole-document prefix whose serialized mass first reaches or exceeds the target.

    Also returns the first ranked-but-not-taken document. That is the other half of the cut, and
    without it an auditor cannot check the boundary without rescanning the whole corpus.
    """
    picked: list[Representative] = []
    crossing: Representative | None = None
    mass = 0
    for representative in ordered:
        if mass >= target:
            break
        picked.append(representative)
        mass += representative.serialized_token_count
        if mass >= target:
            crossing = representative
    next_unselected = ordered[len(picked)] if len(ordered) > len(picked) else None
    return picked, crossing, next_unselected


def _boundary_evidence(
    *,
    mode: str,
    seed: int,
    stage: str,
    source_id: str,
    crossing: Representative | None,
    next_unselected: Representative | None,
) -> dict[str, Any]:
    """Compact, exact, lossless evidence for both sides of the rank cut."""

    def rank_of(representative: Representative | None) -> str | None:
        if representative is None or mode != "SEEDED_HASH":
            return None
        return selection_rank_v1(
            seed=seed,
            stage=stage,
            source_id=source_id,
            canonical_fingerprint=representative.candidate.canonical_fingerprint,
        )

    def bits_hex(representative: Representative | None) -> str | None:
        if representative is None or representative.candidate.score_bits is None:
            return None
        return f"{representative.candidate.score_bits:016x}"

    def float_hex(representative: Representative | None) -> str | None:
        if representative is None or representative.candidate.score_bits is None:
            return None
        return bits_to_score_v1(representative.candidate.score_bits).hex()

    return {
        "representative_rule": REPRESENTATIVE_RULE,
        "crossing_selection_rank": rank_of(crossing),
        "crossing_score_bits_hex": bits_hex(crossing),
        "crossing_score_hex": float_hex(crossing),
        "next_unselected_identity": (
            None if next_unselected is None else next_unselected.cleaned_sha256
        ),
        "next_unselected_serialized_tokens": (
            None if next_unselected is None else next_unselected.serialized_token_count
        ),
        "next_unselected_selection_rank": rank_of(next_unselected),
        "next_unselected_score_bits_hex": bits_hex(next_unselected),
        "next_unselected_score_hex": float_hex(next_unselected),
    }


def realize_selection(
    graph: Any,
    candidates_by_binding: Mapping[str, list[Candidate]],
    reference_exclusion: frozenset[str] | set[str],
) -> list[NodeSelection]:
    """Derive the authoritative Stage-I selection for every node of the frozen graph.

    Nodes are walked in the graph's own frozen execution order, which is ascending
    ``(stage_priority, source_id)``. That places every Stage-B node before every Stage-A node, so
    Stage B commits first and Stage A sees those commitments as exclusions. Stage A itself draws
    from the full eligible population; no Stage-A complement of Stage-B membership is constructed.
    """
    if not isinstance(reference_exclusion, (set, frozenset)):
        raise SelectionError("reference_exclusion must be a set of cleaned identities")

    consumed = {b for node in graph.nodes for b in node.input_binding_ids}
    for binding_id in sorted(consumed):
        if binding_id not in candidates_by_binding:
            raise SelectionError(f"missing candidates for input binding {binding_id!r}")
    for binding_id, records in candidates_by_binding.items():
        if binding_id not in graph.bindings:
            raise SelectionError(f"candidates supplied for unknown input binding {binding_id!r}")
        if not isinstance(records, list):
            raise SelectionError(f"candidates for {binding_id!r} must be a list")
        for record in records:
            if not isinstance(record, Candidate):
                raise SelectionError(f"{binding_id!r}: candidates must be Candidate instances")
            validate_candidate(record)
            if record.input_binding_id != binding_id:
                raise SelectionError(
                    f"{binding_id!r}: candidate claims binding {record.input_binding_id!r}"
                )

    committed_owner: dict[str, str] = {}
    results: list[NodeSelection] = []

    for node in graph.nodes:
        # A union node pools its frozen bindings before collapsing. Because the representative key
        # carries no positional component, pooling needs no per-binding ordinal and no copy: the
        # winner over the union is the object v1 would have kept had the union been one file in
        # any concatenation order.
        pooled: list[Candidate] = []
        for binding_id in node.input_binding_ids:
            pooled.extend(candidates_by_binding[binding_id])
        universe = choose_representatives(pooled)
        del pooled

        if node.selection_mode == BRANCH_DEPENDENT:
            primary = node.branch_primary
            available, matched_n, g2_n, by_owner = _available(
                universe, primary["candidate_predicate"], reference_exclusion, committed_owner
            )
            if sum(r.serialized_token_count for r in available) >= node.target_serialized_tokens:
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

        capacity = sum(r.serialized_token_count for r in available)
        ordered = sorted(
            available,
            key=lambda r: _order_key(
                r, mode=mode, seed=graph.seed, stage=node.stage, source_id=node.source_id
            ),
        )
        picked, crossing, next_unselected = _select_prefix(
            ordered, target=node.target_serialized_tokens
        )
        selected_mass = sum(r.serialized_token_count for r in picked)
        feasible = selected_mass >= node.target_serialized_tokens

        for representative in picked:
            committed_owner[representative.cleaned_sha256] = node.source_id

        selected = tuple(
            SelectedDocument(
                cleaned_sha256=r.cleaned_sha256,
                selection_ordinal_within_node=ordinal,
                input_binding_id=r.locator[0],
                stable_input_record_ordinal=r.locator[1],
                raw_sha256=r.candidate.raw_sha256,
                input_record_sha256=r.candidate.input_record_sha256,
                canonical_fingerprint=r.candidate.canonical_fingerprint,
                content_token_count=r.candidate.content_token_count,
                serialized_token_count=r.candidate.serialized_token_count,
            )
            for ordinal, r in enumerate(picked)
        )

        results.append(
            NodeSelection(
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
                crossing_document_serialized_tokens=(
                    crossing.serialized_token_count if crossing else None
                ),
                actual_overshoot_tokens=(
                    selected_mass - node.target_serialized_tokens if feasible else 0
                ),
                residual_identities=len(available) - len(picked),
                residual_serialized_tokens=capacity - selected_mass,
                selection_fingerprint=selection_fingerprint_v1(r.cleaned_sha256 for r in picked),
                feasible=feasible,
                boundary_evidence=_boundary_evidence(
                    mode=mode,
                    seed=graph.seed,
                    stage=node.stage,
                    source_id=node.source_id,
                    crossing=crossing,
                    next_unselected=next_unselected,
                ),
                selected=selected,
            )
        )
    return results


def ownership_matrix_v1(results: Iterable[NodeSelection]) -> dict[str, dict[str, int]]:
    """Consumer -> owner -> excluded identity count, omitting nodes that excluded nothing."""
    return {r.source_id: dict(r.exclusions_by_owner) for r in results if r.exclusions_by_owner}


__all__ = [
    "BOUNDARY_EVIDENCE_FIELDS",
    "BRANCHES",
    "BRANCH_DEPENDENT",
    "CANONICAL_DOCUMENT_DOMAIN",
    "PHYSICAL_LOCATOR_RULE",
    "REPRESENTATIVE_RULE",
    "SELECTION_FINGERPRINT_DOMAIN",
    "SELECTION_MODES",
    "SELECTION_RANK_DOMAIN",
    "Candidate",
    "NodeSelection",
    "Representative",
    "SelectedDocument",
    "SelectionError",
    "bits_to_score_v1",
    "canonical_document_fingerprint_v1",
    "choose_representatives",
    "ownership_matrix_v1",
    "predicate_matches",
    "read_score_v1",
    "realize_selection",
    "representative_key_v1",
    "score_to_bits_v1",
    "selection_fingerprint_v1",
    "selection_rank_v1",
    "validate_candidate",
]
