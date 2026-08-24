"""Contract tests for the authoritative Stage-I realization tooling v1.

Expected values here come from literal fixtures, from the frozen selector v1, or from small
oracles written out longhand in this file. Nothing is ever checked against the production function
that computed it, because an assertion of the form ``f(x) == f(x)`` proves only that ``f`` is
deterministic and would pass with ``f`` entirely wrong.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
import contextlib
import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from pretrain.h_census_v2 import CensusError
from pretrain.select_pretrain_documents import (
    canonical_document_fingerprint as v1_fingerprint,
    selection_rank as v1_selection_rank,
)
from pretrain.stage_i_audit_v1 import AuditError
from pretrain.stage_i_graph_v2 import (
    GraphError,
    canonical_json_bytes,
    load_source_graph,
    sha256_file,
)
from pretrain.stage_i_output_v1 import (
    COMPLETE_MARKER,
    DOCUMENTS_DIRNAME,
    FROZEN_ENVIRONMENT,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    PASS1_RESULT_SCHEMA,
    RECORD_SCHEMA,
    RECORDS_PER_SHARD,
    SHARD_POLICY_VERSION,
    OutputError,
    TrustedExpectedResult,
    binding_document_digests_sha256,
    build_record,
    environment_sha256,
    iter_records,
    load_published_realization,
    load_trusted_expected_result,
    node_binding_projection_sha256,
    plan_shards,
    publish_atomic,
    recompute_stage_i_run_identity,
    reconcile_manifest_with_audit,
    record_sort_key,
    selection_sequence_commitment_map_sha256,
    stage_i_published_run_identity,
    trusted_expected_result,
    validate_manifest,
    validate_pass1_result,
    validate_record,
)
from pretrain.stage_i_realize_v1 import (
    ACCEPTED_H_COMPLETE_SHA256,
    ACCEPTED_H_RUN_DIR,
    CANONICAL_REPO_ROOT,
    IMPLEMENTATION_BUNDLE_FILES,
    PLAN_SCHEMA,
    REQUIRED_PYTHON_EXECUTABLE,
    REQUIRED_PYTHON_VERSION,
    REQUIRED_TOKENIZERS_VERSION,
    SHARD_POLICY_RULE,
    AcceptedH,
    AuthorizedIContext,
    Environment,
    RealizationError,
    authorization_block,
    authorize_plan,
    build_manifest,
    build_materialization_targets,
    build_pass1_result,
    compare_with_h,
    generate_candidate_plan,
    implementation_bundle_sha256,
    implementation_files,
    load_accepted_h,
    materialize_binding,
    require_h_i_equality,
    resolve_authorized_state,
    scan_binding_candidates,
    verify_environment,
)
from pretrain.stage_i_select_v1 import (
    Candidate,
    NodeSelection,
    SelectedDocument,
    SelectionError,
    canonical_document_fingerprint_v1,
    choose_representatives,
    ownership_matrix_v1,
    realize_selection,
    representative_key_v1,
    score_to_bits_v1,
    selection_fingerprint_v1,
    selection_rank_v1,
)

ROOT = Path("/workspace/petitgpt")
TOKENIZER = ROOT / "runs/g_production_2026-08-21/release/tokenizer.json"
H_RUN_DIR = ROOT / "runs/h_production_v2_2026-08-23"
GRAPH = ROOT / "runs/h_tooling_repair_v2_2026-08-21/policy/stage_i_source_graph_v1.json"
SEED = 5088999448999271579
FP_DOMAIN = b"PetitGPT-stage-i-selection-fingerprint-v1\0"
TUTORIAL_BINDING = "ib_structured_tutorial"


# ---------------------------------------------------------------- independent oracles


def oracle_fingerprint(identities: list[str]) -> str:
    """Written out longhand so a fingerprint comparison is evidence, not a tautology."""
    values = sorted(identities)
    digest = hashlib.sha256(FP_DOMAIN)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def oracle_sequence_commitment(source_id: str, stage: str, pairs) -> str:
    """Longhand reimplementation of the order-sensitive selection-sequence commitment.

    Written out here so a commitment comparison is evidence rather than a tautology; the
    production implementation is never called to produce an expected value.
    """
    pairs = list(pairs)
    digest = hashlib.sha256(b"PetitGPT-stage-i-selection-sequence-v1\0")
    digest.update(b"petitgpt-stage-i-selection-sequence-v1")
    digest.update(b"\0")
    digest.update(source_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(stage.encode("utf-8"))
    digest.update(b"\0")
    for ordinal, identity in pairs:
        digest.update(ordinal.to_bytes(8, "big"))
        digest.update(bytes.fromhex(identity))
    digest.update(b"\0")
    digest.update(len(pairs).to_bytes(8, "big"))
    return digest.hexdigest()


def oracle_canonical_record_sha256(record: dict) -> str:
    """Selector v1 hashes the ASCII canonical JSON of the raw record; re-derived here."""
    payload = json.dumps(
        record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def oracle_raw_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def oracle_ranked_prefix(rows: list[dict], target: int) -> dict:
    """Standalone exact-binary64 descending ranked prefix. Never calls the production core."""
    for row in rows:
        assert type(row["score"]) is float
    ordered = sorted(rows, key=lambda row: (-row["score"], row["sha"]))
    picked: list[dict] = []
    mass = 0
    for row in ordered:
        if mass >= target:
            break
        picked.append(row)
        mass += row["tokens"]
    return {
        "selected": [row["sha"] for row in picked],
        "mass": mass,
        "crossing": picked[-1]["sha"] if picked and mass >= target else None,
        "overshoot": mass - target if mass >= target else 0,
        "feasible": mass >= target,
        "fingerprint": oracle_fingerprint([row["sha"] for row in picked]),
    }


def oracle_seeded_prefix(rows: list[dict], target: int, *, stage: str, source_id: str) -> dict:
    """Standalone seeded-hash prefix, ranked through the FROZEN selector v1 rank function."""
    ordered = sorted(
        rows,
        key=lambda row: (
            v1_selection_rank(
                seed=SEED,
                stage=stage,
                source_id=source_id,
                canonical_fingerprint=row["fp"],
            ),
            row["fp"],
        ),
    )
    picked: list[dict] = []
    mass = 0
    for row in ordered:
        if mass >= target:
            break
        picked.append(row)
        mass += row["tokens"]
    return {
        "selected": [row["sha"] for row in picked],
        "mass": mass,
        "crossing": picked[-1]["sha"] if picked and mass >= target else None,
        "overshoot": mass - target if mass >= target else 0,
        "feasible": mass >= target,
        "fingerprint": oracle_fingerprint([row["sha"] for row in picked]),
    }


# ---------------------------------------------------------------- fixtures


def cand(
    name: str,
    tokens: int,
    *,
    binding: str = "ib_x",
    ordinal: int = 0,
    raw: str | None = None,
    row: str | None = None,
    identity: str | None = None,
    int_score: int | None = None,
    score: float | None = None,
    content: int | None = None,
) -> Candidate:
    label = identity or name
    return Candidate(
        input_binding_id=binding,
        stable_input_record_ordinal=ordinal,
        raw_sha256=hashlib.sha256((raw if raw is not None else f"raw-{name}").encode()).hexdigest(),
        input_record_sha256=hashlib.sha256((row or f"row-{name}").encode()).hexdigest(),
        cleaned_sha256=hashlib.sha256(f"c-{label}".encode()).hexdigest(),
        canonical_fingerprint=hashlib.sha256(f"f-{label}".encode()).hexdigest(),
        content_token_count=tokens - 2 if content is None else content,
        serialized_token_count=tokens,
        int_score=int_score,
        score_bits=None if score is None else score_to_bits_v1(score),
    )


def binding_entry(tmp: Path, key: str) -> dict:
    doc = tmp / f"{key}.jsonl"
    doc.write_text('{"text":"x"}\n', encoding="utf-8")
    idx = tmp / f"{key}.u32.raw"
    idx.write_bytes(b"")
    return {
        "input_binding_id": key,
        "release_key": key.replace("ib_", ""),
        "documents_path": str(doc),
        "documents_sha256": sha256_file(doc),
        "documents_size_bytes": doc.stat().st_size,
        "release_manifest_path": str(doc),
        "release_manifest_sha256": sha256_file(doc),
        "eligibility_index_path": str(idx),
        "eligibility_index_sha256": sha256_file(idx),
        "eligibility_index_element_width_bytes": 4,
        "eligibility_index_dtype": "<u4",
        "total_physical_rows": 1,
        "excluded_rows": 0,
        "expected_eligible_rows": 1,
        "schema_accessor": {"accessor_id": f"{key}_acc"},
        "text_field": "text",
        "cleaning_contract": {
            "strip_leading_noise": False,
            "normalize_quotes": False,
            "underscores_policy": "keep",
            "min_chars": 0,
            "min_ascii_ratio": 0.0,
        },
    }


def node(
    sid,
    stage,
    target,
    ibs=("ib_x",),
    mode="SEEDED_HASH",
    predicate=None,
    primary=None,
    fallback=None,
):
    entry = {
        "node_id": sid,
        "source_id": sid,
        "stage": stage,
        "stage_priority": 0 if stage == "stage_b" else 1,
        "target_serialized_tokens": target,
        "input_binding_ids": list(ibs),
        "selection_mode": mode,
        "candidate_predicate": predicate or {"kind": "ALL_ELIGIBLE"},
    }
    if primary:
        entry["branch_primary"] = primary
        entry["branch_fallback"] = fallback
    return entry


PRIMARY = {
    "branch": "PRIMARY_GE4",
    "selection_mode": "SEEDED_HASH",
    "candidate_predicate": {"kind": "INTEGER_SCORE_AT_LEAST", "field": "s", "value": 4},
}
FALLBACK = {
    "branch": "FALLBACK_RANKED_GE3",
    "selection_mode": "EXACT_SCORE_DESC_SHA_ASC",
    "candidate_predicate": {"kind": "ALL_ELIGIBLE"},
    "rank_order": ["exact finite binary64 continuous score DESC", "cleaned_text_sha256 ASC"],
    "continuous_score_field": "s",
}


def mini_graph(tmp: Path, nodes, bindings=("ib_x",), *, tutorial_ids=None, **over) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    keys = list(bindings)
    if TUTORIAL_BINDING not in keys and tutorial_ids is None:
        keys.append(TUTORIAL_BINDING)
    graph = {
        "schema_version": "petitgpt-stage-i-source-graph-v1",
        "policy_status": "OWNER_FROZEN",
        "authority": "test",
        "date": "2026-08-23",
        "note": "fixture",
        "bound_authorities": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in sorted([
                "d2_d3_eligibility_manifest_sha256",
                "g2_exclusion_manifest_sha256",
                "g2_manifest_sha256",
                "g_manifest_sha256",
                "hq_policy_sha256",
                "selector_v1_sha256_preserved",
                "stage_e_allocation_sha256",
                "tokenizer_sha256",
            ])
        },
        "selection_seed": {
            "domain_utf8": "petitgpt-stage-i-selection-seed-v1",
            "domain_sha256": hashlib.sha256(b"petitgpt-stage-i-selection-seed-v1").hexdigest(),
            "derivation": "sha256 prefix",
            "seed": SEED,
            "seed_hex": hex(SEED),
        },
        "stage_priority": {"stage_b": 0, "stage_a": 1},
        "execution_order_rule": "ascending (stage_priority, source_id)",
        "selection_modes_closed_enum": ["SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC"],
        "control_namespace": {"exists": False, "note": "no control node"},
        "stage_a_population_rule": "full eligible population",
        "ownership_rule": "commit transfers ownership",
        "structured_tutorial": {
            "input_binding_ids": list(tutorial_ids or [TUTORIAL_BINDING]),
            "input_binding_order_source": "fixture",
            "realization": "SINGLE_LOGICAL_NODE",
            "selection_mode": "SEEDED_HASH",
            "sub_targets": None,
        },
        "h_boundary": {
            "h_canonical_output_label": "NON_AUTHORITATIVE_FEASIBILITY_REPLAY",
            "h_publishes_physical_views": False,
        },
        "resume_supported": False,
        "input_bindings": {key: binding_entry(tmp, key) for key in keys},
        "nodes": nodes,
    }
    graph.update(over)
    path = tmp / "graph.json"
    path.write_bytes(canonical_json_bytes(graph))
    return path


def with_candidates(graph, by_binding):
    payload = dict(by_binding)
    for key in graph.bindings:
        payload.setdefault(key, [])
    return payload


# ============================================================ 1-6 representative semantics


def test_01_one_binding_duplicate_representative_is_the_minimum_tuple():
    """Requirement 1. The kept duplicate is min(raw_sha256, input_record_sha256), nothing else."""
    a = cand("a", 10, ordinal=0, raw="alpha", row="r-alpha", identity="dup")
    b = cand("b", 10, ordinal=1, raw="beta", row="r-beta", identity="dup")
    expected = min([a, b], key=lambda c: (c.raw_sha256, c.input_record_sha256))
    chosen = choose_representatives([a, b])
    assert set(chosen) == {a.cleaned_sha256}
    winner = chosen[a.cleaned_sha256].candidate
    assert (winner.raw_sha256, winner.input_record_sha256) == (
        expected.raw_sha256,
        expected.input_record_sha256,
    )


def test_02_opposite_traversal_order_selects_the_same_representative():
    """Requirement 2. The key has no positional component, so arrival order cannot matter."""
    group = [
        cand("a", 10, ordinal=0, raw="alpha", row="r1", identity="dup"),
        cand("b", 10, ordinal=1, raw="beta", row="r2", identity="dup"),
        cand("c", 10, ordinal=2, raw="gamma", row="r3", identity="dup"),
    ]
    forward = choose_representatives(group)
    backward = choose_representatives(list(reversed(group)))
    key = group[0].cleaned_sha256
    assert forward[key].candidate == backward[key].candidate
    assert forward[key].locator == backward[key].locator


def test_03_repeated_identity_at_different_ordinals_collapses_to_one():
    """Requirement 3. Row ordinal never enters the winning key."""
    low = cand("z", 10, ordinal=9, raw="aaa", row="aaa", identity="dup")
    high = cand("y", 10, ordinal=0, raw="zzz", row="zzz", identity="dup")
    chosen = choose_representatives([high, low])
    assert len(chosen) == 1
    # "aaa" hashes are not necessarily smaller; decide with the oracle, not with the ordinal.
    expected = min([low, high], key=lambda c: (c.raw_sha256, c.input_record_sha256))
    assert chosen[low.cleaned_sha256].candidate.raw_sha256 == expected.raw_sha256


def test_04_duplicates_across_structured_tutorial_bindings_collapse(tmp_path: Path):
    """Requirement 4. A union node's duplicate spanning two bindings is one logical document."""
    graph = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_tut", "stage_b", 10, ["ib_cos", "ib_fin"])],
            bindings=("ib_cos", "ib_fin"),
            tutorial_ids=["ib_cos", "ib_fin"],
        ),
        verify_hashes=True,
    )
    shared_a = cand("s", 10, binding="ib_cos", ordinal=3, raw="same", row="same", identity="dup")
    shared_b = cand("s", 10, binding="ib_fin", ordinal=1, raw="same", row="same", identity="dup")
    out = realize_selection(
        graph, with_candidates(graph, {"ib_cos": [shared_a], "ib_fin": [shared_b]}), set()
    )
    assert out[0].selected_identities == 1
    assert out[0].selection_fingerprint == oracle_fingerprint([shared_a.cleaned_sha256])


def test_05_identical_winning_tuple_uses_the_deterministic_physical_locator():
    """Requirement 5. Same winning tuple twice: locator is min(binding, ordinal), provenance only."""
    left = cand("s", 10, binding="ib_b", ordinal=2, raw="same", row="same", identity="dup")
    right = cand("s", 10, binding="ib_a", ordinal=7, raw="same", row="same", identity="dup")
    later = cand("s", 10, binding="ib_a", ordinal=4, raw="same", row="same", identity="dup")
    chosen = choose_representatives([left, right, later])
    representative = chosen[left.cleaned_sha256]
    assert representative.occurrences == 3
    assert representative.locator == ("ib_a", 4)
    # Order of arrival must not move the locator either.
    assert choose_representatives([later, left, right])[left.cleaned_sha256].locator == ("ib_a", 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("serialized_token_count", 12),
        ("canonical_fingerprint", hashlib.sha256(b"other").hexdigest()),
    ],
)
def test_06_inconsistent_duplicate_payload_stops(field, value):
    """Requirement 6. Occurrences of the winning tuple must agree, or the run stops."""
    base = cand("s", 10, binding="ib_a", ordinal=0, raw="same", row="same", identity="dup")
    kwargs = {
        "input_binding_id": "ib_b",
        "stable_input_record_ordinal": 1,
        "raw_sha256": base.raw_sha256,
        "input_record_sha256": base.input_record_sha256,
        "cleaned_sha256": base.cleaned_sha256,
        "canonical_fingerprint": base.canonical_fingerprint,
        "content_token_count": base.content_token_count,
        "serialized_token_count": base.serialized_token_count,
    }
    kwargs[field] = value
    if field == "serialized_token_count":
        kwargs["content_token_count"] = value - 2
    other = Candidate(**kwargs)
    expected = "canonical fingerprints" if field == "canonical_fingerprint" else "disagree on"
    with pytest.raises(SelectionError, match=expected):
        choose_representatives([base, other])


# ============================================================ 7-9 ownership


def test_07_shared_binding_feeds_both_stages_without_duplicating(tmp_path: Path):
    """Requirement 7. One frozen binding serves a Stage-B and a Stage-A node; no A/B copy."""
    graph = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_py", "stage_b", 10, ["ib_x"]), node("a_py", "stage_a", 10, ["ib_x"])],
        ),
        verify_hashes=True,
    )
    docs = [cand(f"d{i}", 10, ordinal=i) for i in range(4)]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": docs}), set())
    b_ids = {d.cleaned_sha256 for d in out[0].selected}
    a_ids = {d.cleaned_sha256 for d in out[1].selected}
    assert b_ids and a_ids
    assert b_ids.isdisjoint(a_ids)
    targets = build_materialization_targets(out)
    assert len(targets["ib_x"]) == len(b_ids) + len(a_ids)


def test_08_stage_b_commit_excludes_stage_a(tmp_path: Path):
    """Requirement 8. Stage B commits first and Stage A sees that commitment as an exclusion."""
    graph = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_x", "stage_b", 10, ["ib_x"]), node("a_x", "stage_a", 10, ["ib_x"])],
        ),
        verify_hashes=True,
    )
    only = cand("only", 10, ordinal=0)
    out = realize_selection(graph, with_candidates(graph, {"ib_x": [only]}), set())
    assert out[0].source_id == "b_x" and out[0].selected_identities == 1
    assert out[1].source_id == "a_x"
    assert out[1].prior_commit_excluded_identities == 1
    assert out[1].exclusions_by_owner == {"b_x": 1}
    assert out[1].post_exclusion_candidate_identities == 0
    assert out[1].feasible is False
    assert ownership_matrix_v1(out) == {"a_x": {"b_x": 1}}


def test_09_cross_source_ownership_is_recorded(tmp_path: Path):
    """Requirement 9. A document committed by one source is excluded from a different source."""
    graph = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_one", "stage_b", 10, ["ib_x"]), node("b_two", "stage_b", 10, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        ),
        verify_hashes=True,
    )
    shared = cand("s", 10, binding="ib_x", ordinal=0, identity="dup")
    echo = cand("s", 10, binding="ib_y", ordinal=5, identity="dup")
    private = cand("p", 10, binding="ib_y", ordinal=6)
    out = realize_selection(
        graph, with_candidates(graph, {"ib_x": [shared], "ib_y": [echo, private]}), set()
    )
    assert out[0].selected_identities == 1
    assert out[1].exclusions_by_owner == {"b_one": 1}
    assert ownership_matrix_v1(out) == {"b_two": {"b_one": 1}}


def test_09b_seen_but_uncommitted_identity_leaves_no_ownership(tmp_path: Path):
    """Ownership transfers on commit, never on mere occurrence."""
    graph = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_early", "stage_b", 10, ["ib_x"]), node("b_late", "stage_b", 10, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        ),
        verify_hashes=True,
    )
    shared = cand("shared", 10, binding="ib_x", ordinal=0, identity="dup")

    def rank(candidate):
        return v1_selection_rank(
            seed=SEED,
            stage="stage_b",
            source_id="b_early",
            canonical_fingerprint=candidate.canonical_fingerprint,
        )

    # Deterministically find a companion that OUTRANKS the shared identity, so the shared one is
    # seen by b_early but never committed. Searched rather than skipped: the whole point of this
    # test is the seen-but-uncommitted case, so it must actually be constructed.
    other = next(
        c
        for c in (cand(f"other{i}", 10, binding="ib_x", ordinal=1) for i in range(200))
        if rank(c) < rank(shared)
    )
    echo = cand("shared", 10, binding="ib_y", ordinal=0, identity="dup")
    out = realize_selection(
        graph, with_candidates(graph, {"ib_x": [shared, other], "ib_y": [echo]}), set()
    )
    assert out[0].selected_identities == 1
    assert out[1].prior_commit_excluded_identities == 0
    assert out[1].exclusions_by_owner == {}


# ============================================================ 10-14 branch, ranking, overshoot


def _branch_graph(tmp_path: Path, target: int):
    return load_source_graph(
        mini_graph(
            tmp_path,
            [
                node(
                    "b_dclm",
                    "stage_b",
                    target,
                    ["ib_x"],
                    mode="BRANCH_DEPENDENT",
                    primary=PRIMARY,
                    fallback=FALLBACK,
                )
            ],
        ),
        verify_hashes=True,
    )


def test_10_primary_insufficient_falls_back_to_ranked_ge3(tmp_path: Path):
    """Requirement 10. The DCLM case: too little >=4 mass, so the ranked >=3 branch is taken."""
    graph = _branch_graph(tmp_path, 100)
    rows = [
        cand("hi", 30, ordinal=0, int_score=4, score=9.0),
        cand("lo1", 60, ordinal=1, int_score=3, score=8.0),
        cand("lo2", 60, ordinal=2, int_score=3, score=7.0),
    ]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": rows}), set())[0]
    assert out.branch == "FALLBACK_RANKED_GE3"
    assert out.selection_mode == "EXACT_SCORE_DESC_SHA_ASC"
    oracle = oracle_ranked_prefix(
        [
            {"sha": c.cleaned_sha256, "score": s, "tokens": t}
            for c, s, t in zip(rows, [9.0, 8.0, 7.0], [30, 60, 60], strict=True)
        ],
        100,
    )
    assert out.selected_serialized_tokens == oracle["mass"]
    assert out.selection_fingerprint == oracle["fingerprint"]
    assert out.crossing_identity == oracle["crossing"]
    assert out.actual_overshoot_tokens == oracle["overshoot"]


def test_11_equal_score_crossing_is_resolved_by_sha_ordering(tmp_path: Path):
    """Requirement 11. The accepted DCLM boundary sits inside a score tie; SHA ASC decides it."""
    graph = _branch_graph(tmp_path, 100)
    tied = [
        cand("t1", 60, ordinal=0, int_score=3, score=3.484375),
        cand("t2", 60, ordinal=1, int_score=3, score=3.484375),
        cand("t3", 60, ordinal=2, int_score=3, score=3.484375),
    ]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": tied}), set())[0]
    assert out.branch == "FALLBACK_RANKED_GE3"
    by_sha = sorted(tied, key=lambda c: c.cleaned_sha256)
    # 100 target, 60 tokens each -> exactly the two lowest SHAs, in SHA ascending order.
    assert out.selected_identities == 2
    assert out.crossing_identity == by_sha[1].cleaned_sha256
    assert out.selection_fingerprint == oracle_fingerprint([
        by_sha[0].cleaned_sha256,
        by_sha[1].cleaned_sha256,
    ])
    assert out.boundary_evidence["crossing_score_bits_hex"] == f"{score_to_bits_v1(3.484375):016x}"
    assert out.boundary_evidence["crossing_score_hex"] == (3.484375).hex()
    assert out.boundary_evidence["next_unselected_identity"] == by_sha[2].cleaned_sha256


def test_12_sufficient_primary_takes_the_ge4_branch(tmp_path: Path):
    """Requirement 12. The FineWeb case: enough >=4 mass, so the seeded primary branch is taken."""
    graph = _branch_graph(tmp_path, 50)
    rows = [
        cand("h1", 40, ordinal=0, int_score=4, score=9.0),
        cand("h2", 40, ordinal=1, int_score=5, score=8.0),
        cand("l1", 90, ordinal=2, int_score=3, score=7.0),
    ]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": rows}), set())[0]
    assert out.branch == "PRIMARY_GE4"
    assert out.selection_mode == "SEEDED_HASH"
    # Only the two >=4 documents are candidates at all.
    assert out.pre_exclusion_unique_identities == 2
    assert out.post_exclusion_candidate_serialized_tokens == 80
    oracle = oracle_seeded_prefix(
        [{"sha": c.cleaned_sha256, "fp": c.canonical_fingerprint, "tokens": 40} for c in rows[:2]],
        50,
        stage="stage_b",
        source_id="b_dclm",
    )
    assert out.selected_serialized_tokens == oracle["mass"]
    assert out.selection_fingerprint == oracle["fingerprint"]
    assert out.crossing_identity == oracle["crossing"]


def test_13_whole_document_overshoot_is_measured_not_bounded(tmp_path: Path):
    """Requirement 13. The prefix is whole-document, so overshoot is exact and reported."""
    graph = load_source_graph(
        mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])]), verify_hashes=True
    )
    rows = [cand(f"d{i}", 30, ordinal=i) for i in range(6)]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": rows}), set())[0]
    # 30-token documents against a 100 target: four documents, 120 tokens, 20 overshoot.
    assert out.selected_identities == 4
    assert out.selected_serialized_tokens == 120
    assert out.actual_overshoot_tokens == 20
    assert out.crossing_document_serialized_tokens == 30
    assert out.residual_identities == 2
    assert out.residual_serialized_tokens == 60
    assert out.feasible is True


def test_13b_exact_target_has_zero_overshoot_and_infeasible_reports_none(tmp_path: Path):
    graph = load_source_graph(
        mini_graph(tmp_path, [node("b_x", "stage_b", 60, ["ib_x"])]), verify_hashes=True
    )
    exact = realize_selection(
        graph,
        with_candidates(graph, {"ib_x": [cand("a", 30, ordinal=0), cand("b", 30, ordinal=1)]}),
        set(),
    )[0]
    assert exact.selected_serialized_tokens == 60 and exact.actual_overshoot_tokens == 0

    short = load_source_graph(
        mini_graph(tmp_path / "s", [node("b_x", "stage_b", 500, ["ib_x"])]), verify_hashes=True
    )
    thin = realize_selection(
        short, with_candidates(short, {"ib_x": [cand("a", 30, ordinal=0)]}), set()
    )[0]
    assert thin.feasible is False
    assert thin.actual_overshoot_tokens == 0
    assert thin.crossing_identity is None
    assert thin.crossing_document_serialized_tokens is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "3.0", None])
def test_14_non_finite_or_mistyped_scores_are_rejected(bad):
    """Requirement 14. A score that cannot participate in a total order is a corpus defect."""
    with pytest.raises(SelectionError):
        score_to_bits_v1(bad)


def test_14b_exact_binary64_ordering_has_no_decimal_quantisation(tmp_path: Path):
    """Two scores one ULP apart must order strictly, not collapse into a rounded bucket."""
    graph = _branch_graph(tmp_path, 10)
    import math

    low = 3.484375
    high = math.nextafter(low, math.inf)
    assert low != high
    rows = [
        cand("lo", 10, ordinal=0, int_score=3, score=low),
        cand("hi", 10, ordinal=1, int_score=3, score=high),
    ]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": rows}), set())[0]
    assert out.branch == "FALLBACK_RANKED_GE3"
    assert out.selected_identities == 1
    # DESC by exact value: the nextafter neighbour must win.
    assert out.selected[0].cleaned_sha256 == rows[1].cleaned_sha256


def test_14c_signed_zero_ties_and_is_broken_by_sha_not_by_bits(tmp_path: Path):
    """+0.0 and -0.0 are numerically equal, so the SHA tiebreak must decide, not the bit pattern."""
    graph = _branch_graph(tmp_path, 10)
    rows = [
        cand("pz", 10, ordinal=0, int_score=3, score=0.0),
        cand("nz", 10, ordinal=1, int_score=3, score=-0.0),
    ]
    out = realize_selection(graph, with_candidates(graph, {"ib_x": rows}), set())[0]
    expected = min(rows, key=lambda c: c.cleaned_sha256)
    assert out.selected[0].cleaned_sha256 == expected.cleaned_sha256


# ============================================================ 15-19 H/I mismatch gates


def _fake_accepted(selections, **mutate) -> AcceptedH:
    """An accepted-H stand-in built from the realized selections, then optionally corrupted.

    Building the "prediction" from the realization and then breaking one field is exactly the
    right shape for these tests: the agreeing case proves the comparator accepts a true match, and
    each mutation proves it rejects one specific disagreement.
    """
    nodes = []
    for selection in selections:
        projected = selection.comparable()
        nodes.append(dict(projected))
    matrix = ownership_matrix_v1(selections)
    for source_id, changes in mutate.items():
        for node_entry in nodes:
            if node_entry["source_id"] == source_id:
                node_entry.update(changes)
    return AcceptedH(
        run_dir=Path("/nonexistent"),
        census={"nodes": nodes, "ownership_matrix": matrix},
        predictions={},
        census_sha256="0" * 64,
        predictions_sha256="0" * 64,
        complete_sha256="0" * 64,
        run_identity="0" * 64,
    )


def _two_node_selection(tmp_path: Path):
    graph = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_x", "stage_b", 60, ["ib_x"]), node("a_x", "stage_a", 30, ["ib_x"])],
        ),
        verify_hashes=True,
    )
    rows = [cand(f"d{i}", 30, ordinal=i) for i in range(6)]
    return realize_selection(graph, with_candidates(graph, {"ib_x": rows}), set())


def test_15_h_branch_mismatch_stops_before_publication(tmp_path: Path):
    """Requirement 15."""
    selections = _two_node_selection(tmp_path)
    accepted = _fake_accepted(selections, b_x={"branch": "PRIMARY_GE4"})
    comparison = compare_with_h(selections, accepted)
    assert comparison["ALL_H_I_BRANCHES_MATCH"] is False
    with pytest.raises(RealizationError, match="refusing to materialize"):
        require_h_i_equality(comparison)


def test_16_h_token_count_mismatch_stops_before_publication(tmp_path: Path):
    """Requirement 16."""
    selections = _two_node_selection(tmp_path)
    accepted = _fake_accepted(selections, b_x={"selected_serialized_tokens": 999})
    comparison = compare_with_h(selections, accepted)
    assert comparison["ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH"] is False
    with pytest.raises(RealizationError):
        require_h_i_equality(comparison)


def test_17_h_fingerprint_mismatch_stops_before_publication(tmp_path: Path):
    """Requirement 17."""
    selections = _two_node_selection(tmp_path)
    accepted = _fake_accepted(selections, b_x={"selection_fingerprint": "a" * 64})
    comparison = compare_with_h(selections, accepted)
    assert comparison["ALL_H_I_FINGERPRINTS_MATCH"] is False
    with pytest.raises(RealizationError):
        require_h_i_equality(comparison)


def test_18_h_crossing_mismatch_stops_before_publication(tmp_path: Path):
    """Requirement 18."""
    selections = _two_node_selection(tmp_path)
    accepted = _fake_accepted(selections, b_x={"crossing_identity": "b" * 64})
    comparison = compare_with_h(selections, accepted)
    assert comparison["ALL_H_I_CROSSING_IDENTITIES_MATCH"] is False
    with pytest.raises(RealizationError):
        require_h_i_equality(comparison)


def test_18b_h_overshoot_mismatch_stops_before_publication(tmp_path: Path):
    selections = _two_node_selection(tmp_path)
    accepted = _fake_accepted(selections, b_x={"actual_overshoot_tokens": 12345})
    comparison = compare_with_h(selections, accepted)
    assert comparison["ALL_H_I_OVERSHOOTS_MATCH"] is False
    with pytest.raises(RealizationError):
        require_h_i_equality(comparison)


def test_19_h_ownership_mismatch_stops_before_publication(tmp_path: Path):
    """Requirement 19. Both the per-node owner map and the global matrix are load bearing."""
    selections = _two_node_selection(tmp_path)
    accepted = _fake_accepted(selections)
    accepted.census["ownership_matrix"]["a_x"] = {"b_x": 99999}
    comparison = compare_with_h(selections, accepted)
    assert comparison["OWNERSHIP_MATRIX_MATCH"] is False
    with pytest.raises(RealizationError):
        require_h_i_equality(comparison)

    per_node = _fake_accepted(selections, a_x={"exclusions_by_owner": {"b_x": 4242}})
    per_node_comparison = compare_with_h(selections, per_node)
    assert per_node_comparison["ALL_NODES_MATCH"] is False
    with pytest.raises(RealizationError):
        require_h_i_equality(per_node_comparison)


def test_19b_matching_h_passes_the_gate(tmp_path: Path):
    """The comparator must accept a true match, or every mismatch test above proves nothing."""
    selections = _two_node_selection(tmp_path)
    comparison = compare_with_h(selections, _fake_accepted(selections))
    for key in (
        "ALL_H_I_BRANCHES_MATCH",
        "ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH",
        "ALL_H_I_FINGERPRINTS_MATCH",
        "ALL_H_I_CROSSING_IDENTITIES_MATCH",
        "ALL_H_I_OVERSHOOTS_MATCH",
        "OWNERSHIP_MATRIX_MATCH",
        "ALL_NODES_MATCH",
    ):
        assert comparison[key] is True, key
    require_h_i_equality(comparison)


# ============================================================ 20-21 materialization


def _real_binding(
    tmp: Path, rows: list[str], *, key: str = "ib_x", excluded: list[int] | None = None
):
    """A frozen-shaped binding backed by a real JSONL file and a real u32 eligibility index."""
    tmp.mkdir(parents=True, exist_ok=True)
    # binding_entry writes placeholder files at these same paths, so it must run BEFORE the real
    # corpus and index are written or it would clobber them.
    entry = binding_entry(tmp, key)
    doc = tmp / f"{key}.jsonl"
    doc.write_text("".join(rows), encoding="utf-8")
    idx = tmp / f"{key}.u32.raw"
    excluded = excluded or []
    idx.write_bytes(b"".join(struct.pack("<I", r) for r in excluded))
    # The release manifest must be its own JSON object; the corpus is JSONL and cannot serve as one.
    manifest = tmp / f"{key}.manifest.json"
    manifest.write_bytes(canonical_json_bytes({"release_key": key, "rows": len(rows)}))
    entry.update({
        "documents_path": str(doc),
        "documents_sha256": sha256_file(doc),
        "documents_size_bytes": doc.stat().st_size,
        "release_manifest_path": str(manifest),
        "release_manifest_sha256": sha256_file(manifest),
        "eligibility_index_path": str(idx),
        "eligibility_index_sha256": sha256_file(idx),
        "total_physical_rows": len(rows),
        "excluded_rows": len(excluded),
        "expected_eligible_rows": len(rows) - len(excluded),
    })
    return entry


def _scan_graph(tmp: Path, rows: list[str], nodes, *, excluded=None):
    entry = _real_binding(tmp / "bind", rows, excluded=excluded)
    path = mini_graph(tmp / "g", nodes)
    graph = json.loads(path.read_text())
    graph["input_bindings"]["ib_x"] = entry
    out = tmp / "graph_real.json"
    out.write_bytes(canonical_json_bytes(graph))
    return load_source_graph(out, verify_hashes=True)


def test_20_materialization_recomputes_hashes_and_token_counts(tmp_path: Path):
    """Requirement 20. Pass 2 trusts nothing from pass 1; it re-derives from the bytes it writes."""
    texts = [
        "Tides arise from gravitation.",
        "Erosion moves sediment downstream.",
        "Kilns fire clay.",
    ]
    rows = [json.dumps({"text": t}) + "\n" for t in texts]
    graph = _scan_graph(tmp_path, rows, [node("b_x", "stage_b", 10, ["ib_x"])])
    binding = graph.bindings["ib_x"]

    candidates, counters = scan_binding_candidates(
        binding, str(TOKENIZER), graph.validated_eligibility_rows("ib_x")
    )
    assert counters["physical_rows"] == 3 and counters["eligible_rows"] == 3
    # Independent oracle for the identity fields, not the production function.
    for candidate, text, raw in zip(candidates, texts, rows, strict=True):
        assert candidate.raw_sha256 == oracle_raw_sha256(text)
        assert candidate.input_record_sha256 == oracle_canonical_record_sha256(json.loads(raw))
        assert candidate.cleaned_sha256 == hashlib.sha256(text.encode()).hexdigest()
        assert candidate.canonical_fingerprint == v1_fingerprint(text)
        assert candidate.serialized_token_count == candidate.content_token_count + 2

    selections = realize_selection(graph, with_candidates(graph, {"ib_x": candidates}), set())
    targets = build_materialization_targets(selections)
    records = list(materialize_binding(binding, str(TOKENIZER), targets["ib_x"]))
    assert records
    for record in records:
        assert record["raw_sha256"] == oracle_raw_sha256(record["training_text"])
        assert (
            record["cleaned_text_sha256"]
            == hashlib.sha256(record["training_text"].encode()).hexdigest()
        )
        assert record["canonical_fingerprint"] == v1_fingerprint(record["training_text"])
        assert record["serialized_token_count"] == record["content_token_count"] + 2


def test_20b_materialization_stops_if_the_frozen_input_moved(tmp_path: Path):
    """A committed selection whose bytes no longer hash the same must not be published."""
    rows = [json.dumps({"text": f"Document number {i} about sediment."}) + "\n" for i in range(3)]
    graph = _scan_graph(tmp_path, rows, [node("b_x", "stage_b", 10, ["ib_x"])])
    binding = graph.bindings["ib_x"]
    candidates, _ = scan_binding_candidates(
        binding, str(TOKENIZER), graph.validated_eligibility_rows("ib_x")
    )
    selections = realize_selection(graph, with_candidates(graph, {"ib_x": candidates}), set())
    targets = build_materialization_targets(selections)
    # Corrupt exactly one committed target so the recomputation disagrees.
    ordinal = next(iter(targets["ib_x"]))
    import dataclasses

    targets["ib_x"][ordinal] = dataclasses.replace(
        targets["ib_x"][ordinal], cleaned_sha256="f" * 64
    )
    with pytest.raises(RealizationError, match="recomputed cleaned_sha256 disagrees"):
        list(materialize_binding(binding, str(TOKENIZER), targets["ib_x"]))


def test_21_duplicate_cleaned_identity_cannot_reach_the_output(tmp_path: Path):
    """Requirement 21. Global identity uniqueness is now established by the streaming audit.

    It used to be an in-memory set inside the writer -- ~13.8M entries at production scale, and a
    statement about objects that passed through a function rather than about the bytes that would
    be published. The audit derives it from the staged shards in bounded memory instead, so this
    test asserts the stronger property: a duplicate cannot survive publication.
    """
    manifest, records, expected = _publishable(tmp_path, count=2)
    twin = dict(records[1])
    twin["cleaned_text_sha256"] = records[0]["cleaned_text_sha256"]
    with pytest.raises(AuditError, match="appears more than once"):
        publish_atomic(tmp_path / "out", "run-dup", manifest, [records[0], twin], expected=expected)
    assert not (tmp_path / "out" / "run-dup").exists()


def test_21b_records_reaching_the_writer_out_of_order_are_rejected(tmp_path: Path):
    from pretrain.stage_i_output_v1 import write_shards

    first = build_record(
        stage="stage_b",
        source_id="b_x",
        input_binding_id="ib_x",
        stable_input_record_ordinal=5,
        input_record_sha256="1" * 64,
        raw_sha256="2" * 64,
        cleaned_text_sha256="3" * 64,
        canonical_fingerprint="4" * 64,
        selection_ordinal_within_node=0,
        content_token_count=5,
        serialized_token_count=7,
        training_text="a",
    )
    second = build_record(
        stage="stage_b",
        source_id="b_x",
        input_binding_id="ib_x",
        stable_input_record_ordinal=1,
        input_record_sha256="5" * 64,
        raw_sha256="6" * 64,
        cleaned_text_sha256="7" * 64,
        canonical_fingerprint="8" * 64,
        selection_ordinal_within_node=1,
        content_token_count=5,
        serialized_token_count=7,
        training_text="b",
    )
    with pytest.raises(OutputError, match="out of canonical physical order"):
        write_shards(tmp_path / "stage", [first, second])


# ============================================================ 22-24 output contract


def _record(**over):
    base = dict(
        stage="stage_b",
        source_id="b_x",
        input_binding_id="ib_x",
        stable_input_record_ordinal=0,
        input_record_sha256="1" * 64,
        raw_sha256="2" * 64,
        cleaned_text_sha256="3" * 64,
        canonical_fingerprint="4" * 64,
        selection_ordinal_within_node=0,
        content_token_count=5,
        serialized_token_count=7,
        training_text="hello",
    )
    base.update(over)
    return build_record(**base)


def test_22_record_schema_rejects_extra_missing_and_wrong_type_fields():
    """Requirement 22. The schema is closed in all three directions."""
    good = _record()
    validate_record(good)

    extra = dict(good)
    extra["surprise"] = 1
    with pytest.raises(OutputError, match="unknown field"):
        validate_record(extra)

    for field in sorted(good):
        missing = dict(good)
        del missing[field]
        with pytest.raises(OutputError, match="missing field"):
            validate_record(missing)

    for field, bad in (
        ("stable_input_record_ordinal", "0"),
        ("content_token_count", 5.0),
        ("serialized_token_count", True),
        ("training_text", 123),
        ("cleaned_text_sha256", "XYZ"),
        ("schema_version", "petitgpt-stage-i-document-v999"),
        ("stage", "stage_q"),
    ):
        wrong = dict(good)
        wrong[field] = bad
        with pytest.raises(OutputError):
            validate_record(wrong)


def test_22b_serialized_token_count_must_include_both_boundary_tokens():
    with pytest.raises(OutputError, match="boundary tokens"):
        _record(content_token_count=5, serialized_token_count=6)


SEQ_SCHEMA = "petitgpt-stage-i-selection-sequence-v1"

_FIXTURE_AUTHORIZATION = {
    "candidate_i_plan_sha256": "e" * 64,
    "authorized_state_sha256": "2" * 64,
    "implementation_commit": "f" * 40,
    "implementation_bundle_sha256": "1" * 64,
    "plan_schema_version": PLAN_SCHEMA,
    "output_schema_version": RECORD_SCHEMA,
    "manifest_schema_version": MANIFEST_SCHEMA,
    "shard_policy_version": SHARD_POLICY_VERSION,
    "records_per_shard": RECORDS_PER_SHARD,
    "h_run_identity": "7" * 64,
    "h_complete_sha256": "d" * 64,
    "h_census_sha256": "8" * 64,
    "h_predictions_sha256": "9" * 64,
    "owner_graph_sha256": "c" * 64,
    "node_binding_projection_sha256": node_binding_projection_sha256({"b_x": ["ib_x"]}),
    "environment_sha256": environment_sha256(FROZEN_ENVIRONMENT),
    "binding_document_digests_sha256": binding_document_digests_sha256({"ib_x": "6" * 64}),
}

# R4-D: the trusted environment and the trusted input-binding document digests the fixture's
# Layer-2 expectation commits to, before any of its Layer-3 bytes exist.
_FIXTURE_ENVIRONMENT = dict(FROZEN_ENVIRONMENT)
_FIXTURE_BINDING_DIGESTS = {"ib_x": "6" * 64}

_FIXTURE_H_BINDING = {
    "h_run_identity": "7" * 64,
    "h_census_sha256": "8" * 64,
    "h_predictions_sha256": "9" * 64,
    "h_complete_sha256": "d" * 64,
    "h_candidate_plan_sha256": "a" * 64,
    "h_implementation_bundle_sha256": "b" * 64,
    "owner_graph_sha256": "c" * 64,
}

_FIXTURE_GATE = {
    "ALL_H_I_BRANCHES_MATCH": True,
    "ALL_H_I_SELECTED_TOKEN_COUNTS_MATCH": True,
    "ALL_H_I_FINGERPRINTS_MATCH": True,
    "ALL_H_I_CROSSING_IDENTITIES_MATCH": True,
    "ALL_H_I_OVERSHOOTS_MATCH": True,
    "OWNERSHIP_MATRIX_MATCH": True,
    "ALL_NODES_MATCH": True,
}


def _pass1_for(
    nodes: list[dict],
    projection: dict,
    totals: dict,
    *,
    environment: dict | None = None,
    binding_digests: dict | None = None,
    **over,
) -> dict:
    """A post-Pass-1 expected result for a synthetic fixture, built the way Pass 1 builds one."""
    commitments = {n["source_id"]: n["selection_sequence_commitment"] for n in nodes}
    environment = dict(_FIXTURE_ENVIRONMENT if environment is None else environment)
    binding_digests = dict(_FIXTURE_BINDING_DIGESTS if binding_digests is None else binding_digests)
    authorization = dict(_FIXTURE_AUTHORIZATION)
    authorization["node_binding_projection_sha256"] = node_binding_projection_sha256(projection)
    authorization["environment_sha256"] = environment_sha256(environment)
    authorization["binding_document_digests_sha256"] = binding_document_digests_sha256(
        binding_digests
    )
    pass1 = {
        "schema_version": PASS1_RESULT_SCHEMA,
        "authorization": authorization,
        "selection_sequence_commitment_version": SEQ_SCHEMA,
        "selection_sequence_commitments": commitments,
        "selection_sequence_commitment_map_sha256": selection_sequence_commitment_map_sha256(
            commitments
        ),
        "node_binding_projection": projection,
        "authorized_input_binding_ids": sorted(binding_digests),
        "environment": environment,
        "binding_document_digests": binding_digests,
        "nodes": nodes,
        "totals": totals,
        "ownership_matrix": {},
        "h_i_gate": dict(_FIXTURE_GATE),
        "h_binding": dict(_FIXTURE_H_BINDING),
    }
    pass1.update(over)
    return pass1


def _expected_for(pass1: dict) -> TrustedExpectedResult:
    return trusted_expected_result(
        pass1, expected_sha256=hashlib.sha256(canonical_json_bytes(pass1)).hexdigest()
    )


def _publishable(tmp_path: Path, count: int = 3):
    """A publishable realization plus the Layer-2 expectation frozen before it existed.

    The expectation is derived from the records this fixture is about to emit, exactly as Pass 1
    derives it from its selection ledger -- and then held fixed. Tests that tamper with the
    published bytes reload with this same pre-tamper expectation, which is the whole point: the
    trusted value must not travel with the thing it constrains.
    """
    records = [
        _record(
            stable_input_record_ordinal=i,
            selection_ordinal_within_node=i,
            cleaned_text_sha256=hashlib.sha256(f"id-{i}".encode()).hexdigest(),
            raw_sha256=hashlib.sha256(f"raw-{i}".encode()).hexdigest(),
            input_record_sha256=hashlib.sha256(f"row-{i}".encode()).hexdigest(),
            canonical_fingerprint=hashlib.sha256(f"fp-{i}".encode()).hexdigest(),
            training_text=f"document {i}",
        )
        for i in range(count)
    ]
    projection = {"b_x": ["ib_x"]}
    node = {
        "source_id": "b_x",
        "stage": "stage_b",
        "target_serialized_tokens": 10,
        "branch": "ORDINARY",
        "selection_mode": "SEEDED_HASH",
        "selected_identities": count,
        "selected_serialized_tokens": 7 * count,
        # Computed with the longhand oracle from the records this fixture actually emits.
        # A placeholder is no longer acceptable: publication now reconciles the manifest
        # against a fingerprint reconstructed from the staged bytes.
        "selection_fingerprint": oracle_fingerprint([r["cleaned_text_sha256"] for r in records]),
        "selection_sequence_commitment": oracle_sequence_commitment(
            "b_x",
            "stage_b",
            [(r["selection_ordinal_within_node"], r["cleaned_text_sha256"]) for r in records],
        ),
        "crossing_identity": None,
        "actual_overshoot_tokens": 0,
        "input_binding_ids": ["ib_x"],
    }
    totals = {
        "records": count,
        "content_tokens": 5 * count,
        "serialized_tokens": 7 * count,
        "unique_cleaned_identities": count,
    }
    pass1 = _pass1_for([dict(node)], projection, dict(totals))
    expected = _expected_for(pass1)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "shards": [],
        "totals": {**totals, "shards": 1},
        "nodes": [dict(node)],
        "node_binding_projection": projection,
        "ownership_matrix": {},
        "bindings": {"ib_x": "6" * 64},
        "environment": {
            "python_executable": REQUIRED_PYTHON_EXECUTABLE,
            "python_version": REQUIRED_PYTHON_VERSION,
            "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
        },
        "h_binding": dict(_FIXTURE_H_BINDING),
        "stage_i_run": _stage_i_run_block(expected),
    }
    return manifest, records, expected


def _stage_i_run_block(expected: TrustedExpectedResult, **over) -> dict:
    """The published-run identity block generated by the trusted expectation's own fields."""
    block = {
        **dict(expected.authorization),
        "post_pass1_result_identity_schema": expected.result_identity_schema,
        "post_pass1_result_identity_sha256": expected.result_identity_sha256,
        "selection_sequence_commitment_version": expected.selection_sequence_commitment_version,
        "selection_sequence_commitment_map_sha256": (
            expected.selection_sequence_commitment_map_sha256
        ),
    }
    block.update(over)
    block["run_identity"] = recompute_stage_i_run_identity(block)
    return block


def test_23_partial_failure_leaves_no_complete_result(tmp_path: Path):
    """Requirement 23. A run that dies mid-write must leave nothing discoverable."""
    manifest, records, expected = _publishable(tmp_path)

    def exploding():
        yield records[0]
        raise RuntimeError("simulated mid-write failure")

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="simulated mid-write failure"):
        publish_atomic(out_dir, "run-x", manifest, exploding(), expected=expected)
    assert not (out_dir / "run-x").exists()
    assert list(out_dir.glob("*")) == [], "a staging directory survived the failure"


def test_23b_successful_publication_is_atomic_and_complete(tmp_path: Path):
    manifest, records, expected = _publishable(tmp_path)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    assert (final / COMPLETE_MARKER).is_file()
    assert (final / MANIFEST_FILENAME).is_file()
    assert (final / DOCUMENTS_DIRNAME).is_dir()
    loaded = load_published_realization(final, expected=expected)
    assert loaded["totals"]["records"] == 3
    assert [r["cleaned_text_sha256"] for r in iter_records(final, loaded)] == [
        hashlib.sha256(f"id-{i}".encode()).hexdigest() for i in range(3)
    ]


def test_23c_publication_refuses_to_overwrite_an_existing_result(tmp_path: Path):
    manifest, records, expected = _publishable(tmp_path)
    publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    _, again, _ = _publishable(tmp_path)
    with pytest.raises(OutputError, match="refusing to overwrite"):
        publish_atomic(tmp_path / "out", "run-x", manifest, again, expected=expected)


def test_24_strict_consumer_rejects_an_inconsistent_result(tmp_path: Path):
    """Requirement 24. Missing marker, tampered manifest, tampered shard and drifted totals."""
    manifest, records, expected = _publishable(tmp_path)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    load_published_realization(final, expected=expected)

    # 1. no COMPLETE marker at all
    marker = (final / COMPLETE_MARKER).read_bytes()
    (final / COMPLETE_MARKER).unlink()
    with pytest.raises(OutputError, match="no COMPLETE marker"):
        load_published_realization(final, expected=expected)
    (final / COMPLETE_MARKER).write_bytes(marker)

    # 2. manifest edited after the fact, so the marker no longer describes it
    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["totals"]["records"] = 99
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final, expected=expected)

    # 3. shard bytes changed underneath a valid manifest
    fresh_manifest, fresh_records, fresh_expected = _publishable(tmp_path)
    other = publish_atomic(
        tmp_path / "out2", "run-y", fresh_manifest, fresh_records, expected=expected
    )
    loaded = load_published_realization(other, expected=expected)
    shard = other / DOCUMENTS_DIRNAME / loaded["shards"][0]["name"]
    corrupted = shard.read_bytes().replace(b"document 0", b"document Z")
    shard.write_bytes(corrupted)
    with pytest.raises(OutputError, match="SHA-256 disagrees"):
        list(iter_records(other, loaded))


def test_24b_manifest_totals_must_reconcile_with_nodes_and_shards(tmp_path: Path):
    manifest, _, expected = _publishable(tmp_path)
    manifest["shards"] = [
        {"name": "documents-00000.jsonl", "records": 3, "bytes": 10, "sha256": "d" * 64}
    ]
    validate_manifest(manifest)

    drifted = json.loads(json.dumps(manifest))
    drifted["nodes"][0]["selected_identities"] = 4
    with pytest.raises(OutputError, match="per-node selected identity counts"):
        validate_manifest(drifted)

    tokens = json.loads(json.dumps(manifest))
    tokens["nodes"][0]["selected_serialized_tokens"] = 999
    with pytest.raises(OutputError, match="per-node selected token counts"):
        validate_manifest(tokens)


def test_24c_shard_policy_is_versioned_and_deterministic():
    """The declared sharding rule is arithmetic on a record count, not a property of the host."""
    assert plan_shards(0) == 0
    assert plan_shards(1) == 1
    assert plan_shards(RECORDS_PER_SHARD) == 1
    assert plan_shards(RECORDS_PER_SHARD + 1) == 2
    assert plan_shards(3 * RECORDS_PER_SHARD) == 3


# ============================================================ 25 environment contract


def test_25_wrong_tokenizers_or_python_version_stops():
    """Requirement 25. Token counts depend on the tokenizer build, so this must fail fast."""
    good = Environment(
        python_executable="/workspace/petitgpt/.venv/bin/python",
        python_version=REQUIRED_PYTHON_VERSION,
        tokenizers_version=REQUIRED_TOKENIZERS_VERSION,
    )
    verify_environment(good)

    with pytest.raises(RealizationError, match="tokenizers 0.22.1 is not the frozen"):
        verify_environment(
            Environment(
                python_executable=good.python_executable,
                python_version=good.python_version,
                tokenizers_version="0.22.1",
            )
        )
    with pytest.raises(RealizationError, match="python 3.11.9 is not the frozen"):
        verify_environment(
            Environment(
                python_executable=good.python_executable,
                python_version="3.11.9",
                tokenizers_version=good.tokenizers_version,
            )
        )
    with pytest.raises(RealizationError, match="python executable"):
        verify_environment(
            Environment(
                python_executable="/usr/bin/python3",
                python_version=good.python_version,
                tokenizers_version=good.tokenizers_version,
            )
        )


# ============================================================ authorized context fixture


@pytest.fixture(scope="module")
def live_plan_path(tmp_path_factory, accepted_h) -> Path:
    """A real candidate plan generated at the current HEAD, so it can actually be authorized."""
    plan = generate_candidate_plan(
        repo_root=ROOT,
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=True),
        accepted=accepted_h,
        environment=Environment(
            python_executable="/workspace/petitgpt/.venv/bin/python",
            python_version=REQUIRED_PYTHON_VERSION,
            tokenizers_version=REQUIRED_TOKENIZERS_VERSION,
        ),
    )
    path = tmp_path_factory.mktemp("live-plan") / "candidate_i_plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    return path


@pytest.fixture(scope="module")
def authorized(live_plan_path) -> AuthorizedIContext:
    """A genuinely authorized context: the positive path of requirement K."""
    return authorize_plan(live_plan_path, sha256_file(live_plan_path), ROOT)


def _mutated_plan(live_plan_path: Path, tmp_path: Path, mutate) -> tuple[Path, str]:
    """Copy the live plan, apply one mutation, and return its new path and true digest."""
    plan = json.loads(live_plan_path.read_text())
    mutate(plan)
    path = tmp_path / "mutated_plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    return path, sha256_file(path)


# ============================================================ 26-27 candidate I plan


@pytest.fixture(scope="module")
def accepted_h():
    """The real accepted Stage-H run, bound through the production loader."""
    return load_accepted_h(H_RUN_DIR)


def _plan(accepted, commit="0" * 40):
    return generate_candidate_plan(
        repo_root=ROOT,
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=True),
        accepted=accepted,
        environment=Environment(
            python_executable="/workspace/petitgpt/.venv/bin/python",
            python_version=REQUIRED_PYTHON_VERSION,
            tokenizers_version=REQUIRED_TOKENIZERS_VERSION,
        ),
        implementation_commit=commit,
    )


def test_26_candidate_i_plan_is_byte_reproducible(accepted_h):
    """Requirement 26. Same inputs, same bytes -- so the owner can regenerate and diff it."""
    first = canonical_json_bytes(_plan(accepted_h))
    second = canonical_json_bytes(_plan(accepted_h))
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


def test_27_candidate_i_plan_is_not_authorized_and_binds_everything(accepted_h):
    """Requirement 27. The plan cannot authorise itself, and it binds the full frozen context."""
    plan = _plan(accepted_h)
    assert plan["authorization_status"] == "NOT_AUTHORIZED"
    assert plan["schema_version"] == PLAN_SCHEMA
    assert plan["resume_supported"] is False
    # No owner authorization may live inside the plan bytes.
    serialized = canonical_json_bytes(plan).decode()
    assert "expected_plan_sha256" not in serialized
    assert "AUTHORIZED" not in serialized.replace("NOT_AUTHORIZED", "")

    assert plan["output_schema_version"] == RECORD_SCHEMA
    assert plan["manifest_schema_version"] == MANIFEST_SCHEMA
    assert plan["shard_policy"]["version"] == SHARD_POLICY_VERSION
    assert plan["shard_policy"]["records_per_shard"] == RECORDS_PER_SHARD

    assert plan["accepted_h"]["run_identity"] == accepted_h.run_identity
    assert plan["accepted_h"]["census_sha256"] == accepted_h.census_sha256
    assert plan["accepted_h"]["predictions_sha256"] == accepted_h.predictions_sha256
    assert (
        plan["accepted_h"]["candidate_plan_sha256"]
        == accepted_h.census["authorization"]["candidate_plan_sha256"]
    )
    assert (
        plan["accepted_h"]["implementation_bundle_sha256"]
        == accepted_h.census["authorization"]["implementation_bundle_sha256"]
    )

    assert plan["graph_sha256"] == (
        "e7cf8eafd117521660e898576d70e5873088129e0378afd34a6d8f56ea983986"
    )
    assert set(plan["authorities"]) == {
        "d2_d3_eligibility_manifest",
        "g2_release_manifest",
        "g_release_manifest",
        "hq_policy",
        "reference_exclusion",
        "selector_v1",
        "stage_e_allocation",
        "tokenizer",
    }
    assert len(plan["input_bindings"]) == 8
    assert plan["environment_contract"]["python_version"] == REQUIRED_PYTHON_VERSION
    assert plan["environment_contract"]["tokenizers_version"] == REQUIRED_TOKENIZERS_VERSION
    assert plan["implementation_bundle_sha256"] == implementation_bundle_sha256(
        implementation_files(ROOT)
    )


def test_27b_accepted_h_binding_requires_both_artifacts(accepted_h):
    """H_PREDICTIONS.json alone is never the authority; it is checked against the census."""
    assert accepted_h.census["status"] == "COMPLETE"
    assert accepted_h.predictions_sha256 == (
        "fff205494b1379eaf0e77a5d58591c085af2764712a024ed25c3767c10382f87"
    )
    with pytest.raises(RealizationError, match="is not the accepted"):
        load_accepted_h(H_RUN_DIR, expected_predictions_sha256="0" * 64)
    with pytest.raises(CensusError, match="run identity is not the expected one"):
        load_accepted_h(H_RUN_DIR, expected_run_identity="1" * 64)


# ============================================================ 28 selection independence


def test_28_stage_i_selection_has_no_h_replay_dependency():
    """Requirement 28. Structural, over the parsed AST -- not a promise in a docstring.

    The selection core must have no import path to Stage H whatsoever, and the driver may take
    exactly one name from the Stage-H module: the reviewed strict CONSUMER used to load the
    accepted result. Anything that decides a selection would make the H/I comparison circular.
    """
    forbidden_module = "pretrain.stage_i_replay_v2"
    h_module = "pretrain.h_census_v2"
    driver_allowlist = {"load_published_run"}

    def imports_of(path: Path) -> dict[str, set[str]]:
        tree = ast.parse(path.read_text())
        found: dict[str, set[str]] = {}
        for statement in ast.walk(tree):
            if isinstance(statement, ast.ImportFrom) and statement.module:
                found.setdefault(statement.module, set()).update(
                    alias.name for alias in statement.names
                )
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    found.setdefault(alias.name, set())
        return found

    core = imports_of(ROOT / "pretrain/stage_i_select_v1.py")
    assert forbidden_module not in core, "the selection core must never import the H replay core"
    assert h_module not in core, "the selection core must have no import path to Stage H"
    assert not any(m.startswith("pretrain.") for m in core), (
        "the selection core must not depend on any pipeline module; it imports only stdlib"
    )

    driver = imports_of(ROOT / "pretrain/stage_i_realize_v1.py")
    assert forbidden_module not in driver, (
        "the Stage-I driver must not import the H replay selection core"
    )
    assert driver.get(h_module, set()) <= driver_allowlist, (
        f"the Stage-I driver may import only {driver_allowlist} from {h_module}, "
        f"found {driver.get(h_module)}"
    )

    output = imports_of(ROOT / "pretrain/stage_i_output_v1.py")
    assert forbidden_module not in output and h_module not in output

    # No Stage-H selection symbol may be BOUND anywhere in the Stage-I implementation. Checking
    # bound imports rather than bare identifiers is what makes this precise: Stage I legitimately
    # defines its own `representative_key_v1` and its own ownership helpers, and flagging those by
    # name would be a collision, not a dependency. A symbol from a forbidden module can only reach
    # this code through an import, and module-level imports of those modules are already banned
    # above -- so an empty intersection here closes the remaining door.
    forbidden_names = {
        "replay",
        "census_body",
        "scan_binding",
        "read_score",
        "ownership_matrix",
        "representative_key",
        "representative_key_of",
        "selection_rank",
        "score_to_bits",
        "bits_to_score",
        "CandidateRecord",
        "NodeResult",
        "_representatives",
        "_select",
        "_order_key",
        "_available",
        "_fingerprint",
    }
    for relative in IMPLEMENTATION_BUNDLE_FILES:
        tree = ast.parse((ROOT / relative).read_text())
        bound: set[str] = set()
        for statement in ast.walk(tree):
            if isinstance(statement, ast.ImportFrom):
                bound.update(alias.asname or alias.name for alias in statement.names)
            elif isinstance(statement, ast.Import):
                bound.update(alias.asname or alias.name.split(".")[0] for alias in statement.names)
        leaked = forbidden_names & bound
        assert not leaked, f"{relative} imports H replay selection symbol(s): {sorted(leaked)}"

    # ...and no dynamic-import escape hatch may reintroduce them at runtime.
    for relative in IMPLEMENTATION_BUNDLE_FILES:
        source = (ROOT / relative).read_text()
        tree = ast.parse(source)
        called = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "__import__" not in called, f"{relative} uses __import__"
        assert "importlib" not in {
            alias.name.split(".")[0]
            for s in ast.walk(tree)
            if isinstance(s, ast.Import)
            for alias in s.names
        }, f"{relative} imports importlib"


# ============================================================ 29 selector v1 differential


@pytest.mark.parametrize(
    "text",
    ["", "a", "hello world", "héllo wörld", "你好，世界", "x" * 4096, "line\nbreak\ttab"],
)
def test_29_canonical_fingerprint_matches_frozen_selector_v1(text):
    """Requirement 29. The re-derived identity primitives agree with the frozen authority."""
    assert canonical_document_fingerprint_v1(text) == v1_fingerprint(text)


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
@pytest.mark.parametrize(
    "source_id", ["b_dclm_edu", "a_fineweb_edu_dedup", "b_structured_tutorial"]
)
def test_29b_selection_rank_matches_frozen_selector_v1(stage, source_id):
    fingerprint = v1_fingerprint(f"{stage}/{source_id}")
    assert selection_rank_v1(
        seed=SEED, stage=stage, source_id=source_id, canonical_fingerprint=fingerprint
    ) == v1_selection_rank(
        seed=SEED, stage=stage, source_id=source_id, canonical_fingerprint=fingerprint
    )


def test_29c_representative_key_order_matches_the_hex_pair_order():
    """The packed key must induce exactly the (raw_sha256, input_record_sha256) lexical order."""
    pairs = [
        (hashlib.sha256(f"r{i}".encode()).hexdigest(), hashlib.sha256(f"x{i}".encode()).hexdigest())
        for i in range(40)
    ]
    by_packed = sorted(
        pairs, key=lambda p: representative_key_v1(raw_sha256=p[0], input_record_sha256=p[1])
    )
    assert by_packed == sorted(pairs)


def test_29d_selection_fingerprint_matches_the_longhand_oracle():
    identities = [hashlib.sha256(f"i{i}".encode()).hexdigest() for i in range(7)]
    assert selection_fingerprint_v1(identities) == oracle_fingerprint(identities)
    # Order independence: the fingerprint identifies the SET that was committed.
    assert selection_fingerprint_v1(reversed(identities)) == oracle_fingerprint(identities)


# ============================================================ 30 physical order & ordinals


def test_30_physical_order_and_selection_ordinal_reconstruction(tmp_path: Path):
    """Requirement 30. Layout order is physical; rank order survives in the ordinal."""
    texts = [f"Document {i} concerning sediment, tides and kilns." for i in range(8)]
    rows = [json.dumps({"text": t}) + "\n" for t in texts]
    graph = _scan_graph(
        tmp_path,
        rows,
        [node("b_x", "stage_b", 40, ["ib_x"]), node("a_x", "stage_a", 40, ["ib_x"])],
    )
    binding = graph.bindings["ib_x"]
    candidates, _ = scan_binding_candidates(
        binding, str(TOKENIZER), graph.validated_eligibility_rows("ib_x")
    )
    selections = realize_selection(graph, with_candidates(graph, {"ib_x": candidates}), set())
    targets = build_materialization_targets(selections)
    records = list(materialize_binding(binding, str(TOKENIZER), targets["ib_x"]))

    ordered = sorted(records, key=record_sort_key)
    # Stage B (priority 0) must precede Stage A (priority 1) in the physical layout.
    stages = [r["stage"] for r in ordered]
    assert stages == sorted(stages, key=lambda s: 0 if s == "stage_b" else 1)
    # Within a node the physical order is by input ordinal, which is NOT the selection order.
    for source_id in ("b_x", "a_x"):
        group = [r for r in ordered if r["source_id"] == source_id]
        assert [r["stable_input_record_ordinal"] for r in group] == sorted(
            r["stable_input_record_ordinal"] for r in group
        )
        # The frozen selection sequence is reconstructible from the retained ordinal.
        by_rank = sorted(group, key=lambda r: r["selection_ordinal_within_node"])
        assert [r["selection_ordinal_within_node"] for r in by_rank] == list(range(len(group)))
        selection = next(s for s in selections if s.source_id == source_id)
        assert [r["cleaned_text_sha256"] for r in by_rank] == [
            d.cleaned_sha256 for d in selection.selected
        ]
        # ...and reconstructing the fingerprint from the published records agrees with the ledger.
        assert oracle_fingerprint([r["cleaned_text_sha256"] for r in group]) == (
            selection.selection_fingerprint
        )


def test_30b_manifest_reflects_the_realized_selection(authorized):
    """The published manifest must restate the ledger and bind the trusted expected result.

    Selections are built for the freshly re-derived authorized state's own graph nodes:
    `build_manifest` takes its graph from that state, never from a caller-supplied object, so
    feeding it a synthetic graph is not possible -- which is the point of that change.
    """
    state = authorized.revalidate()
    graph = state.graph
    selections = []
    for index, node_spec in enumerate(graph.nodes):
        identity = hashlib.sha256(f"n-{node_spec.source_id}".encode()).hexdigest()
        document = SelectedDocument(
            cleaned_sha256=identity,
            selection_ordinal_within_node=0,
            input_binding_id=node_spec.input_binding_ids[0],
            stable_input_record_ordinal=index,
            raw_sha256=hashlib.sha256(f"r-{node_spec.source_id}".encode()).hexdigest(),
            input_record_sha256=hashlib.sha256(f"w-{node_spec.source_id}".encode()).hexdigest(),
            canonical_fingerprint=hashlib.sha256(f"f-{node_spec.source_id}".encode()).hexdigest(),
            content_token_count=5,
            serialized_token_count=7,
        )
        selections.append(
            NodeSelection(
                source_id=node_spec.source_id,
                stage=node_spec.stage,
                target_serialized_tokens=node_spec.target_serialized_tokens,
                branch="ORDINARY",
                selection_mode="SEEDED_HASH",
                pre_exclusion_unique_identities=1,
                g2_excluded_identities=0,
                prior_commit_excluded_identities=0,
                exclusions_by_owner={},
                post_exclusion_candidate_identities=1,
                post_exclusion_candidate_serialized_tokens=7,
                selected_identities=1,
                selected_serialized_tokens=7,
                crossing_identity=None,
                crossing_document_serialized_tokens=None,
                actual_overshoot_tokens=0,
                residual_identities=0,
                residual_serialized_tokens=0,
                selection_fingerprint=oracle_fingerprint([identity]),
                feasible=True,
                boundary_evidence={},
                selected=(document,),
            )
        )

    gate = dict.fromkeys(_FIXTURE_GATE, True)
    pass1 = build_pass1_result(selections, state, gate)
    expected = _expected_for(pass1)
    manifest = build_manifest(selections, state, expected)
    assert manifest["totals"]["records"] == len(selections)
    assert manifest["totals"]["serialized_tokens"] == 7 * len(selections)
    assert manifest["nodes"][0]["selection_fingerprint"] == selections[0].selection_fingerprint

    # The identity block must be generated by its own fields and by the trusted expectation.
    block = manifest["stage_i_run"]
    assert block["run_identity"] == expected.stage_i_run_identity
    assert recompute_stage_i_run_identity(block) == expected.stage_i_run_identity
    assert block["h_complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256
    assert block["candidate_i_plan_sha256"] == authorized.plan_sha256
    assert block["owner_graph_sha256"] == state.graph.graph_sha256
    assert block["authorized_state_sha256"] == authorized.authorized_state_sha256
    assert block["post_pass1_result_identity_schema"] == PASS1_RESULT_SCHEMA
    assert block["post_pass1_result_identity_sha256"] == expected.result_identity_sha256
    assert manifest["h_binding"]["h_complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256

    # R3-A: the published run identity is downstream of the post-Pass-1 result and of the
    # sequence-commitment map. Changing either changes the name of the run, so a resealed result
    # cannot restate the expectation and keep its identity.
    assert (
        stage_i_published_run_identity(
            expected.authorization,
            post_pass1_result_identity_sha256="0" * 64,
            selection_sequence_commitment_map_sha256=(
                expected.selection_sequence_commitment_map_sha256
            ),
        )
        != expected.stage_i_run_identity
    )
    assert (
        stage_i_published_run_identity(
            expected.authorization,
            post_pass1_result_identity_sha256=expected.result_identity_sha256,
            selection_sequence_commitment_map_sha256="0" * 64,
        )
        != expected.stage_i_run_identity
    )


def test_30d_publishing_the_spooled_stream_round_trips(tmp_path: Path):
    """End-to-end: derive -> spool -> publish -> strict consumer -> identical records back."""
    from pretrain.stage_i_realize_v1 import iter_records_in_physical_order

    rows = [json.dumps({"text": f"Sediment study number {i}."}) + "\n" for i in range(8)]
    graph = _scan_graph(tmp_path, rows, [node("b_x", "stage_b", 30, ["ib_x"])])
    binding = graph.bindings["ib_x"]
    candidates, _ = scan_binding_candidates(
        binding, str(TOKENIZER), graph.validated_eligibility_rows("ib_x")
    )
    selections = realize_selection(graph, with_candidates(graph, {"ib_x": candidates}), set())
    manifest, expected = _manifest_for(selections)
    stream = iter_records_in_physical_order(graph, str(TOKENIZER), selections, tmp_path / "work")
    final = publish_atomic(tmp_path / "out", "run-i", manifest, stream, expected=expected)

    loaded = load_published_realization(final, expected=expected)
    published = list(iter_records(final, loaded))
    assert loaded["totals"]["records"] == len(published)
    assert loaded["totals"]["serialized_tokens"] == sum(
        r["serialized_token_count"] for r in published
    )
    assert [record_sort_key(r) for r in published] == sorted(record_sort_key(r) for r in published)
    assert {r["cleaned_text_sha256"] for r in published} == {
        d.cleaned_sha256 for s in selections for d in s.selected
    }


def _manifest_for(selections) -> tuple[dict, TrustedExpectedResult]:
    """A valid manifest and its Layer-2 expectation, both built without build_manifest.

    Written out here so a publication test is not checking the production manifest builder
    against the production publisher; the fixture states independently what the realization is,
    and states the expectation the publisher must prove it equal to.
    """
    total = sum(s.selected_identities for s in selections)
    nodes = [
        {
            "source_id": s.source_id,
            "stage": s.stage,
            "target_serialized_tokens": s.target_serialized_tokens,
            "branch": s.branch,
            "selection_mode": s.selection_mode,
            "selected_identities": s.selected_identities,
            "selected_serialized_tokens": s.selected_serialized_tokens,
            "selection_fingerprint": oracle_fingerprint([d.cleaned_sha256 for d in s.selected]),
            "selection_sequence_commitment": oracle_sequence_commitment(
                s.source_id,
                s.stage,
                [(d.selection_ordinal_within_node, d.cleaned_sha256) for d in s.selected],
            ),
            "crossing_identity": s.crossing_identity,
            "actual_overshoot_tokens": s.actual_overshoot_tokens,
            "input_binding_ids": sorted({d.input_binding_id for d in s.selected}),
        }
        for s in selections
    ]
    projection = {s.source_id: sorted({d.input_binding_id for d in s.selected}) for s in selections}
    totals = {
        "records": total,
        "content_tokens": sum(d.content_token_count for s in selections for d in s.selected),
        "serialized_tokens": sum(s.selected_serialized_tokens for s in selections),
        "unique_cleaned_identities": total,
    }
    pass1 = _pass1_for(
        [dict(n) for n in nodes],
        dict(projection),
        dict(totals),
        ownership_matrix=ownership_matrix_v1(selections),
    )
    expected = _expected_for(pass1)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "shards": [],
        "totals": {**totals, "shards": 0},
        "nodes": [dict(n) for n in nodes],
        "node_binding_projection": projection,
        "ownership_matrix": ownership_matrix_v1(selections),
        "bindings": {"ib_x": "6" * 64},
        "environment": {
            "python_executable": REQUIRED_PYTHON_EXECUTABLE,
            "python_version": REQUIRED_PYTHON_VERSION,
            "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
        },
        "h_binding": dict(_FIXTURE_H_BINDING),
        "stage_i_run": _stage_i_run_block(expected),
    }
    return manifest, expected


# ================================================================================
# R1 REPAIR REGRESSIONS — one per Codex I-tooling-v1 finding
# ================================================================================


# ---- A/B: true streaming spool replay, no whole-spool read ----------------------


def test_r1_a_spool_replay_streams_through_many_bounded_reads(tmp_path: Path):
    """A. Replay must advance in bounded windows, not slurp the spool.

    The reviewed path read a whole ``(stage, source, binding)`` spool with one 16 GiB-capped call,
    which is not an implementation for a source the size of FineWeb Stage A. Here the spool is
    deliberately much larger than the read window, so a whole-file read would take one call while
    a streaming read must take many.
    """
    from pretrain.stage_i_audit_v1 import ShardReader, stream_lines

    spool = tmp_path / "spool.jsonl"
    payload = b"".join(canonical_json_bytes({"n": i, "pad": "x" * 200}) for i in range(500))
    spool.write_bytes(payload)
    assert len(payload) > 64 * 1024

    window = 4096
    lines = list(stream_lines(spool, read_window_bytes=window))
    assert len(lines) == 500
    assert json.loads(lines[0])["n"] == 0
    assert json.loads(lines[-1])["n"] == 499

    reader = ShardReader(spool, read_window_bytes=window)
    consumed = list(reader)
    assert len(consumed) == 500
    # Many bounded reads, not one: this is the property the repair is about.
    assert reader.read_calls >= len(payload) // window
    assert reader.read_calls > 1
    assert reader.bytes_read == len(payload)
    assert reader.sha256 == hashlib.sha256(payload).hexdigest()


def test_r1_b_production_spool_path_contains_no_unbounded_whole_file_read():
    """B. Structural: the spool replay path must not call a whole-file read primitive."""
    source = (ROOT / "pretrain/stage_i_realize_v1.py").read_text()
    tree = ast.parse(source)
    replay = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "iter_records_in_physical_order"
    )
    called = {
        n.func.id
        for n in ast.walk(replay)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    } | {
        n.func.attr
        for n in ast.walk(replay)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in ("read_authoritative_bytes", "read_bytes", "read_text", "readlines"):
        assert forbidden not in called, (
            f"iter_records_in_physical_order calls {forbidden}, which materialises a whole spool"
        )
    assert "stream_lines" in called

    # The audit's own shard streaming must be equally bounded.
    audit = ast.parse((ROOT / "pretrain/stage_i_audit_v1.py").read_text())
    audit_called = {
        n.func.attr
        for n in ast.walk(audit)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "read_bytes" not in audit_called and "readlines" not in audit_called


def test_r1_a2_spool_larger_than_window_round_trips_through_the_driver(tmp_path: Path):
    """A. The driver's own replay, over a spool deliberately larger than its read window."""
    from pretrain.stage_i_realize_v1 import iter_records_in_physical_order

    rows = [
        json.dumps({"text": f"Sediment core sample {i} with layered deposits."}) + "\n"
        for i in range(60)
    ]
    graph = _scan_graph(tmp_path, rows, [node("b_x", "stage_b", 200, ["ib_x"])])
    binding = graph.bindings["ib_x"]
    candidates, _ = scan_binding_candidates(
        binding, str(TOKENIZER), graph.validated_eligibility_rows("ib_x")
    )
    selections = realize_selection(graph, with_candidates(graph, {"ib_x": candidates}), set())
    streamed = list(
        iter_records_in_physical_order(
            graph, str(TOKENIZER), selections, tmp_path / "work", read_window_bytes=64
        )
    )
    assert len(streamed) == sum(s.selected_identities for s in selections)
    assert [record_sort_key(r) for r in streamed] == sorted(record_sort_key(r) for r in streamed)
    assert list((tmp_path / "work").glob("spool-*")) == []


# ---- C: publication refuses when totals disagree with the physical records ------


def test_r1_c_publication_refused_when_manifest_totals_exceed_actual_records(tmp_path: Path):
    """C. The exact reviewed case: manifest claims 22 serialized tokens, records sum to 21."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    # Three records at 7 serialized tokens each = 21 actual.
    assert sum(r["serialized_token_count"] for r in records) == 21
    manifest["totals"]["serialized_tokens"] = 22
    manifest["nodes"][0]["selected_serialized_tokens"] = 22

    with pytest.raises(OutputError, match="but the staged records actually contain 21"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    assert not (tmp_path / "out" / "run-x").exists()
    assert list((tmp_path / "out").glob("*")) == [], "staging or audit scratch survived the failure"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("records", 99, "manifest.totals.records is 99"),
        ("content_tokens", 999, "manifest.totals.content_tokens is 999"),
        ("unique_cleaned_identities", 42, "unique_cleaned_identities is 42"),
    ],
)
def test_r1_c2_every_total_is_reconciled_against_the_physical_records(
    tmp_path: Path, field, value, match
):
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["totals"][field] = value
    with pytest.raises(OutputError, match=match):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    assert not (tmp_path / "out" / "run-x").exists()


def test_r1_c3_per_node_totals_and_fingerprints_are_reconciled(tmp_path: Path):
    """A node whose declared token total or fingerprint disagrees is refused -- twice over.

    Both layers are exercised: a manifest that contradicts the trusted post-Pass-1 result is
    refused on that ground (R3-A), and a manifest that agrees with a *wrong* expectation is still
    refused because neither describes the bytes on disk.
    """
    manifest, records, expected = _publishable(tmp_path, count=3)

    # R4-E: physical reconciliation runs first, so a node whose declared numbers contradict its
    # own records fails on that concrete mismatch rather than on a downstream identity check.
    tokens = json.loads(json.dumps(manifest))
    tokens["nodes"][0]["selected_serialized_tokens"] = 20
    with pytest.raises(OutputError, match="but the records sum to 21"):
        publish_atomic(tmp_path / "out1", "run-x", tokens, list(records), expected=expected)

    fingerprint = json.loads(json.dumps(manifest))
    fingerprint["nodes"][0]["selection_fingerprint"] = "0" * 64
    with pytest.raises(OutputError, match="fingerprint disagrees"):
        publish_atomic(tmp_path / "out2", "run-x", fingerprint, list(records), expected=expected)

    # Now make the expectation agree with the lie. The physical reconstruction still refuses.
    # And a manifest that is physically honest but whose expectation lies is refused on the
    # Layer-2 comparison, which is what runs once the physical facts have checked out.
    node = dict(manifest["nodes"][0])
    node["selection_fingerprint"] = "0" * 64
    lying = _expected_for(
        _pass1_for(
            [dict(node)],
            {"b_x": ["ib_x"]},
            {k: v for k, v in manifest["totals"].items() if k != "shards"},
        )
    )
    honest = json.loads(json.dumps(manifest))
    honest["stage_i_run"] = _stage_i_run_block(lying)
    with pytest.raises(OutputError, match="trusted post-Pass-1 result committed to"):
        publish_atomic(tmp_path / "out3", "run-x", honest, list(records), expected=lying)


def test_r1_c3b_per_node_token_totals_are_reconciled_against_the_records(tmp_path: Path):
    """The per-node physical token reconciliation, on a fixture where totals stay truthful.

    With a single node the global total already catches a wrong per-node count, so the per-node
    path needs two nodes: one understated and one overstated by the same amount, leaving the
    realization total honest and the node numbers both wrong.
    """
    records = []
    nodes = []
    projection = {}
    for source_id, claimed in (("b_p", 20), ("b_q", 22)):
        identities = [hashlib.sha256(f"{source_id}-{i}".encode()).hexdigest() for i in range(3)]
        for index, identity in enumerate(identities):
            records.append(
                _record(
                    source_id=source_id,
                    input_binding_id=f"ib_{source_id}",
                    stable_input_record_ordinal=index,
                    selection_ordinal_within_node=index,
                    cleaned_text_sha256=identity,
                    raw_sha256=hashlib.sha256(f"r-{identity}".encode()).hexdigest(),
                    input_record_sha256=hashlib.sha256(f"w-{identity}".encode()).hexdigest(),
                    canonical_fingerprint=hashlib.sha256(f"f-{identity}".encode()).hexdigest(),
                    training_text=f"{source_id} {index}",
                )
            )
        projection[source_id] = [f"ib_{source_id}"]
        nodes.append({
            "source_id": source_id,
            "stage": "stage_b",
            "target_serialized_tokens": 30,
            "branch": "ORDINARY",
            "selection_mode": "SEEDED_HASH",
            "selected_identities": 3,
            # Wrong per node, right in aggregate: 20 + 22 == 42 == 6 records x 7 tokens.
            "selected_serialized_tokens": claimed,
            "selection_fingerprint": oracle_fingerprint(identities),
            "selection_sequence_commitment": oracle_sequence_commitment(
                source_id, "stage_b", list(enumerate(identities))
            ),
            "crossing_identity": None,
            "actual_overshoot_tokens": 0,
            "input_binding_ids": [f"ib_{source_id}"],
        })
    totals = {
        "records": 6,
        "content_tokens": 30,
        "serialized_tokens": 42,
        "unique_cleaned_identities": 6,
    }
    binding_digests = {"ib_b_p": "6" * 64, "ib_b_q": "7" * 64}
    expected = _expected_for(
        _pass1_for(
            [dict(n) for n in nodes],
            projection,
            dict(totals),
            binding_digests=binding_digests,
        )
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "shards": [],
        "totals": {**totals, "shards": 0},
        "nodes": [dict(n) for n in nodes],
        "node_binding_projection": projection,
        "ownership_matrix": {},
        "bindings": dict(binding_digests),
        "environment": {
            "python_executable": REQUIRED_PYTHON_EXECUTABLE,
            "python_version": REQUIRED_PYTHON_VERSION,
            "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
        },
        "h_binding": dict(_FIXTURE_H_BINDING),
        "stage_i_run": _stage_i_run_block(expected),
    }
    with pytest.raises(OutputError, match="but the records sum to 21"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


# ---- D-J: the strict consumer must enforce physical invariants ------------------


def _corrupt_published(tmp_path: Path, name: str, mutate) -> tuple[Path, TrustedExpectedResult]:
    """Publish a valid realization, then rewrite one shard through `mutate`.

    The manifest is left describing the original, which is exactly the situation a consumer must
    detect: bytes on disk that no longer match what the result claims about itself. The trusted
    expectation is returned alongside, captured before the tampering, because that is how a real
    consumer gets one: from the frozen post-Pass-1 artifact, not from the result it is checking.
    """
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / name, "run-x", manifest, records, expected=expected)
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    lines = shard.read_bytes().split(b"\n")[:-1]
    shard.write_bytes(mutate(lines))
    return final, expected


def test_r1_d_consumer_rejects_reordered_selection_ordinals(tmp_path: Path):
    """D. Selection ordinals must form the exact contiguous domain, reconstructed from records."""

    def swap(lines):
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        first["selection_ordinal_within_node"], second["selection_ordinal_within_node"] = (
            second["selection_ordinal_within_node"],
            first["selection_ordinal_within_node"],
        )
        # Physical order is untouched; only the retained selection order is permuted.
        return (
            canonical_json_bytes(first)
            + canonical_json_bytes(second)
            + b"\n".join(lines[2:])
            + b"\n"
        )

    final, expected = _corrupt_published(tmp_path, "out", swap)
    # A permutation keeps the domain contiguous, so it must be caught by the fingerprint, which is
    # what actually pins the selected sequence.
    with pytest.raises(OutputError, match="fingerprint disagrees|SHA-256|shards"):
        load_published_realization(final, expected=expected)


def test_r1_d2_consumer_rejects_a_gap_in_selection_ordinals(tmp_path: Path):
    """A missing ordinal breaks contiguity and must be named as such."""

    def punch(lines):
        first = json.loads(lines[0])
        first["selection_ordinal_within_node"] = 99
        return canonical_json_bytes(first) + b"\n".join(lines[1:]) + b"\n"

    final, expected = _corrupt_published(tmp_path, "out", punch)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(OutputError, match="COMPLETE marker|not the contiguous domain"):
        load_published_realization(final, expected=expected)


def test_r1_e_consumer_rejects_false_manifest_totals(tmp_path: Path):
    """E. A published manifest whose totals do not describe its records must not load."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["totals"]["serialized_tokens"] = 22
    payload["nodes"][0]["selected_serialized_tokens"] = 22
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final, expected=expected)


def test_r1_f_consumer_rejects_duplicate_cleaned_identities(tmp_path: Path):
    """F. Global identity uniqueness, established from the published bytes."""

    def duplicate(lines):
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        second["cleaned_text_sha256"] = first["cleaned_text_sha256"]
        return (
            canonical_json_bytes(first)
            + canonical_json_bytes(second)
            + b"\n".join(lines[2:])
            + b"\n"
        )

    final, expected = _corrupt_published(tmp_path, "out", duplicate)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((OutputError, AuditError), match="COMPLETE marker|appears more than once"):
        load_published_realization(final, expected=expected)


def test_r1_g_consumer_rejects_noncanonical_record_bytes(tmp_path: Path):
    """G. A record whose bytes are not the canonical serialisation of its own content."""

    def spacer(lines):
        first = json.loads(lines[0])
        # Same content, non-canonical separators: a digest over these bytes says nothing.
        noncanonical = json.dumps(first, sort_keys=True, separators=(", ", ": ")).encode()
        return noncanonical + b"\n" + b"\n".join(lines[1:]) + b"\n"

    final, expected = _corrupt_published(tmp_path, "out", spacer)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((OutputError, AuditError), match="COMPLETE marker|not canonical"):
        load_published_realization(final, expected=expected)


def test_r1_h_consumer_rejects_wrong_physical_order(tmp_path: Path):
    """H. Physical layout order must be strictly ascending across the whole realization."""

    def reverse_two(lines):
        return b"\n".join([lines[1], lines[0]] + list(lines[2:])) + b"\n"

    final, expected = _corrupt_published(tmp_path, "out", reverse_two)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((OutputError, AuditError), match="COMPLETE marker|physical order"):
        load_published_realization(final, expected=expected)


def test_r1_i_consumer_rejects_missing_and_extra_records(tmp_path: Path):
    """I. A dropped or added record changes the physical truth and must be detected."""

    def drop(lines):
        return b"\n".join(lines[1:]) + b"\n"

    final, expected = _corrupt_published(tmp_path, "missing", drop)
    with pytest.raises((OutputError, AuditError)):
        load_published_realization(final, expected=expected)

    def add(lines):
        extra = json.loads(lines[-1])
        extra["stable_input_record_ordinal"] += 1000
        extra["cleaned_text_sha256"] = hashlib.sha256(b"extra").hexdigest()
        extra["selection_ordinal_within_node"] += 1000
        return b"\n".join(lines) + b"\n" + canonical_json_bytes(extra)

    final2, expected2 = _corrupt_published(tmp_path, "extra", add)
    with pytest.raises((OutputError, AuditError)):
        load_published_realization(final2, expected=expected2)


def test_r1_j_consumer_reconstructs_and_verifies_per_node_fingerprints(tmp_path: Path):
    """J. The node fingerprint is recomputed from the published records, via a longhand oracle."""
    manifest, records, expected = _publishable(tmp_path, count=5)
    fingerprint = oracle_fingerprint([r["cleaned_text_sha256"] for r in records])
    assert manifest["nodes"][0]["selection_fingerprint"] == fingerprint
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    loaded = load_published_realization(final, expected=expected)
    assert loaded["nodes"][0]["selection_fingerprint"] == fingerprint

    # And a manifest claiming any other fingerprint must not load.
    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["nodes"][0]["selection_fingerprint"] = "0" * 64
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final, expected=expected)


def test_r1_u_strict_consumer_validates_a_genuine_multi_shard_publication(tmp_path: Path):
    """U. End to end over more than one shard, with every physical invariant enforced."""
    count = RECORDS_PER_SHARD + 7
    manifest, _, _ = _publishable(tmp_path, count=1)
    identities = [hashlib.sha256(f"u-{i}".encode()).hexdigest() for i in range(count)]
    totals = {
        "records": count,
        "content_tokens": 5 * count,
        "serialized_tokens": 7 * count,
        "unique_cleaned_identities": count,
    }
    node = dict(manifest["nodes"][0])
    node.update({
        "selected_identities": count,
        "selected_serialized_tokens": 7 * count,
        "selection_fingerprint": oracle_fingerprint(identities),
        "selection_sequence_commitment": oracle_sequence_commitment(
            "b_x", "stage_b", [(i, identities[i]) for i in range(count)]
        ),
    })
    expected = _expected_for(_pass1_for([dict(node)], {"b_x": ["ib_x"]}, dict(totals)))
    manifest["totals"] = {**totals, "shards": 0}
    manifest["nodes"] = [node]
    manifest["stage_i_run"] = _stage_i_run_block(expected)

    def gen():
        for i in range(count):
            yield _record(
                stable_input_record_ordinal=i,
                selection_ordinal_within_node=i,
                cleaned_text_sha256=identities[i],
                raw_sha256=hashlib.sha256(f"ur-{i}".encode()).hexdigest(),
                input_record_sha256=hashlib.sha256(f"uw-{i}".encode()).hexdigest(),
                canonical_fingerprint=hashlib.sha256(f"uf-{i}".encode()).hexdigest(),
                training_text=f"doc {i}",
            )

    final = publish_atomic(tmp_path / "out", "run-multi", manifest, gen(), expected=expected)
    loaded = load_published_realization(final, expected=expected)
    assert loaded["totals"]["shards"] == 2 == len(loaded["shards"])
    assert loaded["shards"][0]["records"] == RECORDS_PER_SHARD
    assert loaded["shards"][1]["records"] == 7
    assert sum(1 for _ in iter_records(final, loaded)) == count
    # No consumer scratch may survive inside the published result.
    assert sorted(p.name for p in final.iterdir()) == [
        "COMPLETE",
        DOCUMENTS_DIRNAME,
        MANIFEST_FILENAME,
    ]


# ---- K-T: the authorized plan must govern runtime -------------------------------


def test_r1_k_authorize_plan_positive_path_yields_a_sealed_context(authorized, live_plan_path):
    """K. The exact positive path: a real plan, its true digest, a usable capability."""
    assert isinstance(authorized, AuthorizedIContext)
    assert authorized.plan_sha256 == sha256_file(live_plan_path)

    # R3-C/D: the handle carries no load-bearing state at all. Everything comes from a fresh
    # re-derivation off the authorization-time plan bytes.
    state = authorized.revalidate()
    assert state.plan["authorization_status"] == "NOT_AUTHORIZED"
    assert state.graph.graph_sha256 == state.plan["graph_sha256"]
    assert state.accepted.run_identity == state.plan["accepted_h"]["run_identity"]
    assert state.accepted.complete_sha256 == ACCEPTED_H_COMPLETE_SHA256
    assert state.tokenizer_path.name == "tokenizer.json"
    assert state.state_sha256 == authorized.authorized_state_sha256

    # The Layer-1 anchors are generated by the state's own bound fields.
    block = authorization_block(state)
    assert block["candidate_i_plan_sha256"] == authorized.plan_sha256
    assert block["authorized_state_sha256"] == state.state_sha256
    assert block["owner_graph_sha256"] == state.graph.graph_sha256
    assert block["h_complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256
    assert block["node_binding_projection_sha256"] == node_binding_projection_sha256(
        state.node_binding_projection
    )

    # Re-deriving twice must land on exactly the same canonical state.
    assert authorized.revalidate().state_sha256 == state.state_sha256


def test_r3_c_no_module_scope_registration_or_minting_api_exists(authorized):
    """R3-C. Codex promoted a hand-built lookalike with the module-visible registration helper.

    The reproduction was::

        manual = <field-for-field lookalike>
        _register_authorized(manual, _canonical_state_digest(manual))

    after which the manual object was accepted as an authorization. The repair is not a rename:
    the registry, its record type and the only operation that can write to it live inside a
    closure, and nothing that escapes the module can add an entry. This asserts the route is gone
    AND that no other module-scope callable can grant authority.
    """
    import inspect

    import pretrain.stage_i_realize_v1 as module

    for name in ("_AUTHORIZED", "_register_authorized", "_canonical_state_digest", "_SEAL"):
        assert not hasattr(module, name), f"module-scope authority route {name!r} is back"

    # Nothing at module scope may be a registration/minting API by any spelling.
    banned = ("register", "mint", "seal", "stamp", "grant")
    offenders = [
        name
        for name in dir(module)
        if callable(getattr(module, name, None))
        and name != "authorize_plan"
        and any(word in name.lower() for word in banned)
    ]
    assert not offenders, f"module scope exposes a registration-shaped API: {offenders}"

    # authorize_plan is the sole mint, and it takes the exact plan path and owner-held digest.
    parameters = list(inspect.signature(module.authorize_plan).parameters)
    assert parameters[:3] == ["plan_path", "expected_plan_sha256", "repo_root"]

    # A hand-built lookalike cannot be constructed, cannot be given a field, and is not authority.
    with pytest.raises(RealizationError, match="cannot be constructed"):
        AuthorizedIContext()
    manual = object.__new__(AuthorizedIContext)
    with pytest.raises(AttributeError):
        object.__setattr__(manual, "authorized", True)
    with pytest.raises(RealizationError, match="exact Stage-I context instance"):
        module._require_authorized(manual, "test")
    with pytest.raises(RealizationError, match="exact Stage-I context instance"):
        module.resolve_authorized_state(manual, "test")
    with pytest.raises(RealizationError, match="exact Stage-I context instance"):
        module.revalidate_authorized(manual)

    # ...and the genuine instance still works, or every negative above proves nothing.
    module._require_authorized(authorized, "test")


def test_r1_k2_only_the_exact_authorized_instance_is_an_authorization(authorized, tmp_path):
    """R2-C/R3-C. Authority is the exact registered instance, not a value carried in a field."""
    import copy

    from pretrain.stage_i_realize_v1 import _require_authorized, realize_and_publish

    # Copying is refused outright rather than yielding a powerless twin.
    with pytest.raises(RealizationError, match="must not be copied"):
        copy.copy(authorized)
    with pytest.raises(RealizationError, match="must not be copied"):
        copy.deepcopy(authorized)
    with pytest.raises(RealizationError, match="must not be pickled"):
        authorized.__reduce__()

    lookalike = object.__new__(AuthorizedIContext)
    with pytest.raises(RealizationError, match="exact Stage-I context instance"):
        _require_authorized(lookalike, "test")
    with pytest.raises(RealizationError, match="exact Stage-I context instance"):
        realize_and_publish(lookalike, out_dir=tmp_path / "o", work_dir=tmp_path / "w")

    # A subclass is a different instance too, isinstance notwithstanding.
    class Pretender(AuthorizedIContext):
        __slots__ = ()

    with pytest.raises(RealizationError, match="exact Stage-I context instance"):
        _require_authorized(object.__new__(Pretender), "test")

    _require_authorized(authorized, "test")


def test_r3_c_context_handle_exposes_no_load_bearing_state(authorized):
    """R3-D. There is no mutable graph/H/plan/path on the handle to substitute in the first place.

    Codex's R2 mutations all began ``object.__setattr__(context, "graph", ...)`` or
    ``context.accepted.census[...] = ...``. The handle now has empty ``__slots__``, so it has no
    ``__dict__`` and no data attribute to reach for.
    """
    for attribute in (
        "graph",
        "accepted",
        "plan",
        "authorities",
        "tokenizer_path",
        "reference_exclusion_path",
        "node_binding_projection",
        "environment",
        "bundle_files",
        "run_identity",
        "__dict__",
    ):
        assert not hasattr(authorized, attribute), (
            f"the context still exposes load-bearing state as {attribute!r}"
        )
    assert AuthorizedIContext.__slots__ == ("__weakref__",)
    for attribute in ("graph", "accepted", "plan", "run_identity"):
        with pytest.raises(AttributeError):
            object.__setattr__(authorized, attribute, None)


def test_r2_c_plan_bytes_changing_after_authorization_is_detected(live_plan_path, tmp_path):
    """A plan edited on disk after authorization must fail revalidation."""
    copied = tmp_path / "plan.json"
    copied.write_bytes(live_plan_path.read_bytes())
    context = authorize_plan(copied, sha256_file(copied), ROOT)
    context.revalidate()
    payload = json.loads(copied.read_text())
    payload["seed"] = payload["seed"] + 1
    copied.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(RealizationError):
        context.revalidate()


def test_r1_l_wrong_externally_supplied_plan_sha_is_rejected(live_plan_path):
    """L. Authorization is against the owner's digest, not the file's self-description."""
    with pytest.raises(RealizationError, match="is not the owner-supplied"):
        authorize_plan(live_plan_path, "0" * 64, ROOT)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.__setitem__("graph_sha256", "0" * 64), "SHA-256 mismatch"),
        (lambda p: p.__setitem__("seed", 1), "seed disagrees"),
        (
            lambda p: p["bound_authorities"].__setitem__("tokenizer_sha256", "0" * 64),
            "bound authorities disagree",
        ),
        (
            lambda p: p["authorities"]["tokenizer"].__setitem__("sha256", "0" * 64),
            "plan authority tokenizer",
        ),
        (
            lambda p: p["authorities"]["tokenizer"].__setitem__("path", "runs/elsewhere/tok.json"),
            "frozen canonical location",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("run_dir", "runs/some_other_h_run"),
            "frozen accepted Stage-H run directory",
        ),
        (
            lambda p: p["input_bindings"]["ib_dclm_edu"].__setitem__("documents_sha256", "0" * 64),
            "documents_sha256 disagrees",
        ),
        (
            lambda p: p["input_bindings"]["ib_dclm_edu"].__setitem__("total_physical_rows", 1),
            "total_physical_rows disagrees",
        ),
        (lambda p: p.__setitem__("node_order", ["nope"]), "node order disagrees"),
        (
            lambda p: p["nodes"][0].__setitem__("target_serialized_tokens", 1),
            "disagrees with the frozen graph",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("complete_sha256", "0" * 64),
            "exact accepted Stage-H COMPLETE bytes",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("predictions_sha256", "0" * 64),
            "is not the accepted",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("census_sha256", "0" * 64),
            "census SHA-256",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("candidate_plan_sha256", "0" * 64),
            "candidate plan SHA disagrees",
        ),
        (
            lambda p: p["shard_policy"].__setitem__("records_per_shard", 7),
            "shard policy record count disagrees",
        ),
        (
            lambda p: p["shard_policy"].__setitem__("version", "nope"),
            "shard policy version disagrees",
        ),
        (
            lambda p: p["shard_policy"].__setitem__("rule", "records, whatever"),
            "not the frozen shard rule literal",
        ),
        (
            lambda p: p["selection_rules"].__setitem__("representative_rule", "min(a, b)"),
            "not the frozen selector-v1 literal",
        ),
        (
            lambda p: p["selection_rules"].__setitem__("physical_locator_rule", "first one"),
            "physical locator rule is not the frozen literal",
        ),
        (
            lambda p: p.__setitem__("output_schema_version", "nope"),
            "schema versions disagree",
        ),
        (
            lambda p: p.__setitem__("implementation_bundle_sha256", "0" * 64),
            "different Stage-I implementation bundle",
        ),
        (
            lambda p: p["implementation_files"].__setitem__(
                "pretrain/stage_i_select_v1.py", "0" * 64
            ),
            "implementation file digests disagree",
        ),
        (
            lambda p: p.__setitem__("implementation_commit", "0" * 40),
            "is not the plan's implementation commit",
        ),
        (
            lambda p: p["environment_contract"].__setitem__("tokenizers_version", "0.22.1"),
            "environment contract disagrees",
        ),
        (
            lambda p: p.__setitem__("authorization_status", "AUTHORIZED"),
            "must not carry an owner authorization",
        ),
        (lambda p: p.__setitem__("resume_supported", True), "must not enable resume"),
    ],
)
def test_r1_mnopqrs_plan_mismatches_are_all_rejected(live_plan_path, tmp_path, mutation, match):
    """M-S. Every binding the plan makes is checked; a mutated plan cannot authorize a run.

    The mutated plan is re-hashed and authorized against its OWN true digest, so these prove the
    binding checks fire rather than merely re-proving the digest check from test L.
    """
    path, digest = _mutated_plan(live_plan_path, tmp_path, mutation)
    # GraphError is included because the owner graph is now loaded *bound* to the plan's declared
    # digest, so a mutated graph_sha256 is refused by the loader before any of it is parsed.
    with pytest.raises((RealizationError, GraphError), match=match):
        authorize_plan(path, digest, ROOT)


def test_r1_t_published_run_identity_must_be_generated_by_its_own_fields(tmp_path: Path):
    """T. A result claiming an identity its own bound fields do not generate must not load."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    loaded = load_published_realization(final, expected=expected)
    block = loaded["stage_i_run"]
    assert recompute_stage_i_run_identity(block) == block["run_identity"]

    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["stage_i_run"]["h_complete_sha256"] = "0" * 64
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final, expected=expected)


def test_r1_t2_run_identity_changes_with_every_bound_field():
    """Each field genuinely participates: flipping any one must move the identity.

    R3-A extends the identity with the authorization-time canonical state digest and the
    post-Pass-1 result, so those are exercised here too. Without that, a resealed realization
    could carry a different expectation under the same name.
    """
    authorization = dict(
        candidate_i_plan_sha256="1" * 64,
        authorized_state_sha256="9" * 64,
        implementation_commit="a" * 40,
        implementation_bundle_sha256="2" * 64,
        plan_schema_version=PLAN_SCHEMA,
        output_schema_version=RECORD_SCHEMA,
        manifest_schema_version=MANIFEST_SCHEMA,
        shard_policy_version=SHARD_POLICY_VERSION,
        records_per_shard=RECORDS_PER_SHARD,
        h_run_identity="3" * 64,
        h_complete_sha256="4" * 64,
        h_census_sha256="5" * 64,
        h_predictions_sha256="6" * 64,
        owner_graph_sha256="7" * 64,
        node_binding_projection_sha256="8" * 64,
        environment_sha256="d" * 64,
        binding_document_digests_sha256="e" * 64,
    )
    downstream = dict(
        post_pass1_result_identity_sha256="b" * 64,
        selection_sequence_commitment_map_sha256="c" * 64,
        post_pass1_result_identity_schema=PASS1_RESULT_SCHEMA,
        selection_sequence_commitment_version=SEQ_SCHEMA,
    )
    reference = stage_i_published_run_identity(authorization, **downstream)
    seen = {reference}
    for key, value in authorization.items():
        altered = dict(authorization)
        altered[key] = 99 if isinstance(value, int) else "z" * len(str(value))
        identity = stage_i_published_run_identity(altered, **downstream)
        assert identity != reference, f"{key} does not participate in the run identity"
        seen.add(identity)
    for key, value in downstream.items():
        altered = dict(downstream)
        altered[key] = "z" * len(value)
        identity = stage_i_published_run_identity(authorization, **altered)
        assert identity != reference, f"{key} does not participate in the run identity"
        seen.add(identity)
    assert len(seen) == len(authorization) + len(downstream) + 1


def test_r1_v_implementation_bundle_covers_every_stage_i_production_module():
    """The bundle must name every module whose bytes can change a Stage-I result.

    Caught during R1: the new audit module decides whether a realization may be published and what
    the consumer accepts, but was initially absent from the bundle -- so editing it would not have
    invalidated an authorized plan. Enumerating the directory keeps this honest as modules are
    added rather than relying on someone remembering to update a tuple.
    """
    on_disk = sorted(
        f"pretrain/{path.name}" for path in (ROOT / "pretrain").glob("stage_i_*_v1.py")
    )
    assert on_disk, "no Stage-I v1 production modules found"
    assert sorted(IMPLEMENTATION_BUNDLE_FILES) == on_disk, (
        "IMPLEMENTATION_BUNDLE_FILES does not cover every Stage-I v1 production module"
    )
    # And every named member must actually exist and hash.
    files = implementation_files(ROOT)
    assert set(files) == set(IMPLEMENTATION_BUNDLE_FILES)
    assert implementation_bundle_sha256(files) == implementation_bundle_sha256(files)


# ================================================================================
# R2 REPAIR REGRESSIONS
# ================================================================================


def _reseal(final: Path, mutate_lines) -> Path:
    """Rewrite shard 0 through `mutate_lines`, then reseal EVERY unrelated digest and total.

    Resealing matters. Codex found R1's ordinal regression could fail for a stale shard digest or
    a marker mismatch and never reach the invariant it claimed to test. Here the shard digest,
    byte count, manifest and COMPLETE marker are all made self-consistent again, so the fixture
    gets past every generic check and can only fail on the semantic invariant under test.
    """
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    lines = shard.read_bytes().split(b"\n")[:-1]
    shard.write_bytes(mutate_lines(lines))
    payload = shard.read_bytes()

    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    manifest["shards"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["shards"][0]["bytes"] = len(payload)
    manifest["shards"][0]["records"] = payload.count(b"\n")
    manifest_bytes = canonical_json_bytes(manifest)
    (final / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (final / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes({
            "marker": "COMPLETE",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "record_schema_version": RECORD_SCHEMA,
            "manifest_schema_version": MANIFEST_SCHEMA,
            "h_run_identity": manifest["h_binding"]["h_run_identity"],
            "stage_i_run_identity": manifest["stage_i_run"]["run_identity"],
        })
    )
    return final


def test_r2_a_self_consistent_ordinal_swap_is_rejected(tmp_path: Path):
    """R2-A. The exact Codex case: a fully resealed publication with ordinals permuted.

    The ordinal domain stays contiguous, the frozen set fingerprint is unchanged (it sorts), every
    digest and total is resealed. The ONLY thing that differs is which identity sits at which
    ordinal -- and that is precisely what the new order-sensitive commitment binds.
    """
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    load_published_realization(final, expected=expected)  # valid before tampering

    def swap(lines):
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        first["selection_ordinal_within_node"], second["selection_ordinal_within_node"] = (
            second["selection_ordinal_within_node"],
            first["selection_ordinal_within_node"],
        )
        return (
            canonical_json_bytes(first)
            + canonical_json_bytes(second)
            + b"\n".join(lines[2:])
            + b"\n"
        )

    _reseal(final, swap)
    with pytest.raises(OutputError, match="selection-sequence commitment disagrees"):
        load_published_realization(final, expected=expected)


def test_r2_a_identity_moved_to_another_ordinal_is_rejected(tmp_path: Path):
    """A contiguous domain with the wrong sequence must still fail."""
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    def rotate(lines):
        parsed = [json.loads(line) for line in lines]
        ordinals = [r["selection_ordinal_within_node"] for r in parsed]
        for record, ordinal in zip(parsed, ordinals[1:] + ordinals[:1], strict=True):
            record["selection_ordinal_within_node"] = ordinal
        return b"".join(canonical_json_bytes(r) for r in parsed)

    _reseal(final, rotate)
    with pytest.raises(OutputError, match="selection-sequence commitment disagrees"):
        load_published_realization(final, expected=expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_ordinal", "not contiguous"),
        ("gap_ordinal", "not contiguous"),
        ("extra_ordinal", "not contiguous"),
    ],
)
def test_r2_a_ordinal_domain_violations_are_rejected(tmp_path: Path, mutation, match):
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    def mutate(lines):
        parsed = [json.loads(line) for line in lines]
        if mutation == "duplicate_ordinal":
            parsed[1]["selection_ordinal_within_node"] = 0
        elif mutation == "gap_ordinal":
            parsed[2]["selection_ordinal_within_node"] = 9
        else:
            parsed[3]["selection_ordinal_within_node"] = 7
        return b"".join(canonical_json_bytes(r) for r in parsed)

    _reseal(final, mutate)
    with pytest.raises((OutputError, AuditError), match=match):
        load_published_realization(final, expected=expected)


def test_r2_a_sequence_commitment_is_order_sensitive_and_versioned():
    """The commitment must distinguish permutations the frozen fingerprint cannot."""
    from pretrain.stage_i_audit_v1 import (
        SELECTION_SEQUENCE_SCHEMA,
        selection_sequence_commitment,
    )

    ids = [hashlib.sha256(f"s{i}".encode()).hexdigest() for i in range(4)]
    straight = [(i, ids[i]) for i in range(4)]
    swapped = [(0, ids[1]), (1, ids[0]), (2, ids[2]), (3, ids[3])]

    assert selection_sequence_commitment(
        source_id="b_x", stage="stage_b", pairs=straight
    ) == oracle_sequence_commitment("b_x", "stage_b", straight)
    assert selection_sequence_commitment(
        source_id="b_x", stage="stage_b", pairs=swapped
    ) == oracle_sequence_commitment("b_x", "stage_b", swapped)
    assert oracle_sequence_commitment("b_x", "stage_b", straight) != oracle_sequence_commitment(
        "b_x", "stage_b", swapped
    )
    # ...while the frozen fingerprint genuinely cannot tell them apart. That is why this is an
    # addition rather than a redefinition.
    assert selection_fingerprint_v1([i for _, i in straight]) == selection_fingerprint_v1([
        i for _, i in swapped
    ])
    # Source, stage and length all participate.
    assert oracle_sequence_commitment("b_y", "stage_b", straight) != oracle_sequence_commitment(
        "b_x", "stage_b", straight
    )
    assert oracle_sequence_commitment("b_x", "stage_a", straight) != oracle_sequence_commitment(
        "b_x", "stage_b", straight
    )
    assert oracle_sequence_commitment("b_x", "stage_b", straight[:3]) != oracle_sequence_commitment(
        "b_x", "stage_b", straight
    )
    assert SELECTION_SEQUENCE_SCHEMA == "petitgpt-stage-i-selection-sequence-v1"


# ---- R2-B: closed node -> allowed-binding membership ----------------------------


def test_r2_b_undeclared_input_binding_is_rejected(tmp_path: Path):
    """R2-B. Codex published and consumed `ib_not_declared` while the manifest declared `ib_x`."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    tampered = [dict(r) for r in records]
    tampered[0]["input_binding_id"] = "ib_not_declared"
    with pytest.raises(AuditError, match="not authorized to draw from input binding"):
        publish_atomic(tmp_path / "out", "run-x", manifest, tampered, expected=expected)
    assert not (tmp_path / "out" / "run-x").exists()


def test_r2_b_undeclared_binding_is_rejected_by_the_consumer_too(tmp_path: Path):
    """The same record must not survive on the read side either, fully resealed."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    def swap_binding(lines):
        parsed = [json.loads(line) for line in lines]
        parsed[0]["input_binding_id"] = "ib_not_declared"
        return b"".join(canonical_json_bytes(r) for r in parsed)

    _reseal(final, swap_binding)
    with pytest.raises((OutputError, AuditError), match="not authorized|physical order"):
        load_published_realization(final, expected=expected)


def test_r2_b_authorized_binding_attached_to_the_wrong_node_is_rejected(tmp_path: Path):
    """A globally-declared binding used by a node it is not authorized for."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["node_binding_projection"] = {"b_x": ["ib_x"], "b_other": ["ib_other"]}
    manifest["nodes"].append({
        "source_id": "b_other",
        "stage": "stage_b",
        "target_serialized_tokens": 10,
        "branch": "ORDINARY",
        "selection_mode": "SEEDED_HASH",
        "selected_identities": 0,
        "selected_serialized_tokens": 0,
        "selection_fingerprint": "0" * 64,
        "selection_sequence_commitment": "0" * 64,
        "input_binding_ids": ["ib_other"],
        "crossing_identity": None,
        "actual_overshoot_tokens": 0,
    })
    manifest["stage_i_run"] = _stage_i_run_block(
        expected,
        node_binding_projection_sha256=node_binding_projection_sha256({
            "b_x": ["ib_x"],
            "b_other": ["ib_other"],
        }),
    )
    tampered = [dict(r) for r in records]
    tampered[0]["input_binding_id"] = "ib_other"  # authorized, but not for b_x
    with pytest.raises(AuditError, match="not authorized to draw from input binding"):
        publish_atomic(tmp_path / "out", "run-x", manifest, tampered, expected=expected)


def test_r2_b_manifest_projection_must_match_the_run_identity_digest(tmp_path: Path):
    """A widened projection is refused against the trusted authority, and against its digest."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    widened = json.loads(json.dumps(manifest))
    widened["node_binding_projection"] = {"b_x": ["ib_extra", "ib_x"]}
    with pytest.raises(OutputError, match="disagrees with the trusted node/binding authority"):
        publish_atomic(tmp_path / "out", "run-x", widened, list(records), expected=expected)

    # And the manifest's own projection digest must still describe its own projection, so a
    # widened projection cannot pass even the self-consistency check it carries.
    final = publish_atomic(tmp_path / "ok", "run-x", manifest, list(records), expected=expected)
    published = json.loads((final / MANIFEST_FILENAME).read_text())
    published["node_binding_projection"] = {"b_x": ["ib_extra", "ib_x"]}
    with pytest.raises(OutputError, match="node_binding_projection_sha256 does not describe"):
        validate_manifest(published)


def test_r2_b_projection_must_cover_exactly_the_declared_nodes(tmp_path: Path):
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["node_binding_projection"] = {"b_other": ["ib_x"]}
    manifest["stage_i_run"] = _stage_i_run_block(
        expected,
        node_binding_projection_sha256=node_binding_projection_sha256({"b_other": ["ib_x"]}),
    )
    # The manifest's projection now contradicts the trusted authority, which is checked before
    # any of the manifest's self-consistency claims; if it did not fire, the streaming audit
    # would reject the record's unknown node. Either is a correct refusal.
    with pytest.raises(
        (OutputError, AuditError),
        match=(
            "trusted post-Pass-1 result binds|not in the authorized node/binding projection"
            "|must cover exactly the declared nodes"
        ),
    ):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


def test_r2_b_plan_projection_is_derived_from_the_frozen_graph(authorized):
    """The authorized projection must be exactly the graph's node -> binding relation."""
    state = authorized.revalidate()
    from_graph = {
        node.source_id: tuple(sorted(set(node.input_binding_ids))) for node in state.graph.nodes
    }
    assert dict(state.node_binding_projection) == from_graph
    # The authorized plan is held deeply immutable, so its JSON arrays are tuples.
    assert dict(state.plan["node_binding_projection"]) == from_graph
    # R3-B: every projected binding must be a plan-authorized global input binding.
    assert set(state.authorized_input_binding_ids) == set(state.graph.bindings)
    assert {b for allowed in from_graph.values() for b in allowed} <= set(
        state.authorized_input_binding_ids
    )
    # structured_tutorial is the union node; it must carry both of its frozen bindings.
    assert len(from_graph["b_structured_tutorial"]) == 2


# ---- R2-D: closed candidate-plan schema -----------------------------------------


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p["authorities"].pop("hq_policy"), "authority set is not the exact required"),
        (lambda p: p.pop("selection_rules"), "missing field"),
        (lambda p: p.__setitem__("surprise", 1), "unknown field"),
        (lambda p: p["accepted_h"].__setitem__("surprise", 1), "accepted_h carries unknown"),
        (lambda p: p["shard_policy"].__setitem__("surprise", 1), "shard_policy carries unknown"),
        (
            lambda p: p["authorities"].__setitem__(
                "bogus", {"path": "README.md", "sha256": "0" * 64}
            ),
            "authority set is not the exact required",
        ),
        (lambda p: p.pop("realization_label"), "missing field"),
        (lambda p: p.pop("node_order"), "missing field"),
        (lambda p: p.pop("node_binding_projection"), "missing field"),
        (lambda p: p["accepted_h"].pop("complete_sha256"), "accepted_h is missing"),
        (lambda p: p["authorities"]["tokenizer"].pop("path"), "authorities.tokenizer is missing"),
        (
            lambda p: p["authorities"]["tokenizer"].__setitem__("path", "README.md"),
            "frozen canonical location",
        ),
        (
            lambda p: p["shard_policy"].__setitem__("rule", 50000),
            "shard_policy.rule must be an exact string",
        ),
        (
            lambda p: p["selection_rules"].__setitem__("representative_rule", 1),
            "selection_rules.representative_rule must be an exact string",
        ),
        (
            lambda p: p["environment_contract"]["observed_at_generation"].__setitem__(
                "unknown_nested_field", "accepted"
            ),
            "observed_at_generation carries unknown",
        ),
        (
            lambda p: p["environment_contract"]["observed_at_generation"].pop("python_version"),
            "observed_at_generation is missing",
        ),
        (
            lambda p: p["input_bindings"]["ib_dclm_edu"].__setitem__(
                "total_physical_rows", "3545224"
            ),
            "total_physical_rows must be an exact integer",
        ),
        (
            lambda p: p.__setitem__("seed", "5088999448999271579"),
            "seed must be an exact integer",
        ),
        (
            lambda p: p["bound_authorities"].pop("tokenizer_sha256"),
            "not the exact frozen graph authority key set",
        ),
        (
            lambda p: p["implementation_files"].__setitem__("pretrain/other.py", "0" * 64),
            "not the exact Stage-I bundle member list",
        ),
        (
            lambda p: p["authorities"]["hq_policy"].__setitem__("sha256", "0" * 64),
            "plan authority hq_policy",
        ),
        (lambda p: p["nodes"][0].pop("selection_mode"), "nodes\\[0\\] is missing"),
        (lambda p: p["nodes"][0].__setitem__("surprise", 1), "nodes\\[0\\] carries unknown"),
        (
            lambda p: p["input_bindings"]["ib_dclm_edu"].__setitem__("surprise", 1),
            "input_bindings.ib_dclm_edu carries unknown",
        ),
        (
            lambda p: p["node_binding_projection"].__setitem__("b_dclm_edu", ["ib_finewiki_en"]),
            "disagrees with that node",
        ),
        (
            lambda p: p["node_binding_projection"].__setitem__("b_dclm_edu", ["ib_wrong"]),
            "outside the plan-authorized global input-binding set",
        ),
        (
            lambda p: p["environment_contract"].pop("tokenizers_version"),
            "environment_contract is missing",
        ),
        (lambda p: p.__setitem__("schema_version", "petitgpt-i-candidate-plan-v1"), "is not"),
    ],
)
def test_r2_d_closed_plan_schema_rejects(live_plan_path, tmp_path, mutation, match):
    """R2-D. Every level of the plan is a closed schema, and the authority set is exact.

    Each mutated plan is rehashed and authorized against its OWN true digest, so these prove the
    schema checks fire rather than re-proving the digest check.
    """
    path, digest = _mutated_plan(live_plan_path, tmp_path, mutation)
    with pytest.raises(RealizationError, match=match):
        authorize_plan(path, digest, ROOT)


def test_r2_d_required_authority_set_is_closed_and_exact(live_plan_path):
    from pretrain.stage_i_realize_v1 import REQUIRED_AUTHORITIES

    plan = json.loads(live_plan_path.read_text())
    assert frozenset(plan["authorities"]) == REQUIRED_AUTHORITIES
    assert "hq_policy" in REQUIRED_AUTHORITIES
    assert plan["schema_version"] == "petitgpt-i-candidate-plan-v4"


# ---- R2-E: bounded audit memory --------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2, 3, 17, 500])
def test_r2_e_streaming_fingerprint_is_byte_identical_to_the_frozen_one(count):
    """R2-E. The streaming reproduction must equal `selection_fingerprint_v1` exactly.

    The frozen fingerprint is not being redefined -- only computed without buffering the node's
    identity list. Equality against the frozen function is the whole point.
    """
    import random

    from pretrain.stage_i_audit_v1 import StreamingNodeFingerprint

    identities = [hashlib.sha256(f"f{i}".encode()).hexdigest() for i in range(count)]
    shuffled = list(identities)
    random.Random(1234).shuffle(shuffled)

    streaming = StreamingNodeFingerprint(count)
    for identity in sorted(shuffled):
        streaming.update(identity)
    assert streaming.hexdigest() == selection_fingerprint_v1(shuffled)
    assert streaming.hexdigest() == oracle_fingerprint(identities)


def test_r2_e_streaming_fingerprint_refuses_wrong_count_or_order():
    from pretrain.stage_i_audit_v1 import StreamingNodeFingerprint

    ids = sorted(hashlib.sha256(f"g{i}".encode()).hexdigest() for i in range(3))
    short = StreamingNodeFingerprint(3)
    short.update(ids[0])
    with pytest.raises(AuditError, match="expected 3"):
        short.hexdigest()

    over = StreamingNodeFingerprint(1)
    over.update(ids[0])
    with pytest.raises(AuditError, match="more identities than the declared count"):
        over.update(ids[1])

    unordered = StreamingNodeFingerprint(2)
    unordered.update(ids[1])
    with pytest.raises(AuditError, match="ascending order"):
        unordered.update(ids[0])


def test_r2_e_audit_builds_no_node_sized_identity_list():
    """Structural: neither fingerprint reconstruction may accumulate a per-node list."""
    source = (ROOT / "pretrain/stage_i_audit_v1.py").read_text()
    tree = ast.parse(source)
    for name in ("_node_fingerprints", "_selection_sequences"):
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        appends = {
            n.func.attr
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "append" not in appends, f"{name} accumulates a per-node list"
        assert "sorted" not in {
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }, f"{name} re-sorts an in-memory collection"


def test_r2_e_external_merge_uses_bounded_fanin(tmp_path: Path):
    """R2-E. Many spill runs must merge in generations, not all be opened at once."""
    from pretrain.stage_i_audit_v1 import MAX_MERGE_FANIN, ExternalSorter

    assert MAX_MERGE_FANIN <= 16
    sorter = ExternalSorter(tmp_path / "sort", chunk_lines=4)
    total = 4 * MAX_MERGE_FANIN * 3  # forces several merge generations
    keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(total)]
    for key in keys:
        sorter.add((key,), "p")
    real_open = open
    peak = {"open": 0, "now": 0}

    class _Counting:
        def __init__(self, handle):
            self._handle = handle
            peak["now"] += 1
            peak["open"] = max(peak["open"], peak["now"])

        def __iter__(self):
            return iter(self._handle)

        def close(self):
            peak["now"] -= 1
            self._handle.close()

        def write(self, data):
            return self._handle.write(data)

        def flush(self):
            return self._handle.flush()

        def fileno(self):
            return self._handle.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    import pretrain.stage_i_audit_v1 as audit_module

    audit_module.open = lambda *a, **k: _Counting(real_open(*a, **k))
    try:
        merged = [key for (key,), _payload in sorter.sorted_items(1)]
    finally:
        audit_module.open = real_open
        sorter.close()

    assert merged == sorted(keys)
    assert peak["open"] <= MAX_MERGE_FANIN + 1, (
        f"peak simultaneously-open runs was {peak['open']}, above the declared fan-in"
    )


def test_r2_e_audit_result_is_correct_under_many_spill_generations(tmp_path: Path):
    """A tiny sort chunk forces multipass merging; the audit result must be unchanged."""
    manifest, records, expected = _publishable(tmp_path, count=40)
    final = publish_atomic(
        tmp_path / "out", "run-x", manifest, records, sort_chunk_lines=3, expected=expected
    )
    loaded = load_published_realization(final, sort_chunk_lines=3, expected=expected)
    assert loaded["totals"]["records"] == 40
    assert loaded["nodes"][0]["selection_fingerprint"] == oracle_fingerprint([
        r["cleaned_text_sha256"] for r in records
    ])


# ================================================================================
# R3 REPAIR REGRESSIONS — one per Codex R2 re-review finding
# ================================================================================


def _fully_reseal(final: Path, mutate_lines, mutate_manifest=None) -> Path:
    """Rewrite shard 0, then reseal EVERY digest, total, identity and marker field.

    Stronger than ``_reseal``: this also regenerates the manifest's own sequence-commitment map
    digest and the published run identity, so a fixture cannot fail on a leftover
    self-inconsistency instead of on the invariant under test.
    """
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    lines = shard.read_bytes().split(b"\n")[:-1]
    shard.write_bytes(mutate_lines(lines))
    payload = shard.read_bytes()

    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    manifest["shards"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["shards"][0]["bytes"] = len(payload)
    manifest["shards"][0]["records"] = payload.count(b"\n")
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    run_block = dict(manifest["stage_i_run"])
    run_block["selection_sequence_commitment_map_sha256"] = (
        selection_sequence_commitment_map_sha256({
            entry["source_id"]: entry["selection_sequence_commitment"]
            for entry in manifest["nodes"]
        })
    )
    run_block["node_binding_projection_sha256"] = node_binding_projection_sha256(
        manifest["node_binding_projection"]
    )
    run_block.pop("run_identity", None)
    run_block["run_identity"] = recompute_stage_i_run_identity(run_block)
    manifest["stage_i_run"] = run_block

    manifest_bytes = canonical_json_bytes(manifest)
    (final / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (final / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes({
            "marker": "COMPLETE",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "record_schema_version": RECORD_SCHEMA,
            "manifest_schema_version": MANIFEST_SCHEMA,
            "h_run_identity": manifest["h_binding"]["h_run_identity"],
            "stage_i_run_identity": manifest["stage_i_run"]["run_identity"],
        })
    )
    return final


def _function_source(module_relative: str, name: str):
    tree = ast.parse((ROOT / module_relative).read_text())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    )


def _call_lines(function: ast.AST, name: str) -> list[int]:
    lines = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            func = node.func
            target = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if target == name:
                lines.append(node.lineno)
    return sorted(lines)


# ---- R3-A: the trusted post-Pass-1 expected result ------------------------------


def test_r3_a_fully_resealed_sequence_change_is_rejected(tmp_path: Path):
    """R3-A. The exact Codex R2 case, with nothing left inconsistent to trip over.

    Codex showed that a fully resealed physical result could permute the ordinal -> identity
    sequence, restate the manifest's own expected commitment to describe the permutation, reseal
    every digest and the COMPLETE marker, and still be accepted -- because the audit's expected
    value came from the same document it was checking.

    Here the reseal is complete: shard bytes, shard digest, byte count, the manifest's per-node
    commitment, the commitment-map digest and the published run identity are all regenerated. The
    only thing held fixed is the trusted post-Pass-1 result, which was frozen before the bytes
    existed and lives outside the realization. That is what refuses it.
    """
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    load_published_realization(final, expected=expected)  # valid before tampering

    identities = [hashlib.sha256(f"id-{i}".encode()).hexdigest() for i in range(4)]
    permuted = [(0, identities[1]), (1, identities[0]), (2, identities[2]), (3, identities[3])]

    def swap(lines):
        first, second = json.loads(lines[0]), json.loads(lines[1])
        first["selection_ordinal_within_node"], second["selection_ordinal_within_node"] = (
            second["selection_ordinal_within_node"],
            first["selection_ordinal_within_node"],
        )
        return (
            canonical_json_bytes(first)
            + canonical_json_bytes(second)
            + b"\n".join(lines[2:])
            + b"\n"
        )

    def restate(obj):
        obj["nodes"][0]["selection_sequence_commitment"] = oracle_sequence_commitment(
            "b_x", "stage_b", permuted
        )

    _fully_reseal(final, swap, restate)
    with pytest.raises(OutputError, match="disagrees with the trusted post-Pass-1 expectation"):
        load_published_realization(final, expected=expected)


def test_r3_a_resealed_sequence_without_restating_the_map_is_rejected(tmp_path: Path):
    """The other half of the same door: restate the node but not the map digest."""
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    def rotate(lines):
        parsed = [json.loads(line) for line in lines]
        ordinals = [r["selection_ordinal_within_node"] for r in parsed]
        for record, ordinal in zip(parsed, ordinals[1:] + ordinals[:1], strict=True):
            record["selection_ordinal_within_node"] = ordinal
        return b"".join(canonical_json_bytes(r) for r in parsed)

    _reseal(final, rotate)
    with pytest.raises(OutputError, match="selection-sequence commitment"):
        load_published_realization(final, expected=expected)


def test_r3_a_expected_result_is_frozen_before_any_materialization(tmp_path: Path):
    """Layer 2 must exist, and its digest must be fixed, before Pass 2 writes anything.

    Structural because the real thing needs the 151 GB corpus: the driver must freeze the
    post-Pass-1 result before it builds the record stream or calls the publisher, and must never
    reach for the run name before that.
    """
    function = _function_source("pretrain/stage_i_realize_v1.py", "realize_and_publish")
    freeze = _call_lines(function, "_freeze_pass1_result")
    build = _call_lines(function, "build_pass1_result")
    records = _call_lines(function, "iter_records_in_physical_order")
    publish = _call_lines(function, "publish_atomic")
    trusted = _call_lines(function, "trusted_expected_result")
    assert build and freeze and records and publish and trusted
    assert max(build) < min(freeze), "the expected result is built before it is frozen"
    assert max(freeze) < min(records), "Pass 2 must not start before the expectation is frozen"
    assert max(freeze) < min(publish), "publication must not precede the frozen expectation"
    assert max(trusted) < min(records)

    # And freezing refuses to overwrite, so a later run cannot replace a frozen expectation.
    from pretrain.stage_i_realize_v1 import _freeze_pass1_result

    payload = {"schema_version": PASS1_RESULT_SCHEMA}
    first, installed_digest = _freeze_pass1_result(tmp_path, payload, None)
    assert first.parent == tmp_path
    assert first.read_bytes() == canonical_json_bytes(payload)
    # R4-A: the returned identity is the INSTALLED object's, re-read from disk.
    assert installed_digest == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert installed_digest == sha256_file(first)
    with pytest.raises(RealizationError, match="refusing to replace"):
        _freeze_pass1_result(tmp_path, payload, first)


def test_r3_a_expected_result_never_comes_from_the_published_realization():
    """The consumer must not be able to discover its own expectation. Structural and by signature."""
    import inspect

    from pretrain import stage_i_output_v1 as module

    for name in ("publish_atomic", "load_published_realization"):
        parameter = inspect.signature(getattr(module, name)).parameters["expected"]
        assert parameter.default is inspect.Parameter.empty, (
            f"{name} must require a trusted expected result, not default one into existence"
        )
    with pytest.raises(TypeError):
        module.load_published_realization(Path("/nonexistent"))

    # The audit's expectation must be the trusted projection, never the manifest's own copy.
    for name in ("publish_atomic", "load_published_realization"):
        function = _function_source("pretrain/stage_i_output_v1.py", name)
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "audit_staged_realization"
        ]
        assert len(calls) == 1, f"{name} must run exactly one audit"
        projection = calls[0].args[3]
        assert isinstance(projection, ast.Attribute), (
            f"{name} passes {ast.dump(projection)} as the audit's authority"
        )
        assert projection.attr == "node_binding_projection"
        assert getattr(projection.value, "id", None) == "expected", (
            f"{name} must hand the audit the TRUSTED projection, not the manifest's own"
        )


def test_r3_a_trusted_expected_result_requires_an_externally_supplied_digest(tmp_path: Path):
    """An artifact that certifies itself is not an expectation."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    pass1 = _pass1_for(
        [dict(manifest["nodes"][0])],
        {"b_x": ["ib_x"]},
        {k: v for k, v in manifest["totals"].items() if k != "shards"},
    )
    true_digest = hashlib.sha256(canonical_json_bytes(pass1)).hexdigest()
    assert trusted_expected_result(pass1, expected_sha256=true_digest).result_identity_sha256 == (
        true_digest
    )
    with pytest.raises(OutputError, match="is not the supplied"):
        trusted_expected_result(pass1, expected_sha256="0" * 64)

    path = tmp_path / "pass1.json"
    path.write_bytes(canonical_json_bytes(pass1))
    assert load_trusted_expected_result(path, expected_sha256=true_digest)
    with pytest.raises(OutputError, match="is not the supplied"):
        load_trusted_expected_result(path, expected_sha256="1" * 64)


def test_r3_a_manifest_naming_another_expected_result_is_rejected(tmp_path: Path):
    """A realization may not point at an expectation other than the one it is checked against."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["stage_i_run"] = _stage_i_run_block(
        expected, post_pass1_result_identity_sha256="0" * 64
    )
    with pytest.raises(OutputError, match="trusted expectation supplied out of band"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.__setitem__("surprise", 1), "carries unknown field"),
        (lambda r: r.pop("h_i_gate"), "is missing field"),
        (lambda r: r["h_i_gate"].__setitem__("ALL_NODES_MATCH", False), "never passed the H/I"),
        (
            lambda r: r["authorization"].__setitem__("records_per_shard", 7),
            "records_per_shard disagrees",
        ),
        (
            lambda r: r["authorization"].__setitem__("surprise", 1),
            "pass-1 authorization carries unknown",
        ),
        (
            lambda r: r.__setitem__("selection_sequence_commitment_version", "nope"),
            "different selection-sequence commitment version",
        ),
        (
            lambda r: r["selection_sequence_commitments"].__setitem__("b_x", "9" * 64),
            "does not describe its own map|disagrees with the committed map",
        ),
        (
            lambda r: r.__setitem__("node_binding_projection", {"b_x": ["ib_extra", "ib_x"]}),
            "does not generate its bound projection digest",
        ),
        (
            lambda r: r.__setitem__("authorized_input_binding_ids", ["ib_other"]),
            "outside the plan-authorized global set",
        ),
        (lambda r: r["totals"].__setitem__("records", 99), "disagrees with the per-node"),
        (lambda r: r["nodes"][0].__setitem__("branch", "NOPE"), "branch is invalid"),
        (
            lambda r: r.__setitem__("schema_version", "petitgpt-stage-i-pass1-result-v0"),
            "wrong sch",
        ),
    ],
)
def test_r3_a_pass1_result_schema_is_closed(tmp_path: Path, mutation, match):
    """The Layer-2 artifact is closed at every level, exactly like the plan and the manifest."""
    manifest, _records, _expected = _publishable(tmp_path, count=3)
    pass1 = _pass1_for(
        [dict(manifest["nodes"][0])],
        {"b_x": ["ib_x"]},
        {k: v for k, v in manifest["totals"].items() if k != "shards"},
    )
    validate_pass1_result(json.loads(json.dumps(pass1)))  # the unmutated fixture is valid
    mutated = json.loads(json.dumps(pass1))
    mutation(mutated)
    with pytest.raises(OutputError, match=match):
        validate_pass1_result(mutated)


# ---- R3-B: trusted node -> binding authority ------------------------------------


@pytest.mark.parametrize("rebind_manifest", [True, False])
def test_r3_b_fully_resealed_projection_change_is_rejected(tmp_path: Path, rebind_manifest):
    """R3-B. The exact Codex R2 case: records moved to an undeclared binding, everything resealed.

    Codex changed every record's ``input_binding_id`` to ``ib_not_declared``, changed the
    manifest's ``node_binding_projection`` to match, recomputed the projection digest and the
    published run identity, and kept the original candidate-plan SHA. Publisher and consumer both
    accepted, because the audit's notion of "authorized" was the manifest's own projection.

    Both spellings are exercised: with the manifest restated to match the records, and without.
    The trusted plan/context projection is held fixed throughout.
    """
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    def rebind(lines):
        parsed = [json.loads(line) for line in lines]
        for record in parsed:
            record["input_binding_id"] = "ib_not_declared"
        return b"".join(canonical_json_bytes(r) for r in parsed)

    def restate(obj):
        obj["node_binding_projection"] = {"b_x": ["ib_not_declared"]}
        obj["nodes"][0]["input_binding_ids"] = ["ib_not_declared"]
        obj["bindings"] = {"ib_not_declared": "6" * 64}

    _fully_reseal(final, rebind, restate if rebind_manifest else None)
    with pytest.raises(
        (OutputError, AuditError),
        match="not authorized to draw from input binding|trusted node/binding authority",
    ):
        load_published_realization(final, expected=expected)


def test_r3_b_binding_authorized_globally_but_for_another_node_is_rejected(tmp_path: Path):
    """A real binding attached to a node that was never authorized to draw from it."""
    node = {
        "source_id": "b_x",
        "stage": "stage_b",
        "target_serialized_tokens": 10,
        "branch": "ORDINARY",
        "selection_mode": "SEEDED_HASH",
        "selected_identities": 3,
        "selected_serialized_tokens": 21,
        "selection_fingerprint": "0" * 64,
        "selection_sequence_commitment": "0" * 64,
        "crossing_identity": None,
        "actual_overshoot_tokens": 0,
        "input_binding_ids": ["ib_y"],  # globally real, but not authorized for b_x
    }
    pass1 = _pass1_for(
        [dict(node)],
        {"b_x": ["ib_x"]},
        {
            "records": 3,
            "content_tokens": 15,
            "serialized_tokens": 21,
            "unique_cleaned_identities": 3,
        },
        binding_digests={"ib_x": "6" * 64, "ib_y": "7" * 64},
    )
    with pytest.raises(OutputError, match="outside its authorized projection"):
        validate_pass1_result(pass1)


def test_r3_b_manifest_may_not_declare_a_binding_the_plan_never_authorized(tmp_path: Path):
    """R3-B global reconciliation: manifest binding declarations vs the plan-authorized set."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["bindings"] = {"ib_x": "6" * 64, "ib_smuggled": "7" * 64}
    with pytest.raises(OutputError, match="binding\\(s\\) the plan never authorized"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


def test_r3_b_manifest_omitting_an_in_use_binding_is_rejected(tmp_path: Path):
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["bindings"] = {"ib_unused": "6" * 64}
    with pytest.raises(OutputError, match="the plan never authorized|omits the release digest"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


def test_r3_b_changed_projection_digest_and_identity_do_not_help(tmp_path: Path):
    """Recomputing the projection digest and the run identity cannot manufacture authority."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    widened = {"b_x": ["ib_extra", "ib_x"]}
    manifest["node_binding_projection"] = widened
    manifest["stage_i_run"] = _stage_i_run_block(
        expected, node_binding_projection_sha256=node_binding_projection_sha256(widened)
    )
    with pytest.raises(OutputError, match="trusted post-Pass-1 result binds"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


# ---- R3-D: complete, immutable canonical authorized state -----------------------


def _payload_of(state) -> dict:
    from pretrain.stage_i_realize_v1 import _thaw

    return json.loads(canonical_json_bytes(_thaw(state.canonical_payload)))


def _mutate_path(payload: dict, path: list, value):
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


@pytest.fixture(scope="module")
def authorized_state(authorized):
    return authorized.revalidate()


def test_r3_d_canonical_payload_is_the_state_digest(authorized, authorized_state):
    """The digest must be over the whole projection, computed the same way a reviewer would."""
    payload = _payload_of(authorized_state)
    assert (
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        == authorized_state.state_sha256
        == authorized.authorized_state_sha256
    )


@pytest.mark.parametrize(
    "label,path_of,value",
    [
        (
            "branch_threshold_4_to_5",
            lambda p: [
                "graph",
                "nodes",
                next(i for i, n in enumerate(p["graph"]["nodes"]) if n["branch_primary"]),
                "branch_primary",
                "candidate_predicate",
                "value",
            ],
            5,
        ),
        ("h_census_value", lambda p: ["accepted_h", "census", "nodes", 0, "branch"], "NOPE"),
        (
            "h_census_selected_tokens",
            lambda p: ["accepted_h", "gate_projection", 0, "selected_serialized_tokens"],
            1,
        ),
        (
            "h_prediction",
            lambda p: [
                "accepted_h",
                "predictions",
                "predictions",
                sorted(p["accepted_h"]["predictions"]["predictions"])[0],
                "predicted_branch",
            ],
            "NOPE",
        ),
        (
            "input_document_path",
            lambda p: ["paths", "input_documents", sorted(p["paths"]["input_documents"])[0]],
            "/tmp/not-the-frozen-corpus.jsonl",
        ),
        (
            "input_document_path_in_graph",
            lambda p: [
                "graph",
                "bindings",
                sorted(p["graph"]["bindings"])[0],
                "documents_path",
            ],
            "/tmp/not-the-frozen-corpus.jsonl",
        ),
        ("tokenizer_path", lambda p: ["paths", "tokenizer"], "runs/elsewhere/tokenizer.json"),
        (
            "reference_path",
            lambda p: ["paths", "reference_exclusion"],
            "runs/elsewhere/exclusion.json",
        ),
        (
            "authority_path",
            lambda p: ["paths", "authorities", "hq_policy"],
            "runs/elsewhere/policy.json",
        ),
        ("authority_digest", lambda p: ["authority_sha256", "tokenizer"], "0" * 64),
        ("graph_seed", lambda p: ["graph", "seed"], 1),
        (
            "cleaning_contract",
            lambda p: [
                "graph",
                "bindings",
                sorted(p["graph"]["bindings"])[0],
                "cleaning_contract",
                "min_chars",
            ],
            9999,
        ),
        (
            "schema_accessor",
            lambda p: [
                "graph",
                "bindings",
                sorted(p["graph"]["bindings"])[0],
                "schema_accessor",
                "accessor_id",
            ],
            "not-the-accessor",
        ),
        ("ownership_matrix", lambda p: ["accepted_h", "ownership_matrix"], {}),
        ("node_binding_projection", lambda p: ["node_binding_projection"], {"b_x": ["ib_x"]}),
        ("shard_rule_literal", lambda p: ["contract_literals", "shard_policy_rule"], "anything"),
        ("plan_digest", lambda p: ["plan", "sha256"], "0" * 64),
    ],
)
def test_r3_d_canonical_state_covers_every_codex_mutation(authorized_state, label, path_of, value):
    """R3-D. Every value Codex mutated past R2's revalidation is inside the closed projection.

    R2's projection named fields reactively, so a nested graph branch threshold, an H census
    value, an H prediction, an input-document path, the tokenizer and reference paths and an
    authority path were all invisible to it and every one of those substitutions revalidated.
    Changing any of them must now change the authorized-state digest -- which is what
    ``resolve_authorized_state`` compares before every load-bearing use.
    """
    payload = _payload_of(authorized_state)
    reference = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert reference == authorized_state.state_sha256

    path = path_of(payload)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    assert cursor[path[-1]] != value, f"{label}: the fixture value is already the mutation"
    _mutate_path(payload, path, value)
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != reference, (
        f"{label} is not covered by the canonical authorized-state projection"
    )


@pytest.mark.parametrize(
    "label,mutate",
    [
        ("h_census_node", lambda s: s.accepted.census["nodes"][0].__setitem__("branch", "X")),
        ("h_census_totals", lambda s: s.accepted.census["totals"].__setitem__("x", 1)),
        ("h_predictions", lambda s: s.accepted.predictions.__setitem__("x", 1)),
        ("graph_raw_nested", lambda s: s.graph.raw["nodes"][0].__setitem__("stage", "X")),
        ("graph_bindings", lambda s: s.graph.bindings.__setitem__("z", None)),
        ("graph_bound_authorities", lambda s: s.graph.bound_authorities.__setitem__("z", "0")),
        (
            "branch_predicate",
            lambda s: (
                next(n for n in s.graph.nodes if n.branch_primary)
                .branch_primary["candidate_predicate"]
                .__setitem__("value", 5)
            ),
        ),
        ("plan_authorities", lambda s: s.plan["authorities"]["tokenizer"].__setitem__("path", "x")),
        ("plan_top_level", lambda s: s.plan.__setitem__("seed", 1)),
        ("node_binding_projection", lambda s: s.node_binding_projection.__setitem__("z", ())),
        ("authority_paths", lambda s: s.authority_paths.__setitem__("z", "x")),
        ("bundle_files", lambda s: s.bundle_files.__setitem__("z", "0" * 64)),
        ("canonical_payload", lambda s: s.canonical_payload.__setitem__("z", 1)),
    ],
)
def test_r3_d_authorized_state_is_deeply_immutable(authorized_state, label, mutate):
    """R3-D. Nested lists and dicts must not remain mutation surfaces."""
    with pytest.raises((TypeError, AttributeError)):
        mutate(authorized_state)


def test_r3_d_authorized_state_fields_cannot_be_reassigned(authorized_state):
    import dataclasses

    for field in ("graph", "accepted", "plan", "state_sha256", "tokenizer_path"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(authorized_state, field, None)


def test_r3_d_runtime_rederives_state_and_never_reads_it_off_the_handle():
    """R3-D. Load-bearing truth is re-derived, not read off a long-lived caller-reachable object."""
    function = _function_source("pretrain/stage_i_realize_v1.py", "realize_and_publish")
    assert _call_lines(function, "resolve_authorized_state"), (
        "realize_and_publish must resolve the authorized state through the registry"
    )
    forbidden = {
        "graph",
        "accepted",
        "plan",
        "authorities",
        "tokenizer_path",
        "reference_exclusion_path",
        "node_binding_projection",
        "bundle_files",
        "environment",
        "run_identity",
        "run_name",
    }
    offenders = sorted(
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and getattr(node.value, "id", None) == "context"
        and node.attr in forbidden
    )
    assert not offenders, (
        f"realize_and_publish reads load-bearing state off the handle: {offenders}"
    )

    # The re-derivation is the full authorization path, not a cheaper subset.
    derive = _function_source("pretrain/stage_i_realize_v1.py", "_derive_authorized_state")
    for required in (
        "validate_plan_schema",
        "implementation_files",
        "load_source_graph",
        "verify_binding_inputs",
        "load_accepted_h",
        "verify_environment",
        "_authorized_state_payload",
    ):
        assert _call_lines(derive, required), f"re-derivation skips {required}"


def test_r3_d_a_second_authorization_of_the_same_plan_agrees(live_plan_path):
    """Two independent authorizations of identical bytes must land on the same canonical state."""
    first = authorize_plan(live_plan_path, sha256_file(live_plan_path), ROOT)
    second = authorize_plan(live_plan_path, sha256_file(live_plan_path), ROOT)
    assert first is not second
    assert first.authorized_state_sha256 == second.authorized_state_sha256
    assert first.revalidate().state_sha256 == second.revalidate().state_sha256


# ---- R3-E: fully closed candidate-plan schema -----------------------------------


def test_r3_e_plan_schema_is_versioned_and_exact(live_plan_path):
    """R3-E. v3 closes values as well as keys, with exact types and frozen literals."""
    from pretrain.stage_i_realize_v1 import validate_plan_schema

    assert PLAN_SCHEMA == "petitgpt-i-candidate-plan-v4"
    plan = json.loads(live_plan_path.read_text())
    assert validate_plan_schema(json.loads(json.dumps(plan)))

    # `type(x) is int` rather than isinstance: a bool is not an integer here.
    for key, value in (("seed", True), ("resume_supported", 0)):
        mutated = json.loads(json.dumps(plan))
        mutated[key] = value
        with pytest.raises(RealizationError):
            validate_plan_schema(mutated)

    # The frozen structured rules are literals, not "some JSON value".
    assert plan["shard_policy"]["rule"] == SHARD_POLICY_RULE
    assert plan["accepted_h"]["run_dir"] == ACCEPTED_H_RUN_DIR
    assert plan["accepted_h"]["complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256


@pytest.mark.parametrize(
    "path",
    [
        "./runs/g_production_2026-08-21/release/tokenizer.json",
        "runs/g_production_2026-08-21/release/../release/tokenizer.json",
        "/workspace/petitgpt/runs/g_production_2026-08-21/release/tokenizer.json",
        "runs/g_production_2026-08-21//release/tokenizer.json",
        "runs/g_production_2026-08-21/release/tokenizer.json/",
    ],
)
def test_r3_e_equivalent_but_noncanonical_authority_paths_do_not_authorize(live_plan_path, path):
    """R3-E. The same bytes under a different spelling is not the frozen canonical mapping."""
    from pretrain.stage_i_realize_v1 import validate_plan_schema

    plan = json.loads(live_plan_path.read_text())
    canonical = plan["authorities"]["tokenizer"]["path"]
    assert path != canonical
    plan["authorities"]["tokenizer"]["path"] = path
    with pytest.raises(
        RealizationError,
        match="canonical|repository-relative|forward slashes|path segments",
    ):
        validate_plan_schema(plan)


def test_r3_e_authority_mapping_is_name_to_path_to_digest(live_plan_path):
    """Every required authority is pinned to one canonical location and one digest."""
    from pretrain.stage_i_realize_v1 import _AUTHORITY_PATHS, REQUIRED_AUTHORITIES

    plan = json.loads(live_plan_path.read_text())
    assert frozenset(_AUTHORITY_PATHS) == REQUIRED_AUTHORITIES
    for name, entry in plan["authorities"].items():
        assert entry["path"] == _AUTHORITY_PATHS[name]
        assert sha256_file(ROOT / entry["path"]) == entry["sha256"]


# ---- R3: the published identity binds every layer -------------------------------


def test_r3_published_identity_binds_plan_state_and_expected_result(tmp_path: Path):
    """The closed published identity must bind Layer 1 and Layer 2, and nothing may be missing."""
    manifest, _records, expected = _publishable(tmp_path, count=3)
    block = manifest["stage_i_run"]
    for key in (
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
        "post_pass1_result_identity_schema",
        "post_pass1_result_identity_sha256",
        "selection_sequence_commitment_version",
        "selection_sequence_commitment_map_sha256",
    ):
        assert key in block, f"the published identity does not bind {key}"
    assert recompute_stage_i_run_identity(block) == expected.stage_i_run_identity
    assert block["post_pass1_result_identity_sha256"] == expected.result_identity_sha256


# ================================================================================
# R4 REPAIR REGRESSIONS — one per Codex R3 re-review finding
# ================================================================================


# ---- R4-A: atomic, durable, no-replace Layer-2 freeze ---------------------------


def test_r4_a_concurrent_writers_cannot_both_install_the_layer_2_object(tmp_path: Path):
    """R4-A. The exact Codex race: two writers, one destination, deterministic interleaving.

    R3 asked ``path.exists()`` and then opened the destination truncating. Codex ran two writers
    that both saw an absent file; both wrote, both reported success, and the second silently
    replaced the first's frozen Layer-2 object -- the one thing a freeze exists to prevent.

    The barrier here sits exactly where the check-then-act window used to be, and thread A is
    then allowed to run to completion -- including its read-back -- before B writes at all. That
    is the interleaving that reproduced the defect. With an atomic no-replace install there is no
    window to lose: exactly one payload becomes the installed object, and the loser is refused.
    """
    import threading

    import pretrain.stage_i_realize_v1 as module
    from pretrain.stage_i_realize_v1 import _freeze_pass1_result

    destination = tmp_path / "pass1_result-raced.json"
    original = module._write_durable
    reached = threading.Barrier(2)
    a_finished = threading.Event()
    local = threading.local()

    def barriered(path, payload):
        reached.wait(timeout=30)
        if getattr(local, "label", None) == "B":
            a_finished.wait(timeout=30)
        return original(path, payload)

    outcomes: dict[str, tuple[str, str]] = {}

    def run(label: str, marker: str):
        local.label = label
        payload = {"schema_version": PASS1_RESULT_SCHEMA, "probe": marker}
        try:
            _, digest = _freeze_pass1_result(tmp_path, payload, destination)
            outcomes[label] = ("SUCCESS", digest)
        except BaseException as exc:  # noqa: BLE001 - both classes are meaningful here
            outcomes[label] = ("FAILED", f"{type(exc).__name__}: {exc}")
        finally:
            if label == "A":
                a_finished.set()

    module._write_durable = barriered
    try:
        threads = [
            threading.Thread(target=run, args=("A", "payload-a")),
            threading.Thread(target=run, args=("B", "payload-b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    finally:
        module._write_durable = original
        a_finished.set()

    successes = [label for label, (status, _) in outcomes.items() if status == "SUCCESS"]
    assert len(successes) == 1, f"both writers installed a Layer-2 object: {outcomes}"
    losers = [label for label in outcomes if label not in successes]
    assert losers, "the race produced no loser; the fixture did not race"
    assert "refusing to replace" in outcomes[losers[0]][1], outcomes[losers[0]][1]

    # The installed bytes are exactly the winner's, and the digest it was handed is theirs.
    winner = successes[0]
    marker = "payload-a" if winner == "A" else "payload-b"
    installed = json.loads(destination.read_text())
    assert installed["probe"] == marker
    assert outcomes[winner][1] == sha256_file(destination)
    assert (
        outcomes[winner][1]
        == hashlib.sha256(
            canonical_json_bytes({"schema_version": PASS1_RESULT_SCHEMA, "probe": marker})
        ).hexdigest()
    )

    # No temp artifact survived, in either direction.
    assert sorted(p.name for p in tmp_path.iterdir()) == [destination.name]


def test_r4_a_preexisting_destination_is_never_replaced(tmp_path: Path):
    """A Layer-2 object already at the destination survives, byte for byte."""
    from pretrain.stage_i_realize_v1 import _freeze_pass1_result

    destination = tmp_path / "pass1_result-existing.json"
    destination.write_bytes(b"ORIGINAL-LAYER-2-OBJECT\n")
    before = sha256_file(destination)
    with pytest.raises(RealizationError, match="refusing to replace"):
        _freeze_pass1_result(tmp_path, {"schema_version": PASS1_RESULT_SCHEMA}, destination)
    assert sha256_file(destination) == before
    assert destination.read_bytes() == b"ORIGINAL-LAYER-2-OBJECT\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [destination.name]


def test_r4_a_install_uses_an_atomic_no_replace_primitive(tmp_path: Path):
    """Structural + behavioural: the install is `os.link`, and it really is no-replace here.

    ``os.replace`` is deliberately absent from this path -- it replaces by definition. The
    no-replace property is the kernel's, so it is asserted against the actual filesystem the
    frozen environment runs on rather than assumed.
    """
    import os

    from pretrain.stage_i_realize_v1 import _install_no_replace

    function = _function_source("pretrain/stage_i_realize_v1.py", "_install_no_replace")
    linked = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute) and node.attr in {"link", "replace", "rename"}
    ]
    assert [node.attr for node in linked] == ["link"], (
        "installation must use os.link; os.replace/os.rename replace an existing destination"
    )
    freeze = _function_source("pretrain/stage_i_realize_v1.py", "_freeze_pass1_result")
    assert not [
        node
        for node in ast.walk(freeze)
        if isinstance(node, ast.Attribute) and node.attr in {"replace", "rename"}
    ], "the freeze must not fall back to a replacing primitive"
    # ...and no check-then-act existence test survives in the freeze.
    assert not [
        node
        for node in ast.walk(freeze)
        if isinstance(node, ast.Attribute) and node.attr == "exists"
    ], "an exists() pre-check reintroduces the TOCTOU window"

    source, destination = tmp_path / "a.tmp", tmp_path / "dest"
    source.write_bytes(b"first")
    _install_no_replace(source, destination)
    assert destination.read_bytes() == b"first"
    other = tmp_path / "b.tmp"
    other.write_bytes(b"second")
    with pytest.raises(RealizationError, match="refusing to replace"):
        _install_no_replace(other, destination)
    assert destination.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        os.link(other, destination)


def test_r4_a_freeze_flushes_and_fsyncs_the_file_and_the_directory(tmp_path: Path):
    """Durability: the payload is fsynced before install, and the directory after it."""
    import os

    import pretrain.stage_i_realize_v1 as module
    from pretrain.stage_i_realize_v1 import _freeze_pass1_result

    synced: list[str] = []
    original = os.fsync

    def record(fd):
        try:
            synced.append("dir" if os.path.isdir(f"/proc/self/fd/{fd}") else "file")
        except OSError:  # pragma: no cover - defensive
            synced.append("unknown")
        return original(fd)

    monkey = module.os
    monkey.fsync = record
    try:
        path, _digest = _freeze_pass1_result(
            tmp_path, {"schema_version": PASS1_RESULT_SCHEMA}, None
        )
    finally:
        monkey.fsync = original
    assert "file" in synced, "the Layer-2 payload was not fsynced before installation"
    assert "dir" in synced, "the containing directory was not fsynced after installation"
    assert synced.index("file") < synced.index("dir"), (
        "the directory must be fsynced after the file, not before"
    )
    assert path.is_file()


# ---- R4-B: the frozen executable is not negotiable ------------------------------


def test_r4_b_no_public_executable_bypass_exists():
    """R4-B. Codex passed ``require_executable=False`` and authorized under a foreign interpreter."""
    import inspect

    import pretrain.stage_i_realize_v1 as module

    for name in ("authorize_plan", "verify_environment", "_derive_authorized_state"):
        parameters = inspect.signature(getattr(module, name)).parameters
        assert "require_executable" not in parameters, (
            f"{name} still exposes a caller-controlled executable relaxation: {list(parameters)}"
        )
    # Nothing anywhere in the module may still thread that keyword through.
    source = (ROOT / "pretrain/stage_i_realize_v1.py").read_text()
    tree = ast.parse(source)
    keywords = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "require_executable"
    ]
    assert not keywords, "a call still passes require_executable"
    # The CLI cannot ask for it either.
    assert "--require-executable" not in source


def test_r4_b_wrong_executable_fails_through_the_default_path(live_plan_path, monkeypatch):
    """The default path -- the only path -- refuses a foreign interpreter."""
    import pretrain.stage_i_realize_v1 as module

    monkeypatch.setattr(
        module,
        "current_environment",
        lambda: Environment(
            python_executable="/definitely/not/the/frozen/python",
            python_version=REQUIRED_PYTHON_VERSION,
            tokenizers_version=REQUIRED_TOKENIZERS_VERSION,
        ),
    )
    with pytest.raises(RealizationError, match="python executable .* is not the frozen"):
        authorize_plan(live_plan_path, sha256_file(live_plan_path))
    with pytest.raises(RealizationError, match="python executable .* is not the frozen"):
        module.verify_environment(module.current_environment())


def test_r4_b_runtime_rederivation_rechecks_the_executable(live_plan_path, monkeypatch):
    """A context authorized under the frozen interpreter must not revalidate under another."""
    import pretrain.stage_i_realize_v1 as module

    context = authorize_plan(live_plan_path, sha256_file(live_plan_path))
    context.revalidate()
    monkeypatch.setattr(
        module,
        "current_environment",
        lambda: Environment(
            python_executable="/definitely/not/the/frozen/python",
            python_version=REQUIRED_PYTHON_VERSION,
            tokenizers_version=REQUIRED_TOKENIZERS_VERSION,
        ),
    )
    with pytest.raises(RealizationError, match="python executable .* is not the frozen"):
        context.revalidate()
    with pytest.raises(RealizationError, match="python executable .* is not the frozen"):
        resolve_authorized_state(context, "test")


def test_r4_b_relaxing_the_environment_cannot_mint_registry_authority(live_plan_path, monkeypatch):
    """Even with the check monkeypatched away in-process, no relaxed instance is an authority.

    This is the strongest form of the requirement: a test that neutralises ``verify_environment``
    entirely still cannot obtain a context that the registry recognises through any public route
    other than a genuine ``authorize_plan``, and ``verify_environment`` itself mints nothing.
    """
    import pretrain.stage_i_realize_v1 as module

    monkeypatch.setattr(module, "verify_environment", lambda environment: None)
    monkeypatch.setattr(
        module,
        "current_environment",
        lambda: Environment(
            python_executable="/definitely/not/the/frozen/python",
            python_version=REQUIRED_PYTHON_VERSION,
            tokenizers_version=REQUIRED_TOKENIZERS_VERSION,
        ),
    )
    # A relaxed derivation produces a plain state object, never a registry entry.
    state = module._derive_authorized_state(live_plan_path, sha256_file(live_plan_path))
    assert isinstance(state, module.CanonicalAuthorizedState)
    assert not isinstance(state, AuthorizedIContext)
    with pytest.raises(RealizationError, match="requires an authorized Stage-I context"):
        module._require_authorized(state, "test")
    # And the relaxed environment is recorded in the state, so it cannot masquerade as frozen.
    assert state.environment.python_executable == "/definitely/not/the/frozen/python"


# ---- R4-C: the authority base is the executing installation ---------------------


def test_r4_c_canonical_root_is_derived_from_the_executing_module():

    assert CANONICAL_REPO_ROOT == ROOT.resolve()
    assert CANONICAL_REPO_ROOT == Path(ROOT / "pretrain/stage_i_realize_v1.py").resolve().parents[1]


def test_r4_c_a_caller_cannot_relocate_the_authority_base(live_plan_path, tmp_path):
    """R4-C. Codex authorized against an alternate root of hard links to identical resources."""
    digest = sha256_file(live_plan_path)
    # Omitting the root, or naming the real one, both work.
    assert authorize_plan(live_plan_path, digest) is not None
    assert authorize_plan(live_plan_path, digest, ROOT) is not None
    for other in (tmp_path, Path("/tmp"), ROOT / "pretrain"):
        with pytest.raises(RealizationError, match="not the executing Stage-I installation root"):
            authorize_plan(live_plan_path, digest, other)


def test_r4_c_alternate_hardlinked_root_cannot_authorize(accepted_h):
    """The full Codex reproduction: byte-identical hard links under a different root.

    Every content digest matches, because the files are literally the same inodes. What differs is
    where they live -- and that is now load-bearing, so the alternate root is refused before any
    comparable authorized state exists.
    """
    import os
    import shutil
    import tempfile

    # Hard links require the same filesystem as the repository, so the alternate root is staged
    # beside it rather than under the system temp directory.
    staging = Path(tempfile.mkdtemp(dir=str(ROOT), prefix=".pytest-alt-root-"))
    try:
        alt_root = staging / "alt"
        for relative in [
            *IMPLEMENTATION_BUNDLE_FILES,
            "runs/h_tooling_repair_v2_2026-08-21/policy/stage_i_source_graph_v1.json",
        ]:
            source = ROOT / relative
            target = alt_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        # The links really are the same bytes, so nothing content-addressed distinguishes them.
        for relative in IMPLEMENTATION_BUNDLE_FILES:
            assert sha256_file(alt_root / relative) == sha256_file(ROOT / relative)
            assert (alt_root / relative).stat().st_ino == (ROOT / relative).stat().st_ino

        plan = _plan(accepted_h, commit=_head_commit())
        path = staging / "alt_plan.json"
        path.write_bytes(canonical_json_bytes(plan))
        with pytest.raises(RealizationError, match="not the executing Stage-I installation root"):
            authorize_plan(path, sha256_file(path), alt_root)
        # ...and the same plan bytes under the real root are refused on their own terms, so the
        # rejection above is about WHERE it was authorized from, not about the plan.
        with pytest.raises(RealizationError):
            authorize_plan(path, "0" * 64)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@pytest.mark.parametrize(
    "path_of,value",
    [
        (lambda p: ["resolved_paths", "repository_root"], "/somewhere/else"),
        (lambda p: ["resolved_paths", "owner_graph"], "/somewhere/else/graph.json"),
        (lambda p: ["resolved_paths", "candidate_plan"], "/somewhere/else/plan.json"),
        (lambda p: ["resolved_paths", "tokenizer"], "/somewhere/else/tokenizer.json"),
        (lambda p: ["resolved_paths", "reference_exclusion"], "/somewhere/else/excl.json"),
        (lambda p: ["resolved_paths", "accepted_h_run_dir"], "/somewhere/else/h"),
        (lambda p: ["resolved_paths", "accepted_h_census"], "/somewhere/else/census.json"),
        (lambda p: ["resolved_paths", "accepted_h_predictions"], "/somewhere/else/pred.json"),
        (lambda p: ["resolved_paths", "accepted_h_complete"], "/somewhere/else/COMPLETE"),
        (
            lambda p: ["resolved_paths", "accepted_h_evidence_manifest"],
            "/somewhere/else/SHA256SUMS",
        ),
        (lambda p: ["resolved_paths", "authorities", "hq_policy"], "/somewhere/else/policy.json"),
        (
            lambda p: ["resolved_paths", "implementation_files", "pretrain/stage_i_select_v1.py"],
            "/somewhere/else/select.py",
        ),
        (
            lambda p: [
                "resolved_paths",
                "input_documents",
                sorted(p["resolved_paths"]["input_documents"])[0],
            ],
            "/somewhere/else/documents.jsonl",
        ),
        (
            lambda p: [
                "resolved_paths",
                "input_release_manifests",
                sorted(p["resolved_paths"]["input_release_manifests"])[0],
            ],
            "/somewhere/else/manifest.json",
        ),
        (
            lambda p: [
                "resolved_paths",
                "input_eligibility_indexes",
                sorted(p["resolved_paths"]["input_eligibility_indexes"])[0],
            ],
            "/somewhere/else/index.bin",
        ),
    ],
)
def test_r4_c_every_resolved_path_is_bound_into_the_authorized_state(
    authorized_state, path_of, value
):
    """R4-C. Path identity is as load-bearing as content identity, resource by resource."""
    payload = _payload_of(authorized_state)
    reference = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert reference == authorized_state.state_sha256

    path = path_of(payload)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    assert cursor[path[-1]] != value
    assert str(cursor[path[-1]]).startswith("/"), "resolved paths must be absolute"
    cursor[path[-1]] = value
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != reference, (
        f"{path} is not bound into the canonical authorized state"
    )


def test_r4_c_resolved_paths_agree_with_the_state_and_are_canonical(authorized_state):
    resolved = authorized_state.resolved_paths
    assert resolved["repository_root"] == str(ROOT.resolve())
    assert resolved["tokenizer"] == str(authorized_state.tokenizer_path)
    assert resolved["reference_exclusion"] == str(authorized_state.reference_exclusion_path)
    assert resolved["owner_graph"] == str(authorized_state.graph_path)
    assert resolved["accepted_h_run_dir"] == str(authorized_state.h_run_dir)
    assert resolved["accepted_h_census"] == str(authorized_state.accepted.census_path)
    assert resolved["accepted_h_predictions"] == str(authorized_state.accepted.predictions_path)
    assert resolved["accepted_h_complete"] == str(authorized_state.accepted.complete_path)
    # One canonical rule, applied everywhere: fully resolved, absolute, no relative segments.

    def walk(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for child in value.values():
                yield from walk(child)

    for spelling in walk(resolved):
        assert spelling.startswith("/"), spelling
        assert str(Path(spelling).resolve()) == spelling, spelling
    assert set(authorized_state.binding_document_digests) == set(authorized_state.graph.bindings)


# ---- R4-D: Layer-3 environment and binding digests must be trusted --------------


def test_r4_d_layer2_carries_the_trusted_environment_and_binding_digests(authorized_state):
    """The expectations come from canonical Layer-1 state, never from the manifest."""
    gate = dict.fromkeys(_FIXTURE_GATE, True)
    selections = _selections_for(authorized_state)
    pass1 = build_pass1_result(selections, authorized_state, gate)
    assert pass1["environment"] == dict(authorized_state.environment.as_canonical())
    assert pass1["binding_document_digests"] == {
        binding_id: authorized_state.graph.bindings[binding_id].documents_sha256
        for binding_id in authorized_state.graph.bindings
    }
    assert pass1["authorization"]["environment_sha256"] == environment_sha256(pass1["environment"])
    assert pass1["authorization"]["binding_document_digests_sha256"] == (
        binding_document_digests_sha256(pass1["binding_document_digests"])
    )
    expected = _expected_for(pass1)
    assert dict(expected.environment) == dict(FROZEN_ENVIRONMENT)
    assert expected.environment_sha256 == pass1["authorization"]["environment_sha256"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda m: m.__setitem__(
                "environment",
                {
                    "python_executable": "/usr/bin/python3",
                    "python_version": REQUIRED_PYTHON_VERSION,
                    "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
                },
            ),
            "python_executable",
        ),
        (
            lambda m: m["environment"].__setitem__("python_version", "3.13.9"),
            "python_version",
        ),
        (
            lambda m: m["environment"].__setitem__("tokenizers_version", "0.99.0"),
            "tokenizers_version",
        ),
        (
            lambda m: m["environment"].__setitem__("unknown_environment_field", "accepted"),
            "carries unknown field",
        ),
        (lambda m: m["environment"].pop("python_executable"), "is missing field"),
        (lambda m: m.__setitem__("environment", {}), "is missing field"),
    ],
)
def test_r4_d_false_layer3_environment_is_rejected(tmp_path: Path, mutation, match):
    """R4-D. Codex changed the manifest's environment and resealed everything else."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    mutation(manifest)
    with pytest.raises(OutputError, match=match):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    assert not (tmp_path / "out" / "run-x").exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.__setitem__("bindings", {"ib_x": "0" * 64}), "trusted input-binding"),
        (
            lambda m: m.__setitem__("bindings", {"ib_x": "6" * 64, "ib_extra": "7" * 64}),
            "the plan never authorized|trusted input-binding",
        ),
        (lambda m: m.__setitem__("bindings", {"ib_other": "6" * 64}), "never authorized"),
    ],
)
def test_r4_d_false_layer3_binding_digests_are_rejected(tmp_path: Path, mutation, match):
    """R4-D. Codex replaced a binding release digest with zeros and resealed the rest."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    mutation(manifest)
    with pytest.raises(OutputError, match=match):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)


@pytest.mark.parametrize("what", ["environment", "bindings"])
def test_r4_d_consumer_rejects_a_fully_resealed_provenance_change(tmp_path: Path, what):
    """The strict consumer must refuse the same thing, on a completely resealed publication.

    Every unrelated Layer-3 digest -- shard, manifest, COMPLETE, run identity -- is regenerated,
    so the fixture reaches the trusted-provenance comparison rather than a generic check. The
    Layer-1/Layer-2 expectation supplied out of band is held fixed throughout.
    """
    manifest, records, expected = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    load_published_realization(final, expected=expected)  # valid before tampering

    def falsify(obj):
        if what == "environment":
            obj["environment"] = {
                "python_executable": "/usr/bin/python3",
                "python_version": "3.13.9",
                "tokenizers_version": "0.99.0",
            }
        else:
            obj["bindings"] = {"ib_x": "0" * 64}

    _fully_reseal(final, lambda lines: b"".join(line + b"\n" for line in lines), falsify)
    # R5-C moved full manifest validation after reconciliation, so the trusted Layer-2 comparison
    # is what fires now rather than the static frozen-value check. Both are correct refusals; the
    # trusted one is the stronger statement.
    with pytest.raises(
        OutputError,
        match=("trusted post-Pass-1 result binds|trusted input-binding|frozen Stage-I environment"),
    ):
        load_published_realization(final, expected=expected)


def test_r4_d_published_identity_binds_environment_and_binding_digests(tmp_path: Path):
    """Both new anchors participate in the published run identity."""
    manifest, _records, expected = _publishable(tmp_path, count=3)
    block = manifest["stage_i_run"]
    assert block["environment_sha256"] == expected.environment_sha256
    assert block["binding_document_digests_sha256"] == expected.binding_document_digests_sha256
    assert recompute_stage_i_run_identity(block) == expected.stage_i_run_identity
    for key in ("environment_sha256", "binding_document_digests_sha256"):
        altered = {**dict(expected.authorization), key: "0" * 64}
        assert (
            stage_i_published_run_identity(
                altered,
                post_pass1_result_identity_sha256=expected.result_identity_sha256,
                selection_sequence_commitment_map_sha256=(
                    expected.selection_sequence_commitment_map_sha256
                ),
            )
            != expected.stage_i_run_identity
        )


def test_r4_d_reconciliation_never_takes_the_manifest_as_its_own_expectation():
    """Structural: the environment and binding comparisons read `expected`, not the manifest."""
    for name in ("_reconcile_environment", "_reconcile_binding_digests"):
        function = _function_source("pretrain/stage_i_output_v1.py", name)
        attributes = {
            node.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "expected"
        }
        assert attributes, f"{name} does not consult the trusted expectation at all"


# ---- R4-E: physical facts are reconciled first ----------------------------------


def test_r4_e_physical_mismatch_is_reported_before_identity(tmp_path: Path):
    """R4-E. A false total plus an invalid identity must fail on the false total.

    Codex found the combined fixture failing on the run identity, which hides the concrete,
    diagnosable defect behind a downstream one. Both faults are present here; the physical one is
    the one that must surface.
    """
    manifest, records, expected = _publishable(tmp_path, count=3)
    manifest["totals"]["serialized_tokens"] = 22
    manifest["nodes"][0]["selected_serialized_tokens"] = 22
    manifest["stage_i_run"] = {**manifest["stage_i_run"], "run_identity": "0" * 64}
    with pytest.raises(OutputError, match="serialized_tokens is 22 but the staged records"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    assert not (tmp_path / "out" / "run-x").exists()

    # The isolated identity fault is still rejected on its own terms.
    clean, records2, expected2 = _publishable(tmp_path, count=3)
    clean["stage_i_run"] = {**clean["stage_i_run"], "run_identity": "0" * 64}
    with pytest.raises(OutputError, match="run identity"):
        publish_atomic(tmp_path / "out2", "run-x", clean, records2, expected=expected2)


def test_r4_e_reconciliation_runs_physical_checks_before_layer_2_checks():
    """Structural: the order is part of the contract, not an accident of call sites."""
    function = _function_source("pretrain/stage_i_output_v1.py", "reconcile_manifest_with_audit")
    order = [
        line
        for line in _call_lines_by_name(function)
        if line[1]
        in {
            "_reconcile_physical",
            "_reconcile_run_identity",
            "_reconcile_projection",
            "_reconcile_environment",
            "_reconcile_binding_digests",
            "_reconcile_expected_result",
        }
    ]
    names = [name for _line, name in sorted(order)]
    assert names[0] == "_reconcile_physical", f"physical reconciliation must run first, got {names}"
    assert set(names[1:]) == {
        "_reconcile_run_identity",
        "_reconcile_projection",
        "_reconcile_environment",
        "_reconcile_binding_digests",
        "_reconcile_expected_result",
    }


# ---- R4-F: the manifest's shape is closed before anything indexes it -------------


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.pop("stage_i_run"), r"missing field\(s\): \['stage_i_run'\]"),
        (lambda m: m.__setitem__("stage_i_run", ["not", "an", "object"]), "must be a JSON object"),
        (lambda m: m.__setitem__("stage_i_run", None), "must be a JSON object"),
        (lambda m: m.__setitem__("stage_i_run", "text"), "must be a JSON object"),
        (lambda m: m.pop("environment"), r"missing field\(s\): \['environment'\]"),
        (lambda m: m.__setitem__("environment", []), "must be a JSON object"),
        (lambda m: m.pop("bindings"), r"missing field\(s\): \['bindings'\]"),
        (lambda m: m.__setitem__("totals", None), "must be a JSON object"),
        (lambda m: m.__setitem__("nodes", {}), "must be a JSON array"),
        (lambda m: m.__setitem__("shards", None), "must be a JSON array"),
        (lambda m: m.__setitem__("surprise", 1), "unknown field"),
    ],
)
def test_r4_f_malformed_manifest_shape_fails_closed(tmp_path: Path, mutation, match):
    """R4-F. An absent or mistyped block is a controlled refusal, never a raw KeyError."""
    manifest, _records, expected = _publishable(tmp_path, count=3)
    mutation(manifest)
    audit = _StubAudit()
    with pytest.raises(OutputError, match=match):
        reconcile_manifest_with_audit(manifest, audit, expected)
    with pytest.raises(OutputError, match=match):
        validate_manifest(manifest)


def test_r4_f_no_raw_keyerror_escapes_reconciliation(tmp_path: Path):
    """Every top-level manifest field can go missing without an uncontrolled exception."""
    manifest, _records, expected = _publishable(tmp_path, count=3)
    audit = _StubAudit()
    for field in sorted(manifest):
        broken = json.loads(json.dumps(manifest))
        broken.pop(field)
        with pytest.raises(OutputError):
            reconcile_manifest_with_audit(broken, audit, expected)


class _StubAudit:
    """Just enough audit surface for shape checks to be the thing that fires."""

    schema_version = "stub"
    records = 3
    content_tokens = 15
    serialized_tokens = 21
    shards = 1
    unique_cleaned_identities = 3
    per_shard = ()
    nodes = ()


def _head_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _call_lines_by_name(function: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name:
                found.append((node.lineno, name))
    return found


def _selections_for(state):
    """One selected document per graph node, enough to build a real Pass-1 result."""
    selections = []
    for index, node_spec in enumerate(state.graph.nodes):
        identity = hashlib.sha256(f"r4-{node_spec.source_id}".encode()).hexdigest()
        document = SelectedDocument(
            cleaned_sha256=identity,
            selection_ordinal_within_node=0,
            input_binding_id=node_spec.input_binding_ids[0],
            stable_input_record_ordinal=index,
            raw_sha256=hashlib.sha256(f"raw-{node_spec.source_id}".encode()).hexdigest(),
            input_record_sha256=hashlib.sha256(f"row-{node_spec.source_id}".encode()).hexdigest(),
            canonical_fingerprint=hashlib.sha256(f"fp-{node_spec.source_id}".encode()).hexdigest(),
            content_token_count=5,
            serialized_token_count=7,
        )
        selections.append(
            NodeSelection(
                source_id=node_spec.source_id,
                stage=node_spec.stage,
                target_serialized_tokens=node_spec.target_serialized_tokens,
                branch="ORDINARY",
                selection_mode="SEEDED_HASH",
                pre_exclusion_unique_identities=1,
                g2_excluded_identities=0,
                prior_commit_excluded_identities=0,
                exclusions_by_owner={},
                post_exclusion_candidate_identities=1,
                post_exclusion_candidate_serialized_tokens=7,
                selected_identities=1,
                selected_serialized_tokens=7,
                crossing_identity=None,
                crossing_document_serialized_tokens=None,
                actual_overshoot_tokens=0,
                residual_identities=0,
                residual_serialized_tokens=0,
                selection_fingerprint=oracle_fingerprint([identity]),
                feasible=True,
                boundary_evidence={},
                selected=(document,),
            )
        )
    return selections


# ================================================================================
# R5 REPAIR REGRESSIONS — one per Codex R4 re-review finding
# ================================================================================


@contextlib.contextmanager
def _in_directory(path: Path):
    """Run a block with the process CWD somewhere else, and always put it back."""
    here = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(here)


def _release_manifest_relatives() -> list[str]:
    """The eight relative release_manifest_path values the frozen owner graph declares."""
    graph = json.loads(GRAPH.read_text())
    values = sorted(b["release_manifest_path"] for b in graph["input_bindings"].values())
    assert len(values) == 8
    assert all(not Path(v).is_absolute() for v in values), "fixture assumes relative values"
    return values


def _shadow_tree(root: Path) -> Path:
    """A directory whose relative layout shadows every graph release_manifest_path.

    The files are hard links to the genuine ones, so they are byte-identical and every content
    digest matches. Only their location differs -- which is exactly the thing that must decide.
    """
    for relative in _release_manifest_relatives():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(ROOT / relative, target)
    return root


@pytest.fixture
def shadow_cwd():
    """A shadow tree on the repository's own filesystem, so hard links are possible."""
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(dir=str(ROOT), prefix=".pytest-shadow-cwd-"))
    try:
        yield _shadow_tree(staging / "alt")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---- R5-A: the process CWD is not a resource-location base ----------------------


def test_r5_a_relative_graph_paths_resolve_against_the_repository_not_the_cwd(shadow_cwd):
    """R5-A. All eight release manifests are relative; they must locate under the repository.

    Codex authorized from a directory whose relative layout shadowed the graph's and observed
    every one of the eight release manifests read from there instead. The CWD was an unstated,
    unbound resource-location base.
    """
    from pretrain.stage_i_graph_v2 import CANONICAL_REPO_ROOT, canonical_resource_path

    assert CANONICAL_REPO_ROOT == ROOT.resolve()
    with _in_directory(shadow_cwd):
        graph = load_source_graph(GRAPH, verify_hashes=True)
        for binding_id, binding in sorted(graph.bindings.items()):
            resolved = binding.release_manifest_path
            assert resolved.is_absolute(), binding_id
            assert str(resolved).startswith(str(ROOT)), (
                f"{binding_id} release manifest resolved outside the repository: {resolved}"
            )
            assert not str(resolved).startswith(str(shadow_cwd)), (
                f"{binding_id} release manifest was read from the process CWD: {resolved}"
            )
    # The helper itself is the single rule, and it ignores the CWD entirely.
    with _in_directory(shadow_cwd):
        assert canonical_resource_path("runs/x/y.json") == ROOT / "runs/x/y.json"
        assert canonical_resource_path("/tmp/absolute.json") == Path("/tmp/absolute.json")


def test_r5_a_no_graph_resource_path_is_built_from_a_raw_relative_value():
    """Structural: every graph resource path goes through the one canonical helper.

    A single surviving `Path(relative_value)` would restore CWD-dependent behaviour for whichever
    resource it built, so this is checked over the parsed source rather than by sampling.
    """
    tree = ast.parse((ROOT / "pretrain/stage_i_graph_v2.py").read_text())
    binding = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_binding"
    )
    for field in ("documents_path", "release_manifest_path", "eligibility_index_path"):
        keyword = next(
            kw
            for node in ast.walk(binding)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == field
        )
        call = keyword.value
        assert isinstance(call, ast.Call), f"{field} is not built by a call"
        name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        assert name == "canonical_resource_path", (
            f"{field} is built with {name}(), not the canonical resolver"
        )


def test_r5_a_authorization_and_revalidation_are_cwd_invariant(live_plan_path, shadow_cwd):
    """R5-A. Same canonical state from the repository, from a shadow tree, from anywhere.

    This is the invariant, and it is stronger than detecting a later CWD change: changing
    ``os.getcwd()`` must not change any resolved Stage-I resource path, so revalidation from a
    different directory simply succeeds.
    """
    digest = sha256_file(live_plan_path)
    with _in_directory(ROOT):
        context = authorize_plan(live_plan_path, digest)
        from_repo = context.revalidate()
    with _in_directory(shadow_cwd):
        # The very same context, revalidated from the shadow tree.
        from_alt = context.revalidate()
        fresh = authorize_plan(live_plan_path, digest).revalidate()

    assert from_repo.state_sha256 == from_alt.state_sha256 == fresh.state_sha256
    for state in (from_alt, fresh):
        assert dict(state.resolved_paths["input_release_manifests"]) == dict(
            from_repo.resolved_paths["input_release_manifests"]
        )
    for value in from_alt.resolved_paths["input_release_manifests"].values():
        assert str(value).startswith(str(ROOT))
        assert not str(value).startswith(str(shadow_cwd))


def test_r5_a_every_resolved_stage_i_path_is_cwd_invariant(live_plan_path, shadow_cwd):
    """No resolved Stage-I resource path may move when the process CWD moves."""
    digest = sha256_file(live_plan_path)
    with _in_directory(ROOT):
        baseline = _payload_of(authorize_plan(live_plan_path, digest).revalidate())
    with _in_directory(shadow_cwd):
        shifted = _payload_of(authorize_plan(live_plan_path, digest).revalidate())
    assert baseline["resolved_paths"] == shifted["resolved_paths"]
    assert baseline["paths"] == shifted["paths"]
    assert baseline["graph"]["bindings"] == shifted["graph"]["bindings"]


def test_r5_a_candidate_plan_generation_is_cwd_independent(accepted_h, shadow_cwd):
    """The plan's bytes must not depend on where the generator was invoked from."""
    with _in_directory(ROOT):
        first = canonical_json_bytes(_plan(accepted_h))
    with _in_directory(shadow_cwd):
        second = canonical_json_bytes(_plan(accepted_h))
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


# ---- R5-B: the accepted-H evidence manifest is a bound input --------------------


@pytest.fixture
def h_run_copy():
    """A hard-linked copy of the accepted H run whose SHA256SUMS can be rewritten."""
    import shutil
    import tempfile

    staging = Path(tempfile.mkdtemp(dir=str(ROOT), prefix=".pytest-h-run-"))

    def build(mutate=None) -> Path:
        target = Path(tempfile.mkdtemp(dir=str(staging))) / "h"
        for child in sorted(H_RUN_DIR.rglob("*")):
            if not child.is_file() or child.name == "SHA256SUMS":
                continue
            dest = target / child.relative_to(H_RUN_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(child, dest)
        lines = (H_RUN_DIR / "SHA256SUMS").read_text().splitlines()
        if mutate is not None:
            lines = mutate(list(lines))
        (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        return target

    try:
        yield build
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def test_r5_b_evidence_manifest_identity_is_bound(accepted_h):
    """R5-B. The manifest's bytes, its exact entry set and its closure are all bound."""
    from pretrain.stage_i_realize_v1 import (
        H_EVIDENCE_MANIFEST_SCHEMA,
        h_evidence_manifest_identity,
        verify_h_evidence_manifest,
    )

    evidence = verify_h_evidence_manifest(H_RUN_DIR)
    assert evidence.file_sha256 == sha256_file(H_RUN_DIR / "SHA256SUMS")
    assert len(evidence.entries) == 16
    assert evidence.entries == tuple(sorted(evidence.entries))
    # The identity is generated by its own bound fields, via a separately written expectation.
    payload = {
        "schema_version": H_EVIDENCE_MANIFEST_SCHEMA,
        "manifest_file_sha256": evidence.file_sha256,
        "entry_count": len(evidence.entries),
        "entries": [[name, digest] for name, digest in sorted(evidence.entries)],
    }
    assert evidence.identity == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert evidence.identity == h_evidence_manifest_identity(
        file_sha256=evidence.file_sha256, entries=evidence.entries
    )

    # The accepted-H record keeps it, and it agrees with the independently pinned artifacts.
    assert accepted_h.evidence_identity == evidence.identity
    assert accepted_h.evidence_manifest_sha256 == evidence.file_sha256
    assert accepted_h.evidence_entries == evidence.entries
    assert evidence.digest_of("evidence/H_PREDICTIONS.json") == accepted_h.predictions_sha256
    published = accepted_h.published_dir.relative_to(accepted_h.run_dir)
    assert evidence.digest_of(str(published / "COMPLETE")) == ACCEPTED_H_COMPLETE_SHA256
    assert evidence.digest_of(str(published / "census.json")) == accepted_h.census_sha256


def test_r5_b_entry_set_enumerates_the_run(accepted_h):
    """The closure rule: the manifest lists exactly the run's files, minus itself."""
    present = {
        str(path.relative_to(H_RUN_DIR))
        for path in H_RUN_DIR.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert {name for name, _digest in accepted_h.evidence_entries} == present


@pytest.mark.parametrize(
    "label,mutate,match",
    [
        (
            "reduced_16_to_1",
            lambda lines: lines[:1],
            "does not enumerate the accepted H run",
        ),
        (
            "entry_removed",
            lambda lines: [ln for ln in lines if not ln.endswith("  NOTES.md")],
            r"unlisted file\(s\): \['NOTES.md'\]",
        ),
        (
            "entry_added",
            lambda lines: [*lines, f"{'0' * 64}  evidence/NOT_A_REAL_FILE.txt"],
            "lists file",
        ),
        (
            "entry_digest_changed",
            lambda lines: [
                (f"{'1' * 64}  NOTES.md" if ln.endswith("  NOTES.md") else ln) for ln in lines
            ],
            "NOTES.md changed",
        ),
        (
            "entry_path_renamed",
            lambda lines: [
                (ln.replace("  NOTES.md", "  RENAMED.md") if ln.endswith("  NOTES.md") else ln)
                for ln in lines
            ],
            r"unlisted file\(s\): \['NOTES.md'\]|lists file",
        ),
        (
            "entry_duplicated",
            lambda lines: [*lines, lines[0]],
            "duplicate evidence entry",
        ),
        (
            "manifest_emptied",
            lambda lines: [],
            "evidence manifest is empty",
        ),
    ],
)
def test_r5_b_mutated_evidence_manifest_is_rejected(h_run_copy, label, mutate, match):
    """R5-B. Every shape of evidence-manifest tampering Codex named, plus the near neighbours."""
    directory = h_run_copy(mutate)
    with pytest.raises(RealizationError, match=match):
        load_accepted_h(directory)


def test_r5_b_evidence_manifest_bytes_alone_move_the_identity(h_run_copy, accepted_h):
    """SHA256SUMS bytes changed while every referenced artifact is untouched.

    Reordering the lines leaves the entry set identical and every artifact byte-for-byte as it
    was, but the file itself is different -- and the file is a bound input, so the identity moves.
    """
    from pretrain.stage_i_realize_v1 import verify_h_evidence_manifest

    directory = h_run_copy(lambda lines: list(reversed(lines)))
    reordered = verify_h_evidence_manifest(directory)
    assert reordered.file_sha256 != accepted_h.evidence_manifest_sha256
    assert reordered.entries == accepted_h.evidence_entries  # same set, same digests
    assert reordered.identity != accepted_h.evidence_identity


def test_r5_b_candidate_plan_binds_the_evidence_manifest(live_plan_path, accepted_h):
    """The plan the owner authorizes carries the evidence scope, so authorization covers it."""
    from pretrain.stage_i_realize_v1 import H_EVIDENCE_MANIFEST_SCHEMA

    plan = json.loads(live_plan_path.read_text())
    block = plan["accepted_h"]
    assert block["evidence_manifest_schema"] == H_EVIDENCE_MANIFEST_SCHEMA
    assert block["evidence_manifest_sha256"] == accepted_h.evidence_manifest_sha256
    assert block["evidence_manifest_identity"] == accepted_h.evidence_identity
    assert block["evidence_entry_count"] == len(accepted_h.evidence_entries)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda p: p["accepted_h"].__setitem__("evidence_manifest_sha256", "0" * 64),
            "evidence manifest is .* on disk but the Stage-I plan binds",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("evidence_manifest_identity", "0" * 64),
            "evidence-manifest identity is .* but the Stage-I plan binds",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("evidence_entry_count", 1),
            "evidence manifest lists 16 entries but the Stage-I plan binds 1",
        ),
        (
            lambda p: p["accepted_h"].__setitem__("evidence_manifest_schema", "nope"),
            "different accepted-H evidence-manifest schema",
        ),
        (lambda p: p["accepted_h"].pop("evidence_manifest_identity"), "accepted_h is missing"),
    ],
)
def test_r5_b_plan_evidence_binding_is_checked_at_authorization(
    live_plan_path, tmp_path, mutation, match
):
    """A plan whose evidence binding disagrees with the run on disk cannot authorize."""
    path, digest = _mutated_plan(live_plan_path, tmp_path, mutation)
    with pytest.raises(RealizationError, match=match):
        authorize_plan(path, digest)


def test_r5_b_evidence_identity_is_bound_into_the_canonical_state(authorized_state):
    """The canonical state carries the projection, and every part of it is load-bearing."""
    payload = _payload_of(authorized_state)
    evidence = payload["accepted_h"]["evidence_manifest"]
    assert evidence["identity"] == authorized_state.accepted.evidence_identity
    assert evidence["entry_count"] == 16
    assert len(evidence["entries"]) == 16

    reference = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert reference == authorized_state.state_sha256
    for path in (
        ["accepted_h", "evidence_manifest", "identity"],
        ["accepted_h", "evidence_manifest", "file_sha256"],
        ["accepted_h", "evidence_manifest", "entry_count"],
        ["accepted_h", "evidence_manifest", "path"],
    ):
        mutated = json.loads(json.dumps(payload))
        cursor = mutated
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = 1 if path[-1] == "entry_count" else "changed"
        assert hashlib.sha256(canonical_json_bytes(mutated)).hexdigest() != reference, path
    # Removing a single entry from the projection moves it too.
    mutated = json.loads(json.dumps(payload))
    mutated["accepted_h"]["evidence_manifest"]["entries"].pop()
    assert hashlib.sha256(canonical_json_bytes(mutated)).hexdigest() != reference


def test_r5_b_evidence_identity_reaches_publication_transitively(authorized_state):
    """R5-B. The chain from the evidence manifest to the published run identity, link by link.

    No new Layer-2 or run-identity field is needed: `authorized_state_sha256` already commits to
    the whole canonical state, and it is already inside the authorization block that the
    post-Pass-1 result and the published run identity both bind. This proves that rather than
    assuming it.
    """
    gate = dict.fromkeys(_FIXTURE_GATE, True)
    selections = _selections_for(authorized_state)
    pass1 = build_pass1_result(selections, authorized_state, gate)
    expected = _expected_for(pass1)

    # link 1: evidence identity -> canonical state digest
    payload = _payload_of(authorized_state)
    assert (
        payload["accepted_h"]["evidence_manifest"]["identity"]
        == authorized_state.accepted.evidence_identity
    )
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == (
        authorized_state.state_sha256
    )

    # link 2: canonical state digest -> Layer-2 authorization block
    assert pass1["authorization"]["authorized_state_sha256"] == authorized_state.state_sha256
    assert expected.authorization["authorized_state_sha256"] == authorized_state.state_sha256

    # link 3: Layer-2 -> published run identity
    moved = {**dict(expected.authorization), "authorized_state_sha256": "0" * 64}
    assert (
        stage_i_published_run_identity(
            moved,
            post_pass1_result_identity_sha256=expected.result_identity_sha256,
            selection_sequence_commitment_map_sha256=(
                expected.selection_sequence_commitment_map_sha256
            ),
        )
        != expected.stage_i_run_identity
    )


# ---- R5-C: the strict consumer reconciles physical facts first ------------------


def test_r5_c_strict_consumer_reports_physical_mismatch_before_identity(tmp_path: Path):
    """R5-C. A false total plus an invalid run identity must surface the false total.

    The shared reconciler was already physical-first; the consumer was not, because it ran full
    manifest validation -- which recomputes the published run identity -- before the audit.
    """
    manifest, records, expected = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    load_published_realization(final, expected=expected)  # valid before tampering

    def falsify(obj):
        obj["totals"]["serialized_tokens"] = 22
        obj["nodes"][0]["selected_serialized_tokens"] = 22
        obj["stage_i_run"] = {**obj["stage_i_run"], "run_identity": "0" * 64}

    _fully_reseal(final, lambda lines: b"".join(line + b"\n" for line in lines), falsify)
    with pytest.raises(OutputError, match="serialized_tokens is 22 but the staged records"):
        load_published_realization(final, expected=expected)


def _reseal_preserving_fault(final: Path, mutate) -> None:
    """Reseal the marker around a deliberately faulty manifest, leaving the fault in place.

    ``_fully_reseal`` regenerates the run identity from the manifest's own fields, which would
    quietly repair an injected identity fault. Here the mutation is applied last and only the
    COMPLETE marker is brought back into agreement.
    """
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    mutate(manifest)
    manifest_bytes = canonical_json_bytes(manifest)
    (final / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    (final / COMPLETE_MARKER).write_bytes(
        canonical_json_bytes({
            "marker": "COMPLETE",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "record_schema_version": RECORD_SCHEMA,
            "manifest_schema_version": MANIFEST_SCHEMA,
            "h_run_identity": manifest["h_binding"]["h_run_identity"],
            "stage_i_run_identity": manifest["stage_i_run"]["run_identity"],
        })
    )


def test_r5_c_strict_consumer_still_reports_a_lone_identity_fault(tmp_path: Path):
    """A physically honest realization with a bad identity must still fail on the identity."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)
    load_published_realization(final, expected=expected)  # valid before tampering

    _reseal_preserving_fault(
        final,
        lambda obj: obj.__setitem__(
            "stage_i_run", {**obj["stage_i_run"], "run_identity": "0" * 64}
        ),
    )
    with pytest.raises(OutputError, match="run identity|run_identity"):
        load_published_realization(final, expected=expected)


def test_r5_c_strict_consumer_keeps_malformed_structure_controlled(tmp_path: Path):
    """Structural failures stay controlled where the physical audit cannot safely begin."""
    manifest, records, expected = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records, expected=expected)

    pristine = (final / MANIFEST_FILENAME).read_bytes()
    for mutation, match in (
        (lambda obj: obj.pop("shards"), r"missing field\(s\)"),
        (lambda obj: obj.__setitem__("shards", None), "must be a JSON array"),
        (lambda obj: obj.__setitem__("stage_i_run", None), "must be a JSON object"),
        (
            lambda obj: obj["shards"][0].__setitem__("name", "documents-99999.jsonl"),
            "canonical shard order",
        ),
        (lambda obj: obj["shards"][0].__setitem__("records", 0), "must be >= 1"),
    ):
        # Each case starts from the pristine manifest; otherwise the mutations accumulate and a
        # later case fails on an earlier one's damage.
        payload = json.loads(pristine)
        mutation(payload)
        (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
        with pytest.raises(OutputError, match=match):
            load_published_realization(final, expected=expected)
    (final / MANIFEST_FILENAME).write_bytes(pristine)


def test_r5_c_consumer_orders_structure_then_physical_then_identity():
    """Structural: the order is part of the contract, not an accident of call sites."""
    function = _function_source("pretrain/stage_i_output_v1.py", "load_published_realization")
    calls = _call_lines_by_name(function)
    first = {}
    for line, name in sorted(calls):
        first.setdefault(name, line)
    for name in (
        "require_manifest_shape",
        "validate_shard_list",
        "audit_staged_realization",
        "reconcile_manifest_with_audit",
        "validate_manifest",
    ):
        assert name in first, f"the consumer does not call {name}"
    assert first["require_manifest_shape"] < first["validate_shard_list"]
    assert first["validate_shard_list"] < first["audit_staged_realization"]
    assert first["audit_staged_realization"] < first["reconcile_manifest_with_audit"]
    assert first["reconcile_manifest_with_audit"] < first["validate_manifest"], (
        "full manifest validation (which recomputes the run identity) must follow reconciliation"
    )
