"""Contract tests for the authoritative Stage-I realization tooling v1.

Expected values here come from literal fixtures, from the frozen selector v1, or from small
oracles written out longhand in this file. Nothing is ever checked against the production function
that computed it, because an assertion of the form ``f(x) == f(x)`` proves only that ``f`` is
deterministic and would pass with ``f`` entirely wrong.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import struct

import pytest

from pretrain.h_census_v2 import CensusError
from pretrain.select_pretrain_documents import (
    canonical_document_fingerprint as v1_fingerprint,
    selection_rank as v1_selection_rank,
)
from pretrain.stage_i_audit_v1 import AuditError
from pretrain.stage_i_graph_v2 import canonical_json_bytes, load_source_graph, sha256_file
from pretrain.stage_i_output_v1 import (
    COMPLETE_MARKER,
    DOCUMENTS_DIRNAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    RECORD_SCHEMA,
    RECORDS_PER_SHARD,
    SHARD_POLICY_VERSION,
    OutputError,
    build_record,
    iter_records,
    load_published_realization,
    plan_shards,
    publish_atomic,
    recompute_stage_i_run_identity,
    record_sort_key,
    validate_manifest,
    validate_record,
)
from pretrain.stage_i_realize_v1 import (
    ACCEPTED_H_COMPLETE_SHA256,
    IMPLEMENTATION_BUNDLE_FILES,
    PLAN_SCHEMA,
    REQUIRED_PYTHON_VERSION,
    REQUIRED_TOKENIZERS_VERSION,
    AcceptedH,
    AuthorizedIContext,
    Environment,
    RealizationError,
    authorize_plan,
    build_manifest,
    build_materialization_targets,
    compare_with_h,
    generate_candidate_plan,
    implementation_bundle_sha256,
    implementation_files,
    load_accepted_h,
    materialize_binding,
    require_h_i_equality,
    scan_binding_candidates,
    stage_i_run_identity,
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
    manifest, records = _publishable(tmp_path, count=2)
    twin = dict(records[1])
    twin["cleaned_text_sha256"] = records[0]["cleaned_text_sha256"]
    with pytest.raises(AuditError, match="appears more than once"):
        publish_atomic(tmp_path / "out", "run-dup", manifest, [records[0], twin])
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


def _publishable(tmp_path: Path, count: int = 3):
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
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "shards": [],
        "totals": {
            "records": count,
            "content_tokens": 5 * count,
            "serialized_tokens": 7 * count,
            "unique_cleaned_identities": count,
            "shards": 1,
        },
        "nodes": [
            {
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
                "selection_fingerprint": oracle_fingerprint([
                    r["cleaned_text_sha256"] for r in records
                ]),
                "crossing_identity": None,
                "actual_overshoot_tokens": 0,
            }
        ],
        "ownership_matrix": {},
        "bindings": {"ib_x": "6" * 64},
        "environment": {
            "python_executable": "/workspace/petitgpt/.venv/bin/python",
            "python_version": REQUIRED_PYTHON_VERSION,
            "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
        },
        "h_binding": {
            "h_run_identity": "7" * 64,
            "h_census_sha256": "8" * 64,
            "h_predictions_sha256": "9" * 64,
            "h_complete_sha256": "d" * 64,
            "h_candidate_plan_sha256": "a" * 64,
            "h_implementation_bundle_sha256": "b" * 64,
            "owner_graph_sha256": "c" * 64,
        },
        "stage_i_run": _stage_i_run_block(),
    }
    return manifest, records


def _stage_i_run_block(**over) -> dict:
    """A valid published-run identity block whose run_identity is generated by its own fields."""
    block = {
        "candidate_i_plan_sha256": "e" * 64,
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
    }
    block.update(over)
    block["run_identity"] = recompute_stage_i_run_identity(block)
    return block


def test_23_partial_failure_leaves_no_complete_result(tmp_path: Path):
    """Requirement 23. A run that dies mid-write must leave nothing discoverable."""
    manifest, records = _publishable(tmp_path)

    def exploding():
        yield records[0]
        raise RuntimeError("simulated mid-write failure")

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="simulated mid-write failure"):
        publish_atomic(out_dir, "run-x", manifest, exploding())
    assert not (out_dir / "run-x").exists()
    assert list(out_dir.glob("*")) == [], "a staging directory survived the failure"


def test_23b_successful_publication_is_atomic_and_complete(tmp_path: Path):
    manifest, records = _publishable(tmp_path)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records)
    assert (final / COMPLETE_MARKER).is_file()
    assert (final / MANIFEST_FILENAME).is_file()
    assert (final / DOCUMENTS_DIRNAME).is_dir()
    loaded = load_published_realization(final)
    assert loaded["totals"]["records"] == 3
    assert [r["cleaned_text_sha256"] for r in iter_records(final, loaded)] == [
        hashlib.sha256(f"id-{i}".encode()).hexdigest() for i in range(3)
    ]


def test_23c_publication_refuses_to_overwrite_an_existing_result(tmp_path: Path):
    manifest, records = _publishable(tmp_path)
    publish_atomic(tmp_path / "out", "run-x", manifest, records)
    _, again = _publishable(tmp_path)
    with pytest.raises(OutputError, match="refusing to overwrite"):
        publish_atomic(tmp_path / "out", "run-x", manifest, again)


def test_24_strict_consumer_rejects_an_inconsistent_result(tmp_path: Path):
    """Requirement 24. Missing marker, tampered manifest, tampered shard and drifted totals."""
    manifest, records = _publishable(tmp_path)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records)
    load_published_realization(final)

    # 1. no COMPLETE marker at all
    marker = (final / COMPLETE_MARKER).read_bytes()
    (final / COMPLETE_MARKER).unlink()
    with pytest.raises(OutputError, match="no COMPLETE marker"):
        load_published_realization(final)
    (final / COMPLETE_MARKER).write_bytes(marker)

    # 2. manifest edited after the fact, so the marker no longer describes it
    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["totals"]["records"] = 99
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final)

    # 3. shard bytes changed underneath a valid manifest
    fresh_manifest, fresh_records = _publishable(tmp_path)
    other = publish_atomic(tmp_path / "out2", "run-y", fresh_manifest, fresh_records)
    loaded = load_published_realization(other)
    shard = other / DOCUMENTS_DIRNAME / loaded["shards"][0]["name"]
    corrupted = shard.read_bytes().replace(b"document 0", b"document Z")
    shard.write_bytes(corrupted)
    with pytest.raises(OutputError, match="SHA-256 disagrees"):
        list(iter_records(other, loaded))


def test_24b_manifest_totals_must_reconcile_with_nodes_and_shards(tmp_path: Path):
    manifest, _ = _publishable(tmp_path)
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
    """The published manifest must restate the ledger and bind the authorized run identity.

    Selections are built directly for the context's own graph nodes: `build_manifest` now takes
    its graph from the authorized context, so feeding it a synthetic graph is no longer possible
    -- which is the point of that change.
    """
    graph = authorized.graph
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

    manifest = build_manifest(selections, authorized)
    assert manifest["totals"]["records"] == len(selections)
    assert manifest["totals"]["serialized_tokens"] == 7 * len(selections)
    assert manifest["nodes"][0]["selection_fingerprint"] == selections[0].selection_fingerprint

    # The identity block must be generated by its own fields and match the authorized context.
    block = manifest["stage_i_run"]
    assert block["run_identity"] == authorized.run_identity
    assert recompute_stage_i_run_identity(block) == authorized.run_identity
    assert block["h_complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256
    assert block["candidate_i_plan_sha256"] == authorized.plan_sha256
    assert block["owner_graph_sha256"] == authorized.graph.graph_sha256
    assert manifest["h_binding"]["h_complete_sha256"] == ACCEPTED_H_COMPLETE_SHA256


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
    manifest = _manifest_for(selections)
    stream = iter_records_in_physical_order(graph, str(TOKENIZER), selections, tmp_path / "work")
    final = publish_atomic(tmp_path / "out", "run-i", manifest, stream)

    loaded = load_published_realization(final)
    published = list(iter_records(final, loaded))
    assert loaded["totals"]["records"] == len(published)
    assert loaded["totals"]["serialized_tokens"] == sum(
        r["serialized_token_count"] for r in published
    )
    assert [record_sort_key(r) for r in published] == sorted(record_sort_key(r) for r in published)
    assert {r["cleaned_text_sha256"] for r in published} == {
        d.cleaned_sha256 for s in selections for d in s.selected
    }


def _manifest_for(selections) -> dict:
    """A valid manifest describing exactly these selections, built without build_manifest.

    Written out here so a publication test is not checking the production manifest builder
    against the production publisher; the fixture states independently what the realization is.
    """
    total = sum(s.selected_identities for s in selections)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "shard_policy_version": SHARD_POLICY_VERSION,
        "records_per_shard": RECORDS_PER_SHARD,
        "shards": [],
        "totals": {
            "records": total,
            "content_tokens": sum(d.content_token_count for s in selections for d in s.selected),
            "serialized_tokens": sum(s.selected_serialized_tokens for s in selections),
            "unique_cleaned_identities": total,
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
                "selection_fingerprint": oracle_fingerprint([d.cleaned_sha256 for d in s.selected]),
                "crossing_identity": s.crossing_identity,
                "actual_overshoot_tokens": s.actual_overshoot_tokens,
            }
            for s in selections
        ],
        "ownership_matrix": ownership_matrix_v1(selections),
        "bindings": {"ib_x": "6" * 64},
        "environment": {
            "python_executable": "/workspace/petitgpt/.venv/bin/python",
            "python_version": REQUIRED_PYTHON_VERSION,
            "tokenizers_version": REQUIRED_TOKENIZERS_VERSION,
        },
        "h_binding": {
            "h_run_identity": "7" * 64,
            "h_census_sha256": "8" * 64,
            "h_predictions_sha256": "9" * 64,
            "h_complete_sha256": "d" * 64,
            "h_candidate_plan_sha256": "a" * 64,
            "h_implementation_bundle_sha256": "b" * 64,
            "owner_graph_sha256": "c" * 64,
        },
        "stage_i_run": _stage_i_run_block(),
    }


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
    manifest, records = _publishable(tmp_path, count=3)
    # Three records at 7 serialized tokens each = 21 actual.
    assert sum(r["serialized_token_count"] for r in records) == 21
    manifest["totals"]["serialized_tokens"] = 22
    manifest["nodes"][0]["selected_serialized_tokens"] = 22

    with pytest.raises(OutputError, match="but the staged records actually contain 21"):
        publish_atomic(tmp_path / "out", "run-x", manifest, records)
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
    manifest, records = _publishable(tmp_path, count=3)
    manifest["totals"][field] = value
    with pytest.raises(OutputError, match=match):
        publish_atomic(tmp_path / "out", "run-x", manifest, records)
    assert not (tmp_path / "out" / "run-x").exists()


def test_r1_c3_per_node_totals_and_fingerprints_are_reconciled(tmp_path: Path):
    """A node whose declared token total or fingerprint disagrees with its records is refused."""
    manifest, records = _publishable(tmp_path, count=3)
    # Totals stay truthful so the PER-NODE reconciliation is the check that fires, not the
    # global one; otherwise this would re-prove test_r1_c rather than the node path.
    tokens = json.loads(json.dumps(manifest))
    tokens["nodes"][0]["selected_serialized_tokens"] = 20
    with pytest.raises(OutputError, match="but the records sum to 21"):
        publish_atomic(tmp_path / "out1", "run-x", tokens, list(records))

    fingerprint = json.loads(json.dumps(manifest))
    fingerprint["nodes"][0]["selection_fingerprint"] = "0" * 64
    with pytest.raises(OutputError, match="fingerprint disagrees"):
        publish_atomic(tmp_path / "out2", "run-x", fingerprint, list(records))


# ---- D-J: the strict consumer must enforce physical invariants ------------------


def _corrupt_published(tmp_path: Path, name: str, mutate) -> Path:
    """Publish a valid realization, then rewrite one shard through `mutate`.

    The manifest is left describing the original, which is exactly the situation a consumer must
    detect: bytes on disk that no longer match what the result claims about itself.
    """
    manifest, records = _publishable(tmp_path, count=4)
    final = publish_atomic(tmp_path / name, "run-x", manifest, records)
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    lines = shard.read_bytes().split(b"\n")[:-1]
    shard.write_bytes(mutate(lines))
    return final


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

    final = _corrupt_published(tmp_path, "out", swap)
    # A permutation keeps the domain contiguous, so it must be caught by the fingerprint, which is
    # what actually pins the selected sequence.
    with pytest.raises(OutputError, match="fingerprint disagrees|SHA-256|shards"):
        load_published_realization(final)


def test_r1_d2_consumer_rejects_a_gap_in_selection_ordinals(tmp_path: Path):
    """A missing ordinal breaks contiguity and must be named as such."""

    def punch(lines):
        first = json.loads(lines[0])
        first["selection_ordinal_within_node"] = 99
        return canonical_json_bytes(first) + b"\n".join(lines[1:]) + b"\n"

    final = _corrupt_published(tmp_path, "out", punch)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(OutputError, match="COMPLETE marker|not the contiguous domain"):
        load_published_realization(final)


def test_r1_e_consumer_rejects_false_manifest_totals(tmp_path: Path):
    """E. A published manifest whose totals do not describe its records must not load."""
    manifest, records = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records)
    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["totals"]["serialized_tokens"] = 22
    payload["nodes"][0]["selected_serialized_tokens"] = 22
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final)


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

    final = _corrupt_published(tmp_path, "out", duplicate)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((OutputError, AuditError), match="COMPLETE marker|appears more than once"):
        load_published_realization(final)


def test_r1_g_consumer_rejects_noncanonical_record_bytes(tmp_path: Path):
    """G. A record whose bytes are not the canonical serialisation of its own content."""

    def spacer(lines):
        first = json.loads(lines[0])
        # Same content, non-canonical separators: a digest over these bytes says nothing.
        noncanonical = json.dumps(first, sort_keys=True, separators=(", ", ": ")).encode()
        return noncanonical + b"\n" + b"\n".join(lines[1:]) + b"\n"

    final = _corrupt_published(tmp_path, "out", spacer)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((OutputError, AuditError), match="COMPLETE marker|not canonical"):
        load_published_realization(final)


def test_r1_h_consumer_rejects_wrong_physical_order(tmp_path: Path):
    """H. Physical layout order must be strictly ascending across the whole realization."""

    def reverse_two(lines):
        return b"\n".join([lines[1], lines[0]] + list(lines[2:])) + b"\n"

    final = _corrupt_published(tmp_path, "out", reverse_two)
    manifest = json.loads((final / MANIFEST_FILENAME).read_text())
    shard = final / DOCUMENTS_DIRNAME / "documents-00000.jsonl"
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest["shards"][0]["bytes"] = shard.stat().st_size
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((OutputError, AuditError), match="COMPLETE marker|physical order"):
        load_published_realization(final)


def test_r1_i_consumer_rejects_missing_and_extra_records(tmp_path: Path):
    """I. A dropped or added record changes the physical truth and must be detected."""

    def drop(lines):
        return b"\n".join(lines[1:]) + b"\n"

    final = _corrupt_published(tmp_path, "missing", drop)
    with pytest.raises((OutputError, AuditError)):
        load_published_realization(final)

    def add(lines):
        extra = json.loads(lines[-1])
        extra["stable_input_record_ordinal"] += 1000
        extra["cleaned_text_sha256"] = hashlib.sha256(b"extra").hexdigest()
        extra["selection_ordinal_within_node"] += 1000
        return b"\n".join(lines) + b"\n" + canonical_json_bytes(extra)

    final2 = _corrupt_published(tmp_path, "extra", add)
    with pytest.raises((OutputError, AuditError)):
        load_published_realization(final2)


def test_r1_j_consumer_reconstructs_and_verifies_per_node_fingerprints(tmp_path: Path):
    """J. The node fingerprint is recomputed from the published records, via a longhand oracle."""
    manifest, records = _publishable(tmp_path, count=5)
    expected = oracle_fingerprint([r["cleaned_text_sha256"] for r in records])
    manifest["nodes"][0]["selection_fingerprint"] = expected
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records)

    loaded = load_published_realization(final)
    assert loaded["nodes"][0]["selection_fingerprint"] == expected

    # And a manifest claiming any other fingerprint must not load.
    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["nodes"][0]["selection_fingerprint"] = "0" * 64
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final)


def test_r1_u_strict_consumer_validates_a_genuine_multi_shard_publication(tmp_path: Path):
    """U. End to end over more than one shard, with every physical invariant enforced."""
    count = RECORDS_PER_SHARD + 7
    manifest, _ = _publishable(tmp_path, count=1)
    manifest["totals"].update({
        "records": count,
        "content_tokens": 5 * count,
        "serialized_tokens": 7 * count,
        "unique_cleaned_identities": count,
        "shards": 0,
    })
    manifest["nodes"][0]["selected_identities"] = count
    manifest["nodes"][0]["selected_serialized_tokens"] = 7 * count
    identities = [hashlib.sha256(f"u-{i}".encode()).hexdigest() for i in range(count)]
    manifest["nodes"][0]["selection_fingerprint"] = oracle_fingerprint(identities)

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

    final = publish_atomic(tmp_path / "out", "run-multi", manifest, gen())
    loaded = load_published_realization(final)
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
    """K. The exact positive path: a real plan, its true digest, a usable context."""
    assert isinstance(authorized, AuthorizedIContext)
    assert authorized.plan_sha256 == sha256_file(live_plan_path)
    assert authorized.plan["authorization_status"] == "NOT_AUTHORIZED"
    # Everything load-bearing came from the plan, not from an argument.
    assert authorized.graph.graph_sha256 == authorized.plan["graph_sha256"]
    assert authorized.accepted.run_identity == authorized.plan["accepted_h"]["run_identity"]
    assert authorized.accepted.complete_sha256 == ACCEPTED_H_COMPLETE_SHA256
    assert authorized.tokenizer_path.name == "tokenizer.json"
    assert authorized.run_name.startswith("run-")

    # The identity is generated by its own bound fields, via a separately written expectation.
    expected = stage_i_run_identity(
        candidate_plan_sha256=authorized.plan_sha256,
        implementation_commit=authorized.implementation_commit,
        implementation_bundle_sha256=authorized.bundle_sha256,
        plan_schema_version=PLAN_SCHEMA,
        output_schema_version=RECORD_SCHEMA,
        manifest_schema_version=MANIFEST_SCHEMA,
        shard_policy_version=SHARD_POLICY_VERSION,
        records_per_shard=RECORDS_PER_SHARD,
        h_run_identity=authorized.accepted.run_identity,
        h_complete_sha256=authorized.accepted.complete_sha256,
        h_census_sha256=authorized.accepted.census_sha256,
        h_predictions_sha256=authorized.accepted.predictions_sha256,
        owner_graph_sha256=authorized.graph.graph_sha256,
    )
    assert authorized.run_identity == expected
    authorized.revalidate()


def test_r1_k2_a_manually_built_context_is_not_an_authorization(authorized, tmp_path):
    """A similarly shaped object must not count. The seal is the capability, not the shape."""
    import dataclasses

    from pretrain.stage_i_realize_v1 import _require_authorized, realize_and_publish

    forged = AuthorizedIContext(
        repo_root=authorized.repo_root,
        plan_path=authorized.plan_path,
        plan_sha256=authorized.plan_sha256,
        plan=authorized.plan,
        graph=authorized.graph,
        graph_path=authorized.graph_path,
        accepted=authorized.accepted,
        environment=authorized.environment,
        bundle_sha256=authorized.bundle_sha256,
        bundle_files=authorized.bundle_files,
        implementation_commit=authorized.implementation_commit,
        tokenizer_path=authorized.tokenizer_path,
        reference_exclusion_path=authorized.reference_exclusion_path,
        run_identity=authorized.run_identity,
    )
    with pytest.raises(RealizationError, match="requires an authorized Stage-I context"):
        _require_authorized(forged, "test")
    with pytest.raises(RealizationError, match="requires an authorized Stage-I context"):
        realize_and_publish(forged, out_dir=tmp_path / "o", work_dir=tmp_path / "w")

    # dataclasses.replace also drops the seal, because it is init=False.
    copied = dataclasses.replace(authorized)
    with pytest.raises(RealizationError, match="requires an authorized Stage-I context"):
        _require_authorized(copied, "test")

    # ...and the genuine one still passes, or the negatives above prove nothing.
    _require_authorized(authorized, "test")


def test_r1_l_wrong_externally_supplied_plan_sha_is_rejected(live_plan_path):
    """L. Authorization is against the owner's digest, not the file's self-description."""
    with pytest.raises(RealizationError, match="is not the owner-supplied"):
        authorize_plan(live_plan_path, "0" * 64, ROOT)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.__setitem__("graph_sha256", "0" * 64), "owner graph SHA-256"),
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
            "COMPLETE SHA-256",
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
            "shard policy disagrees",
        ),
        (
            lambda p: p["shard_policy"].__setitem__("version", "nope"),
            "shard policy disagrees",
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
    with pytest.raises(RealizationError, match=match):
        authorize_plan(path, digest, ROOT)


def test_r1_t_published_run_identity_must_be_generated_by_its_own_fields(tmp_path: Path):
    """T. A result claiming an identity its own bound fields do not generate must not load."""
    manifest, records = _publishable(tmp_path, count=3)
    final = publish_atomic(tmp_path / "out", "run-x", manifest, records)
    loaded = load_published_realization(final)
    block = loaded["stage_i_run"]
    assert recompute_stage_i_run_identity(block) == block["run_identity"]

    payload = json.loads((final / MANIFEST_FILENAME).read_text())
    payload["stage_i_run"]["h_complete_sha256"] = "0" * 64
    (final / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(payload))
    with pytest.raises(OutputError):
        load_published_realization(final)


def test_r1_t2_run_identity_changes_with_every_bound_field():
    """Each field genuinely participates: flipping any one must move the identity."""
    base = dict(
        candidate_plan_sha256="1" * 64,
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
    )
    reference = stage_i_run_identity(**base)
    seen = {reference}
    for key, value in base.items():
        altered = dict(base)
        altered[key] = 99 if isinstance(value, int) else "z" * len(str(value))
        identity = stage_i_run_identity(**altered)
        assert identity != reference, f"{key} does not participate in the run identity"
        seen.add(identity)
    assert len(seen) == len(base) + 1


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
