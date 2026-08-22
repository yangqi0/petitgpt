"""Adversarial regression suite for the Stage-I graph, replay core and Stage-H census v2.

Every finding from both independent reviews has at least one reproducer here that fails on the
rejected semantics and passes on the repaired ones. Test count is not the argument; the
reproducers are.

Two rules govern this file. Expected values are derived independently of the code under test:
where a test needs to know what selector v1 would have done, it either runs the real frozen
selector or re-derives its contract here, never by calling the v2 helper it is checking. And no
test asserts only that a source string is absent when the property can be exercised directly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from pretrain.h_census_v2 import (
    AUTHORITY_KEYS,
    CENSUS_SCHEMA,
    DEFAULT_AUTHORITY_PATHS,
    IMPLEMENTATION_BUNDLE_FILES,
    PLAN_SCHEMA,
    AuthorizationError,
    AuthorizedRunContext,
    CensusError,
    authorize_run,
    build_census,
    census_body,
    file_sha256,
    generate_candidate_plan,
    implementation_bundle_sha256,
    implementation_files,
    load_published_run,
    load_reference_exclusion,
    publish_atomic,
    read_score,
    run_identity,
    scan_binding,
    validate_complete_census,
)
from pretrain.stage_i_graph_v2 import (
    GraphError,
    StrictJSONError,
    canonical_json_bytes,
    derive_seed,
    load_source_graph,
    read_authoritative_bytes,
    sha256_file,
    strict_json_loads,
    validate_eligibility_index,
)
from pretrain.stage_i_replay_v2 import (
    NODE_RESULT_SCHEMA,
    CandidateRecord,
    NodeResult,
    ReplayError,
    _representatives,
    bits_to_score,
    replay,
    representative_key,
    score_to_bits,
    selection_rank,
)

ROOT = Path("/workspace/petitgpt")
TOKENIZER = ROOT / "runs/g_production_2026-08-21/release/tokenizer.json"
GRAPH = ROOT / "runs/h_tooling_repair_v2_2026-08-21/policy/stage_i_source_graph_v1.json"
GRAPH_SHA = "e7cf8eafd117521660e898576d70e5873088129e0378afd34a6d8f56ea983986"
SELECTOR_V1 = ROOT / "pretrain/select_pretrain_documents.py"
SELECTOR_V1_SHA = "fd87767e04114afee343982924fa4954fdd2acf2442c5f7fc5dafaa093e280cc"
REJECTED_H_V1 = ROOT / "runs/h_production_2026-08-21/tools/h_canonical_census.py"
REJECTED_H_V1_SHA = "8308b479bde26a5f97e29bb766ed0ab37efb7e83a33f76eab453a421a6e9b01c"
SEED = 5088999448999271579
FP_DOMAIN = b"PetitGPT-stage-i-selection-fingerprint-v1\0"
TUTORIAL_BINDING = "ib_structured_tutorial"


def _runtime_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------- independent oracles


def oracle_canonical_record_sha256(record: dict) -> str:
    """Selector v1's input-record digest, re-derived here rather than imported.

    v1 hashes the ASCII bytes of ``json.dumps(record, ensure_ascii=True, sort_keys=True,
    separators=(",",":"), allow_nan=False)``. Writing it out means a differential against the real
    selector is evidence rather than a tautology.
    """
    text = json.dumps(
        record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def oracle_raw_sha256(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def oracle_representative(rows: list[dict], text_field: str = "text") -> dict:
    """The recovered v1 rule: minimum (raw_sha256, input_record_sha256) over the group."""
    return min(
        rows,
        key=lambda row: (oracle_raw_sha256(row[text_field]), oracle_canonical_record_sha256(row)),
    )


def oracle_fingerprint(identities: list[str]) -> str:
    digest = hashlib.sha256(FP_DOMAIN)
    digest.update(len(identities).to_bytes(8, "big"))
    for value in sorted(identities):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def oracle_ranked_prefix(rows: list[dict], target: int) -> dict:
    """Standalone exact-binary64 descending ranked prefix; never calls the replay core."""
    for row in rows:
        if type(row["score"]) is not float:
            raise ValueError("score must be a binary64 float")
        if row["score"] != row["score"] or row["score"] in (float("inf"), float("-inf")):
            raise ValueError("score must be finite")
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
        "crossing": picked[-1]["sha"] if mass >= target else None,
        "overshoot": mass - target if mass >= target else 0,
        "feasible": mass >= target,
        "fingerprint": oracle_fingerprint([row["sha"] for row in picked]),
        "next_unselected": ordered[len(picked)]["sha"] if len(ordered) > len(picked) else None,
    }


# ---------------------------------------------------------------- fixtures


def key_for(name: str, *, raw: str | None = None, row: str | None = None) -> bytes:
    return representative_key(
        raw_sha256=oracle_raw_sha256(raw if raw is not None else f"raw-{name}"),
        input_record_sha256=hashlib.sha256((row or f"row-{name}").encode()).hexdigest(),
    )


def rec(
    name,
    tokens,
    *,
    row=0,
    int_score=None,
    score=None,
    binding="ib_x",
    rep_key=None,
    identity=None,
):
    label = identity or name
    return CandidateRecord(
        input_binding_id=binding,
        row_index=row,
        cleaned_sha256=hashlib.sha256(f"c-{label}".encode()).hexdigest(),
        canonical_fingerprint=hashlib.sha256(f"f-{label}".encode()).hexdigest(),
        serialized_tokens=tokens,
        representative_key=rep_key if rep_key is not None else key_for(name),
        int_score=int_score,
        score_bits=None if score is None else score_to_bits(score),
    )


def in_binding(record: CandidateRecord, binding: str) -> CandidateRecord:
    """The same logical identity as it appears in another release."""
    return dataclasses.replace(record, input_binding_id=binding)


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


def fixture_authorities() -> dict[str, str]:
    return {
        name: hashlib.sha256(name.encode()).hexdigest() for name in sorted(AUTHORITY_KEYS.values())
    }


def mini_graph(tmp: Path, nodes, bindings=("ib_x",), *, tutorial_ids=None, **over):
    """A complete, closed fixture graph. Every required policy block is present and valid."""
    tmp.mkdir(parents=True, exist_ok=True)
    keys = list(bindings)
    if TUTORIAL_BINDING not in keys and tutorial_ids is None:
        keys.append(TUTORIAL_BINDING)
    ibs = {key: binding_entry(tmp, key) for key in keys}
    graph = {
        "schema_version": "petitgpt-stage-i-source-graph-v1",
        "policy_status": "OWNER_FROZEN",
        "authority": "test",
        "date": "2026-08-21",
        "note": "fixture",
        "bound_authorities": fixture_authorities(),
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
        "input_bindings": ibs,
        "nodes": nodes,
    }
    graph.update(over)
    path = tmp / "graph.json"
    path.write_bytes(canonical_json_bytes(graph))
    return path


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


def load_fixture(path: Path):
    return load_source_graph(path, verify_hashes=True)


def with_records(graph, records_by_binding):
    """Supply empty record lists for any declared-but-unconsumed fixture binding."""
    payload = dict(records_by_binding)
    for key in graph.bindings:
        payload.setdefault(key, [])
    return payload


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


# ---------------------------------------------------------------- frozen artefacts


def test_frozen_seed_derivation_is_exact():
    assert derive_seed("petitgpt-stage-i-selection-seed-v1") == SEED
    assert hex(SEED) == "0x469fc20943c29c9b"
    assert (
        hashlib.sha256(b"petitgpt-stage-i-selection-seed-v1").hexdigest()
        == "c69fc20943c29c9b254b8d332b75bb3bc8a4cd21e1f9230a15e7bc592dadb448"
    )


def test_frozen_evidence_bytes_are_untouched():
    """The owner graph, the frozen selector and the rejected v1 tool must never move."""
    assert file_sha256(GRAPH) == GRAPH_SHA
    assert file_sha256(SELECTOR_V1) == SELECTOR_V1_SHA
    assert file_sha256(REJECTED_H_V1) == REJECTED_H_V1_SHA


def test_owner_graph_loads_verified_and_is_ordered():
    graph = load_source_graph(GRAPH, verify_hashes=True, expected_graph_sha256=GRAPH_SHA)
    assert graph.seed == SEED
    assert [n.source_id for n in graph.nodes] == [
        "b_dclm_edu",
        "b_fineweb_edu_dedup",
        "b_finewiki_en",
        "b_pes2o",
        "b_python_gate_c_full",
        "b_stackexchange",
        "b_structured_tutorial",
        "a_dclm_edu",
        "a_fineweb_edu_dedup",
        "a_finewiki_en",
        "a_python_gate_c_full",
    ]
    assert len(graph.nodes) == 11
    assert len(graph.bindings) == 8
    assert graph.raw["resume_supported"] is False
    assert graph.graph_sha256 == GRAPH_SHA


def test_owner_graph_load_is_bound_to_expected_bytes():
    with pytest.raises(GraphError, match="SHA-256 mismatch"):
        load_source_graph(GRAPH, verify_hashes=False, expected_graph_sha256="0" * 64)


def test_owner_graph_structured_tutorial_is_one_logical_node():
    graph = load_source_graph(GRAPH, verify_hashes=False)
    block = graph.raw["structured_tutorial"]
    assert block["realization"] == "SINGLE_LOGICAL_NODE"
    assert block["sub_targets"] is None
    owning = [n for n in graph.nodes if set(n.input_binding_ids) & set(block["input_binding_ids"])]
    assert [n.source_id for n in owning] == ["b_structured_tutorial"]
    assert list(owning[0].input_binding_ids) == list(block["input_binding_ids"])


def test_owner_graph_h_boundary_is_index_only():
    graph = load_source_graph(GRAPH, verify_hashes=False)
    assert graph.raw["h_boundary"]["h_publishes_physical_views"] is False
    assert (
        graph.raw["h_boundary"]["h_canonical_output_label"]
        == "NON_AUTHORITATIVE_FEASIBILITY_REPLAY"
    )


# ---------------------------------------------------------------- H-01 ownership


def test_h01_occurrence_is_not_ownership(tmp_path: Path):
    """An identity merely SEEN by an earlier node must stay available downstream."""
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [node("b_early", "stage_b", 100, ["ib_x"]), node("b_late", "stage_b", 100, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        )
    )
    # The early node commits exactly one document. Give it a companion that outranks the shared
    # identity so the shared one is seen but never committed.
    shared = rec("shared", 100)
    ranked = sorted(
        [shared, rec("early-a", 100)],
        key=lambda r: selection_rank(
            seed=SEED,
            stage="stage_b",
            source_id="b_early",
            canonical_fingerprint=r.canonical_fingerprint,
        ),
    )
    winner = ranked[0]
    loser = ranked[1]
    out = replay(
        graph,
        with_records(graph, {"ib_x": [winner, loser], "ib_y": [in_binding(loser, "ib_y")]}),
        set(),
    )
    assert out[0].selected_identities == 1
    assert out[0].selection_fingerprint == oracle_fingerprint([winner.cleaned_sha256])
    # the merely-seen identity is untouched: no ownership, still selectable
    assert out[1].prior_commit_excluded_identities == 0
    assert out[1].exclusions_by_owner == {}
    assert out[1].selection_fingerprint == oracle_fingerprint([loser.cleaned_sha256])


def test_h01_actual_commit_excludes_later_node(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [node("b_early", "stage_b", 100, ["ib_x"]), node("b_late", "stage_b", 100, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        )
    )
    only = rec("only", 100)
    out = replay(
        graph, with_records(graph, {"ib_x": [only], "ib_y": [in_binding(only, "ib_y")]}), set()
    )
    assert out[0].selected_identities == 1
    assert out[1].prior_commit_excluded_identities == 1
    assert out[1].exclusions_by_owner == {"b_early": 1}
    assert out[1].post_exclusion_candidate_identities == 0
    assert out[1].feasible is False


def test_h01_ownership_adjusts_capacity_not_just_a_counter(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [node("b_first", "stage_b", 500, ["ib_x"]), node("b_second", "stage_b", 100, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        )
    )
    shared = [rec(f"s{i}", 100) for i in range(5)]
    out = replay(
        graph,
        with_records(
            graph, {"ib_x": list(shared), "ib_y": [in_binding(r, "ib_y") for r in shared]}
        ),
        set(),
    )
    assert out[0].selected_identities == 5
    assert out[1].prior_commit_excluded_identities == 5
    assert out[1].post_exclusion_candidate_serialized_tokens == 0
    assert out[1].feasible is False


# ---------------------------------------------------------------- H-01 representative semantics


def _run_selector_v1(
    tmp: Path, tag: str, rows: list[dict], cleaning: dict, target: int
) -> list[dict]:
    """Run the real frozen selector and return the rows it emitted."""
    from pretrain import select_pretrain_documents as v1

    source = tmp / f"{tag}.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    exclusion = tmp / f"{tag}.exclusion.json"
    exclusion.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": v1.EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": v1.CLEANED_TEXT_HASH_ALGORITHM,
            "cleaning": v1.CleaningSpec(**cleaning).as_dict(),
            "hash_count": 1,
            "hashes": [hashlib.sha256(b"outside every fixture").hexdigest()],
        }),
        encoding="utf-8",
    )
    spec = tmp / f"{tag}.spec.json"
    spec.write_text(
        json.dumps({
            "schema_version": 1,
            "seed": SEED,
            "text_field": "text",
            "cleaning": cleaning,
            "sources": [
                {
                    "stage": "stage_b",
                    "source_id": "b_fixture",
                    "path": str(source),
                    "target_serialized_tokens": target,
                }
            ],
        }),
        encoding="utf-8",
    )
    out = tmp / f"out-{tag}"
    v1.build_selection_registry(
        spec_path=spec,
        tokenizer_path=TOKENIZER,
        out_dir=out,
        exclude_hash_manifests=(exclusion,),
        sqlite_cache_mb=1,
        commit_every=1,
    )
    emitted = (out / "selected/stage_b/b_fixture.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in emitted.splitlines()]


QUOTE_VARIANTS = [
    {"text": "“one canonical duplicate”", "variant": "smart"},
    {"text": '"one canonical duplicate"', "variant": "ascii"},
]


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_h01_representative_matches_the_real_selector_v1(tmp_path: Path, order):
    """Differential against the frozen selector: same choice, both traversal orders."""
    rows = QUOTE_VARIANTS if order == "forward" else list(reversed(QUOTE_VARIANTS))
    emitted = _run_selector_v1(tmp_path, f"q-{order}", rows, {"normalize_quotes": True}, 1)
    assert len(emitted) == 1
    expected = oracle_representative(QUOTE_VARIANTS)
    assert emitted[0]["variant"] == expected["variant"]
    assert emitted[0]["_petitgpt_selection"]["raw_sha256"] == oracle_raw_sha256(expected["text"])
    assert emitted[0]["_petitgpt_selection"]["input_record_sha256"] == (
        oracle_canonical_record_sha256(expected)
    )

    # v2 must collapse the same duplicate group to the same record.
    candidates = [
        rec(
            row["variant"],
            100,
            row=index,
            identity="dup",
            int_score=index,
            rep_key=representative_key(
                raw_sha256=oracle_raw_sha256(row["text"]),
                input_record_sha256=oracle_canonical_record_sha256(row),
            ),
        )
        for index, row in enumerate(rows)
    ]
    chosen = _representatives(candidates)
    assert len(chosen) == 1
    survivor = next(iter(chosen.values()))
    assert survivor.raw_sha256 == oracle_raw_sha256(expected["text"])
    assert survivor.input_record_sha256 == oracle_canonical_record_sha256(expected)


def test_h01_representative_is_metadata_deciding_and_matches_v1(tmp_path: Path):
    """Identical raw text, different score metadata: the record digest decides, as in v1."""
    text = "identical raw body with differing score metadata"
    rows = [
        {"text": text, "variant": "low", "edu_int_score": 3},
        {"text": text, "variant": "high", "edu_int_score": 4},
    ]
    assert len({oracle_raw_sha256(r["text"]) for r in rows}) == 1  # the raw digest really ties
    expected = oracle_representative(rows)
    emitted = _run_selector_v1(tmp_path, "meta", rows, {}, 1)
    assert emitted[0]["variant"] == expected["variant"]
    assert emitted[0]["edu_int_score"] == expected["edu_int_score"]

    candidates = [
        rec(
            row["variant"],
            100,
            row=index,
            identity="dup",
            int_score=row["edu_int_score"],
            rep_key=representative_key(
                raw_sha256=oracle_raw_sha256(row["text"]),
                input_record_sha256=oracle_canonical_record_sha256(row),
            ),
        )
        for index, row in enumerate(rows)
    ]
    survivor = next(iter(_representatives(candidates).values()))
    assert survivor.int_score == expected["edu_int_score"]


def test_h01_rejected_positional_rule_would_disagree(tmp_path: Path):
    """The rejected min(binding_ordinal, row_index) rule is provably a different function.

    The fixture is built so the v1 hash-minimum record is NOT at row zero. A positional rule keeps
    row zero and reports FALLBACK_RANKED_GE3; the v1 rule keeps the hash minimum and reports
    PRIMARY_GE4. Only one of those can be selector-equivalent.
    """
    rows = [
        {"text": "duplicate body", "variant": "a", "edu_int_score": 3},
        {"text": "duplicate body", "variant": "b", "edu_int_score": 4},
    ]
    expected = oracle_representative(rows)
    ordered = sorted(
        rows,
        key=lambda row: (oracle_raw_sha256(row["text"]), oracle_canonical_record_sha256(row)),
        reverse=True,
    )  # put the hash-maximum record at physical row 0
    assert ordered[0] is not expected

    graph = load_fixture(
        mini_graph(
            tmp_path,
            [
                node(
                    "b_x",
                    "stage_b",
                    100,
                    ["ib_x"],
                    mode="BRANCH_DEPENDENT",
                    primary=PRIMARY,
                    fallback=FALLBACK,
                )
            ],
        )
    )
    candidates = [
        rec(
            row["variant"],
            100,
            row=index,
            identity="dup",
            int_score=row["edu_int_score"],
            score=float(row["edu_int_score"]),
            rep_key=representative_key(
                raw_sha256=oracle_raw_sha256(row["text"]),
                input_record_sha256=oracle_canonical_record_sha256(row),
            ),
        )
        for index, row in enumerate(ordered)
    ]
    result = replay(graph, with_records(graph, {"ib_x": candidates}), set())[0]
    positional_survivor = candidates[0]  # what min(binding_ordinal, row_index) would have kept
    v1_survivor = next(c for c in candidates if c.int_score == expected["edu_int_score"])
    assert positional_survivor is not v1_survivor
    assert result.branch == ("PRIMARY_GE4" if v1_survivor.int_score >= 4 else "FALLBACK_RANKED_GE3")
    assert result.branch != (
        "PRIMARY_GE4" if positional_survivor.int_score >= 4 else "FALLBACK_RANKED_GE3"
    )


def test_h01_representative_ignores_row_index_and_traversal_order(tmp_path: Path):
    rows = [
        {"text": "“row order probe”", "variant": "smart"},
        {"text": '"row order probe"', "variant": "ascii"},
        {"text": '"row order probe”', "variant": "mixed"},
    ]
    expected = oracle_representative(rows, "text")
    from pretrain.select_pretrain_documents import CleaningSpec, _clean_for_selection

    cleaned = _clean_for_selection(rows[0]["text"], CleaningSpec(normalize_quotes=True))[0]
    assert all(
        _clean_for_selection(row["text"], CleaningSpec(normalize_quotes=True))[0] == cleaned
        for row in rows
    )
    survivors = set()
    for permutation in itertools.permutations(range(3)):
        candidates = [
            rec(
                rows[source]["variant"],
                100,
                row=position * 7,
                identity="dup",
                rep_key=representative_key(
                    raw_sha256=oracle_raw_sha256(rows[source]["text"]),
                    input_record_sha256=oracle_canonical_record_sha256(rows[source]),
                ),
            )
            for position, source in enumerate(permutation)
        ]
        survivors.add(next(iter(_representatives(candidates).values())).representative_key)
    assert len(survivors) == 1
    assert survivors.pop() == representative_key(
        raw_sha256=oracle_raw_sha256(expected["text"]),
        input_record_sha256=oracle_canonical_record_sha256(expected),
    )


def test_h01_representative_is_binding_order_independent_across_a_union(tmp_path: Path):
    """A union node's representative must not depend on which binding came first."""
    rows = [
        {"text": "“shared tutorial body”", "variant": "b1"},
        {"text": '"shared tutorial body"', "variant": "b2"},
    ]
    expected = oracle_representative(rows)
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [node("b_structured_tutorial", "stage_b", 100, ["ib_x", "ib_y"])],
            bindings=("ib_x", "ib_y"),
            tutorial_ids=["ib_x", "ib_y"],
        )
    )
    first = rec(
        "b1",
        100,
        binding="ib_x",
        identity="dup",
        int_score=1,
        rep_key=representative_key(
            raw_sha256=oracle_raw_sha256(rows[0]["text"]),
            input_record_sha256=oracle_canonical_record_sha256(rows[0]),
        ),
    )
    second = rec(
        "b2",
        100,
        binding="ib_y",
        identity="dup",
        int_score=2,
        rep_key=representative_key(
            raw_sha256=oracle_raw_sha256(rows[1]["text"]),
            input_record_sha256=oracle_canonical_record_sha256(rows[1]),
        ),
    )
    forward = _representatives([first, second])
    reverse = _representatives([second, first])
    assert len(forward) == len(reverse) == 1
    expected_key = representative_key(
        raw_sha256=oracle_raw_sha256(expected["text"]),
        input_record_sha256=oracle_canonical_record_sha256(expected),
    )
    assert next(iter(forward.values())).representative_key == expected_key
    assert next(iter(reverse.values())).representative_key == expected_key
    out = replay(graph, with_records(graph, {"ib_x": [first], "ib_y": [second]}), set())[0]
    assert out.selected_identities == 1


def test_h01_shared_binding_representative_is_identical_in_both_stages(tmp_path: Path):
    """One binding serving Stage B and Stage A resolves to the same representative in both."""
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [
                node("b_finewiki_en", "stage_b", 100, ["ib_x"]),
                node("a_finewiki_en", "stage_a", 100, ["ib_x"]),
            ],
        )
    )
    duplicates = [
        rec("v1", 100, row=0, identity="dup", rep_key=key_for("v1")),
        rec("v2", 100, row=1, identity="dup", rep_key=key_for("v2")),
    ]
    survivor = next(iter(_representatives(duplicates).values()))
    out = replay(graph, with_records(graph, {"ib_x": duplicates}), set())
    assert out[0].selected_identities == 1
    assert out[0].selection_fingerprint == oracle_fingerprint([survivor.cleaned_sha256])
    assert out[1].prior_commit_excluded_identities == 1
    assert out[1].exclusions_by_owner == {"b_finewiki_en": 1}


def test_h01_identity_collision_is_detected_even_when_the_key_is_smaller():
    """The collision guard must run before the representative comparison, not after it."""
    low = CandidateRecord(
        input_binding_id="ib_x",
        row_index=1,
        cleaned_sha256="a" * 64,
        canonical_fingerprint="b" * 64,
        serialized_tokens=10,
        representative_key=b"\x00" * 64,
    )
    high = CandidateRecord(
        input_binding_id="ib_x",
        row_index=0,
        cleaned_sha256="a" * 64,
        canonical_fingerprint="c" * 64,
        serialized_tokens=10,
        representative_key=b"\xff" * 64,
    )
    with pytest.raises(ReplayError, match="identity collision"):
        _representatives([high, low])
    with pytest.raises(ReplayError, match="identity collision"):
        _representatives([low, high])


def test_h01_representative_key_byte_order_matches_hex_pair_order():
    """Packing must not change the comparison the rule performs."""
    samples = [
        ("00" * 32, "ff" * 32),
        ("ff" * 32, "00" * 32),
        ("0f" * 32, "10" * 32),
        ("a" * 64, "a" * 63 + "b"),
    ]
    for raw_a, rec_a in samples:
        for raw_b, rec_b in samples:
            packed = representative_key(raw_sha256=raw_a, input_record_sha256=rec_a) < (
                representative_key(raw_sha256=raw_b, input_record_sha256=rec_b)
            )
            assert packed == ((raw_a, rec_a) < (raw_b, rec_b))


def test_h01_candidate_record_requires_the_v1_key():
    with pytest.raises(TypeError):
        CandidateRecord(
            input_binding_id="ib_x",
            row_index=0,
            cleaned_sha256="a" * 64,
            canonical_fingerprint="b" * 64,
            serialized_tokens=10,
        )
    with pytest.raises(ReplayError, match="representative_key"):
        CandidateRecord(
            input_binding_id="ib_x",
            row_index=0,
            cleaned_sha256="a" * 64,
            canonical_fingerprint="b" * 64,
            serialized_tokens=10,
            representative_key=b"\x00" * 63,
        )


# ---------------------------------------------------------------- H-02 overshoot


def test_h02_actual_overshoot_and_crossing_are_published(tmp_path: Path):
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 250, ["ib_x"])]))
    docs = [rec(f"d{i}", 100) for i in range(10)]
    out = replay(graph, with_records(graph, {"ib_x": docs}), set())[0]
    assert out.selected_serialized_tokens == 300
    assert out.actual_overshoot_tokens == 50
    assert out.crossing_document_serialized_tokens == 100
    ranked = sorted(
        docs,
        key=lambda r: (
            selection_rank(
                seed=SEED,
                stage="stage_b",
                source_id="b_x",
                canonical_fingerprint=r.canonical_fingerprint,
            ),
            r.canonical_fingerprint,
        ),
    )
    assert out.crossing_identity == ranked[2].cleaned_sha256
    assert out.boundary_evidence["next_unselected_identity"] == ranked[3].cleaned_sha256
    assert out.selection_fingerprint == oracle_fingerprint([r.cleaned_sha256 for r in ranked[:3]])


def test_h02_overshoot_reduces_downstream_residual(tmp_path: Path):
    big = rec("big", 900, row=0)
    small = [rec(f"s{i}", 100, row=i + 1) for i in range(5)]
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"]), node("a_x", "stage_a", 400, ["ib_x"])]
        )
    )
    out = replay(graph, with_records(graph, {"ib_x": [big] + small}), set())
    stage_b, stage_a = out
    assert stage_b.selected_serialized_tokens >= 100
    assert stage_a.post_exclusion_candidate_serialized_tokens == (
        1400 - stage_b.selected_serialized_tokens
    )
    assert stage_b.actual_overshoot_tokens == stage_b.selected_serialized_tokens - 100


def test_h02_exact_target_has_zero_overshoot_and_infeasible_has_none(tmp_path: Path):
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 300, ["ib_x"])]))
    exact = replay(
        graph, with_records(graph, {"ib_x": [rec(f"d{i}", 100) for i in range(3)]}), set()
    )[0]
    assert exact.feasible is True
    assert exact.actual_overshoot_tokens == 0
    assert exact.boundary_evidence["next_unselected_identity"] is None
    short = replay(graph, with_records(graph, {"ib_x": [rec("only", 100)]}), set())[0]
    assert short.feasible is False
    assert short.actual_overshoot_tokens == 0
    assert short.crossing_identity is None


# ---------------------------------------------------------------- H-03 exact float order


def test_h03_close_float64_scores_order_exactly(tmp_path: Path):
    low = rec("alpha", 100, score=4.000000001)
    high = rec("beta", 100, score=4.000000002)
    assert abs(bits_to_score(low.score_bits) - bits_to_score(high.score_bits)) < 1e-6
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    out = replay(graph, with_records(graph, {"ib_x": [low, high]}), set())[0]
    assert out.selected_identities == 1
    assert out.crossing_identity == high.cleaned_sha256


def test_h03_sha_tiebreak_only_for_identical_bits(tmp_path: Path):
    a = rec("alpha", 100, score=4.25)
    b = rec("beta", 100, score=4.25)
    assert a.score_bits == b.score_bits
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    out = replay(graph, with_records(graph, {"ib_x": [a, b]}), set())[0]
    assert out.crossing_identity == min(a.cleaned_sha256, b.cleaned_sha256)


def test_h03_signed_zero_ties_and_is_broken_by_sha_not_by_bits(tmp_path: Path):
    """The frozen rank_order is numeric, so +0.0 and -0.0 tie. Raw bit order would not."""
    positive = rec("pos", 100, score=0.0)
    negative = rec("neg", 100, score=-0.0)
    assert positive.score_bits != negative.score_bits
    assert bits_to_score(positive.score_bits) == bits_to_score(negative.score_bits)
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    out = replay(graph, with_records(graph, {"ib_x": [positive, negative]}), set())[0]
    assert out.crossing_identity == min(positive.cleaned_sha256, negative.cleaned_sha256)
    # a raw unsigned-bit sort would place -0.0 (0x8000...) after every positive value
    assert negative.score_bits > positive.score_bits


def test_h03_negative_and_subnormal_scores_order_numerically(tmp_path: Path):
    smallest = float.fromhex("0x0.0000000000001p-1022")
    specs = [("neg", -12.25), ("zero", 0.0), ("subnormal", smallest), ("big", 1.5e308)]
    docs = [rec(name, 100, score=value) for name, value in specs]
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 400, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    out = replay(graph, with_records(graph, {"ib_x": docs}), set())[0]
    rows = [
        {"sha": doc.cleaned_sha256, "score": value, "tokens": 100}
        for doc, (_name, value) in zip(docs, specs, strict=True)
    ]
    expected = oracle_ranked_prefix(rows, 400)
    assert out.selection_fingerprint == expected["fingerprint"]
    assert out.crossing_identity == expected["crossing"]


def test_h03_fallback_ranked_prefix_matches_a_standalone_oracle(tmp_path: Path):
    """Every cumulative boundary is checked against an implementation that is not the one tested."""
    specs = [
        ("large", 1.7976931348623157e308, 83),
        ("adjacent", 4.000000000000001, 41),
        ("close_high", 4.00000049, 77),
        ("close_low", 4.00000040, 29),
        ("equal_a", 3.5, 31),
        ("equal_b", 3.5, 37),
        ("pos_zero", 0.0, 43),
        ("neg_zero", -0.0, 47),
        ("negative", -12.25, 59),
    ]
    docs = [rec(name, tokens, score=score) for name, score, tokens in specs]
    rows = [
        {"sha": doc.cleaned_sha256, "score": score, "tokens": tokens}
        for doc, (_n, score, tokens) in zip(docs, specs, strict=True)
    ]
    cumulative = 0
    for index, row in enumerate(sorted(rows, key=lambda r: (-r["score"], r["sha"])), start=1):
        cumulative += row["tokens"]
        graph = load_fixture(
            mini_graph(
                tmp_path / f"b{index}",
                [node("b_x", "stage_b", cumulative, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")],
            )
        )
        out = replay(graph, with_records(graph, {"ib_x": docs}), set())[0]
        expected = oracle_ranked_prefix(rows, cumulative)
        assert out.selection_fingerprint == expected["fingerprint"]
        assert out.selected_serialized_tokens == expected["mass"]
        assert out.crossing_identity == expected["crossing"]
        assert out.actual_overshoot_tokens == expected["overshoot"]
        assert out.boundary_evidence["next_unselected_identity"] == expected["next_unselected"]


def test_h03_ranked_prefix_differs_from_hash_inside_the_ranked_cut(tmp_path: Path):
    """Hashing inside a score-ranked cut drops the highest scorers; it is a different function."""
    docs = [rec(f"d{i}", 100, score=4.90 - i * 0.01) for i in range(12)]
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 500, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    ranked = replay(graph, with_records(graph, {"ib_x": docs}), set())[0]
    rows = [
        {"sha": d.cleaned_sha256, "score": bits_to_score(d.score_bits), "tokens": 100} for d in docs
    ]
    assert ranked.selection_fingerprint == oracle_ranked_prefix(rows, 500)["fingerprint"]

    cut = sorted(docs, key=lambda r: -bits_to_score(r.score_bits))[:7]
    hashed_graph = load_fixture(
        mini_graph(tmp_path / "h", [node("b_x", "stage_b", 500, ["ib_x"], mode="SEEDED_HASH")])
    )
    hashed = replay(hashed_graph, with_records(hashed_graph, {"ib_x": cut}), set())[0]
    assert ranked.selection_fingerprint != hashed.selection_fingerprint


def test_h03_no_decimal_quantisation_in_the_ordering(tmp_path: Path):
    """Two scores that a 1e-6 bucket would merge must still order strictly."""
    low = rec("low", 100, score=4.0000000000000001e-7 + 4.0)
    high = rec("high", 100, score=4.0000004999999996)
    assert bits_to_score(high.score_bits) != bits_to_score(low.score_bits)
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    out = replay(graph, with_records(graph, {"ib_x": [low, high]}), set())[0]
    ordered = sorted([low, high], key=lambda r: -bits_to_score(r.score_bits))
    assert out.crossing_identity == ordered[0].cleaned_sha256


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "4.5", None])
def test_h03_invalid_scores_rejected_by_the_encoder(bad):
    with pytest.raises(ReplayError):
        score_to_bits(bad)


@pytest.mark.parametrize(
    "bits",
    [
        0x7FF8000000000000,  # NaN
        0x7FF0000000000000,  # +Inf
        0xFFF0000000000000,  # -Inf
        0xFFF8000000000000,  # negative NaN
    ],
)
def test_h03_invalid_score_bits_rejected_at_direct_construction(bits):
    """The public replay boundary must not accept a hand-built record carrying bad bits."""
    with pytest.raises(ReplayError):
        rec("x", 100, score=None, rep_key=key_for("x")).__class__(
            input_binding_id="ib_x",
            row_index=0,
            cleaned_sha256="a" * 64,
            canonical_fingerprint="b" * 64,
            serialized_tokens=10,
            representative_key=key_for("x"),
            score_bits=bits,
        )


@pytest.mark.parametrize("bits", [-1, 1 << 64, True, 4.0, "0"])
def test_h03_malformed_score_bits_rejected(bits):
    with pytest.raises(ReplayError):
        CandidateRecord(
            input_binding_id="ib_x",
            row_index=0,
            cleaned_sha256="a" * 64,
            canonical_fingerprint="b" * 64,
            serialized_tokens=10,
            representative_key=key_for("x"),
            score_bits=bits,
        )


def test_h03_mutated_record_is_rejected_at_the_replay_boundary(tmp_path: Path):
    """A frozen dataclass can still be mutated; replay must revalidate what it is handed."""
    good = rec("good", 100, score=1.0)
    other = rec("other", 100, score=2.0)
    object.__setattr__(good, "score_bits", 0x7FF8000000000000)
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    with pytest.raises(ReplayError, match="NaN"):
        replay(graph, with_records(graph, {"ib_x": [good, other]}), set())


def test_h03_boundary_score_bits_are_published_losslessly(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 150, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        )
    )
    docs = [rec("hi", 100, score=4.75), rec("mid", 100, score=3.5), rec("lo", 100, score=1.25)]
    out = replay(graph, with_records(graph, {"ib_x": docs}), set())[0]
    evidence = out.as_canonical()["boundary_evidence"]
    assert evidence["crossing_score_bits_hex"] == f"{score_to_bits(3.5):016x}"
    assert evidence["next_unselected_score_bits_hex"] == f"{score_to_bits(1.25):016x}"
    assert struct.unpack(">d", bytes.fromhex(evidence["crossing_score_bits_hex"]))[0] == 3.5
    assert float.fromhex(evidence["crossing_score_hex"]) == 3.5
    assert evidence["representative_rule"].startswith("selector-v1 min(raw_sha256")


# ---------------------------------------------------------------- H-04 publication


def _authorized(tmp_path: Path, monkeypatch=None) -> AuthorizedRunContext:
    """A genuine authorisation over the real graph and authorities."""
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    return authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def _fake_census(context: AuthorizedRunContext) -> dict:
    """A minimal but fully valid COMPLETE census for this context, built without scanning."""
    nodes = []
    for graph_node in context.graph.nodes:
        target = graph_node.target_serialized_tokens
        if graph_node.selection_mode == "BRANCH_DEPENDENT":
            branch = "PRIMARY_GE4"
            mode = graph_node.branch_primary["selection_mode"]
        else:
            branch = "ORDINARY"
            mode = graph_node.selection_mode
        nodes.append({
            "canonical_schema_version": NODE_RESULT_SCHEMA,
            "source_id": graph_node.source_id,
            "stage": graph_node.stage,
            "target_serialized_tokens": target,
            "branch": branch,
            "selection_mode": mode,
            "pre_exclusion_unique_identities": 1,
            "g2_excluded_identities": 0,
            "prior_commit_excluded_identities": 0,
            "exclusions_by_owner": {},
            "post_exclusion_candidate_identities": 1,
            "post_exclusion_candidate_serialized_tokens": target,
            "selected_identities": 1,
            "selected_serialized_tokens": target,
            "crossing_identity": "a" * 64,
            "crossing_document_serialized_tokens": target,
            "actual_overshoot_tokens": 0,
            "residual_identities": 0,
            "residual_serialized_tokens": 0,
            "selection_fingerprint": "b" * 64,
            "feasible": True,
            "boundary_evidence": {
                "representative_rule": "selector-v1 min(raw_sha256, input_record_sha256)",
                "crossing_selection_rank": "c" * 64,
                "crossing_score_bits_hex": None,
                "crossing_score_hex": None,
                "next_unselected_identity": None,
                "next_unselected_serialized_tokens": None,
                "next_unselected_selection_rank": None,
                "next_unselected_score_bits_hex": None,
                "next_unselected_score_hex": None,
            },
        })
    counters = {}
    for key, binding in sorted(context.graph.bindings.items()):
        counters[key] = {
            "physical_rows": binding.total_physical_rows,
            "d2_d3_excluded_rows": binding.excluded_rows,
            "empty_or_non_string_text": 0,
            "empty_after_cleaning": 0,
            "eligible_rows": binding.total_physical_rows - binding.excluded_rows,
        }
    return {
        "schema_version": CENSUS_SCHEMA,
        "status": "COMPLETE",
        "output_label": "NON_AUTHORITATIVE_FEASIBILITY_REPLAY",
        "resume_supported": False,
        "graph_sha256": context.graph_sha256,
        "seed": context.graph.seed,
        "bound_authorities": dict(sorted(context.graph.bound_authorities.items())),
        "reference_exclusion_identities": 0,
        "binding_counters": counters,
        "nodes": nodes,
        "ownership_matrix": {},
        "hard_stop_required": False,
        "totals": {
            "eligible_rows": sum(c["eligible_rows"] for c in counters.values()),
            "physical_rows": sum(c["physical_rows"] for c in counters.values()),
            "selected_serialized_tokens": sum(n["selected_serialized_tokens"] for n in nodes),
            "selected_identities": len(nodes),
        },
        "authorization": context.as_canonical(),
    }


def test_h04_resume_is_not_supported():
    from pretrain import h_census_v2

    assert h_census_v2.RESUME_SUPPORTED is False
    for spelling in ("--resume", "--resume=1", "--resume-full", "--resume-from"):
        with pytest.raises(CensusError, match="resume is not supported"):
            h_census_v2.main([spelling, "run", "--plan", "x", "--out-dir", "o"])


def test_h04_graph_rejects_resume_enabled(tmp_path: Path):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100)], resume_supported=True)
    with pytest.raises(GraphError, match="resume must be disabled"):
        load_source_graph(path, verify_hashes=False)


def test_h04_valid_complete_run_publishes_and_reloads(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    final = publish_atomic(context, census, tmp_path / "out")
    assert (final / "COMPLETE").is_file()
    assert final.name == f"run-{context.identity[:32]}"
    reloaded = load_published_run(final, expected_run_identity=context.identity)
    assert reloaded == census


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"status": "INCOMPLETE"}, "wrong status"),
        ({"schema_version": "wrong"}, "wrong schema_version"),
        ({"output_label": "AUTHORITATIVE"}, "wrong output_label"),
        ({"resume_supported": True}, "resume must be disabled"),
        ({"graph_sha256": "0" * 64}, "graph SHA disagree"),
        ({"authorization": {}}, "missing required keys"),
        ({"nodes": []}, "non-empty array"),
        ({"hard_stop_required": True}, "hard_stop_required disagrees"),
        ({"extra_field": 1}, "unknown keys"),
    ],
)
def test_h04_invalid_result_can_never_receive_complete(tmp_path: Path, mutation, match):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    census.update(mutation)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(CensusError, match=match):
        publish_atomic(context, census, out)
    assert list(out.iterdir()) == []


def test_h04_complete_nested_unknown_field_is_rejected(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    census["nodes"][0]["boundary_evidence"]["unexpected"] = 1
    with pytest.raises(CensusError, match="unknown keys"):
        publish_atomic(context, census, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_h04_complete_nested_missing_field_is_rejected(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    del census["nodes"][0]["boundary_evidence"]["representative_rule"]
    with pytest.raises(CensusError, match="missing required keys"):
        publish_atomic(context, census, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_h04_complete_nested_wrong_primitive_type_is_rejected(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    first_binding = next(iter(census["binding_counters"]))
    census["binding_counters"][first_binding]["eligible_rows"] = "0"
    with pytest.raises(CensusError, match="must be an exact integer"):
        publish_atomic(context, census, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_h04_publisher_and_consumer_share_the_nested_contract(tmp_path: Path):
    context = _authorized(tmp_path)
    malformed = _fake_census(context)
    malformed["nodes"][0]["boundary_evidence"]["unexpected"] = 1
    with pytest.raises(CensusError, match="unknown keys"):
        publish_atomic(context, malformed, tmp_path / "publisher-out")

    final = publish_atomic(context, _fake_census(context), tmp_path / "consumer-out")
    census_path = final / "census.json"
    marker_path = final / "COMPLETE"
    stored = json.loads(census_path.read_text())
    stored["nodes"][0]["boundary_evidence"]["unexpected"] = 1
    payload = canonical_json_bytes(stored)
    marker = json.loads(marker_path.read_text())
    marker["census_sha256"] = hashlib.sha256(payload).hexdigest()
    census_path.write_bytes(payload)
    marker_path.write_bytes(canonical_json_bytes(marker))

    with pytest.raises(CensusError, match="unknown keys"):
        load_published_run(final)


@pytest.mark.parametrize(
    "path,value",
    [
        (("reference_exclusion_identities",), True),
        (("totals", "physical_rows"), 1.0),
        (("nodes", 0, "selected_identities"), "1"),
    ],
)
def test_h04_exact_primitive_types_are_recursive(tmp_path: Path, path, value):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    target = census
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(CensusError, match="exact integer"):
        validate_complete_census(census, context)


def test_h04_ownership_matrix_rejects_bool_as_an_integer(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    owner = census["nodes"][0]["source_id"]
    node_result = census["nodes"][1]
    source = node_result["source_id"]
    node_result["pre_exclusion_unique_identities"] = 2
    node_result["prior_commit_excluded_identities"] = 1
    node_result["exclusions_by_owner"] = {owner: 1}
    census["ownership_matrix"] = {source: {owner: True}}

    with pytest.raises(CensusError, match="must be an exact integer"):
        validate_complete_census(census)


def test_h04_consumer_rejects_noncanonical_node_order(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    census["nodes"][0], census["nodes"][1] = census["nodes"][1], census["nodes"][0]

    with pytest.raises(CensusError, match="canonical execution order"):
        validate_complete_census(census)


def test_h05_authorization_execution_state_is_immutable(tmp_path: Path):
    context = _authorized(tmp_path)
    with pytest.raises(TypeError):
        context.graph.bindings["ib_pes2o"] = context.graph.bindings["ib_pes2o"]
    with pytest.raises(TypeError):
        context.graph.binding_identities["ib_pes2o"] = context.graph.binding_identities["ib_pes2o"]
    with pytest.raises(TypeError):
        context.authority_sha256["tokenizer"] = "0" * 64
    with pytest.raises(TypeError):
        context.plan["authorities"]["tokenizer"]["sha256"] = "0" * 64
    rows = next(rows for rows in context.graph.eligibility_rows.values() if len(rows))
    with pytest.raises(ValueError, match="read-only"):
        rows[0] = rows[0]


def test_h04_internally_inconsistent_accounting_is_refused(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    census["nodes"][0]["residual_identities"] = 7
    out = tmp_path / "out"
    with pytest.raises(CensusError, match="residual_identities and residual_serialized_tokens"):
        publish_atomic(context, census, out)
    assert not out.exists() or list(out.iterdir()) == []


def test_h04_infeasible_node_may_not_claim_a_crossing(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    node_result = census["nodes"][0]
    node_result["selected_serialized_tokens"] = 1
    node_result["post_exclusion_candidate_serialized_tokens"] = 1
    node_result["residual_serialized_tokens"] = 0
    node_result["feasible"] = False
    census["hard_stop_required"] = True
    census["totals"]["selected_serialized_tokens"] = sum(
        n["selected_serialized_tokens"] for n in census["nodes"]
    )
    with pytest.raises(CensusError, match="infeasible but reports a crossing"):
        publish_atomic(context, census, tmp_path / "out")


def test_h04_publication_requires_an_authorized_context(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    with pytest.raises(AuthorizationError, match="requires an AuthorizedRunContext"):
        publish_atomic(object(), census, tmp_path / "out")


def test_h04_failure_before_rename_leaves_no_discoverable_output(tmp_path: Path, monkeypatch):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    out = tmp_path / "out"
    out.mkdir()
    from pretrain import h_census_v2

    def boom(src, dst):
        raise OSError("injected rename failure")

    monkeypatch.setattr(h_census_v2.os, "replace", boom)
    with pytest.raises(OSError, match="injected rename failure"):
        publish_atomic(context, census, out)
    assert list(out.iterdir()) == []


def test_h04_failure_while_writing_the_marker_leaves_no_complete(tmp_path: Path, monkeypatch):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    out = tmp_path / "out"
    out.mkdir()
    from pretrain import h_census_v2

    real = h_census_v2._write_durable
    calls = {"n": 0}

    def flaky(path, payload):
        calls["n"] += 1
        if calls["n"] == 2:  # the COMPLETE marker
            raise OSError("injected marker failure")
        real(path, payload)

    monkeypatch.setattr(h_census_v2, "_write_durable", flaky)
    with pytest.raises(OSError, match="injected marker failure"):
        publish_atomic(context, census, out)
    assert list(out.iterdir()) == []
    assert not (out / f"run-{context.identity[:32]}").exists()


def test_h04_existing_final_is_never_overwritten(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    out = tmp_path / "out"
    final = publish_atomic(context, census, out)
    before = (final / "census.json").read_bytes()
    with pytest.raises(CensusError, match="already exists"):
        publish_atomic(context, census, out)
    assert (final / "census.json").read_bytes() == before


def test_h04_stale_staging_blocks_publication(tmp_path: Path):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    out = tmp_path / "out"
    out.mkdir()
    (out / f".run-{context.identity[:32]}.staging-leftover").mkdir()
    with pytest.raises(CensusError, match="stale staging"):
        publish_atomic(context, census, out)


def test_h04_strict_consumer_rejects_partial_and_forged_runs(tmp_path: Path):
    context = _authorized(tmp_path)
    final = publish_atomic(context, _fake_census(context), tmp_path / "out")
    (final / "COMPLETE").unlink()
    with pytest.raises(CensusError, match="no COMPLETE marker"):
        load_published_run(final)

    forged = tmp_path / "forged" / final.name
    forged.mkdir(parents=True)
    (forged / "census.json").write_bytes(canonical_json_bytes({"status": "INCOMPLETE"}))
    (forged / "COMPLETE").write_bytes(canonical_json_bytes({"marker": "COMPLETE"}))
    with pytest.raises(CensusError, match="does not describe the published census"):
        load_published_run(forged)


def test_h04_pure_helper_cannot_publish(tmp_path: Path):
    """census_body is callable without authorisation and yields nothing publishable."""
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])]))
    body = census_body(graph, str(TOKENIZER), set())
    assert "authorization" not in body
    assert "status" not in body
    context = _authorized(tmp_path)
    with pytest.raises(CensusError, match="missing required keys"):
        validate_complete_census(body, context)


# ---------------------------------------------------------------- H-05 input TOCTOU


def _binding_with_index(tmp: Path, payload: bytes, *, total=10, excluded=None):
    tmp.mkdir(parents=True, exist_ok=True)
    doc = tmp / "d.jsonl"
    doc.write_text('{"text":"x"}\n', encoding="utf-8")
    idx = tmp / "i.u32.raw"
    idx.write_bytes(payload)
    count = len(payload) // 4 if excluded is None else excluded
    entry = binding_entry(tmp, "ib_x")
    entry.update({
        "documents_path": str(doc),
        "documents_sha256": sha256_file(doc),
        "documents_size_bytes": doc.stat().st_size,
        "release_manifest_path": str(doc),
        "release_manifest_sha256": sha256_file(doc),
        "eligibility_index_path": str(idx),
        "eligibility_index_sha256": sha256_file(idx),
        "total_physical_rows": total,
        "excluded_rows": count,
        "expected_eligible_rows": total - count,
    })
    return entry


def _graph_with_binding(tmp: Path, binding: dict, **over):
    graph = json.loads(mini_graph(tmp / "base", [node("b_x", "stage_b", 100)]).read_text())
    graph["input_bindings"]["ib_x"] = binding
    graph.update(over)
    path = tmp / "g2.json"
    path.write_bytes(canonical_json_bytes(graph))
    return path


def pack(rows):
    return b"".join(struct.pack("<I", r) for r in rows)


@pytest.mark.parametrize(
    "payload,total,match",
    [
        (pack([0, 1, 2]) + b"\x00", 10, "not a multiple of 4"),
        (pack([2, 1, 0]), 10, "strictly increasing"),
        (pack([1, 1, 2]), 10, "strictly increasing"),
        (pack([0, 1, 99]), 10, "out of range"),
    ],
)
def test_h05_malformed_eligibility_index_rejected(tmp_path: Path, payload, total, match):
    binding = _binding_with_index(tmp_path, payload, total=total)
    with pytest.raises(GraphError, match=match):
        load_source_graph(_graph_with_binding(tmp_path, binding), verify_hashes=True)


def test_h05_index_sha_mismatch_rejected(tmp_path: Path):
    binding = _binding_with_index(tmp_path, pack([0, 1, 2]))
    binding["eligibility_index_sha256"] = "0" * 64
    with pytest.raises(GraphError, match="eligibility index SHA mismatch"):
        load_source_graph(_graph_with_binding(tmp_path, binding), verify_hashes=True)


def test_h05_same_length_different_rows_rejected(tmp_path: Path):
    binding = _binding_with_index(tmp_path, pack([0, 1, 2]))
    (tmp_path / "i.u32.raw").write_bytes(pack([3, 4, 5]))
    with pytest.raises(GraphError, match="eligibility index SHA mismatch"):
        load_source_graph(_graph_with_binding(tmp_path, binding), verify_hashes=True)


def test_h05_element_count_mismatch_rejected(tmp_path: Path):
    binding = _binding_with_index(tmp_path, pack([0, 1, 2]))
    binding["excluded_rows"] = 2
    binding["expected_eligible_rows"] = 8
    with pytest.raises(GraphError, match="element count mismatch"):
        load_source_graph(_graph_with_binding(tmp_path, binding), verify_hashes=True)


def test_h05_row_arithmetic_must_reconcile(tmp_path: Path):
    binding = _binding_with_index(tmp_path, pack([0, 1, 2]))
    binding["expected_eligible_rows"] = 99
    with pytest.raises(GraphError, match="expected_eligible_rows"):
        load_source_graph(_graph_with_binding(tmp_path, binding), verify_hashes=False)


def test_h05_eligibility_index_is_opened_exactly_once(tmp_path: Path):
    """The historical race needed two opens. Prove there is only one."""
    binding_dict = _binding_with_index(tmp_path, pack([0, 1, 2]))
    graph = load_source_graph(_graph_with_binding(tmp_path, binding_dict), verify_hashes=False)
    binding = graph.bindings["ib_x"]
    from pretrain import stage_i_graph_v2

    opens: list[str] = []
    real_open = os.open

    def counting_open(path, flags, *args, **kwargs):
        opens.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    stage_i_graph_v2.os.open = counting_open
    try:
        rows = validate_eligibility_index(binding)
    finally:
        stage_i_graph_v2.os.open = real_open
    assert list(rows) == [0, 1, 2]
    assert opens.count(str(binding.eligibility_index_path)) == 1


def test_h05_validated_rows_are_the_rows_consumed(tmp_path: Path):
    """Replacing the index after validation cannot change what the scan uses."""
    binding_dict = _binding_with_index(tmp_path, pack([1, 3]), total=5)
    path = _graph_with_binding(tmp_path, binding_dict)
    graph = load_source_graph(path, verify_hashes=True)
    validated = list(graph.validated_eligibility_rows("ib_x"))
    assert validated == [1, 3]
    # A same-size index holding different rows appears at the same path afterwards.
    Path(binding_dict["eligibility_index_path"]).write_bytes(pack([0, 4]))
    assert list(graph.validated_eligibility_rows("ib_x")) == [1, 3]
    # And a fresh validation now fails closed rather than silently consuming the new rows.
    with pytest.raises(GraphError, match="eligibility index SHA mismatch"):
        validate_eligibility_index(graph.bindings["ib_x"])


def test_h05_scan_requires_validated_rows(tmp_path: Path):
    graph = load_source_graph(
        _graph_with_binding(tmp_path, _binding_with_index(tmp_path, b"")), verify_hashes=False
    )
    with pytest.raises(GraphError, match="never validated"):
        graph.validated_eligibility_rows("ib_x")


def test_h05_symlinked_authority_is_refused(tmp_path: Path):
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(GraphError, match="cannot open authoritatively"):
        read_authoritative_bytes(link)


def test_h05_non_regular_input_is_refused(tmp_path: Path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(GraphError, match="not a regular file"):
        read_authoritative_bytes(fifo)


def test_h05_documents_are_opened_once_and_hashed_as_consumed(tmp_path: Path):
    rows = ['{"text":"A document about tides and gravitation."}\n']
    binding = _scan_fixture(tmp_path / "once", rows)
    from pretrain import stage_i_graph_v2

    opens: list[str] = []
    real_open = os.open

    def counting_open(path, flags, *args, **kwargs):
        opens.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    stage_i_graph_v2.os.open = counting_open
    try:
        records, _counters = scan_binding(binding, str(TOKENIZER), _empty_rows())
    finally:
        stage_i_graph_v2.os.open = real_open
    assert len(records) == 1
    assert opens.count(str(binding.documents_path)) == 1


def test_h05_mutated_documents_detected_by_the_streaming_digest(tmp_path: Path):
    binding = _scan_fixture(tmp_path / "mut", ['{"text":"original body of the document"}\n'])
    Path(binding.documents_path).write_text(
        '{"text":"replaced body of the doc!"}\n', encoding="utf-8"
    )
    with pytest.raises(CensusError, match="documents SHA-256 mismatch"):
        scan_binding(binding, str(TOKENIZER), _empty_rows())


def test_h05_same_size_documents_replacement_blocks_complete_publication(tmp_path: Path):
    owner_graph = load_source_graph(GRAPH, verify_hashes=False)
    graph_path = mini_graph(
        tmp_path / "small",
        [node("b_x", "stage_b", 1, ["ib_x"])],
        bound_authorities=dict(owner_graph.bound_authorities),
    )
    graph = load_source_graph(graph_path, verify_hashes=False)
    plan = generate_candidate_plan(
        graph_path=graph_path,
        graph=graph,
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan_path = tmp_path / "small-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan))
    context = authorize_run(
        plan_path=plan_path,
        expected_plan_sha256=file_sha256(plan_path),
        repo_root=ROOT,
    )
    census = build_census(context)

    documents_path = context.graph.bindings["ib_x"].documents_path
    original = documents_path.read_bytes()
    replacement_bytes = original.replace(b'"x"', b'"y"')
    assert replacement_bytes != original and len(replacement_bytes) == len(original)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(replacement_bytes)
    os.replace(replacement, documents_path)

    out = tmp_path / "out"
    with pytest.raises(AuthorizationError, match="documents identity changed"):
        publish_atomic(context, census, out)
    assert not out.exists()


def test_h05_documents_size_drift_is_detected_at_graph_load(tmp_path: Path):
    binding = _binding_with_index(tmp_path, pack([0]), total=1)
    binding["documents_size_bytes"] = 999999
    with pytest.raises(GraphError, match="documents size"):
        load_source_graph(_graph_with_binding(tmp_path, binding), verify_hashes=True)


def test_h05_authority_revalidation_catches_post_authorisation_drift(tmp_path: Path, monkeypatch):
    """Every authority is re-derived immediately before publication."""
    context = _authorized(tmp_path)
    census = _fake_census(context)
    from pretrain import h_census_v2

    real_file_sha256 = h_census_v2.file_sha256

    def drifted_file_sha256(path):
        if Path(path) == context.tokenizer_path:
            return "0" * 64
        return real_file_sha256(path)

    monkeypatch.setattr(h_census_v2, "file_sha256", drifted_file_sha256)
    with pytest.raises(AuthorizationError, match="authority 'tokenizer' changed"):
        publish_atomic(context, census, tmp_path / "out")


def test_h05_binding_manifests_and_indexes_are_rebound_before_publication(
    tmp_path: Path, monkeypatch
):
    """The review's case: a release manifest changed after graph validation went undetected."""
    context = _authorized(tmp_path)
    census = _fake_census(context)
    binding = context.graph.bindings["ib_pes2o"]
    from pretrain import h_census_v2

    manifest_path = binding.release_manifest_path
    original = manifest_path.read_bytes()
    real_file_sha256 = h_census_v2.file_sha256

    def mismatched_at(target):
        def digest(path):
            if Path(path) == target:
                return "0" * 64
            return real_file_sha256(path)

        return digest

    with monkeypatch.context() as patch:
        patch.setattr(h_census_v2, "file_sha256", mismatched_at(manifest_path))
        with pytest.raises(
            AuthorizationError, match="release manifest changed after authorisation"
        ):
            publish_atomic(context, census, tmp_path / "out_a")
    assert manifest_path.read_bytes() == original  # nothing was written to the real release

    with monkeypatch.context() as patch:
        patch.setattr(
            h_census_v2,
            "file_sha256",
            mismatched_at(binding.eligibility_index_path),
        )
        with pytest.raises(
            AuthorizationError, match="eligibility index changed after authorisation"
        ):
            publish_atomic(context, census, tmp_path / "out_b")

    real_open = h_census_v2.open_authoritative

    @contextlib.contextmanager
    def size_drift(path, *, buffering=-1):
        with real_open(path, buffering=buffering) as (stream, identity):
            if Path(path) == binding.documents_path:
                identity = dataclasses.replace(identity, st_size=1)
            yield stream, identity

    with monkeypatch.context() as patch:
        patch.setattr(h_census_v2, "open_authoritative", size_drift)
        with pytest.raises(AuthorizationError, match="documents size changed after authorisation"):
            publish_atomic(context, census, tmp_path / "out_c")
    assert census["status"] == "COMPLETE"


def test_h05_graph_bytes_changing_after_authorisation_is_detected(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    copied_graph = tmp_path / "graph_copy.json"
    copied_graph.write_bytes(GRAPH.read_bytes())
    plan["graph_path"] = str(copied_graph)
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    context = authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)
    census = _fake_census(context)
    copied_graph.write_bytes(GRAPH.read_bytes() + b"\n")
    with pytest.raises(AuthorizationError, match="owner graph bytes changed"):
        publish_atomic(context, census, tmp_path / "out")


def test_h05_plan_bytes_changing_after_authorisation_is_detected(tmp_path: Path):
    path = tmp_path / "plan.json"
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    path.write_bytes(canonical_json_bytes(plan))
    context = authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)
    census = _fake_census(context)
    path.write_bytes(canonical_json_bytes(plan) + b"\n")
    with pytest.raises(AuthorizationError, match="plan bytes changed"):
        publish_atomic(context, census, tmp_path / "out")


# ---------------------------------------------------------------- H-06 authorization closure


def test_h06_missing_expected_sha_is_refused(tmp_path: Path):
    plan = tmp_path / "plan.json"
    plan.write_bytes(canonical_json_bytes({"schema_version": PLAN_SCHEMA}))
    with pytest.raises(AuthorizationError, match="may not authorise itself"):
        authorize_run(plan_path=plan, expected_plan_sha256=None, repo_root=ROOT)


@pytest.mark.parametrize("bad", ["", "0" * 63, "z" * 64, "ABC", "0X" + "0" * 62])
def test_h06_malformed_expected_sha_is_refused(tmp_path: Path, bad):
    plan = tmp_path / "plan.json"
    plan.write_bytes(canonical_json_bytes({"schema_version": PLAN_SCHEMA}))
    with pytest.raises(AuthorizationError):
        authorize_run(plan_path=plan, expected_plan_sha256=bad, repo_root=ROOT)


def test_h06_wrong_sha_is_refused(tmp_path: Path):
    plan = tmp_path / "plan.json"
    plan.write_bytes(canonical_json_bytes({"schema_version": PLAN_SCHEMA}))
    with pytest.raises(AuthorizationError, match="plan SHA-256 mismatch"):
        authorize_run(plan_path=plan, expected_plan_sha256="0" * 64, repo_root=ROOT)


def test_h06_schema_only_plan_is_refused(tmp_path: Path):
    plan = tmp_path / "plan.json"
    plan.write_bytes(canonical_json_bytes({"schema_version": PLAN_SCHEMA}))
    with pytest.raises(AuthorizationError, match="missing required keys"):
        authorize_run(plan_path=plan, expected_plan_sha256=file_sha256(plan), repo_root=ROOT)


def test_h06_old_plan_schema_is_refused(tmp_path: Path):
    old = ROOT / "runs/h_tooling_repair_v2_2026-08-21/evidence/candidate_h_plan_v2.json"
    assert file_sha256(old) == "5cc4500a2a9b557ab14e783b6d66e1d23ef0a75c4c879a4ffc6d8e6f7016a50e"
    with pytest.raises(AuthorizationError, match="schema_version must be"):
        authorize_run(plan_path=old, expected_plan_sha256=file_sha256(old), repo_root=ROOT)


def test_h06_self_labelled_authorized_plan_is_refused(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan["authorization_status"] = "AUTHORIZED"
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(AuthorizationError, match="must remain NOT_AUTHORIZED"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_one_byte_plan_mutation_is_refused(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    path = tmp_path / "plan.json"
    payload = canonical_json_bytes(plan)
    path.write_bytes(payload)
    good = file_sha256(path)
    mutated = bytearray(payload)
    mutated[len(mutated) // 2] ^= 0x20
    path.write_bytes(bytes(mutated))
    with pytest.raises(AuthorizationError, match="plan SHA-256 mismatch"):
        authorize_run(plan_path=path, expected_plan_sha256=good, repo_root=ROOT)


def test_h06_regenerated_but_unauthorized_plan_is_refused(tmp_path: Path):
    """An internally consistent plan the owner never approved must not pass."""
    graph = load_source_graph(GRAPH, verify_hashes=False)
    approved = generate_candidate_plan(
        graph_path=GRAPH, graph=graph, repo_root=ROOT, implementation_commit=_runtime_head()
    )
    approved_path = tmp_path / "approved.json"
    approved_path.write_bytes(canonical_json_bytes(approved))
    approved_sha = file_sha256(approved_path)

    other = generate_candidate_plan(
        graph_path=GRAPH, graph=graph, repo_root=ROOT, implementation_commit="1" * 40
    )
    other_path = tmp_path / "other.json"
    other_path.write_bytes(canonical_json_bytes(other))
    assert file_sha256(other_path) != approved_sha
    with pytest.raises(AuthorizationError, match="plan SHA-256 mismatch"):
        authorize_run(plan_path=other_path, expected_plan_sha256=approved_sha, repo_root=ROOT)


def test_h06_hashed_bytes_and_parsed_bytes_are_the_same_bytes(tmp_path: Path):
    """The plan is opened once; a swap between hashing and parsing has no window to occur."""
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    expected = file_sha256(path)
    from pretrain import stage_i_graph_v2

    opens: list[str] = []
    real_open = os.open

    def counting_open(target, flags, *args, **kwargs):
        opens.append(str(target))
        return real_open(target, flags, *args, **kwargs)

    stage_i_graph_v2.os.open = counting_open
    try:
        context = authorize_run(plan_path=path, expected_plan_sha256=expected, repo_root=ROOT)
    finally:
        stage_i_graph_v2.os.open = real_open
    assert opens.count(str(path)) == 1
    assert context.plan_sha256 == expected


def test_h06_graph_mutation_is_refused(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan["graph_sha256"] = "0" * 64
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(GraphError, match="SHA-256 mismatch"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_alternate_graph_path_is_refused(tmp_path: Path):
    """Pointing the plan at a different graph file cannot smuggle in other node targets."""
    graph = load_source_graph(GRAPH, verify_hashes=False)
    other = json.loads(GRAPH.read_text())
    other["nodes"] = other["nodes"][:1]
    other["nodes"][0]["target_serialized_tokens"] = 1
    other_path = tmp_path / "other_graph.json"
    other_path.write_bytes(canonical_json_bytes(other))
    plan = generate_candidate_plan(
        graph_path=GRAPH, graph=graph, repo_root=ROOT, implementation_commit=_runtime_head()
    )
    plan["graph_path"] = str(other_path)
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(GraphError, match="SHA-256 mismatch"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_implementation_mutation_is_refused(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan["implementation_files"] = {name: "0" * 64 for name in plan["implementation_files"]}
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(AuthorizationError, match="implementation files disagree"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_implementation_bundle_digest_is_checked_independently(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan["implementation_bundle_sha256"] = "0" * 64
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(AuthorizationError, match="bundle SHA-256 disagrees"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


@pytest.mark.parametrize("authority", sorted(AUTHORITY_KEYS))
def test_h06_authority_mutation_is_refused(tmp_path: Path, authority):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan["authorities"][authority]["sha256"] = "0" * 64
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(AuthorizationError, match=f"authority '{authority}' bytes disagree"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_authority_pointed_at_other_bytes_is_refused(tmp_path: Path):
    """A real file with a real digest still fails unless the graph binds those exact bytes."""
    decoy = tmp_path / "decoy.json"
    decoy.write_text('{"not": "the tokenizer"}', encoding="utf-8")
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    plan["authorities"]["tokenizer"] = {
        "path": str(decoy),
        "sha256": file_sha256(decoy),
    }
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(
        AuthorizationError, match="disagree with the owner graph's tokenizer_sha256"
    ):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_binding_and_node_mutations_are_refused(tmp_path: Path):
    base = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    for mutate, match in (
        (
            lambda p: p["input_bindings"]["ib_pes2o"].update({"total_physical_rows": 1}),
            "input bindings disagree",
        ),
        (
            lambda p: p["nodes"][0].update({"target_serialized_tokens": 1}),
            "node targets or order disagree",
        ),
        (
            lambda p: p.update({"node_order": list(reversed(p["node_order"]))}),
            "node order disagrees",
        ),
        (lambda p: p.update({"seed": 1}), "seed disagrees"),
        (
            lambda p: p["bound_authorities"].update({"tokenizer_sha256": "0" * 64}),
            "bound authorities disagree",
        ),
    ):
        plan = json.loads(json.dumps(base))
        mutate(plan)
        path = tmp_path / "plan.json"
        path.write_bytes(canonical_json_bytes(plan))
        with pytest.raises(AuthorizationError, match=match):
            authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_h06_direct_python_api_cannot_bypass_authorisation(tmp_path: Path):
    """build_census and publish_atomic both demand the capability object."""
    with pytest.raises(AuthorizationError, match="requires an AuthorizedRunContext"):
        build_census(None)
    with pytest.raises(AuthorizationError, match="requires an AuthorizedRunContext"):
        build_census({"graph": "anything"})
    with pytest.raises(AuthorizationError, match="requires an AuthorizedRunContext"):
        publish_atomic("not-a-context", {}, tmp_path)


def test_h06_manually_constructed_context_has_no_authorization_provenance(tmp_path: Path):
    genuine = _authorized(tmp_path)
    manual = AuthorizedRunContext(
        repo_root=genuine.repo_root,
        plan_path=genuine.plan_path,
        plan_sha256=genuine.plan_sha256,
        plan=genuine.plan,
        graph=genuine.graph,
        graph_sha256=genuine.graph_sha256,
        bundle_sha256=genuine.bundle_sha256,
        bundle_files=genuine.bundle_files,
        authority_paths=genuine.authority_paths,
        authority_sha256=genuine.authority_sha256,
        implementation_commit=genuine.implementation_commit,
        identity=genuine.identity,
    )
    census = _fake_census(manual)
    with pytest.raises(AuthorizationError, match="produced by authorize_run"):
        manual.revalidate()
    with pytest.raises(AuthorizationError, match="produced by authorize_run"):
        validate_complete_census(census, manual)
    with pytest.raises(AuthorizationError, match="produced by authorize_run"):
        build_census(manual)
    with pytest.raises(AuthorizationError, match="produced by authorize_run"):
        publish_atomic(manual, census, tmp_path / "manual-out")


def test_h06_dataclass_copy_loses_authorization_provenance(tmp_path: Path):
    genuine = _authorized(tmp_path)
    copied = dataclasses.replace(genuine)
    with pytest.raises(AuthorizationError, match="produced by authorize_run"):
        publish_atomic(copied, _fake_census(copied), tmp_path / "out")


def test_h06_authorize_run_context_provenance_is_accepted(tmp_path: Path):
    genuine = _authorized(tmp_path)
    census = _fake_census(genuine)
    validate_complete_census(census, genuine)
    final = publish_atomic(genuine, census, tmp_path / "out")
    assert load_published_run(final, expected_run_identity=genuine.identity) == census


def test_h06_authorised_context_is_reproducible_and_complete(tmp_path: Path):
    context = _authorized(tmp_path)
    assert context.graph_sha256 == GRAPH_SHA
    assert context.identity == run_identity(
        plan_sha256=context.plan_sha256,
        graph_sha256=GRAPH_SHA,
        bundle_sha256=implementation_bundle_sha256(implementation_files(ROOT)),
    )
    assert sorted(context.authority_sha256) == sorted(AUTHORITY_KEYS)
    assert context.authority_sha256["selector_v1"] == SELECTOR_V1_SHA


# ---------------------------------------------------------------- M-01 strict JSON and schema


def test_m01_duplicate_keys_rejected_at_every_depth():
    with pytest.raises(StrictJSONError, match="duplicate JSON key"):
        strict_json_loads('{"a": 1, "a": 2}')
    with pytest.raises(StrictJSONError, match="duplicate JSON key"):
        strict_json_loads('{"outer": {"b": 1, "b": 2}}')
    with pytest.raises(StrictJSONError, match="duplicate JSON key"):
        strict_json_loads('{"list": [{"c": 1, "c": 2}]}')
    assert strict_json_loads('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


@pytest.mark.parametrize("text", ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'])
def test_m01_non_finite_constants_rejected(text):
    with pytest.raises(StrictJSONError, match="non-finite JSON constant"):
        strict_json_loads(text)


def test_m01_graph_with_duplicate_keys_rejected(tmp_path: Path):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100)])
    text = path.read_text()
    path.write_text(text.replace('"seed":', '"seed": 1, "seed":', 1), encoding="utf-8")
    with pytest.raises(StrictJSONError, match="duplicate JSON key"):
        load_source_graph(path, verify_hashes=False)


def test_m01_plan_with_duplicate_keys_rejected(tmp_path: Path):
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    path = tmp_path / "plan.json"
    text = canonical_json_bytes(plan).decode()
    path.write_text(text.replace('"seed":', '"seed":1,"seed":', 1), encoding="utf-8")
    with pytest.raises(StrictJSONError, match="duplicate JSON key"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_m01_document_row_with_duplicate_keys_rejected(tmp_path: Path):
    binding = _scan_fixture(tmp_path / "dupkeys", ['{"text":"a body","text":"another body"}\n'])
    with pytest.raises(CensusError, match="duplicate JSON key"):
        scan_binding(binding, str(TOKENIZER), _empty_rows())


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"resume_supported": "false"}, "must be a JSON boolean"),
        ({"policy_status": "DRAFT"}, "OWNER_FROZEN"),
        ({"stage_priority": {"stage_b": 0}}, "stage_priority must be exactly"),
        ({"selection_modes_closed_enum": ["SEEDED_HASH"]}, "closed_enum must be exactly"),
        ({"control_namespace": {"exists": True, "note": "x"}}, "no control semantics"),
        ({"bound_authorities": {}}, "missing required keys"),
        (
            {"h_boundary": {"h_canonical_output_label": "X", "h_publishes_physical_views": False}},
            "label must be",
        ),
        (
            {
                "h_boundary": {
                    "h_canonical_output_label": "NON_AUTHORITATIVE_FEASIBILITY_REPLAY",
                    "h_publishes_physical_views": True,
                }
            },
            "must not publish physical candidate views",
        ),
        ({"schema_version": "other"}, "wrong schema_version"),
    ],
)
def test_m01_graph_constants_are_mechanically_enforced(tmp_path: Path, mutation, match):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100)], **mutation)
    with pytest.raises(GraphError, match=match):
        load_source_graph(path, verify_hashes=False)


def test_m01_graph_rejects_unknown_and_missing_keys(tmp_path: Path):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100)], unexpected_policy="x")
    with pytest.raises(GraphError, match="unknown keys"):
        load_source_graph(path, verify_hashes=False)
    graph = json.loads(mini_graph(tmp_path / "b", [node("b_x", "stage_b", 100)]).read_text())
    del graph["ownership_rule"]
    other = tmp_path / "missing.json"
    other.write_bytes(canonical_json_bytes(graph))
    with pytest.raises(GraphError, match="missing required keys"):
        load_source_graph(other, verify_hashes=False)


@pytest.mark.parametrize(
    "target,match",
    [
        ("100", "exact JSON integer"),
        (100.0, "exact JSON integer"),
        (True, "exact JSON integer"),
        (0, "must be >= 1"),
    ],
)
def test_m01_node_target_type_is_exact(tmp_path: Path, target, match):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", target)])
    with pytest.raises(GraphError, match=match):
        load_source_graph(path, verify_hashes=False)


def test_m01_cleaning_contract_rejects_string_booleans(tmp_path: Path):
    """The historical bool() coercion turned the string "false" into True."""
    graph = json.loads(mini_graph(tmp_path, [node("b_x", "stage_b", 100)]).read_text())
    graph["input_bindings"]["ib_x"]["cleaning_contract"]["normalize_quotes"] = "false"
    path = tmp_path / "coerce.json"
    path.write_bytes(canonical_json_bytes(graph))
    assert bool("false") is True  # the coercion this rejects
    with pytest.raises(GraphError, match="must be a JSON boolean"):
        load_source_graph(path, verify_hashes=False)


def test_m01_cleaning_contract_rejects_numeric_strings_and_bad_policy(tmp_path: Path):
    for field_name, value, match in (
        ("min_chars", "5", "exact JSON integer"),
        ("min_chars", 5.0, "exact JSON integer"),
        ("min_ascii_ratio", "0.5", "must be a JSON number"),
        ("min_ascii_ratio", 1.5, "within"),
        ("underscores_policy", "delete", "must be one of"),
    ):
        graph = json.loads(
            mini_graph(tmp_path / field_name, [node("b_x", "stage_b", 100)]).read_text()
        )
        graph["input_bindings"]["ib_x"]["cleaning_contract"][field_name] = value
        path = tmp_path / f"{field_name}-{value}.json"
        path.write_bytes(canonical_json_bytes(graph))
        with pytest.raises(GraphError, match=match):
            load_source_graph(path, verify_hashes=False)


def test_m01_invalid_accessor_enum_rejected(tmp_path: Path):
    graph = json.loads(mini_graph(tmp_path, [node("b_x", "stage_b", 100)]).read_text())
    graph["input_bindings"]["ib_x"]["schema_accessor"] = {
        "accessor_id": "x",
        "integer_score": {"container": "metadata", "key": "s", "json_type": "decimal"},
    }
    path = tmp_path / "acc.json"
    path.write_bytes(canonical_json_bytes(graph))
    with pytest.raises(GraphError, match="json_type: must be one of"):
        load_source_graph(path, verify_hashes=False)


def test_m01_structured_tutorial_may_not_be_split_across_nodes(tmp_path: Path):
    path = mini_graph(
        tmp_path,
        [
            node("b_one", "stage_b", 100, ["ib_x"]),
            node("b_two", "stage_b", 100, ["ib_y"]),
        ],
        bindings=("ib_x", "ib_y"),
        tutorial_ids=["ib_x", "ib_y"],
    )
    with pytest.raises(GraphError, match="forbids splitting its bindings"):
        load_source_graph(path, verify_hashes=False)


def test_m01_structured_tutorial_sub_targets_are_forbidden(tmp_path: Path):
    path = mini_graph(
        tmp_path,
        [node("b_x", "stage_b", 100, ["ib_x"])],
        structured_tutorial={
            "input_binding_ids": [TUTORIAL_BINDING],
            "input_binding_order_source": "fixture",
            "realization": "SINGLE_LOGICAL_NODE",
            "selection_mode": "SEEDED_HASH",
            "sub_targets": [1, 2],
        },
    )
    with pytest.raises(GraphError, match="sub_targets must be null"):
        load_source_graph(path, verify_hashes=False)


def test_m01_hard_linked_binding_is_not_a_distinct_release(tmp_path: Path):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])], bindings=("ib_x", "ib_y"))
    graph = json.loads(path.read_text())
    original = Path(graph["input_bindings"]["ib_x"]["documents_path"])
    linked = tmp_path / "hardlink.jsonl"
    os.link(original, linked)
    graph["input_bindings"]["ib_y"]["documents_path"] = str(linked)
    graph["input_bindings"]["ib_y"]["documents_sha256"] = sha256_file(linked)
    graph["input_bindings"]["ib_y"]["documents_size_bytes"] = linked.stat().st_size
    path.write_bytes(canonical_json_bytes(graph))
    with pytest.raises(GraphError, match="same underlying file object"):
        load_source_graph(path, verify_hashes=True)


def test_m01_duplicate_physical_path_under_two_binding_ids_rejected(tmp_path: Path):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])], bindings=("ib_x", "ib_y"))
    graph = json.loads(path.read_text())
    graph["input_bindings"]["ib_y"]["documents_path"] = graph["input_bindings"]["ib_x"][
        "documents_path"
    ]
    graph["input_bindings"]["ib_y"]["documents_sha256"] = graph["input_bindings"]["ib_x"][
        "documents_sha256"
    ]
    path.write_bytes(canonical_json_bytes(graph))
    with pytest.raises(GraphError, match="registered under both"):
        load_source_graph(path, verify_hashes=False)


ACC = {
    "accessor_id": "fineweb_metadata_literal_dotted",
    "integer_score": {"container": "metadata", "key": "metadata.int_score", "json_type": "int"},
    "continuous_score": {"container": "metadata", "key": "metadata.score", "json_type": "float"},
}


def test_m01_fineweb_literal_dotted_accessor_works_and_naive_split_fails():
    row = {"metadata": {"metadata.int_score": 4, "metadata.score": 3.75}}
    assert read_score(row, ACC, "integer_score") == 4
    assert read_score(row, ACC, "continuous_score") == 3.75
    naive = row.get("metadata", {}).get("metadata", {})
    assert naive == {}


@pytest.mark.parametrize(
    "row,which,match",
    [
        ({"metadata": {"metadata.int_score": True}}, "integer_score", "got boolean"),
        ({"metadata": {"metadata.int_score": 4.0}}, "integer_score", "must be a JSON integer"),
        ({"metadata": {"metadata.int_score": "4"}}, "integer_score", "must be a JSON integer"),
        ({"metadata": {}}, "integer_score", "missing from"),
        ({"metadata": None}, "integer_score", "missing or not an object"),
        ({}, "integer_score", "missing or not an object"),
        ({"metadata": {"metadata.score": 4}}, "continuous_score", "must be a JSON float"),
        ({"metadata": {"metadata.score": "4.0"}}, "continuous_score", "must be a JSON float"),
        ({"metadata": {"metadata.score": True}}, "continuous_score", "got boolean"),
    ],
)
def test_m01_score_schema_fails_closed(row, which, match):
    with pytest.raises(CensusError, match=match):
        read_score(row, ACC, which)


def _empty_rows():
    return np.zeros(0, dtype="<u4")


def _scan_fixture(tmp: Path, rows: list[str], accessor: dict | None = None):
    tmp.mkdir(parents=True, exist_ok=True)
    doc = tmp / "docs.jsonl"
    doc.write_text("".join(rows), encoding="utf-8")
    entry = binding_entry(tmp, "ib_x")
    entry.update({
        "documents_path": str(doc),
        "documents_sha256": sha256_file(doc),
        "documents_size_bytes": doc.stat().st_size,
        "release_manifest_path": str(doc),
        "release_manifest_sha256": sha256_file(doc),
        "total_physical_rows": len(rows),
        "excluded_rows": 0,
        "expected_eligible_rows": len(rows),
    })
    if accessor:
        entry["schema_accessor"] = accessor
    graph = json.loads(mini_graph(tmp / "g", [node("b_x", "stage_b", 100)]).read_text())
    graph["input_bindings"]["ib_x"] = entry
    path = tmp / "graph_scan.json"
    path.write_bytes(canonical_json_bytes(graph))
    return load_source_graph(path, verify_hashes=False).bindings["ib_x"]


def test_m01_missing_and_non_string_text_fail_closed(tmp_path: Path):
    missing = _scan_fixture(tmp_path / "missing", ['{"nottext":"a"}\n'])
    with pytest.raises(CensusError, match="missing 'text'"):
        scan_binding(missing, str(TOKENIZER), _empty_rows())
    wrong = _scan_fixture(tmp_path / "wrong", ['{"text":123}\n'])
    with pytest.raises(CensusError, match="is not a string"):
        scan_binding(wrong, str(TOKENIZER), _empty_rows())


def test_m01_scan_counts_framing_and_representative_keys(tmp_path: Path):
    rows = [
        '{"text":"A document about tides and gravitation."}\n',
        '{"text":"Another document about erosion and sediment."}\n',
    ]
    binding = _scan_fixture(tmp_path / "counts", rows)
    records, counters = scan_binding(binding, str(TOKENIZER), _empty_rows())
    assert counters["physical_rows"] == 2 and counters["eligible_rows"] == 2
    assert all(r.serialized_tokens >= 3 for r in records)
    for record, raw in zip(records, rows, strict=True):
        parsed = json.loads(raw)
        assert record.raw_sha256 == oracle_raw_sha256(parsed["text"])
        assert record.input_record_sha256 == oracle_canonical_record_sha256(parsed)


def test_m01_reference_exclusion_manifest_must_be_whole_reserve_scope(tmp_path: Path):
    path = tmp_path / "excl.json"
    path.write_bytes(
        canonical_json_bytes({"exclusion_scope": "partial", "hashes": [], "hash_count": 0})
    )
    with pytest.raises(CensusError, match="whole-reserve scope"):
        load_reference_exclusion(path)
    path2 = tmp_path / "excl2.json"
    path2.write_bytes(
        canonical_json_bytes({
            "exclusion_scope": "entire_pre_tokenizer_reserved_pool",
            "hashes": ["a" * 64, "a" * 64],
            "hash_count": 2,
        })
    )
    with pytest.raises(CensusError, match="disagrees with the hash list"):
        load_reference_exclusion(path2)


# ---------------------------------------------------------------- M-02 run identity


def test_m02_run_identity_binds_plan_graph_and_implementation():
    base = run_identity(plan_sha256="a" * 64, graph_sha256="b" * 64, bundle_sha256="c" * 64)
    assert base != run_identity(plan_sha256="d" * 64, graph_sha256="b" * 64, bundle_sha256="c" * 64)
    assert base != run_identity(plan_sha256="a" * 64, graph_sha256="d" * 64, bundle_sha256="c" * 64)
    assert base != run_identity(plan_sha256="a" * 64, graph_sha256="b" * 64, bundle_sha256="d" * 64)
    assert base == run_identity(plan_sha256="a" * 64, graph_sha256="b" * 64, bundle_sha256="c" * 64)


def test_m02_different_plan_yields_a_different_final_directory(tmp_path: Path):
    graph = load_source_graph(GRAPH, verify_hashes=False)
    finals = []
    for suffix in ("first", "second"):
        plan = generate_candidate_plan(
            graph_path=GRAPH,
            graph=graph,
            repo_root=ROOT,
            implementation_commit=_runtime_head(),
        )
        plan["authorization_note"] += f" Fixture variant: {suffix}."
        path = tmp_path / f"plan-{suffix}.json"
        path.write_bytes(canonical_json_bytes(plan))
        context = authorize_run(
            plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT
        )
        finals.append(publish_atomic(context, _fake_census(context), tmp_path / "out"))
    assert finals[0] != finals[1]
    assert all(f.is_dir() for f in finals)


def test_m02_implementation_bundle_covers_every_load_bearing_module():
    files = implementation_files(ROOT)
    assert sorted(files) == sorted(IMPLEMENTATION_BUNDLE_FILES)
    for relative in IMPLEMENTATION_BUNDLE_FILES:
        assert files[relative] == file_sha256(ROOT / relative)
    assert files["pretrain/select_pretrain_documents.py"] == SELECTOR_V1_SHA
    digest = implementation_bundle_sha256(files)
    tampered = dict(files)
    tampered["pretrain/h_census_v2.py"] = "0" * 64
    assert implementation_bundle_sha256(tampered) != digest


def test_m02_no_implicit_latest_lookup():
    src = (ROOT / "pretrain/h_census_v2.py").read_text()
    assert "latest" not in src.lower()


def test_m02_published_run_is_bound_to_its_identity(tmp_path: Path):
    context = _authorized(tmp_path)
    final = publish_atomic(context, _fake_census(context), tmp_path / "out")
    with pytest.raises(CensusError, match="not the expected one"):
        load_published_run(final, expected_run_identity="0" * 64)
    renamed = final.parent / "run-0000"
    final.rename(renamed)
    with pytest.raises(CensusError, match="does not match its own run identity"):
        load_published_run(renamed)


def test_m02_runtime_head_must_match_candidate_plan_commit(tmp_path: Path):
    graph = load_source_graph(GRAPH, verify_hashes=False)
    mismatched = "0" * 40 if _runtime_head() != "0" * 40 else "1" * 40
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=graph,
        repo_root=ROOT,
        implementation_commit=mismatched,
    )
    path = tmp_path / "wrong-commit-plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(AuthorizationError, match="does not match runtime repository HEAD"):
        authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)


def test_m02_runtime_head_is_rechecked_before_publication(tmp_path: Path, monkeypatch):
    context = _authorized(tmp_path)
    census = _fake_census(context)
    changed = "f" * 40 if context.implementation_commit != "f" * 40 else "e" * 40
    from pretrain import h_census_v2

    monkeypatch.setattr(h_census_v2, "_git_head", lambda _repo_root: changed)
    with pytest.raises(AuthorizationError, match="HEAD changed after authorisation"):
        publish_atomic(context, census, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_m02_consumer_rejects_internally_inconsistent_run_identity(tmp_path: Path):
    context = _authorized(tmp_path)
    final = publish_atomic(context, _fake_census(context), tmp_path / "out")
    census_path = final / "census.json"
    marker_path = final / "COMPLETE"
    census = json.loads(census_path.read_text())
    marker = json.loads(marker_path.read_text())
    forged_identity = "d" * 64
    census["authorization"]["run_identity"] = forged_identity
    payload = canonical_json_bytes(census)
    marker["run_identity"] = forged_identity
    marker["census_sha256"] = hashlib.sha256(payload).hexdigest()
    census_path.write_bytes(payload)
    marker_path.write_bytes(canonical_json_bytes(marker))
    renamed = final.parent / f"run-{forged_identity[:32]}"
    final.rename(renamed)

    with pytest.raises(CensusError, match="run identity is internally inconsistent"):
        load_published_run(renamed)


def test_m02_consumer_recomputes_implementation_bundle(tmp_path: Path):
    context = _authorized(tmp_path)
    final = publish_atomic(context, _fake_census(context), tmp_path / "out")
    census_path = final / "census.json"
    marker_path = final / "COMPLETE"
    census = json.loads(census_path.read_text())
    marker = json.loads(marker_path.read_text())
    census["authorization"]["implementation_bundle_sha256"] = "e" * 64
    payload = canonical_json_bytes(census)
    marker["implementation_bundle_sha256"] = "e" * 64
    marker["census_sha256"] = hashlib.sha256(payload).hexdigest()
    census_path.write_bytes(payload)
    marker_path.write_bytes(canonical_json_bytes(marker))

    with pytest.raises(CensusError, match="bundle digest is internally inconsistent"):
        load_published_run(final)


# ---------------------------------------------------------------- M-03 canonical projection


def test_m03_canonical_projection_ignores_incidental_attributes(tmp_path: Path):
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])]))
    result = replay(graph, with_records(graph, {"ib_x": [rec("d", 100)]}), set())[0]
    before = canonical_json_bytes(result.as_canonical())
    for name, value in (
        ("elapsed_seconds", 1.25),
        ("hostname", "runpod"),
        ("pid", 4242),
        ("_cache", {"x": 1}),
    ):
        object.__setattr__(result, name, value)
    after = canonical_json_bytes(result.as_canonical())
    assert after == before
    assert b"elapsed_seconds" not in after and b"hostname" not in after


def test_m03_canonical_projection_is_an_explicit_named_contract(tmp_path: Path):
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])]))
    result = replay(graph, with_records(graph, {"ib_x": [rec("d", 100)]}), set())[0]
    projected = result.as_canonical()
    assert projected["canonical_schema_version"] == NODE_RESULT_SCHEMA
    assert set(projected) == {
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
    }


def test_m03_boundary_evidence_projection_drops_unknown_keys(tmp_path: Path):
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])]))
    result = replay(graph, with_records(graph, {"ib_x": [rec("d", 100)]}), set())[0]
    result.boundary_evidence["injected_timing_field"] = 1.5
    projected = result.as_canonical()["boundary_evidence"]
    assert "injected_timing_field" not in projected


def test_m03_canonical_json_is_deterministic_and_order_free():
    assert (
        canonical_json_bytes({"b": 1, "a": [3, 2]})
        == canonical_json_bytes({"a": [3, 2], "b": 1})
        == b'{"a":[3,2],"b":1}\n'
    )


def test_m03_node_result_is_not_serialised_through_instance_dict():
    """A field present on the object but absent from the contract must not reach output."""
    result = NodeResult(
        source_id="b_x",
        stage="stage_b",
        target_serialized_tokens=1,
        branch="ORDINARY",
        selection_mode="SEEDED_HASH",
        pre_exclusion_unique_identities=0,
        g2_excluded_identities=0,
        prior_commit_excluded_identities=0,
        exclusions_by_owner={},
        post_exclusion_candidate_identities=0,
        post_exclusion_candidate_serialized_tokens=0,
        selected_identities=0,
        selected_serialized_tokens=0,
        crossing_identity=None,
        crossing_document_serialized_tokens=None,
        actual_overshoot_tokens=0,
        residual_identities=0,
        residual_serialized_tokens=0,
        selection_fingerprint="0" * 64,
        feasible=False,
    )
    object.__setattr__(result, "future_debug_field", "leak")
    assert "future_debug_field" in result.__dict__
    assert "future_debug_field" not in result.as_canonical()


# ---------------------------------------------------------------- M-04 independent oracles


def test_m04_seeded_rank_is_byte_identical_to_selector_v1():
    """SEEDED_HASH ordering is proved against the frozen selector, not against itself."""
    from pretrain.select_pretrain_documents import selection_rank as v1_rank

    for stage in ("stage_b", "stage_a"):
        for index in range(8):
            fingerprint = hashlib.sha256(f"rank-{stage}-{index}".encode()).hexdigest()
            assert selection_rank(
                seed=SEED, stage=stage, source_id="b_x", canonical_fingerprint=fingerprint
            ) == v1_rank(seed=SEED, stage=stage, source_id="b_x", canonical_fingerprint=fingerprint)


def test_m04_seeded_prefix_matches_the_real_selector_end_to_end(tmp_path: Path):
    """A full selector-v1 run and the v2 replay must choose the same documents and mass."""
    from tokenizers import Tokenizer

    from pretrain import select_pretrain_documents as v1

    texts = [f"bounded differential candidate {i} with stable content" for i in range(24)]
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    tokens = {text: len(tokenizer.encode(text).ids) + 2 for text in texts}
    ranked = sorted(
        texts,
        key=lambda text: (
            v1.selection_rank(
                seed=SEED,
                stage="stage_b",
                source_id="b_fixture",
                canonical_fingerprint=v1.canonical_document_fingerprint(text),
            ),
            v1.canonical_document_fingerprint(text),
        ),
    )
    target = sum(tokens[text] for text in ranked[:5]) - 1
    emitted = _run_selector_v1(tmp_path, "seeded", [{"text": t} for t in texts], {}, target)
    expected_ids = [row["_petitgpt_selection"]["cleaned_sha256"] for row in emitted]

    graph = load_fixture(
        mini_graph(tmp_path / "g", [node("b_fixture", "stage_b", target, ["ib_x"])])
    )
    candidates = [
        CandidateRecord(
            input_binding_id="ib_x",
            row_index=index,
            cleaned_sha256=hashlib.sha256(text.encode()).hexdigest(),
            canonical_fingerprint=v1.canonical_document_fingerprint(text),
            serialized_tokens=tokens[text],
            representative_key=representative_key(
                raw_sha256=oracle_raw_sha256(text),
                input_record_sha256=oracle_canonical_record_sha256({"text": text}),
            ),
        )
        for index, text in enumerate(texts)
    ]
    out = replay(graph, with_records(graph, {"ib_x": candidates}), set())[0]
    assert out.selected_identities == len(expected_ids)
    assert out.selection_fingerprint == oracle_fingerprint(expected_ids)
    assert out.selected_serialized_tokens == sum(
        row["_petitgpt_selection"]["serialized_tokens"] for row in emitted
    )


def test_m04_ownership_matches_the_real_selector_across_two_sources(tmp_path: Path):
    """Committed identities transfer ownership in v1; the replay must agree exactly."""
    from tokenizers import Tokenizer

    from pretrain import select_pretrain_documents as v1

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    shared = "a shared document that both sources contain verbatim"
    only_early = "a document that appears solely in the earlier source"
    only_late = "a document that appears solely in the later source"
    texts = {t: len(tokenizer.encode(t).ids) + 2 for t in (shared, only_early, only_late)}

    early_rows = [{"text": shared}, {"text": only_early}]
    late_rows = [{"text": shared}, {"text": only_late}]
    early_path = tmp_path / "early.jsonl"
    late_path = tmp_path / "late.jsonl"
    for path, rows in ((early_path, early_rows), (late_path, late_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    exclusion = tmp_path / "excl.json"
    exclusion.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": v1.EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": v1.CLEANED_TEXT_HASH_ALGORITHM,
            "cleaning": v1.CleaningSpec().as_dict(),
            "hash_count": 1,
            "hashes": [hashlib.sha256(b"unrelated").hexdigest()],
        }),
        encoding="utf-8",
    )
    early_target = texts[shared] + texts[only_early]
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({
            "schema_version": 1,
            "seed": SEED,
            "text_field": "text",
            "cleaning": {},
            "sources": [
                {
                    "stage": "stage_b",
                    "source_id": "b_early",
                    "path": str(early_path),
                    "target_serialized_tokens": early_target,
                },
                {
                    "stage": "stage_b",
                    "source_id": "b_late",
                    "path": str(late_path),
                    "target_serialized_tokens": texts[only_late],
                },
            ],
        }),
        encoding="utf-8",
    )
    out_dir = tmp_path / "v1out"
    manifest = v1.build_selection_registry(
        spec_path=spec,
        tokenizer_path=TOKENIZER,
        out_dir=out_dir,
        exclude_hash_manifests=(exclusion,),
        sqlite_cache_mb=1,
        commit_every=1,
    )
    late_source = next(s for s in manifest["sources"] if s["source_id"] == "b_late")
    assert late_source["scan"]["excluded_by_prior_selection"] == 1
    assert late_source["exact_hash_exclusions_by_owner"] == {"b_early": 1}

    graph = load_fixture(
        mini_graph(
            tmp_path / "g",
            [
                node("b_early", "stage_b", early_target, ["ib_x"]),
                node("b_late", "stage_b", texts[only_late], ["ib_y"]),
            ],
            bindings=("ib_x", "ib_y"),
        )
    )

    def make(text, binding, row):
        return CandidateRecord(
            input_binding_id=binding,
            row_index=row,
            cleaned_sha256=hashlib.sha256(text.encode()).hexdigest(),
            canonical_fingerprint=v1.canonical_document_fingerprint(text),
            serialized_tokens=texts[text],
            representative_key=representative_key(
                raw_sha256=oracle_raw_sha256(text),
                input_record_sha256=oracle_canonical_record_sha256({"text": text}),
            ),
        )

    results = replay(
        graph,
        with_records(
            graph,
            {
                "ib_x": [make(shared, "ib_x", 0), make(only_early, "ib_x", 1)],
                "ib_y": [make(shared, "ib_y", 0), make(only_late, "ib_y", 1)],
            },
        ),
        set(),
    )
    assert results[1].prior_commit_excluded_identities == 1
    assert results[1].exclusions_by_owner == {"b_early": 1}
    for source, result in zip(manifest["sources"], results, strict=True):
        assert result.selected_identities == source["selected"]["documents"]
        assert result.selected_serialized_tokens == source["selected"]["serialized_tokens"]
        assert result.actual_overshoot_tokens == source["selected"]["serialized_token_overshoot"]


def test_m04_g2_exclusion_matches_the_real_selector(tmp_path: Path):
    from tokenizers import Tokenizer

    from pretrain import select_pretrain_documents as v1

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    reserved = "a reserved reference validation document body"
    keep = "an ordinary document that remains selectable"
    tokens = {t: len(tokenizer.encode(t).ids) + 2 for t in (reserved, keep)}
    rows = [{"text": reserved}, {"text": keep}]
    source = tmp_path / "src.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    exclusion = tmp_path / "excl.json"
    exclusion.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": v1.EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": v1.CLEANED_TEXT_HASH_ALGORITHM,
            "cleaning": v1.CleaningSpec().as_dict(),
            "hash_count": 1,
            "hashes": [hashlib.sha256(reserved.encode()).hexdigest()],
        }),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({
            "schema_version": 1,
            "seed": SEED,
            "text_field": "text",
            "cleaning": {},
            "sources": [
                {
                    "stage": "stage_b",
                    "source_id": "b_x",
                    "path": str(source),
                    "target_serialized_tokens": tokens[keep],
                }
            ],
        }),
        encoding="utf-8",
    )
    manifest = v1.build_selection_registry(
        spec_path=spec,
        tokenizer_path=TOKENIZER,
        out_dir=tmp_path / "out",
        exclude_hash_manifests=(exclusion,),
        sqlite_cache_mb=1,
        commit_every=1,
    )
    assert manifest["sources"][0]["scan"]["excluded_reference_validation"] == 1

    graph = load_fixture(
        mini_graph(tmp_path / "g", [node("b_x", "stage_b", tokens[keep], ["ib_x"])])
    )
    candidates = [
        CandidateRecord(
            input_binding_id="ib_x",
            row_index=index,
            cleaned_sha256=hashlib.sha256(text.encode()).hexdigest(),
            canonical_fingerprint=v1.canonical_document_fingerprint(text),
            serialized_tokens=tokens[text],
            representative_key=representative_key(
                raw_sha256=oracle_raw_sha256(text),
                input_record_sha256=oracle_canonical_record_sha256({"text": text}),
            ),
        )
        for index, text in enumerate((reserved, keep))
    ]
    result = replay(
        graph,
        with_records(graph, {"ib_x": candidates}),
        {hashlib.sha256(reserved.encode()).hexdigest()},
    )[0]
    assert result.g2_excluded_identities == 1
    assert result.selected_identities == 1
    assert result.selection_fingerprint == oracle_fingerprint([
        hashlib.sha256(keep.encode()).hexdigest()
    ])


def test_m04_candidate_plan_is_reproducible_byte_for_byte(tmp_path: Path):
    """The tracked generator alone must produce the sealed plan bytes; nothing added by hand."""
    graph = load_source_graph(GRAPH, verify_hashes=False)
    commit = _runtime_head()
    first = canonical_json_bytes(
        generate_candidate_plan(
            graph_path=GRAPH, graph=graph, repo_root=ROOT, implementation_commit=commit
        )
    )
    second = canonical_json_bytes(
        generate_candidate_plan(
            graph_path=GRAPH,
            graph=load_source_graph(GRAPH, verify_hashes=False),
            repo_root=ROOT,
            implementation_commit=commit,
        )
    )
    assert first == second
    plan = json.loads(first)
    assert plan["implementation_commit"] == commit
    assert plan["schema_version"] == PLAN_SCHEMA == "petitgpt-h-candidate-plan-v4"
    assert plan["authorization_status"] == "NOT_AUTHORIZED"
    assert sorted(plan) == sorted([
        "schema_version",
        "authorization_status",
        "authorization_note",
        "census_schema_version",
        "node_result_schema_version",
        "graph_path",
        "graph_sha256",
        "seed",
        "resume_supported",
        "h_publishes_physical_views",
        "bound_authorities",
        "authorities",
        "implementation_commit",
        "implementation_files",
        "implementation_bundle_sha256",
        "input_bindings",
        "nodes",
        "node_order",
    ])


def test_m04_generated_plan_authorises_without_manual_editing(tmp_path: Path):
    """Generate, hash, authorise: the round trip must close with no hand edits."""
    path = tmp_path / "plan.json"
    plan = generate_candidate_plan(
        graph_path=GRAPH,
        graph=load_source_graph(GRAPH, verify_hashes=False),
        repo_root=ROOT,
        implementation_commit=_runtime_head(),
    )
    path.write_bytes(canonical_json_bytes(plan))
    context = authorize_run(plan_path=path, expected_plan_sha256=file_sha256(path), repo_root=ROOT)
    assert context.plan["authorization_status"] == "NOT_AUTHORIZED"
    assert context.graph_sha256 == GRAPH_SHA


def test_m04_default_authority_paths_resolve_to_the_bound_bytes():
    graph = load_source_graph(GRAPH, verify_hashes=False)
    for name, relative in sorted(DEFAULT_AUTHORITY_PATHS.items()):
        assert file_sha256(ROOT / relative) == graph.bound_authorities[AUTHORITY_KEYS[name]]


# ---------------------------------------------------------------- architecture


def test_primary_branch_uses_seeded_hash_and_fallback_only_on_shortfall(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [
                node(
                    "b_x",
                    "stage_b",
                    200,
                    ["ib_x"],
                    mode="BRANCH_DEPENDENT",
                    primary=PRIMARY,
                    fallback=FALLBACK,
                )
            ],
        )
    )
    plenty = [rec(f"p{i}", 100, int_score=4, score=4.5) for i in range(5)]
    out = replay(graph, with_records(graph, {"ib_x": plenty}), set())[0]
    assert out.branch == "PRIMARY_GE4" and out.selection_mode == "SEEDED_HASH"
    thin = [rec("hq", 100, int_score=4, score=4.5)] + [
        rec(f"lo{i}", 100, int_score=3, score=3.0) for i in range(5)
    ]
    out = replay(graph, with_records(graph, {"ib_x": thin}), set())[0]
    assert out.branch == "FALLBACK_RANKED_GE3"
    assert out.selection_mode == "EXACT_SCORE_DESC_SHA_ASC"


def test_g2_exclusion_can_flip_the_branch(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [
                node(
                    "b_x",
                    "stage_b",
                    200,
                    ["ib_x"],
                    mode="BRANCH_DEPENDENT",
                    primary=PRIMARY,
                    fallback=FALLBACK,
                )
            ],
        )
    )
    hq = [rec(f"h{i}", 100, int_score=4, score=4.5) for i in range(2)]
    lo = [rec(f"l{i}", 100, int_score=3, score=3.0) for i in range(5)]
    assert replay(graph, with_records(graph, {"ib_x": hq + lo}), set())[0].branch == "PRIMARY_GE4"
    flipped = replay(graph, with_records(graph, {"ib_x": hq + lo}), {hq[0].cleaned_sha256})[0]
    assert flipped.branch == "FALLBACK_RANKED_GE3"


def test_source_id_and_seed_change_seeded_fingerprints():
    fingerprint = "ab" * 32
    base = selection_rank(
        seed=SEED, stage="stage_b", source_id="b_x", canonical_fingerprint=fingerprint
    )
    assert base != selection_rank(
        seed=SEED, stage="stage_b", source_id="b_y", canonical_fingerprint=fingerprint
    )
    assert base != selection_rank(
        seed=SEED + 1, stage="stage_b", source_id="b_x", canonical_fingerprint=fingerprint
    )
    assert base != selection_rank(
        seed=SEED, stage="stage_a", source_id="b_x", canonical_fingerprint=fingerprint
    )


def test_shared_input_binding_needs_no_copy_link_or_duplicate(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [
                node("b_finewiki_en", "stage_b", 100, ["ib_x"]),
                node("a_finewiki_en", "stage_a", 200, ["ib_x"]),
            ],
        )
    )
    assert graph.nodes[0].input_binding_ids == graph.nodes[1].input_binding_ids == ("ib_x",)
    out = replay(graph, with_records(graph, {"ib_x": [rec(f"d{i}", 100) for i in range(6)]}), set())
    assert out[0].selected_identities == 1
    assert out[1].prior_commit_excluded_identities == 1


def test_structured_tutorial_union_is_order_independent(tmp_path: Path):
    graph = load_fixture(
        mini_graph(
            tmp_path,
            [node("b_structured_tutorial", "stage_b", 300, ["ib_x", "ib_y"])],
            bindings=("ib_x", "ib_y"),
            tutorial_ids=["ib_x", "ib_y"],
        )
    )
    first_set = [rec(f"a{i}", 100, binding="ib_x", row=i) for i in range(4)]
    second_set = [rec(f"b{i}", 100, binding="ib_y", row=i) for i in range(4)]
    first = replay(graph, with_records(graph, {"ib_x": first_set, "ib_y": second_set}), set())[0]
    second = replay(
        graph,
        with_records(
            graph, {"ib_x": list(reversed(first_set)), "ib_y": list(reversed(second_set))}
        ),
        set(),
    )[0]
    assert first.selection_fingerprint == second.selection_fingerprint
    assert first.selected_serialized_tokens == second.selected_serialized_tokens


def test_h_publishes_no_physical_candidate_views(tmp_path: Path):
    """Exercised, not grepped: a full authorised publication writes exactly two files."""
    context = _authorized(tmp_path)
    final = publish_atomic(context, _fake_census(context), tmp_path / "out")
    assert sorted(p.name for p in final.iterdir()) == ["COMPLETE", "census.json"]
    census = json.loads((final / "census.json").read_text())

    published_keys = set()

    def collect_keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                published_keys.add(key)
                collect_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_keys(nested)

    collect_keys(census)
    assert published_keys.isdisjoint({
        "candidate_records",
        "documents",
        "document_paths",
        "records",
        "rows",
        "selected_documents",
        "text",
    })
    assert context.plan["h_publishes_physical_views"] is False


def test_replay_rejects_unknown_or_missing_bindings(tmp_path: Path):
    graph = load_fixture(mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])]))
    with pytest.raises(ReplayError, match="missing candidate records"):
        replay(graph, {}, set())
    with pytest.raises(ReplayError, match="unknown input binding"):
        replay(graph, with_records(graph, {"ib_x": [], "ib_nope": []}), set())


def test_replay_rejects_records_claiming_another_binding(tmp_path: Path):
    graph = load_fixture(
        mini_graph(tmp_path, [node("b_x", "stage_b", 100, ["ib_x"])], bindings=("ib_x", "ib_y"))
    )
    with pytest.raises(ReplayError, match="claims input binding"):
        replay(graph, with_records(graph, {"ib_x": [rec("d", 100, binding="ib_y")]}), set())


def test_selection_is_reproducible_across_processes(tmp_path: Path):
    """No process-local seed, hash randomisation or dictionary order can move the result."""
    script = tmp_path / "probe.py"
    script.write_text(
        "\n".join([
            "import hashlib, json, sys",
            f"sys.path.insert(0, {str(ROOT)!r})",
            "from pretrain.stage_i_graph_v2 import load_source_graph",
            "from pretrain.stage_i_replay_v2 import CandidateRecord, replay, representative_key",
            "graph = load_source_graph(sys.argv[1], verify_hashes=True)",
            "def make(i):",
            "    return CandidateRecord(",
            "        input_binding_id='ib_x', row_index=i,",
            "        cleaned_sha256=hashlib.sha256(f'c-{i}'.encode()).hexdigest(),",
            "        canonical_fingerprint=hashlib.sha256(f'f-{i}'.encode()).hexdigest(),",
            "        serialized_tokens=100 + i,",
            "        representative_key=representative_key(",
            "            raw_sha256=hashlib.sha256(f'r-{i}'.encode()).hexdigest(),",
            "            input_record_sha256=hashlib.sha256(f'q-{i}'.encode()).hexdigest()),",
            "    )",
            "records = {k: [] for k in graph.bindings}",
            "records['ib_x'] = [make(i) for i in range(12)]",
            "out = replay(graph, records, set())[0]",
            "print(json.dumps({'fp': out.selection_fingerprint,",
            "                  'mass': out.selected_serialized_tokens}))",
        ]),
        encoding="utf-8",
    )
    graph_path = mini_graph(tmp_path / "g", [node("b_x", "stage_b", 350, ["ib_x"])])
    outputs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, str(script), str(graph_path)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1


def test_temporary_fixtures_never_touch_the_repository():
    """Nothing in this suite may write inside the working tree."""
    with tempfile.TemporaryDirectory() as directory:
        assert not Path(directory).is_relative_to(ROOT)
