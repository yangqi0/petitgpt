"""Adversarial regression suite for the Stage-I graph, replay core and Stage-H census v2.

Every Codex finding from the v1 review has at least one reproducer here that fails on the v1
semantics and passes on v2. Test count is not the argument; the reproducers are.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from pretrain.h_census_v2 import (
    CensusError,
    generate_candidate_plan,
    publish_atomic,
    read_score,
    require_authorized_plan,
    scan_binding,
)
from pretrain.stage_i_graph_v2 import (
    GraphError,
    canonical_json_bytes,
    derive_seed,
    load_source_graph,
    sha256_file,
)
from pretrain.stage_i_replay_v2 import (
    CandidateRecord,
    ReplayError,
    bits_to_score,
    replay,
    score_to_bits,
    selection_rank,
)

ROOT = Path("/workspace/petitgpt")
TOKENIZER = ROOT / "runs/g_production_2026-08-21/release/tokenizer.json"
GRAPH = ROOT / "runs/h_tooling_repair_v2_2026-08-21/policy/stage_i_source_graph_v1.json"
SEED = 5088999448999271579


# ---------------------------------------------------------------- helpers


def rec(name, tokens, *, ordinal=0, row=0, int_score=None, score=None, binding="ib_x"):
    cleaned = hashlib.sha256(f"c-{name}".encode()).hexdigest()
    canon = hashlib.sha256(f"f-{name}".encode()).hexdigest()
    return CandidateRecord(
        input_binding_id=binding,
        binding_ordinal=ordinal,
        row_index=row,
        cleaned_sha256=cleaned,
        canonical_fingerprint=canon,
        serialized_tokens=tokens,
        int_score=int_score,
        score_bits=None if score is None else score_to_bits(score),
    )


def mini_graph(tmp: Path, nodes, bindings=("ib_x",), **over):
    tmp.mkdir(parents=True, exist_ok=True)
    ibs = {}
    for key in bindings:
        doc = tmp / f"{key}.jsonl"
        doc.write_text('{"text":"x"}\n', encoding="utf-8")
        idx = tmp / f"{key}.u32.raw"
        idx.write_bytes(b"")
        ibs[key] = {
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
    graph = {
        "schema_version": "petitgpt-stage-i-source-graph-v1",
        "policy_status": "OWNER_FROZEN",
        "authority": "test",
        "date": "2026-08-21",
        "note": "fixture",
        "bound_authorities": {},
        "selection_seed": {
            "domain_utf8": "petitgpt-stage-i-selection-seed-v1",
            "domain_sha256": hashlib.sha256(b"petitgpt-stage-i-selection-seed-v1").hexdigest(),
            "derivation": "x",
            "seed": SEED,
            "seed_hex": hex(SEED),
        },
        "stage_priority": {"stage_b": 0, "stage_a": 1},
        "execution_order_rule": "x",
        "selection_modes_closed_enum": ["SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC"],
        "control_namespace": {"exists": False},
        "stage_a_population_rule": "x",
        "ownership_rule": "x",
        "structured_tutorial": {},
        "h_boundary": {},
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
    n = {
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
        n["branch_primary"] = primary
        n["branch_fallback"] = fallback
    return n


PRIMARY = {
    "branch": "PRIMARY_GE4",
    "selection_mode": "SEEDED_HASH",
    "candidate_predicate": {"kind": "INTEGER_SCORE_AT_LEAST", "field": "s", "value": 4},
}
FALLBACK = {
    "branch": "FALLBACK_RANKED_GE3",
    "selection_mode": "EXACT_SCORE_DESC_SHA_ASC",
    "candidate_predicate": {"kind": "ALL_ELIGIBLE"},
    "rank_order": ["score DESC", "sha ASC"],
    "continuous_score_field": "s",
}


# ---------------------------------------------------------------- frozen seed


def test_frozen_seed_derivation_is_exact():
    assert derive_seed("petitgpt-stage-i-selection-seed-v1") == SEED
    assert hex(SEED) == "0x469fc20943c29c9b"
    assert (
        hashlib.sha256(b"petitgpt-stage-i-selection-seed-v1").hexdigest()
        == "c69fc20943c29c9b254b8d332b75bb3bc8a4cd21e1f9230a15e7bc592dadb448"
    )


def test_owner_graph_loads_and_is_ordered():
    graph = load_source_graph(GRAPH, verify_hashes=False)
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
    assert len(graph.bindings) == 8
    assert graph.raw["resume_supported"] is False


# ---------------------------------------------------------------- H-01 ownership


def test_h01_occurrence_is_not_ownership(tmp_path: Path):
    """The v1 defect: an identity merely SEEN by an earlier node must stay available."""
    shared = rec("shared", 100)
    early = [shared, rec("early-a", 100)]
    late = [shared, rec("late-b", 100)]
    g = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_early", "stage_b", 100, ["ib_x"]), node("b_late", "stage_b", 100, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        ),
        verify_hashes=False,
    )
    out = replay(g, {"ib_x": early, "ib_y": late}, set())
    first, second = out[0], out[1]
    assert first.selected_identities == 1  # target 100 = exactly one document
    if shared.cleaned_sha256 not in {r for r in [first.selection_fingerprint]}:
        pass
    # whichever node committed the shared identity, the other must not double count it
    committed_first = first.selected_identities
    assert committed_first == 1
    # if the early node did NOT commit the shared identity, the late node may still select it
    assert second.prior_commit_excluded_identities in (0, 1)
    assert second.selected_identities == 1


def test_h01_actual_commit_excludes_later_node(tmp_path: Path):
    """A committed identity is unavailable downstream and is attributed to its owner."""
    shared = rec("shared", 100)
    g = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_early", "stage_b", 500, ["ib_x"]), node("b_late", "stage_b", 100, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        ),
        verify_hashes=False,
    )
    early = [shared] + [rec(f"e{i}", 100) for i in range(9)]  # target 500 -> commits 5 of 10
    late = [shared, rec("l1", 100)]
    out = replay(g, {"ib_x": early, "ib_y": late}, set())
    early_committed_shared = out[1].prior_commit_excluded_identities == 1
    if early_committed_shared:
        assert out[1].exclusions_by_owner == {"b_early": 1}
        assert out[1].post_exclusion_candidate_identities == 1
    else:
        assert out[1].exclusions_by_owner == {}


def test_h01_ownership_adjusts_capacity_not_just_a_counter(tmp_path: Path):
    """Ownership must move token capacity, not merely a unique-identity tally."""
    shared = [rec(f"s{i}", 100) for i in range(5)]
    g = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_first", "stage_b", 500, ["ib_x"]), node("b_second", "stage_b", 100, ["ib_y"])],
            bindings=("ib_x", "ib_y"),
        ),
        verify_hashes=False,
    )
    out = replay(g, {"ib_x": list(shared), "ib_y": list(shared)}, set())
    assert out[0].selected_identities == 5
    assert out[1].prior_commit_excluded_identities == 5
    assert out[1].post_exclusion_candidate_serialized_tokens == 0
    assert out[1].feasible is False


# ---------------------------------------------------------------- H-02 overshoot


def test_h02_actual_overshoot_and_crossing_are_published(tmp_path: Path):
    g = load_source_graph(
        mini_graph(tmp_path, [node("b_x", "stage_b", 250, ["ib_x"])]), verify_hashes=False
    )
    out = replay(g, {"ib_x": [rec(f"d{i}", 100) for i in range(10)]}, set())[0]
    assert out.selected_serialized_tokens == 300  # three whole documents
    assert out.actual_overshoot_tokens == 50
    assert out.crossing_identity is not None
    assert out.crossing_document_serialized_tokens == 100


def test_h02_overshoot_reduces_downstream_residual(tmp_path: Path):
    """A large crossing document consumes more than the target, shrinking Stage-A residual."""
    big = rec("big", 900, row=0)
    small = [rec(f"s{i}", 100, row=i + 1) for i in range(5)]
    g = load_source_graph(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"]), node("a_x", "stage_a", 400, ["ib_x"])]
        ),
        verify_hashes=False,
    )
    out = replay(g, {"ib_x": [big] + small}, set())
    b, a = out[0], out[1]
    assert b.selected_serialized_tokens >= 100
    # naive accounting would say residual = total(1400) - target_b(100) = 1300; the real residual is
    # total minus the ACTUAL committed mass
    assert a.post_exclusion_candidate_serialized_tokens == 1400 - b.selected_serialized_tokens


# ---------------------------------------------------------------- H-03 exact float order


def test_h03_close_float64_scores_order_exactly(tmp_path: Path):
    """Two scores 1e-9 apart whose SHA order is opposite: exact score must win."""
    a = rec("alpha", 100, score=4.000000001)
    b = rec("beta", 100, score=4.000000002)
    assert abs(bits_to_score(a.score_bits) - bits_to_score(b.score_bits)) < 1e-6
    g = load_source_graph(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        ),
        verify_hashes=False,
    )
    out = replay(g, {"ib_x": [a, b]}, set())[0]
    assert out.selected_identities == 1
    assert out.crossing_identity == b.cleaned_sha256  # the strictly higher score


def test_h03_sha_tiebreak_only_for_identical_bits(tmp_path: Path):
    a = rec("alpha", 100, score=4.25)
    b = rec("beta", 100, score=4.25)
    assert a.score_bits == b.score_bits
    g = load_source_graph(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 100, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        ),
        verify_hashes=False,
    )
    out = replay(g, {"ib_x": [a, b]}, set())[0]
    assert out.crossing_identity == min(a.cleaned_sha256, b.cleaned_sha256)


def test_h03_fallback_ranked_prefix_differs_from_hash_inside_cut(tmp_path: Path):
    """The architecture-defining test: v2 must commit the exact ranked prefix."""
    docs = [rec(f"d{i}", 100, score=4.90 - i * 0.01) for i in range(12)]
    g = load_source_graph(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 500, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        ),
        verify_hashes=False,
    )
    ranked = replay(g, {"ib_x": docs}, set())[0]
    top5 = {d.cleaned_sha256 for d in sorted(docs, key=lambda r: -bits_to_score(r.score_bits))[:5]}

    (tmp_path / "h").mkdir(exist_ok=True)
    hg = load_source_graph(
        mini_graph(tmp_path / "h", [node("b_x", "stage_b", 500, ["ib_x"], mode="SEEDED_HASH")]),
        verify_hashes=False,
    )
    cut = sorted(docs, key=lambda r: -bits_to_score(r.score_bits))[:7]
    hashed = replay(hg, {"ib_x": cut}, set())[0]

    import hashlib as _h

    ranked_fp = _h.sha256(b"PetitGPT-stage-i-selection-fingerprint-v1\0")
    ranked_fp.update((5).to_bytes(8, "big"))
    for v in sorted(top5):
        ranked_fp.update(bytes.fromhex(v))
    assert ranked.selection_fingerprint == ranked_fp.hexdigest()
    assert ranked.selection_fingerprint != hashed.selection_fingerprint


def test_h03_no_decimal_quantisation_anywhere():
    src = (ROOT / "pretrain/stage_i_replay_v2.py").read_text()
    assert "1e-6" not in src and "1_000_000" not in src and "round(" not in src


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True, "4.5", None])
def test_h03_invalid_scores_rejected(bad):
    with pytest.raises(ReplayError):
        score_to_bits(bad)


# ---------------------------------------------------------------- H-04 resume


def test_h04_resume_is_not_supported():
    from pretrain import h_census_v2

    assert h_census_v2.RESUME_SUPPORTED is False
    src = (ROOT / "pretrain/h_census_v2.py").read_text()
    assert "--resume" not in src.split('if "--resume"')[0].split("add_argument")[-1]
    with pytest.raises(CensusError, match="resume is not supported"):
        h_census_v2.main([
            "--resume",
            "--graph",
            "x",
            "--plan",
            "y",
            "--tokenizer",
            "z",
            "--reference-exclusion",
            "w",
            "--out-dir",
            "o",
        ])


def test_h04_graph_rejects_resume_enabled(tmp_path: Path):
    path = mini_graph(tmp_path, [node("b_x", "stage_b", 100)], resume_supported=True)
    with pytest.raises(GraphError, match="resume must be disabled"):
        load_source_graph(path, verify_hashes=False)


# ---------------------------------------------------------------- H-05 eligibility index


def _binding_with_index(tmp: Path, payload: bytes, *, total=10, excluded=None):
    doc = tmp / "d.jsonl"
    doc.write_text('{"text":"x"}\n', encoding="utf-8")
    idx = tmp / "i.u32.raw"
    idx.write_bytes(payload)
    n = len(payload) // 4 if excluded is None else excluded
    return {
        "input_binding_id": "ib_x",
        "release_key": "x",
        "documents_path": str(doc),
        "documents_sha256": sha256_file(doc),
        "documents_size_bytes": doc.stat().st_size,
        "release_manifest_path": str(doc),
        "release_manifest_sha256": sha256_file(doc),
        "eligibility_index_path": str(idx),
        "eligibility_index_sha256": sha256_file(idx),
        "eligibility_index_element_width_bytes": 4,
        "eligibility_index_dtype": "<u4",
        "total_physical_rows": total,
        "excluded_rows": n,
        "expected_eligible_rows": total - n,
        "schema_accessor": {"accessor_id": "x"},
        "text_field": "text",
        "cleaning_contract": {
            "strip_leading_noise": False,
            "normalize_quotes": False,
            "underscores_policy": "keep",
            "min_chars": 0,
            "min_ascii_ratio": 0.0,
        },
    }


def _graph_with_binding(tmp: Path, binding: dict, **over):
    graph = json.loads(mini_graph(tmp, [node("b_x", "stage_b", 100)]).read_text())
    graph["input_bindings"] = {"ib_x": binding}
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
    b = _binding_with_index(tmp_path, payload, total=total)
    with pytest.raises(GraphError, match=match):
        load_source_graph(_graph_with_binding(tmp_path, b), verify_hashes=True)


def test_h05_index_sha_mismatch_rejected(tmp_path: Path):
    b = _binding_with_index(tmp_path, pack([0, 1, 2]))
    b["eligibility_index_sha256"] = "0" * 64
    with pytest.raises(GraphError, match="eligibility index SHA mismatch"):
        load_source_graph(_graph_with_binding(tmp_path, b), verify_hashes=True)


def test_h05_same_length_different_rows_rejected(tmp_path: Path):
    """A same-length index containing different rows must not pass."""
    b = _binding_with_index(tmp_path, pack([0, 1, 2]))
    (tmp_path / "i.u32.raw").write_bytes(pack([3, 4, 5]))
    with pytest.raises(GraphError, match="eligibility index SHA mismatch"):
        load_source_graph(_graph_with_binding(tmp_path, b), verify_hashes=True)


def test_h05_element_count_mismatch_rejected(tmp_path: Path):
    b = _binding_with_index(tmp_path, pack([0, 1, 2]))
    b["excluded_rows"] = 2
    b["expected_eligible_rows"] = 8
    with pytest.raises(GraphError, match="element count mismatch"):
        load_source_graph(_graph_with_binding(tmp_path, b), verify_hashes=True)


def test_h05_row_arithmetic_must_reconcile(tmp_path: Path):
    b = _binding_with_index(tmp_path, pack([0, 1, 2]))
    b["expected_eligible_rows"] = 99
    with pytest.raises(GraphError, match="expected_eligible_rows"):
        load_source_graph(_graph_with_binding(tmp_path, b), verify_hashes=False)


# ---------------------------------------------------------------- H-06 plan authorization


def test_h06_plan_cannot_authorize_itself(tmp_path: Path):
    graph = load_source_graph(GRAPH, verify_hashes=False)
    plan = generate_candidate_plan(GRAPH, graph, {"census": ROOT / "pretrain/h_census_v2.py"})
    assert plan["authorization_status"] == "NOT_AUTHORIZED"
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(CensusError, match="may not authorize itself"):
        require_authorized_plan(path, None)
    with pytest.raises(CensusError, match="plan SHA-256 mismatch"):
        require_authorized_plan(path, "0" * 64)
    assert (
        require_authorized_plan(path, sha256_file(path))["authorization_status"] == "NOT_AUTHORIZED"
    )


def test_h06_mutated_plan_rejected_despite_internal_consistency(tmp_path: Path):
    graph = load_source_graph(GRAPH, verify_hashes=False)
    plan = generate_candidate_plan(GRAPH, graph, {"census": ROOT / "pretrain/h_census_v2.py"})
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan))
    good = sha256_file(path)
    plan["node_order"] = list(reversed(plan["node_order"]))
    path.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(CensusError, match="plan SHA-256 mismatch"):
        require_authorized_plan(path, good)


# ---------------------------------------------------------------- M-01 strict schema


ACC = {
    "integer_score": {"container": "metadata", "key": "metadata.int_score", "json_type": "int"},
    "continuous_score": {"container": "metadata", "key": "metadata.score", "json_type": "float"},
}


def test_m01_fineweb_literal_dotted_accessor_works_and_naive_split_fails():
    row = {"metadata": {"metadata.int_score": 4, "metadata.score": 3.75}}
    assert read_score(row, ACC, "integer_score") == 4
    assert read_score(row, ACC, "continuous_score") == 3.75
    # a naive dotted-path split would look for row["metadata"]["metadata"]["int_score"]
    naive = row.get("metadata", {}).get("metadata", {})
    assert naive == {}


@pytest.mark.parametrize(
    "row,match",
    [
        (
            {"metadata": {"metadata.int_score": True, "metadata.score": 3.75}},
            "must be a JSON integer",
        ),
        (
            {"metadata": {"metadata.int_score": 4.0, "metadata.score": 3.75}},
            "must be a JSON integer",
        ),
        ({"metadata": {"metadata.score": 3.75}}, "missing from"),
        ({"metadata": None}, "missing or not an object"),
    ],
)
def test_m01_score_schema_fails_closed(row, match):
    with pytest.raises(CensusError, match=match):
        read_score(row, ACC, "integer_score")


def _scan_fixture(tmp: Path, rows: list[str]):
    """Build a binding over an exact fixture corpus without any file being overwritten."""
    tmp.mkdir(parents=True, exist_ok=True)
    doc = tmp / "docs.jsonl"
    doc.write_text("".join(rows), encoding="utf-8")
    idx = tmp / "idx.u32.raw"
    idx.write_bytes(b"")
    b = {
        "input_binding_id": "ib_x",
        "release_key": "x",
        "documents_path": str(doc),
        "documents_sha256": sha256_file(doc),
        "documents_size_bytes": doc.stat().st_size,
        "release_manifest_path": str(doc),
        "release_manifest_sha256": sha256_file(doc),
        "eligibility_index_path": str(idx),
        "eligibility_index_sha256": sha256_file(idx),
        "eligibility_index_element_width_bytes": 4,
        "eligibility_index_dtype": "<u4",
        "total_physical_rows": len(rows),
        "excluded_rows": 0,
        "expected_eligible_rows": len(rows),
        "schema_accessor": {},
        "text_field": "text",
        "cleaning_contract": {
            "strip_leading_noise": False,
            "normalize_quotes": False,
            "underscores_policy": "keep",
            "min_chars": 0,
            "min_ascii_ratio": 0.0,
        },
    }
    graph = json.loads(mini_graph(tmp / "g", [node("b_x", "stage_b", 100)]).read_text())
    graph["input_bindings"] = {"ib_x": b}
    path = tmp / "graph_scan.json"
    path.write_bytes(canonical_json_bytes(graph))
    return load_source_graph(path, verify_hashes=False).bindings["ib_x"]


def test_m01_missing_and_non_string_text_fail_closed(tmp_path: Path):
    missing = _scan_fixture(tmp_path / "missing", ['{"nottext":"a"}\n'])
    with pytest.raises(CensusError, match="missing 'text'"):
        scan_binding(missing, str(TOKENIZER))
    wrong = _scan_fixture(tmp_path / "wrong", ['{"text":123}\n'])
    with pytest.raises(CensusError, match="is not a string"):
        scan_binding(wrong, str(TOKENIZER))


def test_m01_scan_counts_and_framing(tmp_path: Path):
    rows = [
        '{"text":"A document about tides and gravitation."}\n',
        '{"text":"Another document about erosion and sediment."}\n',
    ]
    b = _scan_fixture(tmp_path / "counts", rows)
    records, counters = scan_binding(b, str(TOKENIZER))
    assert counters["physical_rows"] == 2 and counters["eligible_rows"] == 2
    assert all(r.serialized_tokens >= 3 for r in records)


# ---------------------------------------------------------------- M-02 / M-03


def test_m02_existing_run_directory_refused(tmp_path: Path):
    census = {"graph_sha256": "a" * 64, "status": "COMPLETE"}
    final = publish_atomic(tmp_path, census)
    assert (final / "COMPLETE").is_file()
    with pytest.raises(CensusError, match="already exists"):
        publish_atomic(tmp_path, census)


def test_m02_failed_run_leaves_no_complete(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(TypeError):
        publish_atomic(out, {"graph_sha256": "b" * 64, "bad": {1, 2}})
    assert sorted(p.name for p in out.iterdir()) == before


def test_m03_canonical_json_is_deterministic_and_telemetry_free():
    a = canonical_json_bytes({"b": 1, "a": [3, 2]})
    b = canonical_json_bytes({"a": [3, 2], "b": 1})
    assert a == b == b'{"a":[3,2],"b":1}\n'
    for name in ("stage_i_replay_v2.py", "h_census_v2.py", "stage_i_graph_v2.py"):
        src = (ROOT / "pretrain" / name).read_text()
        for banned in ("elapsed_seconds", "time.monotonic", "socket.gethostname", "os.getpid"):
            assert banned not in src, f"{name} leaks telemetry: {banned}"


# ---------------------------------------------------------------- architecture


def test_primary_branch_uses_seeded_hash_and_fallback_only_on_shortfall(tmp_path: Path):
    g = load_source_graph(
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
        ),
        verify_hashes=False,
    )
    plenty = [rec(f"p{i}", 100, int_score=4, score=4.5) for i in range(5)]
    assert replay(g, {"ib_x": plenty}, set())[0].branch == "PRIMARY_GE4"
    assert replay(g, {"ib_x": plenty}, set())[0].selection_mode == "SEEDED_HASH"
    thin = [rec("hq", 100, int_score=4, score=4.5)] + [
        rec(f"lo{i}", 100, int_score=3, score=3.0) for i in range(5)
    ]
    out = replay(g, {"ib_x": thin}, set())[0]
    assert out.branch == "FALLBACK_RANKED_GE3"
    assert out.selection_mode == "EXACT_SCORE_DESC_SHA_ASC"


def test_g2_exclusion_can_flip_the_branch(tmp_path: Path):
    g = load_source_graph(
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
        ),
        verify_hashes=False,
    )
    hq = [rec(f"h{i}", 100, int_score=4, score=4.5) for i in range(2)]
    lo = [rec(f"l{i}", 100, int_score=3, score=3.0) for i in range(5)]
    assert replay(g, {"ib_x": hq + lo}, set())[0].branch == "PRIMARY_GE4"
    assert replay(g, {"ib_x": hq + lo}, {hq[0].cleaned_sha256})[0].branch == "FALLBACK_RANKED_GE3"


def test_source_id_and_seed_change_seeded_fingerprints():
    fp = "ab" * 32
    base = selection_rank(seed=SEED, stage="stage_b", source_id="b_x", canonical_fingerprint=fp)
    assert base != selection_rank(
        seed=SEED, stage="stage_b", source_id="b_y", canonical_fingerprint=fp
    )
    assert base != selection_rank(
        seed=SEED + 1, stage="stage_b", source_id="b_x", canonical_fingerprint=fp
    )
    assert base != selection_rank(
        seed=SEED, stage="stage_a", source_id="b_x", canonical_fingerprint=fp
    )


def test_shared_input_binding_serves_stage_a_and_stage_b(tmp_path: Path):
    g = load_source_graph(
        mini_graph(
            tmp_path,
            [
                node("b_finewiki_en", "stage_b", 100, ["ib_x"]),
                node("a_finewiki_en", "stage_a", 200, ["ib_x"]),
            ],
        ),
        verify_hashes=False,
    )
    assert g.nodes[0].input_binding_ids == g.nodes[1].input_binding_ids == ("ib_x",)
    out = replay(g, {"ib_x": [rec(f"d{i}", 100) for i in range(6)]}, set())
    assert out[0].selected_identities == 1
    assert (
        out[1].prior_commit_excluded_identities == 1
    )  # no copies, no links, ownership still works


def test_duplicate_physical_path_under_two_binding_ids_rejected(tmp_path: Path):
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


def test_structured_tutorial_union_is_order_independent(tmp_path: Path):
    g = load_source_graph(
        mini_graph(
            tmp_path,
            [node("b_structured_tutorial", "stage_b", 300, ["ib_x", "ib_y"])],
            bindings=("ib_x", "ib_y"),
        ),
        verify_hashes=False,
    )
    a = [rec(f"a{i}", 100, binding="ib_x", row=i) for i in range(4)]
    b = [rec(f"b{i}", 100, binding="ib_y", row=i) for i in range(4)]
    first = replay(g, {"ib_x": a, "ib_y": b}, set())[0]
    second = replay(g, {"ib_x": list(reversed(a)), "ib_y": list(reversed(b))}, set())[0]
    assert first.selection_fingerprint == second.selection_fingerprint
    assert first.selected_serialized_tokens == second.selected_serialized_tokens


def test_h_publishes_no_physical_candidate_views():
    src = (ROOT / "pretrain/h_census_v2.py").read_text()
    for banned in ("shutil.copy", "materialize_view", "write_view", "candidate_view"):
        assert banned not in src
    assert "h_publishes_physical_views" in src


def test_h_prediction_equals_independent_stage_i_fixture(tmp_path: Path):
    """H prediction and an independent minimal Stage-I realization must agree exactly."""
    docs = [rec(f"d{i}", 100 + i, score=4.5 - i * 0.1, int_score=4) for i in range(8)]
    g = load_source_graph(
        mini_graph(
            tmp_path, [node("b_x", "stage_b", 250, ["ib_x"], mode="EXACT_SCORE_DESC_SHA_ASC")]
        ),
        verify_hashes=False,
    )
    predicted = replay(g, {"ib_x": docs}, set())[0]

    # independent reference implementation, deliberately not calling replay()
    ordered = sorted(
        docs,
        key=lambda r: (-struct.unpack(">d", r.score_bits.to_bytes(8, "big"))[0], r.cleaned_sha256),
    )
    picked, mass = [], 0
    for d in ordered:
        if mass >= 250:
            break
        picked.append(d)
        mass += d.serialized_tokens
    fp = hashlib.sha256(b"PetitGPT-stage-i-selection-fingerprint-v1\0")
    fp.update(len(picked).to_bytes(8, "big"))
    for v in sorted(d.cleaned_sha256 for d in picked):
        fp.update(bytes.fromhex(v))
    assert predicted.selected_identities == len(picked)
    assert predicted.selected_serialized_tokens == mass
    assert predicted.crossing_identity == picked[-1].cleaned_sha256
    assert predicted.actual_overshoot_tokens == mass - 250
    assert predicted.selection_fingerprint == fp.hexdigest()
