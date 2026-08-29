"""P-PILOT-CONTRACT-V2.3 contract and executor tests.

Pure contract logic, tiny non-training fixtures, and mocks for executor orchestration. No real
model-training gradient update is performed anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

import pretrain.pilot_contract_v2_3 as C
import pretrain.pilot_runner_v2_3 as R

REPO_ROOT = Path(__file__).resolve().parent.parent
A_BLOCKS, B_BLOCKS = 4882814, 1464845
VRAM_4090 = 24564 * 1024 * 1024


# ------------------------------------------------------------------ supersession


def test_v2_3_supersedes_only_the_v2_2_optimizer_decision():
    d = C.contract_document()
    assert d["contract_version"] == "P-PILOT-CONTRACT-V2.3"
    assert "P-PILOT-CONTRACT-V2.2" in d["supersedes"]
    assert "optimizer" in d["supersedes"]["P-PILOT-CONTRACT-V2.2"]
    assert d["owner_optimizer_verdict"] == "FREEZE_MUON_DIRECTLY"
    assert d["optimizer_family_comparison_required"] is False
    for retained in ("effective-batch geometry", "production warmup 500"):
        assert retained in d["retained_from_v2_2"]
    for closed in ("model architecture", "tokenizer", "effective batch decision"):
        assert closed in d["not_reopened"]


def test_ignored_local_authority_has_no_runtime_role():
    """No executable code path may reference the gitignored local authority documents.

    Docstrings may name them -- both modules state explicitly that they are non-load-bearing --
    so this walks the AST and inspects only string constants that are NOT docstrings.
    """
    import ast

    assert C.contract_document()["ignored_local_authority_runtime_role"] == "NONE"
    local_names = ("CLAUDE.md", "DECISIONS.md", "PLAYBOOK.md", "RETRAIN_PLAN.md")
    for module in (C, R):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                for local in local_names:
                    assert local not in node.value, (
                        f"{module.__name__} references {local} in executable code"
                    )


def test_contract_digest_stable():
    assert (
        C.contract_sha256()
        == hashlib.sha256(C.canonical_json_bytes(C.contract_document())).hexdigest()
    )


# ------------------------------------------------------------------ Muon binding


def test_optimizer_is_muon_with_frozen_flags():
    assert C.FROZEN_OPTIMIZER == "muon"
    assert C.MUON_LR_ARG == 0.0
    assert C.MUON_MOMENTUM == 0.95
    cli = C.realized_muon_config()["cli"]
    assert cli["--optimizer"] == "muon"
    assert cli["--muon_lr"] == 0.0
    assert cli["--muon_momentum"] == 0.95
    assert cli["explicit_optimizer_flag_required"] is True


def test_realized_muon_grouping_matches_src_optim():
    cfg = C.realized_muon_config()
    names = [g["name"] for g in cfg["param_groups"]]
    assert names == ["muon_matrices", "aux_adamw_decay", "aux_adamw_no_decay"]
    assert cfg["adam_param_name_keys"] == ["tok_emb", "lm_head", ".gate."]
    muon, aux_d, aux_nd = cfg["param_groups"]
    assert muon["state_keys"] == ["momentum_buffer"]
    assert aux_d["state_keys"] == ["step", "exp_avg", "exp_avg_sq"]
    assert aux_nd["weight_decay"] == 0.0
    assert all(g["lr_ratio"] == 1.0 for g in cfg["param_groups"])
    assert "0.2 * sqrt(max(fan_in, fan_out))" in cfg["rms_matching"]["formula"]
    assert cfg["rms_matching"]["newton_schulz_steps"] == 5


def test_muon_lr_zero_yields_ratio_one_on_a_real_tiny_model():
    """Construct a tiny model and optimizer; no update is performed."""
    import torch

    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=64, n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=16
        )
    )
    opt = build_optimizer(
        model,
        name="muon",
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        muon_lr=0.0,
        muon_momentum=0.95,
        verbose=False,
    )
    v = C.verify_realized_grouping(opt)
    assert v["matches_frozen_realization"], v["failures"]
    assert v["all_lr_ratios_are_one"] is True
    assert v["muon_group_count"] == 1
    assert v["aux_adamw_group_count"] >= 1


def test_verify_grouping_rejects_wrong_ratio_and_momentum():
    fake = mock.Mock()
    fake.param_groups = [
        {"use_muon": True, "lr_ratio": 2.0, "momentum": 0.95, "params": []},
        {"use_muon": False, "lr_ratio": 1.0, "params": []},
    ]
    v = C.verify_realized_grouping(fake)
    assert not v["matches_frozen_realization"]
    assert any("lr_ratio" in f for f in v["failures"])
    fake.param_groups = [
        {"use_muon": True, "lr_ratio": 1.0, "momentum": 0.9, "params": []},
        {"use_muon": False, "lr_ratio": 1.0, "params": []},
    ]
    assert any("momentum" in f for f in C.verify_realized_grouping(fake)["failures"])


def test_apply_scheduled_lr_drives_all_groups_equally():
    fake = mock.Mock()
    fake.param_groups = [{"lr_ratio": 1.0}, {"lr_ratio": 1.0}, {"lr_ratio": 1.0}]
    assert R.apply_scheduled_lr(fake, 4e-4) == [4e-4, 4e-4, 4e-4]


# ------------------------------------------------------------------ Phase MB


def test_mb_grid_ten_candidates_exact_order():
    grid = C.mb_candidate_grid()
    assert len(grid) == 10
    assert [(c["micro_bsz"], c["compile"]) for c in grid] == [
        (16, False),
        (16, True),
        (8, False),
        (8, True),
        (4, False),
        (4, True),
        (2, False),
        (2, True),
        (1, False),
        (1, True),
    ]
    assert all(c["optimizer"] == "muon" and c["muon_lr"] == 0.0 for c in grid)
    assert all(c["micro_bsz"] * c["grad_accum"] == 128 for c in grid)


@pytest.mark.parametrize(("u", "mult"), [(1, 0.1), (9, 0.9), (10, 1.0), (11, 1.0), (40, 1.0)])
def test_mb_lr(u, mult):
    assert C.mb_lr(u) == pytest.approx(3e-4 * mult)


def _mb(cid, micro, comp, tps, vram_gib, **over):
    r = {
        "candidate_id": cid,
        "micro_bsz": micro,
        "compile": comp,
        "median_tokens_per_sec": tps,
        "max_memory_reserved_bytes": int(vram_gib * 1024**3),
        "completed_updates": 40,
        "oom": False,
        "uncontrolled_exception": False,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "all_optimizer_states_instantiated": True,
        "grouping_matches_contract": True,
        "all_lr_ratios_are_one": True,
        "canonical_compile_path": True,
    }
    r.update(over)
    return r


def _full_grid(**overrides):
    out = []
    for c in C.mb_candidate_grid():
        r = _mb(c["candidate_id"], c["micro_bsz"], c["compile"], 1000.0, 8.0)
        r.update(overrides.get(c["candidate_id"], {}))
        out.append(r)
    return out


def test_mb_requires_the_complete_grid():
    partial = _full_grid()[:3]
    with pytest.raises(C.PilotContractError, match="incomplete"):
        C.mb_select(partial, VRAM_4090)


def test_mb_rejects_duplicate_candidate_identity():
    dup = _full_grid() + [_mb("mb_micro8_compileoff", 8, False, 9999.0, 8.0)]
    with pytest.raises(C.PilotContractError, match="duplicate"):
        C.mb_select(dup, VRAM_4090)


def test_mb_rejects_unknown_candidate_identity():
    grid = _full_grid()
    grid[0] = _mb("mb_micro99_compileoff", 99, False, 1000.0, 8.0)
    with pytest.raises(C.PilotContractError, match="incomplete|unknown"):
        C.mb_select(grid, VRAM_4090)


def test_mb_selection_and_tie_ladder():
    out = C.mb_select(_full_grid(mb_micro8_compileoff={"median_tokens_per_sec": 5000.0}), VRAM_4090)
    assert out["FROZEN_MICRO_BSZ"] == 8 and out["FROZEN_GRAD_ACCUM"] == 16
    assert out["FROZEN_COMPILE"] is False and out["tie_break"] == "fastest_unique"
    # VRAM tie-break
    out2 = C.mb_select(
        _full_grid(
            mb_micro8_compileoff={
                "median_tokens_per_sec": 5000.0,
                "max_memory_reserved_bytes": int(12 * 1024**3),
            },
            mb_micro4_compileoff={
                "median_tokens_per_sec": 4950.0,
                "max_memory_reserved_bytes": int(8 * 1024**3),
            },
        ),
        VRAM_4090,
    )
    assert out2["tie_break"] == "lowest_peak_reserved_vram" and out2["FROZEN_MICRO_BSZ"] == 4


def test_mb_all_ineligible_aborts():
    out = C.mb_select(
        _full_grid(**{c["candidate_id"]: {"oom": True} for c in C.mb_candidate_grid()}), VRAM_4090
    )
    assert out["outcome"] == "PHASE_MB_ABORT"


@pytest.mark.parametrize(
    "bad",
    [
        {"grouping_matches_contract": False},
        {"all_lr_ratios_are_one": False},
        {"all_optimizer_states_instantiated": False},
    ],
)
def test_mb_eligibility_requires_muon_grouping_facts(bad):
    ok, failures = C.mb_candidate_eligible(_mb("x", 8, False, 1.0, 8.0, **bad), VRAM_4090)
    assert not ok and failures


# ------------------------------------------------------------------ Phase Muon-LR


def test_lr_grid_and_run_geometry():
    assert C.LR_GRID_SEED1 == (2e-4, 3e-4, 4e-4)
    assert C.LR_RUN_UPDATES == 200 and C.LR_WARMUP_UPDATES == 25
    assert C.LR_BLOCKS_PER_RUN == 25600
    assert C.LR_TOKENS_PER_RUN == 52428800


@pytest.mark.parametrize(("u", "mult"), [(1, 1 / 25), (24, 24 / 25), (25, 1.0), (26, 1.0)])
def test_lr_warmup_values(u, mult):
    assert C.lr_schedule(u, 4e-4) == pytest.approx(4e-4 * mult)


def test_lr_schedule_rejects_update_zero():
    with pytest.raises(C.PilotContractError, match="1-based"):
        C.lr_schedule(0, 4e-4)


def test_score_formula():
    assert C.lr_score(2.0, 1.0) == pytest.approx((10 * 2.0 + 3 * 1.0) / 13)


def test_sustained_divergence_baseline_and_window_semantics():
    calm = {u: 3.0 for u in range(1, 201)}
    r = C.sustained_divergence(calm)
    assert r["baseline"] == pytest.approx(3.0) and r["threshold"] == pytest.approx(4.5)
    assert r["diverged"] is False
    blown = dict(calm)
    for u in range(100, 140):
        blown[u] = 9.0
    d = C.sustained_divergence(blown)
    assert d["diverged"] is True
    assert all(80 <= v["window_final_update"] <= 200 for v in d["violations"])


def test_sustained_divergence_window_must_end_within_80_to_200():
    losses = {u: 3.0 for u in range(1, 201)}
    for u in range(1, 40):
        losses[u] = 50.0
    assert C.sustained_divergence(losses)["diverged"] is False


def _lr(peak, score, **over):
    r = {
        "peak_lr": peak,
        "score": score,
        "completed_updates": 200,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "all_parameters_finite": True,
        "muon_momentum_states_present": True,
        "aux_adamw_states_present": True,
        "grouping_matches_contract": True,
        "all_lr_ratios_are_one": True,
        "eval_loss_stage_a": 3.0,
        "eval_loss_stage_b": 3.0,
        "sustained_divergence": False,
    }
    r.update(over)
    return r


def test_lr_seed1_requires_complete_grid():
    with pytest.raises(C.PilotContractError, match="incomplete"):
        C.lr_select_seed1([_lr(2e-4, 3.2), _lr(3e-4, 3.1)])


def test_lr_seed1_selection_and_tie():
    out = C.lr_select_seed1([_lr(2e-4, 3.5), _lr(3e-4, 3.2), _lr(4e-4, 3.4)])
    assert out["winner_peak_lr"] == 3e-4
    tie = C.lr_select_seed1([_lr(2e-4, 3.21), _lr(3e-4, 3.20), _lr(4e-4, 3.9)])
    assert tie["winner_peak_lr"] == 2e-4


def test_lr_fewer_than_two_eligible_aborts():
    out = C.lr_select_seed1([
        _lr(2e-4, 3.5, completed_updates=199),
        _lr(3e-4, 3.2, sustained_divergence=True),
        _lr(4e-4, 3.4),
    ])
    assert out["outcome"] == "PHASE_MUON_LR_ABORT"


@pytest.mark.parametrize(("winner", "neighbor"), [(2e-4, 3e-4), (3e-4, 2e-4), (4e-4, 3e-4)])
def test_confirmation_neighbor(winner, neighbor):
    assert C.confirmation_neighbor(winner) == pytest.approx(neighbor)


def test_confirmation_one_ineligible_other_wins():
    out = C.lr_confirm([
        {"peak_lr": 3e-4, "seed1_score": 3.2, "seed2_score": 3.2, "seed2_eligible": False},
        {"peak_lr": 2e-4, "seed1_score": 3.6, "seed2_score": 3.6, "seed2_eligible": True},
    ])
    assert out["outcome"] == "CONFIRMED" and out["confirmed_peak_lr"] == 2e-4


def test_confirmation_both_ineligible_aborts():
    out = C.lr_confirm([
        {"peak_lr": 3e-4, "seed1_score": 3.2, "seed2_score": 3.2, "seed2_eligible": False},
        {"peak_lr": 2e-4, "seed1_score": 3.6, "seed2_score": 3.6, "seed2_eligible": False},
    ])
    assert out["outcome"] == "PHASE_MUON_LR_ABORT"


@pytest.mark.parametrize(("confirmed", "edge"), [(2e-4, 1e-4), (4e-4, 6e-4), (3e-4, None)])
def test_edge_candidate_rule(confirmed, edge):
    assert C.edge_candidate(confirmed) == (pytest.approx(edge) if edge else None)


def test_edge_requires_both_seeds_eligible():
    out = C.lr_resolve_edge(
        incumbent_lr=2e-4,
        incumbent_final_score=3.0,
        edge_lr=1e-4,
        edge_seed1_eligible=True,
        edge_seed2_eligible=False,
        edge_seed1_score=2.0,
        edge_seed2_score=2.0,
    )
    assert out["FROZEN_PEAK_LR"] == 2e-4
    assert out["rule"] == "edge_not_comparison_eligible_incumbent_remains"


def test_edge_wins_when_better_and_both_eligible():
    out = C.lr_resolve_edge(
        incumbent_lr=4e-4,
        incumbent_final_score=3.0,
        edge_lr=6e-4,
        edge_seed1_eligible=True,
        edge_seed2_eligible=True,
        edge_seed1_score=2.0,
        edge_seed2_score=2.0,
    )
    assert out["FROZEN_PEAK_LR"] == 6e-4 and out["edge_final_score"] == pytest.approx(2.0)


def test_edge_tie_goes_to_lower_lr():
    out = C.lr_resolve_edge(
        incumbent_lr=2e-4,
        incumbent_final_score=3.00,
        edge_lr=1e-4,
        edge_seed1_eligible=True,
        edge_seed2_eligible=True,
        edge_seed1_score=3.01,
        edge_seed2_score=3.01,
    )
    assert out["FROZEN_PEAK_LR"] == 1e-4  # within 0.5%, lower LR wins


def test_no_second_expansion_permitted():
    for lr in (1e-4, 6e-4, 3e-4):
        out = C.lr_resolve_edge(
            incumbent_lr=lr, incumbent_final_score=3.0, edge_lr=C.edge_candidate(lr)
        )
        assert out["second_expansion_permitted"] is False


# ------------------------------------------------------------------ token ledger


def test_token_ceilings():
    assert C.PHASE_MB_TOKEN_CEILING == 105_000_000
    assert C.PHASE_MB_EXPECTED_MAX == 104_857_600 == C.phase_mb_projected_tokens()
    assert C.PHASE_MUON_LR_TOKEN_CEILING == 370_000_000
    assert C.GLOBAL_PILOT_TOKEN_CEILING == 500_000_000


def test_ceiling_checked_before_each_update():
    ok = C.check_update_within_ceilings(phase="MB", phase_tokens_so_far=0, global_tokens_so_far=0)
    assert ok["may_execute"] is True
    at_edge = C.check_update_within_ceilings(
        phase="MB", phase_tokens_so_far=C.PHASE_MB_TOKEN_CEILING, global_tokens_so_far=0
    )
    assert at_edge["may_execute"] is False and at_edge["outcome"] == "PILOT_ABORT"
    glob = C.check_update_within_ceilings(
        phase="LR", phase_tokens_so_far=0, global_tokens_so_far=C.GLOBAL_PILOT_TOKEN_CEILING
    )
    assert "global_pilot_token_ceiling" in glob["breaches"]


def _ledger_identity(**over):
    identity = {
        "contract_sha256": "c",
        "implementation_head": "h",
        "execution_bundle_sha256": "b",
        "pilot_index_manifest_file_sha256": "m",
        "authorization_sha256": "a",
        "session_id": "s",
        "authorized_output_root": "/tmp/root",
        "authorized_scope": "FULL_V2_3_PILOT",
    }
    identity.update(over)
    return identity


def _ceilings(global_ceiling=C.GLOBAL_PILOT_TOKEN_CEILING):
    return {
        "MB": min(C.PHASE_MB_TOKEN_CEILING, global_ceiling),
        "LR": min(C.PHASE_MUON_LR_TOKEN_CEILING, global_ceiling),
        "GLOBAL": global_ceiling,
    }


def test_persistent_ledger_round_trip(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.reserve("MB")
    ledger.complete("MB")
    reloaded = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    assert reloaded.state["reserved_tokens"]["GLOBAL"] == C.EFFECTIVE_BATCH_TOKENS
    assert reloaded.state["completed_tokens"]["MB"] == C.EFFECTIVE_BATCH_TOKENS
    assert reloaded.state["reserved_updates"] == 1
    assert reloaded.state["completed_updates"] == 1


@pytest.mark.parametrize("field", list(R.TokenLedger.IDENTITY_FIELDS))
def test_ledger_reload_validates_every_bound_identity_field(tmp_path, field):
    R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    other = _ledger_identity(**{field: "DIFFERENT"})
    with pytest.raises(C.PilotContractError, match="does not bind this execution"):
        R.TokenLedger(tmp_path / "l.json", other, _ceilings())


def test_ledger_identity_must_be_complete(tmp_path):
    incomplete = _ledger_identity()
    incomplete.pop("session_id")
    with pytest.raises(C.PilotContractError, match="ledger identity is incomplete"):
        R.TokenLedger(tmp_path / "l.json", incomplete, _ceilings())


def test_ledger_refuses_update_over_ceiling(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.state["reserved_tokens"]["MB"] = C.PHASE_MB_TOKEN_CEILING
    ledger._write(ledger.state)  # noqa: SLF001 - exercising the persisted path deliberately
    with pytest.raises(C.PilotContractError, match="PILOT_ABORT"):
        ledger.reserve("MB")


def test_ledger_uses_interprocess_locking(tmp_path):
    """Both ledger transitions take an exclusive flock before reload/check/write."""
    import inspect

    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    assert "flock" in inspect.getsource(R.TokenLedger._lock)  # noqa: SLF001
    assert "LOCK_EX" in inspect.getsource(R.TokenLedger._lock)  # noqa: SLF001
    for method in (R.TokenLedger.reserve, R.TokenLedger.complete):
        assert "self._lock()" in inspect.getsource(method)
    ledger.reserve("MB")
    ledger.complete("MB")
    assert ledger.lock_path.exists()


@pytest.mark.parametrize("ceiling", [0, -1, C.GLOBAL_PILOT_TOKEN_CEILING + 1, "x", None])
def test_authorized_ceiling_validation(ceiling):
    with pytest.raises(C.PilotContractError):
        R.authorized_effective_ceilings({"pilot_trained_token_ceiling": ceiling})


def test_effective_ceiling_is_the_minimum(tmp_path):
    low = 1_000_000
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings(low))
    assert ledger.effective_ceiling("MB") == low
    assert ledger.effective_ceiling("LR") == low
    eff = R.authorized_effective_ceilings({"pilot_trained_token_ceiling": low})
    assert eff == {"MB": low, "LR": low, "GLOBAL": low}


# ------------------------------------------------------------------ indices


@pytest.fixture(scope="module")
def indices():
    return C.generate_pilot_indices(A_BLOCKS, B_BLOCKS)


def test_index_counts_hashes_and_disjointness(indices):
    assert len(indices["stage_a_eval"]) == 4096
    assert len(indices["stage_a_train"]) == 131072
    assert len(indices["stage_b_eval"]) == 4096
    assert not (set(indices["stage_a_eval"]) & set(indices["stage_a_train"]))
    assert indices["stage_a_eval"] == sorted(indices["stage_a_eval"])
    assert indices["stage_b_eval"] == sorted(indices["stage_b_eval"])
    assert indices["stage_a_train"] != sorted(indices["stage_a_train"])
    for key in ("stage_a_eval", "stage_a_train", "stage_b_eval"):
        payload = ("\n".join(str(v) for v in indices[key]) + "\n").encode()
        assert hashlib.sha256(payload).hexdigest() == indices[f"{key}_sha256"]


def test_index_generation_is_deterministic(indices):
    again = C.generate_pilot_indices(A_BLOCKS, B_BLOCKS)
    assert again["stage_a_train"] == indices["stage_a_train"]


def test_seed_orders_differ(indices):
    o1 = C.train_order(indices["stage_a_train"], 20260829)
    o2 = C.train_order(indices["stage_a_train"], 20260830)
    assert set(o1) == set(o2) and o1 != o2


def test_numpy_exact_version_required():
    assert C.REQUIRED_NUMPY_VERSION == "2.2.6"
    assert C.require_numpy_version() == "2.2.6"
    with mock.patch("numpy.__version__", "2.1.0"):
        with pytest.raises(C.PilotContractError, match="exactly 2.2.6"):
            C.require_numpy_version()


def test_pilot_requirements_file_pins_numpy():
    text = (REPO_ROOT / "requirements-pilot-v2_3.txt").read_text()
    assert "numpy==2.2.6" in text


# ------------------------------------------------------------------ runtime gate


def test_rtx4090_gate_accepts_only_the_exact_device():
    ok = C.check_training_authority({
        "name": "NVIDIA GeForce RTX 4090",
        "total_vram_mib": 24564,
        "cuda_available": True,
        "bf16_supported": True,
    })
    assert ok["granted"] and ok["training_authority"] == "GRANTED"


@pytest.mark.parametrize(
    "gpu",
    [
        {
            "name": "NVIDIA RTX 4000 Ada Generation",
            "total_vram_mib": 20146,
            "cuda_available": True,
            "bf16_supported": True,
        },
        {
            "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
            "total_vram_mib": 16384,
            "cuda_available": True,
            "bf16_supported": True,
        },
        {
            "name": "NVIDIA GeForce RTX 4090",
            "total_vram_mib": 24564,
            "cuda_available": False,
            "bf16_supported": True,
        },
        {
            "name": "NVIDIA GeForce RTX 4090",
            "total_vram_mib": 24564,
            "cuda_available": True,
            "bf16_supported": False,
        },
    ],
)
def test_rtx4090_gate_rejects(gpu):
    verdict = C.check_training_authority(gpu)
    assert not verdict["granted"] and verdict["training_authority"] == "NONE"
    with pytest.raises(C.PilotContractError, match="no training authority"):
        C.require_training_authority(gpu)


def test_substring_only_check_is_insufficient():
    """A 4090 Laptop GPU contains '4090' but is not the intended device."""
    assert "4090" in "NVIDIA GeForce RTX 4090 Laptop GPU"
    assert not C.check_training_authority({
        "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
        "total_vram_mib": 16384,
        "cuda_available": True,
        "bf16_supported": True,
    })["granted"]
    assert C.contract_document()["runtime_gate"]["substring_check_insufficient"] is True


# ------------------------------------------------------------------ authorization


def _authorized(observed, scope="FULL_V2_3_PILOT"):
    return {
        "schema_version": C.AUTHORIZATION_SCHEMA,
        "authorization_status": "AUTHORIZED",
        "allowed_scope": scope,
        "repository_branch": observed["branch"],
        "repository_head": observed["head"],
        "contract_version": C.CONTRACT_VERSION,
        "contract_sha256": observed["contract_sha256"],
        "execution_implementation_bundle_sha256": observed["execution_bundle_sha256"],
        "serialized_index_lists_digest": observed["serialized_index_lists_digest"],
        "pilot_index_manifest_file_sha256": observed["pilot_index_manifest_file_sha256"],
        "accepted_stage_a_meta_sha256": observed["stage_a_meta_sha256"],
        "accepted_stage_b_meta_sha256": observed["stage_b_meta_sha256"],
        "allowed_output_root": observed["output_root"],
        "pilot_trained_token_ceiling": C.GLOBAL_PILOT_TOKEN_CEILING,
    }


@pytest.fixture
def observed():
    return {
        "branch": "agent/retrain-pipeline-contracts",
        "head": "deadbeef",
        "contract_sha256": C.contract_sha256(),
        "execution_bundle_sha256": "bundle",
        "serialized_index_lists_digest": "idx",
        "pilot_index_manifest_file_sha256": "mfile",
        "stage_a_meta_sha256": "a",
        "stage_b_meta_sha256": "b",
        "output_root": "/tmp/out",
    }


def test_authorization_template_is_not_authorized():
    t = C.authorization_template()
    assert t["authorization_status"] == "NOT_AUTHORIZED"
    assert t["allowed_scope"] is None and t["repository_head"] is None
    assert set(C.ALLOWED_SCOPES) == {"PHASE_MB_ONLY", "FULL_V2_3_PILOT"}


def test_authorization_missing_refuses(observed):
    v = C.validate_authorization(None, requested_scope="PHASE_MB_ONLY", observed=observed)
    assert not v["authorized"] and v["failures"] == ["authorization_missing"]


def test_not_authorized_status_refuses(observed):
    m = _authorized(observed) | {"authorization_status": "NOT_AUTHORIZED"}
    v = C.validate_authorization(m, requested_scope="FULL_V2_3_PILOT", observed=observed)
    assert not v["authorized"]
    assert "authorization_status_not_authorized" in v["failures"]


@pytest.mark.parametrize(
    "field",
    [
        "repository_branch",
        "repository_head",
        "contract_sha256",
        "execution_implementation_bundle_sha256",
        "serialized_index_lists_digest",
        "pilot_index_manifest_file_sha256",
        "accepted_stage_a_meta_sha256",
        "accepted_stage_b_meta_sha256",
        "allowed_output_root",
    ],
)
def test_authorization_mismatch_refuses(observed, field):
    m = _authorized(observed) | {field: "WRONG"}
    v = C.validate_authorization(m, requested_scope="FULL_V2_3_PILOT", observed=observed)
    assert not v["authorized"] and any(field in f for f in v["failures"])


def test_scope_escalation_refused(observed):
    m = _authorized(observed, scope="PHASE_MB_ONLY")
    v = C.validate_authorization(m, requested_scope="FULL_V2_3_PILOT", observed=observed)
    assert not v["authorized"]
    assert "requested_scope_exceeds_authorization" in v["failures"]


def test_matching_authorized_fixture_passes_without_training(observed):
    """A synthetic matching AUTHORIZED manifest validates -- no candidate is invoked."""
    v = C.validate_authorization(
        _authorized(observed), requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert v["authorized"] is True and v["failures"] == []
    assert v["allowed_scope"] == "FULL_V2_3_PILOT"


def test_token_ceiling_above_contract_refused(observed):
    m = _authorized(observed) | {"pilot_trained_token_ceiling": 10**12}
    v = C.validate_authorization(m, requested_scope="FULL_V2_3_PILOT", observed=observed)
    assert "token_ceiling_invalid_or_above_contract" in v["failures"]


MEASURED_SECONDS = [0.25] * (C.MB_PROBE_UPDATES - C.MB_MEASURED_FIRST_UPDATE + 1)
MEASURED_TPS = C.EFFECTIVE_BATCH_TOKENS / 0.25


def _fake_session(tmp_path, scope="FULL_V2_3_PILOT", root=None):
    """A validated ExecutionSession assembled directly, as open_session would produce one.

    Building one by hand grants nothing: R2 moved the authority into
    validate_execution_artifacts, which every real execution root calls on artifact bytes. The
    tests below prove that separately; this fixture only avoids re-reading a 26 GB release.
    """
    root = (Path(root) if root is not None else (tmp_path / "root")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "contract_sha256": "c",
        "implementation_head": "h",
        "execution_bundle_sha256": "b",
        "pilot_index_manifest_file_sha256": "m",
        "authorization_sha256": "a",
        "session_id": "s",
        "authorized_output_root": str(root),
        "authorized_scope": scope,
    }
    validated = {
        "observed": {
            "branch": "agent/retrain-pipeline-contracts",
            "head": "h",
            "contract_sha256": "c",
            "execution_bundle_sha256": "b",
            "serialized_index_lists_digest": "i",
            "pilot_index_manifest_file_sha256": "m",
            "stage_a_meta_sha256": "sa",
            "stage_b_meta_sha256": "sb",
            "output_root": str(root),
        },
        "serialized_index_lists_digest": "i",
        "pilot_index_manifest_file_sha256": "m",
        "authorization_sha256": "a",
        "authorization_path": str(tmp_path / "authorization.json"),
        "index_manifest_path": str(tmp_path / R.PILOT_INDEX_MANIFEST_FILENAME),
        "session_id": "s",
        "scope": scope,
        "authorized_root": root,
        "phase": "MB",
        "indices": {
            "stage_a_eval": [0],
            "stage_a_train": [0],
            "stage_b_eval": [0],
            "stage_a_eval_sha256": "x",
            "stage_a_train_sha256": "y",
            "stage_b_eval_sha256": "z",
        },
        "fingerprint": {"fingerprint_sha256": "fp", "gpu": {"total_vram_bytes": VRAM_4090}},
        "stage_a": {"dataset": None},
        "stage_b": {"dataset": None},
        "effective_ceilings": _ceilings(),
        "identity": identity,
    }
    ledger = R.TokenLedger(root / R.LEDGER_FILENAME, identity, _ceilings())
    return R.ExecutionSession(validated, ledger)


def _bound(session, extra):
    payload = dict(extra)
    payload.update(dict(R._session_bindings(session)))  # noqa: SLF001
    payload["ledger_identity"] = dict(session.ledger.identity)
    payload["eligible"] = True
    payload["run_meta_sha256"] = "rm"
    return payload


def _bound_mb_result(candidate, session, **over):
    r = _bound(
        session,
        {
            "phase": "MB",
            "candidate_id": candidate["candidate_id"],
            "seed_label": candidate["seed_label"],
            "micro_bsz": candidate["micro_bsz"],
            "grad_accum": candidate["grad_accum"],
            "compile": candidate["compile"],
            "completed_updates": 40,
            "measured_update_wall_seconds": list(MEASURED_SECONDS),
            "median_tokens_per_sec": MEASURED_TPS,
            "max_memory_reserved_bytes": 8 * 1024**3,
            "oom": False,
            "uncontrolled_exception": False,
            "all_losses_finite": True,
            "all_grad_norms_finite": True,
            "all_optimizer_states_instantiated": True,
            "grouping_matches_contract": True,
            "all_lr_ratios_are_one": True,
            "canonical_compile_path": True,
            "output_dir": candidate["output_dir"],
        },
    )
    r.update(over)
    return r


def _bound_lr_result(candidate, session, score, **over):
    weight = 1024.0
    numerator = float(score) * weight
    loss = numerator / max(1.0, weight)
    r = _bound(
        session,
        {
            "phase": "LR",
            "candidate_id": candidate["candidate_id"],
            "peak_lr": candidate["peak_lr"],
            "seed_label": candidate["seed_label"],
            "completed_updates": 200,
            "all_losses_finite": True,
            "all_grad_norms_finite": True,
            "all_parameters_finite": True,
            "muon_momentum_states_present": True,
            "aux_adamw_states_present": True,
            "grouping_matches_contract": True,
            "all_lr_ratios_are_one": True,
            "eval_stage_a_numerator": numerator,
            "eval_stage_a_weight": weight,
            "eval_stage_b_numerator": numerator,
            "eval_stage_b_weight": weight,
            "eval_loss_stage_a": loss,
            "eval_loss_stage_b": loss,
            "score": C.lr_score(loss, loss),
            "sustained_divergence": False,
            "output_dir": candidate["output_dir"],
        },
    )
    r.update(over)
    return r


def _write_result(candidate, payload):
    out = Path(candidate["output_dir"])
    out.mkdir(parents=True)
    (out / "result.json").write_bytes(C.canonical_json_bytes(payload))


def test_candidate_functions_reject_a_plain_authorized_dict(tmp_path):
    """The `context['authorized'] = True` pattern is gone; a mapping is not a session."""
    candidate = R.plan_phase_mb(output_root=tmp_path / "root")[0]
    for fn in (R.run_phase_mb_candidate, R.run_phase_lr_candidate):
        with pytest.raises(C.PilotContractError, match="validated ExecutionSession"):
            fn(candidate, {"authorized": True, "train_view": None, "ledger": None})


def test_phase_mb_only_scope_cannot_run_lr(tmp_path):
    session = _fake_session(tmp_path, scope="PHASE_MB_ONLY")
    with pytest.raises(R.BindingFailure, match="PHASE_MB_ONLY"):
        R.orchestrate_phase_muon_lr(session, backend=lambda *_: None)


def test_orchestrator_runs_exactly_the_ten_candidates(tmp_path):
    """A fake backend records what the orchestrator chose to launch."""
    session = _fake_session(tmp_path)
    launched = []

    def backend(candidate, given):
        assert isinstance(given, R.ExecutionSession)
        launched.append(candidate["candidate_id"])
        _write_result(candidate, _bound_mb_result(candidate, given))

    report = R.orchestrate_phase_mb(session, backend=backend)
    assert launched == list(C.MB_REQUIRED_CANDIDATE_IDS)
    assert len(launched) == 10
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"


def test_candidate_local_failure_becomes_ineligible_evidence_and_grid_continues(tmp_path):
    session = _fake_session(tmp_path)
    launched = []

    def backend(candidate, given):
        launched.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "mb_micro16_compileoff":
            raise R.CandidateFailure("simulated candidate-local CUDA failure")
        _write_result(candidate, _bound_mb_result(candidate, given))

    report = R.orchestrate_phase_mb(session, backend=backend)
    assert len(launched) == 10, "the grid must continue after a candidate-local failure"
    failed = [c for c in report["candidates"] if c["candidate_id"] == "mb_micro16_compileoff"]
    assert failed and failed[0]["eligible"] is False
    assert failed[0]["reason"] == "candidate_runtime_exception"
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"


def test_phase_level_binding_failure_aborts(tmp_path):
    session = _fake_session(tmp_path)

    def backend(candidate, given):
        raise C.PilotContractError("accepted-release binding failed")

    with pytest.raises(R.PhaseAbort, match="accepted-release binding failed"):
        R.orchestrate_phase_mb(session, backend=backend)


def test_result_must_bind_this_execution(tmp_path):
    session = _fake_session(tmp_path)

    def backend(candidate, given):
        _write_result(candidate, _bound_mb_result(candidate, given, contract_sha256="WRONG"))

    with pytest.raises(R.BindingFailure, match="does not bind this execution"):
        R.orchestrate_phase_mb(session, backend=backend)


def _frozen_mb_report(session, micro_bsz=8, compile_on=False):
    """Publish an authoritative Phase-MB report that freezes a specific geometry."""

    def backend(candidate, given):
        fast = candidate["micro_bsz"] == micro_bsz and candidate["compile"] is compile_on
        seconds = [0.25 if fast else 1.0] * len(MEASURED_SECONDS)
        _write_result(
            candidate,
            _bound_mb_result(
                candidate,
                given,
                measured_update_wall_seconds=seconds,
                median_tokens_per_sec=C.EFFECTIVE_BATCH_TOKENS / seconds[0],
            ),
        )

    return R.orchestrate_phase_mb(session, backend=backend)


def test_lr_orchestrator_derives_grid_confirmation_and_edge(tmp_path):
    """Nothing about which LR candidates run, or with what geometry, comes from caller input."""
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=8, compile_on=False)
    launched = []
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}

    def backend(candidate, given):
        launched.append((candidate["peak_lr"], candidate["seed_label"]))
        assert candidate["micro_bsz"] == 8 and candidate["compile"] is False
        _write_result(candidate, _bound_lr_result(candidate, given, scores[candidate["peak_lr"]]))

    report = R.orchestrate_phase_muon_lr(session, backend=backend)
    seed1_lrs = sorted(lr for lr, seed in launched if seed == "seed-1" and lr in C.LR_GRID_SEED1)
    assert seed1_lrs == sorted(C.LR_GRID_SEED1), "initial grid derived internally"
    # winner is 2e-4 (lowest score) -> confirmation neighbour is the HIGHER 3e-4
    seed2_lrs = sorted({lr for lr, seed in launched if seed == "seed-2"})
    assert 2e-4 in seed2_lrs and 3e-4 in seed2_lrs
    # 2e-4 confirmed -> edge 1e-4 must have run under BOTH seeds
    assert (1e-4, "seed-1") in launched and (1e-4, "seed-2") in launched
    assert report["final"]["second_expansion_permitted"] is False
    assert report["terminal_status"] == "SUCCESS"


def test_runner_run_subcommand_refuses_without_authorization(tmp_path):
    """No authorization manifest exists in this repository, so `run` cannot start."""
    code = R.main([
        "run",
        "--phase",
        "MB",
        "--authorization",
        str(tmp_path / "absent.json"),
        "--pilot-index-manifest",
        str(tmp_path / "absent_indices.json"),
        "--output-root",
        str(tmp_path / "x"),
    ])
    assert code == R.BINDING_FAILURE


# ------------------------------------------------------------------ checkpoint isolation


def test_pilot_checkpointing_is_disabled():
    assert C.PILOT_CHECKPOINTING == "DISABLED"
    policy = C.CHECKPOINT_ISOLATION
    assert policy["executor_creates_training_checkpoints"] is False
    assert policy["executor_accepts_pilot_resume"] is False
    assert policy["stage_n_o_consumer_integration_required"] is False
    for action in ("save a pilot checkpoint", "resume from a pilot checkpoint"):
        with pytest.raises(C.PilotContractError, match="PILOT_CHECKPOINTING=DISABLED"):
            C.require_checkpointing_disabled(action)
    with pytest.raises(C.PilotContractError, match="PILOT_CHECKPOINTING=DISABLED"):
        R.require_checkpointing_disabled("resume")


def test_no_pilot_checkpoint_save_or_resume_path_exists():
    """The executor exposes no checkpoint save/resume entry point and no resume CLI option."""
    import inspect

    assert not hasattr(R, "resume_pilot_checkpoint")
    assert not hasattr(R, "pilot_checkpoint_identity")
    assert not hasattr(C, "check_pilot_resume")
    src = inspect.getsource(R)
    assert "torch.save" not in src
    assert "--resume" not in src


# ------------------------------------------------------------------ output isolation


def test_output_dir_new_and_outside_releases(tmp_path):
    assert R.require_new_output_dir(tmp_path / "fresh")
    existing = tmp_path / "e"
    existing.mkdir()
    with pytest.raises(C.PilotContractError, match="must not exist"):
        R.require_new_output_dir(existing)


@pytest.mark.parametrize(
    "bad",
    [
        "runs/m_production_v1_2026-08-29/release/stage_a/x",
        "runs/m_production_v1_2026-08-29/release",
        "runs/i_production_v1_2026-08-25/y",
        "runs/g_production_2026-08-21/release/z",
    ],
)
def test_accepted_release_paths_rejected(bad):
    with pytest.raises(C.PilotContractError, match="accepted release"):
        R.require_new_output_dir(Path(bad))


def test_every_planned_candidate_has_its_own_output(tmp_path):
    mb = R.plan_phase_mb(output_root=tmp_path / "mb")
    assert len({c["output_dir"] for c in mb}) == len(mb) == 10
    lr = R.plan_phase_lr(output_root=tmp_path / "lr", micro_bsz=8, compile_on=False)
    assert len({c["output_dir"] for c in lr}) == len(lr) == 3
    for c in mb + lr:
        assert c["authorization_status"] == "NOT_AUTHORIZED"


def test_repo_root_cwd_independent(tmp_path, monkeypatch):
    before = R.repo_root()
    monkeypatch.chdir(tmp_path)
    assert R.repo_root() == before == REPO_ROOT


# ------------------------------------------------------------------ git policy


def test_allowed_probe_identity():
    assert R.ALLOWED_UNTRACKED == ".codex_r1_manual_context_probe.py"
    assert R.file_sha256(REPO_ROOT / R.ALLOWED_UNTRACKED) == R.ALLOWED_UNTRACKED_SHA256


def test_changed_probe_bytes_rejected(monkeypatch):
    monkeypatch.setattr(R, "ALLOWED_UNTRACKED_SHA256", "0" * 64)
    assert "allowed_historical_untracked_file_bytes_changed" in R.git_policy_status()["failures"]


def test_extra_untracked_source_rejected(monkeypatch):
    real = R._git

    def fake(*args):
        if args == ("status", "--porcelain"):
            return "?? .codex_r1_manual_context_probe.py\n?? pretrain/rogue.py"
        return real(*args)

    monkeypatch.setattr(R, "_git", fake)
    s = R.git_policy_status()
    assert s["unexpected_untracked"] == ["pretrain/rogue.py"]
    assert "uncontrolled_untracked_files_present" in s["failures"]


# ------------------------------------------------------------------ fingerprint / closure


def test_fingerprint_separates_base_from_per_run():
    fp = R.base_runtime_fingerprint()
    for per_run in ("compile", "micro_bsz", "grad_accum", "peak_lr", "seed", "phase"):
        assert per_run not in fp
    for required in (
        "gpu",
        "torch_version",
        "numpy_version",
        "python_executable",
        "contract_sha256",
        "execution_implementation_bundle_sha256",
        "fingerprint_sha256",
    ):
        assert required in fp


def test_execution_closure_includes_real_dependencies():
    closure = R.execution_closure()
    for required in (
        "src/model.py",
        "src/optim.py",
        "pretrain/dataset_pretrain.py",
        "src/special_tokens.py",
        "pretrain/pilot_contract_v2_3.py",
        "pretrain/pilot_runner_v2_3.py",
    ):
        assert required in closure["derived_closure"], required
    assert closure["unbound_load_bearing_module_count"] == 0
    assert closure["derived_closure_count"] > 4, "must not be the old 4-file preflight closure"
    assert len(closure["EXECUTION_IMPLEMENTATION_BUNDLE_SHA256"]) == 64


# ------------------------------------------------------------------ schedule semantics


def test_planner_geometry_matches_expected():
    g = C.expected_planner_geometry(A_BLOCKS, B_BLOCKS)
    assert g["schedule_total_steps"] == 49590
    assert g["decay_steps"] == 4959
    assert (g["decay_start_step"], g["decay_end_step"]) == (44631, 49590)
    assert g["matches_expected"] is True


def test_final_lr_discrete_semantics():
    """Cosine parity, and the deliberate endpoint-vs-last-applied distinction."""
    peak = 4e-4
    s = C.final_lr_semantics(
        peak_lr=peak, warmup_steps=500, decay_start_step=44631, decay_end_step=49590
    )
    assert s["PRODUCTION_MIN_LR_INTENT_RATIO"] == 0.10
    assert s["mathematical_endpoint_step"] == 49590
    assert s["mathematical_endpoint_ratio"] == 0.10
    assert s["last_applied_optimizer_update"] == 49589
    # The frozen expected values: exactly 0.10 at the endpoint, ~0.10000009 one step before.
    assert s["last_applied_ratio"] == pytest.approx(0.10000009030130619, rel=1e-12)
    assert s["last_applied_ratio"] > s["mathematical_endpoint_ratio"]
    assert "cosine" in s["schedule_family"]


def test_wsd_helper_matches_the_production_scheduler_exactly():
    """The V2.3 helper delegates to the SAME pure function the trainer imports."""
    from src.canonical_schedule import lr_schedule as canonical

    kw = {"peak_lr": 4e-4, "warmup_steps": 500, "decay_start_step": 44631, "decay_end_step": 49590}
    for step in (0, 1, 250, 499, 500, 1000, 44630, 44631, 44632, 49588, 49589, 49590):
        assert C.wsd_lr(step, **kw) == canonical(
            step,
            500,
            4e-4,
            schedule="wsd",
            schedule_total_steps=49590,
            decay_start_step=44631,
            decay_end_step=49590,
            min_lr_ratio=0.1,
        )


def test_production_trainer_imports_the_same_schedule():
    import importlib.util
    import sys as _sys

    from src.canonical_schedule import lr_schedule as shared

    _sys.path.insert(0, "pretrain")
    spec = importlib.util.spec_from_file_location(
        "_tpwb_parity", "pretrain/train_pretrain_with_bench.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.lr_schedule is shared


def test_wsd_lr_shape():
    kw = {"peak_lr": 4e-4, "warmup_steps": 500, "decay_start_step": 44631, "decay_end_step": 49590}
    assert C.wsd_lr(499, **kw) == pytest.approx(4e-4)
    assert C.wsd_lr(1000, **kw) == pytest.approx(4e-4)
    assert C.wsd_lr(44631, **kw) == pytest.approx(4e-4)
    assert C.wsd_lr(49590, **kw) == pytest.approx(4e-5)


def test_frozen_warmup_is_convention():
    assert C.FROZEN_WARMUP_STEPS == 500
    assert "NOT pilot-derived" in C.FROZEN_WARMUP_STEPS_AUTHORITY


def test_planner_decay_geometry_no_regression(tmp_path, monkeypatch):
    """The real planner still reproduces the frozen decay relationship."""
    import pretrain.plan_pretrain_run as planner
    from tests.test_plan_pretrain_run import _write_full_provenance, _write_stage

    a = _write_stage(tmp_path, "pa", [list(range(3))] * 80)
    b = _write_stage(tmp_path, "pb", [list(range(3))] * 48)
    prov = _write_full_provenance(tmp_path, a, b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda p: None)
    plan = planner.build_run_plan(
        stage_a_dir=a,
        stage_b_dir=b,
        seq_len=2,
        micro_bsz=1,
        grad_accum=2,
        warmup_steps=1,
        decay_fraction=C.PLANNER_DECAY_FRACTION_INPUT,
        reference_val_dir=prov["reference_val_dir"],
        tokenizer_release_manifest=prov["tokenizer_release_manifest"],
        selection_manifest=prov["selection_manifest"],
    )
    assert plan["wsd_candidate"]["decay_end_step"] == plan["boundaries"]["schedule_total_steps"]
    assert plan["wsd_candidate"]["decay_start_step"] >= plan["boundaries"]["stage_b_start_step"]


# ------------------------------------------------------------------ release binding


def test_release_identity_derived_not_supplied():
    """Universes come from the accepted releases, never from caller-supplied numbers."""
    import inspect

    src = inspect.getsource(R.validate_execution_artifacts)
    assert "verify_accepted_release" in src
    assert "verify_accepted_release" in inspect.getsource(R.derive_universes)
    sig = inspect.signature(R.derive_universes)
    assert not sig.parameters, "derive_universes must take no caller-supplied counts"


def test_accepted_release_constants_match_owner_values():
    assert R.ACCEPTED_STAGE_A_META_SHA256 == (
        "334564305f0b5bb058ff4fda9b2f13a9fba01046818ed3600ce2a6ca3cc5c81c"
    )
    assert R.ACCEPTED_STAGE_B_META_SHA256 == (
        "fe634de7690a1ab56bb7e478b161eb4916724ac648ba09338b55f044fa6e0a2e"
    )


# ------------------------------------------------------------------ R1: shared canonical loss


def test_canonical_loss_is_shared_by_trainer_and_pilot():
    """Both import the SAME implementation; the pilot no longer pulls in the whole trainer."""
    import importlib.util
    import inspect
    import sys as _sys

    from src.canonical_loss import masked_weighted_ce_components, masked_weighted_ce_loss

    _sys.path.insert(0, "pretrain")
    spec = importlib.util.spec_from_file_location(
        "_tpwb_loss_parity", "pretrain/train_pretrain_with_bench.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.masked_weighted_ce_components is masked_weighted_ce_components
    assert mod.masked_weighted_ce_loss is masked_weighted_ce_loss
    # The pilot imports the shared module, not the trainer. Checked at the AST import level
    # so a docstring mentioning the trainer cannot fail (or silently pass) this test.
    import ast

    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "src.canonical_loss" in imported
    assert not any("train_pretrain_with_bench" in m for m in imported)
    assert inspect.getsource(R.canonical_loss_components).count("canonical_loss") >= 1


def test_canonical_loss_parity_with_previous_production_behaviour():
    """Pin the extracted primitives against an independent reference computation."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    from src.canonical_loss import masked_weighted_ce_components, masked_weighted_ce_loss

    torch.manual_seed(7)
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 0.0]])
    per_token = F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1), reduction="none")
    expected_num = (per_token * mask.reshape(-1)).sum()
    expected_w = mask.sum()
    num, w = masked_weighted_ce_components(logits, labels, mask, eos_id=3, eos_weight=1.0)
    assert torch.allclose(num, expected_num)
    assert torch.allclose(w, expected_w)
    assert torch.allclose(
        masked_weighted_ce_loss(logits, labels, mask, eos_id=3, eos_weight=1.0),
        expected_num / expected_w,
    )


def test_global_token_mean_rejects_batch_mean_averaging():
    """With unequal per-batch weights the two quantities genuinely differ."""
    components = [(10.0, 2.0), (30.0, 10.0)]  # batch means 5.0 and 3.0
    global_mean = R.global_token_mean(components)
    batch_mean_of_means = (10.0 / 2.0 + 30.0 / 10.0) / 2
    assert global_mean == pytest.approx(40.0 / 12.0)
    assert batch_mean_of_means == pytest.approx(4.0)
    assert global_mean != pytest.approx(batch_mean_of_means)
    import inspect

    src = inspect.getsource(R.evaluate)
    assert "total_numerator" in src and "total_weight" in src
    assert "/ max(1.0, total_weight)" in src


# ------------------------------------------------------------------ R1: dtype + ordering


def test_int16_packed_ids_are_converted_for_embedding_and_ce():
    """Canonical packed storage surfaces as int16; the model needs long index tensors."""
    import torch

    from src.model import GPT, GPTConfig

    packed = torch.zeros(2, 4, dtype=torch.int16)
    converted = R.to_model_ids(packed, "cpu")
    assert converted.dtype == torch.long
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            vocab_size=32, n_layers=1, d_model=16, n_heads=2, n_kv_heads=1, d_ff=32, max_seq_len=8
        )
    )
    with torch.no_grad():
        logits = model(converted)  # would raise on an int16 index tensor
    assert logits.shape == (2, 4, 32)
    labels = R.to_model_ids(torch.zeros(2, 4, dtype=torch.int16), "cpu")
    mask = torch.ones(2, 4)
    num, w = R.canonical_loss_components(logits, labels, mask)
    assert torch.isfinite(num) and float(w) == 8.0


class _FakeDataset:
    """Minimal stand-in for PackedBinDataset: returns (ids, labels, mask) at int16 storage."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch

        return (
            torch.full((4,), i % 7, dtype=torch.int16),
            torch.full((4,), i % 5, dtype=torch.int16),
            torch.ones(4, dtype=torch.float32),
        )


def test_index_view_consumes_the_fixed_order_without_wrap():
    order = [5, 3, 9, 1]
    view = R.IndexView(_FakeDataset(20), order)
    assert len(view) == 4
    assert [int(view[i][0][0]) for i in range(4)] == [v % 7 for v in order]
    with pytest.raises(C.PilotContractError, match="no replay or wrap"):
        view[4]


def test_exact_block_consumption_mb_and_lr():
    assert C.MB_PROBE_UPDATES * C.SEQUENCES_PER_OPTIMIZER_UPDATE == 5120
    assert C.LR_RUN_UPDATES * C.SEQUENCES_PER_OPTIMIZER_UPDATE == 25600
    assert C.LR_BLOCKS_PER_RUN == 25600
    import inspect

    src = inspect.getsource(R._run_updates)  # noqa: SLF001
    assert "required_blocks" in src
    assert "no replay" in src or "without replay" in src
    assert "contract requires exactly" in src


def test_run_updates_enforces_exact_consumption_and_ledger(tmp_path):
    """Drive the real update loop on CPU with fakes: no model, no optimizer step."""
    import torch

    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    view = R.IndexView(_FakeDataset(64), list(range(8)))
    with pytest.raises(C.PilotContractError, match="without replay"):
        R._run_updates(  # noqa: SLF001
            model=mock.Mock(),
            optimizer=mock.Mock(),
            view=view,
            micro_bsz=2,
            grad_accum=64,
            updates=40,
            lr_fn=C.mb_lr,
            ledger=ledger,
            phase="MB",
            device="cpu",
        )
    assert ledger.state["reserved_updates"] == 0, "a refused run must not reserve tokens"
    assert ledger.state["completed_updates"] == 0
    assert torch is not None


# ------------------------------------------------------------------ R1: output containment


def test_candidate_output_must_resolve_beneath_the_authorized_root(tmp_path):
    root = (tmp_path / "authorized").resolve()
    root.mkdir()
    assert R.require_candidate_output_dir(root / "cand", root)
    outside = (tmp_path / "elsewhere").resolve()
    with pytest.raises(C.PilotContractError, match="beneath the authorized root"):
        R.require_candidate_output_dir(outside / "cand", root)
    # traversal that escapes by resolution, not by string prefix
    with pytest.raises(C.PilotContractError, match="beneath the authorized root"):
        R.require_candidate_output_dir(root / ".." / "escape", root)
    existing = root / "already"
    existing.mkdir()
    with pytest.raises(C.PilotContractError, match="must not exist"):
        R.require_candidate_output_dir(existing, root)


def test_containment_uses_resolved_paths_not_string_prefixes(tmp_path):
    root = (tmp_path / "root").resolve()
    root.mkdir()
    sibling = (tmp_path / "root_evil").resolve()
    sibling.mkdir()
    # "root_evil" starts with the string "root" but is not contained in it
    with pytest.raises(C.PilotContractError, match="beneath the authorized root"):
        R.require_candidate_output_dir(sibling / "c", root)


# ------------------------------------------------------------------ R1: closure


def test_execution_closure_is_derived_and_minimal():
    closure = R.execution_closure()
    derived = closure["derived_closure"]
    for required in (
        "pretrain/pilot_contract_v2_3.py",
        "pretrain/pilot_runner_v2_3.py",
        "src/model.py",
        "src/optim.py",
        "pretrain/dataset_pretrain.py",
        "src/special_tokens.py",
        "src/canonical_loss.py",
        "src/canonical_schedule.py",
    ):
        assert required in derived, required
    # the shared-loss refactor removed the broad trainer dependency
    assert "pretrain/train_pretrain_with_bench.py" not in derived
    assert closure["unbound_load_bearing_module_count"] == 0
    assert len(derived) == closure["derived_closure_count"]
    assert len(set(derived)) == len(derived)


# =========================================================== R2 Part 1: one validation gate


def _runner_ast():
    import ast

    return ast.parse(Path(R.__file__).read_text(encoding="utf-8"))


def _enclosing_functions(tree, predicate):
    """Names of the functions whose body contains a node matching `predicate`."""
    import ast

    hits = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if predicate(inner):
                    hits.add(node.name)
    return hits


def test_every_backend_path_passes_through_the_shared_artifact_validator():
    """R2 Part 1: the real training backends are reachable only from the validated worker."""
    import ast

    tree = _runner_ast()
    # The backends are selected by name and then invoked, so a reference is the reachable edge.
    callers = _enclosing_functions(
        tree,
        lambda n: (
            isinstance(n, ast.Name) and n.id in ("run_phase_mb_candidate", "run_phase_lr_candidate")
        ),
    )
    assert callers == {"_cli_execute_candidate"}, callers
    worker = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_cli_execute_candidate"
    )
    assert any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open_session"
        for n in ast.walk(worker)
    )
    opener = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "open_session"
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "validate_execution_artifacts"
        for n in ast.walk(opener)
    )


def test_the_validator_accepts_only_artifact_paths_and_the_requested_phase():
    """No caller-supplied manifest object, hash, count, authority or context is accepted."""
    import inspect

    params = inspect.signature(R.validate_execution_artifacts).parameters
    assert set(params) == {
        "authorization_path",
        "candidate_spec_path",
        "pilot_index_manifest_path",
        "output_dir",
        "requested_phase",
        "gpu_required",
    }
    for name in ("authorized", "manifest", "context", "stage_a_blocks", "expected_sha256"):
        assert name not in params


def test_the_old_trust_in_context_api_is_gone():
    for removed in ("build_validated_context", "ValidatedExecutionContext"):
        assert not hasattr(R, removed), removed
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "manifest_path=None" not in src
    assert '"authorized"' not in src or 'context["authorized"]' not in src


def test_validator_refuses_when_the_authorization_file_is_absent(tmp_path):
    with pytest.raises(R.BindingFailure, match="NOT_AUTHORIZED"):
        R.validate_execution_artifacts(
            authorization_path=tmp_path / "absent.json",
            candidate_spec_path=None,
            pilot_index_manifest_path=tmp_path / "absent_indices.json",
            output_dir=tmp_path / "out",
            requested_phase="MB",
            gpu_required=False,
        )


def test_binding_failure_is_never_downgraded_to_an_ineligible_candidate(tmp_path):
    session = _fake_session(tmp_path)

    def backend(candidate, given):
        raise R.BindingFailure("forged release identity")

    with pytest.raises(R.BindingFailure, match="forged release identity"):
        R.orchestrate_phase_mb(session, backend=backend)


# ==================================================== R2 Part 2: real paths reach the child


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_subprocess(monkeypatch, handler):
    import types

    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return handler(list(argv), kwargs)

    monkeypatch.setattr(R, "subprocess", types.SimpleNamespace(run=run))
    return calls


def test_parent_hands_the_child_the_real_manifest_spec_and_index_paths(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]

    def handler(argv, kwargs):
        spec_path = Path(argv[argv.index("--spec") + 1])
        R.write_terminal_result(
            spec_path,
            {
                "terminal_status": "SUCCESS",
                "error_class": None,
                "error_message": None,
                "completed_updates": 40,
                "reserved_tokens": {},
                "completed_tokens": {},
                "run_meta_sha256": "rm",
                "candidate_id": candidate["candidate_id"],
                "phase": "MB",
            },
        )
        return _FakeCompleted(0, "out", "err")

    calls = _fake_subprocess(monkeypatch, handler)
    R._subprocess_backend(candidate, session)  # noqa: SLF001
    argv = calls[0][0]
    assert argv[argv.index("--authorization") + 1] == session.validated["authorization_path"]
    assert (
        argv[argv.index("--pilot-index-manifest") + 1] == session.validated["index_manifest_path"]
    )
    spec = json.loads(Path(argv[argv.index("--spec") + 1]).read_text(encoding="utf-8"))
    assert spec["authorization_path"] == session.validated["authorization_path"]
    assert spec["pilot_index_manifest_path"] == session.validated["index_manifest_path"]
    # The spec carries no hashes, counts or authority the child could take on trust.
    assert set(spec) == {
        "candidate",
        "authorization_path",
        "pilot_index_manifest_path",
        "authorized_output_root",
        "session_id",
    }


def test_the_child_cli_requires_both_artifact_paths(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        R.main(["execute-candidate", "--spec", str(tmp_path / "s.json")])
    assert excinfo.value.code == 2


def test_child_refuses_a_spec_whose_paths_disagree_with_its_launch(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_bytes(
        C.canonical_json_bytes({
            "candidate": {"phase": "MB", "candidate_id": "mb_micro8_compileoff"},
            "authorization_path": "/somewhere/else.json",
            "pilot_index_manifest_path": str(tmp_path / "PILOT_INDICES.json"),
            "authorized_output_root": str(tmp_path / "root"),
            "session_id": "s",
        })
    )
    code = R.main([
        "execute-candidate",
        "--spec",
        str(spec),
        "--authorization",
        str(tmp_path / "auth.json"),
        "--pilot-index-manifest",
        str(tmp_path / "PILOT_INDICES.json"),
    ])
    assert code == R.BINDING_FAILURE
    terminal = json.loads(R.terminal_result_path(spec).read_text(encoding="utf-8"))
    assert terminal["terminal_status"] == "BINDING_FAILURE"
    assert "does not match the path this process was launched with" in terminal["error_message"]


# ================================================= R2 Part 3: session and scope semantics


def test_scope_is_derived_from_the_requested_phase(tmp_path):
    assert R._scope_for("MB") == "PHASE_MB_ONLY"  # noqa: SLF001
    assert R._scope_for("LR") == "FULL_V2_3_PILOT"  # noqa: SLF001


def test_phase_mb_only_session_terminates_after_its_report(tmp_path):
    session = _fake_session(tmp_path, scope="PHASE_MB_ONLY")
    report = _frozen_mb_report(session)
    assert report["session_terminates_after_this_phase"] is True
    assert report["next_phase_requires_new_authorization"] is True
    with pytest.raises(R.BindingFailure, match="never promoted"):
        R.orchestrate_phase_muon_lr(session, backend=lambda *_: None)


def test_full_pilot_runs_both_phases_under_one_session_and_ledger(tmp_path):
    session = _fake_session(tmp_path, scope="FULL_V2_3_PILOT")
    report = _frozen_mb_report(session)
    assert report["session_terminates_after_this_phase"] is False
    frozen = R.load_authoritative_mb_report(session)
    assert frozen["micro_bsz"] == 8
    assert report["session_id"] == session.session_id
    assert session.ledger.identity["session_id"] == session.session_id


def test_a_report_from_another_session_is_refused(tmp_path):
    first = _fake_session(tmp_path, scope="FULL_V2_3_PILOT")
    _frozen_mb_report(first)
    other = _fake_session(tmp_path, scope="FULL_V2_3_PILOT")
    other.validated["session_id"] = "DIFFERENT"
    other.validated["authorization_sha256"] = "DIFFERENT"
    with pytest.raises(R.BindingFailure, match="does not bind this execution"):
        R.load_authoritative_mb_report(other)


def test_a_new_authorization_may_not_reuse_an_existing_session_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / R.SESSION_FILENAME).write_bytes(
        C.canonical_json_bytes({"session_id": "OLD", "scope": "PHASE_MB_ONLY"})
    )
    validated = _fake_session(tmp_path, root=tmp_path / "other").validated
    validated["authorized_root"] = root.resolve()
    monkeypatch.setattr(R, "validate_execution_artifacts", lambda **kw: validated)
    with pytest.raises(R.BindingFailure, match="different session"):
        R.open_session(
            authorization_path=tmp_path / "a.json",
            candidate_spec_path=None,
            pilot_index_manifest_path=tmp_path / "i.json",
            output_dir=root,
            phase="MB",
            gpu_required=False,
        )


# ================================================== R2 Part 4: result classes and exit codes


def test_four_result_classes_map_to_process_exit_codes():
    assert dict(R.RESULT_CLASSES) == {
        "SUCCESS": 0,
        "CANDIDATE_INELIGIBLE": 3,
        "PHASE_ABORT": 4,
        "BINDING_FAILURE": 5,
    }
    assert (R.SUCCESS, R.CANDIDATE_INELIGIBLE, R.PHASE_ABORT, R.BINDING_FAILURE) == (0, 3, 4, 5)
    assert set(R.TERMINAL_STATUSES) == set(R.RESULT_CLASSES)


def test_result_classes_subcommand_reports_the_mapping(capsys):
    assert R.main(["result-classes"]) == 0
    assert json.loads(capsys.readouterr().out) == dict(R.RESULT_CLASSES)


def test_phase_abort_when_no_candidate_is_eligible(tmp_path):
    session = _fake_session(tmp_path)

    def backend(candidate, given):
        raise R.CandidateFailure("CUDA out of memory")

    report = R.orchestrate_phase_mb(session, backend=backend)
    assert report["selection"]["outcome"] == "PHASE_MB_ABORT"
    assert all(c["reason"] == "oom" for c in report["candidates"])
    with pytest.raises(R.PhaseAbort, match="did not freeze a geometry"):
        R.load_authoritative_mb_report(session)


def test_exception_hierarchy_separates_binding_phase_and_candidate_failures():
    assert issubclass(R.BindingFailure, C.PilotContractError)
    assert issubclass(R.PhaseAbort, C.PilotContractError)
    assert not issubclass(R.CandidateFailure, C.PilotContractError)


# ================================================== R2 Part 5: reserve before every update


class _TinyLM:
    """A minimal real module so the update loop runs end to end on CPU."""

    def __new__(cls):
        import torch

        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(8, 8)

            def forward(self, x):
                return self.emb(x)

        return _M()


def _observing_optimizer(model, ledger, observations):
    import torch

    class _Observing(torch.optim.SGD):
        def step(self, *args, **kwargs):
            state = json.loads(ledger.path.read_text(encoding="utf-8"))
            observations.append((state["reserved_updates"], state["completed_updates"]))
            return super().step(*args, **kwargs)

    return _Observing(model.parameters(), lr=0.0)


def test_the_ledger_reserves_before_the_optimizer_update_and_completes_after(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    model = _TinyLM()
    observations = []
    optimizer = _observing_optimizer(model, ledger, observations)
    view = R.IndexView(_FakeDataset(300), list(range(256)))
    result = R._run_updates(  # noqa: SLF001
        model=model,
        optimizer=optimizer,
        view=view,
        micro_bsz=2,
        grad_accum=64,
        updates=2,
        lr_fn=C.mb_lr,
        ledger=ledger,
        phase="MB",
        device="cpu",
    )
    assert result["completed_updates"] == 2
    assert observations == [(1, 0), (2, 1)], observations
    assert ledger.state["reserved_updates"] == ledger.state["completed_updates"] == 2
    assert ledger.state["reserved_tokens"]["MB"] == 2 * C.EFFECTIVE_BATCH_TOKENS
    assert ledger.state["completed_tokens"]["GLOBAL"] == 2 * C.EFFECTIVE_BATCH_TOKENS


def test_completed_tokens_can_never_exceed_reserved(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    with pytest.raises(C.PilotContractError, match="reserve must precede"):
        ledger.complete("MB")


def test_a_crashed_candidate_does_not_hand_reserved_budget_back(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.reserve("MB")  # the process dies here, before the update finishes
    reopened = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    assert reopened.state["reserved_tokens"]["MB"] == C.EFFECTIVE_BATCH_TOKENS
    assert reopened.state["completed_tokens"]["MB"] == 0
    assert reopened.state["reserved_updates"] == 1


# ============================== R2 Part 6: the authoritative, immutable Phase-MB report


def test_the_phase_mb_report_is_published_once(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    path = session.output_root / R.MB_REPORT_FILENAME
    assert path.is_file() and path.with_suffix(".sha256").is_file()
    with pytest.raises(R.BindingFailure, match="immutable"):
        R.write_immutable_report(path, {"anything": 1})


def test_phase_lr_geometry_comes_only_from_the_verified_report(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=4, compile_on=True)
    frozen = R.load_authoritative_mb_report(session)
    assert (frozen["micro_bsz"], frozen["compile"]) == (4, True)
    assert frozen["grad_accum"] == C.frozen_grad_accum(4)
    launched = []

    def backend(candidate, given):
        launched.append((candidate["micro_bsz"], candidate["compile"]))
        _write_result(candidate, _bound_lr_result(candidate, given, 3.0))

    R.orchestrate_phase_muon_lr(session, backend=backend)
    assert set(launched) == {(4, True)}


def _run_subcommand_options():
    """The option strings the `run` subparser declares, read from main() in source order."""
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(R.main)).body[0]
    options, seen_run = set(), False
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and [
            t.id for t in stmt.targets if isinstance(t, ast.Name)
        ] == ["r"]:
            seen_run = True
            continue
        if not seen_run:
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                and getattr(call.func.value, "id", "") == "r"
            ):
                for arg in call.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        options.add(arg.value)
    assert options, "the run subcommand must declare options"
    return options


def test_run_subcommand_requires_both_artifact_paths():
    options = _run_subcommand_options()
    assert {"--phase", "--authorization", "--pilot-index-manifest", "--output-root"} <= options
    assert "--micro-bsz" not in options and "--compile" not in options


def test_lr_refuses_without_an_authoritative_report(tmp_path):
    session = _fake_session(tmp_path)
    with pytest.raises(R.BindingFailure, match="never taken from the command line"):
        R.orchestrate_phase_muon_lr(session, backend=lambda *_: None)


def test_lr_refuses_a_report_whose_bytes_changed(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    path = session.output_root / R.MB_REPORT_FILENAME
    path.write_bytes(path.read_bytes().replace(b'"phase":"MB"', b'"phase":"mb"'))
    with pytest.raises(R.BindingFailure, match="does not match its published sidecar"):
        R.load_authoritative_mb_report(session)


def test_lr_refuses_a_report_whose_selection_was_forged(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=8, compile_on=False)
    path = session.output_root / R.MB_REPORT_FILENAME
    report = json.loads(path.read_text(encoding="utf-8"))
    report["selection"] = dict(report["selection"]) | {"FROZEN_MICRO_BSZ": 1}
    body = C.canonical_json_bytes(report)
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(body).hexdigest()}  {path.name}\n", encoding="utf-8"
    )
    with pytest.raises(R.BindingFailure, match="does not reproduce the published selection"):
        R.load_authoritative_mb_report(session)


def test_lr_refuses_a_report_with_an_incomplete_grid(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    path = session.output_root / R.MB_REPORT_FILENAME
    report = json.loads(path.read_text(encoding="utf-8"))
    report["candidates"] = report["candidates"][:9]
    body = C.canonical_json_bytes(report)
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(body).hexdigest()}  {path.name}\n", encoding="utf-8"
    )
    with pytest.raises(C.PilotContractError, match="grid incomplete"):
        R.load_authoritative_mb_report(session)


# ================================ R2 Part 7: every selection number is recomputed from evidence


def test_mb_throughput_is_recomputed_from_the_raw_per_update_timings():
    seconds = [0.2, 0.4, 0.3]
    assert R.median_tokens_per_sec(seconds) == C.EFFECTIVE_BATCH_TOKENS / 0.3


def test_mb_recomputation_rejects_a_forged_median(tmp_path):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]
    forged = _bound_mb_result(candidate, session, median_tokens_per_sec=1e9)
    with pytest.raises(R.BindingFailure, match="disagrees with the value recomputed"):
        R.verify_recomputed_mb_result(forged)


def test_mb_recomputation_rejects_the_wrong_number_of_measured_updates(tmp_path):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]
    short = _bound_mb_result(
        candidate,
        session,
        measured_update_wall_seconds=[0.25] * 5,
        median_tokens_per_sec=C.EFFECTIVE_BATCH_TOKENS / 0.25,
    )
    with pytest.raises(R.BindingFailure, match="the contract measures exactly"):
        R.verify_recomputed_mb_result(short)


def test_mb_recomputation_rejects_missing_timings(tmp_path):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]
    with pytest.raises(C.PilotContractError, match="no measured per-update wall timings"):
        R.verify_recomputed_mb_result(
            _bound_mb_result(candidate, session, measured_update_wall_seconds=[])
        )


def test_lr_score_is_recomputed_from_the_raw_evaluation_components(tmp_path):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_lr(
        output_root=session.output_root, micro_bsz=8, compile_on=False, peak_lrs=[2e-4]
    )[0]
    good = _bound_lr_result(candidate, session, 3.25)
    recomputed = R.verify_recomputed_lr_result(good)
    assert recomputed["score"] == good["score"]
    assert recomputed["score_weights"] == [10, 3]


@pytest.mark.parametrize(
    "over,message",
    [
        ({"score": 0.001}, "serialized score"),
        ({"eval_loss_stage_a": 0.001}, "eval_loss_stage_a"),
        ({"eval_loss_stage_b": 0.001}, "eval_loss_stage_b"),
        ({"eval_stage_a_numerator": None}, "raw stage-a eval numerator"),
        ({"eval_stage_b_weight": 0.0}, "positive stage-b eval weight"),
    ],
)
def test_lr_recomputation_rejects_disagreeing_or_missing_evidence(tmp_path, over, message):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_lr(
        output_root=session.output_root, micro_bsz=8, compile_on=False, peak_lrs=[2e-4]
    )[0]
    forged = _bound_lr_result(candidate, session, 3.25)
    forged.update(over)
    with pytest.raises(R.BindingFailure, match=message):
        R.verify_recomputed_lr_result(forged)


def test_a_forged_result_is_caught_by_the_orchestrator_not_only_in_isolation(tmp_path):
    session = _fake_session(tmp_path)

    def backend(candidate, given):
        _write_result(candidate, _bound_mb_result(candidate, given, median_tokens_per_sec=1e9))

    with pytest.raises(R.BindingFailure, match="disagrees with the value recomputed"):
        R.orchestrate_phase_mb(session, backend=backend)


# ==================================================== R2 Part 8: evidence completeness


def test_the_initial_lr_grid_must_be_complete_evidence(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)

    def backend(candidate, given):
        if candidate["peak_lr"] == 4e-4:
            raise R.CandidateFailure("simulated")
        _write_result(candidate, _bound_lr_result(candidate, given, 3.0))

    # An ineligible record is still a record: the grid is complete and selection may proceed.
    report = R.orchestrate_phase_muon_lr(session, backend=backend)
    assert {float(r["peak_lr"]) for r in report["seed1"]} == set(C.LR_GRID_SEED1)


def test_a_missing_initial_grid_record_aborts(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    with mock.patch.object(R, "plan_phase_lr", _plan_without(4e-4)):
        with pytest.raises(C.PilotContractError, match="initial grid incomplete"):
            R.orchestrate_phase_muon_lr(
                session,
                backend=lambda c, s: _write_result(c, _bound_lr_result(c, s, 3.0)),
            )


def _plan_without(dropped):
    real = R.plan_phase_lr

    def planner(**kwargs):
        return [c for c in real(**kwargs) if float(c["peak_lr"]) != dropped]

    return planner


def test_incomplete_confirmation_evidence_aborts(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}
    real = R.plan_phase_lr

    def planner(**kwargs):
        planned = real(**kwargs)
        if kwargs.get("seed_label") == "seed-2":
            planned = [c for c in planned if float(c["peak_lr"]) != 3e-4]
        return planned

    with mock.patch.object(R, "plan_phase_lr", planner):
        with pytest.raises(R.PhaseAbort, match="confirmation evidence incomplete"):
            R.orchestrate_phase_muon_lr(
                session,
                backend=lambda c, s: _write_result(c, _bound_lr_result(c, s, scores[c["peak_lr"]])),
            )


def test_incomplete_edge_evidence_aborts(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}
    real = R.plan_phase_lr

    def planner(**kwargs):
        planned = real(**kwargs)
        if kwargs.get("seed_label") == "seed-2" and list(kwargs.get("peak_lrs") or []) == [1e-4]:
            planned = []
        return planned

    with mock.patch.object(R, "plan_phase_lr", planner):
        with pytest.raises(R.PhaseAbort, match="edge evidence incomplete"):
            R.orchestrate_phase_muon_lr(
                session,
                backend=lambda c, s: _write_result(c, _bound_lr_result(c, s, scores[c["peak_lr"]])),
            )


# ============================== R2 Part 9: the pilot index manifest FILE hash comes from disk


def test_the_index_manifest_file_hash_is_computed_from_the_bytes_on_disk(tmp_path):
    body = C.canonical_json_bytes({"schema_version": C.PILOT_INDEX_SCHEMA, "x": 1})
    path = tmp_path / R.PILOT_INDEX_MANIFEST_FILENAME
    path.write_bytes(body)
    assert R.file_sha256(path) == hashlib.sha256(body).hexdigest()
    import inspect

    src = inspect.getsource(R.validate_execution_artifacts)
    assert "_sha256_file(index_path)" in src
    assert "observed_index_file_sha" in src
    # never taken from the manifest the authorization is being checked against
    assert 'manifest.get("pilot_index_manifest_file_sha256")' not in src


def test_the_index_manifest_uses_the_canonical_filename(tmp_path):
    assert R.PILOT_INDEX_MANIFEST_FILENAME == "PILOT_INDICES.json"
    with pytest.raises(C.PilotContractError, match="canonical name"):
        R.write_pilot_index_manifest(tmp_path / "whatever.json")


def _fake_release(stage):
    return {
        "stage": stage,
        "blocks": A_BLOCKS if stage == "stage_a" else B_BLOCKS,
        "meta_sha256": (
            R.ACCEPTED_STAGE_A_META_SHA256 if stage == "stage_a" else R.ACCEPTED_STAGE_B_META_SHA256
        ),
    }


def test_the_index_manifest_document_carries_no_self_hash(monkeypatch):
    monkeypatch.setattr(R, "verify_accepted_release", _fake_release)
    doc = R.pilot_index_manifest_document()
    assert "pilot_index_manifest_file_sha256" not in doc
    assert doc["schema_version"] == C.PILOT_INDEX_SCHEMA
    assert doc["stage_a_blocks"] == A_BLOCKS and doc["stage_b_blocks"] == B_BLOCKS
    assert doc["stage_a_meta_sha256"] == R.ACCEPTED_STAGE_A_META_SHA256
    assert doc["stage_b_meta_sha256"] == R.ACCEPTED_STAGE_B_META_SHA256
    assert doc["seed"] == C.PILOT_INDEX_SEED
    assert doc["counts"] == {"stage_a_eval": 4096, "stage_a_train": 131072, "stage_b_eval": 4096}
    expected = C.generate_pilot_indices(A_BLOCKS, B_BLOCKS)
    for key in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256"):
        assert doc[key] == expected[key]
    assert (
        doc["serialized_index_lists_digest"]
        == hashlib.sha256(
            C.canonical_json_bytes({
                k: expected[k]
                for k in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256")
            })
        ).hexdigest()
    )


def test_the_index_manifest_is_single_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "verify_accepted_release", _fake_release)
    out = tmp_path / R.PILOT_INDEX_MANIFEST_FILENAME
    published = R.write_pilot_index_manifest(out)
    assert published["pilot_index_manifest_file_sha256"] == R.file_sha256(out)
    with pytest.raises(C.PilotContractError, match="already exists"):
        R.write_pilot_index_manifest(out)


# ================================ R2 Part 10: the authorized root is validated before any write


def test_the_authorized_root_is_validated_before_anything_is_written(tmp_path):
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(R.open_session)).body[0]
    order = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("validate_execution_artifacts", "mkdir", "write_bytes"):
                order.append((node.lineno, name))
    order.sort()
    assert order[0][1] == "validate_execution_artifacts", order
    validator = ast.parse(inspect.getsource(R.validate_execution_artifacts)).body[0]
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "validate_authorized_output_root"
        for n in ast.walk(validator)
    )


@pytest.mark.parametrize(
    "relative",
    [
        "runs/m_production_v1_2026-08-29/release/stage_a",
        "runs/m_production_v1_2026-08-29/release/stage_a/inner",
        "runs/g_production_2026-08-21/release",
    ],
)
def test_the_authorized_root_may_not_be_or_sit_inside_an_accepted_release(relative):
    with pytest.raises(C.PilotContractError, match="accepted release"):
        R.validate_authorized_output_root(REPO_ROOT / relative)


def test_the_authorized_root_may_not_contain_an_accepted_release():
    with pytest.raises(C.PilotContractError, match="contains an accepted release"):
        R.validate_authorized_output_root(REPO_ROOT / "runs")


def test_the_authorized_root_needs_an_existing_parent(tmp_path):
    assert (
        R.validate_authorized_output_root(tmp_path / "new_root")
        == (tmp_path / "new_root").resolve()
    )
    with pytest.raises(C.PilotContractError, match="parent must already exist"):
        R.validate_authorized_output_root(tmp_path / "absent" / "deeper")


def test_no_write_happens_when_the_root_is_refused(tmp_path, monkeypatch):
    """A refused root leaves no session file, ledger or lock behind."""
    root = REPO_ROOT / "runs" / "m_production_v1_2026-08-29" / "release"
    before = sorted(p.name for p in root.iterdir()) if root.is_dir() else None
    with pytest.raises(C.PilotContractError):
        R.validate_authorized_output_root(root)
    after = sorted(p.name for p in root.iterdir()) if root.is_dir() else None
    assert before == after


# ============================================ R2 Part 11: the subprocess terminal protocol


def _terminal(**over):
    doc = {
        "terminal_status": "SUCCESS",
        "error_class": None,
        "error_message": None,
        "completed_updates": 40,
        "reserved_tokens": {"MB": 1},
        "completed_tokens": {"MB": 1},
        "run_meta_sha256": "rm",
        "candidate_id": "mb_micro16_compileoff",
        "phase": "MB",
    }
    doc.update(over)
    return doc


def test_terminal_result_round_trip(tmp_path):
    spec = tmp_path / "spec.json"
    path = R.write_terminal_result(spec, _terminal())
    assert path == tmp_path / "spec.terminal.json"
    doc = R.read_terminal_result(spec)
    assert doc["terminal_status"] == "SUCCESS"
    assert doc["schema_version"] == R.TERMINAL_RESULT_SCHEMA
    assert set(R.TERMINAL_RESULT_FIELDS) <= set(doc)


def test_terminal_result_requires_every_field(tmp_path):
    incomplete = _terminal()
    incomplete.pop("completed_tokens")
    with pytest.raises(C.PilotContractError, match="terminal result is incomplete"):
        R.write_terminal_result(tmp_path / "s.json", incomplete)


def test_an_unknown_terminal_status_is_refused(tmp_path):
    with pytest.raises(C.PilotContractError, match="unknown terminal status"):
        R.write_terminal_result(tmp_path / "s.json", _terminal(terminal_status="FINE"))


def test_a_missing_terminal_result_is_a_phase_abort_not_a_pass(tmp_path):
    with pytest.raises(R.PhaseAbort, match="left no terminal result"):
        R.read_terminal_result(tmp_path / "never_ran.json")


@pytest.mark.parametrize(
    "status,expected",
    [
        ("CANDIDATE_INELIGIBLE", R.CandidateFailure),
        ("PHASE_ABORT", R.PhaseAbort),
        ("BINDING_FAILURE", R.BindingFailure),
    ],
)
def test_the_terminal_status_decides_how_the_parent_proceeds(
    tmp_path, monkeypatch, status, expected
):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]

    def handler(argv, kwargs):
        spec_path = Path(argv[argv.index("--spec") + 1])
        R.write_terminal_result(
            spec_path,
            _terminal(
                terminal_status=status,
                error_class="RuntimeError",
                error_message="simulated",
                completed_updates=0,
            ),
        )
        return _FakeCompleted(R.RESULT_CLASSES[status], "", "")

    _fake_subprocess(monkeypatch, handler)
    with pytest.raises(expected):
        R._subprocess_backend(candidate, session)  # noqa: SLF001


def test_a_terminal_status_that_disagrees_with_the_exit_code_aborts(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]

    def handler(argv, kwargs):
        spec_path = Path(argv[argv.index("--spec") + 1])
        R.write_terminal_result(spec_path, _terminal(terminal_status="SUCCESS"))
        return _FakeCompleted(3, "", "")  # claims success, exits ineligible

    _fake_subprocess(monkeypatch, handler)
    with pytest.raises(R.PhaseAbort, match="terminal protocol is broken"):
        R._subprocess_backend(candidate, session)  # noqa: SLF001


def test_child_stdout_and_stderr_are_preserved_verbatim(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]

    def handler(argv, kwargs):
        spec_path = Path(argv[argv.index("--spec") + 1])
        R.write_terminal_result(spec_path, _terminal(candidate_id=candidate["candidate_id"]))
        return _FakeCompleted(0, "child stdout line\n", "child stderr line\n")

    _fake_subprocess(monkeypatch, handler)
    R._subprocess_backend(candidate, session)  # noqa: SLF001
    spec_dir = session.output_root / "_specs"
    stem = candidate["candidate_id"]
    assert (spec_dir / f"{stem}.stdout").read_text(encoding="utf-8") == "child stdout line\n"
    assert (spec_dir / f"{stem}.stderr").read_text(encoding="utf-8") == "child stderr line\n"


def test_the_worker_writes_a_terminal_result_even_when_it_fails_early(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_bytes(
        C.canonical_json_bytes({
            "candidate": {"phase": "MB", "candidate_id": "mb_micro8_compileoff"},
            "authorization_path": str(tmp_path / "auth.json"),
            "pilot_index_manifest_path": str(tmp_path / "PILOT_INDICES.json"),
            "authorized_output_root": str(tmp_path / "root"),
            "session_id": "s",
        })
    )
    code = R.main([
        "execute-candidate",
        "--spec",
        str(spec),
        "--authorization",
        str(tmp_path / "auth.json"),
        "--pilot-index-manifest",
        str(tmp_path / "PILOT_INDICES.json"),
    ])
    assert code == R.BINDING_FAILURE
    doc = R.read_terminal_result(spec)
    assert doc["terminal_status"] == "BINDING_FAILURE"
    assert doc["error_class"] == "BindingFailure"
    assert doc["reserved_tokens"] is None and doc["completed_tokens"] is None


# ============================================ R2 Part 12: timing names and honest evidence


def test_timing_fields_use_their_r2_names(tmp_path):
    import inspect

    src = inspect.getsource(R.run_phase_mb_candidate)
    for name in (
        "torch_compile_wrapper_seconds",
        "first_optimizer_update_wall_seconds",
        "measured_update_wall_seconds",
    ):
        assert f'"{name}"' in src, name
    for retired in ('"compile_wrapper_seconds"', "compile_materialization_wall_seconds"):
        assert retired not in src, retired
    result = R._run_updates  # noqa: SLF001
    assert "measured_update_wall_seconds" in inspect.getsource(result)


def test_no_host_device_scalar_transfer_inside_the_timed_region():
    """R2 Part 12: the timed region measures the training step, not a synchronizing readback."""
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(R._run_updates)).body[0]  # noqa: SLF001
    loop = next(
        n for n in ast.walk(fn) if isinstance(n, ast.For) and getattr(n.target, "id", None) == "u"
    )
    body = loop.body
    start = next(
        i
        for i, s in enumerate(body)
        if isinstance(s, ast.Assign) and any(getattr(t, "id", None) == "t0" for t in s.targets)
    )
    stop = next(
        i
        for i, s in enumerate(body)
        if isinstance(s, ast.Assign)
        and any(isinstance(t, ast.Subscript) for t in s.targets)
        and getattr(getattr(s.targets[0], "value", None), "id", None) == "per_update_seconds"
    )
    assert start < stop
    for statement in body[start + 1 : stop]:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                assert getattr(node.func, "id", None) != "float", ast.dump(node)
                assert getattr(node.func, "attr", None) not in ("item", "tolist")


def test_compile_evidence_is_observed_not_self_reported(tmp_path):
    import torch

    class _Plain(torch.nn.Module):
        def forward(self, x):
            return x

    cache = tmp_path / "inductor_cache"
    plain = _Plain()
    off = R.compile_path_evidence(plain, requested=False, cache_dir=cache)
    assert off["canonical_compile_path"] is True
    assert off["compiled_module_is_optimized_module"] is False
    # requesting compile while eager silently ran: no graphs, no artifacts -> rejected
    on = R.compile_path_evidence(plain, requested=True, cache_dir=cache)
    assert on["observed_compiled_execution"] is False
    assert on["canonical_compile_path"] is False
    assert on["inductor_artifact_count"] == 0


def test_a_compiled_module_with_artifacts_is_accepted(tmp_path):
    class _OptimizedModule:  # the name torch._dynamo gives its wrapper
        pass

    cache = tmp_path / "inductor_cache"
    cache.mkdir()
    (cache / "output_code.py").write_text("# compiled", encoding="utf-8")
    evidence = R.compile_path_evidence(_OptimizedModule(), requested=True, cache_dir=cache)
    assert evidence["compiled_module_is_optimized_module"] is True
    assert evidence["inductor_artifact_count"] == 1
    assert evidence["canonical_compile_path"] is True
    # the same module with compile=off is a silent-compile failure in the other direction
    assert (
        R.compile_path_evidence(_OptimizedModule(), requested=False, cache_dir=cache)[
            "canonical_compile_path"
        ]
        is False
    )


def test_the_contract_rejects_a_compile_candidate_that_fell_back(tmp_path):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[1]
    assert candidate["compile"] is True
    result = _bound_mb_result(candidate, session, canonical_compile_path=False)
    eligible, failures = C.mb_candidate_eligible(result, VRAM_4090)
    assert not eligible and "compile_silent_fallback" in failures


# ================================================= R2 Part 13: Muon RMS-matching verification


def test_the_rms_matched_lr_formula():
    import math

    assert R.RMS_MATCHING_CONSTANT == 0.2
    for fan_out, fan_in in R.RMS_MATCHING_SHAPE_CASES:
        assert R.expected_rms_matched_lr(3e-4, fan_out, fan_in) == (
            3e-4 * 0.2 * math.sqrt(max(fan_in, fan_out))
        )


def test_the_realized_muon_update_applies_the_rms_matched_lr():
    """One real Muon step per shape case, reconstructed from the closed form."""
    evidence = R.verify_rms_matching(lr=3e-4)
    assert evidence["rule"] == "adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))"
    assert evidence["all_cases_match"] is True
    assert len(evidence["cases"]) == len(R.RMS_MATCHING_SHAPE_CASES)
    for case in evidence["cases"]:
        assert case["matches_rms_matching_rule"] is True


def test_the_rms_matching_cases_would_catch_an_unscaled_implementation():
    """Non-vacuous: each case's RMS-matched update differs materially from the unscaled one."""
    evidence = R.verify_rms_matching(lr=3e-4)
    assert evidence["all_cases_discriminating"] is True
    for case in evidence["cases"]:
        assert case["rms_matching_factor"] >= 2.0
        assert case["unscaled_lr_would_differ_by"] > 20.0 * max(case["max_abs_error"], 1e-12)


def test_the_rms_matching_shape_cases_include_non_square_weights():
    shapes = set(R.RMS_MATCHING_SHAPE_CASES)
    assert (576, 576) in shapes, "the canonical square attention projection"
    assert any(a != b for a, b in shapes), "a square-only case set proves nothing about max()"
    assert (1536, 576) in shapes and (576, 1536) in shapes, "both orientations"


def test_the_rms_matching_subcommand_runs_without_training(capsys):
    assert R.main(["rms-matching"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["all_cases_match"] and payload["all_cases_discriminating"]


# ============================================== R2 Part 14: the recomputed execution closure


def test_the_execution_closure_is_rederived_after_r2():
    closure = R.execution_closure()
    assert closure["derived_closure"] == sorted(closure["derived_closure"])
    assert closure["derived_closure_count"] == len(closure["derived_closure"])
    assert closure["unbound_load_bearing_module_count"] == 0
    assert "pretrain/train_pretrain_with_bench.py" not in closure["derived_closure"]
    for required in (
        "pretrain/pilot_contract_v2_3.py",
        "pretrain/pilot_runner_v2_3.py",
        "pretrain/dataset_pretrain.py",
        "src/canonical_loss.py",
        "src/canonical_schedule.py",
        "src/model.py",
        "src/optim.py",
        "src/special_tokens.py",
    ):
        assert required in closure["derived_closure"], required


def test_the_bundle_hash_is_stable_and_covers_every_closure_file():
    first = R.execution_closure()
    second = R.execution_closure()
    assert (
        first["EXECUTION_IMPLEMENTATION_BUNDLE_SHA256"]
        == (second["EXECUTION_IMPLEMENTATION_BUNDLE_SHA256"])
    )
    assert R.execution_bundle_sha256() == first["EXECUTION_IMPLEMENTATION_BUNDLE_SHA256"]
    assert set(first["files"]) == set(first["derived_closure"])
    for relative, digest in first["files"].items():
        assert digest == R.file_sha256(REPO_ROOT / relative), relative


def test_the_same_bundle_hash_binds_authorization_parent_child_and_fingerprint(tmp_path):
    """One value, four places: the manifest field, the observed set, run_meta and the print."""
    bundle = R.execution_bundle_sha256()
    assert "execution_implementation_bundle_sha256" in C.AUTHORIZATION_REQUIRED_FIELDS
    assert R.base_runtime_fingerprint()["execution_implementation_bundle_sha256"] == bundle
    import inspect

    validator = inspect.getsource(R.validate_execution_artifacts)
    assert '"execution_bundle_sha256": execution_bundle_sha256()' in validator
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]
    meta = R.run_meta(candidate=candidate, session=session)
    assert (
        meta["execution_implementation_bundle_sha256"]
        == session.validated["observed"]["execution_bundle_sha256"]
    )
    assert "execution_bundle_sha256" in dict(R._session_bindings(session))  # noqa: SLF001
    assert "execution_bundle_sha256" in R.REQUIRED_RESULT_BINDINGS


def test_run_meta_binds_the_session_and_the_index_manifest_file_hash(tmp_path):
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]
    meta = R.run_meta(candidate=candidate, session=session)
    assert meta["session_id"] == session.session_id
    assert (
        meta["pilot_index_manifest_file_sha256"]
        == session.validated["pilot_index_manifest_file_sha256"]
    )
    assert meta["authorization_sha256"] == session.validated["authorization_sha256"]
    assert meta["schema_version"] == C.RUN_META_SCHEMA


# ================================================ R2: nothing in this file trains a real model


def test_this_repository_still_publishes_no_authorized_manifest():
    template = C.authorization_template()
    assert template["authorization_status"] == "NOT_AUTHORIZED"
    assert template["allowed_scope"] is None
    tracked = R._git("ls-files").splitlines()  # noqa: SLF001
    assert not [f for f in tracked if "pilot_authorization" in f]


def test_accepted_release_identity_mismatch_is_a_binding_failure(monkeypatch):
    """A wrong release meta hash is an identity binding, not a runtime condition."""
    monkeypatch.setattr(R, "ACCEPTED_STAGE_A_META_SHA256", "0" * 64)
    with pytest.raises(R.BindingFailure, match="meta SHA-256 mismatch"):
        R.verify_accepted_release("stage_a")


def test_evaluate_returns_the_raw_components_the_parent_recomputes_from():
    """R2 Part 7 depends on this shape: the numerator and weight, plus the single division."""
    model = _TinyLM()
    view = R.IndexView(_FakeDataset(20), [0, 1, 2, 3])
    out = R.evaluate(model, view, micro_bsz=2, device="cpu")
    assert set(out) == {"numerator", "weight", "loss", "blocks"}
    assert out["blocks"] == 4
    assert out["weight"] > 0.0
    assert out["loss"] == out["numerator"] / max(1.0, out["weight"])
    # the same shape the LR candidate serializes and the parent recomputes
    assert R.verify_recomputed_lr_result({
        "candidate_id": "probe",
        "eval_stage_a_numerator": out["numerator"],
        "eval_stage_a_weight": out["weight"],
        "eval_stage_b_numerator": out["numerator"],
        "eval_stage_b_weight": out["weight"],
        "eval_loss_stage_a": out["loss"],
        "eval_loss_stage_b": out["loss"],
        "score": C.lr_score(out["loss"], out["loss"]),
    })["score"] == C.lr_score(out["loss"], out["loss"])


def test_a_stale_terminal_result_is_cleared_before_a_launch(tmp_path, monkeypatch):
    """A terminal result left by an earlier attempt may not be read as this launch's."""
    session = _fake_session(tmp_path)
    candidate = R.plan_phase_mb(output_root=session.output_root)[0]
    spec_dir = session.output_root / "_specs"
    spec_dir.mkdir(parents=True)
    stale = spec_dir / f"{candidate['candidate_id']}.json"
    R.write_terminal_result(stale, _terminal(terminal_status="SUCCESS"))

    def handler(argv, kwargs):
        return _FakeCompleted(0, "", "")  # the child writes nothing at all

    _fake_subprocess(monkeypatch, handler)
    with pytest.raises(R.PhaseAbort, match="left no terminal result"):
        R._subprocess_backend(candidate, session)  # noqa: SLF001
