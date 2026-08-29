"""P-PILOT-CONTRACT-V2.3 contract and executor tests.

Pure contract logic, tiny non-training fixtures, and mocks for executor orchestration. No real
model-training gradient update is performed anywhere in this file.
"""

from __future__ import annotations

import hashlib
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


def _ledger_identity():
    return {
        "contract_sha256": "c",
        "head": "h",
        "execution_bundle_sha256": "b",
        "serialized_index_lists_digest": "i",
        "authorization_sha256": "a",
        "authorized_output_root": "/tmp/root",
        "authorized_scope": "FULL_V2_3_PILOT",
    }


def _ceilings(global_ceiling=C.GLOBAL_PILOT_TOKEN_CEILING):
    return {
        "MB": min(C.PHASE_MB_TOKEN_CEILING, global_ceiling),
        "LR": min(C.PHASE_MUON_LR_TOKEN_CEILING, global_ceiling),
        "GLOBAL": global_ceiling,
    }


def test_persistent_ledger_round_trip(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.commit_update("MB")
    reloaded = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    assert reloaded.state["global_tokens"] == C.EFFECTIVE_BATCH_TOKENS
    assert reloaded.state["phase_tokens"]["MB"] == C.EFFECTIVE_BATCH_TOKENS
    assert reloaded.state["updates"] == 1


def test_ledger_reload_validates_bound_identity(tmp_path):
    R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    other = _ledger_identity() | {"head": "DIFFERENT"}
    with pytest.raises(C.PilotContractError, match="does not bind this execution"):
        R.TokenLedger(tmp_path / "l.json", other, _ceilings())


def test_ledger_refuses_update_over_ceiling(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.state["phase_tokens"]["MB"] = C.PHASE_MB_TOKEN_CEILING
    ledger._write(ledger.state)  # noqa: SLF001 - exercising the persisted path deliberately
    with pytest.raises(C.PilotContractError, match="PILOT_ABORT"):
        ledger.commit_update("MB")


def test_ledger_uses_interprocess_locking(tmp_path):
    """The commit path takes an exclusive flock before reload/check/commit."""
    import inspect

    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    assert "flock" in inspect.getsource(R.TokenLedger._lock)  # noqa: SLF001
    assert "LOCK_EX" in inspect.getsource(R.TokenLedger._lock)  # noqa: SLF001
    assert "self._lock()" in inspect.getsource(R.TokenLedger.commit_update)
    ledger.commit_update("MB")
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


def _fake_context(tmp_path, scope="FULL_V2_3_PILOT"):
    """A ValidatedExecutionContext built directly, as build_validated_context would."""
    root = (tmp_path / "root").resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "contract_sha256": "c",
        "head": "h",
        "execution_bundle_sha256": "b",
        "serialized_index_lists_digest": "i",
        "authorization_sha256": "a",
        "authorized_output_root": str(root),
        "authorized_scope": scope,
    }
    ledger = R.TokenLedger(root / R.LEDGER_FILENAME, identity, _ceilings())
    return R.ValidatedExecutionContext(
        scope=scope,
        phase="MB",
        manifest_sha256="a",
        observed={
            "branch": "b",
            "head": "h",
            "contract_sha256": "c",
            "execution_bundle_sha256": "b",
            "serialized_index_lists_digest": "i",
            "pilot_index_manifest_file_sha256": "m",
            "stage_a_meta_sha256": "sa",
            "stage_b_meta_sha256": "sb",
            "output_root": str(root),
        },
        indices={
            "stage_a_eval": [0],
            "stage_a_train": [0],
            "stage_b_eval": [0],
            "stage_a_eval_sha256": "x",
            "stage_a_train_sha256": "y",
            "stage_b_eval_sha256": "z",
        },
        fingerprint={"fingerprint_sha256": "fp", "gpu": {"total_vram_bytes": VRAM_4090}},
        stage_a={"dataset": None},
        stage_b={"dataset": None},
        ledger=ledger,
        output_root=root,
        effective_ceilings=_ceilings(),
        train_order_by_seed={"seed-1": [0], "seed-2": [0]},
    )


def test_candidate_functions_reject_a_plain_authorized_dict(tmp_path):
    """The `context['authorized'] = True` pattern is gone; a mapping is not a context."""
    candidate = R.plan_phase_mb(output_root=tmp_path / "root")[0]
    for fn in (R.run_phase_mb_candidate, R.run_phase_lr_candidate):
        with pytest.raises(C.PilotContractError, match="ValidatedExecutionContext"):
            fn(candidate, {"authorized": True, "train_view": None, "ledger": None})


def test_phase_mb_only_scope_cannot_run_lr(tmp_path):
    ctx = _fake_context(tmp_path, scope="PHASE_MB_ONLY")
    ctx.require_phase_allowed("MB")
    with pytest.raises(C.PilotContractError, match="PHASE_MB_ONLY may not execute"):
        ctx.require_phase_allowed("LR")
    with pytest.raises(C.PilotContractError, match="PHASE_MB_ONLY may not execute"):
        R.orchestrate_phase_muon_lr(ctx, micro_bsz=8, compile_on=False, backend=lambda *_: None)


def test_orchestrator_runs_exactly_the_ten_candidates(tmp_path):
    """A fake backend records what the orchestrator chose to launch."""
    ctx = _fake_context(tmp_path)
    launched = []

    def backend(candidate, context):
        assert isinstance(context, R.ValidatedExecutionContext)
        launched.append(candidate["candidate_id"])
        out = Path(candidate["output_dir"])
        out.mkdir(parents=True)
        (out / "result.json").write_bytes(
            C.canonical_json_bytes(_bound_mb_result(candidate, context))
        )

    report = R.orchestrate_phase_mb(ctx, backend=backend)
    assert launched == list(C.MB_REQUIRED_CANDIDATE_IDS)
    assert len(launched) == 10
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"


def _bound_mb_result(candidate, context, **over):
    r = {
        "phase": "MB",
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "completed_updates": 40,
        "median_tokens_per_sec": 1000.0,
        "max_memory_reserved_bytes": 8 * 1024**3,
        "oom": False,
        "uncontrolled_exception": False,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "all_optimizer_states_instantiated": True,
        "grouping_matches_contract": True,
        "all_lr_ratios_are_one": True,
        "canonical_compile_path": True,
        "run_meta_sha256": "rm",
        "contract_sha256": context.observed["contract_sha256"],
        "implementation_head": context.observed["head"],
        "execution_bundle_sha256": context.observed["execution_bundle_sha256"],
        "serialized_index_lists_digest": context.observed["serialized_index_lists_digest"],
        "runtime_fingerprint_sha256": context.fingerprint["fingerprint_sha256"],
        "authorization_sha256": context.manifest_sha256,
        "ledger_identity": dict(context.ledger.identity),
        "output_dir": candidate["output_dir"],
    }
    r.update(over)
    return r


def test_candidate_local_failure_becomes_ineligible_evidence_and_grid_continues(tmp_path):
    ctx = _fake_context(tmp_path)
    launched = []

    def backend(candidate, context):
        launched.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "mb_micro16_compileoff":
            raise RuntimeError("simulated candidate-local CUDA failure")
        out = Path(candidate["output_dir"])
        out.mkdir(parents=True)
        (out / "result.json").write_bytes(
            C.canonical_json_bytes(_bound_mb_result(candidate, context))
        )

    report = R.orchestrate_phase_mb(ctx, backend=backend)
    assert len(launched) == 10, "the grid must continue after a candidate-local failure"
    failed = [c for c in report["candidates"] if c["candidate_id"] == "mb_micro16_compileoff"]
    assert failed and failed[0]["eligible"] is False
    assert failed[0]["reason"] == "candidate_runtime_exception"
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"


def test_phase_level_binding_failure_aborts(tmp_path):
    ctx = _fake_context(tmp_path)

    def backend(candidate, context):
        raise C.PilotContractError("accepted-release binding failed")

    with pytest.raises(C.PilotContractError, match="accepted-release binding failed"):
        R.orchestrate_phase_mb(ctx, backend=backend)


def test_result_must_bind_this_execution(tmp_path):
    ctx = _fake_context(tmp_path)

    def backend(candidate, context):
        out = Path(candidate["output_dir"])
        out.mkdir(parents=True)
        (out / "result.json").write_bytes(
            C.canonical_json_bytes(_bound_mb_result(candidate, context, contract_sha256="WRONG"))
        )

    with pytest.raises(C.PilotContractError, match="does not bind this execution"):
        R.orchestrate_phase_mb(ctx, backend=backend)


def test_lr_orchestrator_derives_grid_confirmation_and_edge(tmp_path):
    """Nothing about which LR candidates run comes from caller input."""
    ctx = _fake_context(tmp_path)
    launched = []
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}

    def backend(candidate, context):
        launched.append((candidate["peak_lr"], candidate["seed_label"]))
        out = Path(candidate["output_dir"])
        out.mkdir(parents=True)
        r = {
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
            "eval_loss_stage_a": 3.0,
            "eval_loss_stage_b": 3.0,
            "score": scores[candidate["peak_lr"]],
            "sustained_divergence": False,
            "run_meta_sha256": "rm",
            "contract_sha256": context.observed["contract_sha256"],
            "implementation_head": context.observed["head"],
            "execution_bundle_sha256": context.observed["execution_bundle_sha256"],
            "serialized_index_lists_digest": context.observed["serialized_index_lists_digest"],
            "runtime_fingerprint_sha256": context.fingerprint["fingerprint_sha256"],
            "authorization_sha256": context.manifest_sha256,
            "ledger_identity": dict(context.ledger.identity),
            "output_dir": candidate["output_dir"],
        }
        (out / "result.json").write_bytes(C.canonical_json_bytes(r))

    report = R.orchestrate_phase_muon_lr(ctx, micro_bsz=8, compile_on=False, backend=backend)
    seed1_lrs = sorted(lr for lr, seed in launched if seed == "seed-1" and lr in C.LR_GRID_SEED1)
    assert seed1_lrs == sorted(C.LR_GRID_SEED1), "initial grid derived internally"
    # winner is 2e-4 (lowest score) -> confirmation neighbour is the HIGHER 3e-4
    seed2_lrs = sorted({lr for lr, seed in launched if seed == "seed-2"})
    assert 2e-4 in seed2_lrs and 3e-4 in seed2_lrs
    # 2e-4 confirmed -> edge 1e-4 must have run under BOTH seeds
    assert (1e-4, "seed-1") in launched and (1e-4, "seed-2") in launched
    assert report["final"]["second_expansion_permitted"] is False


def test_runner_run_subcommand_refuses(tmp_path):
    assert R.main(["run", "--phase", "MB", "--output-root", str(tmp_path / "x")]) == 2


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

    src = inspect.getsource(R.build_validated_context)
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
    assert ledger.state["updates"] == 0, "a refused run must not consume ledger tokens"
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
