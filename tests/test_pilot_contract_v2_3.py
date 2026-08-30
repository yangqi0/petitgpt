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
SYNTHETIC_VRAM_BYTES = 24564 * 1024 * 1024


# ------------------------------------------------------------------ supersession


def test_v2_3_records_the_optimizer_and_hardware_policy_amendments():
    d = C.contract_document()
    assert d["contract_version"] == "P-PILOT-CONTRACT-V2.3"
    assert "P-PILOT-CONTRACT-V2.2" in d["supersedes"]
    supersession = d["supersedes"]["P-PILOT-CONTRACT-V2.2"]
    assert "optimizer" in supersession and "GPU-product" in supersession
    assert d["owner_optimizer_verdict"] == "FREEZE_MUON_DIRECTLY"
    assert d["runtime_gate"]["TRAINING_GPU_MODEL"] == ("DEFERRED_UNTIL_OWNER_PILOT_AUTHORIZATION")
    assert d["runtime_gate"]["future_stage_n_o"]["exact_gpu_product_frozen_now"] is False
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


def _timings(tps):
    """Per-update records whose median RATE is exactly ``tps`` (updates 11..40)."""
    seconds = C.TRAINED_TOKENS_PER_UPDATE / float(tps)
    return [
        {"update": u, "trained_tokens": C.TRAINED_TOKENS_PER_UPDATE, "wall_seconds": seconds}
        for u in C.MB_MEASURED_UPDATES
    ]


def _mb(cid, micro, comp, tps, vram_gib, **over):
    r = {
        "candidate_id": cid,
        "micro_bsz": micro,
        "compile": comp,
        "median_update_tokens_per_second": tps,
        "update_timings": _timings(tps),
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
        over = dict(overrides.get(c["candidate_id"], {}))
        tps = over.pop("median_update_tokens_per_second", 1000.0)
        r = _mb(c["candidate_id"], c["micro_bsz"], c["compile"], tps, 8.0)
        r.update(over)
        out.append(r)
    return out


def test_mb_requires_the_complete_grid():
    partial = _full_grid()[:3]
    with pytest.raises(C.PilotContractError, match="incomplete"):
        C.mb_select(partial, SYNTHETIC_VRAM_BYTES)


def test_mb_rejects_duplicate_candidate_identity():
    dup = [*_full_grid(), _mb("mb_micro8_compileoff", 8, False, 9999.0, 8.0)]
    with pytest.raises(C.PilotContractError, match="duplicate"):
        C.mb_select(dup, SYNTHETIC_VRAM_BYTES)


def test_mb_rejects_unknown_candidate_identity():
    grid = _full_grid()
    grid[0] = _mb("mb_micro99_compileoff", 99, False, 1000.0, 8.0)
    with pytest.raises(C.PilotContractError, match="incomplete|unknown"):
        C.mb_select(grid, SYNTHETIC_VRAM_BYTES)


def test_mb_selection_and_tie_ladder():
    out = C.mb_select(
        _full_grid(mb_micro8_compileoff={"median_update_tokens_per_second": 5000.0}),
        SYNTHETIC_VRAM_BYTES,
    )
    assert out["FROZEN_MICRO_BSZ"] == 8 and out["FROZEN_GRAD_ACCUM"] == 16
    assert out["FROZEN_COMPILE"] is False and out["tie_break"] == "fastest_unique"
    # VRAM tie-break
    out2 = C.mb_select(
        _full_grid(
            mb_micro8_compileoff={
                "median_update_tokens_per_second": 5000.0,
                "max_memory_reserved_bytes": int(12 * 1024**3),
            },
            mb_micro4_compileoff={
                "median_update_tokens_per_second": 4950.0,
                "max_memory_reserved_bytes": int(8 * 1024**3),
            },
        ),
        SYNTHETIC_VRAM_BYTES,
    )
    assert out2["tie_break"] == "lowest_peak_reserved_vram" and out2["FROZEN_MICRO_BSZ"] == 4


def test_mb_all_ineligible_aborts():
    out = C.mb_select(
        _full_grid(**{c["candidate_id"]: {"oom": True} for c in C.mb_candidate_grid()}),
        SYNTHETIC_VRAM_BYTES,
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
    ok, failures = C.mb_candidate_eligible(
        _mb("x", 8, False, 1.0, 8.0, **bad), SYNTHETIC_VRAM_BYTES
    )
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
        "compile": False,
        "canonical_compile_path": True,
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
    """A structurally consistent ledger sitting one update below the ceiling refuses the next."""
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    updates = C.PHASE_MB_TOKEN_CEILING // C.TRAINED_TOKENS_PER_UPDATE
    tokens = updates * C.TRAINED_TOKENS_PER_UPDATE
    ledger.state["reserved_tokens"]["MB"] = tokens
    ledger.state["reserved_tokens"]["GLOBAL"] = tokens
    ledger.state["completed_tokens"]["MB"] = tokens
    ledger.state["completed_tokens"]["GLOBAL"] = tokens
    ledger.state["reserved_updates"] = ledger.state["completed_updates"] = updates
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

SYNTHETIC_GPU_NAME = "Owner-selected synthetic NVIDIA CUDA device"


def _synthetic_runtime_fingerprint(
    *,
    name=SYNTHETIC_GPU_NAME,
    device_count=1,
    cuda_available=True,
    bf16_supported=True,
    driver="synthetic-driver",
):
    body = {
        "schema_version": C.BASE_FINGERPRINT_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "gpu": {
            "cuda_available": cuda_available,
            "device_count": device_count,
            "selected_device_index": 0,
            "name": name,
            "total_vram_mib": SYNTHETIC_VRAM_BYTES // (1024 * 1024),
            "total_vram_bytes": SYNTHETIC_VRAM_BYTES,
            "capability": "9.0",
            "driver": driver,
            "cuda_runtime": "12.8",
            "bf16_supported": bf16_supported,
        },
        "torch_version": "2.synthetic+cu128",
        "torch_build": {"cuda": "12.8", "git_version": "synthetic"},
        "numpy_version": C.REQUIRED_NUMPY_VERSION,
        "required_numpy_version": C.REQUIRED_NUMPY_VERSION,
        "tokenizers_version": "synthetic",
        "python_version": "3.synthetic",
        "python_executable": "/synthetic/python",
        "python_implementation": "CPython",
        "platform": "synthetic-platform",
        "container_template": "synthetic-container",
        "repository": {
            "branch": "agent/retrain-pipeline-contracts",
            "head": "deadbeef",
            "tracked_clean": True,
            "allowed_untracked_sha256": "a" * 64,
            "allowed_untracked_unchanged": True,
            "unexpected_untracked": [],
        },
        "contract_sha256": C.contract_sha256(),
        "execution_implementation_bundle_sha256": "b" * 64,
    }
    return {
        **body,
        "fingerprint_sha256": hashlib.sha256(C.canonical_json_bytes(body)).hexdigest(),
    }


def _hardware_binding(fingerprint, *, expected_name=None):
    return {
        "expected_gpu_device_name": expected_name or fingerprint["gpu"]["name"],
        "expected_cuda_device_count": 1,
        "cuda_required": True,
        "bf16_required": True,
        "expected_base_runtime_fingerprint_sha256": fingerprint["fingerprint_sha256"],
    }


def test_owner_selected_gpu_identity_and_runtime_binding_pass():
    fingerprint = _synthetic_runtime_fingerprint()
    verdict = C.check_training_authority(fingerprint, _hardware_binding(fingerprint))
    assert verdict["granted"] and verdict["training_authority"] == "GRANTED"


def test_a_different_gpu_identity_than_the_owner_selected_one_fails():
    fingerprint = _synthetic_runtime_fingerprint()
    binding = _hardware_binding(fingerprint, expected_name="A different selected device")
    verdict = C.check_training_authority(fingerprint, binding)
    assert not verdict["granted"]
    assert "gpu_device_identity_mismatch" in verdict["failures"]


@pytest.mark.parametrize(
    ("gpu_over", "failure"),
    [
        ({"device_count": 2}, "cuda_device_count_not_exactly_1"),
        ({"cuda_available": False}, "cuda_unavailable"),
        ({"bf16_supported": False}, "bf16_unsupported"),
    ],
)
def test_hardware_independent_runtime_requirements_fail_closed(gpu_over, failure):
    original = _synthetic_runtime_fingerprint()
    body = {k: v for k, v in original.items() if k != "fingerprint_sha256"}
    body["gpu"] = {**body["gpu"], **gpu_over}
    fingerprint = {
        **body,
        "fingerprint_sha256": hashlib.sha256(C.canonical_json_bytes(body)).hexdigest(),
    }
    binding = _hardware_binding(fingerprint)
    verdict = C.check_training_authority(fingerprint, binding)
    assert not verdict["granted"] and failure in verdict["failures"]
    with pytest.raises(C.PilotContractError, match="no training authority"):
        C.require_training_authority(fingerprint, binding)


def test_incomplete_self_hashed_runtime_fingerprint_fails_closed():
    original = _synthetic_runtime_fingerprint()
    body = {k: v for k, v in original.items() if k not in {"fingerprint_sha256", "repository"}}
    incomplete = {
        **body,
        "fingerprint_sha256": hashlib.sha256(C.canonical_json_bytes(body)).hexdigest(),
    }
    verdict = C.check_training_authority(incomplete, _hardware_binding(incomplete))
    assert not verdict["granted"]
    assert any("base_runtime_fingerprint_missing_fields" in f for f in verdict["failures"])


def test_active_v2_3_authority_has_no_permanent_product_gate():
    authority_surfaces = [
        (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "docs/PILOT_CONTRACT_V2_3.md",
            "pretrain/pilot_contract_v2_3.py",
            "pretrain/pilot_runner_v2_3.py",
            "requirements-pilot-v2_3.txt",
        )
    ]
    authority_surfaces.append(
        (REPO_ROOT / "README.md").read_text(encoding="utf-8").split("---", 1)[0]
    )
    authority = "\n".join(authority_surfaces)
    assert "REQUIRED_GPU_NAME_EXACT" not in authority
    assert "required_vram_mib_range" not in authority
    for product_number in ("40" + "90", "50" + "90"):
        assert product_number not in authority


# ------------------------------------------------------------------ authorization


def _authorized(observed, scope="FULL_V2_3_PILOT"):
    fingerprint = observed["base_runtime_fingerprint"]
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
        "training_hardware": _hardware_binding(fingerprint),
    }


@pytest.fixture
def observed():
    return {
        "branch": "agent/retrain-pipeline-contracts",
        "head": "deadbeef",
        "contract_sha256": C.contract_sha256(),
        "execution_bundle_sha256": "b" * 64,
        "serialized_index_lists_digest": "idx",
        "pilot_index_manifest_file_sha256": "mfile",
        "stage_a_meta_sha256": "a",
        "stage_b_meta_sha256": "b",
        "output_root": "/tmp/out",
        "base_runtime_fingerprint": _synthetic_runtime_fingerprint(),
    }


def test_authorization_template_is_not_authorized_and_selects_no_hardware():
    template = C.authorization_template()
    assert template["authorization_status"] == "NOT_AUTHORIZED"
    assert template["allowed_scope"] is None and template["repository_head"] is None
    assert template["training_hardware"] is None
    assert C.TRAINING_GPU_MODEL == "DEFERRED_UNTIL_OWNER_PILOT_AUTHORIZATION"
    assert set(C.ALLOWED_SCOPES) == {"PHASE_MB_ONLY", "FULL_V2_3_PILOT"}


def test_authorization_missing_refuses(observed):
    verdict = C.validate_authorization(None, requested_scope="PHASE_MB_ONLY", observed=observed)
    assert not verdict["authorized"] and verdict["failures"] == ["authorization_missing"]


def test_not_authorized_status_refuses(observed):
    manifest = _authorized(observed) | {"authorization_status": "NOT_AUTHORIZED"}
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert not verdict["authorized"]
    assert "authorization_status_not_authorized" in verdict["failures"]


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
    manifest = _authorized(observed) | {field: "WRONG"}
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert not verdict["authorized"] and any(field in f for f in verdict["failures"])


def test_scope_escalation_refused(observed):
    manifest = _authorized(observed, scope="PHASE_MB_ONLY")
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert not verdict["authorized"]
    assert "requested_scope_exceeds_authorization" in verdict["failures"]


def test_matching_authorized_fixture_passes_without_training(observed):
    """A synthetic matching AUTHORIZED manifest validates -- no candidate is invoked."""
    verdict = C.validate_authorization(
        _authorized(observed), requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert verdict["authorized"] is True and verdict["failures"] == []
    assert verdict["allowed_scope"] == "FULL_V2_3_PILOT"


def test_authorization_rejects_a_mismatching_selected_gpu_identity(observed):
    manifest = _authorized(observed)
    manifest["training_hardware"] = {
        **manifest["training_hardware"],
        "expected_gpu_device_name": "A different selected device",
    }
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert "gpu_device_identity_mismatch" in verdict["failures"]


@pytest.mark.parametrize(
    ("runtime_over", "failure"),
    [
        ({"cuda_available": False}, "cuda_unavailable"),
        ({"bf16_supported": False}, "bf16_unsupported"),
    ],
)
def test_authorization_rejects_missing_cuda_or_bf16(observed, runtime_over, failure):
    bad_observed = dict(observed)
    bad_fingerprint = _synthetic_runtime_fingerprint(**runtime_over)
    bad_observed["base_runtime_fingerprint"] = bad_fingerprint
    manifest = _authorized(bad_observed)
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=bad_observed
    )
    assert failure in verdict["failures"]


def test_authorization_rejects_runtime_fingerprint_mismatch(observed):
    manifest = _authorized(observed)
    manifest["training_hardware"] = {
        **manifest["training_hardware"],
        "expected_base_runtime_fingerprint_sha256": "0" * 64,
    }
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert "base_runtime_fingerprint_sha256_mismatch" in verdict["failures"]


def test_authorization_rejects_fingerprint_repository_identity_divergence(observed):
    divergent_observed = dict(observed)
    original = observed["base_runtime_fingerprint"]
    body = {k: v for k, v in original.items() if k != "fingerprint_sha256"}
    body["repository"] = {**body["repository"], "head": "different-reviewed-head"}
    divergent_observed["base_runtime_fingerprint"] = {
        **body,
        "fingerprint_sha256": hashlib.sha256(C.canonical_json_bytes(body)).hexdigest(),
    }
    verdict = C.validate_authorization(
        _authorized(divergent_observed),
        requested_scope="FULL_V2_3_PILOT",
        observed=divergent_observed,
    )
    assert "base_runtime_fingerprint_repository_head_mismatch" in verdict["failures"]


def test_token_ceiling_above_contract_refused(observed):
    manifest = _authorized(observed) | {"pilot_trained_token_ceiling": 10**12}
    verdict = C.validate_authorization(
        manifest, requested_scope="FULL_V2_3_PILOT", observed=observed
    )
    assert "token_ceiling_invalid_or_above_contract" in verdict["failures"]


MEASURED_SECONDS = [0.25] * len(C.MB_MEASURED_UPDATES)
MEASURED_TPS = C.EFFECTIVE_BATCH_TOKENS / 0.25


def _mb_timings(seconds=None):
    """R3 Part 6 evidence shape: one record per measured update, 11..40 inclusive."""
    seconds = list(seconds if seconds is not None else MEASURED_SECONDS)
    assert len(seconds) == len(C.MB_MEASURED_UPDATES)
    return [
        {"update": u, "trained_tokens": C.TRAINED_TOKENS_PER_UPDATE, "wall_seconds": float(sec)}
        for u, sec in zip(C.MB_MEASURED_UPDATES, seconds, strict=True)
    ]


def _compile_evidence(compile_on):
    """A recorded execution-path observation consistent with the requested compile mode."""
    if compile_on:
        return {
            "compile_requested": True,
            "invoked_compiled_callable": True,
            "invoked_uncompiled_module": False,
            "realized_module_is_optimized_module": True,
            "compilation_materialized": True,
            "dynamo_unique_graphs": 3,
            "inductor_artifact_count": 7,
            "forward_invocations_match_geometry": True,
        }
    return {
        "compile_requested": False,
        "invoked_compiled_callable": False,
        "invoked_uncompiled_module": True,
        "realized_module_is_optimized_module": False,
        "compilation_materialized": False,
        "dynamo_unique_graphs": 0,
        "inductor_artifact_count": 0,
        "forward_invocations_match_geometry": True,
    }


def _fake_session(tmp_path, scope="FULL_V2_3_PILOT", root=None):
    """A validated ExecutionSession assembled directly, as open_session would produce one.

    Building one by hand grants nothing: R4's sole real executor accepts only canonical artifact
    paths, revalidates their bytes, and keeps model construction plus the update loop lexical-local.
    ExecutionSession is orchestrator metadata; this fixture only avoids re-reading a 26 GB release.
    """
    root = (Path(root) if root is not None else (tmp_path / "root")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "contract_sha256": "c" * 64,
        "implementation_head": "h",
        "execution_bundle_sha256": "b" * 64,
        "pilot_index_manifest_file_sha256": "d" * 64,
        "authorization_sha256": "a" * 64,
        "session_id": "s",
        "authorized_output_root": str(root),
        "authorized_scope": scope,
    }
    validated = {
        "observed": {
            "branch": "agent/retrain-pipeline-contracts",
            "head": "h",
            "contract_sha256": "c" * 64,
            "execution_bundle_sha256": "b" * 64,
            "serialized_index_lists_digest": "e" * 64,
            "pilot_index_manifest_file_sha256": "d" * 64,
            "stage_a_meta_sha256": "1" * 64,
            "stage_b_meta_sha256": "2" * 64,
            "output_root": str(root),
        },
        "serialized_index_lists_digest": "e" * 64,
        "pilot_index_manifest_file_sha256": "d" * 64,
        "authorization_sha256": "a" * 64,
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
            "stage_a_eval_sha256": "3" * 64,
            "stage_a_train_sha256": "4" * 64,
            "stage_b_eval_sha256": "5" * 64,
        },
        "fingerprint": _synthetic_runtime_fingerprint(),
        "stage_a": {"dataset": None, "blocks": 1},
        "stage_b": {"dataset": None, "blocks": 1},
        "effective_ceilings": _ceilings(),
        "identity": identity,
    }
    session_path = root / R.SESSION_FILENAME
    if session_path.is_file():
        session_sha256 = hashlib.sha256(session_path.read_bytes()).hexdigest()
    else:
        session_sha256 = R.write_immutable_artifact(
            session_path,
            {
                "schema_version": R.SESSION_SCHEMA,
                "contract_version": C.CONTRACT_VERSION,
                "session_id": validated["session_id"],
                "authorized_scope": scope,
                "authorized_output_root": str(root),
                "ledger_identity": dict(identity),
            },
        )
    ledger = R.TokenLedger(root / R.LEDGER_FILENAME, identity, _ceilings())
    return R.ExecutionSession(
        validated, ledger, session_path=session_path, session_sha256=session_sha256
    )


def _plan_entry(plan, candidate_id):
    return next(e for e in plan["plan"]["candidates"] if e["candidate_id"] == candidate_id)


def _bound(session, plan, candidate, extra):
    payload = dict(extra)
    payload["schema_version"] = R.CANDIDATE_RESULT_SCHEMA
    payload["contract_version"] = C.CONTRACT_VERSION
    payload.update(dict(R._session_bindings(session)))  # noqa: SLF001
    payload["ledger_identity"] = dict(session.ledger.identity)
    payload["peak_lr"] = candidate["peak_lr"]
    payload["candidate_spec_sha256"] = _plan_entry(plan, candidate["candidate_id"])[
        "candidate_spec_sha256"
    ]
    payload["phase_plan_sha256"] = plan["plan_sha256"]
    payload["session_sha256"] = session.session_sha256
    payload.setdefault("compile_evidence", _compile_evidence(bool(candidate["compile"])))
    payload.setdefault("canonical_compile_path", True)
    return payload


def _run_meta_doc(candidate, session, plan, payload, **over):
    bindings = dict(R._session_bindings(session))  # noqa: SLF001
    doc = {
        "schema_version": C.RUN_META_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "peak_lr": candidate["peak_lr"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": bool(candidate["compile"]),
        "model_seed": candidate["model_init_seed"],
        "train_order_seed": candidate["train_order_seed"],
        "output_dir": str(Path(candidate["output_dir"]).resolve()),
        "candidate_spec_sha256": payload["candidate_spec_sha256"],
        "phase_plan_sha256": payload["phase_plan_sha256"],
        "session_sha256": payload["session_sha256"],
        "session_id": session.session_id,
        "contract_sha256": bindings["contract_sha256"],
        "implementation_head": bindings["implementation_head"],
        "execution_bundle_sha256": bindings["execution_bundle_sha256"],
        "pilot_index_manifest_file_sha256": bindings["pilot_index_manifest_file_sha256"],
        "runtime_fingerprint_sha256": bindings["runtime_fingerprint_sha256"],
        "authorization_sha256": bindings["authorization_sha256"],
        "lr_configuration": {
            "peak_lr": candidate["peak_lr"],
            "warmup_updates": candidate["warmup_updates"],
            "updates": candidate["updates"],
        },
        "ledger_identity": dict(session.ledger.identity),
    }
    doc.update(over)
    return doc


def _bound_mb_result(candidate, session, plan, **over):
    r = _bound(
        session,
        plan,
        candidate,
        {
            "phase": "MB",
            "candidate_id": candidate["candidate_id"],
            "seed_label": candidate["seed_label"],
            "micro_bsz": candidate["micro_bsz"],
            "grad_accum": candidate["grad_accum"],
            "compile": candidate["compile"],
            "completed_updates": 40,
            "update_timings": _mb_timings(),
            "median_update_tokens_per_second": MEASURED_TPS,
            "max_memory_reserved_bytes": 8 * 1024**3,
            "oom": False,
            "uncontrolled_exception": False,
            "all_losses_finite": True,
            "all_grad_norms_finite": True,
            "all_optimizer_states_instantiated": True,
            "grouping_matches_contract": True,
            "all_lr_ratios_are_one": True,
            "output_dir": candidate["output_dir"],
        },
    )
    r.update(over)
    eligible, failures = C.mb_candidate_eligible(
        r, _synthetic_runtime_fingerprint()["gpu"]["total_vram_bytes"]
    )
    r.update({"eligible": eligible, "eligibility_failures": list(failures)})
    r["terminal_status"] = "SUCCESS" if eligible else "CANDIDATE_INELIGIBLE"
    return r


def _bound_lr_result(candidate, session, plan, base_score, **over):
    weight = 1024.0
    numerator = float(base_score) * weight
    loss = numerator / weight
    losses_by_update = {str(update): 3.0 for update in range(1, C.LR_RUN_UPDATES + 1)}
    divergence_detail = C.sustained_divergence({int(k): v for k, v in losses_by_update.items()})
    r = _bound(
        session,
        plan,
        candidate,
        {
            "phase": "LR",
            "candidate_id": candidate["candidate_id"],
            "peak_lr": candidate["peak_lr"],
            "seed_label": candidate["seed_label"],
            "micro_bsz": candidate["micro_bsz"],
            "grad_accum": candidate["grad_accum"],
            "compile": candidate["compile"],
            "completed_updates": 200,
            "losses_by_update": losses_by_update,
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
            "divergence_detail": divergence_detail,
            "output_dir": candidate["output_dir"],
        },
    )
    r.update(over)
    eligible, failures = C.lr_candidate_eligible(r)
    r.update({"eligible": eligible, "eligibility_failures": list(failures)})
    r["terminal_status"] = "SUCCESS" if eligible else "CANDIDATE_INELIGIBLE"
    return r


def _ineligible_mb_result(candidate, session, plan, *, message="simulated failure", oom=False):
    return _bound_mb_result(
        candidate,
        session,
        plan,
        completed_updates=0,
        update_timings=[],
        median_update_tokens_per_second=0.0,
        max_memory_reserved_bytes=0,
        oom=oom,
        uncontrolled_exception=not oom,
        all_losses_finite=False,
        all_grad_norms_finite=False,
        all_optimizer_states_instantiated=False,
        grouping_matches_contract=False,
        all_lr_ratios_are_one=False,
        compile_evidence={
            **_compile_evidence(bool(candidate["compile"])),
            "forward_invocations_match_geometry": False,
        },
        canonical_compile_path=False,
        reason="oom" if oom else "candidate_runtime_exception",
        detail=message,
    )


def _ineligible_lr_result(candidate, session, plan, *, message="simulated failure", oom=False):
    result = _bound_lr_result(
        candidate,
        session,
        plan,
        3.0,
        completed_updates=0,
        all_losses_finite=False,
        all_grad_norms_finite=False,
        all_parameters_finite=False,
        muon_momentum_states_present=False,
        aux_adamw_states_present=False,
        grouping_matches_contract=False,
        all_lr_ratios_are_one=False,
        eval_stage_a_numerator=None,
        eval_stage_a_weight=None,
        eval_stage_b_numerator=None,
        eval_stage_b_weight=None,
        eval_loss_stage_a=None,
        eval_loss_stage_b=None,
        score=None,
        compile_evidence={
            **_compile_evidence(bool(candidate["compile"])),
            "forward_invocations_match_geometry": False,
        },
        canonical_compile_path=False,
        reason="oom" if oom else "candidate_runtime_exception",
        detail=message,
        oom=oom,
        uncontrolled_exception=not oom,
    )
    result.pop("losses_by_update")
    result.pop("divergence_detail")
    return result


def _write_result(candidate, session, plan, payload, meta_over=None):
    """Publish an immutable synthetic run_meta -> result -> receipt -> terminal chain."""
    ledger = session.ledger
    ledger.begin_candidate({
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "peak_lr": candidate["peak_lr"],
        "planned_updates": candidate["updates"],
        "candidate_spec_sha256": payload["candidate_spec_sha256"],
        "phase_plan_sha256": plan["plan_sha256"],
        "session_id": session.session_id,
        "session_sha256": session.session_sha256,
        "authorization_sha256": session.validated["authorization_sha256"],
    })
    completed = int(payload["completed_updates"])
    with ledger._lock():  # noqa: SLF001 - fast synthetic accounting, never a training path
        ledger._reload_locked()  # noqa: SLF001
        active = ledger.state["active_candidate"]
        tokens = completed * C.TRAINED_TOKENS_PER_UPDATE
        phase = candidate["phase"]
        for bucket in (phase, "GLOBAL"):
            ledger.state["reserved_tokens"][bucket] += tokens
            ledger.state["completed_tokens"][bucket] += tokens
        ledger.state["reserved_updates"] += completed
        ledger.state["completed_updates"] += completed
        active["candidate_reserved_updates"] = completed
        active["candidate_completed_updates"] = completed
        ledger._require_structural_invariants(ledger.state)  # noqa: SLF001
        ledger._write(ledger.state)  # noqa: SLF001

    out = Path(candidate["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    meta = _run_meta_doc(candidate, session, plan, payload, **(meta_over or {}))
    meta_sha256 = R.write_immutable_artifact(out / "run_meta.json", meta)
    published = dict(payload)
    published.setdefault("run_meta_sha256", meta_sha256)
    published["ledger_snapshot"] = ledger.snapshot()
    result_sha256 = R.write_immutable_artifact(out / "result.json", published)
    receipt = ledger.finalize_candidate(
        terminal_status=published["terminal_status"],
        run_meta_sha256=meta_sha256,
        result_sha256=result_sha256,
    )
    R.write_terminal_result(
        out,
        {
            "terminal_status": published["terminal_status"],
            "error_class": None,
            "error_message": None,
            "phase": candidate["phase"],
            "candidate_id": candidate["candidate_id"],
            "peak_lr": candidate["peak_lr"],
            "planned_updates": candidate["updates"],
            "candidate_spec_sha256": payload["candidate_spec_sha256"],
            "phase_plan_sha256": plan["plan_sha256"],
            "session_id": session.session_id,
            "session_sha256": session.session_sha256,
            "authorization_sha256": session.validated["authorization_sha256"],
            "ledger_identity": dict(ledger.identity),
            "ledger_receipt_sha256": receipt["receipt_sha256"],
            "reserved_updates": receipt["candidate_reserved_updates"],
            "completed_updates": receipt["candidate_completed_updates"],
            "reserved_tokens": receipt["after_reserved_tokens"],
            "completed_tokens": receipt["after_completed_tokens"],
            "ledger_reserved_updates": receipt["after_reserved_updates"],
            "ledger_completed_updates": receipt["after_completed_updates"],
            "run_meta_sha256": meta_sha256,
            "result_sha256": result_sha256,
        },
    )
    return published


def _published_mb_plan(session):
    """Publish the immutable Phase-MB plan the launcher tests need."""
    candidates = R.plan_phase_mb(output_root=session.output_root)
    plan = R.publish_phase_plan(
        root=session.output_root,
        plan_kind="PHASE_MB_PLAN",
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        candidates=candidates,
        derived_from={"source": "the frozen P-PILOT-CONTRACT-V2.3 Phase-MB grid"},
    )
    return candidates, plan


def _mb_launcher(**over):
    """A fake launcher. It cannot reach the real training backend: the signature differs."""

    def launcher(candidate, session, plan):
        _write_result(candidate, session, plan, _bound_mb_result(candidate, session, plan, **over))

    return launcher


def test_no_real_backend_accepts_a_constructed_session_or_context(tmp_path):
    """R4 Part 1: caller-built state has no callable route to candidate update logic."""
    for removed in (
        "WorkerAuthority",
        "_WORKER_AUTHORITY_MINT",
        "REAL_TRAINING_ENTRYPOINTS",
        "execute_validated_candidate",
        "_run_updates",
        "_train_phase_mb",
        "_train_phase_lr",
        "build_pilot_model",
        "build_pilot_optimizer",
    ):
        assert not hasattr(R, removed), removed
    with pytest.raises(TypeError):
        R.execute_candidate_from_artifact_paths(session=_fake_session(tmp_path))


def test_phase_mb_only_scope_cannot_run_lr(tmp_path):
    session = _fake_session(tmp_path, scope="PHASE_MB_ONLY")
    with pytest.raises(R.BindingFailure, match="PHASE_MB_ONLY"):
        R.orchestrate_phase_muon_lr(session, launcher=lambda *_: None)


def test_orchestrator_runs_exactly_the_ten_candidates(tmp_path):
    """A fake launcher records what the orchestrator chose to launch."""
    session = _fake_session(tmp_path)
    launched = []

    def launcher(candidate, given, plan):
        assert isinstance(given, R.ExecutionSession)
        assert plan["plan"]["plan_kind"] == "PHASE_MB_PLAN"
        launched.append(candidate["candidate_id"])
        _write_result(candidate, given, plan, _bound_mb_result(candidate, given, plan))

    report = R.orchestrate_phase_mb(session, launcher=launcher)
    assert launched == list(C.MB_REQUIRED_CANDIDATE_IDS)
    assert len(launched) == 10
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"


def test_candidate_local_failure_becomes_ineligible_evidence_and_grid_continues(tmp_path):
    session = _fake_session(tmp_path)
    launched = []

    def launcher(candidate, given, plan):
        launched.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "mb_micro16_compileoff":
            _write_result(candidate, given, plan, _ineligible_mb_result(candidate, given, plan))
            return
        _write_result(candidate, given, plan, _bound_mb_result(candidate, given, plan))

    report = R.orchestrate_phase_mb(session, launcher=launcher)
    assert len(launched) == 10, "the grid must continue after a candidate-local failure"
    failed = [c for c in report["candidates"] if c["candidate_id"] == "mb_micro16_compileoff"]
    assert failed and failed[0]["eligible"] is False
    assert failed[0]["reason"] == "candidate_runtime_exception"
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"


def test_phase_level_binding_failure_aborts(tmp_path):
    session = _fake_session(tmp_path)

    def launcher(candidate, given, plan):
        raise C.PilotContractError("accepted-release binding failed")

    with pytest.raises(R.PhaseAbort, match="accepted-release binding failed"):
        R.orchestrate_phase_mb(session, launcher=launcher)


def test_result_must_bind_this_execution(tmp_path):
    session = _fake_session(tmp_path)

    def launcher(candidate, given, plan):
        _write_result(
            candidate,
            given,
            plan,
            _bound_mb_result(candidate, given, plan, contract_sha256="WRONG"),
        )

    with pytest.raises(R.BindingFailure, match="does not bind this execution"):
        R.orchestrate_phase_mb(session, launcher=launcher)


def _frozen_mb_report(session, micro_bsz=8, compile_on=False):
    """Publish an authoritative Phase-MB report that freezes a specific geometry."""

    def launcher(candidate, given, plan):
        fast = candidate["micro_bsz"] == micro_bsz and candidate["compile"] is compile_on
        seconds = [0.25 if fast else 1.0] * len(C.MB_MEASURED_UPDATES)
        timings = _mb_timings(seconds)
        _write_result(
            candidate,
            given,
            plan,
            _bound_mb_result(
                candidate,
                given,
                plan,
                update_timings=timings,
                median_update_tokens_per_second=C.mb_median_update_tokens_per_second(timings),
            ),
        )

    return R.orchestrate_phase_mb(session, launcher=launcher)


def test_lr_orchestrator_derives_grid_confirmation_and_edge(tmp_path):
    """Nothing about which LR candidates run, or with what geometry, comes from caller input."""
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=8, compile_on=False)
    launched = []
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}

    def launcher(candidate, given, plan):
        launched.append((candidate["peak_lr"], candidate["seed_label"]))
        assert candidate["micro_bsz"] == 8 and candidate["compile"] is False
        _write_result(
            candidate,
            given,
            plan,
            _bound_lr_result(candidate, given, plan, scores[candidate["peak_lr"]]),
        )

    report = R.orchestrate_phase_muon_lr(session, launcher=launcher)
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


def test_fingerprint_records_and_binds_the_exact_selected_cuda_runtime(monkeypatch):
    import torch

    props = mock.Mock(total_memory=SYNTHETIC_VRAM_BYTES, major=9, minor=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: SYNTHETIC_GPU_NAME)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(R, "_nvidia_smi", lambda field: "synthetic-driver")

    fingerprint = R.base_runtime_fingerprint()
    assert fingerprint["gpu"] == {
        "cuda_available": True,
        "device_count": 1,
        "selected_device_index": 0,
        "name": SYNTHETIC_GPU_NAME,
        "total_vram_mib": SYNTHETIC_VRAM_BYTES // (1024 * 1024),
        "total_vram_bytes": SYNTHETIC_VRAM_BYTES,
        "capability": "9.0",
        "driver": "synthetic-driver",
        "cuda_runtime": "12.8",
        "bf16_supported": True,
    }
    assert fingerprint["python_implementation"]
    assert (
        R.base_runtime_fingerprint(
            gpu_required=True,
            hardware_binding=_hardware_binding(fingerprint),
        )
        == fingerprint
    )


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
    # R3 Part 5: the producer divides by the accumulated weight exactly, guarded by a
    # positive-weight requirement, so the parent's recomputation can agree exactly.
    assert "total_numerator / total_weight" in src
    assert "max(1.0, total_weight)" not in src


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


def _executor_nested_source(name):
    """Return one lexical-local function from R4's sole artifact-path executor."""
    import ast
    import inspect

    source = inspect.getsource(R.execute_candidate_from_artifact_paths)
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


def test_exact_block_consumption_mb_and_lr_is_local_to_the_sole_executor():
    assert C.MB_PROBE_UPDATES * C.SEQUENCES_PER_OPTIMIZER_UPDATE == 5120
    assert C.LR_RUN_UPDATES * C.SEQUENCES_PER_OPTIMIZER_UPDATE == 25600
    assert C.LR_BLOCKS_PER_RUN == 25600
    src = _executor_nested_source("run_updates")
    assert "required_blocks" in src
    assert "no replay" in src or "without replay" in src
    assert "consumed {cursor} blocks, expected exactly {required_blocks}" in src
    assert not hasattr(R, "_run_updates"), "the update loop must not be module-callable"


def test_update_loop_preflights_consumption_before_any_ledger_reservation():
    """An undersized view is refused by the lexical loop before budget can be consumed."""
    import ast

    src = _executor_nested_source("run_updates")
    fn = ast.parse(src).body[0]
    update_loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and getattr(n.target, "id", None) == "update"
    )
    reserve = next(
        n
        for n in ast.walk(update_loop)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "reserve"
    )
    preflight = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "require"
        and n.lineno < update_loop.lineno
    )
    assert preflight.lineno < reserve.lineno
    assert "len(view) >= required_blocks" in src
    assert "cursor == required_blocks" in src


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
    """R4: one path-only executor owns validation, construction, and every update primitive."""
    import ast
    import inspect

    tree = _runner_ast()
    executor = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "execute_candidate_from_artifact_paths"
    )
    lexical_locals = {n.name for n in executor.body if isinstance(n, ast.FunctionDef)}
    assert {"construct_model", "construct_optimizer", "run_updates", "train_candidate"} <= (
        lexical_locals
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "validate_worker_execution"
        for n in ast.walk(executor)
    )
    source = inspect.getsource(R.execute_candidate_from_artifact_paths)
    assert source.index("execution = validate_worker_execution(") < source.index("import torch")
    update_source = _executor_nested_source("run_updates")
    for operation in ("loss.backward()", "optimizer.step()", "ledger.reserve(phase)"):
        assert operation in update_source
    for removed in (
        "WorkerAuthority",
        "_WORKER_AUTHORITY_MINT",
        "execute_validated_candidate",
        "_train_phase_mb",
        "_train_phase_lr",
        "_run_updates",
    ):
        assert not hasattr(R, removed), removed
    cli = inspect.getsource(R._cli_internal_worker)  # noqa: SLF001
    assert "execute_candidate_from_artifact_paths(" in cli
    assert "validate_worker_execution(" not in cli


def test_the_validator_accepts_only_artifact_paths_and_the_requested_phase():
    """No caller-supplied manifest object, hash, count, authority or context is accepted."""
    import inspect

    params = inspect.signature(R.validate_execution_artifacts).parameters
    assert set(params) == {
        "authorization_path",
        "pilot_index_manifest_path",
        "output_dir",
        "requested_phase",
        "gpu_required",
    }
    for name in ("authorized", "manifest", "context", "stage_a_blocks", "expected_sha256"):
        assert name not in params


def test_the_real_worker_entry_takes_canonical_artifact_paths_only(tmp_path):
    """R3 Part 1: every raw input to the real worker is a path to a canonical artifact."""
    import inspect

    params = inspect.signature(R.validate_worker_execution).parameters
    assert set(params) == set(R.REAL_WORKER_ARTIFACT_INPUTS) | {"gpu_required"}
    for required in (
        "authorization_path",
        "session_manifest_path",
        "phase_plan_path",
        "candidate_spec_path",
        "pilot_index_manifest_path",
        "accepted_stage_a_path",
        "accepted_stage_b_path",
        "ledger_path",
        "candidate_output_path",
    ):
        assert required in params, required
    for banned in ("session", "context", "validated", "authorized", "candidate", "backend"):
        assert banned not in params
    # It revalidates the bytes itself rather than trusting anything about its caller.
    src = inspect.getsource(R.validate_worker_execution)
    assert "validate_execution_artifacts(" in src
    assert "validate_session_manifest(" in src
    assert "validate_phase_plan(" in src
    assert "validate_complete_phase_plan_membership(" in src


def test_the_old_trust_in_context_api_is_gone():
    for removed in (
        "build_validated_context",
        "ValidatedExecutionContext",
        "run_phase_mb_candidate",
        "run_phase_lr_candidate",
    ):
        assert not hasattr(R, removed), removed
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "manifest_path=None" not in src
    assert '"authorized"' not in src or 'context["authorized"]' not in src


def test_validator_refuses_when_the_authorization_file_is_absent(tmp_path):
    with pytest.raises(R.BindingFailure, match="NOT_AUTHORIZED"):
        R.validate_execution_artifacts(
            authorization_path=tmp_path / "absent.json",
            pilot_index_manifest_path=tmp_path / "absent_indices.json",
            output_dir=tmp_path / "out",
            requested_phase="MB",
            gpu_required=False,
        )


def test_binding_failure_is_never_downgraded_to_an_ineligible_candidate(tmp_path):
    session = _fake_session(tmp_path)

    def launcher(candidate, given, plan):
        raise R.BindingFailure("forged release identity")

    with pytest.raises(R.BindingFailure, match="forged release identity"):
        R.orchestrate_phase_mb(session, launcher=launcher)


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


def _publish_worker_chain(candidate, session, plan, status="SUCCESS"):
    """Publish the immutable raw result, receipt, and strict candidate-local terminal."""
    if status == "SUCCESS":
        payload = _bound_mb_result(candidate, session, plan)
    else:
        payload = _bound_mb_result(
            candidate,
            session,
            plan,
            completed_updates=0,
            update_timings=[],
            median_update_tokens_per_second=0.0,
            all_losses_finite=False,
            all_grad_norms_finite=False,
            all_optimizer_states_instantiated=False,
            grouping_matches_contract=False,
            all_lr_ratios_are_one=False,
        )
        payload["terminal_status"] = status
    return _write_result(candidate, session, plan, payload)


def test_parent_hands_the_child_only_canonical_artifact_paths(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]

    def handler(argv, kwargs):
        _publish_worker_chain(candidate, session, plan)
        return _FakeCompleted(0, "out", "err")

    calls = _fake_subprocess(monkeypatch, handler)
    R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001
    argv = calls[0][0]
    assert argv[2] == "internal-worker"
    flags = {argv[i]: argv[i + 1] for i in range(len(argv)) if str(argv[i]).startswith("--")}
    assert flags["--authorization"] == session.validated["authorization_path"]
    assert flags["--pilot-index-manifest"] == session.validated["index_manifest_path"]
    assert flags["--session-manifest"] == str(session.session_path)
    assert flags["--phase-plan"] == str(plan["plan_path"])
    assert flags["--ledger"] == str(session.output_root / R.LEDGER_FILENAME)
    assert flags["--candidate-output"] == candidate["output_dir"]
    assert flags["--accepted-stage-a"].endswith(R.ACCEPTED_STAGE_A)
    assert flags["--accepted-stage-b"].endswith(R.ACCEPTED_STAGE_B)
    # Every raw input is a path; nothing else is passed at all.
    assert set(flags) == {f"--{n.rsplit('_path', 1)[0].replace('_', '-')}" for n in ()} | {
        "--authorization",
        "--session-manifest",
        "--phase-plan",
        "--candidate-spec",
        "--pilot-index-manifest",
        "--accepted-stage-a",
        "--accepted-stage-b",
        "--ledger",
        "--candidate-output",
    }
    spec = json.loads(Path(flags["--candidate-spec"]).read_text(encoding="utf-8"))
    # The spec carries the candidate and its session binding -- no hashes, counts or authority.
    assert set(spec) == {
        "schema_version",
        "contract_version",
        "session_id",
        "session_sha256",
        "plan_kind",
        "phase",
        "candidate",
    }
    assert spec["candidate"] == candidate


def test_the_worker_cli_requires_every_artifact_path(tmp_path):
    """R3 Part 2: the internal worker has no optional inputs and no legacy public command."""
    with pytest.raises(SystemExit) as excinfo:
        R.main(["execute-candidate", "--spec", str(tmp_path / "s.json")])
    assert excinfo.value.code == 2  # the old public command no longer exists
    with pytest.raises(SystemExit) as excinfo:
        R.main(["internal-worker", "--candidate-spec", str(tmp_path / "s.json")])
    assert excinfo.value.code == 2


def _worker_argv(tmp_path, spec, **over):
    argv = {
        "--authorization": str(tmp_path / "auth.json"),
        "--session-manifest": str(tmp_path / "root" / R.SESSION_FILENAME),
        "--phase-plan": str(tmp_path / "root" / R.PHASE_MB_PLAN_FILENAME),
        "--candidate-spec": str(spec),
        "--pilot-index-manifest": str(tmp_path / "PILOT_INDICES.json"),
        "--accepted-stage-a": str(REPO_ROOT / R.ACCEPTED_STAGE_A),
        "--accepted-stage-b": str(REPO_ROOT / R.ACCEPTED_STAGE_B),
        "--ledger": str(tmp_path / "root" / R.LEDGER_FILENAME),
        "--candidate-output": str(tmp_path / "root" / "mb_micro8_compileoff"),
    }
    argv.update(over)
    return ["internal-worker", *[v for kv in argv.items() for v in kv]]


def _self_declared_spec(tmp_path, **over):
    spec = tmp_path / "self_declared.json"
    doc = {
        "schema_version": R.CANDIDATE_SPEC_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "session_id": "s",
        "session_sha256": "deadbeef",
        "plan_kind": "PHASE_MB_PLAN",
        "phase": "MB",
        "candidate": R.plan_phase_mb(output_root=tmp_path / "root")[4],
    }
    doc.update(over)
    spec.write_bytes(C.canonical_json_bytes(doc))
    return spec


def test_a_self_declared_candidate_spec_is_never_executable(tmp_path):
    """R4: prevalidation refusal confers no capability and writes no terminal artifact."""
    spec = _self_declared_spec(tmp_path)
    code = R.main(_worker_argv(tmp_path, spec))
    assert code == R.BINDING_FAILURE
    candidate_output = tmp_path / "root" / "mb_micro8_compileoff"
    assert not candidate_output.exists()
    assert not R.terminal_result_path(candidate_output).exists()
    assert not spec.with_suffix(".terminal.json").exists()


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
        R.orchestrate_phase_muon_lr(session, launcher=lambda *_: None)


def test_full_pilot_matching_mb_to_lr_runtime_binding_passes(tmp_path):
    session = _fake_session(tmp_path, scope="FULL_V2_3_PILOT")
    report = _frozen_mb_report(session)
    assert report["session_terminates_after_this_phase"] is False
    assert report["base_runtime_fingerprint"] == session.validated["fingerprint"]
    assert report["physical_vram_bytes"] == SYNTHETIC_VRAM_BYTES
    frozen = R.load_authoritative_mb_report(session)
    assert frozen["micro_bsz"] == 8
    assert report["session_id"] == session.session_id
    assert session.ledger.identity["session_id"] == session.session_id


def test_full_pilot_mb_to_lr_runtime_mismatch_aborts_before_lr(tmp_path):
    session = _fake_session(tmp_path, scope="FULL_V2_3_PILOT")
    _frozen_mb_report(session)
    session.validated["fingerprint"] = _synthetic_runtime_fingerprint(
        name="A different owner-selected synthetic device",
        driver="a different synthetic driver",
    )
    launched = []

    with pytest.raises(R.BindingFailure, match="runtime fingerprint is incompatible"):
        R.orchestrate_phase_muon_lr(session, launcher=lambda *args: launched.append(args))
    assert launched == []


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
    R.write_immutable_artifact(
        root / R.SESSION_FILENAME,
        {
            "schema_version": R.SESSION_SCHEMA,
            "session_id": "OLD",
            "authorized_scope": "PHASE_MB_ONLY",
        },
    )
    validated = _fake_session(tmp_path, root=tmp_path / "other").validated
    validated["authorized_root"] = root.resolve()
    monkeypatch.setattr(R, "validate_execution_artifacts", lambda **kw: validated)
    with pytest.raises(R.BindingFailure, match="does not bind this execution"):
        R.open_session(
            authorization_path=tmp_path / "a.json",
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

    def launcher(candidate, given, plan):
        _write_result(
            candidate, given, plan, _ineligible_mb_result(candidate, given, plan, oom=True)
        )

    with pytest.raises(R.PhaseAbort, match="did not freeze a geometry"):
        R.orchestrate_phase_mb(session, launcher=launcher)
    report_path = session.output_root / R.MB_REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.with_suffix(".sha256").is_file()
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


def test_the_ledger_reserves_before_the_optimizer_update_and_completes_after():
    """The non-callable lexical loop keeps the conservative transition order."""
    import ast

    fn = ast.parse(_executor_nested_source("run_updates")).body[0]
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and getattr(n.target, "id", None) == "update"
    )
    calls = {
        getattr(n.func, "attr", None): n.lineno
        for n in ast.walk(loop)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) in {"reserve", "step", "complete"}
    }
    assert calls["reserve"] < calls["step"] < calls["complete"]
    progress_write = next(
        n
        for n in ast.walk(loop)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript) and getattr(t.value, "id", None) == "progress"
            for t in n.targets
        )
    )
    assert calls["complete"] < progress_write.lineno


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

    def launcher(candidate, given, plan):
        launched.append((candidate["micro_bsz"], candidate["compile"]))
        _write_result(candidate, given, plan, _bound_lr_result(candidate, given, plan, 3.0))

    R.orchestrate_phase_muon_lr(session, launcher=launcher)
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
        R.orchestrate_phase_muon_lr(session, launcher=lambda *_: None)


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
    with pytest.raises(R.BindingFailure, match="raw candidate evidence"):
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
    with pytest.raises(C.PilotContractError, match="raw candidate evidence"):
        R.load_authoritative_mb_report(session)


# ================================ R2 Part 7: every selection number is recomputed from evidence


def test_mb_throughput_is_the_median_of_the_per_update_rates():
    """R3 Part 6: median(tokens/seconds), computed from records -- not tokens/median(seconds)."""
    records = _mb_timings([0.2, 0.4] + [0.3] * 28)
    rates = [C.TRAINED_TOKENS_PER_UPDATE / r["wall_seconds"] for r in records]
    import statistics

    assert C.mb_median_update_tokens_per_second(records) == statistics.median(rates)
    assert [C.mb_update_tokens_per_second(r) for r in records] == rates


def test_median_of_rates_is_not_tokens_over_median_seconds():
    """A regression fixture where the two statistics genuinely disagree."""
    import statistics

    # An even sample: the median averages the two middle values, and averaging rates is not
    # the reciprocal of averaging seconds.
    seconds = [0.10] * 15 + [1.00] * 15
    records = _mb_timings(seconds)
    median_of_rates = C.mb_median_update_tokens_per_second(records)
    tokens_over_median_seconds = C.TRAINED_TOKENS_PER_UPDATE / statistics.median(seconds)
    assert median_of_rates == pytest.approx(
        (C.TRAINED_TOKENS_PER_UPDATE / 0.10 + C.TRAINED_TOKENS_PER_UPDATE / 1.00) / 2
    )
    assert tokens_over_median_seconds == pytest.approx(C.TRAINED_TOKENS_PER_UPDATE / 0.55)
    assert median_of_rates != pytest.approx(tokens_over_median_seconds, rel=0.1)
    # The contract names the former; the recomputation must reproduce exactly that.
    assert median_of_rates > tokens_over_median_seconds


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t[:-1], "no record for measured update"),
        (lambda t: [*t, dict(t[-1])], "more than once"),
        (lambda t: [*t, {**t[-1], "update": 41}], "unmeasured update"),
        (lambda t: [{**r, "update": r["update"] + 1} for r in t], "no record for measured update"),
        (lambda t: [{**t[0], "trained_tokens": 1}, *t[1:]], "trained tokens"),
        (lambda t: [{**t[0], "wall_seconds": 0.0}, *t[1:]], "non-positive"),
        (lambda t: [{k: v for k, v in t[0].items() if k != "update"}, *t[1:]], "is missing"),
    ],
)
def test_mb_timing_records_must_cover_11_to_40_exactly(mutate, message):
    with pytest.raises(C.PilotContractError, match=message):
        C.require_exact_mb_timing_records(mutate(_mb_timings()))


def test_mb_timing_records_accept_exactly_the_measured_window():
    records = C.require_exact_mb_timing_records(_mb_timings())
    assert [r["update"] for r in records] == list(range(11, 41))
    assert len(records) == 30


def test_mb_recomputation_rejects_a_forged_median(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    forged = _bound_mb_result(candidates[0], session, plan, median_update_tokens_per_second=1e9)
    with pytest.raises(R.BindingFailure, match="disagrees with the median of the per-update rates"):
        R.verify_recomputed_mb_result(forged)


def test_mb_recomputation_rejects_a_median_computed_the_wrong_way(tmp_path):
    """tokens / median(seconds) is a different statistic and is refused as a stored median."""
    import statistics

    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    seconds = [0.10] * 15 + [1.00] * 15
    timings = _mb_timings(seconds)
    wrong = C.TRAINED_TOKENS_PER_UPDATE / statistics.median(seconds)
    forged = _bound_mb_result(
        candidates[0],
        session,
        plan,
        update_timings=timings,
        median_update_tokens_per_second=wrong,
    )
    with pytest.raises(R.BindingFailure, match="disagrees with the median of the per-update rates"):
        R.verify_recomputed_mb_result(forged)
    good = _bound_mb_result(
        candidates[0],
        session,
        plan,
        update_timings=timings,
        median_update_tokens_per_second=C.mb_median_update_tokens_per_second(timings),
    )
    assert R.verify_recomputed_mb_result(good)["measured_updates"] == 30


def test_mb_recomputation_rejects_the_wrong_number_of_measured_updates(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    short = _bound_mb_result(candidates[0], session, plan, update_timings=_mb_timings()[:5])
    with pytest.raises(R.BindingFailure, match="timing evidence is unusable"):
        R.verify_recomputed_mb_result(short)


def test_mb_recomputation_rejects_missing_timings(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    with pytest.raises(R.BindingFailure, match="timing evidence is unusable"):
        R.verify_recomputed_mb_result(
            _bound_mb_result(candidates[0], session, plan, update_timings=[])
        )


def _lr_plan(session, peak_lrs=(2e-4,), seed_label="seed-1", plan_kind="PHASE_LR_INITIAL_PLAN"):
    candidates = R.plan_phase_lr(
        output_root=session.output_root,
        micro_bsz=8,
        compile_on=False,
        peak_lrs=list(peak_lrs),
        seed_label=seed_label,
    )
    plan = R.publish_phase_plan(
        root=session.output_root,
        plan_kind=plan_kind,
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        candidates=candidates,
        derived_from={
            "frozen_geometry": {"micro_bsz": 8, "grad_accum": 16, "compile": False},
            "peak_lrs": [float(v) for v in peak_lrs],
            "seed_labels": [seed_label],
        },
    )
    return candidates, plan


def test_lr_score_is_recomputed_from_the_raw_evaluation_components(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _lr_plan(session)
    good = _bound_lr_result(candidates[0], session, plan, 3.25)
    recomputed = R.verify_recomputed_lr_result(good)
    assert recomputed["score"] == good["score"]
    assert recomputed["score_weights"] == [10, 3]
    assert recomputed["score_formula"] == "(10*loss_A + 3*loss_B)/13"


def test_lr_score_is_the_weighted_recomputation_of_the_raw_components(tmp_path):
    """R3 Part 5: SCORE = (10*loss_A + 3*loss_B)/13, from the raw numerators and weights."""
    session = _fake_session(tmp_path)
    candidates, plan = _lr_plan(session)
    result = _bound_lr_result(
        candidates[0],
        session,
        plan,
        3.0,
        eval_stage_a_numerator=2400.0,
        eval_stage_a_weight=800.0,
        eval_stage_b_numerator=1200.0,
        eval_stage_b_weight=300.0,
        eval_loss_stage_a=3.0,
        eval_loss_stage_b=4.0,
        score=(10 * 3.0 + 3 * 4.0) / 13,
    )
    recomputed = R.verify_recomputed_lr_result(result)
    assert recomputed["eval_loss_stage_a"] == 2400.0 / 800.0
    assert recomputed["eval_loss_stage_b"] == 1200.0 / 300.0
    assert recomputed["score"] == pytest.approx((10 * 3.0 + 3 * 4.0) / 13)


def test_internally_consistent_losses_with_a_wrong_stored_score_are_rejected(tmp_path):
    """The raw losses agree with their own numerators; only the summary SCORE is forged."""
    session = _fake_session(tmp_path)
    candidates, plan = _lr_plan(session)
    result = _bound_lr_result(
        candidates[0],
        session,
        plan,
        3.0,
        eval_stage_a_numerator=2400.0,
        eval_stage_a_weight=800.0,
        eval_stage_b_numerator=1200.0,
        eval_stage_b_weight=300.0,
        eval_loss_stage_a=3.0,
        eval_loss_stage_b=4.0,
        score=0.5,  # internally consistent losses, deliberately wrong summary
    )
    with pytest.raises(R.BindingFailure, match="serialized score"):
        R.verify_recomputed_lr_result(result)


def test_selectors_consume_only_the_recomputed_lr_values(tmp_path):
    """The admitted record carries the recomputed score, so a stored one cannot be selected on."""
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    seen = {}

    def launcher(candidate, given, plan):
        payload = _bound_lr_result(candidate, given, plan, 3.0 + candidate["peak_lr"] * 1000)
        seen[candidate["candidate_id"]] = payload["score"]
        _write_result(candidate, given, plan, payload)

    report = R.orchestrate_phase_muon_lr(session, launcher=launcher)
    for record in report["seed1"]:
        recomputed = C.lr_score(
            record["eval_stage_a_numerator"] / record["eval_stage_a_weight"],
            record["eval_stage_b_numerator"] / record["eval_stage_b_weight"],
        )
        assert record["score"] == recomputed
        assert record["score_formula"] == "(10*loss_A + 3*loss_B)/13"


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
    candidates, plan = _lr_plan(session)
    forged = _bound_lr_result(candidates[0], session, plan, 3.25)
    forged.update(over)
    with pytest.raises(R.BindingFailure, match=message):
        R.verify_recomputed_lr_result(forged)


def test_a_forged_result_is_caught_by_the_orchestrator_not_only_in_isolation(tmp_path):
    session = _fake_session(tmp_path)

    def launcher(candidate, given, plan):
        _write_result(
            candidate,
            given,
            plan,
            _bound_mb_result(candidate, given, plan, median_update_tokens_per_second=1e9),
        )

    with pytest.raises(R.BindingFailure, match="disagrees with the median of the per-update rates"):
        R.orchestrate_phase_mb(session, launcher=launcher)


# ==================================================== R2 Part 8: evidence completeness


def test_the_initial_lr_grid_must_be_complete_evidence(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)

    def launcher(candidate, given, plan):
        if candidate["peak_lr"] == 4e-4:
            _write_result(candidate, given, plan, _ineligible_lr_result(candidate, given, plan))
            return
        _write_result(candidate, given, plan, _bound_lr_result(candidate, given, plan, 3.0))

    # An ineligible record is still a record: the grid is complete and selection may proceed.
    report = R.orchestrate_phase_muon_lr(session, launcher=launcher)
    assert {float(r["peak_lr"]) for r in report["seed1"]} == set(C.LR_GRID_SEED1)


def test_a_missing_initial_grid_record_aborts(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session)
    with mock.patch.object(R, "plan_phase_lr", _plan_without(4e-4)):
        with pytest.raises(C.PilotContractError, match="initial grid incomplete"):
            R.orchestrate_phase_muon_lr(
                session,
                launcher=lambda c, s, pl: _write_result(c, s, pl, _bound_lr_result(c, s, pl, 3.0)),
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
                launcher=lambda c, s, pl: _write_result(
                    c, s, pl, _bound_lr_result(c, s, pl, scores[c["peak_lr"]])
                ),
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
                launcher=lambda c, s, pl: _write_result(
                    c, s, pl, _bound_lr_result(c, s, pl, scores[c["peak_lr"]])
                ),
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


def _read_candidate_terminal(candidate, session, plan):
    return R.read_terminal_result(
        Path(candidate["output_dir"]),
        expected=R.terminal_expectations(session, planned=candidate, plan=plan),
        ledger=session.ledger,
    )


def _terminal_payload(candidate):
    doc = json.loads(
        R.terminal_result_path(Path(candidate["output_dir"])).read_text(encoding="utf-8")
    )
    doc.pop("schema_version")
    return doc


def test_terminal_result_round_trip(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _publish_worker_chain(candidate, session, plan)
    path = R.terminal_result_path(Path(candidate["output_dir"]))
    assert path == Path(candidate["output_dir"]) / "terminal.json"
    assert path.with_suffix(".sha256").is_file()
    doc = _read_candidate_terminal(candidate, session, plan)
    assert doc["terminal_status"] == "SUCCESS"
    assert doc["schema_version"] == R.TERMINAL_RESULT_SCHEMA
    assert set(doc) == set(R.TERMINAL_RESULT_FIELDS)


def test_terminal_result_requires_every_field(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _publish_worker_chain(candidate, session, plan)
    incomplete = _terminal_payload(candidate)
    incomplete.pop("completed_tokens")
    probe = tmp_path / "terminal_probe"
    probe.mkdir()
    with pytest.raises(C.PilotContractError, match="terminal result is incomplete"):
        R.write_terminal_result(probe, incomplete)


def test_an_unknown_terminal_status_is_refused(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _publish_worker_chain(candidate, session, plan)
    payload = _terminal_payload(candidate) | {"terminal_status": "FINE"}
    probe = tmp_path / "terminal_probe"
    probe.mkdir()
    with pytest.raises(C.PilotContractError, match="unknown terminal status"):
        R.write_terminal_result(probe, payload)


def test_a_missing_terminal_result_is_a_binding_failure_not_a_pass(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    with pytest.raises(R.BindingFailure, match="required artifact is missing"):
        R.read_terminal_result(
            tmp_path / "never_ran",
            expected=R.terminal_expectations(session, planned=candidates[0], plan=plan),
            ledger=session.ledger,
        )


@pytest.mark.parametrize(
    "status,expected",
    [
        ("CANDIDATE_INELIGIBLE", None),
        ("PHASE_ABORT", R.PhaseAbort),
        ("BINDING_FAILURE", R.BindingFailure),
    ],
)
def test_the_terminal_status_decides_how_the_parent_proceeds(
    tmp_path, monkeypatch, status, expected
):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]

    def handler(argv, kwargs):
        _publish_worker_chain(candidate, session, plan, status=status)
        return _FakeCompleted(R.RESULT_CLASSES[status], "", "")

    _fake_subprocess(monkeypatch, handler)
    if expected is None:
        terminal = R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001
        assert terminal["terminal_status"] == "CANDIDATE_INELIGIBLE"
    else:
        with pytest.raises(expected):
            R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001


def test_a_terminal_status_that_disagrees_with_the_exit_code_aborts(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]

    def handler(argv, kwargs):
        _publish_worker_chain(candidate, session, plan)
        return _FakeCompleted(3, "", "")  # claims success, exits ineligible

    _fake_subprocess(monkeypatch, handler)
    with pytest.raises(R.PhaseAbort, match="terminal protocol is broken"):
        R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001


def test_child_stdout_and_stderr_are_preserved_verbatim(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]

    def handler(argv, kwargs):
        _publish_worker_chain(candidate, session, plan)
        return _FakeCompleted(0, "child stdout line\n", "child stderr line\n")

    _fake_subprocess(monkeypatch, handler)
    R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001
    output = Path(candidate["output_dir"])
    assert (output / "worker.stdout").read_text(encoding="utf-8") == "child stdout line\n"
    assert (output / "worker.stderr").read_text(encoding="utf-8") == "child stderr line\n"


def test_the_worker_writes_nothing_when_artifact_validation_fails_early(tmp_path):
    spec = _self_declared_spec(tmp_path)
    code = R.main(_worker_argv(tmp_path, spec))
    assert code == R.BINDING_FAILURE
    output = tmp_path / "root" / "mb_micro8_compileoff"
    assert not output.exists()
    assert not R.terminal_result_path(output).exists()


# ============================================ R2 Part 12: timing names and honest evidence


def test_timing_fields_use_their_r2_names():
    src = _executor_nested_source("train_candidate") + _executor_nested_source("run_updates")
    for name in (
        "torch_compile_wrapper_seconds",
        "first_optimizer_update_wall_seconds",
        "update_timings",
    ):
        assert f'"{name}"' in src, name
    for retired in (
        '"compile_wrapper_seconds"',
        "compile_materialization_wall_seconds",
        '"measured_update_wall_seconds"',
        '"median_tokens_per_sec"',
    ):
        assert retired not in src, retired
    assert "update_timings" in _executor_nested_source("run_updates")
    assert not hasattr(R, "_train_phase_mb")


def test_no_host_device_scalar_transfer_inside_the_timed_region():
    """R2 Part 12: the timed region measures the training step, not a synchronizing readback."""
    import ast

    fn = ast.parse(_executor_nested_source("run_updates")).body[0]
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For) and getattr(n.target, "id", None) == "update"
    )
    body = loop.body
    start = next(
        i
        for i, s in enumerate(body)
        if isinstance(s, ast.Assign) and any(getattr(t, "id", None) == "started" for t in s.targets)
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
    """R3 Part 7: compile=off must prove the UNCOMPILED callable was the one invoked."""
    import torch

    class _Plain(torch.nn.Module):
        def forward(self, x):
            return x

    cache = tmp_path / "inductor_cache"
    plain = _Plain()
    forward = R.ObservedForward(plain)
    forward(torch.zeros(1))
    off = R.compile_path_evidence(
        plain, forward, requested=False, cache_dir=cache, expected_forward_invocations=1
    )
    assert off["canonical_compile_path"] is True
    assert off["invoked_uncompiled_module"] is True
    assert off["realized_module_is_optimized_module"] is False
    assert off["forward_invocations"] == 1
    # requesting compile while eager silently ran: no compiled object was ever invoked
    on = R.compile_path_evidence(
        plain, forward, requested=True, cache_dir=cache, expected_forward_invocations=1
    )
    assert on["invoked_compiled_callable"] is False
    assert on["canonical_compile_path"] is False
    assert on["inductor_artifact_count"] == 0


def test_the_compiled_callable_must_be_the_one_actually_invoked(tmp_path):
    """R3 Part 7: compile=on binds the object torch.compile returned, not a Boolean."""

    class _OptimizedModule:  # the name torch._dynamo gives its wrapper
        def __call__(self, x):
            return x

    cache = tmp_path / "inductor_cache"
    cache.mkdir()
    (cache / "output_code.py").write_text("# compiled", encoding="utf-8")
    compiled = _OptimizedModule()
    forward = R.ObservedForward(compiled, compiled_object=compiled)
    for _ in range(3):
        forward(0)
    evidence = R.compile_path_evidence(
        compiled, forward, requested=True, cache_dir=cache, expected_forward_invocations=3
    )
    assert evidence["realized_module_is_optimized_module"] is True
    assert evidence["invoked_compiled_callable"] is True
    assert evidence["inductor_artifact_count"] == 1
    assert evidence["canonical_compile_path"] is True
    assert R.recheck_compile_path_evidence({"compile": True, "compile_evidence": evidence}) is True
    # the same module with compile=off is a silent-compile failure in the other direction
    off = R.compile_path_evidence(
        compiled, forward, requested=False, cache_dir=cache, expected_forward_invocations=3
    )
    assert off["canonical_compile_path"] is False
    # a compiled object that was NOT the invoked callable is rejected too
    eager = R.ObservedForward(object(), compiled_object=compiled)
    stray = R.compile_path_evidence(
        compiled, eager, requested=True, cache_dir=cache, expected_forward_invocations=0
    )
    assert stray["invoked_compiled_callable"] is False
    assert stray["canonical_compile_path"] is False


def test_forward_invocation_count_must_match_the_frozen_geometry(tmp_path):
    """The observed callable must have been invoked exactly updates x grad_accum times."""

    class _Plain:
        def __call__(self, x):
            return x

    plain = _Plain()
    forward = R.ObservedForward(plain)
    forward(0)
    evidence = R.compile_path_evidence(
        plain, forward, requested=False, cache_dir=tmp_path / "c", expected_forward_invocations=640
    )
    assert evidence["forward_invocations_match_geometry"] is False
    assert evidence["canonical_compile_path"] is False
    assert (
        R.recheck_compile_path_evidence({"compile": False, "compile_evidence": evidence}) is False
    )


def test_the_contract_rejects_a_compile_candidate_that_fell_back(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[1]
    assert candidate["compile"] is True
    result = _bound_mb_result(candidate, session, plan, canonical_compile_path=False)
    eligible, failures = C.mb_candidate_eligible(result, SYNTHETIC_VRAM_BYTES)
    assert not eligible and "compile_silent_fallback" in failures
    # and the other direction: compile=off that somehow ran a compiled path
    off = _bound_mb_result(candidates[0], session, plan, canonical_compile_path=False)
    eligible_off, failures_off = C.mb_candidate_eligible(off, SYNTHETIC_VRAM_BYTES)
    assert not eligible_off and "unrequested_compiled_path" in failures_off


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
    candidates, plan = _published_mb_plan(session)
    payload = _bound_mb_result(candidates[0], session, plan)
    meta = _run_meta_doc(candidates[0], session, plan, payload)
    assert (
        meta["execution_bundle_sha256"] == session.validated["observed"]["execution_bundle_sha256"]
    )
    assert "execution_bundle_sha256" in R.RUN_META_RESULT_BINDINGS
    assert "execution_bundle_sha256" in dict(R._session_bindings(session))  # noqa: SLF001
    assert "execution_bundle_sha256" in R.REQUIRED_RESULT_BINDINGS


def test_run_meta_binds_the_session_and_the_index_manifest_file_hash(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    payload = _bound_mb_result(candidates[0], session, plan)
    execution = {
        "validated": session.validated,
        "candidate": candidates[0],
        "output_dir": Path(candidates[0]["output_dir"]),
        "spec_sha256": payload["candidate_spec_sha256"],
        "plan": plan["plan"],
        "plan_sha256": plan["plan_sha256"],
        "session_sha256": session.session_sha256,
        "ledger": session.ledger,
    }
    meta = R.run_meta(execution)
    assert meta["session_id"] == session.session_id
    assert (
        meta["pilot_index_manifest_file_sha256"]
        == session.validated["pilot_index_manifest_file_sha256"]
    )
    assert meta["authorization_sha256"] == session.validated["authorization_sha256"]
    assert meta["schema_version"] == C.RUN_META_SCHEMA
    # A session object itself grants nothing; only the validator-derived execution mapping fits.
    import inspect

    assert set(inspect.signature(R.run_meta).parameters) == {"execution"}
    with pytest.raises((TypeError, KeyError)):
        R.run_meta(session)


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
    """R3 Part 5 depends on this shape: the numerator and weight, plus the single division."""
    model = _TinyLM()
    view = R.IndexView(_FakeDataset(20), [0, 1, 2, 3])
    out = R.evaluate(model, R.ObservedForward(model), view, micro_bsz=2, device="cpu")
    assert set(out) == {"numerator", "weight", "loss", "blocks"}
    assert out["blocks"] == 4
    assert out["weight"] > 0.0
    assert out["loss"] == out["numerator"] / out["weight"]
    # the same shape the LR candidate serializes and the parent recomputes
    assert R.verify_recomputed_lr_result({
        "candidate_id": "probe",
        "compile": False,
        "compile_evidence": _compile_evidence(False),
        "canonical_compile_path": True,
        "eval_stage_a_numerator": out["numerator"],
        "eval_stage_a_weight": out["weight"],
        "eval_stage_b_numerator": out["numerator"],
        "eval_stage_b_weight": out["weight"],
        "eval_loss_stage_a": out["loss"],
        "eval_loss_stage_b": out["loss"],
        "score": C.lr_score(out["loss"], out["loss"]),
    })["score"] == C.lr_score(out["loss"], out["loss"])


def test_a_preexisting_terminal_is_never_deleted_or_overwritten(tmp_path, monkeypatch):
    """R4 rejects a stale-attempt protocol mismatch while preserving immutable evidence."""
    session = _fake_session(tmp_path)
    import inspect

    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _publish_worker_chain(candidate, session, plan)
    stale = R.terminal_result_path(Path(candidate["output_dir"]))
    before = stale.read_bytes()

    def handler(argv, kwargs):
        return _FakeCompleted(R.BINDING_FAILURE, "", "output already exists")

    _fake_subprocess(monkeypatch, handler)
    with pytest.raises(R.PhaseAbort, match="terminal protocol is broken"):
        R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001
    assert stale.read_bytes() == before
    assert "unlink" not in inspect.getsource(R._subprocess_launcher)  # noqa: SLF001


# ==================================================================================== R3
# ============================ R3 Part 1/2/3: the immutable session and phase-plan chain


def _worker_validated(session, monkeypatch):
    """Isolate the plan/spec/ledger layer by stubbing the artifact-bytes layer beneath it.

    The artifact-bytes layer has its own coverage above; opening the real 26 GB accepted
    releases here would prove nothing extra about phase-plan membership.
    """
    validated = dict(session.validated)

    def fake(**kwargs):
        assert set(kwargs) == {
            "authorization_path",
            "pilot_index_manifest_path",
            "output_dir",
            "requested_phase",
            "gpu_required",
        }
        assert Path(kwargs["output_dir"]).resolve() == session.output_root
        return validated

    monkeypatch.setattr(R, "validate_execution_artifacts", fake)
    return validated


def _worker_kwargs(session, plan, candidate, **over):
    kwargs = {
        "authorization_path": Path(session.validated["authorization_path"]),
        "session_manifest_path": session.session_path,
        "phase_plan_path": plan["plan_path"],
        "candidate_spec_path": plan["specs"][candidate["candidate_id"]]["path"],
        "pilot_index_manifest_path": Path(session.validated["index_manifest_path"]),
        "accepted_stage_a_path": REPO_ROOT / R.ACCEPTED_STAGE_A,
        "accepted_stage_b_path": REPO_ROOT / R.ACCEPTED_STAGE_B,
        "ledger_path": session.output_root / R.LEDGER_FILENAME,
        "candidate_output_path": Path(candidate["output_dir"]),
    }
    kwargs.update(over)
    return kwargs


def _session_for_worker(tmp_path, monkeypatch, scope="FULL_V2_3_PILOT"):
    session = _fake_session(tmp_path, scope=scope)
    monkeypatch.setattr(
        R,
        "validate_session_manifest",
        lambda path, validated: (
            json.loads(Path(path).read_bytes().decode("utf-8")),
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ),
    )
    _worker_validated(session, monkeypatch)
    return session


def test_a_valid_plan_and_matching_spec_return_only_non_authorizing_data(tmp_path, monkeypatch):
    """R4 validation returns derived facts, never a capability accepted by a backend."""
    import inspect

    session = _session_for_worker(tmp_path, monkeypatch)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    execution = R.validate_worker_execution(
        **_worker_kwargs(session, plan, candidate), gpu_required=False
    )
    assert isinstance(execution, dict)
    assert execution["candidate"] == candidate
    assert execution["plan_sha256"] == plan["plan_sha256"]
    assert (
        execution["spec_sha256"]
        == _plan_entry(plan, candidate["candidate_id"])["candidate_spec_sha256"]
    )
    assert execution["output_dir"] == Path(candidate["output_dir"]).resolve()
    assert execution["phase"] == "MB"
    assert not hasattr(R, "WorkerAuthority")
    assert not hasattr(R, "execute_validated_candidate")
    assert "execution" not in inspect.signature(R.execute_candidate_from_artifact_paths).parameters


def test_an_unlisted_candidate_spec_is_refused_before_model_construction(tmp_path, monkeypatch):
    session = _session_for_worker(tmp_path, monkeypatch)
    candidates, plan = _published_mb_plan(session)
    rogue = tmp_path / "rogue.json"
    doc = json.loads(
        plan["specs"][candidates[0]["candidate_id"]]["path"].read_bytes().decode("utf-8")
    )
    doc["candidate"] = {**doc["candidate"], "updates": 1}  # a cheaper, self-declared candidate
    rogue.write_bytes(C.canonical_json_bytes(doc))
    with pytest.raises(R.BindingFailure, match="launched spec SHA"):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, candidates[0], candidate_spec_path=rogue),
            gpu_required=False,
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("micro_bsz", 3),
        ("grad_accum", 1),
        ("compile", True),
        ("updates", 1),
        ("peak_lr", 9e-4),
        ("model_init_seed", 1),
        ("seed_label", "seed-2"),
    ],
)
def test_a_spec_whose_fields_leave_the_contract_derivation_is_refused(
    tmp_path, monkeypatch, field, value
):
    """R3 Part 2: every candidate field is re-derived, not accepted as published."""
    session = _session_for_worker(tmp_path, monkeypatch)
    candidates = R.plan_phase_mb(output_root=session.output_root)
    tampered = [{**candidates[0], field: value}, *candidates[1:]]
    plan = R.publish_phase_plan(
        root=session.output_root,
        plan_kind="PHASE_MB_PLAN",
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        candidates=tampered,
        derived_from={"source": "tampered"},
    )
    with pytest.raises(R.BindingFailure, match="differ from the contract derivation"):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, tampered[0]), gpu_required=False
        )


@pytest.mark.parametrize(
    "over, message",
    [
        ({"ledger_path": "elsewhere"}, "token ledger must be"),
        ({"session_manifest_path": "elsewhere"}, "session manifest must be"),
        ({"accepted_stage_a_path": "elsewhere"}, "is not the frozen release"),
        ({"accepted_stage_b_path": "elsewhere"}, "is not the frozen release"),
        ({"phase_plan_path": "elsewhere"}, "must live directly under"),
    ],
)
def test_the_worker_refuses_a_path_that_is_not_the_canonical_artifact(
    tmp_path, monkeypatch, over, message
):
    session = _session_for_worker(tmp_path, monkeypatch)
    candidates, plan = _published_mb_plan(session)
    resolved = {k: tmp_path / "elsewhere" / k for k in over}
    with pytest.raises(R.BindingFailure, match=message):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, candidates[0], **resolved), gpu_required=False
        )


def test_the_worker_refuses_an_output_directory_the_plan_did_not_assign(tmp_path, monkeypatch):
    session = _session_for_worker(tmp_path, monkeypatch)
    candidates, plan = _published_mb_plan(session)
    with pytest.raises(R.BindingFailure, match="is not the directory the phase plan assigns"):
        R.validate_worker_execution(
            **_worker_kwargs(
                session,
                plan,
                candidates[0],
                candidate_output_path=session.output_root / "somewhere_else",
            ),
            gpu_required=False,
        )


def test_the_worker_refuses_a_spec_bound_to_another_session(tmp_path, monkeypatch):
    session = _session_for_worker(tmp_path, monkeypatch)
    candidates, plan = _published_mb_plan(session)
    spec_path = plan["specs"][candidates[0]["candidate_id"]]["path"]
    doc = json.loads(spec_path.read_bytes().decode("utf-8"))
    doc["session_sha256"] = "0" * 64
    spec_path.write_bytes(C.canonical_json_bytes(doc))
    with pytest.raises(R.BindingFailure, match="does not bind the session manifest"):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, candidates[0]), gpu_required=False
        )


def test_the_session_manifest_binds_the_whole_execution(tmp_path):
    """R3 Part 3: SESSION.json is derived from the validated artifacts, byte for byte."""
    session = _fake_session(tmp_path)
    doc = R.session_manifest_document(session.validated)
    for field in (
        "session_id",
        "authorized_scope",
        "authorization_sha256",
        "contract_sha256",
        "implementation_head",
        "execution_bundle_sha256",
        "pilot_index_manifest_file_sha256",
        "serialized_index_lists_digest",
        "accepted_releases",
        "runtime_fingerprint_sha256",
        "base_runtime_fingerprint",
        "authorized_output_root",
        "ledger_identity",
        "effective_ceilings",
        "session_hard_ceiling",
    ):
        assert field in doc, field
    assert doc["session_hard_ceiling"] == C.FULL_V2_3_PILOT_SESSION_HARD_CEILING
    assert set(doc["accepted_releases"]) == {"stage_a", "stage_b"}
    assert doc["base_runtime_fingerprint"] == session.validated["fingerprint"]
    path = tmp_path / "SESSION_probe.json"
    R.write_immutable_artifact(path, doc)
    assert R.validate_session_manifest(path, session.validated)[0] == doc
    tampered = dict(session.validated)
    tampered["session_id"] = "OTHER"
    with pytest.raises(R.BindingFailure, match="does not bind this execution"):
        R.validate_session_manifest(path, tampered)


def test_the_phase_mb_plan_lists_exactly_the_ten_required_specs(tmp_path):
    session = _fake_session(tmp_path)
    _, plan = _published_mb_plan(session)
    doc = plan["plan"]
    assert doc["plan_kind"] == "PHASE_MB_PLAN"
    assert doc["session_sha256"] == session.session_sha256
    assert doc["candidate_ids"] == list(C.MB_REQUIRED_CANDIDATE_IDS)
    assert len(doc["candidate_spec_sha256s"]) == 10
    assert len(set(doc["candidate_spec_sha256s"])) == 10
    for entry in doc["candidates"]:
        spec = session.output_root / entry["candidate_spec_relpath"]
        assert hashlib.sha256(spec.read_bytes()).hexdigest() == entry["candidate_spec_sha256"]
    # published once
    with pytest.raises(R.BindingFailure, match="already exists"):
        R.publish_phase_plan(
            root=session.output_root,
            plan_kind="PHASE_MB_PLAN",
            session_sha256=session.session_sha256,
            session_id=session.session_id,
            candidates=R.plan_phase_mb(output_root=session.output_root),
            derived_from={},
        )


def _lr_chain(tmp_path, scores=None):
    """Run a full FULL-scope Phase-LR orchestration and return (session, report)."""
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=8, compile_on=False)
    scores = scores or {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}

    def launcher(candidate, given, plan):
        _write_result(
            candidate,
            given,
            plan,
            _bound_lr_result(candidate, given, plan, scores[candidate["peak_lr"]]),
        )

    return session, R.orchestrate_phase_muon_lr(session, launcher=launcher)


def test_the_lr_plan_chain_binds_only_validated_preceding_evidence(tmp_path):
    """R3 Part 3: initial <- MB report; confirmation <- initial report; edge <- confirmation."""
    session, report = _lr_chain(tmp_path)
    root = session.output_root
    mb_report_sha = hashlib.sha256((root / R.MB_REPORT_FILENAME).read_bytes()).hexdigest()

    initial = json.loads((root / R.PHASE_LR_INITIAL_PLAN_FILENAME).read_text(encoding="utf-8"))
    assert initial["plan_kind"] == "PHASE_LR_INITIAL_PLAN"
    assert initial["session_sha256"] == session.session_sha256
    assert initial["derived_from"]["phase_mb_report_sha256"] == mb_report_sha
    assert initial["derived_from"]["frozen_geometry"] == {
        "micro_bsz": 8,
        "grad_accum": 16,
        "compile": False,
    }
    assert initial["derived_from"]["peak_lrs"] == list(C.LR_GRID_SEED1)
    assert initial["derived_from"]["seed_labels"] == ["seed-1"]
    assert len(initial["candidates"]) == 3

    initial_report_sha = hashlib.sha256(
        (root / R.LR_INITIAL_REPORT_FILENAME).read_bytes()
    ).hexdigest()
    confirm = json.loads((root / R.PHASE_LR_CONFIRMATION_PLAN_FILENAME).read_text(encoding="utf-8"))
    assert confirm["derived_from"]["preceding_report_sha256"] == initial_report_sha
    assert confirm["derived_from"]["seed_labels"] == ["seed-2"]
    assert sorted(confirm["derived_from"]["peak_lrs"]) == [2e-4, 3e-4]

    confirmation_report_sha = hashlib.sha256(
        (root / R.LR_CONFIRMATION_REPORT_FILENAME).read_bytes()
    ).hexdigest()
    edge = json.loads((root / R.PHASE_LR_EDGE_PLAN_FILENAME).read_text(encoding="utf-8"))
    assert edge["derived_from"]["preceding_report_sha256"] == confirmation_report_sha
    assert edge["derived_from"]["peak_lrs"] == [1e-4]
    assert sorted(edge["derived_from"]["seed_labels"]) == ["seed-1", "seed-2"]
    assert report["terminal_status"] == "SUCCESS"
    for name in (
        R.LR_INITIAL_REPORT_FILENAME,
        R.LR_CONFIRMATION_REPORT_FILENAME,
        R.LR_EDGE_REPORT_FILENAME,
        R.LR_REPORT_FILENAME,
    ):
        assert (root / name).is_file() and (root / name).with_suffix(".sha256").is_file()


def test_a_plan_referencing_a_tampered_prior_report_is_refused(tmp_path, monkeypatch):
    session, _ = _lr_chain(tmp_path)
    root = session.output_root
    plan_path = root / R.PHASE_LR_CONFIRMATION_PLAN_FILENAME
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate_id = plan["candidates"][0]["candidate_id"]
    candidate = next(
        c
        for c in R.plan_phase_lr(
            output_root=root,
            micro_bsz=8,
            compile_on=False,
            peak_lrs=plan["derived_from"]["peak_lrs"],
            seed_label="seed-2",
        )
        if c["candidate_id"] == candidate_id
    )
    # rewrite the preceding report's bytes; the plan's recorded SHA no longer matches
    report_path = root / R.LR_INITIAL_REPORT_FILENAME
    report_path.write_bytes(
        report_path.read_bytes().replace(b'"step":"INITIAL"', b'"step":"FORGED"')
    )
    _worker_validated(session, monkeypatch)
    monkeypatch.setattr(
        R,
        "validate_session_manifest",
        lambda path, validated: (
            json.loads(Path(path).read_bytes().decode("utf-8")),
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ),
    )
    bundle = {
        "plan": plan,
        "plan_path": plan_path,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "specs": {
            candidate_id: {
                "path": root / plan["candidates"][0]["candidate_spec_relpath"],
                "candidate_spec_sha256": plan["candidates"][0]["candidate_spec_sha256"],
            }
        },
    }
    with pytest.raises(R.BindingFailure, match="preceding_lr_report SHA-256"):
        R.validate_worker_execution(
            **_worker_kwargs(session, bundle, candidate), gpu_required=False
        )


# ============================================ R3 Part 4: run_meta and planned-candidate binding


def _admit(session, plan, candidate):
    return R.load_completed_result(session, planned=candidate, plan=plan)


def test_an_admitted_result_requires_a_real_run_meta_file(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    payload = _write_result(candidate, session, plan, _bound_mb_result(candidate, session, plan))
    assert _admit(session, plan, candidate)["run_meta_verified_from_disk"] is True
    (Path(candidate["output_dir"]) / "run_meta.json").unlink()
    with pytest.raises(R.PhaseAbort, match="terminal run_meta_sha256"):
        _admit(session, plan, candidate)
    assert payload["run_meta_sha256"]


def test_a_fabricated_run_meta_digest_without_a_file_is_rejected(tmp_path):
    """The literal ``run_meta_sha256='rm'`` shortcut can never be admitted again."""
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    out = Path(candidate["output_dir"])
    out.mkdir(parents=True)
    payload = _bound_mb_result(candidate, session, plan, run_meta_sha256="rm")
    (out / "result.json").write_bytes(C.canonical_json_bytes(payload))
    with pytest.raises(R.BindingFailure, match="no published SHA-256 sidecar"):
        _admit(session, plan, candidate)


def test_the_run_meta_sha_is_recomputed_from_the_bytes_on_disk(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _write_result(candidate, session, plan, _bound_mb_result(candidate, session, plan))
    meta_path = Path(candidate["output_dir"]) / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["injected"] = "after the digest was recorded"
    meta_path.write_bytes(C.canonical_json_bytes(meta))
    with pytest.raises(R.PhaseAbort, match="terminal run_meta_sha256"):
        _admit(session, plan, candidate)


@pytest.mark.parametrize(
    "meta_over, message",
    [
        ({"candidate_id": "mb_micro1_compileon"}, "disagrees with result"),
        ({"phase": "LR"}, "disagrees with result"),
        ({"seed_label": "seed-2"}, "disagrees with result"),
        ({"micro_bsz": 4}, "disagrees with result"),
        ({"grad_accum": 4}, "disagrees with result"),
        ({"compile": True}, "disagrees with result"),
        ({"model_seed": 1}, "does not match planned"),
        ({"train_order_seed": 1}, "does not match planned"),
        ({"phase_plan_sha256": "0" * 64}, "disagrees with result"),
        ({"candidate_spec_sha256": "0" * 64}, "disagrees with result"),
        ({"session_sha256": "0" * 64}, "disagrees with result"),
        ({"session_id": "OTHER"}, "disagrees with result"),
        ({"output_dir": "/tmp/elsewhere"}, "disagrees with result"),
        ({"runtime_fingerprint_sha256": "x"}, "disagrees with result"),
        ({"lr_configuration": {"peak_lr": 1.0}}, "LR configuration"),
        ({"ledger_identity": {}}, "ledger identity does not bind"),
        ({"schema_version": "other"}, "schema .* != expected"),
        ({"contract_version": "other"}, "contract version mismatch"),
    ],
)
def test_run_meta_that_does_not_describe_the_planned_candidate_is_rejected(
    tmp_path, meta_over, message
):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _write_result(
        candidate, session, plan, _bound_mb_result(candidate, session, plan), meta_over=meta_over
    )
    with pytest.raises(R.BindingFailure, match=message):
        _admit(session, plan, candidate)


@pytest.mark.parametrize(
    "over, message",
    [
        ({"candidate_id": "mb_micro1_compileon"}, "does not match planned"),
        ({"phase_plan_sha256": "0" * 64}, "does not match planned"),
        ({"candidate_spec_sha256": "0" * 64}, "does not match planned"),
        ({"session_sha256": "0" * 64}, "does not match planned"),
        ({"micro_bsz": 2}, "does not match planned"),
        ({"grad_accum": 2}, "does not match planned"),
        ({"compile": True}, "does not match planned"),
        ({"seed_label": "seed-2"}, "does not match planned"),
        ({"output_dir": "/tmp/elsewhere"}, "does not match planned"),
    ],
)
def test_a_result_that_does_not_match_its_planned_identity_is_rejected(tmp_path, over, message):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _write_result(candidate, session, plan, _bound_mb_result(candidate, session, plan, **over))
    with pytest.raises(R.BindingFailure, match=message):
        _admit(session, plan, candidate)


# ============================================================= R3 Part 8: ledger invariants


def _consistent_state(ledger, mb_updates=2, lr_updates=1, completed_updates=2):
    t = C.TRAINED_TOKENS_PER_UPDATE
    reserved_updates = mb_updates + lr_updates
    state = dict(ledger.state)
    state["reserved_tokens"] = {
        "MB": mb_updates * t,
        "LR": lr_updates * t,
        "GLOBAL": reserved_updates * t,
    }
    state["completed_tokens"] = {
        "MB": completed_updates * t,
        "LR": 0,
        "GLOBAL": completed_updates * t,
    }
    state["reserved_updates"] = reserved_updates
    state["completed_updates"] = completed_updates
    return state


def test_the_parent_snapshot_reloads_state_a_child_wrote(tmp_path):
    """R3 Part 8: snapshot() never answers from stale in-memory counters."""
    parent = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    child = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    stale = dict(parent.state["reserved_tokens"])
    child.reserve("MB")
    child.complete("MB")
    assert parent.state["reserved_tokens"] == stale, "the parent's in-memory copy is stale"
    fresh = parent.snapshot()
    assert fresh["reloaded_from_disk"] is True
    assert fresh["reserved_tokens"]["MB"] == C.TRAINED_TOKENS_PER_UPDATE
    assert fresh["completed_tokens"]["GLOBAL"] == C.TRAINED_TOKENS_PER_UPDATE
    assert fresh["reserved_updates"] == fresh["completed_updates"] == 1
    assert fresh["session_hard_ceiling"] == C.FULL_V2_3_PILOT_SESSION_HARD_CEILING
    assert fresh["trained_tokens_per_update"] == C.TRAINED_TOKENS_PER_UPDATE


def test_a_crash_between_reserve_and_complete_stays_legal(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.reserve("MB")  # the process dies here
    reopened = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    snap = reopened.snapshot()
    assert snap["reserved_updates"] == 1 and snap["completed_updates"] == 0
    assert snap["reserved_tokens"]["MB"] == C.TRAINED_TOKENS_PER_UPDATE
    assert snap["completed_tokens"]["MB"] == 0


def test_the_ledger_snapshot_is_reentrant_under_the_lock(tmp_path):
    """The lock guard must not deadlock a nested read on the same process."""
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.reserve("MB")
    with ledger._lock():  # noqa: SLF001
        assert ledger.snapshot()["reserved_updates"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda st: st["completed_tokens"].__setitem__("MB", 99 * C.TRAINED_TOKENS_PER_UPDATE),
            "exceeds reserved_tokens",
        ),
        (
            lambda st: st["reserved_tokens"].__setitem__(
                "GLOBAL", 99 * C.TRAINED_TOKENS_PER_UPDATE
            ),
            "not the sum of the per-phase reservations",
        ),
        (
            lambda st: st["completed_tokens"].__setitem__("GLOBAL", 0),
            "not the sum of the per-phase completions",
        ),
        (lambda st: st.__setitem__("reserved_updates", 99), "disagrees with reserved_tokens"),
        (lambda st: st.__setitem__("completed_updates", 99), "exceeds reserved_updates"),
        (
            lambda st: st["reserved_tokens"].__setitem__("MB", -C.TRAINED_TOKENS_PER_UPDATE),
            "is negative",
        ),
        (
            lambda st: st["reserved_tokens"].__setitem__("MB", 5),
            "not a whole number of optimizer updates",
        ),
        (
            lambda st: st.__setitem__("trained_tokens_per_update", 1),
            "does not equal the frozen batch geometry",
        ),
        (
            lambda st: st.__setitem__("session_hard_ceiling", 1),
            "does not equal the frozen FULL-session ceiling",
        ),
        (
            lambda st: st["effective_ceilings"].__setitem__("MB", 7),
            "do not equal the ceilings frozen",
        ),
        (
            lambda st: st["reserved_tokens"].__setitem__(
                "MB",
                (C.PHASE_MB_TOKEN_CEILING // C.TRAINED_TOKENS_PER_UPDATE + 1)
                * C.TRAINED_TOKENS_PER_UPDATE,
            ),
            "exceeds its ceiling",
        ),
    ],
)
def test_every_structural_invariant_is_enforced_on_reload(tmp_path, mutate, message):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    state = _consistent_state(ledger)
    mutate(state)
    ledger._write(state)  # noqa: SLF001 - a corrupted ledger on disk is exactly the case
    with pytest.raises(R.LedgerIntegrityFailure, match=message):
        ledger.snapshot()
    with pytest.raises(R.LedgerIntegrityFailure, match=message):
        R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())


def test_a_consistent_ledger_passes_every_invariant(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger._write(_consistent_state(ledger))  # noqa: SLF001
    snap = ledger.snapshot()
    t = C.TRAINED_TOKENS_PER_UPDATE
    assert snap["reserved_tokens"]["GLOBAL"] == 3 * t
    assert snap["completed_tokens"]["GLOBAL"] == 2 * t
    assert snap["reserved_updates"] == 3 and snap["completed_updates"] == 2


def test_ledger_integrity_failure_is_a_phase_level_failure():
    assert issubclass(R.LedgerIntegrityFailure, R.PhaseAbort)
    assert not issubclass(R.LedgerIntegrityFailure, R.CandidateFailure)


def test_the_ledger_refuses_a_non_geometry_token_amount(tmp_path):
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    with pytest.raises(C.PilotContractError, match="refusing to reserve"):
        ledger.reserve("MB", tokens=1)
    ledger.reserve("MB")
    with pytest.raises(C.PilotContractError, match="refusing to complete"):
        ledger.complete("MB", tokens=1)


# =============================================== R3 Part 9/10: terminal accounting and process


def test_partial_progress_survives_a_candidate_local_exception(tmp_path):
    """R4 seals the conservative 3-reserved/2-completed state in a durable receipt."""
    ledger = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    ledger.begin_candidate({
        "phase": "MB",
        "candidate_id": "mb_partial_probe",
        "peak_lr": 3e-4,
        "planned_updates": 4,
        "candidate_spec_sha256": "1" * 64,
        "phase_plan_sha256": "2" * 64,
        "session_id": "s",
        "session_sha256": "3" * 64,
        "authorization_sha256": "4" * 64,
    })
    for _ in range(2):
        ledger.reserve("MB")
        ledger.complete("MB")
    ledger.reserve("MB")  # update 3 failed after its conservative reservation
    receipt = ledger.finalize_candidate(
        terminal_status="CANDIDATE_INELIGIBLE",
        run_meta_sha256="5" * 64,
        result_sha256="6" * 64,
    )
    assert receipt["candidate_reserved_updates"] == 3
    assert receipt["candidate_completed_updates"] == 2
    reopened = R.TokenLedger(tmp_path / "l.json", _ledger_identity(), _ceilings())
    assert reopened.receipt(receipt["receipt_sha256"]) == receipt
    snap = reopened.snapshot()
    assert snap["reserved_updates"] == 3 and snap["completed_updates"] == 2


def test_the_parent_preserves_receipt_backed_terminal_accounting(tmp_path, monkeypatch):
    """R4 returns the child's strict terminal only after matching its durable receipt."""
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]

    def handler(argv, kwargs):
        _publish_worker_chain(candidate, session, plan, status="CANDIDATE_INELIGIBLE")
        return _FakeCompleted(R.CANDIDATE_INELIGIBLE, "", "")

    _fake_subprocess(monkeypatch, handler)
    terminal = R._subprocess_launcher(candidate, session, plan)  # noqa: SLF001
    receipt = session.ledger.receipt(terminal["ledger_receipt_sha256"])
    assert terminal["terminal_status"] == "CANDIDATE_INELIGIBLE"
    assert terminal["reserved_updates"] == receipt["candidate_reserved_updates"]
    assert terminal["completed_updates"] == receipt["candidate_completed_updates"]
    assert terminal["reserved_tokens"] == receipt["after_reserved_tokens"]
    assert terminal["completed_tokens"] == receipt["after_completed_tokens"]
    assert terminal["ledger_reserved_updates"] == receipt["after_reserved_updates"]
    assert terminal["ledger_completed_updates"] == receipt["after_completed_updates"]


@pytest.mark.parametrize(
    "message, reason",
    [
        ("CUDA out of memory", "oom"),
        ("TorchDynamo failed to compile the graph", "compile_failure"),
        ("loss became non-finite at update 12", "nonfinite"),
        ("some other candidate-local runtime problem", "candidate_runtime_exception"),
    ],
)
def test_candidate_local_failure_classification(tmp_path, message, reason):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)

    def launcher(c, s, pl):
        raise R.CandidateFailure(message)

    evidence = R._launch(candidates[0], session, plan, launcher)  # noqa: SLF001
    assert evidence["reason"] == reason
    assert evidence["eligible"] is False


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_process_control_events_are_never_downgraded_to_ineligible(tmp_path, exc):
    """R3 Part 10: no BaseException handler turns process control into candidate evidence."""
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)

    def launcher(c, s, pl):
        raise exc("interrupted")

    with pytest.raises(exc):
        R._launch(candidates[0], session, plan, launcher)  # noqa: SLF001


def test_no_broad_base_exception_handler_remains_on_the_candidate_paths():
    import ast
    import inspect

    for fn in (R._launch, R._cli_internal_worker, R._cli_run):  # noqa: SLF001
        tree = ast.parse(inspect.getsource(fn).lstrip())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                names = [n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)]
                assert "BaseException" not in names, (fn.__name__, names)


@pytest.mark.parametrize(
    "body",
    [b"not json at all", C.canonical_json_bytes({"schema_version": "wrong"}), b"[]"],
)
def test_a_malformed_terminal_artifact_aborts_the_phase(tmp_path, body):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _publish_worker_chain(candidate, session, plan)
    R.terminal_result_path(Path(candidate["output_dir"])).write_bytes(body)
    with pytest.raises(R.BindingFailure, match="published sidecar"):
        _read_candidate_terminal(candidate, session, plan)


def test_a_terminal_artifact_with_an_unusable_progress_count_aborts(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[0]
    _publish_worker_chain(candidate, session, plan)
    path = R.terminal_result_path(Path(candidate["output_dir"]))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["completed_updates"] = -1
    body = C.canonical_json_bytes(doc)
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(body).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(R.PhaseAbort, match="completed_updates must be a nonnegative integer"):
        _read_candidate_terminal(candidate, session, plan)


# ======================================================== R3 Part 11: the exact Muon verifier


def _tiny_muon():
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
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR_ARG,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    return model, opt


def _group_of(opt, role):
    return next(g for g in opt.param_groups if C.optimizer_group_role(g) == role)


def test_the_exact_verifier_accepts_the_canonical_realization():
    model, opt = _tiny_muon()
    v = C.verify_realized_grouping(opt, model)
    assert v["matches_frozen_realization"], v["failures"]
    assert v["membership_verified"] is True
    assert sorted(v["group_roles"]) == sorted(C.OPTIMIZER_GROUP_ROLES)
    assert v["membership"] == v["expected_membership"]


@pytest.mark.parametrize(
    "role, key, value, fragment",
    [
        ("aux_adamw_decay", "weight_decay", 0.05, "aux_adamw_decay weight_decay must be exactly"),
        ("aux_adamw_decay", "weight_decay", 0.0, "aux_adamw_decay weight_decay must be exactly"),
        ("aux_adamw_no_decay", "weight_decay", 1e-9, "aux_adamw_no_decay weight_decay"),
        ("aux_adamw_no_decay", "weight_decay", -0.1, "aux_adamw_no_decay weight_decay"),
        ("muon_matrices", "weight_decay", 0.05, "muon_matrices weight_decay must be exactly"),
        ("aux_adamw_decay", "betas", (0.9, 0.999), "betas must be"),
        ("aux_adamw_no_decay", "betas", (0.95, 0.95), "betas must be"),
        ("aux_adamw_decay", "eps", 1e-6, "eps must be"),
        ("aux_adamw_no_decay", "eps", 0.0, "eps must be"),
        ("muon_matrices", "momentum", 0.9, "momentum must be"),
        ("muon_matrices", "nesterov", False, "Nesterov"),
        ("muon_matrices", "ns_steps", 4, "Newton-Schulz steps must be"),
        ("muon_matrices", "lr_ratio", 1.5, "lr_ratio must be"),
    ],
)
def test_the_exact_verifier_rejects_every_single_field_mutation(role, key, value, fragment):
    model, opt = _tiny_muon()
    _group_of(opt, role)[key] = value
    v = C.verify_realized_grouping(opt, model)
    assert not v["matches_frozen_realization"], (role, key, value)
    assert any(fragment in f for f in v["failures"]), v["failures"]


def test_the_exact_verifier_rejects_membership_damage():
    model, opt = _tiny_muon()
    muon = _group_of(opt, "muon_matrices")
    decay = _group_of(opt, "aux_adamw_decay")
    # a parameter in two groups at once
    muon["params"] = [*muon["params"], decay["params"][0]]
    assert any(
        "more than one optimizer group" in f
        for f in C.verify_realized_grouping(opt, model)["failures"]
    )
    model, opt = _tiny_muon()
    muon = _group_of(opt, "muon_matrices")
    dropped = muon["params"].pop()
    v = C.verify_realized_grouping(opt, model)
    assert any("in no optimizer group" in f for f in v["failures"])
    assert dropped is not None
    model, opt = _tiny_muon()
    import torch

    _group_of(opt, "muon_matrices")["params"].append(torch.nn.Parameter(torch.zeros(4, 4)))
    assert any(
        "not trainable model parameters" in f
        for f in C.verify_realized_grouping(opt, model)["failures"]
    )


def test_the_role_of_a_group_is_not_derived_from_its_weight_decay():
    """A mutated decay must not let a group masquerade as the role whose decay it carries."""
    model, opt = _tiny_muon()
    no_decay = _group_of(opt, "aux_adamw_no_decay")
    no_decay["weight_decay"] = C.WEIGHT_DECAY
    assert C.optimizer_group_role(no_decay) == "aux_adamw_no_decay"
    v = C.verify_realized_grouping(opt, model)
    assert not v["matches_frozen_realization"]
    assert any("aux_adamw_no_decay weight_decay" in f for f in v["failures"])


def _executable_names(fn):
    """Every name the function's CODE references, with docstrings and comments excluded."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fn).lstrip())
    return (
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        | {
            alias.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for alias in n.names
        }
    )


def test_the_rms_oracle_never_calls_the_realization_it_verifies():
    """R3 Part 11: neither half of the expectation comes from src.optim."""
    import ast
    import inspect

    for fn in (R.expected_rms_matched_lr, R.expected_newton_schulz_scalar_gain):
        names = _executable_names(fn)
        assert "zeropower_via_newtonschulz5" not in names
        assert not any(n.startswith("src") for n in names), names
    verifier = ast.parse(inspect.getsource(R.verify_rms_matching).lstrip())
    imported = {
        alias.name
        for node in ast.walk(verifier)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert imported == {"Muon"}, imported  # the optimizer under test, never its NS helper
    assert "zeropower_via_newtonschulz5" not in _executable_names(R.verify_rms_matching)


def test_the_independent_closed_form_matches_the_realized_muon_update():
    evidence = R.verify_rms_matching()
    assert evidence["all_cases_match"] and evidence["all_cases_discriminating"]
    for case in evidence["cases"]:
        assert case["relative_error"] <= R.RMS_MATCHING_RELATIVE_TOLERANCE
        # the margin an unscaled implementation would leave is orders of magnitude larger
        assert case["unscaled_lr_would_differ_by"] > 1000.0 * max(case["max_abs_error"], 1e-15)
        assert case["expected_adjusted_lr"] == R.expected_rms_matched_lr(
            evidence["lr"], case["fan_out"], case["fan_in"]
        )


def test_the_closed_form_newton_schulz_gain_is_a_pure_scalar_recursion():
    a, b, c = C.NEWTON_SCHULZ_COEFFICIENTS
    beta = 1.0 / ((24.0) + 1e-7)  # sigma = 1, short side 576 -> sqrt = 24
    for _ in range(5):
        beta = a * beta + b * beta**3 + c * beta**5
    assert R.expected_newton_schulz_scalar_gain(576) == pytest.approx(beta, rel=1e-12)


def test_optimizer_verification_failure_blocks_the_first_training_update(monkeypatch):
    """R4 keeps both realization gates before its non-callable update loop."""
    model, opt = _tiny_muon()
    monkeypatch.setattr(
        R,
        "verify_realized_grouping",
        lambda o, m=None: {
            "failures": ["aux_adamw_decay weight_decay must be exactly 0.1, got 0.05"],
            "matches_frozen_realization": False,
            "all_lr_ratios_are_one": True,
        },
    )
    with pytest.raises(R.PhaseAbort, match="optimizer realization verification FAILED"):
        R.verify_muon_realization(model, opt, peak_lr=3e-4)
    train = _executor_nested_source("train_candidate")
    assert (
        train.index("optimizer = construct_optimizer(module)")
        < train.index("verify_muon_realization(")
        < train.index("update_result = run_updates(")
    )
    assert not hasattr(R, "REAL_TRAINING_ENTRYPOINTS")


def test_lexical_optimizer_constructor_refuses_a_non_canonical_realization():
    src = _executor_nested_source("construct_optimizer")
    assert src.index("optimizer = build_optimizer(") < src.index(
        "grouping = verify_realized_grouping(optimizer, model)"
    )
    assert 'if not grouping["matches_frozen_realization"]' in src
    assert "raise PhaseAbort(" in src
    assert src.index("raise PhaseAbort(") < src.index("return optimizer")
    assert not hasattr(R, "build_pilot_optimizer")


# ================================================= R3 Part 12: the FULL session budget semantics


def test_full_session_hard_ceiling_and_semantics():
    assert C.FULL_V2_3_PILOT_SESSION_HARD_CEILING == 500_000_000
    assert C.FULL_V2_3_PILOT_SESSION_HARD_CEILING == C.GLOBAL_PILOT_TOKEN_CEILING
    full = C.SESSION_BUDGET_SEMANTICS["FULL_V2_3_PILOT"]
    assert full["session_hard_ceiling_tokens"] == 500_000_000
    assert full["phase_ceilings"] == {"MB": 105_000_000, "LR": 370_000_000}
    assert full["one_authorization_sha256"] is True
    assert full["one_session_identity"] is True
    assert full["one_token_ledger"] is True
    assert full["automatic_second_session"] is False
    assert full["automatic_cross_authorization_retry"] is False
    assert "NEW owner decision" in full["second_authorization_requires"]
    assert "consumed" in full["second_authorization_requires"]
    mb_only = C.SESSION_BUDGET_SEMANTICS["PHASE_MB_ONLY"]
    assert mb_only["may_execute_phase_lr"] is False
    assert mb_only["may_be_promoted_to_full"] is False
    assert mb_only["automatic_mb_only_to_full_transition"] is False
    assert C.SESSION_BUDGET_SEMANTICS["cross_authorization_aggregate_accounting"] == (
        "NOT_IMPLEMENTED_BY_DESIGN"
    )
    budget = C.contract_document()["token_budget"]
    assert budget["FULL_V2_3_PILOT_SESSION_HARD_CEILING"] == 500_000_000
    assert budget["TRAINED_TOKENS_PER_UPDATE"] == 262144


def test_session_budget_subcommand_reports_the_frozen_semantics(capsys):
    assert R.main(["session-budget"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["FULL_V2_3_PILOT_SESSION_HARD_CEILING"] == 500_000_000
    assert out["phase_ceilings"] == {"MB": 105_000_000, "LR": 370_000_000}
    assert out["semantics"]["FULL_V2_3_PILOT"]["automatic_second_session"] is False


def test_no_automatic_mb_only_to_full_transition_exists(tmp_path):
    """A PHASE_MB_ONLY session ends at its report; nothing promotes it."""
    session = _fake_session(tmp_path, scope="PHASE_MB_ONLY")
    report = _frozen_mb_report(session)
    assert report["session_terminates_after_this_phase"] is True
    assert report["next_phase_requires_new_authorization"] is True
    with pytest.raises(R.BindingFailure, match="never promoted"):
        R.orchestrate_phase_muon_lr(session, launcher=lambda *_: None)
    # and the artifact layer refuses the scope outright, before any orchestration
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "can never be" in src and "promoted" in src


def test_the_documented_session_semantics_are_in_the_contract_document():
    doc = Path(REPO_ROOT / "docs" / "PILOT_CONTRACT_V2_3.md").read_text(encoding="utf-8")
    assert "FULL_V2_3_PILOT_SESSION_HARD_CEILING   500,000,000" in doc
    assert "no automatic second FULL session" in doc
    assert "new owner decision that must explicitly take the prior session" in doc


# ==================================================== R3 Part 7: compile evidence at admission


def test_admission_revalidates_the_compile_observation(tmp_path):
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[1]
    assert candidate["compile"] is True
    forged = _bound_mb_result(
        candidate,
        session,
        plan,
        compile_evidence={**_compile_evidence(True), "invoked_compiled_callable": False},
    )
    with pytest.raises(R.BindingFailure, match="does not support the verdict recomputed"):
        R.verify_recomputed_mb_result(forged)
    missing = _bound_mb_result(candidate, session, plan, compile_evidence=None)
    with pytest.raises(R.BindingFailure, match="does not support the verdict recomputed"):
        R.verify_recomputed_mb_result(missing)


def test_an_honest_silent_fallback_is_ineligible_not_a_phase_abort(tmp_path):
    """A candidate that truthfully recorded an eager fallback is ineligible, not fatal."""
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    candidate = candidates[1]
    assert candidate["compile"] is True
    honest = _bound_mb_result(
        candidate,
        session,
        plan,
        compile_evidence={**_compile_evidence(True), "invoked_compiled_callable": False},
        canonical_compile_path=False,
    )
    recomputed = R.verify_recomputed_mb_result(honest)
    assert recomputed["canonical_compile_path"] is False
    eligible, failures = C.mb_candidate_eligible({**honest, **recomputed}, SYNTHETIC_VRAM_BYTES)
    assert not eligible and "compile_silent_fallback" in failures


def test_a_silent_fallback_candidate_does_not_abort_the_grid(tmp_path):
    """The Phase-MB grid completes and simply selects one of the eligible candidates."""
    session = _fake_session(tmp_path)

    def launcher(candidate, given, plan):
        payload = _bound_mb_result(candidate, given, plan)
        if candidate["candidate_id"] == "mb_micro16_compileon":
            payload = _bound_mb_result(
                candidate,
                given,
                plan,
                compile_evidence={
                    **_compile_evidence(True),
                    "invoked_compiled_callable": False,
                    "compilation_materialized": False,
                },
                canonical_compile_path=False,
            )
        _write_result(candidate, given, plan, payload)

    report = R.orchestrate_phase_mb(session, launcher=launcher)
    assert report["selection"]["outcome"] == "PHASE_MB_FROZEN"
    fell_back = next(c for c in report["candidates"] if c["candidate_id"] == "mb_micro16_compileon")
    assert fell_back["canonical_compile_path"] is False
    assert (
        "compile_silent_fallback"
        in report["selection_trace"]["eligibility"]["mb_micro16_compileon"]
    )


def test_selection_uses_only_the_measured_window(tmp_path):
    """Throughput is measured over updates 11..40 and nothing else."""
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)
    timings = _mb_timings()
    result = _bound_mb_result(
        candidates[0],
        session,
        plan,
        update_timings=timings,
        # a very slow "update 1" is recorded but is NOT part of the measured window
        per_update_wall_seconds={"1": 900.0},
        first_optimizer_update_wall_seconds=900.0,
    )
    recomputed = R.verify_recomputed_mb_result(result)
    assert recomputed["measured_update_ids"] == list(range(11, 41))
    assert recomputed["median_update_tokens_per_second"] == MEASURED_TPS


# ==================== R3 re-review closure: the LR chain is derived, never echoed back


def _mb_report_for_worker(session, micro_bsz=8, compile_on=False):
    """Publish a real Phase-MB report the worker can derive the frozen geometry from."""
    report = _frozen_mb_report(session, micro_bsz=micro_bsz, compile_on=compile_on)
    path = session.output_root / R.MB_REPORT_FILENAME
    return {
        "phase_mb_report_relpath": R.MB_REPORT_FILENAME,
        "phase_mb_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, report


def _publish_lr_plan(session, *, plan_kind, peak_lrs, seed_labels, derived_from):
    candidates = []
    geometry = derived_from["frozen_geometry"]
    for seed_label in seed_labels:
        candidates.extend(
            R.plan_phase_lr(
                output_root=session.output_root,
                micro_bsz=geometry["micro_bsz"],
                compile_on=geometry["compile"],
                peak_lrs=list(peak_lrs),
                seed_label=seed_label,
            )
        )
    return candidates, R.publish_phase_plan(
        root=session.output_root,
        plan_kind=plan_kind,
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        candidates=candidates,
        derived_from={
            **derived_from,
            "peak_lrs": [float(v) for v in peak_lrs],
            "seed_labels": list(seed_labels),
        },
    )


def test_an_lr_plan_with_no_bound_mb_report_is_refused(tmp_path, monkeypatch):
    """The worker never accepts a plan's own declared geometry as the derivation."""
    session = _session_for_worker(tmp_path, monkeypatch)
    candidates, plan = _publish_lr_plan(
        session,
        plan_kind="PHASE_LR_CONFIRMATION_PLAN",
        peak_lrs=[0.5],
        seed_labels=["seed-1"],
        derived_from={"frozen_geometry": {"micro_bsz": 1, "grad_accum": 128, "compile": True}},
    )
    with pytest.raises(R.BindingFailure, match="not recorded as a path plus SHA-256"):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, candidates[0]), gpu_required=False
        )


def test_an_lr_plan_whose_geometry_contradicts_the_mb_report_is_refused(tmp_path, monkeypatch):
    """Even a correctly hash-bound Phase-MB report is CONSULTED, not merely re-hashed."""
    session = _fake_session(tmp_path)
    mb, _ = _mb_report_for_worker(session, micro_bsz=8, compile_on=False)
    _worker_validated(session, monkeypatch)
    monkeypatch.setattr(
        R,
        "validate_session_manifest",
        lambda path, validated: (
            json.loads(Path(path).read_bytes().decode("utf-8")),
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ),
    )
    candidates, plan = _publish_lr_plan(
        session,
        plan_kind="PHASE_LR_INITIAL_PLAN",
        peak_lrs=C.LR_GRID_SEED1,
        seed_labels=["seed-1"],
        # a geometry the bound report does not freeze
        derived_from={
            **mb,
            "frozen_geometry": {"micro_bsz": 1, "grad_accum": 128, "compile": True},
        },
    )
    with pytest.raises(R.BindingFailure, match="the Phase-MB report it binds freezes"):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, candidates[0]), gpu_required=False
        )


def test_an_lr_plan_with_an_underived_peak_lr_is_refused(tmp_path, monkeypatch):
    session = _fake_session(tmp_path)
    mb, _ = _mb_report_for_worker(session, micro_bsz=8, compile_on=False)
    _worker_validated(session, monkeypatch)
    monkeypatch.setattr(
        R,
        "validate_session_manifest",
        lambda path, validated: (
            json.loads(Path(path).read_bytes().decode("utf-8")),
            hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        ),
    )
    geometry = {"micro_bsz": 8, "grad_accum": 16, "compile": False}
    candidates, plan = _publish_lr_plan(
        session,
        plan_kind="PHASE_LR_INITIAL_PLAN",
        peak_lrs=[0.5],
        seed_labels=["seed-1"],
        derived_from={**mb, "frozen_geometry": geometry},
    )
    with pytest.raises(R.BindingFailure, match="but the contract derives"):
        R.validate_worker_execution(
            **_worker_kwargs(session, plan, candidates[0]), gpu_required=False
        )


def test_a_confirmation_plan_must_match_the_lr_the_initial_report_selected(tmp_path):
    """CONFIRMATION LRs are re-derived from the bound initial report, not declared."""
    session, report = _lr_chain(tmp_path)
    root = session.output_root
    assert report["terminal_status"] == "SUCCESS"
    published = json.loads(
        (root / R.PHASE_LR_CONFIRMATION_PLAN_FILENAME).read_text(encoding="utf-8")
    )
    derived = R.derive_planned_candidates(plan=published, session=session, authorized_root=root)
    # the seed-1 winner is 2e-4, so the neighbour is the higher 3e-4, both at seed-2
    assert sorted({c["peak_lr"] for c in derived}) == [2e-4, 3e-4]
    assert {c["seed_label"] for c in derived} == {"seed-2"}
    assert {c["micro_bsz"] for c in derived} == {8}
    assert {c["compile"] for c in derived} == {False}
    # forging the confirmation LR set is refused by the derivation from the bound report
    forged = {**published, "derived_from": {**published["derived_from"], "peak_lrs": [4e-4, 3e-4]}}
    with pytest.raises(R.BindingFailure, match="but the contract derives"):
        R.derive_planned_candidates(plan=forged, session=session, authorized_root=root)
    # so is running the confirmation at seed-1
    wrong_seed = {
        **published,
        "derived_from": {**published["derived_from"], "seed_labels": ["seed-1"]},
    }
    with pytest.raises(R.BindingFailure, match="but the contract derives"):
        R.derive_planned_candidates(plan=wrong_seed, session=session, authorized_root=root)
    # and so is pointing it at a preceding report whose bytes no longer hash to the recorded SHA
    initial = root / R.LR_INITIAL_REPORT_FILENAME
    initial.write_bytes(initial.read_bytes().replace(b'"step":"INITIAL"', b'"step":"FORGED!"'))
    with pytest.raises(R.BindingFailure, match="preceding_lr_report SHA-256"):
        R.derive_planned_candidates(plan=published, session=session, authorized_root=root)


def test_an_edge_plan_must_match_the_edge_the_confirmation_report_implies(tmp_path):
    """EDGE LRs are re-derived from the bound confirmation report's own pairs."""
    session, _ = _lr_chain(tmp_path)
    root = session.output_root
    edge_plan = json.loads((root / R.PHASE_LR_EDGE_PLAN_FILENAME).read_text(encoding="utf-8"))
    derived = R.derive_planned_candidates(plan=edge_plan, session=session, authorized_root=root)
    assert sorted({c["peak_lr"] for c in derived}) == [1e-4]
    assert sorted({c["seed_label"] for c in derived}) == ["seed-1", "seed-2"]
    # forge the edge LR: the derivation from the bound confirmation report refuses it
    forged = dict(edge_plan)
    forged["derived_from"] = {**edge_plan["derived_from"], "peak_lrs": [6e-4]}
    with pytest.raises(R.BindingFailure, match="but the contract derives"):
        R.derive_planned_candidates(plan=forged, session=session, authorized_root=root)
    # forge the geometry: the bound Phase-MB report refuses it
    forged2 = dict(edge_plan)
    forged2["derived_from"] = {
        **edge_plan["derived_from"],
        "frozen_geometry": {"micro_bsz": 1, "grad_accum": 128, "compile": True},
    }
    with pytest.raises(R.BindingFailure, match="the Phase-MB report it binds freezes"):
        R.derive_planned_candidates(plan=forged2, session=session, authorized_root=root)


def test_the_mb_report_verifier_is_session_free_and_recomputes(tmp_path):
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=4, compile_on=True)
    report = json.loads((session.output_root / R.MB_REPORT_FILENAME).read_text(encoding="utf-8"))
    frozen = R.verify_mb_report_document(report)
    assert (frozen["micro_bsz"], frozen["compile"]) == (4, True)
    assert frozen["grad_accum"] == C.frozen_grad_accum(4)
    forged = dict(report)
    forged["selection"] = {**report["selection"], "FROZEN_MICRO_BSZ": 1}
    with pytest.raises(R.BindingFailure, match="does not reproduce the published selection"):
        R.verify_mb_report_document(forged)
    with pytest.raises(R.BindingFailure, match="unusable physical VRAM"):
        R.verify_mb_report_document({**report, "physical_vram_bytes": None})


# ================= R3 re-review closure: unreadable artifacts are binding failures


@pytest.mark.parametrize("body", [b"", b"{", b"not json", b"[]", b'"a string"', b"\xff\xfe\x00"])
def test_an_unreadable_canonical_artifact_is_a_binding_failure(tmp_path, body):
    path = tmp_path / "artifact.json"
    path.write_bytes(body)
    with pytest.raises(R.BindingFailure):
        R.load_json_artifact(path, label="probe artifact")
    with pytest.raises(R.BindingFailure, match="could not be read"):
        R.load_json_artifact(tmp_path / "absent.json", label="probe artifact")


def test_a_corrupt_immutable_artifact_is_a_binding_failure(tmp_path):
    path = tmp_path / "PROBE.json"
    digest = R.write_immutable_artifact(path, {"schema_version": "probe", "a": 1})
    assert R.read_immutable_artifact(path, schema_version="probe")[1] == digest
    path.with_suffix(".sha256").write_text("", encoding="utf-8")
    with pytest.raises(R.BindingFailure, match="unreadable or empty"):
        R.read_immutable_artifact(path, schema_version="probe")
    path.with_suffix(".sha256").write_text("not-a-digest  PROBE.json\n", encoding="utf-8")
    with pytest.raises(R.BindingFailure, match="not a hex digest"):
        R.read_immutable_artifact(path, schema_version="probe")


def test_an_unpublishable_payload_is_a_named_phase_abort(tmp_path):
    """A non-finite value may never enter the immutable chain as a raw ValueError."""
    with pytest.raises(R.PhaseAbort, match="not canonically serializable"):
        R.write_immutable_artifact(tmp_path / "BAD.json", {"score": float("inf")})
    assert not (tmp_path / "BAD.json").exists()


def test_a_referenced_artifact_may_not_escape_the_authorized_root(tmp_path):
    root = tmp_path / "root"
    (root / "inner").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(R.BindingFailure, match="resolves outside the authorized root"):
        R._require_referenced_artifact(  # noqa: SLF001
            root, "../outside.json", hashlib.sha256(b"{}").hexdigest(), "probe"
        )


def test_a_pre_execution_fault_is_never_an_ineligible_candidate(tmp_path):
    """R4 artifact validation precedes imports/model construction and produces no output."""
    import inspect

    executor = inspect.getsource(R.execute_candidate_from_artifact_paths)
    assert executor.index("execution = validate_worker_execution(") < executor.index("import torch")
    cli = inspect.getsource(R._cli_internal_worker)  # noqa: SLF001
    assert "execute_candidate_from_artifact_paths(" in cli
    assert "write_terminal_result(" not in cli
    # a spec that is not even readable JSON exits BINDING_FAILURE, not CANDIDATE_INELIGIBLE
    spec = tmp_path / "corrupt.json"
    spec.write_bytes(b"{ this is not json")
    code = R.main(_worker_argv(tmp_path, spec))
    assert code == R.BINDING_FAILURE
    output = tmp_path / "root" / "mb_micro8_compileoff"
    assert not output.exists()
    assert not R.terminal_result_path(output).exists()
    assert not spec.with_suffix(".terminal.json").exists()


# ============ R3 re-review closure: an ineligible confirmation neighbour still publishes


def test_an_ineligible_confirmation_neighbour_still_publishes_its_report(tmp_path):
    """The confirmation pairs must stay canonically serializable (no inf)."""
    session = _fake_session(tmp_path)
    _frozen_mb_report(session, micro_bsz=8, compile_on=False)
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}

    def launcher(candidate, given, plan):
        # the seed-2 rerun of the confirmation NEIGHBOUR fails locally
        if candidate["seed_label"] == "seed-2" and candidate["peak_lr"] == 3e-4:
            _write_result(
                candidate,
                given,
                plan,
                _ineligible_lr_result(candidate, given, plan, oom=True),
            )
            return
        _write_result(
            candidate,
            given,
            plan,
            _bound_lr_result(candidate, given, plan, scores[candidate["peak_lr"]]),
        )

    report = R.orchestrate_phase_muon_lr(session, launcher=launcher)
    root = session.output_root
    confirmation = json.loads(
        (root / R.LR_CONFIRMATION_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    neighbour = next(p for p in confirmation["confirmation_pairs"] if p["peak_lr"] == 3e-4)
    assert neighbour["seed2_score"] is None and neighbour["seed2_eligible"] is False
    assert confirmation["selection"]["outcome"] == "CONFIRMED"
    assert confirmation["selection"]["confirmed_peak_lr"] == 2e-4
    assert report["terminal_status"] == "SUCCESS"
    # every published artifact in the chain round-trips through canonical JSON
    for name in (
        R.LR_INITIAL_REPORT_FILENAME,
        R.LR_CONFIRMATION_REPORT_FILENAME,
        R.LR_EDGE_REPORT_FILENAME,
        R.LR_REPORT_FILENAME,
    ):
        body = (root / name).read_bytes()
        assert C.canonical_json_bytes(json.loads(body.decode("utf-8"))) == body


def test_lr_eligibility_enforces_the_compile_path_in_both_directions():
    fell_back = _lr(3e-4, 3.0, compile=True, canonical_compile_path=False)
    ok, failures = C.lr_candidate_eligible(fell_back)
    assert not ok and "compile_silent_fallback" in failures
    unrequested = _lr(3e-4, 3.0, compile=False, canonical_compile_path=False)
    ok2, failures2 = C.lr_candidate_eligible(unrequested)
    assert not ok2 and "unrequested_compiled_path" in failures2
    assert C.lr_candidate_eligible(_lr(3e-4, 3.0))[0] is True


def test_ineligible_evidence_marks_a_missing_terminal_artifact(tmp_path):
    """An invented zero may never look like a real measurement."""
    session = _fake_session(tmp_path)
    candidates, plan = _published_mb_plan(session)

    def launcher(c, s, pl):
        raise R.CandidateFailure("simulated parent-side fault")

    evidence = R._launch(candidates[0], session, plan, launcher)  # noqa: SLF001
    assert evidence["terminal_evidence_present"] is False
    assert evidence["completed_updates"] == 0
    assert evidence["terminal_reserved_tokens"] is None


def test_the_launcher_reads_the_terminal_artifact_before_the_logs():
    """A fault while preserving logs must not discard the child's progress accounting."""
    import inspect

    src = inspect.getsource(R._subprocess_launcher)  # noqa: SLF001
    assert src.index("terminal = read_terminal_result(") < src.index("worker.stdout")
