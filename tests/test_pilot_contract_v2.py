"""P-PILOT-CONTRACT-V2.2 contract tests.

Pure contract logic and tiny synthetic CPU fixtures only. No 124,635,456-parameter training
path, no real gradient update, no CUDA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pretrain.pilot_contract_v2 as C
import pretrain.pilot_runner_v2 as R

REPO_ROOT = Path(__file__).resolve().parent.parent
A_BLOCKS, B_BLOCKS = 4882814, 1464845


# ------------------------------------------------------------------ schema + supersession


def test_contract_schema_and_version():
    d = C.contract_document()
    assert d["contract_version"] == "P-PILOT-CONTRACT-V2.2"
    assert d["schema_version"] == "petitgpt-pilot-contract-v2.2"
    assert d["authorization_status"] == "NOT_AUTHORIZED"
    for key in (
        "hardware",
        "effective_batch",
        "phase_mb",
        "pilot_indices",
        "optimizer",
        "phase_lr",
        "production_schedule",
        "budget",
        "checkpoint_isolation",
    ):
        assert key in d, key
    assert "PLAYBOOK.md 11.1" in d["supersedes"]
    assert "final-model-architecture" in d["does_not_supersede"]
    assert (
        C.contract_sha256()
        == hashlib.sha256(C.canonical_json_bytes(C.contract_document())).hexdigest()
    )


def test_contract_digest_is_stable_across_calls():
    assert C.contract_sha256() == C.contract_sha256()


# ------------------------------------------------------------------ effective batch


def test_effective_batch_constants():
    assert C.EFFECTIVE_BATCH_TOKENS == 262144
    assert C.SEQUENCES_PER_OPTIMIZER_UPDATE == 128
    assert C.SEQ_LEN == 2048
    assert C.SEQUENCES_PER_OPTIMIZER_UPDATE * C.SEQ_LEN == C.EFFECTIVE_BATCH_TOKENS


@pytest.mark.parametrize(("micro", "accum"), [(16, 8), (8, 16), (4, 32), (2, 64), (1, 128)])
def test_frozen_grad_accum(micro, accum):
    assert C.frozen_grad_accum(micro) == accum
    assert micro * accum == 128


def test_grad_accum_rejects_non_divisor():
    with pytest.raises(C.PilotContractError, match="does not divide"):
        C.frozen_grad_accum(5)


# ------------------------------------------------------------------ Phase MB grid + LR


def test_mb_grid_is_ten_unconditional_probes_in_fixed_order():
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
    for c in grid:
        assert c["updates"] == 40
        assert c["model_init_seed"] == 20260829
        assert c["micro_bsz"] * c["grad_accum"] == 128
    assert C.contract_document()["phase_mb"]["prior_measurements_may_exclude_a_candidate"] is False


@pytest.mark.parametrize(
    ("u", "expected"), [(1, 3e-4 * 0.1), (9, 3e-4 * 0.9), (10, 3e-4), (11, 3e-4), (40, 3e-4)]
)
def test_mb_lr_schedule(u, expected):
    assert C.mb_lr(u) == pytest.approx(expected)


def test_mb_lr_rejects_update_zero():
    with pytest.raises(C.PilotContractError, match="1-based"):
        C.mb_lr(0)


# ------------------------------------------------------------------ Phase MB selection


def _mb(cid, micro, compile_on, tps, vram_gib, **over):
    r = {
        "candidate_id": cid,
        "micro_bsz": micro,
        "compile": compile_on,
        "median_tokens_per_sec": tps,
        "max_memory_reserved_bytes": int(vram_gib * 1024**3),
        "completed_updates": 40,
        "oom": False,
        "uncontrolled_exception": False,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "adamw_state_complete": True,
        "canonical_compile_path": True,
    }
    r.update(over)
    return r


VRAM = 24 * 1024**3


def test_mb_select_fastest_unique():
    out = C.mb_select([_mb("a", 8, False, 50000, 10), _mb("b", 4, False, 40000, 8)], VRAM)
    assert out["outcome"] == "PHASE_MB_FROZEN"
    assert out["FROZEN_MICRO_BSZ"] == 8 and out["FROZEN_GRAD_ACCUM"] == 16
    assert out["FROZEN_COMPILE"] is False
    assert out["tie_break"] == "fastest_unique"


def test_mb_tie_break_lowest_vram():
    out = C.mb_select([_mb("a", 8, False, 50000, 12), _mb("b", 4, False, 49500, 8)], VRAM)
    assert out["tie_break"] == "lowest_peak_reserved_vram"
    assert out["FROZEN_MICRO_BSZ"] == 4


def test_mb_tie_break_compile_off_within_256mib():
    out = C.mb_select([_mb("a", 8, True, 50000, 8.0), _mb("b", 8, False, 49500, 8.1)], VRAM)
    assert out["tie_break"] == "compile_off_preferred"
    assert out["FROZEN_COMPILE"] is False


def test_mb_tie_break_larger_micro_bsz_last():
    out = C.mb_select([_mb("a", 4, False, 50000, 8.0), _mb("b", 8, False, 49500, 8.05)], VRAM)
    assert out["tie_break"] == "larger_micro_bsz"
    assert out["FROZEN_MICRO_BSZ"] == 8


@pytest.mark.parametrize(
    "bad",
    [
        {"completed_updates": 39},
        {"oom": True},
        {"all_losses_finite": False},
        {"all_grad_norms_finite": False},
        {"adamw_state_complete": False},
        {"uncontrolled_exception": True},
    ],
)
def test_mb_eligibility_rejects_each_failure(bad):
    ok, failures = C.mb_candidate_eligible(_mb("x", 8, False, 1000, 8, **bad), VRAM)
    assert not ok and failures


def test_mb_vram_ceiling_is_ninety_percent():
    ok, _ = C.mb_candidate_eligible(_mb("x", 8, False, 1000, 21.5), VRAM)
    assert ok
    ok2, failures = C.mb_candidate_eligible(_mb("x", 8, False, 1000, 22.0), VRAM)
    assert not ok2 and "vram_reserved_above_90_percent" in failures


def test_mb_compile_silent_fallback_rejected():
    ok, failures = C.mb_candidate_eligible(
        _mb("x", 8, True, 1000, 8, canonical_compile_path=False), VRAM
    )
    assert not ok and "compile_silent_fallback" in failures


def test_mb_all_ineligible_aborts():
    out = C.mb_select([_mb("a", 8, False, 1, 8, oom=True)], VRAM)
    assert out["outcome"] == "PHASE_MB_ABORT"


# ------------------------------------------------------------------ indices


@pytest.fixture(scope="module")
def indices():
    return C.generate_pilot_indices(A_BLOCKS, B_BLOCKS)


def test_index_counts_and_ranges(indices):
    assert len(indices["stage_a_eval"]) == 4096
    assert len(indices["stage_a_train"]) == 131072
    assert len(indices["stage_b_eval"]) == 4096
    assert all(0 <= v < A_BLOCKS for v in indices["stage_a_eval"])
    assert all(0 <= v < A_BLOCKS for v in indices["stage_a_train"])
    assert all(0 <= v < B_BLOCKS for v in indices["stage_b_eval"])


def test_index_disjointness_and_serialization(indices):
    assert not (set(indices["stage_a_eval"]) & set(indices["stage_a_train"]))
    assert indices["stage_a_eval"] == sorted(indices["stage_a_eval"])
    assert indices["stage_b_eval"] == sorted(indices["stage_b_eval"])
    # train is stored in draw order, so it must NOT be sorted
    assert indices["stage_a_train"] != sorted(indices["stage_a_train"])


def test_index_generation_is_deterministic(indices):
    again = C.generate_pilot_indices(A_BLOCKS, B_BLOCKS)
    for key in ("stage_a_eval_sha256", "stage_a_train_sha256", "stage_b_eval_sha256"):
        assert again[key] == indices[key]
    assert again["stage_a_train"] == indices["stage_a_train"]


def test_canonical_index_hashes_match_the_serialized_form(indices):
    payload = ("\n".join(str(v) for v in indices["stage_a_eval"]) + "\n").encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == indices["stage_a_eval_sha256"]


def test_seed_1_and_seed_2_train_orders_differ_but_share_the_set(indices):
    o1 = C.train_order(indices["stage_a_train"], 20260829)
    o2 = C.train_order(indices["stage_a_train"], 20260830)
    assert set(o1) == set(o2) == set(indices["stage_a_train"])
    assert o1 != o2
    assert C.train_order(indices["stage_a_train"], 20260829) == o1


def test_train_order_rejects_unknown_seed(indices):
    with pytest.raises(C.PilotContractError, match="train-order seed"):
        C.train_order(indices["stage_a_train"][:16], 12345)


def test_400_updates_consume_exactly_51200_stage_a_blocks():
    assert R.blocks_consumed(C.LR_RUN_UPDATES) == 51200
    assert 51200 <= C.STAGE_A_TRAIN_COUNT


def test_seed_semantics():
    assert C.SEED_SEMANTICS["seed-1"] == {"model_init": 20260829, "train_order": 20260829}
    assert C.SEED_SEMANTICS["seed-2"] == {"model_init": 20260830, "train_order": 20260830}


def test_index_generation_rejects_too_small_universe():
    with pytest.raises(C.PilotContractError, match="needs at least"):
        C.generate_pilot_indices(1000, 100000)


# ------------------------------------------------------------------ optimizer


def test_full_adamw_config_binding():
    cfg = C.realized_adamw_config()
    assert cfg["optimizer"] == "adamw"
    assert cfg["betas"] == [0.9, 0.95]
    assert cfg["eps"] == 1e-8
    assert cfg["weight_decay"] == 0.1
    assert cfg["grad_clip"] == 1.0
    assert cfg["fused_on_cuda"] is True
    assert cfg["muon_in_scope"] is False
    groups = cfg["param_group_membership"]
    assert groups["adam_decay_names"] == ["tok_emb", "lm_head", ".gate."]
    assert "ndim < 2" in groups["no_decay"]


def test_muon_default_must_be_overridden_explicitly():
    import inspect

    from src.optim import build_optimizer

    assert inspect.signature(build_optimizer).parameters["name"].default == "muon"
    assert "must be passed explicitly" in C.realized_adamw_config()["explicit_cli_requirement"]


# ------------------------------------------------------------------ Phase LR


@pytest.mark.parametrize(("u", "mult"), [(1, 1 / 50), (49, 49 / 50), (50, 1.0), (51, 1.0)])
def test_lr_schedule(u, mult):
    assert C.lr_schedule(u, 3e-4) == pytest.approx(3e-4 * mult)


def test_lr_schedule_rejects_update_zero():
    with pytest.raises(C.PilotContractError, match="1-based"):
        C.lr_schedule(0, 3e-4)


def test_lr_score_weights():
    assert C.lr_score(1.0, 1.0) == pytest.approx(1.0)
    assert C.lr_score(2.0, 1.0) == pytest.approx((10 * 2.0 + 3 * 1.0) / 13)


def _lr(peak, score, **over):
    r = {
        "peak_lr": peak,
        "score": score,
        "completed_updates": 400,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "eval_loss_stage_a": 3.0,
        "eval_loss_stage_b": 3.0,
        "sustained_divergence": False,
    }
    r.update(over)
    return r


def test_lr_seed1_selection_lowest_score():
    out = C.lr_select_seed1([_lr(2e-4, 3.5), _lr(3e-4, 3.2), _lr(4e-4, 3.3), _lr(6e-4, 3.9)])
    assert out["outcome"] == "SEED1_WINNER" and out["winner_peak_lr"] == 3e-4


def test_lr_half_percent_tie_goes_to_lower_lr():
    out = C.lr_select_seed1([_lr(2e-4, 3.210), _lr(3e-4, 3.200), _lr(4e-4, 3.9), _lr(6e-4, 4.0)])
    assert out["winner_peak_lr"] == 2e-4  # 3.210 is within 0.5% of 3.200


def test_lr_fewer_than_two_eligible_aborts():
    out = C.lr_select_seed1([_lr(2e-4, 3.5), _lr(3e-4, 3.2, completed_updates=399)])
    assert out["outcome"] == "PHASE_LR_ABORT"


@pytest.mark.parametrize(
    ("winner", "neighbor"), [(2e-4, 3e-4), (3e-4, 2e-4), (4e-4, 3e-4), (6e-4, 4e-4)]
)
def test_confirmation_neighbor_rule(winner, neighbor):
    assert C.confirmation_neighbor(winner) == pytest.approx(neighbor)


def test_confirmation_final_score_and_tie():
    out = C.lr_confirm([
        {"peak_lr": 3e-4, "seed1_score": 3.20, "seed2_score": 3.20, "seed2_eligible": True},
        {"peak_lr": 2e-4, "seed1_score": 3.21, "seed2_score": 3.21, "seed2_eligible": True},
    ])
    assert out["FROZEN_PEAK_LR"] == 2e-4  # within 0.5%, lower LR wins


def test_confirmation_one_ineligible_other_wins():
    out = C.lr_confirm([
        {"peak_lr": 3e-4, "seed1_score": 3.2, "seed2_score": 3.2, "seed2_eligible": False},
        {"peak_lr": 2e-4, "seed1_score": 3.6, "seed2_score": 3.6, "seed2_eligible": True},
    ])
    assert out["outcome"] == "PHASE_LR_FROZEN" and out["FROZEN_PEAK_LR"] == 2e-4


def test_confirmation_both_ineligible_aborts():
    out = C.lr_confirm([
        {"peak_lr": 3e-4, "seed1_score": 3.2, "seed2_score": 3.2, "seed2_eligible": False},
        {"peak_lr": 2e-4, "seed1_score": 3.6, "seed2_score": 3.6, "seed2_eligible": False},
    ])
    assert out["outcome"] == "PHASE_LR_ABORT"


@pytest.mark.parametrize(
    ("winner", "edge"), [(2e-4, 1e-4), (6e-4, 8e-4), (3e-4, None), (4e-4, None)]
)
def test_edge_expansion(winner, edge):
    assert C.edge_expansion_candidate(winner) == (pytest.approx(edge) if edge else None)


def test_sustained_divergence_guard():
    calm = {u: 3.0 for u in range(1, 401)}
    assert C.sustained_divergence(calm)["diverged"] is False
    blown = dict(calm)
    for u in range(120, 160):
        blown[u] = 9.0
    result = C.sustained_divergence(blown)
    assert result["diverged"] is True
    assert result["threshold"] == pytest.approx(4.5)
    assert all(v["window_start"] >= 101 for v in result["violations"])


def test_sustained_divergence_ignores_pre_101_spike():
    losses = {u: 3.0 for u in range(1, 401)}
    for u in range(60, 80):
        losses[u] = 20.0
    assert C.sustained_divergence(losses)["diverged"] is False


def test_lr_eligibility_flags_divergence():
    ok, failures = C.lr_candidate_eligible(_lr(3e-4, 3.2, sustained_divergence=True))
    assert not ok and "sustained_divergence" in failures


# ------------------------------------------------------------------ schedule + decay


def test_frozen_warmup_is_convention_not_pilot_derived():
    assert C.FROZEN_WARMUP_STEPS == 500
    assert "NOT pilot-derived" in C.FROZEN_WARMUP_STEPS_AUTHORITY


def test_owner_intent_and_planner_input_are_named_separately():
    d = C.contract_document()["production_schedule"]
    assert d["OWNER_DECAY_INTENT_FRACTION_OF_TOTAL"] == 0.10
    assert d["PLANNER_DECAY_FRACTION_INPUT"] == 0.10
    assert d["MIN_LR_RATIO"] == 0.10
    assert "schedule_total_steps" in d["decay_encoding"]["planner_flag_scope"]


def test_decay_verification_against_a_local_planner_fixture(tmp_path, monkeypatch):
    """Drive the REAL planner on tiny synthetic shards and verify the owner decay intent."""
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
    result = C.verify_decay_encoding(plan["boundaries"], plan["wsd_candidate"])
    assert result["all_passed"], result
    assert result["rounding_difference_steps"] <= 1
    assert plan["wsd_candidate"]["decay_end_step"] == plan["boundaries"]["schedule_total_steps"]


def test_decay_verification_rejects_decay_outside_stage_b():
    bad = C.verify_decay_encoding(
        {"schedule_total_steps": 1000, "stage_b_start_step": 800, "stage_b_global_stop_step": 1000},
        {"decay_start_step": 700, "decay_end_step": 1000},
    )
    assert bad["checks"]["decay_interval_wholly_in_stage_b"] is False
    assert bad["all_passed"] is False


def test_decay_verification_rejects_wrong_end_step():
    bad = C.verify_decay_encoding(
        {"schedule_total_steps": 1000, "stage_b_start_step": 800, "stage_b_global_stop_step": 1000},
        {"decay_start_step": 900, "decay_end_step": 990},
    )
    assert bad["checks"]["decay_end_equals_schedule_total"] is False


# ------------------------------------------------------------------ budget + isolation


def test_phase_mb_projection_within_ceiling():
    assert C.phase_mb_projected_tokens() == 10 * 40 * 262144 == 104857600
    assert C.phase_mb_projected_tokens() <= C.PHASE_MB_TRAINED_TOKEN_CEILING


def test_budget_breaches():
    assert C.budget_status(10**8, 10**8, 8)["outcome"] == "WITHIN_BUDGET"
    assert "global_pilot_token_ceiling" in C.budget_status(10**9, 10**9, 8)["breaches"]
    assert "phase_lr_run_ceiling" in C.budget_status(0, 0, 9)["breaches"]
    assert "phase_mb_token_ceiling" in C.budget_status(2 * 10**8, 0, 1)["breaches"]
    assert C.budget_status(0, 0, 9)["outcome"] == "PILOT_ABORT"
    assert C.budget_status(0, 0, 1)["ceiling_may_be_increased_inside_this_contract"] is False


@pytest.mark.parametrize("purpose", ["another candidate", "stage_n", "stage_o"])
def test_pilot_checkpoint_never_initializes_anything(purpose):
    with pytest.raises(C.PilotContractError, match="forbids initializing"):
        C.reject_pilot_checkpoint_as_initialization(purpose)


def test_checkpoint_isolation_flags():
    iso = C.CHECKPOINT_ISOLATION
    assert iso["candidates_always_start_fresh"] is True
    assert iso["pilot_checkpoint_may_initialize_stage_n"] is False
    assert iso["pilot_checkpoint_may_initialize_stage_o"] is False
    assert iso["pilot_checkpoint_may_initialize_another_candidate"] is False


# ------------------------------------------------------------------ authorization


def test_authorization_template_is_never_authorized():
    t = C.authorization_template()
    assert t["authorization_status"] == "NOT_AUTHORIZED"
    assert t["authorized_implementation_head"] is None
    assert t["authorized_contract_sha256"] is None
    assert t["authorized_by"] is None
    # No field anywhere in the template may carry the bare AUTHORIZED status value.
    values = [v for v in json.loads(json.dumps(t)).values() if isinstance(v, str)]
    assert "AUTHORIZED" not in [v for v in values if v != "NOT_AUTHORIZED"]


def test_launch_authorization_always_refuses():
    with pytest.raises(C.PilotContractError, match="NOT_AUTHORIZED"):
        C.require_launch_authorization()
    with pytest.raises(C.PilotContractError):
        C.require_launch_authorization({"authorization_status": "AUTHORIZED"})


def test_runner_execute_refuses():
    with pytest.raises(C.PilotContractError):
        R.execute_candidate(candidate="mb_micro8_compileoff")


def test_generated_plans_serialize_not_authorized(tmp_path):
    for plan in R.plan_phase_mb(output_root=tmp_path / "out"):
        assert plan["authorization_status"] == "NOT_AUTHORIZED"
    for plan in R.plan_phase_lr(output_root=tmp_path / "out", micro_bsz=8, compile_on=False):
        assert plan["authorization_status"] == "NOT_AUTHORIZED"


# ------------------------------------------------------------------ hardware authority


def test_gpu_training_authority():
    assert C.gpu_has_training_authority("NVIDIA GeForce RTX 4090") is True
    assert C.gpu_has_training_authority("NVIDIA RTX 4000 Ada Generation") is False
    assert C.HARDWARE["rtx_4000_ada_training_authority"] == "NONE"


# ------------------------------------------------------------------ runner: paths + policy


def test_output_dir_must_be_new(tmp_path):
    existing = tmp_path / "already"
    existing.mkdir()
    with pytest.raises(C.PilotContractError, match="must not exist"):
        R.require_new_output_dir(existing)
    assert R.require_new_output_dir(tmp_path / "fresh")


@pytest.mark.parametrize(
    "protected",
    [
        "runs/m_production_v1_2026-08-29/release/stage_a/pilot",
        "runs/m_production_v1_2026-08-29/release",
        "runs/i_production_v1_2026-08-25/x",
        "runs/g_production_2026-08-21/release/y",
    ],
)
def test_output_dir_never_inside_an_accepted_release(protected):
    with pytest.raises(C.PilotContractError, match="accepted release"):
        R.require_new_output_dir(Path(protected))


def test_repo_root_is_cwd_independent(tmp_path, monkeypatch):
    before = R.repo_root()
    monkeypatch.chdir(tmp_path)
    assert R.repo_root() == before == REPO_ROOT


def test_git_policy_detects_changed_probe_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "ALLOWED_UNTRACKED_SHA256", "0" * 64)
    status = R.git_policy_status()
    assert status["allowed_untracked_unchanged"] is False
    assert "allowed_historical_untracked_file_bytes_changed" in status["failures"]


def test_git_policy_rejects_extra_untracked(monkeypatch):
    real = R._git

    def fake(*args):
        if args[:2] == ("status", "--porcelain") and len(args) == 2:
            return "?? .codex_r1_manual_context_probe.py\n?? pretrain/rogue_module.py"
        return real(*args)

    monkeypatch.setattr(R, "_git", fake)
    status = R.git_policy_status()
    assert status["unexpected_untracked"] == ["pretrain/rogue_module.py"]
    assert "uncontrolled_untracked_files_present" in status["failures"]
    with pytest.raises(C.PilotContractError, match="Git policy"):
        R.require_git_policy()


def test_allowed_untracked_identity_is_the_recorded_probe():
    assert R.ALLOWED_UNTRACKED == ".codex_r1_manual_context_probe.py"
    probe = REPO_ROOT / R.ALLOWED_UNTRACKED
    assert R.file_sha256(probe) == R.ALLOWED_UNTRACKED_SHA256


# ------------------------------------------------------------------ fingerprint + run_meta


def test_base_fingerprint_has_no_global_compile_value():
    fp = R.base_runtime_fingerprint()
    assert "compile" not in fp
    assert "compile" not in fp.get("gpu", {})
    for key in (
        "torch_version",
        "numpy_version",
        "python_version",
        "python_executable",
        "repository",
        "contract_sha256",
        "implementation_bundle_sha256",
        "fingerprint_sha256",
    ):
        assert key in fp, key


def test_run_meta_binds_per_candidate_compile_separately(tmp_path):
    fp = R.base_runtime_fingerprint()
    idx = C.generate_pilot_indices(200000, 10000)
    off, on = R.plan_phase_mb(output_root=tmp_path / "o")[0:2]
    m_off = R.run_meta(candidate=off, fingerprint=fp, index_manifest=idx, implementation_head="x")
    m_on = R.run_meta(candidate=on, fingerprint=fp, index_manifest=idx, implementation_head="x")
    assert m_off["compile"] is False and m_on["compile"] is True
    assert m_off["base_fingerprint_sha256"] == m_on["base_fingerprint_sha256"]
    for key in (
        "phase",
        "candidate_id",
        "micro_bsz",
        "grad_accum",
        "optimizer",
        "lr_configuration",
        "model_seed",
        "train_order_seed",
        "pilot_index_hashes",
        "contract_sha256",
        "implementation_head",
    ):
        assert key in m_off, key
    assert m_off["authorization_status"] == "NOT_AUTHORIZED"


def test_implementation_bundle_binds_both_modules():
    files, digest = R.implementation_bundle()
    assert sorted(files) == ["pretrain/pilot_contract_v2.py", "pretrain/pilot_runner_v2.py"]
    assert len(digest) == 64


# ------------------------------------------------------------------ production loss-mask parity


def test_pilot_reuses_the_production_loss_mask_semantics():
    """The pilot must not define its own mask; it reuses the canonical production helper."""
    import inspect

    from pretrain.dataset_pretrain import PackedBinDataset  # noqa: F401

    src = inspect.getsource(R)
    assert "loss_mask" not in src or "canonical" in src
    # The runner must not reimplement the model or the optimizer.
    assert "class GPT" not in src
    assert "torch.optim.AdamW(" not in src
