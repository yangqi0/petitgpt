"""P-PRODUCTION-LAUNCH-CONTRACT-V1 contract tests.

No real model training: parser/namespace fixtures, tiny synthetic modules and
non-training construction only. No forward, backward or optimizer update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

from pretrain import production_launch_contract_v1 as C  # noqa: E402

EXACT_PLAN_SHA = C.EXACT_RUN_PLAN_SHA256
ACCEPTANCE_SHA = C.PILOT_OWNER_ACCEPTANCE_SHA256


# --------------------------------------------------------------------- fixtures


def _runtime() -> dict:
    return {
        "gpu_uuid": "GPU-0000dead-beef-0000-0000-000000000000",
        "gpu_pci_bus_id": "00000000:02:00.0",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "visible_cuda_device_count": 1,
        "total_vram_bytes": 25250627584,
        "compute_capability": "8.9",
        "driver_version": "580.126.20",
        "cuda_runtime_version": "12.6",
        "torch_version": "2.11.0+cu126",
        "python_version": "3.10.12",
        "numpy_version": "2.2.6",
        "trainer_head": "0" * 40,
        "trainer_execution_bundle_sha256": "1" * 64,
        "canonical_cwd": C.CANONICAL_CWD,
        "num_workers": 2,
    }


def governed_args(stage: str = "stage_a", **overrides) -> argparse.Namespace:
    """A namespace that satisfies every governed check, before per-test mutation."""
    bounds = C.STAGE_BOUNDARIES[stage]
    ns = argparse.Namespace(
        layers=30,
        d_model=576,
        n_heads=9,
        n_kv_heads=3,
        d_ff=1536,
        seq_len=2048,
        vocab_size=32000,
        dropout=0.0,
        optimizer="muon",
        lr=0.0006,
        muon_lr=0.0,
        muon_momentum=0.95,
        min_lr_ratio=0.10,
        warmup_steps=500,
        lr_schedule="wsd",
        weight_decay=0.1,
        grad_clip=1.0,
        micro_bsz=8,
        grad_accum=16,
        precision="bf16",
        compile=True,
        schedule_total_steps=49590,
        decay_start_step=44631,
        decay_end_step=49590,
        run_plan_stage=stage,
        data_stage_start_step=bounds["start_step"],
        max_steps=bounds["stop_step"],
        seed=C.MODEL_INIT_SEED,
        val_seed=C.VALIDATION_SEED,
        stage_a_sampler_seed=C.STAGE_A_SAMPLER_SEED,
        stage_b_sampler_seed=C.STAGE_B_SAMPLER_SEED,
        # R1 owner clarification 4: the governed path normalizes the legacy field to the
        # ACTIVE stage seed, so a correct governed namespace carries that value.
        sampler_seed=(C.STAGE_A_SAMPLER_SEED if stage == "stage_a" else C.STAGE_B_SAMPLER_SEED),
        eval_steps=list(C.EVALUATION_MILESTONES),
        eval_every=0,
        bench_eval_every=0,
        bench_eval_path="",
        val_samples=0,
        val_samples_per_source=0,
        eos_weight=1.0,
        eos_weight_warmup_steps=0,
        save_steps=list(C.CHECKPOINT_MILESTONES),
        save_every=0,
        mask_last_label_in_loss=False,
        no_mask_bos_in_loss=False,
        no_mask_last_label_in_loss=False,
        allow_weights_only_resume=False,
        allow_data_branch=False,
        allow_schedule_branch=False,
        strict_resume_contract=True,
        out_dir=str(REPO / "outputs" / "governed"),
        bos_id=C.CANONICAL_BOS_ID,
        eos_id=C.CANONICAL_EOS_ID,
        num_workers=2,
        resume_path="",
        resume_full=False,
        resume_step=-1,
        log_every=20,
        debug_every=500,
        bench_eval_script="pretrain/eval_bench_v5.py",
        bench_eval_out_dir="",
        bench_eval_max_seq_len=1024,
        bench_eval_max_new_tokens=192,
        bench_eval_min_new_tokens=1,
        bench_eval_ban_first_steps=4,
        add_bos_to_prompts=True,
        sample_temperature=0.7,
        sample_top_p=0.9,
        sample_top_k=0,
        sample_max_new_tokens=256,
        sample_min_new_tokens=0,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def authorization(scope: str = "STAGE_N", **overrides) -> dict:
    manifest = C.authorization_template()
    manifest.update({
        "authorization_status": "AUTHORIZED",
        "allowed_scope": scope,
        "repository_branch": "agent/retrain-pipeline-contracts",
        "trainer_head": "0" * 40,
        "trainer_execution_bundle_sha256": "1" * 64,
        "launch_contract_sha256": C.contract_sha256(),
        "exact_run_plan_sha256": EXACT_PLAN_SHA,
        "pilot_owner_acceptance_sha256": ACCEPTANCE_SHA,
        "allowed_output_root": str(REPO / "outputs" / "governed"),
        "canonical_cwd": C.CANONICAL_CWD,
        "training_runtime": _runtime(),
        "resume": {"mode": "FRESH"},
        "authorized_by": "Yang Qi",
        "authorized_at": "2026-08-31T00:00:00Z",
    })
    manifest.update(overrides)
    return manifest


def observed(scope_stage: str = "stage_a") -> dict:
    return {
        "branch": "agent/retrain-pipeline-contracts",
        "head": "0" * 40,
        "trainer_execution_bundle_sha256": "1" * 64,
        "launch_contract_sha256": C.contract_sha256(),
        "exact_run_plan_sha256": EXACT_PLAN_SHA,
        "pilot_owner_acceptance_sha256": ACCEPTANCE_SHA,
        "output_root": str(REPO / "outputs" / "governed"),
        "canonical_cwd": C.CANONICAL_CWD,
        "training_runtime": _runtime(),
    }


# --------------------------------------------------------------------- contract schema


def test_contract_schema_and_identity_are_stable():
    doc = C.contract_document()
    assert doc["schema_version"] == C.CONTRACT_SCHEMA
    assert doc["contract_version"] == C.CONTRACT_VERSION
    assert doc["authorization_status"] == "NOT_AUTHORIZED"
    assert doc["authorizes_training"] is False
    assert C.contract_sha256() == C.contract_sha256()
    assert len(C.contract_sha256()) == 64


def test_contract_binds_accepted_stage_p_inputs():
    acc = C.contract_document()["accepted_stage_p_inputs"]
    assert acc["exact_run_plan_sha256"] == EXACT_PLAN_SHA
    assert acc["pilot_owner_acceptance_sha256"] == ACCEPTANCE_SHA
    assert acc["plan_generation_head"] == "4306f1db60b2c283f504404627e74f921c601800"
    assert acc["exact_plan_is_not_a_training_authorization"] is True


def test_exact_plan_and_acceptance_bytes_match_on_disk():
    for relpath, expected in (
        (C.EXACT_RUN_PLAN_RELPATH, EXACT_PLAN_SHA),
        (C.PILOT_OWNER_ACCEPTANCE_RELPATH, ACCEPTANCE_SHA),
    ):
        assert C.file_sha256(REPO / relpath) == expected


# --------------------------------------------------------------------- classification


def _parser_dests() -> list[str]:
    captured = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        import train_pretrain_with_bench as trainer

        try:
            trainer.parse_args()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return [a.dest for a in captured["parser"]._actions if a.dest != "help"]


def test_every_parser_field_is_classified_exactly_once():
    result = C.classify_parser_namespace(_parser_dests())
    assert result["unclassified_fields"] == []
    assert result["classified_but_absent_from_parser"] == []
    assert result["complete"] is True


def test_every_classification_uses_a_known_class():
    for dest, spec in C.PARSER_FIELD_CLASSIFICATION.items():
        assert spec["class"] in C.FIELD_CLASSES, dest


# --------------------------------------------------------------------- value rejection


@pytest.mark.parametrize(
    "field,bad",
    [
        ("lr", 0.0003),
        ("optimizer", "adamw"),
        ("muon_lr", 0.0006),
        ("muon_momentum", 0.9),
        ("micro_bsz", 4),
        ("grad_accum", 8),
        ("precision", "fp32"),
        ("compile", False),
        ("warmup_steps", 1000),
        ("min_lr_ratio", 0.0),
        ("lr_schedule", "cosine"),
        ("weight_decay", 0.05),
        ("grad_clip", 0.5),
        ("layers", 16),
        ("d_model", 768),
        ("n_heads", 12),
        ("n_kv_heads", 12),
        ("d_ff", 2048),
        ("seq_len", 1024),
        ("vocab_size", 32768),
        ("dropout", 0.1),
    ],
)
def test_wrong_governed_value_is_rejected(field, bad):
    failures = C.validate_governed_args(governed_args(**{field: bad}), stage="stage_a")
    assert any(f.startswith(field + ":") or field in f for f in failures), failures


def test_correct_governed_namespace_has_no_failures():
    assert C.validate_governed_args(governed_args("stage_a"), stage="stage_a") == []
    assert C.validate_governed_args(governed_args("stage_b"), stage="stage_b") == []


def test_effective_batch_tokens_must_be_262144():
    # micro_bsz*grad_accum*seq_len is checked as a derived product, so the pair must
    # actually move it: 16x8 still yields 262144 and is caught only by the field checks.
    failures = C.validate_governed_args(governed_args(micro_bsz=4, grad_accum=16), stage="stage_a")
    assert any("effective_batch_tokens" in f for f in failures)

    invariant = C.validate_governed_args(governed_args(micro_bsz=16, grad_accum=8), stage="stage_a")
    assert not any("effective_batch_tokens" in f for f in invariant)
    assert any(f.startswith("micro_bsz:") for f in invariant)


# --------------------------------------------------------------------- seeds


@pytest.mark.parametrize(
    "field,bad",
    [
        ("seed", 1234),
        ("stage_a_sampler_seed", 1234),
        ("stage_b_sampler_seed", 1234),
        ("val_seed", 1234),
    ],
)
def test_seed_mismatch_is_rejected(field, bad):
    failures = C.validate_governed_args(governed_args(**{field: bad}), stage="stage_a")
    assert any(field in f for f in failures), failures


def test_pilot_seeds_are_rejected_as_production_seeds():
    failures = C.validate_governed_args(governed_args(seed=20260829), stage="stage_a")
    assert any("pilot seed" in f for f in failures)


def test_stage_seeds_are_distinct_and_stage_scoped():
    assert C.STAGE_A_SAMPLER_SEED != C.STAGE_B_SAMPLER_SEED
    assert C.stage_sampler_seed("stage_a") == C.STAGE_A_SAMPLER_SEED
    assert C.stage_sampler_seed("stage_b") == C.STAGE_B_SAMPLER_SEED
    with pytest.raises(C.LaunchContractError):
        C.stage_sampler_seed("stage_c")


def test_legacy_shared_sampler_seed_is_normalized_not_an_independent_authority():
    """R1 owner clarification 4 supersedes the earlier 'must not equal a stage seed' rule.

    The legacy field is mechanically normalized to the ACTIVE stage seed and validated, so it
    can never select a different permutation. A value that would select a different
    permutation is still rejected.
    """
    normalized = C.validate_governed_args(
        governed_args(sampler_seed=C.STAGE_A_SAMPLER_SEED), stage="stage_a"
    )
    assert not any("sampler_seed" in f for f in normalized)

    divergent = C.validate_governed_args(governed_args(sampler_seed=999999), stage="stage_a")
    assert any("different permutation" in f for f in divergent)


def test_trainer_resolves_only_the_matching_stage_seed():
    import train_pretrain_with_bench as trainer

    args = governed_args("stage_a")
    assert trainer.resolve_stage_sampler_seed(args, "stage_a") == C.STAGE_A_SAMPLER_SEED
    assert trainer.resolve_stage_sampler_seed(args, "stage_b") == C.STAGE_B_SAMPLER_SEED


def test_worker_rng_seed_is_derived_from_the_frozen_seed():
    assert C.worker_rng_seed(C.STAGE_A_SAMPLER_SEED, purpose="train_loader") == (
        C.STAGE_A_SAMPLER_SEED + 17
    )
    assert C.worker_rng_seed(C.VALIDATION_SEED, purpose="val_by_source_loader") == (
        C.VALIDATION_SEED + 19
    )


# --------------------------------------------------------------------- evaluation


def test_evaluation_milestones_are_exactly_the_owner_list():
    assert C.EVALUATION_MILESTONES == (
        500,
        3815,
        11445,
        22889,
        38146,
        38147,
        43868,
        43870,
        44631,
        49590,
    )
    assert C.EVALUATION_POLICY["mode"] == "explicit_milestones_only"
    assert C.EVALUATION_POLICY["periodic_eval_status"] == "DISABLED"
    assert C.EVALUATION_POLICY["benchmark_eval_status"] == "DISABLED"


@pytest.mark.parametrize(
    "steps",
    [
        list(C.EVALUATION_MILESTONES)[:-1],  # missing
        [*C.EVALUATION_MILESTONES, 49591],  # extra
        list(reversed(C.EVALUATION_MILESTONES)),  # reordered
        [500, 500, *C.EVALUATION_MILESTONES[1:]],  # duplicated
    ],
)
def test_bad_evaluation_milestone_list_is_rejected(steps):
    failures = C.validate_governed_args(governed_args(eval_steps=steps), stage="stage_a")
    assert any("eval_steps" in f for f in failures), failures


def test_periodic_eval_enabled_is_rejected():
    failures = C.validate_governed_args(governed_args(eval_every=1000), stage="stage_a")
    assert any("eval_every" in f for f in failures)


def test_benchmark_evaluation_enabled_is_rejected():
    assert any(
        "bench_eval_every" in f
        for f in C.validate_governed_args(governed_args(bench_eval_every=1000), stage="stage_a")
    )
    assert any(
        "bench_eval_path" in f
        for f in C.validate_governed_args(
            governed_args(bench_eval_path="pretrain/bench_v1.jsonl"), stage="stage_a"
        )
    )


@pytest.mark.parametrize("field", ["val_samples", "val_samples_per_source"])
def test_validation_subset_or_truncation_is_rejected(field):
    failures = C.validate_governed_args(governed_args(**{field: 200}), stage="stage_a")
    assert any(field in f for f in failures)


def test_validation_policy_is_full_deterministic_no_subset():
    policy = C.EVALUATION_POLICY
    assert policy["validation_full_set"] is True
    assert policy["random_validation_subset"] is False
    assert policy["validation_shuffle"] is False
    assert policy["validation_order"] == "deterministic_no_shuffle"
    assert policy["validation_seed"] == C.VALIDATION_SEED


def test_trainer_evaluates_only_at_milestones_when_periodic_is_disabled():
    import train_pretrain_with_bench as trainer

    steps = list(C.EVALUATION_MILESTONES)
    assert trainer.should_evaluate(500, eval_every=0, eval_steps=steps) is True
    assert trainer.should_evaluate(49590, eval_every=0, eval_steps=steps) is True
    for off_milestone in (1, 499, 501, 1000, 20000, 49589):
        assert trainer.should_evaluate(off_milestone, eval_every=0, eval_steps=steps) is False


def test_trainer_normalizes_eval_steps_from_csv_and_repeated_flags():
    import train_pretrain_with_bench as trainer

    csv = trainer.normalize_eval_steps(["500,3815,11445"], schedule_total_steps=49590)
    repeated = trainer.normalize_eval_steps(["500", "3815", "11445"], schedule_total_steps=49590)
    assert csv == repeated == [500, 3815, 11445]
    with pytest.raises(ValueError):
        trainer.normalize_eval_steps(["3815,500"], schedule_total_steps=49590)
    with pytest.raises(ValueError):
        trainer.normalize_eval_steps(["500,500"], schedule_total_steps=49590)


# --------------------------------------------------------------------- checkpoints


def test_checkpoint_milestones_come_from_the_exact_plan():
    plan = json.loads((REPO / C.EXACT_RUN_PLAN_RELPATH).read_text())
    assert tuple(plan["checkpoint_milestones"]["absolute_steps"]) == C.CHECKPOINT_MILESTONES


def test_periodic_save_enabled_is_rejected():
    failures = C.validate_governed_args(governed_args(save_every=1000), stage="stage_a")
    assert any("save_every" in f for f in failures)


@pytest.mark.parametrize(
    "steps",
    [
        [*C.CHECKPOINT_MILESTONES, 40000],  # unplanned extra checkpoint
        list(C.CHECKPOINT_MILESTONES)[:-1],  # missing milestone
        list(reversed(C.CHECKPOINT_MILESTONES)),  # reordered
    ],
)
def test_unplanned_or_incomplete_save_list_is_rejected(steps):
    failures = C.validate_governed_args(governed_args(save_steps=steps), stage="stage_a")
    assert any("save_steps" in f for f in failures)


def test_repeated_save_step_flag_derivation_is_mechanical():
    flags = C.save_steps_cli_flags()
    assert len(flags) == 2 * len(C.CHECKPOINT_MILESTONES)
    assert flags[0::2] == ["--save_steps"] * len(C.CHECKPOINT_MILESTONES)
    assert [int(v) for v in flags[1::2]] == list(C.CHECKPOINT_MILESTONES)
    # The governed serialization must satisfy load_run_plan_binding's per-element int().
    assert [int(v) for v in flags[1::2]] == [int(v) for v in flags[1::2]]


def test_eval_step_flag_derivation_is_mechanical():
    flags = C.eval_steps_cli_flags()
    assert flags[0::2] == ["--eval_steps"] * len(C.EVALUATION_MILESTONES)
    assert [int(v) for v in flags[1::2]] == list(C.EVALUATION_MILESTONES)


def test_periodic_save_disabled_emits_only_planned_checkpoints():
    import train_pretrain_with_bench as trainer

    steps = list(C.CHECKPOINT_MILESTONES)
    assert trainer.should_save_checkpoint(3815, save_every=0, save_steps=steps) is True
    for off in (1, 1000, 2000, 40000, 49589):
        assert trainer.should_save_checkpoint(off, save_every=0, save_steps=steps) is False


# --------------------------------------------------------------------- model contract


def test_model_contract_and_parameter_count_match_the_real_model():
    from src.model import GPT, GPTConfig, audit_gpt_parameter_count

    cfg = GPTConfig()
    model = GPT(cfg)
    audit = audit_gpt_parameter_count(model, cfg)
    assert int(audit["actual_total"]) == C.MODEL_PARAMETER_COUNT
    for field, value in C.MODEL_CONTRACT.items():
        if field == "seq_len":
            assert cfg.max_seq_len == value
        else:
            assert getattr(cfg, field) == value


def test_wrong_model_parameter_count_is_rejected():
    from src.model import GPT, GPTConfig, audit_gpt_parameter_count

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    audit = audit_gpt_parameter_count(model, cfg)
    assert int(audit["actual_total"]) != C.MODEL_PARAMETER_COUNT


def test_realized_muon_grouping_matches_the_governed_contract():
    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    optimizer = build_optimizer(
        model,
        name="muon",
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    result = C.verify_realized_optimizer(optimizer, model)
    assert result["failures"] == []
    assert result["matches_governed_realization"] is True
    assert len(optimizer.state) == 0  # no optimizer update was performed


def test_realized_muon_contract_reports_frozen_mechanics():
    realized = C.realized_muon_contract()
    assert realized["muon_momentum"] == 0.95
    assert realized["nesterov"] is True
    assert realized["newton_schulz_steps"] == 5
    assert realized["aux_adamw_betas"] == [0.9, 0.95]
    assert realized["aux_adamw_eps"] == 1e-8
    assert realized["role_weight_decay"]["aux_adamw_no_decay"] == 0.0
    assert realized["lr_ratio_required_on_every_group"] == 1.0


# --------------------------------------------------------------------- authorization


def test_authorization_template_is_not_authorized_and_scoped():
    template = C.authorization_template()
    assert template["authorization_status"] == "NOT_AUTHORIZED"
    assert template["allowed_scope"] is None
    assert template["allowed_scope_values"] == ["STAGE_N", "STAGE_O"]


def test_stage_n_authorization_cannot_execute_stage_o():
    verdict = C.validate_authorization(
        authorization("STAGE_N"), requested_scope="STAGE_O", observed=observed()
    )
    assert verdict["authorized"] is False
    assert any("scope_mismatch" in f for f in verdict["failures"])


def test_stage_o_authorization_cannot_execute_stage_n():
    verdict = C.validate_authorization(
        authorization("STAGE_O"), requested_scope="STAGE_N", observed=observed()
    )
    assert verdict["authorized"] is False


def test_not_authorized_manifest_is_refused():
    verdict = C.validate_authorization(
        authorization("STAGE_N", authorization_status="NOT_AUTHORIZED"),
        requested_scope="STAGE_N",
        observed=observed(),
    )
    assert verdict["authorized"] is False
    assert "authorization_status_not_authorized" in verdict["failures"]


def test_missing_authorization_is_refused():
    verdict = C.validate_authorization(None, requested_scope="STAGE_N", observed=observed())
    assert verdict["authorized"] is False


def test_complete_authorization_is_accepted():
    verdict = C.validate_authorization(
        authorization("STAGE_N"), requested_scope="STAGE_N", observed=observed()
    )
    assert verdict["failures"] == []
    assert verdict["authorized"] is True


# --------------------------------------------------------------------- runtime binding


@pytest.mark.parametrize("field", ["gpu_uuid", "gpu_pci_bus_id", "gpu_name", "torch_version"])
def test_runtime_mismatch_is_rejected(field):
    manifest = authorization("STAGE_N")
    manifest["training_runtime"] = {**_runtime(), field: "DIFFERENT"}
    verdict = C.validate_authorization(manifest, requested_scope="STAGE_N", observed=observed())
    assert verdict["authorized"] is False
    assert any(f"runtime_mismatch:{field}" in f for f in verdict["failures"])


def test_runtime_binding_requires_every_field():
    incomplete = {k: v for k, v in _runtime().items() if k != "gpu_uuid"}
    failures = C.check_training_runtime_binding(incomplete)
    assert any("gpu_uuid" in f for f in failures)


def test_visible_device_count_must_be_one():
    failures = C.check_training_runtime_binding({**_runtime(), "visible_cuda_device_count": 2})
    assert "visible_cuda_device_count_must_equal_1" in failures


def test_no_materiality_exception_exists():
    assert C.MATERIALITY_EXCEPTION_IMPLEMENTED is False
    assert "must be rerun" in C.STAGE_N_TO_STAGE_O_CONTINUITY


# --------------------------------------------------------------------- launch gate


def _live_authorization(scope: str = "STAGE_N") -> dict:
    """An authorization bound to the values the gate actually observes at runtime."""
    repo = C.observed_repository()
    manifest = authorization(scope)
    manifest["repository_branch"] = repo["branch"]
    manifest["trainer_head"] = repo["head"]
    manifest["trainer_execution_bundle_sha256"] = C.trainer_execution_bundle_sha256()
    runtime = {
        **_runtime(),
        "trainer_head": repo["head"],
        "trainer_execution_bundle_sha256": C.trainer_execution_bundle_sha256(),
    }
    manifest["training_runtime"] = runtime
    return manifest


def _live_runtime() -> dict:
    repo = C.observed_repository()
    return {
        **_runtime(),
        "trainer_head": repo["head"],
        "trainer_execution_bundle_sha256": C.trainer_execution_bundle_sha256(),
    }


def _launch(stage="stage_a", scope="STAGE_N", **arg_overrides):
    return C.validate_governed_launch(
        args=governed_args(stage, **arg_overrides),
        stage=stage,
        launch_contract=C.contract_document(),
        authorization=_live_authorization(scope),
        exact_plan_sha256=EXACT_PLAN_SHA,
        pilot_owner_acceptance_sha256=ACCEPTANCE_SHA,
        observed_runtime=_live_runtime(),
        cwd=C.CANONICAL_CWD,
    )


def test_governed_launch_accepts_a_correct_stage_a_launch():
    assert _launch()["failures"] == []


def test_governed_launch_requires_canonical_cwd():
    result = C.validate_governed_launch(
        args=governed_args("stage_a"),
        stage="stage_a",
        launch_contract=C.contract_document(),
        authorization=_live_authorization("STAGE_N"),
        exact_plan_sha256=EXACT_PLAN_SHA,
        pilot_owner_acceptance_sha256=ACCEPTANCE_SHA,
        observed_runtime=_live_runtime(),
        cwd="/tmp",
    )
    assert any("canonical_cwd" in f for f in result["failures"])


def test_governed_launch_rejects_a_wrong_exact_plan_sha():
    result = C.validate_governed_launch(
        args=governed_args("stage_a"),
        stage="stage_a",
        launch_contract=C.contract_document(),
        authorization=_live_authorization("STAGE_N"),
        exact_plan_sha256="0" * 64,
        pilot_owner_acceptance_sha256=ACCEPTANCE_SHA,
        observed_runtime=_live_runtime(),
        cwd=C.CANONICAL_CWD,
    )
    assert any("exact_run_plan_sha256" in f for f in result["failures"])


def test_governed_launch_rejects_a_wrong_acceptance_sha():
    result = C.validate_governed_launch(
        args=governed_args("stage_a"),
        stage="stage_a",
        launch_contract=C.contract_document(),
        authorization=_live_authorization("STAGE_N"),
        exact_plan_sha256=EXACT_PLAN_SHA,
        pilot_owner_acceptance_sha256="0" * 64,
        observed_runtime=_live_runtime(),
        cwd=C.CANONICAL_CWD,
    )
    assert any("pilot_owner_acceptance_sha256" in f for f in result["failures"])


def test_require_governed_launch_raises_on_mismatch():
    with pytest.raises(C.LaunchContractError):
        C.require_governed_launch(
            args=governed_args("stage_a", lr=0.0003),
            stage="stage_a",
            launch_contract=C.contract_document(),
            authorization=_live_authorization("STAGE_N"),
            exact_plan_sha256=EXACT_PLAN_SHA,
            pilot_owner_acceptance_sha256=ACCEPTANCE_SHA,
            observed_runtime=_live_runtime(),
            cwd=C.CANONICAL_CWD,
        )


# --------------------------------------------------------------------- compile


class _Eager:
    def __call__(self, *a, **k):  # pragma: no cover - never invoked
        raise AssertionError("no forward is performed in these tests")


def test_compile_failure_aborts_a_governed_run(monkeypatch):
    import torch

    def boom(_module):
        raise RuntimeError("inductor exploded")

    monkeypatch.setattr(torch, "compile", boom)
    with pytest.raises(C.LaunchContractError, match="eager fallback is forbidden"):
        C.bind_compiled_callable(_Eager(), compile_requested=True)


def test_compile_returning_the_eager_module_is_rejected(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "compile", lambda module: module)
    with pytest.raises(C.LaunchContractError, match="eager fallback"):
        C.bind_compiled_callable(_Eager(), compile_requested=True)


def test_compiled_callable_is_bound_as_the_training_callable(monkeypatch):
    import torch

    sentinel = _Eager()
    monkeypatch.setattr(torch, "compile", lambda module: sentinel)
    eager = _Eager()
    evidence = C.bind_compiled_callable(eager, compile_requested=True)
    assert evidence["module"] is sentinel
    assert evidence["compiled"] is True
    assert evidence["eager_fallback_occurred"] is False


def test_eager_fallback_cannot_pass_a_compile_true_binding():
    contract = {"training": {"compile": True}}
    with pytest.raises(C.LaunchContractError):
        C.assert_compile_binding(contract, {"compiled": False, "eager_fallback_occurred": True})
    with pytest.raises(C.LaunchContractError):
        C.assert_compile_binding(
            contract,
            {
                "compiled": True,
                "eager_fallback_occurred": True,
                "compiled_callable_is_training_callable": True,
            },
        )
    C.assert_compile_binding(
        contract,
        {
            "compiled": True,
            "eager_fallback_occurred": False,
            "compiled_callable_is_training_callable": True,
        },
    )


# --------------------------------------------------------------------- run contract


def _run_contract(stage="stage_a", **overrides):
    contract = C.governed_run_contract(
        stage=stage,
        launch_contract_sha256=C.contract_sha256(),
        stage_authorization_sha256="a" * 64,
        exact_plan_sha256=EXACT_PLAN_SHA,
        pilot_owner_acceptance_sha256=ACCEPTANCE_SHA,
        trainer_head="0" * 40,
        trainer_execution_bundle_sha256="1" * 64,
        runtime=_runtime(),
    )
    contract.update(overrides)
    return contract


def test_run_contract_binds_every_governed_identity():
    contract = _run_contract()
    assert contract["launch_contract_sha256"] == C.contract_sha256()
    assert contract["exact_run_plan_sha256"] == EXACT_PLAN_SHA
    assert contract["pilot_owner_acceptance_sha256"] == ACCEPTANCE_SHA
    assert contract["seeds"] == dict(C.SEED_TUPLE)
    assert contract["stage_sampler_seed"] == C.STAGE_A_SAMPLER_SEED
    assert contract["gpu_uuid"] == _runtime()["gpu_uuid"]
    assert contract["gpu_pci_bus_id"] == _runtime()["gpu_pci_bus_id"]
    assert contract["evaluation_policy"]["periodic_eval_status"] == "DISABLED"
    assert contract["checkpoint_policy"]["periodic_save_status"] == "DISABLED"
    assert contract["stage_start_step"] == 0
    assert contract["stage_stop_step"] == 38146


def test_run_contract_is_published_before_any_optimizer_update():
    """The artifact must exist at the pre-update boundary, with no updates recorded."""
    contract = _run_contract()
    digest = C.governed_run_contract_sha256(contract)
    assert contract["completed_evaluation_milestones"] == []
    assert len(digest) == 64
    # the digest deliberately excludes the mutable completed-milestone list
    contract["completed_evaluation_milestones"] = [500]
    assert C.governed_run_contract_sha256(contract) == digest


def test_checkpoint_resume_digest_equality():
    a, b = _run_contract(), _run_contract()
    verdict = C.validate_governed_resume(a, b)
    assert verdict["compatible"] is True
    assert verdict["checkpoint_run_contract_sha256"] == verdict["current_run_contract_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("launch_contract_sha256", "0" * 64),
        ("stage_authorization_sha256", "0" * 64),
        ("exact_run_plan_sha256", "0" * 64),
        ("trainer_head", "9" * 40),
        ("trainer_execution_bundle_sha256", "0" * 64),
        ("gpu_uuid", "GPU-other"),
        ("gpu_pci_bus_id", "00000000:03:00.0"),
        ("stage", "stage_b"),
    ],
)
def test_resume_rejects_a_mismatched_governed_value(field, value):
    verdict = C.validate_governed_resume(_run_contract(), _run_contract(**{field: value}))
    assert verdict["compatible"] is False
    assert any(field in f for f in verdict["failures"])


def test_resume_rejects_a_cli_override_of_a_bound_value():
    """A CLI change that alters any governed value cannot be accepted on resume."""
    checkpoint = _run_contract()
    overridden = _run_contract()
    overridden["training"] = {**overridden["training"], "peak_lr": 0.0003}
    verdict = C.validate_governed_resume(checkpoint, overridden)
    assert verdict["compatible"] is False
    assert any("training" in f for f in verdict["failures"])


def test_resume_rejects_mismatched_seed_and_policy():
    for field, value in (
        ("seeds", {**dict(C.SEED_TUPLE), "model_init_seed": 1234}),
        ("evaluation_policy", {"mode": "periodic"}),
        ("checkpoint_policy", {"mode": "periodic"}),
    ):
        verdict = C.validate_governed_resume(_run_contract(), _run_contract(**{field: value}))
        assert verdict["compatible"] is False


# --------------------------------------------------------------------- closure


def test_governed_trainer_closure_has_zero_unbound_modules():
    closure = C.trainer_execution_closure()
    assert closure["unbound_load_bearing_module_count"] == 0
    assert closure["derived_closure_count"] >= 10
    assert len(closure["TRAINER_EXECUTION_BUNDLE_SHA256"]) == 64


def test_governed_closure_is_not_the_pilot_or_stage_p_closure():
    closure = C.trainer_execution_closure()
    assert closure["reused_pilot_execution_closure"] is False
    assert closure["reused_stage_p_plan_closure"] is False


def test_governed_closure_covers_every_load_bearing_area():
    closure = set(C.trainer_execution_closure()["derived_closure"])
    for required in (
        "src/model.py",  # model
        "src/optim.py",  # optimizer
        "pretrain/dataset_pretrain.py",  # dataset + sampler
        "src/canonical_loss.py",  # loss
        "src/canonical_schedule.py",  # schedule
        "src/special_tokens.py",  # special tokens
        "pretrain/run_plan_contract.py",  # run-plan loading
        "pretrain/production_launch_contract_v1.py",  # launch-contract validation
        "pretrain/train_pretrain_with_bench.py",  # trainer + eval + checkpointing
    ):
        assert required in closure, required


def test_closure_bundle_is_stable_across_calls():
    assert (
        C.trainer_execution_closure()["TRAINER_EXECUTION_BUNDLE_SHA256"]
        == C.trainer_execution_bundle_sha256()
    )
