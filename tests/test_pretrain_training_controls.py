from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import sys

import pytest
import torch

from pretrain.dataset_pretrain import ResumablePermutationSampler
from src.model import GPT, GPTConfig

PRETRAIN_DIR = str(Path(__file__).resolve().parents[1] / "pretrain")
_inserted_pretrain_path = PRETRAIN_DIR not in sys.path
if _inserted_pretrain_path:
    sys.path.insert(0, PRETRAIN_DIR)
try:
    train = importlib.import_module("train_pretrain")
    train_bench = importlib.import_module("train_pretrain_with_bench")
finally:
    if _inserted_pretrain_path:
        sys.path.remove(PRETRAIN_DIR)


def _schedule_spec(**overrides):
    spec = {
        "name": "wsd",
        "warmup_steps": 2,
        "schedule_total_steps": 12,
        "decay_start_step": 8,
        "decay_end_step": 12,
        "min_lr_ratio": 0.1,
    }
    spec.update(overrides)
    return spec


def _validation_args(**overrides):
    values = {
        "max_steps": 12,
        "schedule_total_steps": 12,
        "warmup_steps": 2,
        "min_lr_ratio": 0.1,
        "data_stage_start_step": 0,
        "micro_bsz": 2,
        "grad_accum": 4,
        "lr_schedule": "wsd",
        "decay_start_step": 8,
        "decay_end_step": 12,
        "mask_last_label_in_loss": False,
        "no_mask_last_label_in_loss": False,
        "lr": 1e-3,
        "num_workers": 0,
        "eos_weight_warmup_steps": 0,
        "eos_weight": 1.0,
        "log_every": 1,
        "eval_every": 2,
        "save_every": 2,
        "save_steps": [],
        "debug_every": 2,
        "resume_step": -1,
        "resume_path": "",
        "resume_full": False,
        "allow_weights_only_resume": False,
        "allow_schedule_branch": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("module", [train, train_bench])
def test_parameter_count_audit_locks_canonical_and_allows_explicit_experiments(module):
    canonical_cfg = GPTConfig()
    with torch.device("meta"):
        canonical_model = GPT(canonical_cfg)
    audit = module.audit_gpt_parameter_count(canonical_model, canonical_cfg)

    assert audit["status"] == "passed"
    assert audit["actual_total"] == 133_128_960
    assert audit["derived_expected_total"] == 133_128_960
    assert audit["canonical_expected_total"] == 133_128_960
    assert audit["canonical_parameterization"] is True
    assert audit["canonical_match"] is True

    canonical_model.register_parameter(
        "unexpected_parameter",
        torch.nn.Parameter(torch.empty(1, device="meta")),
    )
    with pytest.raises(RuntimeError, match="architecture-derived count"):
        module.audit_gpt_parameter_count(canonical_model, canonical_cfg)

    experiment_cfg = GPTConfig(
        n_layers=2,
        d_model=128,
        n_heads=4,
        d_ff=320,
    )
    with torch.device("meta"):
        experiment_model = GPT(experiment_cfg)
    experiment = module.audit_gpt_parameter_count(experiment_model, experiment_cfg)
    assert experiment["actual_total"] == experiment["derived_expected_total"]
    assert experiment["canonical_parameterization"] is False
    assert experiment["canonical_match"] is False


@pytest.mark.parametrize("module", [train, train_bench])
def test_wsd_uses_one_absolute_continuous_timeline_across_stage_boundary(module):
    base_lr = 1e-3
    values = [
        module.lr_schedule(
            step,
            2,
            base_lr,
            schedule="wsd",
            schedule_total_steps=12,
            decay_start_step=8,
            decay_end_step=12,
            min_lr_ratio=0.1,
        )
        for step in range(14)
    ]

    assert values[:2] == pytest.approx([base_lr / 2, base_lr])
    assert values[2:9] == pytest.approx([base_lr] * 7)
    assert base_lr * 0.1 < values[9] < base_lr
    assert values[9:13] == sorted(values[9:13], reverse=True)
    assert values[12] == values[13] == pytest.approx(base_lr * 0.1)


@pytest.mark.parametrize("module", [train, train_bench])
def test_schedule_branch_accepts_future_only_change_and_rejects_changed_history(module):
    saved = _schedule_spec()
    future_only = _schedule_spec(decay_start_step=10)
    module.validate_schedule_branch(
        saved,
        future_only,
        checkpoint_step=8,
        base_lr=1e-3,
    )

    with pytest.raises(RuntimeError, match="histories diverge"):
        module.validate_schedule_branch(
            saved,
            _schedule_spec(decay_start_step=6),
            checkpoint_step=8,
            base_lr=1e-3,
        )


@pytest.mark.parametrize("module", [train, train_bench])
def test_save_steps_accept_repeatable_csv_and_full_global_horizon(module):
    args = _validation_args(
        max_steps=8,
        schedule_total_steps=12,
        save_steps=["1,3", "6,8,10,12"],
    )

    module.validate_training_args(args)

    assert args.save_steps == [1, 3, 6, 8, 10, 12]
    # Stage-A invocations may carry later Stage-B milestones unchanged.
    assert args.save_steps[-1] > args.max_steps


@pytest.mark.parametrize("module", [train, train_bench])
@pytest.mark.parametrize(
    ("save_steps", "message"),
    [
        (["0"], "positive absolute"),
        (["1,1"], "strictly increasing and unique"),
        (["3,2"], "strictly increasing and unique"),
        (["1,,2"], "empty comma-separated"),
        (["1,nope"], "only integer"),
        (["1,13"], "cannot exceed --schedule_total_steps"),
    ],
)
def test_save_steps_reject_invalid_contract(module, save_steps, message):
    args = _validation_args(save_steps=save_steps)

    with pytest.raises(ValueError, match=message):
        module.validate_training_args(args)


@pytest.mark.parametrize("module", [train, train_bench])
def test_checkpoint_trigger_is_union_of_periodic_and_explicit_steps(module):
    explicit = [3, 8, 11]

    assert module.should_save_checkpoint(3, save_every=5, save_steps=explicit)
    assert module.should_save_checkpoint(5, save_every=5, save_steps=explicit)
    # A step present in both policies still produces one boolean save event.
    assert module.should_save_checkpoint(10, save_every=5, save_steps=[10])
    assert not module.should_save_checkpoint(4, save_every=5, save_steps=explicit)

    assert not module.should_retain_step_checkpoint(
        5, save_steps=explicit, invocation_final_step=12
    )
    assert module.should_retain_step_checkpoint(3, save_steps=explicit, invocation_final_step=12)
    assert module.should_retain_step_checkpoint(12, save_steps=explicit, invocation_final_step=12)


@pytest.mark.parametrize("module", [train, train_bench])
def test_checkpoint_writer_retains_only_named_or_final_steps(module, tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    writes = []

    def capture_write(payload, path):
        writes.append((payload, Path(path)))

    monkeypatch.setattr(module, "_atomic_torch_save", capture_write)
    kwargs = {
        "out_dir": tmp_path,
        "global_step": 7,
        "local_step": 2,
        "model": model,
        "optim": optim,
        "scaler": None,
        "model_config": {"vocab_size": 4},
        "train_args": {"save_every": 5},
        "run_contract": {"checkpointing": {}},
        "position_stats": {"serialized_positions": 1},
        "sampler_state": {"committed_position": 1},
        "data_contract": {"fingerprint": "fixture"},
    }

    module.save_ckpt(**kwargs, retain_step=False)
    assert [path.name for _, path in writes] == ["latest.pt"]
    latest_payload = writes[0][0]
    assert latest_payload["checkpoint_retention"] == {"retain_step": False}
    assert {"model", "optim", "rng_state", "data_sampler", "data_contract"} <= set(latest_payload)

    writes.clear()
    module.save_ckpt(**kwargs, retain_step=True)
    assert [path.name for _, path in writes] == ["step_000007.pt", "latest.pt"]
    assert all(payload["checkpoint_retention"] == {"retain_step": True} for payload, _ in writes)


def test_two_trainers_have_identical_save_step_controls():
    raw = ["1,3", "6,8,10,12"]
    left = train.normalize_save_steps(raw, schedule_total_steps=12)
    right = train_bench.normalize_save_steps(raw, schedule_total_steps=12)
    assert left == right == [1, 3, 6, 8, 10, 12]

    for step in range(1, 14):
        assert train.should_save_checkpoint(
            step, save_every=5, save_steps=left
        ) == train_bench.should_save_checkpoint(step, save_every=5, save_steps=right)


@pytest.mark.parametrize("module", [train, train_bench])
def test_stage_b_and_schedule_branch_require_full_state_resume(module):
    stage_b_weights_only = _validation_args(
        data_stage_start_step=8,
        resume_path="stage_a.pt",
        resume_full=False,
        allow_weights_only_resume=True,
    )
    with pytest.raises(ValueError, match="requires --resume_full"):
        module.validate_training_args(stage_b_weights_only)

    schedule_branch_weights_only = _validation_args(
        resume_path="stage_a.pt",
        resume_full=False,
        allow_weights_only_resume=True,
        allow_schedule_branch=True,
    )
    with pytest.raises(ValueError, match="allow_schedule_branch requires"):
        module.validate_training_args(schedule_branch_weights_only)


@pytest.mark.parametrize("module", [train, train_bench])
def test_same_stage_resume_restores_verified_committed_sampler_suffix(module):
    dataset = range(11)
    saved = ResumablePermutationSampler(dataset, seed=7, num_samples=16)
    saved.commit(8)
    current = ResumablePermutationSampler(
        dataset,
        seed=7,
        start_position=8,
        num_samples=8,
    )
    contract = {
        "fingerprint": "fixture",
        "dataset_length": 11,
        "sampling_mode": "deterministic",
        "sampler_seed": 7,
        "data_stage_start_step": 0,
        "samples_per_optimizer_step": 4,
    }

    module.validate_data_resume_state(
        saved_data_contract=contract,
        current_data_contract=dict(contract),
        saved_sampler_state=saved.state_dict(),
        current_sampler=current,
        global_step=2,
        data_stage_start_step=0,
        strict=True,
    )

    assert current.state_dict() == saved.state_dict()
    assert list(current) == list(saved)


@pytest.mark.parametrize("module", [train, train_bench])
def test_same_stage_resume_rejects_sampler_gap_or_replay(module):
    dataset = range(11)
    current = ResumablePermutationSampler(
        dataset,
        seed=7,
        start_position=8,
        num_samples=8,
    )
    saved_state = current.state_dict()
    saved_state["committed_position"] = 4
    contract = {
        "fingerprint": "fixture",
        "dataset_length": 11,
        "sampling_mode": "deterministic",
        "sampler_seed": 7,
        "data_stage_start_step": 0,
        "samples_per_optimizer_step": 4,
    }

    with pytest.raises(RuntimeError, match="committed_position"):
        module.validate_data_resume_state(
            saved_data_contract=contract,
            current_data_contract=dict(contract),
            saved_sampler_state=saved_state,
            current_sampler=current,
            global_step=2,
            data_stage_start_step=0,
            strict=True,
        )


@pytest.mark.parametrize("module", [train, train_bench])
def test_new_stage_boundary_intentionally_starts_new_sampler_at_zero(module):
    current = ResumablePermutationSampler(range(13), seed=9, num_samples=8)
    completed_previous = ResumablePermutationSampler(range(7), seed=1, num_samples=20)
    completed_previous.commit(20)

    module.validate_data_resume_state(
        saved_data_contract={
            "fingerprint": "stage-a",
            "data_stage_start_step": 0,
            "samples_per_optimizer_step": 2,
        },
        current_data_contract={
            "fingerprint": "stage-b",
            "data_stage_start_step": 10,
            "samples_per_optimizer_step": 2,
        },
        saved_sampler_state=completed_previous.state_dict(),
        current_sampler=current,
        global_step=10,
        data_stage_start_step=10,
        strict=True,
    )

    assert current.position == 0
    assert len(current) == 8


@pytest.mark.parametrize("module", [train, train_bench])
def test_new_stage_boundary_rejects_incomplete_previous_stage(module):
    current = ResumablePermutationSampler(range(13), seed=9, num_samples=8)
    incomplete = ResumablePermutationSampler(range(7), seed=1, num_samples=20)
    incomplete.commit(18)

    with pytest.raises(RuntimeError, match="previous-stage sampler is incomplete"):
        module.validate_data_resume_state(
            saved_data_contract={
                "fingerprint": "stage-a",
                "data_stage_start_step": 0,
                "samples_per_optimizer_step": 2,
            },
            current_data_contract={
                "fingerprint": "stage-b",
                "data_stage_start_step": 10,
                "samples_per_optimizer_step": 2,
            },
            saved_sampler_state=incomplete.state_dict(),
            current_sampler=current,
            global_step=10,
            data_stage_start_step=10,
            strict=True,
        )


def test_two_trainers_have_identical_core_training_control_results():
    for step in range(16):
        left = train.lr_schedule(
            step,
            2,
            1e-3,
            schedule="wsd",
            schedule_total_steps=12,
            decay_start_step=8,
            decay_end_step=12,
            min_lr_ratio=0.1,
        )
        right = train_bench.lr_schedule(
            step,
            2,
            1e-3,
            schedule="wsd",
            schedule_total_steps=12,
            decay_start_step=8,
            decay_end_step=12,
            min_lr_ratio=0.1,
        )
        assert math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
