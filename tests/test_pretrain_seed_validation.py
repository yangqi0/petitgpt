from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

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


MODULES = [train, train_bench]
REQUIRED_ARGS = [
    "--train_dir",
    "train",
    "--val_dir",
    "val",
    "--out_dir",
    "out",
    "--samples_dir",
    "samples",
    "--tokenizer_path",
    "tokenizer.json",
]


def _parse(module, monkeypatch: pytest.MonkeyPatch, *extra: str):
    monkeypatch.setattr(sys, "argv", ["trainer", *REQUIRED_ARGS, *extra])
    return module.parse_args()


@pytest.mark.parametrize("module", MODULES)
def test_data_and_validation_seeds_are_independent_of_model_seed(module, monkeypatch):
    args = _parse(module, monkeypatch, "--seed", "99", "--micro_bsz", "7")

    assert args.seed == 99
    assert args.sampler_seed == 1234
    assert args.val_seed == 1234
    assert args.val_samples == 200
    assert args.val_samples_per_source == 80

    selected = module.resolve_validation_sample_count(101, args.val_samples)
    assert selected == 101
    val_ids = list(module.FixedSubsetSampler(range(101), num_samples=17, seed=args.val_seed))

    other = _parse(module, monkeypatch, "--seed", "7", "--micro_bsz", "1")
    other_ids = list(module.FixedSubsetSampler(range(101), num_samples=17, seed=other.val_seed))
    assert other_ids == val_ids


@pytest.mark.parametrize("module", MODULES)
def test_zero_validation_count_means_full_stream(module, monkeypatch):
    args = _parse(
        module,
        monkeypatch,
        "--val_samples",
        "0",
        "--val_samples_per_source",
        "0",
    )
    module.validate_training_args(args)

    assert module.resolve_validation_sample_count(37, args.val_samples) == 37
    assert module.resolve_validation_sample_count(11, args.val_samples_per_source) == 11


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--seed", "must be non-negative"),
        ("--sampler_seed", "must be non-negative"),
        ("--val_seed", "must be non-negative"),
        ("--val_samples", "must be non-negative"),
        ("--val_samples_per_source", "must be non-negative"),
    ],
)
def test_negative_seed_and_validation_controls_fail(module, monkeypatch, flag, message):
    args = _parse(module, monkeypatch, flag, "-1")
    with pytest.raises(ValueError, match=message):
        module.validate_training_args(args)


@pytest.mark.parametrize("module", MODULES)
def test_canonical_vocab_and_prompt_bos_are_fail_fast(module, monkeypatch):
    wrong_vocab = _parse(module, monkeypatch, "--vocab_size", "31999")
    with pytest.raises(ValueError, match="vocab_size must be exactly 32000"):
        module.validate_training_args(wrong_vocab)

    missing_bos = _parse(module, monkeypatch, "--no-add_bos_to_prompts")
    with pytest.raises(ValueError, match="canonical BOS prompt contract"):
        module.validate_training_args(missing_bos)


@pytest.mark.parametrize("module", MODULES)
def test_strict_run_contract_records_all_seed_and_validation_controls(module, monkeypatch):
    args = _parse(
        module,
        monkeypatch,
        "--seed",
        "7",
        "--sampler_seed",
        "11",
        "--val_seed",
        "13",
        "--val_samples",
        "17",
        "--val_samples_per_source",
        "19",
    )
    module.validate_training_args(args)
    contract = module.build_run_contract(
        args,
        {"vocab_size": 32_000},
        "tokenizer-sha256",
        parameter_count={"status": "passed", "actual_total": 133_128_960},
    )

    assert contract["model_seed"] == 7
    assert contract["sampler_seed"] == 11
    assert contract["val_seed"] == 13
    assert contract["validation_selection"] == {
        "combined_samples": 17,
        "samples_per_source": 19,
    }
    assert contract["rng_consumers"]["add_bos_to_prompts"] is True
    assert contract["parameter_count"] == {
        "status": "passed",
        "actual_total": 133_128_960,
    }


@pytest.mark.parametrize("module", MODULES)
def test_strict_run_contract_records_checkpoint_policy(module, monkeypatch):
    args = _parse(
        module,
        monkeypatch,
        "--save_every",
        "7",
        "--save_steps",
        "1,3",
        "--save_steps",
        "8,12",
    )
    module.validate_training_args(args)
    contract = module.build_run_contract(
        args,
        {"vocab_size": 32_000},
        "tokenizer-sha256",
        run_plan_binding={"plan_sha256": "a" * 64, "stage": "stage_a"},
    )

    assert contract["schema_version"] == 3
    assert contract["run_plan"] == {"plan_sha256": "a" * 64, "stage": "stage_a"}
    assert contract["checkpointing"] == {
        "save_every": 7,
        "save_steps": [1, 3, 8, 12],
        "retention_policy": {
            "periodic": "atomic_latest_only",
            "explicit_save_step": "atomic_named_step_then_latest",
            "invocation_final": "atomic_named_step_then_latest",
        },
    }

    changed = {
        **contract,
        "checkpointing": {"save_every": 7, "save_steps": [1, 3, 8, 13]},
    }
    with pytest.raises(RuntimeError, match="checkpointing"):
        module.validate_resume_contract(
            {"run_contract": contract},
            changed,
            strict=True,
            checkpoint_step=3,
            allow_schedule_branch=False,
        )


@pytest.mark.parametrize("module", MODULES)
def test_trainer_resume_contract_dispatches_exact_stage_handoff(module, tmp_path):
    common = {
        "schema_version": 1,
        "status": "validated",
        "plan_path": str(tmp_path / "run_plan.json"),
        "plan_sha256": "1" * 64,
        "plan_schema_version": 3,
        "plan_type": "deterministic_no_replacement_stage_a_b",
        "schedule_total_steps": 8,
        "tokenizer_sha256": "2" * 64,
        "reference_release_manifest_sha256": "3" * 64,
        "checkpoint_milestones_schema_version": 1,
        "checkpoint_milestone_steps": [2, 5, 8],
        "checkpoint_milestones_sha256": "4" * 64,
    }
    stage_a = {
        **common,
        "stage": "stage_a",
        "stage_start_step": 0,
        "stage_stop_step": 5,
    }
    stage_b = {
        **common,
        "stage": "stage_b",
        "stage_start_step": 5,
        "stage_stop_step": 8,
    }

    module.validate_resume_contract(
        {"run_contract": {"run_plan": stage_a}},
        {"run_plan": stage_b},
        strict=True,
        checkpoint_step=5,
        allow_schedule_branch=False,
    )

    changed = {**stage_b, "plan_sha256": "5" * 64}
    with pytest.raises(RuntimeError, match="run-plan identity changed"):
        module.validate_resume_contract(
            {"run_contract": {"run_plan": stage_a}},
            {"run_plan": changed},
            strict=True,
            checkpoint_step=5,
            allow_schedule_branch=False,
        )


@pytest.mark.parametrize("module", MODULES)
def test_cli_save_steps_preserves_repeatable_argument_order(module, monkeypatch):
    args = _parse(
        module,
        monkeypatch,
        "--save_steps",
        "1,3",
        "--save_steps",
        "8,12",
    )

    assert args.save_steps == ["1,3", "8,12"]


class _FixedLogitModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*input_ids.shape, 4, dtype=torch.float32)
        logits[..., 3] = torch.where(input_ids == 1, 2.0, 0.0)
        return logits


@pytest.mark.parametrize("module", MODULES)
def test_evaluate_uses_global_effective_target_weights_including_eos(module):
    model = _FixedLogitModel()
    batches = [
        (
            torch.tensor([[1, 1]], dtype=torch.int32),
            torch.tensor([[3, 1]], dtype=torch.int32),
            torch.tensor([[1.0, 0.0]]),
        ),
        (
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            torch.tensor([[0, 0, 0]], dtype=torch.int32),
            torch.ones(1, 3),
        ),
    ]

    loss = module.evaluate(
        model,
        batches,
        torch.device("cpu"),
        "fp32",
        eos_id=3,
        eos_weight=4.0,
    )

    eos_logits = torch.tensor([[0.0, 0.0, 0.0, 2.0]])
    eos_ce = float(F.cross_entropy(eos_logits, torch.tensor([3])).item())
    ordinary_ce = math.log(4.0)
    expected = (4.0 * eos_ce + 3.0 * ordinary_ce) / 7.0
    batch_mean_of_means = (eos_ce + ordinary_ce) / 2.0

    assert loss == pytest.approx(expected, rel=1e-7, abs=1e-7)
    assert loss != pytest.approx(batch_mean_of_means, rel=1e-4, abs=1e-4)
    assert model.training is True
