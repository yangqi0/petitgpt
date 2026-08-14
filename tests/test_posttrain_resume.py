"""Exact-resume contracts for SFT, DPO, and GRPO."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import distill.train_distill as distill_train
from dpo.dpo import (
    build_model_from_ckpt as build_dpo_model_from_ckpt,
    validate_dpo_args,
)
from grpo.grpo import build_model_from_ckpt as build_grpo_model_from_ckpt
from sft.train_sft import validate_sft_args
from src.model import GPT
from src.posttrain_resume import (
    DeterministicEpochBatchSampler,
    build_resume_contract_base,
    capture_rng_state,
    make_loader_generator,
    require_resume_step,
    restore_rng_state,
    restore_training_state,
    resume_contract_for_step,
    validate_resume_contract,
)


def _collect_batches(*, start_batch: int, count: int) -> list[list[int]]:
    sampler = DeterministicEpochBatchSampler(
        dataset_size=11,
        batch_size=3,
        seed=9182,
        start_batch=start_batch,
        drop_last=True,
    )
    batches: list[list[int]] = []
    iterator = iter(sampler)
    while len(batches) < count:
        try:
            batches.append(next(iterator))
        except StopIteration:
            iterator = iter(sampler)
    return batches


def test_deterministic_batch_sampler_reconstructs_mid_epoch_cursor():
    full = _collect_batches(start_batch=0, count=17)
    for consumed in (1, 3, 4, 7, 12):
        resumed = _collect_batches(
            start_batch=consumed,
            count=len(full) - consumed,
        )
        assert resumed == full[consumed:]


def test_loader_generator_does_not_advance_global_torch_rng():
    before = torch.get_rng_state().clone()
    first = torch.rand(3, generator=make_loader_generator(123, 7))
    second = torch.rand(3, generator=make_loader_generator(123, 7))
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(first, second)


def test_capture_and_restore_all_cpu_rng_streams():
    original = capture_rng_state()
    try:
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        state = capture_rng_state()
        expected_python = [random.random() for _ in range(3)]
        expected_numpy = np.random.random(3)
        expected_torch = torch.rand(3)

        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restore_rng_state(state)

        assert [random.random() for _ in range(3)] == expected_python
        assert np.array_equal(np.random.random(3), expected_numpy)
        assert torch.equal(torch.rand(3), expected_torch)
    finally:
        restore_rng_state(original)


def test_resume_contract_binds_args_inputs_and_consumed_batches(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"row": 1}\n', encoding="utf-8")
    args = {
        "lr": 1e-5,
        "out_dir": "first-output",
        "resume": "old.pt",
    }
    base = build_resume_contract_base(
        stage="sft",
        args=args,
        input_paths={"train": train},
        dataset_size=17,
        batch_size=2,
        batches_per_step=4,
        seed=123,
    )
    contract = resume_contract_for_step(base, 9)
    assert contract["data_order"]["consumed_batches"] == 36
    assert contract["runtime"]["torch"] == str(torch.__version__)

    runtime_only_changed = build_resume_contract_base(
        stage="sft",
        args={**args, "out_dir": "second-output", "resume": "new.pt"},
        input_paths={"train": train},
        dataset_size=17,
        batch_size=2,
        batches_per_step=4,
        seed=123,
    )
    assert runtime_only_changed == base

    checkpoint = {"resume_contract": contract}
    validate_resume_contract(
        checkpoint,
        resume_contract_for_step(runtime_only_changed, 9),
        weights_only_hint="--init_from_pretrain",
    )

    train.write_text('{"row": 2}\n', encoding="utf-8")
    changed_input = build_resume_contract_base(
        stage="sft",
        args=args,
        input_paths={"train": train},
        dataset_size=17,
        batch_size=2,
        batches_per_step=4,
        seed=123,
    )
    with pytest.raises(RuntimeError, match="exact resume contract mismatch"):
        validate_resume_contract(
            checkpoint,
            resume_contract_for_step(changed_input, 9),
            weights_only_hint="--init_from_pretrain",
        )


def _minimal_resume_checkpoint() -> dict:
    return {
        "step": 3,
        "kind": "sft",
        "resume_contract": {"schema_version": 1},
        "optimizer": {},
        "scaler": None,
        "rng_state": {},
        "loop_state": {},
        "aux_rng_state": {},
    }


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("optimizer", "optimizer state"),
        ("scaler", "scaler state"),
        ("rng_state", "RNG state"),
        ("resume_contract", "weights-only run"),
        ("loop_state", "loop state"),
        ("aux_rng_state", "auxiliary sampling RNG state"),
    ],
)
def test_require_resume_step_rejects_partial_checkpoint(missing, message):
    checkpoint = _minimal_resume_checkpoint()
    checkpoint.pop(missing)
    with pytest.raises(RuntimeError, match=message):
        require_resume_step(
            checkpoint,
            stage="sft",
            weights_only_hint="--init_from_pretrain",
        )


def test_require_resume_step_rejects_bad_step_and_stage():
    checkpoint = _minimal_resume_checkpoint()
    checkpoint["step"] = "3"
    with pytest.raises(RuntimeError, match="invalid or missing step"):
        require_resume_step(
            checkpoint,
            stage="sft",
            weights_only_hint="--init_from_pretrain",
        )

    checkpoint = _minimal_resume_checkpoint()
    with pytest.raises(RuntimeError, match="requires a 'dpo' checkpoint"):
        require_resume_step(
            checkpoint,
            stage="dpo",
            weights_only_hint="--init_ckpt",
        )


def test_distill_resume_accepts_only_distill_kind():
    checkpoint = _minimal_resume_checkpoint()
    checkpoint["kind"] = "distill"
    assert (
        require_resume_step(
            checkpoint,
            stage="distill",
            weights_only_hint="--init_from_pretrain",
        )
        == 3
    )
    with pytest.raises(RuntimeError, match="requires a 'sft' checkpoint"):
        require_resume_step(
            checkpoint,
            stage="sft",
            weights_only_hint="--init_from_pretrain",
        )


def test_distill_wrapper_routes_explicit_stage_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(
        distill_train, "_sft_main", lambda argv, *, stage_kind: calls.append((argv, stage_kind))
    )
    distill_train.main(["--sentinel"])
    assert calls == [(["--sentinel"], "distill")]


def _valid_supervised_controls(**updates):
    values = {
        "max_steps": 10,
        "lr": 1e-5,
        "warmup_steps": 2,
        "micro_bsz": 1,
        "grad_accum": 1,
        "eval_batches": 1,
        "sample_max_new_tokens": 1,
        "num_workers": 0,
        "eval_every": 0,
        "save_every": 0,
        "sample_every": 0,
        "sample_in_domain_n": 0,
        "seq_len": 8,
        "resume": "",
        "sample_only_ckpt": "",
    }
    values.update(updates)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("validator", [validate_sft_args, validate_dpo_args])
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"micro_bsz": 0}, "--micro_bsz"),
        ({"grad_accum": 0}, "--grad_accum"),
        ({"lr": 0.0}, "--lr"),
        ({"warmup_steps": 11}, "--warmup_steps"),
        ({"num_workers": -1}, "--num_workers"),
        ({"save_every": -1}, "--save_every"),
    ],
)
def test_supervised_posttrain_control_gates(validator, updates, message):
    with pytest.raises(ValueError, match=message):
        validator(_valid_supervised_controls(**updates))


@pytest.mark.parametrize("validator", [validate_sft_args, validate_dpo_args])
def test_supervised_posttrain_allows_disabled_cadences(validator):
    validator(_valid_supervised_controls())


def test_sft_rejects_conflicting_resume_and_sample_only():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_sft_args(
            _valid_supervised_controls(resume="resume.pt", sample_only_ckpt="sample.pt")
        )


def _toy_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(1, 8),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.35),
        torch.nn.Linear(8, 1),
    )


def _train_toy_steps(model, optimizer, *, start_step: int, end_step: int) -> None:
    batches_per_step = 2
    sampler = DeterministicEpochBatchSampler(
        dataset_size=13,
        batch_size=2,
        seed=771,
        start_batch=start_step * batches_per_step,
        drop_last=True,
    )
    iterator = iter(sampler)
    model.train()
    for _step in range(start_step, end_step):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(batches_per_step):
            try:
                indices = next(iterator)
            except StopIteration:
                iterator = iter(sampler)
                indices = next(iterator)
            x = torch.tensor(indices, dtype=torch.float32).unsqueeze(1) / 13.0
            y = x.square() + 0.25
            loss = torch.nn.functional.mse_loss(model(x), y) / batches_per_step
            loss.backward()
        optimizer.step()


def test_exact_resume_matches_uninterrupted_dropout_training():
    original = capture_rng_state()
    try:
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        model = _toy_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _train_toy_steps(model, optimizer, start_step=0, end_step=4)
        checkpoint = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scaler": None,
            "rng_state": capture_rng_state(),
        }

        _train_toy_steps(model, optimizer, start_step=4, end_step=9)
        expected = copy.deepcopy(model.state_dict())

        resumed_model = _toy_model()
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
        restore_training_state(
            checkpoint,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scaler=None,
            use_fp16=False,
        )
        restore_rng_state(checkpoint["rng_state"])
        _train_toy_steps(resumed_model, resumed_optimizer, start_step=4, end_step=9)

        for name, tensor in resumed_model.state_dict().items():
            assert torch.equal(tensor, expected[name]), name
    finally:
        restore_rng_state(original)


def test_incompatible_optimizer_state_fails_immediately():
    model = _toy_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": {"state": {}, "param_groups": []},
        "scaler": None,
    }
    with pytest.raises(RuntimeError, match="optimizer state is incompatible"):
        restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=None,
            use_fp16=False,
        )


@pytest.mark.parametrize(
    "builder",
    [build_dpo_model_from_ckpt, build_grpo_model_from_ckpt],
)
def test_posttrain_initialization_rejects_partial_model_state(builder, tiny_cfg):
    model = GPT(tiny_cfg)
    state = model.state_dict()
    state.pop(next(iter(state)))
    checkpoint = {
        "cfg": asdict(tiny_cfg),
        "model": state,
    }
    with pytest.raises(RuntimeError, match="Missing key"):
        builder(
            checkpoint,
            vocab_size=tiny_cfg.vocab_size,
            seq_len=tiny_cfg.max_seq_len,
            device="cpu",
        )


def test_posttrain_entrypoints_have_no_legacy_resume_fallback():
    root = Path(__file__).resolve().parents[1]
    for relative in ("sft/train_sft.py", "dpo/dpo.py", "grpo/grpo.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "starting with fresh optimizer" not in source
        assert "if args.resume and os.path.exists" not in source
        assert "shuffle=True" not in source
        assert "restore_training_state(" in source
        assert "restore_rng_state(" in source
        assert "DeterministicEpochBatchSampler(" in source
        assert '"resume_contract": resume_contract_for_step(' in source
        assert '"rng_state": capture_rng_state()' in source
        assert "strict=False" not in source
        assert "last_saved_step == checkpoint_step" in source
        assert source.count("save_training_checkpoint(step)") == 2

    sft_source = (root / "sft/train_sft.py").read_text(encoding="utf-8")
    assert '"stage_kind": stage_kind' in sft_source
    assert '"kind": stage_kind' in sft_source
    assert "stage=stage_kind" in sft_source
