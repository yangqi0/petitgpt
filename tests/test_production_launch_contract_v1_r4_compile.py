"""R4 compile lifecycle regressions; all executions are bounded and take no optimizer step."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

import train_pretrain_with_bench as trainer  # noqa: E402

from pretrain import production_launch_contract_v1 as C  # noqa: E402

from .test_production_launch_contract_v1_real_path import (  # noqa: E402
    _production_compile_facts,
)


class OptimizedModule:
    def __call__(self, *args, **kwargs):
        return None


OptimizedModule.__module__ = "torch._dynamo.eval_frame"


def _realized_evidence() -> dict:
    module = OptimizedModule()
    forward = C.ObservedForward(module, compiled_object=module)
    forward()
    draft = C.compile_realization_evidence(
        module,
        forward,
        requested=True,
        cache_dir=None,
        expected_forward_invocations=1,
        counters={"stats": {"unique_graphs": 1}},
        finalize=False,
    )
    observed_failures = draft.pop("failures")
    draft.update(_production_compile_facts())
    draft["inductor_cache_dir"] = draft["isolated_cache"]["cache_dir"]
    return C.finalize_compile_evidence(draft, additional_failures=observed_failures)


@pytest.mark.parametrize(
    ("field", "value"),
    (("failures", ["anything"]), ("verdict", "FAIL")),
)
def test_compile_self_hash_covers_failures_and_verdict(field, value):
    evidence = _realized_evidence()
    original_sha = evidence["compile_evidence_sha256"]
    tampered = copy.deepcopy(evidence)
    tampered[field] = value

    assert C.seal_compile_evidence(tampered) != original_sha
    failures = C.verify_compile_evidence_document(tampered)
    assert "compile_evidence_sha256_does_not_match_its_own_document" in failures
    with pytest.raises(C.LaunchContractError):
        C.require_compile_realized(tampered)


def test_compile_acceptance_requires_empty_failures_even_when_resealed():
    evidence = _realized_evidence()
    evidence.pop("compile_evidence_sha256")
    evidence["failures"] = ["observed_failure"]
    evidence["verdict"] = "FAIL"
    evidence["compile_realized"] = False
    evidence["compile_evidence_sha256"] = C.seal_compile_evidence(evidence)

    with pytest.raises(C.LaunchContractError, match="not lazily realized"):
        C.require_compile_realized(evidence)


def test_finalizer_refuses_to_reseal_a_published_document():
    with pytest.raises(C.LaunchContractError, match="unsealed observation"):
        C.finalize_compile_evidence(_realized_evidence())


@pytest.mark.parametrize("field", C.COMPILE_PRODUCTION_REQUIRED_FACTS)
def test_compile_verifier_rejects_each_removed_production_fact_even_when_resealed(field):
    evidence = _realized_evidence()
    del evidence[field]
    evidence["compile_evidence_sha256"] = C.seal_compile_evidence(evidence)

    failures = C.verify_compile_evidence_document(evidence)
    assert f"compile_evidence_missing_subfact:{field}" in failures
    with pytest.raises(C.LaunchContractError):
        C.require_compile_realized(evidence)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("probe_geometry", "micro_bsz"), C.MICRO_BSZ - 1),
        (("probe_signature", "grad_enabled"), False),
        (("probe_signature", "autocast_enabled"), False),
        (("probe_signature", "autocast_dtype"), "torch.float16"),
        (("production_shape_probe",), False),
        (("isolated_cache", "isolated"), False),
        (("cache_was_empty_before_realization",), False),
        (("precompile_causal_diagnostic", "executed"), False),
        (("precompile_causal_diagnostic", "max_abs_difference"), 1.0),
        (("fail_closed_stance", "suppress_errors"), True),
        (("fail_closed_stance", "fail_on_recompile_limit_hit"), False),
        (("post_realization_stance", "armed"), False),
    ),
)
def test_compile_verifier_rederives_each_required_fact_after_reseal(path, value):
    evidence = _realized_evidence()
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    evidence["compile_evidence_sha256"] = C.seal_compile_evidence(evidence)

    failures = C.verify_compile_evidence_document(evidence)
    assert failures
    assert "compile_evidence_sha256_does_not_match_its_own_document" not in failures
    with pytest.raises(C.LaunchContractError):
        C.require_compile_realized(evidence)


@pytest.mark.parametrize("initial_training", (True, False))
def test_causal_diagnostic_uses_no_grad_and_restores_exact_mode(initial_training):
    class CausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[bool, bool]] = []

        def forward(self, tokens):
            self.calls.append((bool(self.training), bool(torch.is_grad_enabled())))
            return torch.nn.functional.one_hot(tokens, num_classes=32).float()

    model = CausalModel()
    model.train(initial_training)
    tokens = (torch.arange(137, dtype=torch.long) % 32).unsqueeze(0)

    difference = trainer.causal_leak_check(
        model,
        tokens,
        torch.device("cpu"),
        vocab_size=32,
        check_pos=128,
        delta_pos=8,
    )

    assert difference == 0.0
    assert model.training is initial_training
    assert model.calls == [(False, False), (False, False)]


def test_noncausal_diagnostic_fails_closed_and_restores_training_mode():
    class NonCausalModel(torch.nn.Module):
        def forward(self, tokens):
            future = tokens[:, -1:].expand_as(tokens)
            return torch.nn.functional.one_hot((tokens + future) % 32, num_classes=32).float()

    model = NonCausalModel().train()
    tokens = (torch.arange(C.CAUSAL_DIAGNOSTIC_SEQ_LEN, dtype=torch.long) % 32).unsqueeze(0)

    with pytest.raises(RuntimeError, match="causal leak diagnostic failed"):
        trainer.causal_leak_check(
            model,
            tokens,
            torch.device("cpu"),
            vocab_size=32,
            check_pos=C.CAUSAL_DIAGNOSTIC_CHECK_POS,
            delta_pos=C.CAUSAL_DIAGNOSTIC_DELTA_POS,
            max_abs_tolerance=C.CAUSAL_LEAK_MAX_ABS_TOLERANCE,
        )
    assert model.training is True


def test_governed_inference_requires_shared_eager_base_and_sampling_restores_mode():
    base = torch.nn.Linear(2, 2).train()
    wrapper = type("Wrapper", (), {"_orig_mod": base})()
    assert (
        trainer.select_inference_model(
            wrapper,
            base,
            governed=True,
            compile_enabled=True,
        )
        is base
    )
    with pytest.raises(RuntimeError, match="exact eager base"):
        trainer.select_inference_model(
            wrapper,
            torch.nn.Linear(2, 2),
            governed=True,
            compile_enabled=True,
        )

    with trainer.preserve_model_training_mode(base):
        base.eval()
    assert base.training is True


def test_compile_claiming_save_rejects_placeholder_before_writing(tmp_path, monkeypatch):
    evidence = _realized_evidence()
    contract = {
        "kind": C.GOVERNED_CHECKPOINT_KIND,
        "training": {"compile": True},
        "compile_evidence_sha256": evidence["compile_evidence_sha256"],
    }
    sampler = type(
        "LiveSampler",
        (),
        {
            "seed": C.STAGE_A_SAMPLER_SEED,
            "range_start_position": 0,
            "end_position": 128,
            "committed_position": 0,
        },
    )()
    rng_state = trainer.capture_rng_state()
    if rng_state["torch_cuda"] is None:
        rng_state["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    placeholder_state = C.build_checkpoint_state(
        stage="stage_a",
        sampler=sampler,
        global_step=0,
        completed_evaluation_milestones=[],
        completed_checkpoint_milestones=[],
        rng_state=rng_state,
        compile_evidence=evidence,
    )
    placeholder_state["compile_evidence"] = {"compile_requested": True, "compiled": False}
    placeholder_state["compile_evidence_sha256"] = None
    writes: list[Path] = []
    monkeypatch.setattr(trainer, "_atomic_torch_save", lambda obj, path: writes.append(path))

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match="compile_evidence"):
        trainer.save_ckpt(
            out_dir=tmp_path,
            global_step=0,
            local_step=0,
            model=model,
            optim=optimizer,
            scaler=None,
            model_config={},
            train_args={},
            run_contract={},
            position_stats={},
            sampler_state={
                "version": 2,
                "data_length": 1,
                "seed": C.STAGE_A_SAMPLER_SEED,
                "range_start_position": 0,
                "committed_position": 0,
                "end_position": 128,
            },
            data_contract={"dataset_length": 1},
            governed_run_contract=contract,
            governed_run_contract_sha256="a" * 64,
            governed_checkpoint_state=placeholder_state,
        )
    assert writes == []


def test_post_arm_eager_eval_and_first_training_signature_do_not_recompile():
    """Bounded 8x2048 regression for eager inference plus the armed training graph."""
    if not hasattr(torch.compiler, "set_stance"):
        pytest.skip("this Torch runtime has no compiler stance API")

    class TinyTrainingCallable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.projection = torch.nn.Linear(8, 32)

        def forward(self, token_ids):
            return self.projection(self.embedding(token_ids))

    torch.compiler.set_stance("default")
    torch._dynamo.reset()
    C.reset_dynamo_counters()
    base = TinyTrainingCallable().train()
    compiled = torch.compile(base, backend="eager", fullgraph=True)
    inputs = torch.arange(C.MICRO_BSZ * C.MODEL_CONTRACT["seq_len"], dtype=torch.long)
    inputs = (inputs % 32).reshape(C.MICRO_BSZ, C.MODEL_CONTRACT["seq_len"])

    try:
        with torch.enable_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            realized = compiled(inputs)
        assert realized.requires_grad
        graphs_after_realization = int(
            torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)
        )
        assert graphs_after_realization >= 1

        stance = C.arm_fail_on_recompile()
        assert stance["armed"] is True

        inference_model = trainer.select_inference_model(
            compiled,
            base,
            governed=True,
            compile_enabled=True,
        )
        assert inference_model is base
        labels = ((inputs + 1) % 32).to(torch.uint16)
        loss_mask = torch.ones_like(inputs, dtype=torch.float32)
        eval_loss = trainer.evaluate(
            inference_model,
            [(inputs.to(torch.uint16), labels, loss_mask)],
            torch.device("cpu"),
            "bf16",
            eos_id=1,
            eos_weight=1.0,
        )
        assert eval_loss >= 0.0
        assert base.training is True
        assert (
            int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))
            == graphs_after_realization
        )

        with torch.enable_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            first_training_call = compiled(inputs)
        assert first_training_call.requires_grad
        assert (
            int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))
            == graphs_after_realization
        )

        # The same stance must fail closed for a genuinely new mandatory signature. This
        # distinguishes "the second call happened to work" from an actually armed policy.
        changed_shape = inputs[:, :-1]
        with pytest.raises(RuntimeError, match="Detected recompile.*fail_on_recompile"):
            with torch.enable_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                compiled(changed_shape)
    finally:
        torch.compiler.set_stance("default")
        torch._dynamo.reset()
        C.reset_dynamo_counters()
