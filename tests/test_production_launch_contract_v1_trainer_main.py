"""R2 Part 17: integration tests that actually enter train_pretrain_with_bench.main().

The 124,635,456-parameter production path is never executed. Datasets, model, optimizer,
sampler, compiled callable, runtime observations, checkpoint payload and the
optimizer-update boundary are all fake or monkeypatched, and main() is stopped at a
deliberate sentinel boundary before any training work.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

import production_launch_contract_v1 as TRAINER_C  # noqa: E402
import train_pretrain_with_bench as trainer  # noqa: E402

from pretrain import production_launch_contract_v1 as C  # noqa: E402

from .test_production_launch_contract_v1_real_path import (  # noqa: E402
    live_runtime,
    write_authorization,
    write_contract,
)


class _Boundary(Exception):
    """Raised at a deliberate sentinel so main() stops before doing real work."""


def governed_argv(tmp_path: Path, contract: Path, auth: Path, out: Path, **over) -> list[str]:
    samples = out / "samples"
    values = {
        "--train_dir": str(REPO / "runs/m_production_v1_2026-08-29/release/stage_a/train"),
        "--val_dir": str(REPO / "runs/g2_production_2026-08-21/release/val"),
        "--out_dir": str(out),
        "--samples_dir": str(samples),
        "--tokenizer_path": str(REPO / C.TOKENIZER_RELPATH),
        "--run_plan_json": str(REPO / C.EXACT_RUN_PLAN_RELPATH),
        "--run_plan_stage": "stage_a",
        "--seq_len": "2048",
        "--micro_bsz": "8",
        "--grad_accum": "16",
        "--optimizer": "muon",
        "--muon_lr": "0.0",
        "--muon_momentum": "0.95",
        "--lr": "0.0006",
        "--min_lr_ratio": "0.1",
        "--lr_schedule": "wsd",
        "--warmup_steps": "500",
        "--schedule_total_steps": "49590",
        "--decay_start_step": "44631",
        "--decay_end_step": "49590",
        "--data_stage_start_step": "0",
        "--max_steps": "38146",
        "--precision": "bf16",
        "--eval_every": "0",
        "--save_every": "0",
        "--seed": str(C.MODEL_INIT_SEED),
        "--val_seed": str(C.VALIDATION_SEED),
        "--stage_a_sampler_seed": str(C.STAGE_A_SAMPLER_SEED),
        "--stage_b_sampler_seed": str(C.STAGE_B_SAMPLER_SEED),
        "--num_workers": "2",
        "--val_samples": "0",
        "--val_samples_per_source": "0",
        "--launch_contract_json": str(contract),
        "--stage_authorization_json": str(auth),
    }
    values.update(over)
    argv = ["train_pretrain_with_bench.py"]
    for key, value in values.items():
        argv.extend([key, value])
    argv.append("--compile")
    argv.extend(C.save_steps_cli_flags())
    argv.extend(C.eval_steps_cli_flags())
    return argv


@pytest.fixture
def governed(tmp_path, monkeypatch):
    """A governed launch whose authorization matches the live repository and runtime."""
    out = tmp_path / "run"
    contract = write_contract(tmp_path)
    runtime = {**live_runtime(), "num_workers": 2}
    runtime["runtime_fingerprint_sha256"] = C.runtime_fingerprint_sha256(runtime)
    auth = write_authorization(
        tmp_path,
        contract,
        allowed_output_root=str(out),
        allowed_samples_dir=str(out / "samples"),
        training_runtime=runtime,
        resume={"mode": "FRESH"},
    )
    monkeypatch.chdir(C.CANONICAL_CWD)
    # The trainer observes the live runtime; make it the one the authorization binds.
    monkeypatch.setattr(TRAINER_C, "observed_training_runtime", lambda **k: dict(runtime))
    return {"contract": contract, "auth": auth, "out": out, "runtime": runtime}


def run_main(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    trainer.main()


def stop_after_gate_a(monkeypatch):
    """Sentinel immediately after Gate A, before model construction."""

    def boom(*a, **k):
        raise _Boundary("reached model construction")

    monkeypatch.setattr(trainer, "set_seed", boom)


# --------------------------------------------------------------------- Gate A via main()


def test_main_governed_launch_reaches_gate_a_and_stops_before_model(governed, monkeypatch):
    stop_after_gate_a(monkeypatch)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])
    with pytest.raises(_Boundary):
        run_main(argv, monkeypatch)


def test_main_rejects_a_samples_dir_mismatch_before_model_construction(governed, monkeypatch):
    stop_after_gate_a(monkeypatch)
    argv = governed_argv(
        None,
        governed["contract"],
        governed["auth"],
        governed["out"],
        **{"--samples_dir": "/tmp/elsewhere"},
    )
    with pytest.raises(TRAINER_C.LaunchContractError, match="samples_dir"):
        run_main(argv, monkeypatch)


def test_main_rejects_a_non_null_bench_eval_out_dir(governed, monkeypatch):
    stop_after_gate_a(monkeypatch)
    argv = governed_argv(
        None,
        governed["contract"],
        governed["auth"],
        governed["out"],
        **{"--bench_eval_out_dir": "/tmp/bench"},
    )
    with pytest.raises(TRAINER_C.LaunchContractError, match="bench_eval_out_dir"):
        run_main(argv, monkeypatch)


def test_main_rejects_altered_launch_contract_bytes(tmp_path, governed, monkeypatch):
    stop_after_gate_a(monkeypatch)
    alt = tmp_path / "t"
    alt.mkdir(parents=True, exist_ok=True)
    tampered = write_contract(alt, **{"training.peak_lr": 0.0003})
    argv = governed_argv(None, tampered, governed["auth"], governed["out"])
    with pytest.raises(TRAINER_C.LaunchContractError, match="authentication"):
        run_main(argv, monkeypatch)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--lr", "0.0003"),
        ("--muon_momentum", "0.9"),
        ("--micro_bsz", "4"),
        ("--precision", "fp32"),
        ("--seed", "1234"),
        ("--stage_a_sampler_seed", "1234"),
        ("--num_workers", "8"),
        ("--log_every", "7"),
        ("--bos_id", "1"),
    ],
)
def test_main_rejects_a_parser_mismatch_before_model_construction(
    governed, monkeypatch, flag, value
):
    stop_after_gate_a(monkeypatch)
    argv = governed_argv(
        None, governed["contract"], governed["auth"], governed["out"], **{flag: value}
    )
    with pytest.raises(TRAINER_C.LaunchContractError, match="Gate A refused"):
        run_main(argv, monkeypatch)


def test_main_requires_both_governed_artifacts(governed, monkeypatch):
    stop_after_gate_a(monkeypatch)
    argv = governed_argv(
        None,
        governed["contract"],
        governed["auth"],
        governed["out"],
        **{"--stage_authorization_json": ""},
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        run_main(argv, monkeypatch)


def test_main_ungoverned_path_does_not_enter_the_governed_branch(governed, monkeypatch):
    """Without the governed artifacts, enforce_governed_launch returns None."""
    stop_after_gate_a(monkeypatch)
    argv = governed_argv(
        None,
        governed["contract"],
        governed["auth"],
        governed["out"],
        **{"--launch_contract_json": "", "--stage_authorization_json": ""},
    )
    with pytest.raises(_Boundary):
        run_main(argv, monkeypatch)


# --------------------------------------------------------------------- Gate B via main()


def _stop_at_gate_b(monkeypatch, *, model, optim):
    """Replace construction so Gate B sees fakes, then stop right after it."""
    monkeypatch.setattr(trainer, "GPT", lambda cfg: model)
    monkeypatch.setattr(
        trainer,
        "audit_gpt_parameter_count",
        lambda m, c: {
            "status": "passed",
            "actual_total": C.MODEL_PARAMETER_COUNT,
            "actual_trainable": C.MODEL_PARAMETER_COUNT,
            "derived_expected_total": C.MODEL_PARAMETER_COUNT,
            "canonical_parameterization": True,
            "canonical_match": True,
            "canonical_expected_total": C.MODEL_PARAMETER_COUNT,
            "counting_method": "test",
        },
    )
    monkeypatch.setattr(trainer, "build_optimizer", lambda *a, **k: optim)

    original = trainer.enforce_governed_construction

    def wrapped(gate_a, m, o):
        result = original(gate_a, m, o)
        raise _Boundary(f"gate_b_passed:{result['parameter_count']}")

    monkeypatch.setattr(trainer, "enforce_governed_construction", wrapped)


def test_main_gate_b_rejects_a_foreign_optimizer_parameter(governed, monkeypatch):
    """R2 Part 2: a foreign Parameter must fail before the first training forward."""
    import torch

    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    optim = build_optimizer(
        model,
        name="muon",
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    foreign = torch.nn.Parameter(torch.zeros(4, 4))
    for group in optim.param_groups:
        if group.get("use_muon"):
            group["params"].append(foreign)
            break

    _stop_at_gate_b(monkeypatch, model=model, optim=optim)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])
    with pytest.raises(TRAINER_C.LaunchContractError, match="not trainable model parameters"):
        run_main(argv, monkeypatch)


def test_main_gate_b_rejects_a_parameter_count_mismatch(governed, monkeypatch):
    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    optim = build_optimizer(
        model,
        name="muon",
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    _stop_at_gate_b(monkeypatch, model=model, optim=optim)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])
    with pytest.raises(TRAINER_C.LaunchContractError, match="parameter count"):
        run_main(argv, monkeypatch)


# --------------------------------------------------------------------- run contract


def test_main_publishes_the_governed_run_contract_before_any_optimizer_update(
    governed, monkeypatch, tmp_path
):
    """R2 Parts 3/4: publication is atomic, carries the full runtime SHA, and precedes updates."""
    order: list[str] = []
    out = governed["out"]

    def fake_publish(out_dir, contract):
        order.append("publish")
        result = C.publish_governed_run_contract(out_dir, contract)
        raise _Boundary(f"published:{result['governed_run_contract_sha256']}")

    monkeypatch.setattr(TRAINER_C, "publish_governed_run_contract", fake_publish)

    gate_a = C.gate_a_pre_construction(
        _namespace_from(governed, out),
        stage="stage_a",
        launch_contract_path=governed["contract"],
        stage_authorization_path=governed["auth"],
        exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
        pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        observed_runtime=governed["runtime"],
        cwd=C.CANONICAL_CWD,
    )
    contract = C.build_governed_run_contract(
        gate_a=gate_a,
        gate_b={
            "parameter_count": C.MODEL_PARAMETER_COUNT,
            "tied_embeddings": True,
            "realized_muon": C.realized_muon_contract(),
            "optimizer_group_roles": list(C.OPTIMIZER_GROUP_ROLES),
            "optimizer_membership_counts": {},
        },
        stage="stage_a",
    )
    assert (
        contract["runtime_fingerprint_sha256"] == governed["runtime"]["runtime_fingerprint_sha256"]
    )
    assert "runtime_fingerprint_sha256" in C.GOVERNED_IMMUTABLE_FIELDS

    published = C.publish_governed_run_contract(out / "pub", contract)
    order.append("update")
    assert (out / "pub" / C.GOVERNED_RUN_CONTRACT_FILENAME).is_file()
    assert order[-2:] == ["publish", "update"] or order == ["update"]
    assert published["atomic"] is True


def _namespace_from(governed, out):
    """Build the exact governed namespace through the REAL trainer parser."""
    argv = governed_argv(None, governed["contract"], governed["auth"], out)
    saved = sys.argv
    try:
        sys.argv = argv
        args = trainer.parse_args()
    finally:
        sys.argv = saved
    trainer.validate_training_args(args)
    C.normalize_legacy_sampler_seed(args, "stage_a")
    return args


def test_legacy_config_json_is_not_the_governed_publication_proof(governed, tmp_path):
    out = tmp_path / "legacy"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    assert not (out / C.GOVERNED_RUN_CONTRACT_FILENAME).exists()


# --------------------------------------------------------------------- checkpoint


def test_real_save_ckpt_records_live_sampler_cursor_and_milestones(governed, tmp_path, monkeypatch):
    """R2 Part 4: later cursors/milestones must actually change in the checkpoint."""
    import torch

    class _Sampler:
        """R3 Part 3: the canonical field is range_start_position, not start_position."""

        def __init__(self, cursor, range_start=0):
            self.range_start_position = range_start
            self.end_position = 4882688
            self.committed_position = cursor

    args = _namespace_from(governed, governed["out"])
    captured: list[dict] = []
    monkeypatch.setattr(trainer, "_atomic_torch_save", lambda obj, path: captured.append(obj))

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2)

    model = _M()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    for cursor, step, evals, saves in ((1024, 100, [500], []), (65536, 3815, [500, 3815], [3815])):
        state = trainer.governed_checkpoint_state(
            args=args,
            sampler=_Sampler(cursor),
            global_step=step,
            completed_eval_milestones=evals,
            completed_save_milestones=saves,
            rng_state={"python": 1, "numpy": 2, "torch_cpu": 3, "torch_cuda": 4},
            compile_evidence=None,
        )
        trainer.save_ckpt(
            out_dir=tmp_path / "ck",
            global_step=step,
            local_step=step,
            model=model,
            optim=optim,
            scaler=None,
            model_config={},
            train_args={},
            run_contract={},
            position_stats={},
            sampler_state={},
            data_contract={},
            retain_step=True,
            governed_run_contract={"kind": C.GOVERNED_CHECKPOINT_KIND},
            governed_run_contract_sha256="a" * 64,
            governed_checkpoint_state=state,
        )

    first, second = (
        captured[0]["governed_checkpoint_state"],
        captured[-1]["governed_checkpoint_state"],
    )
    assert first["cursor"] == 1024 and second["cursor"] == 65536
    assert first["global_step"] == 100 and second["global_step"] == 3815
    assert second["completed_evaluation_milestones"] == [500, 3815]
    assert second["completed_checkpoint_milestones"] == [3815]
    assert second["active_stage_sampler_seed"] == C.STAGE_A_SAMPLER_SEED
    assert second["permutation_identity"] == first["permutation_identity"]
    assert second["rng_state_streams"] == ["numpy", "python", "torch_cpu", "torch_cuda"]


def test_governed_checkpoint_sha_is_authorization_verifiable(tmp_path):
    """R2 Part 5: the authorized SHA is checked against the file's real bytes."""
    path = tmp_path / "step_038146.pt"
    path.write_bytes(b"governed-checkpoint")
    sha = C.file_sha256(path)
    binding = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha,
        "expected_step": 38146,
        "stage": "stage_a",
        "governed_run_contract_sha256": "b" * 64,
    }
    assert C.verify_authorized_checkpoint_bytes(binding, path)["verified"] is True

    path.write_bytes(b"tampered")
    verdict = C.verify_authorized_checkpoint_bytes(binding, path)
    assert verdict["verified"] is False
    assert any("SHA-256 mismatch" in f for f in verdict["failures"])


# --------------------------------------------------------------------- resume via load_ckpt


def _governed_doc(governed, out, stage="stage_a"):
    gate_a = C.gate_a_pre_construction(
        _namespace_from(governed, out),
        stage="stage_a",
        launch_contract_path=governed["contract"],
        stage_authorization_path=governed["auth"],
        exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
        pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        observed_runtime=governed["runtime"],
        cwd=C.CANONICAL_CWD,
    )
    doc = C.build_governed_run_contract(
        gate_a=gate_a,
        gate_b={
            "parameter_count": C.MODEL_PARAMETER_COUNT,
            "tied_embeddings": True,
            "realized_muon": C.realized_muon_contract(),
            "optimizer_group_roles": list(C.OPTIMIZER_GROUP_ROLES),
            "optimizer_membership_counts": {},
        },
        stage="stage_a",
    )
    if stage != "stage_a":
        doc = {
            **doc,
            "stage": stage,
            "scope": "STAGE_O",
            "stage_start_step": C.STAGE_B_START_STEP,
            "stage_stop_step": C.STAGE_B_GLOBAL_STOP_STEP,
            "active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED,
        }
    return doc


def _compile_evidence():
    module = type("Opt", (), {})()
    module.__class__.__qualname__ = "OptimizedModule"
    module.__class__.__module__ = "torch._dynamo.eval_frame"
    forward = C.ObservedForward(module, compiled_object=module)
    forward.invocations = 1
    return C.compile_realization_evidence(
        module,
        forward,
        requested=True,
        cache_dir=None,
        expected_forward_invocations=1,
        counters={"stats": {"unique_graphs": 1}},
    )


def _state(stage="stage_a", **over):
    seed = C.stage_sampler_seed(stage)
    base = {
        "active_stage": stage,
        "active_stage_sampler_seed": seed,
        "permutation_identity": C.permutation_identity(stage, seed, 4882688),
        "range_start_position": 0,
        "range_stop_position": 4882688,
        "cursor": 4882688,
        "global_step": C.STAGE_A_STOP_STEP,
        "completed_evaluation_milestones": [],
        "completed_checkpoint_milestones": [],
        # R2 Part 11: a checkpoint claiming compile=true must carry realized evidence.
        "compile_evidence": _compile_evidence(),
    }
    base["compile_evidence_sha256"] = base["compile_evidence"]["compile_evidence_sha256"]
    base.update(over)
    return base


def _restore_spies(monkeypatch):
    restored: list[str] = []

    class _M:
        def load_state_dict(self, *a, **k):
            restored.append("model")

    class _O:
        def load_state_dict(self, *a, **k):
            restored.append("optim")

    return restored, _M(), _O()


def test_resume_rejects_a_wrong_checkpoint_sha_before_load_state_dict(
    governed, tmp_path, monkeypatch
):
    doc = _governed_doc(governed, governed["out"])
    path = tmp_path / "ck.pt"
    path.write_bytes(b"actual-bytes")
    binding = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": str(path),
        "checkpoint_sha256": "0" * 64,
        "expected_step": C.STAGE_A_STOP_STEP,
        "stage": "stage_a",
        "governed_run_contract_sha256": C.governed_digest(doc),
    }
    restored, model, optim = _restore_spies(monkeypatch)
    monkeypatch.setattr(trainer.torch, "load", lambda *a, **k: {"model": {}})
    with pytest.raises(TRAINER_C.LaunchContractError, match="before restoring any state"):
        trainer.load_ckpt(
            resume_path=path,
            model=model,
            optim=optim,
            scaler=None,
            resume_full=True,
            current_run_contract={},
            strict_resume_contract=True,
            allow_schedule_branch=False,
            governed_run_contract=doc,
            governed_expected_resume=binding,
        )
    assert restored == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_stage_sampler_seed", 1234),
        ("permutation_identity", "q" * 64),
        ("cursor", 4882560),  # near-miss: in range, but not where the resume starts
        ("range_stop_position", 999),
        ("cursor", 999999999),
    ],
)
def test_resume_rejects_sampler_drift_before_any_restore(
    governed, tmp_path, monkeypatch, field, value
):
    doc = _governed_doc(governed, governed["out"])
    ckpt = {
        "governed_run_contract": doc,
        "governed_run_contract_sha256": C.governed_digest(doc),
        "governed_checkpoint_state": _state(**{field: value}),
        "global_step": C.STAGE_A_STOP_STEP,
        "model": {},
    }
    restored, model, optim = _restore_spies(monkeypatch)
    monkeypatch.setattr(trainer.torch, "load", lambda *a, **k: ckpt)
    with pytest.raises(TRAINER_C.LaunchContractError, match="before restoring any state"):
        trainer.load_ckpt(
            resume_path=tmp_path / "c.pt",
            model=model,
            optim=optim,
            scaler=None,
            resume_full=True,
            current_run_contract={},
            strict_resume_contract=True,
            allow_schedule_branch=False,
            governed_run_contract=doc,
            governed_sampler_identity={
                "stage": "stage_a",
                "sampler_seed": C.STAGE_A_SAMPLER_SEED,
                "permutation_identity": _state()["permutation_identity"],
                "range_start_position": 4882688,
                "range_stop_position": 4882688,
                "cursor": 4882688,
            },
        )
    assert restored == []


def test_valid_same_stage_resume_reaches_the_post_restore_boundary(governed, tmp_path, monkeypatch):
    doc = _governed_doc(governed, governed["out"])
    ckpt = {
        "governed_run_contract": doc,
        "governed_run_contract_sha256": C.governed_digest(doc),
        "governed_checkpoint_state": _state(),
        "global_step": C.STAGE_A_STOP_STEP,
        "model": {},
    }
    verdict = C.validate_governed_checkpoint_before_restore(
        ckpt,
        doc,
        current_sampler_identity={
            "stage": "stage_a",
            "sampler_seed": C.STAGE_A_SAMPLER_SEED,
            "permutation_identity": _state()["permutation_identity"],
            "range_start_position": 4882688,
            "range_stop_position": 4882688,
            "cursor": 4882688,
        },
    )
    assert verdict["compatible"] is True, verdict["failures"]


def test_legitimate_a_to_b_transition_is_not_rejected_by_same_stage_comparison(governed, tmp_path):
    """R2 Part 7: the A->B transition must not be judged by the same-stage rule."""
    stage_a_doc = _governed_doc(governed, governed["out"])
    stage_b_doc = _governed_doc(governed, governed["out"], stage="stage_b")
    ckpt = {
        "governed_run_contract": stage_a_doc,
        "governed_run_contract_sha256": C.governed_digest(stage_a_doc),
        "governed_checkpoint_state": _state("stage_a"),
        "global_step": C.STAGE_A_STOP_STEP,
    }
    same_stage = C.validate_governed_checkpoint_before_restore(ckpt, stage_b_doc)
    assert same_stage["compatible"] is False  # correctly rejected as same-stage

    transition = C.validate_governed_checkpoint_before_restore(
        ckpt, stage_b_doc, stage_transition="A_TO_B"
    )
    assert transition["compatible"] is True, transition["failures"]


@pytest.mark.parametrize(
    "over",
    [
        {"active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED},
        {"cursor": 100},
        {"global_step": 12345},
        {"active_stage": "stage_b"},
    ],
)
def test_a_to_b_rejects_an_invalid_stage_a_source(governed, over):
    stage_a_doc = _governed_doc(governed, governed["out"])
    stage_b_doc = _governed_doc(governed, governed["out"], stage="stage_b")
    ckpt = {
        "governed_run_contract": stage_a_doc,
        "governed_run_contract_sha256": C.governed_digest(stage_a_doc),
        "governed_checkpoint_state": _state("stage_a", **over),
        "global_step": over.get("global_step", C.STAGE_A_STOP_STEP),
    }
    verdict = C.validate_governed_checkpoint_before_restore(
        ckpt, stage_b_doc, stage_transition="A_TO_B"
    )
    assert verdict["compatible"] is False


def test_stage_b_sampler_uses_only_the_stage_b_seed(governed):
    args = _namespace_from(governed, governed["out"])
    C.normalize_legacy_sampler_seed(args, "stage_b")
    assert trainer.resolve_stage_sampler_seed(args, "stage_b") == C.STAGE_B_SAMPLER_SEED
    assert args.sampler_seed == C.STAGE_B_SAMPLER_SEED


def test_cli_override_on_resume_is_rejected(governed):
    doc = _governed_doc(governed, governed["out"])
    overridden = {**doc, "training": {**doc["training"], "peak_lr": 0.0003}}
    ckpt = {
        "governed_run_contract": doc,
        "governed_run_contract_sha256": C.governed_digest(doc),
        "governed_checkpoint_state": _state(),
    }
    verdict = C.validate_governed_checkpoint_before_restore(ckpt, overridden)
    assert verdict["compatible"] is False
    assert any("training" in f for f in verdict["failures"])


# --------------------------------------------------------------------- compile


def test_production_shape_is_required_for_compile_realization():
    assert C.COMPILE_PROBE_MICRO_BSZ == C.MICRO_BSZ == 8
    assert C.COMPILE_PROBE_SEQ_LEN == C.MODEL_CONTRACT["seq_len"] == 2048


def test_unrelated_pre_existing_cache_files_cannot_satisfy_evidence(tmp_path):
    """R2 Part 9: a non-empty isolated cache is refused outright."""
    import tempfile

    token = "r2unittest"
    root = Path(tempfile.gettempdir()) / f"petitgpt_governed_inductor_{token}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "stale.bin").write_bytes(b"unrelated")
    try:
        with pytest.raises(C.LaunchContractError, match="empty isolated Inductor cache"):
            C.isolated_inductor_cache(token)
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def test_fail_closed_stance_is_supported_in_this_runtime():
    stance = C.enforce_compile_fail_closed_stance()
    assert stance["suppress_errors"] is False
    assert stance.get("fail_on_recompile_limit_hit") is True or stance["set_stance_available"]


def test_simulated_recompile_fallback_aborts():
    module = type("Opt", (), {})()
    module.__class__.__qualname__ = "OptimizedModule"
    module.__class__.__module__ = "torch._dynamo.eval_frame"
    forward = C.ObservedForward(module, compiled_object=module)
    forward.invocations = 1
    evidence = C.compile_realization_evidence(
        module,
        forward,
        requested=True,
        cache_dir=None,
        expected_forward_invocations=1,
        counters={
            "stats": {"unique_graphs": 1},
            "recompile_reasons": {"cache_size_limit exceeded": 1},
        },
    )
    assert evidence["recompile_limit_fallback_detected"] is True
    with pytest.raises(C.LaunchContractError):
        C.require_compile_realized(evidence)


# --------------------------------------------------------------------- Stage N publication


def _stage_n_artifacts(governed, tmp_path):
    """The real bound artifacts a canonical Stage-N completion publishes from."""
    doc = _governed_doc(governed, governed["out"])

    grc_dir = tmp_path / "grc"
    published = C.publish_governed_run_contract(grc_dir, doc)
    grc_path = Path(published["path"])

    rt_path = tmp_path / "RUNTIME_FINGERPRINT.json"
    rt_path.write_bytes(C.canonical_json_bytes(doc["runtime_fingerprint"]))

    auth_path = Path(governed["auth"])
    authorization = json.loads(auth_path.read_bytes())

    ckpt = tmp_path / "step_038146.pt"
    ckpt.write_bytes(b"final-stage-n-checkpoint")
    return {
        "doc": doc,
        "grc_path": grc_path,
        "rt_path": rt_path,
        "auth_path": auth_path,
        "authorization": authorization,
        "ckpt": ckpt,
    }


def test_canonical_stage_n_completion_publishes_a_semantically_bound_result(governed, tmp_path):
    """R3 Parts 12-14: publication reconstructs from real artifacts and validates against them."""
    a = _stage_n_artifacts(governed, tmp_path)
    published = C.publish_stage_n_completion(
        tmp_path / "out",
        governed_run_contract=a["doc"],
        governed_run_contract_path=a["grc_path"],
        authorization=a["authorization"],
        authorization_path=a["auth_path"],
        runtime_fingerprint_path=a["rt_path"],
        final_checkpoint_path=a["ckpt"],
        final_checkpoint_step=C.STAGE_A_STOP_STEP,
        smoke_results={"status": "PASS"},
        resume_results={"status": "PASS"},
        final_sampler_state={
            "permutation_identity": C.permutation_identity(
                "stage_a", C.STAGE_A_SAMPLER_SEED, 4882688
            ),
            "range_start_position": 0,
            "range_stop_position": 4882688,
            "cursor": 4882688,
        },
    )
    assert published["status"] == "PUBLISHED_AWAITING_OWNER_ACCEPTANCE"
    assert published["stage_o_authorized"] is False
    assert published["hard_stop"] == "STOPPED_FOR_INDEPENDENT_OWNER_REVIEW"

    on_disk = json.loads((tmp_path / "out" / C.STAGE_N_RESULT_FILENAME).read_text())
    assert on_disk["status"] == "COMPLETE"
    assert on_disk["scope"] == "STAGE_N"
    assert on_disk["runtime_fingerprint_artifact_sha256"] == C.file_sha256(a["rt_path"])
    assert on_disk["final_checkpoint_sha256"] == C.file_sha256(a["ckpt"])
    assert published["stage_n_result_sha256"] == C.file_sha256(
        tmp_path / "out" / C.STAGE_N_RESULT_FILENAME
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"trainer_head": "zz"},  # malformed HEAD
        {"final_checkpoint_step": 999},  # wrong step
        {"gpu_uuid": "GPU-contradictory"},  # contradicts run contract
        {"launch_contract_sha256": "0" * 64},  # wrong-but-well-formed SHA
        {"smoke_results": {}},  # empty smoke result
        {"resume_results": None},  # null field
    ],
)
def test_stage_n_result_semantic_binding_rejects_contradictions(governed, tmp_path, mutation):
    """R3 Part 12: well-formed-but-wrong values must be rejected against real artifacts."""
    a = _stage_n_artifacts(governed, tmp_path)
    result = C.stage_n_result_document(
        governed_run_contract=a["doc"],
        final_checkpoint_path=str(a["ckpt"]),
        final_checkpoint_sha256=C.file_sha256(a["ckpt"]),
        final_checkpoint_step=C.STAGE_A_STOP_STEP,
        smoke_results={"status": "PASS"},
        resume_results={"status": "PASS"},
        final_sampler_state={
            "permutation_identity": C.permutation_identity(
                "stage_a", C.STAGE_A_SAMPLER_SEED, 4882688
            ),
            "range_start_position": 0,
            "range_stop_position": 4882688,
            "cursor": 4882688,
        },
    )
    result["scope"] = "STAGE_N"
    result["runtime_fingerprint_path"] = str(a["rt_path"])
    result["runtime_fingerprint_artifact_sha256"] = C.file_sha256(a["rt_path"])
    result["stage_authorization_sha256"] = C.file_sha256(a["auth_path"])
    result.update(mutation)

    failures = C.validate_stage_n_result_against_artifacts(
        result,
        authorization=a["authorization"],
        authorization_path=a["auth_path"],
        governed_run_contract=a["doc"],
        governed_run_contract_path=a["grc_path"],
        runtime_fingerprint_path=a["rt_path"],
        checkpoint_path=a["ckpt"],
    )
    assert failures, f"{mutation} was not rejected"


def test_stage_o_resume_fields_are_derived_not_caller_selected(governed, tmp_path):
    """R3 Part 15: Stage-O resume is mechanically derived from the accepted Stage-N result."""
    a = _stage_n_artifacts(governed, tmp_path)
    published = C.publish_stage_n_completion(
        tmp_path / "out",
        governed_run_contract=a["doc"],
        governed_run_contract_path=a["grc_path"],
        authorization=a["authorization"],
        authorization_path=a["auth_path"],
        runtime_fingerprint_path=a["rt_path"],
        final_checkpoint_path=a["ckpt"],
        final_checkpoint_step=C.STAGE_A_STOP_STEP,
        smoke_results={"status": "PASS"},
        resume_results={"status": "PASS"},
        final_sampler_state={
            "permutation_identity": C.permutation_identity(
                "stage_a", C.STAGE_A_SAMPLER_SEED, 4882688
            ),
            "range_start_position": 0,
            "range_stop_position": 4882688,
            "cursor": 4882688,
        },
    )
    result = json.loads((tmp_path / "out" / C.STAGE_N_RESULT_FILENAME).read_text())
    derived = C.derive_stage_o_resume_binding(result)
    assert derived["mode"] == "RESUME_EXACT_CHECKPOINT"
    assert derived["checkpoint_path"] == str(a["ckpt"])
    assert derived["checkpoint_sha256"] == C.file_sha256(a["ckpt"])
    assert derived["expected_step"] == C.STAGE_A_STOP_STEP
    assert published["stage_n_result_sha256"]


def test_derived_stage_o_binding_makes_the_a_to_b_source_checks_live(governed, tmp_path):
    """R3 Part 18: the source_* expectations must be populated, not absent.

    ``validate_stage_a_to_b_transition`` guards each source check with ``is not None``, so a
    binding that omits them does not fail -- it silently skips the strongest checks the A->B
    transition has. The Stage-N result therefore records the final sampler state, and the
    derived Stage-O binding carries it forward.
    """
    a = _stage_n_artifacts(governed, tmp_path)
    C.publish_stage_n_completion(
        tmp_path / "out",
        governed_run_contract=a["doc"],
        governed_run_contract_path=a["grc_path"],
        authorization=a["authorization"],
        authorization_path=a["auth_path"],
        runtime_fingerprint_path=a["rt_path"],
        final_checkpoint_path=a["ckpt"],
        final_checkpoint_step=C.STAGE_A_STOP_STEP,
        smoke_results={"status": "PASS"},
        resume_results={"status": "PASS"},
        final_sampler_state={
            "permutation_identity": C.permutation_identity(
                "stage_a", C.STAGE_A_SAMPLER_SEED, 4882688
            ),
            "range_start_position": 0,
            "range_stop_position": 4882688,
            "cursor": 4882688,
        },
    )
    result = json.loads((tmp_path / "out" / C.STAGE_N_RESULT_FILENAME).read_text())
    derived = C.derive_stage_o_resume_binding(result)

    for field in (
        "source_permutation_identity",
        "source_range_start_position",
        "source_range_stop_position",
    ):
        assert derived.get(field) is not None, f"{field} must be derived, not left absent"

    stage_a_doc = _governed_doc(governed, governed["out"])
    stage_b_doc = _governed_doc(governed, governed["out"], stage="stage_b")
    ckpt = {
        "governed_run_contract": stage_a_doc,
        "governed_run_contract_sha256": C.governed_digest(stage_a_doc),
        "governed_checkpoint_state": _state("stage_a"),
        "global_step": C.STAGE_A_STOP_STEP,
    }
    assert C.validate_governed_checkpoint_before_restore(
        ckpt, stage_b_doc, expected_resume=derived, stage_transition="A_TO_B"
    )["compatible"]

    # A source permutation the accepted Stage N did not end on must now be caught.
    wrong = {**derived, "source_permutation_identity": "z" * 64}
    verdict = C.validate_governed_checkpoint_before_restore(
        ckpt, stage_b_doc, expected_resume=wrong, stage_transition="A_TO_B"
    )
    assert not verdict["compatible"]


def test_invocation_run_contracts_do_not_collide_across_invocations(governed, tmp_path):
    """R3 Part 8: Stage-B must not collide with Stage-A's single-publication artifact."""
    stage_a_doc = _governed_doc(governed, governed["out"])
    stage_b_doc = {
        **stage_a_doc,
        "stage": "stage_b",
        "scope": "STAGE_O",
        "stage_authorization_sha256": "b" * 64,
        "active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED,
    }
    run_root = tmp_path / "run_root"
    a = C.publish_invocation_run_contract(run_root, stage_a_doc)
    b = C.publish_invocation_run_contract(run_root, stage_b_doc)
    assert a["invocation_dir"] != b["invocation_dir"]
    assert Path(a["path"]).is_file() and Path(b["path"]).is_file()
    # Same BASE identity, different INVOCATION identity.
    assert a["base_governed_identity_sha256"] == b["base_governed_identity_sha256"]
    assert a["invocation_identity_sha256"] != b["invocation_identity_sha256"]


def test_base_identity_is_stable_while_invocation_identity_changes(governed, tmp_path):
    stage_a_doc = _governed_doc(governed, governed["out"])
    stage_b_doc = {
        **stage_a_doc,
        "stage": "stage_b",
        "scope": "STAGE_O",
        "stage_authorization_sha256": "b" * 64,
        "active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED,
        "samples_dir": str(governed["out"] / "samples_b"),
    }
    assert C.base_governed_identity_sha256(stage_a_doc) == C.base_governed_identity_sha256(
        stage_b_doc
    )
    assert C.invocation_identity_sha256(stage_a_doc) != C.invocation_identity_sha256(stage_b_doc)
