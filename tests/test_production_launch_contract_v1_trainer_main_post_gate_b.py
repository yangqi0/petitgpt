"""R3 Part 18: integration tests that enter real ``main()`` and proceed PAST Gate B.

The existing ``..._trainer_main`` suite stops at or before Gate B, so nothing previously
exercised the governed surfaces that live *downstream* of construction from inside
``main()`` itself. Covered here: Gate C compile realization, atomic run-contract
publication, the same-stage resume path, and the authorized A->B (Stage O) transition.

NOT covered here, and not claimed to be: the governed save. Its call site inside ``main()``
is only reachable by executing the training loop, which this segment forbids; ``save_ckpt``'s
governed payload is covered at helper level in ``..._trainer_main`` instead.

Nothing production-shaped runs. The 124,635,456-parameter model is never built, Gate B's
verdict and compile realization are stubbed, and every test stops at a deliberate sentinel.
No test performs an optimizer update or a forward/backward pass: ``optim.zero_grad`` and
``optim.step`` are patched to abort, so a test that slips past its sentinel fails
immediately rather than silently training.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

import production_launch_contract_v1 as TRAINER_C  # noqa: E402
import train_pretrain_with_bench as trainer  # noqa: E402

from pretrain import production_launch_contract_v1 as C  # noqa: E402

from .test_production_launch_contract_v1_real_path import write_authorization  # noqa: E402
from .test_production_launch_contract_v1_trainer_main import (  # noqa: E402
    _Boundary,
    _compile_evidence,
    _ordinary_stage_a_run_contract,
    _stage_a_data_contract_for_fixture,
    governed,  # noqa: F401,F811  (pytest fixture re-export)
    governed_argv,
    run_main,
)

STAGE_A_SAMPLES = C.STAGE_A_STOP_STEP * 128  # micro_bsz 8 x grad_accum 16
STAGE_B_SAMPLES = (C.STAGE_B_GLOBAL_STOP_STEP - C.STAGE_B_START_STEP) * 128


def _tiny_model():
    from src.model import GPT, GPTConfig

    return GPT(GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128))


def _passing_audit(model, cfg):
    return {
        "status": "passed",
        "actual_total": C.MODEL_PARAMETER_COUNT,
        "actual_trainable": C.MODEL_PARAMETER_COUNT,
        "derived_expected_total": C.MODEL_PARAMETER_COUNT,
        "canonical_parameterization": True,
        "canonical_match": True,
        "canonical_expected_total": C.MODEL_PARAMETER_COUNT,
        "counting_method": "test",
    }


def past_gate_b(monkeypatch):
    """Let main() carry a fake model/optimizer through Gate B and keep going.

    Returns the record dict the tests assert on: which governed surfaces were reached, in
    what order, and whether any optimizer update happened.
    """
    from src.optim import build_optimizer

    model = _tiny_model()
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
    record: dict = {"model": model, "optim": optim, "order": [], "updates": 0, "realize": []}

    monkeypatch.setattr(trainer, "GPT", lambda cfg: model)
    monkeypatch.setattr(trainer, "audit_gpt_parameter_count", _passing_audit)
    monkeypatch.setattr(trainer, "build_optimizer", lambda *a, **k: optim)

    # Gate B legitimately refuses a 2-layer stand-in: it reads the real architecture and
    # parameter count. That refusal is exactly what the ..._trainer_main suite asserts. Here
    # the subject is everything DOWNSTREAM of Gate B, so its verdict is stubbed with the
    # result the production model would have produced.
    def stub_gate_b(gate_a, m, o):
        record["order"].append("gate_b")
        return {
            "parameter_count": C.MODEL_PARAMETER_COUNT,
            "tied_embeddings": True,
            "realized_muon": C.realized_muon_contract(),
            "optimizer_group_roles": list(C.OPTIMIZER_GROUP_ROLES),
            "optimizer_membership_counts": {},
        }

    monkeypatch.setattr(trainer, "enforce_governed_construction", stub_gate_b)

    # No test in this file may reach a training step. These are hard stops, not counters:
    # a test that slips past its sentinel fails instantly instead of silently training.
    def no_updates(*a, **k):
        record["updates"] += 1
        record["order"].append("optimizer_update")
        raise AssertionError("a governed test reached an optimizer update")

    def no_loop(*a, **k):
        record["order"].append("training_loop")
        raise _Boundary("entered the training loop")

    monkeypatch.setattr(optim, "step", no_updates)
    monkeypatch.setattr(optim, "zero_grad", no_loop)

    # Gate C's realization is the one genuinely expensive step; record its arguments instead
    # of compiling. Everything it feeds -- require_compile_realized, the stance arming, the
    # evidence seal and the published contract -- still runs for real.
    def fake_realize(m, *, device, micro_bsz, seq_len, vocab_size, cache, finalize=True):
        record["realize"].append({
            "micro_bsz": int(micro_bsz),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
            "cache_dir": cache.get("cache_dir"),
        })
        record["order"].append("compile_realized")
        assert finalize is False, "Gate C must receive observations before the one final seal"
        evidence = dict(_compile_evidence())
        for final_field in ("compile_evidence_sha256", "compile_realized", "verdict"):
            evidence.pop(final_field)
        return evidence

    monkeypatch.setattr(TRAINER_C, "realize_compile_production_shape", fake_realize)

    # An identity stub is correctly refused: the contract forbids claiming compile=true when
    # torch.compile hands back the eager module. The stub must therefore be a real distinct
    # wrapper, shaped like the OptimizedModule dynamo would return.
    class OptimizedModule(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self._orig_mod = mod

        def forward(self, *a, **k):
            return self._orig_mod(*a, **k)

    OptimizedModule.__module__ = "torch._dynamo.eval_frame"
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: OptimizedModule(m))

    original_publish = TRAINER_C.publish_governed_run_contract

    def spy_publish(out_dir, contract):
        record["order"].append("publish")
        result = original_publish(out_dir, contract)
        record["published"] = result
        record["contract"] = contract
        return result

    monkeypatch.setattr(TRAINER_C, "publish_governed_run_contract", spy_publish)
    return record


def stop_after_publication(monkeypatch, record):
    original = trainer.publish_governed_run_contract_now

    def wrapped(**kw):
        contract, sha = original(**kw)
        record["sampler"] = kw["sampler"]
        raise _Boundary(f"published:{sha}")

    monkeypatch.setattr(trainer, "publish_governed_run_contract_now", wrapped)


# ------------------------------------------------------- Gate C reached from inside main()


def test_main_reaches_gate_c_and_publishes_before_any_optimizer_update(governed, monkeypatch):  # noqa: F811
    """A fresh governed launch realizes compile and publishes, with zero updates so far."""
    record = past_gate_b(monkeypatch)
    stop_after_publication(monkeypatch, record)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])

    with pytest.raises(_Boundary, match="published:"):
        run_main(argv, monkeypatch)

    assert record["order"] == ["gate_b", "compile_realized", "publish"]
    assert record["updates"] == 0
    invocation = C.invocation_directory(governed["out"], "stage_a", C.file_sha256(governed["auth"]))
    assert (invocation / C.GOVERNED_RUN_CONTRACT_FILENAME).is_file()
    assert record["published"]["atomic"] is True


def test_main_realizes_compile_at_the_frozen_production_geometry(governed, monkeypatch):  # noqa: F811
    """R2 Parts 8-10 driven through main(): batch 1 probes must never reach realization."""
    record = past_gate_b(monkeypatch)
    stop_after_publication(monkeypatch, record)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])

    with pytest.raises(_Boundary):
        run_main(argv, monkeypatch)

    assert len(record["realize"]) == 1, "Gate C realizes exactly once"
    probe = record["realize"][0]
    assert probe["micro_bsz"] == C.MICRO_BSZ, "batch-1 probes must never reach realization"
    assert probe["seq_len"] == C.MODEL_CONTRACT["seq_len"]
    assert probe["vocab_size"] == C.MODEL_CONTRACT["vocab_size"]
    assert probe["cache_dir"], "Gate C must use an isolated Inductor cache"


def test_published_contract_binds_the_live_sampler_main_actually_built(governed, monkeypatch):  # noqa: F811
    """The published sampler identity must come from the real sampler, not a placeholder."""
    record = past_gate_b(monkeypatch)
    stop_after_publication(monkeypatch, record)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])

    with pytest.raises(_Boundary):
        run_main(argv, monkeypatch)

    identity = record["contract"]["sampler_identity"]
    sampler = record["sampler"]
    assert identity == C.sampler_identity_document("stage_a", sampler)
    assert identity["sampler_seed"] == C.STAGE_A_SAMPLER_SEED
    assert identity["range_start_position"] == 0
    assert identity["range_stop_position"] == STAGE_A_SAMPLES


def test_main_aborts_when_gate_c_realization_is_not_evidenced(governed, monkeypatch):  # noqa: F811
    """An unrealized compile must stop main() at Gate C, before publication."""
    record = past_gate_b(monkeypatch)

    def unrealized(m, *, device, micro_bsz, seq_len, vocab_size, cache, finalize=True):
        # The authoritative sub-facts, not a cosmetic flag: this is what an eager fallback
        # actually looks like.
        assert finalize is False
        evidence = dict(_compile_evidence())
        for final_field in ("compile_evidence_sha256", "compile_realized", "verdict"):
            evidence.pop(final_field)
        evidence.update({
            "compilation_materialized": False,
            "dynamo_unique_graphs": 0,
            "failures": ["compile_never_materialized_lazily_eager_fallback"],
        })
        return evidence

    monkeypatch.setattr(TRAINER_C, "realize_compile_production_shape", unrealized)
    argv = governed_argv(None, governed["contract"], governed["auth"], governed["out"])

    with pytest.raises(TRAINER_C.LaunchContractError):
        run_main(argv, monkeypatch)

    assert "publish" not in record["order"]
    assert not (governed["out"] / C.GOVERNED_RUN_CONTRACT_FILENAME).exists()
    assert record["updates"] == 0


# ------------------------------------------------- governed resume reached from inside main()

RESUME_STEP = 10000
RESUME_CURSOR = RESUME_STEP * 128


def _fresh_doc(governed):  # noqa: F811
    """The governed contract the interrupted FRESH invocation would have published."""
    from .test_production_launch_contract_v1_trainer_main import _governed_doc

    doc = _governed_doc(governed, governed["out"])
    doc["sampler_identity"] = {
        "stage": "stage_a",
        "sampler_seed": C.STAGE_A_SAMPLER_SEED,
        "permutation_identity": C.permutation_identity(
            "stage_a", C.STAGE_A_SAMPLER_SEED, STAGE_A_SAMPLES
        ),
        "range_start_position": 0,
        "invocation_range_start_position": 0,
        "range_stop_position": STAGE_A_SAMPLES,
        "cursor": 0,
        "consumed": 0,
        "remaining": STAGE_A_SAMPLES,
    }
    return doc


def _saved_state(doc, **over):
    """The sampler state a FRESH Stage-A run holds after committing RESUME_CURSOR samples."""
    state = {
        "schema_version": "petitgpt-governed-checkpoint-state-v1",
        "active_stage": "stage_a",
        "active_stage_sampler_seed": C.STAGE_A_SAMPLER_SEED,
        "permutation_identity": C.permutation_identity(
            "stage_a", C.STAGE_A_SAMPLER_SEED, STAGE_A_SAMPLES
        ),
        "range_start_position": 0,
        "invocation_range_start_position": 0,
        "range_stop_position": STAGE_A_SAMPLES,
        "cursor": RESUME_CURSOR,
        "consumed": RESUME_CURSOR,
        "remaining": STAGE_A_SAMPLES - RESUME_CURSOR,
        "global_step": RESUME_STEP,
        "completed_evaluation_milestones": [
            value for value in C.EVALUATION_MILESTONES if value <= RESUME_STEP
        ],
        "completed_checkpoint_milestones": [
            value for value in C.CHECKPOINT_MILESTONES if value <= RESUME_STEP
        ],
        "rng_state": trainer.capture_rng_state(),
        "compile_evidence": _compile_evidence(),
    }
    state["compile_evidence_sha256"] = state["compile_evidence"]["compile_evidence_sha256"]
    state.update(over)
    return state


def _restart(tmp_path, governed, monkeypatch, **state_over):  # noqa: F811
    """Authorize a same-stage restart of the interrupted run, and stage its checkpoint.

    The restart is a NEW invocation: a new authorization file whose resume binding names the
    interrupted run's contract digest. That is what a real crash restart looks like.
    """
    doc = _fresh_doc(governed)
    publication = C.publish_invocation_run_contract(doc)
    ckpt_dir = Path(publication["invocation_dir"])
    ckpt_path = ckpt_dir / f"step_{RESUME_STEP:06d}.pt"
    ckpt_path.write_bytes(b"opaque governed checkpoint bytes")

    auth_dir = tmp_path / "restart"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth = write_authorization(
        auth_dir,
        governed["contract"],
        allowed_output_root=str(governed["out"]),
        allowed_samples_dir=str(governed["out"] / "samples"),
        training_runtime=governed["runtime"],
        resume={
            "mode": "RESUME_EXACT_CHECKPOINT",
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": C.file_sha256(ckpt_path),
            "expected_step": RESUME_STEP,
            "stage": "stage_a",
            "governed_run_contract_sha256": C.governed_digest(doc),
            "source_stage_authorization_path": doc["stage_authorization_path"],
            "source_stage_authorization_sha256": doc["stage_authorization_sha256"],
            "source_invocation_run_contract_path": publication["path"],
            "source_invocation_run_contract_sha256": publication["file_sha256"],
            "source_base_governed_identity_digest": C.base_governed_identity_sha256(doc),
            "source_checkpoint_path": str(ckpt_path),
            "source_checkpoint_sha256": C.file_sha256(ckpt_path),
            "source_checkpoint_step": RESUME_STEP,
            "source_checkpoint_stage": "stage_a",
            "source_active_stage": "stage_a",
            "source_sampler_seed": C.STAGE_A_SAMPLER_SEED,
            "source_permutation_identity": C.permutation_identity(
                "stage_a", C.STAGE_A_SAMPLER_SEED, STAGE_A_SAMPLES
            ),
            "source_range_start_position": 0,
            "source_invocation_range_start_position": 0,
            "source_range_stop_position": STAGE_A_SAMPLES,
            "source_cursor": RESUME_CURSOR,
        },
    )
    state = _saved_state(doc, **state_over)
    data_contract = _stage_a_data_contract_for_fixture(governed)
    ckpt = {
        "governed_run_contract": doc,
        "governed_run_contract_sha256": C.governed_digest(doc),
        "governed_checkpoint_state": state,
        "global_step": RESUME_STEP,
        "local_step": RESUME_STEP,
        "model": {},
        "optim": {},
        "scaler": None,
        "run_contract": _ordinary_stage_a_run_contract(governed, governed["out"]),
        "data_contract": data_contract,
        "data_sampler": {
            "version": 2,
            "data_length": data_contract["dataset_length"],
            "seed": C.STAGE_A_SAMPLER_SEED,
            "range_start_position": 0,
            "committed_position": state["cursor"],
            "end_position": STAGE_A_SAMPLES,
        },
    }
    monkeypatch.setattr(trainer.torch, "load", lambda *a, **k: ckpt)
    return auth, ckpt_path, ckpt


def _resume_argv(governed, auth, ckpt_path):  # noqa: F811
    return governed_argv(
        None,
        governed["contract"],
        auth,
        governed["out"],
        **{
            "--resume_path": str(ckpt_path),
            "--resume_step": str(RESUME_STEP),
        },
    ) + ["--resume_full"]


def _trip_after_governed_validation(monkeypatch, record):
    """Sentinel at the first NON-governed resume check, so passing it proves the gate ran."""

    def boom(*a, **k):
        record["order"].append("governed_resume_validated")
        raise _Boundary("governed resume accepted")

    monkeypatch.setattr(trainer, "validate_resume_contract", boom)


def test_main_accepts_a_real_same_stage_restart(governed, tmp_path, monkeypatch):  # noqa: F811
    """A crash restart is a NEW authorization; it must still be resumable.

    Requiring the restart's authorization SHA and resume binding to equal the interrupted
    run's made every same-stage restart impossible. This is the regression test for that.
    """
    record = past_gate_b(monkeypatch)
    auth, ckpt_path, _ = _restart(tmp_path, governed, monkeypatch)
    _trip_after_governed_validation(monkeypatch, record)

    with pytest.raises(_Boundary, match="governed resume accepted"):
        run_main(_resume_argv(governed, auth, ckpt_path), monkeypatch)

    assert "governed_resume_validated" in record["order"]
    assert record["updates"] == 0


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"cursor": RESUME_CURSOR - 128}, "checkpoint_state_differs_from_source_authority"),
        ({"cursor": RESUME_CURSOR + 128}, "checkpoint_state_differs_from_source_authority"),
        (
            {"global_step": RESUME_STEP + 1},
            "checkpoint_state_differs_from_source_authority",
        ),
        (
            {"active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED},
            "checkpoint_state_differs_from_source_authority",
        ),
        (
            {"range_stop_position": 999},
            "checkpoint_state_differs_from_source_authority",
        ),
        ({"active_stage": "stage_b"}, "checkpoint_state_differs_from_source_authority"),
    ],
)
def test_main_refuses_a_drifted_restart_before_restoring_state(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
    over,
    expected,
):
    """The trainer's OWN derived expectation -- not a hand-passed one -- must catch drift."""
    record = past_gate_b(monkeypatch)
    auth, ckpt_path, _ = _restart(tmp_path, governed, monkeypatch, **over)
    _trip_after_governed_validation(monkeypatch, record)

    with pytest.raises(TRAINER_C.LaunchContractError) as excinfo:
        run_main(_resume_argv(governed, auth, ckpt_path), monkeypatch)

    assert "before restoring any state" in str(excinfo.value)
    assert expected in str(excinfo.value)
    assert "governed_resume_validated" not in record["order"]
    assert record["updates"] == 0


def test_restart_authorization_is_still_pinned_to_the_interrupted_contract(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """The compensating control for excluding `resume` from the same-stage match.

    A restart may carry a different authorization, but its resume binding must name the
    exact contract digest of the run it claims to continue.
    """
    record = past_gate_b(monkeypatch)
    auth, ckpt_path, ckpt = _restart(tmp_path, governed, monkeypatch)
    raw = json.loads(auth.read_bytes())
    raw["resume"]["governed_run_contract_sha256"] = "e" * 64
    auth.write_bytes(C.canonical_json_bytes(raw))
    _trip_after_governed_validation(monkeypatch, record)

    with pytest.raises(TRAINER_C.LaunchContractError) as excinfo:
        run_main(_resume_argv(governed, auth, ckpt_path), monkeypatch)

    assert "source_authority_resume_run_contract_digest_mismatch" in str(excinfo.value)
    assert "governed_resume_validated" not in record["order"]


def test_main_refuses_a_checkpoint_at_the_wrong_step(governed, tmp_path, monkeypatch):  # noqa: F811
    """The top-level checkpoint step is a separate guard from the sampler's recorded step."""
    record = past_gate_b(monkeypatch)
    auth, ckpt_path, ckpt = _restart(tmp_path, governed, monkeypatch)
    ckpt["global_step"] = RESUME_STEP + 1
    _trip_after_governed_validation(monkeypatch, record)

    with pytest.raises(TRAINER_C.LaunchContractError) as excinfo:
        run_main(_resume_argv(governed, auth, ckpt_path), monkeypatch)

    assert "checkpoint_step_differs_from_source_authority" in str(excinfo.value)
    assert "governed_resume_validated" not in record["order"]
    assert record["updates"] == 0


# ------------------------------------------------ A_TO_B (Stage O) reached from inside main()

STAGE_B_TRAIN = REPO / "runs/m_production_v1_2026-08-29/release/stage_b/train"


def _stage_o(tmp_path, governed, monkeypatch):  # noqa: F811
    """Publish a Stage-N result, accept it, and authorize the Stage-O A->B continuation.

    Everything is derived from the SAME live runtime the trainer observes, so the BASE
    identity genuinely matches across the transition rather than matching by construction.
    """
    from .test_production_launch_contract_v1_trainer_main import _stage_n_artifacts

    work = tmp_path / "stage_n"
    work.mkdir(parents=True, exist_ok=True)
    artifacts = _stage_n_artifacts(governed, work)
    stage_a_doc = artifacts["doc"]
    ckpt_path = artifacts["ckpt"]
    source_invocation = artifacts["invocation"]

    C.publish_stage_n_completion(
        source_invocation,
        governed_run_contract=stage_a_doc,
        governed_run_contract_path=artifacts["grc_path"],
        authorization=artifacts["authorization"],
        authorization_path=artifacts["auth_path"],
        runtime_fingerprint_path=artifacts["rt_path"],
        final_checkpoint_path=ckpt_path,
        final_checkpoint_step=C.STAGE_A_STOP_STEP,
        smoke_results_path=artifacts["smoke_path"],
        resume_results_path=artifacts["resume_path"],
        final_sampler_state={
            "permutation_identity": C.permutation_identity(
                "stage_a", C.STAGE_A_SAMPLER_SEED, STAGE_A_SAMPLES
            ),
            "range_start_position": 0,
            "invocation_range_start_position": 0,
            "range_stop_position": STAGE_A_SAMPLES,
            "cursor": STAGE_A_SAMPLES,
        },
    )
    result_path = source_invocation / C.STAGE_N_RESULT_FILENAME
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)

    acceptance = {
        "stage_n_result_owner_verdict": "ACCEPTED",
        "accepted_stage_n_result_sha256": C._sha256_bytes(result_bytes),
    }
    acc_path = work / "STAGE_N_ACCEPTANCE.json"
    acc_bytes = C.canonical_json_bytes(acceptance)
    acc_path.write_bytes(acc_bytes)

    chain = {
        "accepted_stage_n_result_path": str(result_path),
        "accepted_stage_n_result_sha256": C._sha256_bytes(result_bytes),
        "stage_n_owner_acceptance_path": str(acc_path),
        "stage_n_owner_acceptance_sha256": C._sha256_bytes(acc_bytes),
        "stage_n_authorization_sha256": result["stage_authorization_sha256"],
        "stage_n_governed_run_contract_sha256": result["governed_run_contract_sha256"],
        "stage_n_governed_run_contract_artifact_sha256": result[
            "governed_run_contract_artifact_sha256"
        ],
        "stage_n_runtime_fingerprint": result["runtime_fingerprint"],
        "stage_n_runtime_fingerprint_sha256": result["runtime_fingerprint_sha256"],
        "stage_n_gpu_uuid": result["gpu_uuid"],
        "stage_n_gpu_pci_bus_id": result["gpu_pci_bus_id"],
        "stage_n_trainer_head": result["trainer_head"],
        "stage_n_trainer_execution_bundle_sha256": result["trainer_execution_bundle_sha256"],
        "stage_n_exact_run_plan_sha256": result["exact_run_plan_sha256"],
        "stage_n_final_checkpoint_path": result["final_checkpoint_path"],
        "stage_n_final_checkpoint_sha256": result["final_checkpoint_sha256"],
        "stage_n_final_checkpoint_step": result["final_checkpoint_step"],
        "stage_n_runtime_fingerprint_path": result["runtime_fingerprint_path"],
        "stage_n_runtime_fingerprint_artifact_sha256": result[
            "runtime_fingerprint_artifact_sha256"
        ],
    }

    # Stage B is a new invocation under the SAME governed production run root. Its
    # authorization SHA derives a collision-free invocation directory.
    out_b = governed["out"]
    auth_dir = tmp_path / "stage_o"
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_b = write_authorization(
        auth_dir,
        governed["contract"],
        allowed_scope="STAGE_O",
        transition="A_TO_B",
        stage_n_chain=chain,
        allowed_output_root=str(out_b),
        allowed_samples_dir=str(out_b / "samples"),
        training_runtime=governed["runtime"],
        resume=C.derive_stage_o_resume_binding(result),
    )

    real_torch_load = torch.load
    stage_a_ckpt = real_torch_load(ckpt_path, map_location="cpu", weights_only=False)

    def load_stage_a_fixture(path, *args, **kwargs):
        if Path(path).resolve() == ckpt_path.resolve():
            return stage_a_ckpt
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(trainer.torch, "load", load_stage_a_fixture)
    return {"auth": auth_b, "out": out_b, "ckpt": ckpt_path, "state": stage_a_ckpt}


def _stage_b_argv(governed, o):  # noqa: F811
    return governed_argv(
        None,
        governed["contract"],
        o["auth"],
        o["out"],
        **{
            "--train_dir": str(STAGE_B_TRAIN),
            "--run_plan_stage": "stage_b",
            "--data_stage_start_step": str(C.STAGE_A_STOP_STEP),
            "--max_steps": str(C.STAGE_B_GLOBAL_STOP_STEP),
            "--resume_path": str(o["ckpt"]),
            "--resume_step": str(C.STAGE_A_STOP_STEP),
        },
    ) + ["--resume_full"]


def test_main_accepts_the_authorized_a_to_b_transition(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Stage O reached through real main(): the A->B handoff must pass pre-restore validation."""
    record = past_gate_b(monkeypatch)
    o = _stage_o(tmp_path, governed, monkeypatch)
    _trip_after_governed_validation(monkeypatch, record)

    with pytest.raises(_Boundary, match="governed resume accepted"):
        run_main(_stage_b_argv(governed, o), monkeypatch)

    assert "governed_resume_validated" in record["order"]
    assert record["updates"] == 0


def test_main_a_to_b_reaches_the_verified_stage_b_sampler_boundary(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Real main restores the accepted source, then builds and verifies Stage B at cursor 0."""
    record = past_gate_b(monkeypatch)
    o = _stage_o(tmp_path, governed, monkeypatch)

    # The tiny Stage-N artifact carries a real ordinary Stage-A run contract. Delegate the
    # complete generic plus governed load_ckpt control flow while making only the two
    # tensor-container restores compatible with the tiny objects.

    def restore_model(state, *, strict=True):
        assert strict is True
        assert set(state) == {"fixture_weight"}
        record["order"].append("model_restored")

    def restore_optim(state):
        assert state == {"state": {}, "param_groups": []}
        record["order"].append("optimizer_restored")

    monkeypatch.setattr(record["model"], "load_state_dict", restore_model)
    monkeypatch.setattr(record["optim"], "load_state_dict", restore_optim)

    real_restore_rng_state = trainer.restore_rng_state

    def restore_rng_state_spy(state):
        real_restore_rng_state(state)
        record["order"].append("rng_restored")

    monkeypatch.setattr(trainer, "restore_rng_state", restore_rng_state_spy)

    real_validate_data_resume_state = trainer.validate_data_resume_state

    def validate_stage_b_sampler(**kwargs):
        real_validate_data_resume_state(**kwargs)
        sampler = kwargs["current_sampler"]
        record["stage_b_sampler"] = sampler.state_dict()
        record["order"].append("stage_b_sampler_verified")
        raise _Boundary("verified Stage-B sampler boundary")

    monkeypatch.setattr(trainer, "validate_data_resume_state", validate_stage_b_sampler)

    with pytest.raises(_Boundary, match="verified Stage-B sampler boundary"):
        run_main(_stage_b_argv(governed, o), monkeypatch)

    assert record["order"][-4:] == [
        "model_restored",
        "optimizer_restored",
        "rng_restored",
        "stage_b_sampler_verified",
    ]
    sampler = record["stage_b_sampler"]
    assert sampler["seed"] == C.STAGE_B_SAMPLER_SEED
    assert sampler["range_start_position"] == 0
    assert sampler["committed_position"] == 0
    assert sampler["end_position"] == STAGE_B_SAMPLES
    assert record["updates"] == 0


def test_main_refuses_an_a_to_b_source_the_accepted_stage_n_did_not_end_on(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """The derived source_* expectations must actually bite on the real path."""
    record = past_gate_b(monkeypatch)
    o = _stage_o(tmp_path, governed, monkeypatch)
    o["state"]["governed_checkpoint_state"]["cursor"] = STAGE_A_SAMPLES - 128
    _trip_after_governed_validation(monkeypatch, record)

    with pytest.raises(TRAINER_C.LaunchContractError) as excinfo:
        run_main(_stage_b_argv(governed, o), monkeypatch)

    assert "stage_n_final_checkpoint" in str(excinfo.value)
    assert "governed_resume_validated" not in record["order"]
