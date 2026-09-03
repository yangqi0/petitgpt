"""R4 Stage-N/Stage-O artifact and CUDA identity regressions.

All checkpoints and evidence are tiny local fixtures. No production model forward, backward,
or optimizer update is executed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

import production_launch_contract_v1 as TRAINER_C  # noqa: E402
import train_pretrain_with_bench as trainer  # noqa: E402

from pretrain import production_launch_contract_v1 as C  # noqa: E402

from .test_production_launch_contract_v1_r4_resume_save import (  # noqa: E402
    _install_real_stage_a_data_resume_state,
    _restart_authorization,
    _source_invocation,
    _stage_b_source_invocation,
)
from .test_production_launch_contract_v1_real_path import write_authorization  # noqa: E402
from .test_production_launch_contract_v1_trainer_main import (  # noqa: E402
    _compile_evidence,
    _governed_doc,
    _ordinary_stage_a_run_contract,
    _publish_fixture_resume_check,
    _stage_a_data_contract_for_fixture,
    governed,  # noqa: F401,F811 - pytest fixture re-export
    governed_argv,
    run_main,
)
from .test_production_launch_contract_v1_trainer_main_post_gate_b import (  # noqa: E402
    past_gate_b,
)

STAGE_A_SAMPLES = C.STAGE_A_STOP_STEP * C.SEQUENCES_PER_UPDATE


def _write_json(path: Path, document: dict) -> Path:
    path.write_bytes(C.canonical_json_bytes(document))
    return path


def _restorable_rng_state() -> dict:
    import random

    import numpy as np

    torch_state = torch.Generator(device="cpu").manual_seed(20260831).get_state()
    if torch.cuda.is_available():
        cuda_state = torch.Generator(device="cuda:0").manual_seed(20260831).get_state()
    else:
        cuda_state = torch.tensor(
            list((20260831).to_bytes(8, "little") + (0).to_bytes(8, "little")),
            dtype=torch.uint8,
        )
    return {
        "python": random.Random(20260831).getstate(),
        "numpy": np.random.RandomState(20260831).get_state(),
        "torch_cpu": torch_state.clone(),
        # CUDA Philox state is exactly 16 bytes under the frozen PyTorch runtime.
        "torch_cuda": [cuda_state],
    }


def _stage_n_artifacts(governed, tmp_path: Path) -> dict:  # noqa: F811
    authorization_path = Path(governed["auth"])
    authorization = json.loads(authorization_path.read_bytes())
    authorization["stage_n_completion"] = {"expected_final_step": C.STAGE_A_STOP_STEP}
    authorization_path.write_bytes(C.canonical_json_bytes(authorization))

    contract = _governed_doc(governed, governed["out"])
    evidence = _compile_evidence()
    assert C.verify_compile_evidence_document(evidence) == []
    contract["compile_evidence"] = evidence
    contract["compile_evidence_sha256"] = evidence["compile_evidence_sha256"]

    initial_sampler = SimpleNamespace(
        seed=C.STAGE_A_SAMPLER_SEED,
        range_start_position=0,
        end_position=STAGE_A_SAMPLES,
        committed_position=0,
    )
    contract["sampler_identity"] = C.sampler_identity_document("stage_a", initial_sampler)
    invocation = Path(contract["invocation_root"])
    grc_publication = C.publish_governed_run_contract(invocation, contract)
    runtime_publication = C.publish_stage_n_runtime_artifact(
        invocation, contract["runtime_fingerprint"]
    )

    sampler = SimpleNamespace(
        seed=C.STAGE_A_SAMPLER_SEED,
        range_start_position=0,
        end_position=STAGE_A_SAMPLES,
        committed_position=STAGE_A_SAMPLES,
    )
    dynamic = C.build_checkpoint_state(
        stage="stage_a",
        sampler=sampler,
        global_step=C.STAGE_A_STOP_STEP,
        completed_evaluation_milestones=[
            step for step in C.EVALUATION_MILESTONES if step <= C.STAGE_A_STOP_STEP
        ],
        completed_checkpoint_milestones=[
            step for step in C.CHECKPOINT_MILESTONES if step <= C.STAGE_A_STOP_STEP
        ],
        rng_state=_restorable_rng_state(),
        compile_evidence=evidence,
    )
    checkpoint_path = invocation / f"step_{C.STAGE_A_STOP_STEP:06d}.pt"
    ordinary_run_contract = _ordinary_stage_a_run_contract(governed, governed["out"])
    data_contract = _stage_a_data_contract_for_fixture(governed)
    torch.save(
        {
            "kind": C.GOVERNED_CHECKPOINT_KIND,
            "global_step": C.STAGE_A_STOP_STEP,
            "local_step": C.STAGE_A_STOP_STEP,
            "governed_run_contract": contract,
            "governed_run_contract_sha256": C.governed_digest(contract),
            "governed_checkpoint_state": dynamic,
            "run_contract": ordinary_run_contract,
            "data_contract": data_contract,
            "data_sampler": {
                "version": 2,
                "data_length": data_contract["dataset_length"],
                "seed": C.STAGE_A_SAMPLER_SEED,
                "range_start_position": 0,
                "committed_position": STAGE_A_SAMPLES,
                "end_position": STAGE_A_SAMPLES,
            },
            "model": {},
            "optim": {},
            "scaler": None,
        },
        checkpoint_path,
    )

    smoke_log = invocation / "smoke.log"
    smoke_log.write_bytes(b"bounded smoke evidence\n")
    smoke_path = Path(
        C.publish_stage_n_check_result(
            invocation,
            kind=C.STAGE_N_SMOKE_RESULT_KIND,
            authorization_path=authorization_path,
            governed_run_contract_path=grc_publication["path"],
            runtime_fingerprint_path=runtime_publication["path"],
            checkpoint_path=checkpoint_path,
            checkpoint_step=C.STAGE_A_STOP_STEP,
            evidence_artifact_path=smoke_log,
        )["path"]
    )
    fixture_source = {
        "doc": contract,
        "dynamic": dynamic,
        "ckpt": checkpoint_path,
        "auth_path": authorization_path,
        "grc_path": Path(grc_publication["path"]),
        "rt_path": Path(runtime_publication["path"]),
        "invocation": invocation,
    }
    resume_path = _publish_fixture_resume_check(fixture_source, governed, tmp_path)
    published = C.publish_stage_n_completion(
        invocation,
        governed_run_contract=contract,
        governed_run_contract_path=grc_publication["path"],
        authorization=authorization,
        authorization_path=authorization_path,
        runtime_fingerprint_path=runtime_publication["path"],
        final_checkpoint_path=checkpoint_path,
        final_checkpoint_step=C.STAGE_A_STOP_STEP,
        smoke_results_path=smoke_path,
        resume_results_path=resume_path,
        final_sampler_state=C.sampler_identity_document("stage_a", sampler),
    )
    result_path = Path(published["path"])
    return {
        "contract": contract,
        "authorization": authorization,
        "authorization_path": authorization_path,
        "grc_path": Path(grc_publication["path"]),
        "runtime_path": Path(runtime_publication["path"]),
        "checkpoint_path": checkpoint_path,
        "smoke_path": smoke_path,
        "resume_path": resume_path,
        "resume_publication_context": fixture_source["resume_publication_context"],
        "result_path": result_path,
        "result": json.loads(result_path.read_bytes()),
    }


def _validate(bundle: dict, **overrides) -> list[str]:
    args = {
        "authorization_path": bundle["authorization_path"],
        "governed_run_contract_path": bundle["grc_path"],
        "exact_plan_path": REPO / C.EXACT_RUN_PLAN_RELPATH,
        "pilot_acceptance_path": REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        "runtime_fingerprint_path": bundle["runtime_path"],
        "checkpoint_path": bundle["checkpoint_path"],
        "smoke_results_path": bundle["smoke_path"],
        "resume_results_path": bundle["resume_path"],
        "expected_final_step": C.STAGE_A_STOP_STEP,
    }
    args.update(overrides)
    return C.validate_stage_n_result_against_artifacts(bundle["result"], **args)


def test_stage_n_completion_authenticates_every_actual_artifact(governed, tmp_path):  # noqa: F811
    bundle = _stage_n_artifacts(governed, tmp_path)
    assert _validate(bundle) == []
    result = bundle["result"]
    assert result["governed_run_contract_artifact_sha256"] == C.file_sha256(bundle["grc_path"])
    assert result["runtime_fingerprint_artifact_sha256"] == C.file_sha256(bundle["runtime_path"])

    checkpoint = torch.load(bundle["checkpoint_path"], map_location="cpu", weights_only=False)
    checkpoint.pop("data_sampler")
    torch.save(checkpoint, bundle["checkpoint_path"])
    missing_operational = C.validate_stage_n_checkpoint_artifact(
        bundle["checkpoint_path"],
        result=result,
        governed_run_contract=json.loads(bundle["grc_path"].read_bytes()),
        expected_final_step=C.STAGE_A_STOP_STEP,
    )
    assert "governed_checkpoint_operational_sampler_state_missing" in missing_operational
    checkpoint["data_sampler"] = {
        "version": 2,
        "data_length": checkpoint["data_contract"]["dataset_length"],
        "seed": C.STAGE_A_SAMPLER_SEED,
        "range_start_position": 0,
        "committed_position": STAGE_A_SAMPLES,
        "end_position": STAGE_A_SAMPLES,
    }
    torch.save(checkpoint, bundle["checkpoint_path"])

    with pytest.raises(C.LaunchContractError, match="caller final checkpoint step differs"):
        C.publish_stage_n_completion(
            bundle["result_path"].parent,
            governed_run_contract=bundle["contract"],
            governed_run_contract_path=bundle["grc_path"],
            authorization=bundle["authorization"],
            authorization_path=bundle["authorization_path"],
            runtime_fingerprint_path=bundle["runtime_path"],
            final_checkpoint_path=bundle["checkpoint_path"],
            final_checkpoint_step=C.STAGE_A_STOP_STEP - 1,
            smoke_results_path=bundle["smoke_path"],
            resume_results_path=bundle["resume_path"],
            final_sampler_state={
                "permutation_identity": result["final_sampler_permutation_identity"],
                "range_start_position": result["final_sampler_range_start_position"],
                "invocation_range_start_position": result[
                    "final_sampler_invocation_range_start_position"
                ],
                "range_stop_position": result["final_sampler_range_stop_position"],
                "cursor": result["final_sampler_cursor"],
            },
        )

    # A result and checkpoint cannot collude on a different sampler stop: the GRC's
    # pre-training sampler identity and the step-derived cursor remain independent authority.
    checkpoint = torch.load(bundle["checkpoint_path"], map_location="cpu", weights_only=False)
    state = checkpoint["governed_checkpoint_state"]
    drift_stop = state["range_stop_position"] + C.SEQUENCES_PER_UPDATE
    state["range_stop_position"] = drift_stop
    state["cursor"] = drift_stop
    state["consumed"] = drift_stop
    state["remaining"] = 0
    state["permutation_identity"] = C.permutation_identity(
        "stage_a", C.STAGE_A_SAMPLER_SEED, drift_stop
    )
    torch.save(checkpoint, bundle["checkpoint_path"])
    colluding_result = copy.deepcopy(result)
    colluding_result["final_sampler_permutation_identity"] = state["permutation_identity"]
    colluding_result["final_sampler_range_stop_position"] = drift_stop
    colluding_result["final_sampler_cursor"] = drift_stop
    failures = C.validate_stage_n_checkpoint_artifact(
        bundle["checkpoint_path"],
        result=colluding_result,
        governed_run_contract=json.loads(bundle["grc_path"].read_bytes()),
        expected_final_step=C.STAGE_A_STOP_STEP,
    )
    assert any("contract_sampler_mismatch" in failure for failure in failures)
    assert "stage_n_final_checkpoint_cursor_step_mismatch" in failures


def test_resume_proof_retry_and_zero_update_state_are_load_bearing(
    governed,  # noqa: F811
    tmp_path,
):
    bundle = _stage_n_artifacts(governed, tmp_path)
    context = bundle["resume_publication_context"]
    evidence_path = bundle["resume_path"].with_name(C.STAGE_N_RESUME_EVIDENCE_FILENAME)

    # Simulate a crash after evidence publication but before the result rename. A retry
    # reuses byte-identical evidence and recreates the missing result; a further retry is a
    # complete no-op.
    bundle["resume_path"].unlink()
    retried = C.publish_stage_n_resume_check_from_verified_invocation(**context)
    assert retried["evidence"]["reused"] is True
    assert retried["result"]["reused"] is False
    repeated = C.publish_stage_n_resume_check_from_verified_invocation(**context)
    assert repeated["evidence"]["reused"] is True
    assert repeated["result"]["reused"] is True

    canonical_evidence = evidence_path.read_bytes()
    contradictory = json.loads(canonical_evidence)
    contradictory["optimizer_updates"] = 1
    evidence_path.write_bytes(C.canonical_json_bytes(contradictory))
    with pytest.raises(C.LaunchContractError, match="contradictory pre-existing"):
        C.publish_stage_n_resume_check_from_verified_invocation(**context)
    evidence_path.write_bytes(canonical_evidence)

    original_checkpoint = Path(context["resume_final_checkpoint_path"]).read_bytes()
    original_evidence = json.loads(canonical_evidence)
    original_result = json.loads(bundle["resume_path"].read_bytes())

    def validation_failures(mutator) -> list[str]:
        current_path = Path(context["resume_final_checkpoint_path"])
        current_path.write_bytes(original_checkpoint)
        checkpoint = torch.load(current_path, map_location="cpu", weights_only=False)
        mutator(checkpoint)
        torch.save(checkpoint, current_path)
        evidence = copy.deepcopy(original_evidence)
        evidence["resume_final_checkpoint_sha256"] = C.file_sha256(current_path)
        evidence_path.write_bytes(C.canonical_json_bytes(evidence))
        result = copy.deepcopy(original_result)
        result["evidence_artifact_sha256"] = C.file_sha256(evidence_path)
        source = bundle["result"]
        return C.validate_stage_n_check_result(
            result,
            expected_kind=C.STAGE_N_RESUME_RESULT_KIND,
            expected_authorization_sha256=source["stage_authorization_sha256"],
            expected_governed_run_contract_sha256=source["governed_run_contract_sha256"],
            expected_runtime_fingerprint_sha256=source["runtime_fingerprint_sha256"],
            expected_checkpoint_path=bundle["checkpoint_path"],
            expected_checkpoint_sha256=C.file_sha256(bundle["checkpoint_path"]),
            expected_checkpoint_step=C.STAGE_A_STOP_STEP,
            expected_authorization_path=bundle["authorization_path"],
            expected_governed_run_contract_path=bundle["grc_path"],
        )

    def mutate_model(checkpoint):
        checkpoint["model"]["drift"] = torch.tensor([9.0])

    def mutate_optimizer(checkpoint):
        checkpoint["optim"]["state"] = {1: {"step": 1}}

    def mutate_rng(checkpoint):
        import random

        checkpoint["governed_checkpoint_state"]["rng_state"]["python"] = random.Random(7).getstate()

    def mutate_milestones(checkpoint):
        checkpoint["governed_checkpoint_state"]["completed_evaluation_milestones"].pop()

    def mutate_cursor(checkpoint):
        state = checkpoint["governed_checkpoint_state"]
        state["invocation_range_start_position"] -= 1
        state["cursor"] -= 1

    def remove_operational_sampler(checkpoint):
        checkpoint.pop("data_sampler")

    for mutator, expected in (
        (mutate_model, "zero_update_mismatch:model"),
        (mutate_optimizer, "zero_update_mismatch:optim"),
        (mutate_rng, "zero_update_mismatch:rng_state"),
        (mutate_milestones, "zero_update_mismatch:completed_evaluation_milestones"),
        (mutate_cursor, "sampler_resume_discontinuity"),
        (remove_operational_sampler, "operational_sampler_state_missing"),
    ):
        failures = validation_failures(mutator)
        assert any(expected in failure for failure in failures), failures


@pytest.mark.parametrize("artifact", ("authorization_path", "grc_path"))
def test_stage_n_rejects_non_json_authority_artifacts(governed, tmp_path, artifact):  # noqa: F811
    bundle = _stage_n_artifacts(governed, tmp_path)
    malformed = tmp_path / f"bad-{artifact}.bin"
    malformed.write_bytes(b"not json")
    keyword = (
        "authorization_path" if artifact == "authorization_path" else "governed_run_contract_path"
    )
    failures = _validate(bundle, **{keyword: malformed})
    assert any("not_valid_json" in failure for failure in failures)


def test_stage_n_rejects_path_substitution_even_for_identical_runtime_bytes(
    governed,  # noqa: F811
    tmp_path,
):
    bundle = _stage_n_artifacts(governed, tmp_path)
    substituted = tmp_path / "substituted-runtime.json"
    substituted.write_bytes(bundle["runtime_path"].read_bytes())
    failures = _validate(bundle, runtime_fingerprint_path=substituted)
    assert "stage_n_result_runtime_fingerprint_path_mismatch" in failures


def test_stage_n_rejects_arbitrary_non_checkpoint_bytes(governed, tmp_path):  # noqa: F811
    bundle = _stage_n_artifacts(governed, tmp_path)
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"arbitrary non-checkpoint bytes")
    failures = _validate(bundle, checkpoint_path=fake)
    assert any("not_loadable" in failure for failure in failures)


def test_stage_n_checkpoint_embedded_contract_must_equal_the_actual_grc_artifact(
    governed,  # noqa: F811
    tmp_path,
):
    bundle = _stage_n_artifacts(governed, tmp_path)
    checkpoint = torch.load(bundle["checkpoint_path"], map_location="cpu", weights_only=False)
    checkpoint["governed_run_contract"]["out_dir"] = "/tmp/substituted"
    torch.save(checkpoint, bundle["checkpoint_path"])
    loaded_grc = json.loads(bundle["grc_path"].read_bytes())
    failures = C.validate_stage_n_checkpoint_artifact(
        bundle["checkpoint_path"],
        result=bundle["result"],
        governed_run_contract=loaded_grc,
        expected_final_step=C.STAGE_A_STOP_STEP,
    )
    assert "stage_n_final_checkpoint_embedded_contract_not_exact_artifact" in failures


def test_bare_status_maps_are_not_stage_n_smoke_or_resume_results():
    failures = C.validate_stage_n_check_result(
        {"status": "PASS"},
        expected_kind=C.STAGE_N_SMOKE_RESULT_KIND,
        expected_authorization_sha256="a" * 64,
        expected_governed_run_contract_sha256="b" * 64,
        expected_runtime_fingerprint_sha256="c" * 64,
    )
    assert any("missing_field" in failure for failure in failures)


def test_stage_o_reopens_runtime_and_derives_the_accepted_checkpoint(
    governed,  # noqa: F811
    tmp_path,
):
    bundle = _stage_n_artifacts(governed, tmp_path)
    result = bundle["result"]
    acceptance = {
        "stage_n_result_owner_verdict": "ACCEPTED",
        "accepted_stage_n_result_sha256": C.file_sha256(bundle["result_path"]),
    }
    acceptance_path = _write_json(tmp_path / "STAGE_N_ACCEPTANCE.json", acceptance)
    chain = {
        "accepted_stage_n_result_path": str(bundle["result_path"]),
        "accepted_stage_n_result_sha256": C.file_sha256(bundle["result_path"]),
        "stage_n_owner_acceptance_path": str(acceptance_path),
        "stage_n_owner_acceptance_sha256": C.file_sha256(acceptance_path),
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
    derived = C.derive_stage_o_resume_binding(result)
    verdict = C.validate_stage_o_chain(
        {"stage_n_chain": chain, "resume": derived},
        observed_runtime=result["runtime_fingerprint"],
    )
    assert verdict["valid"] is True, verdict["failures"]
    assert verdict["derived_resume"] == derived
    assert derived["checkpoint_path"] == result["final_checkpoint_path"]
    assert derived["checkpoint_sha256"] == result["final_checkpoint_sha256"]
    assert derived["expected_step"] == result["final_checkpoint_step"]


def test_stage_o_stage_b_crash_restart_authenticates_source_before_gate_a(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    # R2 reserves the live repair branch for the exact successor/N3 path.  This older R4
    # regression exercises the ordinary non-incident Stage-O chain, so select that policy
    # explicitly without weakening the production successor-branch dispatch.
    non_incident_branch = "agent/test-only-non-incident-stage-o"
    monkeypatch.setattr(C, "STAGE_N_SUCCESSOR_COMPATIBILITY_BRANCH", non_incident_branch)
    monkeypatch.setattr(
        TRAINER_C,
        "STAGE_N_SUCCESSOR_COMPATIBILITY_BRANCH",
        non_incident_branch,
    )
    accepted = _stage_n_artifacts(governed, tmp_path)
    source_step = C.STAGE_B_START_STEP + 1
    stage_b = _stage_b_source_invocation(
        tmp_path,
        accepted_stage_n=accepted,
        governed=governed,
        source_step=source_step,
    )
    auth_dir = tmp_path / "stage_b_restart_authorization"
    auth_dir.mkdir(parents=True, exist_ok=True)
    restart_auth_path = write_authorization(
        auth_dir,
        governed["contract"],
        allowed_scope="STAGE_O",
        transition=None,
        allowed_output_root=str(governed["out"]),
        allowed_samples_dir=str(governed["out"] / "samples"),
        training_runtime=governed["runtime"],
        resume=stage_b["source_binding"],
        stage_n_chain=stage_b["stage_n_chain"],
    )
    authorization = json.loads(restart_auth_path.read_bytes())

    source_verdict = C.verify_resume_source_authority(authorization["resume"], transition=None)
    assert source_verdict["verified"], source_verdict["failures"]
    chain_verdict = C.validate_stage_o_chain(authorization, observed_runtime=governed["runtime"])
    assert chain_verdict["valid"], chain_verdict["failures"]
    assert chain_verdict["derived_resume"] == stage_b["source_binding"]

    def gate_args():
        argv = governed_argv(
            None,
            governed["contract"],
            restart_auth_path,
            governed["out"],
            **{
                "--train_dir": str(REPO / "runs/m_production_v1_2026-08-29/release/stage_b/train"),
                "--run_plan_stage": "stage_b",
                "--data_stage_start_step": str(C.STAGE_B_START_STEP),
                "--max_steps": str(C.STAGE_B_GLOBAL_STOP_STEP),
                "--resume_path": str(stage_b["checkpoint_path"]),
                "--resume_step": str(source_step),
            },
        ) + ["--resume_full"]
        saved_argv = sys.argv
        try:
            sys.argv = argv
            args = trainer.parse_args()
        finally:
            sys.argv = saved_argv
        trainer.validate_training_args(args)
        C.normalize_legacy_sampler_seed(args, "stage_b")
        return args

    gate_a = C.gate_a_pre_construction(
        gate_args(),
        stage="stage_b",
        launch_contract_path=governed["contract"],
        stage_authorization_path=restart_auth_path,
        exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
        pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        observed_runtime=governed["runtime"],
        cwd=C.CANONICAL_CWD,
    )
    assert gate_a["passed"] is True
    assert gate_a["transition"] is None
    assert gate_a["resume"] == stage_b["source_binding"]

    tampered_grc = copy.deepcopy(stage_b["grc"])
    tampered_grc["out_dir"] = str(tmp_path / "substituted-stage-b-root")
    stage_b["grc_path"].write_bytes(C.canonical_json_bytes(tampered_grc))
    with pytest.raises(C.LaunchContractError, match="Gate A refused"):
        C.gate_a_pre_construction(
            gate_args(),
            stage="stage_b",
            launch_contract_path=governed["contract"],
            stage_authorization_path=restart_auth_path,
            exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
            pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
            observed_runtime=governed["runtime"],
            cwd=C.CANONICAL_CWD,
        )


def test_fresh_stage_n_main_final_save_waits_cleanly_for_independent_checks(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
    capsys,  # noqa: F811
):
    """A controlled fresh authorization reaches real final save and awaits both checks."""
    authorization_path = Path(governed["auth"])
    authorization = json.loads(authorization_path.read_bytes())
    authorization["stage_n_completion"] = {"expected_final_step": C.STAGE_A_STOP_STEP}
    authorization_path.write_bytes(C.canonical_json_bytes(authorization))

    record = past_gate_b(monkeypatch)
    fake_progress = tmp_path / "bounded-fresh-progress.pt"
    fake_progress.write_bytes(b"test-only controlled progress sentinel")
    real_enforce = trainer.enforce_governed_launch

    def enforce_then_fast_forward(args):
        gate_a = real_enforce(args)
        # Test-only control after the real FRESH Gate-A verdict: skip 38,146 production
        # updates while still reaching the actual Gate-C/final-save/awaiting call sites.
        args.resume_path = str(fake_progress)
        args.resume_step = C.STAGE_A_STOP_STEP
        return gate_a

    monkeypatch.setattr(trainer, "enforce_governed_launch", enforce_then_fast_forward)
    monkeypatch.setattr(
        trainer,
        "load_ckpt",
        lambda **kwargs: (
            C.STAGE_A_STOP_STEP,
            C.STAGE_A_STOP_STEP,
            trainer.empty_position_stats(),
            {
                "governed_checkpoint_state": {
                    "completed_evaluation_milestones": [
                        step for step in C.EVALUATION_MILESTONES if step <= C.STAGE_A_STOP_STEP
                    ],
                    "completed_checkpoint_milestones": [
                        step for step in C.CHECKPOINT_MILESTONES if step <= C.STAGE_A_STOP_STEP
                    ],
                },
                "data_contract": {},
                "data_sampler": {},
            },
        ),
    )
    monkeypatch.setattr(trainer, "validate_data_resume_state", lambda **kwargs: None)

    run_main(
        governed_argv(
            None,
            governed["contract"],
            authorization_path,
            governed["out"],
        ),
        monkeypatch,
    )

    assert record["updates"] == 0
    invocation = C.invocation_directory(
        governed["out"], "stage_a", C.file_sha256(authorization_path)
    )
    assert (invocation / f"step_{C.STAGE_A_STOP_STEP:06d}.pt").is_file()
    assert (invocation / C.STAGE_N_RUNTIME_FILENAME).is_file()
    assert not (invocation / C.STAGE_N_RESULT_FILENAME).exists()
    output = capsys.readouterr().out
    assert "AWAITING_SMOKE_AND_RESUME_CHECKS" in output
    assert C.STAGE_N_SMOKE_RESULT_FILENAME in output
    assert C.STAGE_N_RESUME_RESULT_FILENAME in output


def test_stage_n_terminal_resume_predicate_excludes_midstage_and_unverified_sources():
    launch = {
        "transition": None,
        "resume": {
            "mode": "RESUME_EXACT_CHECKPOINT",
            "source_checkpoint_step": C.STAGE_A_STOP_STEP,
        },
        "verified_source_authority": {"verified": True},
    }
    assert trainer.is_stage_n_terminal_zero_update_resume(launch, C.STAGE_A_STOP_STEP)

    launch["resume"]["source_checkpoint_step"] = C.STAGE_A_STOP_STEP - 1
    assert not trainer.is_stage_n_terminal_zero_update_resume(launch, C.STAGE_A_STOP_STEP)
    launch["resume"]["source_checkpoint_step"] = C.STAGE_A_STOP_STEP
    launch["transition"] = "A_TO_B"
    assert not trainer.is_stage_n_terminal_zero_update_resume(launch, C.STAGE_A_STOP_STEP)
    launch["transition"] = None
    launch["verified_source_authority"]["verified"] = False
    assert not trainer.is_stage_n_terminal_zero_update_resume(launch, C.STAGE_A_STOP_STEP)


def test_stage_n_midstage_restart_completion_awaits_checks_for_current_invocation(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,
    capsys,  # noqa: F811
):
    """A completed crash restart becomes a new source, not a zero-update proof of the old one."""
    source_authorization_path = Path(governed["auth"])
    source_authorization = json.loads(source_authorization_path.read_bytes())
    source_authorization["stage_n_completion"] = {"expected_final_step": C.STAGE_A_STOP_STEP}
    source_authorization_path.write_bytes(C.canonical_json_bytes(source_authorization))

    source_step = C.STAGE_A_STOP_STEP - 1
    source = _source_invocation(governed, tmp_path, source_step=source_step)
    _install_real_stage_a_data_resume_state(governed, source)
    authorization_path = _restart_authorization(governed, tmp_path, source)
    authorization = json.loads(authorization_path.read_bytes())
    authorization["stage_n_completion"] = {"expected_final_step": C.STAGE_A_STOP_STEP}
    authorization_path.write_bytes(C.canonical_json_bytes(authorization))

    record = past_gate_b(monkeypatch)
    monkeypatch.setattr(trainer, "validate_resume_contract", lambda *a, **k: None)
    monkeypatch.setattr(trainer, "validate_data_resume_state", lambda **kwargs: None)
    monkeypatch.setattr(
        trainer,
        "load_ckpt",
        lambda **kwargs: (
            C.STAGE_A_STOP_STEP,
            C.STAGE_A_STOP_STEP,
            trainer.empty_position_stats(),
            {
                "governed_checkpoint_state": {
                    "completed_evaluation_milestones": [
                        step for step in C.EVALUATION_MILESTONES if step <= C.STAGE_A_STOP_STEP
                    ],
                    "completed_checkpoint_milestones": [
                        step for step in C.CHECKPOINT_MILESTONES if step <= C.STAGE_A_STOP_STEP
                    ],
                },
                "data_contract": {},
                "data_sampler": {},
                "rng_state": source["dynamic_state"]["rng_state"],
            },
        ),
    )

    def forbid_zero_update_publication(**kwargs):
        raise AssertionError("a mid-stage restart was misclassified as a zero-update proof")

    monkeypatch.setattr(
        TRAINER_C,
        "publish_stage_n_resume_check_from_verified_invocation",
        forbid_zero_update_publication,
    )
    argv = governed_argv(
        None,
        governed["contract"],
        authorization_path,
        governed["out"],
        **{
            "--resume_path": str(source["checkpoint_path"]),
            "--resume_step": str(source_step),
        },
    ) + ["--resume_full"]
    run_main(argv, monkeypatch)

    assert record["updates"] == 0
    current_invocation = C.invocation_directory(
        governed["out"], "stage_a", C.file_sha256(authorization_path)
    )
    assert (current_invocation / f"step_{C.STAGE_A_STOP_STEP:06d}.pt").is_file()
    assert (current_invocation / C.STAGE_N_RUNTIME_FILENAME).is_file()
    assert not (current_invocation / C.STAGE_N_RESUME_RESULT_FILENAME).exists()
    assert not (current_invocation / C.STAGE_N_RESULT_FILENAME).exists()
    assert not (
        Path(source["publication"]["invocation_dir"]) / C.STAGE_N_RESUME_RESULT_FILENAME
    ).exists()
    output = capsys.readouterr().out
    assert "AWAITING_SMOKE_AND_RESUME_CHECKS" in output


def test_trainer_main_zero_update_path_publishes_canonical_stage_n_completion(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,  # noqa: F811
):
    """A verified real-main resume publishes source evidence/result with zero updates."""
    source_authorization_path = Path(governed["auth"])
    source_authorization = json.loads(source_authorization_path.read_bytes())
    source_authorization["stage_n_completion"] = {"expected_final_step": C.STAGE_A_STOP_STEP}
    source_authorization_path.write_bytes(C.canonical_json_bytes(source_authorization))
    source = _source_invocation(governed, tmp_path)
    _install_real_stage_a_data_resume_state(governed, source)
    source_invocation = Path(source["publication"]["invocation_dir"])
    source_runtime = C.publish_stage_n_runtime_artifact(
        source_invocation, source["contract"]["runtime_fingerprint"]
    )
    smoke_evidence = source_invocation / "smoke.log"
    smoke_evidence.write_bytes(b"independent bounded Stage-N smoke evidence\n")
    C.publish_stage_n_check_result(
        source_invocation,
        kind=C.STAGE_N_SMOKE_RESULT_KIND,
        authorization_path=source_authorization_path,
        governed_run_contract_path=source["publication"]["path"],
        runtime_fingerprint_path=source_runtime["path"],
        checkpoint_path=source["checkpoint_path"],
        checkpoint_step=C.STAGE_A_STOP_STEP,
        evidence_artifact_path=smoke_evidence,
    )
    authorization_path = _restart_authorization(governed, tmp_path, source)

    record = past_gate_b(monkeypatch)
    monkeypatch.setattr(trainer, "validate_resume_contract", lambda *a, **k: None)
    real_validate_data_resume_state = trainer.validate_data_resume_state
    resume_validation: dict = {"calls": 0}

    def validate_data_resume_state_spy(**kwargs):
        real_validate_data_resume_state(**kwargs)
        resume_validation["calls"] += 1
        resume_validation["sampler_state"] = kwargs["current_sampler"].state_dict()

    monkeypatch.setattr(trainer, "validate_data_resume_state", validate_data_resume_state_spy)
    argv = governed_argv(
        None,
        governed["contract"],
        authorization_path,
        governed["out"],
        **{
            "--resume_path": str(source["checkpoint_path"]),
            "--resume_step": str(C.STAGE_A_STOP_STEP),
        },
    ) + ["--resume_full"]
    run_main(argv, monkeypatch)

    assert record["updates"] == 0
    assert resume_validation["calls"] == 1
    assert resume_validation["sampler_state"]["range_start_position"] == STAGE_A_SAMPLES
    assert resume_validation["sampler_state"]["committed_position"] == STAGE_A_SAMPLES
    result_path = source_invocation / C.STAGE_N_RESULT_FILENAME
    assert result_path.is_file()
    resume_evidence_path = source_invocation / C.STAGE_N_RESUME_EVIDENCE_FILENAME
    resume_result_path = source_invocation / C.STAGE_N_RESUME_RESULT_FILENAME
    assert resume_evidence_path.is_file()
    assert resume_result_path.is_file()
    resume_evidence = json.loads(resume_evidence_path.read_bytes())
    assert resume_evidence["kind"] == C.STAGE_N_RESUME_EVIDENCE_KIND
    assert resume_evidence["source_checkpoint_path"] == str(source["checkpoint_path"])
    assert resume_evidence["resume_stage_authorization_path"] == str(authorization_path)
    result = json.loads(result_path.read_bytes())
    assert result["status"] == "COMPLETE"
    assert result["final_checkpoint_step"] == C.STAGE_A_STOP_STEP
    assert (
        C.validate_stage_n_result_against_artifacts(
            result,
            authorization_path=source_authorization_path,
            governed_run_contract_path=result["governed_run_contract_path"],
            exact_plan_path=result["exact_plan_path"],
            pilot_acceptance_path=result["pilot_acceptance_path"],
            runtime_fingerprint_path=result["runtime_fingerprint_path"],
            checkpoint_path=result["final_checkpoint_path"],
            smoke_results_path=result["smoke_results_path"],
            resume_results_path=result["resume_results_path"],
            expected_final_step=C.STAGE_A_STOP_STEP,
        )
        == []
    )


NVML = [
    {
        "physical_index": 0,
        "uuid": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "pci_bus_id": "00000000:01:00.0",
        "name": "GPU",
        "memory_total": "1 MiB",
        "driver_version": "1",
    },
    {
        "physical_index": 1,
        "uuid": "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "pci_bus_id": "00000000:02:00.0",
        "name": "GPU",
        "memory_total": "1 MiB",
        "driver_version": "1",
    },
]


def _cuda(uuid: str, pci: str) -> dict:
    return {"gpu_uuid": uuid, "gpu_pci_bus_id": pci, "gpu_name": "GPU"}


def test_uuid_form_cvd_requires_one_exact_full_cuda_uuid():
    identity = _cuda(NVML[0]["uuid"], NVML[0]["pci_bus_id"])
    assert C.resolve_selected_gpu_identity(
        cuda_visible_devices=NVML[0]["uuid"], records=NVML, cuda_identity=identity
    )["resolved"]
    ambiguous = C.resolve_selected_gpu_identity(
        cuda_visible_devices="GPU-", records=NVML, cuda_identity=identity
    )
    assert ambiguous["resolved"] is False
    assert any(
        "uuid_not" in failure or "contradicts" in failure for failure in ambiguous["failures"]
    )


def test_uuid_form_cvd_fails_when_cuda_exposes_no_uuid():
    result = C.resolve_selected_gpu_identity(
        cuda_visible_devices=NVML[0]["uuid"],
        records=NVML,
        cuda_identity=_cuda("", NVML[0]["pci_bus_id"]),
    )
    assert result["resolved"] is False
    assert "cuda_visible_devices_uuid_unverifiable_without_cuda_uuid" in result["failures"]


def test_numeric_cvd_is_still_not_an_nvml_ordinal():
    identity = _cuda(NVML[0]["uuid"], NVML[0]["pci_bus_id"])
    result = C.resolve_selected_gpu_identity(
        cuda_visible_devices="1", records=NVML, cuda_identity=identity
    )
    assert result["resolved"] is True
    assert result["selected_physical_index"] == 0
