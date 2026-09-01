"""R4 source-authority, invocation-layout, and real-main save regressions.

All trainer-main execution uses a two-layer stand-in and resumes at the exact final step, so
the training loop performs no forward, backward, or optimizer update. The production model
is never constructed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

import train_pretrain_with_bench as trainer  # noqa: E402

from pretrain import production_launch_contract_v1 as C  # noqa: E402

from .test_production_launch_contract_v1_real_path import write_authorization  # noqa: E402
from .test_production_launch_contract_v1_trainer_main import (  # noqa: E402
    _compile_evidence,
    _data_contract_for_args,
    _namespace_from,
    _ordinary_run_contract_for_args,
    _ordinary_stage_a_run_contract,
    _stage_a_data_contract_for_fixture,
    governed,  # noqa: F401,F811
    governed_argv,
    run_main,
)
from .test_production_launch_contract_v1_trainer_main_post_gate_b import (  # noqa: E402
    _tiny_model,
    past_gate_b,
)

STAGE_A_SAMPLES = C.STAGE_A_STOP_STEP * C.MICRO_BSZ * C.GRAD_ACCUM
STAGE_B_SAMPLES = (C.STAGE_B_GLOBAL_STOP_STEP - C.STAGE_B_START_STEP) * C.MICRO_BSZ * C.GRAD_ACCUM


@dataclass
class _LiveSampler:
    seed: int = C.STAGE_A_SAMPLER_SEED
    start: int = 0
    cursor: int = 0
    stop: int = STAGE_A_SAMPLES

    @property
    def range_start_position(self) -> int:
        return self.start

    @property
    def committed_position(self) -> int:
        return self.cursor

    @property
    def end_position(self) -> int:
        return self.stop


def _milestone_prefix(values: tuple[int, ...], step: int) -> list[int]:
    return [value for value in values if value <= step]


def _restorable_rng_state() -> dict:
    state = trainer.capture_rng_state()
    if state["torch_cuda"] is None:
        # CPU-only test runners still need a structurally complete governed fixture. The
        # production path requires one visible CUDA device and captures its real state.
        seed_and_offset = (20260831).to_bytes(8, "little") + (0).to_bytes(8, "little")
        state["torch_cuda"] = [torch.tensor(list(seed_and_offset), dtype=torch.uint8)]
    return state


def _gate_b() -> dict:
    return {
        "parameter_count": C.MODEL_PARAMETER_COUNT,
        "tied_embeddings": True,
        "realized_muon": C.realized_muon_contract(),
        "optimizer_group_roles": list(C.OPTIMIZER_GROUP_ROLES),
        "optimizer_membership_counts": {},
    }


def _source_invocation(
    governed,  # noqa: F811
    tmp_path: Path,
    *,
    source_step: int = C.STAGE_A_STOP_STEP,
) -> dict:
    """Publish a genuine source invocation artifact and exact tiny checkpoint."""
    assert 0 <= source_step <= C.STAGE_A_STOP_STEP
    source_cursor = source_step * C.SEQUENCES_PER_UPDATE
    args = _namespace_from(governed, governed["out"])
    gate_a = C.gate_a_pre_construction(
        args,
        stage="stage_a",
        launch_contract_path=governed["contract"],
        stage_authorization_path=governed["auth"],
        exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
        pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        observed_runtime=governed["runtime"],
        cwd=C.CANONICAL_CWD,
    )
    evidence = _compile_evidence()
    source_contract = C.build_governed_run_contract(
        gate_a=gate_a,
        gate_b=_gate_b(),
        stage="stage_a",
        sampler_identity=C.sampler_identity_document("stage_a", _LiveSampler()),
        compile_evidence=evidence,
    )
    publication = C.publish_invocation_run_contract(source_contract, gate_a=gate_a)

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
    rng_state = _restorable_rng_state()
    dynamic_state = C.build_checkpoint_state(
        stage="stage_a",
        sampler=_LiveSampler(cursor=source_cursor),
        global_step=source_step,
        completed_evaluation_milestones=_milestone_prefix(C.EVALUATION_MILESTONES, source_step),
        completed_checkpoint_milestones=_milestone_prefix(C.CHECKPOINT_MILESTONES, source_step),
        rng_state=rng_state,
        compile_evidence=evidence,
    )
    ordinary_run_contract = _ordinary_stage_a_run_contract(governed, governed["out"])
    data_contract = _stage_a_data_contract_for_fixture(governed)
    checkpoint = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": None,
        "global_step": source_step,
        "local_step": source_step,
        "config": {},
        "train_args": {},
        "run_contract": ordinary_run_contract,
        "data_contract": data_contract,
        "rng_state": rng_state,
        "position_stats": {},
        "data_sampler": {
            "version": 2,
            "data_length": data_contract["dataset_length"],
            "seed": C.STAGE_A_SAMPLER_SEED,
            "range_start_position": 0,
            "committed_position": source_cursor,
            "end_position": STAGE_A_SAMPLES,
        },
        "governed_run_contract": source_contract,
        "governed_run_contract_sha256": C.governed_digest(source_contract),
        "governed_checkpoint_state": dynamic_state,
        "kind": C.GOVERNED_CHECKPOINT_KIND,
    }
    checkpoint_path = Path(publication["invocation_dir"]) / f"step_{source_step:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return {
        "gate_a": gate_a,
        "contract": source_contract,
        "publication": publication,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "dynamic_state": dynamic_state,
    }


def _source_resume_binding(source: dict, **overrides) -> dict:
    contract = source["contract"]
    publication = source["publication"]
    checkpoint_path = source["checkpoint_path"]
    checkpoint_step = int(source["checkpoint"]["global_step"])
    dynamic_state = source["dynamic_state"]
    binding = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": C.file_sha256(checkpoint_path),
        "expected_step": checkpoint_step,
        "stage": "stage_a",
        "governed_run_contract_sha256": C.governed_digest(contract),
        "source_stage_authorization_path": contract["stage_authorization_path"],
        "source_stage_authorization_sha256": contract["stage_authorization_sha256"],
        "source_invocation_run_contract_path": publication["path"],
        "source_invocation_run_contract_sha256": publication["file_sha256"],
        "source_base_governed_identity_digest": C.base_governed_identity_sha256(contract),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": C.file_sha256(checkpoint_path),
        "source_checkpoint_step": checkpoint_step,
        "source_checkpoint_stage": "stage_a",
        "source_active_stage": "stage_a",
        "source_sampler_seed": C.STAGE_A_SAMPLER_SEED,
        "source_permutation_identity": dynamic_state["permutation_identity"],
        "source_range_start_position": dynamic_state["range_start_position"],
        "source_invocation_range_start_position": dynamic_state["invocation_range_start_position"],
        "source_range_stop_position": dynamic_state["range_stop_position"],
        "source_cursor": dynamic_state["cursor"],
    }
    binding.update(overrides)
    return binding


def _install_real_stage_a_data_resume_state(governed, source: dict) -> None:  # noqa: F811
    """Give the tiny governed source the exact ordinary sampler/data metadata main reopens."""
    args = _namespace_from(governed, governed["out"])
    train_dir = Path(trainer._resolve_path(args.train_dir))
    dataset = trainer.PackedBinDataset(
        str(train_dir),
        sampling_mode="deterministic",
        seq_len=int(args.seq_len),
        bos_id=int(args.bos_id),
        eos_id=int(args.eos_id),
        mask_bos_in_loss=not bool(args.no_mask_bos_in_loss),
        mask_last_label_in_loss=bool(args.mask_last_label_in_loss),
        require_release_manifest=True,
    )
    sampler = trainer.ResumablePermutationSampler(
        dataset,
        seed=C.STAGE_A_SAMPLER_SEED,
        start_position=0,
        num_samples=STAGE_A_SAMPLES,
    )
    sampler.commit(int(source["dynamic_state"]["cursor"]))
    source["checkpoint"]["data_contract"] = trainer.build_data_contract(train_dir, dataset, args)
    source["checkpoint"]["data_sampler"] = sampler.state_dict()
    torch.save(source["checkpoint"], source["checkpoint_path"])


def _stage_b_source_invocation(
    tmp_path: Path,
    *,
    accepted_stage_n: dict,
    governed: dict,  # noqa: F811 - fixture-shaped helper input
    source_step: int,
) -> dict:
    """Publish a real Stage-B crash source rooted in an accepted Stage-N chain."""
    assert C.STAGE_B_START_STEP <= source_step <= C.STAGE_B_GLOBAL_STOP_STEP
    source_dir = tmp_path / f"stage_b_source_{source_step}"
    source_dir.mkdir(parents=True, exist_ok=True)

    stage_n_result = accepted_stage_n["result"]
    owner_acceptance = {
        "stage_n_result_owner_verdict": "ACCEPTED",
        "accepted_stage_n_result_sha256": C.file_sha256(accepted_stage_n["result_path"]),
    }
    owner_acceptance_path = source_dir / "STAGE_N_ACCEPTANCE.json"
    owner_acceptance_path.write_bytes(C.canonical_json_bytes(owner_acceptance))
    stage_n_chain = {
        "accepted_stage_n_result_path": str(accepted_stage_n["result_path"]),
        "accepted_stage_n_result_sha256": C.file_sha256(accepted_stage_n["result_path"]),
        "stage_n_owner_acceptance_path": str(owner_acceptance_path),
        "stage_n_owner_acceptance_sha256": C.file_sha256(owner_acceptance_path),
        "stage_n_authorization_sha256": stage_n_result["stage_authorization_sha256"],
        "stage_n_governed_run_contract_sha256": stage_n_result["governed_run_contract_sha256"],
        "stage_n_governed_run_contract_artifact_sha256": stage_n_result[
            "governed_run_contract_artifact_sha256"
        ],
        "stage_n_runtime_fingerprint": stage_n_result["runtime_fingerprint"],
        "stage_n_runtime_fingerprint_sha256": stage_n_result["runtime_fingerprint_sha256"],
        "stage_n_gpu_uuid": stage_n_result["gpu_uuid"],
        "stage_n_gpu_pci_bus_id": stage_n_result["gpu_pci_bus_id"],
        "stage_n_trainer_head": stage_n_result["trainer_head"],
        "stage_n_trainer_execution_bundle_sha256": stage_n_result[
            "trainer_execution_bundle_sha256"
        ],
        "stage_n_exact_run_plan_sha256": stage_n_result["exact_run_plan_sha256"],
        "stage_n_final_checkpoint_path": stage_n_result["final_checkpoint_path"],
        "stage_n_final_checkpoint_sha256": stage_n_result["final_checkpoint_sha256"],
        "stage_n_final_checkpoint_step": stage_n_result["final_checkpoint_step"],
        "stage_n_runtime_fingerprint_path": stage_n_result["runtime_fingerprint_path"],
        "stage_n_runtime_fingerprint_artifact_sha256": stage_n_result[
            "runtime_fingerprint_artifact_sha256"
        ],
    }
    stage_a_resume = C.derive_stage_o_resume_binding(stage_n_result)
    authorization_path = write_authorization(
        source_dir,
        governed["contract"],
        allowed_scope="STAGE_O",
        transition="A_TO_B",
        allowed_output_root=str(governed["out"]),
        allowed_samples_dir=str(governed["out"] / "samples"),
        training_runtime=governed["runtime"],
        resume=stage_a_resume,
        stage_n_chain=stage_n_chain,
    )
    authorization_sha = C.file_sha256(authorization_path)
    invocation = C.invocation_directory(governed["out"], "stage_b", authorization_sha)
    evidence = _compile_evidence()
    initial_sampler = _LiveSampler(
        seed=C.STAGE_B_SAMPLER_SEED,
        stop=STAGE_B_SAMPLES,
    )
    contract = copy.deepcopy(accepted_stage_n["contract"])
    contract.update({
        "stage_authorization_path": str(authorization_path.resolve()),
        "stage_authorization_sha256": authorization_sha,
        "stage": "stage_b",
        "scope": "STAGE_O",
        "governed_run_root": str(Path(governed["out"]).resolve()),
        "invocation_root": str(invocation),
        "out_dir": str(invocation),
        "samples_dir": str(invocation / "samples"),
        "resume": stage_a_resume,
        "active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED,
        "stage_start_step": C.STAGE_B_START_STEP,
        "stage_stop_step": C.STAGE_B_GLOBAL_STOP_STEP,
        "sampler_identity": C.sampler_identity_document("stage_b", initial_sampler),
        "compile_evidence": evidence,
        "compile_evidence_sha256": evidence["compile_evidence_sha256"],
    })
    publication = C.publish_invocation_run_contract(contract)

    stage_b_argv = governed_argv(
        None,
        governed["contract"],
        authorization_path,
        governed["out"],
        **{
            "--train_dir": str(REPO / "runs/m_production_v1_2026-08-29/release/stage_b/train"),
            "--run_plan_stage": "stage_b",
            "--data_stage_start_step": str(C.STAGE_B_START_STEP),
            "--max_steps": str(C.STAGE_B_GLOBAL_STOP_STEP),
            "--resume_path": stage_a_resume["checkpoint_path"],
            "--resume_step": str(C.STAGE_A_STOP_STEP),
        },
    ) + ["--resume_full"]
    saved_argv = sys.argv
    try:
        sys.argv = stage_b_argv
        stage_b_args = trainer.parse_args()
    finally:
        sys.argv = saved_argv
    trainer.validate_training_args(stage_b_args)
    C.normalize_legacy_sampler_seed(stage_b_args, "stage_b")
    ordinary_run_contract = _ordinary_run_contract_for_args(stage_b_args)
    data_contract = _data_contract_for_args(stage_b_args)

    cursor = (source_step - C.STAGE_B_START_STEP) * C.SEQUENCES_PER_UPDATE
    live_sampler = _LiveSampler(
        seed=C.STAGE_B_SAMPLER_SEED,
        cursor=cursor,
        stop=STAGE_B_SAMPLES,
    )
    dynamic = C.build_checkpoint_state(
        stage="stage_b",
        sampler=live_sampler,
        global_step=source_step,
        completed_evaluation_milestones=_milestone_prefix(C.EVALUATION_MILESTONES, source_step),
        completed_checkpoint_milestones=_milestone_prefix(C.CHECKPOINT_MILESTONES, source_step),
        rng_state=_restorable_rng_state(),
        compile_evidence=evidence,
    )
    checkpoint = {
        "kind": C.GOVERNED_CHECKPOINT_KIND,
        "global_step": source_step,
        "local_step": source_step - C.STAGE_B_START_STEP,
        "governed_run_contract": contract,
        "governed_run_contract_sha256": C.governed_digest(contract),
        "governed_checkpoint_state": dynamic,
        "run_contract": ordinary_run_contract,
        "data_contract": data_contract,
        "data_sampler": {
            "version": 2,
            "data_length": data_contract["dataset_length"],
            "seed": C.STAGE_B_SAMPLER_SEED,
            "range_start_position": 0,
            "committed_position": cursor,
            "end_position": STAGE_B_SAMPLES,
        },
        "model": {},
        "optim": {},
        "scaler": None,
    }
    checkpoint_path = invocation / f"step_{source_step:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    source_binding = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": C.file_sha256(checkpoint_path),
        "expected_step": source_step,
        "stage": "stage_b",
        "governed_run_contract_sha256": C.governed_digest(contract),
        "source_stage_authorization_path": str(authorization_path.resolve()),
        "source_stage_authorization_sha256": authorization_sha,
        "source_invocation_run_contract_path": publication["path"],
        "source_invocation_run_contract_sha256": publication["file_sha256"],
        "source_base_governed_identity_digest": C.base_governed_identity_sha256(contract),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": C.file_sha256(checkpoint_path),
        "source_checkpoint_step": source_step,
        "source_checkpoint_stage": "stage_b",
        "source_active_stage": "stage_b",
        "source_sampler_seed": C.STAGE_B_SAMPLER_SEED,
        "source_permutation_identity": dynamic["permutation_identity"],
        "source_range_start_position": dynamic["range_start_position"],
        "source_invocation_range_start_position": dynamic["invocation_range_start_position"],
        "source_range_stop_position": dynamic["range_stop_position"],
        "source_cursor": dynamic["cursor"],
    }
    return {
        "authorization": json.loads(authorization_path.read_bytes()),
        "authorization_path": authorization_path,
        "grc": contract,
        "grc_path": Path(publication["path"]),
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "source_binding": source_binding,
        "stage_n_chain": stage_n_chain,
        "owner_acceptance_path": owner_acceptance_path,
    }


def _restart_authorization(governed, tmp_path: Path, source: dict, **resume_overrides):  # noqa: F811
    auth_dir = tmp_path / "restart_authorization"
    auth_dir.mkdir(parents=True, exist_ok=True)
    return write_authorization(
        auth_dir,
        governed["contract"],
        allowed_output_root=str(governed["out"]),
        allowed_samples_dir=str(governed["out"] / "samples"),
        training_runtime=governed["runtime"],
        resume=_source_resume_binding(source, **resume_overrides),
    )


def _restart_gate(governed, auth: Path, source: dict):  # noqa: F811
    argv = governed_argv(
        None,
        governed["contract"],
        auth,
        governed["out"],
        **{
            "--resume_path": str(source["checkpoint_path"]),
            "--resume_step": str(C.STAGE_A_STOP_STEP),
        },
    ) + ["--resume_full"]
    saved = sys.argv
    try:
        sys.argv = argv
        args = trainer.parse_args()
    finally:
        sys.argv = saved
    trainer.validate_training_args(args)
    C.normalize_legacy_sampler_seed(args, "stage_a")
    gate_a = C.gate_a_pre_construction(
        args,
        stage="stage_a",
        launch_contract_path=governed["contract"],
        stage_authorization_path=auth,
        exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
        pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        observed_runtime=governed["runtime"],
        cwd=C.CANONICAL_CWD,
    )
    return args, gate_a


def test_live_sampler_seed_and_full_rng_are_load_bearing():
    evidence = _compile_evidence()
    rng_state = _restorable_rng_state()
    state = C.build_checkpoint_state(
        stage="stage_a",
        sampler=_LiveSampler(cursor=128),
        global_step=1,
        completed_evaluation_milestones=[],
        completed_checkpoint_milestones=[],
        rng_state=rng_state,
        compile_evidence=evidence,
    )
    assert state["active_stage_sampler_seed"] == C.STAGE_A_SAMPLER_SEED
    assert set(state["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert C.validate_governed_checkpoint_state(state, checkpoint_global_step=1) == []

    invalid_rng = dict(rng_state)
    invalid_rng["python"] = 1
    with pytest.raises(C.LaunchContractError, match="python_rng_not_restorable"):
        C.build_checkpoint_state(
            stage="stage_a",
            sampler=_LiveSampler(cursor=128),
            global_step=1,
            completed_evaluation_milestones=[],
            completed_checkpoint_milestones=[],
            rng_state=invalid_rng,
            compile_evidence=evidence,
        )
    with pytest.raises(C.LaunchContractError, match="global_step"):
        C.build_checkpoint_state(
            stage="stage_a",
            sampler=_LiveSampler(cursor=128),
            global_step=True,
            completed_evaluation_milestones=[],
            completed_checkpoint_milestones=[],
            rng_state=rng_state,
            compile_evidence=evidence,
        )

    string_position = dict(state)
    string_position["cursor"] = "128"
    assert (
        "governed_checkpoint_dynamic_state_sampler_positions_invalid"
        in C.validate_governed_checkpoint_state(string_position, checkpoint_global_step=1)
    )

    with pytest.raises(C.LaunchContractError, match="live sampler seed"):
        C.sampler_identity_document("stage_a", _LiveSampler(seed=C.STAGE_B_SAMPLER_SEED))


def test_cpu_rng_serialization_is_not_a_cuda_rng_state():
    state = _restorable_rng_state()
    state["torch_cuda"] = [state["torch_cpu"].clone()]
    assert (
        "governed_checkpoint_dynamic_state_torch_cuda_rng_not_restorable"
        in C.validate_restorable_rng_state(state)
    )


@pytest.mark.parametrize("num_bytes", [15, 17])
def test_cuda_rng_state_requires_exact_frozen_serialization_size(num_bytes):
    assert C.validate_cuda_rng_states([torch.zeros(num_bytes, dtype=torch.uint8)]) == [
        "governed_checkpoint_dynamic_state_torch_cuda_rng_not_restorable"
    ]


def test_cuda_rng_structural_validation_cannot_claim_live_restorability(monkeypatch):
    state = torch.zeros(C.CUDA_RNG_STATE_NUM_BYTES, dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert C.validate_cuda_rng_states([state]) == []
    assert C.validate_cuda_rng_states([state], require_live_validation=True) == [
        "governed_checkpoint_dynamic_state_torch_cuda_rng_not_restorable"
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA generator")
def test_cuda_rng_live_validation_uses_an_isolated_generator():
    process_state = torch.cuda.get_rng_state(0).clone()
    candidate = torch.Generator(device="cuda:0").manual_seed(20260831).get_state()
    assert C.validate_cuda_rng_states([candidate], require_live_validation=True) == []
    assert torch.equal(torch.cuda.get_rng_state(0), process_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA restore preflight")
def test_restore_rejects_bad_cuda_rng_before_mutating_global_streams():
    state = _restorable_rng_state()
    state["torch_cuda"] = [state["torch_cpu"].clone()]
    python_before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="CUDA RNG state is incompatible"):
        trainer.restore_rng_state(state)
    assert random.getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_source_bound_same_stage_resume_rejects_coupled_base_rewrite(
    governed,  # noqa: F811
    tmp_path,  # noqa: F811
):
    source = _source_invocation(governed, tmp_path)
    auth = _restart_authorization(governed, tmp_path, source)
    _, gate_a = _restart_gate(governed, auth, source)
    current = C.build_governed_run_contract(
        gate_a=gate_a,
        gate_b=_gate_b(),
        stage="stage_a",
    )
    expected_sampler = C.expected_sampler_identity_for_resume(
        "stage_a",
        expected_step=C.STAGE_A_STOP_STEP,
        data_stage_start_step=0,
        micro_bsz=C.MICRO_BSZ,
        grad_accum=C.GRAD_ACCUM,
        planned_stage_samples=STAGE_A_SAMPLES,
    )
    valid = C.validate_governed_checkpoint_before_restore(
        source["checkpoint"],
        current,
        expected_resume=gate_a["resume"],
        current_sampler_identity=expected_sampler,
        expected_global_step=C.STAGE_A_STOP_STEP,
        verified_source_authority=gate_a["verified_source_authority"],
    )
    assert valid["compatible"], valid["failures"]

    rewritten = copy.deepcopy(source["checkpoint"])
    rewritten["governed_run_contract"]["training"]["peak_lr"] = 0.0003
    rewritten["governed_run_contract_sha256"] = C.governed_digest(
        rewritten["governed_run_contract"]
    )
    refused = C.validate_governed_checkpoint_before_restore(
        rewritten,
        {**current, "training": rewritten["governed_run_contract"]["training"]},
        expected_resume=gate_a["resume"],
        current_sampler_identity=expected_sampler,
        expected_global_step=C.STAGE_A_STOP_STEP,
        verified_source_authority=gate_a["verified_source_authority"],
    )
    assert not refused["compatible"]
    assert "current_base_identity_differs_from_verified_source" in refused["failures"]
    assert "checkpoint_contract_differs_from_verified_source_artifact" in refused["failures"]


def test_source_invocation_range_start_is_authenticated_not_merely_before_cursor(
    governed,  # noqa: F811
    tmp_path,  # noqa: F811
):
    source = _source_invocation(governed, tmp_path)
    auth = _restart_authorization(
        governed,
        tmp_path,
        source,
        source_invocation_range_start_position=777,
    )
    with pytest.raises(C.LaunchContractError, match="source_authority_sampler_binding_mismatch"):
        _restart_gate(governed, auth, source)


def test_source_authorization_runtime_must_match_published_contract(
    governed,  # noqa: F811
    tmp_path,  # noqa: F811
):
    """Two rehashed source artifacts still fail if their bound runtime claims disagree."""
    source = _source_invocation(governed, tmp_path)

    authorization = json.loads(
        Path(source["contract"]["stage_authorization_path"]).read_text(encoding="utf-8")
    )
    authorization["training_runtime"] = {
        **authorization["training_runtime"],
        "torch_version": "coordinated-but-contradictory-runtime",
    }
    drift_auth_path = tmp_path / "drift_source_authorization.json"
    drift_auth_path.write_bytes(C.canonical_json_bytes(authorization))
    drift_auth_sha = C.file_sha256(drift_auth_path)

    drift_contract = copy.deepcopy(source["contract"])
    drift_invocation = C.invocation_directory(
        drift_contract["governed_run_root"], "stage_a", drift_auth_sha
    )
    drift_contract.update({
        "stage_authorization_path": str(drift_auth_path.resolve()),
        "stage_authorization_sha256": drift_auth_sha,
        "invocation_root": str(drift_invocation),
        "out_dir": str(drift_invocation),
        "samples_dir": str(drift_invocation / "samples"),
    })
    drift_publication = C.publish_invocation_run_contract(drift_contract)

    drift_checkpoint = copy.deepcopy(source["checkpoint"])
    drift_checkpoint["governed_run_contract"] = drift_contract
    drift_checkpoint["governed_run_contract_sha256"] = C.governed_digest(drift_contract)
    drift_checkpoint_path = drift_invocation / f"step_{C.STAGE_A_STOP_STEP:06d}.pt"
    torch.save(drift_checkpoint, drift_checkpoint_path)

    binding = _source_resume_binding(source)
    binding.update({
        "checkpoint_path": str(drift_checkpoint_path),
        "checkpoint_sha256": C.file_sha256(drift_checkpoint_path),
        "governed_run_contract_sha256": C.governed_digest(drift_contract),
        "source_stage_authorization_path": str(drift_auth_path.resolve()),
        "source_stage_authorization_sha256": drift_auth_sha,
        "source_invocation_run_contract_path": drift_publication["path"],
        "source_invocation_run_contract_sha256": drift_publication["file_sha256"],
        "source_base_governed_identity_digest": C.base_governed_identity_sha256(drift_contract),
        "source_checkpoint_path": str(drift_checkpoint_path),
        "source_checkpoint_sha256": C.file_sha256(drift_checkpoint_path),
    })
    verdict = C.verify_resume_source_authority(binding, transition=None)
    assert not verdict["verified"]
    assert "source_authority_authorization_run_contract_runtime_mismatch" in verdict["failures"]


def test_stage_b_exact_restart_uses_authenticated_same_stage_rule(
    governed,  # noqa: F811
    tmp_path,  # noqa: F811
):
    """A Stage-O crash checkpoint is not misclassified as another Stage-A handoff."""
    stage_a_source = _source_invocation(governed, tmp_path)
    evidence = _compile_evidence()
    step = C.STAGE_B_START_STEP + 1
    saved_sampler = _LiveSampler(
        seed=C.STAGE_B_SAMPLER_SEED,
        start=0,
        cursor=C.MICRO_BSZ * C.GRAD_ACCUM,
        stop=STAGE_B_SAMPLES,
    )
    state = C.build_checkpoint_state(
        stage="stage_b",
        sampler=saved_sampler,
        global_step=step,
        completed_evaluation_milestones=_milestone_prefix(C.EVALUATION_MILESTONES, step),
        completed_checkpoint_milestones=_milestone_prefix(C.CHECKPOINT_MILESTONES, step),
        rng_state=_restorable_rng_state(),
        compile_evidence=evidence,
    )

    source_contract = copy.deepcopy(stage_a_source["contract"])
    source_contract.update({
        "stage": "stage_b",
        "scope": "STAGE_O",
        "active_stage_sampler_seed": C.STAGE_B_SAMPLER_SEED,
        "stage_start_step": C.STAGE_B_START_STEP,
        "stage_stop_step": C.STAGE_B_GLOBAL_STOP_STEP,
        "sampler_identity": C.sampler_identity_document(
            "stage_b",
            _LiveSampler(seed=C.STAGE_B_SAMPLER_SEED, stop=STAGE_B_SAMPLES),
        ),
        "compile_evidence": evidence,
        "compile_evidence_sha256": evidence["compile_evidence_sha256"],
    })
    digest = C.governed_digest(source_contract)
    base_digest = C.base_governed_identity_sha256(source_contract)
    ordinary_run_contract = copy.deepcopy(stage_a_source["checkpoint"]["run_contract"])
    ordinary_run_contract["sampler_seed"] = C.STAGE_B_SAMPLER_SEED
    ordinary_run_contract["run_plan"] = {
        **ordinary_run_contract["run_plan"],
        "stage": "stage_b",
    }
    data_contract = copy.deepcopy(stage_a_source["checkpoint"]["data_contract"])
    data_contract.update({
        "sampler_seed": C.STAGE_B_SAMPLER_SEED,
        "active_stage": "stage_b",
        "data_stage_start_step": C.STAGE_B_START_STEP,
    })
    checkpoint = {
        "kind": C.GOVERNED_CHECKPOINT_KIND,
        "global_step": step,
        "governed_run_contract": source_contract,
        "governed_run_contract_sha256": digest,
        "governed_checkpoint_state": state,
        "model": {},
        "optim": {},
        "scaler": None,
        "run_contract": ordinary_run_contract,
        "data_contract": data_contract,
        "data_sampler": {
            "version": 2,
            "data_length": data_contract["dataset_length"],
            "seed": C.STAGE_B_SAMPLER_SEED,
            "range_start_position": 0,
            "committed_position": C.SEQUENCES_PER_UPDATE,
            "end_position": STAGE_B_SAMPLES,
        },
    }
    expected_resume = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "expected_step": step,
        "stage": "stage_b",
        "governed_run_contract_sha256": digest,
        "source_base_governed_identity_digest": base_digest,
        "source_checkpoint_step": step,
        "source_checkpoint_stage": "stage_b",
        "source_active_stage": "stage_b",
        "source_sampler_seed": C.STAGE_B_SAMPLER_SEED,
        "source_permutation_identity": state["permutation_identity"],
        "source_range_start_position": 0,
        "source_invocation_range_start_position": 0,
        "source_range_stop_position": STAGE_B_SAMPLES,
        "source_cursor": C.MICRO_BSZ * C.GRAD_ACCUM,
    }
    verified_source = {
        "verified": True,
        "source_invocation_run_contract": source_contract,
        "source_base_governed_identity_digest": base_digest,
    }
    authorization = {"transition": None, "resume": expected_resume}
    assert (
        C.validate_transition_declaration(
            authorization,
            scope="STAGE_O",
            verified_source_authority=verified_source,
        )
        == []
    )
    assert C.validate_transition_declaration(
        {**authorization, "transition": "A_TO_B"},
        scope="STAGE_O",
        verified_source_authority=verified_source,
    )

    current_sampler = C.expected_sampler_identity_for_resume(
        "stage_b",
        expected_step=step,
        data_stage_start_step=C.STAGE_B_START_STEP,
        micro_bsz=C.MICRO_BSZ,
        grad_accum=C.GRAD_ACCUM,
        planned_stage_samples=STAGE_B_SAMPLES,
    )
    verdict = C.validate_governed_checkpoint_before_restore(
        checkpoint,
        source_contract,
        expected_resume=expected_resume,
        current_sampler_identity=current_sampler,
        stage_transition=None,
        expected_global_step=step,
        verified_source_authority=verified_source,
    )
    assert verdict["compatible"], verdict["failures"]


def test_a_to_b_source_schema_is_mandatory():
    args = type(
        "Args",
        (),
        {"resume_path": "/tmp/source.pt", "resume_full": True, "resume_step": 38146},
    )()
    legacy = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": "/tmp/source.pt",
        "checkpoint_sha256": "a" * 64,
        "expected_step": 38146,
        "stage": "stage_a",
        "governed_run_contract_sha256": "b" * 64,
    }
    failures = C.validate_resume_binding(
        args,
        legacy,
        require_source_authority=True,
        transition="A_TO_B",
    )
    assert len([f for f in failures if "missing_source_field" in f]) == len(
        C.A_TO_B_SOURCE_REQUIRED_FIELDS
    )


def test_restarted_stage_a_sampler_proves_completion_at_a_to_b_boundary():
    """The source invocation start must not be added twice to its absolute cursor."""
    restart_position = 10_000 * C.MICRO_BSZ * C.GRAD_ACCUM
    source = trainer.ResumablePermutationSampler(
        range(11),
        seed=C.STAGE_A_SAMPLER_SEED,
        start_position=restart_position,
        num_samples=STAGE_A_SAMPLES - restart_position,
    )
    source.commit(STAGE_A_SAMPLES - restart_position)
    stage_b = trainer.ResumablePermutationSampler(
        range(13),
        seed=C.STAGE_B_SAMPLER_SEED,
        start_position=0,
        num_samples=STAGE_B_SAMPLES,
    )

    trainer.validate_data_resume_state(
        saved_data_contract={
            "fingerprint": "stage-a",
            "dataset_length": 11,
            "data_stage_start_step": 0,
            "samples_per_optimizer_step": C.MICRO_BSZ * C.GRAD_ACCUM,
        },
        current_data_contract={
            "fingerprint": "stage-b",
            "data_stage_start_step": C.STAGE_A_STOP_STEP,
            "samples_per_optimizer_step": C.MICRO_BSZ * C.GRAD_ACCUM,
        },
        saved_sampler_state=source.state_dict(),
        current_sampler=stage_b,
        global_step=C.STAGE_A_STOP_STEP,
        data_stage_start_step=C.STAGE_A_STOP_STEP,
        strict=True,
        preserve_invocation_range_start=True,
        governed_checkpoint_state={
            "active_stage_sampler_seed": C.STAGE_A_SAMPLER_SEED,
            "invocation_range_start_position": restart_position,
            "cursor": STAGE_A_SAMPLES,
            "range_stop_position": STAGE_A_SAMPLES,
        },
    )

    assert stage_b.seed == C.STAGE_B_SAMPLER_SEED
    assert stage_b.range_start_position == 0
    assert stage_b.committed_position == 0


def test_a_to_b_boundary_rejects_source_invocation_start_after_cursor():
    source = trainer.ResumablePermutationSampler(
        range(11),
        seed=C.STAGE_A_SAMPLER_SEED,
        start_position=0,
        num_samples=STAGE_A_SAMPLES,
    ).state_dict()
    source.update({
        "range_start_position": STAGE_A_SAMPLES + 1,
        "committed_position": STAGE_A_SAMPLES,
        "end_position": STAGE_A_SAMPLES,
    })
    stage_b = trainer.ResumablePermutationSampler(
        range(13),
        seed=C.STAGE_B_SAMPLER_SEED,
        num_samples=STAGE_B_SAMPLES,
    )

    with pytest.raises(RuntimeError, match="operational_sampler_range_invalid"):
        trainer.validate_data_resume_state(
            saved_data_contract={
                "fingerprint": "stage-a",
                "dataset_length": 11,
                "data_stage_start_step": 0,
                "samples_per_optimizer_step": C.MICRO_BSZ * C.GRAD_ACCUM,
            },
            current_data_contract={
                "fingerprint": "stage-b",
                "data_stage_start_step": C.STAGE_A_STOP_STEP,
                "samples_per_optimizer_step": C.MICRO_BSZ * C.GRAD_ACCUM,
            },
            saved_sampler_state=source,
            current_sampler=stage_b,
            global_step=C.STAGE_A_STOP_STEP,
            data_stage_start_step=C.STAGE_A_STOP_STEP,
            strict=True,
            preserve_invocation_range_start=True,
            governed_checkpoint_state={
                "active_stage_sampler_seed": C.STAGE_A_SAMPLER_SEED,
                "invocation_range_start_position": STAGE_A_SAMPLES + 1,
                "cursor": STAGE_A_SAMPLES,
                "range_stop_position": STAGE_A_SAMPLES,
            },
        )


def test_governed_operational_sampler_start_must_match_dynamic_source_identity():
    saved = trainer.ResumablePermutationSampler(
        range(11),
        seed=C.STAGE_A_SAMPLER_SEED,
        start_position=777,
        num_samples=1_271,
    )
    saved.commit(247)
    current = trainer.ResumablePermutationSampler(
        range(11),
        seed=C.STAGE_A_SAMPLER_SEED,
        start_position=1_024,
        num_samples=1_024,
    )
    contract = {
        "fingerprint": "stage-a",
        "dataset_length": 11,
        "sampling_mode": "deterministic",
        "sampler_seed": C.STAGE_A_SAMPLER_SEED,
        "data_stage_start_step": 0,
        "samples_per_optimizer_step": C.SEQUENCES_PER_UPDATE,
    }

    with pytest.raises(RuntimeError, match="range_start_position:invocation_range_start_position"):
        trainer.validate_data_resume_state(
            saved_data_contract=contract,
            current_data_contract=dict(contract),
            saved_sampler_state=saved.state_dict(),
            current_sampler=current,
            global_step=8,
            data_stage_start_step=0,
            strict=True,
            preserve_invocation_range_start=True,
            governed_checkpoint_state={
                "active_stage_sampler_seed": C.STAGE_A_SAMPLER_SEED,
                "invocation_range_start_position": 0,
                "cursor": 1_024,
                "range_stop_position": 2_048,
            },
        )


def test_real_main_final_save_has_live_resumable_state_and_zero_updates(
    governed,  # noqa: F811
    tmp_path,
    monkeypatch,  # noqa: F811
):
    source = _source_invocation(governed, tmp_path)
    _install_real_stage_a_data_resume_state(governed, source)
    auth = _restart_authorization(governed, tmp_path, source)
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
        auth,
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
    restart_auth_sha = C.file_sha256(auth)
    invocation_root = C.invocation_directory(governed["out"], "stage_a", restart_auth_sha)
    saved_path = invocation_root / f"step_{C.STAGE_A_STOP_STEP:06d}.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    dynamic = saved["governed_checkpoint_state"]
    assert saved["data_sampler"]["range_start_position"] == STAGE_A_SAMPLES
    assert saved["data_sampler"]["committed_position"] == STAGE_A_SAMPLES
    assert dynamic["range_start_position"] == 0
    assert dynamic["invocation_range_start_position"] == STAGE_A_SAMPLES
    assert dynamic["cursor"] == STAGE_A_SAMPLES
    assert dynamic["completed_evaluation_milestones"] == _milestone_prefix(
        C.EVALUATION_MILESTONES, C.STAGE_A_STOP_STEP
    )
    assert dynamic["completed_checkpoint_milestones"] == _milestone_prefix(
        C.CHECKPOINT_MILESTONES, C.STAGE_A_STOP_STEP
    )
    assert set(dynamic["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert C.verify_compile_evidence_document(dynamic["compile_evidence"]) == []
    assert (
        C.validate_governed_checkpoint_state(
            dynamic,
            governed_run_contract=saved["governed_run_contract"],
            checkpoint_global_step=C.STAGE_A_STOP_STEP,
        )
        == []
    )

    # The source of A->B is this restarted invocation, whose LIVE invocation start is the
    # recovered terminal cursor while its canonical Stage-A permutation range still starts 0.
    source_contract = saved["governed_run_contract"]
    source_contract_path = invocation_root / C.GOVERNED_RUN_CONTRACT_FILENAME
    a_to_b_binding = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": str(saved_path),
        "checkpoint_sha256": C.file_sha256(saved_path),
        "expected_step": C.STAGE_A_STOP_STEP,
        "stage": "stage_a",
        "governed_run_contract_sha256": C.governed_digest(source_contract),
        "source_stage_authorization_path": str(auth.resolve()),
        "source_stage_authorization_sha256": restart_auth_sha,
        "source_invocation_run_contract_path": str(source_contract_path),
        "source_invocation_run_contract_sha256": C.file_sha256(source_contract_path),
        "source_base_governed_identity_digest": C.base_governed_identity_sha256(source_contract),
        "source_checkpoint_path": str(saved_path),
        "source_checkpoint_sha256": C.file_sha256(saved_path),
        "source_checkpoint_step": C.STAGE_A_STOP_STEP,
        "source_checkpoint_stage": "stage_a",
        "source_active_stage": "stage_a",
        "source_sampler_seed": C.STAGE_A_SAMPLER_SEED,
        "source_permutation_identity": dynamic["permutation_identity"],
        "source_range_start_position": 0,
        "source_invocation_range_start_position": STAGE_A_SAMPLES,
        "source_range_stop_position": STAGE_A_SAMPLES,
        "source_cursor": STAGE_A_SAMPLES,
    }
    source_verdict = C.verify_resume_source_authority(a_to_b_binding, transition="A_TO_B")
    assert source_verdict["verified"], source_verdict["failures"]
    assert (
        C.validate_stage_a_to_b_transition(
            source_contract,
            dynamic,
            source_binding=a_to_b_binding,
        )
        == []
    )
