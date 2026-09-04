"""Real, bounded Stage-O parser/chain/Gate-A rehearsal for the launch adapter.

The opt-in rehearsal deliberately supplies a NOT_AUTHORIZED Stage-O artifact.  It enters
the accepted trainer's real parser and Gate A and requires every machine-checkable binding
to pass, leaving only the owner's authorization-state refusal.  It never calls trainer
``main`` and installs tripwires on every construction or training surface anyway.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

ADAPTER_ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_ROOT = Path("/workspace/petitgpt_stage_n_result_publication_recovery_v1")
HISTORICAL_ROOT = Path("/workspace/petitgpt")
STAGE_N_RESULT_PATH = (
    ACCEPTED_ROOT / "runs/n3_bridge_output_r3_2026-09-04/STAGE_N_COMPLETE_RESULT.json"
)
STAGE_N_RESULT_SHA256 = "3f2d9029286bf9d0f8abe704aedef60e812d98efdac4049d6fbdff16895398d2"
STAGE_N_ACCEPTANCE_PATH = (
    ACCEPTED_ROOT
    / "runs/n_stage_n_owner_closeout_and_stage_o_preflight_v1_2026-09-04"
    / "STAGE_N_OWNER_ACCEPTANCE.json"
)
STAGE_N_ACCEPTANCE_SHA256 = "0aec8cffd6e7f3395017b887523ad3a9dc0a109cf7744901a08320d58bfd90a6"
LAUNCH_CONTRACT_PATH = (
    HISTORICAL_ROOT
    / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03"
    / "LAUNCH_CONTRACT.json"
)
EXPECTED_ACCEPTED_HEAD = "7686fd811642dd6246ca3a3c21a4bf43bc28cd3b"
EXPECTED_ACCEPTED_BUNDLE = "1086af0b6821b2fdc4b2850371845c992f831dfcd84a6d504d2938fad003e75d"

sys.path.insert(0, str(ADAPTER_ROOT))
from tools import stage_o_successor_launch_adapter_v1 as A  # noqa: E402


class _ForbiddenBoundary(BaseException):
    """A BaseException tripwire cannot be mistaken for a handled trainer failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical_json(contract: Any, path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(document))
    return path


def _accepted_stage_o_chain(contract: Any, stage_n: dict[str, Any]) -> dict[str, Any]:
    bridge = stage_n["stage_n_compatibility_bridge"]
    return {
        "accepted_stage_n_result_path": str(STAGE_N_RESULT_PATH),
        "accepted_stage_n_result_sha256": STAGE_N_RESULT_SHA256,
        "stage_n_owner_acceptance_path": str(STAGE_N_ACCEPTANCE_PATH),
        "stage_n_owner_acceptance_sha256": STAGE_N_ACCEPTANCE_SHA256,
        "stage_n_authorization_sha256": stage_n["stage_authorization_sha256"],
        "stage_n_governed_run_contract_sha256": stage_n["governed_run_contract_sha256"],
        "stage_n_governed_run_contract_artifact_sha256": stage_n[
            "governed_run_contract_artifact_sha256"
        ],
        "stage_n_runtime_fingerprint": stage_n["runtime_fingerprint"],
        "stage_n_runtime_fingerprint_sha256": stage_n["runtime_fingerprint_sha256"],
        "stage_n_runtime_fingerprint_path": stage_n["runtime_fingerprint_path"],
        "stage_n_runtime_fingerprint_artifact_sha256": stage_n[
            "runtime_fingerprint_artifact_sha256"
        ],
        "stage_n_gpu_uuid": stage_n["gpu_uuid"],
        "stage_n_gpu_pci_bus_id": stage_n["gpu_pci_bus_id"],
        "stage_n_trainer_head": stage_n["trainer_head"],
        "stage_n_trainer_execution_bundle_sha256": stage_n["trainer_execution_bundle_sha256"],
        "stage_n_exact_run_plan_sha256": stage_n["exact_run_plan_sha256"],
        "stage_n_final_checkpoint_path": stage_n["final_checkpoint_path"],
        "stage_n_final_checkpoint_sha256": stage_n["final_checkpoint_sha256"],
        "stage_n_final_checkpoint_step": stage_n["final_checkpoint_step"],
        "n3_bridge_authorization_path": bridge["authorization_path"],
        "n3_bridge_authorization_sha256": bridge["authorization_sha256"],
        "n3_bridge_semantic_comparison_manifest_path": bridge["semantic_comparison_manifest_path"],
        "n3_bridge_semantic_comparison_manifest_sha256": bridge[
            "semantic_comparison_manifest_sha256"
        ],
        "n3_bridge_governed_run_contract_path": bridge["governed_run_contract_path"],
        "n3_bridge_governed_run_contract_artifact_sha256": bridge[
            "governed_run_contract_artifact_sha256"
        ],
        "n3_bridge_governed_run_contract_sha256": bridge["governed_run_contract_sha256"],
        "n3_bridge_terminal_checkpoint_path": bridge["terminal_checkpoint_path"],
        "n3_bridge_terminal_checkpoint_sha256": bridge["terminal_checkpoint_sha256"],
        "n3_bridge_terminal_checkpoint_step": bridge["terminal_checkpoint_step"],
        "stage_n_training_execution_head": stage_n["stage_n_training_execution_head"],
        "stage_n_training_execution_bundle": stage_n["stage_n_training_execution_bundle"],
        "stage_n_compatibility_bridge_head": stage_n["stage_n_compatibility_bridge_head"],
        "stage_n_compatibility_bridge_bundle": stage_n["stage_n_compatibility_bridge_bundle"],
        "stage_o_execution_head": stage_n["stage_o_execution_head"],
        "stage_o_execution_bundle": stage_n["stage_o_execution_bundle"],
        "stage_n_source_history": stage_n["stage_n_source_history"],
    }


def _write_stage_o_not_authorized(
    contract: Any,
    tmp_path: Path,
    *,
    runtime: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    assert _sha256(STAGE_N_RESULT_PATH) == STAGE_N_RESULT_SHA256
    assert _sha256(STAGE_N_ACCEPTANCE_PATH) == STAGE_N_ACCEPTANCE_SHA256
    stage_n = json.loads(STAGE_N_RESULT_PATH.read_bytes())
    run_root = tmp_path / "stage_o_governed"

    authorization = contract.authorization_template()
    authorization.update({
        "allowed_scope": "STAGE_O",
        "repository_branch": contract.STAGE_N_SUCCESSOR_COMPATIBILITY_BRANCH,
        "trainer_head": stage_n["stage_o_execution_head"],
        "trainer_execution_bundle_sha256": stage_n["stage_o_execution_bundle"],
        "launch_contract_sha256": contract.file_sha256(LAUNCH_CONTRACT_PATH),
        "allowed_output_root": str(run_root),
        "allowed_samples_dir": str(run_root / "samples"),
        "training_runtime": runtime,
        "resume": contract.derive_stage_o_resume_binding(stage_n),
        "transition": "A_TO_B",
        "stage_n_successor_bridge_chain": _accepted_stage_o_chain(contract, stage_n),
    })
    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert authorization["authorized_by"] is None
    assert authorization["authorized_at"] is None
    path = tmp_path / "STAGE_O_AUTHORIZATION_NOT_AUTHORIZED.json"
    return path, authorization, stage_n


def _trainer_argv(
    contract: Any,
    *,
    stage_o_authorization_path: Path,
    stage_n: dict[str, Any],
    run_root: Path,
) -> list[str]:
    values = {
        "--train_dir": str(
            HISTORICAL_ROOT / "runs/m_production_v1_2026-08-29/release/stage_b/train"
        ),
        "--val_dir": str(HISTORICAL_ROOT / "runs/g2_production_2026-08-21/release/val"),
        "--out_dir": str(run_root),
        "--samples_dir": str(run_root / "samples"),
        "--tokenizer_path": str(HISTORICAL_ROOT / contract.TOKENIZER_RELPATH),
        "--run_plan_json": str(HISTORICAL_ROOT / contract.EXACT_RUN_PLAN_RELPATH),
        "--run_plan_stage": "stage_b",
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
        "--data_stage_start_step": "38146",
        "--max_steps": "49590",
        "--precision": "bf16",
        "--eval_every": "0",
        "--save_every": "0",
        "--seed": str(contract.MODEL_INIT_SEED),
        "--val_seed": str(contract.VALIDATION_SEED),
        "--stage_a_sampler_seed": str(contract.STAGE_A_SAMPLER_SEED),
        "--stage_b_sampler_seed": str(contract.STAGE_B_SAMPLER_SEED),
        "--num_workers": "2",
        "--val_samples": "0",
        "--val_samples_per_source": "0",
        "--launch_contract_json": str(LAUNCH_CONTRACT_PATH),
        "--stage_authorization_json": str(stage_o_authorization_path),
        "--resume_path": stage_n["final_checkpoint_path"],
    }
    argv: list[str] = []
    for key, value in values.items():
        argv.extend((key, value))
    argv.extend(("--resume_full", "--compile"))
    argv.extend(contract.save_steps_cli_flags())
    argv.extend(contract.eval_steps_cli_flags())
    return argv


def _forbidden_boundary_profiler(
    topology: Any,
) -> tuple[Any, list[str]]:
    trainer = topology.trainer
    contract = topology.launch_contract
    forbidden_codes: dict[Any, str] = {}

    def register(label: str, target: Any) -> None:
        code = getattr(target, "__code__", None)
        assert code is not None, f"forbidden boundary has no Python code object:{label}"
        assert code not in forbidden_codes, f"forbidden boundary code is not unique:{label}"
        forbidden_codes[code] = label

    # set_seed is the first call after Gate A in the accepted trainer main path. The
    # remaining code-object sentinels make later construction/training boundaries
    # explicit without mutating any adapter-owned module namespace or class.
    register("trainer.set_seed", trainer.set_seed)
    register("trainer.GPTConfig.__init__", trainer.GPTConfig.__init__)
    register("trainer.GPT.__init__", trainer.GPT.__init__)
    register("trainer.GPT.forward", trainer.GPT.forward)
    register("trainer.build_optimizer", trainer.build_optimizer)
    register("trainer.load_ckpt", trainer.load_ckpt)
    register(
        "launch.bind_compiled_callable_governed",
        contract.bind_compiled_callable_governed,
    )
    register(
        "launch.realize_compile_production_shape",
        contract.realize_compile_production_shape,
    )
    register("torch.compile", trainer.torch.compile)
    register("torch.Tensor.backward", trainer.torch.Tensor.backward)
    register("torch.optim.Optimizer.step", trainer.torch.optim.Optimizer.step)

    violations: list[str] = []

    def profiler(frame: Any, event: str, _arg: Any) -> Any:
        if event == "call" and frame.f_code in forbidden_codes:
            label = forbidden_codes[frame.f_code]
            violations.append(label)
            raise _ForbiddenBoundary(f"construction or training boundary reached:{label}")
        return profiler

    return profiler, violations


def test_adapter_authorization_template_is_not_execution_or_training_authority() -> None:
    template = A.adapter_authorization_template()

    assert (
        template["schema_version"] == "petitgpt-stage-o-successor-launch-adapter-authorization-v1"
    )
    assert template["authorization_status"] == "NOT_AUTHORIZED"
    assert template["authorizes_adapter_execution"] is False
    assert template["authorizes_training"] is False
    assert template["canonical_cwd"] == str(HISTORICAL_ROOT)
    assert template["accepted_stage_o_trainer"]["head"] == EXPECTED_ACCEPTED_HEAD
    assert (
        template["accepted_stage_o_trainer"]["trainer_execution_bundle_sha256"]
        == EXPECTED_ACCEPTED_BUNDLE
    )
    assert template["module_names"]["canonical_launch_contract"] == (
        "pretrain.production_launch_contract_v1"
    )
    assert template["module_names"]["bare_launch_contract"] == ("production_launch_contract_v1")


def test_adapter_not_authorized_template_has_only_owner_state_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The committed adapter requires its live script to be tracked and the worktree
    # tracked-clean even in preflight.  Preserve every real identity byte here while
    # modeling that final reviewed state for this pre-commit structural test.
    adapter_identity = A.adapter_identity()
    adapter_identity.update({"tracked_clean": True, "script_tracked": True})
    monkeypatch.setattr(A, "adapter_identity", lambda: dict(adapter_identity))
    template = A.adapter_authorization_template()
    verdict = A.validate_adapter_authorization(
        template,
        require_execution=False,
    )

    assert verdict["identity_valid"] is True, verdict
    assert verdict["authorized"] is False
    assert verdict["identity_failures"] == []
    assert verdict["binding_failures"] == []
    assert verdict["owner_state_failures"] == [
        "adapter_authorization_status_not_authorized",
        "adapter_execution_not_authorized",
    ]


def test_preflight_adapter_binding_rejects_trainer_argv_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_identity = A.adapter_identity()
    adapter_identity.update({"tracked_clean": True, "script_tracked": True})
    monkeypatch.setattr(A, "adapter_identity", lambda: dict(adapter_identity))

    expected_argv = ["--run_plan_stage", "stage_b"]
    stage_o_document = {
        "authorization_status": "NOT_AUTHORIZED",
        "allowed_scope": "STAGE_O",
        "stage_o_launch_adapter_identity": {
            field: adapter_identity[field]
            for field in (
                "head",
                "adapter_tool_bundle_sha256",
                "adapter_tool_path",
                "adapter_tool_sha256",
            )
        },
        "stage_o_trainer_argv": expected_argv,
    }
    stage_o_path = tmp_path / "STAGE_O_AUTHORIZATION_NOT_AUTHORIZED.json"
    stage_o_path.write_bytes(
        json.dumps(stage_o_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    adapter_document = A.adapter_authorization_template(
        stage_o_authorization_path=stage_o_path,
    )

    verdict = A.validate_adapter_authorization(
        adapter_document,
        require_execution=False,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=stage_o_document,
        trainer_argv=["--run_plan_stage", "stage_a"],
    )

    assert verdict["identity_valid"] is True, verdict
    assert verdict["binding_failures"] == ["stage_o_trainer_argv_mismatch"]


@pytest.mark.skipif(
    os.environ.get("PETITGPT_RUN_STAGE_O_GATE_A_REHEARSAL") != "1",
    reason="real accepted Stage-N chain revalidation is opt-in",
)
def test_real_parser_successor_chain_and_gate_a_stop_on_owner_state_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(HISTORICAL_ROOT)
    topology = A.install_accepted_module_topology()
    contract = topology.launch_contract
    trainer = topology.trainer

    assert topology.launch_contract is sys.modules["pretrain.production_launch_contract_v1"]
    assert topology.launch_contract is sys.modules["production_launch_contract_v1"]
    assert topology.pretrain_package.production_launch_contract_v1 is topology.launch_contract

    runtime = contract.observed_training_runtime(num_workers=2)
    stage_o_path, stage_o_authorization, stage_n = _write_stage_o_not_authorized(
        contract,
        tmp_path,
        runtime=runtime,
    )
    assert stage_n["runtime_fingerprint"] == runtime
    run_root = Path(stage_o_authorization["allowed_output_root"])
    trainer_argv = _trainer_argv(
        contract,
        stage_o_authorization_path=stage_o_path,
        stage_n=stage_n,
        run_root=run_root,
    )
    adapter_identity = A.adapter_authorization_template(runtime_fingerprint=runtime)[
        "stage_o_launch_adapter"
    ]
    stage_o_authorization["stage_o_trainer_argv"] = list(trainer_argv)
    stage_o_authorization["stage_o_launch_adapter_identity"] = {
        field: adapter_identity[field]
        for field in (
            "head",
            "adapter_tool_path",
            "adapter_tool_sha256",
            "adapter_tool_bundle_sha256",
        )
    }
    _write_canonical_json(contract, stage_o_path, stage_o_authorization)

    # Exercise the accepted parser independently before the adapter repeats the exact path.
    parsed = A.parse_trainer_args(trainer, trainer_argv)
    assert parsed.run_plan_stage == "stage_b"
    assert parsed.stage_authorization_json == str(stage_o_path)
    assert parsed.resume_path == stage_n["final_checkpoint_path"]

    adapter_document = A.adapter_authorization_template(
        runtime_fingerprint=runtime,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization_sha256=contract.file_sha256(stage_o_path),
    )
    adapter_path = _write_canonical_json(
        contract,
        tmp_path / "ADAPTER_AUTHORIZATION_NOT_AUTHORIZED.json",
        adapter_document,
    )
    adapter_verdict = A.validate_adapter_authorization(
        adapter_document,
        observed_runtime=runtime,
        require_execution=False,
        adapter_authorization_path=adapter_path,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=stage_o_authorization,
        trainer_argv=trainer_argv,
    )
    assert adapter_verdict["identity_valid"] is True, adapter_verdict
    assert adapter_verdict["identity_failures"] == []
    assert adapter_verdict["binding_failures"] == []

    profiler, boundary_violations = _forbidden_boundary_profiler(topology)
    previous_profile = sys.getprofile()
    previous_trace = sys.gettrace()
    sys.setprofile(profiler)
    try:
        result = A.run_preflight(adapter_path, stage_o_path, trainer_argv)
    finally:
        sys.setprofile(previous_profile)

    assert sys.getprofile() is previous_profile
    assert sys.gettrace() is previous_trace
    assert boundary_violations == []

    assert result["schema_version"] == "petitgpt-stage-o-successor-launch-adapter-preflight-v1"
    assert result["mode"] == "PREFLIGHT"
    assert result["adapter_authorization"]["identity_valid"] is True
    assert result["adapter_authorization"]["binding_failures"] == []
    assert result["adapter_authorization"]["owner_state_failures"] == [
        "adapter_authorization_status_not_authorized",
        "adapter_execution_not_authorized",
    ]
    assert result["trainer_parser_reached"] is True
    assert result["stage_o_chain"]["valid"] is True, result["stage_o_chain"]
    assert result["stage_o_chain"]["failures"] == []
    assert result["gate_a_reached"] is True
    assert result["gate_a_passed"] is False
    assert result["gate_a_exception_type"] == "LaunchContractError"
    assert result["gate_a_failures"] == ["authorization_status_not_authorized"]
    assert result["module_origin_failures"] == []
    assert result["binding_failures"] == []
    assert result["owner_state_failures"] == ["authorization_status_not_authorized"]
    assert result["module_topology"]["canonical_to_bare_same_object"] is True
    assert result["module_topology"]["parent_package_binding_same_object"] is True
    assert result["module_topology"]["duplicate_launch_module_object_count"] == 0
    assert result["module_topology"]["bridge_launch_same_object"] is True
    class_family = result["module_topology"]["class_family_identity"]
    assert set(class_family) == {"LaunchContractError", "_Missing", "ObservedForward"}
    assert all(class_family.values())
    for boundary in (
        "model_construction_reached",
        "checkpoint_restore_reached",
        "compile_realization_reached",
        "model_forward_reached",
        "backward_reached",
        "optimizer_update_reached",
    ):
        assert result[boundary] is False

    evidence_path = os.environ.get("PETITGPT_STAGE_O_REHEARSAL_RESULT_PATH")
    if evidence_path:
        destination = Path(evidence_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(A.canonical_json_bytes(result))
