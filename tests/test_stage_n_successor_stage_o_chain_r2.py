"""R2 successor Stage-N provenance and Stage-O compatibility-chain regressions.

These tests use only tiny JSON artifacts and opaque checkpoint bytes.  They never restore a
checkpoint, construct a model, touch CUDA, or execute a training-loop iteration.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pretrain import (  # noqa: E402, I001
    production_launch_contract_v1 as C,
    stage_n_successor_head_compatibility_bridge_v1 as B,
)


HISTORICAL_HEAD = "6d80423adc16d4a160a7fe42660020c585b5185d"
HISTORICAL_BUNDLE = "bbd49b9d73d3cb2fa18aacb3eee861a901e5a7511ed334b85b37239ab1d50043"
SUCCESSOR_HEAD = "a" * 40
SUCCESSOR_BUNDLE = "b" * 64
BRIDGE_BUNDLE = "c" * 64


def _write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(C.canonical_json_bytes(document))
    return path


def _runtime() -> dict:
    wrapper = json.loads(
        Path(B.HISTORICAL_ARTIFACTS["n2_runtime_fingerprint"]["path"]).read_bytes()
    )
    runtime = copy.deepcopy(wrapper["runtime_fingerprint"])
    runtime.update({
        "trainer_head": SUCCESSOR_HEAD,
        "trainer_execution_bundle_sha256": SUCCESSOR_BUNDLE,
    })
    runtime["runtime_fingerprint_sha256"] = C.runtime_fingerprint_sha256(runtime)
    return runtime


def _code_identity() -> dict:
    return {
        "branch": B.SUCCESSOR_BRANCH,
        "repository_root": str(B.SUCCESSOR_REPOSITORY_ROOT),
        "production_module_path": str(
            B.SUCCESSOR_REPOSITORY_ROOT / "pretrain/production_launch_contract_v1.py"
        ),
        "bridge_module_path": str(
            B.SUCCESSOR_REPOSITORY_ROOT
            / "pretrain/stage_n_successor_head_compatibility_bridge_v1.py"
        ),
        "all_loaded_local_modules_under_repository_root": True,
        "head": SUCCESSOR_HEAD,
        "trainer_execution_bundle_sha256": SUCCESSOR_BUNDLE,
        "bridge_tool_bundle_sha256": BRIDGE_BUNDLE,
        "tracked_clean": True,
    }


def _source_history() -> dict:
    """The one immutable N1/N2 history authorized by the owner supersession."""
    return {
        "source_execution": {
            "branch": B.HISTORICAL_TRAINING_BRANCH,
            "head": B.HISTORICAL_TRAINING_HEAD,
            "trainer_execution_bundle_sha256": B.HISTORICAL_TRAINING_BUNDLE_SHA256,
        },
        "exact_run_plan_sha256": B.EXACT_RUN_PLAN_SHA256,
        "pilot_owner_acceptance_sha256": B.PILOT_OWNER_ACCEPTANCE_SHA256,
        "n1_governed_run_contract_semantic_sha256": B.N1_GRC_SEMANTIC_SHA256,
        "n1_base_governed_identity_sha256": B.N1_BASE_GOVERNED_IDENTITY_SHA256,
        "n2_governed_run_contract_semantic_sha256": B.N2_GRC_SEMANTIC_SHA256,
        "source_runtime_fingerprint_sha256": B.HISTORICAL_RUNTIME_FINGERPRINT_SHA256,
        "artifacts": {name: dict(binding) for name, binding in B.HISTORICAL_ARTIFACTS.items()},
    }


def _semantic_manifest(tmp_path: Path) -> tuple[Path, dict]:
    records = []
    for index, relative in enumerate(B.HISTORICAL_GOVERNED_CLOSURE_FILES, start=1):
        source_sha = f"{index:064x}"
        successor_sha = source_sha if relative != B.SUCCESSOR_POLICY_FILE else "f" * 64
        records.append({
            "path": relative,
            "source_sha256": source_sha,
            "successor_sha256": successor_sha,
            "byte_identical": source_sha == successor_sha,
        })
    document = {
        "schema_version": B.SEMANTIC_COMPARISON_SCHEMA,
        "source_head": B.HISTORICAL_TRAINING_HEAD,
        "source_trainer_execution_bundle_sha256": B.HISTORICAL_TRAINING_BUNDLE_SHA256,
        "successor_head": SUCCESSOR_HEAD,
        "successor_trainer_execution_bundle_sha256": SUCCESSOR_BUNDLE,
        "exact_run_plan_sha256": B.EXACT_RUN_PLAN_SHA256,
        "pilot_owner_acceptance_sha256": B.PILOT_OWNER_ACCEPTANCE_SHA256,
        "files": records,
        "file_count": len(records),
        "changed_files": [B.SUCCESSOR_POLICY_FILE],
        "changed_file_count": 1,
        "core_training_semantics_files_changed": [],
        "semantic_isolation_pass": True,
    }
    document["manifest_content_sha256"] = B.sha256_bytes(B.canonical_json_bytes(document))
    path = _write_json(tmp_path / "SEMANTIC_COMPARISON_MANIFEST.json", document)
    return path, document


def _compile_evidence() -> dict:
    """A sealed production-shape observation document; no compilation is executed."""

    cache_dir = "/tmp/petitgpt-successor-stage-o-test-cache"
    observations = {
        "schema_version": "petitgpt-governed-compile-evidence-v1",
        "compile_requested": True,
        "invoked_compiled_callable": True,
        "realized_module_is_optimized_module": True,
        "compilation_materialized": True,
        "forward_invocations": 1,
        "expected_forward_invocations": 1,
        "forward_invocations_match": True,
        "dynamo_unique_graphs": 1,
        "inductor_artifact_count": 0,
        "inductor_cache_dir": cache_dir,
        "recompile_limit_fallback_detected": False,
        "eager_fallback_occurred": False,
        "compiled_callable_is_training_callable": True,
        "probe_geometry": {
            "micro_bsz": C.MICRO_BSZ,
            "seq_len": C.MODEL_CONTRACT["seq_len"],
        },
        "probe_signature": {
            "device_type": "cuda",
            "module_training_mode": True,
            "grad_enabled": True,
            "autocast_enabled": True,
            "autocast": "bf16",
            "autocast_dtype": "torch.bfloat16",
            "input_dtype": "torch.int64",
            "input_shape": [C.MICRO_BSZ, C.MODEL_CONTRACT["seq_len"]],
            "output_dtype": "torch.bfloat16",
            "output_requires_grad": True,
            "optimizer_step_taken": False,
            "matches_training_forward_signature": True,
        },
        "production_shape_probe": True,
        "isolated_cache": {
            "cache_dir": cache_dir,
            "triton_cache_dir": f"{cache_dir}/triton",
            "was_empty_before_realization": True,
            "isolated": True,
        },
        "cache_was_empty_before_realization": True,
        "precompile_causal_diagnostic": {
            "executed": True,
            "used_uncompiled_base_model": True,
            "executed_before_training_compile_realization": True,
            "grad_enabled": False,
            "input_shape": [1, C.CAUSAL_DIAGNOSTIC_SEQ_LEN],
            "check_pos": C.CAUSAL_DIAGNOSTIC_CHECK_POS,
            "delta_pos": C.CAUSAL_DIAGNOSTIC_DELTA_POS,
            "max_abs_difference": 0.0,
            "max_abs_tolerance": C.CAUSAL_LEAK_MAX_ABS_TOLERANCE,
            "within_tolerance": True,
            "mode_before": "train",
            "mode_after": "train",
            "mode_restored": True,
        },
        "fail_closed_stance": {
            "suppress_errors": False,
            "fail_on_recompile_limit_hit": True,
            "set_stance_available": True,
        },
        "post_realization_stance": {
            "armed": True,
            "stance": "fail_on_recompile",
        },
        "bridge_zero_update_observations": {
            "execution_counters": dict(B.ZERO_EXECUTION_COUNTERS),
            "before_compile": {
                "model_parameters_bitwise_identical": True,
                "optimizer_state_equivalent": True,
                "scaler_state_equivalent": True,
                "rng_state_preserved": True,
                "all_parameter_grads_absent": True,
                "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
            },
            "after_compile": {
                "model_parameters_bitwise_identical": True,
                "optimizer_state_equivalent": True,
                "scaler_state_equivalent": True,
                "rng_state_preserved": True,
                "all_parameter_grads_absent": True,
                "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
            },
            "training_loop_constructed": False,
            "data_loader_constructed": False,
            "sampler_constructed": False,
            "scheduler_constructed": False,
        },
    }
    return C.finalize_compile_evidence(observations)


def _result(tmp_path: Path) -> tuple[dict, dict]:
    """Create a structurally complete successor result and its N3 artifacts."""
    runtime = _runtime()
    bridge_dir = tmp_path / "n3"
    bridge_dir.mkdir(parents=True)
    semantic_path, _ = _semantic_manifest(tmp_path)
    bridge_authorization = B.n3_authorization_template(
        successor_head=SUCCESSOR_HEAD,
        successor_trainer_bundle_sha256=SUCCESSOR_BUNDLE,
        bridge_tool_bundle_sha256=BRIDGE_BUNDLE,
        successor_runtime_fingerprint_sha256=runtime["runtime_fingerprint_sha256"],
        semantic_comparison_manifest_path=semantic_path,
        semantic_comparison_manifest_sha256=C.file_sha256(semantic_path),
        output_root=bridge_dir,
    )
    bridge_authorization.update({
        "authorization_status": "AUTHORIZED",
        "authorizes_bridge_execution": True,
        "authorized_by": "synthetic-owner",
        "authorized_at": "2026-09-03T00:00:00Z",
    })
    authorization_path = _write_json(bridge_dir / "N3_AUTHORIZATION.json", bridge_authorization)
    authorization_sha = C.file_sha256(authorization_path)
    permutation = C.permutation_identity(
        "stage_a",
        B.COMPLETION_SAMPLER_SEED,
        B.COMPLETION_SAMPLER_ENDPOINT,
    )
    compile_evidence = _compile_evidence()
    state_equivalence = {
        "schema_version": "petitgpt-stage-n-successor-state-equivalence-v1",
        "equivalent": True,
        "failures": [],
        "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
        "optimizer_state_equivalent": True,
        "scaler_state_equivalent": True,
        "rng_state_preserved": True,
        "global_step": B.COMPLETION_STEP,
        "sampler_cursor": B.COMPLETION_SAMPLER_ENDPOINT,
    }

    grc = {
        "schema_version": C.RUN_CONTRACT_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "kind": C.GOVERNED_CHECKPOINT_KIND,
        "governed": True,
        "stage": "stage_a",
        "scope": "STAGE_N",
        "stage_authorization_path": str(authorization_path),
        "stage_authorization_sha256": authorization_sha,
        "trainer_branch": B.SUCCESSOR_BRANCH,
        "successor_repository_root": str(B.SUCCESSOR_REPOSITORY_ROOT),
        "trainer_head": SUCCESSOR_HEAD,
        "trainer_execution_bundle_sha256": SUCCESSOR_BUNDLE,
        "bridge_tool_bundle_sha256": BRIDGE_BUNDLE,
        "launch_contract_sha256": C.contract_sha256(),
        "exact_plan_path": B.HISTORICAL_ARTIFACTS["exact_plan"]["path"],
        "exact_run_plan_sha256": B.EXACT_RUN_PLAN_SHA256,
        "pilot_acceptance_path": B.HISTORICAL_ARTIFACTS["pilot_acceptance"]["path"],
        "pilot_owner_acceptance_sha256": B.PILOT_OWNER_ACCEPTANCE_SHA256,
        "runtime_fingerprint": runtime,
        "runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        "canonical_cwd": B.RUNTIME_INVARIANTS["canonical_cwd"],
        "gpu_uuid": runtime["gpu_uuid"],
        "gpu_pci_bus_id": runtime["gpu_pci_bus_id"],
        "num_workers": runtime["num_workers"],
        "governed_run_root": str(bridge_dir),
        "invocation_root": str(bridge_dir),
        "out_dir": str(bridge_dir),
        "samples_dir": str(bridge_dir / "samples"),
        "stage_start_step": C.STAGE_A_START_STEP,
        "stage_stop_step": B.COMPLETION_STEP,
        "active_stage_sampler_seed": B.COMPLETION_SAMPLER_SEED,
        "resume": {
            "mode": "RESUME_EXACT_CHECKPOINT",
            "checkpoint_path": B.HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
            "checkpoint_sha256": B.N2_TERMINAL_CHECKPOINT_SHA256,
            "expected_step": B.COMPLETION_STEP,
            "stage": "stage_a",
            "governed_run_contract_sha256": B.N2_GRC_SEMANTIC_SHA256,
            "source_stage_authorization_path": B.HISTORICAL_ARTIFACTS["n2_authorization"]["path"],
            "source_stage_authorization_sha256": B.N2_AUTHORIZATION_SHA256,
            "source_invocation_run_contract_path": B.HISTORICAL_ARTIFACTS[
                "n2_governed_run_contract"
            ]["path"],
            "source_invocation_run_contract_sha256": B.N2_GRC_ARTIFACT_SHA256,
            "source_base_governed_identity_digest": B.N1_BASE_GOVERNED_IDENTITY_SHA256,
            "source_checkpoint_path": B.HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
            "source_checkpoint_sha256": B.N2_TERMINAL_CHECKPOINT_SHA256,
            "source_checkpoint_step": B.COMPLETION_STEP,
            "source_checkpoint_stage": "stage_a",
            "source_active_stage": "stage_a",
            "source_sampler_seed": B.COMPLETION_SAMPLER_SEED,
            "source_permutation_identity": permutation,
            "source_range_start_position": 0,
            "source_invocation_range_start_position": B.COMPLETION_SAMPLER_ENDPOINT,
            "source_range_stop_position": B.COMPLETION_SAMPLER_ENDPOINT,
            "source_cursor": B.COMPLETION_SAMPLER_ENDPOINT,
        },
        "sampler_identity": {
            "stage": "stage_a",
            "sampler_seed": B.COMPLETION_SAMPLER_SEED,
            "permutation_identity": permutation,
            "range_start_position": 0,
            "invocation_range_start_position": B.COMPLETION_SAMPLER_ENDPOINT,
            "range_stop_position": B.COMPLETION_SAMPLER_ENDPOINT,
            "cursor": B.COMPLETION_SAMPLER_ENDPOINT,
        },
        "compile_evidence": compile_evidence,
        "compile_evidence_sha256": compile_evidence["compile_evidence_sha256"],
        "compatibility_bridge": {
            "schema_version": B.BRIDGE_GRC_SCHEMA,
            "source_head": B.HISTORICAL_TRAINING_HEAD,
            "source_bundle_sha256": B.HISTORICAL_TRAINING_BUNDLE_SHA256,
            "source_checkpoint_sha256": B.N2_TERMINAL_CHECKPOINT_SHA256,
            "successor_repository_root": str(B.SUCCESSOR_REPOSITORY_ROOT),
            "successor_head": SUCCESSOR_HEAD,
            "successor_trainer_execution_bundle_sha256": SUCCESSOR_BUNDLE,
            "successor_bridge_tool_bundle_sha256": BRIDGE_BUNDLE,
            "successor_runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
            "semantic_comparison_manifest_path": str(semantic_path),
            "semantic_comparison_manifest_sha256": C.file_sha256(semantic_path),
            "start_step": B.COMPLETION_STEP,
            "stop_step": B.COMPLETION_STEP,
            "optimizer_updates": 0,
            "trained_tokens": 0,
            "sampler_advances": 0,
            "training_loop_iterations": 0,
            "backward_calls": 0,
            "scheduler_advances": 0,
            "data_batches_consumed": 0,
        },
    }
    grc["governed_run_contract_sha256"] = C.governed_digest(grc)
    grc_path = _write_json(bridge_dir / C.GOVERNED_RUN_CONTRACT_FILENAME, grc)

    checkpoint_path = bridge_dir / f"step_{C.STAGE_A_STOP_STEP:06d}.pt"
    checkpoint_path.write_bytes(b"tiny successor bridge checkpoint fixture\n")
    runtime_path = _write_json(
        bridge_dir / C.STAGE_N_RUNTIME_FILENAME,
        C.stage_n_runtime_artifact_document(runtime),
    )
    smoke_evidence_path = _write_json(
        bridge_dir / "STAGE_N_BRIDGE_COMPILE_EVIDENCE.json",
        compile_evidence,
    )
    resume_evidence = {
        "schema_version": B.BRIDGE_RESUME_EVIDENCE_SCHEMA,
        "kind": B.BRIDGE_RESUME_EVIDENCE_KIND,
        "status": "PASS",
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha,
        "semantic_comparison_manifest_path": str(semantic_path),
        "semantic_comparison_manifest_sha256": C.file_sha256(semantic_path),
        "governed_run_contract_path": str(grc_path),
        "governed_run_contract_artifact_sha256": C.file_sha256(grc_path),
        "governed_run_contract_sha256": C.governed_digest(grc),
        "terminal_checkpoint_path": str(checkpoint_path),
        "terminal_checkpoint_sha256": C.file_sha256(checkpoint_path),
        "terminal_checkpoint_step": B.COMPLETION_STEP,
        "runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        "source_n2_checkpoint_path": B.HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
        "source_n2_checkpoint_sha256": B.N2_TERMINAL_CHECKPOINT_SHA256,
        "execution_counters": dict(B.ZERO_EXECUTION_COUNTERS),
        "state_equivalence": state_equivalence,
    }
    resume_evidence_path = _write_json(
        bridge_dir / "STAGE_N_BRIDGE_STATE_EQUIVALENCE.json",
        resume_evidence,
    )
    check = C.stage_n_check_result_document(
        kind=C.STAGE_N_SMOKE_RESULT_KIND,
        stage_authorization_sha256=authorization_sha,
        governed_run_contract_sha256=C.governed_digest(grc),
        checkpoint_path=checkpoint_path,
        checkpoint_step=C.STAGE_A_STOP_STEP,
        runtime_fingerprint_sha256=runtime["runtime_fingerprint_sha256"],
        evidence_artifact_path=smoke_evidence_path,
    )
    resume_check = C.stage_n_check_result_document(
        kind=C.STAGE_N_RESUME_RESULT_KIND,
        stage_authorization_sha256=authorization_sha,
        governed_run_contract_sha256=C.governed_digest(grc),
        checkpoint_path=checkpoint_path,
        checkpoint_step=C.STAGE_A_STOP_STEP,
        runtime_fingerprint_sha256=runtime["runtime_fingerprint_sha256"],
        evidence_artifact_path=resume_evidence_path,
    )
    smoke_path = _write_json(bridge_dir / "STAGE_N_SMOKE_RESULT.json", check)
    resume_path = _write_json(bridge_dir / "STAGE_N_RESUME_RESULT.json", resume_check)

    bridge_binding = {
        "schema_version": B.BRIDGE_RESULT_SCHEMA,
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha,
        "semantic_comparison_manifest_path": str(semantic_path),
        "semantic_comparison_manifest_sha256": C.file_sha256(semantic_path),
        "governed_run_contract_path": str(grc_path),
        "governed_run_contract_artifact_sha256": C.file_sha256(grc_path),
        "governed_run_contract_sha256": C.governed_digest(grc),
        "terminal_checkpoint_path": str(checkpoint_path),
        "terminal_checkpoint_sha256": C.file_sha256(checkpoint_path),
        "terminal_checkpoint_step": C.STAGE_A_STOP_STEP,
        "runtime_fingerprint_path": str(runtime_path),
        "runtime_fingerprint_artifact_sha256": C.file_sha256(runtime_path),
        "runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        "bridge_tool_bundle_sha256": BRIDGE_BUNDLE,
        "optimizer_updates": 0,
        "trained_tokens": 0,
        "sampler_advances": 0,
        "training_loop_iterations": 0,
        "backward_calls": 0,
        "scheduler_advances": 0,
        "data_batches_consumed": 0,
        "state_equivalence": state_equivalence,
    }
    result = {
        "schema_version": C.STAGE_N_RESULT_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "status": "COMPLETE",
        "stage": "stage_a",
        "scope": "STAGE_N",
        "stage_authorization_path": str(authorization_path),
        "stage_authorization_sha256": authorization_sha,
        "launch_contract_sha256": C.contract_sha256(),
        "exact_plan_path": B.HISTORICAL_ARTIFACTS["exact_plan"]["path"],
        "exact_run_plan_sha256": C.EXACT_RUN_PLAN_SHA256,
        "pilot_acceptance_path": B.HISTORICAL_ARTIFACTS["pilot_acceptance"]["path"],
        "pilot_owner_acceptance_sha256": C.PILOT_OWNER_ACCEPTANCE_SHA256,
        # Legacy invocation fields now describe N3, never the historical training execution.
        "trainer_head": SUCCESSOR_HEAD,
        "trainer_execution_bundle_sha256": SUCCESSOR_BUNDLE,
        "governed_run_contract_path": str(grc_path),
        "governed_run_contract_artifact_sha256": C.file_sha256(grc_path),
        "governed_run_contract_sha256": C.governed_digest(grc),
        "base_governed_identity_sha256": C.base_governed_identity_sha256(grc),
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": C.file_sha256(checkpoint_path),
        "final_checkpoint_step": C.STAGE_A_STOP_STEP,
        "runtime_fingerprint": runtime,
        "runtime_fingerprint_path": str(runtime_path),
        "runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        "runtime_fingerprint_artifact_sha256": C.file_sha256(runtime_path),
        "gpu_uuid": runtime["gpu_uuid"],
        "gpu_pci_bus_id": runtime["gpu_pci_bus_id"],
        "num_workers": runtime["num_workers"],
        "smoke_results_path": str(smoke_path),
        "smoke_results_sha256": C.file_sha256(smoke_path),
        "smoke_results": check,
        "resume_results_path": str(resume_path),
        "resume_results_sha256": C.file_sha256(resume_path),
        "resume_results": resume_check,
        "final_sampler_permutation_identity": C.permutation_identity(
            "stage_a",
            B.COMPLETION_SAMPLER_SEED,
            B.COMPLETION_SAMPLER_ENDPOINT,
        ),
        "final_sampler_range_start_position": 0,
        "final_sampler_invocation_range_start_position": C.STAGE_A_TERMINAL_RANGE_STOP,
        "final_sampler_range_stop_position": C.STAGE_A_TERMINAL_RANGE_STOP,
        "final_sampler_cursor": C.STAGE_A_TERMINAL_RANGE_STOP,
        "stage_n_training_execution_head": HISTORICAL_HEAD,
        "stage_n_training_execution_bundle": HISTORICAL_BUNDLE,
        "stage_n_compatibility_bridge_head": SUCCESSOR_HEAD,
        "stage_n_compatibility_bridge_bundle": BRIDGE_BUNDLE,
        "stage_o_execution_head": SUCCESSOR_HEAD,
        "stage_o_execution_bundle": SUCCESSOR_BUNDLE,
        "stage_n_source_history": _source_history(),
        "stage_n_compatibility_bridge": bridge_binding,
        "stage_n_owner_accepted": False,
        "stage_o_authorized": False,
    }
    return result, {
        "authorization_path": authorization_path,
        "semantic_path": semantic_path,
        "grc_path": grc_path,
        "checkpoint_path": checkpoint_path,
        "runtime_path": runtime_path,
    }


def _chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    result, artifacts = _result(tmp_path)
    synthetic_grc = json.loads(artifacts["grc_path"].read_bytes())
    monkeypatch.setattr(
        B,
        "_expected_successor_grc_from_authorized_source",
        lambda **_kwargs: (
            {"synthetic": "authenticated-source"},
            copy.deepcopy(synthetic_grc),
        ),
    )
    result_path = _write_json(
        artifacts["checkpoint_path"].parent / "STAGE_N_COMPLETE_RESULT.json",
        result,
    )
    result_sha = C.file_sha256(result_path)
    acceptance = {
        "stage_n_result_owner_verdict": "ACCEPTED",
        "accepted_stage_n_result_sha256": result_sha,
    }
    acceptance_path = _write_json(tmp_path / "STAGE_N_ACCEPTANCE.json", acceptance)

    # The bridge suite exercises the authoritative live-byte preflight. This suite isolates
    # Stage-O's independent artifact-chain validation using a committed-identity observation
    # seam; neither seam restores the tiny checkpoint.
    def accepted_preflight(authorization_path, **kwargs):
        n3_authorization = json.loads(Path(authorization_path).read_bytes())
        manifest_path = (
            kwargs.get("semantic_comparison_manifest_path")
            or n3_authorization["semantic_comparison_manifest"]["path"]
        )
        semantic_manifest = json.loads(Path(manifest_path).read_bytes())
        return {
            "schema_version": "petitgpt-stage-n-successor-bridge-preflight-v1",
            "authorized": True,
            "failures": [],
            "authorization_path": str(Path(authorization_path).resolve()),
            "authorization_sha256": C.file_sha256(authorization_path),
            "authorization_document": copy.deepcopy(n3_authorization),
            "semantic_comparison_manifest_path": str(Path(manifest_path).resolve()),
            "semantic_comparison_manifest_sha256": C.file_sha256(manifest_path),
            "semantic_comparison_manifest_document": copy.deepcopy(semantic_manifest),
            "source_head": B.HISTORICAL_TRAINING_HEAD,
            "successor_head": n3_authorization["successor"]["head"],
            "completion_step": B.COMPLETION_STEP,
            "sampler_endpoint": B.COMPLETION_SAMPLER_ENDPOINT,
        }

    monkeypatch.setattr(
        B,
        "validate_bridge_authorization",
        accepted_preflight,
    )
    # Bridge-focused tests independently exercise authenticated snapshot deserialization,
    # both checkpoint envelopes, and the 213-tensor/full-state equivalence recomputation.
    # This Stage-O fixture keeps opaque checkpoint bytes so it can isolate chain topology
    # without restoring even a tiny checkpoint.
    monkeypatch.setattr(
        B,
        "_revalidate_successor_checkpoint_state",
        lambda *args, **kwargs: {"failures": [], "state_equivalence": None},
    )
    monkeypatch.setattr(
        C,
        "_observed_successor_stage_o_code_identity",
        _code_identity,
    )
    monkeypatch.setattr(C, "validate_stage_n_check_result", lambda *args, **kwargs: [])

    chain = {
        "accepted_stage_n_result_path": str(result_path),
        "accepted_stage_n_result_sha256": result_sha,
        "stage_n_owner_acceptance_path": str(acceptance_path),
        "stage_n_owner_acceptance_sha256": C.file_sha256(acceptance_path),
        "stage_n_authorization_sha256": result["stage_authorization_sha256"],
        "stage_n_governed_run_contract_sha256": result["governed_run_contract_sha256"],
        "stage_n_governed_run_contract_artifact_sha256": result[
            "governed_run_contract_artifact_sha256"
        ],
        "stage_n_runtime_fingerprint": result["runtime_fingerprint"],
        "stage_n_runtime_fingerprint_sha256": result["runtime_fingerprint_sha256"],
        "stage_n_runtime_fingerprint_path": result["runtime_fingerprint_path"],
        "stage_n_runtime_fingerprint_artifact_sha256": result[
            "runtime_fingerprint_artifact_sha256"
        ],
        "stage_n_gpu_uuid": result["gpu_uuid"],
        "stage_n_gpu_pci_bus_id": result["gpu_pci_bus_id"],
        "stage_n_trainer_head": result["trainer_head"],
        "stage_n_trainer_execution_bundle_sha256": result["trainer_execution_bundle_sha256"],
        "stage_n_exact_run_plan_sha256": result["exact_run_plan_sha256"],
        "stage_n_final_checkpoint_path": result["final_checkpoint_path"],
        "stage_n_final_checkpoint_sha256": result["final_checkpoint_sha256"],
        "stage_n_final_checkpoint_step": result["final_checkpoint_step"],
        "n3_bridge_authorization_path": str(artifacts["authorization_path"]),
        "n3_bridge_authorization_sha256": C.file_sha256(artifacts["authorization_path"]),
        "n3_bridge_semantic_comparison_manifest_path": str(artifacts["semantic_path"]),
        "n3_bridge_semantic_comparison_manifest_sha256": C.file_sha256(artifacts["semantic_path"]),
        "n3_bridge_governed_run_contract_path": str(artifacts["grc_path"]),
        "n3_bridge_governed_run_contract_artifact_sha256": C.file_sha256(artifacts["grc_path"]),
        "n3_bridge_governed_run_contract_sha256": result["governed_run_contract_sha256"],
        "n3_bridge_terminal_checkpoint_path": str(artifacts["checkpoint_path"]),
        "n3_bridge_terminal_checkpoint_sha256": C.file_sha256(artifacts["checkpoint_path"]),
        "n3_bridge_terminal_checkpoint_step": C.STAGE_A_STOP_STEP,
        "stage_n_training_execution_head": result["stage_n_training_execution_head"],
        "stage_n_training_execution_bundle": result["stage_n_training_execution_bundle"],
        "stage_n_compatibility_bridge_head": result["stage_n_compatibility_bridge_head"],
        "stage_n_compatibility_bridge_bundle": result["stage_n_compatibility_bridge_bundle"],
        "stage_o_execution_head": result["stage_o_execution_head"],
        "stage_o_execution_bundle": result["stage_o_execution_bundle"],
        "stage_n_source_history": result["stage_n_source_history"],
    }
    authorization = {
        "repository_branch": C.STAGE_N_SUCCESSOR_COMPATIBILITY_BRANCH,
        "stage_n_successor_bridge_chain": chain,
        "resume": C.derive_stage_o_resume_binding(result),
    }
    return authorization, result, artifacts


def test_successor_complete_result_separates_training_bridge_and_stage_o_provenance(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(C, "validate_stage_n_check_result", lambda *args, **kwargs: [])
    result, _ = _result(tmp_path)
    assert C.validate_stage_n_result(result) == []
    assert result["trainer_head"] == result["stage_n_compatibility_bridge_head"]
    assert result["trainer_execution_bundle_sha256"] == result["stage_o_execution_bundle"]
    assert result["stage_n_compatibility_bridge_bundle"] == BRIDGE_BUNDLE
    assert (
        result["stage_n_compatibility_bridge_bundle"] != result["trainer_execution_bundle_sha256"]
    )
    assert result["stage_n_training_execution_head"] == HISTORICAL_HEAD
    assert result["stage_n_training_execution_bundle"] == HISTORICAL_BUNDLE
    assert result["stage_n_training_execution_head"] != result["trainer_head"]


def test_successor_publication_accepts_absent_completion_with_exact_artifact_agreement(
    tmp_path,
    monkeypatch,
):
    """The late publisher must not invent a requirement Gate A never imposed."""

    monkeypatch.setattr(C, "validate_stage_n_check_result", lambda *args, **kwargs: [])
    result, artifacts = _result(tmp_path)
    authorization = json.loads(artifacts["authorization_path"].read_bytes())
    assert "stage_n_completion" not in authorization

    publication = C.publish_stage_n_result(tmp_path / "successor-publication", result)

    assert publication["status"] == "PUBLISHED_AWAITING_OWNER_ACCEPTANCE"
    assert Path(publication["path"]).read_bytes() == C.canonical_json_bytes(result)


@pytest.mark.parametrize(
    "field",
    [
        "stage_n_training_execution_head",
        "stage_n_training_execution_bundle",
        "stage_n_compatibility_bridge_head",
        "stage_n_compatibility_bridge_bundle",
        "stage_o_execution_head",
        "stage_o_execution_bundle",
        "stage_n_source_history",
        "stage_n_compatibility_bridge",
    ],
)
def test_successor_complete_result_requires_each_provenance_authority(
    tmp_path,
    monkeypatch,
    field,
):
    monkeypatch.setattr(C, "validate_stage_n_check_result", lambda *args, **kwargs: [])
    result, _ = _result(tmp_path)
    result.pop(field)
    assert C.validate_stage_n_result(result)


def test_successor_complete_result_cannot_relabel_historical_updates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(C, "validate_stage_n_check_result", lambda *args, **kwargs: [])
    result, _ = _result(tmp_path)
    result["stage_n_training_execution_head"] = SUCCESSOR_HEAD
    result["stage_n_source_history"]["stage_n_training_execution_head"] = SUCCESSOR_HEAD
    assert C.validate_stage_n_result(result)


def test_successor_stage_o_accepts_matching_n3_and_resumes_bridge_checkpoint(
    tmp_path,
    monkeypatch,
):
    authorization, _, artifacts = _chain(tmp_path, monkeypatch)
    n3_authorization = json.loads(artifacts["authorization_path"].read_bytes())
    assert "stage_n_completion" not in n3_authorization
    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())
    assert verdict["valid"] is True, verdict["failures"]
    assert verdict["derived_resume"]["checkpoint_path"] == str(artifacts["checkpoint_path"])
    assert verdict["derived_resume"]["checkpoint_sha256"] == C.file_sha256(
        artifacts["checkpoint_path"]
    )
    assert verdict["derived_resume"]["expected_step"] == C.STAGE_A_STOP_STEP


def test_successor_stage_o_rejects_relocated_copy_of_accepted_complete(
    tmp_path,
    monkeypatch,
):
    authorization, _, artifacts = _chain(tmp_path, monkeypatch)
    chain = authorization["stage_n_successor_bridge_chain"]
    authorized_result = Path(chain["accepted_stage_n_result_path"])
    relocated_result = tmp_path / "relocated" / authorized_result.name
    relocated_result.parent.mkdir()
    relocated_result.write_bytes(authorized_result.read_bytes())
    chain["accepted_stage_n_result_path"] = str(relocated_result)
    chain["accepted_stage_n_result_sha256"] = C.file_sha256(relocated_result)

    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())

    assert verdict["valid"] is False
    assert verdict["derived_resume"]["checkpoint_path"] == str(artifacts["checkpoint_path"])
    assert any(
        "complete_result_path_not_authorized_destination" in failure
        or "COMPLETE path is not the exact authorized destination" in failure
        for failure in verdict["failures"]
    )


def test_successor_stage_o_without_accepted_n3_is_rejected(tmp_path, monkeypatch):
    authorization, _, _ = _chain(tmp_path, monkeypatch)
    chain = authorization.pop("stage_n_successor_bridge_chain")
    # Supplying the former pre-N3 shape must not silently select the superseded old-head path.
    authorization["stage_n_chain"] = chain
    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())
    assert verdict["valid"] is False
    assert any(
        "n3" in failure.lower() or "successor" in failure.lower() for failure in verdict["failures"]
    )


def test_successor_execution_authority_cannot_select_an_ordinary_legacy_chain():
    authorization = {
        "repository_branch": C.STAGE_N_SUCCESSOR_COMPATIBILITY_BRANCH,
        "stage_n_chain": {"shape": "ordinary-pre-n3-chain"},
    }

    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())

    assert verdict == {
        "valid": False,
        "failures": ["stage_o_successor_branch_requires_stage_n_successor_bridge_chain"],
        "derived_resume": None,
        "requires_new_stage_n": False,
    }


def test_exact_incident_source_requires_n3_when_successor_head_is_detached():
    authorization = {
        "repository_branch": "HEAD",
        "trainer_head": SUCCESSOR_HEAD,
        "stage_n_chain": {
            "stage_n_trainer_head": C.STAGE_N_SUCCESSOR_COMPATIBILITY_SOURCE_HEAD,
            "stage_n_trainer_execution_bundle_sha256": (
                C.STAGE_N_SUCCESSOR_COMPATIBILITY_SOURCE_BUNDLE_SHA256
            ),
            "stage_n_exact_run_plan_sha256": C.EXACT_RUN_PLAN_SHA256,
            "stage_n_final_checkpoint_sha256": (
                C.STAGE_N_SUCCESSOR_COMPATIBILITY_SOURCE_N2_TERMINAL_CHECKPOINT_SHA256
            ),
        },
    }

    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())

    assert verdict == {
        "valid": False,
        "failures": ["stage_o_exact_historical_source_requires_stage_n_successor_bridge_chain"],
        "derived_resume": None,
        "requires_new_stage_n": False,
    }


def test_successor_stage_o_cannot_resume_directly_from_old_n2_checkpoint(
    tmp_path,
    monkeypatch,
):
    authorization, _, artifacts = _chain(tmp_path, monkeypatch)
    old_checkpoint = tmp_path / "old-n2-step-038146.pt"
    old_checkpoint.write_bytes(b"historical N2 checkpoint fixture\n")
    authorization["resume"].update({
        "checkpoint_path": str(old_checkpoint),
        "checkpoint_sha256": C.file_sha256(old_checkpoint),
        "source_checkpoint_path": str(old_checkpoint),
        "source_checkpoint_sha256": C.file_sha256(old_checkpoint),
    })
    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())
    assert verdict["valid"] is False
    assert verdict["derived_resume"]["checkpoint_path"] == str(artifacts["checkpoint_path"])


def test_successor_stage_o_rejects_a_second_unreviewed_successor(tmp_path, monkeypatch):
    authorization, _, _ = _chain(tmp_path, monkeypatch)
    changed_runtime = {**_runtime(), "trainer_head": "f" * 40}
    changed_runtime["runtime_fingerprint_sha256"] = C.runtime_fingerprint_sha256(changed_runtime)
    verdict = C.validate_stage_o_chain(authorization, observed_runtime=changed_runtime)
    assert verdict["valid"] is False
    assert any(
        "trainer_head" in failure or "successor" in failure for failure in verdict["failures"]
    )


def test_successor_stage_o_rejects_historical_loaded_module_origin(tmp_path, monkeypatch):
    authorization, _, _ = _chain(tmp_path, monkeypatch)
    wrong_identity = {
        **_code_identity(),
        "production_module_path": ("/workspace/petitgpt/pretrain/production_launch_contract_v1.py"),
        "all_loaded_local_modules_under_repository_root": False,
    }
    monkeypatch.setattr(
        C,
        "_observed_successor_stage_o_code_identity",
        lambda: wrong_identity,
    )

    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())

    assert verdict["valid"] is False
    assert any(
        "observed_code_identity_mismatch:production_module_path" in failure
        or "observed_code_identity_mismatch:all_loaded_local_modules" in failure
        for failure in verdict["failures"]
    )


def test_successor_stage_o_requires_owner_acceptance_of_the_exact_n3_result(
    tmp_path,
    monkeypatch,
):
    authorization, _, _ = _chain(tmp_path, monkeypatch)
    chain = authorization["stage_n_successor_bridge_chain"]
    acceptance_path = Path(chain["stage_n_owner_acceptance_path"])
    acceptance = json.loads(acceptance_path.read_bytes())
    acceptance["stage_n_result_owner_verdict"] = "REJECTED"
    _write_json(acceptance_path, acceptance)
    chain["stage_n_owner_acceptance_sha256"] = C.file_sha256(acceptance_path)
    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())
    assert verdict["valid"] is False
    assert any("not_owner_accepted" in failure for failure in verdict["failures"])


def test_successor_stage_o_rejects_self_consistent_result_provenance_rewrite(
    tmp_path,
    monkeypatch,
):
    authorization, _, _ = _chain(tmp_path, monkeypatch)
    chain = authorization["stage_n_successor_bridge_chain"]
    result_path = Path(chain["accepted_stage_n_result_path"])
    result = json.loads(result_path.read_bytes())
    result["stage_o_execution_head"] = "f" * 40
    _write_json(result_path, result)
    result_sha = C.file_sha256(result_path)
    chain["accepted_stage_n_result_sha256"] = result_sha
    acceptance_path = Path(chain["stage_n_owner_acceptance_path"])
    acceptance = json.loads(acceptance_path.read_bytes())
    acceptance["accepted_stage_n_result_sha256"] = result_sha
    _write_json(acceptance_path, acceptance)
    chain["stage_n_owner_acceptance_sha256"] = C.file_sha256(acceptance_path)
    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())
    assert verdict["valid"] is False
    assert any("stage_o_execution_head" in failure for failure in verdict["failures"])


def test_successor_stage_o_rejects_noncanonical_complete_with_consistent_hashes(
    tmp_path,
    monkeypatch,
):
    authorization, _, _ = _chain(tmp_path, monkeypatch)
    chain = authorization["stage_n_successor_bridge_chain"]
    result_path = Path(chain["accepted_stage_n_result_path"])
    result = json.loads(result_path.read_bytes())
    result["stage_n_owner_accepted"] = True
    _write_json(result_path, result)
    result_sha = C.file_sha256(result_path)
    chain["accepted_stage_n_result_sha256"] = result_sha
    acceptance_path = Path(chain["stage_n_owner_acceptance_path"])
    acceptance = json.loads(acceptance_path.read_bytes())
    acceptance["accepted_stage_n_result_sha256"] = result_sha
    _write_json(acceptance_path, acceptance)
    chain["stage_n_owner_acceptance_sha256"] = C.file_sha256(acceptance_path)

    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())

    assert verdict["valid"] is False
    assert (
        "stage_o_successor_complete:successor_complete_not_exact_authorized_projection"
        in verdict["failures"]
    )


def test_successor_stage_o_rejects_noncanonical_grc_with_consistent_artifact_hashes(
    tmp_path,
    monkeypatch,
):
    authorization, _, artifacts = _chain(tmp_path, monkeypatch)
    chain = authorization["stage_n_successor_bridge_chain"]
    grc_path = artifacts["grc_path"]
    grc = json.loads(grc_path.read_bytes())
    grc["unreviewed_extension"] = True
    _write_json(grc_path, grc)
    grc_artifact_sha = C.file_sha256(grc_path)
    chain["n3_bridge_governed_run_contract_artifact_sha256"] = grc_artifact_sha

    result_path = Path(chain["accepted_stage_n_result_path"])
    result = json.loads(result_path.read_bytes())
    result["governed_run_contract_artifact_sha256"] = grc_artifact_sha
    result["stage_n_compatibility_bridge"]["governed_run_contract_artifact_sha256"] = (
        grc_artifact_sha
    )
    _write_json(result_path, result)
    result_sha = C.file_sha256(result_path)
    chain["accepted_stage_n_result_sha256"] = result_sha
    acceptance_path = Path(chain["stage_n_owner_acceptance_path"])
    acceptance = json.loads(acceptance_path.read_bytes())
    acceptance["accepted_stage_n_result_sha256"] = result_sha
    _write_json(acceptance_path, acceptance)
    chain["stage_n_owner_acceptance_sha256"] = C.file_sha256(acceptance_path)

    verdict = C.validate_stage_o_chain(authorization, observed_runtime=_runtime())

    assert verdict["valid"] is False
    assert (
        "stage_o_successor_complete:successor_complete_grc_not_exact_authorized_projection"
        in verdict["failures"]
    )


def test_fixture_never_invokes_training_surfaces(monkeypatch, tmp_path):
    """An explicit tripwire keeps this suite bounded as production code evolves."""
    import torch

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint restored"))
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: pytest.fail("compile invoked"))
    result, artifacts = _result(tmp_path)
    assert result["final_checkpoint_step"] == C.STAGE_A_STOP_STEP
    assert Path(artifacts["checkpoint_path"]).read_bytes().startswith(b"tiny successor")
    assert copy.deepcopy(result) == result
