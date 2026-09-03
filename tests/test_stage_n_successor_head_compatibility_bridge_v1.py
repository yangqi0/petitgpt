"""Bounded tests for the one-time Stage-N successor-HEAD compatibility bridge."""

from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest
import torch

from pretrain import (
    production_launch_contract_v1 as launch,
    stage_n_successor_head_compatibility_bridge_v1 as B,
)

SUCCESSOR_HEAD = "a" * 40
SUCCESSOR_TRAINER_BUNDLE = "b" * 64
SUCCESSOR_BRIDGE_BUNDLE = "c" * 64


def _exact_plan() -> dict:
    return {
        "schema_version": 3,
        "plan_type": "deterministic_no_replacement_stage_a_b",
        "invariants": {
            "sampling_mode": "deterministic",
            "replacement": False,
            "implicit_replay": False,
            "explicit_replay": False,
        },
        "inputs": {"micro_bsz": 8, "grad_accum": 16, "seq_len": 2048},
        "batch": {
            "sequences_per_optimizer_step": 128,
            "serialized_target_positions_per_optimizer_step": 262144,
        },
        "boundaries": {
            "stage_a_start_step": 0,
            "stage_a_stop_step": B.COMPLETION_STEP,
            "stage_b_start_step": B.COMPLETION_STEP,
            "stage_b_global_stop_step": 49590,
            "schedule_total_steps": 49590,
        },
        "stages": {
            "stage_a": {
                "planned_optimizer_steps": B.COMPLETION_STEP,
                "consumed_blocks": B.COMPLETION_SAMPLER_ENDPOINT,
                "consumed_exposure_blocks": B.COMPLETION_SAMPLER_ENDPOINT,
                "consumed_serialized_target_positions": 9999745024,
            }
        },
        "checkpoint_milestones": {
            "schema_version": 1,
            "absolute_steps": [3815, B.COMPLETION_STEP, 49590],
            "entries": [
                {
                    "absolute_step": B.COMPLETION_STEP,
                    "actual_cumulative_consumed_transitions": 9999745024,
                    "reasons": ["stage_a_end"],
                }
            ],
        },
    }


def _write_semantic_roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    successor = tmp_path / "successor"
    for relative in B.HISTORICAL_GOVERNED_CLOSURE_FILES:
        source_path = source / relative
        successor_path = successor / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        successor_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(f"old:{relative}\n".encode())
        successor_path.write_bytes(source_path.read_bytes())
    (successor / B.SUCCESSOR_POLICY_FILE).write_bytes(b"reviewed successor policy\n")
    return source, successor


def _semantic_manifest(source: Path, successor: Path) -> dict:
    return B.build_semantic_comparison_manifest(
        source,
        successor,
        successor_head=SUCCESSOR_HEAD,
        successor_trainer_bundle_sha256=SUCCESSOR_TRAINER_BUNDLE,
    )


def _n2_evidence() -> dict:
    return {
        "schema_version": B.N2_EVIDENCE_SCHEMA,
        "N2_ZERO_UPDATE_STATUS": "VERIFIED",
        "n2_optimizer_step_calls": 0,
        "n2_trained_tokens": 0,
        "n2_sampler_advances": 0,
        "n2_training_loop_iterations": 0,
        "terminal_compile_evidence_verifies": True,
        "n2_governed_run_contract_artifact_sha256": B.N2_GRC_ARTIFACT_SHA256,
        "n2_governed_run_contract_digest": B.N2_GRC_SEMANTIC_SHA256,
        "source_checkpoint": dict(B.HISTORICAL_ARTIFACTS["n2_source_checkpoint"]),
        "terminal_checkpoint": dict(B.HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]),
        "state_invariance": {
            "base_governed_identity_identical": True,
            "checkpoint_milestone_prefix_unchanged": True,
            "evaluation_milestone_prefix_unchanged": True,
            "model_parameters_bitwise_identical": True,
            "optimizer_param_groups_identical": True,
            "optimizer_state_identical": True,
            "permutation_identity_unchanged": True,
            "range_start_unchanged": True,
            "range_stop_unchanged": True,
            "sampler_seed_unchanged": True,
            "scaler_identical": True,
            "trained_tokens_unchanged": True,
            "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
            "model_max_abs_diff": 0.0,
            "global_step": {
                "source": B.COMPLETION_STEP,
                "terminal": B.COMPLETION_STEP,
                "unchanged": True,
            },
            "sampler_cursor": {
                "source": B.COMPLETION_SAMPLER_ENDPOINT,
                "terminal": B.COMPLETION_SAMPLER_ENDPOINT,
                "unchanged": True,
            },
        },
        "rng_state": {
            "all_streams_identical": True,
            "per_stream_identical": {
                "python": True,
                "numpy": True,
                "torch_cpu": True,
                "torch_cuda": True,
            },
        },
    }


def _historical_runtime() -> dict:
    artifact_path = Path(B.HISTORICAL_ARTIFACTS["n2_runtime_fingerprint"]["path"])
    wrapper = json.loads(artifact_path.read_bytes())
    return copy.deepcopy(wrapper["runtime_fingerprint"])


def _observed_successor() -> dict:
    runtime = copy.deepcopy(_historical_runtime())
    runtime.update({
        "trainer_head": SUCCESSOR_HEAD,
        "trainer_execution_bundle_sha256": SUCCESSOR_TRAINER_BUNDLE,
    })
    runtime["runtime_fingerprint_sha256"] = launch.runtime_fingerprint_sha256(runtime)
    return {
        "branch": B.SUCCESSOR_BRANCH,
        "repository_root": str(B.SUCCESSOR_REPOSITORY_ROOT),
        "head": SUCCESSOR_HEAD,
        "trainer_execution_bundle_sha256": SUCCESSOR_TRAINER_BUNDLE,
        "bridge_tool_bundle_sha256": SUCCESSOR_BRIDGE_BUNDLE,
        "runtime": runtime,
    }


def _authorized_candidate(
    tmp_path: Path, manifest: dict, *, output_root: Path | None = None
) -> dict:
    manifest_path = tmp_path / "SEMANTIC_COMPARISON.json"
    manifest_path.write_bytes(B.canonical_json_bytes(manifest))
    observed = _observed_successor()
    candidate = B.n3_authorization_template(
        successor_head=SUCCESSOR_HEAD,
        successor_trainer_bundle_sha256=SUCCESSOR_TRAINER_BUNDLE,
        bridge_tool_bundle_sha256=SUCCESSOR_BRIDGE_BUNDLE,
        successor_runtime_fingerprint_sha256=observed["runtime"]["runtime_fingerprint_sha256"],
        semantic_comparison_manifest_path=manifest_path,
        semantic_comparison_manifest_sha256=B.file_sha256(manifest_path),
        output_root=output_root or tmp_path / "bridge-output",
    )
    candidate.update(
        authorization_status="AUTHORIZED",
        authorizes_bridge_execution=True,
        authorized_by="bounded-test-owner",
        authorized_at="2026-09-03T00:00:00Z",
    )
    return candidate


def _validate(candidate: dict, manifest: dict, source: Path, successor: Path) -> dict:
    return B._validate_bridge_authorization_claims(
        candidate,
        observed_successor=_observed_successor(),
        semantic_comparison_manifest=manifest,
        exact_plan=_exact_plan(),
        existing_n2_evidence=_n2_evidence(),
        historical_runtime_fingerprint=_historical_runtime(),
        reopen_source_artifacts=False,
        source_root=source,
        successor_root=successor,
    )


def _fake_checkpoint(*, exact_plan_path: str = "/exact/plan.json") -> dict:
    rng = {
        "python": (3, (1, 2, 3), None),
        "numpy": ("MT19937", torch.arange(4).numpy(), 0, 0, 0.0),
        "torch_cpu": torch.arange(8, dtype=torch.uint8),
        "torch_cuda": [torch.arange(16, dtype=torch.uint8)],
    }
    sampler = {
        "stage": B.COMPLETION_STAGE,
        "sampler_seed": B.COMPLETION_SAMPLER_SEED,
        "permutation_identity": "exact-permutation",
        "range_start_position": 0,
        "invocation_range_start_position": B.COMPLETION_SAMPLER_ENDPOINT,
        "range_stop_position": B.COMPLETION_SAMPLER_ENDPOINT,
        "cursor": B.COMPLETION_SAMPLER_ENDPOINT,
    }
    grc = {
        "schema_version": launch.RUN_CONTRACT_SCHEMA,
        "kind": launch.GOVERNED_CHECKPOINT_KIND,
        "governed": True,
        "stage": B.COMPLETION_STAGE,
        "scope": "STAGE_N",
        "stage_start_step": 0,
        "stage_stop_step": B.COMPLETION_STEP,
        "stage_authorization_path": "/source/N2_AUTH.json",
        "stage_authorization_sha256": B.N2_AUTHORIZATION_SHA256,
        "trainer_branch": B.HISTORICAL_TRAINING_BRANCH,
        "trainer_head": B.HISTORICAL_TRAINING_HEAD,
        "trainer_execution_bundle_sha256": B.HISTORICAL_TRAINING_BUNDLE_SHA256,
        "launch_contract_sha256": "d" * 64,
        "exact_plan_path": exact_plan_path,
        "exact_run_plan_sha256": B.EXACT_RUN_PLAN_SHA256,
        "pilot_acceptance_path": "/pilot/acceptance.json",
        "pilot_owner_acceptance_sha256": B.PILOT_OWNER_ACCEPTANCE_SHA256,
        "runtime_fingerprint": {},
        "runtime_fingerprint_sha256": B.HISTORICAL_RUNTIME_FINGERPRINT_SHA256,
        "num_workers": 2,
        "sampler_identity": sampler,
        "training": {"compile": True},
        "compile_evidence": {"old": True},
        "compile_evidence_sha256": "e" * 64,
    }
    grc["governed_run_contract_sha256"] = launch.governed_digest(grc)
    dynamic = {
        "schema_version": "petitgpt-governed-checkpoint-state-v1",
        "active_stage": B.COMPLETION_STAGE,
        "active_stage_sampler_seed": B.COMPLETION_SAMPLER_SEED,
        "permutation_identity": "exact-permutation",
        "range_start_position": 0,
        "invocation_range_start_position": B.COMPLETION_SAMPLER_ENDPOINT,
        "range_stop_position": B.COMPLETION_SAMPLER_ENDPOINT,
        "cursor": B.COMPLETION_SAMPLER_ENDPOINT,
        "consumed": B.COMPLETION_SAMPLER_ENDPOINT,
        "remaining": 0,
        "global_step": B.COMPLETION_STEP,
        "completed_evaluation_milestones": [500, 3815, 11445, 22889, 38146],
        "completed_checkpoint_milestones": [3815, 11445, 22889, 38146],
        "rng_state": rng,
        "rng_state_present": True,
        "rng_state_streams": ["numpy", "python", "torch_cpu", "torch_cuda"],
        "compile_evidence": {"old": True},
        "compile_evidence_sha256": "e" * 64,
    }
    return {
        "kind": launch.GOVERNED_CHECKPOINT_KIND,
        "model": {
            f"tensor_{index:03d}": torch.tensor([index], dtype=torch.int64)
            for index in range(B.EXPECTED_MODEL_TENSOR_COUNT)
        },
        "optim": {"state": {0: {"momentum": torch.arange(3)}}, "param_groups": [{"lr": 0.1}]},
        "scaler": {"scale": 65536.0},
        "global_step": B.COMPLETION_STEP,
        "run_contract": {"tokens": 9999745024},
        "data_contract": {"release": "stage-a"},
        "rng_state": rng,
        "position_stats": {"serialized": 9999745024},
        "data_sampler": sampler,
        "governed_run_contract": grc,
        "governed_run_contract_sha256": grc["governed_run_contract_sha256"],
        "governed_checkpoint_state": dynamic,
    }


def test_template_is_not_authorized_and_exactly_locks_historical_chain():
    candidate = B.n3_authorization_template()
    assert candidate["authorization_status"] == "NOT_AUTHORIZED"
    assert candidate["authorizes_bridge_execution"] is False
    assert candidate["authorizes_training"] is False
    assert candidate["scope"] == B.N3_SCOPE
    assert candidate["source_execution"]["head"] == B.HISTORICAL_TRAINING_HEAD
    assert candidate["source_execution"]["trainer_execution_bundle_sha256"] == (
        B.HISTORICAL_TRAINING_BUNDLE_SHA256
    )
    assert candidate["source_checkpoint"]["sha256"] == B.N2_TERMINAL_CHECKPOINT_SHA256
    assert candidate["zero_update"] == dict(B.ZERO_UPDATE_LIMITS)
    assert candidate["successor"]["head"] is None


def test_exact_semantic_manifest_and_authorized_policy_pass(tmp_path):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    candidate = _authorized_candidate(tmp_path, manifest)
    assert manifest["file_count"] == 12
    assert manifest["changed_files"] == [B.SUCCESSOR_POLICY_FILE]
    assert _validate(candidate, manifest, source, successor) == {
        "schema_version": "petitgpt-stage-n-successor-bridge-preflight-v1",
        "authorized": True,
        "failures": [],
        "source_head": B.HISTORICAL_TRAINING_HEAD,
        "successor_head": SUCCESSOR_HEAD,
        "completion_step": B.COMPLETION_STEP,
        "sampler_endpoint": B.COMPLETION_SAMPLER_ENDPOINT,
    }

    candidate["stage_n_completion"] = {
        "expected_final_step": B.COMPLETION_STEP,
    }
    assert _validate(candidate, manifest, source, successor)["authorized"] is True


def test_bridge_rejects_present_mismatching_stage_n_completion(tmp_path):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    candidate = _authorized_candidate(tmp_path, manifest)
    candidate["stage_n_completion"] = {
        "expected_final_step": B.COMPLETION_STEP - 1,
    }

    verdict = _validate(candidate, manifest, source, successor)

    assert verdict["authorized"] is False
    assert "stage_n_completion_expected_final_step_differs_from_exact_plan" in verdict["failures"]


@pytest.mark.parametrize(
    "mutation",
    [
        "source_head",
        "source_bundle",
        "successor_head",
        "unreviewed_successor_head",
        "successor_bundle",
        "runtime",
        "optimizer_updates",
        "optimizer_updates_boolean_alias",
        "trained_tokens",
        "n2_terminal_checkpoint",
        "non_boolean_control",
        "top_level_schema_extension",
        "history_schema_extension",
        "successor_schema_extension",
        "manifest_binding_schema_extension",
        "destination_numeric_alias",
        "completion_schema_extension",
    ],
)
def test_exact_policy_rejects_identity_runtime_and_zero_update_drift(tmp_path, mutation):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    candidate = _authorized_candidate(tmp_path, manifest)
    observed = _observed_successor()
    n2 = _n2_evidence()
    if mutation == "source_head":
        candidate["source_execution"]["head"] = "f" * 40
    elif mutation == "source_bundle":
        candidate["source_execution"]["trainer_execution_bundle_sha256"] = "f" * 64
    elif mutation == "successor_head":
        observed["head"] = "f" * 40
    elif mutation == "unreviewed_successor_head":
        candidate["successor"]["head"] = "f" * 40
    elif mutation == "successor_bundle":
        candidate["successor"]["trainer_execution_bundle_sha256"] = "f" * 64
    elif mutation == "runtime":
        observed["runtime"]["driver_version"] = "changed"
        observed["runtime"]["runtime_fingerprint_sha256"] = launch.runtime_fingerprint_sha256(
            observed["runtime"]
        )
    elif mutation == "optimizer_updates":
        n2["n2_optimizer_step_calls"] = 1
    elif mutation == "optimizer_updates_boolean_alias":
        n2["n2_optimizer_step_calls"] = False
    elif mutation == "trained_tokens":
        n2["n2_trained_tokens"] = 1
    elif mutation == "non_boolean_control":
        candidate["authorizes_bridge_execution"] = 1
    elif mutation == "top_level_schema_extension":
        candidate["unreviewed_extension"] = True
    elif mutation == "history_schema_extension":
        candidate["history"]["unreviewed_extension"] = True
    elif mutation == "successor_schema_extension":
        candidate["successor"]["unreviewed_extension"] = True
    elif mutation == "manifest_binding_schema_extension":
        candidate["semantic_comparison_manifest"]["unreviewed_extension"] = True
    elif mutation == "destination_numeric_alias":
        candidate["destination"]["expected_checkpoint_step"] = float(B.COMPLETION_STEP)
    elif mutation == "completion_schema_extension":
        candidate["stage_n_completion"] = {
            "expected_final_step": B.COMPLETION_STEP,
            "unreviewed_extension": True,
        }
    else:
        n2["terminal_checkpoint"]["sha256"] = "f" * 64
    verdict = B._validate_bridge_authorization_claims(
        candidate,
        observed_successor=observed,
        semantic_comparison_manifest=manifest,
        exact_plan=_exact_plan(),
        existing_n2_evidence=n2,
        historical_runtime_fingerprint=_historical_runtime(),
        reopen_source_artifacts=False,
        source_root=source,
        successor_root=successor,
    )
    assert verdict["authorized"] is False
    assert verdict["failures"]


def test_exact_policy_rejects_nested_device_resolution_drift(tmp_path):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    candidate = _authorized_candidate(tmp_path, manifest)
    observed = _observed_successor()
    observed["runtime"]["selected_device_resolution"]["cuda_visible_devices"] = "0"
    changed_runtime_sha = launch.runtime_fingerprint_sha256(observed["runtime"])
    observed["runtime"]["runtime_fingerprint_sha256"] = changed_runtime_sha
    candidate["successor"]["runtime_fingerprint_sha256"] = changed_runtime_sha

    verdict = B._validate_bridge_authorization_claims(
        candidate,
        observed_successor=observed,
        semantic_comparison_manifest=manifest,
        exact_plan=_exact_plan(),
        existing_n2_evidence=_n2_evidence(),
        historical_runtime_fingerprint=_historical_runtime(),
        reopen_source_artifacts=False,
        source_root=source,
        successor_root=successor,
    )

    assert verdict["authorized"] is False
    assert "n3_successor_runtime_not_exact_historical_projection" in verdict["failures"]


def test_changed_training_semantics_file_rejects_bridge(tmp_path):
    source, successor = _write_semantic_roots(tmp_path)
    (successor / B.CORE_TRAINING_SEMANTICS_FILES[0]).write_bytes(b"changed training semantics")
    manifest = _semantic_manifest(source, successor)
    candidate = _authorized_candidate(tmp_path, manifest)
    verdict = _validate(candidate, manifest, source, successor)
    assert verdict["authorized"] is False
    assert "semantic_comparison_mismatch:changed_file_count" in verdict["failures"]
    assert (
        "semantic_comparison_mismatch:core_training_semantics_files_changed" in verdict["failures"]
    )


def test_semantic_manifest_rejects_boolean_integer_alias(tmp_path):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    manifest["changed_file_count"] = True
    manifest["manifest_content_sha256"] = B.sha256_bytes(
        B.canonical_json_bytes(B._manifest_projection(manifest))
    )
    artifact_sha = B.sha256_bytes(B.canonical_json_bytes(manifest))

    failures = B.validate_semantic_comparison_manifest(
        manifest,
        expected_artifact_sha256=artifact_sha,
        expected_successor_head=SUCCESSOR_HEAD,
        expected_successor_trainer_bundle_sha256=SUCCESSOR_TRAINER_BUNDLE,
        source_root=source,
        successor_root=successor,
    )

    assert "semantic_comparison_mismatch:changed_file_count" in failures


@pytest.mark.parametrize(
    ("location", "expected_failure"),
    [
        ("document", "semantic_comparison_manifest_field_set_mismatch"),
        (
            "record",
            "semantic_comparison_record_field_set_mismatch:pretrain/dataset_pretrain.py",
        ),
    ],
)
def test_semantic_manifest_rejects_unknown_fields(tmp_path, location, expected_failure):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    if location == "document":
        manifest["unreviewed_extension"] = True
    else:
        manifest["files"][0]["unreviewed_extension"] = True
    manifest["manifest_content_sha256"] = B.sha256_bytes(
        B.canonical_json_bytes(B._manifest_projection(manifest))
    )
    artifact_sha = B.sha256_bytes(B.canonical_json_bytes(manifest))

    failures = B.validate_semantic_comparison_manifest(
        manifest,
        expected_artifact_sha256=artifact_sha,
        expected_successor_head=SUCCESSOR_HEAD,
        expected_successor_trainer_bundle_sha256=SUCCESSOR_TRAINER_BUNDLE,
        source_root=source,
        successor_root=successor,
    )

    assert expected_failure in failures


def test_state_equivalence_accepts_only_compile_and_provenance_changes():
    source = _fake_checkpoint()
    destination = copy.deepcopy(source)
    destination["governed_run_contract"] = {"successor": True}
    destination["governed_run_contract_sha256"] = "f" * 64
    destination["governed_checkpoint_state"]["compile_evidence"] = {"successor": True}
    destination["governed_checkpoint_state"]["compile_evidence_sha256"] = "f" * 64
    counters = {
        "optimizer_updates": 0,
        "trained_tokens": 0,
        "sampler_advances": 0,
        "training_loop_iterations": 0,
        "backward_calls": 0,
        "scheduler_advances": 0,
        "data_batches_consumed": 0,
    }
    verdict = B.validate_state_equivalence(source, destination, execution_counters=counters)
    assert verdict["equivalent"] is True
    assert verdict["model_tensors_compared"] == B.EXPECTED_MODEL_TENSOR_COUNT

    for field in ("model", "optim", "scaler"):
        changed = copy.deepcopy(destination)
        if field == "model":
            changed[field]["tensor_000"][0] += 1
        elif field == "optim":
            changed[field]["param_groups"][0]["lr"] = 0.2
        else:
            changed[field]["scale"] = 1.0
        rejected = B.validate_state_equivalence(source, changed, execution_counters=counters)
        assert rejected["equivalent"] is False
        assert f"bridge_state_difference:{field}" in rejected["failures"]

    changed_cursor = copy.deepcopy(destination)
    changed_cursor["governed_checkpoint_state"]["cursor"] -= 1
    rejected = B.validate_state_equivalence(source, changed_cursor, execution_counters=counters)
    assert rejected["equivalent"] is False
    assert "bridge_dynamic_state_difference:cursor" in rejected["failures"]

    changed_unknown = copy.deepcopy(destination)
    changed_unknown["future_unknown_field"] = "not-authorized"
    rejected = B.validate_state_equivalence(source, changed_unknown, execution_counters=counters)
    assert "bridge_checkpoint_field_set_difference" in rejected["failures"]

    changed_dynamic_unknown = copy.deepcopy(destination)
    changed_dynamic_unknown["governed_checkpoint_state"]["future_unknown"] = True
    rejected = B.validate_state_equivalence(
        source, changed_dynamic_unknown, execution_counters=counters
    )
    assert "bridge_dynamic_state_field_set_difference" in rejected["failures"]

    missing_tensor = copy.deepcopy(destination)
    missing_tensor["model"].pop("tensor_212")
    rejected = B.validate_state_equivalence(source, missing_tensor, execution_counters=counters)
    assert "bridge_model_tensor_count_mismatch" not in rejected["failures"]
    assert "bridge_state_difference:model" in rejected["failures"]

    missing_counter = dict(counters)
    missing_counter.pop("backward_calls")
    rejected = B.validate_state_equivalence(source, destination, execution_counters=missing_counter)
    assert "bridge_execution_counter_set_mismatch" in rejected["failures"]


def test_public_execution_has_no_callbacks_and_stops_before_checkpoint_load(tmp_path, monkeypatch):
    source, successor = _write_semantic_roots(tmp_path)
    manifest = _semantic_manifest(source, successor)
    candidate = _authorized_candidate(tmp_path, manifest)
    candidate["authorization_status"] = "NOT_AUTHORIZED"
    candidate["authorizes_bridge_execution"] = False
    auth_path = tmp_path / "N3_NOT_AUTHORIZED.json"
    auth_path.write_bytes(B.canonical_json_bytes(candidate))
    plan_path = tmp_path / "EXACT_RUN_PLAN.json"
    plan_path.write_bytes(B.canonical_json_bytes(_exact_plan()))
    n2_path = tmp_path / "N2.json"
    n2_path.write_bytes(B.canonical_json_bytes(_n2_evidence()))
    called = False

    def forbidden_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("checkpoint must not be opened")

    public_parameters = inspect.signature(B.execute_authorized_bridge).parameters
    assert "compile_realizer" not in public_parameters
    assert "checkpoint_loader" not in public_parameters
    assert "checkpoint_saver" not in public_parameters
    assert "expected_model_tensor_count" not in public_parameters
    assert not hasattr(B, "_execute_authorized_bridge_with_adapters")
    validator_parameters = inspect.signature(B.validate_bridge_authorization).parameters
    assert "observed_successor" not in validator_parameters
    assert "source_root" not in validator_parameters
    assert "successor_root" not in validator_parameters
    monkeypatch.setattr(B, "_read_bound_checkpoint_snapshot", forbidden_loader)
    with pytest.raises(B.CompatibilityBridgeError):
        B.execute_authorized_bridge(
            authorization_path=auth_path,
            semantic_comparison_manifest_path=candidate["semantic_comparison_manifest"]["path"],
            exact_plan_path=plan_path,
            existing_n2_evidence_path=n2_path,
        )
    with pytest.raises(B.CompatibilityBridgeError):
        B.main([
            "execute",
            "--authorization-path",
            str(auth_path),
            "--semantic-comparison-manifest-path",
            candidate["semantic_comparison_manifest"]["path"],
            "--exact-plan-path",
            str(plan_path),
            "--existing-n2-evidence-path",
            str(n2_path),
        ])
    assert called is False


@pytest.mark.parametrize("reject_staged_complete", [False, True])
def test_tiny_authorized_state_machine_publishes_canonical_zero_update_results(
    tmp_path, monkeypatch, reject_staged_complete
):
    semantic_source, semantic_successor = _write_semantic_roots(tmp_path)
    monkeypatch.setattr(B, "CANONICAL_TRAINING_ROOT", semantic_source)
    monkeypatch.setattr(B, "SUCCESSOR_REPOSITORY_ROOT", semantic_successor)
    plan_path = tmp_path / "historical" / "EXACT_RUN_PLAN.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(B.canonical_json_bytes(_exact_plan()))
    plan_sha = B.file_sha256(plan_path)
    monkeypatch.setattr(B, "EXACT_RUN_PLAN_SHA256", plan_sha)
    monkeypatch.setattr(launch, "EXACT_RUN_PLAN_SHA256", plan_sha)

    source_checkpoint = _fake_checkpoint(exact_plan_path=str(plan_path))
    source_checkpoint["governed_run_contract"]["exact_run_plan_sha256"] = plan_sha
    source_checkpoint["governed_run_contract"]["governed_run_contract_sha256"] = (
        launch.governed_digest(source_checkpoint["governed_run_contract"])
    )
    source_checkpoint["governed_run_contract_sha256"] = source_checkpoint["governed_run_contract"][
        "governed_run_contract_sha256"
    ]
    checkpoint_path = tmp_path / "historical" / "N2_TERMINAL.pt"
    checkpoint_path.write_bytes(pickle.dumps(source_checkpoint))
    checkpoint_sha = B.file_sha256(checkpoint_path)
    monkeypatch.setattr(B, "N2_TERMINAL_CHECKPOINT_SHA256", checkpoint_sha)

    historical_runtime = _historical_runtime()
    monkeypatch.setattr(
        B,
        "HISTORICAL_RUNTIME_FINGERPRINT_SHA256",
        historical_runtime["runtime_fingerprint_sha256"],
    )

    n2_evidence = _n2_evidence()
    n2_evidence["terminal_checkpoint"] = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
    }
    n2_path = tmp_path / "historical" / "N2_V2_ZERO_UPDATE_INVARIANCE.json"
    n2_path.write_bytes(B.canonical_json_bytes(n2_evidence))

    artifacts: dict[str, dict[str, str]] = {}
    for name in B.HISTORICAL_ARTIFACTS:
        path = tmp_path / "historical" / f"{name}.artifact"
        if name == "exact_plan":
            path = plan_path
        elif name == "n2_terminal_checkpoint":
            path = checkpoint_path
        elif name == "n2_zero_update_invariance":
            path = n2_path
        elif name in ("n1_evidence_manifest", "n2_evidence_manifest"):
            path.write_bytes(b"")
        elif name == "n2_process_exit":
            path.write_bytes(b"EXIT=0\n")
        elif name == "n2_final_state":
            path.write_bytes(
                B.canonical_json_bytes({
                    "schema_version": "petitgpt-n2-v2-final-state-v1",
                    "branch": B.HISTORICAL_TRAINING_BRANCH,
                    "head": B.HISTORICAL_TRAINING_HEAD,
                    "head_unchanged": True,
                    "runtime_unchanged": True,
                    "trainer_bundle_unchanged": True,
                    "protected_probe_unchanged": True,
                    "tracked_clean": True,
                    "stage_n_owner_acceptance_published": False,
                    "stage_n_result_published": False,
                    "stage_o_authorized": False,
                    "stage_o_started": False,
                    "gpu_uuid": B.RUNTIME_INVARIANTS["gpu_uuid"],
                    "gpu_pci": B.RUNTIME_INVARIANTS["gpu_pci_bus_id"],
                })
            )
        elif name == "n2_launch_record":
            path.write_bytes(
                B.canonical_json_bytes({
                    "schema_version": "petitgpt-n2-v2-launch-record-v1",
                    "authorization_sha256": B.N2_AUTHORIZATION_SHA256,
                    "cwd": str(semantic_source),
                    "internal_helpers_called_directly": False,
                })
            )
        elif name == "n2_runtime_fingerprint":
            path.write_bytes(
                B.canonical_json_bytes(launch.stage_n_runtime_artifact_document(historical_runtime))
            )
        else:
            path.write_bytes(f"{name}\n".encode())
        artifacts[name] = {"path": str(path), "sha256": B.file_sha256(path)}
    monkeypatch.setattr(B, "HISTORICAL_ARTIFACTS", artifacts)
    n2_evidence["source_checkpoint"] = dict(artifacts["n2_source_checkpoint"])
    n2_evidence["terminal_checkpoint"] = dict(artifacts["n2_terminal_checkpoint"])
    n2_path.write_bytes(B.canonical_json_bytes(n2_evidence))
    artifacts["n2_zero_update_invariance"]["sha256"] = B.file_sha256(n2_path)

    manifest = _semantic_manifest(semantic_source, semantic_successor)
    observed = _observed_successor()
    observed["repository_root"] = str(semantic_successor)
    historical = {
        "repository_root": str(semantic_source),
        "branch": B.HISTORICAL_TRAINING_BRANCH,
        "head": B.HISTORICAL_TRAINING_HEAD,
        "trainer_execution_bundle_sha256": B.HISTORICAL_TRAINING_BUNDLE_SHA256,
        "tracked_clean": True,
        "untracked": [B.HISTORICAL_UNTRACKED_PROBE_RELPATH],
        "untracked_probe_sha256": B.HISTORICAL_UNTRACKED_PROBE_SHA256,
    }
    monkeypatch.setattr(B, "_assert_successor_module_origins", lambda: None)
    monkeypatch.setattr(B, "observe_historical_identity", lambda: copy.deepcopy(historical))
    monkeypatch.setattr(B, "observe_successor_identity", lambda: observed)
    monkeypatch.setattr(launch, "verify_compile_evidence_document", lambda _evidence: [])
    monkeypatch.setattr(launch, "validate_governed_checkpoint_state", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launch, "validate_governed_checkpoint_resume_envelope", lambda _checkpoint: []
    )

    candidate = _authorized_candidate(tmp_path, manifest, output_root=tmp_path / "bridge-output")
    candidate["history"]["exact_run_plan_sha256"] = plan_sha
    candidate["history"]["artifacts"] = copy.deepcopy(artifacts)
    candidate["source_checkpoint"] = {
        **dict(artifacts["n2_terminal_checkpoint"]),
        "step": B.COMPLETION_STEP,
        "stage": B.COMPLETION_STAGE,
        "sampler_cursor": B.COMPLETION_SAMPLER_ENDPOINT,
        "sampler_seed": B.COMPLETION_SAMPLER_SEED,
    }
    auth_path = tmp_path / "N3_AUTHORIZED.json"
    auth_path.write_bytes(B.canonical_json_bytes(candidate))

    def fake_realizer(_checkpoint, *, authorization_sha256):
        assert authorization_sha256 == B.file_sha256(auth_path)
        state_proof = {
            "model_parameters_bitwise_identical": True,
            "optimizer_state_equivalent": True,
            "scaler_state_equivalent": True,
            "rng_state_preserved": True,
            "all_parameter_grads_absent": True,
            "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
        }
        return {
            "compile_evidence": {
                "compile_evidence_sha256": "f" * 64,
                "verdict": "PASS",
                "bridge_zero_update_observations": {
                    "execution_counters": dict(B.ZERO_EXECUTION_COUNTERS),
                    "before_compile": copy.deepcopy(state_proof),
                    "after_compile": copy.deepcopy(state_proof),
                    "training_loop_constructed": False,
                    "data_loader_constructed": False,
                    "sampler_constructed": False,
                    "scheduler_constructed": False,
                },
            },
            "execution_counters": dict(B.ZERO_EXECUTION_COUNTERS),
            "live_state_proof": state_proof,
        }

    monkeypatch.setattr(B, "_canonical_zero_update_compile_realization", fake_realizer)
    monkeypatch.setattr(B, "_deserialize_checkpoint_snapshot", pickle.loads)
    monkeypatch.setattr(
        B,
        "_serialize_checkpoint_document",
        lambda document: pickle.dumps(dict(document)),
    )
    real_complete_validator = B._validate_successor_complete_result_document
    output_root = Path(candidate["destination"]["output_root"])

    def staged_complete_validator(document, **kwargs):
        if kwargs.get("physical_output_root") is not None:
            assert not output_root.exists()
            if reject_staged_complete:
                return ["injected_staged_tree_failure"]
        return real_complete_validator(document, **kwargs)

    monkeypatch.setattr(
        B,
        "_validate_successor_complete_result_document",
        staged_complete_validator,
    )

    def execution():
        return B.execute_authorized_bridge(
            authorization_path=auth_path,
            semantic_comparison_manifest_path=candidate["semantic_comparison_manifest"]["path"],
            exact_plan_path=plan_path,
            existing_n2_evidence_path=n2_path,
        )

    if reject_staged_complete:
        with pytest.raises(B.CompatibilityBridgeError, match="injected_staged_tree_failure"):
            execution()
        assert not output_root.exists()
        return
    result = execution()
    assert result["status"] == "COMPLETE_STOPPED_FOR_INDEPENDENT_REVIEW"
    assert set(result["execution_counters"].values()) == {0}
    assert result["state_equivalence"]["equivalent"] is True
    complete = json.loads(Path(result["results"]["complete"]["path"]).read_text())
    assert complete["schema_version"] == launch.STAGE_N_RESULT_SCHEMA
    assert complete["stage_n_training_execution_head"] == B.HISTORICAL_TRAINING_HEAD
    assert complete["stage_n_compatibility_bridge_head"] == SUCCESSOR_HEAD
    assert complete["stage_o_execution_head"] == SUCCESSOR_HEAD
    assert complete["stage_n_compatibility_bridge"]["optimizer_updates"] == 0
    assert complete["stage_n_compatibility_bridge"]["state_equivalence"]["equivalent"] is True
    resume_evidence = json.loads(Path(candidate["destination"]["resume_evidence_path"]).read_text())
    assert resume_evidence["schema_version"] == B.BRIDGE_RESUME_EVIDENCE_SCHEMA
    assert resume_evidence["kind"] == B.BRIDGE_RESUME_EVIDENCE_KIND
    assert (
        resume_evidence["governed_run_contract_artifact_sha256"]
        == (complete["governed_run_contract_artifact_sha256"])
    )
    assert (
        resume_evidence["governed_run_contract_sha256"]
        == (complete["governed_run_contract_sha256"])
    )
    assert output_root.is_dir()
    assert not list(output_root.parent.glob(f".{output_root.name}.n3-staging-*"))
    assert {path.name for path in output_root.iterdir() if path.is_file()} == {
        "GOVERNED_RUN_CONTRACT.json",
        "step_038146.pt",
        launch.STAGE_N_RUNTIME_FILENAME,
        "STAGE_N_BRIDGE_COMPILE_EVIDENCE.json",
        "STAGE_N_BRIDGE_STATE_EQUIVALENCE.json",
        "STAGE_N_SMOKE_RESULT.json",
        "STAGE_N_RESUME_RESULT.json",
        "STAGE_N_COMPLETE_RESULT.json",
    }


def test_checkpoint_deserialization_uses_the_authenticated_snapshot(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source.pt"
    original = {"identity": "authenticated"}
    replacement = {"identity": "raced-path"}
    original_bytes = pickle.dumps(original)
    checkpoint.write_bytes(original_bytes)

    def mutate_path_then_deserialize(snapshot):
        checkpoint.write_bytes(pickle.dumps(replacement))
        return pickle.loads(snapshot)

    monkeypatch.setattr(B, "_deserialize_checkpoint_snapshot", mutate_path_then_deserialize)
    snapshot, loaded = B._read_bound_checkpoint_snapshot(
        checkpoint,
        expected_sha256=B.sha256_bytes(original_bytes),
        label="test source",
    )
    assert snapshot == original_bytes
    assert loaded == original
    assert pickle.loads(checkpoint.read_bytes()) == replacement


def test_malformed_checkpoint_snapshot_fails_closed(tmp_path):
    checkpoint = tmp_path / "malformed.pt"
    checkpoint.write_bytes(b"not a torch checkpoint\n")

    with pytest.raises(B.CompatibilityBridgeError, match="cannot be deserialized"):
        B._read_bound_checkpoint_snapshot(
            checkpoint,
            expected_sha256=B.file_sha256(checkpoint),
            label="malformed",
        )


def test_checkpoint_binding_requires_one_compile_evidence_graph(monkeypatch):
    checkpoint = _fake_checkpoint()
    smoke_evidence = {
        "compile_evidence_sha256": "a" * 64,
        "origin": "smoke-and-checkpoint",
    }
    governed_run_contract = copy.deepcopy(checkpoint["governed_run_contract"])
    governed_run_contract["compile_evidence"] = {
        "compile_evidence_sha256": "b" * 64,
        "origin": "different-grc",
    }
    governed_run_contract["compile_evidence_sha256"] = "b" * 64
    governed_run_contract["governed_run_contract_sha256"] = launch.governed_digest(
        governed_run_contract
    )
    checkpoint["governed_run_contract"] = governed_run_contract
    checkpoint["governed_run_contract_sha256"] = governed_run_contract[
        "governed_run_contract_sha256"
    ]
    checkpoint["governed_checkpoint_state"]["compile_evidence"] = smoke_evidence
    checkpoint["governed_checkpoint_state"]["compile_evidence_sha256"] = "a" * 64
    monkeypatch.setattr(launch, "validate_governed_checkpoint_state", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launch, "validate_governed_checkpoint_resume_envelope", lambda _checkpoint: []
    )

    failures = B._validate_successor_checkpoint_binding(
        checkpoint,
        governed_run_contract=governed_run_contract,
        compile_evidence=smoke_evidence,
        require_live_cuda_validation=False,
    )

    assert "successor_grc_compile_evidence_mismatch" in failures
    assert "successor_grc_compile_evidence_sha256_mismatch" in failures
    assert "successor_checkpoint_compile_evidence_mismatch" not in failures


def test_bridge_compile_evidence_requires_exact_zero_work_observations(monkeypatch):
    """Bridge-only facts remain mandatory even when the generic compile seal is valid."""

    monkeypatch.setattr(launch, "verify_compile_evidence_document", lambda _evidence: [])
    state_proof = {
        "model_parameters_bitwise_identical": True,
        "optimizer_state_equivalent": True,
        "scaler_state_equivalent": True,
        "rng_state_preserved": True,
        "all_parameter_grads_absent": True,
        "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
    }
    observations = {
        "execution_counters": dict(B.ZERO_EXECUTION_COUNTERS),
        "before_compile": copy.deepcopy(state_proof),
        "after_compile": copy.deepcopy(state_proof),
        "training_loop_constructed": False,
        "data_loader_constructed": False,
        "sampler_constructed": False,
        "scheduler_constructed": False,
    }
    baseline = {"bridge_zero_update_observations": observations}
    assert B.validate_bridge_compile_evidence_document(baseline) == []

    missing = {}
    assert "bridge_zero_update_observations_missing" in (
        B.validate_bridge_compile_evidence_document(missing)
    )

    nonzero = copy.deepcopy(baseline)
    nonzero["bridge_zero_update_observations"]["execution_counters"]["optimizer_updates"] = 1
    assert "bridge_compile_execution_counters_nonzero_or_malformed" in (
        B.validate_bridge_compile_evidence_document(nonzero)
    )

    boolean_alias = copy.deepcopy(baseline)
    boolean_alias["bridge_zero_update_observations"]["execution_counters"]["optimizer_updates"] = (
        False
    )
    assert "bridge_compile_execution_counters_nonzero_or_malformed" in (
        B.validate_bridge_compile_evidence_document(boolean_alias)
    )

    proof_drift = copy.deepcopy(baseline)
    proof_drift["bridge_zero_update_observations"]["after_compile"]["model_tensors_compared"] = (
        float(B.EXPECTED_MODEL_TENSOR_COUNT)
    )
    assert "bridge_compile_after_compile_state_proof_mismatch" in (
        B.validate_bridge_compile_evidence_document(proof_drift)
    )

    constructed = copy.deepcopy(baseline)
    constructed["bridge_zero_update_observations"]["training_loop_constructed"] = True
    assert "bridge_compile_forbidden_surface_observed:training_loop_constructed" in (
        B.validate_bridge_compile_evidence_document(constructed)
    )

    extra = copy.deepcopy(baseline)
    extra["bridge_zero_update_observations"]["unreviewed"] = False
    assert "bridge_zero_update_observations_schema_mismatch" in (
        B.validate_bridge_compile_evidence_document(extra)
    )


def test_authoritative_checkpoint_revalidation_rejects_forged_equivalence_claim(
    tmp_path, monkeypatch
):
    source = _fake_checkpoint()
    destination = copy.deepcopy(source)
    destination["model"]["tensor_000"] = torch.tensor([-1], dtype=torch.int64)
    source_path = tmp_path / "source.pt"
    destination_path = tmp_path / "successor.pt"
    source_bytes = pickle.dumps(source)
    destination_bytes = pickle.dumps(destination)
    source_path.write_bytes(source_bytes)
    destination_path.write_bytes(destination_bytes)
    artifacts = copy.deepcopy(B.HISTORICAL_ARTIFACTS)
    artifacts["n2_terminal_checkpoint"] = {
        "path": str(source_path),
        "sha256": B.sha256_bytes(source_bytes),
    }
    monkeypatch.setattr(B, "HISTORICAL_ARTIFACTS", artifacts)
    monkeypatch.setattr(B, "_deserialize_checkpoint_snapshot", pickle.loads)
    monkeypatch.setattr(launch, "validate_governed_checkpoint_state", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        launch, "validate_governed_checkpoint_resume_envelope", lambda _checkpoint: []
    )
    forged_equivalence = B.validate_state_equivalence(
        source,
        source,
        execution_counters=B.ZERO_EXECUTION_COUNTERS,
    )
    result = B._revalidate_successor_checkpoint_state(
        destination_path,
        destination_sha256=B.sha256_bytes(destination_bytes),
        governed_run_contract=source["governed_run_contract"],
        compile_evidence=source["governed_checkpoint_state"]["compile_evidence"],
        recorded_state_equivalence=forged_equivalence,
    )
    assert any("bridge_state_difference:model" in failure for failure in result["failures"])
    assert "successor_checkpoint_recorded_state_equivalence_mismatch" in result["failures"]


def test_atomic_directory_publish_never_replaces_a_raced_destination(tmp_path):
    staging = tmp_path / ".result.n3-staging-test"
    destination = tmp_path / "result"
    staging.mkdir()
    (staging / "COMPLETE.json").write_text("staged", encoding="utf-8")
    destination.mkdir()
    marker = destination / "raced-owner-file"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(B.CompatibilityBridgeError, match="no-replace"):
        B._rename_directory_noreplace(staging, destination)

    assert staging.is_dir()
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_dual_root_import_pins_launch_contract_to_successor_worktree():
    repair_root = Path(B.__file__).resolve().parent.parent
    canonical_cwd = B.CANONICAL_TRAINING_ROOT.resolve()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repair_root)
    command = (
        "import json; from pathlib import Path; "
        "from pretrain import production_launch_contract_v1 as historical_launch; "
        "from pretrain import stage_n_successor_head_compatibility_bridge_v1 as B; "
        "print(json.dumps({'cwd': str(Path.cwd().resolve()), "
        "'bridge': str(Path(B.__file__).resolve()), "
        "'historical_launch': str(Path(historical_launch.__file__).resolve()), "
        "'launch': str(Path(B.launch.__file__).resolve())}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=canonical_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert Path(observed["cwd"]) == canonical_cwd
    assert (
        Path(observed["bridge"])
        == (repair_root / "pretrain/stage_n_successor_head_compatibility_bridge_v1.py").resolve()
    )
    assert (
        Path(observed["historical_launch"])
        == (canonical_cwd / "pretrain/production_launch_contract_v1.py").resolve()
    )
    assert (
        Path(observed["launch"])
        == (repair_root / "pretrain/production_launch_contract_v1.py").resolve()
    )
    model_module = B._load_exact_successor_module(
        "_petitgpt_successor_src_model_test", "src/model.py"
    )
    optimizer_module = B._load_exact_successor_module(
        "_petitgpt_successor_src_optim_test", "src/optim.py"
    )
    assert Path(model_module.__file__).resolve() == (repair_root / "src/model.py").resolve()
    assert Path(optimizer_module.__file__).resolve() == (repair_root / "src/optim.py").resolve()


def test_bridge_tool_closure_is_closed_and_includes_successor_trainer():
    closure = B.bridge_tool_closure()
    assert B.BRIDGE_TOOL_RELPATH in closure["derived_closure"]
    assert closure["unbound_load_bearing_module_count"] == 0
    assert len(closure["BRIDGE_TOOL_BUNDLE_SHA256"]) == 64
    assert closure["successor_trainer_bundle_sha256"] == (launch.trainer_execution_bundle_sha256())
