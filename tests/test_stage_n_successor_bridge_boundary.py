"""Focused successor-head tests for the shared exact-plan Stage-N boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import pytest

from pretrain import production_launch_contract_v1 as C


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
            "stage_a_stop_step": 38146,
            "stage_b_start_step": 38146,
            "stage_b_global_stop_step": 49590,
            "schedule_total_steps": 49590,
        },
        "stages": {
            "stage_a": {
                "planned_optimizer_steps": 38146,
                "consumed_blocks": 4882688,
                "consumed_exposure_blocks": 4882688,
                "consumed_serialized_target_positions": 9999745024,
            }
        },
        "checkpoint_milestones": {
            "schema_version": 1,
            "absolute_steps": [3815, 38146, 49590],
            "entries": [
                {
                    "absolute_step": 38146,
                    "actual_cumulative_consumed_transitions": 9999745024,
                    "reasons": ["stage_a_end"],
                }
            ],
        },
    }


def _write_authenticated_plan(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    path = tmp_path / "EXACT_RUN_PLAN.json"
    path.write_bytes(C.canonical_json_bytes(_exact_plan()))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(C, "EXACT_RUN_PLAN_SHA256", sha)
    return path, C.load_stage_n_completion_boundary(path)


def _grc(path: Path, sha: str) -> dict:
    return {
        "stage": "stage_a",
        "stage_start_step": 0,
        "stage_stop_step": 38146,
        "exact_plan_path": str(path),
        "exact_run_plan_sha256": sha,
        "sampler_identity": {
            "stage": "stage_a",
            "range_start_position": 0,
            "range_stop_position": 4882688,
        },
    }


def test_exact_plan_derives_the_single_stage_n_completion_boundary():
    boundary = C.derive_stage_n_completion_boundary(_exact_plan())
    assert boundary["stage"] == "stage_a"
    assert boundary["stop_step"] == 38146
    assert boundary["sampler_endpoint"] == 4882688
    assert boundary["sampler_endpoint"] == (
        boundary["stop_step"] * boundary["sequences_per_optimizer_step"]
    )


def test_plan_loader_authenticates_bytes_before_derivation(tmp_path, monkeypatch):
    path, boundary = _write_authenticated_plan(tmp_path, monkeypatch)
    assert boundary["exact_plan_path"] == str(path)

    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(C.LaunchContractError, match="SHA-256 mismatch"):
        C.load_stage_n_completion_boundary(path)


def test_plan_loader_hashes_and_parses_one_byte_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "EXACT_RUN_PLAN.json"
    snapshot = C.canonical_json_bytes(_exact_plan())
    path.write_bytes(snapshot)
    monkeypatch.setattr(C, "EXACT_RUN_PLAN_SHA256", hashlib.sha256(snapshot).hexdigest())
    original_read_bytes = Path.read_bytes
    reads = 0

    def alternating_read_bytes(candidate: Path) -> bytes:
        nonlocal reads
        if candidate == path:
            reads += 1
            return snapshot if reads == 1 else b"{}\n"
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", alternating_read_bytes)

    boundary = C.load_stage_n_completion_boundary(path)

    assert boundary["stop_step"] == C.STAGE_A_STOP_STEP
    assert reads == 1


def test_coherently_changed_plan_boundary_still_fails_closed():
    plan = copy.deepcopy(_exact_plan())
    stop = 38147
    endpoint = stop * 128
    positions = endpoint * 2048
    plan["boundaries"]["stage_a_stop_step"] = stop
    plan["boundaries"]["stage_b_start_step"] = stop
    plan["stages"]["stage_a"].update(
        planned_optimizer_steps=stop,
        consumed_blocks=endpoint,
        consumed_exposure_blocks=endpoint,
        consumed_serialized_target_positions=positions,
    )
    plan["checkpoint_milestones"]["absolute_steps"][1] = stop
    plan["checkpoint_milestones"]["entries"][0].update(
        absolute_step=stop,
        actual_cumulative_consumed_transitions=positions,
    )

    with pytest.raises(C.LaunchContractError, match="frozen launch boundary"):
        C.derive_stage_n_completion_boundary(plan)


@pytest.mark.parametrize(
    ("completion", "expected_failure"),
    [
        (None, None),
        ({"expected_final_step": 38146}, None),
        (
            {"expected_final_step": 38145},
            "stage_n_completion_expected_final_step_differs_from_exact_plan",
        ),
        (
            {"expected_final_step": 38146, "unreviewed_extension": True},
            "stage_n_completion_field_set_mismatch",
        ),
    ],
)
def test_completion_is_an_optional_redundant_cross_check(
    tmp_path, monkeypatch, completion, expected_failure
):
    path, boundary = _write_authenticated_plan(tmp_path, monkeypatch)
    auth = {"exact_run_plan_sha256": boundary["exact_run_plan_sha256"]}
    if completion is not None:
        auth["stage_n_completion"] = completion
    failures = C.validate_stage_n_boundary_agreement(
        auth,
        boundary,
        governed_run_contract=_grc(path, boundary["exact_run_plan_sha256"]),
        checkpoint_step=38146,
        sampler_state={
            "range_start_position": 0,
            "range_stop_position": 4882688,
            "cursor": 4882688,
        },
    )
    if expected_failure is None:
        assert failures == []
    else:
        assert expected_failure in failures


def test_explicit_null_completion_is_rejected_instead_of_treated_as_absent():
    authorization = {"stage_n_completion": None}
    boundary = C.derive_stage_n_completion_boundary(_exact_plan())

    assert C.validate_stage_n_completion_cross_check(authorization, boundary) == [
        "stage_n_completion_must_be_an_object"
    ]
    assert C.validate_stage_n_completion_binding(
        authorization,
        argparse.Namespace(max_steps=C.STAGE_A_STOP_STEP),
        stage="stage_a",
    ) == ["stage_n_completion_must_be_an_object"]


def test_execution_artifacts_cannot_collude_on_a_different_sampler_endpoint(tmp_path, monkeypatch):
    path, boundary = _write_authenticated_plan(tmp_path, monkeypatch)
    auth = {"exact_run_plan_sha256": boundary["exact_run_plan_sha256"]}
    grc = _grc(path, boundary["exact_run_plan_sha256"])
    grc["sampler_identity"]["range_stop_position"] -= 1
    failures = C.validate_stage_n_boundary_agreement(
        auth,
        boundary,
        governed_run_contract=grc,
        checkpoint_step=38146,
        sampler_state={
            "range_start_position": 0,
            "range_stop_position": 4882688,
            "cursor": 4882687,
        },
    )
    assert "stage_n_governed_run_contract_boundary_mismatch:range_stop_position" in failures
    assert "stage_n_sampler_state_differs_from_exact_plan_boundary:cursor" in failures
