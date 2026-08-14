from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain.run_plan_contract import (
    load_run_plan_binding,
    resolve_run_plan_sample_budget,
    synchronize_validated_run_plan_binding,
    validate_run_plan_args,
    validate_run_plan_dataset,
    validate_run_plan_resume_transition,
    validate_run_plan_validation_dataset,
)

TOKENIZER_SHA256 = "a" * 64
RELEASE_SHA256 = "b" * 64
REFERENCE_SHA256 = "d" * 64


def _args(tmp_path: Path, **updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "run_plan_json": str(tmp_path / "run_plan.json"),
        "run_plan_stage": "stage_a",
        "val_dir": str(tmp_path / "reference" / "val"),
        "strict_resume_contract": True,
        "seq_len": 8,
        "micro_bsz": 2,
        "grad_accum": 2,
        "warmup_steps": 1,
        "data_stage_start_step": 0,
        "max_steps": 5,
        "schedule_total_steps": 8,
        "lr_schedule": "wsd",
        "decay_start_step": 5,
        "decay_end_step": 8,
        "allow_schedule_branch": False,
        "allow_data_branch": False,
        "save_steps": [2, 5, 8],
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _plan(train_a: Path, train_b: Path) -> dict:
    reference_val = train_a.parent.parent / "reference" / "val"
    stage_a_release = {
        "manifest_sha256": RELEASE_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    stage_b_release = {
        "manifest_sha256": "c" * 64,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    return {
        "schema_version": 3,
        "plan_type": "deterministic_no_replacement_stage_a_b",
        "invariants": {
            "sampling_mode": "deterministic",
            "replacement": False,
            "implicit_replay": False,
            "explicit_replay": False,
            "full_production_provenance_chain": True,
        },
        "release_provenance": {
            "full_chain_validated": True,
            "stage_b_selection_stage": "stage_b",
            "shared_tokenizer_sha256": TOKENIZER_SHA256,
            "selection": {
                "manifest_sha256": "e" * 64,
                "stage_b_selection_stage": "stage_b",
            },
            "source_bindings": {
                "validated": True,
                "stage_b_selection_stage": "stage_b",
                "stage_a": [],
                "stage_b": [{"source_id": "premium"}],
            },
            "stage_a": {"manifest_sha256": RELEASE_SHA256},
            "stage_b": {"manifest_sha256": "c" * 64},
            "reference_validation": {
                "manifest_sha256": REFERENCE_SHA256,
            },
        },
        "inputs": {
            "stage_a_dir": str(train_a.resolve()),
            "stage_b_dir": str(train_b.resolve()),
            "stage_b_selection_stage": "stage_b",
            "seq_len": 8,
            "micro_bsz": 2,
            "grad_accum": 2,
            "warmup_steps": 1,
            "stage_a_exposures": 1,
            "stage_b_exposures": 1,
            "reference_val_dir": str(reference_val.resolve()),
        },
        "batch": {
            "sequences_per_optimizer_step": 4,
            "serialized_target_positions_per_optimizer_step": 32,
        },
        "boundaries": {
            "stage_a_start_step": 0,
            "stage_a_stop_step": 5,
            "stage_b_start_step": 5,
            "stage_b_global_stop_step": 8,
            "schedule_total_steps": 8,
        },
        "wsd_candidate": {
            "warmup_steps": 1,
            "decay_start_step": 5,
            "decay_end_step": 8,
        },
        "checkpoint_milestones": {
            "schema_version": 1,
            "absolute_steps": [2, 5, 8],
            "cli_save_steps": "2,5,8",
        },
        "stages": {
            "stage_a": {
                "source_dir": str(train_a.resolve()),
                "planned_optimizer_steps": 5,
                "requested_exposures": 1,
                "unique_blocks": 23,
                "candidate_exposure_blocks": 23,
                "consumed_blocks": 20,
                "dataset": {"release_validation": stage_a_release},
            },
            "stage_b": {
                "source_dir": str(train_b.resolve()),
                "planned_optimizer_steps": 3,
                "requested_exposures": 1,
                "unique_blocks": 15,
                "candidate_exposure_blocks": 15,
                "consumed_blocks": 12,
                "dataset": {"release_validation": stage_b_release},
            },
        },
    }


def _write_plan(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "run_plan.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_load_run_plan_binding_records_exact_artifact_and_budget(tmp_path: Path) -> None:
    train_a = tmp_path / "stage_a" / "train"
    train_b = tmp_path / "stage_b" / "train"
    train_a.mkdir(parents=True)
    train_b.mkdir(parents=True)
    path = _write_plan(tmp_path, _plan(train_a, train_b))

    binding = load_run_plan_binding(
        _args(tmp_path), train_dir=train_a, tokenizer_sha256=TOKENIZER_SHA256
    )

    assert binding is not None
    assert binding["plan_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert binding["stage"] == "stage_a"
    assert binding["expected_stage_samples"] == 20
    assert binding["stage_release_manifest_sha256"] == RELEASE_SHA256
    assert binding["reference_release_manifest_sha256"] == REFERENCE_SHA256
    assert binding["checkpoint_milestone_steps"] == [2, 5, 8]
    assert len(binding["checkpoint_milestones_sha256"]) == 64


def test_strict_mode_requires_paired_plan_and_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strict production"):
        validate_run_plan_args(_args(tmp_path, run_plan_json="", run_plan_stage=""))
    with pytest.raises(ValueError, match="supplied together"):
        validate_run_plan_args(_args(tmp_path, run_plan_stage=""))
    assert (
        load_run_plan_binding(
            _args(
                tmp_path,
                run_plan_json="",
                run_plan_stage="",
                strict_resume_contract=False,
            ),
            train_dir=tmp_path,
            tokenizer_sha256=TOKENIZER_SHA256,
        )
        is None
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda plan: plan["release_provenance"].update(full_chain_validated=False), "full_chain"),
        (lambda plan: plan["inputs"].update(seq_len=9), "inputs.seq_len"),
        (lambda plan: plan["stages"]["stage_a"].update(consumed_blocks=19), "arithmetic"),
        (
            lambda plan: plan["release_provenance"]["stage_a"].update(manifest_sha256="d" * 64),
            "hashes disagree",
        ),
    ],
)
def test_run_plan_rejects_tampered_launch_fields(tmp_path: Path, mutate, match: str) -> None:
    train_a = tmp_path / "stage_a" / "train"
    train_b = tmp_path / "stage_b" / "train"
    train_a.mkdir(parents=True)
    train_b.mkdir(parents=True)
    payload = _plan(train_a, train_b)
    mutate(payload)
    _write_plan(tmp_path, payload)

    with pytest.raises(RuntimeError, match=match):
        load_run_plan_binding(_args(tmp_path), train_dir=train_a, tokenizer_sha256=TOKENIZER_SHA256)


def test_run_plan_binds_tokenizer_train_dir_and_wsd(tmp_path: Path) -> None:
    train_a = tmp_path / "stage_a" / "train"
    train_b = tmp_path / "stage_b" / "train"
    other = tmp_path / "other"
    train_a.mkdir(parents=True)
    train_b.mkdir(parents=True)
    other.mkdir()
    _write_plan(tmp_path, _plan(train_a, train_b))

    with pytest.raises(RuntimeError, match="tokenizer SHA"):
        load_run_plan_binding(_args(tmp_path), train_dir=train_a, tokenizer_sha256="e" * 64)
    with pytest.raises(RuntimeError, match="source_dir"):
        load_run_plan_binding(_args(tmp_path), train_dir=other, tokenizer_sha256=TOKENIZER_SHA256)
    with pytest.raises(RuntimeError, match="reference_val_dir"):
        load_run_plan_binding(
            _args(tmp_path, val_dir=str(other)),
            train_dir=train_a,
            tokenizer_sha256=TOKENIZER_SHA256,
        )
    with pytest.raises(RuntimeError, match="decay_start_step"):
        load_run_plan_binding(
            _args(tmp_path, decay_start_step=6),
            train_dir=train_a,
            tokenizer_sha256=TOKENIZER_SHA256,
        )
    binding = load_run_plan_binding(
        _args(tmp_path, decay_start_step=6, allow_schedule_branch=True),
        train_dir=train_a,
        tokenizer_sha256=TOKENIZER_SHA256,
    )
    assert binding is not None

    with pytest.raises(RuntimeError, match="save_steps must exactly match"):
        load_run_plan_binding(
            _args(tmp_path, save_steps=[2, 8]), train_dir=train_a, tokenizer_sha256=TOKENIZER_SHA256
        )


def test_sample_budget_is_bound_to_plan_and_resume_suffix() -> None:
    assert resolve_run_plan_sample_budget(
        None,
        stage_sample_position=8,
        step_derived_stage_samples=20,
    ) == (20, 12)
    with pytest.raises(RuntimeError, match="step/batch arithmetic"):
        resolve_run_plan_sample_budget(
            {"expected_stage_samples": 24},
            stage_sample_position=8,
            step_derived_stage_samples=20,
        )
    with pytest.raises(RuntimeError, match="exceeds the frozen"):
        resolve_run_plan_sample_budget(
            {"expected_stage_samples": 20},
            stage_sample_position=21,
            step_derived_stage_samples=20,
        )


class _Dataset:
    def __init__(
        self,
        *,
        length: int = 23,
        manifest_sha256: str = RELEASE_SHA256,
        release_kind: str = "regular",
        split: str = "train",
    ):
        self.length = length
        self.manifest_sha256 = manifest_sha256
        self.release_kind = release_kind
        self.split = split

    def __len__(self) -> int:
        return self.length

    def stats(self) -> dict:
        return {
            "release_validation": {
                "manifest_sha256": self.manifest_sha256,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "release_kind": self.release_kind,
                "split": self.split,
            }
        }


def test_dataset_binding_rejects_changed_release_or_block_count(tmp_path: Path) -> None:
    train_a = tmp_path / "stage_a" / "train"
    train_b = tmp_path / "stage_b" / "train"
    train_a.mkdir(parents=True)
    train_b.mkdir(parents=True)
    _write_plan(tmp_path, _plan(train_a, train_b))
    binding = load_run_plan_binding(
        _args(tmp_path), train_dir=train_a, tokenizer_sha256=TOKENIZER_SHA256
    )
    assert binding is not None
    validate_run_plan_dataset(binding, _Dataset())
    with pytest.raises(RuntimeError, match="block count"):
        validate_run_plan_dataset(binding, _Dataset(length=22))
    with pytest.raises(RuntimeError, match="release manifest"):
        validate_run_plan_dataset(binding, _Dataset(manifest_sha256="f" * 64))

    reference = _Dataset(
        manifest_sha256=REFERENCE_SHA256,
        release_kind="reference",
        split="val",
    )
    validate_run_plan_validation_dataset(binding, reference)
    with pytest.raises(RuntimeError, match="validation release manifest"):
        validate_run_plan_validation_dataset(
            binding,
            _Dataset(manifest_sha256="f" * 64, release_kind="reference", split="val"),
        )


def test_resume_binding_accepts_only_same_plan_a_to_b_boundary(tmp_path: Path) -> None:
    saved = {
        "schema_version": 1,
        "status": "validated",
        "plan_path": str(tmp_path / "run_plan.json"),
        "plan_sha256": "1" * 64,
        "plan_schema_version": 3,
        "plan_type": "deterministic_no_replacement_stage_a_b",
        "stage": "stage_a",
        "stage_start_step": 0,
        "stage_stop_step": 5,
        "schedule_total_steps": 8,
        "expected_stage_samples": 20,
        "requested_exposures": 1,
        "unique_blocks": 23,
        "candidate_exposure_blocks": 23,
        "stage_release_manifest_sha256": RELEASE_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "reference_release_manifest_sha256": REFERENCE_SHA256,
        "checkpoint_milestones_schema_version": 1,
        "checkpoint_milestone_steps": [2, 5, 8],
        "checkpoint_milestones_sha256": "3" * 64,
    }
    validate_run_plan_resume_transition(saved, saved, checkpoint_step=3)

    current = deepcopy(saved)
    current.update(
        stage="stage_b",
        stage_start_step=5,
        stage_stop_step=8,
        expected_stage_samples=12,
        unique_blocks=15,
        candidate_exposure_blocks=15,
        stage_release_manifest_sha256="c" * 64,
    )
    validate_run_plan_resume_transition(saved, current, checkpoint_step=5)
    with pytest.raises(RuntimeError, match="checkpoint step"):
        validate_run_plan_resume_transition(saved, current, checkpoint_step=4)
    changed = deepcopy(current)
    changed["plan_sha256"] = "2" * 64
    with pytest.raises(RuntimeError, match="identity changed"):
        validate_run_plan_resume_transition(saved, changed, checkpoint_step=5)


def _load_data_branch_bindings(tmp_path: Path) -> tuple[dict, dict]:
    train_a = tmp_path / "stage_a" / "train"
    train_b = tmp_path / "stage_b" / "train"
    train_control = tmp_path / "stage_b_control" / "train"
    train_a.mkdir(parents=True)
    train_b.mkdir(parents=True)
    train_control.mkdir(parents=True)

    premium = _plan(train_a, train_b)
    control = deepcopy(premium)
    control["inputs"].update({
        "stage_b_dir": str(train_control.resolve()),
        "stage_b_selection_stage": "control",
    })
    control["release_provenance"]["stage_b_selection_stage"] = "control"
    control["release_provenance"]["selection"]["stage_b_selection_stage"] = "control"
    control["release_provenance"]["source_bindings"]["stage_b_selection_stage"] = "control"
    control["release_provenance"]["source_bindings"]["stage_b"] = [{"source_id": "control"}]
    control["release_provenance"]["stage_b"]["manifest_sha256"] = "f" * 64
    control["stages"]["stage_b"]["source_dir"] = str(train_control.resolve())
    control["stages"]["stage_b"]["dataset"]["release_validation"]["manifest_sha256"] = "f" * 64

    premium_path = tmp_path / "premium_plan.json"
    control_path = tmp_path / "control_plan.json"
    premium_path.write_text(json.dumps(premium, sort_keys=True), encoding="utf-8")
    control_path.write_text(json.dumps(control, sort_keys=True), encoding="utf-8")

    saved = load_run_plan_binding(
        _args(tmp_path, run_plan_json=str(premium_path)),
        train_dir=train_a,
        tokenizer_sha256=TOKENIZER_SHA256,
    )
    current = load_run_plan_binding(
        _args(
            tmp_path,
            run_plan_json=str(control_path),
            run_plan_stage="stage_b",
            data_stage_start_step=5,
            max_steps=8,
            allow_data_branch=True,
        ),
        train_dir=train_control,
        tokenizer_sha256=TOKENIZER_SHA256,
    )
    assert saved is not None
    assert current is not None
    return saved, current


def test_data_branch_accepts_only_a_end_and_persists_parent_lineage(tmp_path: Path) -> None:
    saved, current = _load_data_branch_bindings(tmp_path)
    launch_metadata_binding = deepcopy(current)

    validate_run_plan_resume_transition(
        saved,
        current,
        checkpoint_step=5,
        allow_data_branch=True,
    )

    lineage = current["data_branch_validation"]
    assert lineage["status"] == "validated"
    assert lineage["parent_plan_sha256"] == saved["plan_sha256"]
    assert lineage["current_plan_sha256"] == current["plan_sha256"]
    synchronize_validated_run_plan_binding({"run_plan": current}, launch_metadata_binding)
    assert launch_metadata_binding == current


def test_data_branch_rejects_missing_flag_wrong_boundary_and_budget_change(
    tmp_path: Path,
) -> None:
    saved, current = _load_data_branch_bindings(tmp_path)
    with pytest.raises(RuntimeError, match="identity changed"):
        validate_run_plan_resume_transition(saved, deepcopy(current), checkpoint_step=5)
    with pytest.raises(RuntimeError, match="checkpoint step"):
        validate_run_plan_resume_transition(
            saved,
            deepcopy(current),
            checkpoint_step=4,
            allow_data_branch=True,
        )
    changed = deepcopy(current)
    changed["stage_b_expected_samples"] += 4
    with pytest.raises(RuntimeError, match="immutable training fields"):
        validate_run_plan_resume_transition(
            saved,
            changed,
            checkpoint_step=5,
            allow_data_branch=True,
        )
