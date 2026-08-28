"""Strict launch binding between production trainers and a frozen run plan."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

RUN_PLAN_SCHEMA_VERSION = 3
RUN_PLAN_STAGES = ("stage_a", "stage_b")
STAGE_B_SELECTION_STAGES = ("stage_b", "control")

# Provenance branch discriminators (DECISIONS D-146). Restated as literals rather than imported
# so this launch-path module keeps its zero local-import surface; the authoritative definitions
# live in pretrain/stage_p_native_provenance_v1.py and
# tests/test_stage_p_native_provenance_v1.py pins the two spellings to each other.
NATIVE_CHAIN_KIND = "accepted_stage_i_native_v1"
LEGACY_CHAIN_KIND = "legacy_selector_v1"
PROVENANCE_CHAIN_KINDS = (NATIVE_CHAIN_KIND, LEGACY_CHAIN_KIND)
NATIVE_PROVENANCE_SCHEMA = "petitgpt-accepted-stage-i-native-release-provenance-v2"

# R2-E. Two DISTINCT identities. The pre-merge chain identity covers what the native validator
# itself established; the post-merge data-branch identity covers the COMPLETE assembled native
# data authority this module consumes. They are not the same projection and are not named as if
# they were.
PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA = "petitgpt-stage-i-native-premerge-chain-identity-v1"
POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA = (
    "petitgpt-stage-p-native-post-merge-data-branch-identity-v1"
)

# Mirrors pretrain/stage_p_native_provenance_v1.PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS.
PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS = (
    "accepted_stage_i",
    "accepted_stage_i_identity_sha256",
    "candidate_m_plan_schema",
    "candidate_m_plan_sha256",
    "model_contract",
    "provenance_chain_kind",
    "schema_version",
    "shared_tokenizer_sha256",
    "stage_m_implementation_bundle_sha256",
    "stage_m_ordering_policy",
    "stage_p_native_validator_bundle_sha256",
    "stages",
)

# Serialized objects that only ever exist on the legacy selector-v1 branch. Derived from what
# `_validate_full_provenance` actually emits plus the legacy CLI artifacts, not from flag names.
NATIVE_FORBIDDEN_PROVENANCE_FIELDS = (
    "selection",
    "source_bindings",
    "selection_manifest",
    "selection_manifest_sha256",
    "selection_registry",
    "selection_audit",
    "selection_database",
    "sqlite_registry",
    "selected_jsonl",
    "selection_spec",
)

# Top-level objects a native provenance block must carry. Absence is a controlled failure, not
# a reason to fall back to the legacy branch.
# R2-D. Derived from what the native planner branch actually serializes after the merge, not
# from a prose list. Missing any of these is a controlled failure.
NATIVE_REQUIRED_PROVENANCE_FIELDS = (
    "accepted_stage_i",
    "accepted_stage_i_identity_sha256",
    "candidate_m_plan_schema",
    "candidate_m_plan_sha256",
    "candidate_plan_contract_field_count",
    "model_contract",
    "native_post_merge_data_branch_identity_sha256",
    "native_shared_authority_validated",
    "premerge_native_chain_identity_sha256",
    "provenance_chain_kind",
    "reference_validation",
    "schema_version",
    "shared_exclusion_authority",
    "shared_tokenizer_sha256",
    "stage_m_implementation_bundle_sha256",
    "stage_m_ordering_policy",
    "stage_p_native_validator_bundle_sha256",
    "stages",
    "tokenizer_release",
)

# Mirrors pretrain/stage_p_native_provenance_v1.POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS.
POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS = (
    "accepted_stage_i",
    "accepted_stage_i_identity_sha256",
    "candidate_m_plan_schema",
    "candidate_m_plan_sha256",
    "candidate_plan_contract_field_count",
    "model_contract",
    "native_shared_authority_validated",
    "premerge_native_chain_identity_sha256",
    "provenance_chain_kind",
    "reference_validation",
    "schema_version",
    "shared_exclusion_authority",
    "shared_tokenizer_sha256",
    "stage_m_implementation_bundle_sha256",
    "stage_m_ordering_policy",
    "stage_p_native_validator_bundle_sha256",
    "stages",
    "tokenizer_release",
)

POST_MERGE_STAGE_FIELDS = (
    "expected_accounting",
    "input_sequence_commitment",
    "manifest_sha256",
    "release_dir",
    "semantic_groups_checked",
    "shard_inventory_sha256",
    "shards",
    "stored_token_ids",
)

NATIVE_REQUIRED_STAGE_FIELDS = (
    "expected_accounting",
    "input_sequence_commitment",
    "manifest_sha256",
    "release_dir",
    "semantic_groups_checked",
    "shard_inventory_sha256",
    "shards",
    "stored_token_ids",
)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"run plan {field} must be a JSON object")
    return value


def _integer(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"run plan {field} must be an integer, got {value!r}")
    if value < 0 or (positive and value <= 0):
        relation = "positive" if positive else "non-negative"
        raise RuntimeError(f"run plan {field} must be {relation}, got {value!r}")
    return int(value)


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"run plan {field} must be a lowercase SHA-256")
    return value


def _same_int(actual: Any, expected: int, *, field: str) -> int:
    value = _integer(actual, field=field)
    if value != int(expected):
        raise RuntimeError(
            f"run plan {field} disagrees with launch: plan={value}, launch={int(expected)}"
        )
    return value


def _read_plan(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"--run_plan_json must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read run plan {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("run plan must contain a JSON object")
    return resolved, payload, hashlib.sha256(raw).hexdigest()


def _data_branch_immutable_sha256(plan: Mapping[str, Any]) -> str:
    """Hash every plan field except the explicitly branchable Stage-B data identity."""
    normalized = json.loads(json.dumps(plan, sort_keys=True))
    inputs = _mapping(normalized.get("inputs"), field="inputs")
    provenance = _mapping(normalized.get("release_provenance"), field="release_provenance")
    stages = _mapping(normalized.get("stages"), field="stages")
    marker = "<validated-stage-b-data-branch>"
    inputs["stage_b_dir"] = marker
    inputs["stage_b_selection_stage"] = marker
    provenance["stage_b_selection_stage"] = marker
    provenance["stage_b"] = marker
    # R1-C: branch-aware. A native plan has no selection/source_bindings objects and must not be
    # asked for them; a legacy plan is masked exactly as before, so its digest is unchanged.
    if provenance.get("provenance_chain_kind", LEGACY_CHAIN_KIND) == NATIVE_CHAIN_KIND:
        native_stages = _mapping(provenance.get("stages"), field="release_provenance.stages")
        native_stages["stage_b"] = marker
    else:
        source_bindings = _mapping(
            provenance.get("source_bindings"), field="release_provenance.source_bindings"
        )
        selection = _mapping(provenance.get("selection"), field="release_provenance.selection")
        source_bindings["stage_b_selection_stage"] = marker
        source_bindings["stage_b"] = marker
        selection["stage_b_selection_stage"] = marker
    stages["stage_b"] = marker
    normalized["totals"] = marker
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    """Byte-for-byte the encoding pretrain/stage_m_contract_v1.py uses.

    Restated here rather than imported so this launch-path module keeps its zero local-import
    surface; tests/test_stage_p_native_provenance_v1.py pins the two encoders to each other.
    """
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def premerge_native_chain_identity_sha256(provenance: Mapping[str, Any]) -> str:
    """Recompute the PRE-MERGE native chain identity. Not the final data-branch identity."""
    payload = {
        "schema_version": PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA,
        "fields": {name: provenance.get(name) for name in PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS},
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def post_merge_data_branch_identity_sha256(provenance: Mapping[str, Any]) -> str:
    """Independently recompute the COMPLETE assembled native data-branch identity. R2-E.

    This is the mechanism that makes ``full_chain_validated`` non-authoritative: the digest is
    derived from the provenance object's own assembled contents, so a caller cannot assert a
    validated chain by flipping a flag.
    """
    stages = _mapping(provenance.get("stages"), field="release_provenance.stages")
    projection: dict[str, Any] = {
        "schema_version": POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA,
        "fields": {name: provenance.get(name) for name in POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS},
    }
    projection["fields"]["stages"] = {
        stage: {key: (stages.get(stage) or {}).get(key) for key in POST_MERGE_STAGE_FIELDS}
        for stage in sorted(stages)
    }
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def validate_native_provenance_object(provenance: Mapping[str, Any]) -> str:
    """Fully validate a native provenance block and return its data-branch identity.

    ``full_chain_validated`` is checked *last* and only as a consistency field: everything it
    claims has already had to be proved by the checks above it.
    """
    intruders = [name for name in NATIVE_FORBIDDEN_PROVENANCE_FIELDS if name in provenance]
    if intruders:
        raise RuntimeError(
            f"native run plan carries legacy selector-v1 provenance objects: {intruders}"
        )
    if provenance.get("schema_version") != NATIVE_PROVENANCE_SCHEMA:
        raise RuntimeError(
            "native run plan release_provenance.schema_version must be "
            f"{NATIVE_PROVENANCE_SCHEMA!r}, got {provenance.get('schema_version')!r}"
        )
    missing = [name for name in NATIVE_REQUIRED_PROVENANCE_FIELDS if provenance.get(name) is None]
    if missing:
        raise RuntimeError(f"native run plan release_provenance is missing {missing}")

    _sha256(
        provenance.get("candidate_m_plan_sha256"),
        field="release_provenance.candidate_m_plan_sha256",
    )
    _sha256(
        provenance.get("stage_m_implementation_bundle_sha256"),
        field="release_provenance.stage_m_implementation_bundle_sha256",
    )
    _sha256(
        provenance.get("stage_p_native_validator_bundle_sha256"),
        field="release_provenance.stage_p_native_validator_bundle_sha256",
    )
    _sha256(
        provenance.get("accepted_stage_i_identity_sha256"),
        field="release_provenance.accepted_stage_i_identity_sha256",
    )
    accepted = _mapping(
        provenance.get("accepted_stage_i"), field="release_provenance.accepted_stage_i"
    )
    for field in (
        "run_identity",
        "manifest_sha256",
        "completion_object_sha256",
        "layer2_expected_result_sha256",
    ):
        _sha256(accepted.get(field), field=f"release_provenance.accepted_stage_i.{field}")
    if not isinstance(accepted.get("shard_inventory"), list) or not accepted["shard_inventory"]:
        raise RuntimeError("native run plan accepted_stage_i.shard_inventory must be non-empty")

    stages = _mapping(provenance.get("stages"), field="release_provenance.stages")
    if sorted(stages) != list(RUN_PLAN_STAGES):
        raise RuntimeError(
            f"native run plan must bind exactly {list(RUN_PLAN_STAGES)}, got {sorted(stages)}"
        )
    for stage in RUN_PLAN_STAGES:
        entry = _mapping(stages.get(stage), field=f"release_provenance.stages.{stage}")
        absent = [name for name in NATIVE_REQUIRED_STAGE_FIELDS if entry.get(name) is None]
        if absent:
            raise RuntimeError(f"native run plan stages.{stage} is missing {absent}")
        _sha256(entry.get("manifest_sha256"), field=f"stages.{stage}.manifest_sha256")
        _sha256(entry.get("shard_inventory_sha256"), field=f"stages.{stage}.shard_inventory_sha256")
        _sha256(
            entry.get("input_sequence_commitment"),
            field=f"stages.{stage}.input_sequence_commitment",
        )
        _integer(entry.get("shards"), field=f"stages.{stage}.shards", positive=True)
        _integer(
            entry.get("stored_token_ids"), field=f"stages.{stage}.stored_token_ids", positive=True
        )

    if provenance.get("native_shared_authority_validated") is not True:
        raise RuntimeError(
            "native run plan release_provenance.native_shared_authority_validated must be true"
        )
    exclusion = _mapping(
        provenance.get("shared_exclusion_authority"),
        field="release_provenance.shared_exclusion_authority",
    )
    if not exclusion.get("manifest_sha256s") or not isinstance(exclusion.get("hash_count"), int):
        raise RuntimeError("native run plan shared exclusion authority is incomplete")

    premerge = premerge_native_chain_identity_sha256(provenance)
    if premerge != provenance.get("premerge_native_chain_identity_sha256"):
        raise RuntimeError(
            "native run plan pre-merge chain identity does not recompute: "
            f"recomputed={premerge}, "
            f"declared={provenance.get('premerge_native_chain_identity_sha256')}"
        )
    recomputed = post_merge_data_branch_identity_sha256(provenance)
    declared = provenance.get("native_post_merge_data_branch_identity_sha256")
    if recomputed != declared:
        raise RuntimeError(
            "native run plan post-merge data-branch identity does not recompute: "
            f"recomputed={recomputed}, declared={declared}"
        )
    if provenance.get("full_chain_validated") is not True:
        raise RuntimeError("native run plan release_provenance.full_chain_validated must be true")
    return recomputed


def validate_run_plan_args(args: argparse.Namespace) -> None:
    """Require a paired plan/stage in strict mode; allow only explicit legacy escape."""
    plan_path = str(getattr(args, "run_plan_json", "") or "").strip()
    stage = str(getattr(args, "run_plan_stage", "") or "").strip()
    strict = bool(getattr(args, "strict_resume_contract", True))
    if bool(plan_path) != bool(stage):
        raise ValueError("--run_plan_json and --run_plan_stage must be supplied together")
    if stage and stage not in RUN_PLAN_STAGES:
        raise ValueError(f"--run_plan_stage must be one of {RUN_PLAN_STAGES}, got {stage!r}")
    if strict and not plan_path:
        raise ValueError(
            "strict production training requires --run_plan_json and --run_plan_stage; "
            "only --no_strict_resume_contract may run an explicitly unbound legacy/debug job"
        )


def load_run_plan_binding(
    args: argparse.Namespace,
    *,
    train_dir: str | Path,
    tokenizer_sha256: str,
    val_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load and validate the immutable launch fields of a schema-v3 planner artifact."""
    validate_run_plan_args(args)
    plan_arg = str(getattr(args, "run_plan_json", "") or "").strip()
    if not plan_arg:
        return None

    stage_name = str(args.run_plan_stage)
    path, plan, plan_sha256 = _read_plan(plan_arg)
    if plan.get("schema_version") != RUN_PLAN_SCHEMA_VERSION:
        raise RuntimeError(
            f"run plan schema_version must be {RUN_PLAN_SCHEMA_VERSION}, "
            f"got {plan.get('schema_version')!r}"
        )
    if plan.get("plan_type") not in {
        "deterministic_no_replacement_stage_a_b",
        "deterministic_explicit_multi_exposure_stage_a_b",
    }:
        raise RuntimeError(f"unsupported run plan_type: {plan.get('plan_type')!r}")

    invariants = _mapping(plan.get("invariants"), field="invariants")
    if (
        invariants.get("sampling_mode") != "deterministic"
        or invariants.get("replacement") is not False
        or invariants.get("implicit_replay") is not False
        or invariants.get("full_production_provenance_chain") is not True
    ):
        raise RuntimeError("run plan does not declare strict deterministic production invariants")
    provenance = _mapping(plan.get("release_provenance"), field="release_provenance")
    if provenance.get("full_chain_validated") is not True:
        raise RuntimeError("run plan release_provenance.full_chain_validated must be true")
    shared_tokenizer_sha256 = _sha256(
        provenance.get("shared_tokenizer_sha256"),
        field="release_provenance.shared_tokenizer_sha256",
    )
    if shared_tokenizer_sha256 != tokenizer_sha256:
        raise RuntimeError(
            "run plan tokenizer SHA-256 disagrees with --tokenizer_path: "
            f"plan={shared_tokenizer_sha256}, launch={tokenizer_sha256}"
        )

    inputs = _mapping(plan.get("inputs"), field="inputs")
    stage_b_selection_stage = inputs.get("stage_b_selection_stage")
    if stage_b_selection_stage not in STAGE_B_SELECTION_STAGES:
        raise RuntimeError(
            "run plan inputs.stage_b_selection_stage must be one of "
            f"{STAGE_B_SELECTION_STAGES}, got {stage_b_selection_stage!r}"
        )
    if provenance.get("stage_b_selection_stage") != stage_b_selection_stage:
        raise RuntimeError("run plan Stage-B selection cohort disagrees within provenance")
    # DECISIONS D-146: a run plan declares exactly one provenance branch. The legacy
    # selector-v1 chain still requires its selection artifact and source bindings, unchanged.
    # The accepted-Stage-I native chain has neither, and proves the corresponding facts through
    # the accepted publication, the candidate-M plan and the two Stage-M releases instead.
    provenance_chain_kind = provenance.get("provenance_chain_kind", LEGACY_CHAIN_KIND)
    if provenance_chain_kind not in PROVENANCE_CHAIN_KINDS:
        raise RuntimeError(
            f"unsupported release_provenance.provenance_chain_kind: {provenance_chain_kind!r}"
        )
    native_chain = provenance_chain_kind == NATIVE_CHAIN_KIND
    native_data_branch_identity: str | None = None
    if native_chain:
        # R1-E / R1-C. The whole native provenance object is validated -- schema, required
        # fields, identities, both release bindings -- and its self-derived chain identity is
        # recomputed, before full_chain_validated is even looked at.
        native_data_branch_identity = validate_native_provenance_object(provenance)
        native_stages = _mapping(provenance.get("stages"), field="release_provenance.stages")
        selection_manifest_sha256 = _sha256(
            provenance.get("candidate_m_plan_sha256"),
            field="release_provenance.candidate_m_plan_sha256",
        )
        source_bindings = {
            "validated": True,
            "stage_b_selection_stage": stage_b_selection_stage,
            "stage_b": _mapping(
                native_stages.get("stage_b"), field="release_provenance.stages.stage_b"
            ).get("input_sequence_commitment"),
        }
    else:
        selection_provenance = _mapping(
            provenance.get("selection"), field="release_provenance.selection"
        )
        selection_manifest_sha256 = _sha256(
            selection_provenance.get("manifest_sha256"),
            field="release_provenance.selection.manifest_sha256",
        )
        if selection_provenance.get("stage_b_selection_stage") != stage_b_selection_stage:
            raise RuntimeError("run plan selection artifact uses a different Stage-B cohort")
        source_bindings = _mapping(
            provenance.get("source_bindings"), field="release_provenance.source_bindings"
        )
        if (
            source_bindings.get("validated") is not True
            or source_bindings.get("stage_b_selection_stage") != stage_b_selection_stage
        ):
            raise RuntimeError(
                "run plan source bindings do not validate the selected Stage-B cohort"
            )
    if bool(getattr(args, "allow_data_branch", False)) and (
        stage_name != "stage_b" or stage_b_selection_stage != "control"
    ):
        raise RuntimeError(
            "--allow_data_branch is only valid for a control-cohort Stage-B invocation"
        )
    _same_int(inputs.get("seq_len"), int(args.seq_len), field="inputs.seq_len")
    _same_int(inputs.get("micro_bsz"), int(args.micro_bsz), field="inputs.micro_bsz")
    _same_int(inputs.get("grad_accum"), int(args.grad_accum), field="inputs.grad_accum")
    _same_int(inputs.get("warmup_steps"), int(args.warmup_steps), field="inputs.warmup_steps")
    reference_val_dir = inputs.get("reference_val_dir")
    launch_val_dir = val_dir if val_dir is not None else getattr(args, "val_dir", None)
    if launch_val_dir is None:
        raise RuntimeError("launch has no --val_dir to bind to the run plan")
    if not isinstance(reference_val_dir, str) or Path(reference_val_dir).expanduser().resolve() != (
        Path(launch_val_dir).expanduser().resolve()
    ):
        raise RuntimeError(
            "run plan inputs.reference_val_dir disagrees with --val_dir: "
            f"plan={reference_val_dir!r}, launch={str(Path(launch_val_dir).resolve())!r}"
        )
    reference_provenance = _mapping(
        provenance.get("reference_validation"),
        field="release_provenance.reference_validation",
    )
    reference_manifest_sha256 = _sha256(
        reference_provenance.get("manifest_sha256"),
        field="release_provenance.reference_validation.manifest_sha256",
    )

    sequences_per_step = int(args.micro_bsz) * int(args.grad_accum)
    batch = _mapping(plan.get("batch"), field="batch")
    _same_int(
        batch.get("sequences_per_optimizer_step"),
        sequences_per_step,
        field="batch.sequences_per_optimizer_step",
    )
    _same_int(
        batch.get("serialized_target_positions_per_optimizer_step"),
        sequences_per_step * int(args.seq_len),
        field="batch.serialized_target_positions_per_optimizer_step",
    )

    boundaries = _mapping(plan.get("boundaries"), field="boundaries")
    stage_a_start = _integer(
        boundaries.get("stage_a_start_step"), field="boundaries.stage_a_start_step"
    )
    stage_a_stop = _integer(
        boundaries.get("stage_a_stop_step"),
        field="boundaries.stage_a_stop_step",
        positive=True,
    )
    stage_b_start = _integer(
        boundaries.get("stage_b_start_step"),
        field="boundaries.stage_b_start_step",
        positive=True,
    )
    stage_b_stop = _integer(
        boundaries.get("stage_b_global_stop_step"),
        field="boundaries.stage_b_global_stop_step",
        positive=True,
    )
    schedule_total = _integer(
        boundaries.get("schedule_total_steps"),
        field="boundaries.schedule_total_steps",
        positive=True,
    )
    if stage_a_start != 0 or stage_a_stop != stage_b_start or stage_b_stop != schedule_total:
        raise RuntimeError("run plan Stage A/B boundaries are not one contiguous global timeline")
    if stage_name == "stage_a":
        expected_start, expected_stop = stage_a_start, stage_a_stop
    else:
        expected_start, expected_stop = stage_b_start, stage_b_stop
    _same_int(
        expected_start,
        int(args.data_stage_start_step),
        field=f"boundaries.{stage_name}_start",
    )
    _same_int(expected_stop, int(args.max_steps), field=f"boundaries.{stage_name}_stop")
    _same_int(
        schedule_total,
        int(args.schedule_total_steps),
        field="boundaries.schedule_total_steps",
    )

    milestones = _mapping(plan.get("checkpoint_milestones"), field="checkpoint_milestones")
    if milestones.get("schema_version") != 1:
        raise RuntimeError("run plan checkpoint_milestones.schema_version must be 1")
    raw_save_steps = milestones.get("absolute_steps")
    if not isinstance(raw_save_steps, list) or not raw_save_steps:
        raise RuntimeError("run plan checkpoint_milestones.absolute_steps must be non-empty")
    milestone_steps = [
        _integer(value, field=f"checkpoint_milestones.absolute_steps[{index}]", positive=True)
        for index, value in enumerate(raw_save_steps)
    ]
    if milestone_steps != sorted(set(milestone_steps)):
        raise RuntimeError(
            "run plan checkpoint_milestones.absolute_steps must be strictly increasing and unique"
        )
    if milestone_steps[-1] > schedule_total:
        raise RuntimeError("run plan checkpoint milestone exceeds schedule_total_steps")
    if milestones.get("cli_save_steps") != ",".join(str(step) for step in milestone_steps):
        raise RuntimeError("run plan checkpoint milestone CLI serialization is inconsistent")
    launch_save_steps = [int(step) for step in getattr(args, "save_steps", [])]
    if launch_save_steps != milestone_steps:
        raise RuntimeError(
            "--save_steps must exactly match the complete run-plan checkpoint milestones: "
            f"plan={milestone_steps}, launch={launch_save_steps}"
        )
    milestone_sha256 = hashlib.sha256(
        json.dumps(milestones, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    stages = _mapping(plan.get("stages"), field="stages")
    stage_a_plan = _mapping(stages.get("stage_a"), field="stages.stage_a")
    stage_b_plan = _mapping(stages.get("stage_b"), field="stages.stage_b")
    stage_a_samples = _integer(
        stage_a_plan.get("consumed_blocks"),
        field="stages.stage_a.consumed_blocks",
        positive=True,
    )
    stage_b_samples = _integer(
        stage_b_plan.get("consumed_blocks"),
        field="stages.stage_b.consumed_blocks",
        positive=True,
    )
    if (
        stage_a_samples != (stage_a_stop - stage_a_start) * sequences_per_step
        or stage_b_samples != (stage_b_stop - stage_b_start) * sequences_per_step
    ):
        raise RuntimeError(
            "run plan both-stage consumed blocks disagree with boundary/batch arithmetic"
        )
    stage = _mapping(stages.get(stage_name), field=f"stages.{stage_name}")
    planned_steps = _integer(
        stage.get("planned_optimizer_steps"),
        field=f"stages.{stage_name}.planned_optimizer_steps",
        positive=True,
    )
    if planned_steps != expected_stop - expected_start:
        raise RuntimeError(
            f"run plan {stage_name} planned steps disagree with its absolute boundaries"
        )
    source_dir = stage.get("source_dir")
    if (
        not isinstance(source_dir, str)
        or Path(source_dir).expanduser().resolve() != Path(train_dir).expanduser().resolve()
    ):
        raise RuntimeError(
            f"run plan {stage_name} source_dir disagrees with --train_dir: "
            f"plan={source_dir!r}, launch={str(Path(train_dir).resolve())!r}"
        )
    expected_samples = _integer(
        stage.get("consumed_blocks"),
        field=f"stages.{stage_name}.consumed_blocks",
        positive=True,
    )
    arithmetic_samples = planned_steps * sequences_per_step
    if expected_samples != arithmetic_samples:
        raise RuntimeError(
            f"run plan {stage_name} consumed_blocks disagree with step/batch arithmetic: "
            f"{expected_samples} != {arithmetic_samples}"
        )
    candidate_samples = _integer(
        stage.get("candidate_exposure_blocks"),
        field=f"stages.{stage_name}.candidate_exposure_blocks",
        positive=True,
    )
    if expected_samples > candidate_samples:
        raise RuntimeError(f"run plan {stage_name} consumes beyond its explicit exposure budget")
    requested_exposures = _integer(
        stage.get("requested_exposures"),
        field=f"stages.{stage_name}.requested_exposures",
        positive=True,
    )
    unique_blocks = _integer(
        stage.get("unique_blocks"),
        field=f"stages.{stage_name}.unique_blocks",
        positive=True,
    )
    if candidate_samples != requested_exposures * unique_blocks:
        raise RuntimeError(f"run plan {stage_name} explicit exposure arithmetic is inconsistent")
    stage_a_exposures = _integer(
        inputs.get("stage_a_exposures"), field="inputs.stage_a_exposures", positive=True
    )
    stage_b_exposures = _integer(
        inputs.get("stage_b_exposures"), field="inputs.stage_b_exposures", positive=True
    )
    input_stage_exposures = stage_a_exposures if stage_name == "stage_a" else stage_b_exposures
    if requested_exposures != input_stage_exposures:
        raise RuntimeError(f"run plan {stage_name} requested exposures disagree with inputs")
    explicit_replay = stage_a_exposures > 1 or stage_b_exposures > 1
    if invariants.get("explicit_replay") is not explicit_replay:
        raise RuntimeError("run plan explicit-replay invariant disagrees with its inputs")
    expected_plan_type = (
        "deterministic_explicit_multi_exposure_stage_a_b"
        if explicit_replay
        else "deterministic_no_replacement_stage_a_b"
    )
    if plan.get("plan_type") != expected_plan_type:
        raise RuntimeError("run plan explicit-replay inputs disagree with plan_type")

    release = _mapping(stage.get("dataset"), field=f"stages.{stage_name}.dataset")
    release_validation = _mapping(
        release.get("release_validation"),
        field=f"stages.{stage_name}.dataset.release_validation",
    )
    expected_manifest_sha256 = _sha256(
        release_validation.get("manifest_sha256"),
        field=f"stages.{stage_name}.dataset.release_validation.manifest_sha256",
    )
    release_stage = _mapping(provenance.get(stage_name), field=f"release_provenance.{stage_name}")
    if (
        _sha256(
            release_stage.get("manifest_sha256"),
            field=f"release_provenance.{stage_name}.manifest_sha256",
        )
        != expected_manifest_sha256
    ):
        raise RuntimeError(f"run plan {stage_name} release manifest hashes disagree internally")

    other_name = "stage_b" if stage_name == "stage_a" else "stage_a"
    other_plan = stage_b_plan if other_name == "stage_b" else stage_a_plan
    other_dataset = _mapping(other_plan.get("dataset"), field=f"stages.{other_name}.dataset")
    other_validation = _mapping(
        other_dataset.get("release_validation"),
        field=f"stages.{other_name}.dataset.release_validation",
    )
    other_manifest_sha256 = _sha256(
        other_validation.get("manifest_sha256"),
        field=f"stages.{other_name}.dataset.release_validation.manifest_sha256",
    )
    other_release = _mapping(provenance.get(other_name), field=f"release_provenance.{other_name}")
    other_provenance_sha256 = _sha256(
        other_release.get("manifest_sha256"),
        field=f"release_provenance.{other_name}.manifest_sha256",
    )
    if other_manifest_sha256 != other_provenance_sha256:
        raise RuntimeError(f"run plan {other_name} release manifest hashes disagree internally")
    stage_release_manifest_sha256 = {
        stage_name: expected_manifest_sha256,
        other_name: other_manifest_sha256,
    }

    schedule_name = str(args.lr_schedule)
    if schedule_name not in {"wsd", "cosine"}:
        raise RuntimeError(
            "a strict run plan supports the canonical WSD schedule or an explicit cosine control"
        )
    wsd = _mapping(plan.get("wsd_candidate"), field="wsd_candidate")
    _same_int(wsd.get("warmup_steps"), int(args.warmup_steps), field="wsd_candidate.warmup_steps")
    if schedule_name == "wsd" and not bool(args.allow_schedule_branch):
        _same_int(
            wsd.get("decay_start_step"),
            int(args.decay_start_step),
            field="wsd_candidate.decay_start_step",
        )
        _same_int(
            wsd.get("decay_end_step"),
            int(args.decay_end_step),
            field="wsd_candidate.decay_end_step",
        )

    return {
        "schema_version": 1,
        "status": "validated",
        "plan_path": str(path),
        "plan_sha256": plan_sha256,
        "plan_schema_version": RUN_PLAN_SCHEMA_VERSION,
        "plan_type": str(plan["plan_type"]),
        "stage": stage_name,
        "stage_start_step": expected_start,
        "stage_stop_step": expected_stop,
        "schedule_total_steps": schedule_total,
        "expected_stage_samples": expected_samples,
        "data_branch_immutable_sha256": _data_branch_immutable_sha256(plan),
        "provenance_chain_kind": provenance_chain_kind,
        "native_post_merge_data_branch_identity_sha256": native_data_branch_identity,
        "data_branch_validation": None,
        "selection_manifest_sha256": selection_manifest_sha256,
        "sequences_per_optimizer_step": sequences_per_step,
        "stage_a_stop_step": stage_a_stop,
        "stage_b_stop_step": stage_b_stop,
        "stage_a_expected_samples": stage_a_samples,
        "stage_b_expected_samples": stage_b_samples,
        "stage_a_release_manifest_sha256": stage_release_manifest_sha256["stage_a"],
        "stage_b_release_manifest_sha256": stage_release_manifest_sha256["stage_b"],
        "stage_b_source_bindings": source_bindings["stage_b"],
        "requested_exposures": requested_exposures,
        "unique_blocks": unique_blocks,
        "candidate_exposure_blocks": candidate_samples,
        "stage_b_selection_stage": stage_b_selection_stage,
        "stage_release_manifest_sha256": expected_manifest_sha256,
        "reference_release_manifest_sha256": reference_manifest_sha256,
        "tokenizer_sha256": shared_tokenizer_sha256,
        "checkpoint_milestones_schema_version": 1,
        "checkpoint_milestone_steps": milestone_steps,
        "checkpoint_milestones_sha256": milestone_sha256,
    }


def resolve_run_plan_sample_budget(
    binding: Mapping[str, Any] | None,
    *,
    stage_sample_position: int,
    step_derived_stage_samples: int,
) -> tuple[int, int]:
    """Return the frozen absolute sampler end and remaining suffix."""
    position = int(stage_sample_position)
    step_budget = int(step_derived_stage_samples)
    if position < 0 or step_budget <= 0:
        raise RuntimeError("stage sampler position/budget must be non-negative/positive")
    planned = step_budget
    if binding is not None:
        planned = _integer(
            binding.get("expected_stage_samples"),
            field="binding.expected_stage_samples",
            positive=True,
        )
    if planned != step_budget:
        raise RuntimeError(
            "run-plan sample budget disagrees with stage step/batch arithmetic: "
            f"plan={planned}, steps={step_budget}"
        )
    if position > planned:
        raise RuntimeError(
            "checkpoint data position exceeds the frozen stage sample budget: "
            f"position={position}, budget={planned}"
        )
    return planned, planned - position


def synchronize_validated_run_plan_binding(
    run_contract: Mapping[str, Any],
    launch_binding: dict[str, Any] | None,
) -> None:
    """Propagate validated branch lineage into the binding used by run metadata."""
    if launch_binding is None:
        return
    effective = run_contract.get("run_plan")
    if not isinstance(effective, Mapping):
        raise RuntimeError("run contract has no effective run-plan binding")
    launch_binding.clear()
    launch_binding.update(effective)


def validate_run_plan_dataset(binding: Mapping[str, Any], dataset: Any) -> None:
    """Bind the frozen plan to the release bytes actually opened by the trainer."""
    expected_unique = _integer(
        binding.get("unique_blocks"), field="binding.unique_blocks", positive=True
    )
    if int(len(dataset)) != expected_unique:
        raise RuntimeError(
            "training dataset block count disagrees with run plan: "
            f"dataset={len(dataset)}, plan={expected_unique}"
        )
    stats = dataset.stats()
    release = _mapping(stats.get("release_validation"), field="dataset.release_validation")
    current_sha256 = _sha256(
        release.get("manifest_sha256"), field="dataset.release_validation.manifest_sha256"
    )
    if current_sha256 != binding.get("stage_release_manifest_sha256"):
        raise RuntimeError(
            "training shard release manifest disagrees with run plan: "
            f"dataset={current_sha256}, plan={binding.get('stage_release_manifest_sha256')}"
        )
    if release.get("tokenizer_sha256") != binding.get("tokenizer_sha256"):
        raise RuntimeError("training shard tokenizer SHA-256 disagrees with run plan")


def validate_run_plan_validation_dataset(binding: Mapping[str, Any], dataset: Any) -> None:
    """Bind the combined frozen reference-validation release opened by the trainer."""
    stats = dataset.stats()
    release = _mapping(stats.get("release_validation"), field="validation.release_validation")
    if release.get("release_kind") != "reference" or release.get("split") != "val":
        raise RuntimeError("--val_dir must be the combined frozen reference-validation split")
    current_sha256 = _sha256(
        release.get("manifest_sha256"), field="validation.release_validation.manifest_sha256"
    )
    if current_sha256 != binding.get("reference_release_manifest_sha256"):
        raise RuntimeError(
            "validation release manifest disagrees with run plan: "
            f"dataset={current_sha256}, "
            f"plan={binding.get('reference_release_manifest_sha256')}"
        )
    if release.get("tokenizer_sha256") != binding.get("tokenizer_sha256"):
        raise RuntimeError("validation shard tokenizer SHA-256 disagrees with run plan")


def validate_run_plan_resume_transition(
    saved: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    checkpoint_step: int,
    allow_data_branch: bool = False,
) -> None:
    """Accept exact resume, same-plan handoff, or an explicitly scoped control branch."""
    if not isinstance(saved, Mapping) or not isinstance(current, Mapping):
        raise RuntimeError("[resume] run-plan binding is missing")
    saved_core = dict(saved)
    current_core = dict(current)
    saved_branch = saved_core.pop("data_branch_validation", None)
    current_core.pop("data_branch_validation", None)
    if saved_core == current_core:
        if saved_branch is not None:
            if not isinstance(current, dict):
                raise RuntimeError("[resume] current run-plan binding is not mutable")
            current["data_branch_validation"] = saved_branch
        return
    invariant_fields = (
        "schema_version",
        "status",
        "plan_path",
        "plan_sha256",
        "plan_schema_version",
        "plan_type",
        "schedule_total_steps",
        "tokenizer_sha256",
        "reference_release_manifest_sha256",
        "checkpoint_milestones_schema_version",
        "checkpoint_milestone_steps",
        "checkpoint_milestones_sha256",
    )
    mismatches = [field for field in invariant_fields if saved.get(field) != current.get(field)]
    if mismatches and not allow_data_branch:
        raise RuntimeError(
            "[resume] run-plan identity changed across stage boundary: " + ", ".join(mismatches)
        )
    if saved.get("stage") != "stage_a" or current.get("stage") != "stage_b":
        raise RuntimeError("[resume] only an exact same stage or Stage A -> Stage B is allowed")
    boundary = int(checkpoint_step)
    if (
        int(saved.get("stage_stop_step", -1)) != boundary
        or int(current.get("stage_start_step", -1)) != boundary
    ):
        raise RuntimeError(
            "[resume] Stage A -> Stage B run-plan handoff does not match checkpoint step"
        )
    if mismatches:
        _record_validated_data_branch(
            saved,
            current,
            checkpoint_step=boundary,
        )


def _record_validated_data_branch(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    checkpoint_step: int,
) -> None:
    """Attach audited lineage for the sole legal premium-to-control data branch."""
    if not isinstance(current, dict):
        raise RuntimeError("[resume] current run-plan binding is not mutable")
    branch_fields = (
        "schema_version",
        "status",
        "plan_schema_version",
        "plan_type",
        "data_branch_immutable_sha256",
        # R1-C: for a native plan the immutable data authority is this identity; for a legacy
        # plan it is None on both sides, so the comparison is a no-op there.
        "provenance_chain_kind",
        "native_post_merge_data_branch_identity_sha256",
        "schedule_total_steps",
        "tokenizer_sha256",
        "reference_release_manifest_sha256",
        "selection_manifest_sha256",
        "checkpoint_milestones_schema_version",
        "checkpoint_milestone_steps",
        "checkpoint_milestones_sha256",
        "sequences_per_optimizer_step",
        "stage_a_stop_step",
        "stage_a_expected_samples",
        "stage_a_release_manifest_sha256",
        "stage_b_stop_step",
        "stage_b_expected_samples",
    )
    branch_mismatches = [field for field in branch_fields if saved.get(field) != current.get(field)]
    if branch_mismatches:
        raise RuntimeError(
            "[resume] data branch changed immutable training fields: "
            + ", ".join(branch_mismatches)
        )
    if saved.get("stage_b_selection_stage") != "stage_b":
        raise RuntimeError("[resume] data branch parent must use the premium stage_b cohort")
    if current.get("stage_b_selection_stage") != "control":
        raise RuntimeError("[resume] data branch target must use the control cohort")
    if saved.get("plan_sha256") == current.get("plan_sha256"):
        raise RuntimeError("[resume] data branch requires a distinct frozen control plan")
    parent_stage_b_release = _sha256(
        saved.get("stage_b_release_manifest_sha256"),
        field="saved.stage_b_release_manifest_sha256",
    )
    current_stage_b_release = _sha256(
        current.get("stage_b_release_manifest_sha256"),
        field="current.stage_b_release_manifest_sha256",
    )
    if parent_stage_b_release == current_stage_b_release:
        raise RuntimeError("[resume] data branch must bind a distinct Stage-B shard release")
    current["data_branch_validation"] = {
        "schema_version": 1,
        "status": "validated",
        "kind": "stage_b_data_control",
        "checkpoint_step": int(checkpoint_step),
        "parent_plan_path": saved.get("plan_path"),
        "parent_plan_sha256": saved.get("plan_sha256"),
        "current_plan_path": current.get("plan_path"),
        "current_plan_sha256": current.get("plan_sha256"),
        "parent_stage_b_selection_stage": saved.get("stage_b_selection_stage"),
        "current_stage_b_selection_stage": current.get("stage_b_selection_stage"),
        "parent_stage_b_release_manifest_sha256": parent_stage_b_release,
        "current_stage_b_release_manifest_sha256": current_stage_b_release,
        "parent_stage_b_source_bindings": saved.get("stage_b_source_bindings"),
        "current_stage_b_source_bindings": current.get("stage_b_source_bindings"),
        "validated_immutable_sha256": current.get("data_branch_immutable_sha256"),
        "changed_fields": [
            "inputs.stage_b_dir",
            "inputs.stage_b_selection_stage",
            "release_provenance.stage_b",
            "release_provenance.source_bindings.stage_b",
            "stages.stage_b.dataset_and_source",
            "totals.stage_b_derived_accounting",
        ],
    }
