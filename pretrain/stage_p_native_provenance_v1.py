#!/usr/bin/env python3
"""Stage-P native provenance v1 for accepted H/I-derived releases.

D-026 remains authoritative for legacy selector-v1 releases and this module never touches that
path. What it adds is a second, explicitly versioned provenance branch for releases produced by
the accepted H/I pipeline, which has no selection manifest, no SQLite selection registry and no
schema-2 exact-intersection audit -- and for which fabricating those artefacts would make a
manifest assert a process that did not run (DECISIONS D-146).

The native branch proves the chain directly instead:

    accepted Stage-I publication  ->  candidate-M plan  ->  two Stage-M releases

Every link is verified from bytes on disk. A serialized boolean is never evidence:
``full_chain_validated`` is *derived* here, set true only after every required check has
passed, and a caller-supplied value is rejected outright.

The two branches are disjoint by construction. A run plan declares exactly one
``provenance_chain_kind``; a payload that carries native identities alongside legacy selection
fields, or a legacy kind alongside native Stage-I authority, is refused rather than silently
resolved to whichever branch happens to match first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.stage_m_contract_v1 import (  # noqa: E402
    CANDIDATE_PLAN_SCHEMA,
    MODEL_CONTRACT,
    ORDERING_CONTRACT_ID,
    SEQ_LEN,
    STAGE_STREAMS,
    canonical_json_bytes,
    file_sha256,
    p_native_implementation_bundle,
    sha256_hex,
    stream_accounting,
    validated_sha256,
)
from pretrain.stage_m_output_v1 import MANIFEST_FILENAME, validate_published_release  # noqa: E402

NATIVE_CHAIN_KIND = "accepted_stage_i_native_v1"
LEGACY_CHAIN_KIND = "legacy_selector_v1"
NATIVE_PROVENANCE_SCHEMA = "petitgpt-accepted-stage-i-native-release-provenance-v1"
PROVENANCE_CHAIN_KINDS = (NATIVE_CHAIN_KIND, LEGACY_CHAIN_KIND)

# Fields that only ever belong to the legacy selector-v1 chain. Their presence in a native
# payload is branch mixing, not an extra hint.
LEGACY_ONLY_FIELDS = (
    "selection_manifest",
    "selection_manifest_sha256",
    "selection_registry",
    "selection_registry_sha256",
    "selection_audit",
    "selection_audit_sha256",
    "selection_database",
    "sqlite_registry",
    "selected_jsonl",
    "selection_spec",
)

# Fields that only ever belong to the native chain.
NATIVE_ONLY_FIELDS = (
    "accepted_stage_i",
    "candidate_m_plan_sha256",
    "stage_m_implementation_bundle_sha256",
    "stage_m_ordering_policy",
)


class NativeProvenanceError(RuntimeError):
    """Controlled failure while validating the accepted-Stage-I-native provenance chain."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise NativeProvenanceError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    assert isinstance(value, Mapping)
    return value


def _int(value: object, *, field: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{field} must be an integer, got {value!r}",
    )
    assert isinstance(value, int)
    return value


def assert_single_branch(payload: Mapping[str, Any], *, chain_kind: str) -> None:
    """Refuse any payload that mixes the two provenance branches."""
    _require(
        chain_kind in PROVENANCE_CHAIN_KINDS,
        f"unknown provenance_chain_kind: {chain_kind!r}",
    )
    if chain_kind == NATIVE_CHAIN_KIND:
        intruders = [name for name in LEGACY_ONLY_FIELDS if name in payload]
        _require(
            not intruders,
            f"native provenance payload carries legacy selector-v1 fields: {intruders}",
        )
    else:
        intruders = [name for name in NATIVE_ONLY_FIELDS if name in payload]
        _require(
            not intruders,
            f"legacy provenance payload carries native Stage-I authority fields: {intruders}",
        )
    _require(
        "full_chain_validated" not in payload,
        "full_chain_validated is derived by the validator and must not be caller-supplied",
    )


def _load_release(release_root: Path, *, stage: str) -> tuple[Mapping[str, Any], str, dict]:
    """Strictly validate one Stage-M release and return its manifest, digest and read-back."""
    root = Path(release_root).expanduser().resolve()
    _require(root.is_dir(), f"{stage}: Stage-M release directory is missing: {root}")
    manifest_path = root / MANIFEST_FILENAME
    _require(manifest_path.is_file(), f"{stage}: release has no {MANIFEST_FILENAME}: {root}")

    import json

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(isinstance(manifest, dict), f"{stage}: release manifest must be an object")
    manifest_sha256 = file_sha256(manifest_path)

    # The strict physical consumer check. Passing this means a real PackedBinDataset opens it.
    validation = validate_published_release(root)
    _require(
        validation["manifest_sha256"] == manifest_sha256,
        f"{stage}: release manifest changed during validation",
    )
    _require(
        manifest.get("status") == "complete",
        f"{stage}: release is not in the canonical schema-3 complete state",
    )
    _require(
        manifest.get("schema_version") == 3,
        f"{stage}: release is not canonical schema 3",
    )
    return manifest, manifest_sha256, validation


def _bind_reference_validation(
    reference_val_dir: Path, *, expected_tokenizer_sha256: str
) -> dict[str, Any]:
    """Bind the frozen reference-validation release through the existing frozen validator.

    The native chain does not re-implement reference validation: the G2 release is already a
    canonical schema-2 reference release, and ``validate_shard_release`` is the same check the
    legacy branch and the trainer perform. Only its identity is recorded here.
    """
    from pretrain.dataset_pretrain import validate_shard_release as _validate

    result = _validate(Path(reference_val_dir))
    _require(
        result["release_kind"] == "reference" and result["split"] == "val",
        "reference_val_dir must identify the combined reference val split",
    )
    _require(
        result["tokenizer_sha256"] == expected_tokenizer_sha256,
        "reference validation release uses a different tokenizer than the Stage-M plan",
    )
    return {
        "reference_val_dir": str(Path(reference_val_dir).expanduser().resolve()),
        "manifest_sha256": str(result["manifest_sha256"]),
        "manifest_schema_version": result["manifest_schema_version"],
        "shards": int(result["expected_shards"]),
        "tokens": int(result["expected_tokens"]),
        "tokenizer_sha256": str(result["tokenizer_sha256"]),
    }


def _bind_tokenizer_release(
    tokenizer_release_manifest: Path, *, expected_tokenizer_sha256: str
) -> dict[str, Any]:
    """Bind the canonical tokenizer release manifest by bytes and prove it names our tokenizer."""
    import json

    path = Path(tokenizer_release_manifest).expanduser().resolve()
    _require(path.is_file(), f"tokenizer release manifest is missing: {path}")
    manifest_sha256 = file_sha256(path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(isinstance(manifest, Mapping), "tokenizer release manifest must be an object")
    declared = manifest.get("tokenizer_sha256")
    if declared is None:
        declared = (manifest.get("tokenizer") or {}).get("sha256")
    _require(
        declared == expected_tokenizer_sha256,
        "tokenizer release manifest does not name the tokenizer the Stage-M plan bound: "
        f"release={declared!r}, plan={expected_tokenizer_sha256}",
    )
    return {
        "path": str(path),
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": str(declared),
    }


def validate_native_chain(
    *,
    repo_root: Path,
    accepted_stage_i_dir: Path,
    candidate_m_plan: Path,
    expected_candidate_m_plan_sha256: str,
    stage_releases: Mapping[str, Path],
    reference_val_dir: Path | None = None,
    tokenizer_release_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate the whole accepted-Stage-I-native chain and derive its provenance block.

    Returns a provenance object whose ``full_chain_validated`` is ``True`` only because every
    check below passed. Any failure raises instead of degrading the flag.
    """
    import json

    from pretrain.stage_m_input_v1 import load_accepted_stage_i

    repo_root = Path(repo_root).expanduser().resolve()
    _require(
        sorted(stage_releases) == sorted(STAGE_STREAMS),
        f"native chain requires exactly {list(STAGE_STREAMS)}, got {sorted(stage_releases)}",
    )

    # --- link 1: the candidate-M plan, by owner-held digest -------------------------------
    plan_path = Path(candidate_m_plan).expanduser().resolve()
    _require(plan_path.is_file(), f"candidate Stage-M plan is missing: {plan_path}")
    plan_bytes = plan_path.read_bytes()
    plan_sha256 = sha256_hex(plan_bytes)
    expected_plan = validated_sha256(
        expected_candidate_m_plan_sha256, field="expected_candidate_m_plan_sha256"
    )
    _require(
        plan_sha256 == expected_plan,
        f"candidate Stage-M plan digest mismatch: actual={plan_sha256}, expected={expected_plan}",
    )
    plan = json.loads(plan_bytes.decode("utf-8"))
    _require(
        plan.get("schema_version") == CANDIDATE_PLAN_SCHEMA,
        f"unsupported candidate Stage-M plan schema: {plan.get('schema_version')!r}",
    )
    _require(
        (plan.get("ordering_contract") or {}).get("policy") == ORDERING_CONTRACT_ID,
        "candidate Stage-M plan does not carry the frozen ordering policy",
    )
    _require(
        dict(plan.get("model_contract") or {}) == dict(MODEL_CONTRACT),
        "candidate Stage-M plan model contract differs from the frozen contract",
    )
    _require(
        int(plan["model_contract"]["seq_len"]) == SEQ_LEN,
        "candidate Stage-M plan seq_len is not the frozen context length",
    )

    # --- link 2: the accepted Stage-I publication ----------------------------------------
    bound = _mapping(plan.get("accepted_stage_i"), field="plan.accepted_stage_i")
    accepted = load_accepted_stage_i(
        Path(accepted_stage_i_dir),
        expected_run_identity=str(bound["run_identity"]),
        expected_manifest_sha256=str(bound["manifest_sha256"]),
        expected_completion_sha256=str(bound["completion_object_sha256"]),
        expected_layer2_sha256=str(bound["layer2_expected_result_sha256"]),
        expected_records=int(bound["total_records"]),
        expected_serialized_tokens=int(bound["total_serialized_tokens"]),
        expected_shard_count=int(bound["shard_count"]),
    )
    _require(
        accepted.as_canonical() == dict(bound),
        "the accepted Stage-I publication on disk differs from the one the M plan bound",
    )
    _require(
        accepted.identity_sha256() == plan.get("accepted_stage_i_identity_sha256"),
        "accepted Stage-I identity digest differs from the candidate Stage-M plan",
    )
    _require(
        sorted(accepted.stage_membership) == sorted(STAGE_STREAMS),
        "accepted Stage-I does not declare exactly the two frozen stage streams",
    )

    tokenizer_sha256 = validated_sha256(
        (plan.get("resources") or {}).get("tokenizer", {}).get("sha256"),
        field="plan.resources.tokenizer.sha256",
    )
    bundle_sha256 = validated_sha256(
        plan.get("implementation_bundle_sha256"), field="plan.implementation_bundle_sha256"
    )

    # --- link 3: the two derived Stage-M releases ----------------------------------------
    stages: dict[str, Any] = {}
    for stage in STAGE_STREAMS:
        manifest, manifest_sha256, validation = _load_release(stage_releases[stage], stage=stage)
        binding = _mapping(manifest.get("stage_m"), field=f"{stage}.meta.stage_m")
        _require(
            binding.get("candidate_plan_sha256") == plan_sha256,
            f"{stage}: release was not produced by this candidate Stage-M plan",
        )
        _require(
            binding.get("implementation_bundle_sha256") == bundle_sha256,
            f"{stage}: release implementation bundle differs from the plan",
        )
        _require(
            binding.get("ordering_policy") == ORDERING_CONTRACT_ID,
            f"{stage}: release does not declare the frozen ordering policy",
        )
        _require(
            binding.get("stage") == stage,
            f"{stage}: release stage_m block names a different stage",
        )
        _require(
            dict(binding.get("environment") or {}) == dict(plan.get("environment_contract") or {}),
            f"{stage}: release environment differs from the plan",
        )
        _require(
            manifest.get("tokenizer_sha256") == tokenizer_sha256,
            f"{stage}: release tokenizer differs from the plan",
        )
        _require(
            validation["tokenizer_sha256"] == tokenizer_sha256,
            f"{stage}: validated release tokenizer differs from the plan",
        )
        _require(
            dict(binding.get("model_contract") or {}) == dict(MODEL_CONTRACT),
            f"{stage}: release model contract differs from the frozen contract",
        )

        accepted_block = _mapping(
            binding.get("accepted_stage_i"), field=f"{stage}.stage_m.accepted_stage_i"
        )
        for key, value in (
            ("run_identity", accepted.run_identity),
            ("layer2_expected_result_sha256", accepted.layer2_sha256),
            ("manifest_sha256", accepted.manifest_sha256),
            ("completion_object_sha256", accepted.completion_sha256),
            ("identity_sha256", accepted.identity_sha256()),
        ):
            _require(
                accepted_block.get(key) == value,
                f"{stage}: release accepted Stage-I {key} differs from the verified publication",
            )

        planned = _mapping(
            (plan.get("stage_streams") or {}).get(stage), field=f"plan.stage_streams.{stage}"
        )
        _require(
            binding.get("input_sequence_commitment") == planned.get("input_sequence_commitment"),
            f"{stage}: release input sequence commitment differs from the plan",
        )
        _require(
            _int(binding.get("input_record_count"), field=f"{stage}.input_record_count")
            == int(planned["input_record_count"]),
            f"{stage}: release input record count differs from the plan",
        )
        _require(
            _int(binding.get("input_serialized_tokens"), field=f"{stage}.input_serialized_tokens")
            == int(planned["input_serialized_tokens"]),
            f"{stage}: release input serialized tokens differ from the plan",
        )
        membership = accepted.stage_membership[stage]
        _require(
            int(planned["input_record_count"]) == int(membership["records"])
            and int(planned["input_serialized_tokens"]) == int(membership["serialized_tokens"]),
            f"{stage}: plan input accounting differs from the accepted publication",
        )

        expected = stream_accounting(
            stage, int(membership["serialized_tokens"]), SEQ_LEN
        ).as_canonical()
        _require(
            dict(planned.get("expected_accounting") or {}) == expected,
            f"{stage}: plan expected accounting is not the frozen derivation",
        )
        _require(
            dict(manifest.get("stage_m_accounting") or {}) == expected,
            f"{stage}: release accounting differs from the plan's expected accounting",
        )
        stored = _int(validation["expected_tokens"], field=f"{stage}.expected_tokens")
        _require(
            stored == int(expected["retained_stored_token_ids"]),
            f"{stage}: published stored token IDs differ from the expected accounting",
        )
        _require(
            (stored - 1) // SEQ_LEN == int(expected["training_sequences"])
            and (stored - 1) % SEQ_LEN == 0,
            f"{stage}: published block geometry differs from the expected accounting",
        )

        records = [dict(r) for r in validation.get("shard_file_records") or ()]
        _require(
            len(records) == int(validation["expected_shards"]),
            f"{stage}: validated shard inventory is incomplete",
        )
        stages[stage] = {
            "release_dir": str(Path(stage_releases[stage]).expanduser().resolve()),
            "manifest_sha256": manifest_sha256,
            "shards": int(validation["expected_shards"]),
            "stored_token_ids": stored,
            "shard_inventory_sha256": sha256_hex(canonical_json_bytes(records)),
            "input_sequence_commitment": str(planned["input_sequence_commitment"]),
            "expected_accounting": expected,
        }

    reference_validation = (
        _bind_reference_validation(
            Path(reference_val_dir), expected_tokenizer_sha256=tokenizer_sha256
        )
        if reference_val_dir is not None
        else None
    )
    tokenizer_release = (
        _bind_tokenizer_release(
            Path(tokenizer_release_manifest), expected_tokenizer_sha256=tokenizer_sha256
        )
        if tokenizer_release_manifest is not None
        else None
    )

    _, native_bundle_sha256 = p_native_implementation_bundle(repo_root)
    provenance = {
        "schema_version": NATIVE_PROVENANCE_SCHEMA,
        "provenance_chain_kind": NATIVE_CHAIN_KIND,
        "accepted_stage_i": accepted.as_canonical(),
        "accepted_stage_i_identity_sha256": accepted.identity_sha256(),
        "candidate_m_plan_sha256": plan_sha256,
        "candidate_m_plan_schema": CANDIDATE_PLAN_SCHEMA,
        "stage_m_implementation_bundle_sha256": bundle_sha256,
        "stage_m_ordering_policy": ORDERING_CONTRACT_ID,
        "stage_p_native_validator_bundle_sha256": native_bundle_sha256,
        "shared_tokenizer_sha256": tokenizer_sha256,
        "model_contract": dict(MODEL_CONTRACT),
        "stages": stages,
        "reference_validation": reference_validation,
        "tokenizer_release": tokenizer_release,
        "full_chain_validated": True,
    }
    provenance["native_chain_identity_sha256"] = sha256_hex(
        canonical_json_bytes({k: v for k, v in provenance.items() if k != "full_chain_validated"})
    )
    return provenance


def native_chain_field_names() -> Sequence[str]:
    return NATIVE_ONLY_FIELDS


__all__ = [
    "LEGACY_CHAIN_KIND",
    "LEGACY_ONLY_FIELDS",
    "NATIVE_CHAIN_KIND",
    "NATIVE_ONLY_FIELDS",
    "NATIVE_PROVENANCE_SCHEMA",
    "PROVENANCE_CHAIN_KINDS",
    "NativeProvenanceError",
    "assert_single_branch",
    "native_chain_field_names",
    "validate_native_chain",
]
