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
    EXCLUSION_REFERENCE_FIELDS,
    MODEL_CONTRACT,
    ORDERING_CONTRACT_ID,
    SEQ_LEN,
    STAGE_STREAMS,
    canonical_exclusion_authority,
    canonical_json_bytes,
    canonical_tokenizer_authority,
    derive_exclusion_reference,
    file_sha256,
    p_native_implementation_bundle,
    plan_exclusion_authority,
    require_canonical_file,
    require_identical_exclusion_authorities,
    sha256_hex,
    stream_accounting,
    validate_candidate_plan_contract,
    validated_sha256,
)
from pretrain.stage_m_output_v1 import MANIFEST_FILENAME, validate_published_release  # noqa: E402

NATIVE_CHAIN_KIND = "accepted_stage_i_native_v1"
LEGACY_CHAIN_KIND = "legacy_selector_v1"
# R2 bumps this deliberately: the native provenance object gained shared_exclusion_authority
# and candidate_plan_contract_field_count, and renamed its chain-identity field, so the
# serialized contract changed. Schema discipline, not a version bump for changed source bytes.
NATIVE_PROVENANCE_SCHEMA = "petitgpt-accepted-stage-i-native-release-provenance-v2"

# R2-E. Two DISTINCT identities, named for what they actually cover. The R1 report wrongly
# described them as one mirrored definition; they were never the same projection.
#
#   PRE-MERGE  (here)  covers what validate_native_chain itself commits to, before the planner
#                      merges the shared G/G2 authorities in.
#   POST-MERGE (below) covers the complete assembled native data authority that the
#                      training-facing run-plan contract consumes.
PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA = "petitgpt-stage-i-native-premerge-chain-identity-v1"

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


# R2-C. The complete load-bearing semantic projection of a Stage-M schema-3 release, derived
# from the fields build_release_meta actually emits rather than from the review's examples.
# Every group here is checked by validate_release_semantics; the count is machine-derived from
# this tuple so it cannot drift from the prose.
M_RELEASE_SEMANTIC_GROUPS = (
    "release_schema_version",
    "release_completion_status",
    "release_storage_dtype",
    "release_byte_order_profile",
    "release_vocab_size",
    "release_canonical_contract_block",
    "release_legacy_flags",
    "release_source_exhaustion_policy",
    "release_shard_tokens",
    "release_train_split_geometry",
    "release_no_validation_split",
    "release_shard_inventory_and_digests",
    "tokenizer_identity",
    "tokenizer_path",
    "stage_identity",
    "stage_stream_count",
    "candidate_plan_identity",
    "candidate_plan_schema",
    "ordering_policy",
    "commitment_schema",
    "stage_input_commitment",
    "accepted_i_run_identity",
    "accepted_i_manifest_identity",
    "accepted_i_completion_identity",
    "accepted_i_layer2_identity",
    "accepted_i_binding_identity",
    "accepted_i_record_token_aggregates",
    "per_stage_input_accounting",
    "per_stage_expected_accounting",
    "per_stage_actual_accounting",
    "document_framing_semantics",
    "tail_lookahead_semantics",
    "model_seq_len_authority",
    "shared_exclusion_authority",
    "implementation_identity",
    "environment_identity",
    "validation_completion_claims",
)


def validate_release_semantics(
    *,
    stage: str,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_sha256: str,
    accepted: Any,
    repo_root: Path,
) -> tuple[str, ...]:
    """Check every load-bearing semantic group of one Stage-M release. R2-C.

    Each group is proved against the authorized candidate-M plan, the verified accepted-Stage-I
    authority, or the frozen contract -- whichever is the real authority for that fact. Where a
    fact is serialized in more than one place, the copies must agree, so a release cannot carry
    two contradictory statements of the same thing.
    """
    from pretrain.stage_m_contract_v1 import MODEL_CONTRACT as _MODEL

    checked: list[str] = []

    def group(name: str, condition: object, message: str) -> None:
        _require(condition, f"{stage}: {message}")
        checked.append(name)

    binding = _mapping(manifest.get("stage_m"), field=f"{stage}.meta.stage_m")
    profile = _mapping(plan.get("release_profile"), field="plan.release_profile")
    planned = _mapping(
        (plan.get("stage_streams") or {}).get(stage), field=f"plan.stage_streams.{stage}"
    )
    expected_accounting = dict(planned.get("expected_accounting") or {})
    seq_len = int(_MODEL["seq_len"])

    group(
        "release_schema_version",
        manifest.get("schema_version") == 3,
        "release is not canonical schema 3",
    )
    group(
        "release_completion_status",
        manifest.get("status") == "complete",
        "release is not in the canonical schema-3 complete state",
    )
    group(
        "release_storage_dtype",
        manifest.get("dtype") == profile.get("storage_dtype"),
        f"release dtype {manifest.get('dtype')!r} differs from the plan profile",
    )
    group(
        "release_byte_order_profile",
        profile.get("storage_byte_order") == "little"
        and profile.get("storage_dtype_explicit") == "<u2",
        "plan release profile does not declare the frozen little-endian uint16 byte order",
    )
    group(
        "release_vocab_size",
        manifest.get("vocab_size") == int(_MODEL["vocab_size"]),
        "release vocab_size differs from the frozen model contract",
    )

    contract_block = _mapping(manifest.get("contract"), field=f"{stage}.contract")
    group(
        "release_canonical_contract_block",
        contract_block.get("canonical") is True
        and contract_block.get("issues") == []
        and contract_block.get("legacy_allow_noncanonical_contract") is False,
        "release contract block is not canonical",
    )
    legacy = _mapping(manifest.get("legacy_flags"), field=f"{stage}.legacy_flags")
    group(
        "release_legacy_flags",
        legacy.get("allow_noncanonical_contract") is False
        and legacy.get("replay_on_exhaustion") is False,
        "release declares legacy/debug flags",
    )
    group(
        "release_source_exhaustion_policy",
        manifest.get("source_exhaustion_policy") == "fail_fast",
        "release does not declare fail_fast source exhaustion",
    )
    group(
        "release_shard_tokens",
        manifest.get("shard_tokens") == profile.get("shard_tokens"),
        "release shard_tokens differs from the plan profile",
    )

    stored = _int(validation.get("expected_tokens"), field=f"{stage}.expected_tokens")
    group(
        "release_train_split_geometry",
        manifest.get("train_tokens") == stored
        and manifest.get("train_shards") == validation.get("expected_shards")
        and (manifest.get("accounting") or {}).get("train", {}).get("emitted_shard_tokens")
        == stored,
        "release train split geometry is internally inconsistent",
    )
    # R3-C 8.2: every representation of "there is no validation split here".
    val_accounting = (manifest.get("accounting") or {}).get("val") or {}
    shard_files = manifest.get("shard_files") or {}
    group(
        "release_no_validation_split",
        manifest.get("val_tokens") == 0
        and manifest.get("val_shards") == 0
        and manifest.get("val_by_source") == {}
        and manifest.get("val_ratio") == 0.0
        and manifest.get("val_shard_tokens") == manifest.get("shard_tokens")
        and shard_files.get("val") == []
        and shard_files.get("val_by_source") == {}
        and all(
            int(val_accounting.get(key, -1)) == 0
            for key in (
                "documents",
                "content_tokens",
                "boundary_tokens",
                "serialized_tokens",
                "separator_tokens",
                "emitted_shard_tokens",
            )
        ),
        "a Stage-M release must represent the absence of a validation split consistently "
        "across val_tokens, val_shards, val_ratio, val_shard_tokens, accounting.val, "
        "shard_files.val and val_by_source",
    )
    records = [dict(r) for r in validation.get("shard_file_records") or ()]
    group(
        "release_shard_inventory_and_digests",
        len(records) == int(validation["expected_shards"]) and bool(records),
        "validated shard inventory is incomplete",
    )

    tokenizer_sha256 = ((plan.get("resources") or {}).get("tokenizer") or {}).get("sha256")
    group(
        "tokenizer_identity",
        manifest.get("tokenizer_sha256") == tokenizer_sha256
        and validation.get("tokenizer_sha256") == tokenizer_sha256,
        "release tokenizer differs from the plan",
    )
    # R4-C: the emitted tokenizer PATH, resolved from the repository root and hashed. The R3
    # basename+suffix comparison accepted a nonexistent path, an alternate directory holding a
    # same-named file, and a correct digest declared against the wrong path; all three now fail.
    planned_tokenizer_path = ((plan.get("resources") or {}).get("tokenizer") or {}).get("path")
    emitted_tokenizer_path = manifest.get("tokenizer_path")
    canonical_tokenizer = canonical_tokenizer_authority(repo_root)
    tokenizer_checks: dict[str, bool] = {
        "plan_path_is_canonical": planned_tokenizer_path == canonical_tokenizer["tokenizer_path"],
        "release_path_equals_plan_path": isinstance(emitted_tokenizer_path, str)
        and emitted_tokenizer_path == planned_tokenizer_path,
        "release_digest_is_canonical": manifest.get("tokenizer_sha256")
        == canonical_tokenizer["tokenizer_sha256"],
    }
    if tokenizer_checks["release_path_equals_plan_path"]:
        # Open the file at exactly that repository-relative path and hash the actual bytes.
        assert isinstance(emitted_tokenizer_path, str)
        try:
            actual = require_canonical_file(
                repo_root, emitted_tokenizer_path, label=f"{stage}.tokenizer_path"
            )
        except Exception:  # noqa: BLE001 - a missing/unreadable path is a normal failure here
            actual = None
        tokenizer_checks["bytes_at_path_match"] = actual == canonical_tokenizer["tokenizer_sha256"]
    else:
        tokenizer_checks["bytes_at_path_match"] = False
    failed_tokenizer = sorted(k for k, ok in tokenizer_checks.items() if not ok)
    group(
        "tokenizer_path",
        not failed_tokenizer,
        f"release tokenizer_path {emitted_tokenizer_path!r} is not the canonical accepted-G "
        f"tokenizer {canonical_tokenizer['tokenizer_path']!r}: "
        f"{len(failed_tokenizer)} of {len(tokenizer_checks)} checks failed: "
        + ", ".join(failed_tokenizer),
    )
    group(
        "stage_identity",
        binding.get("stage") == stage,
        "release stage_m block names a different stage",
    )
    group(
        "stage_stream_count",
        binding.get("stage_stream_count") == len(STAGE_STREAMS),
        "release stage_stream_count differs from the frozen two-stream contract",
    )
    group(
        "candidate_plan_identity",
        binding.get("candidate_plan_sha256") == plan_sha256,
        "release was not produced by this candidate Stage-M plan",
    )
    group(
        "candidate_plan_schema",
        binding.get("candidate_plan_schema") == plan.get("schema_version") == CANDIDATE_PLAN_SCHEMA,
        "release candidate_plan_schema disagrees with the plan",
    )
    group(
        "ordering_policy",
        binding.get("ordering_policy") == ORDERING_CONTRACT_ID,
        "release does not declare the frozen ordering policy",
    )
    group(
        "commitment_schema",
        binding.get("input_sequence_commitment_schema")
        == planned.get("input_sequence_commitment_schema"),
        "release input_sequence_commitment_schema differs from the plan",
    )
    group(
        "stage_input_commitment",
        binding.get("input_sequence_commitment") == planned.get("input_sequence_commitment"),
        "release input sequence commitment differs from the plan",
    )

    accepted_block = _mapping(
        binding.get("accepted_stage_i"), field=f"{stage}.stage_m.accepted_stage_i"
    )
    group(
        "accepted_i_run_identity",
        accepted_block.get("run_identity") == accepted.run_identity,
        "release accepted Stage-I run identity differs from the verified publication",
    )
    group(
        "accepted_i_manifest_identity",
        accepted_block.get("manifest_sha256") == accepted.manifest_sha256,
        "release accepted Stage-I manifest digest differs",
    )
    group(
        "accepted_i_completion_identity",
        accepted_block.get("completion_object_sha256") == accepted.completion_sha256,
        "release accepted Stage-I completion digest differs",
    )
    group(
        "accepted_i_layer2_identity",
        accepted_block.get("layer2_expected_result_sha256") == accepted.layer2_sha256,
        "release accepted Stage-I Layer-2 digest differs",
    )
    group(
        "accepted_i_binding_identity",
        accepted_block.get("identity_sha256") == accepted.identity_sha256(),
        "release accepted Stage-I binding identity differs",
    )
    group(
        "accepted_i_record_token_aggregates",
        accepted_block.get("shard_count") == accepted.shard_count
        and accepted_block.get("total_records") == accepted.total_records
        and accepted_block.get("total_serialized_tokens") == accepted.total_serialized_tokens,
        "release accepted Stage-I aggregates differ from the verified publication",
    )

    group(
        "per_stage_input_accounting",
        _int(binding.get("input_record_count"), field=f"{stage}.input_record_count")
        == int(planned["input_record_count"])
        and _int(binding.get("input_serialized_tokens"), field=f"{stage}.input_serialized_tokens")
        == int(planned["input_serialized_tokens"]),
        "release input accounting differs from the plan",
    )
    group(
        "per_stage_expected_accounting",
        dict(manifest.get("stage_m_accounting") or {}) == expected_accounting,
        "release accounting differs from the plan's expected accounting",
    )
    # R3-C 8.3: the whole actual-accounting block, including content tokens.
    documents = manifest.get("documents")
    train_accounting = (manifest.get("accounting") or {}).get("train") or {}
    expected_content = int(planned["input_serialized_tokens"]) - 2 * int(
        planned["input_record_count"]
    )
    group(
        "per_stage_actual_accounting",
        documents == int(planned["input_record_count"])
        and int(train_accounting.get("documents", -1)) == int(planned["input_record_count"])
        and int(train_accounting.get("serialized_tokens", -1))
        == int(planned["input_serialized_tokens"])
        and int(train_accounting.get("content_tokens", -1)) == expected_content
        and int(train_accounting.get("content_tokens", -1)) == int(planned["input_content_tokens"]),
        "release actual accounting differs from the plan "
        "(documents, serialized tokens or content tokens)",
    )

    group(
        "document_framing_semantics",
        contract_block.get("add_bos") is True
        and contract_block.get("add_eos") is True
        and contract_block.get("doc_sep") == ""
        and (manifest.get("accounting") or {}).get("train", {}).get("separator_tokens") == 0
        and (manifest.get("accounting") or {}).get("train", {}).get("boundary_tokens")
        == 2 * int(documents or 0),
        "release document framing declarations are not the frozen contract",
    )
    group(
        "tail_lookahead_semantics",
        stored == int(expected_accounting["retained_stored_token_ids"])
        and (stored - 1) % seq_len == 0
        and (stored - 1) // seq_len == int(expected_accounting["training_sequences"])
        and int(expected_accounting["padding_tokens"]) == 0
        and int(expected_accounting["final_lookahead_tokens"]) == 1,
        "release block geometry differs from the expected tail/lookahead accounting",
    )
    group(
        "model_seq_len_authority",
        dict(binding.get("model_contract") or {}) == dict(_MODEL)
        and int(expected_accounting["seq_len"]) == seq_len,
        "release model/seq_len authority differs from the frozen contract",
    )

    # R3-C 8.1: every serialized exclusion representation, not just presence.
    exclusion_block = _mapping(
        manifest.get("reference_validation_exclusion"),
        field=f"{stage}.reference_validation_exclusion",
    )
    plan_exclusion = _mapping(
        (plan.get("resources") or {}).get("canonical_exclusion_authority"),
        field="plan.resources.canonical_exclusion_authority",
    )
    exclusion_entries = exclusion_block.get("manifests")
    exclusion_entry = (
        exclusion_entries[0]
        if isinstance(exclusion_entries, list) and len(exclusion_entries) == 1
        else {}
    )
    # R4-A/R4-C: the release's exclusion reference is resolved and the real artifact reopened,
    # then EVERY serialized field in both representations is compared against what that file
    # actually contains -- not against the plan's declarations. The plan is compared separately
    # as its own participant, so agreement here is agreement between two independent reads.
    release_reference = derive_exclusion_reference(
        repo_root,
        {
            "artifact_path": exclusion_block.get("canonical_artifact_path"),
            "artifact_sha256": exclusion_block.get("canonical_artifact_sha256"),
            "artifact_schema_version": exclusion_block.get("canonical_artifact_schema_version"),
            "kind": exclusion_block.get("kind"),
            "hash_algorithm": exclusion_block.get("hash_algorithm"),
            "derived_count": exclusion_block.get("union_hash_count"),
        },
        participant=f"stage_m_release[{stage}]",
        label=f"{stage}.reference_validation_exclusion",
    )
    entry = exclusion_entry or {}
    exclusion_checks = {
        "enabled": exclusion_block.get("enabled") is True,
        "manifest_count": exclusion_block.get("manifest_count") == 1,
        "manifests_length": isinstance(exclusion_entries, list) and len(exclusion_entries) == 1,
        "block.canonical_artifact_path": exclusion_block.get("canonical_artifact_path")
        == release_reference["artifact_path"],
        "block.canonical_artifact_sha256": exclusion_block.get("canonical_artifact_sha256")
        == release_reference["artifact_sha256"],
        "block.canonical_artifact_schema_version": exclusion_block.get(
            "canonical_artifact_schema_version"
        )
        == release_reference["artifact_schema_version"],
        "block.kind": exclusion_block.get("kind") == release_reference["kind"],
        "block.hash_algorithm": exclusion_block.get("hash_algorithm")
        == release_reference["hash_algorithm"],
        "block.union_hash_count": exclusion_block.get("union_hash_count")
        == release_reference["derived_count"],
        "block.enforced_at_stage": exclusion_block.get("enforced_at_stage") == "stage_i",
        "block.reapplied_by_stage_m": exclusion_block.get("reapplied_by_stage_m") is False,
        "entry.enabled": entry.get("enabled") is True,
        "entry.path": entry.get("path") == release_reference["artifact_path"],
        "entry.manifest_sha256": entry.get("manifest_sha256")
        == release_reference["artifact_sha256"],
        "entry.hash_count": entry.get("hash_count") == release_reference["derived_count"],
        "entry.kind": entry.get("kind") == release_reference["kind"],
        "entry.hash_algorithm": entry.get("hash_algorithm") == release_reference["hash_algorithm"],
        "entry.schema_version": entry.get("schema_version")
        == release_reference["artifact_schema_version"],
        # The plan is an independent participant; both must have landed on the same reference.
        "agrees_with_plan": all(
            plan_exclusion.get(field) == release_reference[field]
            for field in EXCLUSION_REFERENCE_FIELDS
        ),
    }
    failed_exclusion = sorted(k for k, ok in exclusion_checks.items() if not ok)
    group(
        "shared_exclusion_authority",
        not failed_exclusion,
        f"release exclusion representations disagree with the artifact the release itself "
        f"names ({release_reference['artifact_path']}): "
        f"{len(failed_exclusion)} of {len(exclusion_checks)} checks failed: "
        + ", ".join(failed_exclusion),
    )

    group(
        "implementation_identity",
        binding.get("implementation_bundle_sha256") == plan.get("implementation_bundle_sha256")
        and binding.get("implementation_commit") == plan.get("implementation_commit"),
        "release implementation identity differs from the plan",
    )
    group(
        "environment_identity",
        dict(binding.get("environment") or {}) == dict(plan.get("environment_contract") or {}),
        "release environment differs from the plan",
    )
    group(
        "validation_completion_claims",
        validation.get("manifest_schema_version") == 3
        and validation.get("dtype") == "uint16"
        and validation.get("release_kind") == "regular"
        and validation.get("split") == "train",
        "release did not validate as a canonical schema-3 train split",
    )

    missing = [g for g in M_RELEASE_SEMANTIC_GROUPS if g not in checked]
    _require(not missing, f"{stage}: semantic groups not validated: {missing}")
    _require(
        len(checked) == len(M_RELEASE_SEMANTIC_GROUPS),
        f"{stage}: expected {len(M_RELEASE_SEMANTIC_GROUPS)} groups, checked {len(checked)}",
    )
    return tuple(checked)


def release_exclusion_authority(
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    repo_root: Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """One Stage-M release's independently derived exclusion authority.

    R4-B. The release is a full participant: the path comes from the release's own serialized
    block, the artifact at that path is opened and hashed here, and everything the release
    declares -- including the digest the frozen schema-3 validator independently parsed out of
    it -- is checked against the file that is actually there. Nothing is carried in from the
    plan or from another participant.
    """
    declared = manifest.get("reference_validation_exclusion")
    _require(
        isinstance(declared, Mapping),
        f"{stage}: release has no reference_validation_exclusion object",
    )
    assert isinstance(declared, Mapping)
    digests = list(validation.get("reference_exclusion_manifest_sha256s") or ())
    _require(
        len(digests) == 1,
        f"{stage}: a Stage-M release must name exactly one exclusion manifest, got {digests}",
    )
    entries = declared.get("manifests")
    _require(
        isinstance(entries, list) and len(entries) == 1,
        f"{stage}: release reference_validation_exclusion.manifests must hold exactly one entry",
    )
    assert isinstance(entries, list)
    entry = entries[0] or {}
    # The release states the reference twice: once at block level, once in its single manifest
    # entry. Both spellings must agree before either is trusted as this participant's reference.
    _require(
        entry.get("path") == declared.get("canonical_artifact_path"),
        f"{stage}: release names {entry.get('path')!r} in its manifest entry but "
        f"{declared.get('canonical_artifact_path')!r} at block level",
    )
    _require(
        digests[0] == declared.get("canonical_artifact_sha256"),
        f"{stage}: schema-3 validation parsed exclusion digest {digests[0]} but the release "
        f"block declares {declared.get('canonical_artifact_sha256')!r}",
    )
    reference = {
        "artifact_path": entry.get("path"),
        "artifact_sha256": digests[0],
        "artifact_schema_version": declared.get("canonical_artifact_schema_version"),
        "kind": declared.get("kind"),
        "hash_algorithm": declared.get("hash_algorithm"),
        "derived_count": declared.get("union_hash_count"),
    }
    return derive_exclusion_reference(
        repo_root,
        reference,
        participant=f"stage_m_release[{stage}]",
        label=f"stage_m_release[{stage}]",
    )


def planned_commitment_schema(plan: Mapping[str, Any], stage: str) -> str:
    """The input-sequence-commitment schema the authorized plan declares for one stage."""
    streams = _mapping(plan.get("stage_streams"), field="plan.stage_streams")
    entry = _mapping(streams.get(stage), field=f"plan.stage_streams.{stage}")
    schema = entry.get("input_sequence_commitment_schema")
    _require(
        isinstance(schema, str) and schema,
        f"plan.stage_streams.{stage}.input_sequence_commitment_schema is missing",
    )
    assert isinstance(schema, str)
    return schema


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


def validate_native_chain(
    *,
    repo_root: Path,
    accepted_stage_i_dir: Path,
    candidate_m_plan: Path,
    expected_candidate_m_plan_sha256: str,
    stage_releases: Mapping[str, Path],
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
    # R2-B / section 7: the same closed contract validator the producer uses. A plan whose
    # digest matches but whose declarations contradict the frozen contract is refused here too.
    try:
        plan_contract = validate_candidate_plan_contract(plan, repo_root)
    except Exception as exc:  # normalized into this module's controlled failure type
        raise NativeProvenanceError(f"candidate Stage-M plan contract: {exc}") from exc

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
    # R2-A / R3-B: the candidate-M plan, both releases and the two accepted authorities each
    # derive their own view; the comparison then requires the SAME canonical artifact digest
    # and the SAME independently derived count.
    exclusion_authorities = [plan_exclusion_authority(plan, repo_root)]
    canonical = canonical_exclusion_authority(repo_root)
    exclusion_authorities.append(canonical["accepted_g"])
    exclusion_authorities.append(canonical["accepted_g2"])
    stages: dict[str, Any] = {}
    for stage in STAGE_STREAMS:
        manifest, manifest_sha256, validation = _load_release(stage_releases[stage], stage=stage)
        # R2-C: every load-bearing semantic group, table-driven so the coverage count cannot
        # drift from the prose in the evidence.
        groups = validate_release_semantics(
            stage=stage,
            manifest=manifest,
            validation=validation,
            plan=plan,
            plan_sha256=plan_sha256,
            accepted=accepted,
            repo_root=repo_root,
        )
        planned = _mapping(
            (plan.get("stage_streams") or {}).get(stage), field=f"plan.stage_streams.{stage}"
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
        stored = _int(validation["expected_tokens"], field=f"{stage}.expected_tokens")
        records = [dict(r) for r in validation.get("shard_file_records") or ()]

        exclusion_authorities.append(
            release_exclusion_authority(manifest, validation, repo_root, stage=stage)
        )
        records = [dict(r) for r in validation.get("shard_file_records") or ()]
        _require(
            len(records) == int(validation["expected_shards"]),
            f"{stage}: validated shard inventory is incomplete",
        )
        stages[stage] = {
            "semantic_groups_checked": len(groups),
            "release_dir": str(Path(stage_releases[stage]).expanduser().resolve()),
            "manifest_sha256": manifest_sha256,
            "shards": int(validation["expected_shards"]),
            "stored_token_ids": stored,
            "shard_inventory_sha256": sha256_hex(canonical_json_bytes(records)),
            "input_sequence_commitment": str(planned["input_sequence_commitment"]),
            "expected_accounting": expected,
        }

    agreed_exclusion = require_identical_exclusion_authorities(exclusion_authorities)

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
        "shared_exclusion_authority": agreed_exclusion,
        "candidate_plan_contract_field_count": plan_contract["validated_field_count"],
        "full_chain_validated": True,
    }
    provenance["premerge_native_chain_identity_sha256"] = premerge_native_chain_identity_sha256(
        provenance
    )
    return provenance


def premerge_native_chain_identity_sha256(provenance: Mapping[str, Any]) -> str:
    """Digest over the closed PRE-MERGE projection.

    Covers only what validate_native_chain established. It is deliberately NOT the final
    training-facing data-branch identity; that is computed after the planner merges the shared
    G/G2 authorities, by :func:`post_merge_data_branch_identity_sha256`. The R1 report described
    these two as one mirrored definition, which was never true.
    """
    return sha256_hex(
        canonical_json_bytes({
            "schema_version": PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA,
            "fields": {
                name: provenance.get(name) for name in PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS
            },
        })
    )


# --------------------------------------------------------------------- post-merge identity

POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA = (
    "petitgpt-stage-p-native-post-merge-data-branch-identity-v1"
)

# The complete assembled native data authority as it exists in the final run plan. Producer
# (here) and training-facing consumer (pretrain/run_plan_contract.py) keep separate
# implementations so the launch path retains its zero-local-import surface; field list, field
# order, canonical encoder, schema name and hash algorithm must match exactly, and tests
# deliberately drift each side to prove a mismatch is detected.
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


def post_merge_data_branch_identity_sha256(provenance: Mapping[str, Any]) -> str:
    """Identity of the COMPLETE assembled native data authority. R2-E / section 13."""
    stages = provenance.get("stages")
    _require(isinstance(stages, Mapping), "post-merge identity requires release_provenance.stages")
    assert isinstance(stages, Mapping)
    projection: dict[str, Any] = {
        "schema_version": POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA,
        "fields": {name: provenance.get(name) for name in POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS},
    }
    projection["fields"]["stages"] = {
        stage: {key: (stages.get(stage) or {}).get(key) for key in POST_MERGE_STAGE_FIELDS}
        for stage in sorted(stages)
    }
    return sha256_hex(canonical_json_bytes(projection))


def native_chain_field_names() -> Sequence[str]:
    return NATIVE_ONLY_FIELDS


__all__ = [
    "LEGACY_CHAIN_KIND",
    "LEGACY_ONLY_FIELDS",
    "POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS",
    "POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA",
    "POST_MERGE_STAGE_FIELDS",
    "POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS",
    "POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA",
    "POST_MERGE_STAGE_FIELDS",
    "PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS",
    "PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA",
    "NATIVE_CHAIN_KIND",
    "NATIVE_ONLY_FIELDS",
    "NATIVE_PROVENANCE_SCHEMA",
    "PROVENANCE_CHAIN_KINDS",
    "NativeProvenanceError",
    "assert_single_branch",
    "native_chain_field_names",
    "M_RELEASE_SEMANTIC_GROUPS",
    "release_exclusion_authority",
    "validate_release_semantics",
    "post_merge_data_branch_identity_sha256",
    "premerge_native_chain_identity_sha256",
    "planned_commitment_schema",
    "validate_native_chain",
]
