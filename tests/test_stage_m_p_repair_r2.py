"""R2 repair regressions for the Codex R1 re-review findings.

Bounded synthetic fixtures only. No real Stage-M production, no packed corpus.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import pretrain.run_plan_contract as rpc
import pretrain.stage_m_contract_v1 as contract
from pretrain.stage_m_contract_v1 import (
    MODEL_CONTRACT,
    PACKING_SEMANTICS,
    RELEASE_PROFILE,
    canonical_json_bytes,
    exclusion_authority,
    plan_exclusion_authority,
    require_identical_exclusion_authorities,
    validate_candidate_plan_contract,
)
import pretrain.stage_m_output_v1 as output
import pretrain.stage_p_native_provenance_v1 as native
from pretrain.stage_p_native_provenance_v1 import (
    M_RELEASE_SEMANTIC_GROUPS,
    NativeProvenanceError,
)
from tests._stage_m_fixtures import read_json
from tests.test_stage_m_p_repair_r1 import native_chain, native_plan

# ------------------------------------------------------------------ R2-A shared exclusion


def test_candidate_plan_exclusion_joins_the_shared_comparison(m_run):
    provenance = native_chain(m_run)
    agreed = provenance["shared_exclusion_authority"]
    assert agreed["schema_version"] == contract.CANONICAL_EXCLUSION_AUTHORITY_SCHEMA
    assert "candidate_m_plan" in agreed["participants"]
    assert any(p.startswith("stage_m_release[") for p in agreed["participants"])
    assert "accepted_g" in agreed["participants"]
    assert "accepted_g2" in agreed["participants"]


def test_candidate_exclusion_digest_mismatch_is_rejected(m_run):
    """A well-formed but different digest is a shared-authority disagreement, not a plan-shape
    error: the plan alone cannot know it is wrong, so R2-A's cross-artifact comparison is what
    catches it."""
    plan = read_json(m_run["plan_path"])
    plan["resources"]["canonical_exclusion_authority"]["artifact_sha256"] = "0" * 64
    m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    import hashlib

    sha = hashlib.sha256(m_run["plan_path"].read_bytes()).hexdigest()
    with pytest.raises((NativeProvenanceError, contract.StageMError)):
        native_chain(m_run, expected_candidate_m_plan_sha256=sha)


def test_candidate_exclusion_count_mismatch_is_rejected(m_run):
    plan = read_json(m_run["plan_path"])
    plan["resources"]["canonical_exclusion_authority"]["derived_count"] = 999
    m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    import hashlib

    sha = hashlib.sha256(m_run["plan_path"].read_bytes()).hexdigest()
    with pytest.raises((NativeProvenanceError, contract.StageMError)):
        native_chain(m_run, expected_candidate_m_plan_sha256=sha)


@pytest.mark.parametrize("stage", ["stage_a", "stage_b"])
@pytest.mark.parametrize("field", ["manifest_sha256", "union_hash_count"])
def test_release_exclusion_mismatch_is_rejected(m_run, stage, field):
    path = m_run["releases"][stage] / "meta.json"
    meta = read_json(path)
    block = meta["reference_validation_exclusion"]
    if field == "manifest_sha256":
        block["manifests"][0]["manifest_sha256"] = "0" * 64
        block["canonical_artifact_sha256"] = "0" * 64
    else:
        block["union_hash_count"] = 987
    path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


@pytest.mark.parametrize("authority", ["accepted_g", "accepted_g2"])
@pytest.mark.parametrize("field", ["manifest_sha256", "hash_count"])
def test_accepted_authority_exclusion_mismatch_is_rejected(m_run, authority, field):
    canonical = m_run["canonical_exclusion"]
    path = canonical["g_manifest"] if authority == "accepted_g" else canonical["g2_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = (
        payload["reference_reserve_exclusion"]["manifests"][0]
        if authority == "accepted_g"
        else payload["reserve_provenance"]["reserve_exclusion"]
    )
    entry[field] = "0" * 64 if field == "manifest_sha256" else 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


def test_exclusion_authority_normalizer_is_single_definition():
    common = {
        "artifact_schema_version": 1,
        "kind": "petitgpt_reference_validation_exclusions",
        "hash_algorithm": "sha256-cleaned-text-utf8-v1",
    }
    a = exclusion_authority(
        participant="one", artifact_path="p", artifact_sha256="a" * 64, derived_count=5, **common
    )
    b = exclusion_authority(
        participant="two", artifact_path="p", artifact_sha256="a" * 64, derived_count=5, **common
    )
    assert require_identical_exclusion_authorities([a, b])["derived_count"] == 5
    c = exclusion_authority(
        participant="three", artifact_path="p", artifact_sha256="a" * 64, derived_count=6, **common
    )
    with pytest.raises(contract.StageMError, match="disagree"):
        require_identical_exclusion_authorities([a, c])
    # R3: same count, DIFFERENT artifact -- the substitution candidate v3 made.
    d = exclusion_authority(
        participant="copy",
        artifact_path="other",
        artifact_sha256="b" * 64,
        derived_count=5,
        **common,
    )
    with pytest.raises(contract.StageMError, match="disagree"):
        require_identical_exclusion_authorities([a, d])
    # R4-A: the closed sub-schema is compared whole, so kind and algorithm are load-bearing too.
    for field, bad in (
        ("kind", "petitgpt_something_else"),
        ("hash_algorithm", "sha256"),
        ("artifact_schema_version", 2),
    ):
        variant = exclusion_authority(
            participant=f"wrong_{field}",
            artifact_path="p",
            artifact_sha256="a" * 64,
            derived_count=5,
            **{**common, field: bad},
        )
        with pytest.raises(contract.StageMError, match=field):
            require_identical_exclusion_authorities([a, variant])


# ------------------------------------------------------------------ R2-B candidate plan


def _plan(m_run) -> dict[str, Any]:
    return read_json(m_run["plan_path"])


def test_valid_plan_passes_the_closed_contract(m_run):
    result = validate_candidate_plan_contract(_plan(m_run), m_run["tmp_path"])
    assert result["schema_version"] == contract.CANDIDATE_PLAN_CONTRACT_SCHEMA
    assert result["validated_field_count"] > 80
    assert result["stage_streams"] == list(contract.STAGE_STREAMS)


def _mutations() -> dict[str, Any]:
    """Semantically invalid plans whose file digest would still be 'correct' locally."""

    def at(path: str, value: Any):
        def apply(plan):
            node = plan
            parts = path.split(".")
            for key in parts[:-1]:
                node = node[key]
            node[parts[-1]] = value

        return apply

    def drop(path: str):
        def apply(plan):
            node = plan
            parts = path.split(".")
            for key in parts[:-1]:
                node = node[key]
            node.pop(parts[-1], None)

        return apply

    return {
        "wrong_commitment_schema": at(
            "stage_streams.stage_a.input_sequence_commitment_schema", "other-v1"
        ),
        "wrong_release_schema": at("release_profile.manifest_schema_version", 2),
        "wrong_stream_count": at("packing_semantics.stage_stream_count", 3),
        "wrong_stage_set": drop("stage_streams.stage_b"),
        "wrong_ordering_policy": at("ordering_contract.policy", "SOMETHING_ELSE"),
        "hash_shuffle_enabled": at("ordering_contract.hash_shuffle", True),
        "weighted_interleave_enabled": at("ordering_contract.weighted_interleave", True),
        "new_random_permutation": at("ordering_contract.new_random_permutation", True),
        "wrong_dtype": at("release_profile.storage_dtype", "uint32"),
        "big_endian_profile": at("release_profile.storage_byte_order", "big"),
        "explicit_be_dtype": at("release_profile.storage_dtype_explicit", ">u2"),
        "wrong_T": at("model_contract.seq_len", 1024),
        "wrong_stride": at("packing_semantics.stride", 1024),
        "wrong_read_length": at("packing_semantics.read_length", 2048),
        "wrong_padding_policy": at("packing_semantics.padding", "zero"),
        "separator_introduced": at("packing_semantics.textual_document_separator", "\\n\\n"),
        "wrong_stage_a_totals": at(
            "stage_streams.stage_a.expected_accounting.training_sequences", 1
        ),
        "wrong_stage_b_totals": at("stage_streams.stage_b.expected_accounting.tail_transitions", 0),
        "wrong_total_accounting": at("expected_totals.total_padding_tokens", 7),
        "malformed_exclusion_digest": at(
            "resources.canonical_exclusion_authority.artifact_sha256", "not-a-sha"
        ),
        "wrong_exclusion_count": at("resources.canonical_exclusion_authority.derived_count", 0),
        "wrong_tokenizer_identity": at("resources.tokenizer.sha256", "nothex"),
        "wrong_bundle_identity": at("implementation_bundle_sha256", "0" * 64),
        "wrong_environment": at("environment_contract.byte_order", "big"),
        "wrong_authorization_status": at("authorization_status", "AUTHORIZED"),
        "wrong_text_field": at("text_field", "text"),
        "legacy_orchestration_claimed": at("legacy_orchestration_used", True),
        "accepted_i_totals_inconsistent": at("accepted_stage_i.total_content_tokens", 1),
        "shard_inventory_count_mismatch": at("accepted_stage_i.shard_count", 999),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_semantically_invalid_plan_is_rejected(m_run, name):
    plan = _plan(m_run)
    _mutations()[name](plan)
    with pytest.raises((contract.StageMError, KeyError)):
        validate_candidate_plan_contract(plan, m_run["tmp_path"])


def test_plan_contract_runs_on_the_production_authorization_path(m_run):
    """Section 7: a semantically invalid plan is refused even with a matching digest."""
    import hashlib

    import pretrain.stage_m_realize_v1 as realize

    plan = _plan(m_run)
    plan["ordering_contract"]["hash_shuffle"] = True
    m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    sha = hashlib.sha256(m_run["plan_path"].read_bytes()).hexdigest()
    with pytest.raises(contract.StageMError, match="hash_shuffle"):
        realize.authorize_plan(m_run["plan_path"], sha, m_run["tmp_path"].resolve())


# ------------------------------------------------------------------ R2-C release metadata


def test_release_semantic_groups_are_closed_and_unique():
    assert len(set(M_RELEASE_SEMANTIC_GROUPS)) == len(M_RELEASE_SEMANTIC_GROUPS)
    assert len(M_RELEASE_SEMANTIC_GROUPS) >= 30


def test_every_group_is_checked_for_both_stages(m_run):
    provenance = native_chain(m_run)
    for stage in contract.STAGE_STREAMS:
        assert provenance["stages"][stage]["semantic_groups_checked"] == len(
            M_RELEASE_SEMANTIC_GROUPS
        )


_GROUP_MUTATIONS: dict[str, Any] = {
    "release_schema_version": lambda m: m.__setitem__("schema_version", 2),
    "release_completion_status": lambda m: m.__setitem__("status", "partial"),
    "release_storage_dtype": lambda m: m.__setitem__("dtype", "uint32"),
    "release_vocab_size": lambda m: m.__setitem__("vocab_size", 31_999),
    "release_canonical_contract_block": lambda m: m["contract"].__setitem__("canonical", False),
    "release_legacy_flags": lambda m: m["legacy_flags"].__setitem__("replay_on_exhaustion", True),
    "release_source_exhaustion_policy": lambda m: m.__setitem__(
        "source_exhaustion_policy", "legacy_replay"
    ),
    "release_shard_tokens": lambda m: m.__setitem__("shard_tokens", 12_345),
    "release_train_split_geometry": lambda m: m.__setitem__("train_tokens", 1),
    "release_no_validation_split": lambda m: m.__setitem__("val_ratio", 0.002),
    "tokenizer_identity": lambda m: m.__setitem__("tokenizer_sha256", "0" * 64),
    "stage_identity": lambda m: m["stage_m"].__setitem__("stage", "stage_b"),
    "stage_stream_count": lambda m: m["stage_m"].__setitem__("stage_stream_count", 3),
    "candidate_plan_identity": lambda m: m["stage_m"].__setitem__(
        "candidate_plan_sha256", "0" * 64
    ),
    "candidate_plan_schema": lambda m: m["stage_m"].__setitem__(
        "candidate_plan_schema", "other-v1"
    ),
    "ordering_policy": lambda m: m["stage_m"].__setitem__("ordering_policy", "OTHER"),
    "commitment_schema": lambda m: m["stage_m"].__setitem__(
        "input_sequence_commitment_schema", "other-v1"
    ),
    "stage_input_commitment": lambda m: m["stage_m"].__setitem__(
        "input_sequence_commitment", "0" * 64
    ),
    "accepted_i_run_identity": lambda m: m["stage_m"]["accepted_stage_i"].__setitem__(
        "run_identity", "0" * 64
    ),
    "accepted_i_manifest_identity": lambda m: m["stage_m"]["accepted_stage_i"].__setitem__(
        "manifest_sha256", "0" * 64
    ),
    "accepted_i_completion_identity": lambda m: m["stage_m"]["accepted_stage_i"].__setitem__(
        "completion_object_sha256", "0" * 64
    ),
    "accepted_i_layer2_identity": lambda m: m["stage_m"]["accepted_stage_i"].__setitem__(
        "layer2_expected_result_sha256", "0" * 64
    ),
    "accepted_i_binding_identity": lambda m: m["stage_m"]["accepted_stage_i"].__setitem__(
        "identity_sha256", "0" * 64
    ),
    "accepted_i_record_token_aggregates": lambda m: m["stage_m"]["accepted_stage_i"].__setitem__(
        "total_records", 1
    ),
    "per_stage_input_accounting": lambda m: m["stage_m"].__setitem__("input_record_count", 1),
    "per_stage_expected_accounting": lambda m: m["stage_m_accounting"].__setitem__(
        "training_sequences", 1
    ),
    "per_stage_actual_accounting": lambda m: m["accounting"]["train"].__setitem__(
        "content_tokens", 1
    ),
    "document_framing_semantics": lambda m: m["contract"].__setitem__("doc_sep", "\\n\\n"),
    "tail_lookahead_semantics": lambda m: m["stage_m_accounting"].__setitem__("padding_tokens", 4),
    "model_seq_len_authority": lambda m: m["stage_m"]["model_contract"].__setitem__("n_layers", 31),
    "shared_exclusion_authority": lambda m: m["reference_validation_exclusion"].__setitem__(
        "reapplied_by_stage_m", True
    ),
    "tokenizer_path": lambda m: m.__setitem__("tokenizer_path", "/elsewhere/other.json"),
    "implementation_identity": lambda m: m["stage_m"].__setitem__(
        "implementation_bundle_sha256", "0" * 64
    ),
    "environment_identity": lambda m: m["stage_m"]["environment"].__setitem__("byte_order", "big"),
}


@pytest.mark.parametrize("group", sorted(_GROUP_MUTATIONS))
def test_release_semantic_group_contradiction_is_rejected(m_run, group):
    path = m_run["releases"]["stage_a"] / "meta.json"
    meta = read_json(path)
    _GROUP_MUTATIONS[group](meta)
    path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


def test_every_mutable_group_has_a_contradiction_test():
    """Groups without a mutation are ones the frozen physical validator already forces."""
    structural = {
        "release_byte_order_profile",  # comes from the plan profile, covered by R2-B
        "release_shard_inventory_and_digests",  # forced by validate_shard_release
        "validation_completion_claims",  # forced by validate_shard_release
    }
    covered = set(_GROUP_MUTATIONS) | structural
    assert set(M_RELEASE_SEMANTIC_GROUPS) - covered == set()


# ------------------------------------------------------------------ R2-D native required


def test_native_required_fields_include_the_previously_missing_ones():
    assert "tokenizer_release" in rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS
    assert "native_shared_authority_validated" in rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS
    assert "native_post_merge_data_branch_identity_sha256" in rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS
    assert "shared_exclusion_authority" in rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS
    assert "reference_validation" in rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS


@pytest.mark.parametrize("field", list(rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS))
def test_missing_native_required_field_is_rejected(native_e2e, field):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance.pop(field, None)
    with pytest.raises(RuntimeError):
        rpc.validate_native_provenance_object(provenance)


def test_shared_authority_flag_cannot_stand_alone(native_e2e):
    provenance = json.loads(json.dumps(native_plan(native_e2e)["release_provenance"]))
    assert provenance["native_shared_authority_validated"] is True
    provenance["shared_exclusion_authority"]["manifest_sha256s"] = ["0" * 64]
    with pytest.raises(RuntimeError, match="post-merge data-branch identity"):
        rpc.validate_native_provenance_object(provenance)


def test_shared_authority_flag_false_is_rejected(native_e2e):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance["native_shared_authority_validated"] = False
    with pytest.raises(RuntimeError, match="native_shared_authority_validated"):
        rpc.validate_native_provenance_object(provenance)


# ------------------------------------------------------------------ R2-E identities


def test_the_two_identities_are_distinct_and_truthfully_named(native_e2e):
    provenance = native_plan(native_e2e)["release_provenance"]
    pre = provenance["premerge_native_chain_identity_sha256"]
    post = provenance["native_post_merge_data_branch_identity_sha256"]
    assert pre != post
    assert native.PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA.endswith("premerge-chain-identity-v1")
    assert native.POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA.endswith(
        "post-merge-data-branch-identity-v1"
    )


def test_post_merge_identity_covers_the_shared_authorities(native_e2e):
    provenance = json.loads(json.dumps(native_plan(native_e2e)["release_provenance"]))
    base = rpc.post_merge_data_branch_identity_sha256(provenance)
    for field in ("reference_validation", "tokenizer_release", "shared_exclusion_authority"):
        mutated = json.loads(json.dumps(provenance))
        mutated[field] = {"tampered": True}
        assert rpc.post_merge_data_branch_identity_sha256(mutated) != base, field
    # The pre-merge identity deliberately does NOT cover them.
    for field in ("reference_validation", "tokenizer_release"):
        mutated = json.loads(json.dumps(provenance))
        mutated[field] = {"tampered": True}
        assert rpc.premerge_native_chain_identity_sha256(
            mutated
        ) == rpc.premerge_native_chain_identity_sha256(provenance)


def test_wrong_post_merge_identity_is_rejected(native_e2e):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance["native_post_merge_data_branch_identity_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="post-merge data-branch identity"):
        rpc.validate_native_provenance_object(provenance)


def test_producer_consumer_projection_drift_is_detected(native_e2e, monkeypatch):
    provenance = native_plan(native_e2e)["release_provenance"]
    assert native.post_merge_data_branch_identity_sha256(
        provenance
    ) == rpc.post_merge_data_branch_identity_sha256(provenance)
    # Drift the consumer's field list and prove the two no longer agree.
    monkeypatch.setattr(
        rpc,
        "POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS",
        rpc.POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS[:-1],
    )
    assert native.post_merge_data_branch_identity_sha256(
        provenance
    ) != rpc.post_merge_data_branch_identity_sha256(provenance)


def test_producer_consumer_stage_projection_drift_is_detected(native_e2e, monkeypatch):
    provenance = native_plan(native_e2e)["release_provenance"]
    monkeypatch.setattr(rpc, "POST_MERGE_STAGE_FIELDS", rpc.POST_MERGE_STAGE_FIELDS[:-1])
    assert native.post_merge_data_branch_identity_sha256(
        provenance
    ) != rpc.post_merge_data_branch_identity_sha256(provenance)


def test_projection_definitions_match_exactly():
    assert rpc.POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS == (
        native.POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS
    )
    assert rpc.POST_MERGE_STAGE_FIELDS == native.POST_MERGE_STAGE_FIELDS
    assert rpc.POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA == (
        native.POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA
    )


# ------------------------------------------------------------------ R2-F publication


def _staging(tmp_path, name="staging"):
    root = tmp_path / name
    (root / "train").mkdir(parents=True)
    (root / "train" / "shard_00000.bin").write_bytes(b"\x01\x00")
    return root


def test_normal_publish_succeeds(tmp_path):
    published = output.publish_release_atomic(_staging(tmp_path), tmp_path / "release")
    assert (published / "train" / "shard_00000.bin").exists()


def test_pre_existing_destination_fails(tmp_path):
    (tmp_path / "release").mkdir()
    with pytest.raises(output.StageMOutputError, match="refusing to replace"):
        output.publish_release_atomic(_staging(tmp_path), tmp_path / "release")


def test_destination_created_at_the_publication_boundary_is_untouched(tmp_path, monkeypatch):
    """The race the old exists()+rename could lose: an EMPTY destination appearing late."""
    destination = tmp_path / "release"
    staging = _staging(tmp_path)
    real = output.fsync_dir

    def late(path):
        real(path)
        if path == staging and not destination.exists():
            destination.mkdir()
            (destination / ".sentinel").write_text("pre-existing")

    monkeypatch.setattr(output, "fsync_dir", late)
    with pytest.raises(output.StageMOutputError, match="refusing to replace"):
        output.publish_release_atomic(staging, destination)
    assert (destination / ".sentinel").read_text() == "pre-existing"
    assert (staging / "train" / "shard_00000.bin").exists()


def test_plain_rename_would_have_replaced_it(tmp_path):
    """Guard the guard: prove the defect the no-replace primitive removes was real."""
    import os

    source = tmp_path / "s"
    source.mkdir()
    destination = tmp_path / "d"
    destination.mkdir()
    os.rename(str(source), str(destination))
    assert destination.is_dir() and not source.exists()


def test_no_check_then_act_remains_in_publication():
    import inspect

    body = inspect.getsource(output.publish_release_atomic)
    assert "destination.exists()" not in body
    assert "os.rename(" not in body
    assert "_renameat2_noreplace(" in body


def test_fsync_order_still_correct(tmp_path, monkeypatch):
    synced: list[str] = []
    real = output.fsync_dir
    monkeypatch.setattr(output, "fsync_dir", lambda p: (synced.append(str(p)), real(p))[1])
    staging = _staging(tmp_path)
    destination = tmp_path / "release"
    output.publish_release_atomic(staging, destination)
    assert synced[0] == str(staging / "train")
    assert synced[1] == str(staging)
    assert synced[-1] == str(destination.parent)


def test_unsupported_primitive_is_a_controlled_error(tmp_path, monkeypatch):
    import ctypes
    import errno

    class FakeLib:
        def __getattr__(self, name):
            if name == "renameat2":
                raise AttributeError(name)
            raise AttributeError(name)

    monkeypatch.setattr(output.platform, "machine", lambda: "s390x")
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: FakeLib())
    with pytest.raises(output.AtomicPublicationUnsupported, match="no atomic no-replace"):
        output.publish_release_atomic(_staging(tmp_path), tmp_path / "release")
    assert errno.EEXIST  # keep the import meaningful


# ------------------------------------------------------------------ frozen-contract pins


def test_frozen_profile_declares_byte_order():
    assert RELEASE_PROFILE["storage_byte_order"] == "little"
    assert RELEASE_PROFILE["storage_dtype_explicit"] == "<u2"
    assert PACKING_SEMANTICS["stride"] == MODEL_CONTRACT["seq_len"] == 2048
    assert PACKING_SEMANTICS["read_length"] == 2049
    assert PACKING_SEMANTICS["padding"] == "none"


def test_plan_exclusion_authority_is_extracted_canonically(m_run):
    authority = plan_exclusion_authority(_plan(m_run), m_run["tmp_path"])
    assert authority["schema_version"] == contract.CANONICAL_EXCLUSION_AUTHORITY_SCHEMA
    assert authority["participant"] == "candidate_m_plan"
    assert authority["artifact_sha256"] == m_run["canonical_exclusion"]["artifact_sha256"]
    assert authority["derived_count"] == m_run["canonical_exclusion"]["derived_count"]
