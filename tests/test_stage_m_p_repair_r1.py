"""R1 repair regressions for the Codex findings on the Stage-M / Stage-P native tooling.

Bounded synthetic fixtures only. No real Stage-M production, no real packed corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pretrain.dataset_pretrain import PackedBinDataset
from pretrain.plan_pretrain_run import build_run_plan
import pretrain.run_plan_contract as rpc
import pretrain.stage_m_contract_v1 as contract
from pretrain.stage_m_contract_v1 import (
    CANDIDATE_PLAN_SCHEMA,
    M_IMPLEMENTATION_BUNDLE_FILES,
    REQUIRED_BYTE_ORDER,
    STAGE_STREAMS,
    canonical_json_bytes,
    m_implementation_bundle,
)
from pretrain.stage_m_input_v1 import (
    StageIInputError,
    derive_input_sequence_commitments,
    inspect_accepted_stage_i_metadata_only,
    iter_accepted_records,
    load_accepted_stage_i,
)
import pretrain.stage_m_output_v1 as output
import pretrain.stage_m_realize_v1 as realize
from pretrain.stage_p_native_provenance_v1 import (
    LEGACY_CHAIN_KIND,
    NATIVE_CHAIN_KIND,
    NativeProvenanceError,
    validate_native_chain,
)
from tests._stage_m_fixtures import (
    read_json,
    restore_canonical_exclusion_block,
    same_size_mutation,
)
from tests.test_plan_pretrain_run import _write_full_provenance

REPO_ROOT = Path(__file__).resolve().parent.parent


def native_chain(m_run, **overrides):
    """Run validate_native_chain against a bounded m_run fixture."""
    kwargs = {
        # The fixture writes its own accepted G/G2/L1 set, so the chain resolves them there.
        "repo_root": m_run["tmp_path"],
        "accepted_stage_i_dir": m_run["accepted_dir"],
        "candidate_m_plan": m_run["plan_path"],
        "expected_candidate_m_plan_sha256": m_run["plan_sha"],
        "stage_releases": dict(m_run["releases"]),
    }
    kwargs.update(overrides)
    return validate_native_chain(**kwargs)


def native_plan(native_e2e, **overrides):
    """Build a native run plan through the real planner entrypoint.

    The accepted-authority root is redirected to the fixture's synthetic accepted G/G2/L1 set
    for the duration of the call, and each release's canonical exclusion block is restored
    first because building the legacy G/G2 fixture overwrites it as a side effect.
    """
    import pretrain.plan_pretrain_run as planner

    for stage in ("stage_a", "stage_b"):
        restore_canonical_exclusion_block(
            native_e2e["releases"][stage] / "meta.json", native_e2e["canonical_exclusion"]
        )
    original_root = planner.native_provenance_repo_root
    planner.native_provenance_repo_root = lambda: Path(native_e2e["tmp_path"]).resolve()
    try:
        return _native_plan_inner(native_e2e, **overrides)
    finally:
        planner.native_provenance_repo_root = original_root


def _native_plan_inner(native_e2e, **overrides):
    kwargs: dict[str, Any] = {
        "stage_a_dir": native_e2e["stage_a_dir"],
        "stage_b_dir": native_e2e["stage_b_dir"],
        "seq_len": 2048,
        "micro_bsz": 1,
        "grad_accum": 1,
        "warmup_steps": 1,
        "decay_fraction": 0.5,
        "provenance_chain_kind": NATIVE_CHAIN_KIND,
        "accepted_stage_i_dir": native_e2e["accepted_dir"],
        "candidate_m_plan": native_e2e["plan_path"],
        "expected_candidate_m_plan_sha256": native_e2e["plan_sha"],
        "reference_val_dir": native_e2e["reference_val_dir"],
        "tokenizer_release_manifest": native_e2e["tokenizer_release_manifest"],
    }
    kwargs.update(overrides)
    return build_run_plan(**kwargs)


# ------------------------------------------------------------------ R1-A physical hashing


def test_production_loader_always_hashes_every_shard(accepted):
    accepted_dir, _recs = accepted
    binding = load_accepted_stage_i(accepted_dir)
    assert binding.shard_bytes_verified is True
    assert binding.shard_count == len(binding.shard_inventory)


def test_production_loader_has_no_verify_switch():
    """R1-A: the boolean whose default could silently weaken production no longer exists."""
    import inspect

    assert "verify_shard_bytes" not in inspect.signature(load_accepted_stage_i).parameters
    assert "verify_shard_bytes" not in inspect.signature(iter_accepted_records).parameters
    assert (
        "verify_shard_bytes" not in inspect.signature(derive_input_sequence_commitments).parameters
    )


def test_same_size_shard_byte_change_is_rejected(accepted):
    accepted_dir, _recs = accepted
    binding = load_accepted_stage_i(accepted_dir)
    shard = accepted_dir / "documents" / str(binding.shard_inventory[0]["name"])
    before = shard.stat().st_size
    same_size_mutation(shard)
    assert shard.stat().st_size == before, "the fixture must not change the file length"
    with pytest.raises(StageIInputError, match="SHA-256 mismatch"):
        load_accepted_stage_i(accepted_dir)


def test_fixed_plan_rejects_a_same_size_shard_change(m_run):
    shard_dir = m_run["accepted_dir"] / "documents"
    shard = sorted(shard_dir.glob("*.jsonl"))[0]
    before = shard.stat().st_size
    same_size_mutation(shard)
    assert shard.stat().st_size == before
    with pytest.raises(StageIInputError, match="SHA-256 mismatch"):
        realize.authorize_plan(m_run["plan_path"], m_run["plan_sha"], m_run["tmp_path"].resolve())


def test_native_chain_cannot_validate_after_a_same_size_shard_change(m_run):
    shard = sorted((m_run["accepted_dir"] / "documents").glob("*.jsonl"))[0]
    same_size_mutation(shard)
    with pytest.raises((StageIInputError, NativeProvenanceError)):
        native_chain(m_run)


@pytest.mark.parametrize(
    "mutation",
    ["different_size", "replacement", "omission", "extra", "path_change", "digest_mismatch"],
)
def test_shard_inventory_mutations_are_rejected(accepted, mutation):
    accepted_dir, _recs = accepted
    binding = load_accepted_stage_i(accepted_dir)
    documents = accepted_dir / "documents"
    first = documents / str(binding.shard_inventory[0]["name"])
    second = documents / str(binding.shard_inventory[1]["name"])
    if mutation == "different_size":
        first.write_bytes(first.read_bytes() + b"\n")
    elif mutation == "replacement":
        first.write_bytes(second.read_bytes())
    elif mutation == "omission":
        first.unlink()
    elif mutation == "extra":
        (documents / "documents-99999.jsonl").write_text("{}\n")
    elif mutation == "path_change":
        first.rename(documents / "documents-88888.jsonl")
    else:
        manifest = read_json(accepted_dir / "manifest.json")
        manifest["shards"][0]["sha256"] = "0" * 64
        (accepted_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(StageIInputError):
        load_accepted_stage_i(accepted_dir)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_manifest_sha256", "1" * 64),
        ("expected_completion_sha256", "2" * 64),
        ("expected_run_identity", "3" * 64),
    ],
)
def test_top_level_identity_mismatches_are_rejected(accepted, field, value):
    accepted_dir, _recs = accepted
    with pytest.raises(StageIInputError):
        load_accepted_stage_i(accepted_dir, **{field: value})


def test_diagnostic_binding_cannot_reach_production(accepted):
    accepted_dir, _recs = accepted
    diagnostic = inspect_accepted_stage_i_metadata_only(accepted_dir)
    assert diagnostic.shard_bytes_verified is False
    with pytest.raises(StageIInputError, match="not physically verified"):
        diagnostic.require_physically_verified("test")
    with pytest.raises(StageIInputError, match="not physically verified"):
        list(iter_accepted_records(diagnostic))
    with pytest.raises(StageIInputError, match="not physically verified"):
        derive_input_sequence_commitments(diagnostic)


def test_diagnostic_binding_still_catches_metadata_faults(accepted):
    accepted_dir, _recs = accepted
    with pytest.raises(StageIInputError):
        inspect_accepted_stage_i_metadata_only(accepted_dir, expected_records=1)


# ------------------------------------------------------------------ R1-A commitment re-derivation


def test_runtime_commitment_rederivation_reproduces_the_plan(m_run):
    proof = realize.rederive_and_prove_input_commitments(m_run["context"])
    plan = read_json(m_run["plan_path"])
    for stage in STAGE_STREAMS:
        assert (
            proof[stage]["commitment"] == plan["stage_streams"][stage]["input_sequence_commitment"]
        )
        assert proof[stage]["record_count"] == plan["stage_streams"][stage]["input_record_count"]
        assert (
            proof[stage]["serialized_tokens"]
            == plan["stage_streams"][stage]["input_serialized_tokens"]
        )


def test_runtime_commitment_rederivation_is_not_a_self_comparison(m_run, monkeypatch):
    """Swapping the derived commitment must fail even though the plan is untouched."""
    real = realize.derive_input_sequence_commitments

    def _wrong(accepted):
        derived = real(accepted)
        derived["stage_a"]._sealed = "0" * 64  # noqa: SLF001 - deliberate fault injection
        return derived

    monkeypatch.setattr(realize, "derive_input_sequence_commitments", _wrong)
    with pytest.raises(contract.StageMError, match="input_sequence_commitment"):
        realize.rederive_and_prove_input_commitments(m_run["context"])


def test_publication_is_blocked_when_input_changes_after_authorization(m_run, tmp_path):
    shard = sorted((m_run["accepted_dir"] / "documents").glob("*.jsonl"))[0]
    same_size_mutation(shard)
    with pytest.raises(StageIInputError):
        realize.realize_and_publish(m_run["context"], out_dir=tmp_path / "second")
    assert not (tmp_path / "second" / "stage_a" / "meta.json").exists()


# ------------------------------------------------------------------ R1-D M metadata


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_plan_schema", "petitgpt-m-candidate-plan-v0"),
        ("stage_stream_count", 3),
        ("input_sequence_commitment_schema", "something-else-v1"),
        ("ordering_policy", "SOME_OTHER_ORDER"),
        ("implementation_commit", "f" * 40),
    ],
)
def test_contradictory_stage_m_metadata_is_rejected(m_run, field, value):
    meta_path = m_run["releases"]["stage_a"] / "meta.json"
    meta = read_json(meta_path)
    meta["stage_m"][field] = value
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(NativeProvenanceError):
        native_chain(m_run)


@pytest.mark.parametrize("field", ["dtype", "vocab_size", "shard_tokens"])
def test_contradictory_release_profile_metadata_is_rejected(m_run, field):
    meta_path = m_run["releases"]["stage_b"] / "meta.json"
    meta = read_json(meta_path)
    meta[field] = {"dtype": "uint32", "vocab_size": 31_999, "shard_tokens": 12345}[field]
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises((NativeProvenanceError, RuntimeError)):
        native_chain(m_run)


def test_contradictory_accounting_seq_len_is_rejected(m_run):
    meta_path = m_run["releases"]["stage_a"] / "meta.json"
    meta = read_json(meta_path)
    meta["stage_m_accounting"]["seq_len"] = 1024
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(NativeProvenanceError):
        native_chain(m_run)


# ------------------------------------------------------------------ R1-G byte order


def test_byte_order_is_bound_in_the_environment_contract():
    assert contract.REQUIRED_BYTE_ORDER == "little"
    env = contract.current_environment()
    assert env.byte_order == REQUIRED_BYTE_ORDER
    assert "byte_order" in env.as_canonical()


def test_non_little_endian_runtime_is_refused():
    big = contract.Environment(
        python_executable=contract.REQUIRED_PYTHON_EXECUTABLE,
        python_version=contract.REQUIRED_PYTHON_VERSION,
        tokenizers_version=contract.REQUIRED_TOKENIZERS_VERSION,
        numpy_version="2.2.6",
        byte_order="big",
    )
    with pytest.raises(contract.StageMError, match="little-endian"):
        contract.verify_environment(big)


def test_writer_uses_explicit_little_endian_dtype():
    assert output.STORAGE_DTYPE.byteorder in ("<", "=")
    assert output.STORAGE_DTYPE.str == "<u2"
    assert output.STORAGE_DTYPE.itemsize == 2


def test_writer_refuses_a_non_little_endian_host(tmp_path, monkeypatch):
    monkeypatch.setattr(output.sys, "byteorder", "big")
    with pytest.raises(output.StageMOutputError, match="little-endian"):
        output.ShardWriter(directory=tmp_path / "train", shard_tokens=16)


def test_release_profile_records_the_byte_order():
    assert contract.RELEASE_PROFILE["storage_byte_order"] == "little"
    assert contract.RELEASE_PROFILE["storage_dtype_explicit"] == "<u2"


# ------------------------------------------------------------------ R1-F durability


def test_nested_shard_directory_is_fsynced_before_publication(tmp_path, monkeypatch, tok):
    from tests._stage_m_fixtures import framed_ids

    synced: list[str] = []
    real = output.fsync_dir
    monkeypatch.setattr(output, "fsync_dir", lambda p: (synced.append(str(p)), real(p))[1])

    docs = [framed_ids(tok, f"durability {i}") for i in range(30)]
    total = sum(len(d) for d in docs)
    accounting = contract.stream_accounting("stage_a", total, 8)
    staging = tmp_path / "staging"
    output.pack_stream(
        stage="stage_a",
        documents=iter(docs),
        accounting=accounting,
        directory=staging / "train",
        shard_tokens=64,
    )
    assert str(staging / "train") in synced, "nested train/ directory was not fsynced"
    output.write_manifest(staging, {"placeholder": True})
    synced.clear()
    destination = tmp_path / "release"
    output.publish_release_atomic(staging, destination)
    assert synced[0] == str(staging / "train"), "nested directory must be synced first"
    assert synced[1] == str(staging), "staging root must be synced before the rename"
    assert synced[-1] == str(destination.parent), "destination parent must be synced last"


# ------------------------------------------------------------------ native e2e fixtures


@pytest.fixture
def native_e2e(m_run, monkeypatch):
    """Bounded native route: real M releases plus the frozen G/G2 authorities, no selection."""
    tmp_path = m_run["tmp_path"]
    stage_a_train = m_run["releases"]["stage_a"] / "train"
    stage_b_train = m_run["releases"]["stage_b"] / "train"
    provenance = _write_full_provenance(tmp_path, stage_a_train, stage_b_train)

    # Align every authority on the tokenizer the candidate-M plan actually bound.
    tokenizer_bytes = m_run["tokenizer_path"].read_bytes()
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    release_tokenizer = Path(provenance["tokenizer_release_manifest"]).parent / "tokenizer.json"
    release_tokenizer.write_bytes(tokenizer_bytes)
    for path, key in (
        (Path(provenance["tokenizer_release_manifest"]), "tokenizer_sha256"),
        (Path(provenance["reference_val_dir"]).parent / "manifest.json", "tokenizer_sha256"),
        (stage_a_train.parent / "meta.json", "tokenizer_sha256"),
        (stage_b_train.parent / "meta.json", "tokenizer_sha256"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[key] = tokenizer_sha
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda p: None)
    return {
        **m_run,
        "stage_a_dir": stage_a_train,
        "stage_b_dir": stage_b_train,
        "reference_val_dir": provenance["reference_val_dir"],
        "tokenizer_release_manifest": provenance["tokenizer_release_manifest"],
        "selection_manifest": provenance["selection_manifest"],
    }


# ------------------------------------------------------------------ R1-B planner reachability


def test_native_planner_produces_a_valid_plan_with_gg2_inputs(native_e2e):
    plan = native_plan(native_e2e)
    provenance = plan["release_provenance"]
    assert provenance["provenance_chain_kind"] == NATIVE_CHAIN_KIND
    assert provenance["full_chain_validated"] is True
    assert provenance["native_shared_authority_validated"] is True
    assert "selection" not in provenance
    assert "source_bindings" not in provenance
    assert provenance["reference_validation"]["manifest_sha256"]
    assert provenance["tokenizer_release"]["tokenizer_sha256"]
    assert plan["inputs"]["selection_manifest"] is None
    assert plan["inputs"]["reference_val_dir"]


@pytest.mark.parametrize("missing", ["reference_val_dir", "tokenizer_release_manifest"])
def test_native_planner_requires_the_gg2_authorities(native_e2e, missing):
    with pytest.raises(ValueError, match="reference-validation and tokenizer"):
        native_plan(native_e2e, **{missing: None})


def test_native_planner_rejects_a_legacy_selection_manifest(native_e2e):
    with pytest.raises(ValueError, match="must not be given a legacy selector-v1"):
        native_plan(native_e2e, selection_manifest=native_e2e["selection_manifest"])


def test_native_planner_rejects_a_tokenizer_authority_mismatch(native_e2e):
    manifest = Path(native_e2e["reference_val_dir"]).parent / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tokenizer_sha256"] = "9" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="tokenizer SHA-256 values disagree"):
        native_plan(native_e2e)


# ------------------------------------------------------------------ R1-C strict contract


def _strict_args(plan_path: Path, stage: str, native_e2e):
    import argparse

    return argparse.Namespace(
        run_plan_json=str(plan_path),
        run_plan_stage=stage,
        strict_resume_contract=True,
        seq_len=2048,
        micro_bsz=1,
        grad_accum=1,
        warmup_steps=1,
        val_dir=str(native_e2e["reference_val_dir"]),
        allow_data_branch=False,
    )


def test_strict_contract_accepts_a_validnative_plan(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    plan_path = tmp_path / "native_run_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    provenance = plan["release_provenance"]
    identity = rpc.validate_native_provenance_object(provenance)
    assert len(identity) == 64
    assert (
        rpc.premerge_native_chain_identity_sha256(provenance)
        == provenance["premerge_native_chain_identity_sha256"]
    )
    assert identity == provenance["native_post_merge_data_branch_identity_sha256"]


def test_native_data_branch_identity_is_versioned_and_binds_the_authority(native_e2e):
    plan = native_plan(native_e2e)
    provenance = dict(plan["release_provenance"])
    base = rpc.post_merge_data_branch_identity_sha256(provenance)
    for field in (
        "accepted_stage_i_identity_sha256",
        "candidate_m_plan_sha256",
        "shared_tokenizer_sha256",
        "stage_m_implementation_bundle_sha256",
    ):
        mutated = dict(provenance)
        mutated[field] = "0" * 64
        assert rpc.post_merge_data_branch_identity_sha256(mutated) != base, field
    stages = json.loads(json.dumps(provenance["stages"]))
    stages["stage_a"]["input_sequence_commitment"] = "0" * 64
    assert rpc.post_merge_data_branch_identity_sha256({**provenance, "stages": stages}) != base


def test_data_branch_immutable_sha_does_not_require_legacy_objects(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    digest = rpc._data_branch_immutable_sha256(plan)  # noqa: SLF001 - contract surface
    assert len(digest) == 64


def test_native_chain_identity_field_lists_agree_between_modules():
    from pretrain.stage_p_native_provenance_v1 import (
        POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS,
        POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA,
        POST_MERGE_STAGE_FIELDS,
        PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS,
        PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA,
    )

    assert rpc.PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS == PREMERGE_NATIVE_CHAIN_IDENTITY_FIELDS
    assert rpc.PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA == PREMERGE_NATIVE_CHAIN_IDENTITY_SCHEMA
    assert rpc.POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS == POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS
    assert rpc.POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA == POST_MERGE_DATA_BRANCH_IDENTITY_SCHEMA
    assert rpc.POST_MERGE_STAGE_FIELDS == POST_MERGE_STAGE_FIELDS
    assert rpc.NATIVE_PROVENANCE_SCHEMA == (
        "petitgpt-accepted-stage-i-native-release-provenance-v2"
    )


def test_native_chain_identity_is_computed_identically_by_both_modules(native_e2e):
    from pretrain.stage_p_native_provenance_v1 import (
        post_merge_data_branch_identity_sha256 as post_native,
        premerge_native_chain_identity_sha256 as pre_native,
    )

    provenance = native_plan(native_e2e)["release_provenance"]
    assert pre_native(provenance) == rpc.premerge_native_chain_identity_sha256(provenance)
    assert pre_native(provenance) == provenance["premerge_native_chain_identity_sha256"]
    assert post_native(provenance) == rpc.post_merge_data_branch_identity_sha256(provenance)
    assert post_native(provenance) == provenance["native_post_merge_data_branch_identity_sha256"]
    # The two identities are distinct concepts and must not collide.
    assert pre_native(provenance) != post_native(provenance)


def test_canonical_encoders_agree_between_modules():
    payload = {"b": 1, "a": [1, 2, {"z": None}]}
    assert rpc._canonical_json_bytes(payload) == canonical_json_bytes(payload)  # noqa: SLF001


# ------------------------------------------------------------------ R1-E schema / branch


@pytest.mark.parametrize("field", list(rpc.NATIVE_FORBIDDEN_PROVENANCE_FIELDS))
def test_native_plan_carrying_a_legacy_object_is_rejected(native_e2e, field):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance[field] = {"anything": True}
    with pytest.raises(RuntimeError, match="legacy selector-v1 provenance objects"):
        rpc.validate_native_provenance_object(provenance)


@pytest.mark.parametrize("field", list(rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS))
def test_native_plan_missing_a_required_field_is_rejected(native_e2e, field):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance.pop(field, None)
    with pytest.raises(RuntimeError):
        rpc.validate_native_provenance_object(provenance)


@pytest.mark.parametrize("field", list(rpc.NATIVE_REQUIRED_STAGE_FIELDS))
def test_native_plan_missing_a_required_stage_field_is_rejected(native_e2e, field):
    provenance = json.loads(json.dumps(native_plan(native_e2e)["release_provenance"]))
    provenance["stages"]["stage_b"].pop(field, None)
    with pytest.raises(RuntimeError):
        rpc.validate_native_provenance_object(provenance)


def test_native_plan_with_a_wrong_schema_version_is_rejected(native_e2e):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance["schema_version"] = "petitgpt-something-else-v9"
    with pytest.raises(RuntimeError, match="schema_version must be"):
        rpc.validate_native_provenance_object(provenance)


def test_unknown_chain_kind_is_not_reinterpreted_as_legacy():
    assert "made_up" not in rpc.PROVENANCE_CHAIN_KINDS
    with pytest.raises(ValueError, match="provenance_chain_kind must be one of"):
        build_run_plan(
            stage_a_dir="/x",
            stage_b_dir="/y",
            seq_len=2048,
            micro_bsz=1,
            grad_accum=1,
            warmup_steps=1,
            decay_fraction=0.5,
            provenance_chain_kind="made_up",
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_schema",
        "missing_p_bundle",
        "legacy_field",
        "wrong_accepted_digest",
        "wrong_m_metadata",
        "wrong_release_digest",
    ],
)
def test_serialized_full_chain_validated_cannot_substitute_for_validation(native_e2e, corruption):
    provenance = json.loads(json.dumps(native_plan(native_e2e)["release_provenance"]))
    assert provenance["full_chain_validated"] is True
    if corruption == "wrong_schema":
        provenance["schema_version"] = "nope-v1"
    elif corruption == "missing_p_bundle":
        provenance.pop("stage_p_native_validator_bundle_sha256")
    elif corruption == "legacy_field":
        provenance["selection"] = {"manifest_sha256": "a" * 64}
    elif corruption == "wrong_accepted_digest":
        provenance["accepted_stage_i_identity_sha256"] = "0" * 64
    elif corruption == "wrong_m_metadata":
        provenance["stage_m_ordering_policy"] = "SOMETHING_ELSE"
    else:
        provenance["stages"]["stage_a"]["manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        rpc.validate_native_provenance_object(provenance)


def test_full_chain_validated_false_cannot_be_upgraded_by_caller(native_e2e):
    provenance = dict(native_plan(native_e2e)["release_provenance"])
    provenance["full_chain_validated"] = False
    with pytest.raises(RuntimeError, match="full_chain_validated must be true"):
        rpc.validate_native_provenance_object(provenance)


# ------------------------------------------------------------------ end-to-end (section 15)


def test_end_to_end_native_route(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    provenance = plan["release_provenance"]
    assert provenance["full_chain_validated"] is True

    identity = rpc.validate_native_provenance_object(provenance)
    assert len(identity) == 64

    for stage in STAGE_STREAMS:
        dataset = PackedBinDataset(
            str(native_e2e["releases"][stage] / "train"),
            seq_len=2048,
            require_release_manifest=True,
        )
        stats = dataset.stats()
        assert stats["window_size"] == 2049
        assert stats["block_stride"] == 2048
        assert stats["tail_transitions"] == 0
        assert len(dataset) == plan["stages"][stage]["unique_blocks"]


def test_legacy_route_still_works_unchanged(tmp_path, monkeypatch):
    from tests.test_plan_pretrain_run import _build_full_plan, _write_stage

    stage_a = _write_stage(tmp_path, "la", [list(range(17))])
    stage_b = _write_stage(tmp_path, "lb", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda p: None)
    plan = _build_full_plan(stage_a, stage_b, provenance)
    block = plan["release_provenance"]
    assert block["provenance_chain_kind"] == LEGACY_CHAIN_KIND
    assert block["full_chain_validated"] is True
    assert "selection" in block and "source_bindings" in block
    assert rpc._data_branch_immutable_sha256(plan)  # noqa: SLF001


# ------------------------------------------------------------------ closure after R1


def test_m_bundle_closure_is_still_complete_after_r1():
    import ast

    seen: set[str] = set()
    stack = ["pretrain/stage_m_realize_v1.py"]
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for dotted in names:
                parts = dotted.split(".")
                for depth in range(1, len(parts)):
                    init = REPO_ROOT.joinpath(*parts[:depth], "__init__.py")
                    if init.is_file():
                        stack.append(str(init.relative_to(REPO_ROOT)))
                module = REPO_ROOT.joinpath(*parts).with_suffix(".py")
                if module.is_file():
                    stack.append(str(module.relative_to(REPO_ROOT)))
    assert sorted(seen) == sorted(M_IMPLEMENTATION_BUNDLE_FILES)
    files, digest = m_implementation_bundle(REPO_ROOT)
    assert set(files) == set(M_IMPLEMENTATION_BUNDLE_FILES)
    assert len(digest) == 64
    assert CANDIDATE_PLAN_SCHEMA == "petitgpt-m-candidate-plan-v2"


# ------------------------------------------------------------------ section 16 P bundle scope


P_END_TO_END_DEFERRED_FILES = (
    "pretrain/plan_pretrain_run.py",
    "pretrain/run_plan_contract.py",
)


def test_p_helper_bundle_is_helper_scoped_and_does_not_claim_end_to_end():
    """The seven-file helper bundle is truthfully helper-only.

    Section 16: the planner and the strict run-plan contract are load-bearing for end-to-end
    Stage-P plan generation and validation, and are NOT in the helper bundle. That scope gap is
    deferred to the exact Stage-P implementation bundle rather than papered over by widening
    this one.
    """
    from pretrain.stage_m_contract_v1 import P_NATIVE_IMPLEMENTATION_BUNDLE_FILES

    for deferred in P_END_TO_END_DEFERRED_FILES:
        assert deferred not in P_NATIVE_IMPLEMENTATION_BUNDLE_FILES
        assert (REPO_ROOT / deferred).is_file()


def test_deferred_end_to_end_files_are_exercised_now():
    """Deferred from the *bundle*, not from testing: R1 covers both files end to end."""
    source = (REPO_ROOT / "tests" / "test_stage_m_p_repair_r1.py").read_text(encoding="utf-8")
    assert "build_run_plan(" in source
    assert "rpc.validate_native_provenance_object(" in source
    assert "rpc._data_branch_immutable_sha256(" in source
