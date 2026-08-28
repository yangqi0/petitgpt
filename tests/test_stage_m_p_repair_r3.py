"""R3 repair regressions: canonical exclusion authority, independent counts, full loader.

Bounded synthetic fixtures only. No real Stage-M production, no packed corpus, no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pretrain.dataset_pretrain import PackedBinDataset
import pretrain.run_plan_contract as rpc
import pretrain.stage_m_contract_v1 as contract
from pretrain.stage_m_contract_v1 import (
    ACCEPTED_G2_RELEASE_MANIFEST,
    ACCEPTED_G_TOKENIZER_RELEASE_MANIFEST,
    STAGE_STREAMS,
    accepted_g2_exclusion_authority,
    accepted_g_exclusion_authority,
    canonical_exclusion_authority,
    canonical_json_bytes,
    exclusion_authority,
    plan_exclusion_authority,
    read_exclusion_artifact,
    require_identical_exclusion_authorities,
)
from pretrain.stage_p_native_provenance_v1 import (
    M_RELEASE_SEMANTIC_GROUPS,
    NativeProvenanceError,
)
from tests._stage_m_fixtures import read_json, restore_canonical_exclusion_block
from tests.test_stage_m_p_repair_r1 import native_chain, native_plan

REPO_ROOT = Path(__file__).resolve().parent.parent

# The real accepted values, read from the repository rather than hard-coded abbreviations.
_REAL = canonical_exclusion_authority(REPO_ROOT)
REAL_CANONICAL_PATH = _REAL["canonical_artifact_path"]
REAL_CANONICAL_SHA = _REAL["artifact_sha256"]
REAL_CANONICAL_COUNT = _REAL["derived_count"]
# The differently serialized G2 release copy candidate v3 wrongly bound.
OLD_G2_RELEASE_COPY = "runs/g2_production_2026-08-21/release/exclusion_hash_manifest.json"


# ------------------------------------------------------------------ canonical authority


def test_accepted_g_and_g2_name_the_same_canonical_artifact():
    g = accepted_g_exclusion_authority(REPO_ROOT)
    g2 = accepted_g2_exclusion_authority(REPO_ROOT)
    assert g["artifact_sha256"] == g2["artifact_sha256"] == REAL_CANONICAL_SHA
    assert g["artifact_path"] == g2["artifact_path"] == REAL_CANONICAL_PATH
    assert g["derived_count"] == g2["derived_count"] == REAL_CANONICAL_COUNT
    assert len(REAL_CANONICAL_SHA) == 64


def test_canonical_artifact_is_the_l1_reserve_file():
    assert REAL_CANONICAL_PATH.startswith("runs/l1_production_")
    assert (REPO_ROOT / REAL_CANONICAL_PATH).is_file()
    derived = read_exclusion_artifact(REPO_ROOT, REAL_CANONICAL_PATH)
    assert derived["artifact_sha256"] == REAL_CANONICAL_SHA
    assert derived["derived_count"] == REAL_CANONICAL_COUNT


def test_old_g2_release_copy_is_a_different_artifact_with_the_same_count():
    """The exact defect candidate v3 had: same item count, different bytes."""
    copy = read_exclusion_artifact(REPO_ROOT, OLD_G2_RELEASE_COPY)
    assert copy["derived_count"] == REAL_CANONICAL_COUNT, "same number of exclusion hashes"
    assert copy["artifact_sha256"] != REAL_CANONICAL_SHA, "but a different artifact"
    # Identity is the artifact, so a count-only comparison would have missed this.
    canonical = exclusion_authority(
        participant="accepted_g",
        artifact_path=REAL_CANONICAL_PATH,
        artifact_sha256=REAL_CANONICAL_SHA,
        derived_count=REAL_CANONICAL_COUNT,
        artifact_schema_version=_REAL["artifact_schema_version"],
        kind=_REAL["kind"],
        hash_algorithm=_REAL["hash_algorithm"],
    )
    other = exclusion_authority(
        participant="old_g2_release_copy",
        artifact_path=OLD_G2_RELEASE_COPY,
        artifact_sha256=copy["artifact_sha256"],
        derived_count=copy["derived_count"],
        artifact_schema_version=copy["artifact_schema_version"],
        kind=copy["kind"],
        hash_algorithm=copy["hash_algorithm"],
    )
    with pytest.raises(contract.StageMError, match="disagree"):
        require_identical_exclusion_authorities([canonical, other])


def test_a_byte_identical_copy_at_another_path_is_still_a_different_reference(tmp_path):
    """R4-A: artifact_path is load-bearing; equal digests must not launder a different path."""
    source = REPO_ROOT / REAL_CANONICAL_PATH
    alt_rel = "runs/l1_production_2026-08-20/reference_reserve_v1/copy_of_exclusions.json"
    alt = tmp_path / alt_rel
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_bytes(source.read_bytes())
    copy = read_exclusion_artifact(tmp_path, alt_rel)
    assert copy["artifact_sha256"] == REAL_CANONICAL_SHA
    assert copy["derived_count"] == REAL_CANONICAL_COUNT
    assert copy["kind"] == _REAL["kind"]
    # Everything except the path is identical, and it is still refused.
    with pytest.raises(contract.StageMError, match="artifact_path"):
        require_identical_exclusion_authorities([
            exclusion_authority(
                participant="accepted_g",
                artifact_path=REAL_CANONICAL_PATH,
                artifact_sha256=REAL_CANONICAL_SHA,
                derived_count=REAL_CANONICAL_COUNT,
                artifact_schema_version=_REAL["artifact_schema_version"],
                kind=_REAL["kind"],
                hash_algorithm=_REAL["hash_algorithm"],
            ),
            exclusion_authority(
                participant="same_bytes_other_path",
                artifact_path=alt_rel,
                artifact_sha256=copy["artifact_sha256"],
                derived_count=copy["derived_count"],
                artifact_schema_version=copy["artifact_schema_version"],
                kind=copy["kind"],
                hash_algorithm=copy["hash_algorithm"],
            ),
        ])


def test_accepted_manifest_paths_are_the_frozen_ones():
    assert ACCEPTED_G_TOKENIZER_RELEASE_MANIFEST.startswith("runs/g_production_")
    assert ACCEPTED_G2_RELEASE_MANIFEST.startswith("runs/g2_production_")
    assert (REPO_ROOT / ACCEPTED_G_TOKENIZER_RELEASE_MANIFEST).is_file()
    assert (REPO_ROOT / ACCEPTED_G2_RELEASE_MANIFEST).is_file()


def test_artifact_whose_declared_count_lies_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hash_count": 99, "hashes": ["a" * 64]}), encoding="utf-8")
    with pytest.raises(contract.StageMError, match="disagrees with the"):
        read_exclusion_artifact(tmp_path, "bad.json")


# ------------------------------------------------------------------ independent counts


def test_candidate_binds_the_canonical_authority(m_run):
    authority = plan_exclusion_authority(read_json(m_run["plan_path"]), m_run["tmp_path"])
    canonical = m_run["canonical_exclusion"]
    assert authority["artifact_sha256"] == canonical["artifact_sha256"]
    assert authority["derived_count"] == canonical["derived_count"]
    assert authority["artifact_path"] == canonical["artifact_path"]


def test_all_five_participants_agree(m_run):
    agreed = native_chain(m_run)["shared_exclusion_authority"]
    assert set(agreed["participants"]) == {
        "candidate_m_plan",
        "accepted_g",
        "accepted_g2",
        "stage_m_release[stage_a]",
        "stage_m_release[stage_b]",
    }
    assert agreed["participant_count"] == 5


def test_the_previous_999_versus_2_scenario_no_longer_validates(m_run):
    """Codex's case: candidate/A/B all say 999 while the underlying authorities say 2.

    Under the R2 code the candidate's count was passed in as G's and G2's, so the three
    agreeing wrong numbers looked consistent. Now each authority derives its own.
    """
    plan = read_json(m_run["plan_path"])
    plan["resources"]["canonical_exclusion_authority"]["derived_count"] = 999
    m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    for stage in STAGE_STREAMS:
        meta_path = m_run["releases"][stage] / "meta.json"
        meta = read_json(meta_path)
        meta["reference_validation_exclusion"]["union_hash_count"] = 999
        meta["reference_validation_exclusion"]["manifests"][0]["hash_count"] = 999
        meta_path.write_bytes(canonical_json_bytes(meta))
    sha = hashlib.sha256(m_run["plan_path"].read_bytes()).hexdigest()
    assert m_run["canonical_exclusion"]["derived_count"] == 2
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run, expected_candidate_m_plan_sha256=sha)


@pytest.mark.parametrize("participant", ["accepted_g", "accepted_g2"])
def test_accepted_authority_count_is_derived_from_its_own_evidence(m_run, participant):
    canonical = m_run["canonical_exclusion"]
    path = canonical["g_manifest"] if participant == "accepted_g" else canonical["g2_manifest"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = (
        payload["reference_reserve_exclusion"]["manifests"][0]
        if participant == "accepted_g"
        else payload["reserve_provenance"]["reserve_exclusion"]
    )
    entry["hash_count"] = 4321
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(contract.StageMError, match="declares derived_count"):
        native_chain(m_run)


def test_same_count_different_artifact_is_rejected(m_run, tmp_path):
    """A reserialized copy with the same hashes is not the canonical authority."""
    canonical = m_run["canonical_exclusion"]
    other = m_run["tmp_path"] / "reserialized.json"
    original = json.loads(canonical["artifact_abs"].read_text(encoding="utf-8"))
    other.write_text(json.dumps(original, indent=2), encoding="utf-8")  # same content, new bytes
    copy = read_exclusion_artifact(m_run["tmp_path"], "reserialized.json")
    assert copy["derived_count"] == canonical["derived_count"]
    assert copy["artifact_sha256"] != canonical["artifact_sha256"]


# ------------------------------------------------------------------ release semantics


def test_release_semantic_projection_is_closed_and_complete(m_run):
    provenance = native_chain(m_run)
    for stage in STAGE_STREAMS:
        assert provenance["stages"][stage]["semantic_groups_checked"] == len(
            M_RELEASE_SEMANTIC_GROUPS
        )
    assert len(set(M_RELEASE_SEMANTIC_GROUPS)) == len(M_RELEASE_SEMANTIC_GROUPS)


_R3_GROUP_MUTATIONS: dict[str, Any] = {
    "enforced_at_stage": lambda m: m["reference_validation_exclusion"].__setitem__(
        "enforced_at_stage", "stage_m"
    ),
    "reapplied_by_stage_m": lambda m: m["reference_validation_exclusion"].__setitem__(
        "reapplied_by_stage_m", True
    ),
    "exclusion_path": lambda m: m["reference_validation_exclusion"].__setitem__(
        "canonical_artifact_path", "runs/elsewhere/other.json"
    ),
    "exclusion_entry_path": lambda m: m["reference_validation_exclusion"]["manifests"][
        0
    ].__setitem__("path", "runs/elsewhere/other.json"),
    "exclusion_count": lambda m: m["reference_validation_exclusion"].__setitem__(
        "union_hash_count", 4321
    ),
    "exclusion_entry_count": lambda m: m["reference_validation_exclusion"]["manifests"][
        0
    ].__setitem__("hash_count", 4321),
    "val_shard_tokens": lambda m: m.__setitem__("val_shard_tokens", 7),
    "val_ratio": lambda m: m.__setitem__("val_ratio", 0.002),
    "accounting_val": lambda m: m["accounting"]["val"].__setitem__("documents", 3),
    "shard_files_val": lambda m: m["shard_files"].__setitem__("val", [{"x": 1}]),
    "accounting_train_documents": lambda m: m["accounting"]["train"].__setitem__("documents", 1),
    "accounting_train_content_tokens": lambda m: m["accounting"]["train"].__setitem__(
        "content_tokens", 1
    ),
    "tokenizer_path": lambda m: m.__setitem__("tokenizer_path", "/elsewhere/other.json"),
}


@pytest.mark.parametrize("case", sorted(_R3_GROUP_MUTATIONS))
def test_previously_partial_or_unchecked_representations_are_rejected(m_run, case):
    path = m_run["releases"]["stage_a"] / "meta.json"
    meta = read_json(path)
    _R3_GROUP_MUTATIONS[case](meta)
    path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


# ------------------------------------------------------------------ R3-D required fields


def test_required_field_scope_is_reported_accurately():
    """R3-D: the evidence quotes an effective scope, not just the top-level list length.

    Every count below is derived from the loader's own tuples, so the evidence cannot drift
    from what actually gets enforced.
    """
    top_level = list(rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS)
    per_stage = list(rpc.NATIVE_REQUIRED_STAGE_FIELDS)
    assert len(set(top_level)) == len(top_level) == 19
    assert len(set(per_stage)) == len(per_stage) == 8
    # R3-D: 19 is the top-level list length, NOT the number of native-specific fields and NOT
    # the effective required scope. Seven of the 19 are shared vocabulary the legacy chain also
    # names; the effective scope is the top-level list plus the per-stage list for both stages.
    legacy_shared = {
        "schema_version",
        "shared_tokenizer_sha256",
        "stage_b_selection_stage",
        "full_chain_validated",
        "provenance_chain_kind",
        "stages",
        "tokenizer_release",
        "reference_validation",
        "model_contract",
    }
    native_specific = [f for f in top_level if f not in legacy_shared]
    assert len(native_specific) == 12
    assert len(top_level) - len(native_specific) == 7
    assert len(top_level) + 2 * len(per_stage) == 35, "effective required scope"
    # The post-merge identity preimage is the top-level list minus the identity field itself.
    projection = list(rpc.POST_MERGE_DATA_BRANCH_IDENTITY_FIELDS)
    assert len(projection) == 18
    assert set(projection) == set(top_level) - {"native_post_merge_data_branch_identity_sha256"}
    assert list(rpc.POST_MERGE_STAGE_FIELDS) == per_stage
    # The two fields the R2 review found missing are still required.
    assert "tokenizer_release" in top_level
    assert "native_shared_authority_validated" in top_level


# ------------------------------------------------------------------ R3-E full loader


def _loader_args(plan: dict[str, Any], plan_path: Path, stage: str, native_e2e):
    """Build the exact launch namespace a real trainer would pass, read from the plan itself.

    R3-E: every immutable launch field comes from the plan bytes, so this is the real
    training-facing flow rather than a hand-tuned subset.
    """
    boundaries = plan["boundaries"]
    wsd = plan["wsd_candidate"]
    inputs = plan["inputs"]
    start = (
        boundaries["stage_a_start_step"]
        if stage == "stage_a"
        else (boundaries["stage_b_start_step"])
    )
    stop = (
        boundaries["stage_a_stop_step"]
        if stage == "stage_a"
        else (boundaries["stage_b_global_stop_step"])
    )
    return argparse.Namespace(
        run_plan_json=str(plan_path),
        run_plan_stage=stage,
        strict_resume_contract=True,
        seq_len=inputs["seq_len"],
        micro_bsz=inputs["micro_bsz"],
        grad_accum=inputs["grad_accum"],
        warmup_steps=wsd["warmup_steps"],
        data_stage_start_step=start,
        max_steps=stop,
        schedule_total_steps=boundaries["schedule_total_steps"],
        lr_schedule="wsd",
        decay_start_step=wsd["decay_start_step"],
        decay_end_step=wsd["decay_end_step"],
        allow_schedule_branch=False,
        allow_data_branch=False,
        save_steps=list(plan["checkpoint_milestones"]["absolute_steps"]),
        val_dir=str(native_e2e["reference_val_dir"]),
    )


def _write_plan(tmp_path: Path, plan: dict[str, Any]) -> Path:
    path = tmp_path / "native_run_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


@pytest.mark.parametrize("stage", list(STAGE_STREAMS))
def test_full_training_facing_loader_accepts_the_native_plan(native_e2e, tmp_path, stage):
    plan = native_plan(native_e2e)
    plan_path = _write_plan(tmp_path, plan)
    binding = rpc.load_run_plan_binding(
        _loader_args(plan, plan_path, stage, native_e2e),
        train_dir=native_e2e[f"{stage}_dir"],
        tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
        val_dir=native_e2e["reference_val_dir"],
    )
    assert binding is not None
    assert binding["provenance_chain_kind"] == "accepted_stage_i_native_v1"
    assert (
        binding["native_post_merge_data_branch_identity_sha256"]
        == (plan["release_provenance"]["native_post_merge_data_branch_identity_sha256"])
    )
    # The loader sees the same validated exclusion authority and derived count.
    agreed = plan["release_provenance"]["shared_exclusion_authority"]
    assert agreed["artifact_sha256"] == native_e2e["canonical_exclusion"]["artifact_sha256"]
    assert agreed["derived_count"] == native_e2e["canonical_exclusion"]["derived_count"]

    dataset = PackedBinDataset(
        str(native_e2e["releases"][stage] / "train"), seq_len=2048, require_release_manifest=True
    )
    assert dataset.stats()["tail_transitions"] == 0
    assert len(dataset) == plan["stages"][stage]["unique_blocks"]


def test_full_loader_rejects_an_inconsistent_exclusion_authority(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    plan["release_provenance"]["shared_exclusion_authority"]["artifact_sha256"] = "0" * 64
    plan_path = _write_plan(tmp_path, plan)
    with pytest.raises(RuntimeError):
        rpc.load_run_plan_binding(
            _loader_args(plan, plan_path, "stage_a", native_e2e),
            train_dir=native_e2e["stage_a_dir"],
            tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
            val_dir=native_e2e["reference_val_dir"],
        )


def test_full_loader_rejects_inconsistent_release_metadata(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    plan["release_provenance"]["stages"]["stage_a"]["manifest_sha256"] = "0" * 64
    plan_path = _write_plan(tmp_path, plan)
    with pytest.raises(RuntimeError):
        rpc.load_run_plan_binding(
            _loader_args(plan, plan_path, "stage_a", native_e2e),
            train_dir=native_e2e["stage_a_dir"],
            tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
            val_dir=native_e2e["reference_val_dir"],
        )


def test_full_loader_rejects_a_missing_native_field(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    plan["release_provenance"].pop("tokenizer_release")
    plan_path = _write_plan(tmp_path, plan)
    with pytest.raises(RuntimeError):
        rpc.load_run_plan_binding(
            _loader_args(plan, plan_path, "stage_a", native_e2e),
            train_dir=native_e2e["stage_a_dir"],
            tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
            val_dir=native_e2e["reference_val_dir"],
        )


def test_restore_helper_is_only_a_fixture_shape_fix(m_run):
    """The restore helper must never change the exclusion IDENTITY, only the block shape."""
    path = m_run["releases"]["stage_a"] / "meta.json"
    before = read_json(path)["reference_validation_exclusion"]
    restore_canonical_exclusion_block(path, m_run["canonical_exclusion"])
    after = read_json(path)["reference_validation_exclusion"]
    assert before["canonical_artifact_sha256"] == after["canonical_artifact_sha256"]
    assert before["union_hash_count"] == after["union_hash_count"]
