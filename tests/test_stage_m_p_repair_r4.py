"""R4 repair regressions: closed exclusion sub-schema, literal five-party derivation,
complete release semantics, tokenizer-path binding, P helper bundle schema, loader scope.

Bounded synthetic fixtures only. No real Stage-M production, no packed corpus, no training.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from pretrain.dataset_pretrain import PackedBinDataset
import pretrain.run_plan_contract as rpc
import pretrain.stage_m_contract_v1 as contract
from pretrain.stage_m_contract_v1 import (
    BUNDLE_SCHEMA,
    EXCLUSION_REFERENCE_FIELDS,
    P_NATIVE_HELPER_BUNDLE_SCHEMA,
    P_NATIVE_IMPLEMENTATION_BUNDLE_FILES,
    STAGE_STREAMS,
    bundle_files,
    bundle_sha256,
    canonical_exclusion_authority,
    canonical_json_bytes,
    canonical_tokenizer_authority,
    derive_exclusion_reference,
    m_implementation_bundle,
    p_native_implementation_bundle,
    read_exclusion_artifact,
    validate_candidate_plan_contract,
)
from pretrain.stage_p_native_provenance_v1 import (
    M_RELEASE_SEMANTIC_GROUPS,
    NativeProvenanceError,
)
from tests._stage_m_fixtures import read_json
from tests.test_stage_m_p_repair_r1 import native_chain, native_plan
from tests.test_stage_m_p_repair_r3 import _loader_args, _write_plan

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL = canonical_exclusion_authority(REPO_ROOT)
REAL_TOKENIZER = canonical_tokenizer_authority(REPO_ROOT)


# =============================================================== R4-A closed sub-schema


def test_canonical_exclusion_subschema_is_complete_and_recovered_not_guessed():
    """Every field comes from the accepted artifacts, none from a hard-coded literal."""
    artifact = read_exclusion_artifact(REPO_ROOT, REAL["canonical_artifact_path"])
    assert REAL["canonical_artifact_path"] == (
        "runs/l1_production_2026-08-20/reference_reserve_v1/exclusion_hash_manifest.json"
    )
    assert REAL["artifact_sha256"] == (
        "7e768eb992456cca9b7ba64dd6fda0410f87843faff72d573027139a917e1dd4"
    )
    assert REAL["derived_count"] == 172483
    # kind, hash_algorithm and schema_version are recovered from the artifact itself.
    assert REAL["kind"] == artifact["kind"] == "petitgpt_reference_validation_exclusions"
    assert REAL["hash_algorithm"] == artifact["hash_algorithm"] == "sha256-cleaned-text-utf8-v1"
    assert REAL["artifact_schema_version"] == artifact["artifact_schema_version"] == 1
    assert set(EXCLUSION_REFERENCE_FIELDS) <= set(REAL)


def test_the_candidate_schema_bump_is_deliberate():
    """R4 §19: the bump is for a serialized-contract change, not for moved bytes."""
    assert contract.CANDIDATE_PLAN_SCHEMA == "petitgpt-m-candidate-plan-v3"
    # The exact change: the closed sub-schema gained artifact_schema_version.
    assert "artifact_schema_version" in EXCLUSION_REFERENCE_FIELDS
    v4 = json.loads(
        (
            REPO_ROOT
            / "runs/m_p_native_tooling_repair_r3_2026-08-28/evidence/candidate_m_plan_v4.json"
        ).read_text(encoding="utf-8")
    )
    assert v4["schema_version"] == "petitgpt-m-candidate-plan-v2"
    assert "artifact_schema_version" not in v4["resources"]["canonical_exclusion_authority"]


def test_the_closed_field_set_is_exactly_six():
    assert EXCLUSION_REFERENCE_FIELDS == (
        "artifact_path",
        "artifact_sha256",
        "artifact_schema_version",
        "kind",
        "hash_algorithm",
        "derived_count",
    )


def test_repository_root_resolution_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert canonical_exclusion_authority(REPO_ROOT)["artifact_sha256"] == REAL["artifact_sha256"]
    assert canonical_tokenizer_authority(REPO_ROOT) == REAL_TOKENIZER


@pytest.mark.parametrize(
    "relative",
    [
        "runs/l1_production_2026-08-20/reference_reserve_v1/does_not_exist.json",
        "runs/g_production_2026-08-21/release/exclusion_hash_manifest.json",  # same basename
        "runs/l1_production_2026-08-20/reference_reserve_v1/reserve_manifest.json",  # not it
    ],
)
def test_a_wrong_or_missing_artifact_path_is_refused(relative):
    with pytest.raises(contract.StageMError):
        derive_exclusion_reference(
            REPO_ROOT,
            {**{f: REAL[f] for f in EXCLUSION_REFERENCE_FIELDS}, "artifact_path": relative},
            participant="probe",
            label="probe",
        )


def test_a_path_outside_the_repository_root_is_refused(tmp_path):
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps({"schema_version": 1, "hash_count": 1, "hashes": ["a" * 64]}))
    with pytest.raises(contract.StageMError, match="outside the repository root"):
        read_exclusion_artifact(REPO_ROOT, str(outside))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "petitgpt_something_else"),
        ("hash_algorithm", "sha256"),
        ("artifact_schema_version", 2),
        ("artifact_sha256", "0" * 64),
        ("derived_count", 999),
    ],
)
def test_a_wrong_declaration_against_the_right_path_is_refused(field, value):
    declared = {f: REAL[f] for f in EXCLUSION_REFERENCE_FIELDS}
    declared["artifact_path"] = REAL["canonical_artifact_path"]
    declared[field] = value
    with pytest.raises(contract.StageMError, match=field):
        derive_exclusion_reference(REPO_ROOT, declared, participant="probe", label="probe")


def test_a_non_json_or_directory_reference_is_refused():
    with pytest.raises(contract.StageMError):
        read_exclusion_artifact(REPO_ROOT, "runs/l1_production_2026-08-20/reference_reserve_v1")


# =============================================================== R4-B five-party derivation


FIVE_PARTIES = (
    "candidate_m_plan",
    "stage_m_release[stage_a]",
    "stage_m_release[stage_b]",
    "accepted_g",
    "accepted_g2",
)


def test_all_five_participants_independently_agree(m_run):
    agreed = native_chain(m_run)["shared_exclusion_authority"]
    assert set(agreed["participants"]) == set(FIVE_PARTIES)
    assert agreed["participant_count"] == 5
    assert agreed["compared_fields"] == list(EXCLUSION_REFERENCE_FIELDS)
    canonical = m_run["canonical_exclusion"]
    for field in ("artifact_sha256", "derived_count", "kind", "hash_algorithm"):
        assert agreed[field] == canonical[field]


def _break_participant(m_run, participant: str, mutate) -> None:
    if participant == "candidate_m_plan":
        plan = read_json(m_run["plan_path"])
        mutate(plan["resources"]["canonical_exclusion_authority"])
        m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    elif participant.startswith("stage_m_release"):
        stage = participant[len("stage_m_release[") : -1]
        path = m_run["releases"][stage] / "meta.json"
        meta = read_json(path)
        mutate(meta["reference_validation_exclusion"])
        path.write_bytes(canonical_json_bytes(meta))
    else:
        key = "g_manifest" if participant == "accepted_g" else "g2_manifest"
        path = m_run["canonical_exclusion"][key]
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = (
            payload["reference_reserve_exclusion"]["manifests"][0]
            if participant == "accepted_g"
            else payload["reserve_provenance"]["reserve_exclusion"]
        )
        mutate(entry)
        path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("participant", FIVE_PARTIES)
def test_each_participant_opens_its_own_artifact(m_run, participant):
    """Point one participant at a nonexistent path; only a real reopen can notice."""

    def mutate(entry):
        for key in ("artifact_path", "path", "canonical_artifact_path", "manifest_path"):
            if key in entry:
                entry[key] = "runs/l1_production_2026-08-20/reference_reserve_v1/absent.json"

    _break_participant(m_run, participant, mutate)
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


@pytest.mark.parametrize("participant", FIVE_PARTIES)
def test_each_participant_validates_kind_and_algorithm(m_run, participant):
    def mutate(entry):
        if "kind" in entry:
            entry["kind"] = "petitgpt_not_the_right_kind"
        if "hash_algorithm" in entry:
            entry["hash_algorithm"] = "sha256"

    _break_participant(m_run, participant, mutate)
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


@pytest.mark.parametrize("participant", FIVE_PARTIES)
def test_each_participant_validates_its_own_count(m_run, participant):
    def mutate(entry):
        for key in ("derived_count", "hash_count", "union_hash_count"):
            if key in entry:
                entry[key] = 4321

    _break_participant(m_run, participant, mutate)
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


def test_a_same_basename_alternate_path_is_refused(m_run):
    """Correct declarations attached to a byte-identical copy at another path."""
    canonical = m_run["canonical_exclusion"]
    alt_rel = "runs/l1_production_2026-08-20/copy/exclusion_hash_manifest.json"
    alt = m_run["tmp_path"] / alt_rel
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_bytes(Path(canonical["artifact_abs"]).read_bytes())
    copy = read_exclusion_artifact(m_run["tmp_path"], alt_rel)
    assert copy["artifact_sha256"] == canonical["artifact_sha256"]
    assert copy["derived_count"] == canonical["derived_count"]
    assert copy["kind"] == canonical["kind"]

    plan = read_json(m_run["plan_path"])
    plan["resources"]["canonical_exclusion_authority"]["artifact_path"] = alt_rel
    m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


def test_same_count_different_bytes_is_refused(m_run):
    canonical = m_run["canonical_exclusion"]
    payload = json.loads(Path(canonical["artifact_abs"]).read_text(encoding="utf-8"))
    reserialized = m_run["tmp_path"] / "runs/l1_production_2026-08-20/reserialized.json"
    reserialized.parent.mkdir(parents=True, exist_ok=True)
    reserialized.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    other = read_exclusion_artifact(
        m_run["tmp_path"], "runs/l1_production_2026-08-20/reserialized.json"
    )
    assert other["derived_count"] == canonical["derived_count"]
    assert other["artifact_sha256"] != canonical["artifact_sha256"]


def test_a_serialized_pass_claim_cannot_cover_a_differing_participant(m_run):
    """native_shared_authority_validated is not self-asserting."""
    _break_participant(
        m_run,
        "stage_m_release[stage_b]",
        lambda entry: entry.update({"union_hash_count": 4321}),
    )
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


def test_one_inconsistent_participant_fails_through_the_full_loader(native_e2e, tmp_path):
    plan = native_plan(native_e2e)
    plan["release_provenance"]["shared_exclusion_authority"]["kind"] = "petitgpt_wrong"
    path = _write_plan(tmp_path, plan)
    with pytest.raises(RuntimeError):
        rpc.load_run_plan_binding(
            _loader_args(plan, path, "stage_a", native_e2e),
            train_dir=native_e2e["stage_a_dir"],
            tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
            val_dir=native_e2e["reference_val_dir"],
        )


# =============================================================== R4-A candidate contract


def test_candidate_contract_requires_the_full_subschema(m_run):
    result = validate_candidate_plan_contract(read_json(m_run["plan_path"]), m_run["tmp_path"])
    for field in EXCLUSION_REFERENCE_FIELDS:
        assert (
            result["exclusion_authority"][field]
            == m_run["canonical_exclusion"]["artifact_path" if field == "artifact_path" else field]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_path", "runs/l1_production_2026-08-20/reference_reserve_v1/absent.json"),
        ("kind", "petitgpt_wrong_kind"),
        ("hash_algorithm", "sha256"),
        ("artifact_schema_version", 99),
    ],
)
def test_a_right_digest_with_a_wrong_subschema_field_fails_candidate_validation(
    m_run, field, value
):
    plan = read_json(m_run["plan_path"])
    plan["resources"]["canonical_exclusion_authority"][field] = value
    with pytest.raises(contract.StageMError):
        validate_candidate_plan_contract(plan, m_run["tmp_path"])


@pytest.mark.parametrize("field", list(EXCLUSION_REFERENCE_FIELDS))
def test_a_missing_subschema_field_fails_candidate_validation(m_run, field):
    plan = read_json(m_run["plan_path"])
    plan["resources"]["canonical_exclusion_authority"].pop(field)
    with pytest.raises(contract.StageMError):
        validate_candidate_plan_contract(plan, m_run["tmp_path"])


# =============================================================== R4-C release semantics


def test_the_semantic_group_set_is_derived_from_the_source_not_the_constant():
    """R4-C: independently confirm the schema really has this many groups."""
    src = REPO_ROOT / "pretrain" / "stage_p_native_provenance_v1.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "validate_release_semantics"
    )
    called = [
        node.args[0].value
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "group"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert sorted(called) == sorted(M_RELEASE_SEMANTIC_GROUPS)
    assert len(called) == len(set(called)) == 37


def test_every_group_is_checked_for_both_stages(m_run):
    provenance = native_chain(m_run)
    for stage in STAGE_STREAMS:
        assert provenance["stages"][stage]["semantic_groups_checked"] == 37


_R4_MUTATIONS: dict[str, Any] = {
    # shared_exclusion_authority: previously partial, now complete
    "block_schema_version": lambda m: m["reference_validation_exclusion"].__setitem__(
        "canonical_artifact_schema_version", 99
    ),
    "block_kind": lambda m: m["reference_validation_exclusion"].__setitem__("kind", "wrong"),
    "block_algorithm": lambda m: m["reference_validation_exclusion"].__setitem__(
        "hash_algorithm", "sha256"
    ),
    "entry_kind": lambda m: m["reference_validation_exclusion"]["manifests"][0].__setitem__(
        "kind", "wrong"
    ),
    "entry_algorithm": lambda m: m["reference_validation_exclusion"]["manifests"][0].__setitem__(
        "hash_algorithm", "sha256"
    ),
    "entry_schema_version": lambda m: m["reference_validation_exclusion"]["manifests"][
        0
    ].__setitem__("schema_version", 99),
    "entry_path_disagrees_with_block": lambda m: m["reference_validation_exclusion"]["manifests"][
        0
    ].__setitem__("path", "runs/l1_production_2026-08-20/reference_reserve_v1/other.json"),
    "enforced_at_stage": lambda m: m["reference_validation_exclusion"].__setitem__(
        "enforced_at_stage", "stage_m"
    ),
    "reapplied_by_stage_m": lambda m: m["reference_validation_exclusion"].__setitem__(
        "reapplied_by_stage_m", True
    ),
    "manifest_count": lambda m: m["reference_validation_exclusion"].__setitem__(
        "manifest_count", 2
    ),
    # tokenizer_path: previously basename+suffix only, now resolved and hashed
    "tokenizer_path_nonexistent_same_basename": lambda m: m.__setitem__(
        "tokenizer_path", "runs/g_production_2026-08-21/absent/tokenizer.json"
    ),
    "tokenizer_path_suffix_only": lambda m: m.__setitem__(
        "tokenizer_path", "release/tokenizer.json"
    ),
    "tokenizer_path_alternate_directory": lambda m: m.__setitem__(
        "tokenizer_path", "tok/tokenizer.json"
    ),
    "tokenizer_digest": lambda m: m.__setitem__("tokenizer_sha256", "0" * 64),
}


@pytest.mark.parametrize("case", sorted(_R4_MUTATIONS))
def test_previously_partial_groups_now_reject_every_contradiction(m_run, case):
    path = m_run["releases"]["stage_a"] / "meta.json"
    meta = read_json(path)
    _R4_MUTATIONS[case](meta)
    path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises((NativeProvenanceError, contract.StageMError, RuntimeError)):
        native_chain(m_run)


def test_the_reported_mutation_count_is_the_executed_count():
    """R4-F: evidence must quote the number of cases that actually run."""
    assert len(_R4_MUTATIONS) == 14


# =============================================================== R4-C tokenizer path


def test_canonical_tokenizer_path_is_recovered_from_the_accepted_release():
    assert REAL_TOKENIZER["tokenizer_path"] == "runs/g_production_2026-08-21/release/tokenizer.json"
    assert REAL_TOKENIZER["tokenizer_sha256"] == (
        "d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce"
    )
    assert REAL_TOKENIZER["named_by"] == (
        "runs/g_production_2026-08-21/release/tokenizer_release_manifest.json"
    )
    # It is the bytes at that exact path, not a declaration.
    import hashlib

    actual = hashlib.sha256((REPO_ROOT / REAL_TOKENIZER["tokenizer_path"]).read_bytes()).hexdigest()
    assert actual == REAL_TOKENIZER["tokenizer_sha256"]


def test_candidate_binds_the_canonical_tokenizer_path(m_run):
    plan = read_json(m_run["plan_path"])
    assert plan["resources"]["tokenizer"]["path"] == m_run["canonical_exclusion"]["tokenizer_path"]
    assert (
        plan["resources"]["tokenizer"]["sha256"]
        == (m_run["canonical_exclusion"]["tokenizer_sha256"])
    )


@pytest.mark.parametrize(
    "value",
    [
        "runs/g_production_2026-08-21/release/absent/tokenizer.json",
        "release/tokenizer.json",
        "tok/tokenizer.json",
    ],
)
def test_a_wrong_tokenizer_path_fails_candidate_validation(m_run, value):
    plan = read_json(m_run["plan_path"])
    plan["resources"]["tokenizer"]["path"] = value
    with pytest.raises(contract.StageMError):
        validate_candidate_plan_contract(plan, m_run["tmp_path"])


def test_a_path_pointing_at_different_bytes_fails(m_run):
    decoy = m_run["tmp_path"] / "runs/g_production_2026-08-21/release/decoy.json"
    decoy.write_text("{}", encoding="utf-8")
    plan = read_json(m_run["plan_path"])
    plan["resources"]["tokenizer"]["path"] = "runs/g_production_2026-08-21/release/decoy.json"
    with pytest.raises(contract.StageMError):
        validate_candidate_plan_contract(plan, m_run["tmp_path"])


# =============================================================== R4-D required-field scope


def _scope(native_e2e, tmp_path) -> dict[str, Any]:
    base = native_plan(native_e2e)
    sha = base["release_provenance"]["shared_tokenizer_sha256"]

    def loads(plan) -> bool:
        path = _write_plan(tmp_path, plan)
        try:
            rpc.load_run_plan_binding(
                _loader_args(plan, path, "stage_a", native_e2e),
                train_dir=native_e2e["stage_a_dir"],
                tokenizer_sha256=sha,
                val_dir=native_e2e["reference_val_dir"],
            )
        except Exception:
            return False
        return True

    top = [
        key
        for key in sorted(base["release_provenance"])
        if not loads(_without(base, key, stage_key=None))
    ]
    per_stage = [
        key
        for key in sorted(base["release_provenance"]["stages"]["stage_a"])
        if not loads(_without(base, key, stage_key="stage_a"))
    ]
    return {"top": top, "per_stage": per_stage}


def _without(base, key, *, stage_key):
    plan = json.loads(json.dumps(base))
    if stage_key is None:
        plan["release_provenance"].pop(key, None)
    else:
        plan["release_provenance"]["stages"][stage_key].pop(key, None)
    return plan


def test_required_field_scope_is_measured_not_asserted(native_e2e, tmp_path):
    """R4-D: delete each serialized field in turn and drive the real loader."""
    scope = _scope(native_e2e, tmp_path)
    assert len(scope["top"]) == 23, scope["top"]
    assert len(scope["per_stage"]) == 8, scope["per_stage"]
    assert len(scope["top"]) + 2 * len(scope["per_stage"]) == 39
    # The constants must describe that measurement, not a subset of it.
    assert sorted(rpc.NATIVE_EFFECTIVE_REQUIRED_TOP_LEVEL_FIELDS) == scope["top"]
    assert sorted(rpc.NATIVE_REQUIRED_STAGE_FIELDS) == scope["per_stage"]
    # 19 are checked by name in this module; 4 by dedicated checks earlier in the loader.
    assert len(rpc.NATIVE_REQUIRED_PROVENANCE_FIELDS) == 19
    assert len(rpc.NATIVE_REQUIRED_TOP_LEVEL_FIELDS_ENFORCED_ELSEWHERE) == 4


# =============================================================== R4-E P helper bundle


def test_p_helper_bundle_uses_its_own_schema_in_the_digest_preimage():
    files, digest = p_native_implementation_bundle(REPO_ROOT)
    assert P_NATIVE_HELPER_BUNDLE_SCHEMA == "petitgpt-p-native-helper-bundle-v1"
    assert P_NATIVE_HELPER_BUNDLE_SCHEMA != BUNDLE_SCHEMA
    # The recorded digest reproduces from the P file map under the P schema.
    assert bundle_sha256(files, schema=P_NATIVE_HELPER_BUNDLE_SCHEMA) == digest
    # The same file map under the M schema is a different digest.
    assert bundle_sha256(files, schema=BUNDLE_SCHEMA) != digest
    assert sorted(files) == sorted(P_NATIVE_IMPLEMENTATION_BUNDLE_FILES)
    assert len(files) == 7


def test_the_two_bundles_are_distinct_artifacts():
    _m_files, m_digest = m_implementation_bundle(REPO_ROOT)
    _p_files, p_digest = p_native_implementation_bundle(REPO_ROOT)
    assert m_digest != p_digest
    assert set(P_NATIVE_IMPLEMENTATION_BUNDLE_FILES) != set(
        bundle_files(REPO_ROOT, contract.M_IMPLEMENTATION_BUNDLE_FILES)
    )


def test_native_provenance_binds_the_p_schema_digest(m_run):
    provenance = native_chain(m_run)
    _files, digest = p_native_implementation_bundle(m_run["tmp_path"])
    assert provenance["stage_p_native_validator_bundle_sha256"] == digest


def test_a_m_schema_digest_is_rejected_through_the_full_loader(native_e2e, tmp_path):
    """The P-schema digest is load-bearing: the M-schema preimage must not validate."""
    files, correct = p_native_implementation_bundle(native_e2e["tmp_path"])
    wrong = bundle_sha256(files, schema=BUNDLE_SCHEMA)
    assert wrong != correct
    plan = native_plan(native_e2e)
    assert plan["release_provenance"]["stage_p_native_validator_bundle_sha256"] == correct
    plan["release_provenance"]["stage_p_native_validator_bundle_sha256"] = wrong
    path = _write_plan(tmp_path, plan)
    with pytest.raises(RuntimeError):
        rpc.load_run_plan_binding(
            _loader_args(plan, path, "stage_a", native_e2e),
            train_dir=native_e2e["stage_a_dir"],
            tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
            val_dir=native_e2e["reference_val_dir"],
        )


# =============================================================== full bounded flow


@pytest.mark.parametrize("stage", list(STAGE_STREAMS))
def test_full_bounded_flow_reaches_the_dataset(native_e2e, tmp_path, stage):
    plan = native_plan(native_e2e)
    path = _write_plan(tmp_path, plan)
    binding = rpc.load_run_plan_binding(
        _loader_args(plan, path, stage, native_e2e),
        train_dir=native_e2e[f"{stage}_dir"],
        tokenizer_sha256=plan["release_provenance"]["shared_tokenizer_sha256"],
        val_dir=native_e2e["reference_val_dir"],
    )
    assert binding is not None
    assert binding["provenance_chain_kind"] == "accepted_stage_i_native_v1"
    agreed = plan["release_provenance"]["shared_exclusion_authority"]
    for field in EXCLUSION_REFERENCE_FIELDS:
        assert field in agreed
    dataset = PackedBinDataset(
        str(native_e2e["releases"][stage] / "train"), seq_len=2048, require_release_manifest=True
    )
    assert dataset.stats()["tail_transitions"] == 0
    assert len(dataset) == plan["stages"][stage]["unique_blocks"]
