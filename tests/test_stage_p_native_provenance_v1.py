"""Stage-P native provenance branch tests, plus the legacy selector-v1 regression guard."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pretrain.run_plan_contract import (
    LEGACY_CHAIN_KIND as CONTRACT_LEGACY_KIND,
    NATIVE_CHAIN_KIND as CONTRACT_NATIVE_KIND,
    PROVENANCE_CHAIN_KINDS as CONTRACT_KINDS,
)
from pretrain.stage_m_contract_v1 import (
    MODEL_CONTRACT,
    ORDERING_CONTRACT_ID,
    STAGE_STREAMS,
    canonical_json_bytes,
    p_native_implementation_bundle,
)
import pretrain.stage_m_realize_v1 as realize
from pretrain.stage_p_native_provenance_v1 import (
    LEGACY_CHAIN_KIND,
    LEGACY_ONLY_FIELDS,
    NATIVE_CHAIN_KIND,
    NATIVE_ONLY_FIELDS,
    NATIVE_PROVENANCE_SCHEMA,
    PROVENANCE_CHAIN_KINDS,
    NativeProvenanceError,
    assert_single_branch,
    validate_native_chain,
)
from tests._stage_m_fixtures import (
    make_record,
    read_json,
    save_tokenizer,
    tiny_tokenizer,
    write_accepted_stage_i,
    write_exclusion_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


@pytest.fixture
def native_chain(tmp_path, tok, monkeypatch):
    """A complete accepted-I -> candidate-M plan -> two published Stage-M releases fixture."""
    records = []
    stages = ("stage_b", "stage_a", "stage_b", "stage_a")
    for index in range(400):
        stage = stages[index % len(stages)]
        records.append(
            make_record(
                tok,
                stage=stage,
                source_id=f"{stage[-1]}_src{index % 2}",
                binding=f"ib_src{index % 2}",
                ordinal=index,
                rank=400 - index,
                text=(
                    "The quick brown fox jumps over the lazy dog while a tutorial paragraph "
                    f"explains a concept step by step with examples number {index}."
                ),
            )
        )
    accepted_dir = write_accepted_stage_i(tmp_path / "stage_i", records, records_per_shard=64)

    monkeypatch.setattr(realize, "assert_tokenizer_contract", lambda path: None)
    monkeypatch.setattr(realize, "verify_environment", lambda environment: None)
    monkeypatch.setattr(realize, "resolve_repo_root", lambda explicit=None: tmp_path.resolve())

    from pretrain.stage_m_contract_v1 import M_IMPLEMENTATION_BUNDLE_FILES

    for relative in M_IMPLEMENTATION_BUNDLE_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())

    tokenizer_path = save_tokenizer(tok, tmp_path / "tok" / "tokenizer.json")
    exclusion_path = write_exclusion_manifest(tmp_path / "excl" / "exclusion.json")
    plan_path = tmp_path / "candidate_m_plan.json"
    assert (
        realize.main([
            "plan",
            "--accepted-stage-i-dir",
            str(accepted_dir),
            "--tokenizer",
            str(tokenizer_path.relative_to(tmp_path)),
            "--reference-exclusion-manifest",
            str(exclusion_path.relative_to(tmp_path)),
            "--out",
            str(plan_path),
            "--shard-tokens",
            "4096",
            "--implementation-commit",
            "0" * 40,
        ])
        == 0
    )
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    context = realize.authorize_plan(plan_path, plan_sha256, tmp_path.resolve())
    out_dir = tmp_path / "stage_m"
    realize.realize_and_publish(context, out_dir=out_dir)

    return {
        "accepted_dir": accepted_dir,
        "plan_path": plan_path,
        "plan_sha256": plan_sha256,
        "releases": {stage: out_dir / stage for stage in STAGE_STREAMS},
        "tmp_path": tmp_path,
    }


def _validate(native_chain, **overrides):
    kwargs = {
        "repo_root": REPO_ROOT,
        "accepted_stage_i_dir": native_chain["accepted_dir"],
        "candidate_m_plan": native_chain["plan_path"],
        "expected_candidate_m_plan_sha256": native_chain["plan_sha256"],
        "stage_releases": dict(native_chain["releases"]),
    }
    kwargs.update(overrides)
    return validate_native_chain(**kwargs)


# --------------------------------------------------------------------- 13.12 native chain


def test_valid_native_chain_passes_and_derives_full_chain_validated(native_chain):
    provenance = _validate(native_chain)
    assert provenance["provenance_chain_kind"] == NATIVE_CHAIN_KIND
    assert provenance["schema_version"] == NATIVE_PROVENANCE_SCHEMA
    assert provenance["full_chain_validated"] is True
    assert provenance["candidate_m_plan_sha256"] == native_chain["plan_sha256"]
    assert provenance["stage_m_ordering_policy"] == ORDERING_CONTRACT_ID
    assert provenance["model_contract"] == dict(MODEL_CONTRACT)
    assert sorted(provenance["stages"]) == sorted(STAGE_STREAMS)
    for stage in STAGE_STREAMS:
        entry = provenance["stages"][stage]
        assert entry["shards"] >= 1
        assert (
            entry["stored_token_ids"] == (entry["expected_accounting"]["retained_stored_token_ids"])
        )


def test_changed_candidate_plan_sha_fails(native_chain):
    with pytest.raises(NativeProvenanceError, match="plan digest mismatch"):
        _validate(native_chain, expected_candidate_m_plan_sha256="0" * 64)


def test_changed_accepted_stage_i_identity_fails(native_chain):
    manifest_path = native_chain["accepted_dir"] / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["stage_i_run"]["run_identity"] = "0" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises((NativeProvenanceError, RuntimeError)):
        _validate(native_chain)


def test_changed_stage_m_release_metadata_fails(native_chain):
    meta_path = native_chain["releases"]["stage_a"] / "meta.json"
    meta = read_json(meta_path)
    meta["stage_m"]["candidate_plan_sha256"] = "0" * 64
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(NativeProvenanceError, match="not produced by this candidate"):
        _validate(native_chain)


def test_changed_packed_shard_digest_fails(native_chain):
    train = native_chain["releases"]["stage_a"] / "train"
    shard = sorted(train.glob("shard_*.bin"))[0]
    data = bytearray(shard.read_bytes())
    data[0] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _validate(native_chain)


def test_changed_accounting_fails(native_chain):
    meta_path = native_chain["releases"]["stage_b"] / "meta.json"
    meta = read_json(meta_path)
    meta["stage_m_accounting"]["training_sequences"] += 1
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(NativeProvenanceError, match="accounting differs"):
        _validate(native_chain)


def test_changed_input_sequence_commitment_fails(native_chain):
    meta_path = native_chain["releases"]["stage_a"] / "meta.json"
    meta = read_json(meta_path)
    meta["stage_m"]["input_sequence_commitment"] = "0" * 64
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(NativeProvenanceError, match="commitment differs"):
        _validate(native_chain)


def test_missing_native_field_fails_closed(native_chain):
    meta_path = native_chain["releases"]["stage_a"] / "meta.json"
    meta = read_json(meta_path)
    del meta["stage_m"]["ordering_policy"]
    meta_path.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(NativeProvenanceError, match="ordering policy"):
        _validate(native_chain)


def test_a_missing_stage_release_fails(native_chain):
    releases = dict(native_chain["releases"])
    releases.pop("stage_b")
    with pytest.raises(NativeProvenanceError, match="requires exactly"):
        _validate(native_chain, stage_releases=releases)


def test_swapped_stage_releases_fail(native_chain):
    releases = {
        "stage_a": native_chain["releases"]["stage_b"],
        "stage_b": native_chain["releases"]["stage_a"],
    }
    with pytest.raises(NativeProvenanceError, match="names a different stage"):
        _validate(native_chain, stage_releases=releases)


def test_p_native_declared_bundle_is_the_real_local_dependency_closure():
    """The P validator bundle must cover its whole closure, function-level imports included."""
    import ast

    from pretrain.stage_m_contract_v1 import P_NATIVE_IMPLEMENTATION_BUNDLE_FILES

    seen: set[str] = set()
    stack = ["pretrain/stage_p_native_provenance_v1.py"]
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            for dotted in names:
                parts = dotted.split(".")
                for depth in range(1, len(parts)):
                    init = REPO_ROOT.joinpath(*parts[:depth], "__init__.py")
                    if init.is_file():
                        stack.append(str(init.relative_to(REPO_ROOT)))
                module = REPO_ROOT.joinpath(*parts).with_suffix(".py")
                if module.is_file():
                    stack.append(str(module.relative_to(REPO_ROOT)))
    assert sorted(seen) == sorted(P_NATIVE_IMPLEMENTATION_BUNDLE_FILES)
    unbound = [c for c in seen if c not in set(P_NATIVE_IMPLEMENTATION_BUNDLE_FILES)]
    assert unbound == []


def test_native_validator_bundle_is_reported_and_separate(native_chain):
    provenance = _validate(native_chain)
    _files, digest = p_native_implementation_bundle(REPO_ROOT)
    assert provenance["stage_p_native_validator_bundle_sha256"] == digest
    assert provenance["stage_m_implementation_bundle_sha256"] != digest


# --------------------------------------------------------------------- 13.3 branch mixing


def test_caller_supplied_full_chain_validated_is_refused():
    with pytest.raises(NativeProvenanceError, match="must not be caller-supplied"):
        assert_single_branch({"full_chain_validated": True}, chain_kind=NATIVE_CHAIN_KIND)


@pytest.mark.parametrize("field", list(LEGACY_ONLY_FIELDS))
def test_native_payload_carrying_legacy_fields_is_refused(field):
    with pytest.raises(NativeProvenanceError, match="legacy selector-v1 fields"):
        assert_single_branch({field: "x"}, chain_kind=NATIVE_CHAIN_KIND)


@pytest.mark.parametrize("field", list(NATIVE_ONLY_FIELDS))
def test_legacy_payload_carrying_native_fields_is_refused(field):
    with pytest.raises(NativeProvenanceError, match="native Stage-I authority fields"):
        assert_single_branch({field: "x"}, chain_kind=LEGACY_CHAIN_KIND)


def test_unknown_chain_kind_is_refused():
    with pytest.raises(NativeProvenanceError, match="unknown provenance_chain_kind"):
        assert_single_branch({}, chain_kind="something_else")


def test_run_plan_contract_and_native_module_agree_on_the_discriminators():
    assert CONTRACT_NATIVE_KIND == NATIVE_CHAIN_KIND
    assert CONTRACT_LEGACY_KIND == LEGACY_CHAIN_KIND
    assert CONTRACT_KINDS == PROVENANCE_CHAIN_KINDS


# --------------------------------------------------------------------- planner dispatch


def test_planner_rejects_native_kind_with_a_legacy_selection_manifest(tmp_path):
    from pretrain.plan_pretrain_run import build_run_plan

    with pytest.raises(ValueError, match="must not be given a legacy selector-v1"):
        build_run_plan(
            stage_a_dir=tmp_path,
            stage_b_dir=tmp_path,
            seq_len=2048,
            micro_bsz=4,
            grad_accum=8,
            warmup_steps=10,
            decay_fraction=0.1,
            provenance_chain_kind=NATIVE_CHAIN_KIND,
            accepted_stage_i_dir=tmp_path,
            candidate_m_plan=tmp_path,
            expected_candidate_m_plan_sha256="0" * 64,
            selection_manifest=tmp_path / "selection.json",
        )


def test_planner_rejects_legacy_kind_with_native_inputs(tmp_path):
    from pretrain.plan_pretrain_run import build_run_plan

    with pytest.raises(ValueError, match="must not be given accepted Stage-I native inputs"):
        build_run_plan(
            stage_a_dir=tmp_path,
            stage_b_dir=tmp_path,
            seq_len=2048,
            micro_bsz=4,
            grad_accum=8,
            warmup_steps=10,
            decay_fraction=0.1,
            provenance_chain_kind=LEGACY_CHAIN_KIND,
            accepted_stage_i_dir=tmp_path,
        )


def test_planner_rejects_an_unknown_chain_kind(tmp_path):
    from pretrain.plan_pretrain_run import build_run_plan

    with pytest.raises(ValueError, match="provenance_chain_kind must be one of"):
        build_run_plan(
            stage_a_dir=tmp_path,
            stage_b_dir=tmp_path,
            seq_len=2048,
            micro_bsz=4,
            grad_accum=8,
            warmup_steps=10,
            decay_fraction=0.1,
            provenance_chain_kind="made_up",
        )


def test_planner_requires_all_native_inputs_together(tmp_path):
    from pretrain.plan_pretrain_run import build_run_plan

    with pytest.raises(ValueError, match="must be supplied together"):
        build_run_plan(
            stage_a_dir=tmp_path,
            stage_b_dir=tmp_path,
            seq_len=2048,
            micro_bsz=4,
            grad_accum=8,
            warmup_steps=10,
            decay_fraction=0.1,
            provenance_chain_kind=NATIVE_CHAIN_KIND,
            accepted_stage_i_dir=tmp_path,
        )


def test_no_silent_branch_fallback_default_is_legacy():
    import inspect

    from pretrain.plan_pretrain_run import build_run_plan

    signature = inspect.signature(build_run_plan)
    assert signature.parameters["provenance_chain_kind"].default == LEGACY_CHAIN_KIND


# --------------------------------------------------------------------- 13.11 legacy regression


def test_legacy_cli_still_requires_its_three_provenance_inputs():
    from pretrain.plan_pretrain_run import main

    with pytest.raises(SystemExit) as excinfo:
        main([
            "--stage_a_dir",
            "/nonexistent/a",
            "--stage_b_dir",
            "/nonexistent/b",
            "--seq_len",
            "2048",
            "--micro_bsz",
            "4",
            "--grad_accum",
            "8",
            "--warmup_steps",
            "10",
            "--decay_fraction",
            "0.1",
            "--out_json",
            "/nonexistent/out.json",
        ])
    assert excinfo.value.code == 2


def test_legacy_validate_full_provenance_is_untouched():
    """The legacy validator keeps its exact D-026 signature and required inputs."""
    import inspect

    from pretrain.plan_pretrain_run import _validate_full_provenance

    parameters = inspect.signature(_validate_full_provenance).parameters
    for name in (
        "reference_val_dir",
        "tokenizer_release_manifest",
        "selection_manifest",
        "stage_b_selection_stage",
        "expected_exclusion_sha256s",
        "stage_a_release",
        "stage_b_release",
        "expected_tokenizer_sha256",
    ):
        assert name in parameters
        assert parameters[name].default is inspect.Parameter.empty


def test_run_plan_contract_still_rejects_a_legacy_plan_without_selection():
    """A legacy-kind provenance object with no selection block still fails closed."""
    import pretrain.run_plan_contract as rpc

    provenance = {"provenance_chain_kind": LEGACY_CHAIN_KIND, "full_chain_validated": True}
    with pytest.raises(RuntimeError, match="release_provenance.selection"):
        rpc._mapping(provenance.get("selection"), field="release_provenance.selection")
