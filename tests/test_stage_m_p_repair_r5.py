"""R5 repair regressions: the Stage-M PRODUCER gate compares the whole six-field canonical
exclusion reference, and the symlink behaviour is documented accurately at both layers.

Every test here drives the real producer entrypoint `stage_m_realize_v1.authorize_plan`, not a
Stage-P helper. Bounded synthetic fixtures only: no real Stage-M production, no packed corpus,
no training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import pretrain.stage_m_contract_v1 as contract
from pretrain.stage_m_contract_v1 import (
    EXCLUSION_REFERENCE_FIELDS,
    canonical_exclusion_authority,
    canonical_json_bytes,
    canonical_repo_relative,
    file_sha256,
    read_exclusion_artifact,
    require_identical_exclusion_authorities,
    validate_candidate_plan_contract,
)
import pretrain.stage_m_realize_v1 as realize
from tests._stage_m_fixtures import read_json

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL = canonical_exclusion_authority(REPO_ROOT)


# ------------------------------------------------------------------ helpers


def _authorize(m_run, plan: dict[str, Any] | None = None):
    """Drive the REAL producer entrypoint against the fixture's plan bytes."""
    path = m_run["plan_path"]
    if plan is not None:
        path.write_bytes(canonical_json_bytes(plan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return realize.authorize_plan(path, digest, m_run["tmp_path"].resolve())


def _plan(m_run) -> dict[str, Any]:
    return read_json(m_run["plan_path"])


# ------------------------------------------------------------------ R5-A producer gate


def test_producer_gate_accepts_the_exact_six_field_reference(m_run):
    context = _authorize(m_run)
    canonical = m_run["canonical_exclusion"]
    reference = context.producer_gate["canonical_exclusion_reference"]
    assert sorted(reference) == sorted(EXCLUSION_REFERENCE_FIELDS)
    for field in EXCLUSION_REFERENCE_FIELDS:
        expected = canonical["artifact_path"] if field == "artifact_path" else canonical[field]
        assert reference[field] == expected, field


def test_producer_gate_compares_every_closed_field(m_run):
    """The gate's scope is recorded, not implied: six fields, three participants."""
    context = _authorize(m_run)
    assert context.producer_gate["producer_exclusion_compared_fields"] == list(
        EXCLUSION_REFERENCE_FIELDS
    )
    assert len(context.producer_gate["producer_exclusion_compared_fields"]) == 6
    assert set(context.producer_gate["producer_exclusion_participants"]) == {
        "candidate_m_plan",
        "accepted_g",
        "accepted_g2",
    }


def test_producer_gate_rejects_an_alternate_path_with_identical_bytes(m_run):
    """The exact Codex R4 finding, reproduced through authorize_plan.

    Byte-identical exclusion contents at a different repository path. SHA256 and count both
    match, so the R4 producer gate accepted it; the R5 gate must not.
    """
    canonical = m_run["canonical_exclusion"]
    alt_rel = "runs/l1_production_2026-08-20/reference_reserve_v1/identical_copy.json"
    alt = m_run["tmp_path"] / alt_rel
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_bytes(Path(canonical["artifact_abs"]).read_bytes())

    # The copy really is byte-identical: same digest, same count, same kind and algorithm.
    copy = read_exclusion_artifact(m_run["tmp_path"], alt_rel)
    assert copy["artifact_sha256"] == canonical["artifact_sha256"]
    assert copy["derived_count"] == canonical["derived_count"]
    assert copy["kind"] == canonical["kind"]
    assert copy["hash_algorithm"] == canonical["hash_algorithm"]
    assert copy["artifact_schema_version"] == canonical["artifact_schema_version"]
    # Only the path differs.
    assert copy["artifact_path"] != canonical["artifact_path"]

    plan = _plan(m_run)
    plan["resources"]["canonical_exclusion_authority"]["artifact_path"] = alt_rel
    with pytest.raises(contract.StageMError) as excinfo:
        _authorize(m_run, plan)
    assert "artifact_path" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_sha256", "0" * 64),
        ("artifact_schema_version", 99),
        ("kind", "petitgpt_some_other_kind"),
        ("hash_algorithm", "sha256"),
        ("derived_count", 4321),
    ],
)
def test_producer_gate_rejects_each_individual_field_mismatch(m_run, field, value):
    plan = _plan(m_run)
    plan["resources"]["canonical_exclusion_authority"][field] = value
    with pytest.raises(contract.StageMError):
        _authorize(m_run, plan)


def test_producer_gate_rejects_a_missing_closed_field(m_run):
    for field in EXCLUSION_REFERENCE_FIELDS:
        plan = _plan(m_run)
        plan["resources"]["canonical_exclusion_authority"].pop(field)
        with pytest.raises(contract.StageMError):
            _authorize(m_run, plan)


def test_the_r4_partial_comparison_would_have_accepted_the_alternate_path(m_run):
    """Show the defect is real: SHA + count alone cannot distinguish the two references."""
    canonical = m_run["canonical_exclusion"]
    alt_rel = "runs/l1_production_2026-08-20/reference_reserve_v1/identical_copy2.json"
    alt = m_run["tmp_path"] / alt_rel
    alt.parent.mkdir(parents=True, exist_ok=True)
    alt.write_bytes(Path(canonical["artifact_abs"]).read_bytes())
    copy = read_exclusion_artifact(m_run["tmp_path"], alt_rel)
    # The R4 producer predicate, restated here only to show what it could not see.
    r4_predicate = copy["artifact_sha256"] == canonical["artifact_sha256"] and int(
        copy["derived_count"]
    ) == int(canonical["derived_count"])
    assert r4_predicate is True, "the R4 gate's two comparisons both pass on the copy"
    # The R5 comparison, over the whole closed reference, does see it.
    with pytest.raises(contract.StageMError, match="artifact_path"):
        require_identical_exclusion_authorities([
            contract.exclusion_authority(
                participant="accepted_g_and_g2",
                artifact_path=canonical["artifact_path"],
                artifact_sha256=canonical["artifact_sha256"],
                derived_count=canonical["derived_count"],
                artifact_schema_version=canonical["artifact_schema_version"],
                kind=canonical["kind"],
                hash_algorithm=canonical["hash_algorithm"],
            ),
            contract.exclusion_authority(
                participant="candidate_m_plan",
                artifact_path=copy["artifact_path"],
                artifact_sha256=copy["artifact_sha256"],
                derived_count=copy["derived_count"],
                artifact_schema_version=copy["artifact_schema_version"],
                kind=copy["kind"],
                hash_algorithm=copy["hash_algorithm"],
            ),
        ])


def test_producer_and_native_use_one_shared_comparison_routine():
    """Section 7: one six-field model, not a second partial comparison."""
    import inspect

    import pretrain.stage_p_native_provenance_v1 as native

    producer = inspect.getsource(realize._derive_state)  # noqa: SLF001
    assert "require_identical_exclusion_authorities" in producer
    # The producer reaches the shared routine through the neutral contract layer, never through
    # the Stage-P validator: no Stage-P import exists in the Stage-M producer.
    realize_src = inspect.getsource(realize)
    assert "stage_p_native_provenance" not in realize_src
    # Both callers resolve the same function object.
    assert (
        realize.require_identical_exclusion_authorities
        is native.require_identical_exclusion_authorities
        is contract.require_identical_exclusion_authorities
    )


def test_the_producer_gate_does_not_rely_on_the_candidate_file_sha_alone(m_run):
    """Section 9: a correct plan digest does not excuse a wrong exclusion reference."""
    plan = _plan(m_run)
    plan["resources"]["canonical_exclusion_authority"]["kind"] = "petitgpt_wrong"
    m_run["plan_path"].write_bytes(canonical_json_bytes(plan))
    # The digest is recomputed from the tampered bytes, so it is CORRECT for this file.
    digest = hashlib.sha256(m_run["plan_path"].read_bytes()).hexdigest()
    with pytest.raises(contract.StageMError):
        realize.authorize_plan(m_run["plan_path"], digest, m_run["tmp_path"].resolve())


# ------------------------------------------------------------------ candidate validator


def test_candidate_validator_checks_the_same_six_fields(m_run):
    result = validate_candidate_plan_contract(_plan(m_run), m_run["tmp_path"])
    canonical = m_run["canonical_exclusion"]
    for field in EXCLUSION_REFERENCE_FIELDS:
        expected = canonical["artifact_path"] if field == "artifact_path" else canonical[field]
        assert result["exclusion_authority"][field] == expected, field


def test_candidate_v6_binds_the_exact_canonical_reference():
    """The values R5 must bind, read from the accepted artifacts in this repository."""
    assert REAL["canonical_artifact_path"] == (
        "runs/l1_production_2026-08-20/reference_reserve_v1/exclusion_hash_manifest.json"
    )
    assert REAL["artifact_sha256"] == (
        "7e768eb992456cca9b7ba64dd6fda0410f87843faff72d573027139a917e1dd4"
    )
    assert REAL["artifact_schema_version"] == 1
    assert REAL["kind"] == "petitgpt_reference_validation_exclusions"
    assert REAL["hash_algorithm"] == "sha256-cleaned-text-utf8-v1"
    assert REAL["derived_count"] == 172483


# ------------------------------------------------------------------ symlink scope


def test_the_canonical_l1_artifact_is_not_a_symlink():
    assert not (REPO_ROOT / REAL["canonical_artifact_path"]).is_symlink()


def test_low_level_reader_refuses_a_final_component_symlink(tmp_path):
    """The repository's existing regular-non-symlink rule, now enforced BEFORE resolution.

    R4 evidence claimed this already held. It did not: the check ran after `resolve()` had
    followed the link, so it could never fire.
    """
    real = tmp_path / "runs" / "real.json"
    real.parent.mkdir(parents=True)
    real.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "petitgpt_reference_validation_exclusions",
            "hash_algorithm": "sha256-cleaned-text-utf8-v1",
            "hash_count": 1,
            "hashes": ["a" * 64],
        }),
        encoding="utf-8",
    )
    link = tmp_path / "runs" / "link.json"
    link.symlink_to(real)

    assert read_exclusion_artifact(tmp_path, "runs/real.json")["artifact_path"] == "runs/real.json"
    with pytest.raises(contract.StageMError, match="symlink"):
        read_exclusion_artifact(tmp_path, "runs/link.json")
    with pytest.raises(contract.StageMError, match="symlink"):
        canonical_repo_relative(tmp_path, "runs/link.json", label="probe")


def test_file_sha256_refuses_a_symlink_argument(tmp_path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"abc")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    assert file_sha256(real) == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(contract.StageMError, match="symlink"):
        file_sha256(link)


def test_symlink_refusal_scope_is_the_final_component(tmp_path):
    """Accurate scope, so the evidence does not overclaim.

    An intermediate directory symlink is resolved rather than refused; what protects the
    contract there is the canonical-path comparison, not the symlink check.
    """
    real_dir = tmp_path / "runs" / "real_dir"
    real_dir.mkdir(parents=True)
    payload = json.dumps({
        "schema_version": 1,
        "kind": "petitgpt_reference_validation_exclusions",
        "hash_algorithm": "sha256-cleaned-text-utf8-v1",
        "hash_count": 1,
        "hashes": ["a" * 64],
    })
    (real_dir / "a.json").write_text(payload, encoding="utf-8")
    (tmp_path / "runs" / "link_dir").symlink_to(real_dir, target_is_directory=True)

    # Readable through the directory symlink, and normalized to the REAL repository path.
    got = read_exclusion_artifact(tmp_path, "runs/link_dir/a.json")
    assert got["artifact_path"] == "runs/real_dir/a.json"
    assert got["artifact_path"] != "runs/link_dir/a.json"


def test_an_alternate_symlink_path_fails_the_producer_contract(m_run):
    """Contract layer, not reader layer: the serialized path is not the canonical path."""
    canonical = m_run["canonical_exclusion"]
    link_rel = "runs/l1_production_2026-08-20/reference_reserve_v1/linked_exclusions.json"
    link = m_run["tmp_path"] / link_rel
    link.symlink_to(Path(canonical["artifact_abs"]))
    plan = _plan(m_run)
    plan["resources"]["canonical_exclusion_authority"]["artifact_path"] = link_rel
    with pytest.raises(contract.StageMError):
        _authorize(m_run, plan)


# ------------------------------------------------------------------ preserved surfaces


def test_producer_gate_still_binds_the_implementation_bundle(m_run):
    context = _authorize(m_run)
    _files, digest = contract.m_implementation_bundle(m_run["tmp_path"])
    assert context.producer_gate["bundle_sha256"] == digest
    assert context.bundle_sha256 == digest


@pytest.mark.parametrize("member", list(contract.M_IMPLEMENTATION_BUNDLE_FILES))
def test_a_changed_m_module_byte_fails_the_fixed_plan(m_run, member):
    """Section 12: with candidate bytes fixed, changing any load-bearing M module fails."""
    target = m_run["tmp_path"] / member
    target.write_bytes(target.read_bytes() + b"\n# R5 probe\n")
    with pytest.raises(contract.StageMError):
        _authorize(m_run)
