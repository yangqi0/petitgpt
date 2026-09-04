"""Focused schema and dual-authorization tests for the Stage-O launch adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

ADAPTER_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = Path("/workspace/petitgpt")
STAGE_N_RESULT = Path(
    "/workspace/petitgpt_stage_n_result_publication_recovery_v1/"
    "runs/n3_bridge_output_r3_2026-09-04/STAGE_N_COMPLETE_RESULT.json"
)
LAUNCH_CONTRACT = Path(
    "/workspace/petitgpt/runs/"
    "n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
    "LAUNCH_CONTRACT.json"
)

sys.path.insert(0, str(ADAPTER_ROOT))
from tools import stage_o_successor_launch_adapter_v1 as A  # noqa: E402

_LIVE_IDENTITY_CACHE: tuple[dict[str, Any], dict[str, Any]] | None = None


@pytest.fixture
def reviewed_identity(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Model the post-commit reviewed state for pre-commit schema unit tests only."""

    global _LIVE_IDENTITY_CACHE
    if _LIVE_IDENTITY_CACHE is None:
        _LIVE_IDENTITY_CACHE = (A.adapter_identity(), A.accepted_trainer_identity())
    adapter, accepted = copy.deepcopy(_LIVE_IDENTITY_CACHE)
    adapter.update({"tracked_clean": True, "script_tracked": True})
    monkeypatch.setattr(A, "adapter_identity", lambda: copy.deepcopy(adapter))
    monkeypatch.setattr(A, "accepted_trainer_identity", lambda: copy.deepcopy(accepted))
    return adapter


def _set_nested(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = document
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = value


def _stage_o_document(
    adapter: dict[str, Any], trainer_argv: list[str], *, authorized: bool
) -> dict[str, Any]:
    return {
        "authorization_status": "AUTHORIZED" if authorized else "NOT_AUTHORIZED",
        "allowed_scope": "STAGE_O",
        "authorized_by": "Stage-O owner fixture" if authorized else None,
        "authorized_at": "2026-09-04T00:00:00Z" if authorized else None,
        "stage_o_launch_adapter_identity": {
            "head": adapter["head"],
            "adapter_tool_bundle_sha256": adapter["adapter_tool_bundle_sha256"],
            "adapter_tool_path": adapter["adapter_tool_path"],
            "adapter_tool_sha256": adapter["adapter_tool_sha256"],
        },
        "stage_o_trainer_argv": list(trainer_argv),
    }


def test_template_and_tracked_json_schema_have_the_same_strict_topology(
    reviewed_identity: dict[str, Any],
) -> None:
    template = A.adapter_authorization_template()
    schema_path = (
        ADAPTER_ROOT / "docs/stage_o_successor_launch_adapter_authorization_v1.schema.json"
    )
    schema = json.loads(schema_path.read_bytes())

    assert set(template) == set(schema["required"]) == set(schema["properties"])
    assert schema["$id"] == A.ADAPTER_AUTHORIZATION_SCHEMA
    assert schema["additionalProperties"] is False
    assert template["scope"] == "STAGE_O_SUCCESSOR_LAUNCH_ADAPTER"
    assert template["authorization_status"] == "NOT_AUTHORIZED"
    assert template["authorizes_adapter_execution"] is False
    assert template["authorizes_training"] is False
    assert template["authorized_by"] is None
    assert template["authorized_at"] is None
    assert template["num_workers"] == 2
    assert type(template["num_workers"]) is int
    assert template["stage_o_launch_adapter"]["adapter_tool_closure_count"] == 1
    assert template["stage_o_launch_adapter"]["adapter_tool_unbound_module_count"] == 0
    assert template["canonical_cwd"] == str(HISTORICAL_ROOT)


@pytest.mark.parametrize(
    ("path", "value", "failure"),
    [
        (("accepted_stage_o_trainer", "head"), "0" * 40, "accepted_stage_o_trainer"),
        (
            ("stage_o_launch_adapter", "adapter_tool_sha256"),
            "0" * 64,
            "stage_o_launch_adapter",
        ),
        (
            ("stage_o_launch_adapter", "adapter_tool_bundle_sha256"),
            "f" * 64,
            "stage_o_launch_adapter",
        ),
        (
            ("canonical_sources", "launch_contract", "path"),
            "/tmp/copied_launch_contract.py",
            "canonical_source",
        ),
        (("canonical_cwd",), "/tmp", "canonical_cwd"),
        (("module_names", "bare_launch_contract"), "other_name", "module_name"),
        (("stage_n_chain", "complete_result_sha256"), "0" * 64, "stage_n_chain"),
        (("num_workers",), True, "num_workers"),
        (("authorizes_training",), True, "must_never_authorize_training"),
        (
            ("stage_o_command_derivation", "exact_argv_match_required"),
            False,
            "command_derivation",
        ),
    ],
)
def test_each_reviewed_identity_family_fails_closed(
    reviewed_identity: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
    failure: str,
) -> None:
    document = A.adapter_authorization_template()
    _set_nested(document, path, value)

    verdict = A.validate_adapter_authorization(document)

    assert verdict["identity_valid"] is False
    assert any(failure in item for item in verdict["identity_failures"]), verdict
    assert verdict["authorized"] is False


def test_unknown_top_or_nested_authorization_fields_are_rejected(
    reviewed_identity: dict[str, Any],
) -> None:
    top = A.adapter_authorization_template()
    top["unreviewed"] = True
    nested = A.adapter_authorization_template()
    nested["accepted_stage_o_trainer"]["unreviewed"] = True

    top_verdict = A.validate_adapter_authorization(top)
    nested_verdict = A.validate_adapter_authorization(nested)

    assert "adapter_authorization_field_set_mismatch" in top_verdict["identity_failures"]
    assert "accepted_stage_o_trainer_identity_mismatch" in nested_verdict["identity_failures"]


def test_preflight_binding_authenticates_stage_o_path_bytes_identity_and_exact_argv(
    tmp_path: Path,
    reviewed_identity: dict[str, Any],
) -> None:
    trainer_argv = ["--run_plan_stage", "stage_b", "--num_workers", "2"]
    stage_o_document = _stage_o_document(reviewed_identity, trainer_argv, authorized=False)
    stage_o_path = tmp_path / "STAGE_O_NOT_AUTHORIZED.json"
    stage_o_path.write_bytes(A.canonical_json_bytes(stage_o_document))
    adapter_document = A.adapter_authorization_template(
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization_sha256=A.file_sha256(stage_o_path),
    )

    verdict = A.validate_adapter_authorization(
        adapter_document,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=stage_o_document,
        trainer_argv=trainer_argv,
    )

    assert verdict["identity_valid"] is True, verdict
    assert verdict["binding_failures"] == []
    assert verdict["owner_state_failures"] == [
        "adapter_authorization_status_not_authorized",
        "adapter_execution_not_authorized",
    ]
    assert verdict["authorized"] is False

    stage_o_path.write_bytes(stage_o_path.read_bytes() + b" ")
    drift = A.validate_adapter_authorization(
        adapter_document,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=stage_o_document,
        trainer_argv=trainer_argv,
    )
    assert "stage_o_authorization_sha256_mismatch" in drift["binding_failures"]

    wrong_identity = copy.deepcopy(stage_o_document)
    wrong_identity["stage_o_launch_adapter_identity"]["head"] = "0" * 40
    wrong = A.validate_adapter_authorization(
        adapter_document,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=wrong_identity,
        trainer_argv=trainer_argv,
    )
    assert "stage_o_authorization_adapter_identity_mismatch" in wrong["binding_failures"]


def test_execution_requires_two_authorities_but_adapter_never_authorizes_training(
    tmp_path: Path,
    reviewed_identity: dict[str, Any],
) -> None:
    runtime = json.loads(STAGE_N_RESULT.read_bytes())["runtime_fingerprint"]
    stage_o_path = tmp_path / "STAGE_O_AUTHORIZED.json"
    trainer_argv = [
        "--launch_contract_json",
        str(LAUNCH_CONTRACT),
        "--stage_authorization_json",
        str(stage_o_path),
        "--run_plan_stage",
        "stage_b",
        "--num_workers",
        "2",
    ]
    stage_o_document = _stage_o_document(reviewed_identity, trainer_argv, authorized=True)
    stage_o_path.write_bytes(A.canonical_json_bytes(stage_o_document))
    adapter_document = A.adapter_authorization_template(
        runtime_fingerprint=runtime,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization_sha256=A.file_sha256(stage_o_path),
    )
    adapter_document.update({
        "authorization_status": "AUTHORIZED",
        "authorizes_adapter_execution": True,
        "authorized_by": "Owner fixture",
        "authorized_at": "2026-09-04T00:00:00Z",
    })
    adapter_path = tmp_path / "ADAPTER_AUTHORIZED.json"
    adapter_path.write_bytes(A.canonical_json_bytes(adapter_document))

    verdict = A.validate_adapter_authorization(
        adapter_document,
        observed_runtime=runtime,
        require_execution=True,
        adapter_authorization_path=adapter_path,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=stage_o_document,
        trainer_argv=trainer_argv,
    )

    assert verdict["authorized"] is True, verdict
    assert verdict["failures"] == []
    assert adapter_document["authorizes_training"] is False

    adapter_document["authorizes_training"] = True
    refused = A.validate_adapter_authorization(
        adapter_document,
        observed_runtime=runtime,
        require_execution=True,
        adapter_authorization_path=adapter_path,
        stage_o_authorization_path=stage_o_path,
        stage_o_authorization=stage_o_document,
        trainer_argv=trainer_argv,
    )
    assert refused["authorized"] is False
    assert "adapter_authorization_must_never_authorize_training" in refused["identity_failures"]
