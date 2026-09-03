"""One-time Stage-N old-HEAD to successor-HEAD compatibility bridge.

This module encodes the exact incident-specific N3 policy.  It is deliberately not a
general cross-HEAD resume mechanism: the historical execution identity, plan, N1/N2
chain, runtime, checkpoint and completion boundary are constants.  A future external
owner authorization must bind one exact successor HEAD and its two bundles.  The module
never authorizes itself and :func:`n3_authorization_template` always returns
``NOT_AUTHORIZED``.

The public execution entry point has no caller-supplied executable hook.  Once separately
authorized it can only reopen the terminal N2 checkpoint, restore the closure-bound model,
optimizer, scaler and RNG implementation, realize one compiled production-shape forward,
construct a successor-bound checkpoint, prove zero-update state equivalence, and atomically
publish the bridge artifacts.  This segment creates and tests that future path but does not
execute it against the production checkpoint.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

_BRIDGE_MODULE_PATH = Path(__file__).resolve()
_SUCCESSOR_CODE_ROOT = _BRIDGE_MODULE_PATH.parent.parent
_SUCCESSOR_LAUNCH_PATH = (
    _SUCCESSOR_CODE_ROOT / "pretrain/production_launch_contract_v1.py"
).resolve()


def _load_exact_successor_launch_contract() -> Any:
    """Load the launch contract by reviewed bytes, never by the process CWD.

    The incident deliberately runs with ``cwd=/workspace/petitgpt`` while executing code
    from the successor worktree. A normal namespace-package import can therefore select
    the historical launch module. Reuse an already-loaded canonical module only when its
    origin is exact; otherwise load the successor file under a private, collision-free
    name. A conflicting canonical module is left untouched and can never become this
    bridge's launch authority.
    """

    canonical_name = "pretrain.production_launch_contract_v1"
    loaded = sys.modules.get(canonical_name)
    if loaded is not None:
        origin = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if origin == _SUCCESSOR_LAUNCH_PATH:
            return loaded

    private_name = "_petitgpt_successor_production_launch_contract_v1"
    loaded = sys.modules.get(private_name)
    if loaded is not None:
        origin = Path(str(getattr(loaded, "__file__", ""))).resolve()
        if origin != _SUCCESSOR_LAUNCH_PATH:
            raise RuntimeError("successor launch-contract module origin changed")
        return loaded
    spec = importlib.util.spec_from_file_location(private_name, _SUCCESSOR_LAUNCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create the exact successor launch-contract import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(private_name, None)
        raise
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    if origin != _SUCCESSOR_LAUNCH_PATH:
        raise RuntimeError("loaded launch-contract origin is not the successor worktree")
    return module


launch = _load_exact_successor_launch_contract()


class CompatibilityBridgeError(RuntimeError):
    """A fail-closed N3 compatibility-policy or publication failure."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise CompatibilityBridgeError(message)


N3_SCOPE = "STAGE_N_SUCCESSOR_HEAD_COMPATIBILITY_BRIDGE"
N3_AUTHORIZATION_SCHEMA = "petitgpt-stage-n-successor-head-compatibility-bridge-authorization-v1"
SEMANTIC_COMPARISON_SCHEMA = "petitgpt-successor-semantic-byte-comparison-v1"
N2_EVIDENCE_SCHEMA = "petitgpt-n2-v2-zero-update-invariance-v1"
BRIDGE_GRC_SCHEMA = "petitgpt-stage-n-successor-bridge-governed-run-contract-v1"
BRIDGE_RESULT_SCHEMA = "petitgpt-stage-n-successor-compatibility-result-v1"
BRIDGE_RESUME_EVIDENCE_SCHEMA = "petitgpt-stage-n-successor-resume-evidence-v1"
BRIDGE_RESUME_EVIDENCE_KIND = "STAGE_N_SUCCESSOR_COMPATIBILITY_BRIDGE_RESUME"
BRIDGE_CLOSURE_SCHEMA = "petitgpt-stage-n-successor-bridge-execution-bundle-v1"

# Immutable Stage-N execution history.  These values are policy, not defaults.
HISTORICAL_TRAINING_BRANCH = "agent/retrain-pipeline-contracts"
HISTORICAL_TRAINING_HEAD = launch.STAGE_N_SUCCESSOR_COMPATIBILITY_SOURCE_HEAD
HISTORICAL_TRAINING_BUNDLE_SHA256 = launch.STAGE_N_SUCCESSOR_COMPATIBILITY_SOURCE_BUNDLE_SHA256
SUCCESSOR_BRANCH = launch.STAGE_N_SUCCESSOR_COMPATIBILITY_BRANCH
SUCCESSOR_REPOSITORY_ROOT = Path("/workspace/petitgpt_stage_n_result_publication_recovery_v1")
HISTORICAL_UNTRACKED_PROBE_RELPATH = ".codex_r1_manual_context_probe.py"
HISTORICAL_UNTRACKED_PROBE_SHA256 = (
    "4ea1e8ef471138d9d9cf8076a6fd3bd83ce83c131287bd94c84d998b846ca76c"
)

EXACT_RUN_PLAN_SHA256 = "d673089447b4240ad7d5f7fd97dbf5d57567ad68bfffcc708a08f345fd25c117"
PILOT_OWNER_ACCEPTANCE_SHA256 = "ce5f0366f0f4f276b7ab802006930e3a01c605c023adab6317f0e17755079391"
N1_AUTHORIZATION_SHA256 = "5097f71eb32f8de41667dc274a5e67cb7f5626bc6f300ab7a7652d0e3a0d86d5"
N1_GRC_ARTIFACT_SHA256 = "3cb3f9d39bc78de154f19161293a4009881a9d569a0c51cdb283a833d9b1d977"
N1_GRC_SEMANTIC_SHA256 = "912947fda29fc5114a1b6b0f57983b6ccc6c4d3d4ad088b1aafed6ccc01f5639"
N1_BASE_GOVERNED_IDENTITY_SHA256 = (
    "d5c9f95aa1041f2aae438c8e91fe645b823e4eda5941d272d4ee914e711b9622"
)
N1_FINAL_CHECKPOINT_SHA256 = "529af1e0498fff14ecbda264a16b877c251699f894d57008fbcb3c32c87a70b9"
N1_EVIDENCE_MANIFEST_SHA256 = "95aaf3c5d6c7e2b96d4f8392b89070427f94af3936fbe6124ed25274e187559c"
N2_AUTHORIZATION_SHA256 = "05c79c535a3c94a86d5aa93e2baf6599bd9a2813929bd7b2ee6e182add3c63f2"
N2_GRC_ARTIFACT_SHA256 = "f6314d40c56b563fed6d89613e6f6b4cf702028259ea4baadbd61d9f9216a2d2"
N2_GRC_SEMANTIC_SHA256 = "aaa245475c99447223b08b2c1f13018e72faacf216915764980d00e4749857bf"
N2_SOURCE_CHECKPOINT_SHA256 = N1_FINAL_CHECKPOINT_SHA256
N2_TERMINAL_CHECKPOINT_SHA256 = (
    launch.STAGE_N_SUCCESSOR_COMPATIBILITY_SOURCE_N2_TERMINAL_CHECKPOINT_SHA256
)
N2_EVIDENCE_MANIFEST_SHA256 = "1721681356ae33d0c55b7788fff277377ca322df7caddd1545de3283a0f58cfd"
N2_ZERO_UPDATE_INVARIANCE_SHA256 = (
    "f99c729216025daa2f2987d9057c2c7c58c35f1be3b303da80290ccdc3a11095"
)
N2_RUNTIME_ARTIFACT_SHA256 = "7ba05ea1a491ce7a582a8c4bee9a918be49db6c1388cb0dbb33ee1bbd99b3607"
N2_PROCESS_EXIT_ARTIFACT_SHA256 = "418a5c17f33c70e99b0cc0a07fce69191489cfedc94164bfa903785777c5bd4b"
N2_FINAL_STATE_ARTIFACT_SHA256 = "1544465b5e29ea41257e218d774359e6d84a753678ee90cac22b3ac8c9ec98e4"
N2_LAUNCH_RECORD_ARTIFACT_SHA256 = (
    "e4d67563ef15a6e67cb9e85dc45c8b63d48c57bbdb92af8ab351758db54e5f45"
)
HISTORICAL_RUNTIME_FINGERPRINT_SHA256 = (
    "485f8ce78daea4d3254a6c44dd037d21fea55dd8a258ebd43e21f1cd9045cf26"
)

COMPLETION_STAGE = "stage_a"
COMPLETION_STEP = 38_146
COMPLETION_SAMPLER_ENDPOINT = 4_882_688
COMPLETION_SAMPLER_SEED = 20_260_832
EXPECTED_MODEL_TENSOR_COUNT = 213


def _assert_successor_module_origins() -> None:
    """Fail closed unless all already-loaded bridge authorities are successor bytes."""

    expected_root = SUCCESSOR_REPOSITORY_ROOT.resolve()
    expected_bridge = (
        expected_root / "pretrain/stage_n_successor_head_compatibility_bridge_v1.py"
    ).resolve()
    expected_launch = (expected_root / "pretrain/production_launch_contract_v1.py").resolve()
    _require(_BRIDGE_MODULE_PATH == expected_bridge, "bridge module origin is not successor root")
    _require(
        Path(str(getattr(launch, "__file__", ""))).resolve() == expected_launch,
        "launch-contract module origin is not successor root",
    )
    _require(
        Path(str(getattr(launch, "ROOT", ""))).resolve() == expected_root,
        "launch-contract repository root is not successor root",
    )


def _load_exact_successor_module(private_name: str, relative_path: str) -> Any:
    """Load one realization dependency from the reviewed successor path."""

    _assert_successor_module_origins()
    expected = (SUCCESSOR_REPOSITORY_ROOT / relative_path).resolve()
    loaded = sys.modules.get(private_name)
    if loaded is not None:
        _require(
            Path(str(getattr(loaded, "__file__", ""))).resolve() == expected,
            f"successor dependency module origin changed:{relative_path}",
        )
        return loaded
    spec = importlib.util.spec_from_file_location(private_name, expected)
    _require(
        spec is not None and spec.loader is not None,
        f"cannot create successor dependency import:{relative_path}",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(private_name, None)
        raise
    _require(
        Path(str(getattr(module, "__file__", ""))).resolve() == expected,
        f"loaded dependency origin is not successor root:{relative_path}",
    )
    return module


EXECUTION_COUNTER_FIELDS = (
    "optimizer_updates",
    "trained_tokens",
    "sampler_advances",
    "training_loop_iterations",
    "backward_calls",
    "scheduler_advances",
    "data_batches_consumed",
)
ZERO_EXECUTION_COUNTERS: Mapping[str, int] = {field: 0 for field in EXECUTION_COUNTER_FIELDS}

# The bridge changes only the invocation/compile authority envelope. Every other checkpoint
# field -- including fields unknown to this implementation -- remains byte/value equivalent.
_ALLOWED_TOP_LEVEL_CHECKPOINT_CHANGES = frozenset({
    "governed_run_contract",
    "governed_run_contract_sha256",
})
_ALLOWED_DYNAMIC_CHECKPOINT_CHANGES = frozenset({"compile_evidence", "compile_evidence_sha256"})

CANONICAL_TRAINING_ROOT = Path("/workspace/petitgpt")
N1_INVOCATION = (
    CANONICAL_TRAINING_ROOT
    / "outputs/governed/stage_a_5097f71eb32f8de41667dc274a5e67cb7f5626bc6f300ab7a7652d0e3a0d86d5"
)
N2_INVOCATION = (
    CANONICAL_TRAINING_ROOT
    / "outputs/governed/stage_a_05c79c535a3c94a86d5aa93e2baf6599bd9a2813929bd7b2ee6e182add3c63f2"
)

HISTORICAL_ARTIFACTS: Mapping[str, Mapping[str, str]] = {
    "exact_plan": {
        "path": str(CANONICAL_TRAINING_ROOT / launch.EXACT_RUN_PLAN_RELPATH),
        "sha256": EXACT_RUN_PLAN_SHA256,
    },
    "pilot_acceptance": {
        "path": str(CANONICAL_TRAINING_ROOT / launch.PILOT_OWNER_ACCEPTANCE_RELPATH),
        "sha256": PILOT_OWNER_ACCEPTANCE_SHA256,
    },
    "n1_authorization": {
        "path": str(
            CANONICAL_TRAINING_ROOT / "runs/n_stage_n_n1_owner_authorization_2026-09-01/"
            "OWNER_AUTHORIZATION_STAGE_N_N1.json"
        ),
        "sha256": N1_AUTHORIZATION_SHA256,
    },
    "n1_governed_run_contract": {
        "path": str(N1_INVOCATION / "GOVERNED_RUN_CONTRACT.json"),
        "sha256": N1_GRC_ARTIFACT_SHA256,
    },
    "n1_final_checkpoint": {
        "path": str(N1_INVOCATION / "step_038146.pt"),
        "sha256": N1_FINAL_CHECKPOINT_SHA256,
    },
    "n1_evidence_manifest": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n1_execution_2026-09-01/EVIDENCE_SHA256SUMS.txt"
        ),
        "sha256": N1_EVIDENCE_MANIFEST_SHA256,
    },
    "n2_authorization": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
            "OWNER_AUTHORIZATION_STAGE_N_N2_CORRECTED_V2.json"
        ),
        "sha256": N2_AUTHORIZATION_SHA256,
    },
    "n2_governed_run_contract": {
        "path": str(N2_INVOCATION / "GOVERNED_RUN_CONTRACT.json"),
        "sha256": N2_GRC_ARTIFACT_SHA256,
    },
    "n2_source_checkpoint": {
        "path": str(N1_INVOCATION / "step_038146.pt"),
        "sha256": N2_SOURCE_CHECKPOINT_SHA256,
    },
    "n2_terminal_checkpoint": {
        "path": str(N2_INVOCATION / "step_038146.pt"),
        "sha256": N2_TERMINAL_CHECKPOINT_SHA256,
    },
    "n2_evidence_manifest": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
            "EVIDENCE_SHA256SUMS.txt"
        ),
        "sha256": N2_EVIDENCE_MANIFEST_SHA256,
    },
    "n2_zero_update_invariance": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
            "evidence/N2_V2_ZERO_UPDATE_INVARIANCE.json"
        ),
        "sha256": N2_ZERO_UPDATE_INVARIANCE_SHA256,
    },
    "n2_runtime_fingerprint": {
        "path": str(N2_INVOCATION / "STAGE_N_RUNTIME_FINGERPRINT.json"),
        "sha256": N2_RUNTIME_ARTIFACT_SHA256,
    },
    "n2_process_exit": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
            "evidence/n2v2_exit_code.txt"
        ),
        "sha256": N2_PROCESS_EXIT_ARTIFACT_SHA256,
    },
    "n2_final_state": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
            "evidence/N2_V2_FINAL_STATE.json"
        ),
        "sha256": N2_FINAL_STATE_ARTIFACT_SHA256,
    },
    "n2_launch_record": {
        "path": str(
            CANONICAL_TRAINING_ROOT
            / "runs/n_stage_n_n2_corrected_owner_authorization_and_verification_v2_2026-09-03/"
            "evidence/N2_V2_LAUNCH_RECORD.json"
        ),
        "sha256": N2_LAUNCH_RECORD_ARTIFACT_SHA256,
    },
}

# These hardware/software facts may not drift.  The source and successor code identity is
# validated separately because it is the one reviewed difference this bridge exists to make.
RUNTIME_INVARIANTS: Mapping[str, Any] = {
    "canonical_cwd": "/workspace/petitgpt",
    "compute_capability": "8.9",
    "cuda_runtime_version": "12.6",
    "driver_version": "580.126.20",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_pci_bus_id": "00000000:02:00.0",
    "gpu_uuid": "GPU-4bfe696d-0d99-97e6-c139-4683cc99eee6",
    "num_workers": 2,
    "numpy_version": "2.2.6",
    "python_version": "3.10.12",
    "torch_version": "2.11.0+cu126",
    "total_vram_bytes": 25_250_627_584,
    "visible_cuda_device_count": 1,
    "selected_physical_index": 0,
}

# The launch contract is the sole permitted historical-closure difference.  The other
# eleven members are the production-training semantics whose bytes must remain identical.
CORE_TRAINING_SEMANTICS_FILES = (
    "pretrain/dataset_pretrain.py",
    "pretrain/run_plan_contract.py",
    "pretrain/sample.py",
    "pretrain/train_pretrain_with_bench.py",
    "src/__init__.py",
    "src/canonical_loss.py",
    "src/canonical_schedule.py",
    "src/model.py",
    "src/optim.py",
    "src/special_tokens.py",
    "src/tracking.py",
)
SUCCESSOR_POLICY_FILE = "pretrain/production_launch_contract_v1.py"
HISTORICAL_GOVERNED_CLOSURE_FILES = tuple(
    sorted((*CORE_TRAINING_SEMANTICS_FILES, SUCCESSOR_POLICY_FILE))
)

ZERO_UPDATE_LIMITS: Mapping[str, Any] = {
    "start_step": COMPLETION_STEP,
    "stop_step": COMPLETION_STEP,
    "max_steps": COMPLETION_STEP,
    "expected_optimizer_updates": 0,
    "expected_trained_tokens": 0,
    "expected_sampler_advances": 0,
    "expected_training_loop_iterations": 0,
    "backward_permitted": False,
    "optimizer_step_permitted": False,
    "scheduler_advancement_permitted": False,
    "data_consumption_permitted": False,
    "sampler_advancement_permitted": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_hex(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _artifact_chain_document() -> dict[str, dict[str, str]]:
    return {name: dict(binding) for name, binding in HISTORICAL_ARTIFACTS.items()}


def _destination_document(output_root: str | Path | None) -> dict[str, Any]:
    if output_root is None:
        return {
            "output_root": None,
            "governed_run_contract_path": None,
            "terminal_checkpoint_path": None,
            "runtime_fingerprint_path": None,
            "smoke_evidence_path": None,
            "resume_evidence_path": None,
            "smoke_result_path": None,
            "resume_result_path": None,
            "complete_result_path": None,
            "expected_checkpoint_step": COMPLETION_STEP,
        }
    root = Path(output_root)
    return {
        "output_root": str(root),
        "governed_run_contract_path": str(root / "GOVERNED_RUN_CONTRACT.json"),
        "terminal_checkpoint_path": str(root / "step_038146.pt"),
        "runtime_fingerprint_path": str(root / launch.STAGE_N_RUNTIME_FILENAME),
        "smoke_evidence_path": str(root / "STAGE_N_BRIDGE_COMPILE_EVIDENCE.json"),
        "resume_evidence_path": str(root / "STAGE_N_BRIDGE_STATE_EQUIVALENCE.json"),
        "smoke_result_path": str(root / "STAGE_N_SMOKE_RESULT.json"),
        "resume_result_path": str(root / "STAGE_N_RESUME_RESULT.json"),
        "complete_result_path": str(root / "STAGE_N_COMPLETE_RESULT.json"),
        "expected_checkpoint_step": COMPLETION_STEP,
    }


def n3_authorization_template(
    *,
    successor_head: str | None = None,
    successor_trainer_bundle_sha256: str | None = None,
    bridge_tool_bundle_sha256: str | None = None,
    successor_runtime_fingerprint_sha256: str | None = None,
    semantic_comparison_manifest_path: str | Path | None = None,
    semantic_comparison_manifest_sha256: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fully shaped candidate that can never authorize N3 execution."""
    return {
        "schema_version": N3_AUTHORIZATION_SCHEMA,
        "scope": N3_SCOPE,
        "authorization_status": "NOT_AUTHORIZED",
        "authorizes_bridge_execution": False,
        "authorizes_training": False,
        "one_time_exception": True,
        "source_execution": {
            "branch": HISTORICAL_TRAINING_BRANCH,
            "head": HISTORICAL_TRAINING_HEAD,
            "trainer_execution_bundle_sha256": HISTORICAL_TRAINING_BUNDLE_SHA256,
        },
        "successor": {
            "branch": SUCCESSOR_BRANCH,
            "repository_root": str(SUCCESSOR_REPOSITORY_ROOT),
            "head": successor_head,
            "trainer_execution_bundle_sha256": successor_trainer_bundle_sha256,
            "bridge_tool_bundle_sha256": bridge_tool_bundle_sha256,
            "runtime_fingerprint_sha256": successor_runtime_fingerprint_sha256,
        },
        "semantic_comparison_manifest": {
            "path": (
                str(semantic_comparison_manifest_path)
                if semantic_comparison_manifest_path is not None
                else None
            ),
            "sha256": semantic_comparison_manifest_sha256,
        },
        "history": {
            "exact_run_plan_sha256": EXACT_RUN_PLAN_SHA256,
            "pilot_owner_acceptance_sha256": PILOT_OWNER_ACCEPTANCE_SHA256,
            "n1_governed_run_contract_semantic_sha256": N1_GRC_SEMANTIC_SHA256,
            "n1_base_governed_identity_sha256": N1_BASE_GOVERNED_IDENTITY_SHA256,
            "n2_governed_run_contract_semantic_sha256": N2_GRC_SEMANTIC_SHA256,
            "source_runtime_fingerprint_sha256": HISTORICAL_RUNTIME_FINGERPRINT_SHA256,
            "artifacts": _artifact_chain_document(),
        },
        "runtime_invariants": dict(RUNTIME_INVARIANTS),
        "source_checkpoint": {
            **dict(HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]),
            "step": COMPLETION_STEP,
            "stage": COMPLETION_STAGE,
            "sampler_cursor": COMPLETION_SAMPLER_ENDPOINT,
            "sampler_seed": COMPLETION_SAMPLER_SEED,
        },
        "destination": _destination_document(output_root),
        "zero_update": dict(ZERO_UPDATE_LIMITS),
        "authorized_by": None,
        "authorized_at": None,
        "note": (
            "Candidate only. A future owner-authored artifact must change status to "
            "AUTHORIZED and bind one exact successor identity, manifest and output root."
        ),
    }


def _manifest_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "manifest_content_sha256"}


def build_semantic_comparison_manifest(
    source_root: str | Path,
    successor_root: str | Path,
    *,
    successor_head: str,
    successor_trainer_bundle_sha256: str,
) -> dict[str, Any]:
    """Compare the exact twelve-member historical closure byte-for-byte."""
    source = Path(source_root)
    successor = Path(successor_root)
    records: list[dict[str, Any]] = []
    for relative in HISTORICAL_GOVERNED_CLOSURE_FILES:
        source_path = source / relative
        successor_path = successor / relative
        _require(source_path.is_file(), f"historical comparison source missing: {source_path}")
        _require(successor_path.is_file(), f"successor comparison source missing: {successor_path}")
        source_sha = file_sha256(source_path)
        successor_sha = file_sha256(successor_path)
        records.append({
            "path": relative,
            "source_sha256": source_sha,
            "successor_sha256": successor_sha,
            "byte_identical": source_sha == successor_sha,
        })
    changed = [record["path"] for record in records if not record["byte_identical"]]
    document: dict[str, Any] = {
        "schema_version": SEMANTIC_COMPARISON_SCHEMA,
        "source_head": HISTORICAL_TRAINING_HEAD,
        "source_trainer_execution_bundle_sha256": HISTORICAL_TRAINING_BUNDLE_SHA256,
        "successor_head": successor_head,
        "successor_trainer_execution_bundle_sha256": successor_trainer_bundle_sha256,
        "exact_run_plan_sha256": EXACT_RUN_PLAN_SHA256,
        "pilot_owner_acceptance_sha256": PILOT_OWNER_ACCEPTANCE_SHA256,
        "files": records,
        "file_count": len(records),
        "changed_files": changed,
        "changed_file_count": len(changed),
        "core_training_semantics_files_changed": [
            path for path in changed if path in CORE_TRAINING_SEMANTICS_FILES
        ],
        "semantic_isolation_pass": changed == [SUCCESSOR_POLICY_FILE],
    }
    document["manifest_content_sha256"] = sha256_bytes(
        canonical_json_bytes(_manifest_projection(document))
    )
    return document


def validate_semantic_comparison_manifest(
    document: Mapping[str, Any] | None,
    *,
    expected_artifact_sha256: str,
    expected_successor_head: str,
    expected_successor_trainer_bundle_sha256: str,
    source_root: str | Path | None = None,
    successor_root: str | Path | None = None,
) -> list[str]:
    """Validate both manifest claims and, when roots are supplied, the compared bytes."""
    if not isinstance(document, Mapping):
        return ["semantic_comparison_manifest_missing_or_malformed"]
    failures: list[str] = []
    expected_document_fields = {
        "schema_version",
        "source_head",
        "source_trainer_execution_bundle_sha256",
        "successor_head",
        "successor_trainer_execution_bundle_sha256",
        "exact_run_plan_sha256",
        "pilot_owner_acceptance_sha256",
        "files",
        "file_count",
        "changed_files",
        "changed_file_count",
        "core_training_semantics_files_changed",
        "semantic_isolation_pass",
        "manifest_content_sha256",
    }
    if set(document) != expected_document_fields:
        failures.append("semantic_comparison_manifest_field_set_mismatch")
    exact = (
        ("schema_version", SEMANTIC_COMPARISON_SCHEMA),
        ("source_head", HISTORICAL_TRAINING_HEAD),
        ("source_trainer_execution_bundle_sha256", HISTORICAL_TRAINING_BUNDLE_SHA256),
        ("successor_head", expected_successor_head),
        (
            "successor_trainer_execution_bundle_sha256",
            expected_successor_trainer_bundle_sha256,
        ),
        ("exact_run_plan_sha256", EXACT_RUN_PLAN_SHA256),
        ("pilot_owner_acceptance_sha256", PILOT_OWNER_ACCEPTANCE_SHA256),
        ("file_count", len(HISTORICAL_GOVERNED_CLOSURE_FILES)),
        ("changed_file_count", 1),
        ("changed_files", [SUCCESSOR_POLICY_FILE]),
        ("core_training_semantics_files_changed", []),
        ("semantic_isolation_pass", True),
    )
    for field, expected in exact:
        if not _exact_state_equal(document.get(field), expected):
            failures.append(f"semantic_comparison_mismatch:{field}")

    content_sha = sha256_bytes(canonical_json_bytes(_manifest_projection(document)))
    if document.get("manifest_content_sha256") != content_sha:
        failures.append("semantic_comparison_manifest_content_sha256_mismatch")
    artifact_sha = sha256_bytes(canonical_json_bytes(dict(document)))
    if artifact_sha != expected_artifact_sha256:
        failures.append("semantic_comparison_manifest_artifact_sha256_mismatch")

    records = document.get("files")
    if not isinstance(records, list):
        failures.append("semantic_comparison_files_missing_or_malformed")
        return list(dict.fromkeys(failures))
    expected_paths = list(HISTORICAL_GOVERNED_CLOSURE_FILES)
    actual_paths = [record.get("path") for record in records if isinstance(record, Mapping)]
    if actual_paths != expected_paths or len(records) != len(expected_paths):
        failures.append("semantic_comparison_file_set_mismatch")
    by_path = {str(record.get("path")): record for record in records if isinstance(record, Mapping)}
    expected_record_fields = {
        "path",
        "source_sha256",
        "successor_sha256",
        "byte_identical",
    }
    for relative in expected_paths:
        record = by_path.get(relative)
        if record is None:
            continue
        if set(record) != expected_record_fields:
            failures.append(f"semantic_comparison_record_field_set_mismatch:{relative}")
        for field in ("source_sha256", "successor_sha256"):
            if not _valid_hex(record.get(field), 64):
                failures.append(f"semantic_comparison_invalid_digest:{relative}:{field}")
        expected_identical = relative != SUCCESSOR_POLICY_FILE
        if record.get("byte_identical") is not expected_identical:
            failures.append(f"semantic_comparison_identity_claim_mismatch:{relative}")
        if (record.get("source_sha256") == record.get("successor_sha256")) is not bool(
            record.get("byte_identical")
        ):
            failures.append(f"semantic_comparison_digest_claim_mismatch:{relative}")

        if source_root is not None and successor_root is not None:
            source_path = Path(source_root) / relative
            successor_path = Path(successor_root) / relative
            if not source_path.is_file() or not successor_path.is_file():
                failures.append(f"semantic_comparison_file_not_found:{relative}")
            else:
                if file_sha256(source_path) != record.get("source_sha256"):
                    failures.append(f"semantic_comparison_source_bytes_mismatch:{relative}")
                if file_sha256(successor_path) != record.get("successor_sha256"):
                    failures.append(f"semantic_comparison_successor_bytes_mismatch:{relative}")
    return list(dict.fromkeys(failures))


def _completion_boundary_values(boundary: Any) -> tuple[Any, Any]:
    if isinstance(boundary, Mapping):
        stage_a = boundary.get("stage_a")
        if isinstance(stage_a, Mapping):
            return (
                stage_a.get("stop_step", stage_a.get("final_step")),
                stage_a.get("sampler_endpoint", stage_a.get("range_stop_position")),
            )
        return (
            boundary.get(
                "stop_step",
                boundary.get("stage_a_stop_step", boundary.get("expected_final_step")),
            ),
            boundary.get(
                "sampler_endpoint",
                boundary.get("stage_a_sampler_endpoint", boundary.get("expected_sampler_endpoint")),
            ),
        )
    return None, None


def validate_completion_boundary(
    exact_plan: Mapping[str, Any] | str | Path,
    *,
    authorization: Mapping[str, Any] | None = None,
) -> list[str]:
    """Use the launch contract's one shared pure plan-boundary derivation."""
    try:
        derived = (
            launch.derive_stage_n_completion_boundary(exact_plan)
            if isinstance(exact_plan, Mapping)
            else launch.load_stage_n_completion_boundary(exact_plan)
        )
    except (OSError, ValueError, TypeError, launch.LaunchContractError) as exc:
        return [f"completion_boundary_derivation_failed:{type(exc).__name__}"]
    stop_step, sampler_endpoint = _completion_boundary_values(derived)
    failures: list[str] = []
    if stop_step != COMPLETION_STEP:
        failures.append("completion_boundary_step_mismatch")
    if sampler_endpoint != COMPLETION_SAMPLER_ENDPOINT:
        failures.append("completion_boundary_sampler_endpoint_mismatch")
    if authorization is not None:
        failures.extend(launch.validate_stage_n_completion_cross_check(authorization, derived))
    return list(dict.fromkeys(failures))


def validate_existing_n2_evidence(document: Mapping[str, Any] | None) -> list[str]:
    """Validate the accepted N2 invariance artifact without loading its checkpoint."""
    if not isinstance(document, Mapping):
        return ["n2_evidence_missing_or_malformed"]
    failures: list[str] = []
    if document.get("schema_version") != N2_EVIDENCE_SCHEMA:
        failures.append("n2_evidence_schema_mismatch")
    for field in (
        "n2_optimizer_step_calls",
        "n2_trained_tokens",
        "n2_sampler_advances",
        "n2_training_loop_iterations",
    ):
        if not _exact_state_equal(document.get(field), 0):
            failures.append(f"n2_evidence_nonzero:{field}")
    if document.get("N2_ZERO_UPDATE_STATUS") != "VERIFIED":
        failures.append("n2_zero_update_status_not_verified")
    if document.get("terminal_compile_evidence_verifies") is not True:
        failures.append("n2_compile_evidence_not_verified")

    source = document.get("source_checkpoint")
    terminal = document.get("terminal_checkpoint")
    expected_source = HISTORICAL_ARTIFACTS["n2_source_checkpoint"]
    expected_terminal = HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]
    if not isinstance(source, Mapping) or any(
        source.get(field) != expected_source[field] for field in ("path", "sha256")
    ):
        failures.append("n2_source_checkpoint_binding_mismatch")
    if not isinstance(terminal, Mapping) or any(
        terminal.get(field) != expected_terminal[field] for field in ("path", "sha256")
    ):
        failures.append("n2_terminal_checkpoint_binding_mismatch")

    state = document.get("state_invariance")
    if not isinstance(state, Mapping):
        failures.append("n2_state_invariance_missing_or_malformed")
    else:
        for field in (
            "base_governed_identity_identical",
            "checkpoint_milestone_prefix_unchanged",
            "evaluation_milestone_prefix_unchanged",
            "model_parameters_bitwise_identical",
            "optimizer_param_groups_identical",
            "optimizer_state_identical",
            "permutation_identity_unchanged",
            "range_start_unchanged",
            "range_stop_unchanged",
            "sampler_seed_unchanged",
            "scaler_identical",
            "trained_tokens_unchanged",
        ):
            if state.get(field) is not True:
                failures.append(f"n2_state_invariance_not_true:{field}")
        if not _exact_state_equal(
            state.get("model_tensors_compared"),
            EXPECTED_MODEL_TENSOR_COUNT,
        ):
            failures.append("n2_model_tensor_count_mismatch")
        if not _exact_state_equal(state.get("model_max_abs_diff"), 0.0):
            failures.append("n2_model_state_difference")
        for field, expected in (
            ("global_step", COMPLETION_STEP),
            ("sampler_cursor", COMPLETION_SAMPLER_ENDPOINT),
        ):
            values = state.get(field)
            if not (
                isinstance(values, Mapping)
                and _exact_state_equal(values.get("source"), expected)
                and _exact_state_equal(values.get("terminal"), expected)
                and values.get("unchanged") is True
            ):
                failures.append(f"n2_state_invariance_mismatch:{field}")

    rng = document.get("rng_state")
    if not isinstance(rng, Mapping) or rng.get("all_streams_identical") is not True:
        failures.append("n2_rng_state_not_identical")
    else:
        streams = rng.get("per_stream_identical")
        for name in ("python", "numpy", "torch_cpu", "torch_cuda"):
            if not isinstance(streams, Mapping) or streams.get(name) is not True:
                failures.append(f"n2_rng_stream_not_identical:{name}")
    if document.get("n2_governed_run_contract_artifact_sha256") != N2_GRC_ARTIFACT_SHA256:
        failures.append("n2_governed_run_contract_artifact_sha256_mismatch")
    if document.get("n2_governed_run_contract_digest") != N2_GRC_SEMANTIC_SHA256:
        failures.append("n2_governed_run_contract_semantic_sha256_mismatch")
    return list(dict.fromkeys(failures))


def _exact_state_equal(left: Any, right: Any) -> bool:
    try:
        import torch

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and tuple(left.shape) == tuple(right.shape)
                and left.layout == right.layout == torch.strided
            ):
                return False
            try:
                left_bytes = left.detach().cpu().contiguous().view(torch.uint8)
                right_bytes = right.detach().cpu().contiguous().view(torch.uint8)
                return bool(torch.equal(left_bytes, right_bytes))
            except (RuntimeError, TypeError, ValueError):
                return False
    except ImportError:  # pragma: no cover - real bridge requires torch
        pass
    try:
        import numpy as np

        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(
                isinstance(left, np.ndarray)
                and isinstance(right, np.ndarray)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and left.tobytes(order="C") == right.tobytes(order="C")
            )
    except ImportError:  # pragma: no cover - governed runtime requires NumPy
        pass
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _exact_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_exact_state_equal(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:  # noqa: BLE001 - opaque or executable equality fails closed
        return False
    return result if isinstance(result, bool) else bool(result)


def _model_tensor_count(model_state: Any) -> int:
    if not isinstance(model_state, Mapping):
        return 0
    try:
        import torch

        return sum(isinstance(value, torch.Tensor) for value in model_state.values())
    except ImportError:  # pragma: no cover
        return len(model_state)


def validate_state_equivalence(
    source_checkpoint: Mapping[str, Any] | None,
    destination_checkpoint: Mapping[str, Any] | None,
    *,
    execution_counters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prove that N3 changed only its exact authority and compile-evidence envelope.

    This is deliberately an all-fields comparison. Unknown checkpoint/dynamic-state fields
    are not ignored: their key sets and values must be identical. The two explicit
    allowlists above are the complete exception surface.
    """
    if not isinstance(source_checkpoint, Mapping) or not isinstance(
        destination_checkpoint, Mapping
    ):
        return {
            "equivalent": False,
            "failures": ["checkpoint_missing_or_malformed"],
            "model_tensors_compared": 0,
        }
    failures: list[str] = []
    counters = execution_counters if isinstance(execution_counters, Mapping) else {}
    if set(counters) != set(EXECUTION_COUNTER_FIELDS):
        failures.append("bridge_execution_counter_set_mismatch")
    for field in EXECUTION_COUNTER_FIELDS:
        if type(counters.get(field)) is not int or counters.get(field) != 0:
            failures.append(f"bridge_nonzero_counter:{field}")

    required_top_level = (
        "model",
        "optim",
        "scaler",
        "global_step",
        "run_contract",
        "data_contract",
        "rng_state",
        "position_stats",
        "data_sampler",
        "governed_checkpoint_state",
    )
    for field in required_top_level:
        if field not in source_checkpoint or field not in destination_checkpoint:
            failures.append(f"checkpoint_missing_state_field:{field}")

    if set(source_checkpoint) != set(destination_checkpoint):
        failures.append("bridge_checkpoint_field_set_difference")
    for field in sorted(set(source_checkpoint) & set(destination_checkpoint)):
        if field in _ALLOWED_TOP_LEVEL_CHECKPOINT_CHANGES or field == ("governed_checkpoint_state"):
            continue
        if not _exact_state_equal(source_checkpoint[field], destination_checkpoint[field]):
            failures.append(f"bridge_state_difference:{field}")

    source_state = source_checkpoint.get("governed_checkpoint_state")
    destination_state = destination_checkpoint.get("governed_checkpoint_state")
    if not isinstance(source_state, Mapping) or not isinstance(destination_state, Mapping):
        failures.append("governed_checkpoint_state_missing_or_malformed")
    else:
        if set(source_state) != set(destination_state):
            failures.append("bridge_dynamic_state_field_set_difference")
        for field in sorted(set(source_state) & set(destination_state)):
            if field in _ALLOWED_DYNAMIC_CHECKPOINT_CHANGES:
                continue
            if not _exact_state_equal(source_state[field], destination_state[field]):
                failures.append(f"bridge_dynamic_state_difference:{field}")
        exact_boundary = {
            "active_stage": COMPLETION_STAGE,
            "active_stage_sampler_seed": COMPLETION_SAMPLER_SEED,
            "range_start_position": 0,
            "invocation_range_start_position": COMPLETION_SAMPLER_ENDPOINT,
            "cursor": COMPLETION_SAMPLER_ENDPOINT,
            "range_stop_position": COMPLETION_SAMPLER_ENDPOINT,
            "global_step": COMPLETION_STEP,
        }
        for field, expected in exact_boundary.items():
            if source_state.get(field) != expected or destination_state.get(field) != expected:
                failures.append(f"bridge_completion_boundary_mismatch:{field}")

    if (
        source_checkpoint.get("global_step") != COMPLETION_STEP
        or destination_checkpoint.get("global_step") != COMPLETION_STEP
    ):
        failures.append("bridge_global_step_mismatch")
    model_tensors = _model_tensor_count(source_checkpoint.get("model"))
    if model_tensors != EXPECTED_MODEL_TENSOR_COUNT:
        failures.append("bridge_model_tensor_count_mismatch")
    return {
        "schema_version": "petitgpt-stage-n-successor-state-equivalence-v1",
        "equivalent": not failures,
        "failures": list(dict.fromkeys(failures)),
        "model_tensors_compared": model_tensors,
        "optimizer_state_equivalent": "bridge_state_difference:optim" not in failures,
        "scaler_state_equivalent": "bridge_state_difference:scaler" not in failures,
        "rng_state_preserved": not any("rng_state" in failure for failure in failures),
        "global_step": COMPLETION_STEP,
        "sampler_cursor": COMPLETION_SAMPLER_ENDPOINT,
    }


def _validate_bound_artifacts(artifacts: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(artifacts, Mapping):
        return ["historical_artifact_chain_missing_or_malformed"]
    failures: list[str] = []
    if set(artifacts) != set(HISTORICAL_ARTIFACTS):
        failures.append("historical_artifact_chain_set_mismatch")
    for name, expected in HISTORICAL_ARTIFACTS.items():
        actual = artifacts.get(name)
        if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
            failures.append(f"historical_artifact_binding_mismatch:{name}")
    return failures


def verify_bound_artifact_bytes(artifacts: Mapping[str, Any] | None) -> list[str]:
    """Reopen every bound source artifact and hash its current bytes."""
    failures = _validate_bound_artifacts(artifacts)
    if failures or not isinstance(artifacts, Mapping):
        return failures
    for name, binding in artifacts.items():
        path = Path(str(binding["path"]))
        if not path.is_file():
            failures.append(f"historical_artifact_not_found:{name}")
        elif file_sha256(path) != binding["sha256"]:
            failures.append(f"historical_artifact_sha256_mismatch:{name}")

    exit_path = Path(str(artifacts["n2_process_exit"]["path"]))
    if exit_path.is_file() and exit_path.read_bytes() != b"EXIT=0\n":
        failures.append("n2_process_exit_not_zero")

    def json_artifact(name: str) -> Mapping[str, Any] | None:
        path = Path(str(artifacts[name]["path"]))
        if not path.is_file():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            failures.append(f"historical_artifact_not_valid_json:{name}")
            return None
        if not isinstance(document, Mapping):
            failures.append(f"historical_artifact_not_json_object:{name}")
            return None
        return document

    def verify_evidence_manifest(name: str) -> None:
        manifest_path = Path(str(artifacts[name]["path"]))
        if not manifest_path.is_file():
            return
        root = manifest_path.parent.resolve()
        try:
            rows = manifest_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            failures.append(f"historical_evidence_manifest_unreadable:{name}")
            return
        for index, row in enumerate(rows, start=1):
            pieces = row.split("  ", 1)
            if len(pieces) != 2 or not _valid_hex(pieces[0], 64):
                failures.append(f"historical_evidence_manifest_malformed:{name}:{index}")
                continue
            relative = pieces[1]
            target = (root / relative).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                failures.append(f"historical_evidence_manifest_path_escape:{name}:{index}")
                continue
            if not target.is_file():
                failures.append(f"historical_evidence_manifest_member_missing:{name}:{index}")
            elif file_sha256(target) != pieces[0]:
                failures.append(f"historical_evidence_manifest_member_mismatch:{name}:{index}")

    verify_evidence_manifest("n1_evidence_manifest")
    verify_evidence_manifest("n2_evidence_manifest")

    final_state = json_artifact("n2_final_state")
    if final_state is not None:
        for field, expected in (
            ("schema_version", "petitgpt-n2-v2-final-state-v1"),
            ("branch", HISTORICAL_TRAINING_BRANCH),
            ("head", HISTORICAL_TRAINING_HEAD),
            ("head_unchanged", True),
            ("runtime_unchanged", True),
            ("trainer_bundle_unchanged", True),
            ("protected_probe_unchanged", True),
            ("tracked_clean", True),
            ("stage_n_owner_acceptance_published", False),
            ("stage_n_result_published", False),
            ("stage_o_authorized", False),
            ("stage_o_started", False),
            ("gpu_uuid", RUNTIME_INVARIANTS["gpu_uuid"]),
            ("gpu_pci", RUNTIME_INVARIANTS["gpu_pci_bus_id"]),
        ):
            if final_state.get(field) != expected:
                failures.append(f"n2_final_state_mismatch:{field}")
    launch_record = json_artifact("n2_launch_record")
    if launch_record is not None:
        for field, expected in (
            ("schema_version", "petitgpt-n2-v2-launch-record-v1"),
            ("authorization_sha256", N2_AUTHORIZATION_SHA256),
            ("cwd", str(CANONICAL_TRAINING_ROOT)),
            ("internal_helpers_called_directly", False),
        ):
            if launch_record.get(field) != expected:
                failures.append(f"n2_launch_record_mismatch:{field}")

    n2_invariance = json_artifact("n2_zero_update_invariance")
    failures.extend(validate_existing_n2_evidence(n2_invariance))

    runtime_wrapper = json_artifact("n2_runtime_fingerprint")
    if runtime_wrapper is not None:
        if runtime_wrapper.get("schema_version") != launch.STAGE_N_RUNTIME_ARTIFACT_SCHEMA:
            failures.append("n2_runtime_wrapper_schema_mismatch")
        if runtime_wrapper.get("kind") != launch.STAGE_N_RUNTIME_ARTIFACT_KIND:
            failures.append("n2_runtime_wrapper_kind_mismatch")
        runtime = runtime_wrapper.get("runtime_fingerprint")
        if not isinstance(runtime, Mapping):
            failures.append("n2_runtime_fingerprint_missing_or_malformed")
        else:
            for field, expected in (
                *RUNTIME_INVARIANTS.items(),
                ("trainer_head", HISTORICAL_TRAINING_HEAD),
                (
                    "trainer_execution_bundle_sha256",
                    HISTORICAL_TRAINING_BUNDLE_SHA256,
                ),
            ):
                if runtime.get(field) != expected:
                    failures.append(f"n2_runtime_fingerprint_mismatch:{field}")
            if launch.runtime_fingerprint_sha256(runtime) != HISTORICAL_RUNTIME_FINGERPRINT_SHA256:
                failures.append("n2_runtime_fingerprint_semantic_sha256_mismatch")
        if runtime_wrapper.get("runtime_fingerprint_sha256") != (
            HISTORICAL_RUNTIME_FINGERPRINT_SHA256
        ):
            failures.append("n2_runtime_wrapper_semantic_sha256_mismatch")
    return list(dict.fromkeys(failures))


def _validate_destination_topology(destination: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(destination, Mapping):
        return ["bridge_destination_missing_or_malformed"]
    output = destination.get("output_root")
    if not isinstance(output, str) or not output.strip() or not Path(output).is_absolute():
        return ["bridge_destination_output_root_invalid"]
    expected = _destination_document(output)
    failures = [
        f"bridge_destination_mismatch:{field}"
        for field, value in expected.items()
        if not _exact_state_equal(destination.get(field), value)
    ]
    if set(destination) != set(expected):
        failures.append("bridge_destination_field_set_mismatch")
    return failures


def _validate_bridge_authorization_claims(
    authorization: Mapping[str, Any] | None,
    *,
    observed_successor: Mapping[str, Any],
    semantic_comparison_manifest: Mapping[str, Any] | None,
    exact_plan: Mapping[str, Any] | str | Path,
    existing_n2_evidence: Mapping[str, Any] | None,
    historical_runtime_fingerprint: Mapping[str, Any] | None,
    reopen_source_artifacts: bool,
    source_root: str | Path | None = None,
    successor_root: str | Path | None = None,
) -> dict[str, Any]:
    """Non-authoritative structural helper used only behind live/path-bound preflight."""
    if not isinstance(authorization, Mapping):
        return {"authorized": False, "failures": ["n3_authorization_missing_or_malformed"]}
    failures: list[str] = []
    for field, expected in (
        ("schema_version", N3_AUTHORIZATION_SCHEMA),
        ("scope", N3_SCOPE),
        ("authorization_status", "AUTHORIZED"),
        ("authorizes_bridge_execution", True),
        ("authorizes_training", False),
        ("one_time_exception", True),
    ):
        if not _exact_state_equal(authorization.get(field), expected):
            failures.append(f"n3_authorization_mismatch:{field}")
    for field in ("authorized_by", "authorized_at"):
        value = authorization.get(field)
        if not isinstance(value, str) or not value.strip():
            failures.append(f"n3_authorization_missing_owner_field:{field}")

    exact_source = {
        "branch": HISTORICAL_TRAINING_BRANCH,
        "head": HISTORICAL_TRAINING_HEAD,
        "trainer_execution_bundle_sha256": HISTORICAL_TRAINING_BUNDLE_SHA256,
    }
    if not _exact_state_equal(authorization.get("source_execution"), exact_source):
        failures.append("n3_source_execution_identity_mismatch")

    history = authorization.get("history")
    if not isinstance(history, Mapping):
        failures.append("n3_history_missing_or_malformed")
        artifacts = None
    else:
        for field, expected in (
            ("exact_run_plan_sha256", EXACT_RUN_PLAN_SHA256),
            ("pilot_owner_acceptance_sha256", PILOT_OWNER_ACCEPTANCE_SHA256),
            ("n1_governed_run_contract_semantic_sha256", N1_GRC_SEMANTIC_SHA256),
            ("n1_base_governed_identity_sha256", N1_BASE_GOVERNED_IDENTITY_SHA256),
            ("n2_governed_run_contract_semantic_sha256", N2_GRC_SEMANTIC_SHA256),
            (
                "source_runtime_fingerprint_sha256",
                HISTORICAL_RUNTIME_FINGERPRINT_SHA256,
            ),
        ):
            if history.get(field) != expected:
                failures.append(f"n3_history_mismatch:{field}")
        artifacts = history.get("artifacts")
        failures.extend(_validate_bound_artifacts(artifacts))
        if reopen_source_artifacts:
            failures.extend(verify_bound_artifact_bytes(artifacts))

    expected_source_checkpoint = {
        **dict(HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]),
        "step": COMPLETION_STEP,
        "stage": COMPLETION_STAGE,
        "sampler_cursor": COMPLETION_SAMPLER_ENDPOINT,
        "sampler_seed": COMPLETION_SAMPLER_SEED,
    }
    if not _exact_state_equal(
        authorization.get("source_checkpoint"),
        expected_source_checkpoint,
    ):
        failures.append("n3_source_checkpoint_mismatch")
    if not _exact_state_equal(
        authorization.get("zero_update"),
        dict(ZERO_UPDATE_LIMITS),
    ):
        failures.append("n3_zero_update_limits_mismatch")
    if not _exact_state_equal(
        authorization.get("runtime_invariants"),
        dict(RUNTIME_INVARIANTS),
    ):
        failures.append("n3_runtime_invariants_mismatch")
    destination = authorization.get("destination")
    failures.extend(_validate_destination_topology(destination))
    failures.extend(validate_completion_boundary(exact_plan, authorization=authorization))
    failures.extend(validate_existing_n2_evidence(existing_n2_evidence))

    successor = authorization.get("successor")
    if not isinstance(successor, Mapping):
        failures.append("n3_successor_identity_missing_or_malformed")
        successor = {}
    for field, length in (
        ("head", 40),
        ("trainer_execution_bundle_sha256", 64),
        ("bridge_tool_bundle_sha256", 64),
        ("runtime_fingerprint_sha256", 64),
    ):
        if not _valid_hex(successor.get(field), length):
            failures.append(f"n3_successor_identity_invalid:{field}")
    if successor.get("branch") != SUCCESSOR_BRANCH:
        failures.append("n3_successor_branch_mismatch")
    if successor.get("repository_root") != str(SUCCESSOR_REPOSITORY_ROOT):
        failures.append("n3_successor_repository_root_mismatch")
    if successor.get("head") == HISTORICAL_TRAINING_HEAD:
        failures.append("n3_successor_head_is_not_a_successor")

    for authorization_field, observed_field in (
        ("branch", "branch"),
        ("repository_root", "repository_root"),
        ("head", "head"),
        ("trainer_execution_bundle_sha256", "trainer_execution_bundle_sha256"),
        ("bridge_tool_bundle_sha256", "bridge_tool_bundle_sha256"),
    ):
        if successor.get(authorization_field) != observed_successor.get(observed_field):
            failures.append(f"n3_observed_successor_mismatch:{authorization_field}")

    observed_runtime = observed_successor.get("runtime")
    if not isinstance(observed_runtime, Mapping):
        failures.append("n3_observed_successor_runtime_missing_or_malformed")
    else:
        for field, expected in RUNTIME_INVARIANTS.items():
            if observed_runtime.get(field) != expected:
                failures.append(f"n3_runtime_mismatch:{field}")
        for field, expected in (
            ("trainer_head", successor.get("head")),
            (
                "trainer_execution_bundle_sha256",
                successor.get("trainer_execution_bundle_sha256"),
            ),
        ):
            if observed_runtime.get(field) != expected:
                failures.append(f"n3_runtime_mismatch:{field}")
        calculated_runtime_sha = launch.runtime_fingerprint_sha256(observed_runtime)
        if observed_runtime.get("runtime_fingerprint_sha256") != calculated_runtime_sha:
            failures.append("n3_successor_runtime_self_hash_mismatch")
        if successor.get("runtime_fingerprint_sha256") != calculated_runtime_sha:
            failures.append("n3_successor_runtime_authorization_sha256_mismatch")

        if not isinstance(historical_runtime_fingerprint, Mapping):
            failures.append("n3_historical_runtime_fingerprint_missing_or_malformed")
        else:
            historical_runtime_sha = launch.runtime_fingerprint_sha256(
                historical_runtime_fingerprint
            )
            if (
                historical_runtime_fingerprint.get("runtime_fingerprint_sha256")
                != HISTORICAL_RUNTIME_FINGERPRINT_SHA256
                or historical_runtime_sha != HISTORICAL_RUNTIME_FINGERPRINT_SHA256
            ):
                failures.append("n3_historical_runtime_fingerprint_sha256_mismatch")
            expected_successor_runtime = copy.deepcopy(dict(historical_runtime_fingerprint))
            expected_successor_runtime.update({
                "trainer_head": successor.get("head"),
                "trainer_execution_bundle_sha256": successor.get("trainer_execution_bundle_sha256"),
            })
            expected_successor_runtime["runtime_fingerprint_sha256"] = (
                launch.runtime_fingerprint_sha256(expected_successor_runtime)
            )
            if not _exact_state_equal(observed_runtime, expected_successor_runtime):
                failures.append("n3_successor_runtime_not_exact_historical_projection")

    manifest_binding = authorization.get("semantic_comparison_manifest")
    if not isinstance(manifest_binding, Mapping):
        failures.append("n3_semantic_manifest_binding_missing_or_malformed")
    else:
        if not isinstance(manifest_binding.get("path"), str) or not manifest_binding.get("path"):
            failures.append("n3_semantic_manifest_path_missing")
        if not _valid_hex(manifest_binding.get("sha256"), 64):
            failures.append("n3_semantic_manifest_sha256_invalid")
        failures.extend(
            validate_semantic_comparison_manifest(
                semantic_comparison_manifest,
                expected_artifact_sha256=str(manifest_binding.get("sha256", "")),
                expected_successor_head=str(successor.get("head", "")),
                expected_successor_trainer_bundle_sha256=str(
                    successor.get("trainer_execution_bundle_sha256", "")
                ),
                source_root=source_root,
                successor_root=successor_root,
            )
        )

    expected_authorization = n3_authorization_template(
        successor_head=successor.get("head"),
        successor_trainer_bundle_sha256=successor.get("trainer_execution_bundle_sha256"),
        bridge_tool_bundle_sha256=successor.get("bridge_tool_bundle_sha256"),
        successor_runtime_fingerprint_sha256=successor.get("runtime_fingerprint_sha256"),
        semantic_comparison_manifest_path=(
            manifest_binding.get("path") if isinstance(manifest_binding, Mapping) else None
        ),
        semantic_comparison_manifest_sha256=(
            manifest_binding.get("sha256") if isinstance(manifest_binding, Mapping) else None
        ),
        output_root=(
            destination.get("output_root")
            if isinstance(destination, Mapping) and isinstance(destination.get("output_root"), str)
            else None
        ),
    )
    expected_authorization.update({
        "authorization_status": "AUTHORIZED",
        "authorizes_bridge_execution": True,
        "authorized_by": authorization.get("authorized_by"),
        "authorized_at": authorization.get("authorized_at"),
    })
    if "stage_n_completion" in authorization:
        expected_authorization["stage_n_completion"] = {
            "expected_final_step": COMPLETION_STEP,
        }
    if not _exact_state_equal(authorization, expected_authorization):
        failures.append("n3_authorization_not_exact_template_projection")
    return {
        "schema_version": "petitgpt-stage-n-successor-bridge-preflight-v1",
        "authorized": not failures,
        "failures": list(dict.fromkeys(failures)),
        "source_head": HISTORICAL_TRAINING_HEAD,
        "successor_head": successor.get("head"),
        "completion_step": COMPLETION_STEP,
        "sampler_endpoint": COMPLETION_SAMPLER_ENDPOINT,
    }


def _atomic_publish_json(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    _require(not target.exists(), f"bridge artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(dict(document))
    # Every call is inside the private, mode-0700 staging directory. Writing the final
    # staged name with O_EXCL avoids any replace/clobber primitive before publication.
    with open(target, "xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(target), "sha256": sha256_bytes(body)}


def _deserialize_checkpoint_snapshot(snapshot: bytes) -> Mapping[str, Any]:
    import torch

    # The exact bytes were authenticated before this function is called. Never give
    # torch.load a mutable path that it could reopen after the hash observation.
    document = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=False)
    _require(isinstance(document, Mapping), "bridge source checkpoint is not a mapping")
    return document


def _read_bound_checkpoint_snapshot(
    path: Path, *, expected_sha256: str | None, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    """Read/hash once and deserialize those same authenticated checkpoint bytes."""

    _require(path.is_file(), f"bridge {label} checkpoint not found: {path}")
    try:
        with open(path, "rb") as handle:
            snapshot = handle.read()
    except OSError as exc:
        raise CompatibilityBridgeError(f"cannot snapshot bridge {label} checkpoint") from exc
    observed_sha256 = sha256_bytes(snapshot)
    if expected_sha256 is not None:
        _require(
            observed_sha256 == expected_sha256,
            f"bridge {label} checkpoint bytes do not match their exact SHA-256",
        )
    try:
        document = _deserialize_checkpoint_snapshot(snapshot)
    except Exception as exc:
        raise CompatibilityBridgeError(
            f"bridge {label} checkpoint snapshot cannot be deserialized"
        ) from exc
    _require(isinstance(document, Mapping), f"bridge {label} checkpoint is not a mapping")
    return snapshot, document


def _serialize_checkpoint_document(document: Mapping[str, Any]) -> bytes:
    import torch

    output = io.BytesIO()
    torch.save(dict(document), output)
    return output.getvalue()


def _default_checkpoint_saver(path: Path, document: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"bridge checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _serialize_checkpoint_document(document)
    with open(path, "xb") as handle:
        handle.write(snapshot)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without ever replacing a raced target."""

    import ctypes

    _require(source.parent.resolve() == destination.parent.resolve(), "publish roots differ")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise CompatibilityBridgeError(
            "atomic no-replace bridge publication failed: "
            f"errno={error_number} ({os.strerror(error_number)})"
        )


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(completed.returncode == 0, f"git observation failed: {root} {arguments}")
    return completed.stdout.strip()


def observe_historical_identity() -> dict[str, Any]:
    """Reobserve the immutable old-HEAD worktree, closure, and sole untracked probe."""
    root = CANONICAL_TRAINING_ROOT.resolve()
    branch = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git_output(root, "rev-parse", "HEAD")
    _require(branch == HISTORICAL_TRAINING_BRANCH, "historical branch identity changed")
    _require(head == HISTORICAL_TRAINING_HEAD, "historical HEAD identity changed")
    _require(
        not _git_output(root, "status", "--porcelain", "--untracked-files=no"),
        "historical tracked worktree is not clean",
    )
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    expected_status = f"?? {HISTORICAL_UNTRACKED_PROBE_RELPATH}"
    _require(status == expected_status, "historical untracked file set changed")
    probe = root / HISTORICAL_UNTRACKED_PROBE_RELPATH
    _require(probe.is_file(), "historical manual context probe is missing")
    _require(
        file_sha256(probe) == HISTORICAL_UNTRACKED_PROBE_SHA256,
        "historical manual context probe bytes changed",
    )
    trainer = launch.trainer_execution_closure(root)
    _require(
        trainer.get("unbound_load_bearing_module_count") == 0,
        "historical trainer closure has unbound modules",
    )
    bundle = trainer.get("TRAINER_EXECUTION_BUNDLE_SHA256")
    _require(bundle == HISTORICAL_TRAINING_BUNDLE_SHA256, "historical trainer bundle changed")
    return {
        "repository_root": str(root),
        "branch": branch,
        "head": head,
        "trainer_execution_bundle_sha256": bundle,
        "tracked_clean": True,
        "untracked": [HISTORICAL_UNTRACKED_PROBE_RELPATH],
        "untracked_probe_sha256": HISTORICAL_UNTRACKED_PROBE_SHA256,
    }


def observe_successor_identity() -> dict[str, Any]:
    """Derive code and live-runtime identity inside the future bridge process.

    Execution never accepts caller-selected HEAD or bundle values.  A tracked dirty tree is
    also rejected because neither the Git HEAD nor an authorization could identify its
    actual executable bytes.
    """
    _assert_successor_module_origins()
    root = _SUCCESSOR_CODE_ROOT
    _require(
        Path.cwd().resolve() == CANONICAL_TRAINING_ROOT.resolve(),
        "successor bridge must execute from the canonical /workspace/petitgpt CWD",
    )
    _require(
        root == SUCCESSOR_REPOSITORY_ROOT.resolve(),
        "successor bridge module is not loaded from the exact reviewed successor root",
    )

    _require(
        not _git_output(root, "status", "--porcelain", "--untracked-files=no"),
        "successor tracked worktree is not clean",
    )
    trainer = launch.trainer_execution_closure(root)
    bridge = bridge_tool_closure(root)
    runtime = launch.observed_training_runtime(num_workers=int(RUNTIME_INVARIANTS["num_workers"]))
    return {
        "repository_root": str(root),
        "branch": _git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "head": _git_output(root, "rev-parse", "HEAD"),
        "trainer_execution_bundle_sha256": trainer["TRAINER_EXECUTION_BUNDLE_SHA256"],
        "bridge_tool_bundle_sha256": bridge["BRIDGE_TOOL_BUNDLE_SHA256"],
        "runtime": runtime,
    }


def _failed_preflight(*failures: str) -> dict[str, Any]:
    return {
        "schema_version": "petitgpt-stage-n-successor-bridge-preflight-v1",
        "authorized": False,
        "failures": list(dict.fromkeys(failures)),
        "source_head": HISTORICAL_TRAINING_HEAD,
        "successor_head": None,
        "completion_step": COMPLETION_STEP,
        "sampler_endpoint": COMPLETION_SAMPLER_ENDPOINT,
    }


def _read_json_mapping_snapshot(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, Mapping[str, Any]]:
    """Read, hash, and parse one immutable JSON byte snapshot."""

    _require(path.is_file(), f"bridge {label} not found: {path}")
    try:
        snapshot = path.read_bytes()
        document = json.loads(snapshot.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityBridgeError(f"bridge {label} is not valid JSON: {path}") from exc
    observed_sha256 = sha256_bytes(snapshot)
    if expected_sha256 is not None:
        _require(
            observed_sha256 == expected_sha256,
            f"bridge {label} bytes do not match their exact SHA-256",
        )
    _require(isinstance(document, Mapping), f"bridge {label} is not a JSON object")
    return observed_sha256, document


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    _sha256, document = _read_json_mapping_snapshot(path, label)
    return document


def _load_bound_historical_runtime_fingerprint() -> dict[str, Any]:
    """Read and authenticate the exact N2 runtime bytes in one snapshot."""

    binding = HISTORICAL_ARTIFACTS["n2_runtime_fingerprint"]
    path = Path(str(binding["path"]))
    _require(path.is_file(), f"historical N2 runtime artifact not found: {path}")
    try:
        snapshot = path.read_bytes()
        document = json.loads(snapshot.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityBridgeError(
            f"historical N2 runtime artifact is not valid JSON: {path}"
        ) from exc
    _require(
        sha256_bytes(snapshot) == binding["sha256"],
        "historical N2 runtime artifact bytes changed",
    )
    _require(isinstance(document, Mapping), "historical N2 runtime artifact is not an object")
    _require(
        document.get("schema_version") == launch.STAGE_N_RUNTIME_ARTIFACT_SCHEMA
        and document.get("kind") == launch.STAGE_N_RUNTIME_ARTIFACT_KIND,
        "historical N2 runtime artifact envelope changed",
    )
    runtime = document.get("runtime_fingerprint")
    _require(isinstance(runtime, Mapping), "historical N2 runtime fingerprint is missing")
    _require(
        document.get("runtime_fingerprint_sha256") == HISTORICAL_RUNTIME_FINGERPRINT_SHA256
        and runtime.get("runtime_fingerprint_sha256") == HISTORICAL_RUNTIME_FINGERPRINT_SHA256
        and launch.runtime_fingerprint_sha256(runtime) == HISTORICAL_RUNTIME_FINGERPRINT_SHA256,
        "historical N2 runtime fingerprint identity changed",
    )
    return copy.deepcopy(dict(runtime))


def validate_bridge_authorization(
    authorization_path: str | Path,
    *,
    semantic_comparison_manifest_path: str | Path | None = None,
    exact_plan_path: str | Path | None = None,
    existing_n2_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Authoritatively validate N3 from live bytes, committed code and actual runtime.

    No caller may supply observed identity, runtime, repository roots, plan claims or N2
    claims. The future post-review invocation must use a clean successor code checkout
    while retaining the historical canonical process CWD.
    """
    try:
        _assert_successor_module_origins()
        auth_path = Path(authorization_path).resolve()
        authorization_sha256, authorization = _read_json_mapping_snapshot(
            auth_path, "N3 authorization"
        )
        binding = authorization.get("semantic_comparison_manifest")
        _require(
            isinstance(binding, Mapping),
            "N3 authorization has no semantic comparison binding",
        )
        bound_manifest = Path(str(binding.get("path", ""))).resolve()
        manifest_path = (
            Path(semantic_comparison_manifest_path).resolve()
            if semantic_comparison_manifest_path is not None
            else bound_manifest
        )
        _require(manifest_path == bound_manifest, "semantic manifest path is not authorized")
        manifest_sha256, manifest = _read_json_mapping_snapshot(
            manifest_path,
            "semantic comparison manifest",
            expected_sha256=str(binding.get("sha256", "")),
        )

        exact_path = Path(exact_plan_path or HISTORICAL_ARTIFACTS["exact_plan"]["path"]).resolve()
        n2_path = Path(
            existing_n2_evidence_path or HISTORICAL_ARTIFACTS["n2_zero_update_invariance"]["path"]
        ).resolve()
        _require(
            exact_path == Path(HISTORICAL_ARTIFACTS["exact_plan"]["path"]).resolve(),
            "bridge exact plan path is not the immutable historical plan",
        )
        _require(
            n2_path == Path(HISTORICAL_ARTIFACTS["n2_zero_update_invariance"]["path"]).resolve(),
            "bridge N2 evidence path is not the immutable invariance artifact",
        )
        _exact_plan_sha256, exact_plan = _read_json_mapping_snapshot(
            exact_path,
            "exact Stage-P plan",
            expected_sha256=str(HISTORICAL_ARTIFACTS["exact_plan"]["sha256"]),
        )
        _n2_evidence_sha256, n2_evidence = _read_json_mapping_snapshot(
            n2_path,
            "N2 invariance evidence",
            expected_sha256=str(HISTORICAL_ARTIFACTS["n2_zero_update_invariance"]["sha256"]),
        )
        historical_runtime = _load_bound_historical_runtime_fingerprint()
        historical = observe_historical_identity()
        observed = observe_successor_identity()
        verdict = _validate_bridge_authorization_claims(
            authorization,
            observed_successor=observed,
            semantic_comparison_manifest=manifest,
            exact_plan=exact_plan,
            existing_n2_evidence=n2_evidence,
            historical_runtime_fingerprint=historical_runtime,
            reopen_source_artifacts=True,
            source_root=CANONICAL_TRAINING_ROOT,
            successor_root=Path(str(observed["repository_root"])),
        )
        verdict["authorization_path"] = str(auth_path)
        verdict["authorization_sha256"] = authorization_sha256
        verdict["authorization_document"] = copy.deepcopy(dict(authorization))
        verdict["semantic_comparison_manifest_path"] = str(manifest_path)
        verdict["semantic_comparison_manifest_sha256"] = manifest_sha256
        verdict["semantic_comparison_manifest_document"] = copy.deepcopy(dict(manifest))
        verdict["observed_historical_identity"] = historical
        verdict["observed_successor_identity"] = observed
        return verdict
    except (CompatibilityBridgeError, OSError, ValueError, TypeError) as exc:
        return _failed_preflight(f"authoritative_preflight_failed:{exc}")


# Stable public alias for successor Stage-O/review tooling. It is equally authoritative.
validate_n3_authorization = validate_bridge_authorization


def _restore_canonical_rng_state(state: Mapping[str, Any]) -> None:
    """Restore all four canonical streams after validating them on the live CUDA device."""
    import random

    import numpy as np
    import torch

    failures = launch.validate_restorable_rng_state(state, require_live_cuda_validation=True)
    _require(not failures, "source RNG state is not restorable: " + ", ".join(failures))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _capture_canonical_rng_state() -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _live_restored_state_proof(
    model: Any,
    optimizer: Any,
    source_checkpoint: Mapping[str, Any],
    *,
    expected_rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    rng_state = _capture_canonical_rng_state()
    proof = {
        "model_parameters_bitwise_identical": _exact_state_equal(
            model_state, source_checkpoint.get("model")
        ),
        "optimizer_state_equivalent": _exact_state_equal(
            optimizer_state, source_checkpoint.get("optim")
        ),
        "scaler_state_equivalent": source_checkpoint.get("scaler") is None,
        "rng_state_preserved": _exact_state_equal(rng_state, expected_rng_state),
        "all_parameter_grads_absent": all(
            parameter.grad is None for parameter in model.parameters()
        ),
        "model_tensors_compared": _model_tensor_count(model_state),
    }
    _require(
        proof["model_tensors_compared"] == EXPECTED_MODEL_TENSOR_COUNT,
        "restored live model tensor count is not exactly 213",
    )
    for field in (
        "model_parameters_bitwise_identical",
        "optimizer_state_equivalent",
        "scaler_state_equivalent",
        "rng_state_preserved",
        "all_parameter_grads_absent",
    ):
        _require(proof[field] is True, f"zero-update live-state proof failed: {field}")
    return proof


def _causal_diagnostic(model: Any, device: Any) -> dict[str, Any]:
    """Mirror the trainer's precompile causal diagnostic without importing its script."""
    import math

    import torch

    seq_len = launch.CAUSAL_DIAGNOSTIC_SEQ_LEN
    check_pos = launch.CAUSAL_DIAGNOSTIC_CHECK_POS
    delta_pos = launch.CAUSAL_DIAGNOSTIC_DELTA_POS
    tolerance = launch.CAUSAL_LEAK_MAX_ABS_TOLERANCE
    values = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    values %= int(launch.MODEL_CONTRACT["vocab_size"])
    mode_before = bool(model.training)
    _require(mode_before, "bridge model is not in training mode before causal diagnostic")
    try:
        model.eval()
        with torch.no_grad():
            logits_before = model(values).float()
            perturbed = values.clone()
            position = min(perturbed.shape[1] - 1, check_pos + delta_pos)
            perturbed[:, position] = (perturbed[:, position] + 123) % int(
                launch.MODEL_CONTRACT["vocab_size"]
            )
            logits_after = model(perturbed).float()
            difference = float(
                (logits_before[:, :check_pos, :] - logits_after[:, :check_pos, :])
                .abs()
                .max()
                .item()
            )
    finally:
        model.train(mode_before)
    mode_after = bool(model.training)
    within_tolerance = math.isfinite(difference) and difference <= tolerance
    _require(within_tolerance, "bridge precompile causal diagnostic exceeded tolerance")
    _require(mode_after == mode_before, "bridge causal diagnostic changed model mode")
    return {
        "executed": True,
        "used_uncompiled_base_model": True,
        "executed_before_training_compile_realization": True,
        "grad_enabled": False,
        "input_shape": [1, seq_len],
        "check_pos": check_pos,
        "delta_pos": delta_pos,
        "max_abs_difference": difference,
        "max_abs_tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "mode_before": "train",
        "mode_after": "train",
        "mode_restored": True,
    }


def validate_bridge_compile_evidence_document(
    evidence: Mapping[str, Any] | None,
) -> list[str]:
    """Validate both the governed compile seal and the bridge's zero-work proof."""

    failures = list(launch.verify_compile_evidence_document(evidence))
    if not isinstance(evidence, Mapping):
        failures.append("bridge_compile_evidence_missing")
        return list(dict.fromkeys(failures))

    observations = evidence.get("bridge_zero_update_observations")
    if not isinstance(observations, Mapping):
        failures.append("bridge_zero_update_observations_missing")
        return list(dict.fromkeys(failures))

    if not _exact_state_equal(
        observations.get("execution_counters"),
        dict(ZERO_EXECUTION_COUNTERS),
    ):
        failures.append("bridge_compile_execution_counters_nonzero_or_malformed")

    expected_state_proof = {
        "model_parameters_bitwise_identical": True,
        "optimizer_state_equivalent": True,
        "scaler_state_equivalent": True,
        "rng_state_preserved": True,
        "all_parameter_grads_absent": True,
        "model_tensors_compared": EXPECTED_MODEL_TENSOR_COUNT,
    }
    for phase in ("before_compile", "after_compile"):
        if not _exact_state_equal(observations.get(phase), expected_state_proof):
            failures.append(f"bridge_compile_{phase}_state_proof_mismatch")

    for field in (
        "training_loop_constructed",
        "data_loader_constructed",
        "sampler_constructed",
        "scheduler_constructed",
    ):
        if observations.get(field) is not False:
            failures.append(f"bridge_compile_forbidden_surface_observed:{field}")

    expected_keys = {
        "execution_counters",
        "before_compile",
        "after_compile",
        "training_loop_constructed",
        "data_loader_constructed",
        "sampler_constructed",
        "scheduler_constructed",
    }
    if set(observations) != expected_keys:
        failures.append("bridge_zero_update_observations_schema_mismatch")
    return list(dict.fromkeys(failures))


def _canonical_zero_update_compile_realization(
    source_checkpoint: Mapping[str, Any], *, authorization_sha256: str
) -> dict[str, Any]:
    """Restore the exact canonical state and realize compile with no training operation."""
    import torch

    model_module = _load_exact_successor_module("_petitgpt_successor_src_model", "src/model.py")
    optimizer_module = _load_exact_successor_module("_petitgpt_successor_src_optim", "src/optim.py")
    GPT = model_module.GPT
    gpt_config_from_checkpoint_dict = model_module.gpt_config_from_checkpoint_dict
    build_optimizer = optimizer_module.build_optimizer

    _require(torch.cuda.is_available(), "successor bridge requires its authorized CUDA device")
    _require(torch.cuda.device_count() == 1, "successor bridge requires exactly one visible GPU")
    _require(launch.PRECISION == "bf16", "successor bridge only implements frozen BF16 policy")
    _require(source_checkpoint.get("scaler") is None, "BF16 source scaler must be exactly None")
    source_grc = source_checkpoint.get("governed_run_contract")
    source_state = source_checkpoint.get("governed_checkpoint_state")
    _require(isinstance(source_grc, Mapping), "source governed run contract is missing")
    _require(isinstance(source_state, Mapping), "source governed checkpoint state is missing")
    envelope_failures = launch.validate_governed_checkpoint_resume_envelope(source_checkpoint)
    _require(
        not envelope_failures,
        "source checkpoint resume envelope failed: " + ", ".join(envelope_failures),
    )
    dynamic_failures = launch.validate_governed_checkpoint_state(
        source_state,
        governed_run_contract=source_grc,
        checkpoint_global_step=COMPLETION_STEP,
        require_live_cuda_validation=True,
    )
    _require(
        not dynamic_failures,
        "source governed checkpoint state failed: " + ", ".join(dynamic_failures),
    )
    source_rng = source_state.get("rng_state")
    _require(isinstance(source_rng, Mapping), "source dynamic RNG state is missing")
    _require(
        _exact_state_equal(source_checkpoint.get("rng_state"), source_rng),
        "source outer and governed RNG states differ",
    )

    config = source_checkpoint.get("config")
    _require(isinstance(config, Mapping), "source model config is missing")
    cfg = gpt_config_from_checkpoint_dict(dict(config))
    for field, attribute in (
        ("n_layers", "n_layers"),
        ("d_model", "d_model"),
        ("n_heads", "n_heads"),
        ("n_kv_heads", "n_kv_heads"),
        ("d_ff", "d_ff"),
        ("seq_len", "max_seq_len"),
        ("vocab_size", "vocab_size"),
        ("dropout", "dropout"),
        ("tie_embeddings", "tie_embeddings"),
    ):
        _require(
            getattr(cfg, attribute) == launch.MODEL_CONTRACT[field],
            f"source model configuration mismatch:{field}",
        )

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    model = GPT(cfg)
    source_model = source_checkpoint.get("model")
    _require(isinstance(source_model, Mapping), "source model state is missing")
    normalized_model = {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in source_model.items()
    }
    model.load_state_dict(normalized_model, strict=True)
    model = model.to(device)
    model.train(True)
    optimizer = build_optimizer(
        model,
        name=launch.OPTIMIZER,
        lr=launch.PEAK_LR,
        weight_decay=launch.WEIGHT_DECAY,
        betas=launch.ADAMW_AUX_BETAS,
        eps=launch.ADAMW_AUX_EPS,
        muon_lr=launch.MUON_LR,
        muon_momentum=launch.MUON_MOMENTUM,
        ns_steps=launch.MUON_NS_STEPS,
        verbose=False,
    )
    optimizer_state = source_checkpoint.get("optim")
    _require(isinstance(optimizer_state, Mapping), "source optimizer state is missing")
    optimizer.load_state_dict(dict(optimizer_state))
    launch.gate_b_post_construction(model, optimizer)

    _restore_canonical_rng_state(source_rng)
    before = _live_restored_state_proof(
        model, optimizer, source_checkpoint, expected_rng_state=source_rng
    )
    stance = launch.enforce_compile_fail_closed_stance()
    cache_token = hashlib.sha256(
        f"{os.getpid()}:{authorization_sha256}:{N3_SCOPE}".encode()
    ).hexdigest()[:24]
    cache = launch.isolated_inductor_cache(cache_token)
    bound = launch.bind_compiled_callable_governed(model)
    causal = _causal_diagnostic(bound["eager_module"], device)
    try:
        evidence_draft = launch.realize_compile_production_shape(
            bound["compiled_module"],
            device=device,
            micro_bsz=launch.MICRO_BSZ,
            seq_len=int(launch.MODEL_CONTRACT["seq_len"]),
            vocab_size=int(launch.MODEL_CONTRACT["vocab_size"]),
            cache=cache,
            finalize=False,
        )
    finally:
        # Synthetic causal/compile probes may consume RNG, but the successor checkpoint is
        # the same zero-update resume point and therefore must retain the source streams.
        _restore_canonical_rng_state(source_rng)
    after = _live_restored_state_proof(
        model, optimizer, source_checkpoint, expected_rng_state=source_rng
    )
    counters = dict(ZERO_EXECUTION_COUNTERS)
    evidence_draft["precompile_causal_diagnostic"] = causal
    evidence_draft["fail_closed_stance"] = stance
    evidence_draft["post_realization_stance"] = launch.arm_fail_on_recompile()
    evidence_draft["bridge_zero_update_observations"] = {
        "execution_counters": counters,
        "before_compile": before,
        "after_compile": after,
        "training_loop_constructed": False,
        "data_loader_constructed": False,
        "sampler_constructed": False,
        "scheduler_constructed": False,
    }
    compile_evidence = launch.finalize_compile_evidence(evidence_draft)
    compile_failures = validate_bridge_compile_evidence_document(compile_evidence)
    _require(
        not compile_failures,
        "bridge compile realization evidence failed: " + ", ".join(compile_failures),
    )
    launch.require_compile_realized(compile_evidence)
    return {
        "compile_evidence": compile_evidence,
        "execution_counters": counters,
        "live_state_proof": after,
    }


def _successor_grc(
    source_checkpoint: Mapping[str, Any],
    *,
    authorization_path: Path,
    authorization_sha256: str,
    authorization: Mapping[str, Any],
    observed_runtime: Mapping[str, Any],
    compile_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    source_grc = source_checkpoint.get("governed_run_contract")
    _require(isinstance(source_grc, Mapping), "N2 terminal checkpoint has no governed contract")
    source_state = source_checkpoint.get("governed_checkpoint_state")
    _require(isinstance(source_state, Mapping), "N2 terminal checkpoint has no governed state")
    successor = authorization["successor"]
    destination = authorization["destination"]
    source_resume = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
        "checkpoint_sha256": N2_TERMINAL_CHECKPOINT_SHA256,
        "expected_step": COMPLETION_STEP,
        "stage": COMPLETION_STAGE,
        "governed_run_contract_sha256": N2_GRC_SEMANTIC_SHA256,
        "source_stage_authorization_path": HISTORICAL_ARTIFACTS["n2_authorization"]["path"],
        "source_stage_authorization_sha256": N2_AUTHORIZATION_SHA256,
        "source_invocation_run_contract_path": HISTORICAL_ARTIFACTS["n2_governed_run_contract"][
            "path"
        ],
        "source_invocation_run_contract_sha256": N2_GRC_ARTIFACT_SHA256,
        "source_base_governed_identity_digest": launch.base_governed_identity_sha256(source_grc),
        "source_checkpoint_path": HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
        "source_checkpoint_sha256": N2_TERMINAL_CHECKPOINT_SHA256,
        "source_checkpoint_step": COMPLETION_STEP,
        "source_checkpoint_stage": COMPLETION_STAGE,
        "source_active_stage": COMPLETION_STAGE,
        "source_sampler_seed": COMPLETION_SAMPLER_SEED,
        "source_permutation_identity": source_state.get("permutation_identity"),
        "source_range_start_position": source_state.get("range_start_position"),
        "source_invocation_range_start_position": source_state.get(
            "invocation_range_start_position"
        ),
        "source_range_stop_position": COMPLETION_SAMPLER_ENDPOINT,
        "source_cursor": COMPLETION_SAMPLER_ENDPOINT,
    }
    document = copy.deepcopy(dict(source_grc))
    document.update({
        "schema_version": launch.RUN_CONTRACT_SCHEMA,
        "scope": "STAGE_N",
        "stage": COMPLETION_STAGE,
        "trainer_branch": successor["branch"],
        "trainer_head": successor["head"],
        "trainer_execution_bundle_sha256": successor["trainer_execution_bundle_sha256"],
        "bridge_tool_bundle_sha256": successor["bridge_tool_bundle_sha256"],
        "successor_repository_root": successor["repository_root"],
        "launch_contract_sha256": launch.contract_sha256(),
        "stage_authorization_path": str(authorization_path),
        "stage_authorization_sha256": authorization_sha256,
        "runtime_fingerprint": dict(observed_runtime),
        "runtime_fingerprint_sha256": successor["runtime_fingerprint_sha256"],
        "num_workers": RUNTIME_INVARIANTS["num_workers"],
        "resume": source_resume,
        "governed_run_root": destination["output_root"],
        "invocation_root": destination["output_root"],
        "out_dir": destination["output_root"],
        "samples_dir": str(Path(destination["output_root"]) / "samples"),
        "stage_start_step": 0,
        "stage_stop_step": COMPLETION_STEP,
        "compile_evidence": dict(compile_evidence),
        "compile_evidence_sha256": compile_evidence.get("compile_evidence_sha256"),
        "compatibility_bridge": {
            "schema_version": BRIDGE_GRC_SCHEMA,
            "source_head": HISTORICAL_TRAINING_HEAD,
            "source_bundle_sha256": HISTORICAL_TRAINING_BUNDLE_SHA256,
            "source_checkpoint_sha256": N2_TERMINAL_CHECKPOINT_SHA256,
            "successor_repository_root": successor["repository_root"],
            "successor_head": successor["head"],
            "successor_trainer_execution_bundle_sha256": successor[
                "trainer_execution_bundle_sha256"
            ],
            "successor_bridge_tool_bundle_sha256": successor["bridge_tool_bundle_sha256"],
            "successor_runtime_fingerprint_sha256": successor["runtime_fingerprint_sha256"],
            "semantic_comparison_manifest_path": authorization["semantic_comparison_manifest"][
                "path"
            ],
            "semantic_comparison_manifest_sha256": authorization["semantic_comparison_manifest"][
                "sha256"
            ],
            "start_step": COMPLETION_STEP,
            "stop_step": COMPLETION_STEP,
            "optimizer_updates": 0,
            "trained_tokens": 0,
            "sampler_advances": 0,
            "training_loop_iterations": 0,
            "backward_calls": 0,
            "scheduler_advances": 0,
            "data_batches_consumed": 0,
        },
    })
    document.pop("governed_run_contract_sha256", None)
    document["governed_run_contract_sha256"] = launch.governed_digest(document)
    return document


def _expected_successor_grc_from_authorized_source(
    *,
    authorization_path: Path,
    authorization_sha256: str,
    authorization: Mapping[str, Any],
    observed_runtime: Mapping[str, Any],
    compile_evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Reconstruct the only GRC permitted by the authenticated N2 source bytes."""

    source_binding = HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]
    _source_bytes, source_checkpoint = _read_bound_checkpoint_snapshot(
        Path(str(source_binding["path"])),
        expected_sha256=str(source_binding["sha256"]),
        label="authoritative N2 source for successor GRC",
    )
    expected_grc = _successor_grc(
        source_checkpoint,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        authorization=authorization,
        observed_runtime=observed_runtime,
        compile_evidence=compile_evidence,
    )
    return source_checkpoint, expected_grc


def _bridge_complete_result_document(
    *,
    authorization: Mapping[str, Any],
    authorization_path: Path,
    authorization_sha256: str,
    semantic_manifest_path: Path,
    grc_path: Path,
    grc_artifact_sha256: str,
    grc_semantic_sha256: str,
    governed_run_contract: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    runtime_path: Path,
    runtime_artifact_sha256: str,
    smoke_result: Mapping[str, Any],
    smoke_result_path: Path,
    resume_result: Mapping[str, Any],
    resume_result_path: Path,
    state_equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    successor = authorization["successor"]
    zero = {
        "optimizer_updates": 0,
        "trained_tokens": 0,
        "sampler_advances": 0,
        "training_loop_iterations": 0,
        "backward_calls": 0,
        "scheduler_advances": 0,
        "data_batches_consumed": 0,
    }
    dynamic = governed_run_contract["sampler_identity"]
    complete = launch.stage_n_result_document(
        governed_run_contract=governed_run_contract,
        governed_run_contract_path=grc_path,
        final_checkpoint_path=str(checkpoint_path),
        final_checkpoint_sha256=checkpoint_sha256,
        final_checkpoint_step=COMPLETION_STEP,
        smoke_results=smoke_result,
        smoke_results_path=smoke_result_path,
        resume_results=resume_result,
        resume_results_path=resume_result_path,
        runtime_fingerprint_path=runtime_path,
        final_sampler_state={
            "permutation_identity": dynamic["permutation_identity"],
            "range_start_position": dynamic["range_start_position"],
            "invocation_range_start_position": dynamic["invocation_range_start_position"],
            "range_stop_position": dynamic["range_stop_position"],
            "cursor": COMPLETION_SAMPLER_ENDPOINT,
        },
    )
    complete.update({
        # The whole output directory is staged and atomically renamed, so the final paths do
        # not exist while this document is built. Bind their already-staged exact bytes.
        "governed_run_contract_artifact_sha256": grc_artifact_sha256,
        "runtime_fingerprint_artifact_sha256": runtime_artifact_sha256,
        "smoke_results_sha256": sha256_bytes(canonical_json_bytes(smoke_result)),
        "resume_results_sha256": sha256_bytes(canonical_json_bytes(resume_result)),
        "stage_n_training_execution_head": HISTORICAL_TRAINING_HEAD,
        "stage_n_training_execution_bundle": HISTORICAL_TRAINING_BUNDLE_SHA256,
        "stage_n_compatibility_bridge_head": successor["head"],
        "stage_n_compatibility_bridge_bundle": successor["bridge_tool_bundle_sha256"],
        "stage_o_execution_head": successor["head"],
        "stage_o_execution_bundle": successor["trainer_execution_bundle_sha256"],
        "stage_n_source_history": {
            "source_execution": dict(authorization["source_execution"]),
            **copy.deepcopy(dict(authorization["history"])),
        },
        "stage_n_compatibility_bridge": {
            "schema_version": BRIDGE_RESULT_SCHEMA,
            "authorization_path": str(authorization_path),
            "authorization_sha256": authorization_sha256,
            "semantic_comparison_manifest_path": str(semantic_manifest_path),
            "semantic_comparison_manifest_sha256": authorization["semantic_comparison_manifest"][
                "sha256"
            ],
            "governed_run_contract_path": str(grc_path),
            "governed_run_contract_artifact_sha256": grc_artifact_sha256,
            "governed_run_contract_sha256": grc_semantic_sha256,
            "terminal_checkpoint_path": str(checkpoint_path),
            "terminal_checkpoint_sha256": checkpoint_sha256,
            "terminal_checkpoint_step": COMPLETION_STEP,
            "runtime_fingerprint_path": str(runtime_path),
            "runtime_fingerprint_artifact_sha256": runtime_artifact_sha256,
            "runtime_fingerprint_sha256": successor["runtime_fingerprint_sha256"],
            "bridge_tool_bundle_sha256": successor["bridge_tool_bundle_sha256"],
            **zero,
            "state_equivalence": dict(state_equivalence),
        },
        "stage_n_owner_accepted": False,
        "stage_o_authorized": False,
    })
    return complete


def _bridge_check_result_document(
    *,
    kind: str,
    authorization_sha256: str,
    governed_run_contract_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    runtime_fingerprint_sha256: str,
    evidence_path: Path,
    evidence_sha256: str,
) -> dict[str, Any]:
    _require(
        kind in (launch.STAGE_N_SMOKE_RESULT_KIND, launch.STAGE_N_RESUME_RESULT_KIND),
        "invalid bridge check-result kind",
    )
    return {
        "schema_version": launch.STAGE_N_CHECK_RESULT_SCHEMA,
        "kind": kind,
        "status": "PASS",
        "stage_authorization_sha256": authorization_sha256,
        "governed_run_contract_sha256": governed_run_contract_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": COMPLETION_STEP,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "evidence_artifact_path": str(evidence_path),
        "evidence_artifact_sha256": evidence_sha256,
    }


def _destination_paths(destination: Mapping[str, Any]) -> tuple[Path, ...]:
    fields = (
        "governed_run_contract_path",
        "terminal_checkpoint_path",
        "runtime_fingerprint_path",
        "smoke_evidence_path",
        "resume_evidence_path",
        "smoke_result_path",
        "resume_result_path",
        "complete_result_path",
    )
    return tuple(Path(str(destination[field])) for field in fields)


def _require_destination_absent(
    destination: Mapping[str, Any], *, ignored: Sequence[Path] = ()
) -> None:
    output_root = Path(str(destination["output_root"]))
    _require(not output_root.exists(), f"bridge output root already exists: {output_root}")
    ignored_paths = {path.resolve() for path in ignored}
    for path in _destination_paths(destination):
        if path.resolve() in ignored_paths:
            continue
        _require(not path.exists(), f"bridge destination already exists: {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        _require(not temporary.exists(), f"bridge destination temporary exists: {temporary}")


def _validate_successor_checkpoint_binding(
    checkpoint: Mapping[str, Any],
    *,
    governed_run_contract: Mapping[str, Any],
    compile_evidence: Mapping[str, Any],
    require_live_cuda_validation: bool,
) -> list[str]:
    failures: list[str] = []
    grc_sha = governed_run_contract.get("governed_run_contract_sha256")
    if not _exact_state_equal(checkpoint.get("governed_run_contract"), governed_run_contract):
        failures.append("successor_checkpoint_governed_run_contract_mismatch")
    if checkpoint.get("governed_run_contract_sha256") != grc_sha:
        failures.append("successor_checkpoint_governed_run_contract_sha256_mismatch")
    if checkpoint.get("kind") != governed_run_contract.get("kind"):
        failures.append("successor_checkpoint_kind_mismatch")
    if not _exact_state_equal(governed_run_contract.get("compile_evidence"), compile_evidence):
        failures.append("successor_grc_compile_evidence_mismatch")
    if governed_run_contract.get("compile_evidence_sha256") != compile_evidence.get(
        "compile_evidence_sha256"
    ):
        failures.append("successor_grc_compile_evidence_sha256_mismatch")
    dynamic = checkpoint.get("governed_checkpoint_state")
    if not isinstance(dynamic, Mapping):
        failures.append("successor_checkpoint_dynamic_state_missing")
    else:
        if not _exact_state_equal(dynamic.get("compile_evidence"), compile_evidence):
            failures.append("successor_checkpoint_compile_evidence_mismatch")
        if dynamic.get("compile_evidence_sha256") != compile_evidence.get(
            "compile_evidence_sha256"
        ):
            failures.append("successor_checkpoint_compile_evidence_sha256_mismatch")
        failures.extend(
            launch.validate_governed_checkpoint_state(
                dynamic,
                governed_run_contract=governed_run_contract,
                checkpoint_global_step=COMPLETION_STEP,
                require_live_cuda_validation=require_live_cuda_validation,
            )
        )
    failures.extend(launch.validate_governed_checkpoint_resume_envelope(checkpoint))
    return list(dict.fromkeys(failures))


def _revalidate_successor_checkpoint_state(
    destination_path: Path,
    *,
    destination_sha256: str,
    governed_run_contract: Mapping[str, Any],
    compile_evidence: Mapping[str, Any],
    recorded_state_equivalence: Any,
    authenticated_source_checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute source->successor equivalence from authenticated byte snapshots."""

    failures: list[str] = []
    try:
        source_binding = HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]
        if authenticated_source_checkpoint is None:
            _source_bytes, source = _read_bound_checkpoint_snapshot(
                Path(source_binding["path"]),
                expected_sha256=str(source_binding["sha256"]),
                label="authoritative N2 source",
            )
        else:
            _require(
                file_sha256(source_binding["path"]) == source_binding["sha256"],
                "authoritative N2 source changed after its authenticated snapshot",
            )
            source = authenticated_source_checkpoint
        _destination_bytes, destination = _read_bound_checkpoint_snapshot(
            destination_path,
            expected_sha256=destination_sha256,
            label="authoritative successor",
        )
    except (CompatibilityBridgeError, OSError, ValueError, TypeError) as exc:
        return {
            "failures": [f"successor_checkpoint_snapshot_validation_failed:{exc}"],
            "state_equivalence": None,
        }

    source_grc = source.get("governed_run_contract")
    source_state = source.get("governed_checkpoint_state")
    if not isinstance(source_grc, Mapping) or not isinstance(source_state, Mapping):
        failures.append("successor_checkpoint_source_envelope_missing")
    else:
        failures.extend(
            f"successor_checkpoint_source_dynamic:{failure}"
            for failure in launch.validate_governed_checkpoint_state(
                source_state,
                governed_run_contract=source_grc,
                checkpoint_global_step=COMPLETION_STEP,
                require_live_cuda_validation=False,
            )
        )
    failures.extend(
        f"successor_checkpoint_source_envelope:{failure}"
        for failure in launch.validate_governed_checkpoint_resume_envelope(source)
    )
    equivalence = validate_state_equivalence(
        source,
        destination,
        execution_counters=ZERO_EXECUTION_COUNTERS,
    )
    if equivalence.get("equivalent") is not True:
        failures.extend(
            f"successor_checkpoint_state_equivalence:{failure}"
            for failure in equivalence.get("failures", [])
        )
    if not _exact_state_equal(equivalence, recorded_state_equivalence):
        failures.append("successor_checkpoint_recorded_state_equivalence_mismatch")
    failures.extend(
        f"successor_checkpoint_destination_envelope:{failure}"
        for failure in _validate_successor_checkpoint_binding(
            destination,
            governed_run_contract=governed_run_contract,
            compile_evidence=compile_evidence,
            require_live_cuda_validation=False,
        )
    )
    return {
        "failures": list(dict.fromkeys(failures)),
        "state_equivalence": equivalence,
    }


def _bridge_resume_evidence_document(
    *,
    authorization_path: Path,
    authorization_sha256: str,
    semantic_manifest_path: Path,
    semantic_manifest_sha256: str,
    governed_run_contract_path: Path,
    governed_run_contract_artifact_sha256: str,
    governed_run_contract_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    runtime_fingerprint_sha256: str,
    execution_counters: Mapping[str, Any],
    state_equivalence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": BRIDGE_RESUME_EVIDENCE_SCHEMA,
        "kind": BRIDGE_RESUME_EVIDENCE_KIND,
        "status": "PASS",
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization_sha256,
        "semantic_comparison_manifest_path": str(semantic_manifest_path),
        "semantic_comparison_manifest_sha256": semantic_manifest_sha256,
        "governed_run_contract_path": str(governed_run_contract_path),
        "governed_run_contract_artifact_sha256": governed_run_contract_artifact_sha256,
        "governed_run_contract_sha256": governed_run_contract_sha256,
        "terminal_checkpoint_path": str(checkpoint_path),
        "terminal_checkpoint_sha256": checkpoint_sha256,
        "terminal_checkpoint_step": COMPLETION_STEP,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "source_n2_checkpoint_path": HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
        "source_n2_checkpoint_sha256": N2_TERMINAL_CHECKPOINT_SHA256,
        "execution_counters": dict(execution_counters),
        "state_equivalence": dict(state_equivalence),
    }


def _physical_artifact_path(
    logical_path: Path,
    *,
    authorized_output_root: Path | None,
    physical_output_root: Path | None,
) -> Path:
    """Map one authorized final path into an invisible staged tree."""

    logical = logical_path.resolve()
    if authorized_output_root is None or physical_output_root is None:
        return logical
    authorized = authorized_output_root.resolve()
    physical = physical_output_root.resolve()
    try:
        relative = logical.relative_to(authorized)
    except ValueError:
        return logical
    mapped = (physical / relative).resolve()
    _require(
        mapped == physical or physical in mapped.parents,
        "staged artifact path escaped the physical output root",
    )
    return mapped


def _load_bound_json(
    path_value: Any,
    expected_sha256: Any,
    *,
    label: str,
    failures: list[str],
    authorized_output_root: Path | None = None,
    physical_output_root: Path | None = None,
) -> Mapping[str, Any] | None:
    if not isinstance(path_value, str) or not path_value.strip():
        failures.append(f"successor_complete_missing_path:{label}")
        return None
    path = _physical_artifact_path(
        Path(path_value),
        authorized_output_root=authorized_output_root,
        physical_output_root=physical_output_root,
    )
    if not path.is_file():
        failures.append(f"successor_complete_artifact_missing:{label}")
        return None
    try:
        snapshot = path.read_bytes()
    except OSError:
        failures.append(f"successor_complete_artifact_unreadable:{label}")
        return None
    if sha256_bytes(snapshot) != expected_sha256:
        failures.append(f"successor_complete_artifact_sha256_mismatch:{label}")
        return None
    try:
        value = json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        failures.append(f"successor_complete_artifact_invalid_json:{label}")
        return None
    if not isinstance(value, Mapping):
        failures.append(f"successor_complete_artifact_not_object:{label}")
        return None
    return value


def _validate_successor_complete_result_document(
    result: Mapping[str, Any],
    *,
    authorization_preflight: Mapping[str, Any],
    physical_output_root: Path | None = None,
) -> list[str]:
    """Validate COMPLETE against either published paths or an invisible staged tree."""
    launch_failures = list(
        launch.validate_stage_n_result(result, require_artifacts=physical_output_root is None)
    )
    if physical_output_root is not None:
        # The production validator has no structural-only switch: even with
        # require_artifacts=False its nested check validators open their logical final
        # paths. Suppress only the exact five expected absence reports. Every referenced
        # staged artifact is then independently reopened, hashed and content-validated
        # below through the authorized-root -> physical-root mapping.
        smoke = result.get("smoke_results")
        resume = result.get("resume_results")
        expected_logical_absence = set()
        for label, check in (("smoke", smoke), ("resume", resume)):
            if isinstance(check, Mapping):
                for prefix in ("checkpoint", "evidence_artifact"):
                    value = check.get(f"{prefix}_path")
                    if isinstance(value, str) and value:
                        expected_logical_absence.add(
                            f"stage_n_{label}_result_{prefix}_not_found:{Path(value)}"
                        )
        if isinstance(resume, Mapping):
            evidence_path = resume.get("evidence_artifact_path")
            if isinstance(evidence_path, str) and evidence_path:
                expected_logical_absence.add(
                    f"stage_n_resume_evidence_not_found:{Path(evidence_path)}"
                )
        launch_failures = [
            failure for failure in launch_failures if failure not in expected_logical_absence
        ]
    failures = launch_failures
    failures.extend(launch.validate_successor_stage_n_provenance(result, required=True))
    if authorization_preflight.get("authorized") is not True:
        failures.append("successor_complete_authorization_preflight_failed")
    bridge = result.get("stage_n_compatibility_bridge")
    if not isinstance(bridge, Mapping):
        return list(dict.fromkeys([*failures, "successor_complete_bridge_record_missing"]))

    authorization = _load_bound_json(
        bridge.get("authorization_path"),
        bridge.get("authorization_sha256"),
        label="authorization",
        failures=failures,
    )
    successor_authority = (
        authorization.get("successor")
        if isinstance(authorization, Mapping)
        and isinstance(authorization.get("successor"), Mapping)
        else None
    )
    if isinstance(authorization, Mapping):
        for field, expected in (
            ("authorization_path", bridge.get("authorization_path")),
            ("semantic_comparison_manifest_path", bridge.get("semantic_comparison_manifest_path")),
        ):
            actual = authorization_preflight.get(field)
            if (
                not isinstance(actual, str)
                or not isinstance(expected, str)
                or Path(actual).resolve() != Path(expected).resolve()
            ):
                failures.append(f"successor_complete_preflight_binding_mismatch:{field}")
        for field, expected in (
            ("authorization_sha256", bridge.get("authorization_sha256")),
            (
                "semantic_comparison_manifest_sha256",
                bridge.get("semantic_comparison_manifest_sha256"),
            ),
            ("source_head", HISTORICAL_TRAINING_HEAD),
            (
                "successor_head",
                successor_authority.get("head") if successor_authority is not None else None,
            ),
        ):
            if authorization_preflight.get(field) != expected:
                failures.append(f"successor_complete_preflight_binding_mismatch:{field}")
        if not _exact_state_equal(
            authorization_preflight.get("authorization_document"), authorization
        ):
            failures.append("successor_complete_preflight_authorization_document_mismatch")
    destination = authorization.get("destination") if isinstance(authorization, Mapping) else None
    authorized_output_root = (
        Path(str(destination.get("output_root")))
        if isinstance(destination, Mapping) and destination.get("output_root")
        else None
    )
    manifest = _load_bound_json(
        bridge.get("semantic_comparison_manifest_path"),
        bridge.get("semantic_comparison_manifest_sha256"),
        label="semantic_manifest",
        failures=failures,
    )
    if isinstance(manifest, Mapping) and not _exact_state_equal(
        authorization_preflight.get("semantic_comparison_manifest_document"), manifest
    ):
        failures.append("successor_complete_preflight_manifest_document_mismatch")
    grc = _load_bound_json(
        bridge.get("governed_run_contract_path"),
        bridge.get("governed_run_contract_artifact_sha256"),
        label="governed_run_contract",
        failures=failures,
        authorized_output_root=authorized_output_root,
        physical_output_root=physical_output_root,
    )
    runtime_wrapper = _load_bound_json(
        bridge.get("runtime_fingerprint_path"),
        bridge.get("runtime_fingerprint_artifact_sha256"),
        label="runtime_fingerprint",
        failures=failures,
        authorized_output_root=authorized_output_root,
        physical_output_root=physical_output_root,
    )
    smoke = _load_bound_json(
        result.get("smoke_results_path"),
        result.get("smoke_results_sha256"),
        label="smoke_result",
        failures=failures,
        authorized_output_root=authorized_output_root,
        physical_output_root=physical_output_root,
    )
    resume = _load_bound_json(
        result.get("resume_results_path"),
        result.get("resume_results_sha256"),
        label="resume_result",
        failures=failures,
        authorized_output_root=authorized_output_root,
        physical_output_root=physical_output_root,
    )
    smoke_evidence = (
        _load_bound_json(
            smoke.get("evidence_artifact_path"),
            smoke.get("evidence_artifact_sha256"),
            label="smoke_evidence",
            failures=failures,
            authorized_output_root=authorized_output_root,
            physical_output_root=physical_output_root,
        )
        if isinstance(smoke, Mapping)
        else None
    )
    resume_evidence = (
        _load_bound_json(
            resume.get("evidence_artifact_path"),
            resume.get("evidence_artifact_sha256"),
            label="resume_evidence",
            failures=failures,
            authorized_output_root=authorized_output_root,
            physical_output_root=physical_output_root,
        )
        if isinstance(resume, Mapping)
        else None
    )
    if authorization is not None:
        output_root = destination.get("output_root") if isinstance(destination, Mapping) else None
        if destination != _destination_document(output_root):
            failures.append("successor_complete_authorized_destination_mismatch")
        if authorization.get("semantic_comparison_manifest") != {
            "path": bridge.get("semantic_comparison_manifest_path"),
            "sha256": bridge.get("semantic_comparison_manifest_sha256"),
        }:
            failures.append("successor_complete_manifest_binding_mismatch")
        if isinstance(destination, Mapping):
            topology_bindings = (
                ("governed_run_contract_path", bridge.get("governed_run_contract_path")),
                ("terminal_checkpoint_path", bridge.get("terminal_checkpoint_path")),
                ("runtime_fingerprint_path", bridge.get("runtime_fingerprint_path")),
                (
                    "smoke_evidence_path",
                    smoke.get("evidence_artifact_path") if isinstance(smoke, Mapping) else None,
                ),
                (
                    "resume_evidence_path",
                    resume.get("evidence_artifact_path") if isinstance(resume, Mapping) else None,
                ),
                ("smoke_result_path", result.get("smoke_results_path")),
                ("resume_result_path", result.get("resume_results_path")),
            )
            for destination_field, recorded_path in topology_bindings:
                if (
                    Path(str(destination.get(destination_field, ""))).resolve()
                    != Path(str(recorded_path or "")).resolve()
                ):
                    failures.append(
                        f"successor_complete_destination_path_mismatch:{destination_field}"
                    )
    if manifest is not None and manifest.get("manifest_content_sha256") != sha256_bytes(
        canonical_json_bytes(_manifest_projection(manifest))
    ):
        failures.append("successor_complete_manifest_content_sha256_mismatch")
    if grc is not None:
        if launch.governed_digest(grc) != bridge.get("governed_run_contract_sha256"):
            failures.append("successor_complete_grc_semantic_sha256_mismatch")
        if grc.get("governed_run_contract_sha256") != bridge.get("governed_run_contract_sha256"):
            failures.append("successor_complete_grc_embedded_sha256_mismatch")
        if grc.get("compatibility_bridge") != {
            "schema_version": BRIDGE_GRC_SCHEMA,
            "source_head": HISTORICAL_TRAINING_HEAD,
            "source_bundle_sha256": HISTORICAL_TRAINING_BUNDLE_SHA256,
            "source_checkpoint_sha256": N2_TERMINAL_CHECKPOINT_SHA256,
            "successor_repository_root": (
                successor_authority.get("repository_root")
                if successor_authority is not None
                else None
            ),
            "successor_head": result.get("stage_n_compatibility_bridge_head"),
            "successor_trainer_execution_bundle_sha256": result.get("stage_o_execution_bundle"),
            "successor_bridge_tool_bundle_sha256": result.get(
                "stage_n_compatibility_bridge_bundle"
            ),
            "successor_runtime_fingerprint_sha256": bridge.get("runtime_fingerprint_sha256"),
            "semantic_comparison_manifest_path": bridge.get("semantic_comparison_manifest_path"),
            "semantic_comparison_manifest_sha256": bridge.get(
                "semantic_comparison_manifest_sha256"
            ),
            "start_step": COMPLETION_STEP,
            "stop_step": COMPLETION_STEP,
            "optimizer_updates": 0,
            "trained_tokens": 0,
            "sampler_advances": 0,
            "training_loop_iterations": 0,
            "backward_calls": 0,
            "scheduler_advances": 0,
            "data_batches_consumed": 0,
        }:
            failures.append("successor_complete_grc_bridge_block_mismatch")
    runtime: Mapping[str, Any] | None = None
    if runtime_wrapper is not None:
        runtime = runtime_wrapper.get("runtime_fingerprint")
        if runtime_wrapper.get("schema_version") != launch.STAGE_N_RUNTIME_ARTIFACT_SCHEMA:
            failures.append("successor_complete_runtime_wrapper_schema_mismatch")
        if runtime_wrapper.get("kind") != launch.STAGE_N_RUNTIME_ARTIFACT_KIND:
            failures.append("successor_complete_runtime_wrapper_kind_mismatch")
        if runtime_wrapper.get("runtime_fingerprint_sha256") != bridge.get(
            "runtime_fingerprint_sha256"
        ):
            failures.append("successor_complete_runtime_wrapper_sha256_mismatch")
        if not isinstance(runtime, Mapping) or launch.runtime_fingerprint_sha256(runtime) != (
            bridge.get("runtime_fingerprint_sha256")
        ):
            failures.append("successor_complete_runtime_semantic_sha256_mismatch")
        if runtime != result.get("runtime_fingerprint"):
            failures.append("successor_complete_runtime_document_mismatch")
    if smoke is not None and smoke != result.get("smoke_results"):
        failures.append("successor_complete_smoke_result_document_mismatch")
    if resume is not None and resume != result.get("resume_results"):
        failures.append("successor_complete_resume_result_document_mismatch")
    if smoke_evidence is not None:
        failures.extend(
            f"successor_complete_smoke_evidence:{failure}"
            for failure in validate_bridge_compile_evidence_document(smoke_evidence)
        )
    if resume_evidence is not None:
        expected_resume = {
            "schema_version": BRIDGE_RESUME_EVIDENCE_SCHEMA,
            "kind": BRIDGE_RESUME_EVIDENCE_KIND,
            "status": "PASS",
            "authorization_path": bridge.get("authorization_path"),
            "authorization_sha256": bridge.get("authorization_sha256"),
            "semantic_comparison_manifest_path": bridge.get("semantic_comparison_manifest_path"),
            "semantic_comparison_manifest_sha256": bridge.get(
                "semantic_comparison_manifest_sha256"
            ),
            "governed_run_contract_path": bridge.get("governed_run_contract_path"),
            "governed_run_contract_artifact_sha256": bridge.get(
                "governed_run_contract_artifact_sha256"
            ),
            "governed_run_contract_sha256": bridge.get("governed_run_contract_sha256"),
            "terminal_checkpoint_path": bridge.get("terminal_checkpoint_path"),
            "terminal_checkpoint_sha256": bridge.get("terminal_checkpoint_sha256"),
            "terminal_checkpoint_step": COMPLETION_STEP,
            "runtime_fingerprint_sha256": bridge.get("runtime_fingerprint_sha256"),
            "source_n2_checkpoint_path": HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"],
            "source_n2_checkpoint_sha256": N2_TERMINAL_CHECKPOINT_SHA256,
            "execution_counters": dict(ZERO_EXECUTION_COUNTERS),
            "state_equivalence": bridge.get("state_equivalence"),
        }
        if not _exact_state_equal(resume_evidence, expected_resume):
            failures.append("successor_complete_resume_evidence_mismatch")
    authenticated_source_checkpoint: Mapping[str, Any] | None = None
    semantic_authority = (
        authorization.get("semantic_comparison_manifest")
        if isinstance(authorization, Mapping)
        and isinstance(authorization.get("semantic_comparison_manifest"), Mapping)
        else None
    )
    state_equivalence = bridge.get("state_equivalence")
    exact_projection_ready = (
        isinstance(authorization, Mapping)
        and successor_authority is not None
        and semantic_authority is not None
        and isinstance(destination, Mapping)
        and isinstance(manifest, Mapping)
        and isinstance(grc, Mapping)
        and isinstance(runtime_wrapper, Mapping)
        and isinstance(runtime, Mapping)
        and isinstance(smoke, Mapping)
        and isinstance(resume, Mapping)
        and isinstance(smoke_evidence, Mapping)
        and isinstance(resume_evidence, Mapping)
        and isinstance(state_equivalence, Mapping)
    )
    if not exact_projection_ready:
        failures.append("successor_complete_exact_projection_dependencies_missing")
    else:
        try:
            authenticated_source_checkpoint, expected_grc = (
                _expected_successor_grc_from_authorized_source(
                    authorization_path=Path(str(bridge["authorization_path"])),
                    authorization_sha256=str(bridge["authorization_sha256"]),
                    authorization=authorization,
                    observed_runtime=runtime,
                    compile_evidence=smoke_evidence,
                )
            )
            expected_grc_artifact_sha256 = sha256_bytes(canonical_json_bytes(expected_grc))
            expected_runtime_wrapper = launch.stage_n_runtime_artifact_document(runtime)
            expected_runtime_artifact_sha256 = sha256_bytes(
                canonical_json_bytes(expected_runtime_wrapper)
            )
            expected_smoke_evidence_sha256 = sha256_bytes(canonical_json_bytes(smoke_evidence))
            expected_resume_evidence_sha256 = sha256_bytes(canonical_json_bytes(resume_evidence))
            expected_smoke = _bridge_check_result_document(
                kind=launch.STAGE_N_SMOKE_RESULT_KIND,
                authorization_sha256=str(bridge["authorization_sha256"]),
                governed_run_contract_sha256=str(expected_grc["governed_run_contract_sha256"]),
                checkpoint_path=Path(str(destination["terminal_checkpoint_path"])),
                checkpoint_sha256=str(bridge["terminal_checkpoint_sha256"]),
                runtime_fingerprint_sha256=str(successor_authority["runtime_fingerprint_sha256"]),
                evidence_path=Path(str(destination["smoke_evidence_path"])),
                evidence_sha256=expected_smoke_evidence_sha256,
            )
            expected_resume_check = _bridge_check_result_document(
                kind=launch.STAGE_N_RESUME_RESULT_KIND,
                authorization_sha256=str(bridge["authorization_sha256"]),
                governed_run_contract_sha256=str(expected_grc["governed_run_contract_sha256"]),
                checkpoint_path=Path(str(destination["terminal_checkpoint_path"])),
                checkpoint_sha256=str(bridge["terminal_checkpoint_sha256"]),
                runtime_fingerprint_sha256=str(successor_authority["runtime_fingerprint_sha256"]),
                evidence_path=Path(str(destination["resume_evidence_path"])),
                evidence_sha256=expected_resume_evidence_sha256,
            )

            if not _exact_state_equal(grc, expected_grc):
                failures.append("successor_complete_grc_not_exact_authorized_projection")
            if bridge.get("governed_run_contract_artifact_sha256") != expected_grc_artifact_sha256:
                failures.append("successor_complete_grc_artifact_not_canonical_projection")
            if not _exact_state_equal(runtime_wrapper, expected_runtime_wrapper):
                failures.append("successor_complete_runtime_not_canonical_projection")
            if (
                bridge.get("runtime_fingerprint_artifact_sha256")
                != expected_runtime_artifact_sha256
            ):
                failures.append("successor_complete_runtime_artifact_not_canonical_projection")
            if runtime.get("runtime_fingerprint_sha256") != successor_authority.get(
                "runtime_fingerprint_sha256"
            ):
                failures.append("successor_complete_runtime_not_authorized")
            if not _exact_state_equal(smoke, expected_smoke):
                failures.append("successor_complete_smoke_not_canonical_projection")
            if not _exact_state_equal(resume, expected_resume_check):
                failures.append("successor_complete_resume_not_canonical_projection")

            expected_complete = _bridge_complete_result_document(
                authorization=authorization,
                authorization_path=Path(str(bridge["authorization_path"])),
                authorization_sha256=str(bridge["authorization_sha256"]),
                semantic_manifest_path=Path(str(semantic_authority["path"])),
                grc_path=Path(str(destination["governed_run_contract_path"])),
                grc_artifact_sha256=expected_grc_artifact_sha256,
                grc_semantic_sha256=str(expected_grc["governed_run_contract_sha256"]),
                governed_run_contract=expected_grc,
                checkpoint_path=Path(str(destination["terminal_checkpoint_path"])),
                checkpoint_sha256=str(bridge["terminal_checkpoint_sha256"]),
                runtime_path=Path(str(destination["runtime_fingerprint_path"])),
                runtime_artifact_sha256=expected_runtime_artifact_sha256,
                smoke_result=expected_smoke,
                smoke_result_path=Path(str(destination["smoke_result_path"])),
                resume_result=expected_resume_check,
                resume_result_path=Path(str(destination["resume_result_path"])),
                state_equivalence=state_equivalence,
            )
            if not _exact_state_equal(result, expected_complete):
                failures.append("successor_complete_not_exact_authorized_projection")
        except (
            CompatibilityBridgeError,
            launch.LaunchContractError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            failures.append(f"successor_complete_exact_projection_failed:{type(exc).__name__}")
    checkpoint_path = _physical_artifact_path(
        Path(str(bridge.get("terminal_checkpoint_path", ""))),
        authorized_output_root=authorized_output_root,
        physical_output_root=physical_output_root,
    )
    if not checkpoint_path.is_file():
        failures.append("successor_complete_terminal_checkpoint_missing")
    elif file_sha256(checkpoint_path) != bridge.get("terminal_checkpoint_sha256"):
        failures.append("successor_complete_terminal_checkpoint_sha256_mismatch")
    elif not isinstance(grc, Mapping) or not isinstance(smoke_evidence, Mapping):
        failures.append("successor_complete_checkpoint_validation_dependencies_missing")
    else:
        checkpoint_validation = _revalidate_successor_checkpoint_state(
            checkpoint_path,
            destination_sha256=str(bridge.get("terminal_checkpoint_sha256", "")),
            governed_run_contract=grc,
            compile_evidence=smoke_evidence,
            recorded_state_equivalence=bridge.get("state_equivalence"),
            authenticated_source_checkpoint=authenticated_source_checkpoint,
        )
        failures.extend(checkpoint_validation["failures"])
    return list(dict.fromkeys(failures))


def validate_successor_complete_result(
    result_path: str | Path,
    *,
    expected_result_sha256: str | None = None,
) -> list[str]:
    """Authoritatively reopen and validate a published successor Stage-N COMPLETE result."""
    try:
        result_file = Path(result_path).resolve()
        _result_sha256, result = _read_json_mapping_snapshot(
            result_file,
            "successor Stage-N COMPLETE result",
            expected_sha256=expected_result_sha256,
        )
        bridge = result.get("stage_n_compatibility_bridge")
        _require(isinstance(bridge, Mapping), "successor COMPLETE bridge record is missing")
        preflight = validate_bridge_authorization(
            str(bridge.get("authorization_path", "")),
            semantic_comparison_manifest_path=str(
                bridge.get("semantic_comparison_manifest_path", "")
            ),
        )
        authorization = preflight.get("authorization_document")
        _require(
            isinstance(authorization, Mapping),
            "authoritative N3 preflight returned no authorization snapshot",
        )
        destination = authorization.get("destination")
        _require(isinstance(destination, Mapping), "N3 authorization destination is missing")
        _require(
            result_file == Path(str(destination.get("complete_result_path", ""))).resolve(),
            "successor COMPLETE path is not the exact authorized destination",
        )
        return _validate_successor_complete_result_document(
            result, authorization_preflight=preflight
        )
    except (CompatibilityBridgeError, OSError, ValueError, TypeError) as exc:
        return [f"successor_complete_authoritative_validation_failed:{exc}"]


def execute_authorized_bridge(
    *,
    authorization_path: str | Path,
    semantic_comparison_manifest_path: str | Path | None = None,
    exact_plan_path: str | Path | None = None,
    existing_n2_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Future production N3 entry point and the only artifact-publishing state machine."""
    _assert_successor_module_origins()
    authoritative = validate_bridge_authorization(
        authorization_path,
        semantic_comparison_manifest_path=semantic_comparison_manifest_path,
        exact_plan_path=exact_plan_path,
        existing_n2_evidence_path=existing_n2_evidence_path,
    )
    _require(
        authoritative.get("authorized") is True,
        "N3 authoritative preflight failed: "
        + ", ".join(str(value) for value in authoritative.get("failures", [])),
    )
    auth_path = Path(authorization_path).resolve()
    initial_authorization_sha, authorization = _read_json_mapping_snapshot(
        auth_path, "N3 authorization"
    )
    _require(
        initial_authorization_sha == authoritative.get("authorization_sha256")
        and _exact_state_equal(authorization, authoritative.get("authorization_document")),
        "N3 authorization changed after authoritative preflight",
    )
    manifest_binding = authorization.get("semantic_comparison_manifest")
    _require(isinstance(manifest_binding, Mapping), "N3 semantic manifest binding missing")
    manifest_path = Path(
        semantic_comparison_manifest_path or str(manifest_binding.get("path", ""))
    ).resolve()
    plan_path = Path(exact_plan_path or HISTORICAL_ARTIFACTS["exact_plan"]["path"]).resolve()
    n2_evidence_path = Path(
        existing_n2_evidence_path or HISTORICAL_ARTIFACTS["n2_zero_update_invariance"]["path"]
    ).resolve()
    binding = authorization.get("semantic_comparison_manifest")
    _require(isinstance(binding, Mapping), "N3 semantic manifest binding missing")
    initial_manifest_sha, manifest = _read_json_mapping_snapshot(
        manifest_path,
        "semantic comparison manifest",
        expected_sha256=str(binding.get("sha256", "")),
    )
    _require(
        initial_manifest_sha == authoritative.get("semantic_comparison_manifest_sha256")
        and _exact_state_equal(
            manifest, authoritative.get("semantic_comparison_manifest_document")
        ),
        "semantic manifest changed after authoritative preflight",
    )
    _require(
        plan_path.resolve() == Path(HISTORICAL_ARTIFACTS["exact_plan"]["path"]).resolve(),
        "bridge exact plan path is not the immutable historical plan",
    )
    _require(
        n2_evidence_path.resolve()
        == Path(HISTORICAL_ARTIFACTS["n2_zero_update_invariance"]["path"]).resolve(),
        "bridge N2 evidence path is not the immutable invariance artifact",
    )
    _exact_plan_sha256, exact_plan = _read_json_mapping_snapshot(
        plan_path,
        "exact Stage-P plan",
        expected_sha256=str(HISTORICAL_ARTIFACTS["exact_plan"]["sha256"]),
    )
    _n2_evidence_sha256, n2_evidence = _read_json_mapping_snapshot(
        n2_evidence_path,
        "N2 invariance evidence",
        expected_sha256=str(HISTORICAL_ARTIFACTS["n2_zero_update_invariance"]["sha256"]),
    )
    binding = authorization.get("semantic_comparison_manifest") or {}
    _require(
        Path(str(binding.get("path", ""))).resolve() == manifest_path.resolve(),
        "semantic manifest path is not authorized",
    )
    _require(
        initial_manifest_sha == binding.get("sha256"),
        "semantic manifest bytes are not authorized",
    )
    observed_successor = observe_successor_identity()
    observed_historical = observe_historical_identity()
    historical_runtime = _load_bound_historical_runtime_fingerprint()
    successor_root = Path(str(observed_successor["repository_root"])).resolve()
    preflight = _validate_bridge_authorization_claims(
        authorization,
        observed_successor=observed_successor,
        semantic_comparison_manifest=manifest,
        exact_plan=exact_plan,
        existing_n2_evidence=n2_evidence,
        historical_runtime_fingerprint=historical_runtime,
        reopen_source_artifacts=True,
        source_root=CANONICAL_TRAINING_ROOT,
        successor_root=successor_root,
    )
    _require(
        preflight["authorized"], "N3 bridge preflight failed: " + ", ".join(preflight["failures"])
    )
    destination = authorization.get("destination")
    _require(isinstance(destination, Mapping), "N3 bridge destination is missing")
    _require_destination_absent(destination)

    source_binding = HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]
    source_path = Path(source_binding["path"])
    _source_snapshot, source_checkpoint = _read_bound_checkpoint_snapshot(
        source_path,
        expected_sha256=str(source_binding["sha256"]),
        label="N2 terminal",
    )
    immutable_source = copy.deepcopy(dict(source_checkpoint))
    source_grc = immutable_source.get("governed_run_contract")
    source_state = immutable_source.get("governed_checkpoint_state")
    _require(isinstance(source_grc, Mapping), "N2 terminal checkpoint has no governed contract")
    _require(isinstance(source_state, Mapping), "N2 terminal checkpoint has no governed state")
    boundary = {
        **launch.derive_stage_n_completion_boundary(exact_plan),
        "exact_plan_path": str(plan_path),
        "exact_run_plan_sha256": _exact_plan_sha256,
    }
    boundary_failures = launch.validate_stage_n_boundary_agreement(
        {"exact_run_plan_sha256": EXACT_RUN_PLAN_SHA256},
        boundary,
        governed_run_contract=source_grc,
        checkpoint_step=immutable_source.get("global_step"),
        sampler_state=source_state,
    )
    _require(
        not boundary_failures,
        "N2 terminal checkpoint disagrees with exact-plan boundary: "
        + ", ".join(boundary_failures),
    )

    realizer_input = copy.deepcopy(immutable_source)
    realizer_input_before = copy.deepcopy(realizer_input)
    realization = _canonical_zero_update_compile_realization(
        realizer_input,
        authorization_sha256=initial_authorization_sha,
    )
    _require(isinstance(realization, Mapping), "compile realizer returned no result")
    _require(
        _exact_state_equal(realizer_input, realizer_input_before),
        "compile realization mutated its checkpoint input",
    )
    compile_evidence = realization.get("compile_evidence")
    counters = realization.get("execution_counters")
    live_state_proof = realization.get("live_state_proof")
    _require(isinstance(compile_evidence, Mapping), "compile realizer returned no evidence")
    _require(isinstance(counters, Mapping), "compile realizer returned no counters")
    _require(isinstance(live_state_proof, Mapping), "compile realizer returned no state proof")
    bridge_observations = compile_evidence.get("bridge_zero_update_observations")
    _require(
        isinstance(bridge_observations, Mapping),
        "compile evidence has no bridge zero-update observations",
    )
    _require(
        _exact_state_equal(
            counters,
            bridge_observations.get("execution_counters"),
        ),
        "compile result counters contradict embedded bridge evidence",
    )
    _require(
        _exact_state_equal(
            live_state_proof,
            bridge_observations.get("after_compile"),
        ),
        "compile result state proof contradicts embedded bridge evidence",
    )
    for field in (
        "model_parameters_bitwise_identical",
        "optimizer_state_equivalent",
        "scaler_state_equivalent",
        "rng_state_preserved",
        "all_parameter_grads_absent",
    ):
        _require(live_state_proof.get(field) is True, f"compile live-state proof failed:{field}")
    compile_failures = validate_bridge_compile_evidence_document(compile_evidence)
    _require(
        not compile_failures, "bridge compile realization failed: " + ", ".join(compile_failures)
    )
    _require(
        _exact_state_equal(source_checkpoint, immutable_source),
        "compile realization mutated the reopened source checkpoint",
    )

    authorization_sha = initial_authorization_sha
    successor_grc = _successor_grc(
        immutable_source,
        authorization_path=auth_path,
        authorization_sha256=authorization_sha,
        authorization=authorization,
        observed_runtime=observed_successor["runtime"],
        compile_evidence=compile_evidence,
    )
    grc_path = Path(destination["governed_run_contract_path"])
    checkpoint_path = Path(destination["terminal_checkpoint_path"])

    successor_checkpoint = copy.deepcopy(immutable_source)
    successor_checkpoint["governed_run_contract"] = successor_grc
    successor_checkpoint["governed_run_contract_sha256"] = successor_grc[
        "governed_run_contract_sha256"
    ]
    successor_checkpoint["kind"] = successor_grc.get("kind")
    successor_state = successor_checkpoint.get("governed_checkpoint_state")
    _require(isinstance(successor_state, Mapping), "source dynamic state is missing")
    successor_state = copy.deepcopy(dict(successor_state))
    successor_state["compile_evidence"] = dict(compile_evidence)
    successor_state["compile_evidence_sha256"] = compile_evidence.get("compile_evidence_sha256")
    successor_checkpoint["governed_checkpoint_state"] = successor_state
    before_save = validate_state_equivalence(
        immutable_source,
        successor_checkpoint,
        execution_counters=counters,
    )
    _require(
        before_save["equivalent"],
        "bridge state changed before save: " + ", ".join(before_save["failures"]),
    )
    output_root = Path(str(destination["output_root"]))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.n3-staging-{authorization_sha[:16]}-",
            dir=output_root.parent,
        )
    ).resolve()

    def staged_path(final_path: Path) -> Path:
        try:
            relative = final_path.resolve().relative_to(output_root.resolve())
        except ValueError as exc:
            raise CompatibilityBridgeError(
                f"bridge destination escapes output root: {final_path}"
            ) from exc
        return staging_root / relative

    staged_checkpoint_path = staged_path(checkpoint_path)
    _default_checkpoint_saver(staged_checkpoint_path, successor_checkpoint)
    _require(staged_checkpoint_path.is_file(), "bridge checkpoint saver did not stage a file")
    staged_checkpoint_snapshot, reopened = _read_bound_checkpoint_snapshot(
        staged_checkpoint_path,
        expected_sha256=None,
        label="staged successor",
    )
    after_save = validate_state_equivalence(
        immutable_source,
        reopened,
        execution_counters=counters,
    )
    _require(
        after_save["equivalent"],
        "bridge checkpoint changed on disk: " + ", ".join(after_save["failures"]),
    )
    checkpoint_binding_failures = _validate_successor_checkpoint_binding(
        reopened,
        governed_run_contract=successor_grc,
        compile_evidence=compile_evidence,
        require_live_cuda_validation=True,
    )
    _require(
        not checkpoint_binding_failures,
        "bridge successor checkpoint envelope failed: " + ", ".join(checkpoint_binding_failures),
    )
    checkpoint_sha = sha256_bytes(staged_checkpoint_snapshot)

    staged_grc = _atomic_publish_json(staged_path(grc_path), successor_grc)
    grc_publication = {"path": str(grc_path), "sha256": staged_grc["sha256"]}
    runtime_path = Path(destination["runtime_fingerprint_path"])
    runtime_document = launch.stage_n_runtime_artifact_document(observed_successor["runtime"])
    staged_runtime = _atomic_publish_json(staged_path(runtime_path), runtime_document)
    runtime_publication = {
        "path": str(runtime_path),
        "sha256": staged_runtime["sha256"],
    }
    smoke_evidence_path = Path(destination["smoke_evidence_path"])
    resume_evidence_path = Path(destination["resume_evidence_path"])
    staged_smoke_evidence = _atomic_publish_json(staged_path(smoke_evidence_path), compile_evidence)
    resume_evidence = _bridge_resume_evidence_document(
        authorization_path=auth_path,
        authorization_sha256=authorization_sha,
        semantic_manifest_path=manifest_path,
        semantic_manifest_sha256=initial_manifest_sha,
        governed_run_contract_path=grc_path,
        governed_run_contract_artifact_sha256=grc_publication["sha256"],
        governed_run_contract_sha256=successor_grc["governed_run_contract_sha256"],
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        runtime_fingerprint_sha256=authorization["successor"]["runtime_fingerprint_sha256"],
        execution_counters=counters,
        state_equivalence=after_save,
    )
    staged_resume_evidence = _atomic_publish_json(
        staged_path(resume_evidence_path), resume_evidence
    )
    smoke = _bridge_check_result_document(
        kind=launch.STAGE_N_SMOKE_RESULT_KIND,
        authorization_sha256=authorization_sha,
        governed_run_contract_sha256=successor_grc["governed_run_contract_sha256"],
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        runtime_fingerprint_sha256=authorization["successor"]["runtime_fingerprint_sha256"],
        evidence_path=smoke_evidence_path,
        evidence_sha256=staged_smoke_evidence["sha256"],
    )
    resume = _bridge_check_result_document(
        kind=launch.STAGE_N_RESUME_RESULT_KIND,
        authorization_sha256=authorization_sha,
        governed_run_contract_sha256=successor_grc["governed_run_contract_sha256"],
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        runtime_fingerprint_sha256=authorization["successor"]["runtime_fingerprint_sha256"],
        evidence_path=resume_evidence_path,
        evidence_sha256=staged_resume_evidence["sha256"],
    )
    smoke_result_path = Path(destination["smoke_result_path"])
    resume_result_path = Path(destination["resume_result_path"])
    staged_smoke = _atomic_publish_json(staged_path(smoke_result_path), smoke)
    staged_resume = _atomic_publish_json(staged_path(resume_result_path), resume)
    smoke_publication = {"path": str(smoke_result_path), "sha256": staged_smoke["sha256"]}
    resume_publication = {
        "path": str(resume_result_path),
        "sha256": staged_resume["sha256"],
    }
    complete = _bridge_complete_result_document(
        authorization=authorization,
        authorization_path=auth_path,
        authorization_sha256=authorization_sha,
        semantic_manifest_path=manifest_path,
        grc_path=grc_path,
        grc_artifact_sha256=grc_publication["sha256"],
        grc_semantic_sha256=successor_grc["governed_run_contract_sha256"],
        governed_run_contract=successor_grc,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        runtime_path=runtime_path,
        runtime_artifact_sha256=runtime_publication["sha256"],
        smoke_result=smoke,
        smoke_result_path=smoke_result_path,
        resume_result=resume,
        resume_result_path=resume_result_path,
        state_equivalence=after_save,
    )
    staged_complete = _atomic_publish_json(
        staged_path(Path(destination["complete_result_path"])), complete
    )
    provenance_failures = launch.validate_successor_stage_n_provenance(complete, required=True)
    _require(
        not provenance_failures,
        "bridge staged COMPLETE provenance failed: " + ", ".join(provenance_failures),
    )
    expected_staged_files = {
        staged_path(path).resolve() for path in _destination_paths(destination)
    }
    observed_staged_files = {path.resolve() for path in staging_root.rglob("*") if path.is_file()}
    _require(
        observed_staged_files == expected_staged_files,
        "bridge staging root contains an unexpected or missing artifact",
    )

    # Close the validation/use window immediately before the one atomic directory rename.
    fresh_authorization_sha, fresh_authorization = _read_json_mapping_snapshot(
        auth_path,
        "N3 authorization",
        expected_sha256=initial_authorization_sha,
    )
    fresh_manifest_sha, fresh_manifest = _read_json_mapping_snapshot(
        manifest_path,
        "semantic comparison manifest",
        expected_sha256=initial_manifest_sha,
    )
    fresh_exact_plan_sha, fresh_exact_plan = _read_json_mapping_snapshot(
        plan_path,
        "exact Stage-P plan",
        expected_sha256=_exact_plan_sha256,
    )
    fresh_n2_evidence_sha, fresh_n2_evidence = _read_json_mapping_snapshot(
        n2_evidence_path,
        "N2 invariance evidence",
        expected_sha256=_n2_evidence_sha256,
    )
    fresh_observed = observe_successor_identity()
    _require(
        _exact_state_equal(observe_historical_identity(), observed_historical),
        "historical worktree identity changed during bridge realization",
    )
    _require(
        fresh_authorization_sha == initial_authorization_sha
        and _exact_state_equal(fresh_authorization, authorization),
        "N3 authorization changed during bridge realization",
    )
    _require(
        fresh_manifest_sha == initial_manifest_sha and _exact_state_equal(fresh_manifest, manifest),
        "semantic comparison manifest changed during bridge realization",
    )
    _require(
        fresh_exact_plan_sha == _exact_plan_sha256
        and _exact_state_equal(fresh_exact_plan, exact_plan),
        "exact Stage-P plan changed during bridge realization",
    )
    _require(
        fresh_n2_evidence_sha == _n2_evidence_sha256
        and _exact_state_equal(fresh_n2_evidence, n2_evidence),
        "N2 invariance evidence changed during bridge realization",
    )
    _require(
        _exact_state_equal(fresh_observed, observed_successor),
        "successor code/runtime identity changed during bridge realization",
    )
    _require(
        file_sha256(source_path) == source_binding["sha256"],
        "N2 terminal checkpoint changed during bridge realization",
    )
    fresh_historical_runtime = _load_bound_historical_runtime_fingerprint()
    final_preflight = _validate_bridge_authorization_claims(
        fresh_authorization,
        observed_successor=fresh_observed,
        semantic_comparison_manifest=fresh_manifest,
        exact_plan=fresh_exact_plan,
        existing_n2_evidence=fresh_n2_evidence,
        historical_runtime_fingerprint=fresh_historical_runtime,
        reopen_source_artifacts=True,
        source_root=CANONICAL_TRAINING_ROOT,
        successor_root=successor_root,
    )
    final_preflight.update({
        "authorization_path": str(auth_path),
        "authorization_sha256": fresh_authorization_sha,
        "authorization_document": copy.deepcopy(dict(fresh_authorization)),
        "semantic_comparison_manifest_path": str(manifest_path),
        "semantic_comparison_manifest_sha256": fresh_manifest_sha,
        "semantic_comparison_manifest_document": copy.deepcopy(dict(fresh_manifest)),
    })

    _require(
        final_preflight["authorized"],
        "N3 bridge final preflight failed: " + ", ".join(final_preflight["failures"]),
    )
    _require_destination_absent(destination)
    staged_complete_document = _read_json_mapping(
        staged_path(Path(destination["complete_result_path"])),
        "staged successor Stage-N COMPLETE result",
    )
    _require(
        _exact_state_equal(staged_complete_document, complete),
        "staged COMPLETE bytes differ from the validated document",
    )
    complete_failures = _validate_successor_complete_result_document(
        staged_complete_document,
        authorization_preflight=final_preflight,
        physical_output_root=staging_root,
    )
    _require(
        not complete_failures,
        "bridge staged COMPLETE result failed: " + ", ".join(complete_failures),
    )
    # Nothing under output_root is visible until the fully validated directory is installed.
    # Linux RENAME_NOREPLACE closes the destination absence/use race without clobbering a
    # concurrently-created file, symlink, or directory.
    _fsync_directory(staging_root)
    _rename_directory_noreplace(staging_root, output_root)
    _fsync_directory(output_root.parent)
    publications = {
        "runtime": runtime_publication,
        "smoke": smoke_publication,
        "resume": resume_publication,
        "complete": {
            "path": str(destination["complete_result_path"]),
            "sha256": staged_complete["sha256"],
        },
    }
    return {
        "status": "COMPLETE_STOPPED_FOR_INDEPENDENT_REVIEW",
        "preflight": preflight,
        "execution_counters": counters,
        "state_equivalence": after_save,
        "governed_run_contract": grc_publication,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "results": publications,
        "stage_n_result_published": True,
        "stage_n_owner_accepted": False,
        "stage_o_authorized": False,
    }


BRIDGE_TOOL_RELPATH = "pretrain/stage_n_successor_head_compatibility_bridge_v1.py"


def bridge_tool_closure(root: str | Path | None = None) -> dict[str, Any]:
    """Bind the bridge plus the complete successor governed-trainer closure."""
    base = Path(root) if root is not None else Path(launch.ROOT)
    trainer = launch.trainer_execution_closure(base)
    closure = sorted({*trainer["derived_closure"], BRIDGE_TOOL_RELPATH})
    files: dict[str, str] = {}
    unbound: list[str] = []
    for relative in closure:
        path = base / relative
        if path.is_file():
            files[relative] = file_sha256(path)
        else:
            unbound.append(relative)
    digest = sha256_bytes(
        canonical_json_bytes({
            "schema_version": BRIDGE_CLOSURE_SCHEMA,
            "files": dict(sorted(files.items())),
        })
    )
    # Parse the bridge itself as an additional fail-closed import/syntax assertion.
    ast.parse((base / BRIDGE_TOOL_RELPATH).read_text(encoding="utf-8"), BRIDGE_TOOL_RELPATH)
    return {
        "bundle_schema_version": BRIDGE_CLOSURE_SCHEMA,
        "roots": [BRIDGE_TOOL_RELPATH, *trainer["roots"]],
        "derived_closure": closure,
        "derived_closure_count": len(closure),
        "files": files,
        "unbound_load_bearing_modules": unbound,
        "unbound_load_bearing_module_count": len(unbound),
        "BRIDGE_TOOL_BUNDLE_SHA256": digest,
        "successor_trainer_bundle_sha256": trainer["TRAINER_EXECUTION_BUNDLE_SHA256"],
    }


def bridge_tool_bundle_sha256(root: str | Path | None = None) -> str:
    return str(bridge_tool_closure(root)["BRIDGE_TOOL_BUNDLE_SHA256"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("authorization-template")
    subparsers.add_parser("closure")
    execute = subparsers.add_parser("execute")
    execute.add_argument("--authorization-path", required=True)
    execute.add_argument("--semantic-comparison-manifest-path")
    execute.add_argument("--exact-plan-path")
    execute.add_argument("--existing-n2-evidence-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Expose evidence commands and one reproducible future authorized execution command."""
    args = _parser().parse_args(argv)
    if args.command == "authorization-template":
        document = n3_authorization_template()
    elif args.command == "closure":
        document = bridge_tool_closure()
    else:
        document = execute_authorized_bridge(
            authorization_path=args.authorization_path,
            semantic_comparison_manifest_path=args.semantic_comparison_manifest_path,
            exact_plan_path=args.exact_plan_path,
            existing_n2_evidence_path=args.existing_n2_evidence_path,
        )
    print(canonical_json_bytes(document).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
