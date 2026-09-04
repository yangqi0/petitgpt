#!/usr/bin/env python3
"""Deterministic launch adapter for the accepted successor Stage-O trainer.

This adjacent tool owns only Python import topology and launch delegation.  It never
implements training policy, parser semantics, Gate A, model construction, checkpoint
handling, compilation, or the training loop.  In particular, importing this module imports
stdlib modules only; accepted PetitGPT modules are loaded only after their repository and
byte identities have been checked.
"""

from __future__ import annotations

# ``sys`` is a built-in module and therefore cannot be shadowed by the adjacent tools
# directory.  Refuse direct startup before resolving any filesystem-backed import unless
# Python has removed both the script directory and user site paths from import authority.
import sys

if __name__ == "__main__" and (not sys.flags.isolated or not sys.dont_write_bytecode):
    raise SystemExit("Stage-O adapter CLI requires Python flags -I -B")

import _imp
import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib._bootstrap as _importlib_bootstrap
import importlib._bootstrap_external as _importlib_bootstrap_external
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import CodeType, FunctionType, MappingProxyType, MemberDescriptorType, ModuleType
from typing import Any

ADAPTER_SCOPE = "STAGE_O_SUCCESSOR_LAUNCH_ADAPTER"
ADAPTER_AUTHORIZATION_SCHEMA = "petitgpt-stage-o-successor-launch-adapter-authorization-v1"
ADAPTER_TOOL_BUNDLE_SCHEMA = "petitgpt-stage-o-successor-launch-adapter-bundle-v1"
ADAPTER_PREFLIGHT_SCHEMA = "petitgpt-stage-o-successor-launch-adapter-preflight-v1"

HISTORICAL_CWD = Path("/workspace/petitgpt")
ACCEPTED_SUCCESSOR_ROOT = Path("/workspace/petitgpt_stage_n_result_publication_recovery_v1")
ACCEPTED_SUCCESSOR_BRANCH = "agent/stage-n-result-publication-recovery-v1"
ACCEPTED_SUCCESSOR_HEAD = "7686fd811642dd6246ca3a3c21a4bf43bc28cd3b"
ACCEPTED_TRAINER_BUNDLE_SHA256 = "1086af0b6821b2fdc4b2850371845c992f831dfcd84a6d504d2938fad003e75d"
EXPECTED_ADAPTER_BRANCH = "agent/stage-o-successor-launch-adapter-v1"

CANONICAL_LAUNCH_NAME = "pretrain.production_launch_contract_v1"
BARE_LAUNCH_NAME = "production_launch_contract_v1"
CANONICAL_TRAINER_NAME = "pretrain.train_pretrain_with_bench"
BARE_TRAINER_NAME = "train_pretrain_with_bench"

CANONICAL_LAUNCH_PATH = ACCEPTED_SUCCESSOR_ROOT / "pretrain/production_launch_contract_v1.py"
CANONICAL_LAUNCH_SHA256 = "9e858078e7e492bed6de3b3ce34395d44fb81f3f06aab59c9960d447b7bde861"
CANONICAL_TRAINER_PATH = ACCEPTED_SUCCESSOR_ROOT / "pretrain/train_pretrain_with_bench.py"
CANONICAL_TRAINER_SHA256 = "d9d7e61e5b30b5e24d49d92b2ea6bfc7557b6361d24a89467fdf06634d774fa4"
LAUNCH_RUNTIME_NAMESPACE_SCHEMA = "petitgpt-launch-runtime-namespace-v1"
LAUNCH_RUNTIME_PYTHON_VERSION = (3, 10, 12)
LAUNCH_RUNTIME_NAMESPACE_SHA256 = "99340d72e4bb85d8e62e6908599d0897e087fbdd30d2a48d806ed1dbdf4ccfb7"

STAGE_N_COMPLETE_RESULT_PATH = (
    ACCEPTED_SUCCESSOR_ROOT / "runs/n3_bridge_output_r3_2026-09-04/STAGE_N_COMPLETE_RESULT.json"
)
STAGE_N_COMPLETE_RESULT_SHA256 = "3f2d9029286bf9d0f8abe704aedef60e812d98efdac4049d6fbdff16895398d2"
STAGE_N_OWNER_ACCEPTANCE_PATH = ACCEPTED_SUCCESSOR_ROOT / (
    "runs/n_stage_n_owner_closeout_and_stage_o_preflight_v1_2026-09-04/"
    "STAGE_N_OWNER_ACCEPTANCE.json"
)
STAGE_N_OWNER_ACCEPTANCE_SHA256 = "0aec8cffd6e7f3395017b887523ad3a9dc0a109cf7744901a08320d58bfd90a6"

TRAINER_CLOSURE_ROOTS = (
    "pretrain/train_pretrain_with_bench.py",
    "pretrain/production_launch_contract_v1.py",
    "pretrain/run_plan_contract.py",
    "pretrain/dataset_pretrain.py",
)
TRAINER_BUNDLE_SCHEMA = "petitgpt-production-trainer-execution-bundle-v1"
ADAPTER_SCRIPT_RELPATH = "tools/stage_o_successor_launch_adapter_v1.py"

_EXPECTED_LOCAL_MODULES: Mapping[str, tuple[str, str]] = {
    "dataset_pretrain": (
        "pretrain/dataset_pretrain.py",
        "12a5ca8aceca5ab9b40c0d5a1cf5598abd6d3877de4c81a34ef8ff4b2eae6763",
    ),
    "sample": (
        "pretrain/sample.py",
        "6f6d839f9eaff645411f143144b1b660d6910b359232339fd9ee92e74d8c9b95",
    ),
    "pretrain.run_plan_contract": (
        "pretrain/run_plan_contract.py",
        "870c2f716d2b9d937d2d21d7083d5eb1647c0bd81053f138492dbff9c84a9b4e",
    ),
    "pretrain.stage_n_successor_head_compatibility_bridge_v1": (
        "pretrain/stage_n_successor_head_compatibility_bridge_v1.py",
        "5727573cd86efd3996e70f387c32cb2f3f355b7f1d8d6344e0ed6198e58b061a",
    ),
    "src": (
        "src/__init__.py",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "src.canonical_loss": (
        "src/canonical_loss.py",
        "c755e1af5d807e9bcc89e131ae6e0978e6a8844cdd9ed78626d27d7562ad84c0",
    ),
    "src.canonical_schedule": (
        "src/canonical_schedule.py",
        "fc488c6cf2f30c92f1ee0b7e3563032e8d3a8e8f5e3802140cb304c9a17a2b15",
    ),
    "src.model": (
        "src/model.py",
        "2bc9fa8ae16636837c4a2937301a2419d0ac92faa2cc27560dacbd29a5144dc2",
    ),
    "src.optim": (
        "src/optim.py",
        "13116860174f8557e6ab5a9b21011ecc15dfa0b82e0e6e394fff3554935e264a",
    ),
    "src.special_tokens": (
        "src/special_tokens.py",
        "f767b864d7c8e0cb5e2c166c4f019f3f14666dcbf2b944c7909db050e4cf1e96",
    ),
    "src.tracking": (
        "src/tracking.py",
        "df70f4f2b4ef81c0e85a88c59140c4dd198d82475aa84eeb736f83cf20ee5353",
    ),
}

_LAUNCH_FAMILY_TYPES = ("LaunchContractError", "_Missing", "ObservedForward")
_MISSING = object()
_SETUP_LOCK = threading.RLock()


class AdapterError(RuntimeError):
    """A fail-closed adapter identity, authorization, or import-topology error."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | os.PathLike[str]) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AdapterError(f"path cannot be resolved:{path!r}") from exc


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        completed.returncode == 0,
        f"git {' '.join(args)} failed for {root}: {completed.stderr.strip()}",
    )
    return completed.stdout.strip()


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def accepted_trainer_closure(root: Path | None = None) -> dict[str, Any]:
    """Re-derive the accepted trainer bundle using the accepted contract's AST policy."""

    base = _resolved(root or ACCEPTED_SUCCESSOR_ROOT)

    def resolve(module_name: str) -> list[str]:
        output: list[str] = []
        parts = module_name.split(".")
        for index in range(1, len(parts)):
            package_init = base.joinpath(*parts[:index], "__init__.py")
            if package_init.is_file():
                output.append(str(package_init.relative_to(base)))
        module_file = base.joinpath(*parts).with_suffix(".py")
        if module_file.is_file():
            output.append(str(module_file.relative_to(base)))
        package_file = base.joinpath(*parts, "__init__.py")
        if package_file.is_file():
            output.append(str(package_file.relative_to(base)))
        return output

    def resolve_bare(module_name: str) -> list[str]:
        module_file = base / "pretrain" / f"{module_name.split('.')[0]}.py"
        return [str(module_file.relative_to(base))] if module_file.is_file() else []

    seen: set[str] = set()
    graph: dict[str, list[str]] = {}
    external: set[str] = set()
    stack = list(TRAINER_CLOSURE_ROOTS)
    while stack:
        relative_path = stack.pop()
        if relative_path in seen:
            continue
        source_path = base / relative_path
        _require(source_path.is_file(), f"accepted trainer closure source missing:{source_path}")
        seen.add(relative_path)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative_path)
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_names = [node.module] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            for imported_name in imported_names:
                hits = resolve(imported_name) or resolve_bare(imported_name)
                if hits:
                    dependencies.update(hits)
                else:
                    external.add(imported_name.split(".")[0])
        graph[relative_path] = sorted(dependencies)
        stack.extend(item for item in dependencies if item not in seen)

    closure = sorted(seen)
    files = {relative_path: file_sha256(base / relative_path) for relative_path in closure}
    bundle = _sha256_bytes(
        canonical_json_bytes({
            "schema_version": TRAINER_BUNDLE_SCHEMA,
            "files": dict(sorted(files.items())),
        })
    )
    return {
        "bundle_schema_version": TRAINER_BUNDLE_SCHEMA,
        "roots": list(TRAINER_CLOSURE_ROOTS),
        "derived_closure": closure,
        "derived_closure_count": len(closure),
        "files": files,
        "external_non_repository_modules": sorted(external),
        "local_import_graph": graph,
        "unbound_load_bearing_modules": [],
        "unbound_load_bearing_module_count": 0,
        "TRAINER_EXECUTION_BUNDLE_SHA256": bundle,
    }


def accepted_trainer_identity() -> dict[str, Any]:
    """Validate the immutable successor identity without importing project code."""

    root = _resolved(ACCEPTED_SUCCESSOR_ROOT)
    _require(root == ACCEPTED_SUCCESSOR_ROOT, "accepted successor worktree path moved")
    _require(root.is_dir(), f"accepted successor worktree missing:{root}")
    repository_root = _resolved(_run_git(root, "rev-parse", "--show-toplevel"))
    _require(repository_root == root, "accepted successor git root mismatch")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    _require(branch == ACCEPTED_SUCCESSOR_BRANCH, "accepted successor branch changed")
    _require(head == ACCEPTED_SUCCESSOR_HEAD, "accepted successor HEAD changed")
    _require(not status, "accepted successor tracked bytes are dirty")

    closure = accepted_trainer_closure(root)
    bundle = closure["TRAINER_EXECUTION_BUNDLE_SHA256"]
    _require(bundle == ACCEPTED_TRAINER_BUNDLE_SHA256, "accepted trainer bundle changed")
    _require(
        file_sha256(CANONICAL_LAUNCH_PATH) == CANONICAL_LAUNCH_SHA256,
        "accepted launch-contract bytes changed",
    )
    _require(
        file_sha256(CANONICAL_TRAINER_PATH) == CANONICAL_TRAINER_SHA256,
        "accepted trainer bytes changed",
    )
    for module_name, (relative_path, expected_sha256) in _EXPECTED_LOCAL_MODULES.items():
        actual = file_sha256(root / relative_path)
        _require(actual == expected_sha256, f"accepted dependency bytes changed:{module_name}")
    _require(
        file_sha256(STAGE_N_COMPLETE_RESULT_PATH) == STAGE_N_COMPLETE_RESULT_SHA256,
        "accepted Stage-N COMPLETE result changed",
    )
    _require(
        file_sha256(STAGE_N_OWNER_ACCEPTANCE_PATH) == STAGE_N_OWNER_ACCEPTANCE_SHA256,
        "accepted Stage-N owner acceptance changed",
    )
    return {
        "worktree_path": str(root),
        "branch": branch,
        "head": head,
        "trainer_execution_bundle_sha256": bundle,
        "trainer_closure_count": closure["derived_closure_count"],
        "tracked_clean": True,
        "launch_contract_path": str(CANONICAL_LAUNCH_PATH),
        "launch_contract_sha256": CANONICAL_LAUNCH_SHA256,
        "trainer_path": str(CANONICAL_TRAINER_PATH),
        "trainer_sha256": CANONICAL_TRAINER_SHA256,
    }


def _adapter_root() -> Path:
    return Path(__file__).resolve().parents[1]


def adapter_tool_closure(root: Path | None = None) -> dict[str, Any]:
    """Derive and bind the standalone adapter closure from its exact source AST."""

    base = _resolved(root or _adapter_root())
    script_path = base / ADAPTER_SCRIPT_RELPATH
    _require(script_path.is_file(), f"adapter script missing:{script_path}")
    source = script_path.read_bytes()
    try:
        tree = ast.parse(source.decode("utf-8"), filename=ADAPTER_SCRIPT_RELPATH)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise AdapterError("adapter script cannot be parsed as UTF-8 Python") from exc

    imported_modules: set[str] = set()
    local_resolution_candidates: set[str] = set()
    relative_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
                local_resolution_candidates.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                relative_imports.add(relative_name)
                continue
            if not node.module:
                continue
            imported_modules.add(node.module)
            local_resolution_candidates.add(node.module)
            local_resolution_candidates.update(
                f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*"
            )

    local_dependencies: set[str] = set()
    for module_name in local_resolution_candidates:
        parts = module_name.split(".")
        candidates: set[Path] = set()
        for import_root in (base, script_path.parent):
            candidates.update({
                import_root.joinpath(*parts).with_suffix(".py"),
                import_root.joinpath(*parts, "__init__.py"),
            })
            candidates.update(
                import_root.joinpath(*parts[:index], "__init__.py")
                for index in range(1, len(parts))
            )
        for candidate in candidates:
            if candidate.is_file():
                local_dependencies.add(str(candidate.relative_to(base)))

    _require(
        not local_dependencies,
        "adapter source imports repository-local code:" + ",".join(sorted(local_dependencies)),
    )
    external_roots = {module_name.split(".", 1)[0] for module_name in imported_modules}
    stdlib_roots = sys.stdlib_module_names
    unbound = relative_imports | {name for name in external_roots if name not in stdlib_roots}
    _require(
        not unbound,
        "adapter source has unbound load-bearing imports:" + ",".join(sorted(unbound)),
    )
    _require(script_path.read_bytes() == source, "adapter script changed during closure derivation")

    files = {ADAPTER_SCRIPT_RELPATH: _sha256_bytes(source)}
    bundle = _sha256_bytes(
        canonical_json_bytes({
            "schema_version": ADAPTER_TOOL_BUNDLE_SCHEMA,
            "files": files,
        })
    )
    return {
        "bundle_schema_version": ADAPTER_TOOL_BUNDLE_SCHEMA,
        "roots": [ADAPTER_SCRIPT_RELPATH],
        "derived_closure": [ADAPTER_SCRIPT_RELPATH],
        "derived_closure_count": 1,
        "files": files,
        "external_non_repository_modules": sorted(external_roots),
        "local_import_graph": {ADAPTER_SCRIPT_RELPATH: sorted(local_dependencies)},
        "unbound_load_bearing_modules": sorted(unbound),
        "unbound_load_bearing_module_count": len(unbound),
        "ADAPTER_TOOL_BUNDLE_SHA256": bundle,
    }


def adapter_identity() -> dict[str, Any]:
    root = _adapter_root()
    repository_root = _resolved(_run_git(root, "rev-parse", "--show-toplevel"))
    _require(repository_root == root, "adapter script is not in the adapter worktree root")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _run_git(root, "rev-parse", "HEAD")
    tracked_status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    closure = adapter_tool_closure(root)
    script_path = root / ADAPTER_SCRIPT_RELPATH
    tracked = (
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", ADAPTER_SCRIPT_RELPATH],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    return {
        "worktree_path": str(root),
        "branch": branch,
        "head": head,
        "adapter_tool_path": str(script_path),
        "adapter_tool_sha256": closure["files"][ADAPTER_SCRIPT_RELPATH],
        "adapter_tool_closure_count": closure["derived_closure_count"],
        "adapter_tool_unbound_module_count": closure["unbound_load_bearing_module_count"],
        "adapter_tool_bundle_sha256": closure["ADAPTER_TOOL_BUNDLE_SHA256"],
        "tracked_clean": not tracked_status,
        "script_tracked": tracked,
    }


def adapter_authorization_template(
    *,
    runtime_fingerprint: Mapping[str, Any] | None = None,
    stage_o_authorization_path: str | os.PathLike[str] | None = None,
    stage_o_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an exact, machine-readable template that never authorizes execution."""

    accepted = accepted_trainer_identity()
    adapter = adapter_identity()
    stage_o_path = (
        str(_resolved(stage_o_authorization_path))
        if stage_o_authorization_path is not None
        else None
    )
    if stage_o_path is not None and stage_o_authorization_sha256 is None:
        stage_o_authorization_sha256 = file_sha256(stage_o_path)
    return {
        "schema_version": ADAPTER_AUTHORIZATION_SCHEMA,
        "scope": ADAPTER_SCOPE,
        "authorization_status": "NOT_AUTHORIZED",
        "authorizes_adapter_execution": False,
        "authorizes_training": False,
        "accepted_stage_o_trainer": {
            "worktree_path": accepted["worktree_path"],
            "branch": accepted["branch"],
            "head": accepted["head"],
            "trainer_execution_bundle_sha256": accepted["trainer_execution_bundle_sha256"],
            "tracked_clean_required": True,
        },
        "stage_o_launch_adapter": {
            "worktree_path": adapter["worktree_path"],
            "branch": adapter["branch"],
            "head": adapter["head"],
            "adapter_tool_path": adapter["adapter_tool_path"],
            "adapter_tool_sha256": adapter["adapter_tool_sha256"],
            "adapter_tool_closure_count": adapter["adapter_tool_closure_count"],
            "adapter_tool_unbound_module_count": adapter["adapter_tool_unbound_module_count"],
            "adapter_tool_bundle_sha256": adapter["adapter_tool_bundle_sha256"],
            "tracked_clean_required": True,
        },
        "canonical_sources": {
            "launch_contract": {
                "path": str(CANONICAL_LAUNCH_PATH),
                "sha256": CANONICAL_LAUNCH_SHA256,
            },
            "trainer": {
                "path": str(CANONICAL_TRAINER_PATH),
                "sha256": CANONICAL_TRAINER_SHA256,
            },
        },
        "canonical_cwd": str(HISTORICAL_CWD),
        "module_names": {
            "canonical_launch_contract": CANONICAL_LAUNCH_NAME,
            "bare_launch_contract": BARE_LAUNCH_NAME,
            "canonical_trainer": CANONICAL_TRAINER_NAME,
            "bare_trainer": BARE_TRAINER_NAME,
        },
        "stage_n_chain": {
            "complete_result_path": str(STAGE_N_COMPLETE_RESULT_PATH),
            "complete_result_sha256": STAGE_N_COMPLETE_RESULT_SHA256,
            "owner_acceptance_path": str(STAGE_N_OWNER_ACCEPTANCE_PATH),
            "owner_acceptance_sha256": STAGE_N_OWNER_ACCEPTANCE_SHA256,
        },
        "runtime_fingerprint": (
            dict(runtime_fingerprint) if runtime_fingerprint is not None else None
        ),
        "num_workers": 2,
        "stage_o_authorization_binding": {
            "policy": "EXACT_PATH_AND_SHA256",
            "path": stage_o_path,
            "sha256": stage_o_authorization_sha256,
        },
        "stage_o_command_derivation": {
            "policy": "EXACT_OWNER_BOUND_TRAINER_ARGV_AFTER_DOUBLE_DASH",
            "adapter_python_flags": ["-I", "-B"],
            "trainer_argv_field": "stage_o_trainer_argv",
            "exact_argv_match_required": True,
            "adapter_options_before_separator_only": True,
        },
        "authorized_by": None,
        "authorized_at": None,
        "note": (
            "This adapter authorization can authorize only execution of the reviewed adapter. "
            "It never authorizes Stage-O training. Execution also requires an independently "
            "owner-authorized Stage-O artifact bound by exact path, SHA, adapter identity, "
            "and trainer argv."
        ),
    }


_AUTHORIZATION_FIELDS = frozenset({
    "schema_version",
    "scope",
    "authorization_status",
    "authorizes_adapter_execution",
    "authorizes_training",
    "accepted_stage_o_trainer",
    "stage_o_launch_adapter",
    "canonical_sources",
    "canonical_cwd",
    "module_names",
    "stage_n_chain",
    "runtime_fingerprint",
    "num_workers",
    "stage_o_authorization_binding",
    "stage_o_command_derivation",
    "authorized_by",
    "authorized_at",
    "note",
})


def _expected_stage_o_adapter_identity(adapter: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "head": adapter["head"],
        "adapter_tool_bundle_sha256": adapter["adapter_tool_bundle_sha256"],
        "adapter_tool_path": adapter["adapter_tool_path"],
        "adapter_tool_sha256": adapter["adapter_tool_sha256"],
    }


def validate_adapter_authorization(
    document: Mapping[str, Any] | None,
    *,
    observed_runtime: Mapping[str, Any] | None = None,
    require_execution: bool = False,
    adapter_authorization_path: str | os.PathLike[str] | None = None,
    stage_o_authorization_path: str | os.PathLike[str] | None = None,
    stage_o_authorization: Mapping[str, Any] | None = None,
    stage_o_authorization_snapshot_path: str | os.PathLike[str] | None = None,
    stage_o_authorization_snapshot_sha256: str | None = None,
    trainer_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate identity separately from owner state and future Stage-O authority."""

    identity_failures: list[str] = []
    owner_state_failures: list[str] = []
    binding_failures: list[str] = []
    if not isinstance(document, Mapping):
        return {
            "identity_valid": False,
            "authorized": False,
            "runtime_checked": observed_runtime is not None,
            "failures": ["adapter_authorization_missing_or_malformed"],
            "identity_failures": ["adapter_authorization_missing_or_malformed"],
            "owner_state_failures": [],
            "binding_failures": [],
        }

    try:
        accepted = accepted_trainer_identity()
        adapter = adapter_identity()
    except AdapterError as exc:
        identity_failures.append(f"live_identity_error:{exc}")
        accepted = {}
        adapter = {}

    if set(document) != _AUTHORIZATION_FIELDS:
        identity_failures.append("adapter_authorization_field_set_mismatch")
    if document.get("schema_version") != ADAPTER_AUTHORIZATION_SCHEMA:
        identity_failures.append("adapter_authorization_schema_mismatch")
    if document.get("scope") != ADAPTER_SCOPE:
        identity_failures.append("adapter_authorization_scope_mismatch")
    if document.get("authorizes_training") is not False:
        identity_failures.append("adapter_authorization_must_never_authorize_training")
    if document.get("num_workers") != 2 or type(document.get("num_workers")) is not int:
        identity_failures.append("adapter_num_workers_must_equal_2")
    if document.get("canonical_cwd") != str(HISTORICAL_CWD):
        identity_failures.append("adapter_canonical_cwd_binding_mismatch")

    if accepted:
        expected_accepted = {
            "worktree_path": accepted["worktree_path"],
            "branch": accepted["branch"],
            "head": accepted["head"],
            "trainer_execution_bundle_sha256": accepted["trainer_execution_bundle_sha256"],
            "tracked_clean_required": True,
        }
        if not _strict_equal(document.get("accepted_stage_o_trainer"), expected_accepted):
            identity_failures.append("accepted_stage_o_trainer_identity_mismatch")
    if adapter:
        expected_adapter = {
            "worktree_path": adapter["worktree_path"],
            "branch": adapter["branch"],
            "head": adapter["head"],
            "adapter_tool_path": adapter["adapter_tool_path"],
            "adapter_tool_sha256": adapter["adapter_tool_sha256"],
            "adapter_tool_closure_count": adapter["adapter_tool_closure_count"],
            "adapter_tool_unbound_module_count": adapter["adapter_tool_unbound_module_count"],
            "adapter_tool_bundle_sha256": adapter["adapter_tool_bundle_sha256"],
            "tracked_clean_required": True,
        }
        if not _strict_equal(document.get("stage_o_launch_adapter"), expected_adapter):
            identity_failures.append("stage_o_launch_adapter_identity_mismatch")
        if adapter.get("branch") != EXPECTED_ADAPTER_BRANCH:
            identity_failures.append("adapter_branch_is_not_reviewed_branch")
        if not adapter.get("tracked_clean"):
            identity_failures.append("adapter_tracked_bytes_are_dirty")
        if not adapter.get("script_tracked"):
            identity_failures.append("adapter_script_is_not_tracked")

    expected_sources = {
        "launch_contract": {
            "path": str(CANONICAL_LAUNCH_PATH),
            "sha256": CANONICAL_LAUNCH_SHA256,
        },
        "trainer": {
            "path": str(CANONICAL_TRAINER_PATH),
            "sha256": CANONICAL_TRAINER_SHA256,
        },
    }
    if not _strict_equal(document.get("canonical_sources"), expected_sources):
        identity_failures.append("canonical_source_binding_mismatch")
    expected_names = {
        "canonical_launch_contract": CANONICAL_LAUNCH_NAME,
        "bare_launch_contract": BARE_LAUNCH_NAME,
        "canonical_trainer": CANONICAL_TRAINER_NAME,
        "bare_trainer": BARE_TRAINER_NAME,
    }
    if not _strict_equal(document.get("module_names"), expected_names):
        identity_failures.append("module_name_binding_mismatch")
    expected_stage_n = {
        "complete_result_path": str(STAGE_N_COMPLETE_RESULT_PATH),
        "complete_result_sha256": STAGE_N_COMPLETE_RESULT_SHA256,
        "owner_acceptance_path": str(STAGE_N_OWNER_ACCEPTANCE_PATH),
        "owner_acceptance_sha256": STAGE_N_OWNER_ACCEPTANCE_SHA256,
    }
    if not _strict_equal(document.get("stage_n_chain"), expected_stage_n):
        identity_failures.append("accepted_stage_n_chain_binding_mismatch")
    expected_policy = {
        "policy": "EXACT_OWNER_BOUND_TRAINER_ARGV_AFTER_DOUBLE_DASH",
        "adapter_python_flags": ["-I", "-B"],
        "trainer_argv_field": "stage_o_trainer_argv",
        "exact_argv_match_required": True,
        "adapter_options_before_separator_only": True,
    }
    if not _strict_equal(document.get("stage_o_command_derivation"), expected_policy):
        identity_failures.append("stage_o_command_derivation_policy_mismatch")

    runtime_checked = observed_runtime is not None
    if observed_runtime is not None:
        if not _strict_equal(document.get("runtime_fingerprint"), dict(observed_runtime)):
            identity_failures.append("adapter_runtime_fingerprint_mismatch")
        if observed_runtime.get("num_workers") != 2:
            identity_failures.append("observed_runtime_num_workers_mismatch")
        if observed_runtime.get("canonical_cwd") != str(HISTORICAL_CWD):
            identity_failures.append("observed_runtime_canonical_cwd_mismatch")
    elif require_execution and not isinstance(document.get("runtime_fingerprint"), Mapping):
        identity_failures.append("adapter_runtime_fingerprint_missing")

    if document.get("authorization_status") != "AUTHORIZED":
        owner_state_failures.append("adapter_authorization_status_not_authorized")
    if document.get("authorizes_adapter_execution") is not True:
        owner_state_failures.append("adapter_execution_not_authorized")
    if require_execution:
        for owner_field in ("authorized_by", "authorized_at"):
            value = document.get(owner_field)
            if not isinstance(value, str) or not value.strip():
                owner_state_failures.append(f"adapter_owner_field_missing:{owner_field}")

    binding = document.get("stage_o_authorization_binding")
    if not isinstance(binding, Mapping) or set(binding) != {"policy", "path", "sha256"}:
        binding_failures.append("stage_o_authorization_binding_malformed")
        binding = {}
    elif binding.get("policy") != "EXACT_PATH_AND_SHA256":
        binding_failures.append("stage_o_authorization_binding_policy_mismatch")

    binding_supplied = any(
        value is not None
        for value in (
            stage_o_authorization_path,
            stage_o_authorization,
            stage_o_authorization_snapshot_path,
            stage_o_authorization_snapshot_sha256,
            trainer_argv,
        )
    )
    if require_execution and adapter_authorization_path is None:
        binding_failures.append("adapter_authorization_path_required")
    if require_execution or binding_supplied:
        if stage_o_authorization_path is None:
            binding_failures.append("stage_o_authorization_path_required")
        else:
            actual_stage_o_path = str(_resolved(stage_o_authorization_path))
            if binding.get("path") != actual_stage_o_path:
                binding_failures.append("stage_o_authorization_path_mismatch")
            snapshot_path_supplied = stage_o_authorization_snapshot_path is not None
            snapshot_sha_supplied = stage_o_authorization_snapshot_sha256 is not None
            exact_snapshot_sha: str | None = None
            if snapshot_path_supplied != snapshot_sha_supplied:
                binding_failures.append("stage_o_authorization_snapshot_incomplete")
            elif snapshot_path_supplied:
                exact_snapshot_path = str(_resolved(stage_o_authorization_snapshot_path))
                if exact_snapshot_path != actual_stage_o_path:
                    binding_failures.append("stage_o_authorization_snapshot_path_mismatch")
                exact_snapshot_sha = stage_o_authorization_snapshot_sha256
            else:
                try:
                    snapshot_document, _snapshot_bytes, exact_snapshot_sha, exact_snapshot_path = (
                        _load_json_snapshot(actual_stage_o_path)
                    )
                except AdapterError:
                    binding_failures.append("stage_o_authorization_snapshot_unavailable")
                else:
                    if exact_snapshot_path != actual_stage_o_path:
                        binding_failures.append("stage_o_authorization_snapshot_path_mismatch")
                    if isinstance(stage_o_authorization, Mapping) and not _strict_equal(
                        snapshot_document, stage_o_authorization
                    ):
                        binding_failures.append("stage_o_authorization_snapshot_document_mismatch")
            if exact_snapshot_sha is not None and binding.get("sha256") != exact_snapshot_sha:
                binding_failures.append("stage_o_authorization_sha256_mismatch")
        if not isinstance(stage_o_authorization, Mapping):
            binding_failures.append("stage_o_authorization_document_missing")
        else:
            if (
                require_execution
                and stage_o_authorization.get("authorization_status") != "AUTHORIZED"
            ):
                binding_failures.append("stage_o_authorization_status_not_authorized")
            if require_execution:
                for owner_field in ("authorized_by", "authorized_at"):
                    owner_value = stage_o_authorization.get(owner_field)
                    if not isinstance(owner_value, str) or not owner_value.strip():
                        binding_failures.append(f"stage_o_owner_field_missing:{owner_field}")
            if stage_o_authorization.get("allowed_scope") != "STAGE_O":
                binding_failures.append("stage_o_authorization_scope_mismatch")
            if adapter and not _strict_equal(
                stage_o_authorization.get("stage_o_launch_adapter_identity"),
                _expected_stage_o_adapter_identity(adapter),
            ):
                binding_failures.append("stage_o_authorization_adapter_identity_mismatch")
            bound_argv = stage_o_authorization.get("stage_o_trainer_argv")
            if not isinstance(bound_argv, list) or not all(
                isinstance(item, str) for item in bound_argv
            ):
                binding_failures.append("stage_o_authorization_trainer_argv_missing")
            elif trainer_argv is None or list(trainer_argv) != bound_argv:
                binding_failures.append("stage_o_trainer_argv_mismatch")
            elif require_execution and stage_o_authorization_path is not None:
                try:
                    validate_governed_trainer_argv(
                        trainer_argv,
                        stage_o_authorization_path,
                    )
                except AdapterError as exc:
                    binding_failures.append(f"stage_o_trainer_argv_not_governed:{exc}")

    identity_failures = list(dict.fromkeys(identity_failures))
    owner_state_failures = list(dict.fromkeys(owner_state_failures))
    binding_failures = list(dict.fromkeys(binding_failures))
    failures = identity_failures + binding_failures + owner_state_failures
    return {
        "identity_valid": not identity_failures,
        "authorized": require_execution and not failures,
        "runtime_checked": runtime_checked,
        "failures": failures,
        "identity_failures": identity_failures,
        "owner_state_failures": owner_state_failures,
        "binding_failures": binding_failures,
    }


def _module_file_path(module: object) -> Path | None:
    try:
        value = getattr(module, "__file__", None)
        if value is None or not os.fspath(value):
            return None
        return Path(os.fspath(value)).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _module_origin_path(module: object) -> Path | None:
    try:
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        return Path(os.fspath(origin)).resolve() if origin else None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _validate_module_object(
    module: object,
    *,
    canonical_name: str,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> ModuleType:
    _require(isinstance(module, ModuleType), f"{label} is not a module object")
    _require(
        _module_file_path(module) == expected_path,
        f"{label} is not backed by reviewed path {expected_path}",
    )
    _require(
        _module_origin_path(module) == expected_path,
        f"{label} has missing or different __spec__.origin",
    )
    spec = getattr(module, "__spec__", None)
    _require(getattr(module, "__name__", None) == canonical_name, f"{label} __name__ mismatch")
    _require(
        getattr(spec, "name", None) == canonical_name,
        f"{label} was not created with canonical spec name {canonical_name}",
    )
    is_package = getattr(spec, "submodule_search_locations", None) is not None
    expected_package = canonical_name if is_package else canonical_name.rpartition(".")[0]
    _require(
        getattr(module, "__package__", None) == expected_package,
        f"{label} __package__ mismatch",
    )
    _require(getattr(spec, "parent", None) == expected_package, f"{label} spec parent mismatch")
    loader = getattr(module, "__loader__", None)
    spec_loader = getattr(spec, "loader", None)
    _require(
        loader is spec_loader and isinstance(spec_loader, importlib.machinery.SourceFileLoader),
        f"{label} loader/spec.loader mismatch",
    )
    _require(getattr(spec_loader, "name", None) == canonical_name, f"{label} loader name mismatch")
    try:
        loader_path = Path(os.fspath(spec_loader.path)).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AdapterError(f"{label} loader path is invalid") from exc
    _require(loader_path == expected_path, f"{label} loader path mismatch")
    _require(
        not bool(getattr(spec, "_initializing", False)),
        f"{label} is partially initialized",
    )
    if is_package:
        expected_package_path = expected_path.parent
        try:
            module_paths = tuple(Path(os.fspath(item)).resolve() for item in module.__path__)
            spec_paths = tuple(
                Path(os.fspath(item)).resolve()
                for item in (getattr(spec, "submodule_search_locations", None) or ())
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AdapterError(f"{label} package search metadata is invalid") from exc
        _require(
            module_paths == (expected_package_path,),
            f"{label} package path is not the one accepted source path",
        )
        _require(
            spec_paths == (expected_package_path,),
            f"{label} spec package path is not the one accepted source path",
        )
    _require(expected_path.is_file(), f"reviewed module source missing:{expected_path}")
    _require(
        file_sha256(expected_path) == expected_sha256,
        f"reviewed module source SHA changed:{expected_path}",
    )
    return module


def _loaded_module_path_inventory() -> dict[Path, tuple[ModuleType, ...]]:
    found: dict[Path, dict[int, ModuleType]] = {}
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        for path in {_module_file_path(module), _module_origin_path(module)} - {None}:
            found.setdefault(path, {})[id(module)] = module
    return {path: tuple(modules.values()) for path, modules in found.items()}


def _exact_path_module_objects(
    expected_path: Path,
    inventory: Mapping[Path, Sequence[ModuleType]] | None = None,
) -> list[ModuleType]:
    if inventory is None:
        inventory = _loaded_module_path_inventory()
    return list(inventory.get(expected_path, ()))


def _validate_pretrain_package(module: object, *, label: str) -> ModuleType:
    expected_path = (ACCEPTED_SUCCESSOR_ROOT / "pretrain").resolve()
    _require(isinstance(module, ModuleType), f"{label} is not a module object")
    spec = getattr(module, "__spec__", None)
    _require(getattr(module, "__name__", None) == "pretrain", f"{label} __name__ mismatch")
    _require(getattr(module, "__package__", None) == "pretrain", f"{label} package mismatch")
    _require(getattr(spec, "name", None) == "pretrain", f"{label} spec name mismatch")
    _require(getattr(spec, "parent", None) == "pretrain", f"{label} spec parent mismatch")
    loader = getattr(module, "__loader__", _MISSING)
    spec_loader = getattr(spec, "loader", _MISSING)
    _require(loader is spec_loader, f"{label} loader/spec.loader mismatch")
    _require(
        loader is None or isinstance(loader, _importlib_bootstrap_external._NamespaceLoader),
        f"{label} is not a fixed namespace-package loader",
    )
    _require(getattr(spec, "origin", _MISSING) is None, f"{label} namespace origin mismatch")
    _require(not bool(getattr(spec, "_initializing", False)), f"{label} is partial")
    _require(hasattr(module, "__path__"), f"{label} is not a package")
    try:
        module_paths = tuple(Path(os.fspath(item)).resolve() for item in module.__path__)
        spec_paths = tuple(
            Path(os.fspath(item)).resolve()
            for item in (getattr(spec, "submodule_search_locations", None) or ())
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AdapterError(f"{label} has invalid package search metadata") from exc
    _require(
        module_paths == (expected_path,),
        f"{label} does not have the one accepted successor package path",
    )
    _require(
        spec_paths == (expected_path,),
        f"{label} spec does not have the one accepted successor package path",
    )
    _require(
        getattr(module, "__file__", None) in (None, ""),
        f"{label} is not the expected fixed namespace package",
    )
    return module


def _new_fixed_pretrain_package() -> ModuleType:
    package_path = str((ACCEPTED_SUCCESSOR_ROOT / "pretrain").resolve())
    spec = importlib.machinery.ModuleSpec("pretrain", loader=None, is_package=True)
    spec.submodule_search_locations = [package_path]
    module = ModuleType("pretrain")
    module.__package__ = "pretrain"
    module.__loader__ = None
    module.__spec__ = spec
    module.__path__ = [package_path]  # type: ignore[attr-defined]
    return module


@dataclass(frozen=True)
class AcceptedModuleTopology:
    pretrain_package: ModuleType
    launch_contract: ModuleType
    trainer: ModuleType | None
    launch_imported_symbols: tuple[str, ...]


@dataclass(frozen=True)
class _OwnedModuleBaseline:
    module: ModuleType
    sys_module_bindings: tuple[str, ...]
    namespace_token: tuple[Any, ...]
    retained_objects: tuple[object, ...]


_OWNED_RUNTIME_MODULE_NAMES = tuple(
    dict.fromkeys((CANONICAL_LAUNCH_NAME, *_EXPECTED_LOCAL_MODULES, CANONICAL_TRAINER_NAME))
)
_CONTROLLED_TRAINER_DEPENDENCY_ORDER = (
    "src",
    "src.canonical_loss",
    "src.canonical_schedule",
    "src.model",
    "src.optim",
    "src.special_tokens",
    "src.tracking",
    "pretrain.run_plan_contract",
    "dataset_pretrain",
    "sample",
)
_MODULE_RUNTIME_METADATA = frozenset({
    "__builtins__",
    "__cached__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
})
_OWNED_MODULE_BASELINES: dict[str, _OwnedModuleBaseline] = {}


def _runtime_integrity_token(
    value: object,
    *,
    owner_name: str,
    owner_globals: dict[str, Any],
    active: set[int],
    retained: list[object],
) -> tuple[Any, ...]:
    """Capture identity plus mutable/executable structure without invoking project code."""

    if value is None or isinstance(value, (bool, int, str, bytes)):
        return ("literal", type(value).__name__, value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())

    identity = id(value)
    retained.append(value)
    if identity in active:
        return ("cycle", identity)
    active.add(identity)
    try:
        recurse = lambda item: _runtime_integrity_token(  # noqa: E731
            item,
            owner_name=owner_name,
            owner_globals=owner_globals,
            active=active,
            retained=retained,
        )
        if isinstance(value, ModuleType):
            return ("module", identity, value.__name__)
        if isinstance(value, FunctionType) and (
            value.__globals__ is owner_globals or value.__module__ == owner_name
        ):
            closure: list[tuple[Any, ...]] = []
            for cell in value.__closure__ or ():
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    closure.append(("empty_cell",))
                else:
                    closure.append(recurse(cell_value))
            return (
                "function",
                identity,
                id(value.__code__),
                id(value.__globals__),
                value.__name__,
                value.__qualname__,
                value.__module__,
                recurse(value.__defaults__),
                recurse(value.__kwdefaults__),
                recurse(value.__annotations__),
                recurse(value.__dict__),
                tuple(closure),
            )
        if isinstance(value, type) and value.__module__ == owner_name:
            return (
                "class",
                identity,
                value.__name__,
                value.__qualname__,
                id(type(value)),
                tuple(id(base) for base in value.__bases__),
                tuple((name, recurse(member)) for name, member in sorted(value.__dict__.items())),
            )
        if isinstance(value, property):
            return (
                "property",
                identity,
                recurse(value.fget),
                recurse(value.fset),
                recurse(value.fdel),
                recurse(value.__doc__),
            )
        if isinstance(value, (staticmethod, classmethod)):
            return (type(value).__name__, identity, recurse(value.__func__))
        if isinstance(value, (dict, MappingProxyType)):
            return (
                type(value).__name__,
                identity,
                tuple((recurse(key), recurse(item)) for key, item in value.items()),
            )
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, identity, tuple(recurse(item) for item in value))
        if isinstance(value, (set, frozenset)):
            items = [recurse(item) for item in value]
            return (type(value).__name__, identity, tuple(sorted(items, key=repr)))
        if isinstance(value, bytearray):
            return ("bytearray", identity, bytes(value))
        value_type = type(value)
        if value_type.__module__ == "dataclasses" and value_type.__qualname__ in {
            "Field",
            "_DataclassParams",
        }:
            slots = getattr(value_type, "__slots__", ())
            _require(
                isinstance(slots, tuple) and all(isinstance(name, str) for name in slots),
                f"unsupported dataclass metadata slots:{value_type.__qualname__}",
            )
            return (
                "dataclass_metadata",
                identity,
                value_type.__qualname__,
                tuple((name, recurse(getattr(value, name))) for name in slots),
            )
        return (
            "identity",
            identity,
            value_type.__module__,
            value_type.__qualname__,
        )
    finally:
        active.remove(identity)


def _module_namespace_token(
    module: ModuleType,
    names: Sequence[str] | None = None,
) -> tuple[tuple[Any, ...], tuple[object, ...]]:
    retained: list[object] = []
    active: set[int] = set()
    if names is None:
        names = tuple(sorted(set(module.__dict__) - _MODULE_RUNTIME_METADATA))
    else:
        current_names = tuple(sorted(set(module.__dict__) - _MODULE_RUNTIME_METADATA))
        _require(
            current_names == tuple(names),
            f"adapter-owned module namespace names changed:{module.__name__}",
        )
    token = tuple(
        (
            name,
            _runtime_integrity_token(
                module.__dict__[name],
                owner_name=module.__name__,
                owner_globals=module.__dict__,
                active=active,
                retained=retained,
            ),
        )
        for name in names
    )
    return token, tuple(retained)


def _capture_owned_module_baseline(module: ModuleType) -> _OwnedModuleBaseline:
    _validate_owned_module_builtins(module)
    namespace_token, retained = _module_namespace_token(module)
    bindings = tuple(sorted(name for name, value in sys.modules.items() if value is module))
    return _OwnedModuleBaseline(module, bindings, namespace_token, retained)


def _validate_owned_module_baseline(
    module_name: str,
    baseline: _OwnedModuleBaseline,
) -> None:
    _require(
        sys.modules.get(module_name) is baseline.module,
        f"adapter-owned canonical module binding changed:{module_name}",
    )
    _validate_owned_module_builtins(baseline.module)
    for binding in baseline.sys_module_bindings:
        _require(
            sys.modules.get(binding) is baseline.module,
            f"adapter-owned module alias changed:{binding}",
        )
    baseline_names = tuple(item[0] for item in baseline.namespace_token)
    namespace_token, _retained = _module_namespace_token(baseline.module, baseline_names)
    _require(
        namespace_token == baseline.namespace_token,
        f"adapter-owned module executable/export surface changed:{module_name}",
    )


def _validate_owned_module_builtins(module: ModuleType) -> None:
    builtins_module = sys.modules.get("builtins")
    _require(isinstance(builtins_module, ModuleType), "canonical builtins module is missing")
    _require(
        module.__dict__.get("__builtins__") is builtins_module.__dict__,
        f"adapter-owned module __builtins__ binding changed:{module.__name__}",
    )


@dataclass
class _ImportSnapshot:
    sys_modules: dict[str, object]
    sys_path: list[str]
    package_attributes: dict[tuple[str, str], object]


_PACKAGE_CHILDREN = {
    "pretrain": (
        "production_launch_contract_v1",
        "train_pretrain_with_bench",
        "run_plan_contract",
        "stage_n_successor_head_compatibility_bridge_v1",
    ),
    "src": (
        "canonical_loss",
        "canonical_schedule",
        "model",
        "optim",
        "special_tokens",
        "tracking",
    ),
}


def _snapshot_import_state() -> _ImportSnapshot:
    attributes: dict[tuple[str, str], object] = {}
    for package_name, child_names in _PACKAGE_CHILDREN.items():
        package = sys.modules.get(package_name)
        if not isinstance(package, ModuleType):
            continue
        for child_name in child_names:
            attributes[(package_name, child_name)] = package.__dict__.get(child_name, _MISSING)
    return _ImportSnapshot(dict(sys.modules), list(sys.path), attributes)


def _is_accepted_project_module(module: object) -> bool:
    for path in {_module_file_path(module), _module_origin_path(module)} - {None}:
        try:
            path.relative_to(ACCEPTED_SUCCESSOR_ROOT)
        except ValueError:
            continue
        return True
    return False


def _rollback_import_state(snapshot: _ImportSnapshot, attempted: set[int]) -> None:
    explicit_names = {
        "pretrain",
        CANONICAL_LAUNCH_NAME,
        BARE_LAUNCH_NAME,
        CANONICAL_TRAINER_NAME,
        BARE_TRAINER_NAME,
        *_EXPECTED_LOCAL_MODULES,
    }
    for name, current in tuple(sys.modules.items()):
        previous = snapshot.sys_modules.get(name, _MISSING)
        if previous is current:
            continue
        if (
            name in explicit_names
            or id(current) in attempted
            or _is_accepted_project_module(current)
        ):
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous  # type: ignore[assignment]
    for name in explicit_names:
        previous = snapshot.sys_modules.get(name, _MISSING)
        if previous is _MISSING:
            if name in sys.modules and id(sys.modules[name]) in attempted:
                sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous  # type: ignore[assignment]

    for (package_name, child_name), previous in snapshot.package_attributes.items():
        package = snapshot.sys_modules.get(package_name)
        if not isinstance(package, ModuleType):
            continue
        if previous is _MISSING:
            current = package.__dict__.get(child_name, _MISSING)
            if current is not _MISSING and (
                id(current) in attempted or _is_accepted_project_module(current)
            ):
                package.__dict__.pop(child_name, None)
        else:
            package.__dict__[child_name] = previous
    sys.path[:] = snapshot.sys_path


@contextmanager
def _held_import_locks(names: Sequence[str]):
    # Preserve the caller's explicit public-binding order.  Dependency locks are deliberately
    # left to their normal imports to avoid inversion with an already-running import; a module
    # can enter the trust registry only when this transaction itself recorded its object id.
    locks = [_importlib_bootstrap._get_module_lock(name) for name in dict.fromkeys(names)]
    acquired: list[Any] = []
    try:
        for lock in locks:
            lock.acquire()
            acquired.append(lock)
        _imp.acquire_lock()
        try:
            yield
        finally:
            _imp.release_lock()
    finally:
        for lock in reversed(acquired):
            lock.release()


def _owned_module_candidates(
    module_name: str,
    inventory: Mapping[Path, Sequence[ModuleType]],
) -> list[ModuleType]:
    candidates: dict[int, ModuleType] = {}
    binding_names = [module_name]
    if module_name == CANONICAL_LAUNCH_NAME:
        binding_names.append(BARE_LAUNCH_NAME)
    elif module_name == CANONICAL_TRAINER_NAME:
        binding_names.append(BARE_TRAINER_NAME)
    for binding_name in binding_names:
        value = sys.modules.get(binding_name)
        if isinstance(value, ModuleType):
            candidates[id(value)] = value

    parent_name, separator, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name) if separator else None
    if isinstance(parent, ModuleType):
        value = parent.__dict__.get(child_name)
        if isinstance(value, ModuleType):
            candidates[id(value)] = value

    if module_name == CANONICAL_LAUNCH_NAME:
        expected_path = CANONICAL_LAUNCH_PATH
    elif module_name == CANONICAL_TRAINER_NAME:
        expected_path = CANONICAL_TRAINER_PATH
    else:
        relative_path, _sha256 = _EXPECTED_LOCAL_MODULES[module_name]
        expected_path = (ACCEPTED_SUCCESSOR_ROOT / relative_path).resolve()
    for module in _exact_path_module_objects(expected_path, inventory):
        candidates[id(module)] = module
    return list(candidates.values())


def _validate_owned_module_provenance(
    baselines: Mapping[str, _OwnedModuleBaseline],
    inventory: Mapping[Path, Sequence[ModuleType]],
) -> None:
    """Refuse unowned accepted objects and revalidate every retained owned surface."""

    for module_name in _OWNED_RUNTIME_MODULE_NAMES:
        candidates = _owned_module_candidates(module_name, inventory)
        baseline = baselines.get(module_name)
        if baseline is not None:
            _require(
                isinstance(baseline, _OwnedModuleBaseline),
                f"adapter-owned module baseline changed:{module_name}",
            )
            _validate_owned_module_baseline(module_name, baseline)
        if not candidates:
            continue
        # A clean, exact canonical-first launch import is the sole allowed external
        # preload.  Its full source-derived runtime manifest is checked below before it
        # is transactionally adopted; after adoption this exception can never apply.
        if baseline is None and module_name == CANONICAL_LAUNCH_NAME:
            continue
        _require(
            isinstance(baseline, _OwnedModuleBaseline),
            f"refusing unowned preloaded accepted module:{module_name}",
        )
        _require(
            all(module is baseline.module for module in candidates),
            f"adapter-owned module object changed or split:{module_name}",
        )
        if module_name == CANONICAL_LAUNCH_NAME:
            expected_path = CANONICAL_LAUNCH_PATH
            expected_sha256 = CANONICAL_LAUNCH_SHA256
        elif module_name == CANONICAL_TRAINER_NAME:
            expected_path = CANONICAL_TRAINER_PATH
            expected_sha256 = CANONICAL_TRAINER_SHA256
        else:
            relative_path, expected_sha256 = _EXPECTED_LOCAL_MODULES[module_name]
            expected_path = (ACCEPTED_SUCCESSOR_ROOT / relative_path).resolve()
        _validate_module_object(
            baseline.module,
            canonical_name=module_name,
            expected_path=expected_path,
            expected_sha256=expected_sha256,
            label=f"adapter-owned module {module_name}",
        )


def _adopt_transaction_module(
    module_name: str,
    module: ModuleType,
    *,
    snapshot: _ImportSnapshot,
    attempted: set[int],
    pending: dict[str, _OwnedModuleBaseline],
) -> None:
    baseline = pending.get(module_name)
    if baseline is not None:
        _require(
            baseline.module is module,
            f"refusing to replace adapter-owned module baseline:{module_name}",
        )
        _validate_owned_module_baseline(module_name, baseline)
        return
    _require(
        snapshot.sys_modules.get(module_name, _MISSING) is _MISSING,
        f"refusing to adopt module not created by adapter transaction:{module_name}",
    )
    _require(
        id(module) in attempted,
        f"refusing module loaded outside adapter transaction:{module_name}",
    )
    _require(
        sys.modules.get(module_name) is module,
        f"adapter transaction lost canonical module binding:{module_name}",
    )
    pending[module_name] = _capture_owned_module_baseline(module)


def _adopt_validated_launch_module(
    launch: ModuleType,
    *,
    parent: ModuleType,
    pending: dict[str, _OwnedModuleBaseline],
) -> None:
    """Adopt the sole permitted clean preload after its full runtime-manifest check."""

    baseline = pending.get(CANONICAL_LAUNCH_NAME)
    if baseline is not None:
        _require(
            baseline.module is launch,
            "refusing to replace adapter-owned launch-contract baseline",
        )
        _validate_owned_module_baseline(CANONICAL_LAUNCH_NAME, baseline)
        return
    _require(
        sys.modules.get(CANONICAL_LAUNCH_NAME) is launch,
        "launch adoption lost canonical module binding",
    )
    _require(
        sys.modules.get(BARE_LAUNCH_NAME) is launch,
        "launch adoption lost bare module binding",
    )
    _require(
        parent.__dict__.get("production_launch_contract_v1") is launch,
        "launch adoption lost parent module binding",
    )
    pending[CANONICAL_LAUNCH_NAME] = _capture_owned_module_baseline(launch)


def _validate_existing_topology(
    inventory: Mapping[Path, Sequence[ModuleType]] | None = None,
) -> None:
    if inventory is None:
        inventory = _loaded_module_path_inventory()
    _validate_owned_module_provenance(_OWNED_MODULE_BASELINES, inventory)
    parent = sys.modules.get("pretrain", _MISSING)
    if parent is not _MISSING:
        _validate_pretrain_package(parent, label="sys.modules['pretrain']")

    targets = (
        (
            CANONICAL_LAUNCH_NAME,
            BARE_LAUNCH_NAME,
            CANONICAL_LAUNCH_PATH,
            CANONICAL_LAUNCH_SHA256,
        ),
        (
            CANONICAL_TRAINER_NAME,
            BARE_TRAINER_NAME,
            CANONICAL_TRAINER_PATH,
            CANONICAL_TRAINER_SHA256,
        ),
    )
    for canonical_name, bare_name, path, sha256 in targets:
        candidates: list[ModuleType] = []
        for name in (canonical_name, bare_name):
            if name not in sys.modules:
                continue
            candidates.append(
                _validate_module_object(
                    sys.modules[name],
                    canonical_name=canonical_name,
                    expected_path=path,
                    expected_sha256=sha256,
                    label=f"sys.modules[{name!r}]",
                )
            )
        if isinstance(parent, ModuleType):
            child_name = canonical_name.rpartition(".")[2]
            parent_value = parent.__dict__.get(child_name, _MISSING)
            if parent_value is not _MISSING:
                candidates.append(
                    _validate_module_object(
                        parent_value,
                        canonical_name=canonical_name,
                        expected_path=path,
                        expected_sha256=sha256,
                        label=f"pretrain.{child_name}",
                    )
                )
        exact_path = _exact_path_module_objects(path, inventory)
        for module in exact_path:
            candidates.append(
                _validate_module_object(
                    module,
                    canonical_name=canonical_name,
                    expected_path=path,
                    expected_sha256=sha256,
                    label=f"loaded exact-path object for {canonical_name}",
                )
            )
        canonical = sys.modules.get(canonical_name)
        if candidates:
            _require(
                canonical is not None,
                f"reviewed source is loaded without canonical binding:{path}",
            )
            _require(
                all(candidate is canonical for candidate in candidates),
                f"multiple module objects already define reviewed source:{path}",
            )

    for module_name, (relative_path, sha256) in _EXPECTED_LOCAL_MODULES.items():
        path = (ACCEPTED_SUCCESSOR_ROOT / relative_path).resolve()
        canonical_dependency: ModuleType | None = None
        if module_name in sys.modules:
            canonical_dependency = _validate_module_object(
                sys.modules[module_name],
                canonical_name=module_name,
                expected_path=path,
                expected_sha256=sha256,
                label=f"preloaded dependency {module_name}",
            )
        parent_name, separator, child_name = module_name.rpartition(".")
        dependency_parent = sys.modules.get(parent_name) if separator else None
        if isinstance(dependency_parent, ModuleType) and child_name in dependency_parent.__dict__:
            parent_dependency = _validate_module_object(
                dependency_parent.__dict__[child_name],
                canonical_name=module_name,
                expected_path=path,
                expected_sha256=sha256,
                label=f"preloaded parent dependency {module_name}",
            )
            _require(
                canonical_dependency is not None and parent_dependency is canonical_dependency,
                f"dependency parent attribute differs from canonical binding:{module_name}",
            )
        exact_objects = _exact_path_module_objects(path, inventory)
        if exact_objects:
            canonical = sys.modules.get(module_name)
            _require(
                canonical is not None,
                f"dependency source loaded without canonical binding:{path}",
            )
            _require(
                all(item is canonical for item in exact_objects),
                f"dependency source has multiple module objects:{path}",
            )

    retained_launch = sys.modules.get(CANONICAL_LAUNCH_NAME)
    if isinstance(retained_launch, ModuleType):
        _validate_launch_family(retained_launch)
    retained_trainer = sys.modules.get(CANONICAL_TRAINER_NAME)
    if isinstance(retained_trainer, ModuleType):
        _validate_trainer_runtime_integrity(retained_trainer)
    _validate_retained_bridge_launch()


def _load_or_reuse_reviewed_module(
    *,
    canonical_name: str,
    aliases: Sequence[str],
    expected_path: Path,
    expected_sha256: str,
    parent: ModuleType | None,
    attempted: set[int],
    prevalidated_inventory: Mapping[Path, Sequence[ModuleType]],
) -> ModuleType:
    source = expected_path.read_bytes()
    _require(_sha256_bytes(source) == expected_sha256, f"reviewed bytes changed:{expected_path}")
    existing = sys.modules.get(canonical_name)
    if existing is not None:
        module = _validate_module_object(
            existing,
            canonical_name=canonical_name,
            expected_path=expected_path,
            expected_sha256=expected_sha256,
            label=f"canonical binding {canonical_name}",
        )
    else:
        _require(
            not _exact_path_module_objects(expected_path, prevalidated_inventory),
            f"reviewed source already exists without canonical binding:{expected_path}",
        )
        spec = importlib.util.spec_from_file_location(canonical_name, expected_path)
        _require(
            spec is not None and spec.loader is not None,
            f"cannot create reviewed canonical module:{canonical_name}",
        )
        module = importlib.util.module_from_spec(spec)
        attempted.add(id(module))
        for name in (canonical_name, *aliases):
            sys.modules[name] = module
        if parent is not None:
            parent.__dict__[canonical_name.rpartition(".")[2]] = module
        spec._initializing = True
        try:
            exec(
                compile(source, str(expected_path), "exec", dont_inherit=True),
                module.__dict__,
            )
        finally:
            spec._initializing = False

    for name in (canonical_name, *aliases):
        current = sys.modules.get(name, _MISSING)
        _require(
            current is _MISSING or current is module,
            f"refusing to overwrite existing module binding:{name}",
        )
        sys.modules[name] = module
    if parent is not None:
        parent.__dict__[canonical_name.rpartition(".")[2]] = module
    module = _validate_module_object(
        module,
        canonical_name=canonical_name,
        expected_path=expected_path,
        expected_sha256=expected_sha256,
        label=f"installed module {canonical_name}",
    )
    _require(
        expected_path.read_bytes() == source,
        f"reviewed source changed during initialization:{expected_path}",
    )
    # The caller holds the global import lock.  One consolidated, fresh inventory in final
    # topology validation proves that execution did not create a split exact-path family.
    return module


def _trainer_launch_imported_symbols() -> tuple[str, ...]:
    tree = ast.parse(
        CANONICAL_TRAINER_PATH.read_text(encoding="utf-8"),
        filename=str(CANONICAL_TRAINER_PATH),
    )
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == BARE_LAUNCH_NAME
        for alias in node.names
    }
    return tuple(sorted(names))


def _trainer_top_level_function_contracts() -> dict[str, tuple[CodeType, int]]:
    source = CANONICAL_TRAINER_PATH.read_bytes()
    _require(
        _sha256_bytes(source) == CANONICAL_TRAINER_SHA256,
        "accepted trainer bytes changed while deriving function contracts",
    )
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(CANONICAL_TRAINER_PATH))
    module_code = compile(
        source,
        str(CANONICAL_TRAINER_PATH),
        "exec",
        dont_inherit=True,
    )
    code_by_name = {
        item.co_name: item for item in module_code.co_consts if isinstance(item, CodeType)
    }
    contracts: dict[str, tuple[CodeType, int]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        expected_code = code_by_name.get(node.name)
        _require(expected_code is not None, f"trainer function code missing:{node.name}")
        contracts[node.name] = (expected_code, len(node.decorator_list))
    _require(
        {"parse_args", "validate_training_args", "enforce_governed_launch", "main"}
        <= set(contracts),
        "critical accepted trainer function contract is incomplete",
    )
    return contracts


def _validate_trainer_runtime_integrity(trainer: ModuleType) -> None:
    """Reject mutable retained trainer surfaces without executing its source again."""

    _require(
        type(trainer.__dict__.get("PROJECT_ROOT")) is type(ACCEPTED_SUCCESSOR_ROOT)
        and trainer.__dict__["PROJECT_ROOT"] == ACCEPTED_SUCCESSOR_ROOT,
        "accepted trainer PROJECT_ROOT changed",
    )
    for name, (expected_code, decorator_count) in _trainer_top_level_function_contracts().items():
        bound = trainer.__dict__.get(name)
        _require(isinstance(bound, FunctionType), f"accepted trainer function changed:{name}")
        _require(bound.__name__ == name, f"accepted trainer function name changed:{name}")
        _require(bound.__qualname__ == name, f"accepted trainer function qualname changed:{name}")
        _require(
            bound.__module__ == CANONICAL_TRAINER_NAME,
            f"accepted trainer function module changed:{name}",
        )
        underlying = bound
        unwrap_count = 0
        seen = {id(underlying)}
        while hasattr(underlying, "__wrapped__"):
            candidate = underlying.__wrapped__
            _require(
                isinstance(candidate, FunctionType) and id(candidate) not in seen,
                f"accepted trainer decorator chain changed:{name}",
            )
            underlying = candidate
            seen.add(id(underlying))
            unwrap_count += 1
        _require(
            unwrap_count == decorator_count,
            f"accepted trainer decorator count changed:{name}",
        )
        _require(
            underlying.__globals__ is trainer.__dict__,
            f"accepted trainer function globals changed:{name}",
        )
        _require(
            underlying.__name__ == name
            and underlying.__qualname__ == name
            and underlying.__module__ == CANONICAL_TRAINER_NAME,
            f"accepted trainer underlying function metadata changed:{name}",
        )
        _require(
            underlying.__code__ == expected_code,
            f"accepted trainer function code changed:{name}",
        )

    tree = ast.parse(
        CANONICAL_TRAINER_PATH.read_text(encoding="utf-8"),
        filename=str(CANONICAL_TRAINER_PATH),
    )
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        if node.module not in _EXPECTED_LOCAL_MODULES:
            continue
        dependency = sys.modules.get(node.module)
        _require(
            isinstance(dependency, ModuleType),
            f"accepted trainer dependency binding missing:{node.module}",
        )
        for alias in node.names:
            _require(alias.name != "*", "accepted trainer uses an unreviewable project star import")
            bound_name = alias.asname or alias.name
            _require(
                trainer.__dict__.get(bound_name, _MISSING)
                is getattr(dependency, alias.name, _MISSING),
                f"accepted trainer imported project symbol changed:{bound_name}",
            )


def _validated_launch_import_bindings(launch: ModuleType) -> dict[str, dict[str, str]]:
    """Bind top-level imports to already-loaded canonical module attributes."""

    source = CANONICAL_LAUNCH_PATH.read_bytes()
    _require(
        _sha256_bytes(source) == CANONICAL_LAUNCH_SHA256,
        "accepted launch-contract bytes changed while deriving import bindings",
    )
    tree = ast.parse(source.decode("utf-8"), filename=str(CANONICAL_LAUNCH_PATH))
    bindings: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_module = sys.modules.get(alias.name)
                _require(
                    isinstance(imported_module, ModuleType),
                    f"accepted launch imported module missing:{alias.name}",
                )
                if alias.asname:
                    bound_name = alias.asname
                    expected = imported_module
                else:
                    bound_name = alias.name.split(".", 1)[0]
                    expected = sys.modules.get(bound_name)
                _require(
                    launch.__dict__.get(bound_name, _MISSING) is expected,
                    f"accepted launch imported binding changed:{bound_name}",
                )
                bindings[bound_name] = {
                    "kind": "import_module",
                    "source_module": alias.name,
                }
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            _require(
                node.level == 0 and bool(node.module),
                "accepted launch has an unreviewable relative top-level import",
            )
            imported_module = sys.modules.get(node.module)
            _require(
                isinstance(imported_module, ModuleType),
                f"accepted launch import source missing:{node.module}",
            )
            for alias in node.names:
                _require(alias.name != "*", "accepted launch has a top-level star import")
                bound_name = alias.asname or alias.name
                expected = getattr(imported_module, alias.name, _MISSING)
                _require(
                    expected is not _MISSING
                    and launch.__dict__.get(bound_name, _MISSING) is expected,
                    f"accepted launch imported binding changed:{bound_name}",
                )
                bindings[bound_name] = {
                    "kind": "import_attribute",
                    "source_module": node.module,
                    "source_name": alias.name,
                }
    return bindings


def _launch_runtime_value_manifest(
    value: object,
    *,
    launch: ModuleType,
    active: set[int],
) -> Any:
    """Return an id-independent JSON value for a clean launch runtime object."""

    if value is None or isinstance(value, (bool, int, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real_hex": value.real.hex(),
            "imag_hex": value.imag.hex(),
        }

    identity = id(value)
    _require(identity not in active, "accepted launch runtime namespace contains a cycle")
    active.add(identity)
    try:
        recurse = lambda item: _launch_runtime_value_manifest(  # noqa: E731
            item,
            launch=launch,
            active=active,
        )
        if isinstance(value, FunctionType) and value.__globals__ is launch.__dict__:
            closure: list[Any] = []
            for cell in value.__closure__ or ():
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    closure.append({"kind": "empty_cell"})
                else:
                    closure.append(recurse(cell_value))
            return {
                "kind": "owned_function",
                "name": value.__name__,
                "qualname": value.__qualname__,
                "module": value.__module__,
                "doc": value.__doc__,
                "defaults": recurse(value.__defaults__),
                "kwdefaults": recurse(value.__kwdefaults__),
                "annotations": recurse(value.__annotations__),
                "attributes": recurse(value.__dict__),
                "closure": closure,
            }
        if isinstance(value, type) and value.__module__ == CANONICAL_LAUNCH_NAME:
            members = {name: recurse(member) for name, member in sorted(value.__dict__.items())}
            return {
                "kind": "owned_class",
                "name": value.__name__,
                "qualname": value.__qualname__,
                "module": value.__module__,
                "bases": [
                    {
                        "module": base.__module__,
                        "qualname": base.__qualname__,
                    }
                    for base in value.__bases__
                ],
                "members": members,
            }
        if isinstance(value, property):
            return {
                "kind": "property",
                "fget": recurse(value.fget),
                "fset": recurse(value.fset),
                "fdel": recurse(value.fdel),
                "doc": value.__doc__,
            }
        if isinstance(value, (staticmethod, classmethod)):
            return {"kind": type(value).__name__, "function": recurse(value.__func__)}
        if isinstance(value, MappingProxyType):
            entries = [[recurse(key), recurse(item)] for key, item in value.items()]
            entries.sort(key=lambda item: canonical_json_bytes(item[0]))
            return {"kind": "mappingproxy", "entries": entries}
        if isinstance(value, dict):
            entries = [[recurse(key), recurse(item)] for key, item in value.items()]
            entries.sort(key=lambda item: canonical_json_bytes(item[0]))
            return {"kind": "dict", "entries": entries}
        if isinstance(value, (list, tuple)):
            return {
                "kind": f"{type(value).__module__}.{type(value).__qualname__}",
                "items": [recurse(item) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            items = [recurse(item) for item in value]
            items.sort(key=canonical_json_bytes)
            return {"kind": type(value).__name__, "items": items}
        if isinstance(value, bytearray):
            return {"kind": "bytearray", "hex": bytes(value).hex()}
        value_type = type(value)
        if value_type.__module__ == CANONICAL_LAUNCH_NAME:
            instance_dict = getattr(value, "__dict__", None)
            return {
                "kind": "owned_instance",
                "class": value_type.__qualname__,
                "attributes": recurse(instance_dict),
            }
        if value_type.__module__ == "__future__" and value_type.__qualname__ == "_Feature":
            return {
                "kind": "future_feature",
                "attributes": recurse(value.__dict__),
            }
        descriptor_owner = getattr(value, "__objclass__", None)
        descriptor_name = getattr(value, "__name__", None)
        if isinstance(descriptor_owner, type) and isinstance(descriptor_name, str):
            return {
                "kind": "descriptor",
                "type": f"{value_type.__module__}.{value_type.__qualname__}",
                "owner_module": descriptor_owner.__module__,
                "owner_qualname": descriptor_owner.__qualname__,
                "name": descriptor_name,
            }
        raise AdapterError(
            "accepted launch runtime value cannot be canonicalized:"
            f"{value_type.__module__}.{value_type.__qualname__}"
        )
    finally:
        active.remove(identity)


def _launch_runtime_namespace_manifest(launch: ModuleType) -> dict[str, Any]:
    _require(
        tuple(sys.version_info[:3]) == LAUNCH_RUNTIME_PYTHON_VERSION,
        "accepted launch runtime manifest Python version mismatch",
    )
    builtins_module = sys.modules.get("builtins")
    _require(isinstance(builtins_module, ModuleType), "canonical builtins module is missing")
    _require(
        launch.__dict__.get("__builtins__") is builtins_module.__dict__,
        "accepted launch-contract __builtins__ binding changed",
    )
    imported = _validated_launch_import_bindings(launch)
    names = sorted(set(launch.__dict__) - _MODULE_RUNTIME_METADATA)
    namespace: dict[str, Any] = {}
    for name in names:
        if name in imported:
            namespace[name] = imported[name]
        else:
            namespace[name] = _launch_runtime_value_manifest(
                launch.__dict__[name], launch=launch, active=set()
            )
    return {
        "schema_version": LAUNCH_RUNTIME_NAMESPACE_SCHEMA,
        "python_version": list(LAUNCH_RUNTIME_PYTHON_VERSION),
        "python_cache_tag": sys.implementation.cache_tag,
        "launch_source_sha256": CANONICAL_LAUNCH_SHA256,
        "namespace": namespace,
    }


def _validate_launch_runtime_namespace(launch: ModuleType) -> None:
    try:
        manifest = _launch_runtime_namespace_manifest(launch)
    except AdapterError as exc:
        if str(exc).startswith((
            "accepted launch runtime value cannot be canonicalized:",
            "accepted launch runtime namespace contains a cycle",
        )):
            raise AdapterError("accepted launch-contract runtime namespace changed") from exc
        raise
    manifest_sha256 = _sha256_bytes(canonical_json_bytes(manifest))
    _require(
        manifest_sha256 == LAUNCH_RUNTIME_NAMESPACE_SHA256,
        "accepted launch-contract runtime namespace changed",
    )


def _validate_launch_family(launch: ModuleType) -> tuple[str, ...]:
    source = CANONICAL_LAUNCH_PATH.read_bytes()
    _require(
        _sha256_bytes(source) == CANONICAL_LAUNCH_SHA256,
        "accepted launch-contract bytes changed while deriving function contracts",
    )
    tree = ast.parse(source.decode("utf-8"), filename=str(CANONICAL_LAUNCH_PATH))
    module_code = compile(
        source,
        str(CANONICAL_LAUNCH_PATH),
        "exec",
        dont_inherit=True,
    )
    code_by_name = {
        item.co_name: item for item in module_code.co_consts if isinstance(item, CodeType)
    }
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        expected_code = code_by_name.get(node.name)
        function = launch.__dict__.get(node.name)
        _require(
            expected_code is not None and isinstance(function, FunctionType),
            f"accepted launch-contract function changed:{node.name}",
        )
        _require(
            not node.decorator_list and not hasattr(function, "__wrapped__"),
            f"accepted launch-contract function decorator changed:{node.name}",
        )
        _require(
            function.__name__ == node.name
            and function.__qualname__ == node.name
            and function.__module__ == CANONICAL_LAUNCH_NAME
            and function.__globals__ is launch.__dict__
            and function.__code__ == expected_code,
            f"accepted launch-contract function code or metadata changed:{node.name}",
        )
    class_nodes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    expected_bases = {
        "LaunchContractError": (RuntimeError,),
        "_Missing": (object,),
        "ObservedForward": (object,),
    }
    for name in _LAUNCH_FAMILY_TYPES:
        class_object = getattr(launch, name, None)
        _require(isinstance(class_object, type), f"launch class missing:{name}")
        _require(class_object.__name__ == name, f"launch class name changed:{name}")
        _require(class_object.__qualname__ == name, f"launch class qualname changed:{name}")
        _require(
            class_object.__module__ == CANONICAL_LAUNCH_NAME,
            f"launch class has split module origin:{name}",
        )
        _require(
            class_object.__bases__ == expected_bases[name], f"launch class bases changed:{name}"
        )
        _require(
            sys.modules.get(class_object.__module__) is launch,
            f"launch class does not resolve to canonical module:{name}",
        )
        class_node = class_nodes.get(name)
        class_code = code_by_name.get(name)
        _require(
            class_node is not None and class_code is not None,
            f"launch class source contract missing:{name}",
        )
        expected_method_codes = {
            item.co_name: item for item in class_code.co_consts if isinstance(item, CodeType)
        }
        for method_node in class_node.body:
            if not isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            descriptor = class_object.__dict__.get(method_node.name, _MISSING)
            if (
                len(method_node.decorator_list) == 1
                and isinstance(method_node.decorator_list[0], ast.Name)
                and method_node.decorator_list[0].id == "property"
            ):
                _require(
                    isinstance(descriptor, property)
                    and descriptor.fset is None
                    and descriptor.fdel is None,
                    f"launch class property descriptor changed:{name}.{method_node.name}",
                )
                method = descriptor.fget
            else:
                _require(
                    not method_node.decorator_list,
                    f"launch class method has unsupported decorator:{name}.{method_node.name}",
                )
                method = descriptor
            expected_method = expected_method_codes.get(method_node.name)
            _require(
                isinstance(method, FunctionType)
                and expected_method is not None
                and method.__name__ == method_node.name
                and method.__qualname__ == f"{name}.{method_node.name}"
                and method.__module__ == CANONICAL_LAUNCH_NAME
                and method.__globals__ is launch.__dict__
                and method.__code__ == expected_method,
                f"launch class method changed:{name}.{method_node.name}",
            )
        slots_node = next(
            (
                item
                for item in class_node.body
                if isinstance(item, (ast.Assign, ast.AnnAssign))
                and (
                    any(
                        isinstance(target, ast.Name) and target.id == "__slots__"
                        for target in (
                            item.targets if isinstance(item, ast.Assign) else [item.target]
                        )
                    )
                )
            ),
            None,
        )
        if slots_node is not None:
            slots_value = slots_node.value
            expected_slots = ast.literal_eval(slots_value)
            _require(
                class_object.__dict__.get("__slots__") == expected_slots,
                f"launch class slots changed:{name}",
            )
            for slot in expected_slots:
                _require(
                    isinstance(class_object.__dict__.get(slot), MemberDescriptorType),
                    f"launch class slot descriptor changed:{name}.{slot}",
                )
    _require(
        isinstance(getattr(launch, "_MISSING", None), launch._Missing),
        "launch _MISSING sentinel has a split class family",
    )
    imported_symbols = _trainer_launch_imported_symbols()
    for name in imported_symbols:
        _require(hasattr(launch, name), f"trainer launch import is missing:{name}")
    _validate_launch_runtime_namespace(launch)
    return imported_symbols


def _validate_retained_bridge_launch(launch: ModuleType | None = None) -> None:
    bridge_name = "pretrain.stage_n_successor_head_compatibility_bridge_v1"
    bridge = sys.modules.get(bridge_name)
    if bridge is None:
        return
    relative_path, sha256 = _EXPECTED_LOCAL_MODULES[bridge_name]
    bridge = _validate_module_object(
        bridge,
        canonical_name=bridge_name,
        expected_path=(ACCEPTED_SUCCESSOR_ROOT / relative_path).resolve(),
        expected_sha256=sha256,
        label="retained Stage-N successor bridge",
    )
    canonical_launch = sys.modules.get(CANONICAL_LAUNCH_NAME)
    _require(
        isinstance(canonical_launch, ModuleType),
        "retained Stage-N successor bridge has no canonical launch binding",
    )
    if launch is not None:
        _require(
            canonical_launch is launch,
            "retained Stage-N successor bridge canonical launch binding changed",
        )
    canonical_launch = _validate_module_object(
        canonical_launch,
        canonical_name=CANONICAL_LAUNCH_NAME,
        expected_path=CANONICAL_LAUNCH_PATH,
        expected_sha256=CANONICAL_LAUNCH_SHA256,
        label="retained Stage-N successor bridge launch",
    )
    _require(
        getattr(bridge, "launch", _MISSING) is canonical_launch,
        "retained Stage-N successor bridge holds a stale launch object",
    )
    _validate_launch_family(canonical_launch)


def _install_reviewed_search_path() -> None:
    accepted_root = str(ACCEPTED_SUCCESSOR_ROOT.resolve())
    accepted_pretrain = str((ACCEPTED_SUCCESSOR_ROOT / "pretrain").resolve())
    adapter_root = _adapter_root().resolve()
    adapter_tools = (adapter_root / "tools").resolve()
    retained: list[str] = []
    for item in sys.path:
        try:
            candidate = Path(item or os.getcwd()).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if str(candidate) in {accepted_root, accepted_pretrain}:
            continue
        if candidate in {adapter_root, adapter_tools}:
            continue
        # Never leave the historical worktree, adapter clone, or another repository copy as
        # an import fallback.  The accepted root/pretrain entries below are the only project
        # search authorities; stdlib and site-packages entries are retained unchanged.
        if (candidate / "pretrain/production_launch_contract_v1.py").is_file():
            continue
        if (candidate / "production_launch_contract_v1.py").is_file():
            continue
        retained.append(item)
    sys.path[:] = [accepted_root, accepted_pretrain, *retained]


def _validate_loaded_trainer_dependencies(
    inventory: Mapping[Path, Sequence[ModuleType]],
) -> None:
    required = set(_EXPECTED_LOCAL_MODULES) - {
        "pretrain.stage_n_successor_head_compatibility_bridge_v1"
    }
    for module_name in sorted(required):
        relative_path, sha256 = _EXPECTED_LOCAL_MODULES[module_name]
        _require(module_name in sys.modules, f"trainer dependency not loaded:{module_name}")
        expected_path = (ACCEPTED_SUCCESSOR_ROOT / relative_path).resolve()
        module = _validate_module_object(
            sys.modules[module_name],
            canonical_name=module_name,
            expected_path=expected_path,
            expected_sha256=sha256,
            label=f"trainer dependency {module_name}",
        )
        _require(
            {id(item) for item in _exact_path_module_objects(expected_path, inventory)}
            == {id(module)},
            f"trainer dependency has a split module family:{module_name}",
        )


def _validate_topology(
    topology: AcceptedModuleTopology,
    owned_baselines: Mapping[str, _OwnedModuleBaseline] | None = None,
) -> None:
    inventory = _loaded_module_path_inventory()
    _validate_owned_module_provenance(
        _OWNED_MODULE_BASELINES if owned_baselines is None else owned_baselines,
        inventory,
    )
    parent = _validate_pretrain_package(
        topology.pretrain_package, label="installed pretrain package"
    )
    launch = _validate_module_object(
        topology.launch_contract,
        canonical_name=CANONICAL_LAUNCH_NAME,
        expected_path=CANONICAL_LAUNCH_PATH,
        expected_sha256=CANONICAL_LAUNCH_SHA256,
        label="installed launch contract",
    )
    _require(sys.modules.get(CANONICAL_LAUNCH_NAME) is launch, "canonical launch binding changed")
    _require(sys.modules.get(BARE_LAUNCH_NAME) is launch, "bare launch binding changed")
    _require(
        parent.__dict__.get("production_launch_contract_v1") is launch,
        "parent launch binding changed",
    )
    _require(
        {id(item) for item in _exact_path_module_objects(CANONICAL_LAUNCH_PATH, inventory)}
        == {id(launch)},
        "launch contract has a split exact-path module family",
    )
    imported_symbols = _validate_launch_family(launch)
    _require(
        topology.launch_imported_symbols == imported_symbols,
        "trainer launch-import symbol inventory changed",
    )
    _validate_retained_bridge_launch(launch)
    if topology.trainer is not None:
        trainer = _validate_module_object(
            topology.trainer,
            canonical_name=CANONICAL_TRAINER_NAME,
            expected_path=CANONICAL_TRAINER_PATH,
            expected_sha256=CANONICAL_TRAINER_SHA256,
            label="installed trainer",
        )
        _require(
            sys.modules.get(CANONICAL_TRAINER_NAME) is trainer,
            "canonical trainer binding changed",
        )
        _require(sys.modules.get(BARE_TRAINER_NAME) is trainer, "bare trainer binding changed")
        _require(
            parent.__dict__.get("train_pretrain_with_bench") is trainer,
            "parent trainer binding changed",
        )
        _require(
            {id(item) for item in _exact_path_module_objects(CANONICAL_TRAINER_PATH, inventory)}
            == {id(trainer)},
            "trainer has a split exact-path module family",
        )
        _validate_loaded_trainer_dependencies(inventory)
        _validate_trainer_runtime_integrity(trainer)


def _install_topology(
    *,
    load_trainer: bool,
    before_trainer: Callable[[ModuleType], None] | None = None,
) -> AcceptedModuleTopology:
    global _OWNED_MODULE_BASELINES

    lock_names = (
        "pretrain",
        CANONICAL_LAUNCH_NAME,
        BARE_LAUNCH_NAME,
        CANONICAL_TRAINER_NAME,
        BARE_TRAINER_NAME,
    )
    with _SETUP_LOCK, _held_import_locks(lock_names):
        snapshot = _snapshot_import_state()
        attempted: set[int] = set()
        pending_baselines = dict(_OWNED_MODULE_BASELINES)
        try:
            initial_inventory = _loaded_module_path_inventory()
            _validate_existing_topology(initial_inventory)
            accepted_trainer_identity()
            parent_value = sys.modules.get("pretrain")
            if parent_value is None:
                parent = _new_fixed_pretrain_package()
                attempted.add(id(parent))
                sys.modules["pretrain"] = parent
            else:
                parent = _validate_pretrain_package(parent_value, label="existing pretrain package")
            _install_reviewed_search_path()
            launch = _load_or_reuse_reviewed_module(
                canonical_name=CANONICAL_LAUNCH_NAME,
                aliases=(BARE_LAUNCH_NAME,),
                expected_path=CANONICAL_LAUNCH_PATH,
                expected_sha256=CANONICAL_LAUNCH_SHA256,
                parent=parent,
                attempted=attempted,
                prevalidated_inventory=initial_inventory,
            )
            imported_symbols = _validate_launch_family(launch)
            _adopt_validated_launch_module(
                launch,
                parent=parent,
                pending=pending_baselines,
            )

            bridge_name = "pretrain.stage_n_successor_head_compatibility_bridge_v1"
            bridge_relative_path, bridge_sha256 = _EXPECTED_LOCAL_MODULES[bridge_name]
            bridge = _load_or_reuse_reviewed_module(
                canonical_name=bridge_name,
                aliases=(),
                expected_path=(ACCEPTED_SUCCESSOR_ROOT / bridge_relative_path).resolve(),
                expected_sha256=bridge_sha256,
                parent=parent,
                attempted=attempted,
                prevalidated_inventory=initial_inventory,
            )
            _adopt_transaction_module(
                bridge_name,
                bridge,
                snapshot=snapshot,
                attempted=attempted,
                pending=pending_baselines,
            )
            _validate_retained_bridge_launch(launch)

            if before_trainer is not None:
                before_trainer(launch)
                _validate_owned_module_provenance(
                    pending_baselines,
                    _loaded_module_path_inventory(),
                )
                _validate_launch_family(launch)
            trainer: ModuleType | None = None
            if load_trainer:
                loaded_dependencies: dict[str, ModuleType] = {}
                for module_name in _CONTROLLED_TRAINER_DEPENDENCY_ORDER:
                    relative_path, sha256 = _EXPECTED_LOCAL_MODULES[module_name]
                    dependency_parent: ModuleType | None
                    if module_name.startswith("pretrain."):
                        dependency_parent = parent
                    elif module_name.startswith("src."):
                        src_parent = sys.modules.get("src")
                        _require(
                            isinstance(src_parent, ModuleType),
                            f"controlled dependency parent missing:{module_name}",
                        )
                        dependency_parent = src_parent
                    else:
                        dependency_parent = None
                    dependency = _load_or_reuse_reviewed_module(
                        canonical_name=module_name,
                        aliases=(),
                        expected_path=(ACCEPTED_SUCCESSOR_ROOT / relative_path).resolve(),
                        expected_sha256=sha256,
                        parent=dependency_parent,
                        attempted=attempted,
                        prevalidated_inventory=initial_inventory,
                    )
                    loaded_dependencies[module_name] = dependency
                trainer = _load_or_reuse_reviewed_module(
                    canonical_name=CANONICAL_TRAINER_NAME,
                    aliases=(BARE_TRAINER_NAME,),
                    expected_path=CANONICAL_TRAINER_PATH,
                    expected_sha256=CANONICAL_TRAINER_SHA256,
                    parent=parent,
                    attempted=attempted,
                    prevalidated_inventory=initial_inventory,
                )
                for module_name, dependency in loaded_dependencies.items():
                    _adopt_transaction_module(
                        module_name,
                        dependency,
                        snapshot=snapshot,
                        attempted=attempted,
                        pending=pending_baselines,
                    )
                _adopt_transaction_module(
                    CANONICAL_TRAINER_NAME,
                    trainer,
                    snapshot=snapshot,
                    attempted=attempted,
                    pending=pending_baselines,
                )
            topology = AcceptedModuleTopology(parent, launch, trainer, imported_symbols)
            _validate_topology(topology, pending_baselines)
            _OWNED_MODULE_BASELINES = pending_baselines
            return topology
        except BaseException:
            _rollback_import_state(snapshot, attempted)
            raise


def install_accepted_launch_topology() -> ModuleType:
    """Install and return the one canonical/bare launch-contract module object."""

    return _install_topology(load_trainer=False).launch_contract


def install_accepted_module_topology(
    *, before_trainer: Callable[[ModuleType], None] | None = None
) -> AcceptedModuleTopology:
    return _install_topology(load_trainer=True, before_trainer=before_trainer)


def load_accepted_trainer(
    topology: AcceptedModuleTopology | None = None,
) -> ModuleType:
    if topology is not None:
        _validate_topology(topology)
    installed = install_accepted_module_topology()
    _require(installed.trainer is not None, "accepted trainer was not installed")
    return installed.trainer


def _load_json_snapshot(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], bytes, str, str]:
    resolved = _resolved(path)
    _require(resolved.is_file(), f"JSON artifact not found:{resolved}")
    body = resolved.read_bytes()
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"JSON artifact is malformed:{resolved}") from exc
    _require(isinstance(document, dict), f"JSON artifact is not an object:{resolved}")
    return document, body, _sha256_bytes(body), str(resolved)


def _canonical_authority_path(path: str | os.PathLike[str], *, label: str) -> str:
    try:
        supplied = Path(os.fspath(path))
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{label} path is invalid") from exc
    _require(supplied.is_absolute(), f"{label} path must be absolute")
    resolved = _resolved(supplied)
    _require(supplied == resolved, f"{label} path must be canonical and non-symlinked")
    _require(resolved.is_file(), f"{label} artifact not found:{resolved}")
    return str(resolved)


def _require_unchanged_json_snapshot(
    path: str,
    *,
    expected_document: Mapping[str, Any],
    expected_sha256: str,
    expected_path: str,
    label: str,
) -> None:
    document, _body, sha256, snapshot_path = _load_json_snapshot(path)
    _require(snapshot_path == expected_path, f"{label} resolved path changed")
    _require(sha256 == expected_sha256, f"{label} bytes changed")
    _require(_strict_equal(document, expected_document), f"{label} document changed")


def parse_trainer_args(trainer: ModuleType, trainer_argv: Sequence[str]) -> argparse.Namespace:
    """Invoke the accepted real parser with flags-only argv and no semantic rewrite."""

    _require(
        all(isinstance(item, str) for item in trainer_argv),
        "trainer argv must contain strings only",
    )
    previous = list(sys.argv)
    sys.argv[:] = [str(CANONICAL_TRAINER_PATH), *trainer_argv]
    try:
        parsed = trainer.parse_args()
    finally:
        sys.argv[:] = previous
    _require(isinstance(parsed, argparse.Namespace), "accepted parser returned a non-Namespace")
    return parsed


def _one_exact_option(trainer_argv: Sequence[str], option: str) -> str:
    values: list[str] = []
    for index, token in enumerate(trainer_argv):
        if token == option:
            _require(
                index + 1 < len(trainer_argv), f"governed trainer option missing value:{option}"
            )
            value = trainer_argv[index + 1]
            _require(
                value and not value.startswith("-"),
                f"governed trainer option has an invalid value:{option}",
            )
            values.append(value)
        elif token.startswith(option + "="):
            values.append(token[len(option) + 1 :])
    _require(values, f"governed trainer option is required:{option}")
    _require(len(values) == 1, f"governed trainer option is ambiguous:{option}")
    _require(values[0].strip(), f"governed trainer option is empty:{option}")
    return values[0]


def validate_governed_trainer_argv(
    trainer_argv: Sequence[str],
    stage_o_authorization_path: str | os.PathLike[str],
    *,
    parsed_args: argparse.Namespace | None = None,
) -> dict[str, str]:
    """Require the exact governed Stage-B selectors before delegation.

    The textual pass is intentionally narrow and accepts only argparse's exact
    ``--option value`` and ``--option=value`` spellings.  The accepted parser is
    authoritative for every other option; when its Namespace is supplied, these
    three load-bearing values are re-confirmed against that real parse.
    """

    _require(
        all(isinstance(item, str) for item in trainer_argv),
        "trainer argv must contain strings only",
    )
    checked_argv = list(trainer_argv)
    if "--" in checked_argv:
        checked_argv = checked_argv[: checked_argv.index("--")]
    launch_contract_raw = _one_exact_option(checked_argv, "--launch_contract_json")
    stage_o_from_argv_raw = _one_exact_option(checked_argv, "--stage_authorization_json")
    run_plan_stage_raw = _one_exact_option(checked_argv, "--run_plan_stage")
    launch_contract = launch_contract_raw.strip()
    stage_o_from_argv = stage_o_from_argv_raw.strip()
    run_plan_stage = run_plan_stage_raw.strip()
    _require(launch_contract_raw == launch_contract, "launch-contract path has outer whitespace")
    _require(
        stage_o_from_argv_raw == stage_o_from_argv,
        "Stage-O authorization path has outer whitespace",
    )
    _require(run_plan_stage_raw == run_plan_stage, "run-plan stage has outer whitespace")
    expected_stage_o_path = str(_resolved(stage_o_authorization_path))
    canonical_stage_o_from_argv = _canonical_authority_path(
        stage_o_from_argv, label="trainer Stage-O authorization"
    )
    _require(
        stage_o_from_argv == canonical_stage_o_from_argv == expected_stage_o_path,
        "governed trainer --stage_authorization_json does not match the bound artifact",
    )
    _require(
        run_plan_stage == "stage_b",
        "governed trainer --run_plan_stage must equal stage_b",
    )

    if parsed_args is not None:
        parsed_launch_contract = str(getattr(parsed_args, "launch_contract_json", "") or "").strip()
        parsed_stage_o = str(getattr(parsed_args, "stage_authorization_json", "") or "").strip()
        parsed_stage = str(getattr(parsed_args, "run_plan_stage", "") or "").strip()
        _require(
            parsed_launch_contract == launch_contract,
            "real parser changed --launch_contract_json",
        )
        _require(
            parsed_stage_o
            and parsed_stage_o
            == _canonical_authority_path(parsed_stage_o, label="real-parser Stage-O authorization")
            == expected_stage_o_path
            == stage_o_from_argv,
            "real parser changed --stage_authorization_json",
        )
        _require(parsed_stage == "stage_b", "real parser did not retain Stage B")

    return {
        "launch_contract_json": launch_contract,
        "stage_authorization_json": expected_stage_o_path,
        "run_plan_stage": run_plan_stage,
    }


def _validate_gate_a_snapshot_result(
    gate_a: object,
    *,
    stage_o_document: Mapping[str, Any],
    stage_o_snapshot_path: str,
    stage_o_snapshot_sha256: str,
) -> dict[str, Any]:
    _require(isinstance(gate_a, dict) and gate_a.get("passed") is True, "Gate A did not pass")
    _require(gate_a.get("stage") == "stage_b", "Gate A stage binding changed")
    _require(gate_a.get("scope") == "STAGE_O", "Gate A scope binding changed")
    _require(
        gate_a.get("stage_authorization_path") == stage_o_snapshot_path,
        "Gate A Stage-O authorization path changed",
    )
    _require(
        gate_a.get("stage_authorization_sha256") == stage_o_snapshot_sha256,
        "Gate A Stage-O authorization SHA changed",
    )
    _require(
        _strict_equal(gate_a.get("authorization"), stage_o_document),
        "Gate A validated a different Stage-O authorization document",
    )
    return gate_a


def _run_silent_gate_a(
    topology: AcceptedModuleTopology,
    parsed_args: argparse.Namespace,
    *,
    observed_runtime: Mapping[str, Any],
    stage_o_document: Mapping[str, Any],
    stage_o_snapshot_path: str,
    stage_o_snapshot_sha256: str,
) -> dict[str, Any]:
    """Invoke the accepted canonical Gate A without the trainer wrapper's success log.

    This function only wires the accepted trainer Namespace and pinned repository inputs
    into the accepted launch contract.  All validation remains in the reviewed
    ``gate_a_pre_construction`` implementation.
    """

    trainer = topology.trainer
    launch = topology.launch_contract
    _require(trainer is not None, "execution trainer import did not complete")
    launch.normalize_legacy_sampler_seed(parsed_args, "stage_b")
    gate_a = launch.gate_a_pre_construction(
        parsed_args,
        stage="stage_b",
        launch_contract_path=str(parsed_args.launch_contract_json),
        stage_authorization_path=stage_o_snapshot_path,
        exact_plan_path=trainer._resolve_path(str(parsed_args.run_plan_json)),
        pilot_acceptance_path=(
            trainer.PROJECT_ROOT
            / "runs/p_pilot_acceptance_and_exact_run_plan_v1_2026-08-31/evidence"
            / "PILOT_RESULT_OWNER_ACCEPTANCE.json"
        ),
        observed_runtime=dict(observed_runtime),
    )
    return _validate_gate_a_snapshot_result(
        gate_a,
        stage_o_document=stage_o_document,
        stage_o_snapshot_path=stage_o_snapshot_path,
        stage_o_snapshot_sha256=stage_o_snapshot_sha256,
    )


@contextmanager
def _guard_trainer_gate_a_snapshot(
    topology: AcceptedModuleTopology,
    *,
    stage_o_document: Mapping[str, Any],
    stage_o_snapshot_path: str,
    stage_o_snapshot_sha256: str,
):
    """Bind trainer-main's actual Gate A to the adapter's original Stage-O snapshot."""

    launch = topology.launch_contract
    original_gate_a = getattr(launch, "gate_a_pre_construction", None)
    _require(callable(original_gate_a), "canonical Gate A is unavailable")

    def guarded_gate_a(*args: Any, **kwargs: Any) -> dict[str, Any]:
        gate_a = original_gate_a(*args, **kwargs)
        validated = _validate_gate_a_snapshot_result(
            gate_a,
            stage_o_document=stage_o_document,
            stage_o_snapshot_path=stage_o_snapshot_path,
            stage_o_snapshot_sha256=stage_o_snapshot_sha256,
        )
        _require_unchanged_json_snapshot(
            stage_o_snapshot_path,
            expected_document=stage_o_document,
            expected_sha256=stage_o_snapshot_sha256,
            expected_path=stage_o_snapshot_path,
            label="Stage-O authorization inside trainer Gate A",
        )
        return validated

    launch.gate_a_pre_construction = guarded_gate_a
    try:
        yield
    finally:
        launch.gate_a_pre_construction = original_gate_a


def _gate_a_failures(exc: BaseException) -> list[str]:
    message = str(exc)
    failures = [line[4:] for line in message.splitlines() if line.startswith("  - ")]
    return failures or [message]


def _topology_summary(topology: AcceptedModuleTopology) -> dict[str, Any]:
    launch = topology.launch_contract
    parent = topology.pretrain_package
    trainer = topology.trainer
    bridge = sys.modules.get("pretrain.stage_n_successor_head_compatibility_bridge_v1")
    return {
        "canonical_to_bare_same_object": (
            sys.modules.get(CANONICAL_LAUNCH_NAME) is sys.modules.get(BARE_LAUNCH_NAME) is launch
        ),
        "parent_package_binding_same_object": (
            parent.__dict__.get("production_launch_contract_v1") is launch
        ),
        "duplicate_launch_module_object_count": max(
            0, len(_exact_path_module_objects(CANONICAL_LAUNCH_PATH)) - 1
        ),
        "launch_path": str(_module_file_path(launch)),
        "launch_sha256": file_sha256(CANONICAL_LAUNCH_PATH),
        "launch_spec_name": getattr(getattr(launch, "__spec__", None), "name", None),
        "trainer_path": str(_module_file_path(trainer)) if trainer is not None else None,
        "trainer_sha256": (file_sha256(CANONICAL_TRAINER_PATH) if trainer is not None else None),
        "trainer_spec_name": (
            getattr(getattr(trainer, "__spec__", None), "name", None)
            if trainer is not None
            else None
        ),
        "class_family_identity": {
            name: sys.modules.get(getattr(getattr(launch, name), "__module__", "")) is launch
            for name in _LAUNCH_FAMILY_TYPES
        },
        "bridge_launch_same_object": (
            getattr(bridge, "launch", None) is launch if bridge is not None else None
        ),
        "trainer_launch_imported_symbols": list(topology.launch_imported_symbols),
    }


def _json_safe_namespace(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        try:
            canonical_json_bytes(value)
        except (TypeError, ValueError):
            result[key] = repr(value)
        else:
            result[key] = value
    return result


def run_preflight(
    adapter_authorization_path: str | os.PathLike[str],
    stage_o_authorization_path: str | os.PathLike[str],
    trainer_argv: Sequence[str],
) -> dict[str, Any]:
    """Exercise real parser, Stage-O chain validation, and Gate A; never call trainer.main."""

    stage_o_authorization_path = _canonical_authority_path(
        stage_o_authorization_path, label="Stage-O authorization"
    )
    adapter_authorization_path = _canonical_authority_path(
        adapter_authorization_path, label="adapter authorization"
    )
    validate_governed_trainer_argv(trainer_argv, stage_o_authorization_path)
    adapter_doc, _adapter_bytes, adapter_sha, adapter_snapshot_path = _load_json_snapshot(
        adapter_authorization_path
    )
    stage_o_doc, _stage_o_bytes, stage_o_sha, stage_o_snapshot_path = _load_json_snapshot(
        stage_o_authorization_path
    )
    preliminary = validate_adapter_authorization(adapter_doc, require_execution=False)
    _require(
        preliminary["identity_valid"],
        "adapter preflight refused identity: " + ", ".join(preliminary["identity_failures"]),
    )

    observed_runtime: dict[str, Any] = {}
    final_adapter_verdict: dict[str, Any] = {}

    def verify_before_trainer(launch: ModuleType) -> None:
        nonlocal observed_runtime, final_adapter_verdict
        observed_runtime = launch.observed_training_runtime(num_workers=2)
        final_adapter_verdict = validate_adapter_authorization(
            adapter_doc,
            observed_runtime=observed_runtime,
            require_execution=False,
            adapter_authorization_path=adapter_authorization_path,
            stage_o_authorization_path=stage_o_authorization_path,
            stage_o_authorization=stage_o_doc,
            stage_o_authorization_snapshot_path=stage_o_snapshot_path,
            stage_o_authorization_snapshot_sha256=stage_o_sha,
            trainer_argv=trainer_argv,
        )
        _require(
            final_adapter_verdict["identity_valid"],
            "adapter preflight refused live identity: "
            + ", ".join(final_adapter_verdict["identity_failures"]),
        )
        _require(
            not final_adapter_verdict["binding_failures"],
            "adapter preflight refused Stage-O binding: "
            + ", ".join(final_adapter_verdict["binding_failures"]),
        )

    topology = install_accepted_module_topology(before_trainer=verify_before_trainer)
    trainer = topology.trainer
    _require(trainer is not None, "preflight trainer import did not complete")
    args = parse_trainer_args(trainer, trainer_argv)
    validate_governed_trainer_argv(
        trainer_argv,
        stage_o_authorization_path,
        parsed_args=args,
    )
    stage_o_chain = topology.launch_contract.validate_stage_o_chain(
        stage_o_doc, observed_runtime=observed_runtime
    )

    trainer.validate_training_args(args)
    gate_a_reached = False
    gate_a_passed = False
    gate_a_exception_type: str | None = None
    gate_a_failures: list[str] = []
    try:
        _require_unchanged_json_snapshot(
            stage_o_authorization_path,
            expected_document=stage_o_doc,
            expected_sha256=stage_o_sha,
            expected_path=stage_o_snapshot_path,
            label="Stage-O authorization before preflight Gate A",
        )
        gate_a_reached = True
        trainer.enforce_governed_launch(args)
        gate_a_passed = True
    except topology.launch_contract.LaunchContractError as exc:
        gate_a_exception_type = type(exc).__name__
        gate_a_failures = _gate_a_failures(exc)
    finally:
        _require_unchanged_json_snapshot(
            stage_o_authorization_path,
            expected_document=stage_o_doc,
            expected_sha256=stage_o_sha,
            expected_path=stage_o_snapshot_path,
            label="Stage-O authorization after preflight Gate A",
        )

    module_origin_failures = [
        item for item in gate_a_failures if "module" in item.lower() and "origin" in item.lower()
    ]
    gate_binding_failures = [
        item
        for item in gate_a_failures
        if any(token in item.lower() for token in ("binding", "mismatch"))
        and "authorization_status" not in item
    ]
    gate_owner_failures = [
        item
        for item in gate_a_failures
        if item in {"authorization_status_not_authorized"} or "owner" in item.lower()
    ]
    topology_summary = _topology_summary(topology)
    return {
        "schema_version": ADAPTER_PREFLIGHT_SCHEMA,
        "mode": "PREFLIGHT",
        "adapter_authorization_path": adapter_snapshot_path,
        "adapter_authorization_sha256": adapter_sha,
        "adapter_authorization": final_adapter_verdict,
        "stage_o_authorization_path": stage_o_snapshot_path,
        "stage_o_authorization_sha256": stage_o_sha,
        "module_topology": topology_summary,
        "trainer_parser_reached": True,
        "trainer_namespace": _json_safe_namespace(args),
        "stage_o_chain": stage_o_chain,
        "gate_a_reached": gate_a_reached,
        "gate_a_passed": gate_a_passed,
        "gate_a_exception_type": gate_a_exception_type,
        "gate_a_failures": gate_a_failures,
        "module_origin_failures": module_origin_failures,
        "binding_failures": gate_binding_failures,
        "owner_state_failures": gate_owner_failures,
        "model_construction_reached": False,
        "checkpoint_restore_reached": False,
        "compile_realization_reached": False,
        "model_forward_reached": False,
        "backward_reached": False,
        "optimizer_update_reached": False,
    }


def run_execution(
    adapter_authorization_path: str | os.PathLike[str],
    stage_o_authorization_path: str | os.PathLike[str],
    trainer_argv: Sequence[str],
) -> int:
    """Validate both authorities, then delegate once without capturing output or failures."""

    stage_o_authorization_path = _canonical_authority_path(
        stage_o_authorization_path, label="Stage-O authorization"
    )
    adapter_authorization_path = _canonical_authority_path(
        adapter_authorization_path, label="adapter authorization"
    )
    validate_governed_trainer_argv(trainer_argv, stage_o_authorization_path)
    adapter_doc, _adapter_bytes, _adapter_sha, _adapter_snapshot_path = _load_json_snapshot(
        adapter_authorization_path
    )
    stage_o_doc, _stage_o_bytes, stage_o_sha, stage_o_snapshot_path = _load_json_snapshot(
        stage_o_authorization_path
    )

    static_verdict = validate_adapter_authorization(
        adapter_doc,
        require_execution=True,
        adapter_authorization_path=adapter_authorization_path,
        stage_o_authorization_path=stage_o_authorization_path,
        stage_o_authorization=stage_o_doc,
        stage_o_authorization_snapshot_path=stage_o_snapshot_path,
        stage_o_authorization_snapshot_sha256=stage_o_sha,
        trainer_argv=trainer_argv,
    )
    _require(
        static_verdict["authorized"],
        "adapter execution refused before project import: " + ", ".join(static_verdict["failures"]),
    )

    execution_runtime: dict[str, Any] = {}

    def verify_before_trainer(launch: ModuleType) -> None:
        nonlocal execution_runtime
        execution_runtime = launch.observed_training_runtime(num_workers=2)
        verdict = validate_adapter_authorization(
            adapter_doc,
            observed_runtime=execution_runtime,
            require_execution=True,
            adapter_authorization_path=adapter_authorization_path,
            stage_o_authorization_path=stage_o_authorization_path,
            stage_o_authorization=stage_o_doc,
            stage_o_authorization_snapshot_path=stage_o_snapshot_path,
            stage_o_authorization_snapshot_sha256=stage_o_sha,
            trainer_argv=trainer_argv,
        )
        _require(
            verdict["authorized"],
            "adapter execution refused: " + ", ".join(verdict["failures"]),
        )

    topology = install_accepted_module_topology(before_trainer=verify_before_trainer)
    trainer = topology.trainer
    _require(trainer is not None, "execution trainer import did not complete")
    parsed_args = parse_trainer_args(trainer, trainer_argv)
    validate_governed_trainer_argv(
        trainer_argv,
        stage_o_authorization_path,
        parsed_args=parsed_args,
    )
    trainer.validate_training_args(parsed_args)
    _run_silent_gate_a(
        topology,
        parsed_args,
        observed_runtime=execution_runtime,
        stage_o_document=stage_o_doc,
        stage_o_snapshot_path=stage_o_snapshot_path,
        stage_o_snapshot_sha256=stage_o_sha,
    )
    _require_unchanged_json_snapshot(
        stage_o_authorization_path,
        expected_document=stage_o_doc,
        expected_sha256=stage_o_sha,
        expected_path=stage_o_snapshot_path,
        label="Stage-O authorization after Gate A",
    )
    _validate_topology(topology)
    previous = list(sys.argv)
    with _guard_trainer_gate_a_snapshot(
        topology,
        stage_o_document=stage_o_doc,
        stage_o_snapshot_path=stage_o_snapshot_path,
        stage_o_snapshot_sha256=stage_o_sha,
    ):
        sys.argv[:] = [str(CANONICAL_TRAINER_PATH), *trainer_argv]
        try:
            _require_unchanged_json_snapshot(
                stage_o_authorization_path,
                expected_document=stage_o_doc,
                expected_sha256=stage_o_sha,
                expected_path=stage_o_snapshot_path,
                label="Stage-O authorization before trainer main",
            )
            result = trainer.main()
        finally:
            sys.argv[:] = previous
    if result is None:
        return 0
    _require(type(result) is int, "accepted trainer main returned a non-integer exit code")
    return result


def _write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> None:
    destination = _resolved(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(canonical_json_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _split_adapter_and_trainer_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    if "--" not in values:
        return values, []
    separator = values.index("--")
    return values[:separator], values[separator + 1 :]


def _adapter_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reviewed topology adapter for the accepted successor Stage-O trainer"
    )
    parser.add_argument(
        "mode", choices=("authorization-template", "closure", "preflight", "execute")
    )
    parser.add_argument("--adapter-authorization-path")
    parser.add_argument("--stage-o-authorization-path")
    parser.add_argument("--output", default="-")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        _require(
            bool(sys.flags.isolated and sys.dont_write_bytecode),
            "Stage-O adapter CLI requires Python flags -I -B",
        )
    adapter_argv, trainer_argv = _split_adapter_and_trainer_argv(
        sys.argv[1:] if argv is None else argv
    )
    options = _adapter_parser().parse_args(adapter_argv)

    if options.mode in {"preflight", "execute"}:
        _require(bool(trainer_argv), "trainer argv is required after one -- separator")
        _require(
            options.adapter_authorization_path is not None,
            "--adapter-authorization-path is required",
        )
        _require(
            options.stage_o_authorization_path is not None,
            "--stage-o-authorization-path is required",
        )
    else:
        _require(not trainer_argv, f"{options.mode} does not accept trainer argv")

    if options.mode == "closure":
        document = {
            "accepted_stage_o_trainer": accepted_trainer_identity(),
            "stage_o_launch_adapter": adapter_identity(),
            "adapter_tool_closure": adapter_tool_closure(),
        }
        sys.stdout.buffer.write(canonical_json_bytes(document))
        return 0
    if options.mode == "authorization-template":
        stage_o_path = options.stage_o_authorization_path
        launch = install_accepted_launch_topology()
        runtime = launch.observed_training_runtime(num_workers=2)
        document = adapter_authorization_template(
            runtime_fingerprint=runtime,
            stage_o_authorization_path=stage_o_path,
        )
        if options.output == "-":
            sys.stdout.buffer.write(canonical_json_bytes(document))
        else:
            _write_json(options.output, document)
        return 0
    if options.mode == "preflight":
        report = run_preflight(
            options.adapter_authorization_path,
            options.stage_o_authorization_path,
            trainer_argv,
        )
        if options.output == "-":
            sys.stdout.buffer.write(canonical_json_bytes(report))
        else:
            _write_json(options.output, report)
        return 0
    return run_execution(
        options.adapter_authorization_path,
        options.stage_o_authorization_path,
        trainer_argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
