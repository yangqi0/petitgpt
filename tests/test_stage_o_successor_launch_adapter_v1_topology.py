"""Fresh-process topology tests for the Stage-O successor launch adapter.

The accepted trainer and launch contract are imported only inside disposable subprocesses.
No probe calls the accepted trainer ``main`` path, restores a checkpoint, constructs a model,
realizes compile, executes a forward/backward pass, or takes an optimizer step.  Delegation
tests use an in-memory fake trainer only.

Retained references are intentional.  Looking only at the final ``sys.modules`` dictionary
would miss an adapter that overwrote an invalid binding while leaving its displaced class
family reachable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

ADAPTER_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ADAPTER_ROOT / "tools/stage_o_successor_launch_adapter_v1.py"
ACCEPTED_ROOT = Path("/workspace/petitgpt_stage_n_result_publication_recovery_v1")
HISTORICAL_ROOT = Path("/workspace/petitgpt")
PROJECT_PYTHON = HISTORICAL_ROOT / ".venv/bin/python"

CANONICAL_LAUNCH = "pretrain.production_launch_contract_v1"
BARE_LAUNCH = "production_launch_contract_v1"
CANONICAL_TRAINER = "pretrain.train_pretrain_with_bench"
BARE_TRAINER = "train_pretrain_with_bench"

EXPECTED_SHA256 = {
    "launch": "9e858078e7e492bed6de3b3ce34395d44fb81f3f06aab59c9960d447b7bde861",
    "trainer": "d9d7e61e5b30b5e24d49d92b2ea6bfc7557b6361d24a89467fdf06634d774fa4",
}


_PROBE_PREAMBLE = r"""
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

ADAPTER_PATH = Path(os.environ["PETITGPT_ADAPTER_PATH"]).resolve()
ACCEPTED_ROOT = Path(os.environ["PETITGPT_ACCEPTED_ROOT"]).resolve()
HISTORICAL_ROOT = Path(os.environ["PETITGPT_HISTORICAL_ROOT"]).resolve()
EXPECTED_SHA256 = json.loads(os.environ["PETITGPT_EXPECTED_SHA256"])
CANONICAL_LAUNCH = "pretrain.production_launch_contract_v1"
BARE_LAUNCH = "production_launch_contract_v1"
CANONICAL_TRAINER = "pretrain.train_pretrain_with_bench"
BARE_TRAINER = "train_pretrain_with_bench"
RESULT_PREFIX = "PETITGPT_STAGE_O_ADAPTER_PROBE_RESULT="
ABSENT = object()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accepted_path(relative):
    return (ACCEPTED_ROOT / relative).resolve()


def exact_path_objects(path):
    target = Path(path).resolve()
    found = {}
    for candidate in tuple(sys.modules.values()):
        module_file = getattr(candidate, "__file__", None)
        if not module_file:
            continue
        try:
            origin = Path(str(module_file)).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if origin == target:
            found[id(candidate)] = candidate
    return found


def import_adapter():
    # Use an exact path: importing through the adapter worktree root could accidentally make
    # its byte-identical copies of accepted production modules importable.
    spec = importlib.util.spec_from_file_location(
        "_petitgpt_stage_o_successor_launch_adapter_v1_test_subject", ADAPTER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_accepted_pretrain_parent():
    spec = importlib.machinery.ModuleSpec("pretrain", loader=None, is_package=True)
    spec.submodule_search_locations = [str(accepted_path("pretrain"))]
    parent = importlib.util.module_from_spec(spec)
    parent.__path__ = list(spec.submodule_search_locations)
    sys.modules["pretrain"] = parent
    return parent


def load_source(name, path, *, parent_attribute=True):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
        raise
    if parent_attribute and "." in name:
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    return module


def snapshot_slots():
    names = (
        "pretrain",
        CANONICAL_LAUNCH,
        BARE_LAUNCH,
        CANONICAL_TRAINER,
        BARE_TRAINER,
    )
    modules = {name: sys.modules.get(name, ABSENT) for name in names}
    parent = modules["pretrain"]
    attrs = {}
    if parent is not ABSENT and parent is not None:
        for child in ("production_launch_contract_v1", "train_pretrain_with_bench"):
            attrs[child] = getattr(parent, child, ABSENT)
    return modules, attrs


def assert_snapshot_unchanged(snapshot):
    modules, attrs = snapshot
    for name, previous in modules.items():
        if previous is ABSENT:
            assert name not in sys.modules, (name, sys.modules.get(name))
        else:
            assert name in sys.modules and sys.modules[name] is previous, name
    parent = modules["pretrain"]
    if parent is not ABSENT and parent is not None:
        for child, previous in attrs.items():
            if previous is ABSENT:
                assert child not in parent.__dict__, child
            else:
                assert parent.__dict__.get(child) is previous, child


def assert_reviewed_module(module, relative, canonical_name):
    path = accepted_path(relative)
    assert Path(module.__file__).resolve() == path
    assert Path(module.__spec__.origin).resolve() == path
    assert module.__spec__.name == canonical_name
    assert module.__name__ == canonical_name
    expected_key = "launch" if relative.endswith("production_launch_contract_v1.py") else "trainer"
    assert sha256_file(path) == EXPECTED_SHA256[expected_key]


def assert_success(topology):
    launch = topology.launch_contract
    trainer = topology.trainer
    parent = topology.pretrain_package

    assert sys.modules[CANONICAL_LAUNCH] is launch
    assert sys.modules[BARE_LAUNCH] is launch
    assert sys.modules[CANONICAL_TRAINER] is trainer
    assert sys.modules[BARE_TRAINER] is trainer
    assert sys.modules["pretrain"] is parent
    assert parent.__dict__["production_launch_contract_v1"] is launch
    assert parent.__dict__["train_pretrain_with_bench"] is trainer

    assert_reviewed_module(
        launch, "pretrain/production_launch_contract_v1.py", CANONICAL_LAUNCH
    )
    assert_reviewed_module(
        trainer, "pretrain/train_pretrain_with_bench.py", CANONICAL_TRAINER
    )
    assert len(exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py"))) == 1
    assert len(exact_path_objects(accepted_path("pretrain/train_pretrain_with_bench.py"))) == 1

    for class_name in ("LaunchContractError", "_Missing", "ObservedForward"):
        class_object = getattr(launch, class_name)
        assert class_object.__module__ == CANONICAL_LAUNCH
        assert sys.modules[class_object.__module__] is launch

    tree = ast.parse(
        accepted_path("pretrain/train_pretrain_with_bench.py").read_text(encoding="utf-8")
    )
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == BARE_LAUNCH
        for alias in node.names
    }
    assert imported_names
    assert imported_names == set(topology.launch_imported_symbols)
    assert all(hasattr(launch, name) for name in imported_names)
    return launch, trainer, parent


def governed_argv(stage_o_path, *extra):
    return [
        "--launch_contract_json",
        "/tmp/reviewed-launch-contract.json",
        "--stage_authorization_json",
        str(Path(stage_o_path).resolve()),
        "--run_plan_stage",
        "stage_b",
        *extra,
    ]


def governed_namespace(stage_o_path):
    return A.argparse.Namespace(
        launch_contract_json="/tmp/reviewed-launch-contract.json",
        stage_authorization_json=str(Path(stage_o_path).resolve()),
        run_plan_stage="stage_b",
    )


def emit(payload):
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))


A = import_adapter()
assert issubclass(A.AdapterError, RuntimeError)
"""


def _run_probe(
    body: str,
    *,
    case: str | None = None,
    include_process_output: bool = False,
) -> dict:
    assert ADAPTER_PATH.is_file(), f"adapter implementation not found: {ADAPTER_PATH}"
    assert PROJECT_PYTHON.is_file(), f"project interpreter not found: {PROJECT_PYTHON}"
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        # The adapter itself must select the immutable accepted root.  Never offer either
        # clone through PYTHONPATH as an accidental source of authority.
        "PYTHONPATH": "",
        "PETITGPT_ADAPTER_PATH": str(ADAPTER_PATH),
        "PETITGPT_ACCEPTED_ROOT": str(ACCEPTED_ROOT),
        "PETITGPT_HISTORICAL_ROOT": str(HISTORICAL_ROOT),
        "PETITGPT_EXPECTED_SHA256": json.dumps(EXPECTED_SHA256, sort_keys=True),
    })
    if case is not None:
        environment["PETITGPT_ADAPTER_PROBE_CASE"] = case
    completed = subprocess.run(
        [str(PROJECT_PYTHON), "-B", "-c", _PROBE_PREAMBLE + textwrap.dedent(body)],
        cwd=HISTORICAL_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"fresh-process probe failed with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("PETITGPT_STAGE_O_ADAPTER_PROBE_RESULT=")
    ]
    assert len(result_lines) == 1, (
        f"probe did not emit one result\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    result = json.loads(result_lines[0].split("=", 1)[1])
    if include_process_output:
        result["_process_stdout_lines"] = [
            line
            for line in completed.stdout.splitlines()
            if not line.startswith("PETITGPT_STAGE_O_ADAPTER_PROBE_RESULT=")
        ]
        result["_process_stderr_lines"] = completed.stderr.splitlines()
    return result


def test_importing_adapter_does_not_import_a_petitgpt_project_module():
    observed = _run_probe(
        r"""
        accepted_objects = []
        for relative in (
            "pretrain/production_launch_contract_v1.py",
            "pretrain/train_pretrain_with_bench.py",
            "pretrain/dataset_pretrain.py",
            "pretrain/sample.py",
            "src/model.py",
            "src/optim.py",
        ):
            accepted_objects.extend(exact_path_objects(accepted_path(relative)))
        assert not accepted_objects
        for name in (
            "pretrain",
            CANONICAL_LAUNCH,
            BARE_LAUNCH,
            CANONICAL_TRAINER,
            BARE_TRAINER,
            "dataset_pretrain",
            "sample",
            "src",
        ):
            assert name not in sys.modules
        emit({"adapter_import_is_project_clean": True})
        """
    )
    assert observed == {"adapter_import_is_project_clean": True}


def test_adapter_closure_is_one_script_with_ast_exact_import_roots_and_no_local_graph():
    observed = _run_probe(
        r"""
        adapter_root = ADAPTER_PATH.parents[1]
        source = ADAPTER_PATH.read_bytes()
        relative_path = "tools/stage_o_successor_launch_adapter_v1.py"
        tree = ast.parse(source.decode("utf-8"), filename=relative_path)
        absolute_modules = set()
        local_candidates = set()
        relative_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    absolute_modules.add(alias.name)
                    local_candidates.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_imports.add("." * node.level + (node.module or ""))
                    continue
                assert node.module
                absolute_modules.add(node.module)
                local_candidates.add(node.module)
                local_candidates.update(
                    f"{node.module}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )

        independently_resolved_local = set()
        for module_name in local_candidates:
            parts = module_name.split(".")
            candidates = set()
            for import_root in (adapter_root, ADAPTER_PATH.parent):
                candidates.update({
                    import_root.joinpath(*parts).with_suffix(".py"),
                    import_root.joinpath(*parts, "__init__.py"),
                })
                candidates.update(
                    import_root.joinpath(*parts[:index], "__init__.py")
                    for index in range(1, len(parts))
                )
            independently_resolved_local.update(
                str(candidate.relative_to(adapter_root))
                for candidate in candidates
                if candidate.is_file()
            )

        expected_external_roots = sorted(
            {module_name.split(".", 1)[0] for module_name in absolute_modules}
        )
        assert "__future__" in expected_external_roots
        assert not relative_imports
        assert not independently_resolved_local

        closure = A.adapter_tool_closure()
        script_sha = hashlib.sha256(source).hexdigest()
        expected_bundle = hashlib.sha256((json.dumps(
            {
                "schema_version": "petitgpt-stage-o-successor-launch-adapter-bundle-v1",
                "files": {relative_path: script_sha},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n").encode("utf-8")).hexdigest()
        assert closure["roots"] == [relative_path]
        assert closure["derived_closure"] == [relative_path]
        assert closure["derived_closure_count"] == 1
        assert closure["files"] == {relative_path: script_sha}
        assert closure["local_import_graph"] == {relative_path: []}
        assert closure["external_non_repository_modules"] == expected_external_roots
        assert closure["unbound_load_bearing_modules"] == []
        assert closure["unbound_load_bearing_module_count"] == 0
        assert closure["ADAPTER_TOOL_BUNDLE_SHA256"] == expected_bundle
        emit({
            "closure_count": 1,
            "local_graph_empty": True,
            "external_roots_ast_exact": True,
            "future_import_included": True,
            "bundle_independently_derived": True,
        })
        """
    )
    assert observed == {
        "bundle_independently_derived": True,
        "closure_count": 1,
        "external_roots_ast_exact": True,
        "future_import_included": True,
        "local_graph_empty": True,
    }


@pytest.mark.parametrize("shadow", ["json.py", "importlib/__init__.py"])
def test_adapter_closure_rejects_directly_importable_adjacent_shadow(shadow):
    observed = _run_probe(
        r"""
        shadow = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        temporary = tempfile.TemporaryDirectory()
        try:
            copied_root = Path(temporary.name) / "adapter-copy"
            copied_tools = copied_root / "tools"
            copied_tools.mkdir(parents=True)
            copied_script = copied_tools / "stage_o_successor_launch_adapter_v1.py"
            shutil.copy2(ADAPTER_PATH, copied_script)
            shadow_path = copied_tools / shadow
            shadow_path.parent.mkdir(parents=True, exist_ok=True)
            shadow_path.write_text("SHADOW_EXECUTED = True\n", encoding="utf-8")

            try:
                A.adapter_tool_closure(copied_root)
            except A.AdapterError as exc:
                message = str(exc)
                assert "imports repository-local code" in message
                assert str(shadow_path.relative_to(copied_root)) in message
            else:
                raise AssertionError(f"adjacent import shadow was accepted:{shadow}")
            emit({
                "shadow": shadow,
                "adjacent_shadow_refused": True,
                "shadow_executed": False,
            })
        finally:
            temporary.cleanup()
        """,
        case=shadow,
    )
    assert observed == {
        "adjacent_shadow_refused": True,
        "shadow": shadow,
        "shadow_executed": False,
    }


@pytest.mark.parametrize(
    ("python_flags", "missing_flag"),
    [(("-B",), "-I"), (("-I",), "-B")],
)
def test_direct_script_refuses_missing_isolation_flag_before_adjacent_json_executes(
    tmp_path,
    python_flags,
    missing_flag,
):
    copied_root = tmp_path / "adapter-copy"
    copied_tools = copied_root / "tools"
    copied_tools.mkdir(parents=True)
    copied_script = copied_tools / ADAPTER_PATH.name
    copied_script.write_bytes(ADAPTER_PATH.read_bytes())
    shadow_sentinel = tmp_path / "adjacent-json-executed"
    (copied_tools / "json.py").write_text(
        f"open({str(shadow_sentinel)!r}, 'w', encoding='utf-8').write('executed')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [str(PROJECT_PYTHON), *python_flags, str(copied_script), "closure"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr.strip() == "Stage-O adapter CLI requires Python flags -I -B"
    assert missing_flag in completed.stderr
    assert not shadow_sentinel.exists()


def test_direct_script_isolated_no_bytecode_closure_succeeds(tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [str(PROJECT_PYTHON), "-I", "-B", str(ADAPTER_PATH), "closure"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    document = json.loads(completed.stdout)
    closure = document["adapter_tool_closure"]
    identity = document["stage_o_launch_adapter"]
    assert closure["roots"] == ["tools/stage_o_successor_launch_adapter_v1.py"]
    assert closure["derived_closure_count"] == 1
    assert closure["unbound_load_bearing_module_count"] == 0
    assert identity["adapter_tool_path"] == str(ADAPTER_PATH)
    assert identity["adapter_tool_sha256"] == closure["files"][closure["roots"][0]]


def test_command_derivation_binds_exact_isolated_no_bytecode_python_flags():
    observed = _run_probe(
        r"""
        document = A.adapter_authorization_template()
        command = document["stage_o_command_derivation"]
        assert command["adapter_python_flags"] == ["-I", "-B"]
        assert command["policy"] == "EXACT_OWNER_BOUND_TRAINER_ARGV_AFTER_DOUBLE_DASH"
        emit({
            "adapter_python_flags": command["adapter_python_flags"],
            "exact_argv_match_required": command["exact_argv_match_required"],
        })
        """
    )
    assert observed == {
        "adapter_python_flags": ["-I", "-B"],
        "exact_argv_match_required": True,
    }


def test_adapter_tools_path_is_purged_before_real_trainer_torch_import():
    observed = _run_probe(
        r"""
        temporary = tempfile.TemporaryDirectory()
        try:
            copied_root = Path(temporary.name) / "adapter-copy"
            copied_tools = copied_root / "tools"
            copied_tools.mkdir(parents=True)
            shadow_sentinel = Path(temporary.name) / "adjacent-torch-executed"
            shadow_path = copied_tools / "torch.py"
            shadow_path.write_text(
                f"open({str(shadow_sentinel)!r}, 'w', encoding='utf-8').write('executed')\n"
                "raise RuntimeError('adjacent torch shadow executed')\n",
                encoding="utf-8",
            )
            assert "torch" not in sys.modules
            original_adapter_root = A._adapter_root
            A._adapter_root = lambda: copied_root
            sys.path.insert(0, str(copied_tools))

            topology = A.install_accepted_module_topology()
            _, trainer, _ = assert_success(topology)
            resolved_search_path = {
                Path(item or os.getcwd()).resolve() for item in sys.path
            }
            assert copied_root.resolve() not in resolved_search_path
            assert copied_tools.resolve() not in resolved_search_path
            assert not shadow_sentinel.exists()
            assert Path(trainer.torch.__file__).resolve() != shadow_path.resolve()
            A._adapter_root = original_adapter_root
            emit({
                "adapter_root_purged": True,
                "adapter_tools_purged": True,
                "adjacent_torch_executed": False,
                "real_trainer_imported": True,
            })
        finally:
            temporary.cleanup()
        """
    )
    assert observed == {
        "adapter_root_purged": True,
        "adapter_tools_purged": True,
        "adjacent_torch_executed": False,
        "real_trainer_imported": True,
    }


def test_fresh_adapter_first_installs_one_exact_launch_and_trainer_object():
    observed = _run_probe(
        r"""
        topology = A.install_accepted_module_topology()
        launch, trainer, parent = assert_success(topology)
        emit({
            "launch_identities": 1,
            "trainer_identities": 1,
            "parent_launch_same": parent.production_launch_contract_v1 is launch,
            "parent_trainer_same": parent.train_pretrain_with_bench is trainer,
        })
        """
    )
    assert observed == {
        "launch_identities": 1,
        "parent_launch_same": True,
        "parent_trainer_same": True,
        "trainer_identities": 1,
    }


@pytest.mark.parametrize("bind_bare", [False, True])
def test_exact_canonical_launch_first_reuses_retained_object_and_classes(bind_bare):
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        retained = load_source(
            CANONICAL_LAUNCH,
            accepted_path("pretrain/production_launch_contract_v1.py"),
        )
        retained_exports = (
            retained.LaunchContractError,
            retained._Missing,
            retained.ObservedForward,
            retained.gate_a_pre_construction,
            retained.Path,
            retained.hashlib,
            retained.__all__,
        )
        if json.loads(os.environ["PETITGPT_ADAPTER_PROBE_CASE"]):
            sys.modules[BARE_LAUNCH] = retained

        loaded = A.install_accepted_launch_topology()
        assert loaded is retained
        repeated_exports = (
            loaded.LaunchContractError,
            loaded._Missing,
            loaded.ObservedForward,
            loaded.gate_a_pre_construction,
            loaded.Path,
            loaded.hashlib,
            loaded.__all__,
        )
        assert all(
            original is repeated
            for original, repeated in zip(retained_exports, repeated_exports, strict=True)
        )
        assert sys.modules[CANONICAL_LAUNCH] is retained
        assert sys.modules[BARE_LAUNCH] is retained
        assert parent.production_launch_contract_v1 is retained
        assert len(exact_path_objects(retained.__file__)) == 1
        emit({"bare_prebound": json.loads(os.environ["PETITGPT_ADAPTER_PROBE_CASE"]),
              "retained": True})
        """,
        case=json.dumps(bind_bare),
    )
    assert observed == {"bare_prebound": bind_bare, "retained": True}


def test_exact_parent_package_first_receives_both_canonical_children():
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        parent_id = id(parent)
        topology = A.install_accepted_module_topology()
        launch, trainer, installed_parent = assert_success(topology)
        assert installed_parent is parent
        assert id(installed_parent) == parent_id
        emit({"parent_reused": True, "launch_same": parent.production_launch_contract_v1 is launch,
              "trainer_same": parent.train_pretrain_with_bench is trainer})
        """
    )
    assert observed == {"launch_same": True, "parent_reused": True, "trainer_same": True}


def test_repeated_setup_is_idempotent_and_does_not_reexecute_either_source():
    observed = _run_probe(
        r"""
        first = A.install_accepted_module_topology()
        launch, trainer, parent = assert_success(first)
        retained_exports = (
            launch.gate_a_pre_construction,
            launch.LaunchContractError,
            launch._MISSING,
            trainer.parse_args,
            trainer.validate_training_args,
            trainer.main,
            trainer.GPT,
        )

        second = A.install_accepted_module_topology()
        assert_success(second)
        assert second.launch_contract is launch
        assert second.trainer is trainer
        assert second.pretrain_package is parent
        repeated_exports = (
            second.launch_contract.gate_a_pre_construction,
            second.launch_contract.LaunchContractError,
            second.launch_contract._MISSING,
            second.trainer.parse_args,
            second.trainer.validate_training_args,
            second.trainer.main,
            second.trainer.GPT,
        )
        assert all(
            retained is repeated
            for retained, repeated in zip(retained_exports, repeated_exports, strict=True)
        )
        emit({"launch_reused": True, "trainer_reused": True, "parent_reused": True})
        """
    )
    assert observed == {"launch_reused": True, "parent_reused": True, "trainer_reused": True}


def test_retained_launch_function_mutations_fail_before_trainer_and_preserve_state():
    observed = _run_probe(
        r"""
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage-o.json"
            adapter_path.write_text("{}\n", encoding="utf-8")
            stage_o_path.write_text("{}\n", encoding="utf-8")
            trainer_argv = governed_argv(stage_o_path)
            launch = A.install_accepted_launch_topology()
            parent = sys.modules["pretrain"]
            assert CANONICAL_TRAINER not in sys.modules
            assert BARE_TRAINER not in sys.modules
            A.validate_adapter_authorization = lambda *args, **kwargs: {
                "identity_valid": True,
                "identity_failures": [],
                "binding_failures": [],
            }
            checked = []

            for symbol in (
                "gate_a_pre_construction",
                "validate_stage_o_chain",
                "observed_training_runtime",
            ):
                original = getattr(launch, symbol)

                def replacement(*args, **kwargs):
                    raise AssertionError(f"mutated launch function executed:{symbol}")

                setattr(launch, symbol, replacement)
                retained_snapshot = snapshot_slots()
                accepted_module_bindings = {
                    name: id(module)
                    for name, module in tuple(sys.modules.items())
                    if getattr(module, "__file__", None)
                    and str(Path(module.__file__).resolve()).startswith(
                        str(ACCEPTED_ROOT) + os.sep
                    )
                }
                before_trainer_events = []
                attempts = (
                    ("launch_setup", A.install_accepted_launch_topology),
                    (
                        "full_setup",
                        lambda: A.install_accepted_module_topology(
                            before_trainer=lambda module: before_trainer_events.append(module)
                        ),
                    ),
                    (
                        "preflight",
                        lambda: A.run_preflight(
                            adapter_path,
                            stage_o_path,
                            trainer_argv,
                        ),
                    ),
                )
                for operation, attempt in attempts:
                    try:
                        attempt()
                    except A.AdapterError as exc:
                        assert str(exc) == (
                            "adapter-owned module executable/export surface changed:"
                            + CANONICAL_LAUNCH
                        )
                    else:
                        raise AssertionError(
                            f"{operation} accepted mutated launch.{symbol}"
                        )
                    assert_snapshot_unchanged(retained_snapshot)
                    assert getattr(launch, symbol) is replacement
                    assert before_trainer_events == []
                    assert CANONICAL_TRAINER not in sys.modules
                    assert BARE_TRAINER not in sys.modules
                    assert {
                        name: id(module)
                        for name, module in tuple(sys.modules.items())
                        if getattr(module, "__file__", None)
                        and str(Path(module.__file__).resolve()).startswith(
                            str(ACCEPTED_ROOT) + os.sep
                        )
                    } == accepted_module_bindings
                    assert set(exact_path_objects(
                        accepted_path("pretrain/production_launch_contract_v1.py")
                    )) == {id(launch)}
                    assert not exact_path_objects(
                        accepted_path("pretrain/train_pretrain_with_bench.py")
                    )
                    checked.append(f"{symbol}:{operation}")

                setattr(launch, symbol, original)
                assert A.install_accepted_launch_topology() is launch
                assert sys.modules["pretrain"] is parent

            emit({
                "checked": checked,
                "mutation_preserved_on_refusal": True,
                "trainer_imported": False,
                "duplicate_launch_objects": 0,
                "clean_retry_after_explicit_restore": True,
            })
        finally:
            temporary.cleanup()
        """
    )
    assert observed == {
        "checked": [
            "gate_a_pre_construction:launch_setup",
            "gate_a_pre_construction:full_setup",
            "gate_a_pre_construction:preflight",
            "validate_stage_o_chain:launch_setup",
            "validate_stage_o_chain:full_setup",
            "validate_stage_o_chain:preflight",
            "observed_training_runtime:launch_setup",
            "observed_training_runtime:full_setup",
            "observed_training_runtime:preflight",
        ],
        "clean_retry_after_explicit_restore": True,
        "duplicate_launch_objects": 0,
        "mutation_preserved_on_refusal": True,
        "trainer_imported": False,
    }


def test_preloaded_launch_runtime_global_mutations_fail_closed_and_preserve_state():
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        retained = load_source(
            CANONICAL_LAUNCH,
            accepted_path("pretrain/production_launch_contract_v1.py"),
        )
        sys.modules[BARE_LAUNCH] = retained
        retained_snapshot = snapshot_slots()
        retained_sys_path = list(sys.path)
        retained_module_bindings = {
            name: id(module)
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and str(Path(module.__file__).resolve()).startswith(
                str(ACCEPTED_ROOT) + os.sep
            )
        }
        checked = []

        cases = (
            ("Path", "replacement", "accepted launch imported binding changed:Path"),
            (
                "hashlib",
                "replacement",
                "accepted launch imported binding changed:hashlib",
            ),
            (
                "CONTRACT_VERSION",
                "scalar",
                "accepted launch-contract runtime namespace changed",
            ),
            (
                "__all__",
                "mutable",
                "accepted launch-contract runtime namespace changed",
            ),
        )
        for symbol, mutation_kind, diagnostic in cases:
            original = getattr(retained, symbol)
            original_contents = tuple(original) if mutation_kind == "mutable" else None
            if mutation_kind == "mutable":
                original.append("_stage_o_adapter_mutated_export")
                replacement = original
            elif mutation_kind == "scalar":
                replacement = original + "-MUTATED"
                setattr(retained, symbol, replacement)
            else:
                replacement = object()
                setattr(retained, symbol, replacement)

            before_trainer_events = []
            attempts = (
                ("launch_setup", A.install_accepted_launch_topology),
                (
                    "full_setup",
                    lambda: A.install_accepted_module_topology(
                        before_trainer=lambda module: before_trainer_events.append(module)
                    ),
                ),
            )
            for operation, attempt in attempts:
                try:
                    attempt()
                except A.AdapterError as exc:
                    assert diagnostic in str(exc)
                else:
                    raise AssertionError(
                        f"{operation} accepted mutated launch.{symbol}"
                    )

                assert_snapshot_unchanged(retained_snapshot)
                assert list(sys.path) == retained_sys_path
                assert getattr(retained, symbol) is replacement
                if mutation_kind == "mutable":
                    assert tuple(replacement) == (
                        *original_contents,
                        "_stage_o_adapter_mutated_export",
                    )
                assert before_trainer_events == []
                assert CANONICAL_TRAINER not in sys.modules
                assert BARE_TRAINER not in sys.modules
                assert (
                    "pretrain.stage_n_successor_head_compatibility_bridge_v1"
                    not in sys.modules
                )
                assert {
                    name: id(module)
                    for name, module in tuple(sys.modules.items())
                    if getattr(module, "__file__", None)
                    and str(Path(module.__file__).resolve()).startswith(
                        str(ACCEPTED_ROOT) + os.sep
                    )
                } == retained_module_bindings
                assert set(
                    exact_path_objects(
                        accepted_path("pretrain/production_launch_contract_v1.py")
                    )
                ) == {id(retained)}
                checked.append(f"{symbol}:{operation}")

            if mutation_kind == "mutable":
                original[:] = original_contents
            else:
                setattr(retained, symbol, original)
            A._validate_launch_family(retained)

        loaded = A.install_accepted_launch_topology()
        assert loaded is retained
        assert sys.modules[CANONICAL_LAUNCH] is retained
        assert sys.modules[BARE_LAUNCH] is retained
        assert parent.production_launch_contract_v1 is retained
        assert len(exact_path_objects(retained.__file__)) == 1
        assert CANONICAL_TRAINER not in sys.modules
        assert BARE_TRAINER not in sys.modules
        emit({
            "checked": checked,
            "mutation_preserved_on_refusal": True,
            "no_dependency_or_trainer_residue": True,
            "clean_retry_reused_external_launch": True,
        })
        """
    )
    assert observed == {
        "checked": [
            "Path:launch_setup",
            "Path:full_setup",
            "hashlib:launch_setup",
            "hashlib:full_setup",
            "CONTRACT_VERSION:launch_setup",
            "CONTRACT_VERSION:full_setup",
            "__all__:launch_setup",
            "__all__:full_setup",
        ],
        "clean_retry_reused_external_launch": True,
        "mutation_preserved_on_refusal": True,
        "no_dependency_or_trainer_residue": True,
    }


@pytest.mark.parametrize("symbol", ["LaunchContractError", "_MISSING"])
def test_adapter_owned_launch_family_identity_replacement_fails_closed(symbol):
    observed = _run_probe(
        r"""
        symbol = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        launch = A.install_accepted_launch_topology()
        parent = sys.modules["pretrain"]
        original = getattr(launch, symbol)
        if symbol == "LaunchContractError":
            replacement = type(
                "LaunchContractError",
                (RuntimeError,),
                {
                    "__module__": CANONICAL_LAUNCH,
                    "__qualname__": "LaunchContractError",
                    "__doc__": original.__doc__,
                },
            )
            assert replacement.__name__ == original.__name__
            assert replacement.__qualname__ == original.__qualname__
            assert replacement.__module__ == original.__module__
            assert replacement.__bases__ == original.__bases__
        else:
            replacement = launch._Missing()
            assert type(replacement) is type(original)
            assert replacement.__dict__ == original.__dict__

        setattr(launch, symbol, replacement)
        retained_snapshot = snapshot_slots()
        retained_sys_path = list(sys.path)
        retained_module_bindings = {
            name: id(module)
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and str(Path(module.__file__).resolve()).startswith(
                str(ACCEPTED_ROOT) + os.sep
            )
        }
        before_trainer_events = []
        checked = []
        for operation, attempt in (
            ("launch_setup", A.install_accepted_launch_topology),
            (
                "full_setup",
                lambda: A.install_accepted_module_topology(
                    before_trainer=lambda module: before_trainer_events.append(module)
                ),
            ),
        ):
            try:
                attempt()
            except A.AdapterError as exc:
                assert str(exc) == (
                    "adapter-owned module executable/export surface changed:"
                    + CANONICAL_LAUNCH
                )
            else:
                raise AssertionError(
                    f"{operation} accepted replaced launch.{symbol} identity"
                )

            assert_snapshot_unchanged(retained_snapshot)
            assert list(sys.path) == retained_sys_path
            assert getattr(launch, symbol) is replacement
            assert before_trainer_events == []
            assert {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            } == retained_module_bindings
            assert set(
                exact_path_objects(
                    accepted_path("pretrain/production_launch_contract_v1.py")
                )
            ) == {id(launch)}
            assert CANONICAL_TRAINER not in sys.modules
            assert BARE_TRAINER not in sys.modules
            checked.append(operation)

        setattr(launch, symbol, original)
        assert A.install_accepted_launch_topology() is launch
        assert sys.modules[CANONICAL_LAUNCH] is launch
        assert sys.modules[BARE_LAUNCH] is launch
        assert parent.production_launch_contract_v1 is launch
        assert getattr(launch, symbol) is original
        emit({
            "symbol": symbol,
            "checked": checked,
            "replacement_preserved_on_refusal": True,
            "clean_retry_after_restore": True,
            "trainer_imported": False,
        })
        """,
        case=symbol,
    )
    assert observed == {
        "checked": ["launch_setup", "full_setup"],
        "clean_retry_after_restore": True,
        "replacement_preserved_on_refusal": True,
        "symbol": symbol,
        "trainer_imported": False,
    }


@pytest.mark.parametrize("symbol", ["main", "GPT", "coherent_GPT"])
def test_retained_trainer_symbol_mutation_fails_closed_without_repair_or_duplicates(
    symbol,
):
    observed = _run_probe(
        r"""
        symbol = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        topology = A.install_accepted_module_topology()
        launch, trainer, parent = assert_success(topology)
        checked = []

        trainer_symbol = "GPT" if symbol == "coherent_GPT" else symbol
        original = getattr(trainer, trainer_symbol)
        replacement = (lambda: 71) if symbol == "main" else object()
        dependency = sys.modules.get("src.model")
        dependency_original = None
        if trainer_symbol == "GPT":
            assert dependency is not None
            assert original is sys.modules["src.model"].GPT
        if symbol == "coherent_GPT":
            dependency_original = dependency.GPT
            dependency.GPT = replacement
        setattr(trainer, trainer_symbol, replacement)
        retained_snapshot = snapshot_slots()
        accepted_module_ids = {
            name: id(module)
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and str(Path(module.__file__).resolve()).startswith(str(ACCEPTED_ROOT) + os.sep)
        }

        attempts = (
            ("load", lambda: A.load_accepted_trainer(topology)),
            ("setup", A.install_accepted_module_topology),
        )
        for operation, attempt in attempts:
            try:
                attempt()
            except A.AdapterError as exc:
                expected_module = (
                    "src.model" if symbol == "coherent_GPT" else CANONICAL_TRAINER
                )
                assert (
                    "adapter-owned module executable/export surface changed:"
                    f"{expected_module}"
                ) in str(exc)
            else:
                raise AssertionError(f"{operation} accepted mutated trainer.{symbol}")
            assert_snapshot_unchanged(retained_snapshot)
            assert getattr(trainer, trainer_symbol) is replacement
            if symbol == "coherent_GPT":
                assert dependency.GPT is replacement
            assert {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            } == accepted_module_ids
            assert set(exact_path_objects(
                accepted_path("pretrain/production_launch_contract_v1.py")
            )) == {id(launch)}
            assert set(exact_path_objects(
                accepted_path("pretrain/train_pretrain_with_bench.py")
            )) == {id(trainer)}
            checked.append(f"{symbol}:{operation}")

        assert getattr(trainer, trainer_symbol) is replacement
        setattr(trainer, trainer_symbol, original)
        if symbol == "coherent_GPT":
            assert dependency.GPT is replacement
            dependency.GPT = dependency_original
        A._validate_topology(topology)
        assert_success(topology)
        assert topology.launch_contract is launch
        assert topology.trainer is trainer
        assert topology.pretrain_package is parent

        emit({
            "checked": checked,
            "mutation_preserved_on_refusal": True,
            "duplicate_launch_objects": 0,
            "duplicate_trainer_objects": 0,
            "integrity_valid_after_explicit_restore": True,
        })
        """,
        case=symbol,
    )
    assert observed == {
        "checked": [f"{symbol}:load", f"{symbol}:setup"],
        "duplicate_launch_objects": 0,
        "duplicate_trainer_objects": 0,
        "integrity_valid_after_explicit_restore": True,
        "mutation_preserved_on_refusal": True,
    }


def test_owned_gpt_config_dataclass_field_metadata_mutation_fails_closed():
    observed = _run_probe(
        r"""
        import dataclasses

        topology = A.install_accepted_module_topology()
        launch, trainer, parent = assert_success(topology)
        model_module = sys.modules["src.model"]
        config_class = model_module.GPTConfig
        field = config_class.__dataclass_fields__["vocab_size"]
        original_field_type = field._field_type
        assert original_field_type is dataclasses._FIELD

        retained_snapshot = snapshot_slots()
        retained_sys_path = list(sys.path)
        retained_module_bindings = {
            name: id(module)
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and str(Path(module.__file__).resolve()).startswith(
                str(ACCEPTED_ROOT) + os.sep
            )
        }
        field._field_type = dataclasses._FIELD_CLASSVAR
        attempts = (
            ("setup", A.install_accepted_module_topology),
            ("validate", lambda: A._validate_topology(topology)),
        )
        checked = []
        for operation, attempt in attempts:
            try:
                attempt()
            except A.AdapterError as exc:
                assert (
                    "adapter-owned module executable/export surface changed:src.model"
                    in str(exc)
                )
            else:
                raise AssertionError(
                    f"{operation} accepted mutated GPTConfig dataclass field metadata"
                )

            assert_snapshot_unchanged(retained_snapshot)
            assert list(sys.path) == retained_sys_path
            assert field._field_type is dataclasses._FIELD_CLASSVAR
            assert model_module.GPTConfig is config_class
            assert config_class.__dataclass_fields__["vocab_size"] is field
            assert {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            } == retained_module_bindings
            assert set(
                exact_path_objects(accepted_path("src/model.py"))
            ) == {id(model_module)}
            assert set(
                exact_path_objects(
                    accepted_path("pretrain/train_pretrain_with_bench.py")
                )
            ) == {id(trainer)}
            checked.append(operation)

        field._field_type = original_field_type
        A._validate_topology(topology)
        assert_success(topology)
        assert topology.launch_contract is launch
        assert topology.trainer is trainer
        assert topology.pretrain_package is parent
        emit({
            "checked": checked,
            "metadata_mutation_preserved_on_refusal": True,
            "no_model_or_config_constructed": True,
            "integrity_valid_after_explicit_restore": True,
        })
        """
    )
    assert observed == {
        "checked": ["setup", "validate"],
        "integrity_valid_after_explicit_restore": True,
        "metadata_mutation_preserved_on_refusal": True,
        "no_model_or_config_constructed": True,
    }


def test_owned_dependency_and_trainer_all_binding_deletion_never_reexecutes_source():
    observed = _run_probe(
        r"""
        topology = A.install_accepted_module_topology()
        launch, trainer, pretrain_parent = assert_success(topology)
        src_parent = sys.modules["src"]
        dependency = sys.modules["src.model"]
        checked = []

        cases = (
            (
                "src.model",
                dependency,
                src_parent,
                "model",
                "GPT",
                "adapter-owned module namespace names changed:src",
            ),
            (
                CANONICAL_TRAINER,
                trainer,
                pretrain_parent,
                "train_pretrain_with_bench",
                "main",
                f"adapter-owned canonical module binding changed:{CANONICAL_TRAINER}",
            ),
        )
        for (
            module_name,
            retained,
            parent,
            child_name,
            sentinel_name,
            expected_error,
        ) in cases:
            retained_bindings = {
                name: module
                for name, module in tuple(sys.modules.items())
                if module is retained
            }
            assert retained_bindings
            assert parent.__dict__.get(child_name) is retained
            retained_namespace = retained.__dict__
            retained_sentinel = getattr(retained, sentinel_name)
            expected_path = Path(retained.__file__).resolve()

            for name in retained_bindings:
                sys.modules.pop(name)
            parent.__dict__.pop(child_name)
            assert not exact_path_objects(expected_path)
            deleted_snapshot = snapshot_slots()
            accepted_bindings_after_deletion = {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            }
            before_trainer_events = []
            try:
                A.install_accepted_module_topology(
                    before_trainer=lambda module: before_trainer_events.append(module)
                )
            except A.AdapterError as exc:
                assert expected_error in str(exc)
            else:
                raise AssertionError(f"deleted owned {module_name} was silently reloaded")

            assert before_trainer_events == []
            assert_snapshot_unchanged(deleted_snapshot)
            assert {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            } == accepted_bindings_after_deletion
            assert not exact_path_objects(expected_path)
            assert retained.__dict__ is retained_namespace
            assert getattr(retained, sentinel_name) is retained_sentinel
            assert A._OWNED_MODULE_BASELINES[module_name].module is retained
            checked.append(module_name)

            sys.modules.update(retained_bindings)
            parent.__dict__[child_name] = retained

        A._validate_topology(topology)
        assert_success(topology)
        assert topology.launch_contract is launch
        assert topology.trainer is trainer
        emit({
            "checked": checked,
            "refused_before_trainer_callback": True,
            "deleted_state_preserved": True,
            "source_reexecution_count": 0,
            "baseline_objects_retained": True,
            "integrity_valid_after_explicit_restore": True,
        })
        """
    )
    assert observed == {
        "baseline_objects_retained": True,
        "checked": ["src.model", CANONICAL_TRAINER],
        "deleted_state_preserved": True,
        "integrity_valid_after_explicit_restore": True,
        "refused_before_trainer_callback": True,
        "source_reexecution_count": 0,
    }


@pytest.mark.parametrize("shadow", ["open", "isinstance", "__builtins__"])
def test_retained_trainer_added_global_or_replaced_builtins_fails_closed(shadow):
    observed = _run_probe(
        r"""
        shadow = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        topology = A.install_accepted_module_topology()
        launch, trainer, parent = assert_success(topology)
        retained_snapshot = snapshot_slots()
        accepted_module_bindings = {
            name: id(module)
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and str(Path(module.__file__).resolve()).startswith(
                str(ACCEPTED_ROOT) + os.sep
            )
        }
        sentinel = object()
        if shadow == "__builtins__":
            original = trainer.__dict__[shadow]
            replacement = dict(original)
            replacement["open"] = sentinel
            expected_error = (
                "adapter-owned module __builtins__ binding changed:" + CANONICAL_TRAINER
            )
        else:
            assert shadow not in trainer.__dict__
            original = ABSENT
            replacement = sentinel
            expected_error = (
                "adapter-owned module namespace names changed:" + CANONICAL_TRAINER
            )
        trainer.__dict__[shadow] = replacement

        try:
            A.load_accepted_trainer(topology)
        except A.AdapterError as exc:
            assert expected_error in str(exc)
        else:
            raise AssertionError(f"retained trainer shadow was accepted:{shadow}")

        assert_snapshot_unchanged(retained_snapshot)
        assert trainer.__dict__[shadow] is replacement
        if shadow == "__builtins__":
            assert trainer.__dict__[shadow]["open"] is sentinel
        assert {
            name: id(module)
            for name, module in tuple(sys.modules.items())
            if getattr(module, "__file__", None)
            and str(Path(module.__file__).resolve()).startswith(
                str(ACCEPTED_ROOT) + os.sep
            )
        } == accepted_module_bindings
        assert set(exact_path_objects(
            accepted_path("pretrain/production_launch_contract_v1.py")
        )) == {id(launch)}
        assert set(exact_path_objects(
            accepted_path("pretrain/train_pretrain_with_bench.py")
        )) == {id(trainer)}

        if original is ABSENT:
            trainer.__dict__.pop(shadow)
        else:
            trainer.__dict__[shadow] = original
        A._validate_topology(topology)
        assert_success(topology)
        assert topology.pretrain_package is parent
        emit({
            "shadow": shadow,
            "refused": True,
            "mutation_preserved": True,
            "duplicate_launch_objects": 0,
            "duplicate_trainer_objects": 0,
            "integrity_valid_after_explicit_restore": True,
        })
        """,
        case=shadow,
    )
    assert observed == {
        "duplicate_launch_objects": 0,
        "duplicate_trainer_objects": 0,
        "integrity_valid_after_explicit_restore": True,
        "mutation_preserved": True,
        "refused": True,
        "shadow": shadow,
    }


def test_unowned_exact_dependency_trainer_and_bridge_are_preserved_and_refused():
    observed = _run_probe(
        r"""
        checked = []

        def uninitialized_exact_module(name, relative, *, is_package=False):
            path = accepted_path(relative)
            kwargs = {}
            if is_package:
                kwargs["submodule_search_locations"] = [str(path.parent)]
            spec = importlib.util.spec_from_file_location(name, path, **kwargs)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            assert Path(module.__file__).resolve() == path
            assert Path(module.__spec__.origin).resolve() == path
            assert module.__spec__.name == name
            return module

        def assert_unowned_refused(unowned, module_name, case, launch):
            retained_snapshot = snapshot_slots()
            retained_sys_path = list(sys.path)
            accepted_module_bindings = {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            }
            before_trainer_events = []
            try:
                A.install_accepted_module_topology(
                    before_trainer=lambda module: before_trainer_events.append(module)
                )
            except A.AdapterError as exc:
                assert (
                    f"refusing unowned preloaded accepted module:{module_name}"
                    in str(exc)
                )
            else:
                raise AssertionError(f"unowned exact {case} module was accepted")

            assert before_trainer_events == []
            assert_snapshot_unchanged(retained_snapshot)
            assert sys.path == retained_sys_path
            assert {
                name: id(module)
                for name, module in tuple(sys.modules.items())
                if getattr(module, "__file__", None)
                and str(Path(module.__file__).resolve()).startswith(
                    str(ACCEPTED_ROOT) + os.sep
                )
            } == accepted_module_bindings
            launch_objects = exact_path_objects(
                accepted_path("pretrain/production_launch_contract_v1.py")
            )
            if launch is None:
                assert not launch_objects
            else:
                assert set(launch_objects) == {id(launch)}
            assert id(unowned) in exact_path_objects(Path(unowned.__file__))
            checked.append(case)

        parent = make_accepted_pretrain_parent()
        bridge_name = "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        unowned_bridge = uninitialized_exact_module(
            bridge_name,
            "pretrain/stage_n_successor_head_compatibility_bridge_v1.py",
        )
        sys.modules[bridge_name] = unowned_bridge
        parent.stage_n_successor_head_compatibility_bridge_v1 = unowned_bridge
        assert_unowned_refused(unowned_bridge, bridge_name, "bridge", None)
        assert sys.modules[bridge_name] is unowned_bridge
        sys.modules.pop(bridge_name)
        parent.__dict__.pop("stage_n_successor_head_compatibility_bridge_v1")
        assert CANONICAL_LAUNCH not in sys.modules
        assert BARE_LAUNCH not in sys.modules

        launch = A.install_accepted_launch_topology()
        assert sys.modules["pretrain"] is parent
        for case in ("dependency", "trainer"):
            if case == "dependency":
                src = uninitialized_exact_module(
                    "src", "src/__init__.py", is_package=True
                )
                unowned_model = uninitialized_exact_module("src.model", "src/model.py")
                sys.modules["src"] = src
                sys.modules["src.model"] = unowned_model
                src.model = unowned_model
                unowned = src

                def cleanup():
                    sys.modules.pop("src.model", None)
                    sys.modules.pop("src", None)

                module_name = "src"

            else:
                unowned = uninitialized_exact_module(
                    CANONICAL_TRAINER,
                    "pretrain/train_pretrain_with_bench.py",
                )
                sys.modules[CANONICAL_TRAINER] = unowned
                sys.modules[BARE_TRAINER] = unowned
                parent.train_pretrain_with_bench = unowned

                def cleanup():
                    sys.modules.pop(CANONICAL_TRAINER, None)
                    sys.modules.pop(BARE_TRAINER, None)
                    parent.__dict__.pop("train_pretrain_with_bench", None)

                module_name = CANONICAL_TRAINER

            assert_unowned_refused(unowned, module_name, case, launch)
            cleanup()
            assert A.install_accepted_launch_topology() is launch

        assert CANONICAL_TRAINER not in sys.modules
        assert BARE_TRAINER not in sys.modules
        emit({
            "checked": checked,
            "refused_before_trainer_import": True,
            "unowned_objects_preserved": True,
            "duplicate_launch_objects": 0,
            "clean_retry_after_explicit_removal": True,
        })
        """
    )
    assert observed == {
        "checked": ["bridge", "dependency", "trainer"],
        "clean_retry_after_explicit_removal": True,
        "duplicate_launch_objects": 0,
        "refused_before_trainer_import": True,
        "unowned_objects_preserved": True,
    }


def test_trainer_is_loaded_only_after_launch_topology_and_resolves_every_bare_import():
    observed = _run_probe(
        r"""
        launch = A.install_accepted_launch_topology()
        assert sys.modules[CANONICAL_LAUNCH] is launch
        assert sys.modules[BARE_LAUNCH] is launch
        assert CANONICAL_TRAINER not in sys.modules
        assert BARE_TRAINER not in sys.modules

        trainer = A.load_accepted_trainer()
        topology = A.install_accepted_module_topology()
        checked_launch, checked_trainer, _ = assert_success(topology)
        assert checked_launch is launch
        assert checked_trainer is trainer
        emit({"launch_before_trainer": True,
              "imported_symbol_count": len(topology.launch_imported_symbols)})
        """
    )
    assert observed["launch_before_trainer"] is True
    assert observed["imported_symbol_count"] > 0


@pytest.mark.parametrize(
    "case",
    [
        "historical_bare",
        "historical_canonical",
        "different_bytes_bare",
        "different_bytes_canonical",
        "same_bytes_different_path_bare",
        "same_bytes_different_path_canonical",
    ],
)
def test_wrong_path_or_bytes_launch_binding_fails_without_overwrite_or_residue(case):
    observed = _run_probe(
        r"""
        case = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        name = CANONICAL_LAUNCH if case.endswith("canonical") else BARE_LAUNCH
        temporary = tempfile.TemporaryDirectory()
        try:
            if case.startswith("historical"):
                source = HISTORICAL_ROOT / "pretrain/production_launch_contract_v1.py"
            else:
                source = Path(temporary.name) / "production_launch_contract_v1.py"
                source.write_bytes(accepted_path("pretrain/production_launch_contract_v1.py").read_bytes())
                if case.startswith("different_bytes"):
                    source.write_bytes(source.read_bytes() + b"\n# intentionally different bytes\n")
            retained = load_source(name, source, parent_attribute=False)
            retained_class = retained.LaunchContractError
            snapshot = snapshot_slots()
            exact_before = set(exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py")))
            try:
                A.install_accepted_module_topology()
            except A.AdapterError:
                pass
            else:
                raise AssertionError("invalid launch binding was accepted")
            assert_snapshot_unchanged(snapshot)
            assert sys.modules[name] is retained
            assert retained.LaunchContractError is retained_class
            exact_after = set(exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py")))
            assert exact_after == exact_before
            emit({"case": case, "failed_closed": True, "retained": True, "no_residue": True})
        finally:
            temporary.cleanup()
        """,
        case=case,
    )
    assert observed == {
        "case": case,
        "failed_closed": True,
        "no_residue": True,
        "retained": True,
    }


@pytest.mark.parametrize(
    "case",
    [
        "bare_no_file",
        "canonical_no_file",
        "bare_no_spec",
        "canonical_no_spec",
        "bare_none_partial",
        "canonical_none_partial",
    ],
)
def test_incomplete_launch_binding_fails_without_mutation(case):
    observed = _run_probe(
        r"""
        case = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        name = CANONICAL_LAUNCH if case.startswith("canonical") else BARE_LAUNCH
        if case.endswith("none_partial"):
            retained = None
        else:
            retained = types.ModuleType(name)
            if case.endswith("no_spec"):
                retained.__file__ = str(accepted_path("pretrain/production_launch_contract_v1.py"))
                retained.__spec__ = None
        sys.modules[name] = retained
        snapshot = snapshot_slots()
        exact_before = set(exact_path_objects(
            accepted_path("pretrain/production_launch_contract_v1.py")
        ))
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("incomplete launch binding was accepted")
        assert_snapshot_unchanged(snapshot)
        assert name in sys.modules and sys.modules[name] is retained
        exact_after = set(exact_path_objects(
            accepted_path("pretrain/production_launch_contract_v1.py")
        ))
        assert exact_after == exact_before
        emit({"case": case, "failed_closed": True, "retained": True})
        """,
        case=case,
    )
    assert observed == {"case": case, "failed_closed": True, "retained": True}


def test_missing_name_package_or_loader_metadata_fails_before_trainer_import():
    observed = _run_probe(
        r"""
        path = accepted_path("pretrain/production_launch_contract_v1.py")
        checked = []
        for field in ("__name__", "__package__", "__loader__"):
            spec = importlib.util.spec_from_file_location(CANONICAL_LAUNCH, path)
            assert spec is not None and spec.loader is not None
            retained = importlib.util.module_from_spec(spec)
            retained.__dict__.pop(field, None)
            sys.modules[CANONICAL_LAUNCH] = retained
            snapshot = snapshot_slots()
            exact_before = set(exact_path_objects(path))
            try:
                A.install_accepted_module_topology()
            except A.AdapterError:
                pass
            else:
                raise AssertionError(f"launch binding missing {field} was accepted")
            assert_snapshot_unchanged(snapshot)
            assert sys.modules[CANONICAL_LAUNCH] is retained
            assert set(exact_path_objects(path)) == exact_before
            assert CANONICAL_TRAINER not in sys.modules
            assert BARE_TRAINER not in sys.modules
            sys.modules.pop(CANONICAL_LAUNCH)
            checked.append(field)
        emit({"checked": checked, "failed_before_trainer": True, "no_mutation": True})
        """
    )
    assert observed == {
        "checked": ["__name__", "__package__", "__loader__"],
        "failed_before_trainer": True,
        "no_mutation": True,
    }


def test_two_launch_objects_for_the_exact_source_fail_without_displacing_either_family():
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        canonical = load_source(
            CANONICAL_LAUNCH, accepted_path("pretrain/production_launch_contract_v1.py")
        )
        bare = load_source(
            BARE_LAUNCH,
            accepted_path("pretrain/production_launch_contract_v1.py"),
            parent_attribute=False,
        )
        assert canonical is not bare
        assert canonical.LaunchContractError is not bare.LaunchContractError
        snapshot = snapshot_slots()
        before_ids = set(exact_path_objects(canonical.__file__))
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("split launch families were accepted")
        assert_snapshot_unchanged(snapshot)
        assert set(exact_path_objects(canonical.__file__)) == before_ids
        assert sys.modules[CANONICAL_LAUNCH] is canonical
        assert sys.modules[BARE_LAUNCH] is bare
        assert parent.production_launch_contract_v1 is canonical
        emit({"failed_closed": True, "retained_family_count": len(before_ids)})
        """
    )
    assert observed == {"failed_closed": True, "retained_family_count": 2}


def test_conflicting_parent_attribute_fails_without_creating_aliases():
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        conflict = types.ModuleType("conflicting_launch_contract")
        parent.production_launch_contract_v1 = conflict
        snapshot = snapshot_slots()
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("conflicting parent attribute was overwritten")
        assert_snapshot_unchanged(snapshot)
        assert parent.production_launch_contract_v1 is conflict
        assert CANONICAL_LAUNCH not in sys.modules
        assert BARE_LAUNCH not in sys.modules
        emit({"failed_closed": True, "parent_conflict_retained": True})
        """
    )
    assert observed == {"failed_closed": True, "parent_conflict_retained": True}


def test_historical_pretrain_namespace_fails_without_replacement():
    observed = _run_probe(
        r"""
        historical_path = str((HISTORICAL_ROOT / "pretrain").resolve())
        spec = importlib.machinery.ModuleSpec("pretrain", loader=None, is_package=True)
        spec.submodule_search_locations = [historical_path]
        parent = importlib.util.module_from_spec(spec)
        parent.__path__ = [historical_path]
        sys.modules["pretrain"] = parent
        snapshot = snapshot_slots()
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("historical pretrain namespace was replaced")
        assert_snapshot_unchanged(snapshot)
        assert sys.modules["pretrain"] is parent
        assert tuple(parent.__path__) == (historical_path,)
        assert not exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py"))
        emit({"failed_closed": True, "historical_parent_retained": True})
        """
    )
    assert observed == {"failed_closed": True, "historical_parent_retained": True}


def test_accepted_src_package_with_poisoned_search_path_fails_before_any_child_import():
    observed = _run_probe(
        r"""
        src_path = accepted_path("src/__init__.py")
        src = load_source("src", src_path)
        historical_src = str((HISTORICAL_ROOT / "src").resolve())
        src.__path__[:] = [historical_src]
        src.__spec__.submodule_search_locations[:] = [historical_src]
        retained_path = tuple(src.__path__)
        retained_spec_path = tuple(src.__spec__.submodule_search_locations)
        exact_before = set(exact_path_objects(src_path))
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("accepted src package with poisoned search path was accepted")
        assert sys.modules["src"] is src
        assert tuple(src.__path__) == retained_path
        assert tuple(src.__spec__.submodule_search_locations) == retained_spec_path
        assert set(exact_path_objects(src_path)) == exact_before
        for name in (
            "src.model", "src.optim", "src.canonical_loss", "src.canonical_schedule",
            "src.special_tokens", "src.tracking", CANONICAL_LAUNCH, BARE_LAUNCH,
            CANONICAL_TRAINER, BARE_TRAINER,
        ):
            assert name not in sys.modules
        emit({
            "failed_closed": True,
            "poisoned_paths_retained": True,
            "accepted_child_import_count": 0,
        })
        """
    )
    assert observed == {
        "accepted_child_import_count": 0,
        "failed_closed": True,
        "poisoned_paths_retained": True,
    }


def test_orphan_exact_launch_source_is_not_adopted_or_overwritten():
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        orphan = load_source(
            CANONICAL_LAUNCH, accepted_path("pretrain/production_launch_contract_v1.py")
        )
        sys.modules.pop(CANONICAL_LAUNCH)
        parent.__dict__.pop("production_launch_contract_v1")
        orphan_name = "_petitgpt_unreviewed_orphan_launch_alias"
        sys.modules[orphan_name] = orphan
        snapshot = snapshot_slots()
        before_ids = set(exact_path_objects(orphan.__file__))
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("orphan exact-source object was adopted")
        assert_snapshot_unchanged(snapshot)
        assert sys.modules[orphan_name] is orphan
        assert set(exact_path_objects(orphan.__file__)) == before_ids
        assert CANONICAL_LAUNCH not in sys.modules
        assert BARE_LAUNCH not in sys.modules
        emit({"failed_closed": True, "orphan_retained": True, "family_count": len(before_ids)})
        """
    )
    assert observed == {"failed_closed": True, "family_count": 1, "orphan_retained": True}


def test_retained_accepted_bridge_with_stale_launch_reference_fails_closed():
    observed = _run_probe(
        r"""
        launch = A.install_accepted_launch_topology()
        bridge = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        parent = sys.modules["pretrain"]
        assert bridge.launch is launch
        assert Path(bridge.__file__).resolve() == accepted_path(
            "pretrain/stage_n_successor_head_compatibility_bridge_v1.py"
        )

        sys.modules.pop(CANONICAL_LAUNCH)
        sys.modules.pop(BARE_LAUNCH)
        parent.__dict__.pop("production_launch_contract_v1")
        snapshot = snapshot_slots()
        before_ids = set(exact_path_objects(launch.__file__))
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("retained bridge with stale launch reference was accepted")

        assert_snapshot_unchanged(snapshot)
        assert bridge.launch is launch
        assert sys.modules[
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        ] is bridge
        assert parent.stage_n_successor_head_compatibility_bridge_v1 is bridge
        assert set(exact_path_objects(launch.__file__)) == before_ids
        assert CANONICAL_LAUNCH not in sys.modules
        assert BARE_LAUNCH not in sys.modules
        assert CANONICAL_TRAINER not in sys.modules
        assert BARE_TRAINER not in sys.modules
        emit({
            "failed_closed": True,
            "bridge_retained": True,
            "stale_launch_retained": True,
            "no_trainer_residue": True,
        })
        """
    )
    assert observed == {
        "bridge_retained": True,
        "failed_closed": True,
        "no_trainer_residue": True,
        "stale_launch_retained": True,
    }


@pytest.mark.parametrize("name", [CANONICAL_TRAINER, BARE_TRAINER])
def test_invalid_preexisting_trainer_binding_fails_before_launch_install(name):
    observed = _run_probe(
        r"""
        name = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        retained = types.ModuleType(name)
        retained.__file__ = str(HISTORICAL_ROOT / "pretrain/train_pretrain_with_bench.py")
        retained.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        retained.__spec__.origin = retained.__file__
        sys.modules[name] = retained
        snapshot = snapshot_slots()
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("invalid trainer binding was accepted")
        assert_snapshot_unchanged(snapshot)
        assert sys.modules[name] is retained
        assert not exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py"))
        assert not exact_path_objects(accepted_path("pretrain/train_pretrain_with_bench.py"))
        emit({"failed_closed": True, "name": name, "retained": True})
        """,
        case=name,
    )
    assert observed == {"failed_closed": True, "name": name, "retained": True}


def test_two_canonical_spec_trainer_objects_fail_without_displacement():
    observed = _run_probe(
        r"""
        parent = make_accepted_pretrain_parent()
        path = accepted_path("pretrain/train_pretrain_with_bench.py")

        def unexecuted_canonical_trainer():
            spec = importlib.util.spec_from_file_location(CANONICAL_TRAINER, path)
            assert spec is not None and spec.loader is not None
            return importlib.util.module_from_spec(spec)

        canonical = unexecuted_canonical_trainer()
        bare = unexecuted_canonical_trainer()
        assert canonical is not bare
        sys.modules[CANONICAL_TRAINER] = canonical
        sys.modules[BARE_TRAINER] = bare
        parent.train_pretrain_with_bench = canonical
        snapshot = snapshot_slots()
        before_ids = set(exact_path_objects(path))
        assert len(before_ids) == 2
        try:
            A.install_accepted_module_topology()
        except A.AdapterError:
            pass
        else:
            raise AssertionError("two canonical-spec trainer objects were accepted")
        assert_snapshot_unchanged(snapshot)
        assert set(exact_path_objects(path)) == before_ids
        assert sys.modules[CANONICAL_TRAINER] is canonical
        assert sys.modules[BARE_TRAINER] is bare
        assert parent.train_pretrain_with_bench is canonical
        assert not exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py"))
        emit({"failed_closed": True, "retained_trainer_objects": len(before_ids)})
        """
    )
    assert observed == {"failed_closed": True, "retained_trainer_objects": 2}


def test_initialization_failure_rolls_back_every_created_binding_then_clean_retry_succeeds():
    observed = _run_probe(
        r"""
        snapshot = snapshot_slots()
        sentinel = RuntimeError("intentional launch source compilation failure")
        real_compile = compile
        calls = []

        def fail_launch_compile_once(source, filename, *args, **kwargs):
            if (
                Path(filename).resolve()
                == accepted_path("pretrain/production_launch_contract_v1.py")
                and not calls
            ):
                calls.append(str(Path(filename).resolve()))
                raise sentinel
            return real_compile(source, filename, *args, **kwargs)

        A.compile = fail_launch_compile_once
        try:
            A.install_accepted_module_topology()
        except BaseException as exc:
            assert exc is sentinel
        else:
            raise AssertionError("launch source initialization failure did not stop setup")
        finally:
            A.compile = real_compile
        assert len(calls) == 1
        assert_snapshot_unchanged(snapshot)
        assert not exact_path_objects(accepted_path("pretrain/production_launch_contract_v1.py"))
        assert not exact_path_objects(accepted_path("pretrain/train_pretrain_with_bench.py"))

        topology = A.install_accepted_module_topology()
        assert_success(topology)
        emit({"source_compile_failures": len(calls), "rollback": True, "clean_retry": True})
        """
    )
    assert observed == {"clean_retry": True, "rollback": True, "source_compile_failures": 1}


def test_same_name_classes_from_another_module_are_not_the_canonical_family():
    observed = _run_probe(
        r"""
        topology = A.install_accepted_module_topology()
        launch, _, _ = assert_success(topology)
        fake = types.ModuleType("unreviewed_same_name_family")
        for name, base in (
            ("LaunchContractError", RuntimeError),
            ("_Missing", object),
            ("ObservedForward", object),
        ):
            setattr(fake, name, type(name, (base,), {"__module__": fake.__name__}))
        assert fake.LaunchContractError.__name__ == launch.LaunchContractError.__name__
        assert fake._Missing.__name__ == launch._Missing.__name__
        assert fake.ObservedForward.__name__ == launch.ObservedForward.__name__
        assert fake.LaunchContractError is not launch.LaunchContractError
        assert fake._Missing is not launch._Missing
        assert fake.ObservedForward is not launch.ObservedForward
        assert not isinstance(fake.LaunchContractError("x"), launch.LaunchContractError)
        assert not isinstance(fake._Missing(), launch._Missing)
        assert not isinstance(fake.ObservedForward(), launch.ObservedForward)
        emit({"same_names": True, "identity_rejected": True})
        """
    )
    assert observed == {"identity_rejected": True, "same_names": True}


def test_parse_trainer_args_uses_the_real_parser_and_restores_sys_argv():
    observed = _run_probe(
        r"""
        topology = A.install_accepted_module_topology()
        assert_success(topology)
        trainer_argv = [
            "--train_dir", "/tmp/train dir",
            "--val_dir", "/tmp/val dir",
            "--out_dir", "/tmp/out dir",
            "--samples_dir", "/tmp/samples dir",
            "--tokenizer_path", "/tmp/tokenizer.json",
            "--num_workers", "2",
            "--run_plan_stage", "stage_b",
            "--eval_steps", "38146,44631",
            "--eval_steps", "49590",
            "--compile",
        ]
        retained_argv = sys.argv
        retained_values = list(sys.argv)
        parsed = A.parse_trainer_args(topology.trainer, trainer_argv)
        assert sys.argv is retained_argv
        assert sys.argv == retained_values
        assert parsed.train_dir == "/tmp/train dir"
        assert parsed.val_dir == "/tmp/val dir"
        assert parsed.out_dir == "/tmp/out dir"
        assert parsed.samples_dir == "/tmp/samples dir"
        assert parsed.tokenizer_path == "/tmp/tokenizer.json"
        assert parsed.num_workers == 2
        assert parsed.run_plan_stage == "stage_b"
        assert parsed.eval_steps == ["38146,44631", "49590"]
        assert parsed.compile is True
        emit({"real_parser": True, "sys_argv_restored": True,
              "repeatable_flag_count": len(parsed.eval_steps)})
        """
    )
    assert observed == {
        "real_parser": True,
        "repeatable_flag_count": 2,
        "sys_argv_restored": True,
    }


@pytest.mark.parametrize("trainer_result", [None, 23])
def test_execution_delegates_once_with_exact_argv_streams_and_return_code(trainer_result):
    observed = _run_probe(
        r"""
        expected_result = json.loads(os.environ["PETITGPT_ADAPTER_PROBE_CASE"])
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage_o.json"
            adapter_path.write_text(json.dumps({
                "authorization_status": "AUTHORIZED",
                "authorizes_adapter_execution": True,
                "authorizes_training": False,
                "authorized_by": "test owner",
                "authorized_at": "2026-09-04T00:00:00Z",
            }) + "\n", encoding="utf-8")
            stage_o_document = {
                "authorization_status": "AUTHORIZED",
                "allowed_scope": "STAGE_O",
            }
            stage_o_path.write_text(
                json.dumps(stage_o_document) + "\n", encoding="utf-8"
            )
            stage_o_sha = sha256_file(stage_o_path)
            trainer_argv = governed_argv(
                stage_o_path,
                "--train_dir", "/tmp/path with spaces",
                "--adapter-looking-token", "--adapter-authorization-path",
            )
            calls = []
            parse_calls = []
            validation_calls = []
            gate_a_calls = []
            inner_gate_a_calls = []
            original_argv_object = sys.argv
            original_argv_values = list(sys.argv)
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            launch = types.ModuleType("fake_launch_for_delegation")
            launch.observed_training_runtime = lambda *, num_workers: {
                "num_workers": num_workers
            }
            gate_a_result = {
                "passed": True,
                "stage": "stage_b",
                "scope": "STAGE_O",
                "stage_authorization_path": str(stage_o_path.resolve()),
                "stage_authorization_sha256": stage_o_sha,
                "authorization": stage_o_document,
            }

            def original_gate_a(*args, **kwargs):
                inner_gate_a_calls.append((args, kwargs))
                return gate_a_result

            launch.gate_a_pre_construction = original_gate_a
            trainer = types.ModuleType("fake_trainer_for_delegation")

            def fake_parse_args():
                parse_calls.append(list(sys.argv))
                return governed_namespace(stage_o_path)

            def fake_main():
                calls.append(list(sys.argv))
                assert sys.stdout is original_stdout
                assert sys.stderr is original_stderr
                assert launch.gate_a_pre_construction is not original_gate_a
                assert launch.gate_a_pre_construction(object()) is gate_a_result
                print("TRAINER_STDOUT_SENTINEL", flush=True)
                print("TRAINER_STDERR_SENTINEL", file=sys.stderr, flush=True)
                return expected_result

            trainer.parse_args = fake_parse_args
            trainer.validate_training_args = lambda args: validation_calls.append(args)
            trainer.main = fake_main
            topology = types.SimpleNamespace(
                launch_contract=launch,
                trainer=trainer,
            )

            def fake_install(*, before_trainer=None):
                assert before_trainer is not None
                before_trainer(launch)
                return topology

            A.install_accepted_module_topology = fake_install
            A.validate_adapter_authorization = lambda *args, **kwargs: {
                "authorized": True,
                "failures": [],
            }

            def fake_silent_gate_a(installed, parsed_args, **kwargs):
                gate_a_calls.append(parsed_args)
                return gate_a_result

            A._run_silent_gate_a = fake_silent_gate_a
            A._validate_topology = lambda installed: None
            result = A.run_execution(adapter_path, stage_o_path, trainer_argv)
            assert result == (0 if expected_result is None else expected_result)
            assert len(parse_calls) == 1
            assert parse_calls[0][1:] == trainer_argv
            assert len(validation_calls) == 1
            assert len(gate_a_calls) == 1
            assert gate_a_calls[0] is validation_calls[0]
            assert len(inner_gate_a_calls) == 1
            assert launch.gate_a_pre_construction is original_gate_a
            assert len(calls) == 1
            assert calls[0][0] == str(A.CANONICAL_TRAINER_PATH)
            assert calls[0][1:] == trainer_argv
            assert sys.argv is original_argv_object
            assert sys.argv == original_argv_values
            emit({
                "delegation_calls": len(calls),
                "parser_calls": len(parse_calls),
                "validation_calls": len(validation_calls),
                "gate_a_calls": len(gate_a_calls),
                "inner_gate_a_calls": len(inner_gate_a_calls),
                "gate_a_restored": True,
                "exact_argv": True,
                "streams_same": True,
                "returned": result,
                "sys_argv_restored": True,
            })
        finally:
            temporary.cleanup()
        """,
        case=json.dumps(trainer_result),
        include_process_output=True,
    )
    assert observed == {
        "delegation_calls": 1,
        "exact_argv": True,
        "gate_a_restored": True,
        "gate_a_calls": 1,
        "inner_gate_a_calls": 1,
        "parser_calls": 1,
        "_process_stderr_lines": ["TRAINER_STDERR_SENTINEL"],
        "_process_stdout_lines": ["TRAINER_STDOUT_SENTINEL"],
        "returned": 0 if trainer_result is None else trainer_result,
        "streams_same": True,
        "sys_argv_restored": True,
        "validation_calls": 1,
    }


@pytest.mark.parametrize("case", ["system_exit", "runtime_error"])
def test_execution_propagates_failure_identity_and_never_retries(case):
    observed = _run_probe(
        r"""
        case = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage_o.json"
            adapter_path.write_text(json.dumps({
                "authorization_status": "AUTHORIZED",
                "authorizes_adapter_execution": True,
                "authorizes_training": False,
                "authorized_by": "test owner",
                "authorized_at": "2026-09-04T00:00:00Z",
            }) + "\n", encoding="utf-8")
            stage_o_document = {
                "authorization_status": "AUTHORIZED",
                "allowed_scope": "STAGE_O",
            }
            stage_o_path.write_text(
                json.dumps(stage_o_document) + "\n", encoding="utf-8"
            )
            stage_o_sha = sha256_file(stage_o_path)
            trainer_argv = governed_argv(
                stage_o_path, "--one", "unchanged", "--two=still-unchanged"
            )
            original_argv_object = sys.argv
            original_argv_values = list(sys.argv)
            calls = []
            sentinel = SystemExit(37) if case == "system_exit" else RuntimeError(
                "trainer failure sentinel"
            )

            launch = types.ModuleType("fake_launch_for_failure")
            launch.observed_training_runtime = lambda *, num_workers: {
                "num_workers": num_workers
            }
            inner_gate_a_calls = []
            gate_a_result = {
                "passed": True,
                "stage": "stage_b",
                "scope": "STAGE_O",
                "stage_authorization_path": str(stage_o_path.resolve()),
                "stage_authorization_sha256": stage_o_sha,
                "authorization": stage_o_document,
            }

            def original_gate_a(*args, **kwargs):
                inner_gate_a_calls.append((args, kwargs))
                return gate_a_result

            launch.gate_a_pre_construction = original_gate_a
            trainer = types.ModuleType("fake_trainer_for_failure")
            trainer.parse_args = lambda: governed_namespace(stage_o_path)
            trainer.validate_training_args = lambda args: None

            def fake_main():
                calls.append(list(sys.argv))
                assert launch.gate_a_pre_construction is not original_gate_a
                assert launch.gate_a_pre_construction(object()) is gate_a_result
                raise sentinel

            trainer.main = fake_main

            def fake_install(*, before_trainer=None):
                before_trainer(launch)
                return types.SimpleNamespace(
                    launch_contract=launch,
                    trainer=trainer,
                )

            A.install_accepted_module_topology = fake_install
            A.validate_adapter_authorization = lambda *args, **kwargs: {
                "authorized": True,
                "failures": [],
            }
            A._run_silent_gate_a = lambda *args, **kwargs: gate_a_result
            A._validate_topology = lambda installed: None
            try:
                A.run_execution(adapter_path, stage_o_path, trainer_argv)
            except BaseException as exc:
                assert exc is sentinel
            else:
                raise AssertionError("trainer failure was converted into success")
            assert len(calls) == 1
            assert len(inner_gate_a_calls) == 1
            assert launch.gate_a_pre_construction is original_gate_a
            assert calls[0][1:] == trainer_argv
            assert sys.argv is original_argv_object
            assert sys.argv == original_argv_values
            emit({
                "case": case,
                "delegation_calls": len(calls),
                "inner_gate_a_calls": len(inner_gate_a_calls),
                "gate_a_restored": True,
                "same_exception": True,
                "sys_argv_restored": True,
                "system_exit_code": sentinel.code if case == "system_exit" else None,
            })
        finally:
            temporary.cleanup()
        """,
        case=case,
    )
    assert observed == {
        "case": case,
        "delegation_calls": 1,
        "gate_a_restored": True,
        "inner_gate_a_calls": 1,
        "same_exception": True,
        "sys_argv_restored": True,
        "system_exit_code": 37 if case == "system_exit" else None,
    }


@pytest.mark.parametrize("mismatch", ["path", "sha256", "document"])
def test_execution_refuses_inner_gate_a_snapshot_swap_before_post_gate_work(mismatch):
    observed = _run_probe(
        r"""
        mismatch = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage_o.json"
            adapter_path.write_text("{}\n", encoding="utf-8")
            stage_o_document = {
                "authorization_status": "AUTHORIZED",
                "allowed_scope": "STAGE_O",
            }
            stage_o_path.write_text(
                json.dumps(stage_o_document) + "\n", encoding="utf-8"
            )
            stage_o_sha = sha256_file(stage_o_path)
            trainer_argv = governed_argv(stage_o_path)
            original_argv_object = sys.argv
            original_argv_values = list(sys.argv)
            events = []

            launch = types.ModuleType("fake_launch_for_inner_gate_a_swap")
            launch.observed_training_runtime = lambda *, num_workers: {
                "num_workers": num_workers
            }
            inner_result = {
                "passed": True,
                "stage": "stage_b",
                "scope": "STAGE_O",
                "stage_authorization_path": str(stage_o_path.resolve()),
                "stage_authorization_sha256": stage_o_sha,
                "authorization": stage_o_document,
            }
            if mismatch == "path":
                inner_result["stage_authorization_path"] = "/tmp/swapped-stage-o.json"
            elif mismatch == "sha256":
                inner_result["stage_authorization_sha256"] = "0" * 64
            else:
                inner_result["authorization"] = {
                    **stage_o_document,
                    "authorization_status": "SWAPPED",
                }

            def original_gate_a(*args, **kwargs):
                events.append("inner_gate_a")
                return inner_result

            launch.gate_a_pre_construction = original_gate_a
            trainer = types.ModuleType("fake_trainer_for_inner_gate_a_swap")
            trainer.parse_args = lambda: governed_namespace(stage_o_path)
            trainer.validate_training_args = lambda args: events.append("arg_validation")

            def fake_main():
                events.append("main_entered")
                launch.gate_a_pre_construction(object())
                events.append("post_gate_model_work")
                raise AssertionError("inner Gate-A mismatch reached post-Gate work")

            trainer.main = fake_main
            topology = types.SimpleNamespace(
                launch_contract=launch,
                trainer=trainer,
            )

            def fake_install(*, before_trainer=None):
                before_trainer(launch)
                return topology

            A.install_accepted_module_topology = fake_install
            A.validate_adapter_authorization = lambda *args, **kwargs: {
                "authorized": True,
                "failures": [],
            }
            A._run_silent_gate_a = lambda *args, **kwargs: events.append("silent_gate_a")
            A._validate_topology = lambda installed: None
            try:
                A.run_execution(adapter_path, stage_o_path, trainer_argv)
            except A.AdapterError as exc:
                error_message = str(exc)
            else:
                raise AssertionError("inner Gate-A snapshot swap was accepted")

            assert events == [
                "arg_validation",
                "silent_gate_a",
                "main_entered",
                "inner_gate_a",
            ]
            assert launch.gate_a_pre_construction is original_gate_a
            assert sys.argv is original_argv_object
            assert sys.argv == original_argv_values
            expected_fragment = {
                "path": "path changed",
                "sha256": "SHA changed",
                "document": "different Stage-O authorization document",
            }[mismatch]
            assert expected_fragment in error_message
            emit({
                "mismatch": mismatch,
                "main_calls": events.count("main_entered"),
                "inner_gate_a_calls": events.count("inner_gate_a"),
                "post_gate_model_work": False,
                "gate_a_restored": True,
                "sys_argv_restored": True,
            })
        finally:
            temporary.cleanup()
        """,
        case=mismatch,
    )
    assert observed == {
        "gate_a_restored": True,
        "inner_gate_a_calls": 1,
        "main_calls": 1,
        "mismatch": mismatch,
        "post_gate_model_work": False,
        "sys_argv_restored": True,
    }


def test_execution_refuses_stage_o_snapshot_drift_after_gate_a_before_main():
    observed = _run_probe(
        r"""
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage_o.json"
            adapter_path.write_text("{}\n", encoding="utf-8")
            stage_o_path.write_text("{}\n", encoding="utf-8")
            trainer_argv = governed_argv(stage_o_path)
            events = []
            original_argv_object = sys.argv
            original_argv_values = list(sys.argv)

            launch = types.ModuleType("fake_launch_for_snapshot_drift")
            launch.observed_training_runtime = lambda *, num_workers: {
                "num_workers": num_workers
            }
            trainer = types.ModuleType("fake_trainer_for_snapshot_drift")

            def fake_parse_args():
                events.append("parser")
                return governed_namespace(stage_o_path)

            def fake_main():
                events.append("main")
                raise AssertionError("trainer main must not run after authority drift")

            trainer.parse_args = fake_parse_args
            trainer.validate_training_args = lambda args: events.append("arg_validation")
            trainer.main = fake_main

            def fake_install(*, before_trainer=None):
                before_trainer(launch)
                events.append("trainer_import")
                return types.SimpleNamespace(trainer=trainer)

            def fake_gate_a(*args, **kwargs):
                events.append("gate_a")
                stage_o_path.write_text('{"changed_after_gate_a":true}\n', encoding="utf-8")
                return {"passed": True}

            A.install_accepted_module_topology = fake_install
            A.validate_adapter_authorization = lambda *args, **kwargs: {
                "authorized": True,
                "failures": [],
            }
            A._run_silent_gate_a = fake_gate_a
            A._validate_topology = lambda installed: None
            try:
                A.run_execution(adapter_path, stage_o_path, trainer_argv)
            except A.AdapterError as exc:
                assert "changed" in str(exc)
            else:
                raise AssertionError("Stage-O authorization drift reached trainer main")
            assert events == ["trainer_import", "parser", "arg_validation", "gate_a"]
            assert sys.argv is original_argv_object
            assert sys.argv == original_argv_values
            emit({
                "snapshot_drift_refused": True,
                "main_calls": 0,
                "events": events,
                "sys_argv_restored": True,
            })
        finally:
            temporary.cleanup()
        """
    )
    assert observed == {
        "events": ["trainer_import", "parser", "arg_validation", "gate_a"],
        "main_calls": 0,
        "snapshot_drift_refused": True,
        "sys_argv_restored": True,
    }


@pytest.mark.parametrize(
    "failure",
    [
        "adapter_authorization_status_not_authorized",
        "stage_o_authorization_status_not_authorized",
    ],
)
def test_execution_authority_refusal_happens_before_any_project_import(failure):
    observed = _run_probe(
        r"""
        failure = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage_o.json"
            adapter_path.write_text("{}\n", encoding="utf-8")
            stage_o_path.write_text("{}\n", encoding="utf-8")
            events = []
            launch = types.ModuleType("fake_launch_for_refusal")
            launch.observed_training_runtime = lambda *, num_workers: {
                "num_workers": num_workers
            }

            def fake_install(*, before_trainer=None):
                events.append("launch_topology")
                before_trainer(launch)
                events.append("trainer_import")
                raise AssertionError("trainer import must not be reached")

            A.install_accepted_module_topology = fake_install
            A.validate_adapter_authorization = lambda *args, **kwargs: {
                "authorized": False,
                "failures": [failure],
            }
            try:
                A.run_execution(
                    adapter_path,
                    stage_o_path,
                    governed_argv(stage_o_path, "--never", "delegated"),
                )
            except A.AdapterError as exc:
                assert failure in str(exc)
            else:
                raise AssertionError("unauthorized execution was accepted")
            assert events == []
            emit({
                "failure": failure,
                "project_imported": False,
                "trainer_imported": False,
                "delegated": False,
            })
        finally:
            temporary.cleanup()
        """,
        case=failure,
    )
    assert observed == {
        "delegated": False,
        "failure": failure,
        "project_imported": False,
        "trainer_imported": False,
    }


def test_malformed_governed_argv_is_refused_before_artifact_or_project_import():
    observed = _run_probe(
        r"""
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            expected_stage_o = Path(temporary.name) / "stage-o.json"
            adapter_path.write_text("{}\n", encoding="utf-8")
            expected_stage_o.write_text("{}\n", encoding="utf-8")
            valid = governed_argv(expected_stage_o)
            invalid_cases = {
                "missing_launch_contract": valid[2:],
                "missing_stage_authorization": valid[:2] + valid[4:],
                "missing_stage_selector": valid[:4],
                "wrong_stage_authorization": governed_argv(
                    Path(temporary.name) / "different-stage-o.json"
                ),
                "wrong_stage": valid[:-1] + ["stage_a"],
                "duplicate_launch_contract": valid + [
                    "--launch_contract_json", "/tmp/second-launch-contract.json"
                ],
            }
            events = []

            def forbidden_snapshot(*args, **kwargs):
                events.append("artifact_read")
                raise AssertionError("artifact read must not be reached")

            def forbidden_install(*args, **kwargs):
                events.append("project_import")
                raise AssertionError("project import must not be reached")

            A._load_json_snapshot = forbidden_snapshot
            A.install_accepted_module_topology = forbidden_install
            checked = []
            for runner_name in ("run_preflight", "run_execution"):
                runner = getattr(A, runner_name)
                for case, argv in invalid_cases.items():
                    try:
                        runner(adapter_path, expected_stage_o, argv)
                    except A.AdapterError:
                        pass
                    else:
                        raise AssertionError(
                            f"{runner_name} accepted malformed argv:{case}"
                        )
                    assert events == []
                    checked.append(f"{runner_name}:{case}")
            emit({
                "checked": checked,
                "artifact_read": False,
                "project_imported": False,
            })
        finally:
            temporary.cleanup()
        """
    )
    assert observed["artifact_read"] is False
    assert observed["project_imported"] is False
    assert len(observed["checked"]) == 12


def test_cli_split_preserves_trainer_terminator_and_governed_guard_uses_prefix_only():
    observed = _run_probe(
        r"""
        temporary = tempfile.TemporaryDirectory()
        try:
            stage_o_path = Path(temporary.name) / "stage-o.json"
            stage_o_path.write_text("{}\n", encoding="utf-8")
            governed_prefix = governed_argv(stage_o_path)
            literal_tail = [
                "--",
                "literal-positional",
                "--run_plan_stage",
                "stage_a",
                "--stage_authorization_json",
                "/tmp/not-authority-after-terminator.json",
            ]
            adapter_prefix = [
                "preflight",
                "--adapter-authorization-path",
                "/tmp/adapter.json",
                "--stage-o-authorization-path",
                str(stage_o_path),
            ]
            adapter_argv, trainer_argv = A._split_adapter_and_trainer_argv([
                *adapter_prefix,
                "--",
                *governed_prefix,
                *literal_tail,
            ])
            assert adapter_argv == adapter_prefix
            assert trainer_argv == [*governed_prefix, *literal_tail]
            governed = A.validate_governed_trainer_argv(trainer_argv, stage_o_path)
            assert governed == {
                "launch_contract_json": "/tmp/reviewed-launch-contract.json",
                "stage_authorization_json": str(stage_o_path.resolve()),
                "run_plan_stage": "stage_b",
            }

            _, only_after_terminator = A._split_adapter_and_trainer_argv([
                *adapter_prefix,
                "--",
                "--",
                *governed_prefix,
            ])
            assert only_after_terminator == ["--", *governed_prefix]
            try:
                A.validate_governed_trainer_argv(
                    only_after_terminator,
                    stage_o_path,
                )
            except A.AdapterError as exc:
                assert "required" in str(exc)
            else:
                raise AssertionError("governed flags after terminator were accepted")

            emit({
                "adapter_split_exact": True,
                "trainer_terminator_preserved": True,
                "prefix_governed": True,
                "post_terminator_governed_refused": True,
            })
        finally:
            temporary.cleanup()
        """
    )
    assert observed == {
        "adapter_split_exact": True,
        "post_terminator_governed_refused": True,
        "prefix_governed": True,
        "trainer_terminator_preserved": True,
    }


@pytest.mark.parametrize("runner_name", ["run_preflight", "run_execution"])
def test_stage_o_authorization_symlink_is_refused_before_artifact_or_project_import(
    runner_name,
):
    observed = _run_probe(
        r"""
        runner_name = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            real_path = Path(temporary.name) / "real-stage-o.json"
            alias_path = Path(temporary.name) / "symlink-stage-o.json"
            adapter_path.write_text("{}\n", encoding="utf-8")
            real_path.write_text("{}\n", encoding="utf-8")
            alias_path.symlink_to(real_path)
            events = []

            def forbidden_snapshot(*args, **kwargs):
                events.append("artifact_read")
                raise AssertionError("artifact read must not be reached")

            def forbidden_install(*args, **kwargs):
                events.append("project_import")
                raise AssertionError("project import must not be reached")

            A._load_json_snapshot = forbidden_snapshot
            A.install_accepted_module_topology = forbidden_install
            runner = getattr(A, runner_name)
            try:
                runner(
                    adapter_path,
                    alias_path,
                    governed_argv(alias_path),
                )
            except A.AdapterError as exc:
                assert "symlink" in str(exc).lower() or "canonical" in str(exc).lower()
            else:
                raise AssertionError("Stage-O authorization symlink was accepted")
            assert events == []
            emit({
                "runner": runner_name,
                "symlink_refused": True,
                "artifact_read": False,
                "project_imported": False,
            })
        finally:
            temporary.cleanup()
        """,
        case=runner_name,
    )
    assert observed == {
        "artifact_read": False,
        "project_imported": False,
        "runner": runner_name,
        "symlink_refused": True,
    }


@pytest.mark.parametrize("missing_owner_field", ["authorized_by", "authorized_at"])
def test_authorized_status_with_missing_owner_field_is_refused_before_topology(
    missing_owner_field,
):
    observed = _run_probe(
        r"""
        missing_owner_field = os.environ["PETITGPT_ADAPTER_PROBE_CASE"]
        temporary = tempfile.TemporaryDirectory()
        try:
            adapter_path = Path(temporary.name) / "adapter.json"
            stage_o_path = Path(temporary.name) / "stage_o.json"
            trainer_argv = governed_argv(stage_o_path)
            accepted_identity = {
                "worktree_path": str(ACCEPTED_ROOT),
                "branch": A.ACCEPTED_SUCCESSOR_BRANCH,
                "head": A.ACCEPTED_SUCCESSOR_HEAD,
                "trainer_execution_bundle_sha256": A.ACCEPTED_TRAINER_BUNDLE_SHA256,
            }
            adapter_identity = {
                "worktree_path": str(ADAPTER_PATH.parents[1]),
                "branch": A.EXPECTED_ADAPTER_BRANCH,
                "head": "a" * 40,
                "adapter_tool_path": str(ADAPTER_PATH),
                "adapter_tool_sha256": "b" * 64,
                "adapter_tool_closure_count": 1,
                "adapter_tool_unbound_module_count": 0,
                "adapter_tool_bundle_sha256": "c" * 64,
                "tracked_clean": True,
                "script_tracked": True,
            }
            A.accepted_trainer_identity = lambda: dict(accepted_identity)
            A.adapter_identity = lambda: dict(adapter_identity)
            stage_o_document = {
                "authorization_status": "AUTHORIZED",
                "allowed_scope": "STAGE_O",
                "stage_o_launch_adapter_identity": A._expected_stage_o_adapter_identity(
                    adapter_identity
                ),
                "stage_o_trainer_argv": list(trainer_argv),
            }
            stage_o_path.write_bytes(A.canonical_json_bytes(stage_o_document))
            runtime = {"num_workers": 2, "canonical_cwd": str(HISTORICAL_ROOT)}
            adapter_document = A.adapter_authorization_template(
                runtime_fingerprint=runtime,
                stage_o_authorization_path=stage_o_path,
                stage_o_authorization_sha256=A.file_sha256(stage_o_path),
            )
            adapter_document.update({
                "authorization_status": "AUTHORIZED",
                "authorizes_adapter_execution": True,
                "authorized_by": "Stage-O owner",
                "authorized_at": "2026-09-04T00:00:00Z",
            })
            adapter_document[missing_owner_field] = None
            adapter_path.write_bytes(A.canonical_json_bytes(adapter_document))

            events = []
            A.install_accepted_module_topology = lambda **kwargs: events.append(
                "project_topology"
            )
            try:
                A.run_execution(adapter_path, stage_o_path, trainer_argv)
            except A.AdapterError as exc:
                assert f"adapter_owner_field_missing:{missing_owner_field}" in str(exc)
            else:
                raise AssertionError("AUTHORIZED status with missing owner identity was accepted")
            assert events == []
            emit({
                "missing_owner_field": missing_owner_field,
                "refused_before_topology": True,
                "trainer_main_called": False,
            })
        finally:
            temporary.cleanup()
        """,
        case=missing_owner_field,
    )
    assert observed == {
        "missing_owner_field": missing_owner_field,
        "refused_before_topology": True,
        "trainer_main_called": False,
    }
