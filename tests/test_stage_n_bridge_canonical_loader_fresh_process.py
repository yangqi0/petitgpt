"""Fresh-process matrix for the N3 bridge's canonical reviewed-module loader.

These tests deliberately retain references acquired before the bridge import.  Looking only at
the final ``sys.modules`` mapping is insufficient: replacing a canonical entry can make that
mapping look correct while the displaced module and its class family remain reachable.

Every behavioral case runs under a new interpreter with bytecode writes disabled.  Nothing in
this file invokes N3, constructs the production model/optimizer, realizes compile, or executes a
model forward.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

SUCCESSOR_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = Path("/workspace/petitgpt")
PROJECT_PYTHON = HISTORICAL_ROOT / ".venv/bin/python"

EXPECTED_SOURCE_SHA256 = {
    "pretrain/production_launch_contract_v1.py": (
        "9e858078e7e492bed6de3b3ce34395d44fb81f3f06aab59c9960d447b7bde861"
    ),
    "src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/model.py": "2bc9fa8ae16636837c4a2937301a2419d0ac92faa2cc27560dacbd29a5144dc2",
    "src/optim.py": "13116860174f8557e6ab5a9b21011ecc15dfa0b82e0e6e394fff3554935e264a",
}

_PROBE_PREAMBLE = r"""
import builtins
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(os.environ["PETITGPT_SUCCESSOR_ROOT"]).resolve()
HISTORICAL = Path(os.environ["PETITGPT_HISTORICAL_ROOT"]).resolve()
EXPECTED_SHA256 = json.loads(os.environ["PETITGPT_EXPECTED_SOURCE_SHA256"])
RESULT_PREFIX = "PETITGPT_R3_PROBE_RESULT="


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed_path(relative_path):
    return (ROOT / relative_path).resolve()


def assert_reviewed(module, relative_path):
    expected_path = reviewed_path(relative_path)
    assert Path(module.__file__).resolve() == expected_path
    assert sha256_file(expected_path) == EXPECTED_SHA256[relative_path]


def exact_path_module_objects(relative_path):
    expected_path = reviewed_path(relative_path)
    found = {}
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            actual_path = Path(str(module_file)).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if actual_path == expected_path:
            found[id(module)] = module
    return found


def assert_canonical_binding(module, canonical_name, *private_aliases):
    assert sys.modules[canonical_name] is module
    for private_alias in private_aliases:
        assert sys.modules[private_alias] is module
    parent_name, _, child_name = canonical_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    assert parent.__dict__[child_name] is module
    assert importlib.import_module(canonical_name) is module


def emit(payload):
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
"""


def _run_probe(
    body: str,
    *,
    cwd: Path = SUCCESSOR_ROOT,
    case: str | None = None,
) -> dict:
    assert PROJECT_PYTHON.is_file(), f"project interpreter not found: {PROJECT_PYTHON}"
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(SUCCESSOR_ROOT),
        "PETITGPT_SUCCESSOR_ROOT": str(SUCCESSOR_ROOT),
        "PETITGPT_HISTORICAL_ROOT": str(HISTORICAL_ROOT),
        "PETITGPT_EXPECTED_SOURCE_SHA256": json.dumps(EXPECTED_SOURCE_SHA256, sort_keys=True),
    })
    if case is not None:
        environment["PETITGPT_R3_PROBE_CASE"] = case
    completed = subprocess.run(
        [str(PROJECT_PYTHON), "-B", "-c", _PROBE_PREAMBLE + textwrap.dedent(body)],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"fresh-process probe failed with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("PETITGPT_R3_PROBE_RESULT=")
    ]
    assert len(result_lines) == 1, (
        f"fresh-process probe did not emit one result\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return json.loads(result_lines[0].split("=", 1)[1])


@pytest.mark.parametrize("relative_path, expected_sha256", sorted(EXPECTED_SOURCE_SHA256.items()))
def test_reviewed_source_sha256_constants_are_exact(relative_path, expected_sha256):
    path = SUCCESSOR_ROOT / relative_path
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_fresh_bridge_first_binds_one_reviewed_object_for_all_six_types():
    observed = _run_probe(
        r"""
        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        assert Path(B.__file__).resolve() == reviewed_path(
            "pretrain/stage_n_successor_head_compatibility_bridge_v1.py"
        )

        optim = B._load_exact_successor_module(
            "_petitgpt_successor_src_optim", "src/optim.py"
        )
        model = B._load_exact_successor_module(
            "_petitgpt_successor_src_model", "src/model.py"
        )
        launch = B.launch

        canonical_optim = importlib.import_module("src.optim")
        canonical_model = importlib.import_module("src.model")
        canonical_launch = importlib.import_module(
            "pretrain.production_launch_contract_v1"
        )
        assert optim is canonical_optim
        assert model is canonical_model
        assert launch is canonical_launch

        assert_canonical_binding(
            optim, "src.optim", "_petitgpt_successor_src_optim"
        )
        assert_canonical_binding(
            model, "src.model", "_petitgpt_successor_src_model"
        )
        assert_canonical_binding(
            launch,
            "pretrain.production_launch_contract_v1",
            "_petitgpt_successor_production_launch_contract_v1",
        )

        assert optim.Muon is canonical_optim.Muon
        assert model.GPT is canonical_model.GPT
        assert model.GPTConfig is canonical_model.GPTConfig
        assert launch.LaunchContractError is canonical_launch.LaunchContractError
        assert launch._Missing is canonical_launch._Missing
        assert launch.ObservedForward is canonical_launch.ObservedForward
        assert isinstance(launch._MISSING, launch._Missing)

        for module, names in (
            (optim, ("Muon",)),
            (model, ("GPT", "GPTConfig")),
            (launch, ("LaunchContractError", "_Missing", "ObservedForward")),
        ):
            for name in names:
                class_object = getattr(module, name)
                assert sys.modules[class_object.__module__] is module

        for relative_path in (
            "src/optim.py",
            "src/model.py",
            "pretrain/production_launch_contract_v1.py",
        ):
            assert len(exact_path_module_objects(relative_path)) == 1

        assert_reviewed(optim, "src/optim.py")
        assert_reviewed(model, "src/model.py")
        assert_reviewed(launch, "pretrain/production_launch_contract_v1.py")
        emit(
            {
                "six_type_identity": True,
                "optim_sha256": sha256_file(optim.__file__),
                "model_sha256": sha256_file(model.__file__),
                "launch_sha256": sha256_file(launch.__file__),
            }
        )
        """,
        cwd=HISTORICAL_ROOT,
    )
    assert observed == {
        "launch_sha256": EXPECTED_SOURCE_SHA256["pretrain/production_launch_contract_v1.py"],
        "model_sha256": EXPECTED_SOURCE_SHA256["src/model.py"],
        "optim_sha256": EXPECTED_SOURCE_SHA256["src/optim.py"],
        "six_type_identity": True,
    }


@pytest.mark.parametrize("case", ["optim", "model", "launch"])
def test_fresh_exact_canonical_first_reuses_retained_module_and_classes(case):
    observed = _run_probe(
        r"""
        case = os.environ["PETITGPT_R3_PROBE_CASE"]
        if case == "optim":
            canonical_name = "src.optim"
            relative_path = "src/optim.py"
            private_alias = "_petitgpt_successor_src_optim"
            retained_module = importlib.import_module(canonical_name)
            retained_classes = (retained_module.Muon,)
        elif case == "model":
            canonical_name = "src.model"
            relative_path = "src/model.py"
            private_alias = "_petitgpt_successor_src_model"
            retained_module = importlib.import_module(canonical_name)
            retained_classes = (retained_module.GPT, retained_module.GPTConfig)
        else:
            canonical_name = "pretrain.production_launch_contract_v1"
            relative_path = "pretrain/production_launch_contract_v1.py"
            private_alias = "_petitgpt_successor_production_launch_contract_v1"
            retained_module = importlib.import_module(canonical_name)
            retained_classes = (
                retained_module.LaunchContractError,
                retained_module._Missing,
                retained_module.ObservedForward,
            )

        retained_marker = object()
        retained_module._petitgpt_r3_retained_marker = retained_marker
        retained_module_id = id(retained_module)

        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        if case == "launch":
            loaded = B.launch
            loaded_classes = (
                loaded.LaunchContractError,
                loaded._Missing,
                loaded.ObservedForward,
            )
        else:
            loaded = B._load_exact_successor_module(private_alias, relative_path)
            loaded_classes = (
                (loaded.Muon,)
                if case == "optim"
                else (loaded.GPT, loaded.GPTConfig)
            )

        assert loaded is retained_module
        assert id(loaded) == retained_module_id
        assert loaded_classes == retained_classes
        assert loaded._petitgpt_r3_retained_marker is retained_marker
        assert_canonical_binding(loaded, canonical_name, private_alias)
        assert len(exact_path_module_objects(relative_path)) == 1
        assert_reviewed(loaded, relative_path)
        emit(
            {
                "case": case,
                "retained_module_reused": True,
                "retained_classes_reused": True,
                "module_file": str(Path(loaded.__file__).resolve()),
            }
        )
        """,
        case=case,
    )
    relative_path = {
        "optim": "src/optim.py",
        "model": "src/model.py",
        "launch": "pretrain/production_launch_contract_v1.py",
    }[case]
    assert observed == {
        "case": case,
        "module_file": str((SUCCESSOR_ROOT / relative_path).resolve()),
        "retained_classes_reused": True,
        "retained_module_reused": True,
    }


def test_fresh_parent_packages_first_receive_the_single_canonical_children():
    observed = _run_probe(
        r"""
        src_parent = importlib.import_module("src")
        pretrain_parent = importlib.import_module("pretrain")
        assert "optim" not in src_parent.__dict__
        assert "model" not in src_parent.__dict__
        assert "production_launch_contract_v1" not in pretrain_parent.__dict__

        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        optim = B._load_exact_successor_module(
            "_petitgpt_successor_src_optim", "src/optim.py"
        )
        model = B._load_exact_successor_module(
            "_petitgpt_successor_src_model", "src/model.py"
        )

        assert src_parent.__dict__["optim"] is optim
        assert src_parent.__dict__["model"] is model
        assert pretrain_parent.__dict__["production_launch_contract_v1"] is B.launch
        assert importlib.import_module("src.optim") is optim
        assert importlib.import_module("src.model") is model
        assert (
            importlib.import_module("pretrain.production_launch_contract_v1")
            is B.launch
        )
        assert Path(src_parent.__file__).resolve() == reviewed_path("src/__init__.py")
        emit({"parent_first": True, "normal_reimports_same": True})
        """
    )
    assert observed == {"normal_reimports_same": True, "parent_first": True}


def test_fresh_repeated_bridge_and_different_private_aliases_never_reexecute_sources():
    observed = _run_probe(
        r"""
        bridge_name = "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        B1 = importlib.import_module(bridge_name)
        launch = B1.launch
        launch_classes = (
            launch.LaunchContractError,
            launch._Missing,
            launch.ObservedForward,
        )
        B2 = importlib.import_module(bridge_name)
        assert B2 is B1
        assert B2.launch is launch

        optim_a = B1._load_exact_successor_module(
            "_petitgpt_r3_src_optim_alias_a", "src/optim.py"
        )
        retained_muon = optim_a.Muon
        optim_repeat = B1._load_exact_successor_module(
            "_petitgpt_r3_src_optim_alias_a", "src/optim.py"
        )
        optim_b = B1._load_exact_successor_module(
            "_petitgpt_r3_src_optim_alias_b", "src/optim.py"
        )
        assert optim_repeat is optim_a
        assert optim_b is optim_a
        assert optim_b.Muon is retained_muon

        model_a = B1._load_exact_successor_module(
            "_petitgpt_r3_src_model_alias_a", "src/model.py"
        )
        retained_model_classes = (model_a.GPT, model_a.GPTConfig)
        model_b = B1._load_exact_successor_module(
            "_petitgpt_r3_src_model_alias_b", "src/model.py"
        )
        assert model_b is model_a
        assert (model_b.GPT, model_b.GPTConfig) == retained_model_classes

        second_bridge_name = "_petitgpt_r3_second_bridge_import"
        spec = importlib.util.spec_from_file_location(second_bridge_name, B1.__file__)
        assert spec is not None and spec.loader is not None
        B3 = importlib.util.module_from_spec(spec)
        sys.modules[second_bridge_name] = B3
        spec.loader.exec_module(B3)
        assert B3.launch is launch
        assert (
            B3.launch.LaunchContractError,
            B3.launch._Missing,
            B3.launch.ObservedForward,
        ) == launch_classes
        assert (
            B3._load_exact_successor_module(
                "_petitgpt_r3_src_optim_alias_c", "src/optim.py"
            )
            is optim_a
        )
        assert (
            B3._load_exact_successor_module(
                "_petitgpt_r3_src_model_alias_c", "src/model.py"
            )
            is model_a
        )

        assert len(exact_path_module_objects("src/optim.py")) == 1
        assert len(exact_path_module_objects("src/model.py")) == 1
        assert len(
            exact_path_module_objects("pretrain/production_launch_contract_v1.py")
        ) == 1
        assert_canonical_binding(
            optim_a,
            "src.optim",
            "_petitgpt_r3_src_optim_alias_a",
            "_petitgpt_r3_src_optim_alias_b",
            "_petitgpt_r3_src_optim_alias_c",
        )
        assert_canonical_binding(
            model_a,
            "src.model",
            "_petitgpt_r3_src_model_alias_a",
            "_petitgpt_r3_src_model_alias_b",
            "_petitgpt_r3_src_model_alias_c",
        )
        emit(
            {
                "normal_bridge_repeat": True,
                "second_bridge_execution_reused_types": True,
                "different_private_aliases_reused": True,
            }
        )
        """
    )
    assert observed == {
        "different_private_aliases_reused": True,
        "normal_bridge_repeat": True,
        "second_bridge_execution_reused_types": True,
    }


@pytest.mark.parametrize(
    "case",
    [
        "optim_different_bytes",
        "model_different_bytes",
        "optim_same_bytes_wrong_path",
        "optim_no_metadata",
        "optim_none_partial",
    ],
)
def test_fresh_invalid_canonical_prebind_fails_without_leak_then_clean_retry_succeeds(case):
    observed = _run_probe(
        r"""
        case = os.environ["PETITGPT_R3_PROBE_CASE"]
        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        if case == "model_different_bytes":
            canonical_name = "src.model"
            relative_path = "src/model.py"
            private_alias = "_petitgpt_successor_src_model"
            class_name = "GPT"
            fake_source = "class GPT: pass\nclass GPTConfig: pass\n"
        else:
            canonical_name = "src.optim"
            relative_path = "src/optim.py"
            private_alias = "_petitgpt_successor_src_optim"
            class_name = "Muon"
            fake_source = "class Muon: pass\n"

        parent_name, _, child_name = canonical_name.rpartition(".")
        parent = importlib.import_module(parent_name)
        assert canonical_name not in sys.modules
        assert child_name not in parent.__dict__
        temporary = tempfile.TemporaryDirectory()
        retained_class = None

        if case == "optim_none_partial":
            foreign = None
        elif case == "optim_no_metadata":
            foreign = types.ModuleType(canonical_name)
            exec(fake_source, foreign.__dict__)
            retained_class = getattr(foreign, class_name)
        else:
            foreign_path = Path(temporary.name) / f"{child_name}.py"
            if case == "optim_same_bytes_wrong_path":
                foreign_path.write_bytes(reviewed_path(relative_path).read_bytes())
            else:
                foreign_path.write_text(fake_source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location(canonical_name, foreign_path)
            assert spec is not None and spec.loader is not None
            foreign = importlib.util.module_from_spec(spec)
            sys.modules[canonical_name] = foreign
            spec.loader.exec_module(foreign)
            retained_class = getattr(foreign, class_name)

        sys.modules[canonical_name] = foreign
        parent.__dict__[child_name] = foreign
        failure_type = None
        controlled_failure = False
        try:
            B._load_exact_successor_module(private_alias, relative_path)
        except BaseException as exc:
            failure_type = type(exc).__name__
            controlled_failure = isinstance(exc, B.CompatibilityBridgeError)

        canonical_preserved = (
            canonical_name in sys.modules and sys.modules[canonical_name] is foreign
        )
        parent_preserved = (
            child_name in parent.__dict__ and parent.__dict__[child_name] is foreign
        )
        leaked_aliases = [name for name in (private_alias,) if name in sys.modules]

        if retained_class is not None:
            assert getattr(foreign, class_name) is retained_class

        # Remove only the deliberate fixture/failed-attempt bindings, then prove that the
        # same process can perform a clean retry. A correct loader leaves nothing else to clear.
        sys.modules.pop(canonical_name, None)
        sys.modules.pop(private_alias, None)
        parent.__dict__.pop(child_name, None)
        retried = B._load_exact_successor_module(private_alias, relative_path)

        assert failure_type == "CompatibilityBridgeError"
        assert controlled_failure
        assert canonical_preserved
        assert parent_preserved
        assert not leaked_aliases
        assert retried is sys.modules[canonical_name]
        assert retried is sys.modules[private_alias]
        assert parent.__dict__[child_name] is retried
        assert_reviewed(retried, relative_path)
        assert len(exact_path_module_objects(relative_path)) == 1
        if retained_class is not None:
            assert getattr(retried, class_name) is not retained_class

        temporary.cleanup()
        emit(
            {
                "case": case,
                "controlled_failure": controlled_failure,
                "canonical_preserved": canonical_preserved,
                "parent_preserved": parent_preserved,
                "private_alias_leak_count": len(leaked_aliases),
                "clean_retry_succeeded": True,
            }
        )
        """,
        case=case,
    )
    assert observed == {
        "canonical_preserved": True,
        "case": case,
        "clean_retry_succeeded": True,
        "controlled_failure": True,
        "parent_preserved": True,
        "private_alias_leak_count": 0,
    }


def test_fresh_historical_launch_prebind_fails_closed_then_clean_retry_uses_successor():
    observed = _run_probe(
        r"""
        canonical_name = "pretrain.production_launch_contract_v1"
        private_alias = "_petitgpt_successor_production_launch_contract_v1"
        bridge_name = "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        historical_launch = importlib.import_module(canonical_name)
        parent = importlib.import_module("pretrain")
        assert Path(historical_launch.__file__).resolve() == (
            HISTORICAL / "pretrain/production_launch_contract_v1.py"
        ).resolve()
        retained_error = historical_launch.LaunchContractError
        retained_missing = historical_launch._Missing
        retained_observed = historical_launch.ObservedForward

        failure_type = None
        try:
            importlib.import_module(bridge_name)
        except BaseException as exc:
            failure_type = type(exc).__name__

        canonical_preserved = sys.modules.get(canonical_name) is historical_launch
        parent_preserved = (
            parent.__dict__.get("production_launch_contract_v1") is historical_launch
        )
        private_alias_absent = private_alias not in sys.modules
        retained_classes_preserved = (
            historical_launch.LaunchContractError is retained_error
            and historical_launch._Missing is retained_missing
            and historical_launch.ObservedForward is retained_observed
        )

        # Remove the deliberate conflicting fixture and retry in the same process. The failed
        # bridge import must not have left a private launch family that contaminates this retry.
        sys.modules.pop(bridge_name, None)
        sys.modules.pop(canonical_name, None)
        sys.modules.pop(private_alias, None)
        parent.__dict__.pop("production_launch_contract_v1", None)
        B = importlib.import_module(bridge_name)
        successor_launch = B.launch

        assert failure_type == "CompatibilityBridgeError"
        assert canonical_preserved
        assert parent_preserved
        assert private_alias_absent
        assert retained_classes_preserved
        assert_reviewed(successor_launch, "pretrain/production_launch_contract_v1.py")
        assert_canonical_binding(successor_launch, canonical_name, private_alias)
        assert successor_launch.LaunchContractError is not retained_error
        assert successor_launch._Missing is not retained_missing
        assert successor_launch.ObservedForward is not retained_observed

        historical_error_caught_as_successor = False
        try:
            raise retained_error("historical")
        except successor_launch.LaunchContractError:
            historical_error_caught_as_successor = True
        except RuntimeError:
            pass
        assert not historical_error_caught_as_successor

        successor_error_caught = False
        try:
            raise successor_launch.LaunchContractError("successor")
        except successor_launch.LaunchContractError:
            successor_error_caught = True
        assert successor_error_caught
        emit(
            {
                "failure_type": failure_type,
                "canonical_preserved": canonical_preserved,
                "parent_preserved": parent_preserved,
                "private_alias_absent": private_alias_absent,
                "retained_classes_preserved": retained_classes_preserved,
                "clean_retry_succeeded": True,
                "historical_error_caught_as_successor": (
                    historical_error_caught_as_successor
                ),
                "successor_error_caught": successor_error_caught,
            }
        )
        """,
        cwd=HISTORICAL_ROOT,
    )
    assert observed == {
        "canonical_preserved": True,
        "clean_retry_succeeded": True,
        "failure_type": "CompatibilityBridgeError",
        "historical_error_caught_as_successor": False,
        "parent_preserved": True,
        "private_alias_absent": True,
        "retained_classes_preserved": True,
        "successor_error_caught": True,
    }


def test_fresh_orphan_exact_path_alias_is_preserved_but_not_adopted():
    observed = _run_probe(
        r"""
        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        canonical_name = "src.optim"
        fixed_alias = "_petitgpt_successor_src_optim"
        requested_alias = "_petitgpt_r3_orphan_src_optim_request"
        orphan_alias = "_petitgpt_r3_preexisting_orphan_src_optim"
        parent = importlib.import_module("src")
        source = reviewed_path("src/optim.py")
        spec = importlib.util.spec_from_file_location(orphan_alias, source)
        assert spec is not None and spec.loader is not None
        orphan = importlib.util.module_from_spec(spec)
        sys.modules[orphan_alias] = orphan
        spec.loader.exec_module(orphan)
        retained_muon = orphan.Muon

        failure_type = None
        try:
            B._load_exact_successor_module(requested_alias, "src/optim.py")
        except BaseException as exc:
            failure_type = type(exc).__name__

        assert failure_type == "CompatibilityBridgeError"
        assert sys.modules[orphan_alias] is orphan
        assert orphan.Muon is retained_muon
        assert canonical_name not in sys.modules
        assert fixed_alias not in sys.modules
        assert requested_alias not in sys.modules
        assert "optim" not in parent.__dict__

        emit(
            {
                "controlled_failure": True,
                "orphan_preserved": True,
                "orphan_not_adopted": True,
                "failed_aliases_absent": True,
            }
        )
        """
    )
    assert observed == {
        "controlled_failure": True,
        "failed_aliases_absent": True,
        "orphan_not_adopted": True,
        "orphan_preserved": True,
    }


def test_fresh_reviewed_sha_mismatch_leaves_no_bindings_then_retry_succeeds():
    observed = _run_probe(
        r"""
        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        canonical_name = "src.optim"
        fixed_alias = "_petitgpt_successor_src_optim"
        requested_alias = "_petitgpt_r3_sha_retry_src_optim"
        parent = importlib.import_module("src")
        reviewed_record = B._REVIEWED_SUCCESSOR_MODULES[canonical_name]
        B._REVIEWED_SUCCESSOR_MODULES[canonical_name] = (
            reviewed_record[0],
            "0" * 64,
            reviewed_record[2],
        )
        failure_type = None
        try:
            B._load_exact_successor_module(requested_alias, "src/optim.py")
        except BaseException as exc:
            failure_type = type(exc).__name__
        finally:
            B._REVIEWED_SUCCESSOR_MODULES[canonical_name] = reviewed_record

        assert failure_type == "CompatibilityBridgeError"
        assert canonical_name not in sys.modules
        assert fixed_alias not in sys.modules
        assert requested_alias not in sys.modules
        assert "optim" not in parent.__dict__
        retried = B._load_exact_successor_module(requested_alias, "src/optim.py")
        assert_canonical_binding(
            retried,
            canonical_name,
            fixed_alias,
            requested_alias,
        )
        emit(
            {
                "controlled_failure": True,
                "canonical_absent": True,
                "fixed_alias_absent": True,
                "requested_alias_absent": True,
                "parent_restored": True,
                "clean_retry_succeeded": True,
            }
        )
        """
    )
    assert observed == {
        "canonical_absent": True,
        "clean_retry_succeeded": True,
        "controlled_failure": True,
        "fixed_alias_absent": True,
        "parent_restored": True,
        "requested_alias_absent": True,
    }


def test_fresh_concurrent_private_requests_serialize_to_one_canonical_object():
    observed = _run_probe(
        r"""
        import threading

        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        aliases = (
            "_petitgpt_r3_concurrent_src_optim_a",
            "_petitgpt_r3_concurrent_src_optim_b",
        )
        barrier = threading.Barrier(len(aliases))
        results = []
        failures = []

        def load(alias):
            try:
                barrier.wait(timeout=30)
                results.append(
                    B._load_exact_successor_module(alias, "src/optim.py")
                )
            except BaseException as exc:
                failures.append(f"{type(exc).__name__}:{exc}")

        threads = [threading.Thread(target=load, args=(alias,)) for alias in aliases]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=90)
        assert all(not thread.is_alive() for thread in threads)
        assert failures == []
        assert len(results) == 2
        assert results[0] is results[1]
        module = results[0]
        assert_canonical_binding(
            module,
            "src.optim",
            "_petitgpt_successor_src_optim",
            *aliases,
        )
        assert len(exact_path_module_objects("src/optim.py")) == 1
        emit(
            {
                "thread_count": len(threads),
                "failure_count": len(failures),
                "same_module_object": True,
                "one_exact_path_object": True,
            }
        )
        """
    )
    assert observed == {
        "failure_count": 0,
        "one_exact_path_object": True,
        "same_module_object": True,
        "thread_count": 2,
    }


def test_fresh_cross_owner_reserved_alias_is_rejected_without_mutation():
    observed = _run_probe(
        r"""
        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        launch = B.launch
        launch_canonical = "pretrain.production_launch_contract_v1"
        launch_private = "_petitgpt_successor_production_launch_contract_v1"
        model_private = "_petitgpt_successor_src_model"
        assert model_private not in sys.modules
        failure_type = None
        try:
            B._load_reviewed_successor_module(
                launch_canonical,
                requested_private_name=model_private,
            )
        except BaseException as exc:
            failure_type = type(exc).__name__

        assert failure_type == "CompatibilityBridgeError"
        assert sys.modules[launch_canonical] is launch
        assert sys.modules[launch_private] is launch
        assert model_private not in sys.modules
        assert importlib.import_module("pretrain").production_launch_contract_v1 is launch
        emit(
            {
                "controlled_failure": True,
                "launch_bindings_preserved": True,
                "foreign_reserved_alias_absent": True,
            }
        )
        """
    )
    assert observed == {
        "controlled_failure": True,
        "foreign_reserved_alias_absent": True,
        "launch_bindings_preserved": True,
    }


def test_fresh_normal_import_waits_for_canonical_loader_initialization():
    observed = _run_probe(
        r"""
        import threading
        import time

        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        source = reviewed_path("src/optim.py")
        compile_entered = threading.Event()
        release_compile = threading.Event()
        normal_started = threading.Event()
        normal_finished = threading.Event()
        failures = []
        results = {}
        real_compile = builtins.compile

        def paused_compile(body, filename, *args, **kwargs):
            if Path(str(filename)).resolve() == source:
                compile_entered.set()
                assert release_compile.wait(timeout=30)
            return real_compile(body, filename, *args, **kwargs)

        def loader_import():
            try:
                results["loader"] = B._load_exact_successor_module(
                    "_petitgpt_r3_racing_src_optim", "src/optim.py"
                )
            except BaseException as exc:
                failures.append(f"loader:{type(exc).__name__}:{exc}")

        def normal_import():
            normal_started.set()
            try:
                results["normal"] = importlib.import_module("src.optim")
            except BaseException as exc:
                failures.append(f"normal:{type(exc).__name__}:{exc}")
            finally:
                normal_finished.set()

        builtins.compile = paused_compile
        loader_thread = threading.Thread(target=loader_import)
        normal_thread = threading.Thread(target=normal_import)
        try:
            loader_thread.start()
            assert compile_entered.wait(timeout=30)
            normal_thread.start()
            assert normal_started.wait(timeout=30)
            time.sleep(0.5)
            normal_was_blocked = not normal_finished.is_set()
        finally:
            release_compile.set()
            loader_thread.join(timeout=90)
            normal_thread.join(timeout=90)
            builtins.compile = real_compile

        assert normal_was_blocked
        assert not loader_thread.is_alive()
        assert not normal_thread.is_alive()
        assert failures == []
        assert results["normal"] is results["loader"]
        assert results["normal"].Muon is results["loader"].Muon
        assert_canonical_binding(
            results["loader"],
            "src.optim",
            "_petitgpt_successor_src_optim",
            "_petitgpt_r3_racing_src_optim",
        )
        assert len(exact_path_module_objects("src/optim.py")) == 1
        emit(
            {
                "normal_import_blocked_until_initialized": normal_was_blocked,
                "failure_count": len(failures),
                "same_module_object": True,
                "same_muon_class": True,
            }
        )
        """
    )
    assert observed == {
        "failure_count": 0,
        "normal_import_blocked_until_initialized": True,
        "same_module_object": True,
        "same_muon_class": True,
    }


def test_fresh_canonical_loader_waits_when_normal_import_owns_module_lock_first():
    observed = _run_probe(
        r"""
        import importlib._bootstrap as importlib_bootstrap
        import threading
        import time

        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        canonical_name = "src.optim"
        private_alias = "_petitgpt_r3_reverse_racing_src_optim"
        normal_find_spec_entered = threading.Event()
        release_normal_find_spec = threading.Event()
        loader_started = threading.Event()
        loader_finished = threading.Event()
        failures = []
        results = {}
        real_find_spec = importlib_bootstrap._find_spec

        def paused_find_spec(name, *args, **kwargs):
            if name == canonical_name:
                normal_find_spec_entered.set()
                assert release_normal_find_spec.wait(timeout=30)
            return real_find_spec(name, *args, **kwargs)

        def normal_import():
            try:
                results["normal"] = importlib.import_module(canonical_name)
            except BaseException as exc:
                failures.append(f"normal:{type(exc).__name__}:{exc}")

        def loader_import():
            loader_started.set()
            try:
                results["loader"] = B._load_exact_successor_module(
                    private_alias, "src/optim.py"
                )
            except BaseException as exc:
                failures.append(f"loader:{type(exc).__name__}:{exc}")
            finally:
                loader_finished.set()

        importlib_bootstrap._find_spec = paused_find_spec
        normal_thread = threading.Thread(target=normal_import)
        loader_thread = threading.Thread(target=loader_import)
        try:
            normal_thread.start()
            assert normal_find_spec_entered.wait(timeout=30)
            loader_thread.start()
            assert loader_started.wait(timeout=30)
            time.sleep(0.5)
            loader_was_blocked = not loader_finished.is_set()
        finally:
            release_normal_find_spec.set()
            normal_thread.join(timeout=90)
            loader_thread.join(timeout=90)
            importlib_bootstrap._find_spec = real_find_spec

        assert loader_was_blocked
        assert not normal_thread.is_alive()
        assert not loader_thread.is_alive()
        assert failures == []
        assert results["normal"] is results["loader"]
        assert results["normal"].Muon is results["loader"].Muon
        assert_canonical_binding(
            results["loader"],
            canonical_name,
            "_petitgpt_successor_src_optim",
            private_alias,
        )
        assert len(exact_path_module_objects("src/optim.py")) == 1
        emit(
            {
                "loader_blocked_behind_normal_import": loader_was_blocked,
                "failure_count": len(failures),
                "same_module_object": True,
                "same_muon_class": True,
                "one_exact_path_object": True,
            }
        )
        """
    )
    assert observed == {
        "failure_count": 0,
        "loader_blocked_behind_normal_import": True,
        "one_exact_path_object": True,
        "same_module_object": True,
        "same_muon_class": True,
    }


def test_fresh_source_execution_failure_rolls_back_all_bindings_before_retry():
    observed = _run_probe(
        r"""
        B = importlib.import_module(
            "pretrain.stage_n_successor_head_compatibility_bridge_v1"
        )
        canonical_name = "src.optim"
        fixed_alias = "_petitgpt_successor_src_optim"
        private_alias = "_petitgpt_r3_exec_failure_src_optim"
        parent = importlib.import_module("src")
        assert canonical_name not in sys.modules
        assert private_alias not in sys.modules
        assert "optim" not in parent.__dict__

        real_import = builtins.__import__

        def fail_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("deliberate R3 module-execution rollback probe")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = fail_torch_import
        failure_type = None
        try:
            B._load_exact_successor_module(private_alias, "src/optim.py")
        except BaseException as exc:
            failure_type = type(exc).__name__
        finally:
            builtins.__import__ = real_import

        canonical_absent = canonical_name not in sys.modules
        private_absent = private_alias not in sys.modules
        fixed_private_absent = fixed_alias not in sys.modules
        parent_restored = "optim" not in parent.__dict__
        retried = B._load_exact_successor_module(private_alias, "src/optim.py")

        assert failure_type in {"ImportError", "CompatibilityBridgeError"}
        assert canonical_absent
        assert private_absent
        assert fixed_private_absent
        assert parent_restored
        assert_canonical_binding(retried, canonical_name, private_alias)
        assert_reviewed(retried, "src/optim.py")
        assert len(exact_path_module_objects("src/optim.py")) == 1
        emit(
            {
                "failure_type": failure_type,
                "canonical_absent": canonical_absent,
                "private_absent": private_absent,
                "fixed_private_absent": fixed_private_absent,
                "parent_restored": parent_restored,
                "clean_retry_succeeded": True,
            }
        )
        """
    )
    assert observed["failure_type"] in {"ImportError", "CompatibilityBridgeError"}
    assert observed["canonical_absent"] is True
    assert observed["private_absent"] is True
    assert observed["fixed_private_absent"] is True
    assert observed["parent_restored"] is True
    assert observed["clean_retry_succeeded"] is True
