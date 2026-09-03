"""Runtime-value canonicalization for the successor-head compatibility bridge.

`torch.__version__` is a `TorchVersion`, a str SUBCLASS. A runtime fingerprint reloaded from
canonical JSON therefore holds a plain `builtins.str` while a freshly observed one holds the
subclass. Their text and their serialized bytes are identical, but their concrete classes are
not -- which made the bridge's type-exact comparator reject a runtime that was in fact correct.

These tests pin the narrow fix: strings compare by value, everything else stays type-exact.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pretrain import (  # noqa: E402
    production_launch_contract_v1 as C,
    stage_n_successor_head_compatibility_bridge_v1 as B,
)


class _VersionLike(str):
    """A str subclass standing in for TorchVersion where the real object is not needed."""


# --------------------------------------------------------------- the reproduced defect


def test_plain_str_equals_equal_string_subclass():
    assert B._exact_state_equal("2.11.0+cu126", _VersionLike("2.11.0+cu126"))
    assert B._exact_state_equal(_VersionLike("2.11.0+cu126"), "2.11.0+cu126")


def test_live_torch_version_object_compares_equal_to_its_plain_text():
    """The actual installed torch.__version__ object, not a stand-in."""
    live = torch.__version__
    assert isinstance(live, str)
    assert type(live) is not str, "expected torch to supply a str subclass"
    assert B._exact_state_equal(str(live), live)


def test_different_version_strings_still_fail():
    assert not B._exact_state_equal("2.11.0+cu126", _VersionLike("2.10.0+cu121"))
    assert not B._exact_state_equal("2.11.0+cu126", "2.11.0+cu124")


# --------------------------------------------------------------- non-string strictness kept


def test_bool_and_int_still_fail():
    assert not B._exact_state_equal(True, 1)
    assert not B._exact_state_equal(1, True)


def test_int_and_float_still_fail():
    assert not B._exact_state_equal(1, 1.0)
    assert not B._exact_state_equal(1.0, 1)


def test_list_and_tuple_with_equal_elements_still_fail():
    assert not B._exact_state_equal([1, 2], (1, 2))
    assert not B._exact_state_equal((1, 2), [1, 2])


# --------------------------------------------------------------- nested runtime documents


def _doc(version):
    return {
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "num_workers": 2,
        "torch_version": version,
        "nested": {"visible_cuda_device_count": 1, "torch_version": version},
    }


def test_nested_documents_with_plain_and_subclass_leaves_compare_equal():
    assert B._exact_state_equal(_doc("2.11.0+cu126"), _doc(_VersionLike("2.11.0+cu126")))


def test_nested_documents_with_a_real_value_difference_fail():
    a = _doc("2.11.0+cu126")
    b = _doc("2.11.0+cu126")
    b["nested"]["visible_cuda_device_count"] = 2
    assert not B._exact_state_equal(a, b)


def test_nested_documents_with_a_bool_int_difference_fail():
    a = _doc("2.11.0+cu126")
    b = _doc("2.11.0+cu126")
    b["num_workers"] = True
    assert not B._exact_state_equal(a, b)


# --------------------------------------------------------------- capture canonicalization


def test_observed_runtime_records_torch_version_as_plain_str():
    """Capture must normalize, so the recorded document never carries the subclass."""
    runtime = C.observed_training_runtime(num_workers=2)
    assert type(runtime["torch_version"]) is str
    assert runtime["torch_version"] == str(torch.__version__)


def test_captured_runtime_survives_a_json_round_trip_under_exact_comparison():
    import json

    runtime = C.observed_training_runtime(num_workers=2)
    reloaded = json.loads(C.canonical_json_bytes(runtime).decode("utf-8"))
    assert B._exact_state_equal(runtime, reloaded), (
        "a captured runtime must compare equal to itself after canonical JSON round-trip"
    )


@pytest.mark.parametrize("field", ["gpu_name", "gpu_uuid", "torch_version", "python_version"])
def test_runtime_string_fields_are_plain_str(field):
    runtime = C.observed_training_runtime(num_workers=2)
    if field in runtime:
        assert type(runtime[field]) is str
