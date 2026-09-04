"""Module-origin consistency for the successor-head compatibility bridge.

Loading one file under two module names creates two sets of classes. A verifier doing a plain
`from src.optim import Muon` then rejects an optimizer built from the aliased module -- which is
how N3 failed with "optimizer must be the Muon instance, got Muon". The same split silently
breaks `except LaunchContractError` across the boundary.

These tests pin the invariant (one module object per file) and the provenance strictness that
must survive it: a same-name class from different bytes must still fail.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pretrain import (  # noqa: E402
    production_launch_contract_v1 as C,
    stage_n_successor_head_compatibility_bridge_v1 as B,
)

ALIASED = (
    ("_petitgpt_successor_src_optim", "src/optim.py", "src.optim"),
    ("_petitgpt_successor_src_model", "src/model.py", "src.model"),
)


# ------------------------------------------------------- one module object per file


@pytest.mark.parametrize("alias,rel,canonical", ALIASED)
def test_alias_and_canonical_names_share_one_module_object(alias, rel, canonical):
    module = B._load_exact_successor_module(alias, rel)
    assert sys.modules[alias] is module
    assert sys.modules[canonical] is module, (
        f"{rel} is backed by two module objects; class identity would split"
    )


def test_launch_contract_alias_and_canonical_share_one_module_object():
    launch = B.launch
    assert sys.modules["pretrain.production_launch_contract_v1"] is launch


def test_no_project_file_is_backed_by_two_module_objects():
    B._load_exact_successor_module("_petitgpt_successor_src_optim", "src/optim.py")
    B._load_exact_successor_module("_petitgpt_successor_src_model", "src/model.py")
    by_file: dict[str, set[int]] = {}
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f:
            continue
        resolved = str(Path(f).resolve())
        if str(REPO) in resolved and resolved.endswith(".py"):
            by_file.setdefault(resolved, set()).add(id(module))
    split = {f: ids for f, ids in by_file.items() if len(ids) > 1}
    assert not split, f"files backed by multiple module objects: {sorted(split)}"


# ------------------------------------------------------- class identity consequences


def test_muon_class_identity_is_unified():
    om = B._load_exact_successor_module("_petitgpt_successor_src_optim", "src/optim.py")
    from src.optim import Muon as canonical

    assert om.Muon is canonical


def test_model_class_identity_is_unified():
    mm = B._load_exact_successor_module("_petitgpt_successor_src_model", "src/model.py")
    from src.model import GPT, GPTConfig

    assert mm.GPT is GPT
    assert mm.GPTConfig is GPTConfig


def test_local_exception_crosses_the_module_boundary():
    """A canonical LaunchContractError must be caught by the aliased except."""
    with pytest.raises(B.launch.LaunchContractError):
        raise C.LaunchContractError("canonical")


def test_missing_sentinel_identity_is_unified():
    assert B.launch._Missing is C._Missing


# ------------------------------------------------------- provenance strictness kept


def _muon_from_other_bytes(tmpdir: Path):
    """A same-name Muon defined by a DIFFERENT file."""
    other = tmpdir / "impostor_optim.py"
    other.write_text("class Muon:\n    def __init__(self):\n        self.param_groups = []\n")
    spec = importlib.util.spec_from_file_location("_impostor_optim", other)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Muon


def test_same_name_muon_from_a_different_file_is_not_accepted():
    with tempfile.TemporaryDirectory() as d:
        impostor = _muon_from_other_bytes(Path(d))
        from src.optim import Muon as canonical

        assert impostor is not canonical
        assert impostor.__qualname__ == canonical.__qualname__
        assert not isinstance(impostor(), canonical), (
            "a same-name class from different bytes must never satisfy the check"
        )


def test_realized_optimizer_verifier_rejects_a_lookalike():
    """A lookalike with similar attributes must fail the realized-optimizer verifier."""
    with tempfile.TemporaryDirectory() as d:
        impostor = _muon_from_other_bytes(Path(d))
        verdict = C.verify_realized_optimizer(impostor())
        assert verdict["failures"], "a lookalike optimizer must produce failures"


def test_loader_refuses_to_unify_a_canonical_name_bound_to_different_bytes():
    """The unification must fail closed rather than silently adopt foreign bytes."""
    with tempfile.TemporaryDirectory() as d:
        other = Path(d) / "fake_optim.py"
        other.write_text("VALUE = 1\n")
        spec = importlib.util.spec_from_file_location("_fake_canonical_probe", other)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["_fake_canonical_probe"] = module
        try:
            with pytest.raises(B.CompatibilityBridgeError, match="refusing to unify"):
                B._bind_single_module_object(
                    "_fake_canonical_probe", object(), REPO / "src/optim.py"
                )
        finally:
            sys.modules.pop("_fake_canonical_probe", None)


# ------------------------------------------------------- framework strictness unchanged


def test_framework_types_retain_exact_handling():
    a = torch.zeros(2, dtype=torch.float32)
    assert B._exact_state_equal(a, torch.zeros(2, dtype=torch.float32))
    assert not B._exact_state_equal(a, torch.zeros(2, dtype=torch.float64))
    assert not B._exact_state_equal(a, torch.zeros(3, dtype=torch.float32))


def test_previous_torchversion_repair_is_retained():
    assert B._exact_state_equal("2.11.0+cu126", torch.__version__)
    assert not B._exact_state_equal(True, 1)
    assert not B._exact_state_equal(1, 1.0)
    assert not B._exact_state_equal([1, 2], (1, 2))
    runtime = C.observed_training_runtime(num_workers=2)
    assert type(runtime["torch_version"]) is str
