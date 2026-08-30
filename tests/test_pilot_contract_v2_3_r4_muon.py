"""R4 regressions for the load-bearing Muon ``lr_ratio`` field.

The fixture constructs optimizer groups but never performs a training update.
"""

from __future__ import annotations

import pytest
import torch

import pretrain.pilot_contract_v2_3 as C
from src.optim import build_optimizer


class _EveryMuonRole(torch.nn.Module):
    """Smallest model that realizes each frozen optimizer group role."""

    def __init__(self) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros(4, 4))
        self.tok_emb = torch.nn.Parameter(torch.zeros(8, 4))
        self.bias = torch.nn.Parameter(torch.zeros(4))


def _tiny_muon():
    model = _EveryMuonRole()
    optimizer = build_optimizer(
        model,
        name="muon",
        lr=3e-4,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR_ARG,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    return model, optimizer


def _indexed_group(optimizer, role):
    return next(
        (index, group)
        for index, group in enumerate(optimizer.param_groups)
        if C.optimizer_group_role(group) == role
    )


def test_machine_contract_forbids_a_missing_lr_ratio_default():
    optimizer = C.contract_document()["optimizer"]
    verification = C.contract_document()["optimizer_verification"]

    assert optimizer["lr_ratio_policy"] == {
        "required_on_every_realized_group": True,
        "missing_field_default_permitted": False,
        "required_value": 1.0,
    }
    assert verification["lr_ratio"] == 1.0
    assert verification["lr_ratio_explicitly_required"] is True
    assert verification["missing_lr_ratio_default_permitted"] is False


@pytest.mark.parametrize("role", C.OPTIMIZER_GROUP_ROLES)
def test_missing_lr_ratio_is_rejected_for_every_realized_group_role(role):
    model, optimizer = _tiny_muon()
    group_index, group = _indexed_group(optimizer, role)
    del group["lr_ratio"]

    verdict = C.verify_realized_grouping(optimizer, model)
    expected_ratios = [1.0] * len(optimizer.param_groups)
    expected_ratios[group_index] = None

    assert verdict["matches_frozen_realization"] is False
    assert verdict["all_lr_ratios_are_one"] is False
    assert verdict["lr_ratios"] == expected_ratios
    assert verdict["failures"] == [
        f"optimizer group {group_index} is missing required lr_ratio",
        f"every group lr_ratio must be {C.MUON_LR_RATIO}, got {expected_ratios}",
    ]


@pytest.mark.parametrize("invalid_ratio", [True, "1.0"])
def test_boolean_and_nonnumeric_lr_ratios_are_rejected_exactly(invalid_ratio):
    model, optimizer = _tiny_muon()
    group_index, group = _indexed_group(optimizer, "muon_matrices")
    group["lr_ratio"] = invalid_ratio

    verdict = C.verify_realized_grouping(optimizer, model)
    expected_ratios = [1.0] * len(optimizer.param_groups)
    expected_ratios[group_index] = repr(invalid_ratio)

    assert verdict["matches_frozen_realization"] is False
    assert verdict["all_lr_ratios_are_one"] is False
    assert verdict["lr_ratios"] == expected_ratios
    assert verdict["failures"] == [
        f"optimizer group {group_index} lr_ratio must be a real number, got {invalid_ratio!r}",
        f"every group lr_ratio must be {C.MUON_LR_RATIO}, got {expected_ratios}",
    ]
