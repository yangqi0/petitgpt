"""Tests for the primitives in src/canonical_loss.py and src/canonical_schedule.py.

These pin the exact loss and schedule mathematics used by the accepted pretraining run,
independently of any trainer module.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from src.canonical_loss import masked_weighted_ce_components
from src.canonical_schedule import lr_schedule


def test_loss_matches_an_independent_reference():
    torch.manual_seed(11)
    logits, labels = torch.randn(3, 6, 9), torch.randint(0, 9, (3, 6))
    mask = (torch.rand(3, 6) > 0.3).float()
    per_token = F.cross_entropy(logits.reshape(-1, 9), labels.reshape(-1), reduction="none")
    num, w = masked_weighted_ce_components(logits, labels, mask, eos_id=3, eos_weight=1.0)
    assert torch.allclose(num, (per_token * mask.reshape(-1)).sum())
    assert torch.allclose(w, mask.sum())


def test_eos_weight_reweights_only_eos_targets():
    torch.manual_seed(3)
    logits, labels = torch.randn(1, 4, 5), torch.tensor([[3, 1, 3, 2]])
    mask = torch.ones(1, 4)
    _n1, w1 = masked_weighted_ce_components(logits, labels, mask, eos_id=3, eos_weight=1.0)
    _n2, w2 = masked_weighted_ce_components(logits, labels, mask, eos_id=3, eos_weight=2.0)
    assert float(w1) == 4.0
    assert float(w2) == 6.0  # two EOS targets doubled


def test_wsd_decay_is_cosine_not_linear():
    kw = dict(
        schedule="wsd",
        schedule_total_steps=49590,
        decay_start_step=44631,
        decay_end_step=49590,
        min_lr_ratio=0.1,
    )
    mid = lr_schedule(44631 + 4959 // 2, 500, 1.0, **kw)
    linear_mid = 1.0 + (0.1 - 1.0) * 0.5
    assert mid == pytest.approx(0.55, abs=1e-3)
    assert mid == pytest.approx(linear_mid, abs=1e-3)  # cosine and linear agree at the midpoint
    quarter = lr_schedule(44631 + 4959 // 4, 500, 1.0, **kw)
    linear_quarter = 1.0 + (0.1 - 1.0) * 0.25
    assert quarter != pytest.approx(linear_quarter, abs=1e-3), "must be cosine, not linear"
    expected = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (4959 // 4) / 4959))
    assert quarter == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (499, 1.0),
        (500, 1.0),
        (44631, 1.0),
        (49589, 0.10000009030130619),
        (49590, 0.1),
        (60000, 0.1),
    ],
)
def test_frozen_geometry_exact_values(step, expected):
    kw = dict(
        schedule="wsd",
        schedule_total_steps=49590,
        decay_start_step=44631,
        decay_end_step=49590,
        min_lr_ratio=0.1,
    )
    assert lr_schedule(step, 500, 1.0, **kw) == pytest.approx(expected, rel=1e-15)


def test_warmup_boundary_values():
    kw = dict(
        schedule="wsd",
        schedule_total_steps=49590,
        decay_start_step=44631,
        decay_end_step=49590,
        min_lr_ratio=0.1,
    )
    assert lr_schedule(0, 500, 1.0, **kw) == pytest.approx(1 / 500)
    assert lr_schedule(498, 500, 1.0, **kw) == pytest.approx(499 / 500)
    assert lr_schedule(499, 500, 1.0, **kw) == pytest.approx(1.0)
