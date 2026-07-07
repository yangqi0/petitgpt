"""Numeric tests for the DPO loss and per-sequence log-prob computation
(dpo/dpo.py). No GPU or checkpoint required."""

import math

import torch

from dpo.dpo import dpo_loss, sequence_logps


def test_dpo_loss_at_zero_margin_is_log2():
    """When policy and reference agree (margin 0), loss == log 2 for any beta."""
    z = torch.zeros(4)
    for beta in (0.1, 0.5, 1.0):
        losses, cr, rr = dpo_loss(z, z, z, z, beta=beta)
        assert torch.allclose(losses, torch.full((4,), math.log(2.0)), atol=1e-6)
        # rewards are zero when policy == reference
        assert torch.allclose(cr, z) and torch.allclose(rr, z)


def test_dpo_loss_decreases_as_margin_grows():
    """Larger (chosen - rejected) advantage over the reference => smaller loss."""
    beta = 0.1
    ref = torch.zeros(1)
    # policy strongly prefers chosen over rejected
    good, _, _ = dpo_loss(torch.tensor([2.0]), torch.tensor([-2.0]), ref, ref, beta)
    # policy prefers rejected (wrong) => larger loss
    bad, _, _ = dpo_loss(torch.tensor([-2.0]), torch.tensor([2.0]), ref, ref, beta)
    assert good.item() < math.log(2.0) < bad.item()


def test_dpo_loss_limits():
    beta = 1.0
    ref = torch.zeros(1)
    # margin -> +inf  => loss -> 0
    big, _, _ = dpo_loss(torch.tensor([50.0]), torch.tensor([-50.0]), ref, ref, beta)
    assert big.item() < 1e-3
    # margin -> -inf  => loss grows large
    huge, _, _ = dpo_loss(torch.tensor([-50.0]), torch.tensor([50.0]), ref, ref, beta)
    assert huge.item() > 50.0


def test_dpo_rewards_match_definition():
    beta = 0.3
    pc, pr = torch.tensor([1.5]), torch.tensor([0.5])
    rc, rr_ = torch.tensor([1.0]), torch.tensor([0.2])
    _, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr_, beta)
    assert torch.allclose(chosen_rewards, beta * (pc - rc))
    assert torch.allclose(rejected_rewards, beta * (pr - rr_))


def test_sequence_logps_matches_manual_logsoftmax():
    torch.manual_seed(0)
    B, T, V = 1, 4, 6
    logits = torch.randn(B, T, V)
    # supervise positions 2 and 3 (label at index t is the target for input t-1)
    labels = torch.tensor([[-100, -100, 3, 5]])

    got = sequence_logps(logits, labels)

    # manual: shift, then sum log-prob of supervised targets
    logp = torch.log_softmax(logits[:, :-1], dim=-1)  # positions 0..T-2
    tgt = labels[:, 1:]  # targets for positions 0..T-2
    total = 0.0
    for t in range(T - 1):
        y = tgt[0, t].item()
        if y != -100:
            total += logp[0, t, y].item()
    assert abs(got.item() - total) < 1e-5


def test_sequence_logps_ignores_masked_positions():
    """Positions with label -100 contribute exactly 0."""
    B, T, V = 1, 5, 8
    logits = torch.randn(B, T, V)
    labels_all_masked = torch.full((B, T), -100)
    assert sequence_logps(logits, labels_all_masked).item() == 0.0
