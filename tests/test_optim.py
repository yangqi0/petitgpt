"""Tests for the shared optimizer factory (src/optim.py): Muon + AdamW."""

import pytest
import torch

from src.model import GPT, GPTConfig
from src.optim import Muon, build_optimizer, zeropower_via_newtonschulz5


def _cfg():
    return GPTConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=160,
        max_seq_len=64,
        tie_embeddings=True,
    )


def test_newton_schulz_is_semi_orthogonal():
    """Newton-Schulz should map a random matrix to ~semi-orthogonal (σ ~ 1)."""
    for shape in [(64, 64), (32, 128), (128, 32)]:
        g = torch.randn(*shape)
        u = zeropower_via_newtonschulz5(g, steps=5).float()
        sv = torch.linalg.svdvals(u)
        # Non-trivial singular values land in the Muon design band ~[0.7, 1.2].
        # (Some tiny-input singular values can undershoot; check the bulk.)
        assert sv.max() < 1.3
        assert sv.median() > 0.6


def test_muon_grouping_dedups_and_separates():
    model = GPT(_cfg())
    opt = build_optimizer(model, name="muon", lr=1e-3, weight_decay=0.1, verbose=False)
    # every trainable parameter appears exactly once (tied weights deduped)
    n_grouped = sum(len(g["params"]) for g in opt.param_groups)
    assert n_grouped == len(list(model.parameters()))
    for g in opt.param_groups:
        if g["use_muon"]:
            assert all(p.ndim == 2 for p in g["params"])  # Muon = matrices only
        else:
            # non-muon groups carry AdamW settings
            assert "betas" in g
    # the no-decay group (norm gains) must have weight_decay == 0
    nd = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert nd, "expected a no-weight-decay group for norm/bias params"


def test_adamw_grouping_no_decay_on_norms():
    model = GPT(_cfg())
    opt = build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.1, verbose=False)
    assert isinstance(opt, torch.optim.AdamW)
    wds = sorted({g["weight_decay"] for g in opt.param_groups})
    assert 0.0 in wds and 0.1 in wds  # decay + no-decay groups both present


def test_every_group_has_lr_ratio():
    model = GPT(_cfg())
    for name in ("muon", "adamw"):
        opt = build_optimizer(model, name=name, lr=1e-3, weight_decay=0.1, verbose=False)
        assert all("lr_ratio" in g for g in opt.param_groups)


def _train_toy(name, steps=60):
    torch.manual_seed(1)
    cfg = _cfg()
    m = GPT(cfg)
    o = build_optimizer(m, name=name, lr=3e-3, weight_decay=0.01, verbose=False)
    ids = torch.randint(0, cfg.vocab_size, (4, 32))
    first = last = None
    for s in range(steps):
        lr = 3e-3 * min(1.0, (s + 1) / 10)
        for pg in o.param_groups:
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)
        logits = m(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
        )
        o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        first = loss.item() if first is None else first
        last = loss.item()
    return first, last


def test_muon_reduces_loss():
    first, last = _train_toy("muon")
    assert last < first * 0.7, f"muon did not reduce loss: {first} -> {last}"


def test_adamw_reduces_loss():
    first, last = _train_toy("adamw")
    assert last < first * 0.7, f"adamw did not reduce loss: {first} -> {last}"


def test_muon_state_dict_roundtrip():
    torch.manual_seed(1)
    cfg = _cfg()
    m = GPT(cfg)
    o = build_optimizer(m, name="muon", lr=3e-3, weight_decay=0.01, verbose=False)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    for _ in range(3):
        loss = torch.nn.functional.cross_entropy(
            m(ids)[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
        )
        o.zero_grad()
        loss.backward()
        o.step()
    sd = o.state_dict()

    m2 = GPT(cfg)
    m2.load_state_dict(m.state_dict())
    o2 = build_optimizer(m2, name="muon", lr=3e-3, weight_decay=0.01, verbose=False)
    o2.load_state_dict(sd)
    assert isinstance(o2, Muon)
    # one more step continues without error
    loss = torch.nn.functional.cross_entropy(
        m2(ids)[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
    )
    o2.zero_grad()
    loss.backward()
    o2.step()


def test_cross_optimizer_state_load_raises():
    """Loading an AdamW-shaped state into a Muon optimizer must fail loudly so
    the training scripts can catch it and fall back to fresh state."""
    m = GPT(_cfg())
    muon = build_optimizer(m, name="muon", lr=1e-3, weight_decay=0.1, verbose=False)
    adamw_state = torch.optim.AdamW(m.parameters(), lr=1e-3).state_dict()
    with pytest.raises(ValueError):
        muon.load_state_dict(adamw_state)


def test_muon_rejects_non_2d_param_in_muon_group():
    p1d = torch.nn.Parameter(torch.randn(8))
    with pytest.raises(ValueError):
        Muon([{"params": [p1d], "use_muon": True, "lr": 1e-3, "weight_decay": 0.0}])
