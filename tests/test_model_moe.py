"""Invariants for the Mixture-of-Experts GPT (src/model_moe.py)."""

from dataclasses import asdict

import pytest
import torch

from src.model import SwiGLU
from src.model_moe import MoEConfig, MoEFeedForward, MoEGPT


def _cfg(**overrides) -> MoEConfig:
    base = dict(
        vocab_size=256, n_layers=4, d_model=64, n_heads=4, d_ff=128,
        max_seq_len=64, n_experts=6, n_experts_per_tok=2,
    )
    base.update(overrides)
    return MoEConfig(**base)


def test_config_roundtrips_through_asdict():
    """A MoE checkpoint must be reconstructable via MoEConfig(**cfg_dict)."""
    cfg = _cfg(n_shared_experts=1, n_dense_layers=1, router_jitter=0.1)
    assert MoEConfig(**asdict(cfg)) == cfg


def test_forward_shape_default_returns_logits_only():
    cfg = _cfg()
    model = MoEGPT(cfg).eval()
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(ids)
    assert isinstance(out, torch.Tensor)  # drop-in with the dense model
    assert out.shape == (B, T, cfg.vocab_size)


def test_forward_returns_aux_loss_when_requested():
    cfg = _cfg()
    model = MoEGPT(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, aux = model(ids, return_aux_loss=True)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert aux.ndim == 0 and torch.isfinite(aux)
    # aux is also stashed for callers that use the plain forward signature
    assert torch.allclose(model.aux_loss, aux.detach())


def test_aux_loss_near_one_for_balanced_router():
    """The DeepSeek/Switch load-balancing loss is ~1.0 under perfect balance."""
    cfg = _cfg(n_experts=8, n_experts_per_tok=2)
    moe = MoEFeedForward(cfg).eval()
    torch.nn.init.zeros_(moe.gate.weight)  # uniform routing -> balanced
    x = torch.randn(4, 32, cfg.d_model)
    _, aux = moe(x)
    assert abs(aux.item() - 1.0) < 0.2


def test_backward_trains_router_and_experts():
    cfg = _cfg()
    model = MoEGPT(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, aux = model(ids, return_aux_loss=True)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1)
    ) + cfg.moe_aux_loss_coef * aux
    loss.backward()
    moe = model.blocks[-1].mlp
    assert moe.gate.weight.grad is not None
    assert moe.gate.weight.grad.abs().sum() > 0
    # over a few tokens at least one expert should receive gradient
    assert any(e.w2.weight.grad is not None and e.w2.weight.grad.abs().sum() > 0
               for e in moe.experts)


def test_dense_layers_use_plain_swiglu():
    cfg = _cfg(n_dense_layers=2)
    model = MoEGPT(cfg)
    assert isinstance(model.blocks[0].mlp, SwiGLU)
    assert isinstance(model.blocks[1].mlp, SwiGLU)
    assert isinstance(model.blocks[2].mlp, MoEFeedForward)


def test_active_params_less_than_total():
    cfg = _cfg(n_experts=8, n_experts_per_tok=2)
    counts = MoEGPT(cfg).num_parameters()
    assert counts["active_per_token"] < counts["total"]


def test_invalid_topk_rejected():
    with pytest.raises(ValueError):
        MoEFeedForward(_cfg(n_experts=4, n_experts_per_tok=5))


def test_causality_holds_for_moe():
    cfg = _cfg()
    model = MoEGPT(cfg).eval()
    T = 32
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        l1 = model(ids)
        ids2 = ids.clone()
        pos = 20
        ids2[0, pos] = (ids2[0, pos] + 5) % cfg.vocab_size
        l2 = model(ids2)
    assert (l1[:, :pos] - l2[:, :pos]).abs().max().item() < 1e-5
