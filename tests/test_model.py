"""Correctness invariants for the dense GPT (src/model.py).

These are the kind of properties that a training bug silently violates without
ever raising: output shape, strict causality, tied-embedding weight sharing, and
the two defining properties of RoPE (norm preservation + relative-position
invariance of the q·k inner product). The RoPE tests would FAIL against the old
interleaved/half-split mismatch and PASS after the fix.
"""

from dataclasses import asdict
import math

import pytest
import torch

from src.model import (
    GPT,
    GPTConfig,
    RotaryEmbedding,
    expected_gpt_parameter_count,
    gpt_config_from_checkpoint_dict,
)


def test_forward_shape(tiny_cfg):
    model = GPT(tiny_cfg).eval()
    B, T = 2, 16
    ids = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, tiny_cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_backward_runs(tiny_cfg):
    model = GPT(tiny_cfg).train()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, tiny_cfg.vocab_size), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_tied_embeddings_share_storage():
    cfg = GPTConfig(
        vocab_size=128,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,  # explicit MHA coverage
        d_ff=80,
        max_seq_len=32,
        tie_embeddings=True,
    )
    model = GPT(cfg)
    # Same parameter object, same underlying storage — not just equal values.
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_untied_embeddings_are_separate():
    cfg = GPTConfig(
        vocab_size=128,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        d_ff=80,
        max_seq_len=32,
        tie_embeddings=False,
    )
    model = GPT(cfg)
    assert model.lm_head.weight is not model.tok_emb.weight


def test_causality_no_future_leak(tiny_cfg):
    """Perturbing a future token must not change logits at earlier positions.

    This is the training-loop `causal_leak_check` distilled into an assertion.
    """
    model = GPT(tiny_cfg).eval()
    T = 32
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, T))
    with torch.no_grad():
        logits1 = model(ids)
        ids2 = ids.clone()
        pos = 20
        ids2[0, pos] = (ids2[0, pos] + 7) % tiny_cfg.vocab_size
        logits2 = model(ids2)
    # positions strictly before `pos` must be identical
    max_diff = (logits1[:, :pos] - logits2[:, :pos]).abs().max().item()
    assert max_diff < 1e-5, f"future token leaked into past logits: {max_diff}"
    # sanity: the perturbed position itself DID change (test is meaningful)
    assert (logits1[:, pos] - logits2[:, pos]).abs().max().item() > 0


# --------------------------------------------------------------------------
# RoPE properties
# --------------------------------------------------------------------------
def _apply_rope(rope: RotaryEmbedding, x_tt_hd: torch.Tensor) -> torch.Tensor:
    """x: [T, Hd] -> apply RoPE as if [B=1, nH=1, T, Hd], return [T, Hd]."""
    q = x_tt_hd.unsqueeze(0).unsqueeze(0)
    q_out, _ = rope(q, q, seq_len=x_tt_hd.shape[0])
    return q_out[0, 0]


def test_rope_preserves_norm():
    """RoPE is an orthogonal rotation, so ‖RoPE(q)‖ == ‖q‖ at every position."""
    head_dim, T = 16, 48
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=T)
    q = torch.randn(T, head_dim)
    q_rot = _apply_rope(rope, q)
    err = (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()
    assert err < 1e-5, f"RoPE changed vector norms by {err}"


def test_rope_relative_position_invariance():
    """<RoPE(q,m), RoPE(k,n)> must depend only on the offset (m-n)."""
    head_dim, T = 16, 48
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=T)
    q_vec, k_vec = torch.randn(head_dim), torch.randn(head_dim)

    def dot_at(m, n):
        qq = torch.zeros(T, head_dim)
        kk = torch.zeros(T, head_dim)
        qq[m] = q_vec
        kk[n] = k_vec
        return (_apply_rope(rope, qq)[m] * _apply_rope(rope, kk)[n]).sum().item()

    delta = 4
    vals = [dot_at(m, m - delta) for m in range(delta, 30)]
    spread = max(vals) - min(vals)
    assert spread < 1e-4, f"inner product varies with absolute position: {spread}"


def test_rope_position_zero_is_identity():
    """At position 0 the rotation angle is 0, so RoPE is a no-op there."""
    head_dim, T = 16, 32
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=T)
    x = torch.randn(1, head_dim)
    padded = torch.cat([x, torch.zeros(T - 1, head_dim)])
    out0 = _apply_rope(rope, padded)[0]
    assert torch.allclose(out0, x[0], atol=1e-6)


def test_rope_matches_reference_llama():
    """Byte-for-byte agreement with an independent half-split RoPE reference."""
    head_dim, T, theta = 16, 40, 10000.0
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=T, theta=theta)
    x = torch.randn(T, head_dim)
    got = _apply_rope(rope, x)

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(T).float(), inv_freq)
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    h = head_dim // 2
    rot = torch.cat([-x[:, h:], x[:, :h]], dim=-1)
    ref = x * cos + rot * sin
    assert torch.allclose(got, ref, atol=1e-5)


def test_rope_partial_rotation_rotates_only_prefix():
    """rope_pct<1 rotates a prefix of head_dim and preserves its norm."""
    head_dim, T = 16, 32
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=T, pct=0.5)
    assert rope.rope_dim == 8
    q = torch.randn(1, 1, T, head_dim)
    out, _ = rope(q, q, seq_len=T)
    # untouched tail is passed through unchanged
    assert torch.allclose(out[..., rope.rope_dim :], q[..., rope.rope_dim :], atol=1e-6)
    # rotated prefix keeps its norm
    err = (
        (q[..., : rope.rope_dim].norm(dim=-1) - out[..., : rope.rope_dim].norm(dim=-1)).abs().max()
    )
    assert err < 1e-5


def test_residual_projection_depth_scaling(tiny_cfg):
    """attn.proj / mlp.w2 are initialized with std ~ 0.02/sqrt(2*n_layers)."""
    model = GPT(tiny_cfg)
    expected = 0.02 / math.sqrt(2.0 * tiny_cfg.n_layers)
    for blk in model.blocks:
        for w in (blk.attn.proj.weight, blk.mlp.w2.weight):
            # generous tolerance: it's a random draw, just check the right ballpark
            assert 0.3 * expected < w.std().item() < 3.0 * expected


# --------------------------------------------------------------------------
# GQA (grouped-query attention)
# --------------------------------------------------------------------------
def _gqa_cfg(n_kv_heads: int) -> GPTConfig:
    return GPTConfig(
        vocab_size=128,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        d_ff=160,
        max_seq_len=32,
        tie_embeddings=True,
    )


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_parameter_count_matches_derivation_across_kv_heads(n_kv_heads):
    """MQA (1), GQA (2) and MHA (4) all instantiate exactly the derived count."""
    cfg = _gqa_cfg(n_kv_heads)
    model = GPT(cfg)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected_gpt_parameter_count(cfg)


def test_gqa_fused_projection_layout():
    """K/V are n_kv_heads wide; n_kv_heads == n_heads reproduces the historical
    fused 3*d_model layout, keeping pre-GQA MHA checkpoints loadable."""
    head_dim = 64 // 4
    gqa = GPT(_gqa_cfg(2))
    assert gqa.blocks[0].attn.qkv.weight.shape == (64 + 2 * 2 * head_dim, 64)
    mha = GPT(_gqa_cfg(4))
    assert mha.blocks[0].attn.qkv.weight.shape == (3 * 64, 64)


@pytest.mark.parametrize("n_kv_heads", [2, 4])
def test_fused_qkv_row_order_matches_attention(n_kv_heads):
    """Pins the [Q; K; V] row order of the fused projection — the semantic half
    of the legacy-checkpoint guarantee that weight shapes alone cannot enforce.
    Recomputes attention manually from qkv.weight slices assuming that order;
    any reordering of the fused split makes this fail."""
    cfg = _gqa_cfg(n_kv_heads)
    attn = GPT(cfg).eval().blocks[0].attn
    B, T, C = 2, 12, cfg.d_model
    head_dim = C // cfg.n_heads
    kv_dim = n_kv_heads * head_dim
    torch.manual_seed(3)
    x = torch.randn(B, T, C)
    with torch.no_grad():
        W = attn.qkv.weight
        q = (x @ W[:C].T).view(B, T, cfg.n_heads, head_dim).transpose(1, 2)
        k = (x @ W[C : C + kv_dim].T).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        v = (x @ W[C + kv_dim :].T).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        q, k = attn.rope(q, k, seq_len=T)
        if n_kv_heads != cfg.n_heads:
            rep = cfg.n_heads // n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf")).softmax(dim=-1)
        ref = (att @ v).transpose(1, 2).reshape(B, T, C) @ attn.proj.weight.T
        got = attn(x)
    assert torch.allclose(got, ref, atol=1e-5)


def test_config_from_checkpoint_dict_defaults_legacy_to_mha():
    """Pre-GQA checkpoint config dicts (no n_kv_heads) must rebuild as plain
    MHA; modern dicts keep their stored value."""
    legacy = asdict(_gqa_cfg(4))
    del legacy["n_kv_heads"]
    cfg = gpt_config_from_checkpoint_dict(legacy)
    assert cfg.n_kv_heads == cfg.n_heads == 4
    assert GPT(cfg).blocks[0].attn.qkv.weight.shape == (3 * cfg.d_model, cfg.d_model)
    assert gpt_config_from_checkpoint_dict(asdict(_gqa_cfg(2))).n_kv_heads == 2


def test_invalid_kv_heads_rejected():
    with pytest.raises(AssertionError):
        GPT(GPTConfig(vocab_size=128, n_layers=1, d_model=64, n_heads=4, n_kv_heads=3, d_ff=160))
    with pytest.raises(ValueError):
        expected_gpt_parameter_count(
            GPTConfig(vocab_size=128, n_layers=1, d_model=64, n_heads=4, n_kv_heads=3, d_ff=160)
        )


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_gqa_causality_no_future_leak(n_kv_heads):
    """Strict causality must hold for MQA, GQA, and the plain-MHA branch
    (n_kv_heads == n_heads skips the KV expansion entirely)."""
    cfg = _gqa_cfg(n_kv_heads)
    model = GPT(cfg).eval()
    T = 24
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        logits1 = model(ids)
        ids2 = ids.clone()
        pos = 15
        ids2[0, pos] = (ids2[0, pos] + 7) % cfg.vocab_size
        logits2 = model(ids2)
    max_diff = (logits1[:, :pos] - logits2[:, :pos]).abs().max().item()
    assert max_diff < 1e-5, f"future token leaked into past logits: {max_diff}"
    assert (logits1[:, pos] - logits2[:, pos]).abs().max().item() > 0
