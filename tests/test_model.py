"""Correctness invariants for the dense GPT (src/model.py).

These are the kind of properties that a training bug silently violates without
ever raising: output shape, strict causality, tied-embedding weight sharing, and
the two defining properties of RoPE (norm preservation + relative-position
invariance of the q·k inner product). The RoPE tests would FAIL against the old
interleaved/half-split mismatch and PASS after the fix.
"""

import math

import torch

from src.model import GPT, GPTConfig, RotaryEmbedding


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
    cfg = GPTConfig(vocab_size=128, n_layers=2, d_model=32, n_heads=4, d_ff=80,
                    max_seq_len=32, tie_embeddings=True)
    model = GPT(cfg)
    # Same parameter object, same underlying storage — not just equal values.
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_untied_embeddings_are_separate():
    cfg = GPTConfig(vocab_size=128, n_layers=2, d_model=32, n_heads=4, d_ff=80,
                    max_seq_len=32, tie_embeddings=False)
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
        qq = torch.zeros(T, head_dim); qq[m] = q_vec
        kk = torch.zeros(T, head_dim); kk[n] = k_vec
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
    assert torch.allclose(out[..., rope.rope_dim:], q[..., rope.rope_dim:], atol=1e-6)
    # rotated prefix keeps its norm
    err = (q[..., :rope.rope_dim].norm(dim=-1) - out[..., :rope.rope_dim].norm(dim=-1)).abs().max()
    assert err < 1e-5


def test_residual_projection_depth_scaling(tiny_cfg):
    """attn.proj / mlp.w2 are initialized with std ~ 0.02/sqrt(2*n_layers)."""
    model = GPT(tiny_cfg)
    expected = 0.02 / math.sqrt(2.0 * tiny_cfg.n_layers)
    for blk in model.blocks:
        for w in (blk.attn.proj.weight, blk.mlp.w2.weight):
            # generous tolerance: it's a random draw, just check the right ballpark
            assert 0.3 * expected < w.std().item() < 3.0 * expected
