"""KV-cache / incremental-decoding tests for src/model.py.

The whole point of the cache is to reproduce the exact same logits as a full
recompute while doing O(1)-length forwards per step, so every test here is an
equivalence check against the plain forward. All CPU, tiny models.
"""

import torch

from src.model import GPT, GPTConfig


def _cfg(**kw):
    base = dict(
        vocab_size=64,
        n_layers=3,
        d_model=48,
        n_heads=4,
        d_ff=128,
        max_seq_len=64,
        tie_embeddings=True,
    )
    base.update(kw)
    return GPTConfig(**base)


def test_plain_forward_unchanged_returns_tensor():
    """model(ids) must still return a bare logits tensor (backward compatible)."""
    model = GPT(_cfg()).eval()
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 10, 64)


def test_prefill_cache_matches_plain_forward():
    """use_cache prefill returns the same logits as the plain forward, plus a
    per-layer cache of the right shape."""
    cfg = _cfg()
    model = GPT(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        plain = model(ids)
        cached, past = model(ids, use_cache=True)
    assert torch.allclose(plain, cached, atol=1e-5)
    assert len(past) == cfg.n_layers
    for k, v in past:
        assert k.shape == (2, cfg.n_heads, 12, cfg.d_model // cfg.n_heads)
        assert v.shape == k.shape


def test_incremental_decode_matches_full_recompute():
    """Feeding tokens one at a time through the cache must reproduce, at each
    step, the last-position logits of a full forward over the prefix."""
    cfg = _cfg()
    model = GPT(cfg).eval()
    torch.manual_seed(0)
    full_seq = torch.randint(0, cfg.vocab_size, (1, 20))

    prompt_len = 5
    with torch.no_grad():
        logits, past = model(full_seq[:, :prompt_len], use_cache=True)
        for t in range(prompt_len, full_seq.size(1)):
            # `logits[:, -1]` predicts token t from prefix 0..t-1; the full-forward
            # equivalent is a forward over exactly that prefix (seq[:, :t]).
            ref = model(full_seq[:, :t])[:, -1, :]
            got = logits[:, -1, :]
            assert torch.allclose(got, ref, atol=1e-4), f"mismatch at position {t}"
            # feed the actual token t to advance the cache
            logits, past = model(full_seq[:, t : t + 1], past_kv=past, use_cache=True)
    # cache should now hold the whole sequence length
    assert past[0][0].size(2) == full_seq.size(1)


def test_generate_greedy_matches_naive_recompute():
    """Cached greedy generate == greedy decoding by full recompute each step."""
    cfg = _cfg()
    model = GPT(cfg).eval()
    torch.manual_seed(1)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    n = 12

    fast = model.generate(prompt, max_new_tokens=n, temperature=0.0)

    # naive reference: recompute the full sequence each step, take argmax
    naive = prompt.clone()
    with torch.no_grad():
        for _ in range(n):
            nxt = model(naive)[:, -1, :].argmax(dim=-1, keepdim=True)
            naive = torch.cat([naive, nxt], dim=1)
    assert torch.equal(fast, naive)


def test_generate_respects_eos_and_length():
    cfg = _cfg()
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert out.size(1) <= prompt.size(1) + 8
    assert out.size(1) >= prompt.size(1) + 1


def test_generate_stops_before_exceeding_max_seq_len():
    cfg = _cfg(max_seq_len=10)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model.generate(prompt, max_new_tokens=100, temperature=0.0)
    assert out.size(1) <= cfg.max_seq_len


def test_cache_equivalence_with_partial_rope():
    """Incremental decoding must also match under partial RoPE (rope_pct<1)."""
    cfg = _cfg(rope_pct=0.5)
    model = GPT(cfg).eval()
    torch.manual_seed(2)
    seq = torch.randint(0, cfg.vocab_size, (1, 14))
    with torch.no_grad():
        logits, past = model(seq[:, :4], use_cache=True)
        for t in range(4, seq.size(1)):
            ref = model(seq[:, :t])[:, -1, :]
            assert torch.allclose(logits[:, -1, :], ref, atol=1e-4)
            logits, past = model(seq[:, t : t + 1], past_kv=past, use_cache=True)
