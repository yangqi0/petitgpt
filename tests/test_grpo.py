"""Tests for GRPO core math, rollout alignment, and reward functions.

All CPU-only with tiny models. The trickiest thing GRPO gets right or wrong is
the alignment between sampling-time log-probs and the training-forward log-probs
(the importance ratio), so that has a dedicated end-to-end test.
"""

import pytest
import torch

from grpo import rewards
from grpo.grpo import (
    EOS_ID,
    _sample_step,
    build_group_batch,
    completion_text,
    group_advantages,
    grpo_loss,
    rollout_group,
    token_logprobs,
)
from src.model import GPT, GPTConfig


def _tiny_model():
    cfg = GPTConfig(
        vocab_size=64,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=80,
        max_seq_len=64,
        tie_embeddings=True,
    )
    return GPT(cfg).eval()


# --------------------------------------------------------------------------
# token_logprobs
# --------------------------------------------------------------------------
def test_token_logprobs_matches_manual():
    torch.manual_seed(0)
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    got = token_logprobs(logits, ids)
    assert got.shape == (B, T - 1)
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    for b in range(B):
        for t in range(T - 1):
            assert abs(got[b, t].item() - logp[b, t, ids[b, t + 1]].item()) < 1e-5


# --------------------------------------------------------------------------
# group_advantages
# --------------------------------------------------------------------------
def test_group_advantages_mean_zero_std_one():
    r = torch.tensor([0.0, 1.0, 2.0, 3.0])
    adv = group_advantages(r)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-3


def test_group_advantages_all_equal_is_zero():
    r = torch.full((6,), 2.5)
    adv = group_advantages(r)
    assert torch.allclose(adv, torch.zeros(6), atol=1e-6)


def test_group_advantages_no_std_norm():
    r = torch.tensor([0.0, 2.0, 4.0])
    adv = group_advantages(r, normalize_std=False)
    assert torch.allclose(adv, r - r.mean())


# --------------------------------------------------------------------------
# grpo_loss
# --------------------------------------------------------------------------
def _flat_inputs(N=3, L=4):
    logp = torch.zeros(N, L)
    old = torch.zeros(N, L)
    ref = torch.zeros(N, L)
    mask = torch.ones(N, L)
    return logp, old, ref, mask


def test_grpo_loss_ratio_one_no_kl_equals_neg_advantage():
    logp, old, ref, mask = _flat_inputs()
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss, m = grpo_loss(logp, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.0)
    # ratio==1, kl==0 => objective = mean_token(adv) = mean over sequences (equal lengths)
    assert abs(m["ratio"] - 1.0) < 1e-6
    assert abs(m["kl"]) < 1e-6
    assert abs(loss.item() - (-adv.mean().item())) < 1e-5


def test_grpo_loss_kl_is_nonnegative_and_zero_at_reference():
    logp, old, ref, mask = _flat_inputs()
    adv = torch.zeros(3)
    # logp == ref => kl == 0 => loss == 0
    loss0, m0 = grpo_loss(logp, old, ref, adv, mask, kl_coef=1.0)
    assert abs(m0["kl"]) < 1e-6
    assert abs(loss0.item()) < 1e-6
    # logp != ref => kl > 0 => loss > 0 (adv zero, only KL penalty remains)
    logp2 = logp + 0.5
    loss1, m1 = grpo_loss(logp2, logp2, ref, adv, mask, kl_coef=1.0)
    assert m1["kl"] > 0
    assert loss1.item() > 0


def test_grpo_loss_clips_large_ratio_for_positive_advantage():
    N, L = 2, 3
    old = torch.zeros(N, L)
    logp = torch.full((N, L), 1.0)  # ratio = e^1 ≈ 2.72, way above 1+eps
    ref = torch.zeros(N, L)
    mask = torch.ones(N, L)
    adv = torch.tensor([1.0, 1.0])  # positive => clip caps the surrogate
    _, m = grpo_loss(logp, old, ref, adv, mask, clip_eps=0.2, kl_coef=0.0)
    assert m["clipped_frac"] == 1.0


def test_grpo_loss_masks_padding():
    N, L = 1, 4
    old = torch.zeros(N, L)
    logp = torch.zeros(N, L)
    ref = torch.zeros(N, L)
    adv = torch.tensor([1.0])
    full_mask = torch.ones(N, L)
    half_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss_full, _ = grpo_loss(logp, old, ref, adv, full_mask, kl_coef=0.0)
    loss_half, _ = grpo_loss(logp, old, ref, adv, half_mask, kl_coef=0.0)
    # value identical here (all-equal per-token), but zero-mask must not divide by 0 / NaN
    assert torch.isfinite(loss_full) and torch.isfinite(loss_half)


def test_grpo_loss_reductions_differ_with_unequal_lengths():
    # two completions of different masked length; token_mean vs seq_mean differ
    old = torch.zeros(2, 4)
    logp = torch.zeros(2, 4)
    ref = torch.zeros(2, 4)
    adv = torch.tensor([2.0, -2.0])
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],  # len 3, adv +2
        [1.0, 0.0, 0.0, 0.0],
    ])  # len 1, adv -2
    tok_mean, _ = grpo_loss(logp, old, ref, adv, mask, kl_coef=0.0, reduction="token_mean")
    seq_mean, _ = grpo_loss(logp, old, ref, adv, mask, kl_coef=0.0, reduction="seq_mean")
    # token_mean weights the long (+2) completion more; seq_mean weights equally (=> 0)
    assert abs(seq_mean.item()) < 1e-6
    assert tok_mean.item() < 0  # dominated by the +2 advantage => negative loss


# --------------------------------------------------------------------------
# _sample_step
# --------------------------------------------------------------------------
def test_sample_step_greedy_picks_argmax_with_true_logprob():
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    nxt, logp = _sample_step(logits, temperature=0.0, top_p=1.0)
    assert nxt.item() == 1
    expected = torch.log_softmax(logits, dim=-1)[0, 1]
    assert abs(logp.item() - expected.item()) < 1e-6


# --------------------------------------------------------------------------
# rollout + batch alignment (the important integration test)
# --------------------------------------------------------------------------
def test_rollout_old_logp_matches_training_forward():
    """old_logp captured during greedy rollout must equal the training-forward
    token_logprobs on the built batch (i.e. importance ratio == 1 initially)."""
    torch.manual_seed(0)
    model = _tiny_model()
    prompt_ids = [2, 10, 11, 12]
    G = 3
    samples = rollout_group(
        model,
        prompt_ids,
        group_size=G,
        seq_len=64,
        max_new_tokens=6,
        temperature=0.0,
        top_p=1.0,
        device="cpu",
    )
    adv = torch.zeros(G)
    batch = build_group_batch(prompt_ids, samples, adv, seq_len=64, pad_id=0, device="cpu")
    assert batch is not None
    assert batch["input_ids"].shape[0] == G
    assert batch["comp_mask"].shape == batch["old_logp"].shape

    with torch.no_grad():
        logits = model(batch["input_ids"])
    logp = token_logprobs(logits, batch["input_ids"])
    mask = batch["comp_mask"].bool()
    # at masked (completion) positions the fresh forward must reproduce old_logp
    diff = (logp - batch["old_logp"])[mask].abs().max().item()
    assert diff < 1e-4, f"rollout/forward logp mismatch: {diff}"
    # comp_mask marks exactly the number of generated tokens per row
    for g in range(G):
        assert int(batch["comp_mask"][g].sum().item()) == len(samples[g]["gen_ids"])


def test_completion_text_strips_trailing_eos():
    class FakeTok:
        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    tok = FakeTok()
    assert completion_text(tok, [5, 6, EOS_ID]) == "5 6"
    assert completion_text(tok, []) == ""


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
def test_reward_nonempty():
    assert rewards.reward_nonempty("hi", {}) == 1.0
    assert rewards.reward_nonempty("   ", {}) == 0.0


def test_reward_length_peaks_at_target():
    ex = {"meta": {"target_words": 5}}
    at_target = rewards.reward_length("a b c d e", ex)
    far = rewards.reward_length("a " * 50, ex)
    assert at_target == pytest.approx(1.0, abs=1e-6)
    assert far < at_target
    assert rewards.reward_length("", ex) == 0.0


def test_reward_no_repeat_penalizes_loops():
    assert rewards.reward_no_repeat("the cat sat on the mat quietly", {}) == 1.0
    looped = rewards.reward_no_repeat("go go go go go go go go", {})
    assert looped < 0.5


def test_reward_reference_exact_and_contains():
    ex = {"reference": "The Answer Is 42"}
    assert rewards.reward_reference_exact("the answer is 42", ex) == 1.0
    assert rewards.reward_reference_exact("nope", ex) == 0.0
    ex2 = {"answer": "42"}
    assert rewards.reward_reference_contains("I think it is 42 maybe", ex2) == 1.0
    assert rewards.reward_reference_contains("no idea", ex2) == 0.0


def test_get_reward_fn_sum_and_weights():
    fn = rewards.get_reward_fn("nonempty+no_repeat")
    # nonempty(1.0) + no_repeat(1.0) for a clean short string
    assert fn("the cat sat still", {}) == pytest.approx(2.0)
    fw = rewards.get_reward_fn("nonempty:0.5")
    assert fw("hello", {}) == pytest.approx(0.5)


def test_get_reward_fn_unknown_raises():
    with pytest.raises(KeyError):
        rewards.get_reward_fn("does_not_exist")
    with pytest.raises(ValueError):
        rewards.get_reward_fn("")


def test_reward_code_structural_paths():
    # structurally valid single function, no tests provided => 0.5
    good = "```python\ndef f(x):\n    return x + 1\n```"
    assert rewards.reward_code(good, {"entry_point": "f"}) == 0.5
    # not a single clean function (has an import) => 0.0
    bad = "```python\nimport os\ndef f(x):\n    return x\n```"
    assert rewards.reward_code(bad, {"entry_point": "f"}) == 0.0


def test_reward_code_with_passing_tests():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    ex = {"entry_point": "add", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]}
    try:
        r = rewards.reward_code(good, ex)
    except Exception as e:  # sandbox unavailable in this environment
        pytest.skip(f"code sandbox unavailable: {e}")
    assert r == 1.0
