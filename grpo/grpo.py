#!/usr/bin/env python3

"""
Group Relative Policy Optimization (GRPO) post-training for petitgpt.

GRPO (Shao et al., DeepSeekMath 2024) is an online, critic-free policy-gradient
method: for each prompt it samples a *group* of G completions from the current
policy, scores them with a reward function, and uses the group's mean/std as the
baseline — the advantage of completion i is `(r_i - mean(r)) / (std(r) + eps)`.
No value network is trained (that is the "group relative" trick). A per-token KL
penalty to a frozen reference keeps the policy from drifting.

Per-token objective (PPO-style clipped surrogate + KL, maximized):

    ratio_t   = pi_theta(o_t) / pi_theta_old(o_t)
    surrogate = min(ratio_t * A_i, clip(ratio_t, 1-eps, 1+eps) * A_i)
    kl_t      = exp(ref_t - logp_t) - (ref_t - logp_t) - 1     # k3, unbiased, >=0
    loss      = - mean_over_completion_tokens( surrogate - kl_coef * kl_t )

`pi_theta_old` is the policy at *sampling* time; with the default single update
per rollout the ratio starts at 1 and the clip guards the accumulated drift.

Starts from an SFT/distill/DPO checkpoint (same schema as dpo/dpo.py: keys
"model" + "cfg"/"config"), builds a frozen reference copy, and reuses the plain
chat template ("System: .../User: .../Assistant: ...").

Data format (JSONL, one PROMPT per line) — no chosen/rejected, GRPO generates
its own completions and scores them with `--reward`:

    {"messages": [{"role": "user", "content": "..."}],
     "reference": "optional canonical answer",   # for reference_* rewards
     "tests": ["assert f(1) == 2", ...],          # for the `code` reward
     "entry_point": "f",                          # optional, for `code`
     "meta": {...}}

Example:
    python grpo/grpo.py \\
      --train_jsonl datasets/grpo/train.jsonl --val_jsonl datasets/grpo/val.jsonl \\
      --out_dir outputs/grpo_run --tokenizer_path tokenizer/tokenizer.json \\
      --init_ckpt outputs/sft_v6_general_code/step_003500.pt \\
      --reward code:1.0+no_repeat:0.1 --group_size 8 --groups_per_step 4 \\
      --seq_len 1024 --lr 1e-6 --kl_coef 0.04
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
import random
import sys
import time
from typing import Any

from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # imported as a package (e.g. tests: `from grpo.grpo import ...`)
    from grpo.rewards import get_reward_fn  # noqa: E402
except ModuleNotFoundError:
    # run as a script (`python grpo/grpo.py`), where grpo/ is on sys.path[0]
    from rewards import get_reward_fn  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.optim import build_optimizer  # noqa: E402
from src.special_tokens import assert_special_token_ids  # noqa: E402
from src.tracking import Tracker  # noqa: E402

# -------------------------
# Plain chat template (NO bracket tags; robust to tokenizer)
# Kept identical to sft/train_sft.py, distill/train_distill.py, dpo/dpo.py.
# -------------------------
BOS_ID = 2
EOS_ID = 3

SYS_PREFIX = "System: "
USER_PREFIX = "User: "
ASSIST_PREFIX = "Assistant: "
SEP = "\n\n"


# -------------------------
# Text cleaning + jsonl dataset (prompts only)
# -------------------------
def norm_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def clean_text(s: str) -> str:
    return norm_newlines(s).strip()


class JsonlOffsetsDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        off = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        return json.loads(line)


def collate_passthrough(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


# -------------------------
# Prompt rendering + encoding
# -------------------------
def encode_strip_special(tok: Tokenizer, text: str, bos_id: int, eos_id: int) -> list[int]:
    ids = tok.encode(text).ids
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]
    return ids


def render_prompt_text(messages: list[dict[str, str]], default_system: str) -> str:
    """Render the prompt context and the trailing 'Assistant: ' cue as plain
    text, so the policy generates the assistant completion from there."""
    msgs = list(messages or [])
    if not msgs:
        raise ValueError("missing messages")
    if (msgs[0].get("role") or "").strip().lower() != "system" and default_system:
        msgs = [{"role": "system", "content": default_system}] + msgs

    parts: list[str] = []
    start = 0
    if msgs and (msgs[0].get("role") or "").strip().lower() == "system":
        sys_txt = clean_text(msgs[0].get("content", ""))
        if sys_txt:
            parts.append(SYS_PREFIX + sys_txt + SEP)
        start = 1
    for m in msgs[start:]:
        role = (m.get("role") or "").strip().lower()
        txt = clean_text(m.get("content", ""))
        if not txt:
            continue
        if role == "user":
            parts.append(USER_PREFIX + txt + SEP)
        elif role == "assistant":
            parts.append(ASSIST_PREFIX + txt + SEP)
    parts.append(ASSIST_PREFIX)
    return "".join(parts)


def encode_prompt(
    tok: Tokenizer, messages: list[dict[str, str]], default_system: str, max_prompt_len: int
) -> list[int]:
    text = render_prompt_text(messages, default_system)
    ids = encode_strip_special(tok, text, BOS_ID, EOS_ID)
    # match training: sequences start with BOS
    ids = [BOS_ID] + ids
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]  # keep the tail (most recent turn + cue)
    return ids


# -------------------------
# Core GRPO math (unit-tested)
# -------------------------
def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of each next token: given logits [B,T,V] and the
    realized ids [B,T], returns [B,T-1] where out[:,t] = log p(id[t+1] | <=t)."""
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = input_ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, tgt).squeeze(-1)


def group_advantages(
    rewards: torch.Tensor, eps: float = 1e-4, normalize_std: bool = True
) -> torch.Tensor:
    """Group-relative advantages: (r - mean) [/ (std + eps)].

    A group with identical rewards yields all-zero advantages (no reward signal;
    only the KL term acts), which is the correct GRPO behavior.
    """
    adv = rewards - rewards.mean()
    if normalize_std:
        adv = adv / (rewards.std(unbiased=False) + eps)
    return adv


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    comp_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
    reduction: str = "token_mean",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped GRPO surrogate + per-token KL penalty over completion tokens.

    logp/old_logp/ref_logp: [N, L] per-token log-probs (N = completions in the
    batch, L = max completion-aligned length). advantages: [N] per completion.
    comp_mask: [N, L] 1.0 on real completion tokens, 0.0 on padding/prompt.
    reduction: "token_mean" (mean over all masked tokens, length-unbiased) or
    "seq_mean" (mean over tokens within each completion, then mean over N — the
    original GRPO normalization, which is mildly length-biased).
    """
    ratio = torch.exp(logp - old_logp)
    adv = advantages.unsqueeze(1)  # [N,1]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(surr1, surr2)

    # k3 unbiased KL estimator (DeepSeekMath): >= 0, zero when logp == ref_logp.
    diff = ref_logp - logp
    kl = torch.exp(diff) - diff - 1.0

    per_token = surrogate - kl_coef * kl
    mask = comp_mask.float()

    if reduction == "seq_mean":
        tok_per_seq = mask.sum(dim=1).clamp_min(1.0)
        seq_obj = (per_token * mask).sum(dim=1) / tok_per_seq
        objective = seq_obj.mean()
    elif reduction == "token_mean":
        objective = (per_token * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        raise ValueError(f"unknown reduction {reduction!r}")

    loss = -objective

    with torch.no_grad():
        n_tok = mask.sum().clamp_min(1.0)
        clipped = ((surr2 < surr1).float() * mask).sum() / n_tok
        metrics = {
            "kl": float((kl * mask).sum().item() / n_tok.item()),
            "ratio": float((ratio * mask).sum().item() / n_tok.item()),
            "clipped_frac": float(clipped.item()),
        }
    return loss, metrics


# -------------------------
# Rollout: sample a group of G completions and capture old log-probs
# -------------------------
def _sample_step(
    logits: torch.Tensor, temperature: float, top_p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample next token per row and return (token[B], logp[B]).

    `logp` is the token's log-prob under the model's TRUE distribution
    (temperature 1, no top-p truncation), so it is consistent with the
    training-time `token_logprobs`. Truncation/temperature only shape sampling.
    """
    raw_logp = F.log_softmax(logits.float(), dim=-1)
    if temperature <= 0:
        nxt = logits.argmax(dim=-1)
    else:
        t = logits.float() / temperature
        if top_p and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(t, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            drop_sorted = cum > top_p
            drop_sorted[..., 0] = False
            drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
            t = t.masked_fill(drop, -float("inf"))
        probs = torch.softmax(t, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
    logp = raw_logp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
    return nxt, logp


@torch.no_grad()
def rollout_group(
    policy: torch.nn.Module,
    prompt_ids: list[int],
    group_size: int,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> list[dict[str, Any]]:
    """Generate `group_size` completions for one prompt (batched over the group).

    Returns a list of dicts: {gen_ids, old_logps} where gen_ids are the sampled
    completion token ids (trailing EOS included if produced) and old_logps are
    their per-token log-probs under the sampling policy.
    """
    policy.eval()
    ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :].repeat(group_size, 1)
    gen_ids: list[list[int]] = [[] for _ in range(group_size)]
    old_logps: list[list[float]] = [[] for _ in range(group_size)]
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        window = ids[:, -seq_len:]
        logits = policy(window)[:, -1, :]
        nxt, logp = _sample_step(logits, temperature, top_p)
        active = ~finished
        for g in range(group_size):
            if bool(active[g]):
                gen_ids[g].append(int(nxt[g].item()))
                old_logps[g].append(float(logp[g].item()))
        nxt = torch.where(finished, torch.full_like(nxt, BOS_ID), nxt)  # filler for done rows
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        finished = finished | (nxt == EOS_ID)
        if bool(finished.all()):
            break

    policy.train()
    return [{"gen_ids": gen_ids[g], "old_logps": old_logps[g]} for g in range(group_size)]


def completion_text(tok: Tokenizer, gen_ids: list[int]) -> str:
    ids = gen_ids[:-1] if (gen_ids and gen_ids[-1] == EOS_ID) else gen_ids
    return tok.decode(ids).strip() if ids else ""


# -------------------------
# Build a padded training batch from one prompt's rollout group
# -------------------------
def build_group_batch(
    prompt_ids: list[int],
    samples: list[dict[str, Any]],
    advantages: torch.Tensor,
    seq_len: int,
    pad_id: int,
    device: str,
) -> dict[str, torch.Tensor] | None:
    """Pack a group's completions into padded tensors for the loss forward.

    Returns None if no completion produced any tokens. Tensors:
      input_ids [G, L], comp_mask [G, L-1], old_logp [G, L-1], advantages [G].
    comp_mask marks the completion-token positions in the [L-1] logprob array.
    """
    Tp = len(prompt_ids)
    fulls: list[list[int]] = []
    comp_lens: list[int] = []
    old_seqs: list[list[float]] = []
    for s in samples:
        gen = s["gen_ids"]
        full = (prompt_ids + gen)[:seq_len]
        Lg = len(full) - Tp  # completion tokens actually kept
        fulls.append(full)
        comp_lens.append(max(0, Lg))
        old_seqs.append(s["old_logps"][:Lg])

    if max(comp_lens) == 0:
        return None

    L = max(len(f) for f in fulls)
    G = len(fulls)
    input_ids = torch.full((G, L), pad_id, dtype=torch.long)
    comp_mask = torch.zeros((G, L - 1), dtype=torch.float32)
    old_logp = torch.zeros((G, L - 1), dtype=torch.float32)

    for g, (full, Lg, old) in enumerate(zip(fulls, comp_lens, old_seqs, strict=True)):
        input_ids[g, : len(full)] = torch.tensor(full, dtype=torch.long)
        # completion token full[i] (Tp <= i < Tp+Lg) is predicted at logprob index i-1
        lo, hi = Tp - 1, Tp - 1 + Lg
        comp_mask[g, lo:hi] = 1.0
        if old:
            old_logp[g, lo : lo + len(old)] = torch.tensor(old, dtype=torch.float32)

    return {
        "input_ids": input_ids.to(device),
        "comp_mask": comp_mask.to(device),
        "old_logp": old_logp.to(device),
        "advantages": advantages.to(device),
    }


# -------------------------
# Checkpoint helpers (schema identical to dpo/dpo.py)
# -------------------------
def save_checkpoint_atomic(path: str, obj: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_model_from_ckpt(
    ckpt: dict[str, Any], vocab_size: int, seq_len: int, device: str
) -> tuple[GPT, GPTConfig]:
    cfg_dict = ckpt.get("cfg") or ckpt.get("config")
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("checkpoint missing 'cfg'/'config' dict")
    cfg_dict = dict(cfg_dict)
    cfg_dict["vocab_size"] = vocab_size
    cfg_dict["max_seq_len"] = seq_len
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg).to(device)

    sd = ckpt.get("model")
    if sd is None:
        raise RuntimeError("checkpoint missing 'model'")
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return model, cfg


# -------------------------
# One GRPO step over `groups_per_step` prompts
# -------------------------
def score_and_advantage(
    tok: Tokenizer,
    example: dict[str, Any],
    samples: list[dict[str, Any]],
    reward_fn,
    adv_eps: float,
    normalize_std: bool,
) -> tuple[torch.Tensor, list[float]]:
    rewards = [reward_fn(completion_text(tok, s["gen_ids"]), example) for s in samples]
    r = torch.tensor(rewards, dtype=torch.float32)
    adv = group_advantages(r, eps=adv_eps, normalize_std=normalize_std)
    return adv, rewards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument(
        "--init_ckpt", required=True, help="SFT/distill/DPO checkpoint to init policy+reference"
    )
    ap.add_argument(
        "--ref_ckpt",
        default="",
        help="Optional separate reference checkpoint (default: deep-copy of policy)",
    )

    # Reward
    ap.add_argument(
        "--reward",
        default="nonempty",
        help="reward spec, e.g. 'code:1.0+no_repeat:0.1' (see grpo/rewards.py)",
    )

    # GRPO group sampling
    ap.add_argument("--group_size", type=int, default=8, help="completions sampled per prompt (G)")
    ap.add_argument(
        "--groups_per_step",
        type=int,
        default=4,
        help="prompts per optimizer step (grad-accumulated)",
    )
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--gen_top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--adv_eps", type=float, default=1e-4)
    ap.add_argument(
        "--no_std_norm", action="store_true", help="advantage = r - mean only (skip /std)"
    )

    # Loss
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_coef", type=float, default=0.04)
    ap.add_argument("--loss_reduction", choices=["token_mean", "seq_mean"], default="token_mean")

    # Optim / schedule
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    ap.add_argument("--muon_lr", type=float, default=0.0)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--default_system", default="You are a helpful assistant.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=100)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.max_new_tokens >= args.seq_len:
        raise ValueError("--max_new_tokens must be < --seq_len")
    max_prompt_len = args.seq_len - args.max_new_tokens

    assert_special_token_ids(args.tokenizer_path)
    tok = Tokenizer.from_file(args.tokenizer_path)
    tracker = Tracker(args.out_dir)
    vocab_size = tok.get_vocab_size()
    pad_id = 0
    reward_fn = get_reward_fn(args.reward)
    print(f"[*] reward: {args.reward}")

    print(f"[*] loading policy init from: {args.init_ckpt}")
    init_ckpt = load_ckpt(args.init_ckpt)
    policy, cfg = build_model_from_ckpt(init_ckpt, vocab_size, args.seq_len, device)

    if args.ref_ckpt:
        print(f"[*] loading reference from: {args.ref_ckpt}")
        reference, _ = build_model_from_ckpt(
            load_ckpt(args.ref_ckpt), vocab_size, args.seq_len, device
        )
    else:
        print("[*] reference = frozen deep-copy of the initialized policy")
        reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = build_optimizer(
        policy,
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
    )

    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    print(f"[*] prompts: train={len(train_ds)}")
    print(
        f"[*] completions/step = group_size({args.group_size}) * groups_per_step({args.groups_per_step})"
        f" = {args.group_size * args.groups_per_step}"
    )
    loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_passthrough,
        drop_last=True,
    )

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

    policy.train()
    step = 0
    data_iter = iter(loader)
    t0 = time.time()

    while step < args.max_steps:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)
        optimizer.zero_grad(set_to_none=True)

        agg = {"loss": 0.0, "reward": 0.0, "adv_abs": 0.0, "kl": 0.0, "clipped": 0.0, "std0": 0.0}
        n_groups = 0
        for _ in range(args.groups_per_step):
            try:
                ex = next(data_iter)[0]
            except StopIteration:
                data_iter = iter(loader)
                ex = next(data_iter)[0]

            prompt_ids = encode_prompt(
                tok, ex.get("messages") or [], args.default_system, max_prompt_len
            )
            samples = rollout_group(
                policy,
                prompt_ids,
                args.group_size,
                args.seq_len,
                args.max_new_tokens,
                args.gen_temperature,
                args.gen_top_p,
                device,
            )
            adv, rewards = score_and_advantage(
                tok, ex, samples, reward_fn, args.adv_eps, not args.no_std_norm
            )
            batch = build_group_batch(prompt_ids, samples, adv, args.seq_len, pad_id, device)
            if batch is None:
                continue

            with torch.autocast(
                device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)
            ):
                logits = policy(batch["input_ids"])
            logp = token_logprobs(logits, batch["input_ids"])
            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)
                ):
                    ref_logits = reference(batch["input_ids"])
                ref_logp = token_logprobs(ref_logits, batch["input_ids"])

            loss, m = grpo_loss(
                logp,
                batch["old_logp"],
                ref_logp,
                batch["advantages"],
                batch["comp_mask"],
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
                reduction=args.loss_reduction,
            )
            loss = loss / args.groups_per_step
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            n_groups += 1
            agg["loss"] += float(loss.item())
            agg["reward"] += sum(rewards) / len(rewards)
            agg["adv_abs"] += float(adv.abs().mean().item())
            agg["kl"] += m["kl"]
            agg["clipped"] += m["clipped_frac"]
            agg["std0"] += 1.0 if float(adv.abs().max().item()) == 0.0 else 0.0

        if n_groups == 0:
            step += 1
            continue

        if args.grad_clip > 0:
            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step += 1

        if args.log_every > 0 and step % args.log_every == 0:
            g = n_groups
            print(
                f"[train] step={step} loss={agg['loss']:.4f} "
                f"reward={agg['reward'] / g:.3f} |adv|={agg['adv_abs'] / g:.3f} "
                f"kl={agg['kl'] / g:.4f} clipped={agg['clipped'] / g:.3f} "
                f"zero_std_groups={int(agg['std0'])}/{g} lr={lr:.2e} dt={time.time() - t0:.1f}s"
            )
            tracker.log(
                "train",
                step,
                loss=agg["loss"],
                reward=agg["reward"] / g,
                adv_abs=agg["adv_abs"] / g,
                kl=agg["kl"] / g,
                clipped_frac=agg["clipped"] / g,
                zero_std_frac=agg["std0"] / g,
                lr=lr,
            )
            t0 = time.time()

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt = {
                "step": step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_fp16 else None,
                "cfg": asdict(cfg),
                "args": vars(args),
                "kind": "grpo",
            }
            save_checkpoint_atomic(os.path.join(args.out_dir, f"step_{step:06d}.pt"), ckpt)
            save_checkpoint_atomic(os.path.join(args.out_dir, "latest.pt"), ckpt)
            print(f"[ckpt] saved step_{step:06d}.pt")
            tracker.render()

    tracker.render()
    print("[done]")


if __name__ == "__main__":
    main()
