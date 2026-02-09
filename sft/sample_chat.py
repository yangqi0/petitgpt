#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat sampling for SFT-style prompts.

- Uses the same sampling helpers as pretrain sample.py (top-p/top-k, repetition penalty, no-repeat ngram).
- Renders chat in the SAME template used for SFT:
    [SYS]...[/SYS]
    [USER]...[/USER]
    [ASSISTANT]\n   <-- model continues here

IMPORTANT:
- We encode prompt with add_special_tokens=False, then manually add ONE BOS if needed.
- We make sure the prompt does NOT end with EOS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Optional

import numpy as np
import torch
from tokenizers import Tokenizer

# Make imports work no matter where you run from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


# ---- Chat template (match SFT, no BOS/EOS here) ----
SYS_OPEN = "[SYS]\n"
SYS_CLOSE = "\n[/SYS]\n"
USER_OPEN = "[USER]\n"
USER_CLOSE = "\n[/USER]\n"
ASSIST_OPEN = "[ASSISTANT]\n"
ASSIST_CLOSE = "\n[/ASSISTANT]\n"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p). logits: [vocab]"""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_keep = values[..., -1, None]
        logits = torch.where(logits < min_keep, torch.full_like(logits, -float("inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        sorted_mask = cumprobs > top_p
        sorted_mask[..., 0] = False  # keep at least 1

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = torch.where(mask, torch.full_like(logits, -float("inf")), logits)

    return logits


def _apply_repetition_penalty(logits: torch.Tensor, history: List[int], penalty: float) -> torch.Tensor:
    """Apply repetition penalty to a 1D logits tensor."""
    if penalty is None or penalty <= 1.0 or not history:
        return logits
    for tok in set(history):
        val = logits[tok]
        logits[tok] = val * penalty if val < 0 else val / penalty
    return logits


def _apply_no_repeat_ngram(logits: torch.Tensor, history: List[int], n: int) -> torch.Tensor:
    """Ban next tokens that would create an already-seen n-gram (single sequence)."""
    if n is None or n <= 0:
        return logits
    if len(history) < n - 1:
        return logits

    prefix_len = n - 1
    cur_prefix = tuple(history[-prefix_len:])
    banned = set()
    for i in range(len(history) - n + 1):
        prefix = tuple(history[i : i + prefix_len])
        nxt = history[i + prefix_len]
        if prefix == cur_prefix:
            banned.add(nxt)

    if banned:
        idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, idx, -float("inf"))
    return logits


def _ban_consecutive_repeats(logits: torch.Tensor, history: List[int], max_repeat_token: int) -> torch.Tensor:
    """Prevent too many consecutive repeats of the last token."""
    if max_repeat_token <= 0 or len(history) == 0:
        return logits

    last = history[-1]
    k = 1
    for i in range(len(history) - 2, -1, -1):
        if history[i] == last:
            k += 1
            if k >= max_repeat_token:
                logits[last] = float("-inf")
                break
        else:
            break
    return logits


def _sample_next_id(
    logits_1d: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> int:
    if greedy or temperature <= 0:
        return int(torch.argmax(logits_1d).item())

    logits = logits_1d / max(temperature, 1e-8)
    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)

    if torch.isnan(probs).any() or float(probs.sum().item()) == 0.0:
        return int(torch.argmax(logits_1d).item())

    nxt = torch.multinomial(probs, num_samples=1)
    return int(nxt.item())


def _encode_no_special(tok: Tokenizer, text: str) -> List[int]:
    """
    Encode WITHOUT special tokens if possible.
    tokenizers.Tokenizer.encode supports add_special_tokens in most versions;
    fall back if not available.
    """
    try:
        return tok.encode(text, add_special_tokens=False).ids
    except TypeError:
        # older versions: no keyword, so encode normally and we'll strip later
        return tok.encode(text).ids


def build_chat_prompt(system: str, user: str, add_assistant_open: bool = True) -> str:
    parts = []
    system = (system or "").strip()
    user = (user or "").strip()

    if system:
        parts += [SYS_OPEN, system, SYS_CLOSE]
    parts += [USER_OPEN, user, USER_CLOSE]
    if add_assistant_open:
        parts += [ASSIST_OPEN]
    return "".join(parts)


@torch.no_grad()
def generate_chat(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt_text: str,
    device: str,
    max_seq_len: int,
    bos_id: int,
    eos_id: Optional[int],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    greedy: bool = False,
    repetition_penalty: float = 1.10,
    no_repeat_ngram_size: int = 3,
    max_repeat_token: int = 3,
    seed: Optional[int] = 1234,
    stop_on_assistant_close: bool = True,
) -> dict:
    if seed is not None:
        _set_seed(seed)

    # Encode prompt without specials, then add exactly one BOS
    prompt_ids = _encode_no_special(tok, prompt_text)

    # Ensure exactly one BOS at start
    if len(prompt_ids) == 0 or prompt_ids[0] != bos_id:
        prompt_ids = [bos_id] + prompt_ids

    # Ensure prompt does NOT end with EOS
    if eos_id is not None and len(prompt_ids) > 0 and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]

    # Truncate if prompt is too long (keep room for at least 1 token)
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[-(max_seq_len - 1) :]

    ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]  # [1,t]
    generated: List[int] = []

    # Optional: stop on assistant close tag
    assist_close_ids: List[int] = []
    if stop_on_assistant_close:
        assist_close_ids = _encode_no_special(tok, ASSIST_CLOSE)
        # If the tokenizer would return empty (unlikely), disable this stop
        if len(assist_close_ids) == 0:
            assist_close_ids = []

    for step in range(max_new_tokens):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]

        logits = model(ids)  # your model returns [B,T,V]
        next_logits = logits[0, -1, :].float()

        history = ids[0].tolist()
        next_logits = _apply_repetition_penalty(next_logits, history, repetition_penalty)
        next_logits = _apply_no_repeat_ngram(next_logits, history, no_repeat_ngram_size)
        next_logits = _ban_consecutive_repeats(next_logits, history, max_repeat_token)

        nxt = _sample_next_id(next_logits, temperature=temperature, top_p=top_p, top_k=top_k, greedy=greedy)

        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)

        if eos_id is not None and nxt == eos_id and (step + 1) >= min_new_tokens:
            break

        # Stop if assistant close tag appears at the end (token-level)
        if assist_close_ids:
            if len(generated) >= len(assist_close_ids) and generated[-len(assist_close_ids) :] == assist_close_ids:
                if (step + 1) >= min_new_tokens:
                    break

    full_ids = prompt_ids + generated
    out_text = tok.decode(full_ids)

    return {
        "prompt_text": prompt_text,
        "prompt_tokens": len(prompt_ids),
        "new_tokens": generated,
        "output_text": out_text,
    }


def load_model_from_ckpt(ckpt_path: str, vocab_size: int, seq_len: int, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Support both:
    # - pretrain ckpt: keys include "config" and "model"
    # - sft ckpt: keys include "cfg" (dict) and "model"
    cfg_dict = ckpt.get("config") or ckpt.get("cfg")
    if cfg_dict is None:
        raise RuntimeError("ckpt missing config (expected 'config' or 'cfg')")

    cfg_dict = dict(cfg_dict)
    cfg_dict["vocab_size"] = vocab_size
    cfg_dict["max_seq_len"] = seq_len
    cfg = GPTConfig(**cfg_dict)

    model = GPT(cfg)
    sd = ckpt.get("model")
    if sd is None:
        raise RuntimeError("ckpt missing 'model' state_dict")
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="pretrain or sft checkpoint .pt")
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    ap.add_argument("--system", type=str, default="You are a helpful assistant.")
    ap.add_argument("--user", type=str, required=True, help="user message (single-turn)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repetition_penalty", type=float, default=1.10)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--max_repeat_token", type=int, default=3)
    ap.add_argument("--stop_on_assistant_close", action="store_true", help="stop when [/ASSISTANT] is generated")

    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    model, cfg = load_model_from_ckpt(args.ckpt, vocab_size=vocab_size, seq_len=args.max_seq_len, device=args.device)

    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = None

    prompt = build_chat_prompt(args.system, args.user, add_assistant_open=True)

    with torch.autocast("cuda", dtype=dtype, enabled=(dtype is not None and torch.cuda.is_available())):
        res = generate_chat(
            model=model,
            tok=tok,
            prompt_text=prompt,
            device=args.device,
            max_seq_len=args.max_seq_len,
            bos_id=args.bos_id,
            eos_id=args.eos_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            greedy=args.greedy,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_repeat_token=args.max_repeat_token,
            seed=args.seed,
            stop_on_assistant_close=args.stop_on_assistant_close,
        )

    # Print only assistant continuation (nice UX):
    # We remove the prompt prefix from decoded text if possible.
    out = res["output_text"]
    # naive: find last occurrence of ASSIST_OPEN and print after it
    pos = out.rfind(ASSIST_OPEN)
    if pos != -1:
        print(out[pos + len(ASSIST_OPEN) :].lstrip())
    else:
        print(out)


if __name__ == "__main__":
    main()
