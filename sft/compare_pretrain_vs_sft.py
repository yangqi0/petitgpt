#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple
import sys
import time
import os

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer  # pip install tokenizers

# model
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GPT, GPTConfig


def load_tokenizer(tok_path: str) -> Tokenizer:
    return Tokenizer.from_file(tok_path)


def infer_vocab_size_from_tokenizer_json(path: str) -> int:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    model = obj.get("model", {})
    vocab = model.get("vocab", None)
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return len(vocab)
    added = obj.get("added_tokens", [])
    if isinstance(added, list) and added:
        return max(int(t.get("id", -1)) for t in added) + 1
    raise ValueError(f"Cannot infer vocab_size from tokenizer.json: {path}")


@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float, top_k: int
) -> int:
    # logits: [V]
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)

    # top_k
    if top_k and top_k > 0:
        topk_probs, topk_idx = torch.topk(probs, k=min(top_k, probs.numel()))
        probs2 = torch.zeros_like(probs)
        probs2[topk_idx] = topk_probs
        probs = probs2 / probs2.sum().clamp_min(1e-12)

    # top_p (nucleus)
    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        # always keep at least 1
        keep[..., 0] = True
        kept_idx = sorted_idx[keep]
        probs2 = torch.zeros_like(probs)
        probs2[kept_idx] = probs[kept_idx]
        probs = probs2 / probs2.sum().clamp_min(1e-12)

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_id: int,
    bos_id: int,
    add_bos: bool = False,
) -> str:
    model.eval()

    ids = tok.encode(prompt).ids
    if add_bos:
        ids = [bos_id] + ids

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # [1, T]

    # truncate if too long
    if x.shape[1] > max_seq_len:
        x = x[:, -max_seq_len:]

    out = x
    for step in range(max_new_tokens):
        if out.shape[1] >= max_seq_len:
            break
        logits = model(out)[:, -1, :]  # [1, V]
        nxt = sample_next_token(logits[0], temperature, top_p, top_k)
        out = torch.cat([out, torch.tensor([[nxt]], device=device)], dim=1)
        if step + 1 >= min_new_tokens and nxt == eos_id:
            break

    # decode: strip possible leading BOS and trailing EOS for readability
    out_ids = out[0].tolist()
    if out_ids and out_ids[0] == bos_id:
        out_ids = out_ids[1:]
    if out_ids and out_ids[-1] == eos_id:
        out_ids = out_ids[:-1]
    return tok.decode(out_ids)


def load_ckpt_model(
    ckpt_path: str, *, tok_path: str, seq_len: int, device: torch.device
) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod.") :]: v for k, v in state.items()}

    vocab_size = infer_vocab_size_from_tokenizer_json(tok_path)

    # NOTE: 这里假设你的 120m 配置固定（12L/768/12H/3072）。
    # 如果你以后改了结构，建议从 ckpt["config"] 里读（如果有的话）。
    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_layers=12,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        max_seq_len=seq_len,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = GPT(cfg).to(device)
    model.load_state_dict(state, strict=True)
    return model


def default_prompts() -> List[str]:
    return [
        "Write a short story about a small robot who learns to be kind.",
        "Explain gradient descent in simple terms.",
        "Given a list of integers, how do you find the maximum subarray sum?",
        "Here is a Python function header. Implement Fibonacci with memoization:\n\ndef fib(n: int) -> int:\n    ...\n",
        "User: I'm feeling stressed lately.\nAssistant:",
        "What is the difference between TCP and UDP?",
        "Summarize the following in 2 sentences:\n\nLarge language models can be aligned using SFT and preference optimization.",
        "Solve: If f(x)=x^2+1, what is f(3)?",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--pretrain_ckpt", required=True)
    ap.add_argument("--sft_ckpt", required=True)
    ap.add_argument("--out", default="compare.txt")
    ap.add_argument("--seq_len", type=int, default=1024)

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=32)

    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--add_bos", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.tokenizer_path)

    pre = load_ckpt_model(
        args.pretrain_ckpt, tok_path=args.tokenizer_path, seq_len=args.seq_len, device=device
    )
    sft = load_ckpt_model(
        args.sft_ckpt, tok_path=args.tokenizer_path, seq_len=args.seq_len, device=device
    )

    prompts = default_prompts()
    out_lines: List[str] = []
    out_lines.append(f"tokenizer={args.tokenizer_path}")
    out_lines.append(f"pretrain={args.pretrain_ckpt}")
    out_lines.append(f"sft={args.sft_ckpt}")
    out_lines.append(
        f"sampling: temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"min_new={args.min_new_tokens} max_new={args.max_new_tokens}"
    )
    out_lines.append("=" * 80)

    for i, p in enumerate(prompts):
        pre_txt = generate(
            pre,
            tok,
            p,
            device=device,
            max_seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_id=args.eos_id,
            bos_id=args.bos_id,
            add_bos=args.add_bos,
        )
        sft_txt = generate(
            sft,
            tok,
            p,
            device=device,
            max_seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_id=args.eos_id,
            bos_id=args.bos_id,
            add_bos=args.add_bos,
        )

        out_lines.append(f"[{i}] PROMPT:\n{p}")
        out_lines.append("-" * 80)
        out_lines.append("[PRETRAIN]\n" + pre_txt)
        out_lines.append("-" * 80)
        out_lines.append("[SFT]\n" + sft_txt)
        out_lines.append("=" * 80)

    Path(args.out).write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
