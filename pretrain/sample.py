#!/usr/bin/env python3
"""Sampling / text generation for petitgpt.

Used by training (train_pretrain.py) to dump periodic samples.

Key fixes:
- Ban whitespace-only tokens at the first decoding step (optionally first N steps)
  to avoid "first token = \\n/space" degeneracy.
- Keep repetition_penalty / no_repeat_ngram_size / max_repeat_token as before.

This only affects sampling quality (not training).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from tokenizers import Tokenizer

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: train_pretrain imports generate_default_samples from this file.


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p). logits: [vocab]."""
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


def _apply_repetition_penalty(logits: torch.Tensor, generated: List[int], penalty: float) -> torch.Tensor:
    """Apply repetition penalty to a 1D logits tensor."""
    if penalty is None or penalty <= 1.0 or not generated:
        return logits
    for tok in set(generated):
        val = logits[tok]
        logits[tok] = val * penalty if val < 0 else val / penalty
    return logits


def _apply_no_repeat_ngram(logits: torch.Tensor, generated: List[int], n: int) -> torch.Tensor:
    """Ban next tokens that would create an already-seen n-gram (single sequence)."""
    if n is None or n <= 0:
        return logits
    if len(generated) < n - 1:
        return logits

    prefix_len = n - 1
    cur_prefix = tuple(generated[-prefix_len:])
    banned: Set[int] = set()

    for i in range(len(generated) - n + 1):
        prefix = tuple(generated[i : i + prefix_len])
        nxt = generated[i + prefix_len]
        if prefix == cur_prefix:
            banned.add(nxt)

    if banned:
        idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, idx, -float("inf"))
    return logits


def _ban_consecutive_repeats(logits: torch.Tensor, history: List[int], max_repeat_token: int) -> torch.Tensor:
    """If the last token repeats too many times consecutively, ban generating it again."""
    if max_repeat_token <= 0 or not history:
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


def _compute_whitespace_only_token_ids(tok: Tokenizer, max_scan: Optional[int] = None) -> List[int]:
    """
    Return token ids whose decoded text is non-empty but strip() becomes empty.
    This catches: " ", "\\n", "\\t", combinations, etc.
    """
    vocab_size = tok.get_vocab_size()
    if max_scan is None:
        max_scan = vocab_size
    max_scan = min(max_scan, vocab_size)

    ws_ids: List[int] = []
    # decoding one token at a time is slow-ish but vocab is small enough (your vocab ~32k)
    for i in range(max_scan):
        s = tok.decode([i])
        if s and s.strip() == "":
            ws_ids.append(i)
    return ws_ids


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_seq_len: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    eos_id: Optional[int] = None,
    add_bos: bool = True,
    bos_id: Optional[int] = None,
    greedy: bool = False,
    debug: bool = False,
    debug_topk: int = 10,
    repetition_penalty: float = 1.10,
    no_repeat_ngram_size: int = 3,
    max_repeat_token: int = 3,
    seed: Optional[int] = 1234,
    # New: first-step hygiene
    ban_first_whitespace: bool = True,
    ban_first_steps: int = 1,  # apply ban on step=0..ban_first_steps-1
    extra_ban_token_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Generate a completion.

    Returns dict:
      - prompt_tokens
      - new_tokens
      - output_text
      - debug (optional)
    """
    if seed is not None:
        _set_seed(seed)

    enc = tokenizer.encode(prompt)
    prompt_ids = enc.ids

    if add_bos:
        if bos_id is None:
            raise ValueError("add_bos=True but bos_id is None")
        if len(prompt_ids) == 0 or prompt_ids[0] != bos_id:
            prompt_ids = [bos_id] + prompt_ids

    # Truncate if prompt too long
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[-(max_seq_len - 1) :]

    ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]  # [1, t]
    generated: List[int] = []

    # Prepare ban list once
    ban_ids: List[int] = []
    if ban_first_whitespace:
        ban_ids.extend(_compute_whitespace_only_token_ids(tokenizer))
    if extra_ban_token_ids:
        ban_ids.extend(list(extra_ban_token_ids))
    # Also avoid sampling BOS/EOS spuriously (EOS handled by stopping; BOS shouldn't appear)
    if bos_id is not None:
        ban_ids.append(bos_id)

    # De-dup
    ban_ids = sorted(set(int(x) for x in ban_ids if x is not None and x >= 0))

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["bos_id"] = bos_id
        dbg["eos_id"] = eos_id
        dbg["ban_first_whitespace"] = ban_first_whitespace
        dbg["ban_first_steps"] = ban_first_steps
        dbg["banned_ids_count"] = len(ban_ids)
        # show a few for sanity
        if ban_ids:
            dbg["banned_ids_head"] = ban_ids[:20]

    for step in range(max_new_tokens):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]

        out = model(ids)
        logits = out["logits"] if isinstance(out, dict) else out
        next_logits = logits[0, -1, :].float()  # [vocab]

        # Apply anti-looping guards
        history = ids[0].tolist()
        next_logits = _apply_repetition_penalty(next_logits, history, repetition_penalty)
        next_logits = _apply_no_repeat_ngram(next_logits, history, no_repeat_ngram_size)
        next_logits = _ban_consecutive_repeats(next_logits, history, max_repeat_token)

        # NEW: prevent first token degeneracy (newline/space/etc)
        if ban_ids and ban_first_steps > 0 and step < ban_first_steps:
            idx = torch.tensor(ban_ids, device=next_logits.device, dtype=torch.long)
            next_logits.index_fill_(0, idx, -float("inf"))

        if debug and step == 0:
            tmp = next_logits / max(temperature, 1e-8)
            tmp = _top_k_top_p_filtering(tmp.clone(), top_k=top_k, top_p=top_p)
            probs = torch.softmax(tmp, dim=-1)
            if eos_id is not None:
                dbg["first_step_eos_prob"] = float(probs[eos_id].item())
            k = min(debug_topk, probs.numel())
            topv, topi = torch.topk(probs, k)
            dbg["first_step_topk_ids"] = topi.cpu().tolist()
            dbg["first_step_topk_probs"] = topv.cpu().tolist()
            # decode top ids to quickly spot whitespace / weird prefix
            try:
                dbg["first_step_topk_text"] = [tokenizer.decode([int(i)]) for i in dbg["first_step_topk_ids"]]
            except Exception:
                pass

        nxt = _sample_next_id(next_logits, temperature=temperature, top_p=top_p, top_k=top_k, greedy=greedy)

        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)

        if eos_id is not None and nxt == eos_id and (step + 1) >= min_new_tokens:
            break

    full_ids = prompt_ids + generated
    out_text = tokenizer.decode(full_ids)

    return {
        "prompt_tokens": len(prompt_ids),
        "new_tokens": generated,
        "output_text": out_text,
        "debug": dbg if debug else None,
    }


def generate_default_samples(
    model: torch.nn.Module,
    tokenizer_path: str,
    device: str,
    max_seq_len: int,
    precision: str,
    out_path: Path,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    eos_id: Optional[int],
    add_bos: bool,
    bos_id: Optional[int],
    min_new_tokens: int,
    greedy: bool = False,
    debug: bool = True,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    max_repeat_token: int = 3,
    # New options (safe defaults)
    ban_first_whitespace: bool = True,
    ban_first_steps: int = 1,
    extra_ban_token_ids: Optional[Sequence[int]] = None,
) -> None:
    """Write a standard set of prompts + completions to out_path."""
    tok = Tokenizer.from_file(tokenizer_path)

    model.eval()
    if precision in ("fp16", "float16"):
        autocast_dtype = torch.float16
    elif precision in ("bf16", "bfloat16"):
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    prompts = [
        "Once upon a time, ",
        "In a distant future, humans and robots ",
        "The following is a news report:\n\n\n",
        "Neural networks are a class of machine learning models that ",
        "Complete the following Python function:\n\ndef add(a, b):\n    return",
        "Write a Python function that computes Fibonacci numbers recursively:\n\ndef fib(n):\n    ",
        "If all cats are animals and some animals are black, can we conclude that some cats are black?\nAnswer:",
        "John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Samples generated with tokenizer={tokenizer_path}\n")
        f.write(
            f"precision={precision}, temperature={temperature}, top_p={top_p}, top_k={top_k}, "
            f"max_new_tokens={max_new_tokens}, min_new_tokens={min_new_tokens}, greedy={greedy}\n"
        )
        f.write(
            f"ban_first_whitespace={ban_first_whitespace}, ban_first_steps={ban_first_steps}, "
            f"extra_ban_token_ids={list(extra_ban_token_ids) if extra_ban_token_ids else None}\n"
        )
        f.write("=" * 80 + "\n")

        base_seed = 1234
        for i, prompt in enumerate(prompts, 1):
            f.write(f"[Prompt {i}] (prompt_tokens={len(tok.encode(prompt).ids) + (1 if add_bos else 0)})\n")
            f.write(prompt + "\n\n")

            with torch.autocast(
                "cuda",
                dtype=autocast_dtype,
                enabled=(autocast_dtype is not None and torch.cuda.is_available()),
            ):
                res = generate(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    device=device,
                    max_seq_len=max_seq_len,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    eos_id=eos_id,
                    add_bos=add_bos,
                    bos_id=bos_id,
                    greedy=greedy,
                    debug=debug,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_repeat_token=max_repeat_token,
                    seed=base_seed + i,
                    ban_first_whitespace=ban_first_whitespace,
                    ban_first_steps=ban_first_steps,
                    extra_ban_token_ids=extra_ban_token_ids,
                )

            if debug and res["debug"]:
                f.write("[Debug]\n")
                d = res["debug"]
                f.write(f"bos_id={d.get('bos_id')} eos_id={d.get('eos_id')}\n")
                f.write(
                    f"ban_first_whitespace={d.get('ban_first_whitespace')} "
                    f"ban_first_steps={d.get('ban_first_steps')} "
                    f"banned_ids_count={d.get('banned_ids_count')}\n"
                )
                if "banned_ids_head" in d:
                    f.write(f"banned_ids_head={d['banned_ids_head']}\n")
                if "first_step_eos_prob" in d:
                    f.write(f"first_step_eos_prob={d['first_step_eos_prob']:.3e}\n")
                if "first_step_topk_ids" in d:
                    f.write(f"first_step_topk_ids={d['first_step_topk_ids']}\n")
                if "first_step_topk_probs" in d:
                    f.write(f"first_step_topk_probs={d['first_step_topk_probs']}\n")
                if "first_step_topk_text" in d:
                    f.write(f"first_step_topk_text={d['first_step_topk_text']}\n")
                f.write("\n")

            new_tokens = res["new_tokens"]
            f.write(f"[New tokens {i}] count={len(new_tokens)}")
            if new_tokens:
                f.write(f" first30={new_tokens[:30]}")
            f.write("\n")
            f.write(f"[Full output {i}]\n{res['output_text']}\n\n")
            f.write("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt file")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--eos_id", type=int, default=None)
    parser.add_argument("--add_bos", action="store_true")
    parser.add_argument("--bos_id", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repetition_penalty", type=float, default=1.10)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    # New
    parser.add_argument("--ban_first_whitespace", action="store_true", help="ban whitespace-only tokens at first step(s)")
    parser.add_argument("--ban_first_steps", type=int, default=1, help="apply first-token ban for first N decoding steps")
    parser.add_argument("--extra_ban_token_ids", type=str, default="", help="comma-separated token ids to ban early")

    args = parser.parse_args()

    from src.model import GPT, GPTConfig  # local import

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(args.device)
    model.eval()

    tok = Tokenizer.from_file(args.tokenizer_path)

    if args.precision == "fp16":
        dtype = torch.float16
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = None

    extra_ids: Optional[List[int]] = None
    if args.extra_ban_token_ids.strip():
        extra_ids = [int(x) for x in args.extra_ban_token_ids.split(",") if x.strip()]

    with torch.autocast("cuda", dtype=dtype, enabled=(dtype is not None and torch.cuda.is_available())):
        res = generate(
            model=model,
            tokenizer=tok,
            prompt=args.prompt,
            device=args.device,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            eos_id=args.eos_id,
            add_bos=args.add_bos,
            bos_id=args.bos_id,
            greedy=args.greedy,
            debug=True,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            seed=args.seed,
            ban_first_whitespace=args.ban_first_whitespace,
            ban_first_steps=args.ban_first_steps,
            extra_ban_token_ids=extra_ids,
        )

    print(res["output_text"])
    if res["debug"] is not None:
        print("\n[debug]", res["debug"])


if __name__ == "__main__":
    main()
