#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sample.py (cover version)

Fixes vs the buggy stop-string/whitespace-ban behavior:

1) **avoid_first_whitespace** now bans ONLY tokens that decode to *pure whitespace* (" ", "\n", "\t", etc.)
   instead of banning tokens that merely *start* with whitespace (e.g. " in", " a"), which would
   accidentally ban most of the vocabulary and force junk first tokens.

2) **stop_strings** are implemented as proper stopping criteria (token-suffix match + optional
   decoded-text fallback) and are **NOT** added into the banned-id list.

3) No NameError: stop_token_seqs is always defined.

Drop this file into `pretrain/sample.py` (overwrite).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from tokenizers import Tokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "tokenizers is required. Install with: pip install tokenizers"
    ) from e


# ---------------------------
# Cached helpers
# ---------------------------

_WHITESPACE_BAN_CACHE = {}


def _compute_whitespace_only_ids(tok: "Tokenizer", bos_id: int, eos_id: int) -> List[int]:
    """Token ids that decode to only whitespace.

    IMPORTANT: do **not** ban tokens like " in" or " a"; those are normal BPE tokens.
    We only ban tokens whose decoded string is non-empty and becomes empty after .strip().
    """
    vocab = tok.get_vocab()  # {token_str: id}
    key = (len(vocab), bos_id, eos_id)
    cached = _WHITESPACE_BAN_CACHE.get(key)
    if cached is not None:
        return cached

    ids: List[int] = []
    for _t, i in vocab.items():
        if i in (bos_id, eos_id):
            continue
        s = tok.decode([i])
        # Pure whitespace tokens only
        if s and (s.strip() == ""):
            ids.append(int(i))

    ids = sorted(set(ids))
    _WHITESPACE_BAN_CACHE[key] = ids
    return ids


def _compile_stop_token_seqs(tok: "Tokenizer", stop_strings: Optional[Sequence[str]]) -> List[Tuple[str, List[int]]]:
    """Encode stop strings into token-id sequences (suffix match)."""
    if not stop_strings:
        return []
    out: List[Tuple[str, List[int]]] = []
    for s in stop_strings:
        if s is None:
            continue
        s = str(s)
        if s == "":
            continue
        ids = tok.encode(s).ids
        if ids:
            out.append((s, [int(x) for x in ids]))
    return out


def _endswith_seq(full_ids: Sequence[int], seq: Sequence[int]) -> bool:
    if not seq or len(full_ids) < len(seq):
        return False
    return list(full_ids[-len(seq) :]) == list(seq)


@dataclass
class SampleConfig:
    tokenizer_path: str
    max_seq_len: int
    precision: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    min_new_tokens: int
    greedy: bool
    avoid_first_whitespace: bool
    first_whitespace_resample_tries: int
    ban_first_steps: int
    extra_ban_token_ids: Optional[List[int]]
    stop_strings: Optional[List[str]]
    strip_stop: bool
    add_bos: bool
    bos_id: int
    eos_id: int
    seed_base: int


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _sample_one(
    *,
    model: torch.nn.Module,
    tok: "Tokenizer",
    device: torch.device,
    prompt: str,
    cfg: SampleConfig,
    prompt_index: int,
    debug: bool,
) -> str:
    enc = tok.encode(prompt)
    prompt_ids = [int(x) for x in enc.ids]

    # Optionally prepend BOS
    if cfg.add_bos:
        prompt_ids = [int(cfg.bos_id)] + prompt_ids

    # Truncate from left if prompt too long
    if len(prompt_ids) >= cfg.max_seq_len:
        prompt_ids = prompt_ids[-(cfg.max_seq_len - 1) :]

    stop_token_seqs = _compile_stop_token_seqs(tok, cfg.stop_strings)

    whitespace_only_ids = _compute_whitespace_only_ids(tok, cfg.bos_id, cfg.eos_id)

    # Fixed banned ids (do NOT include stop_token_seqs here)
    fixed_banned: List[int] = []
    if cfg.extra_ban_token_ids:
        fixed_banned.extend(int(x) for x in cfg.extra_ban_token_ids)

    # Keep BOS out of sampling unless user explicitly wants it
    fixed_banned.append(int(cfg.bos_id))

    # NOTE: we do NOT ban EOS globally; EOS is allowed but we enforce min_new_tokens

    # Generation
    full_ids: List[int] = list(prompt_ids)
    new_ids: List[int] = []

    # for decoded-text fallback stop check (handles rare tokenizer mismatch)
    stop_texts = [s for s, _ in stop_token_seqs]
    max_stop_len = max([len(s) for s in stop_texts], default=0)

    if debug:
        print("\n[Debug]")
        print(f"bos_id={cfg.bos_id} eos_id={cfg.eos_id}")
        print(
            f"avoid_first_whitespace={cfg.avoid_first_whitespace} ban_first_steps={cfg.ban_first_steps} "
            f"banned_ids_count={(len(set(fixed_banned)) + (len(whitespace_only_ids) if cfg.avoid_first_whitespace else 0))}"
        )
        head = sorted(set(fixed_banned + (whitespace_only_ids if cfg.avoid_first_whitespace else [])))[:32]
        print(f"banned_ids_head={head}")

    for step in range(int(cfg.max_new_tokens)):
        x = torch.tensor([full_ids[-cfg.max_seq_len :]], device=device, dtype=torch.long)
        logits = model(x)  # [1, T, V]
        logits = logits[:, -1, :]  # [1, V]

        # Temperature / greedy
        if cfg.greedy or cfg.temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1).item())
            # enforce bans by falling back to sampling if greedy picks banned
            if next_id in fixed_banned:
                cfg_greedy = False
            else:
                cfg_greedy = True
        else:
            cfg_greedy = False

        # Build banned ids for THIS step
        banned_now = set(fixed_banned)
        if cfg.avoid_first_whitespace and step < int(cfg.ban_first_steps):
            banned_now.update(whitespace_only_ids)
        if len(new_ids) < int(cfg.min_new_tokens):
            banned_now.add(int(cfg.eos_id))

        # Apply bans by setting logits to -inf
        if banned_now:
            logits = logits.clone()
            logits[:, list(banned_now)] = -float("inf")

        if cfg_greedy:
            # already computed next_id above
            pass
        else:
            # sample
            if cfg.temperature > 0:
                logits = logits / float(cfg.temperature)

            probs = torch.softmax(logits, dim=-1)  # [1, V]

            if cfg.top_k and int(cfg.top_k) > 0:
                topk = int(cfg.top_k)
                vals, idx = torch.topk(probs, k=topk, dim=-1)
                probs2 = torch.zeros_like(probs)
                probs2.scatter_(1, idx, vals)
                probs = probs2
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if cfg.top_p and float(cfg.top_p) < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > float(cfg.top_p)
                # keep at least 1 token
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs2 = torch.zeros_like(probs)
                probs2.scatter_(1, sorted_idx, sorted_probs)
                probs = probs2
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_id = int(torch.multinomial(probs, num_samples=1).item())

        full_ids.append(next_id)
        new_ids.append(next_id)

        # Stop conditions (only after min_new_tokens)
        if len(new_ids) >= int(cfg.min_new_tokens):
            # EOS
            if next_id == int(cfg.eos_id):
                break

            # Token-suffix stop strings
            hit_stop: Optional[Tuple[str, List[int]]] = None
            for s, seq in stop_token_seqs:
                if _endswith_seq(full_ids, seq):
                    hit_stop = (s, seq)
                    break

            # Decoded-text fallback: check only the tail
            if hit_stop is None and stop_texts and max_stop_len > 0:
                tail_text = tok.decode(full_ids[-(cfg.max_seq_len) :])
                # only look near the end to avoid O(n^2)
                tail_slice = tail_text[-(max_stop_len + 32) :]
                for s in stop_texts:
                    if s in tail_slice:
                        # best-effort: treat as hit, but we cannot safely map to tokens here
                        hit_stop = (s, [])
                        break

            if hit_stop is not None:
                s, seq = hit_stop
                if cfg.strip_stop and seq:
                    # remove stop tokens from ids
                    del full_ids[-len(seq) :]
                    del new_ids[-len(seq) :]
                break

    out_text = tok.decode(full_ids)

    # If the user asked to strip stop strings but we hit via text-fallback, do a final text strip
    if cfg.strip_stop and cfg.stop_strings:
        for s in cfg.stop_strings:
            if not s:
                continue
            pos = out_text.find(s)
            if pos != -1:
                out_text = out_text[:pos]
                break

    return out_text


def generate_default_samples(
    *,
    model: torch.nn.Module,
    tokenizer_path: str,
    device: torch.device,
    max_seq_len: int,
    precision: str,
    out_path: str | os.PathLike,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    min_new_tokens: int,
    eos_id: int,
    add_bos: bool,
    bos_id: int,
    greedy: bool,
    debug: bool,
    avoid_first_whitespace: bool = True,
    first_whitespace_resample_tries: int = 32,
    ban_first_steps: int = 4,
    extra_ban_token_ids: Optional[List[int]] = None,
    stop_strings: Optional[List[str]] = None,
    strip_stop: bool = False,
    seed_base: int = 1234,
) -> None:
    tok = Tokenizer.from_file(tokenizer_path)

    cfg = SampleConfig(
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        precision=precision,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        greedy=greedy,
        avoid_first_whitespace=avoid_first_whitespace,
        first_whitespace_resample_tries=first_whitespace_resample_tries,
        ban_first_steps=ban_first_steps,
        extra_ban_token_ids=extra_ban_token_ids,
        stop_strings=stop_strings,
        strip_stop=strip_stop,
        add_bos=add_bos,
        bos_id=bos_id,
        eos_id=eos_id,
        seed_base=seed_base,
    )

    prompts = [
        "Once upon a time, a curious child",
        "Paris is the capital of",
        "In linear algebra, an eigenvalue is",
        "Neural networks are a class of machine learning models that",
        "Complete the following Python function:\n\ndef add(a, b):\n    return",
        "Write a Python function that returns the factorial of n:\n\ndef factorial(n):\n    ",
        "Write a Python function that computes Fibonacci numbers recursively:\n\ndef fib(n):\n    ",
        "If all cats are animals and some animals are black, can we conclude that some cats are black?\nAnswer:",
        "John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:",
        "User: I'm feeling stressed lately.\nAssistant:",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Header
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Samples generated with tokenizer={tokenizer_path}\n")
        f.write(
            f"precision={precision}, temperature={temperature}, top_p={top_p}, top_k={top_k}, "
            f"max_new_tokens={max_new_tokens}, min_new_tokens={min_new_tokens}, greedy={greedy}\n"
        )
        f.write(
            f"avoid_first_whitespace={avoid_first_whitespace}, "
            f"first_whitespace_resample_tries={first_whitespace_resample_tries}, extra_ban_token_ids={extra_ban_token_ids}\n"
        )
        f.write(f"stop_strings={stop_strings} strip_stop={strip_stop}\n")
        f.write(f"seed_base={seed_base} (step_tag=None)\n")
        f.write("=" * 80 + "\n")

    # Generation
    for i, p in enumerate(prompts, start=1):
        try:
            _set_seed(int(seed_base) + i)
            out = _sample_one(
                model=model,
                tok=tok,
                device=device,
                prompt=p,
                cfg=cfg,
                prompt_index=i,
                debug=debug,
            )
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"[Prompt {i}] (prompt_tokens={len(tok.encode(p).ids)})\n")
                f.write(p + "\n\n")
                f.write(f"[Full output {i}]\n")
                f.write(out + "\n\n")
                f.write("=" * 80 + "\n")
        except Exception as e:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"[Prompt {i}] (prompt_tokens={len(tok.encode(p).ids)})\n")
                f.write(p + "\n\n")
                f.write("[ERROR]\n")
                f.write(f"prompt_index={i}\n")
                f.write(f"exception={repr(e)}\n\n")
                f.write("=" * 80 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=32)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--stop_string", action="append", default=[])
    ap.add_argument("--strip_stop", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Lazy import your model builder
    from src.model import GPT, GPTConfig  # type: ignore

    cfg_dict = ckpt.get("config", None)
    if cfg_dict is None:
        raise RuntimeError("checkpoint missing config")

    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg).to(device)

    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod.") :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    generate_default_samples(
        model=model,
        tokenizer_path=args.tokenizer_path,
        device=device,
        max_seq_len=args.max_seq_len,
        precision=args.precision,
        out_path=args.out,
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
        stop_strings=(args.stop_string if args.stop_string else None),
        strip_stop=args.strip_stop,
    )


if __name__ == "__main__":
    main()
