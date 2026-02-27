#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling / text generation for petitgpt.

Used by:
- training (train_pretrain.py) to dump periodic samples via generate_default_samples()
- CLI: python pretrain/sample.py --ckpt ... --out ... [options]

Key features:
- Compatible with ckpt formats produced by this repo (ckpt["config"] + ckpt["model"])
- avoid_first_whitespace: ban whitespace-ish tokens for first N steps
- resample tries if first token still becomes whitespace (defensive)
- repetition_penalty + no_repeat_ngram_size + max_repeat_token (sampling-only quality guards)
- deterministic but varied seeds across prompts / steps
"""

from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------
# Utilities
# -------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p). logits: [vocab]."""
    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        values, _ = torch.topk(logits, k)
        min_keep = values[-1]
        logits = torch.where(logits < min_keep, torch.full_like(logits, -float("inf")), logits)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        sorted_mask = cumprobs > float(top_p)
        sorted_mask[0] = False  # keep at least one

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(0, sorted_indices, sorted_mask)
        logits = torch.where(mask, torch.full_like(logits, -float("inf")), logits)

    return logits


def _apply_repetition_penalty(logits: torch.Tensor, history: List[int], penalty: float) -> torch.Tensor:
    """CTRL-style repetition penalty on logits (1D)."""
    if penalty is None or penalty <= 1.0 or not history:
        return logits
    for tok in set(history):
        v = logits[tok]
        logits[tok] = v * penalty if v < 0 else v / penalty
    return logits


def _apply_no_repeat_ngram(logits: torch.Tensor, history: List[int], n: int) -> torch.Tensor:
    """Ban next tokens that would create an already-seen n-gram (single sequence)."""
    if n is None or n <= 0:
        return logits
    if len(history) < n - 1:
        return logits

    prefix_len = n - 1
    cur_prefix = tuple(history[-prefix_len:])
    banned: set[int] = set()

    for i in range(len(history) - n + 1):
        pref = tuple(history[i : i + prefix_len])
        nxt = history[i + prefix_len]
        if pref == cur_prefix:
            banned.add(nxt)

    if banned:
        idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, idx, -float("inf"))
    return logits


def _ban_consecutive_repeats(logits: torch.Tensor, history: List[int], max_repeat_token: int) -> torch.Tensor:
    """If last token repeats too many times consecutively, ban it."""
    if max_repeat_token is None or max_repeat_token <= 0 or not history:
        return logits
    last = history[-1]
    k = 1
    for i in range(len(history) - 2, -1, -1):
        if history[i] == last:
            k += 1
            if k >= max_repeat_token:
                logits[last] = -float("inf")
                break
        else:
            break
    return logits


def _safe_softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or float(probs.sum().item()) == 0.0:
        # If everything is -inf or numerical issues, fallback to one-hot argmax
        probs = torch.zeros_like(probs)
        probs[int(torch.argmax(logits).item())] = 1.0
    return probs


def _sample_next_id(
    logits_1d: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> int:
    if greedy or temperature <= 0:
        return int(torch.argmax(logits_1d).item())

    logits = logits_1d / max(float(temperature), 1e-8)
    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = _safe_softmax_probs(logits)
    nxt = torch.multinomial(probs, num_samples=1)
    return int(nxt.item())


def _load_ckpt_and_build_model(
    ckpt_path: str,
    device: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Compatible with the checkpoint produced by your train_pretrain.py:
    - ckpt["config"]  (dict)
    - ckpt["model"]   (state_dict)
    Also tries a few common fallbacks.
    """
    from src.model import GPT, GPTConfig  # local import

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected ckpt type: {type(ckpt)}")

    cfg_dict = None
    # your repo format
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg_dict = ckpt["config"]
    # fallback formats
    elif "model_config" in ckpt and isinstance(ckpt["model_config"], dict):
        cfg_dict = ckpt["model_config"]
    elif "args" in ckpt and isinstance(ckpt["args"], dict) and "config" in ckpt["args"]:
        cfg_dict = ckpt["args"]["config"]

    if cfg_dict is None:
        keys = sorted(list(ckpt.keys()))
        raise KeyError(f"Cannot find config in ckpt. Available keys: {keys}")

    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)

    sd = None
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    else:
        # heuristic: first dict that looks like a state_dict
        for k, v in ckpt.items():
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                sd = v
                break

    if sd is None:
        keys = sorted(list(ckpt.keys()))
        raise KeyError(f"Cannot find model state_dict in ckpt. Available keys: {keys}")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def _build_whitespace_ban_ids(tokenizer: Tokenizer) -> List[int]:
    """
    Build a small set of whitespace-ish token ids to ban early.
    We *do not* scan full vocab (too slow). We collect from encodings and a few known ids.
    """
    candidates: set[int] = set()

    # Common whitespace strings -> ids
    for s in [" ", "\n", "\t", "\r", "\n\n", " \n", "\n "]:
        try:
            candidates.update(tokenizer.encode(s).ids)
        except Exception:
            pass

    # Also ban tokens that decode to pure whitespace for a small id window (cheap)
    # This catches things like single-token '\n' if it exists, etc.
    for tid in range(0, 512):
        try:
            txt = tokenizer.decode([tid])
        except Exception:
            continue
        if txt and txt.strip() == "":
            candidates.add(tid)

    return sorted(candidates)

@dataclass
class StopConfig:
    """
    String-level stopping.
    - stop_strings: if any appears in generated text, stop (the stop marker can be removed).
    - stop_on_newline: stop when a newline is generated (first '\\n').
    - include_stop_in_output: whether to keep stop marker in final output text.
    """
    stop_strings: List[str]
    stop_regexes: List["re.Pattern"]
    stop_on_newline: bool = False
    include_stop_in_output: bool = False

def _postprocess_stop(text: str, stop_cfg: StopConfig) -> str:
    """
    If include_stop_in_output is False, truncate text at earliest stop marker.
    """
    if stop_cfg.include_stop_in_output:
        return text
    cut = None
    for s in stop_cfg.stop_strings:
        if not s:
            continue
        i = text.find(s)
        if i >= 0:
            cut = i if cut is None else min(cut, i)
    if cut is not None:
        return text[:cut]
    return text

# -------------------------
# Generation core
# -------------------------

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
    seed: Optional[int] = None,
    # new:
    avoid_first_whitespace: bool = False,
    ban_first_steps: int = 0,
    first_whitespace_resample_tries: int = 32,
    extra_ban_token_ids: Optional[List[int]] = None,
    # stopping:
    stop_strings: Optional[List[str]] = None,
    stop_on_newline: bool = False,
    include_stop_in_output: bool = False,
    stop_regex: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if seed is not None:
        _set_seed(int(seed))

    enc = tokenizer.encode(prompt)
    prompt_ids = enc.ids

    if add_bos:
        if bos_id is None:
            raise ValueError("add_bos=True but bos_id is None")
        if not prompt_ids or prompt_ids[0] != bos_id:
            prompt_ids = [bos_id] + prompt_ids

    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[-(max_seq_len - 1):]

    ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
    generated: List[int] = []

    # stop_cfg = StopConfig(
    #     stop_strings=list(stop_strings or []),
    #     stop_on_newline=bool(stop_on_newline),
    #     include_stop_in_output=bool(include_stop_in_output),
    # )

    # compile regex once
    compiled: List[re.Pattern] = []
    for pat in (stop_regex or []):
        if not pat:
            continue
        compiled.append(re.compile(pat))

    stop_cfg = StopConfig(
        stop_strings=list(stop_strings or []),
        stop_regexes=compiled,
        stop_on_newline=bool(stop_on_newline),
        include_stop_in_output=bool(include_stop_in_output),
    )

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["bos_id"] = bos_id
        dbg["eos_id"] = eos_id
        dbg["avoid_first_whitespace"] = bool(avoid_first_whitespace)
        dbg["ban_first_steps"] = int(ban_first_steps)
        dbg["stop_on_newline"] = bool(stop_cfg.stop_on_newline)
        dbg["stop_strings"] = stop_cfg.stop_strings
        dbg["stop_regex"] = (stop_regex or [])
        dbg["include_stop_in_output"] = bool(stop_cfg.include_stop_in_output)

    whitespace_ban_ids: List[int] = []
    if avoid_first_whitespace and ban_first_steps > 0:
        whitespace_ban_ids = _build_whitespace_ban_ids(tokenizer)
    if extra_ban_token_ids:
        whitespace_ban_ids = sorted(set(whitespace_ban_ids).union(set(extra_ban_token_ids)))

    if debug and avoid_first_whitespace and ban_first_steps > 0:
        dbg["banned_ids_count"] = len(whitespace_ban_ids)
        dbg["banned_ids_head"] = whitespace_ban_ids[:32]

    # incremental decoded after for stop detection (generated part only)
    # gen_text_buf: str = ""
    # stopped_by: Optional[str] = None
    gen_text_buf: str = ""   # generated text only (no prompt)
    stopped_by: Optional[str] = None
    stop_cut: Optional[int] = None  # truncate generated text to this length (in chars)

    for step in range(max_new_tokens):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]

        out = model(ids)
        logits = out["logits"] if isinstance(out, dict) else out
        next_logits = logits[0, -1, :].float()

        history = ids[0].tolist()
        next_logits = _apply_repetition_penalty(next_logits, history, repetition_penalty)
        next_logits = _apply_no_repeat_ngram(next_logits, history, no_repeat_ngram_size)
        next_logits = _ban_consecutive_repeats(next_logits, history, max_repeat_token)

        # Early-step whitespace ban
        if avoid_first_whitespace and step < ban_first_steps and whitespace_ban_ids:
            idx = torch.tensor(whitespace_ban_ids, device=next_logits.device, dtype=torch.long)
            next_logits.index_fill_(0, idx, -float("inf"))

        # Debug preview for first step (after bans/penalties)
        if debug and step == 0:
            tmp = next_logits / max(float(temperature), 1e-8) if temperature > 0 else next_logits.clone()
            tmp = _top_k_top_p_filtering(tmp.clone(), top_k=top_k, top_p=top_p)
            probs = _safe_softmax_probs(tmp)
            if eos_id is not None:
                dbg["first_step_eos_prob"] = float(probs[eos_id].item())
            k = min(int(debug_topk), probs.numel())
            topv, topi = torch.topk(probs, k)
            dbg["first_step_topk_ids"] = topi.detach().cpu().tolist()
            dbg["first_step_topk_probs"] = topv.detach().cpu().tolist()
            try:
                dbg["first_step_topk_text"] = [tokenizer.decode([i]) for i in dbg["first_step_topk_ids"]]
            except Exception:
                dbg["first_step_topk_text"] = None

        # Sample (with extra resample tries if still hits whitespace early)
        if avoid_first_whitespace and step < ban_first_steps and whitespace_ban_ids:
            # We already masked them, but add defensive resampling if tokenizer oddities happen.
            tries = int(first_whitespace_resample_tries)
            local_logits = next_logits
            chosen = None
            for _ in range(max(1, tries)):
                nxt = _sample_next_id(local_logits, temperature, top_p, top_k, greedy)
                if nxt not in set(whitespace_ban_ids):
                    chosen = nxt
                    break
                # ban this one and try again
                local_logits = local_logits.clone()
                local_logits[nxt] = -float("inf")
            if chosen is None:
                chosen = _sample_next_id(next_logits, temperature, top_p, top_k, greedy)
            nxt = int(chosen)
        else:
            nxt = _sample_next_id(next_logits, temperature, top_p, top_k, greedy)

        generated.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)

        # ---- string-level stop detection ----
        if stop_cfg.stop_on_newline or stop_cfg.stop_strings or stop_cfg.stop_regexes:
            # decode only the newly generated token for efficiency
            try:
                piece = tokenizer.decode([nxt])
            except Exception:
                piece = ""
            if piece:
                gen_text_buf += piece

            # stop on newline
            if stopped_by is None and stop_cfg.stop_on_newline and "\n" in gen_text_buf:
                stopped_by = "\n"
                # truncate at first newline unless include_stop_in_output
                if not stop_cfg.include_stop_in_output:
                    stop_cut = gen_text_buf.find("\n")

            # stop on strings
            if stopped_by is None and stop_cfg.stop_strings:
                for s in stop_cfg.stop_strings:
                    if not s:
                        continue
                    if s in gen_text_buf:
                        stopped_by = s
                        if not stop_cfg.include_stop_in_output:
                            stop_cut = gen_text_buf.find(s)
                        break

            # stop on regex (truncate to match end by default)
            if stopped_by is None and stop_cfg.stop_regexes:
                for rx in stop_cfg.stop_regexes:
                    m = rx.search(gen_text_buf)
                    if m is not None:
                        stopped_by = f"re:{rx.pattern}"
                        if not stop_cfg.include_stop_in_output:
                            stop_cut = m.end()
                        break

            if stopped_by is not None and (step + 1) >= min_new_tokens:
                break

        if eos_id is not None and nxt == eos_id and (step + 1) >= min_new_tokens:
            break

    full_ids = prompt_ids + generated
    # build final output with optional truncation
    prompt_text = tokenizer.decode(prompt_ids)
    gen_text = tokenizer.decode(generated)
    if stop_cut is not None:
        gen_text = gen_text[:stop_cut]
    out_text = prompt_text + gen_text

     # if we stopped by string/newline, optionally truncate output_text to remove stop marker
    if stopped_by is not None:
        # out_text includes prompt+generated; we only want to truncate based on generated region.
        # easiest: rebuild final text as prompt_text + processed_generated_text
        prompt_text = tokenizer.decode(prompt_ids)
        gen_text = tokenizer.decode(generated)
        if stop_cfg.stop_on_newline:
            # treat newline as a stop string too
            stop_cfg2 = StopConfig(
                stop_strings=stop_cfg.stop_strings + ["\n"],
                stop_regexes=stop_cfg.stop_regexes,
                stop_on_newline=False,
                include_stop_in_output=stop_cfg.include_stop_in_output,
            )
            gen_text = _postprocess_stop(gen_text, stop_cfg2)
        else:
            gen_text = _postprocess_stop(gen_text, stop_cfg)
        out_text = prompt_text + gen_text

    return {
        "prompt_tokens": len(prompt_ids),
        "new_tokens": generated,
        "completion_text": gen_text,   # IMPORTANT: truncated completion
        "output_text": out_text,
        "debug": dbg if debug else None,
    }


# -------------------------
# Default sampling for training
# -------------------------

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
    # new:
    avoid_first_whitespace: bool = True,
    ban_first_steps: int = 4,
    first_whitespace_resample_tries: int = 32,
    extra_ban_token_ids: Optional[List[int]] = None,
    # stopping (optional, usually leave off for story prompts)
    stop_strings: Optional[List[str]] = None,
    stop_on_newline: bool = False,
    include_stop_in_output: bool = False,
    seed_base: Optional[int] = None,
    step_tag: Optional[int] = None,
) -> None:
    tok = Tokenizer.from_file(tokenizer_path)

    model.eval()
    if precision in ("fp16", "float16"):
        autocast_dtype = torch.float16
    elif precision in ("bf16", "bfloat16"):
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

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

    # Stable-but-varied seed:
    # - If step_tag is given (training), tie seed to step.
    # - Otherwise use seed_base or default.
    if seed_base is None:
        base = 1234
        if step_tag is not None:
            base = int(step_tag) * 1000003 + 868234  # big-ish mix
        seed_base = base

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Samples generated with tokenizer={tokenizer_path}\n")
        f.write(
            f"precision={precision}, temperature={temperature}, top_p={top_p}, top_k={top_k}, "
            f"max_new_tokens={max_new_tokens}, min_new_tokens={min_new_tokens}, greedy={greedy}\n"
        )
        f.write(
            f"avoid_first_whitespace={avoid_first_whitespace}, first_whitespace_resample_tries={first_whitespace_resample_tries}, "
            f"extra_ban_token_ids={extra_ban_token_ids}\n"
        )
        f.write(f"seed_base={seed_base} (step_tag={step_tag})\n")
        f.write("=" * 80 + "\n")

        for i, prompt in enumerate(prompts, 1):
            try:
                f.write(f"[Prompt {i}] (prompt_tokens={len(tok.encode(prompt).ids) + (1 if add_bos else 0)})\n")
                f.write(prompt + "\n\n")
                f.flush()

                device_type = "cuda" if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu"
                with torch.autocast(
                    device_type,
                    dtype=autocast_dtype,
                    enabled=(autocast_dtype is not None and device_type=="cuda"),
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
                        seed=int(seed_base) + i * 9973,  # different per prompt
                        avoid_first_whitespace=avoid_first_whitespace,
                        ban_first_steps=ban_first_steps,
                        first_whitespace_resample_tries=first_whitespace_resample_tries,
                        extra_ban_token_ids=extra_ban_token_ids,
                        stop_strings=stop_strings,
                        stop_on_newline=stop_on_newline,
                        include_stop_in_output=include_stop_in_output,
                    )

                if debug and res.get("debug"):
                    d = res["debug"]
                    f.write("[Debug]\n")
                    f.write(f"bos_id={d.get('bos_id')} eos_id={d.get('eos_id')}\n")
                    f.write(
                        f"avoid_first_whitespace={d.get('avoid_first_whitespace')} "
                        f"ban_first_steps={d.get('ban_first_steps')} "
                        f"banned_ids_count={d.get('banned_ids_count', 0)}\n"
                    )
                    if "banned_ids_head" in d:
                        f.write(f"banned_ids_head={d['banned_ids_head']}\n")
                    if "first_step_eos_prob" in d:
                        f.write(f"first_step_eos_prob={d['first_step_eos_prob']:.3e}\n")
                    if "first_step_topk_ids" in d:
                        f.write(f"first_step_topk_ids={d['first_step_topk_ids']}\n")
                    if "first_step_topk_probs" in d:
                        f.write(f"first_step_topk_probs={d['first_step_topk_probs']}\n")
                    if d.get("first_step_topk_text") is not None:
                        f.write(f"first_step_topk_text={d['first_step_topk_text']}\n")
                    f.write("\n")

                new_tokens = res["new_tokens"]
                f.write(f"[New tokens {i}] count={len(new_tokens)}")
                if new_tokens:
                    f.write(f" first30={new_tokens[:30]}")
                f.write("\n")
                f.write(f"[Full output {i}]\n{res['output_text']}\n\n")
                f.write("=" * 80 + "\n")
                f.flush()

            except Exception as e:
                # Never silently truncate the file: record error and continue.
                f.write("[ERROR]\n")
                f.write(f"prompt_index={i}\n")
                f.write(f"exception={repr(e)}\n\n")
                f.write("=" * 80 + "\n")
                f.flush()


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt file")
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    # generation controls
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=0)

    # tokenizer special ids
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)

    # sampling quality guards
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--max_repeat_token", type=int, default=3)

    # first-token handling
    ap.add_argument("--avoid_first_whitespace", action="store_true")
    ap.add_argument("--ban_first_steps", type=int, default=0)
    ap.add_argument("--first_whitespace_resample_tries", type=int, default=32)
    ap.add_argument("--extra_ban_token_ids", type=str, default=None, help="comma-separated ints, e.g. '202,224'")

    # stopping controls
    ap.add_argument("--stop_on_newline", action="store_true", help="Stop generation when a newline is generated (after min_new_tokens).")
    ap.add_argument("--stop_string", "--stop_strings", dest="stop_strings", action="append", default=None,
                    help="Stop generation when this string appears in generated text. Can be repeated.")
    ap.add_argument("--stop_regex", action="append", default=None,
                    help="Stop when this regex matches generated text (can repeat).")
    ap.add_argument("--include_stop_in_output", action="store_true", help="If set, keep the stop marker in output text (default: truncate it).")

    # mode
    ap.add_argument("--prompt", type=str, default=None, help="If set, generate single completion to stdout.")
    ap.add_argument("--out", type=str, default=None, help="If set, write default prompts to this file (like training).")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print only generated text")

    args = ap.parse_args()

    model, ckpt = _load_ckpt_and_build_model(args.ckpt, device=args.device)

    # dtype / autocast
    if args.precision == "fp16":
        autocast_dtype = torch.float16
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    extra_ban = None
    if args.extra_ban_token_ids:
        extra_ban = [int(x) for x in args.extra_ban_token_ids.split(",") if x.strip()]

    if args.prompt is not None:
        tok = Tokenizer.from_file(args.tokenizer_path)

        device_type = "cuda" if ("cuda" in str(args.device) and torch.cuda.is_available()) else "cpu"
        with torch.autocast(
            device_type,
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None and device_type == "cuda"),
        ):
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
                eos_id=(None if args.eos_id < 0 else args.eos_id),
                add_bos=args.add_bos,
                bos_id=args.bos_id,
                greedy=args.greedy,
                debug=args.debug and (not args.quiet),
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_repeat_token=args.max_repeat_token,
                seed=args.seed,
                avoid_first_whitespace=args.avoid_first_whitespace,
                ban_first_steps=args.ban_first_steps,
                first_whitespace_resample_tries=args.first_whitespace_resample_tries,
                extra_ban_token_ids=extra_ban,
                stop_strings=args.stop_strings,
                stop_on_newline=args.stop_on_newline,
                include_stop_in_output=args.include_stop_in_output,
                stop_regex=args.stop_regex,
            )

        if args.quiet:
            # IMPORTANT: print truncated completion, not raw decoded new_tokens
            print(res.get("completion_text", ""), end="")
        else:
            print(res["output_text"])

        if args.debug and res.get("debug") is not None:
                print("\n[debug]\n" + json.dumps(res["debug"], ensure_ascii=False, indent=2))
        return

    # default sampling to file
    if args.out is None:
        raise SystemExit("Either provide --prompt or --out")

    out_path = Path(args.out)
    generate_default_samples(
        model=model,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        max_seq_len=args.max_seq_len,
        precision=args.precision,
        out_path=out_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        eos_id=(None if args.eos_id < 0 else args.eos_id),
        add_bos=args.add_bos,
        bos_id=args.bos_id,
        min_new_tokens=args.min_new_tokens,
        greedy=args.greedy,
        debug=args.debug,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_repeat_token=args.max_repeat_token,
        avoid_first_whitespace=args.avoid_first_whitespace,
        ban_first_steps=args.ban_first_steps,
        first_whitespace_resample_tries=args.first_whitespace_resample_tries,
        extra_ban_token_ids=extra_ban,
        stop_strings=args.stop_strings,
        stop_on_newline=args.stop_on_newline,
        include_stop_in_output=args.include_stop_in_output,
        stop_regex=args.stop_regex,
        seed_base=args.seed,
        step_tag=ckpt.get("global_step", None),
    )


if __name__ == "__main__":
    main()
