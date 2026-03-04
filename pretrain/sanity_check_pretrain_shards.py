#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanity_check_pretrain_shards.py

Quick sanity checks for token-sharded pretrain datasets produced by build_pretrain_shards.py.

What it checks:
1) Reads <out_dir>/meta.json and prints:
   - total train/val tokens, dtype, shard counts
   - per-source train token share vs configured weight (delta)
2) Samples random blocks from .bin shards, decodes with tokenizer, and reports:
   - ascii ratio
   - url density
   - html signal flag
   - placeholder flag
   - has double newline, has 4-space indent
   - (optional) EOS fraction in sampled tokens

Usage:
  python sanity_check_pretrain_shards.py \
    --out_dir datasets/pretrain_mix_7b_v2 \
    --tokenizer_path tokenizer/tokenizer.json \
    --seq_len 2048 \
    --sample_shards 6 \
    --blocks_per_shard 12 \
    --split both \
    --eos_id 3

Outputs a JSON summary to stdout and optionally to --out_json.

Notes:
- This is a *sanity* tool, not a full audit. Keep sample sizes modest.
- It reads shards via numpy memmap (fast, low RAM).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tokenizers import Tokenizer


_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_HTML_RE = re.compile(r"(<\s*(div|p|span|script|style|table|html|body)\b|&nbsp;|</\s*\w+\s*>)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"(lorem ipsum|as an ai language model|\[deleted\]|\[removed\]|click here|subscribe|sign up)",
    re.IGNORECASE,
)

def ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    n = len(s)
    a = sum(1 for ch in s if ord(ch) < 128)
    return a / n

def url_per_kchars(s: str) -> float:
    if not s:
        return 0.0
    k = max(1e-9, len(s) / 1000.0)
    return float(len(_URL_RE.findall(s))) / k

def has_indent4(s: str) -> bool:
    return "\n    " in s or s.startswith("    ")

def has_dblnl(s: str) -> bool:
    return "\n\n" in s

def html_signal(s: str) -> bool:
    return _HTML_RE.search(s) is not None

def placeholder_signal(s: str) -> bool:
    return _PLACEHOLDER_RE.search(s) is not None

def quantiles(xs: List[float], qs=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    if not xs:
        return {f"q{int(q*100):02d}": 0.0 for q in qs}
    arr = np.asarray(xs, dtype=np.float64)
    out: Dict[str, float] = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = float(np.quantile(arr, q))
    return out

def read_meta(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"meta.json not found at: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def dtype_from_meta(meta: Dict[str, Any]) -> np.dtype:
    dt = str(meta.get("dtype", "uint16"))
    if dt.endswith("uint16"):
        return np.uint16
    if dt.endswith("uint32"):
        return np.uint32
    # fallback
    return np.uint16

def list_shards(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.glob("shard_*.bin") if p.is_file()])

def sample_blocks_from_shard(
    shard_path: Path,
    dtype: np.dtype,
    seq_len: int,
    n_blocks: int,
    rng: random.Random,
) -> List[np.ndarray]:
    mm = np.memmap(shard_path, dtype=dtype, mode="r")
    n = int(mm.shape[0])
    if n < seq_len + 1:
        return []
    blocks = []
    # sample random windows
    for _ in range(n_blocks):
        st = rng.randrange(0, n - seq_len)
        blocks.append(np.asarray(mm[st : st + seq_len], dtype=np.int64))
    return blocks

def decode_block(tok: Tokenizer, ids: np.ndarray) -> str:
    # tokenizers expects python list of ints
    return tok.decode(ids.tolist())

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="dataset dir containing meta.json, train/, val/")
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--sample_shards", type=int, default=6)
    ap.add_argument("--blocks_per_shard", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--out_json", default="", help="optional path to write JSON report")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    meta = read_meta(out_dir)
    dtype = dtype_from_meta(meta)

    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    # ---- meta summary ----
    per_src = meta.get("per_source", {})
    train_total = int(meta.get("train_content_tokens", meta.get("train_tokens", 0)))
    val_total = int(meta.get("val_content_tokens", meta.get("val_tokens", 0)))

    src_rows = []
    for src, st in per_src.items():
        w = float(st.get("weight", 0.0))
        tr = int(st.get("train_tokens", 0))
        vr = int(st.get("val_tokens", 0))
        frac = (tr / train_total) if train_total > 0 else 0.0
        src_rows.append((src, w, tr, vr, frac, frac - w))

    src_rows.sort(key=lambda x: -x[2])

    summary = {
        "out_dir": str(out_dir),
        "dtype": str(dtype),
        "vocab_size": int(vocab_size),
        "meta_train_content_tokens": train_total,
        "meta_val_content_tokens": val_total,
        "meta_train_shards": int(meta.get("train_shards", -1)),
        "meta_val_shards": int(meta.get("val_shards", -1)),
        "sources": [
            {
                "source": r[0],
                "weight": r[1],
                "train_tokens": r[2],
                "val_tokens": r[3],
                "train_frac": r[4],
                "train_frac_minus_weight": r[5],
            }
            for r in src_rows
        ],
    }

    # ---- shard sampling ----
    rng = random.Random(args.seed)

    def analyze_split(split_name: str) -> Dict[str, Any]:
        split_dir = out_dir / split_name
        shards = list_shards(split_dir)
        if not shards:
            return {"split": split_name, "n_shards": 0}

        # basic count check
        n_shards = len(shards)
        sample_n = min(args.sample_shards, n_shards)
        pick = rng.sample(shards, k=sample_n) if sample_n > 0 else []

        ascii_rs: List[float] = []
        url_dens: List[float] = []
        lens: List[float] = []
        html_flags = 0
        placeholder_flags = 0
        dblnl_flags = 0
        indent_flags = 0
        eos_fracs: List[float] = []
        oob_tokens = 0
        total_tokens_checked = 0

        for sp in pick:
            blocks = sample_blocks_from_shard(sp, dtype=dtype, seq_len=args.seq_len, n_blocks=args.blocks_per_shard, rng=rng)
            for ids in blocks:
                # token id check (sampled)
                total_tokens_checked += int(ids.size)
                oob_tokens += int(((ids < 0) | (ids >= vocab_size)).sum())

                # eos fraction (sampled)
                if args.eos_id >= 0:
                    eos_fracs.append(float((ids == args.eos_id).mean()))

                s = decode_block(tok, ids)
                lens.append(float(len(s)))
                ar = ascii_ratio(s)
                ascii_rs.append(ar)
                url_dens.append(url_per_kchars(s))
                if html_signal(s):
                    html_flags += 1
                if placeholder_signal(s):
                    placeholder_flags += 1
                if has_dblnl(s):
                    dblnl_flags += 1
                if has_indent4(s):
                    indent_flags += 1

        n_blocks = sample_n * args.blocks_per_shard
        return {
            "split": split_name,
            "n_shards": n_shards,
            "sampled_shards": sample_n,
            "blocks_sampled": n_blocks,
            "token_ids_oob": int(oob_tokens),
            "token_ids_checked": int(total_tokens_checked),
            "ascii_ratio_quantiles": quantiles(ascii_rs),
            "len_chars_quantiles": quantiles(lens),
            "url_per_kchars_quantiles": quantiles(url_dens),
            "html_signal_frac": float(html_flags / max(1, n_blocks)),
            "placeholder_frac": float(placeholder_flags / max(1, n_blocks)),
            "has_dblnl_frac": float(dblnl_flags / max(1, n_blocks)),
            "has_indent4_frac": float(indent_flags / max(1, n_blocks)),
            "eos_frac_quantiles": quantiles(eos_fracs),
        }

    splits: List[str]
    if args.split == "both":
        splits = ["train", "val"]
    else:
        splits = [args.split]

    split_reports = [analyze_split(s) for s in splits]

    report = {
        "summary": summary,
        "split_reports": split_reports,
    }

    txt = json.dumps(report, ensure_ascii=False, indent=2)
    print(txt)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        Path(args.out_json).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
