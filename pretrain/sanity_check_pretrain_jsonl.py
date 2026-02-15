#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sanity_check_pretrain_jsonl.py

Sanity-check cleaned JSONL files for pretraining.

Checks
------
- JSON parse rate, required field exists and is a string
- Basic length distribution
- ASCII ratio distribution
- Newline / indentation preservation metrics:
  - fraction containing '\n\n'
  - fraction containing indentation pattern ('\\n    ' or startswith '    ')
- Noise indicators (should be low after cleaning):
  - placeholder spam patterns
  - URL count
  - HTML tag/entity signals
- Optional tokenizer roundtrip checks on random samples

Usage
-----
python sanity_check_pretrain_jsonl.py \
  --jsonl datasets/pretrain_clean/fineweb_edu.clean.v1.jsonl \
  --field text \
  --max_lines 200000 \
  --sample 5000

Optional tokenizer check:
python sanity_check_pretrain_jsonl.py \
  --jsonl ... \
  --tokenizer petitgpt/tokenizer/tokenizer.json \
  --roundtrip_samples 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

RE_LONG_UNDERS = re.compile(r"_{20,}")
RE_LONG_DOTS = re.compile(r"\.{20,}")
RE_REPEAT_CHAR = re.compile(r"(.)\1{49,}")
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_HTML_TAG = re.compile(r"</?(html|div|span|script|style|head|body|p|br|table|tr|td|ul|li|a)\b", re.I)
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]{2,10};")
RE_ANGLE_TAG = re.compile(r"<[^>]{1,200}>")


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / max(1, len(s))


def quantiles(xs: List[float], qs=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    if not xs:
        return {f"q{int(q*100):02d}": 0.0 for q in qs}
    xs = sorted(xs)
    out = {}
    n = len(xs)
    for q in qs:
        idx = int(round((n - 1) * q))
        out[f"q{int(q*100):02d}"] = float(xs[idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--max_lines", type=int, default=200000, help="Max lines to scan (0 = all).")
    ap.add_argument("--sample", type=int, default=5000, help="How many valid samples to collect for stats.")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--tokenizer", default="", help="Optional tokenizer.json to run encode->decode roundtrip.")
    ap.add_argument("--roundtrip_samples", type=int, default=200)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer  # local import
        tok = Tokenizer.from_file(args.tokenizer)

    p = Path(args.jsonl)

    stats: Dict[str, Any] = {
        "path": str(p),
        "seen_lines": 0,
        "bad_json": 0,
        "missing_field": 0,
        "non_str": 0,
        "valid": 0,
        "collected": 0,
        "has_dblnl_frac": 0.0,
        "has_indent4_frac": 0.0,
        "ascii_ratio_quantiles": {},
        "len_chars_quantiles": {},
        "url_per_kchars_quantiles": {},
        "placeholder_frac": 0.0,
        "html_signal_frac": 0.0,
        "roundtrip_ok_frac": None,
    }

    ascii_ratios: List[float] = []
    lens: List[float] = []
    url_per_k: List[float] = []

    has_dblnl = 0
    has_indent4 = 0
    placeholder_hits = 0
    html_hits = 0

    # store some random samples for roundtrip
    roundtrip_pool: List[str] = []

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["seen_lines"] += 1
            if args.max_lines and stats["seen_lines"] >= args.max_lines:
                break

            try:
                obj = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue

            if not isinstance(obj, dict) or args.field not in obj:
                stats["missing_field"] += 1
                continue

            t = obj.get(args.field)
            if not isinstance(t, str):
                stats["non_str"] += 1
                continue

            stats["valid"] += 1

            # collect stats up to --sample items
            if stats["collected"] < args.sample:
                stats["collected"] += 1
                lens.append(float(len(t)))

                ar = ascii_ratio(t[:20000])
                ascii_ratios.append(ar)

                if "\n\n" in t:
                    has_dblnl += 1
                if "\n    " in t or t.startswith("    "):
                    has_indent4 += 1

                urls = len(RE_URL.findall(t))
                url_per_k.append(1000.0 * urls / max(1.0, float(len(t))))

                if RE_LONG_UNDERS.search(t) or RE_LONG_DOTS.search(t) or RE_REPEAT_CHAR.search(t):
                    placeholder_hits += 1

                if RE_HTML_TAG.search(t) or RE_HTML_ENTITY.search(t) or len(RE_ANGLE_TAG.findall(t)) >= 10:
                    html_hits += 1

                # reservoir-like pool for roundtrip
                if tok is not None:
                    if len(roundtrip_pool) < args.roundtrip_samples:
                        roundtrip_pool.append(t)
                    else:
                        j = rng.randint(0, stats["valid"] - 1)
                        if j < args.roundtrip_samples:
                            roundtrip_pool[j] = t

    n = max(1, stats["collected"])
    stats["has_dblnl_frac"] = has_dblnl / n
    stats["has_indent4_frac"] = has_indent4 / n
    stats["placeholder_frac"] = placeholder_hits / n
    stats["html_signal_frac"] = html_hits / n
    stats["ascii_ratio_quantiles"] = quantiles(ascii_ratios)
    stats["len_chars_quantiles"] = quantiles(lens)
    stats["url_per_kchars_quantiles"] = quantiles(url_per_k)

    # tokenizer roundtrip
    if tok is not None and roundtrip_pool:
        ok = 0
        for t in roundtrip_pool:
            enc = tok.encode(t)
            dec = tok.decode(enc.ids)
            if dec == t:
                ok += 1
        stats["roundtrip_ok_frac"] = ok / len(roundtrip_pool)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
