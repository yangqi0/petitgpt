#!/usr/bin/env python3

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
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.special_tokens import assert_tokenizer_contract  # noqa: E402

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


def quantiles(xs: list[float], qs=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)) -> dict[str, float]:
    if not xs:
        return {f"q{int(q * 100):02d}": 0.0 for q in qs}
    xs = sorted(xs)
    out = {}
    n = len(xs)
    for q in qs:
        idx = int(round((n - 1) * q))
        out[f"q{int(q * 100):02d}"] = float(xs[idx])
    return out


def reservoir_add(
    reservoir: list[tuple[int, str]],
    item: tuple[int, str],
    *,
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    """Add one item to a uniform deterministic reservoir over all valid rows."""
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < capacity:
        reservoir[replacement] = item


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--max_lines", type=int, default=200000, help="Max lines to scan (0 = all).")
    ap.add_argument("--sample", type=int, default=5000, help="How many valid samples to collect for stats.")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--tokenizer", default="", help="Optional tokenizer.json to run encode->decode roundtrip.")
    ap.add_argument("--roundtrip_samples", type=int, default=200)
    ap.add_argument("--out_json", default="", help="Optional atomically written JSON report.")

    args = ap.parse_args(argv)
    if args.max_lines < 0 or args.sample < 0 or args.roundtrip_samples < 0:
        raise ValueError("--max_lines, --sample, and --roundtrip_samples must be non-negative")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    stats_rng = random.Random(args.seed)
    roundtrip_rng = random.Random(args.seed ^ 0x9E3779B9)

    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer  # local import
        assert_tokenizer_contract(args.tokenizer)
        tok = Tokenizer.from_file(args.tokenizer)
        tok.encode_special_tokens = True

    p = Path(args.jsonl).expanduser().resolve()

    stats: dict[str, Any] = {
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

    ascii_ratios: list[float] = []
    lens: list[float] = []
    url_per_k: list[float] = []

    has_dblnl = 0
    has_indent4 = 0
    placeholder_hits = 0
    html_hits = 0

    sample_pool: list[tuple[int, str]] = []
    roundtrip_pool: list[tuple[int, str]] = []

    with p.open("r", encoding="utf-8", errors="strict") as f:
        for physical_line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if args.max_lines and stats["seen_lines"] >= args.max_lines:
                break
            stats["seen_lines"] += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
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

            item = (physical_line_number, t)
            reservoir_add(
                sample_pool,
                item,
                seen=stats["valid"],
                capacity=args.sample,
                rng=stats_rng,
            )
            if tok is not None:
                reservoir_add(
                    roundtrip_pool,
                    item,
                    seen=stats["valid"],
                    capacity=args.roundtrip_samples,
                    rng=roundtrip_rng,
                )

    stats["collected"] = len(sample_pool)
    stats["sample_line_numbers"] = sorted(line for line, _ in sample_pool)
    for _, t in sample_pool:
        lens.append(float(len(t)))
        ascii_ratios.append(ascii_ratio(t[:20000]))
        has_dblnl += "\n\n" in t
        has_indent4 += "\n    " in t or t.startswith("    ")
        urls = len(RE_URL.findall(t))
        url_per_k.append(1000.0 * urls / max(1.0, float(len(t))))
        placeholder_hits += bool(
            RE_LONG_UNDERS.search(t)
            or RE_LONG_DOTS.search(t)
            or RE_REPEAT_CHAR.search(t)
        )
        html_hits += bool(
            RE_HTML_TAG.search(t)
            or RE_HTML_ENTITY.search(t)
            or len(RE_ANGLE_TAG.findall(t)) >= 10
        )
    n = max(1, len(sample_pool))
    stats["has_dblnl_frac"] = has_dblnl / n
    stats["has_indent4_frac"] = has_indent4 / n
    stats["placeholder_frac"] = placeholder_hits / n
    stats["html_signal_frac"] = html_hits / n
    stats["ascii_ratio_quantiles"] = quantiles(ascii_ratios)
    stats["len_chars_quantiles"] = quantiles(lens)
    stats["url_per_kchars_quantiles"] = quantiles(url_per_k)

    # tokenizer roundtrip
    stats["roundtrip_checked"] = len(roundtrip_pool)
    stats["roundtrip_ok"] = 0
    stats["roundtrip_sample_line_numbers"] = sorted(line for line, _ in roundtrip_pool)
    if tok is not None and roundtrip_pool:
        ok = 0
        for _, t in roundtrip_pool:
            enc = tok.encode(t, add_special_tokens=False)
            dec = tok.decode(enc.ids, skip_special_tokens=False)
            if dec == t:
                ok += 1
        stats["roundtrip_ok"] = ok
        stats["roundtrip_ok_frac"] = ok / len(roundtrip_pool)

    if args.out_json:
        write_json_atomic(Path(args.out_json), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return stats


if __name__ == "__main__":
    main()
