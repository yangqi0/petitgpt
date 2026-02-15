#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pretrain_clean_v1.py

Unified, safe cleaning for pretraining JSONL files of format:
  {"text": "..."}  (one JSON object per line)

Design goals
------------
- Keep formatting that matters for LMs: newlines, indentation, code blocks.
- Remove high-noise artifacts: boilerplate headers, HTML blobs, placeholder lines,
  navigation/menu spam, extremely repetitive characters, and control chars.
- Provide reproducibility: exact-dedup by hash and deterministic filtering.
- Output per-file stats in a sidecar JSON report.

This script is intended for *pretraining* corpora. It is stricter than tokenizer training.

What it does
------------
1) Robust JSONL reading (skips malformed lines).
2) Text normalization:
   - Normalize newlines to '\n'
   - Remove BOM/zero-width characters
   - Remove NULLs and other control chars
3) Optional StarCoder-style code metadata stripping:
   - Strip a consecutive top block of lines matching '^<[^>]{1,64}>.*$'
4) Filters:
   - min_chars / max_chars
   - ascii_ratio threshold (disabled for code by default)
   - placeholder / repetition spam (________, .........., etc.)
   - excessive URL density / "menu" boilerplate heuristics
   - HTML-heavy samples (conservative)
5) Exact dedup (SHA1 hash of normalized text)
6) Write cleaned JSONL and write stats report JSON.

Usage
-----
python pretrain_clean_v1.py \
  --in_jsonl datasets/tokenization/fineweb_edu.tok.v1.jsonl \
  --out_jsonl datasets/pretrain_clean/fineweb_edu.clean.v1.jsonl \
  --domain web \
  --min_chars 200 --max_chars 50000 \
  --min_ascii_ratio 0.75 \
  --dedup \
  --report datasets/pretrain_clean/fineweb_edu.clean.v1.report.json

Domains
-------
--domain {web,wiki,cosmo,code,story}
Domain presets only change safe defaults (ascii_ratio, html threshold, etc.).
You can always override flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -----------------------------
# Regex / helpers
# -----------------------------
RE_ZW = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width + BOM
RE_NULL = re.compile(r"\x00+")
RE_CTRL = re.compile(r"[\x01-\x08\x0B\x0C\x0E-\x1F]")  # keep \t \n \r handled separately

# Placeholder / repetition spam
RE_LONG_UNDERS = re.compile(r"_{20,}")
RE_LONG_DOTS = re.compile(r"\.{20,}")
RE_LONG_DASH = re.compile(r"-{30,}")
RE_LONG_EQ = re.compile(r"={30,}")

# Many repeats of a single char (generic)
RE_REPEAT_CHAR = re.compile(r"(.)\1{49,}")  # 50+ repeats

# HTML-ish signals (conservative)
RE_HTML_TAG = re.compile(r"</?(html|div|span|script|style|head|body|p|br|table|tr|td|ul|li|a)\b", re.I)
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]{2,10};")
RE_ANGLE_TAG = re.compile(r"<[^>]{1,200}>")

# URLs and menu-ish boilerplate signals
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_MENU_WORDS = re.compile(
    r"\b(home|about|contact|privacy|terms|cookie|cookies|subscribe|sign in|login|register|newsletter)\b",
    re.I,
)

# StarCoder meta header lines for code
META_LINE_RE_DEFAULT = r"^<[^>]{1,64}>.*$"


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / max(1, len(s))


def normalize_text(t: str) -> str:
    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Remove BOM/zero-width and NULL/control chars
    t = t.lstrip("\ufeff")
    t = RE_ZW.sub("", t)
    t = RE_NULL.sub("", t)
    t = RE_CTRL.sub("", t)
    return t


def strip_starcoder_meta_block(text: str, meta_line_re: re.Pattern, max_lines: int) -> Tuple[str, int]:
    """
    Strip a consecutive block of StarCoder-style meta lines at the very start.
    Mirrors your clean_code_jsonl.py logic. :contentReference[oaicite:2]{index=2}
    """
    lines = text.splitlines()
    k = 0
    while k < len(lines) and k < max_lines and meta_line_re.match(lines[k] or ""):
        k += 1
    if k == 0:
        return text, 0
    new = "\n".join(lines[k:]).lstrip("\n")
    return new, k


def is_html_heavy(t: str, *, max_tag_frac: float) -> bool:
    """
    Conservative HTML detection: if we see lots of tags/entities relative to length.
    """
    if not t:
        return False
    tag_hits = len(RE_ANGLE_TAG.findall(t))
    ent_hits = len(RE_HTML_ENTITY.findall(t))
    # approx: each tag/entity is "one unit"
    score = tag_hits + 0.5 * ent_hits
    frac = score / max(1, len(t))
    # Also treat explicit HTML sections as heavy
    if RE_HTML_TAG.search(t):
        # allow small snippets
        return frac > max_tag_frac * 0.5
    return frac > max_tag_frac


def has_placeholder_spam(t: str) -> bool:
    return bool(
        RE_LONG_UNDERS.search(t)
        or RE_LONG_DOTS.search(t)
        or RE_LONG_DASH.search(t)
        or RE_LONG_EQ.search(t)
        or RE_REPEAT_CHAR.search(t)
    )


def url_density(t: str) -> float:
    if not t:
        return 0.0
    urls = RE_URL.findall(t)
    return len(urls) / max(1, len(t))


def short_line_ratio(t: str, max_len: int = 30) -> float:
    """
    High short-line ratio is typical for menu dumps, navigation lists, etc.
    """
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    short = sum(1 for ln in lines if len(ln) <= max_len)
    return short / len(lines)


@dataclass
class DomainPreset:
    min_ascii_ratio: float
    html_max_tag_frac: float
    url_density_max: float
    short_line_ratio_max: float
    menu_word_min_hits: int


PRESETS: Dict[str, DomainPreset] = {
    "web": DomainPreset(min_ascii_ratio=0.75, html_max_tag_frac=0.020, url_density_max=0.0025, short_line_ratio_max=0.75, menu_word_min_hits=4),
    "wiki": DomainPreset(min_ascii_ratio=0.80, html_max_tag_frac=0.015, url_density_max=0.0020, short_line_ratio_max=0.80, menu_word_min_hits=4),
    "cosmo": DomainPreset(min_ascii_ratio=0.75, html_max_tag_frac=0.020, url_density_max=0.0025, short_line_ratio_max=0.80, menu_word_min_hits=4),
    "story": DomainPreset(min_ascii_ratio=0.80, html_max_tag_frac=0.020, url_density_max=0.0030, short_line_ratio_max=0.85, menu_word_min_hits=4),
    # For code we typically disable ascii filtering; indentation/newlines must be preserved.
    "code": DomainPreset(min_ascii_ratio=0.0, html_max_tag_frac=0.050, url_density_max=0.0100, short_line_ratio_max=0.90, menu_word_min_hits=10),
}


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL with {'text': ...} per line.")
    ap.add_argument("--out_jsonl", required=True, help="Output cleaned JSONL.")
    ap.add_argument("--field", default="text", help="Field name containing text.")
    ap.add_argument("--domain", choices=list(PRESETS.keys()), required=True)
    ap.add_argument("--report", default="", help="Optional JSON report path.")
    ap.add_argument("--seed", type=int, default=1234, help="Reserved (determinism for future extensions).")

    # Length filters
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=50000)

    # Language-ish filter
    ap.add_argument("--min_ascii_ratio", type=float, default=-1.0, help="Override preset if >=0. Else use preset.")
    ap.add_argument("--ascii_check_max_chars", type=int, default=20000, help="Compute ascii_ratio on a prefix for speed.")

    # Dedup
    ap.add_argument("--dedup", action="store_true", help="Exact dedup by SHA1 of normalized text.")

    # HTML / boilerplate
    ap.add_argument("--drop_html_heavy", action="store_true", default=True)
    ap.add_argument("--html_max_tag_frac", type=float, default=-1.0, help="Override preset if >=0. Else use preset.")
    ap.add_argument("--drop_placeholders", action="store_true", default=True)

    # URL/menu spam
    ap.add_argument("--drop_url_spam", action="store_true", default=True)
    ap.add_argument("--url_density_max", type=float, default=-1.0, help="Override preset if >=0. Else use preset.")
    ap.add_argument("--drop_menu_spam", action="store_true", default=True)
    ap.add_argument("--short_line_ratio_max", type=float, default=-1.0, help="Override preset if >=0. Else use preset.")
    ap.add_argument("--menu_word_min_hits", type=int, default=-1, help="Override preset if >=0. Else use preset.")

    # Code meta stripping
    ap.add_argument("--strip_code_meta", action="store_true", help="Strip StarCoder-style '<...>' header blocks (recommended for code).")
    ap.add_argument("--meta_line_re", default=META_LINE_RE_DEFAULT)
    ap.add_argument("--max_strip_lines", type=int, default=20)
    ap.add_argument("--drop_if_startswith_lt", action="store_true", default=False)

    args = ap.parse_args()

    preset = PRESETS[args.domain]
    min_ascii = preset.min_ascii_ratio if args.min_ascii_ratio < 0 else args.min_ascii_ratio
    html_max_tag_frac = preset.html_max_tag_frac if args.html_max_tag_frac < 0 else args.html_max_tag_frac
    url_density_max = preset.url_density_max if args.url_density_max < 0 else args.url_density_max
    short_line_ratio_max = preset.short_line_ratio_max if args.short_line_ratio_max < 0 else args.short_line_ratio_max
    menu_word_min_hits = preset.menu_word_min_hits if args.menu_word_min_hits < 0 else args.menu_word_min_hits

    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    meta_re = re.compile(args.meta_line_re)

    stats: Dict[str, Any] = {
        "in_path": str(inp),
        "out_path": str(outp),
        "domain": args.domain,
        "seen": 0,
        "kept": 0,
        "bad_json": 0,
        "no_field": 0,
        "non_str": 0,
        "dedup_dropped": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_low_ascii": 0,
        "dropped_html_heavy": 0,
        "dropped_placeholders": 0,
        "dropped_url_spam": 0,
        "dropped_menu_spam": 0,
        "dropped_startswith_lt": 0,
        "code_meta_stripped_docs": 0,
        "code_meta_stripped_lines_total": 0,
        "chars_kept_total": 0,
        "ascii_ratio_kept_avg": 0.0,
    }

    seen_hashes = set() if args.dedup else None

    tmp = outp.with_suffix(outp.suffix + ".tmp")

    def should_drop_menu_spam(t: str) -> bool:
        if not t:
            return False
        hits = len(RE_MENU_WORDS.findall(t))
        if hits < menu_word_min_hits:
            return False
        slr = short_line_ratio(t)
        return slr >= short_line_ratio_max

    with inp.open("r", encoding="utf-8") as rf, tmp.open("w", encoding="utf-8") as wf:
        for line in rf:
            if not line.strip():
                continue
            stats["seen"] += 1
            try:
                obj = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue

            if not isinstance(obj, dict) or args.field not in obj:
                stats["no_field"] += 1
                continue

            t = obj.get(args.field)
            if not isinstance(t, str):
                stats["non_str"] += 1
                continue

            t = normalize_text(t)

            # Optional: strip StarCoder-style code meta headers
            if args.strip_code_meta:
                t2, k = strip_starcoder_meta_block(t, meta_re, args.max_strip_lines)
                if k > 0:
                    stats["code_meta_stripped_docs"] += 1
                    stats["code_meta_stripped_lines_total"] += k
                t = t2

            # Keep internal formatting; only trim at ends
            t = t.strip("\n")
            t = t.strip()

            if not t:
                continue

            if args.drop_if_startswith_lt and t.lstrip().startswith("<"):
                stats["dropped_startswith_lt"] += 1
                continue

            # Length filters
            if args.min_chars > 0 and len(t) < args.min_chars:
                stats["dropped_too_short"] += 1
                continue
            if args.max_chars > 0 and len(t) > args.max_chars:
                stats["dropped_too_long"] += 1
                continue

            # Placeholder/repetition spam
            if args.drop_placeholders and has_placeholder_spam(t):
                stats["dropped_placeholders"] += 1
                continue

            # HTML-heavy
            if args.drop_html_heavy and is_html_heavy(t, max_tag_frac=html_max_tag_frac):
                stats["dropped_html_heavy"] += 1
                continue

            # URL spam
            if args.drop_url_spam:
                if url_density(t) > url_density_max:
                    stats["dropped_url_spam"] += 1
                    continue

            # Menu spam (web-ish)
            if args.drop_menu_spam and args.domain in ("web", "wiki", "cosmo"):
                if should_drop_menu_spam(t):
                    stats["dropped_menu_spam"] += 1
                    continue

            # ASCII ratio (language-ish)
            if min_ascii > 0.0:
                probe = t[: args.ascii_check_max_chars]
                ar = ascii_ratio(probe)
                if ar < min_ascii:
                    stats["dropped_low_ascii"] += 1
                    continue
            else:
                ar = None

            # Dedup
            if seen_hashes is not None:
                h = sha1_hex(t)
                if h in seen_hashes:
                    stats["dedup_dropped"] += 1
                    continue
                seen_hashes.add(h)

            wf.write(json.dumps({args.field: t}, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["chars_kept_total"] += len(t)
            if ar is not None:
                # online average
                prev = stats["ascii_ratio_kept_avg"]
                n = stats["kept"]
                stats["ascii_ratio_kept_avg"] = prev + (ar - prev) / max(1, n)

            if stats["kept"] % 20000 == 0:
                wf.flush()
                os.fsync(wf.fileno())

        wf.flush()
        os.fsync(wf.fileno())

    tmp.replace(outp)

    # finalize report
    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
