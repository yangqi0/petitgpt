#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean code jsonl ({"text": ...}):

Goal: remove StarCoder-style metadata/header blocks that often start with "<...>"
and dominate first-token distribution (e.g. '<' at ~47%).

Rules:
- Drop leading meta lines matching: ^<[^>]{1,64}>.*  (configurable)
- Remove up to --max_strip_lines lines from the start
- Optionally drop samples still starting with '<' after stripping
- Basic sanity: min/max chars
"""

from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from tqdm import tqdm

META_LINE_RE_DEFAULT = r"^<[^>]{1,64}>.*$"
NULL_RE = re.compile(r"\x00+")

def normalize(t: str) -> str:
    t = t.lstrip("\ufeff")
    t = NULL_RE.sub("", t)
    return t

def strip_meta_block(text: str, meta_line_re: re.Pattern, max_lines: int) -> tuple[str, int]:
    lines = text.splitlines()
    k = 0
    # strip a consecutive block of meta-like lines at the very top
    while k < len(lines) and k < max_lines and meta_line_re.match(lines[k] or ""):
        k += 1
    if k == 0:
        return text, 0
    new = "\n".join(lines[k:]).lstrip("\n")
    return new, k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--meta_line_re", default=META_LINE_RE_DEFAULT)
    ap.add_argument("--max_strip_lines", type=int, default=20)
    ap.add_argument("--drop_if_startswith_lt", action="store_true", default=True)
    ap.add_argument("--min_chars", type=int, default=32)
    ap.add_argument("--max_chars", type=int, default=200000)
    args = ap.parse_args()

    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    meta_re = re.compile(args.meta_line_re)

    stats = {
        "seen": 0,
        "kept": 0,
        "bad_json": 0,
        "dropped_too_short": 0,
        "dropped_too_long": 0,
        "dropped_startswith_lt": 0,
        "stripped_any": 0,
        "total_stripped_lines": 0,
    }

    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with inp.open("r", encoding="utf-8") as rf, tmp.open("w", encoding="utf-8") as wf:
        for line in tqdm(rf, desc=f"clean:{inp.name}"):
            stats["seen"] += 1
            try:
                obj = json.loads(line)
                t = obj.get("text", "")
                if not isinstance(t, str):
                    stats["bad_json"] += 1
                    continue
            except Exception:
                stats["bad_json"] += 1
                continue

            t = normalize(t)

            t2, k = strip_meta_block(t, meta_re, args.max_strip_lines)
            if k > 0:
                stats["stripped_any"] += 1
                stats["total_stripped_lines"] += k
            t = t2

            if args.drop_if_startswith_lt and t.lstrip().startswith("<"):
                stats["dropped_startswith_lt"] += 1
                continue

            if len(t) < args.min_chars:
                stats["dropped_too_short"] += 1
                continue
            if len(t) > args.max_chars:
                stats["dropped_too_long"] += 1
                continue

            wf.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            if stats["kept"] % 20000 == 0:
                wf.flush()
                os.fsync(wf.fileno())

        wf.flush()
        os.fsync(wf.fileno())

    tmp.replace(outp)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
