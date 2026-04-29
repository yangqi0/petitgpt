#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming extractor: HF dataset -> jsonl {"text": ...}

Key features:
- Supports generic text datasets (field auto-detect or --field)
- Supports bigcode/starcoderdata: uses max_stars_repo_path + content
- --code_only_python: keep only .py/.pyw/.ipynb (ipynb extracts code cells)
- Optional lightweight cleaning:
  - strip starcoder header lines like "<reponame>.. <gh_stars>.."
  - trim leading BOM / nulls
  - min/max chars
  - ascii_ratio filter (for web/wiki)
  - head pattern filter (very light; keep code intact by default)
- Safe incremental write + periodic flush
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------

RE_STARCODER_META_LINE = re.compile(r"^<reponame>.*?<gh_stars>\d+\s*$")
RE_NULLS = re.compile(r"\x00+")

PY_EXTS = (".py", ".pyw")
IPYNB_EXTS = (".ipynb",)

# strong non-python needles (extra guard for mis-labeled paths)
NON_PY_NEEDLES = [
    "with Ada.", "package body", "SPARK_Mode", "-- { dg-do",  # Ada/GCC tests
    "using System", "namespace ", "public class",             # C#
    "#include ", "std::",                                     # C/C++
    "console.log(", "function(",                              # JS
    "SELECT ", "CREATE TABLE", "INSERT INTO",                 # SQL
]

PY_SIGNAL_RE = [
    re.compile(r"^\s*def\s+\w+\s*\(", re.M),
    re.compile(r"^\s*class\s+\w+\s*[:(]", re.M),
    re.compile(r"^\s*(from|import)\s+\w+", re.M),
    re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]"),
    re.compile(r"^\s*@\w+", re.M),
    re.compile(r"^\s*async\s+def\s+\w+\s*\(", re.M),
]


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    n_ascii = sum(1 for ch in s if ord(ch) < 128)
    return n_ascii / max(1, len(s))


def strip_starcoder_header(text: str) -> str:
    # starcoderdata content often begins with a line like "<reponame>xxx<gh_stars>0"
    lines = text.splitlines()
    if lines and RE_STARCODER_META_LINE.match(lines[0]):
        return "\n".join(lines[1:]).lstrip("\n")
    return text


def extract_ipynb_code_cells(raw: str) -> Optional[str]:
    # raw is the full .ipynb JSON string. Extract code cell sources.
    try:
        nb = json.loads(raw)
        if not isinstance(nb, dict):
            return None
        cells = nb.get("cells", [])
        if not isinstance(cells, list):
            return None
        out_lines = []
        for c in cells:
            if not isinstance(c, dict):
                continue
            if c.get("cell_type") != "code":
                continue
            src = c.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            if isinstance(src, str) and src.strip():
                out_lines.append(src.rstrip())
        if not out_lines:
            return None
        # separator between cells (keeps some structure but not too noisy)
        return "\n\n# --- cell ---\n\n".join(out_lines).strip() + "\n"
    except Exception:
        return None


def looks_like_python_text(t: str) -> bool:
    # quick reject
    for x in NON_PY_NEEDLES:
        if x in t:
            return False
    # accept if shebang python
    s = t.lstrip()
    if s.startswith("#!/usr/bin/env python") or s.startswith("#!/usr/bin/python"):
        return True
    # require at least 2 python signals to reduce false positives
    hits = sum(1 for r in PY_SIGNAL_RE if r.search(t) is not None)
    return hits >= 2


def normalize_text(t: str) -> str:
    # remove BOM, nulls
    t = t.lstrip("\ufeff")
    t = RE_NULLS.sub("", t)
    return t


def detect_text_field(row: Dict[str, Any]) -> Optional[str]:
    # common candidates
    for k in ("text", "content", "raw", "data"):
        if k in row and isinstance(row[k], str):
            return k
    # fallback: first string field
    for k, v in row.items():
        if isinstance(v, str):
            return k
    return None


# -----------------------------
# Main
# -----------------------------

@dataclass
class Counters:
    seen: int = 0
    kept: int = 0
    bad: int = 0
    dropped: int = 0
    dropped_reason: Dict[str, int] = None

    def __post_init__(self):
        if self.dropped_reason is None:
            self.dropped_reason = {}


def drop(c: Counters, reason: str):
    c.dropped += 1
    c.dropped_reason[reason] = c.dropped_reason.get(reason, 0) + 1


def iter_stream(ds_name: str, split: str, subset: Optional[str]) -> Iterable[Dict[str, Any]]:
    kwargs = {"split": split, "streaming": True}
    if subset:
        if "starcoderdata" in ds_name:
            kwargs["data_dir"] = subset
        else:
            kwargs["name"] = subset
    ds = load_dataset(ds_name, **kwargs)
    return iter(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. HuggingFaceFW/fineweb-edu or bigcode/starcoderdata")
    ap.add_argument("--subset", default="", help="optional subset/config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--field", default="", help="text field name; if empty auto-detect from first row")
    ap.add_argument("--max_docs", type=int, default=0, help="0=all")
    ap.add_argument("--seed", type=int, default=1234)

    # generic filters
    ap.add_argument("--min_chars", type=int, default=32)
    ap.add_argument("--max_chars", type=int, default=200000)
    ap.add_argument("--ascii_ratio_min", type=float, default=0.0, help="e.g. 0.9 for mostly English web/wiki")
    ap.add_argument("--strip_starcoder_meta", action="store_true", help="remove first <reponame>..<gh_stars> line")
    ap.add_argument("--head_trim", action="store_true", help="lstrip leading whitespace/newlines")

    # code-specific
    ap.add_argument("--code_only_python", action="store_true", help="for code datasets: keep only python files")
    ap.add_argument("--code_keep_ipynb", action="store_true", help="if path endswith .ipynb, extract code cells")
    ap.add_argument("--code_require_py_signals", action="store_true", help="extra content heuristic for python")

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # streaming iterator
    subset = args.subset.strip() or None
    it = iter_stream(args.dataset, args.split, subset)

    # peek one row for field detection (and to validate keys)
    try:
        first = next(it)
    except StopIteration:
        print("Empty dataset stream", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Failed to read first row: {e}", file=sys.stderr)
        sys.exit(2)

    field = args.field.strip() or detect_text_field(first)
    if not field:
        print(f"Could not detect text field. keys={list(first.keys())}", file=sys.stderr)
        sys.exit(2)

    # starcoderdata path key
    path_key = None
    if "max_stars_repo_path" in first:
        path_key = "max_stars_repo_path"

    # re-chain the first row back
    def chain_first():
        yield first
        for r in it:
            yield r

    counters = Counters()

    # write
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as wf:
        for row in tqdm(chain_first(), desc=f"extract:{args.dataset}:{args.split}"):
            counters.seen += 1
            try:
                text = row.get(field, "")
                if not isinstance(text, str):
                    drop(counters, "non_str")
                    continue
                text = normalize_text(text)

                # starcoder: strip meta line if requested
                if args.strip_starcoder_meta:
                    text = strip_starcoder_header(text)

                if args.head_trim:
                    text = text.lstrip()

                # code_only_python logic (best for starcoderdata)
                if args.code_only_python:
                    if not path_key:
                        drop(counters, "no_path_key")
                        continue
                    p = row.get(path_key, "")
                    if not isinstance(p, str) or not p:
                        drop(counters, "bad_path")
                        continue
                    pl = p.lower()

                    if pl.endswith(PY_EXTS):
                        pass
                    elif args.code_keep_ipynb and pl.endswith(IPYNB_EXTS):
                        code = extract_ipynb_code_cells(text)
                        if not code:
                            drop(counters, "ipynb_no_code")
                            continue
                        text = code
                    else:
                        drop(counters, "not_py_ext")
                        continue

                    if args.code_require_py_signals and not looks_like_python_text(text):
                        drop(counters, "py_signal_fail")
                        continue

                # generic filters
                tlen = len(text)
                if tlen < args.min_chars:
                    drop(counters, "too_short")
                    continue
                if tlen > args.max_chars:
                    drop(counters, "too_long")
                    continue
                if args.ascii_ratio_min > 0.0 and ascii_ratio(text) < args.ascii_ratio_min:
                    drop(counters, "ascii_low")
                    continue

                wf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                counters.kept += 1

                # periodic flush to reduce data loss on abort
                if counters.kept % 20000 == 0:
                    wf.flush()
                    os.fsync(wf.fileno())

                if args.max_docs and counters.kept >= args.max_docs:
                    break

            except Exception:
                counters.bad += 1
                continue

        wf.flush()
        os.fsync(wf.fileno())

    tmp_path.replace(out_path)

    print("\n=== done ===")
    print(json.dumps({
        "seen": counters.seen,
        "kept": counters.kept,
        "dropped": counters.dropped,
        "bad": counters.bad,
        "dropped_reason": counters.dropped_reason,
        "out": str(out_path),
        "text_field": field,
        "path_key": path_key,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
