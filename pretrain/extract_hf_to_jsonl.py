#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract HF dataset (streaming) to simple jsonl: {"text": "..."} per line.

Features:
- streaming=True by default (no full download)
- supports HF config + data_dir
- basic text field fallback
- mode=code: strong cleaning for code snippets
- mode=wiki: trims boilerplate tail sections (References/External links/See also/Category etc.)
- optional exact dedup by sha1
- periodic flush+fsync for safer writes on network filesystems
- --hard_exit: os._exit(0) after printing stats to avoid rare interpreter shutdown aborts
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
from typing import Any, Optional

from datasets import load_dataset
from tqdm import tqdm


# -------------------------
# Text extraction
# -------------------------
def pick_text(ex: dict[str, Any], text_field: str) -> Optional[str]:
    v = ex.get(text_field, None)
    if isinstance(v, str):
        return v
    # fallback fields (common HF schemas)
    for k in ("text", "content", "article", "raw", "completion", "code"):
        v = ex.get(k, None)
        if isinstance(v, str):
            return v
    return None


# -------------------------
# Generic cleaning
# -------------------------
_RE_WS = re.compile(r"[ \t]+")
_RE_REPONAME = re.compile(r"^\s*<reponame>.*?\n", re.IGNORECASE)
_RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # control chars except \n\r\t

def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_CTRL.sub("", t)
    # collapse spaces but keep newlines
    t = "\n".join(_RE_WS.sub(" ", line).rstrip() for line in t.split("\n"))
    return t.strip()


def ascii_ratio(t: str) -> float:
    if not t:
        return 1.0
    n = len(t)
    a = sum(1 for ch in t if ord(ch) < 128)
    return a / n


def non_alnum_symbol_ratio(t: str) -> float:
    if not t:
        return 0.0
    n = len(t)
    sym = sum(1 for ch in t if (not ch.isalnum()) and (not ch.isspace()))
    return sym / n


def looks_like_code(t: str) -> bool:
    needles = (
        "def ", "class ", "import ", "from ", "#include", "public ", "private ",
        "function ", "const ", "let ", "var ", "package ", "namespace ", "using "
    )
    if any(x in t for x in needles):
        return True
    if ("{" in t and "}" in t) or (";" in t and "\n" in t):
        return True
    return False


# -------------------------
# Mode-specific cleaning
# -------------------------
def clean_code(
    t: str,
    *,
    drop_reponame_header: bool,
    min_lines: int,
    max_lines: int,
    max_line_len: int,
    min_ascii_ratio: float,
    max_symbol_ratio: float,
    require_code_signals: bool,
) -> Optional[str]:
    if drop_reponame_header:
        t = _RE_REPONAME.sub("", t)

    t = normalize_text(t)
    if not t:
        return None

    lines = t.split("\n")

    # drop extremely long single lines (often minified / garbage)
    if any(len(line) > max_line_len for line in lines):
        return None

    # trim to avoid giant files; keep head+tail structure
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines - len(head)) :]
        lines = head + ["# ... truncated ..."] + tail
        t = "\n".join(lines).strip()

    if len(lines) < min_lines:
        return None

    if ascii_ratio(t) < min_ascii_ratio:
        return None

    if non_alnum_symbol_ratio(t) > max_symbol_ratio:
        return None

    if require_code_signals and (not looks_like_code(t)):
        return None

    # reject some pathological patterns
    bad_substrings = ("$(N)", "core dumped", "\x00")
    if any(x in t for x in bad_substrings):
        return None

    return t


# Wiki boilerplate trimming:
# Instead of dropping whole pages that contain "References", we cut tail sections.
_WIKI_CUT_PATTERNS = [
    r"\n\s*See also\s*\n",
    r"\n\s*References\s*\n",
    r"\n\s*External links\s*\n",
    r"\n\s*Further reading\s*\n",
    r"\n\s*Notes\s*\n",
]
_WIKI_CUT_RE = re.compile("|".join(_WIKI_CUT_PATTERNS), flags=re.IGNORECASE)

def trim_wiki_tail(t: str) -> str:
    # cut at the earliest occurrence of any boilerplate header
    m = _WIKI_CUT_RE.search(t)
    if m:
        t = t[: m.start()].rstrip()
    return t

def clean_wiki(
    t: str,
    *,
    drop_templates: bool,
    min_paragraphs: int = 2,
) -> Optional[str]:
    t = normalize_text(t)
    if not t:
        return None

    if drop_templates:
        # Trim tail sections first (keeps main content)
        t = trim_wiki_tail(t)
        if not t:
            return None

        low = t.lower()

        # Drop very boilerplate-y stubs/licenses, but be less aggressive than "contains references"
        bad_snippets = (
            "creative commons",
            "this article is",
            "retrieved ",
            "isbn ",
            "citation needed",
            "copyright",
            "licensed under",
        )
        if any(x in low for x in bad_snippets):
            return None

        # drop extremely list-y pages
        if t.count("\n- ") >= 12:
            return None

        # require some paragraph structure
        paras = [p for p in t.split("\n\n") if p.strip()]
        if len(paras) < min_paragraphs:
            return None

    return t


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default=None, help="HF config name (optional)")
    ap.add_argument("--split", default="train", help="split name")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_docs", type=int, default=0, help="stop after N kept docs (0=unlimited)")
    ap.add_argument("--text_field", default="text", help="field to use as text if present")
    ap.add_argument("--min_chars", type=int, default=200, help="drop very short samples (after cleaning)")
    ap.add_argument("--max_chars", type=int, default=0, help="drop very long samples (0=disable)")
    ap.add_argument("--data_dir", default=None, help="HF data_dir (subdirectory), e.g. python for starcoderdata")
    ap.add_argument("--use_auth_token", action="store_true", help="use HF auth token (for gated datasets)")
    ap.add_argument("--mode", choices=["generic", "code", "wiki"], default="generic")

    # code cleaning knobs
    ap.add_argument("--code_drop_reponame", action="store_true", help="remove '<reponame>...\\n' header if present")
    ap.add_argument("--code_require_signals", action="store_true", help="require code-like tokens (def/class/import/{})")
    ap.add_argument("--code_min_lines", type=int, default=6)
    ap.add_argument("--code_max_lines", type=int, default=600)
    ap.add_argument("--code_max_line_len", type=int, default=4000)
    ap.add_argument("--code_min_ascii_ratio", type=float, default=0.90)
    ap.add_argument("--code_max_symbol_ratio", type=float, default=0.45)

    # wiki cleaning
    ap.add_argument("--wiki_drop_templates", action="store_true", help="trim tail boilerplate and drop licensey/listy pages")

    # optional dedup (cheap exact hash)
    ap.add_argument("--dedup", action="store_true", help="drop exact duplicates by sha1(text)")
    ap.add_argument("--dedup_max", type=int, default=2_000_000, help="max hashes kept in memory before clearing")

    # IO safety
    ap.add_argument("--flush_every", type=int, default=10_000, help="flush+fsync every N kept docs (0=disable)")

    # stable exit
    ap.add_argument("--hard_exit", action="store_true",
                    help="force os._exit(0) after printing stats (avoids rare HF streaming shutdown crashes)")

    args = ap.parse_args()
    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # load streaming dataset (compatible auth kwargs)
    load_kwargs = dict(split=args.split, streaming=True)
    if args.data_dir:
        load_kwargs["data_dir"] = args.data_dir
    if args.use_auth_token:
        load_kwargs["use_auth_token"] = True

    ds = load_dataset(args.dataset, args.config, **load_kwargs)

    seen = 0
    kept = 0

    dropped_short = 0
    dropped_long = 0
    dropped_empty = 0
    dropped_bad = 0
    dropped_dup = 0
    no_text = 0
    non_dict = 0

    hashes = set() if args.dedup else None

    pbar_total = args.max_docs if args.max_docs > 0 else None
    pbar = tqdm(total=pbar_total, desc=f"stream {args.dataset}:{args.split} ({args.mode})")

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for ex in ds:
                seen += 1
                if not isinstance(ex, dict):
                    non_dict += 1
                    continue

                t = pick_text(ex, args.text_field)
                if not isinstance(t, str):
                    no_text += 1
                    continue

                if args.mode == "code":
                    t2 = clean_code(
                        t,
                        drop_reponame_header=args.code_drop_reponame,
                        min_lines=args.code_min_lines,
                        max_lines=args.code_max_lines,
                        max_line_len=args.code_max_line_len,
                        min_ascii_ratio=args.code_min_ascii_ratio,
                        max_symbol_ratio=args.code_max_symbol_ratio,
                        require_code_signals=args.code_require_signals,
                    )
                elif args.mode == "wiki":
                    t2 = clean_wiki(t, drop_templates=args.wiki_drop_templates)
                else:
                    t2 = normalize_text(t)

                if not t2:
                    dropped_empty += 1
                    continue

                if len(t2) < args.min_chars:
                    dropped_short += 1
                    continue

                if args.max_chars and len(t2) > args.max_chars:
                    dropped_long += 1
                    continue

                if hashes is not None:
                    h = hashlib.sha1(t2.encode("utf-8")).hexdigest()
                    if h in hashes:
                        dropped_dup += 1
                        continue
                    hashes.add(h)
                    if len(hashes) > args.dedup_max:
                        hashes.clear()

                f.write(json.dumps({"text": t2}, ensure_ascii=False) + "\n")
                kept += 1
                pbar.update(1)

                if args.flush_every and kept % args.flush_every == 0:
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                if args.max_docs and kept >= args.max_docs:
                    break

            # final flush
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    finally:
        try:
            pbar.close()
        except Exception:
            pass
        # help GC / reduce rare interpreter-exit aborts
        try:
            del ds
        except Exception:
            pass
        gc.collect()

    # Print stats BEFORE hard-exit
    print(f"wrote {kept} docs -> {args.out}")
    print(
        json.dumps(
            {
                "seen": seen,
                "kept": kept,
                "no_text": no_text,
                "non_dict": non_dict,
                "dropped_empty": dropped_empty,
                "dropped_short": dropped_short,
                "dropped_long": dropped_long,
                "dropped_bad": dropped_bad,
                "dropped_dup": dropped_dup,
                "mode": args.mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    sys.stdout.flush()

    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
