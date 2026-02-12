#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract HF dataset (streaming) to jsonl: {"text": "..."} per line.

Enhancements (best-effort for higher-quality English-centric pretrain):
- Optional underscore placeholder cleanup (e.g. "____" runs).
- Optional ASCII ratio filtering (global) + HEAD ASCII ratio filtering (first N chars).
- Optional "leading noise" trimming.
- Keeps existing mode=code / mode=wiki logic.

Typical usage:
  python pretrain/extract_hf_to_jsonl.py \
    --dataset HuggingFaceFW/fineweb-edu --split train \
    --out datasets/tokenization/fineweb_edu.clean.jsonl \
    --text_field text --max_docs 5000000 --min_chars 200 \
    --mode generic \
    --min_head_ascii_ratio 0.85 --head_chars 256 \
    --min_ascii_ratio 0.90 \
    --collapse_underscores \
    --strip_leading_noise \
    --dedup --hard_exit
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
_RE_UNDERSCORES = re.compile(r"_{4,}")  # collapse long "____" runs
_RE_LEADING_NOISE = re.compile(r"^\s*([^\w<>{}\[\]().,;:'\"/\-\n]{1,32})\s*")  # conservative


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


def normalize_text(
    t: str,
    *,
    collapse_underscores: bool,
    strip_leading_noise: bool,
) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_CTRL.sub("", t)
    t = "\n".join(_RE_WS.sub(" ", line).rstrip() for line in t.split("\n"))
    t = t.strip()

    if collapse_underscores:
        # Replace long underscore runs with a single space (prevents '____' token becoming a "favorite" first token)
        t = _RE_UNDERSCORES.sub(" ", t).strip()

    if strip_leading_noise:
        # Remove a short leading chunk of weird symbols (common in scraped corpora)
        t = _RE_LEADING_NOISE.sub("", t).lstrip()

    return t


def looks_like_code(t: str) -> bool:
    needles = (
        "def ", "class ", "import ", "from ", "#include", "public ", "private ",
        "function ", "const ", "let ", "var ", "package ", "namespace ", "using ",
        "return ", "if ", "else:", "elif ", "try:", "except", "for ", "while ",
    )
    if any(x in t for x in needles):
        return True
    if ("{" in t and "}" in t) or (";" in t and "\n" in t):
        return True
    if t.count("\n") >= 8 and (t.count("(") + t.count(")") >= 6):
        return True
    return False


# -------------------------
# Mode-specific cleaning: CODE
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
    max_non_ascii_lines_ratio: float,
    collapse_underscores: bool,
    strip_leading_noise: bool,
) -> Optional[str]:
    if drop_reponame_header:
        t = _RE_REPONAME.sub("", t)

    t = normalize_text(t, collapse_underscores=collapse_underscores, strip_leading_noise=strip_leading_noise)
    if not t:
        return None

    lines = t.split("\n")

    if any(len(line) > max_line_len for line in lines):
        return None

    if lines:
        non_ascii_lines = 0
        for ln in lines[: min(len(lines), 200)]:
            if ascii_ratio(ln) < 0.85:
                non_ascii_lines += 1
        if non_ascii_lines / max(1, min(len(lines), 200)) > max_non_ascii_lines_ratio:
            return None

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

    bad_substrings = (
        "$(N)", "core dumped", "\x00", "PyGILState_Release",
        "terminate called without an active exception",
    )
    if any(x in t for x in bad_substrings):
        return None

    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.08:
        return None

    return t


# -------------------------
# Mode-specific cleaning: WIKI
# -------------------------
_WIKI_CUT_PATTERNS = [
    r"\n\s*See also\s*\n",
    r"\n\s*References\s*\n",
    r"\n\s*External links\s*\n",
    r"\n\s*Further reading\s*\n",
    r"\n\s*Notes\s*\n",
    r"\n\s*Bibliography\s*\n",
    r"\n\s*Sources\s*\n",
]
_WIKI_CUT_RE = re.compile("|".join(_WIKI_CUT_PATTERNS), flags=re.IGNORECASE)


def trim_wiki_tail(t: str) -> str:
    m = _WIKI_CUT_RE.search(t)
    if m:
        t = t[: m.start()].rstrip()
    return t


def clean_wiki(
    t: str,
    *,
    drop_templates: bool,
    min_paragraphs: int,
    collapse_underscores: bool,
    strip_leading_noise: bool,
) -> Optional[str]:
    t = normalize_text(t, collapse_underscores=collapse_underscores, strip_leading_noise=strip_leading_noise)
    if not t:
        return None

    t = trim_wiki_tail(t)
    if not t:
        return None

    if drop_templates:
        low = t.lower()
        bad_snippets = (
            "creative commons",
            "this article is",
            "retrieved ",
            "isbn ",
            "citation needed",
            "licensed under",
            "copyright",
        )
        if any(x in low for x in bad_snippets):
            return None

        if t.count("\n- ") >= 12:
            return None

        paras = [p for p in t.split("\n\n") if p.strip()]
        if len(paras) < min_paragraphs:
            return None

    return t


# -------------------------
# HF streaming loader: auth compat
# -------------------------
def load_streaming_dataset(dataset: str, config: Optional[str], split: str, data_dir: Optional[str], use_auth: bool):
    kwargs = dict(split=split, streaming=True)
    if data_dir:
        kwargs["data_dir"] = data_dir

    if use_auth:
        try:
            return load_dataset(dataset, config, token=True, **kwargs)  # type: ignore
        except TypeError:
            return load_dataset(dataset, config, use_auth_token=True, **kwargs)  # type: ignore

    return load_dataset(dataset, config, **kwargs)


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

    # global filtering knobs (generic/wiki)
    ap.add_argument("--min_ascii_ratio", type=float, default=0.90, help="drop docs with ascii_ratio < this (generic/wiki)")
    ap.add_argument("--min_head_ascii_ratio", type=float, default=0.85, help="drop docs if head ascii_ratio < this")
    ap.add_argument("--head_chars", type=int, default=256, help="head chars for head-ascii check")
    ap.add_argument("--max_symbol_ratio", type=float, default=0.55, help="drop docs if non_alnum_symbol_ratio > this")

    # leading noise / placeholder cleanup
    ap.add_argument("--collapse_underscores", action="store_true", help="collapse long ____ runs to a space")
    ap.add_argument("--strip_leading_noise", action="store_true", help="strip a short weird-symbol prefix")

    # code cleaning knobs
    ap.add_argument("--code_drop_reponame", action="store_true", help="remove '<reponame>...\\n' header if present")
    ap.add_argument("--code_require_signals", action="store_true", help="require code-like tokens (def/class/import/{})")
    ap.add_argument("--code_min_lines", type=int, default=6)
    ap.add_argument("--code_max_lines", type=int, default=600)
    ap.add_argument("--code_max_line_len", type=int, default=4000)
    ap.add_argument("--code_min_ascii_ratio", type=float, default=0.90)
    ap.add_argument("--code_max_symbol_ratio", type=float, default=0.45)
    ap.add_argument("--code_max_non_ascii_lines_ratio", type=float, default=0.25)

    # wiki cleaning
    ap.add_argument("--wiki_drop_templates", action="store_true", help="trim tail boilerplate + drop obvious license/listy pages")
    ap.add_argument("--wiki_min_paragraphs", type=int, default=2)

    # optional dedup
    ap.add_argument("--dedup", action="store_true", help="drop exact duplicates by sha1(text)")
    ap.add_argument("--dedup_max", type=int, default=2_000_000, help="max hashes kept in memory before clearing")

    # IO safety
    ap.add_argument("--flush_every", type=int, default=10_000, help="flush+fsync every N kept docs (0=disable)")

    # stable exit
    ap.add_argument(
        "--hard_exit",
        action="store_true",
        help="force os._exit(0) after printing stats (avoids rare HF streaming shutdown crashes)",
    )

    args = ap.parse_args()
    random.seed(args.seed)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ds = load_streaming_dataset(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        data_dir=args.data_dir,
        use_auth=args.use_auth_token,
    )

    seen = 0
    kept = 0

    dropped_short = 0
    dropped_long = 0
    dropped_empty = 0
    dropped_bad = 0
    dropped_dup = 0
    dropped_ascii = 0
    dropped_head_ascii = 0
    dropped_symbol = 0
    no_text = 0
    non_dict = 0

    hashes = set() if args.dedup else None

    pbar_total = args.max_docs if args.max_docs > 0 else None
    pbar = tqdm(total=pbar_total, desc=f"stream {args.dataset}:{args.split} ({args.mode})")

    def global_filters(t: str) -> bool:
        # head-ascii first (targets weird prefixes)
        head = t[: max(1, args.head_chars)]
        if ascii_ratio(head) < args.min_head_ascii_ratio:
            return False
        if ascii_ratio(t) < args.min_ascii_ratio:
            return False
        if non_alnum_symbol_ratio(t) > args.max_symbol_ratio:
            return False
        return True

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
                        max_non_ascii_lines_ratio=args.code_max_non_ascii_lines_ratio,
                        collapse_underscores=args.collapse_underscores,
                        strip_leading_noise=args.strip_leading_noise,
                    )
                elif args.mode == "wiki":
                    t2 = clean_wiki(
                        t,
                        drop_templates=args.wiki_drop_templates,
                        min_paragraphs=args.wiki_min_paragraphs,
                        collapse_underscores=args.collapse_underscores,
                        strip_leading_noise=args.strip_leading_noise,
                    )
                else:
                    t2 = normalize_text(
                        t,
                        collapse_underscores=args.collapse_underscores,
                        strip_leading_noise=args.strip_leading_noise,
                    )

                if not t2:
                    dropped_empty += 1
                    continue

                # generic/wiki filters
                if args.mode in ("generic", "wiki"):
                    head = t2[: max(1, args.head_chars)]
                    if ascii_ratio(head) < args.min_head_ascii_ratio:
                        dropped_head_ascii += 1
                        continue
                    if ascii_ratio(t2) < args.min_ascii_ratio:
                        dropped_ascii += 1
                        continue
                    if non_alnum_symbol_ratio(t2) > args.max_symbol_ratio:
                        dropped_symbol += 1
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
        try:
            del ds
        except Exception:
            pass
        gc.collect()

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
                "dropped_ascii": dropped_ascii,
                "dropped_head_ascii": dropped_head_ascii,
                "dropped_symbol": dropped_symbol,
                "mode": args.mode,
                "min_ascii_ratio": args.min_ascii_ratio,
                "min_head_ascii_ratio": args.min_head_ascii_ratio,
                "head_chars": args.head_chars,
                "max_symbol_ratio": args.max_symbol_ratio,
                "collapse_underscores": bool(args.collapse_underscores),
                "strip_leading_noise": bool(args.strip_leading_noise),
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
