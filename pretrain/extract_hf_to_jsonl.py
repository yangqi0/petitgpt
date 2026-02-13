#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract HF dataset (streaming) to jsonl: {"text": "..."} per line.

Best-effort for better first-token behavior:
- collapse long "____" runs
- optional normalize fancy quotes to ASCII
- optional strip a short weird-symbol prefix
- stronger head-pattern filters for generic/wiki (drop markdown tables / listy boilerplate)
- stronger filters for code (drop markdown/readme, drop html-ish, keep real code)

This file is an overwrite-enhanced version of your current script.  :contentReference[oaicite:1]{index=1}
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
from typing import Any, Optional, Tuple

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
# Generic cleaning helpers
# -------------------------
_RE_WS = re.compile(r"[ \t]+")
_RE_REPONAME = re.compile(r"^\s*<reponame>.*?\n", re.IGNORECASE)
_RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # control chars except \n\r\t
_RE_UNDERSCORES = re.compile(r"_{4,}")
_RE_LEADING_NOISE = re.compile(r"^\s*([^\w<>{}\[\]().,;:'\"/\-\n#|>]{1,32})\s*")

# markdown-ish head patterns
_RE_MD_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_RE_MD_TABLE = re.compile(r"^\s*\|.+\|\s*$")
_RE_DASH_LIST = re.compile(r"^\s*-\s+\S")
_RE_STAR_LIST = re.compile(r"^\s*\*\s+\S")
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s+\S")

# html-ish head patterns
_RE_HTMLISH = re.compile(r"^\s*<\s*/?\s*[a-zA-Z]{1,12}[\s>]")


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


def normalize_quotes_ascii(t: str) -> str:
    # very mild; helps reduce fancy-quote first-token attraction
    repl = {
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "«": '"', "»": '"',
    }
    return "".join(repl.get(ch, ch) for ch in t)


def normalize_text(
    t: str,
    *,
    collapse_underscores: bool,
    strip_leading_noise: bool,
    normalize_quotes: bool,
) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _RE_CTRL.sub("", t)
    if normalize_quotes:
        t = normalize_quotes_ascii(t)
    # collapse spaces but keep newlines
    t = "\n".join(_RE_WS.sub(" ", line).rstrip() for line in t.split("\n"))
    t = t.strip()

    if collapse_underscores:
        t = _RE_UNDERSCORES.sub(" ", t).strip()

    if strip_leading_noise:
        t = _RE_LEADING_NOISE.sub("", t).lstrip()

    return t


def first_nonempty_line(t: str) -> str:
    for ln in t.split("\n"):
        if ln.strip():
            return ln.strip()
    return ""


def head_pattern_label(t: str) -> str:
    ln = first_nonempty_line(t)
    if not ln:
        return "empty"
    if _RE_MD_TABLE.match(ln) or ln.startswith("|"):
        return "pipes_table"
    if _RE_MD_HEADING.match(ln):
        return "md_heading"
    if _RE_DASH_LIST.match(ln):
        return "dash_list"
    if _RE_STAR_LIST.match(ln):
        return "star_list"
    if _RE_BLOCKQUOTE.match(ln):
        return "blockquote"
    if ln.startswith("_"):
        return "underscores"
    if ln[0].isdigit():
        return "digit"
    if ln[0].isalpha():
        return "alpha"
    return "other_symbol"


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


def is_markdownish(t: str) -> bool:
    # README / markdown signals
    if "```" in t:
        return True
    # lots of headings
    head_lines = t.split("\n")[:40]
    h = sum(1 for ln in head_lines if _RE_MD_HEADING.match(ln))
    if h >= 2:
        return True
    # table density early
    pipes = sum(ln.count("|") for ln in head_lines)
    if pipes >= 20:
        return True
    return False


def is_htmlish(t: str) -> bool:
    ln = first_nonempty_line(t)
    return bool(_RE_HTMLISH.match(ln))


def code_line_ratio(t: str) -> float:
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    codey = 0
    for ln in lines[:200]:
        s = ln.strip()
        if s.startswith("#"):
            # comments still count as code context
            codey += 1
            continue
        if any(x in s for x in (";", "{", "}", "(", ")", "=", ":", "->")):
            codey += 1
            continue
        if s.startswith(("def ", "class ", "import ", "from ", "return ")):
            codey += 1
            continue
    return codey / max(1, min(len(lines), 200))


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
    normalize_quotes: bool,
    drop_markdown: bool,
    drop_htmlish: bool,
    min_code_line_ratio: float,
) -> Optional[str]:
    if drop_reponame_header:
        t = _RE_REPONAME.sub("", t)

    t = normalize_text(
        t,
        collapse_underscores=collapse_underscores,
        strip_leading_noise=strip_leading_noise,
        normalize_quotes=normalize_quotes,
    )
    if not t:
        return None

    if drop_htmlish and is_htmlish(t):
        return None
    if drop_markdown and is_markdownish(t):
        return None

    lines = t.split("\n")

    if any(len(line) > max_line_len for line in lines):
        return None

    # too many non-ascii lines in head
    if lines:
        non_ascii_lines = 0
        headn = min(len(lines), 200)
        for ln in lines[:headn]:
            if ascii_ratio(ln) < 0.85:
                non_ascii_lines += 1
        if non_ascii_lines / max(1, headn) > max_non_ascii_lines_ratio:
            return None

    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines - len(head)) :]
        lines = head + ["# ... truncated ..."] + tail
        t = "\n".join(lines).strip()

    if len(lines) < min_lines:
        return None

    # global ratios
    if ascii_ratio(t) < min_ascii_ratio:
        return None
    if non_alnum_symbol_ratio(t) > max_symbol_ratio:
        return None

    if require_code_signals and (not looks_like_code(t)):
        return None

    if code_line_ratio(t) < min_code_line_ratio:
        return None

    bad_substrings = (
        "$(N)", "core dumped", "\x00", "PyGILState_Release",
        "terminate called without an active exception",
    )
    if any(x in t for x in bad_substrings):
        return None

    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.05:
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
    normalize_quotes: bool,
    drop_head_tables: bool,
) -> Optional[str]:
    t = normalize_text(
        t,
        collapse_underscores=collapse_underscores,
        strip_leading_noise=strip_leading_noise,
        normalize_quotes=normalize_quotes,
    )
    if not t:
        return None

    if drop_head_tables:
        lab = head_pattern_label(t)
        if lab in ("pipes_table", "md_heading"):
            return None

    t = trim_wiki_tail(t)
    if not t:
        return None

    if drop_templates:
        low = t.lower()
        bad_snippets = (
            "creative commons",
            "licensed under",
            "copyright",
            "citation needed",
            "retrieved ",
            "isbn ",
        )
        if any(x in low for x in bad_snippets):
            return None

        # listy
        if t.count("\n- ") >= 12:
            return None

        paras = [p for p in t.split("\n\n") if p.strip()]
        if len(paras) < min_paragraphs:
            return None

    return t


# -------------------------
# CLI + main
# -------------------------
def load_streaming_dataset(
    dataset: str,
    config: str | None,
    split: str,
    data_dir: str | None,
    use_auth: bool,
):
    kwargs = dict(streaming=True)
    if config:
        kwargs["name"] = config
    if data_dir:
        kwargs["data_dir"] = data_dir
    if use_auth:
        kwargs["use_auth_token"] = True
    return load_dataset(dataset, split=split, **kwargs)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--use_auth_token", action="store_true")

    ap.add_argument("--out", required=True)
    ap.add_argument("--text_field", default="text")

    ap.add_argument("--mode", choices=["generic", "wiki", "code"], default="generic")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--max_docs", type=int, default=0, help="0=unlimited")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=0)

    # global filters (generic/wiki)
    ap.add_argument("--head_chars", type=int, default=256)
    ap.add_argument("--min_head_ascii_ratio", type=float, default=0.90)
    ap.add_argument("--min_ascii_ratio", type=float, default=0.92)
    ap.add_argument("--max_symbol_ratio", type=float, default=0.55)

    # placeholder / noise / quote
    ap.add_argument("--collapse_underscores", action="store_true")
    ap.add_argument("--strip_leading_noise", action="store_true")
    ap.add_argument("--normalize_quotes", action="store_true", help='convert “ ” ‘ ’ etc to ASCII quotes')

    # head-pattern drops (generic/wiki)
    ap.add_argument("--drop_head_tables", action="store_true", help="drop docs starting with markdown tables/headings")
    ap.add_argument("--drop_head_lists", action="store_true", help="drop docs starting with list markers (-/*)")

    # code knobs
    ap.add_argument("--code_drop_reponame", action="store_true")
    ap.add_argument("--code_require_signals", action="store_true")
    ap.add_argument("--code_min_lines", type=int, default=6)
    ap.add_argument("--code_max_lines", type=int, default=600)
    ap.add_argument("--code_max_line_len", type=int, default=4000)
    ap.add_argument("--code_min_ascii_ratio", type=float, default=0.92)
    ap.add_argument("--code_max_symbol_ratio", type=float, default=0.65)  # code has symbols
    ap.add_argument("--code_max_non_ascii_lines_ratio", type=float, default=0.25)
    ap.add_argument("--code_drop_markdown", action="store_true")
    ap.add_argument("--code_drop_htmlish", action="store_true")
    ap.add_argument("--code_min_code_line_ratio", type=float, default=0.25)

    # wiki knobs
    ap.add_argument("--wiki_drop_templates", action="store_true")
    ap.add_argument("--wiki_min_paragraphs", type=int, default=2)

    # dedup
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--dedup_max", type=int, default=2_000_000)

    # IO
    ap.add_argument("--flush_every", type=int, default=10_000)
    ap.add_argument("--hard_exit", action="store_true")

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

    seen = kept = 0
    no_text = non_dict = bad_json = 0
    dropped_empty = dropped_short = dropped_long = 0
    dropped_dup = dropped_ascii = dropped_head_ascii = dropped_symbol = 0
    dropped_head_pattern = 0

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

                # mode-specific
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
                        normalize_quotes=args.normalize_quotes,
                        drop_markdown=args.code_drop_markdown,
                        drop_htmlish=args.code_drop_htmlish,
                        min_code_line_ratio=args.code_min_code_line_ratio,
                    )
                elif args.mode == "wiki":
                    t2 = clean_wiki(
                        t,
                        drop_templates=args.wiki_drop_templates,
                        min_paragraphs=args.wiki_min_paragraphs,
                        collapse_underscores=args.collapse_underscores,
                        strip_leading_noise=args.strip_leading_noise,
                        normalize_quotes=args.normalize_quotes,
                        drop_head_tables=args.drop_head_tables,
                    )
                else:
                    t2 = normalize_text(
                        t,
                        collapse_underscores=args.collapse_underscores,
                        strip_leading_noise=args.strip_leading_noise,
                        normalize_quotes=args.normalize_quotes,
                    )

                if not t2:
                    dropped_empty += 1
                    continue

                # head-pattern drops for generic/wiki
                if args.mode in ("generic", "wiki"):
                    lab = head_pattern_label(t2)
                    if args.drop_head_tables and lab in ("pipes_table", "md_heading"):
                        dropped_head_pattern += 1
                        continue
                    if args.drop_head_lists and lab in ("dash_list", "star_list"):
                        dropped_head_pattern += 1
                        continue

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

    stats = {
        "seen": seen,
        "kept": kept,
        "no_text": no_text,
        "non_dict": non_dict,
        "bad_json": bad_json,
        "dropped_empty": dropped_empty,
        "dropped_short": dropped_short,
        "dropped_long": dropped_long,
        "dropped_dup": dropped_dup,
        "dropped_ascii": dropped_ascii,
        "dropped_head_ascii": dropped_head_ascii,
        "dropped_symbol": dropped_symbol,
        "dropped_head_pattern": dropped_head_pattern,
        "mode": args.mode,
        "min_ascii_ratio": args.min_ascii_ratio,
        "min_head_ascii_ratio": args.min_head_ascii_ratio,
        "head_chars": args.head_chars,
        "max_symbol_ratio": args.max_symbol_ratio,
        "collapse_underscores": bool(args.collapse_underscores),
        "strip_leading_noise": bool(args.strip_leading_noise),
        "normalize_quotes": bool(args.normalize_quotes),
        "drop_head_tables": bool(args.drop_head_tables),
        "drop_head_lists": bool(args.drop_head_lists),
        "code_drop_markdown": bool(args.code_drop_markdown),
        "code_drop_htmlish": bool(args.code_drop_htmlish),
        "code_min_code_line_ratio": float(args.code_min_code_line_ratio),
    }
    print(f"wrote {kept} docs -> {args.out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    sys.stdout.flush()

    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
