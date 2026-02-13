#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract HF dataset (streaming) to jsonl: {"text": "..."} per line.

Goals:
- generic/wiki/code 三种模式
- 支持 code_only_python：尽量保证 code 数据更像“代码”，且可选只保留 Python
- 重点防止占位符 "____" 变成高频 token（可用 --collapse_underscores）
- 过滤 markdown/表格/README/html-ish 垃圾（可控开关）

Example (code only python):
python scripts/extract_hf_to_jsonl.py \
  --dataset bigcode/starcoderdata \
  --split train \
  --out datasets/tokenization/code_starcoder.clean.v4_py200k.jsonl \
  --mode code \
  --code_only_python \
  --max_docs 200000 \
  --min_chars 80 \
  --collapse_underscores --normalize_quotes --strip_leading_noise \
  --code_drop_markdown --code_drop_htmlish \
  --code_require_signals \
  --code_min_code_line_ratio 0.45
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
from typing import Any, Optional, Iterable

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

# strip short weird-symbol prefix (keeps normal punctuation and code punctuation)
_RE_LEADING_NOISE = re.compile(r"^\s*([^\w<>{}\[\]().,;:'\"/\-\n#|>]{1,32})\s*")

# markdown-ish head patterns
_RE_MD_HEADING = re.compile(r"^\s*#{1,6}\s+\S")     # "# Title"
_RE_MD_TABLE = re.compile(r"^\s*\|.+\|\s*$")        # "| a | b |"
_RE_DASH_LIST = re.compile(r"^\s*-\s+\S")
_RE_STAR_LIST = re.compile(r"^\s*\*\s+\S")
_RE_BLOCKQUOTE = re.compile(r"^\s*>\s+\S")
_RE_FENCE = re.compile(r"^\s*```")

# html/xml-ish head patterns (keep C/C++ include)
_RE_HTMLISH = re.compile(r"^\s*<\s*/?\s*[a-zA-Z]{1,20}[\s>]")


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
        # turn "____" runs into a single space, avoiding the model loving that token
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
    if _RE_FENCE.match(ln):
        return "fence"
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


def is_htmlish_head(t: str) -> bool:
    ln = first_nonempty_line(t)
    if not ln:
        return False
    # Keep C/C++ include lines: "#include <...>"
    if ln.startswith("#include "):
        return False
    return bool(_RE_HTMLISH.match(ln)) or ln.lstrip().startswith("<")


def is_markdownish_early(t: str) -> bool:
    # README / markdown signals
    if "```" in t:
        return True
    head_lines = t.split("\n")[:60]
    # multiple headings early => likely markdown
    h = sum(1 for ln in head_lines if _RE_MD_HEADING.match(ln))
    if h >= 2:
        return True
    # table density early
    pipes = sum(ln.count("|") for ln in head_lines)
    if pipes >= 24:
        return True
    return False


# -------------------------
# Language detection for code-only-python
# -------------------------
def get_lang(ex: dict[str, Any], lang_field: str) -> Optional[str]:
    v = ex.get(lang_field, None)
    if isinstance(v, str):
        return v.strip().lower()
    return None


def guess_python_by_text(t: str) -> bool:
    """
    Heuristic: only when dataset doesn't provide language metadata.
    Try to identify Python-ish code and reject obvious non-python.
    """
    s = t.lstrip()

    # strong python signals
    if s.startswith("#!/usr/bin/env python") or s.startswith("#!/usr/bin/python"):
        return True
    if re.search(r"^\s*def\s+\w+\s*\(", t, flags=re.M):
        return True
    if re.search(r"^\s*class\s+\w+\s*[:(]", t, flags=re.M):
        return True
    if "if __name__ == '__main__':" in t or 'if __name__ == "__main__":' in t:
        return True
    if re.search(r"^\s*(from|import)\s+\w+", t, flags=re.M):
        # but watch out for non-python "import" in other langs - use more checks below
        pass

    # obvious non-python / shell / C# / Java / XML / HTML
    non_py_needles = (
        "using System", "namespace ", "public class", "private ", "protected ",
        "#include ", "BEGIN CERTIFICATE", "<!DOCTYPE html", "<html", "<?xml",
        "package ", "public static void main", "console.log(", "function(",
        "SELECT ", "CREATE TABLE", "echo ", "REM ", "setlocal", "::", "/*", "*/",
    )
    if any(x in t for x in non_py_needles):
        # allow python with type comments containing "/*"? (rare) -> keep strict
        return False

    # braces/semicolons heavy => likely not python
    if t.count("{") + t.count("}") > 8:
        return False
    if t.count(";") > 12:
        return False

    # python-ish punctuation: indentation + colons
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if not lines:
        return False

    indented = sum(1 for ln in lines[:200] if (len(ln) - len(ln.lstrip(" "))) >= 4)
    colon_lines = sum(1 for ln in lines[:200] if ln.rstrip().endswith(":"))
    if indented >= 6 and colon_lines >= 2:
        return True

    # fallback: require some python keywords
    py_kw = sum(t.count(k) for k in ("def ", "class ", "import ", "from ", "None", "True", "False"))
    return py_kw >= 2


# -------------------------
# Code-ness heuristics
# -------------------------
def looks_like_code(t: str) -> bool:
    needles = (
        "def ", "class ", "import ", "from ", "#include", "#define",
        "public ", "private ", "protected ", "template<",
        "function ", "const ", "let ", "var ",
        "package ", "namespace ", "using ",
        "return ", "if ", "else", "elif ", "try", "except",
        "for ", "while ", "switch ", "case ",
    )
    if any(x in t for x in needles):
        return True
    if ("{" in t and "}" in t) or (";" in t and "\n" in t):
        return True
    if t.count("\n") >= 8 and (t.count("(") + t.count(")") >= 6):
        return True
    return False


def code_line_ratio(t: str) -> float:
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    codey = 0
    for ln in lines[:220]:
        s = ln.strip()
        if s.startswith("#!"):  # shebang
            codey += 1
            continue
        if s.startswith("#") and (s.startswith("#include") or s.startswith("#define") or "coding:" in s.lower()):
            codey += 1
            continue
        if s.startswith("#") and len(s) < 200:
            codey += 1
            continue
        if any(x in s for x in (";", "{", "}", "(", ")", "=", ":", "->", "::", "=>")):
            codey += 1
            continue
        if s.startswith(("def ", "class ", "import ", "from ", "return ", "fn ", "func ")):
            codey += 1
            continue
    return codey / max(1, min(len(lines), 220))


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
    drop_md_heading_head: bool,
    drop_head_table_like: bool,
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

    lab = head_pattern_label(t)
    head = first_nonempty_line(t)

    # drop markdown headings at head (but keep python encoding headers)
    if drop_md_heading_head and lab == "md_heading":
        if "coding" not in head.lower():
            return None

    # drop obvious table/fence at head
    if drop_head_table_like and lab in ("pipes_table", "fence"):
        return None

    if drop_htmlish and is_htmlish_head(t):
        return None

    if drop_markdown and is_markdownish_early(t):
        return None

    lines = t.split("\n")
    if any(len(line) > max_line_len for line in lines):
        return None

    # too many non-ascii lines in head
    if lines:
        non_ascii_lines = 0
        headn = min(len(lines), 220)
        for ln in lines[:headn]:
            if ascii_ratio(ln) < 0.85:
                non_ascii_lines += 1
        if non_ascii_lines / max(1, headn) > max_non_ascii_lines_ratio:
            return None

    # truncate very long blocks
    if len(lines) > max_lines:
        head_part = lines[: max_lines // 2]
        tail_part = lines[-(max_lines - len(head_part)) :]
        lines = head_part + ["# ... truncated ..."] + tail_part
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

    # drop obvious crash dumps / binary-ish
    bad_substrings = (
        "core dumped", "\x00", "PyGILState_Release",
        "terminate called without an active exception",
    )
    if any(x in t for x in bad_substrings):
        return None

    letters = sum(ch.isalpha() for ch in t)
    if letters / max(1, len(t)) < 0.03:
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
        if lab in ("pipes_table", "md_heading", "fence"):
            return None

    t = trim_wiki_tail(t)
    if not t:
        return None

    if drop_templates:
        low = t.lower()
        bad_snippets = (
            "creative commons", "licensed under", "copyright",
            "citation needed", "retrieved ", "isbn ",
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
# HF streaming loader
# -------------------------
def load_streaming_dataset(dataset: str, config: str | None, split: str, data_dir: str | None, use_auth: bool):
    kwargs = dict(streaming=True)
    if config:
        kwargs["name"] = config
    if data_dir:
        kwargs["data_dir"] = data_dir
    if use_auth:
        kwargs["use_auth_token"] = True
    return load_dataset(dataset, split=split, **kwargs)


# -------------------------
# main
# -------------------------
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

    ap.add_argument("--max_docs", type=int, default=0, help="0=unlimited (kept docs)")
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
    ap.add_argument("--normalize_quotes", action="store_true")

    # head-pattern drops (generic/wiki)
    ap.add_argument("--drop_head_tables", action="store_true")
    ap.add_argument("--drop_head_lists", action="store_true")

    # code knobs
    ap.add_argument("--code_drop_reponame", action="store_true")
    ap.add_argument("--code_require_signals", action="store_true")
    ap.add_argument("--code_min_lines", type=int, default=6)
    ap.add_argument("--code_max_lines", type=int, default=600)
    ap.add_argument("--code_max_line_len", type=int, default=4000)
    ap.add_argument("--code_min_ascii_ratio", type=float, default=0.92)
    ap.add_argument("--code_max_symbol_ratio", type=float, default=0.75)  # code has symbols
    ap.add_argument("--code_max_non_ascii_lines_ratio", type=float, default=0.22)
    ap.add_argument("--code_drop_markdown", action="store_true")
    ap.add_argument("--code_drop_htmlish", action="store_true")
    ap.add_argument("--code_min_code_line_ratio", type=float, default=0.35)
    ap.add_argument("--code_drop_md_heading_head", action="store_true")
    ap.add_argument("--code_drop_head_table_like", action="store_true")

    # NEW: code language filtering
    ap.add_argument("--lang_field", default="language", help="dataset language field if present")
    ap.add_argument("--code_lang_allow", default="", help="comma-separated allow list, e.g. python,py")
    ap.add_argument("--code_only_python", action="store_true", help="keep only python (uses lang_field if exists; else heuristic)")

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

    allow_langs = {x.strip().lower() for x in args.code_lang_allow.split(",") if x.strip()}
    if args.code_only_python:
        allow_langs |= {"python", "py"}

    ds = load_streaming_dataset(args.dataset, args.config, args.split, args.data_dir, args.use_auth_token)

    seen = kept = 0
    no_text = non_dict = 0
    dropped_empty = dropped_short = dropped_long = 0
    dropped_dup = 0
    dropped_ascii = dropped_head_ascii = dropped_symbol = 0
    dropped_head_pattern = 0
    dropped_lang = 0
    dropped_code = 0

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

                # language filter (CODE only)
                if args.mode == "code" and (args.code_only_python or allow_langs):
                    lang = get_lang(ex, args.lang_field)
                    if lang is not None and allow_langs:
                        if lang not in allow_langs:
                            dropped_lang += 1
                            continue
                    elif args.code_only_python:
                        # no language metadata -> use heuristic later after we extract text
                        pass

                t = pick_text(ex, args.text_field)
                if not isinstance(t, str):
                    no_text += 1
                    continue

                if args.mode == "code" and args.code_only_python:
                    lang = get_lang(ex, args.lang_field)
                    if lang is None:
                        if not guess_python_by_text(t):
                            dropped_lang += 1
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
                        normalize_quotes=args.normalize_quotes,
                        drop_markdown=args.code_drop_markdown,
                        drop_htmlish=args.code_drop_htmlish,
                        min_code_line_ratio=args.code_min_code_line_ratio,
                        drop_md_heading_head=args.code_drop_md_heading_head,
                        drop_head_table_like=args.code_drop_head_table_like,
                    )
                    if not t2:
                        dropped_code += 1
                        continue

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
                    if not t2:
                        dropped_empty += 1
                        continue

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

                    lab = head_pattern_label(t2)
                    if args.drop_head_tables and lab in ("pipes_table", "md_heading", "fence"):
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
        "dropped_empty": dropped_empty,
        "dropped_short": dropped_short,
        "dropped_long": dropped_long,
        "dropped_ascii": dropped_ascii,
        "dropped_head_ascii": dropped_head_ascii,
        "dropped_symbol": dropped_symbol,
        "dropped_head_pattern": dropped_head_pattern,
        "dropped_dup": dropped_dup,
        "dropped_lang": dropped_lang,
        "dropped_code": dropped_code,
        "mode": args.mode,
        "code_only_python": bool(args.code_only_python),
        "code_lang_allow": sorted(list(allow_langs)),
    }
    print(f"wrote {kept} docs -> {args.out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    sys.stdout.flush()

    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
