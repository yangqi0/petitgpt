#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build fixed-size token shards from jsonl {"text": "..."} sources, with mixing weights.

Features:
- Mix multiple jsonl sources with weights (streaming, loops when a file ends)
- Write train/val shards of fixed token counts (.bin, uint16/uint32 auto)
- Optional doc separator (default "\n\n") to reduce doc-glue artifacts
- Optional tokenizer special leak check (BOS/EOS auto-added by tokenizer post_processor)
- Best-practice text normalization / filtering for cleaner "first token" behavior:
  --strip_leading_noise
  --normalize_quotes
  --underscores_policy {keep,space,remove}
  --min_chars
  --min_ascii_ratio

Meta:
- meta.json records fingerprints, per-source stats, first-token topK per-doc (NOT counting doc_sep)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer


# -------------------------
# IO
# -------------------------
def iter_jsonl_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            if isinstance(ex, dict):
                t = ex.get("text", None)
                if isinstance(t, str) and t:
                    yield t


def file_fingerprint(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {
        "path": str(p),
        "resolved": str(p.resolve()),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "inode": int(getattr(st, "st_ino", 0)),
        "device": int(getattr(st, "st_dev", 0)),
    }


def fast_count_lines(p: Path, max_lines: int = 300_000) -> dict[str, Any]:
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
            if n >= max_lines:
                return {"lines": n, "at_least": True, "max_lines": max_lines}
    return {"lines": n, "at_least": False, "max_lines": max_lines}


# -------------------------
# Tokenizer helpers
# -------------------------
def load_tokenizer(tokenizer_path: str):
    return Tokenizer.from_file(tokenizer_path)


def _tokenizer_vocab_size(tok) -> Optional[int]:
    try:
        return int(tok.get_vocab_size())
    except Exception:
        return None


def _encode_base(tok, text: str) -> list[int]:
    enc = tok.encode(text)
    return list(enc.ids)


def encode(tok, text: str, add_bos: bool, add_eos: bool, bos_id: int, eos_id: int) -> list[int]:
    ids = _encode_base(tok, text)
    if add_bos:
        ids = [bos_id] + ids
    if add_eos:
        ids = ids + [eos_id]
    return ids


def _detect_special_leak(tok, bos_id: int, eos_id: int) -> dict[str, Any]:
    probe = "Hello world."
    try:
        ids = tok.encode(probe).ids
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    leak_bos = (len(ids) >= 1 and ids[0] == bos_id)
    leak_eos = (len(ids) >= 1 and ids[-1] == eos_id)
    return {
        "ok": True,
        "probe_ids_head": ids[:8],
        "probe_ids_tail": ids[-8:] if len(ids) >= 8 else ids,
        "leak_bos": bool(leak_bos),
        "leak_eos": bool(leak_eos),
        "leak_any": bool(leak_bos or leak_eos),
    }


# -------------------------
# Text normalization / filters
# -------------------------
_ZW = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width chars + BOM
_WS_HEAD = re.compile(r"^\s+")
_MULTI_NL = re.compile(r"\n{4,}")  # huge blank gaps
_UNDERS = re.compile(r"_{2,}")     # sequences of underscores
# common “noise-y” leading markers in web text
_LEADING_NOISE = re.compile(
    r"""^(
        [>\|\-#\*\+]\s+ |          # markdown quote/table/bullet/heading-ish
        \|{1,}\s* |                # pipe tables
        \*{1,}\s+ |                # bullet stars
        \-{2,}\s+ |                # long dashes
        #{1,6}\s+ |                # headings
        \u2022\s+ |                # bullet dot
        \(\s*\d+\s*\)\s+ |         # (1) style
        \[\s*\d+\s*\]\s+           # [1] style
    )+""",
    re.VERBOSE,
)

_QUOTE_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "‐": "-",   # hyphen variants
        "-": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)

def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    # count ASCII including common whitespace
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / max(1, len(s))

def clean_text(
    t: str,
    *,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
) -> Optional[str]:
    if not isinstance(t, str):
        return None

    # normalize newlines first
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # remove zero-width and BOM
    t = _ZW.sub("", t)

    if normalize_quotes:
        t = t.translate(_QUOTE_MAP)

    # collapse insane blank blocks (keeps some formatting)
    t = _MULTI_NL.sub("\n\n\n", t)

    if strip_leading_noise:
        # strip head whitespace
        t = _WS_HEAD.sub("", t)
        # repeatedly strip common leading markers
        for _ in range(3):
            new_t = _LEADING_NOISE.sub("", t)
            if new_t == t:
                break
            t = new_t
        # after stripping markers, trim again
        t = _WS_HEAD.sub("", t)

    if underscores_policy != "keep":
        if underscores_policy == "space":
            t = _UNDERS.sub(" ", t)
        elif underscores_policy == "remove":
            t = _UNDERS.sub("", t)

    # final trim (but don't nuke internal newlines)
    t = t.strip()

    if min_chars > 0 and len(t) < min_chars:
        return None
    if min_ascii_ratio > 0.0 and ascii_ratio(t) < min_ascii_ratio:
        return None

    return t if t else None


# -------------------------
# Source parsing
# -------------------------
def parse_sources(source_args: list[str]) -> list[tuple[Path, float]]:
    items: list[tuple[Path, float]] = []
    for s in source_args:
        if ":" not in s:
            raise ValueError(f"--source must be like path:weight, got {s!r}")
        p, w = s.rsplit(":", 1)
        p = p.strip()
        w = w.strip()
        wf = float(w)
        if wf <= 0:
            raise ValueError(f"--source weight must be > 0, got {wf} in {s!r}")
        items.append((Path(p), wf))
    tot = sum(w for _, w in items) if items else 1.0
    return [(p, w / tot) for p, w in items]


def topk_counter(c: Counter[int], k: int) -> list[dict[str, Any]]:
    return [{"id": int(tid), "count": int(cnt)} for tid, cnt in c.most_common(k)]


# -------------------------
# Mixing iterator
# -------------------------
@dataclass
class SrcState:
    path: Path
    weight: float
    it: Iterator[str]


def make_mixed_stream(sources: list[tuple[Path, float]], seed: int) -> Iterator[tuple[str, str]]:
    rng = random.Random(seed)
    states: list[SrcState] = []
    for p, w in sources:
        states.append(SrcState(path=p, weight=w, it=iter_jsonl_texts(p)))

    weights = [st.weight for st in states]
    s = sum(weights)
    weights = [w / s for w in weights]

    while True:
        r = rng.random()
        acc = 0.0
        idx = 0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                idx = i
                break
        st = states[idx]
        try:
            t = next(st.it)
        except StopIteration:
            st.it = iter_jsonl_texts(st.path)
            t = next(st.it)
        yield (str(st.path), t)


# -------------------------
# Shard writer
# -------------------------
def write_shards(
    *,
    sources: list[tuple[Path, float]],
    out_dir: Path,
    tokenizer_path: str,
    shard_tokens: int,
    val_shard_tokens: int,
    val_ratio: float,
    seed: int,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    target_train_tokens: int,
    precheck_max_lines: int,
    detect_tokenizer_special_leak: bool,
    doc_sep: str,
    first_token_topk: int,
    # filters
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
) -> None:
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tok = load_tokenizer(tokenizer_path)
    vocab_size = _tokenizer_vocab_size(tok)
    dtype = np.uint32 if (vocab_size is not None and vocab_size > 65535) else np.uint16

    # default doc separator
    if doc_sep == "":
        doc_sep = "\n\n"
    doc_sep_ids = _encode_base(tok, doc_sep) if doc_sep else []

    special_leak = None
    if detect_tokenizer_special_leak:
        special_leak = _detect_special_leak(tok, bos_id=bos_id, eos_id=eos_id)
        if special_leak.get("ok") and special_leak.get("leak_any"):
            print(
                "\n[WARNING] Tokenizer appears to add BOS/EOS via post_processor.\n"
                "          You likely want tokenizer_pretrain_nospecial.json for pretrain.\n"
                f"          leak_bos={special_leak.get('leak_bos')} leak_eos={special_leak.get('leak_eos')}\n"
            )

    buf_train: list[int] = []
    buf_val: list[int] = []
    shard_idx_train = shard_idx_val = 0
    total_train = total_val = 0
    seen_docs = kept_docs = 0
    dropped_short = 0
    dropped_ascii = 0
    dropped_empty = 0

    per_src: dict[str, Any] = {}
    source_fingerprints: dict[str, Any] = {}
    source_line_precheck: dict[str, Any] = {}

    first_tok_all: Counter[int] = Counter()
    first_tok_per_src: dict[str, Counter[int]] = {}

    for p, w in sources:
        sp = str(p)
        per_src[sp] = {
            "weight": float(w),
            "seen_docs": 0,
            "kept_docs": 0,
            "train_tokens": 0,
            "val_tokens": 0,
            "lines_read": 0,
            "bad_json": 0,
            "non_dict": 0,
            "no_text": 0,
            "ok_texts": 0,
            "dropped_short": 0,
            "dropped_ascii": 0,
            "dropped_empty": 0,
        }
        source_fingerprints[sp] = file_fingerprint(p)
        source_line_precheck[sp] = fast_count_lines(p, max_lines=precheck_max_lines)
        first_tok_per_src[sp] = Counter()

    def flush(buf: list[int], out_path: Path) -> int:
        arr = np.asarray(buf, dtype=dtype)
        arr.tofile(out_path)
        return int(arr.size)

    def maybe_flush_train():
        nonlocal buf_train, shard_idx_train, total_train
        while len(buf_train) >= shard_tokens:
            chunk = buf_train[:shard_tokens]
            buf_train = buf_train[shard_tokens:]
            p = out_train / f"shard_{shard_idx_train:05d}.bin"
            n = flush(chunk, p)
            total_train += n
            shard_idx_train += 1

    def maybe_flush_val():
        nonlocal buf_val, shard_idx_val, total_val
        while len(buf_val) >= val_shard_tokens:
            chunk = buf_val[:val_shard_tokens]
            buf_val = buf_val[val_shard_tokens:]
            p = out_val / f"shard_{shard_idx_val:05d}.bin"
            n = flush(chunk, p)
            total_val += n
            shard_idx_val += 1

    def add_doc(src_path: str, ids: list[int]) -> None:
        nonlocal seen_docs, kept_docs, buf_train, buf_val

        seen_docs += 1
        per_src[src_path]["seen_docs"] += 1

        if not ids:
            return

        # record first token per *document* (not counting doc_sep)
        first_tok_all[ids[0]] += 1
        first_tok_per_src[src_path][ids[0]] += 1

        kept_docs += 1
        per_src[src_path]["kept_docs"] += 1

        if rng.random() < val_ratio:
            if doc_sep_ids and buf_val:
                buf_val.extend(doc_sep_ids)
            buf_val.extend(ids)
            per_src[src_path]["val_tokens"] += len(ids)
            maybe_flush_val()
        else:
            if doc_sep_ids and buf_train:
                buf_train.extend(doc_sep_ids)
            buf_train.extend(ids)
            per_src[src_path]["train_tokens"] += len(ids)
            maybe_flush_train()

    stream = make_mixed_stream(sources, seed=seed)

    while total_train < target_train_tokens:
        src, text = next(stream)

        # clean / filter
        ct = clean_text(
            text,
            strip_leading_noise=strip_leading_noise,
            normalize_quotes=normalize_quotes,
            underscores_policy=underscores_policy,
            min_chars=min_chars,
            min_ascii_ratio=min_ascii_ratio,
        )

        if ct is None:
            # best-effort categorize drops
            per_src[src]["dropped_empty"] += 1
            dropped_empty += 1
            continue

        # categorize min_chars/min_ascii_ratio failures precisely (optional but helpful)
        # (we do a second check here on the *cleaned* text)
        if min_chars > 0 and len(ct) < min_chars:
            per_src[src]["dropped_short"] += 1
            dropped_short += 1
            continue
        if min_ascii_ratio > 0.0 and ascii_ratio(ct) < min_ascii_ratio:
            per_src[src]["dropped_ascii"] += 1
            dropped_ascii += 1
            continue

        ids = encode(tok, ct, add_bos=add_bos, add_eos=add_eos, bos_id=bos_id, eos_id=eos_id)
        if not ids:
            per_src[src]["dropped_empty"] += 1
            dropped_empty += 1
            continue

        add_doc(src, ids)

    # flush remainder
    if buf_train:
        p = out_train / f"shard_{shard_idx_train:05d}.bin"
        n = flush(buf_train, p)
        total_train += n
        shard_idx_train += 1
    if buf_val:
        p = out_val / f"shard_{shard_idx_val:05d}.bin"
        n = flush(buf_val, p)
        total_val += n
        shard_idx_val += 1

    first_tok_meta = {
        "topk_all": topk_counter(first_tok_all, first_token_topk),
        "topk_per_source": {k: topk_counter(v, first_token_topk) for k, v in first_tok_per_src.items()},
    }

    meta = {
        "tokenizer_path": tokenizer_path,
        "dtype": "uint32" if dtype == np.uint32 else "uint16",
        "sources": [{"path": str(p), "weight": float(w)} for p, w in sources],
        "shard_tokens": int(shard_tokens),
        "val_shard_tokens": int(val_shard_tokens),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "add_bos": bool(add_bos),
        "add_eos": bool(add_eos),
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "doc_sep": doc_sep,
        "filters": {
            "strip_leading_noise": bool(strip_leading_noise),
            "normalize_quotes": bool(normalize_quotes),
            "underscores_policy": underscores_policy,
            "min_chars": int(min_chars),
            "min_ascii_ratio": float(min_ascii_ratio),
        },
        "seen_docs": int(seen_docs),
        "kept_docs": int(kept_docs),
        "dropped_short": int(dropped_short),
        "dropped_ascii": int(dropped_ascii),
        "dropped_empty": int(dropped_empty),
        "train_tokens": int(total_train),
        "val_tokens": int(total_val),
        "train_shards": int(shard_idx_train),
        "val_shards": int(shard_idx_val),
        "source_fingerprints": source_fingerprints,
        "source_line_precheck": source_line_precheck,
        "tokenizer_special_leak_check": special_leak,
        "first_token_topk": first_tok_meta,
        "target_train_tokens": int(target_train_tokens),
        "per_source": per_src,
    }

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[], help="path:weight (repeatable)")
    ap.add_argument("--target_train_tokens", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--shard_tokens", type=int, default=10_000_000)
    ap.add_argument("--val_shard_tokens", type=int, default=2_000_000)
    ap.add_argument("--val_ratio", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--doc_sep", type=str, default="", help='optional doc separator, default="\\n\\n"')
    ap.add_argument("--first_token_topk", type=int, default=50)

    ap.add_argument("--precheck_max_lines", type=int, default=300_000)
    ap.add_argument("--detect_tokenizer_special_leak", action="store_true")

    # filters / normalization
    ap.add_argument("--strip_leading_noise", action="store_true")
    ap.add_argument("--normalize_quotes", action="store_true")
    ap.add_argument("--underscores_policy", type=str, default="keep", choices=["keep", "space", "remove"])
    ap.add_argument("--min_chars", type=int, default=0)
    ap.add_argument("--min_ascii_ratio", type=float, default=0.0)

    args = ap.parse_args()
    sources = parse_sources(args.source)
    if not sources:
        raise SystemExit("Provide at least one --source path:weight")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.val_shard_tokens <= 0 or args.shard_tokens <= 0:
        raise SystemExit("shard sizes must be > 0")
    if not (0.0 <= args.val_ratio <= 1.0):
        raise SystemExit("--val_ratio must be in [0,1]")
    if args.target_train_tokens <= 0:
        raise SystemExit("--target_train_tokens must be > 0")

    write_shards(
        sources=sources,
        out_dir=out_dir,
        tokenizer_path=args.tokenizer_path,
        shard_tokens=args.shard_tokens,
        val_shard_tokens=args.val_shard_tokens,
        val_ratio=args.val_ratio,
        seed=args.seed,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        target_train_tokens=args.target_train_tokens,
        precheck_max_lines=args.precheck_max_lines,
        detect_tokenizer_special_leak=args.detect_tokenizer_special_leak,
        doc_sep=args.doc_sep,
        first_token_topk=args.first_token_topk,
        strip_leading_noise=args.strip_leading_noise,
        normalize_quotes=args.normalize_quotes,
        underscores_policy=args.underscores_policy,
        min_chars=args.min_chars,
        min_ascii_ratio=args.min_ascii_ratio,
    )


if __name__ == "__main__":
    main()
