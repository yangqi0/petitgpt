#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build mixed pretrain shards from multiple jsonl sources.

Key guarantees (requested):
1) Each source will have at least --min_val_tokens_per_source validation tokens
   (measured on *content tokens*, excluding doc_sep).
2) Training sampling approximates weights by *token quotas* (not doc quotas):
   for each source i, we target ~ weight_i * target_train_tokens content tokens.
3) Before writing shards, assert token ids are in-range:
   - ids are integers
   - 0 <= id < vocab_size (if known)
   - if dtype is uint16, assert id <= 65535
   This prevents silent uint16 overflow / negative dirty ids.

Notes:
- We keep doc_sep tokens out of per_source token accounting (same as your old meta.json),
  but they do contribute to raw shard token count.
- Validation collection is quota-based per source (deterministic), not random val_ratio per doc.
  `val_ratio` is used only to set default val token targets if you don't specify min_val_tokens.
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
from typing import Any, Iterator, Optional

import numpy as np
from tokenizers import Tokenizer


# -------------------------
# IO utils
# -------------------------
def iter_jsonl_texts(path: Path, field: str = "text") -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            t = obj.get(field, None)
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


def fast_count_lines(p: Path, max_lines: int) -> dict[str, Any]:
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
            if n >= max_lines:
                return {"lines": int(n), "at_least": True, "max_lines": int(max_lines)}
    return {"lines": int(n), "at_least": False, "max_lines": int(max_lines)}


# -------------------------
# Tokenizer helpers
# -------------------------
def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def _tokenizer_vocab_size(tok: Tokenizer) -> Optional[int]:
    # tokenizers Tokenizer doesn't expose vocab_size directly in all cases;
    # we can get it via get_vocab() when available.
    try:
        v = tok.get_vocab()
        return int(len(v))
    except Exception:
        return None


def _encode_base(tok: Tokenizer, text: str) -> list[int]:
    enc = tok.encode(text)
    return list(enc.ids)


def encode(
    tok: Tokenizer,
    text: str,
    *,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    ids = _encode_base(tok, text)
    if add_bos:
        if not ids or ids[0] != bos_id:
            ids = [bos_id] + ids
    if add_eos:
        if not ids or ids[-1] != eos_id:
            ids = ids + [eos_id]
    return ids


def assert_token_ids_ok(
    ids: list[int],
    *,
    vocab_size: Optional[int],
    dtype: np.dtype,
    src: str,
    text_preview: str,
) -> None:
    # type + sign checks
    for x in ids:
        if not isinstance(x, int):
            raise AssertionError(f"[token-id] non-int id: {type(x)} from src={src}")
        if x < 0:
            raise AssertionError(f"[token-id] negative id={x} from src={src}")

    # vocab range checks (strong)
    if vocab_size is not None:
        mx = max(ids)
        if mx >= vocab_size:
            raise AssertionError(
                f"[token-id] out-of-range id={mx} >= vocab_size={vocab_size} from src={src}\n"
                f"text_preview={text_preview!r}"
            )

    # dtype overflow checks
    if dtype == np.uint16:
        mx = max(ids)
        if mx > 65535:
            raise AssertionError(
                f"[token-id] uint16 overflow risk: max_id={mx} > 65535 from src={src}\n"
                f"text_preview={text_preview!r}"
            )


# -------------------------
# Cleaning / filters (kept minimal; you already cleaned upstream)
# -------------------------
_RE_LEADING_NOISE = re.compile(r"^\s*(?:\ufeff|<!--.*?-->|<\?xml.*?\?>)+", re.DOTALL)

def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
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
    if t is None:
        return None
    if not isinstance(t, str):
        return None
    t = t.strip("\n\r")
    if strip_leading_noise:
        t = _RE_LEADING_NOISE.sub("", t)
    if normalize_quotes:
        t = (
            t.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
        )
    if underscores_policy == "space":
        t = t.replace("_", " ")
    elif underscores_policy == "remove":
        t = t.replace("_", "")
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
# Per-source iterators + token-quota scheduler
# -------------------------
@dataclass
class SrcState:
    path: Path
    weight: float
    it: Iterator[str]
    name: str


def choose_src_by_remaining(rng: random.Random, states: list[SrcState], remaining: dict[str, int]) -> SrcState:
    # Choose proportional to remaining tokens (token-based mixing)
    live = [(st, max(0, remaining[st.name])) for st in states if remaining[st.name] > 0]
    if not live:
        # fallback (shouldn't happen unless rounding makes everything hit 0 while total still short)
        return rng.choice(states)
    total = sum(r for _, r in live)
    r = rng.randrange(total)
    acc = 0
    for st, rem in live:
        acc += rem
        if r < acc:
            return st
    return live[-1][0]


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
    min_val_tokens_per_source: int,
    seed: int,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    target_train_tokens: int,
    precheck_max_lines: int,
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
    if dtype == np.uint16 and vocab_size is not None and vocab_size > 65535:
        raise AssertionError("Unexpected: vocab_size>65535 but dtype selected uint16")

    if doc_sep == "":
        doc_sep = "\n\n"
    doc_sep_ids = _encode_base(tok, doc_sep) if doc_sep else []

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

    states: list[SrcState] = []
    for p, w in sources:
        sp = str(p)
        states.append(SrcState(path=p, weight=w, it=iter_jsonl_texts(p), name=sp))
        per_src[sp] = {
            "weight": float(w),
            "seen_docs": 0,
            "kept_docs": 0,
            "train_tokens": 0,   # content tokens only
            "val_tokens": 0,     # content tokens only
            "dropped_short": 0,
            "dropped_ascii": 0,
            "dropped_empty": 0,
        }
        source_fingerprints[sp] = file_fingerprint(p)
        source_line_precheck[sp] = fast_count_lines(p, max_lines=precheck_max_lines)
        first_tok_per_src[sp] = Counter()

    # --- Token quotas ---
    # Train content token quotas by weight
    train_target_per_src: dict[str, int] = {}
    # distribute rounding so sum matches exactly
    raw = [(st.name, st.weight * target_train_tokens) for st in states]
    floor_sum = 0
    fracs = []
    for name, x in raw:
        fx = int(x)
        train_target_per_src[name] = fx
        floor_sum += fx
        fracs.append((x - fx, name))
    # assign leftover tokens to largest fractions
    leftover = target_train_tokens - floor_sum
    fracs.sort(reverse=True)
    for i in range(max(0, leftover)):
        train_target_per_src[fracs[i % len(fracs)][1]] += 1

    # Val token targets: default proportional to weights, but ensure min per source
    default_total_val = int(round(target_train_tokens * val_ratio))
    val_target_per_src: dict[str, int] = {}
    rawv = [(st.name, st.weight * default_total_val) for st in states]
    floor_sum = 0
    fracs = []
    for name, x in rawv:
        fx = int(x)
        val_target_per_src[name] = fx
        floor_sum += fx
        fracs.append((x - fx, name))
    leftover = default_total_val - floor_sum
    fracs.sort(reverse=True)
    for i in range(max(0, leftover)):
        val_target_per_src[fracs[i % len(fracs)][1]] += 1

    # Apply per-source minimum (may increase total val beyond default_total_val; that's ok by design)
    for st in states:
        val_target_per_src[st.name] = max(val_target_per_src[st.name], int(min_val_tokens_per_source))

    # Remaining quotas (content tokens)
    train_remaining = dict(train_target_per_src)
    val_remaining = dict(val_target_per_src)

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

    def route_to_val(src_name: str) -> bool:
        # Deterministic quota: if this source still needs val tokens, send to val.
        return val_remaining[src_name] > 0

    def add_doc(src_name: str, ids: list[int]) -> None:
        nonlocal seen_docs, kept_docs, buf_train, buf_val
        seen_docs += 1
        per_src[src_name]["seen_docs"] += 1
        if not ids:
            return

        first_tok_all[ids[0]] += 1
        first_tok_per_src[src_name][ids[0]] += 1

        kept_docs += 1
        per_src[src_name]["kept_docs"] += 1

        if route_to_val(src_name):
            if doc_sep_ids and buf_val:
                buf_val.extend(doc_sep_ids)
            buf_val.extend(ids)
            per_src[src_name]["val_tokens"] += len(ids)
            val_remaining[src_name] = max(0, val_remaining[src_name] - len(ids))
            maybe_flush_val()
        else:
            if doc_sep_ids and buf_train:
                buf_train.extend(doc_sep_ids)
            buf_train.extend(ids)
            per_src[src_name]["train_tokens"] += len(ids)
            train_remaining[src_name] = max(0, train_remaining[src_name] - len(ids))
            maybe_flush_train()

    # --- Main loop: token-quota mixing ---
    # We stop when total TRAIN CONTENT tokens reach target_train_tokens approximately.
    # (train_remaining drives source selection; actual shard token count includes doc_sep)
    train_content_total = 0
    val_content_total = 0

    while train_content_total < target_train_tokens:
        st = choose_src_by_remaining(rng, states, train_remaining)
        src = st.name
        try:
            text = next(st.it)
        except StopIteration:
            st.it = iter_jsonl_texts(st.path)
            text = next(st.it)

        ct = clean_text(
            text,
            strip_leading_noise=strip_leading_noise,
            normalize_quotes=normalize_quotes,
            underscores_policy=underscores_policy,
            min_chars=min_chars,
            min_ascii_ratio=min_ascii_ratio,
        )
        if ct is None:
            per_src[src]["dropped_empty"] += 1
            dropped_empty += 1
            continue
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

        # --- Safety asserts before any write ---
        assert_token_ids_ok(ids, vocab_size=vocab_size, dtype=dtype, src=src, text_preview=ct[:200])

        # Route + write
        to_val = route_to_val(src)
        add_doc(src, ids)

        # Update content totals
        if to_val:
            val_content_total += len(ids)
        else:
            train_content_total += len(ids)

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
        "vocab_size": vocab_size,
        "sources": [{"path": str(p), "weight": float(w)} for p, w in sources],
        "shard_tokens": int(shard_tokens),
        "val_shard_tokens": int(val_shard_tokens),
        "val_ratio": float(val_ratio),
        "min_val_tokens_per_source": int(min_val_tokens_per_source),
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
        # Shard totals include doc_sep tokens
        "train_tokens": int(total_train),
        "val_tokens": int(total_val),
        "train_shards": int(shard_idx_train),
        "val_shards": int(shard_idx_val),
        # Content totals exclude doc_sep tokens
        "train_content_tokens": int(train_content_total),
        "val_content_tokens": int(val_content_total),
        "train_target_tokens": int(target_train_tokens),
        "val_target_tokens_default": int(default_total_val),
        "train_target_per_source": train_target_per_src,
        "val_target_per_source": val_target_per_src,
        "source_fingerprints": source_fingerprints,
        "source_line_precheck": source_line_precheck,
        "first_token_topk": first_tok_meta,
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
    ap.add_argument("--min_val_tokens_per_source", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--doc_sep", type=str, default="", help='optional doc separator, default="\\n\\n"')
    ap.add_argument("--first_token_topk", type=int, default=50)
    ap.add_argument("--precheck_max_lines", type=int, default=300_000)

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
    if args.min_val_tokens_per_source < 0:
        raise SystemExit("--min_val_tokens_per_source must be >= 0")

    write_shards(
        sources=sources,
        out_dir=out_dir,
        tokenizer_path=args.tokenizer_path,
        shard_tokens=args.shard_tokens,
        val_shard_tokens=args.val_shard_tokens,
        val_ratio=args.val_ratio,
        min_val_tokens_per_source=args.min_val_tokens_per_source,
        seed=args.seed,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        target_train_tokens=args.target_train_tokens,
        precheck_max_lines=args.precheck_max_lines,
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
