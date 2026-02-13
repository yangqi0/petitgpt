#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build pretrain binary shards from one or more jsonl sources.

What this script does:
- Weighted multi-source mixing: --source path:weight (repeatable)
- Deterministic train/val split: --seed + --val_ratio
- Train shards of fixed token length: --shard_tokens
- Val shards with independent fixed token length: --val_shard_tokens (lets you have more val files)
- Optional doc separator inserted between documents: --doc_sep (e.g. "\\n\\n")
- Optional tokenizer special leak check: --detect_tokenizer_special_leak
- Writes meta.json containing fingerprints, per-source stats, and first-token distribution.

New (for better text quality & "weird first token" issues):
- Optional quote normalization: --normalize_quotes
- Optional stripping of leading BOM/zero-width/whitespace noise: --strip_leading_noise
- Optional underscore-run handling (e.g., "____"): --underscores_policy {keep,space,drop_doc}
- Optional drop docs by head pattern (cheap but effective): --drop_head_pattern ...
- Optional ASCII ratio filtering: --min_ascii_ratio (good for English pretrain on fineweb/wiki)
- Optional drop if too-short after cleaning: --min_chars

Notes:
- For code-heavy sources, ASCII ratio may reject too much; consider building code jsonl already cleaned
  or run two passes with different policies. This script applies one global policy for simplicity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
from tqdm import tqdm


# -------------------------
# Text extraction
# -------------------------
def guess_text(obj: dict[str, Any]) -> str | None:
    # common single-field schemas
    for k in ("text", "content", "message", "completion"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # instruction-style schemas
    instr = obj.get("instruction")
    inp = obj.get("input")
    out = obj.get("output") or obj.get("response") or obj.get("answer")
    if any(isinstance(x, str) and x.strip() for x in (instr, inp, out)):
        parts: list[str] = []
        if isinstance(instr, str) and instr.strip():
            parts.append(instr.strip())
        if isinstance(inp, str) and inp.strip():
            parts.append(inp.strip())
        if isinstance(out, str) and out.strip():
            parts.append(out.strip())
        return "\n\n".join(parts) if parts else None

    # chat-style schemas
    msgs = obj.get("messages") or obj.get("conversations")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            c = m.get("content") or m.get("value") or m.get("text")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
        if parts:
            return "\n".join(parts)

    return None


def iter_jsonl_texts(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
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
            t = guess_text(obj)
            if t:
                yield t


# -------------------------
# Debug helpers
# -------------------------
def file_fingerprint(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {
        "path": str(p),
        "resolved": str(p.resolve()),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "inode": int(st.st_ino),
        "device": int(st.st_dev),
    }


def fast_count_lines(p: Path, max_lines: int = 300_000) -> dict[str, Any]:
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if n >= max_lines:
                return {"lines": n, "at_least": True, "max_lines": max_lines}
    return {"lines": n, "at_least": False, "max_lines": max_lines}


# -------------------------
# Streaming reader
# -------------------------
class JsonlTextStream:
    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("r", encoding="utf-8")
        self.eof = False

        self.lines_read = 0
        self.bad_json = 0
        self.non_dict = 0
        self.no_text = 0
        self.ok_texts = 0

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def next_text(self) -> str | None:
        if self.eof:
            return None
        while True:
            line = self._f.readline()
            if not line:
                self.eof = True
                return None
            self.lines_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                self.bad_json += 1
                continue
            if not isinstance(obj, dict):
                self.non_dict += 1
                continue
            t = guess_text(obj)
            if not t:
                self.no_text += 1
                continue
            self.ok_texts += 1
            return t


# -------------------------
# Tokenizer
# -------------------------
def load_tokenizer(tokenizer_path: str):
    # project tokenizer (optional)
    try:
        from src.tokenizer import Tokenizer  # type: ignore
        return Tokenizer.from_file(tokenizer_path)
    except Exception:
        pass

    # HuggingFace tokenizers
    try:
        from tokenizers import Tokenizer  # type: ignore
        return Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        raise RuntimeError(
            "Cannot load tokenizer. Please adjust load_tokenizer() to your project.\n"
            f"tokenizer_path={tokenizer_path}\nerror={e}"
        )


def _tokenizer_vocab_size(tokenizer) -> int | None:
    if hasattr(tokenizer, "get_vocab_size"):
        try:
            return int(tokenizer.get_vocab_size())  # type: ignore
        except Exception:
            return None
    for attr in ("vocab_size", "n_vocab"):
        if hasattr(tokenizer, attr):
            try:
                return int(getattr(tokenizer, attr))
            except Exception:
                return None
    return None


def _encode_base(tokenizer, text: str) -> list[int]:
    # tokenizers.Tokenizer path
    if hasattr(tokenizer, "encode") and "tokenizers" in type(tokenizer).__module__:
        return tokenizer.encode(text, add_special_tokens=False).ids
    # project tokenizer path
    return tokenizer.encode(text, add_special_tokens=False)  # type: ignore


def _detect_special_leak(tokenizer, bos_id: int, eos_id: int) -> dict[str, Any]:
    probe = "Hello world."
    try:
        ids = _encode_base(tokenizer, probe)
    except Exception as e:
        return {"ok": False, "error": repr(e)}

    leak_bos = (len(ids) >= 1 and ids[0] == bos_id)
    leak_eos = (len(ids) >= 1 and ids[-1] == eos_id)
    leak_any = leak_bos or leak_eos
    return {
        "ok": True,
        "probe_ids_head": ids[:8],
        "probe_ids_tail": ids[-8:] if len(ids) >= 8 else ids,
        "leak_bos": bool(leak_bos),
        "leak_eos": bool(leak_eos),
        "leak_any": bool(leak_any),
    }


# -------------------------
# Text cleaning / filtering
# -------------------------

# Common zero-width & BOM
_RE_LEADING_NOISE = re.compile(r"^(?:\ufeff|[\u200b\u200c\u200d\u2060\u180e])*")

# normalize fancy quotes to ascii
_QUOTE_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "‘": "'",
        "‚": "'",
        "‐": "-",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)

_RE_UNDERSCORE_RUN = re.compile(r"_{4,}")  # ____ or longer

# Head pattern classifiers (similar to your head_pattern_stats.py)
_RE_PIPES1 = re.compile(r"^\|\s*")
_RE_PIPES2 = re.compile(r"^\|\|\s*")
_RE_HEADING = re.compile(r"^#{1,6}\s+")
_RE_DASH_LIST = re.compile(r"^(?:-\s+|\*\s+)")
_RE_BLOCKQUOTE = re.compile(r"^>\s*")
_RE_DIGIT = re.compile(r"^\d")
_RE_UNDERSCORES = re.compile(r"^_{2,}")
_RE_OTHER_SYMBOL = re.compile(r"^[^\wA-Za-z]")  # broad; use with care


def head_pattern(text: str) -> str:
    s = text.lstrip()
    if not s:
        return "empty"
    if _RE_PIPES2.match(s):
        return "pipes||"
    if _RE_PIPES1.match(s):
        return "pipes|"
    if _RE_HEADING.match(s):
        return "heading#"
    if _RE_BLOCKQUOTE.match(s):
        return "blockquote>"
    if _RE_DASH_LIST.match(s):
        return "dash_list"
    if _RE_UNDERSCORES.match(s):
        return "underscores"
    if _RE_DIGIT.match(s):
        return "digit"
    # alpha: starts with A-Za-z
    if "A" <= s[0] <= "Z" or "a" <= s[0] <= "z":
        return "alpha"
    # otherwise
    if _RE_OTHER_SYMBOL.match(s):
        return "other_symbol"
    return "other"


def ascii_ratio(text: str) -> float:
    # ratio over non-whitespace chars
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    ascii_cnt = 0
    for c in chars:
        o = ord(c)
        if 32 <= o <= 126:
            ascii_cnt += 1
    return ascii_cnt / len(chars)


def clean_text(
    text: str,
    *,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
) -> str:
    t = text

    if strip_leading_noise:
        t = _RE_LEADING_NOISE.sub("", t)
        # also strip leading whitespace (very common)
        t = t.lstrip()

    if normalize_quotes:
        t = t.translate(_QUOTE_MAP)

    if underscores_policy == "keep":
        pass
    elif underscores_policy == "space":
        # turn long runs into a single space (prevents "____" becoming a frequent token)
        t = _RE_UNDERSCORE_RUN.sub(" ", t)
    elif underscores_policy == "drop_doc":
        if _RE_UNDERSCORE_RUN.search(t):
            return ""
    else:
        raise ValueError(f"bad underscores_policy={underscores_policy!r}")

    return t


# -------------------------
# CLI parsing
# -------------------------
def parse_sources(source_args: list[str]) -> list[tuple[Path, float]]:
    items: list[tuple[Path, float]] = []
    for s in source_args:
        if ":" not in s:
            raise ValueError(f"--source must be like path:weight, got {s!r}")
        p, w = s.rsplit(":", 1)
        p = p.strip()
        w = w.strip()
        if not p:
            raise ValueError(f"Bad --source (empty path): {s!r}")
        wf = float(w)
        if wf <= 0:
            raise ValueError(f"--source weight must be > 0, got {wf} in {s!r}")
        items.append((Path(p), wf))
    if not items:
        return []
    tot = sum(w for _, w in items)
    return [(p, w / tot) for p, w in items]


def topk_counter(c: Counter[int], k: int) -> list[dict[str, Any]]:
    return [{"id": int(tid), "count": int(cnt)} for tid, cnt in c.most_common(k)]


# -------------------------
# Shard writer
# -------------------------
def write_shards(
    *,
    jsonl_paths: list[Path],
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
    # new filters
    min_chars: int,
    min_ascii_ratio: float,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    drop_head_patterns: list[str],
) -> None:
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = _tokenizer_vocab_size(tokenizer)
    dtype = np.uint32 if (vocab_size is not None and vocab_size > 65535) else np.uint16

    # doc separator ids (mild boundary)
    doc_sep_ids: list[int] = []
    if doc_sep:
        doc_sep_ids = _encode_base(tokenizer, doc_sep)

    # optional: detect BOS/EOS leakage
    special_leak = None
    if detect_tokenizer_special_leak:
        special_leak = _detect_special_leak(tokenizer, bos_id=bos_id, eos_id=eos_id)
        if special_leak.get("ok") and special_leak.get("leak_any"):
            print(
                "\n[WARNING] Tokenizer appears to add BOS/EOS even with add_special_tokens=False.\n"
                "          This means using --add_bos/--add_eos here can DOUBLE-ADD special tokens.\n"
                "          Strongly consider building shards WITHOUT --add_bos/--add_eos.\n"
                f"          leak_bos={special_leak.get('leak_bos')} leak_eos={special_leak.get('leak_eos')}\n"
                f"          probe_ids_head={special_leak.get('probe_ids_head')} probe_ids_tail={special_leak.get('probe_ids_tail')}\n"
            )

    # buffers
    buf_train: list[int] = []
    buf_val: list[int] = []
    shard_idx_train = 0
    shard_idx_val = 0
    total_train = 0
    total_val = 0

    # counters
    seen_docs = 0
    kept_docs = 0
    dropped_docs = 0
    drop_reasons: Counter[str] = Counter()

    per_src: dict[str, Any] = {}
    source_fingerprints: dict[str, Any] = {}
    source_line_precheck: dict[str, Any] = {}

    # first-token distribution
    first_tok_all: Counter[int] = Counter()
    first_tok_per_src: dict[str, Counter[int]] = {}

    if sources:
        for p, w in sources:
            sp = str(p)
            per_src[sp] = {
                "weight": float(w),
                "seen_docs": 0,
                "kept_docs": 0,
                "dropped_docs": 0,
                "train_tokens": 0,
                "val_tokens": 0,
                "lines_read": 0,
                "bad_json": 0,
                "non_dict": 0,
                "no_text": 0,
                "ok_texts": 0,
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

    def encode_doc(text: str) -> list[int]:
        ids = _encode_base(tokenizer, text)
        if add_bos:
            ids = [bos_id] + ids
        if add_eos:
            ids = ids + [eos_id]
        if doc_sep_ids:
            ids = ids + doc_sep_ids
        return ids

    def accept_or_drop(text: str, *, src_path: str | None) -> str | None:
        nonlocal dropped_docs

        t = clean_text(
            text,
            strip_leading_noise=strip_leading_noise,
            normalize_quotes=normalize_quotes,
            underscores_policy=underscores_policy,
        )

        if not t or not t.strip():
            dropped_docs += 1
            drop_reasons["empty_after_clean"] += 1
            if src_path is not None:
                per_src[src_path]["dropped_docs"] += 1
            return None

        if min_chars > 0 and len(t) < min_chars:
            dropped_docs += 1
            drop_reasons["min_chars"] += 1
            if src_path is not None:
                per_src[src_path]["dropped_docs"] += 1
            return None

        hp = head_pattern(t)
        if drop_head_patterns and hp in drop_head_patterns:
            dropped_docs += 1
            drop_reasons[f"head:{hp}"] += 1
            if src_path is not None:
                per_src[src_path]["dropped_docs"] += 1
            return None

        if min_ascii_ratio > 0:
            ar = ascii_ratio(t)
            if ar < min_ascii_ratio:
                dropped_docs += 1
                drop_reasons["min_ascii_ratio"] += 1
                if src_path is not None:
                    per_src[src_path]["dropped_docs"] += 1
                return None

        return t

    def add_ids(ids: list[int], src_path: str | None):
        nonlocal seen_docs, kept_docs

        seen_docs += 1
        if src_path is not None:
            per_src[src_path]["seen_docs"] += 1

        if not ids:
            dropped = True
        else:
            dropped = False

        if dropped:
            return (False, 0)

        # record first token
        first_tok_all[ids[0]] += 1
        if src_path is not None:
            first_tok_per_src[src_path][ids[0]] += 1

        kept_docs += 1
        if src_path is not None:
            per_src[src_path]["kept_docs"] += 1

        if rng.random() < val_ratio:
            buf_val.extend(ids)
            if src_path is not None:
                per_src[src_path]["val_tokens"] += len(ids)
            maybe_flush_val()
            return (False, 0)
        else:
            buf_train.extend(ids)
            if src_path is not None:
                per_src[src_path]["train_tokens"] += len(ids)
            maybe_flush_train()
            return (True, len(ids))

    # legacy sequential mode
    if not sources:
        for jp in jsonl_paths:
            print(f"Reading: {jp}")
            for text in tqdm(iter_jsonl_texts(jp), desc=jp.name):
                t = accept_or_drop(text, src_path=None)
                if t is None:
                    continue
                ids = encode_doc(t)
                add_ids(ids, src_path=None)

    # weighted multi-source mode
    else:
        src_paths = [p for p, _ in sources]
        src_weights = [w for _, w in sources]
        streams = [JsonlTextStream(p) for p in src_paths]
        exhausted = [False for _ in sources]

        quota = None
        used_train = None
        if target_train_tokens > 0:
            quota = [int(target_train_tokens * w) for w in src_weights]
            diff = target_train_tokens - sum(quota)
            for i in range(abs(diff)):
                quota[i % len(quota)] += 1 if diff > 0 else -1
            used_train = [0 for _ in sources]

        pbar = tqdm(
            total=target_train_tokens if target_train_tokens > 0 else None,
            desc="building (train tokens)",
        )

        def pick_source_idx() -> int | None:
            active = []
            weights = []
            for i in range(len(sources)):
                if exhausted[i]:
                    continue
                if quota is not None and used_train is not None:
                    remain = quota[i] - used_train[i]
                    if remain <= 0:
                        continue
                    active.append(i)
                    weights.append(float(remain))
                else:
                    active.append(i)
                    weights.append(float(src_weights[i]))
            if not active:
                return None
            s = sum(weights)
            r = rng.random() * s
            acc = 0.0
            for idx, w in zip(active, weights):
                acc += w
                if r <= acc:
                    return idx
            return active[-1]

        try:
            while True:
                if quota is not None and used_train is not None and sum(used_train) >= target_train_tokens:
                    break

                i = pick_source_idx()
                if i is None:
                    break

                txt = streams[i].next_text()
                if txt is None:
                    exhausted[i] = True
                    continue

                sp = str(src_paths[i])
                t = accept_or_drop(txt, src_path=sp)
                if t is None:
                    continue

                ids = encode_doc(t)
                is_train, n_train = add_ids(ids, src_path=sp)

                if quota is not None and used_train is not None:
                    used_train[i] += n_train
                pbar.update(n_train)

        finally:
            pbar.close()
            for p, s in zip(src_paths, streams):
                sp = str(p)
                per_src[sp]["lines_read"] = s.lines_read
                per_src[sp]["bad_json"] = s.bad_json
                per_src[sp]["non_dict"] = s.non_dict
                per_src[sp]["no_text"] = s.no_text
                per_src[sp]["ok_texts"] = s.ok_texts
                s.close()

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

    # summarize first-token stats into meta
    first_tok_meta: dict[str, Any] = {
        "topk": int(first_token_topk),
        "global": topk_counter(first_tok_all, first_token_topk),
    }
    if sources:
        per = {}
        for sp, c in first_tok_per_src.items():
            per[sp] = topk_counter(c, first_token_topk)
        first_tok_meta["per_source"] = per

    meta: dict[str, Any] = {
        "tokenizer_path": tokenizer_path,
        "dtype": "uint32" if dtype == np.uint32 else "uint16",
        "jsonl_paths": [str(p) for p in jsonl_paths] if jsonl_paths else [],
        "sources": [{"path": str(p), "weight": float(w)} for p, w in sources] if sources else [],
        "shard_tokens": shard_tokens,
        "val_shard_tokens": val_shard_tokens,
        "val_ratio": val_ratio,
        "seed": seed,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "doc_sep": doc_sep,
        "doc_sep_ids_head": doc_sep_ids[:16],
        "seen_docs": seen_docs,
        "kept_docs": kept_docs,
        "dropped_docs": dropped_docs,
        "drop_reasons_top": [{"reason": k, "count": int(v)} for k, v in drop_reasons.most_common(20)],
        "train_tokens": total_train,
        "val_tokens": total_val,
        "train_shards": shard_idx_train,
        "val_shards": shard_idx_val,
        "source_fingerprints": source_fingerprints,
        "source_line_precheck": source_line_precheck,
        "tokenizer_special_leak_check": special_leak,
        "first_token_topk": first_tok_meta,
        # filters config snapshot
        "filters": {
            "min_chars": int(min_chars),
            "min_ascii_ratio": float(min_ascii_ratio),
            "strip_leading_noise": bool(strip_leading_noise),
            "normalize_quotes": bool(normalize_quotes),
            "underscores_policy": underscores_policy,
            "drop_head_patterns": drop_head_patterns,
        },
    }
    if per_src:
        meta["per_source"] = per_src
    if sources and target_train_tokens > 0:
        meta["target_train_tokens"] = int(target_train_tokens)

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source", action="append", default=[], help="path:weight (repeatable)")
    ap.add_argument("--target_train_tokens", type=int, default=0)
    ap.add_argument("--jsonl", nargs="*", default=[], help="legacy sequential mode")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--shard_tokens", type=int, default=10_000_000, help="train shard size in tokens")
    ap.add_argument(
        "--val_shard_tokens",
        type=int,
        default=2_000_000,
        help="val shard size in tokens (smaller => more val files without increasing val_ratio)",
    )
    ap.add_argument("--val_ratio", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--doc_sep", type=str, default="", help='optional document separator, e.g. "\\n\\n"')
    ap.add_argument("--first_token_topk", type=int, default=50, help="topK first-token ids to store in meta.json")

    ap.add_argument(
        "--precheck_max_lines",
        type=int,
        default=300_000,
        help="Count up to this many lines for each source before building (debug).",
    )
    ap.add_argument(
        "--detect_tokenizer_special_leak",
        action="store_true",
        help="Warn if tokenizer adds BOS/EOS even with add_special_tokens=False (avoid double-add).",
    )

    # ---------- new filters ----------
    ap.add_argument("--min_chars", type=int, default=200, help="drop doc if shorter than this after cleaning (0 disables)")
    ap.add_argument(
        "--min_ascii_ratio",
        type=float,
        default=0.0,
        help="drop doc if ascii_ratio < this (0 disables). For English fineweb/wiki, 0.95~0.98 is common.",
    )
    ap.add_argument("--strip_leading_noise", action="store_true", help="strip BOM/zero-width + leading whitespace")
    ap.add_argument("--normalize_quotes", action="store_true", help="normalize curly quotes/dashes to ASCII")
    ap.add_argument(
        "--underscores_policy",
        type=str,
        default="keep",
        choices=["keep", "space", "drop_doc"],
        help='how to handle long underscore runs like "____": keep | space | drop_doc',
    )
    ap.add_argument(
        "--drop_head_pattern",
        action="append",
        default=[],
        help="drop docs whose head pattern matches (repeatable). "
             "choices include: pipes|, pipes||, heading#, dash_list, blockquote>, underscores, digit, other_symbol",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sources = parse_sources(args.source)

    if sources:
        jsonl_paths: list[Path] = []
    else:
        jsonl_paths = [Path(x) for x in (args.jsonl or [])]
        if not jsonl_paths:
            raise SystemExit("Provide either --source path:weight or --jsonl ...")

    if args.val_shard_tokens <= 0:
        raise SystemExit("--val_shard_tokens must be > 0")
    if args.shard_tokens <= 0:
        raise SystemExit("--shard_tokens must be > 0")
    if args.val_ratio < 0 or args.val_ratio > 1:
        raise SystemExit("--val_ratio must be in [0,1]")
    if args.min_ascii_ratio < 0 or args.min_ascii_ratio > 1:
        raise SystemExit("--min_ascii_ratio must be in [0,1]")

    write_shards(
        jsonl_paths=jsonl_paths,
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
        # filters
        min_chars=args.min_chars,
        min_ascii_ratio=args.min_ascii_ratio,
        strip_leading_noise=args.strip_leading_noise,
        normalize_quotes=args.normalize_quotes,
        underscores_policy=args.underscores_policy,
        drop_head_patterns=args.drop_head_pattern,
    )


if __name__ == "__main__":
    main()
