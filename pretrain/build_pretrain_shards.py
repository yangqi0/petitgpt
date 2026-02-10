# scripts/build_pretrain_shards.py
from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from tqdm import tqdm


def guess_text(obj: dict[str, Any]) -> str | None:
    """Best-effort extraction of text from different jsonl schemas.

    Works for:
      - {"text": "..."} (FineWeb-like)
      - {"content": "..."} / {"message": "..."} / {"completion": "..."}
      - instruction datasets: {"instruction","input","output"/"response"/"answer"}
      - chat datasets: {"messages":[{"role","content"},...]} / {"conversations":[...]}
    """
    # most common
    for k in ("text", "content", "message", "completion"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # instruction-style
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

    # chat-style
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
    """Yield extracted texts from a jsonl file."""
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


class JsonlTextStream:
    """Streaming jsonl reader that yields best-effort extracted texts."""

    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("r", encoding="utf-8")
        self.eof = False

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
                return t


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer.

    Supports:
      - tokenizers.Tokenizer from HuggingFace (tokenizer.json)
      - a project-local Tokenizer at src.tokenizer.Tokenizer with from_file()
    """
    # Option A: your own tokenizer module
    try:
        from src.tokenizer import Tokenizer  # type: ignore

        return Tokenizer.from_file(tokenizer_path)
    except Exception:
        pass

    # Option B: HuggingFace tokenizer.json
    try:
        from tokenizers import Tokenizer  # type: ignore

        return Tokenizer.from_file(tokenizer_path)
    except Exception as e:
        raise RuntimeError(
            "Cannot load tokenizer. Please adjust load_tokenizer() to your project.\n"
            f"tokenizer_path={tokenizer_path}\nerror={e}"
        )


def _tokenizer_vocab_size(tokenizer) -> int | None:
    # tokenizers.Tokenizer
    if hasattr(tokenizer, "get_vocab_size"):
        try:
            return int(tokenizer.get_vocab_size())  # type: ignore
        except Exception:
            return None
    # custom
    for attr in ("vocab_size", "n_vocab"):
        if hasattr(tokenizer, attr):
            try:
                return int(getattr(tokenizer, attr))
            except Exception:
                return None
    return None


def encode(
    tokenizer, text: str, add_bos: bool, add_eos: bool, bos_id: int, eos_id: int
) -> list[int]:
    """Encode a single text into token ids."""
    ids: list[int]
    if hasattr(tokenizer, "encode") and "tokenizers" in type(tokenizer).__module__:
        enc = tokenizer.encode(text)
        ids = enc.ids
    else:
        ids = tokenizer.encode(text)  # type: ignore

    if add_bos:
        ids = [bos_id] + ids
    if add_eos:
        ids = ids + [eos_id]
    return ids


def parse_sources(source_args: list[str]) -> list[tuple[Path, float]]:
    """Parse --source path:weight entries, return normalized weights."""
    items: list[tuple[Path, float]] = []
    for s in source_args:
        if ":" not in s:
            raise ValueError(f"--source must be like path:weight, got {s!r}")
        p, w = s.rsplit(":", 1)
        p = p.strip()
        w = w.strip()
        if not p:
            raise ValueError(f"Bad --source (empty path): {s!r}")
        try:
            wf = float(w)
        except ValueError as e:
            raise ValueError(f"Bad --source weight in {s!r}: {e}") from e
        if wf <= 0:
            raise ValueError(f"--source weight must be > 0, got {wf} in {s!r}")
        items.append((Path(p), wf))

    if not items:
        return []

    tot = sum(w for _, w in items)
    return [(p, w / tot) for p, w in items]


def write_shards(
    *,
    jsonl_paths: list[Path],
    sources: list[tuple[Path, float]],
    out_dir: Path,
    tokenizer_path: str,
    shard_tokens: int,
    val_ratio: float,
    seed: int,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    target_train_tokens: int,
) -> None:
    """Build packed token shards.

    Modes:
      - Legacy mode: provide jsonl_paths (non-empty) and sources empty -> read in order.
      - Mixed mode: provide sources (non-empty) -> sample documents across sources according to weights,
        and optionally stop after ~target_train_tokens tokens have been written to train split.

    Notes:
      - val split is decided per *document* with probability val_ratio (same behavior as before).
      - token accounting is approximate at document granularity.
      - shards are written as uint16 by default if vocab_size <= 65535, else uint32.
    """
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = _tokenizer_vocab_size(tokenizer)
    if vocab_size is not None and vocab_size > 65535:
        dtype = np.uint32
    else:
        dtype = np.uint16

    buf_train: list[int] = []
    buf_val: list[int] = []
    shard_idx_train = 0
    shard_idx_val = 0
    total_train = 0
    total_val = 0
    seen_docs = 0
    kept_docs = 0

    # Per-source bookkeeping (only meaningful in mixed mode)
    per_src = {}
    if sources:
        for p, w in sources:
            per_src[str(p)] = {
                "weight": float(w),
                "seen_docs": 0,
                "kept_docs": 0,
                "train_tokens": 0,
                "val_tokens": 0,
            }

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
        while len(buf_val) >= shard_tokens:
            chunk = buf_val[:shard_tokens]
            buf_val = buf_val[shard_tokens:]
            p = out_val / f"shard_{shard_idx_val:05d}.bin"
            n = flush(chunk, p)
            total_val += n
            shard_idx_val += 1

    def add_ids(ids: list[int], src_path: str | None):
        nonlocal seen_docs, kept_docs
        seen_docs += 1
        if src_path is not None:
            per_src[src_path]["seen_docs"] += 1

        if not ids:
            return

        kept_docs += 1
        if src_path is not None:
            per_src[src_path]["kept_docs"] += 1

        if rng.random() < val_ratio:
            buf_val.extend(ids)
            if src_path is not None:
                per_src[src_path]["val_tokens"] += len(ids)
            maybe_flush_val()
        else:
            buf_train.extend(ids)
            if src_path is not None:
                per_src[src_path]["train_tokens"] += len(ids)
            maybe_flush_train()

    # -----------------------
    # Legacy mode: sequential
    # -----------------------
    if not sources:
        for jp in jsonl_paths:
            print(f"Reading: {jp}")
            for text in tqdm(iter_jsonl_texts(jp), desc=jp.name):
                ids = encode(tokenizer, text, add_bos, add_eos, bos_id, eos_id)
                add_ids(ids, src_path=None)

    # -----------------------
    # Mixed mode: weighted + token quota
    # -----------------------
    else:
        src_paths = [p for p, _ in sources]
        # normalized weights already
        src_weights = [w for _, w in sources]
        streams = [JsonlTextStream(p) for p in src_paths]
        exhausted = [False for _ in sources]

        # token quota is applied to *train* tokens (because that's what drives learning)
        quota = None
        used_train = None
        if target_train_tokens > 0:
            quota = [int(target_train_tokens * w) for w in src_weights]
            # fix rounding to hit exact total
            diff = target_train_tokens - sum(quota)
            # distribute the remainder
            for i in range(abs(diff)):
                quota[i % len(quota)] += 1 if diff > 0 else -1
            used_train = [0 for _ in sources]

        pbar = tqdm(
            total=target_train_tokens if target_train_tokens > 0 else None,
            desc="building (train tokens)",
        )

        def pick_source_idx() -> int | None:
            # active sources: not exhausted and (if quota) not full
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
                    weights.append(float(remain))  # proportional to remaining quota
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
                if quota is not None and used_train is not None and total_train >= target_train_tokens:
                    break

                i = pick_source_idx()
                if i is None:
                    break

                txt = streams[i].next_text()
                if txt is None:
                    exhausted[i] = True
                    continue

                ids = encode(tokenizer, txt, add_bos, add_eos, bos_id, eos_id)
                before_train = total_train
                add_ids(ids, src_path=str(src_paths[i]))

                # update train quota tracker
                if quota is not None and used_train is not None:
                    used_train[i] += max(total_train - before_train, 0)  # only train tokens count
                    # pbar uses train tokens
                    pbar.update(max(total_train - before_train, 0))
                else:
                    # if no quota, still update pbar by train tokens to show progress
                    pbar.update(max(total_train - before_train, 0))

        finally:
            pbar.close()
            for s in streams:
                s.close()

    # flush remainders (keep them as last partial shard)
    if buf_train:
        p = out_train / f"shard_{shard_idx_train:05d}.bin"
        n = flush(buf_train, p)
        total_train += n
        shard_idx_train += 1
        buf_train = []
    if buf_val:
        p = out_val / f"shard_{shard_idx_val:05d}.bin"
        n = flush(buf_val, p)
        total_val += n
        shard_idx_val += 1
        buf_val = []

    meta = {
        "tokenizer_path": tokenizer_path,
        "dtype": "uint32" if dtype == np.uint32 else "uint16",
        "jsonl_paths": [str(p) for p in jsonl_paths] if jsonl_paths else [],
        "sources": [{"path": str(p), "weight": float(w)} for p, w in sources] if sources else [],
        "shard_tokens": shard_tokens,
        "val_ratio": val_ratio,
        "seed": seed,
        "add_bos": add_bos,
        "add_eos": add_eos,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "seen_docs": seen_docs,
        "kept_docs": kept_docs,
        "train_tokens": total_train,
        "val_tokens": total_val,
        "train_shards": shard_idx_train,
        "val_shards": shard_idx_val,
    }
    if per_src:
        meta["per_source"] = per_src
    if sources and target_train_tokens > 0:
        meta["target_train_tokens"] = int(target_train_tokens)

    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()

    # New mode: weighted sources (recommended)
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help="Data source in form path:weight . Can be used multiple times.",
    )
    ap.add_argument(
        "--target_train_tokens",
        type=int,
        default=0,
        help="Stop after producing ~this many train tokens (0 means read all available data).",
    )

    # Legacy mode: sequential jsonl list (kept for backward compatibility)
    ap.add_argument(
        "--jsonl",
        nargs="*",
        default=[],
        help="Input jsonl files (legacy mode). If --source is provided, --jsonl is ignored.",
    )

    ap.add_argument("--out_dir", required=True, help="Output dir for shards")
    ap.add_argument(
        "--tokenizer_path", required=True, help="Path to tokenizer.json or your tokenizer file"
    )
    ap.add_argument(
        "--shard_tokens", type=int, default=10_000_000, help="Tokens per shard (uint16/uint32)"
    )
    ap.add_argument(
        "--val_ratio", type=float, default=0.002, help="Probability a document goes to val"
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    sources = parse_sources(args.source)
    if sources:
        jsonl_paths: list[Path] = []
    else:
        jsonl_paths = [Path(x) for x in (args.jsonl or [])]
        if not jsonl_paths:
            raise SystemExit(
                "You must provide either --source path:weight (recommended) or --jsonl file1 file2 ... (legacy)."
            )

    write_shards(
        jsonl_paths=jsonl_paths,
        sources=sources,
        out_dir=out_dir,
        tokenizer_path=args.tokenizer_path,
        shard_tokens=args.shard_tokens,
        val_ratio=args.val_ratio,
        seed=args.seed,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        target_train_tokens=args.target_train_tokens,
    )


if __name__ == "__main__":
    main()
