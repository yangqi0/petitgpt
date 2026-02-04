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
    """
    Best-effort extraction of text from different jsonl schemas.
    Works for:
      - {"text": "..."} (FineWeb-like)
      - {"content": "..."} / {"message": "..."}
      - instruction datasets: {"instruction","input","output"/"response"}
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
        parts = []
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


def load_tokenizer(tokenizer_path: str):
    """
    Expect your tokenizer can be loaded in one of these ways.
    Adjust this to your project.
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


def encode(
    tokenizer, text: str, add_bos: bool, add_eos: bool, bos_id: int, eos_id: int
) -> list[int]:
    # Support tokenizers.Tokenizer (fast) or custom that has encode returning ids
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


def write_shards(
    jsonl_paths: list[Path],
    out_dir: Path,
    tokenizer_path: str,
    shard_tokens: int,
    val_ratio: float,
    seed: int,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
) -> None:
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tokenizer = load_tokenizer(tokenizer_path)

    buf_train: list[int] = []
    buf_val: list[int] = []
    shard_idx_train = 0
    shard_idx_val = 0
    total_train = 0
    total_val = 0
    seen_docs = 0
    kept_docs = 0

    def flush(buf: list[int], out_path: Path) -> int:
        arr = np.asarray(buf, dtype=np.uint16)
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

    for jp in jsonl_paths:
        print(f"Reading: {jp}")
        for text in tqdm(iter_jsonl_texts(jp), desc=jp.name):
            seen_docs += 1
            ids = encode(tokenizer, text, add_bos, add_eos, bos_id, eos_id)
            if not ids:
                continue
            kept_docs += 1

            if rng.random() < val_ratio:
                buf_val.extend(ids)
                maybe_flush_val()
            else:
                buf_train.extend(ids)
                maybe_flush_train()

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
        "jsonl_paths": [str(p) for p in jsonl_paths],
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
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True, help="Input jsonl files")
    ap.add_argument("--out_dir", required=True, help="Output dir for shards")
    ap.add_argument(
        "--tokenizer_path", required=True, help="Path to tokenizer.json or your tokenizer file"
    )
    ap.add_argument(
        "--shard_tokens", type=int, default=10_000_000, help="Tokens per shard (uint16)"
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

    jsonl_paths = [Path(x) for x in args.jsonl]
    out_dir = Path(args.out_dir)

    write_shards(
        jsonl_paths=jsonl_paths,
        out_dir=out_dir,
        tokenizer_path=args.tokenizer_path,
        shard_tokens=args.shard_tokens,
        val_ratio=args.val_ratio,
        seed=args.seed,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
    )


if __name__ == "__main__":
    main()
