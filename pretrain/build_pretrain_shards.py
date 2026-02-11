# pretrain/build_pretrain_shards.py
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
    """Best-effort extraction of text from different jsonl schemas."""
    for k in ("text", "content", "message", "completion"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

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


def file_fingerprint(p: Path) -> dict[str, Any]:
    """Return a stable-ish identity record to detect 'same path, different file' issues."""
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
    """Count lines up to max_lines; if file is larger, return 'at_least'."""
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if n >= max_lines:
                return {"lines": n, "at_least": True, "max_lines": max_lines}
    return {"lines": n, "at_least": False, "max_lines": max_lines}


class JsonlTextStream:
    def __init__(self, path: Path):
        self.path = path
        self._f = path.open("r", encoding="utf-8")
        self.eof = False

        # counters
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


def load_tokenizer(tokenizer_path: str):
    try:
        from src.tokenizer import Tokenizer  # type: ignore
        return Tokenizer.from_file(tokenizer_path)
    except Exception:
        pass

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


def encode(tokenizer, text: str, add_bos: bool, add_eos: bool, bos_id: int, eos_id: int) -> list[int]:
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
    precheck_max_lines: int,
) -> None:
    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = _tokenizer_vocab_size(tokenizer)
    dtype = np.uint32 if (vocab_size is not None and vocab_size > 65535) else np.uint16

    buf_train: list[int] = []
    buf_val: list[int] = []
    shard_idx_train = 0
    shard_idx_val = 0
    total_train = 0
    total_val = 0
    seen_docs = 0
    kept_docs = 0

    per_src: dict[str, Any] = {}
    source_fingerprints: dict[str, Any] = {}
    source_line_precheck: dict[str, Any] = {}

    if sources:
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
            }
            # record file identity now (helps debug stale mounts)
            source_fingerprints[sp] = file_fingerprint(p)
            source_line_precheck[sp] = fast_count_lines(p, max_lines=precheck_max_lines)

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
            return (False, 0)

        kept_docs += 1
        if src_path is not None:
            per_src[src_path]["kept_docs"] += 1

        # decide split once per doc
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

    if not sources:
        for jp in jsonl_paths:
            print(f"Reading: {jp}")
            for text in tqdm(iter_jsonl_texts(jp), desc=jp.name):
                ids = encode(tokenizer, text, add_bos, add_eos, bos_id, eos_id)
                add_ids(ids, src_path=None)

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

        pbar = tqdm(total=target_train_tokens if target_train_tokens > 0 else None,
                    desc="building (train tokens)")

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

                ids = encode(tokenizer, txt, add_bos, add_eos, bos_id, eos_id)

                # add_ids returns "if extracted in train and how much it contributes to train tokens"
                is_train, n_train = add_ids(ids, src_path=str(src_paths[i]))

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

    meta: dict[str, Any] = {
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
        "source_fingerprints": source_fingerprints,
        "source_line_precheck": source_line_precheck,
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
    ap.add_argument("--shard_tokens", type=int, default=10_000_000)
    ap.add_argument("--val_ratio", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--precheck_max_lines", type=int, default=300_000,
                    help="Count up to this many lines for each source before building (debug).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sources = parse_sources(args.source)

    if sources:
        jsonl_paths: list[Path] = []
    else:
        jsonl_paths = [Path(x) for x in (args.jsonl or [])]
        if not jsonl_paths:
            raise SystemExit("Provide either --source path:weight or --jsonl ...")

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
        precheck_max_lines=args.precheck_max_lines,
    )


if __name__ == "__main__":
    main()
