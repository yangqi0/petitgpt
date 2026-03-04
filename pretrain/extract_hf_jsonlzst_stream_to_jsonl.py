#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream *.jsonl.zst files from a HF dataset repo (no datasets loading script),
decompress on the fly, and write JSONL: {"text": "..."} per line.

Example (Proof-Pile-2 algebraic-stack train):
python pretrain/extract_hf_jsonlzst_stream_to_jsonl.py \
  --dataset EleutherAI/proof-pile-2 \
  --subset algebraic-stack \
  --split train \
  --text_field text \
  --out datasets/raw/proofpile2_algebraic.raw.jsonl \
  --max_rows 800000 \
  --shuffle_buffer 20000 \
  --seed 1234
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from typing import Iterable, List

import requests
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_url


def iter_repo_jsonlzst_files(repo_id: str, prefix: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return [p for p in files if p.startswith(prefix) and p.endswith(".jsonl.zst")]


def stream_jsonl_zst(url: str) -> Iterable[dict]:
    # Stream download + zstd decompress
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(r.raw) as zr:
            buf = io.BufferedReader(zr)
            while True:
                line = buf.readline()
                if not line:
                    break
                # json per line
                try:
                    obj = json.loads(line.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset repo id, e.g. EleutherAI/proof-pile-2")
    ap.add_argument("--subset", required=True, help="subdir, e.g. algebraic-stack / arxiv / open-web-math")
    ap.add_argument("--split", default="train", help="train|validation|test")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--shuffle_buffer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=200000)
    ap.add_argument("--hard_exit", action="store_true", help="os._exit(0) after finishing (avoid finalizing crashes)")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    prefix = f"{args.subset}/{args.split}/"
    paths = iter_repo_jsonlzst_files(args.dataset, prefix)
    if not paths:
        raise RuntimeError(f"No .jsonl.zst files found under {prefix} in {args.dataset}")

    # Shuffle file order a bit to improve mixing
    random.shuffle(paths)

    n = 0
    buffer: List[str] = []

    def emit_text(fh, text: str):
        nonlocal n
        fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        n += 1
        if args.log_every and n % args.log_every == 0:
            print(f"[progress] {n} rows", flush=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for relpath in paths:
            url = hf_hub_url(repo_id=args.dataset, filename=relpath, repo_type="dataset")
            for obj in stream_jsonl_zst(url):
                t = obj.get(args.text_field, None)
                if not isinstance(t, str):
                    continue
                t = t.strip()
                if not t:
                    continue

                if args.shuffle_buffer and args.shuffle_buffer > 0:
                    buffer.append(t)
                    if len(buffer) >= args.shuffle_buffer:
                        j = random.randrange(len(buffer))
                        emit_text(out_f, buffer.pop(j))
                else:
                    emit_text(out_f, t)

                if args.max_rows and n >= args.max_rows:
                    break

            if args.max_rows and n >= args.max_rows:
                break

        # flush remaining buffer
        if buffer:
            random.shuffle(buffer)
            for t in buffer:
                if args.max_rows and n >= args.max_rows:
                    break
                emit_text(out_f, t)

    print(f"[done] wrote {n} rows -> {args.out}", flush=True)

    if args.hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
