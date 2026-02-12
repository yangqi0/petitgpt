#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count the distribution of the *first token id* per document after tokenization.

Usage examples:

python pretrain/first_token_stats.py \
  --tokenizer tokenizer/tokenizer_pretrain_nospecial.json \
  --jsonl datasets/tokenization/fineweb_5m.jsonl \
  --text_field text \
  --max_docs 500000 \
  --topk 50

# Multi-file
python pretrain/first_token_stats.py \
  --tokenizer tokenizer/tokenizer_pretrain_nospecial.json \
  --jsonl datasets/tokenization/fineweb_5m.jsonl datasets/tokenization/wiki.clean.jsonl \
  --max_docs 200000 \
  --topk 80
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def load_tokenizer(tokenizer_path: str):
    try:
        from src.tokenizer import Tokenizer  # type: ignore
        return Tokenizer.from_file(tokenizer_path)
    except Exception:
        pass

    from tokenizers import Tokenizer  # type: ignore
    return Tokenizer.from_file(tokenizer_path)


def encode_ids(tokenizer, text: str) -> list[int]:
    # Force add_special_tokens=False
    if hasattr(tokenizer, "encode") and "tokenizers" in type(tokenizer).__module__:
        return tokenizer.encode(text, add_special_tokens=False).ids
    return tokenizer.encode(text, add_special_tokens=False)  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--max_docs", type=int, default=0, help="0=unlimited")
    ap.add_argument("--topk", type=int, default=50)

    args = ap.parse_args()
    tok = load_tokenizer(args.tokenizer)

    first = Counter()
    empty = 0
    bad_json = 0
    no_text = 0
    total_docs = 0

    for jp in args.jsonl:
        p = Path(jp)
        with p.open("r", encoding="utf-8") as f:
            it = f
            if args.max_docs:
                # tqdm needs a fixed-ish total; we keep it simple
                it = tqdm(f, desc=p.name)
            for line in it:
                if args.max_docs and total_docs >= args.max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    bad_json += 1
                    continue
                if not isinstance(obj, dict):
                    bad_json += 1
                    continue
                t = obj.get(args.text_field, None)
                if not isinstance(t, str) or not t.strip():
                    no_text += 1
                    continue

                ids = encode_ids(tok, t)
                total_docs += 1
                if not ids:
                    empty += 1
                    continue
                first[ids[0]] += 1

    print("\n=== first-token topK ===")
    for tid, c in first.most_common(args.topk):
        try:
            s = tok.decode([tid])
        except Exception:
            s = "<decode-fail>"
        # make whitespace visible
        s_vis = s.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
        print(f"id={tid:<6} count={c:<10} frac={c/max(1,total_docs):.6%} text={repr(s_vis)}")

    print("\n=== summary ===")
    print(
        json.dumps(
            {
                "docs_counted": total_docs,
                "empty_ids": empty,
                "bad_json": bad_json,
                "no_text": no_text,
                "unique_first_ids": len(first),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
