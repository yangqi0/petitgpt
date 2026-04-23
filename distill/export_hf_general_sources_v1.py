from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def maybe_limit(rows: List[Dict[str, Any]], limit: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[:limit]
    idx.sort()
    return [rows[i] for i in idx]

def export_no_robots(split: str, out_jsonl: str, limit: Optional[int], seed: int) -> None:
    ds = load_dataset("HuggingFaceH4/no_robots", split=split)
    rows = []
    for row in ds:
        rows.append({
            "prompt_id": row.get("prompt_id"),
            "prompt": row.get("prompt", ""),
            "messages": row.get("messages", []),
            "category": row.get("category", ""),
        })
    rows = maybe_limit(rows, limit, seed)
    write_jsonl(out_jsonl, rows)
    print(f"[no_robots] wrote {len(rows)} -> {out_jsonl}")

def export_alpaca_cleaned(split: str, out_jsonl: str, limit: Optional[int], seed: int) -> None:
    ds = load_dataset("yahma/alpaca-cleaned", split=split)
    rows = []
    for i, row in enumerate(ds):
        rows.append({
            "id": row.get("id", i),
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
        })
    rows = maybe_limit(rows, limit, seed)
    write_jsonl(out_jsonl, rows)
    print(f"[alpaca_cleaned] wrote {len(rows)} -> {out_jsonl}")

def export_dolly(split: str, out_jsonl: str, limit: Optional[int], seed: int) -> None:
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    rows = []
    for i, row in enumerate(ds):
        rows.append({
            "id": row.get("id", i),
            "instruction": row.get("instruction", ""),
            "context": row.get("context", ""),
            "response": row.get("response", ""),
            "category": row.get("category", ""),
        })
    rows = maybe_limit(rows, limit, seed)
    write_jsonl(out_jsonl, rows)
    print(f"[dolly] wrote {len(rows)} -> {out_jsonl}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory, e.g. data/")
    ap.add_argument("--no_robots_split", default="train_sft")
    ap.add_argument("--alpaca_split", default="train")
    ap.add_argument("--dolly_split", default="train")
    ap.add_argument("--limit_no_robots", type=int, default=0)
    ap.add_argument("--limit_alpaca", type=int, default=0)
    ap.add_argument("--limit_dolly", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_no_robots(
        split=args.no_robots_split,
        out_jsonl=str(out_dir / "no_robots_train_sft.jsonl"),
        limit=args.limit_no_robots or None,
        seed=args.seed,
    )
    export_alpaca_cleaned(
        split=args.alpaca_split,
        out_jsonl=str(out_dir / "alpaca_cleaned_train.jsonl"),
        limit=args.limit_alpaca or None,
        seed=args.seed,
    )
    export_dolly(
        split=args.dolly_split,
        out_jsonl=str(out_dir / "dolly_style_train.jsonl"),
        limit=args.limit_dolly or None,
        seed=args.seed,
    )
    print("Done.")

if __name__ == "__main__":
    main()
