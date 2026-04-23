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

def export_mbpp(config_name: str, split: str, out_jsonl: str, limit: Optional[int], seed: int) -> None:
    if config_name:
        ds = load_dataset("mbpp", config_name, split=split)
    else:
        ds = load_dataset("mbpp", split=split)
    rows = []
    for row in ds:
        rows.append(dict(row))
    rows = maybe_limit(rows, limit, seed)
    write_jsonl(out_jsonl, rows)
    print(f"[mbpp] wrote {len(rows)} -> {out_jsonl}")

def export_apps(split: str, out_jsonl: str, limit: Optional[int], seed: int) -> None:
    ds = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
    rows = []
    for row in ds:
        rows.append(dict(row))
    rows = maybe_limit(rows, limit, seed)
    write_jsonl(out_jsonl, rows)
    print(f"[apps] wrote {len(rows)} -> {out_jsonl}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Output directory, e.g. data/")
    ap.add_argument("--mbpp_config", default="", help="Default '' means load_dataset('mbpp'). Use 'sanitized' if you want sanitized.")
    ap.add_argument("--mbpp_split", default="train")
    ap.add_argument("--apps_split", default="train")
    ap.add_argument("--limit_mbpp", type=int, default=0)
    ap.add_argument("--limit_apps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_mbpp(
        config_name=args.mbpp_config,
        split=args.mbpp_split,
        out_jsonl=str(out_dir / "mbpp_train.jsonl"),
        limit=args.limit_mbpp or None,
        seed=args.seed,
    )
    export_apps(
        split=args.apps_split,
        out_jsonl=str(out_dir / "apps_intro.jsonl"),
        limit=args.limit_apps or None,
        seed=args.seed,
    )
    print("Done.")

if __name__ == "__main__":
    main()
