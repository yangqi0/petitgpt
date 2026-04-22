from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general_canonical_jsonl", required=True)
    ap.add_argument("--code_canonical_jsonl", required=True)
    ap.add_argument("--out_general_smoke_jsonl", required=True)
    ap.add_argument("--out_code_smoke_jsonl", required=True)
    ap.add_argument("--general_per_family", type=int, default=50)
    ap.add_argument("--code_per_family", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    general_rows = read_jsonl(args.general_canonical_jsonl)
    by_gfam = defaultdict(list)
    for row in general_rows:
        by_gfam[row.get("family", "unknown")].append(row)

    general_out = []
    for fam, group in by_gfam.items():
        rng.shuffle(group)
        general_out.extend(group[:args.general_per_family])

    code_rows = read_jsonl(args.code_canonical_jsonl)
    by_cfam = defaultdict(list)
    open_code = []
    for row in code_rows:
        fam = row.get("family")
        if fam:
            by_cfam[fam].append(row)
        else:
            open_code.append(row)

    code_out = []
    for fam, group in by_cfam.items():
        rng.shuffle(group)
        code_out.extend(group[:args.code_per_family])

    rng.shuffle(open_code)
    code_out.extend(open_code[: max(0, 240 - len(code_out))])

    write_jsonl(args.out_general_smoke_jsonl, general_out)
    write_jsonl(args.out_code_smoke_jsonl, code_out)

    print(f"general_smoke={len(general_out)} code_smoke={len(code_out)}")

if __name__ == "__main__":
    main()
