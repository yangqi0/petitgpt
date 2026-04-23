from __future__ import annotations

import argparse
import json
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

def extract_last_user(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "") or ""
    return ""

def extract_last_assistant(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Your existing smol/smoltalk-like JSONL export")
    ap.add_argument("--out_jsonl", required=True, help="Normalized output JSONL for general_extract_open_v1.py")
    args = ap.parse_args()

    src = read_jsonl(args.in_jsonl)
    out = []
    for i, row in enumerate(src):
        prompt = row.get("prompt") or row.get("instruction") or row.get("input") or extract_last_user(row.get("messages"))
        response = row.get("response") or row.get("output") or extract_last_assistant(row.get("messages"))
        out.append({
            "id": row.get("id", i),
            "prompt": prompt or "",
            "response": response or "",
            "category": row.get("category", ""),
        })

    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} rows -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
