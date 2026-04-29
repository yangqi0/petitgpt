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


def maybe_limit(
    rows: List[Dict[str, Any]], limit: Optional[int], seed: int
) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = idx[:limit]
    idx.sort()
    return [rows[i] for i in idx]


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if role in {"system", "user", "assistant"} and str(content).strip():
            out.append({"role": role, "content": str(content).strip()})
    return out


def last_user_assistant(messages: List[Dict[str, str]]) -> tuple[str, str]:
    prompt = ""
    response = ""
    for i, m in enumerate(messages):
        if m["role"] == "user":
            prompt = m["content"]
            response = ""
            for j in range(i + 1, len(messages)):
                if messages[j]["role"] == "assistant":
                    response = messages[j]["content"]
                    break
    if not response:
        for m in reversed(messages):
            if m["role"] == "assistant":
                response = m["content"]
                break
    return prompt.strip(), response.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    rows: List[Dict[str, Any]] = []

    for i, row in enumerate(ds):
        messages = normalize_messages(row.get("messages"))
        prompt, response = last_user_assistant(messages)
        if not prompt:
            continue
        source = row.get("source") or row.get("category") or "smol_smoltalk"
        rows.append(
            {
                "id": row.get("id", i),
                "prompt": prompt,
                "response": response,
                "messages": messages,
                "category": str(source),
                "source": str(source),
            }
        )

    rows = maybe_limit(rows, args.limit or None, args.seed)
    write_jsonl(args.out_jsonl, rows)
    print(f"[export_smol] wrote {len(rows)} -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
