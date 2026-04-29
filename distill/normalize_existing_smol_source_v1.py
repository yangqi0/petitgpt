from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    """Normalize OpenAI/ChatML-style messages."""
    if not isinstance(messages, list):
        return []
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value") or m.get("text") or ""
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role in {"system", "user", "assistant"} and str(content).strip():
            out.append({"role": role, "content": str(content).strip()})
    return out


def last_user_assistant(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """Return the last user message and the following/last assistant response."""
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


def normalize_row(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Convert many smol/smoltalk-like schemas into the local schema expected by general_extract_open_v1.py."""
    messages = normalize_messages(row.get("messages"))
    if not messages:
        messages = normalize_messages(row.get("conversations"))

    prompt, response = "", ""
    if messages:
        prompt, response = last_user_assistant(messages)

    if not prompt:
        prompt = str(
            row.get("prompt")
            or row.get("instruction")
            or row.get("question")
            or row.get("input")
            or ""
        ).strip()

    if not response:
        response = str(
            row.get("response")
            or row.get("output")
            or row.get("answer")
            or row.get("completion")
            or ""
        ).strip()

    if row.get("instruction") and row.get("input"):
        ins = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        if ins and inp and prompt == ins:
            prompt = f"{ins}\n\n{inp}"

    if not prompt:
        return None

    if not messages:
        messages = [{"role": "user", "content": prompt}]
        if response:
            messages.append({"role": "assistant", "content": response})

    source = (
        row.get("source")
        or row.get("dataset")
        or row.get("category")
        or row.get("task")
        or "smol_smoltalk"
    )

    return {
        "id": row.get("id", idx),
        "prompt": prompt,
        "response": response,
        "messages": messages,
        "category": str(source),
        "source": str(source),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    raw_rows = read_jsonl(args.in_jsonl)
    rows = []
    skipped = 0
    for i, row in enumerate(raw_rows):
        out = normalize_row(row, i)
        if out is None:
            skipped += 1
        else:
            rows.append(out)

    rows = maybe_limit(rows, args.limit or None, args.seed)
    write_jsonl(args.out_jsonl, rows)
    print(
        f"[normalize_smol] input={len(raw_rows)} output={len(rows)} skipped={skipped} -> {args.out_jsonl}"
    )


if __name__ == "__main__":
    main()
