from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from general_utils import cleanup_prompt_text, extract_last_user_from_messages, normalize_space, print_counter, read_jsonl, stable_id, write_jsonl

BANNED_KEYWORDS = [
    "python", "javascript", "function", "code", "algorithm", "bug", "debug", "sql",
    "story", "poem", "fiction", "roleplay", "tweet", "social media", "medical advice",
    "legal advice", "investment advice", "leetcode", "programming",
]

def bad_prompt(text: str) -> Optional[str]:
    low = text.lower()
    if any(k in low for k in BANNED_KEYWORDS):
        return "banned_keyword"
    wc = len(low.split())
    if wc < 6:
        return "too_short"
    if wc > 80:
        return "too_long"
    return None

def emit_record(source: str, source_key: str, raw_prompt: str, raw_context: str = "", raw_answer_ref: str = "", raw_category: str = "", split: str = "train") -> Dict[str, Any]:
    return {
        "id": stable_id(source[:3], source_key, raw_prompt[:80]),
        "source": source,
        "source_split": split,
        "source_key": source_key,
        "raw_prompt": cleanup_prompt_text(raw_prompt),
        "raw_context": cleanup_prompt_text(raw_context),
        "raw_answer_ref": raw_answer_ref,
        "raw_category": raw_category,
        "family_hint": None,
        "meta": {"from_template": False},
    }

def iter_rows_from_jsonl(path: Optional[str]) -> Iterator[Dict[str, Any]]:
    if not path:
        return iter(())
    return iter(read_jsonl(path))

def extract_no_robots(rows: Iterable[Dict[str, Any]], limit: int, split: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        category = str(row.get("category", ""))
        if category.lower() == "coding":
            continue
        prompt = cleanup_prompt_text(row.get("prompt", "") or "")
        if not prompt:
            prompt = extract_last_user_from_messages(row.get("messages", []))
        if not prompt:
            continue
        reason = bad_prompt(prompt)
        if reason:
            continue
        out.append(emit_record("no_robots", str(row.get("prompt_id", row.get("id", len(out)))), prompt, "", "", category, split))
        if len(out) >= limit:
            break
    return out

def extract_smol(rows: Iterable[Dict[str, Any]], limit: int, split: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        prompt = row.get("prompt") or row.get("instruction") or row.get("input") or extract_last_user_from_messages(row.get("messages", []))
        prompt = cleanup_prompt_text(prompt or "")
        if not prompt:
            continue
        reason = bad_prompt(prompt)
        if reason:
            continue
        out.append(emit_record("smol_smoltalk", str(row.get("id", len(out))), prompt, "", row.get("response", "") or "", row.get("category", ""), split))
        if len(out) >= limit:
            break
    return out

def extract_alpaca(rows: Iterable[Dict[str, Any]], limit: int, split: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        ins = cleanup_prompt_text(row.get("instruction", "") or "")
        inp = cleanup_prompt_text(row.get("input", "") or "")
        if not ins:
            continue
        prompt = ins if not inp else f"{ins}\n\n{inp}"
        if len(inp.split()) > 60:
            continue
        reason = bad_prompt(prompt)
        if reason:
            continue
        out.append(emit_record("alpaca_cleaned", str(row.get("id", len(out))), prompt, inp, row.get("output", "") or "", row.get("category", ""), split))
        if len(out) >= limit:
            break
    return out

def extract_dolly(rows: Iterable[Dict[str, Any]], limit: int, split: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        ins = cleanup_prompt_text(row.get("instruction", "") or "")
        ctx = cleanup_prompt_text(row.get("context", "") or "")
        if not ins:
            continue
        if len(ctx.split()) > 60:
            continue
        prompt = ins if not ctx else f"{ins}\n\n{ctx}"
        reason = bad_prompt(prompt)
        if reason:
            continue
        out.append(emit_record("dolly_style", str(row.get("id", len(out))), prompt, ctx, row.get("response", "") or "", row.get("category", ""), split))
        if len(out) >= limit:
            break
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_robots_jsonl")
    ap.add_argument("--smol_jsonl")
    ap.add_argument("--alpaca_jsonl")
    ap.add_argument("--dolly_jsonl")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit_no_robots", type=int, default=3000)
    ap.add_argument("--limit_smol", type=int, default=1800)
    ap.add_argument("--limit_alpaca", type=int, default=1400)
    ap.add_argument("--limit_dolly", type=int, default=900)
    args = ap.parse_args()

    all_rows = []
    if args.no_robots_jsonl:
        all_rows.extend(extract_no_robots(iter_rows_from_jsonl(args.no_robots_jsonl), args.limit_no_robots, "train_sft"))
    if args.smol_jsonl:
        all_rows.extend(extract_smol(iter_rows_from_jsonl(args.smol_jsonl), args.limit_smol, "train"))
    if args.alpaca_jsonl:
        all_rows.extend(extract_alpaca(iter_rows_from_jsonl(args.alpaca_jsonl), args.limit_alpaca, "train"))
    if args.dolly_jsonl:
        all_rows.extend(extract_dolly(iter_rows_from_jsonl(args.dolly_jsonl), args.limit_dolly, "train"))

    write_jsonl(args.out_jsonl, all_rows)
    c = Counter([r["source"] for r in all_rows])
    print(f"Wrote {len(all_rows)} raw open prompts to {args.out_jsonl}")
    print_counter("source_counts", c)

if __name__ == "__main__":
    main()
