from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List

from code_utils import infer_entry_point_from_code, infer_entry_point_from_tests, read_jsonl, stable_id, write_jsonl

def get_tests(row: Dict[str, Any]) -> List[str]:
    for key in ["test_list", "tests", "challenge_test_list"]:
        val = row.get(key)
        if isinstance(val, list) and val and all(isinstance(x, str) for x in val):
            return val
    return []

def get_prompt(row: Dict[str, Any]) -> str:
    for key in ["text", "prompt", "question", "instruction"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""

def get_code(row: Dict[str, Any]) -> str:
    for key in ["code", "solution", "reference_solution"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    rows_in = read_jsonl(args.in_jsonl)
    out = []
    reject = Counter()
    for row in rows_in:
        prompt = get_prompt(row)
        tests = get_tests(row)
        code = get_code(row)
        if not prompt or not tests:
            reject["missing_prompt_or_tests"] += 1
            continue
        entry = infer_entry_point_from_tests(tests) or infer_entry_point_from_code(code or "")
        if not entry:
            reject["missing_entry_point"] += 1
            continue
        if len(prompt.split()) > 180:
            reject["too_long_prompt"] += 1
            continue
        canonical = prompt.strip()
        if "return only python code" not in canonical.lower():
            canonical = canonical.rstrip() + "\n\nReturn only Python code."
        out.append({
            "id": stable_id("mbpp", str(row.get("task_id", row.get("id", len(out)))), prompt[:120]),
            "source": "mbpp",
            "source_key": str(row.get("task_id", row.get("id", len(out)))),
            "family": None,
            "raw_prompt": prompt,
            "canonical_prompt": canonical,
            "entry_point": entry,
            "tests": tests[:8],
            "meta": {"from_mbpp": True},
        })
        if len(out) >= args.limit:
            break

    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} MBPP rows to {args.out_jsonl}")
    print("[rejects]")
    for k, v in reject.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
