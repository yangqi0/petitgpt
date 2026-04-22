from __future__ import annotations

import argparse
from collections import Counter

from code_utils import extract_assert_tests_from_apps_io, json_load_maybe, read_jsonl, stable_id, write_jsonl

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--allow_difficulty", nargs="+", default=["introductory"])
    args = ap.parse_args()

    rows_in = read_jsonl(args.in_jsonl)
    out = []
    reject = Counter()
    allowed = {x.lower() for x in args.allow_difficulty}

    for row in rows_in:
        difficulty = str(row.get("difficulty", "")).lower()
        if allowed and difficulty and difficulty not in allowed:
            reject["difficulty_filtered"] += 1
            continue
        question = str(row.get("question", "") or row.get("prompt", "") or "").strip()
        if not question:
            reject["missing_question"] += 1
            continue
        if len(question.split()) > 220:
            reject["too_long_question"] += 1
            continue
        io_obj = json_load_maybe(row.get("input_output"))
        if not isinstance(io_obj, dict):
            reject["missing_input_output"] += 1
            continue
        fn_name = io_obj.get("fn_name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            reject["missing_fn_name"] += 1
            continue
        tests = extract_assert_tests_from_apps_io(io_obj, fn_name.strip(), max_tests=8)
        if len(tests) < 2:
            reject["not_enough_tests"] += 1
            continue
        starter = str(row.get("starter_code", "") or "").strip()
        if starter and "stdin" in starter.lower():
            reject["stdin_style"] += 1
            continue
        canonical = question.rstrip()
        if "return only python code" not in canonical.lower():
            canonical += "\n\nReturn only Python code."
        out.append({
            "id": stable_id("apps", str(row.get("problem_id", row.get("id", len(out)))), question[:120]),
            "source": "apps_intro",
            "source_key": str(row.get("problem_id", row.get("id", len(out)))),
            "family": None,
            "raw_prompt": question,
            "canonical_prompt": canonical,
            "entry_point": fn_name.strip(),
            "tests": tests,
            "meta": {"difficulty": difficulty, "from_apps": True},
        })
        if len(out) >= args.limit:
            break

    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} APPS rows to {args.out_jsonl}")
    print("[rejects]")
    for k, v in reject.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
