from __future__ import annotations

import argparse
from collections import Counter

from code_utils import extract_first_code_block, normalize_text, read_jsonl, run_tests_with_timeout, verify_ast_structure, write_jsonl

def repairable(reasons):
    hard = {"syntax_error", "banned_call", "banned_toplevel_node", "wrong_function_name"}
    return not any(r in hard for r in reasons)

def answer_field(mode: str) -> str:
    return "answer_raw" if mode == "raw" else "answer_repaired"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "repair"], required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_pass_jsonl", required=True)
    ap.add_argument("--out_reject_jsonl", required=True)
    ap.add_argument("--out_repair_candidates_jsonl")
    ap.add_argument("--timeout", type=float, default=0.5)
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    passes, rejects, repairs = [], [], []
    rc = Counter()
    field = answer_field(args.mode)

    for row in rows:
        raw = normalize_text(row.get(field, "") or "")
        code = extract_first_code_block(raw)
        entry = row["entry_point"]
        reasons = verify_ast_structure(code, entry)
        tests_ok = False
        test_err = None
        if not reasons:
            tests_ok, test_err = run_tests_with_timeout(code, entry, row["tests"], timeout=args.timeout)
            if not tests_ok:
                reasons.append("tests_failed")
                if test_err:
                    reasons.append(f"test_detail:{test_err}")
        if not reasons:
            passes.append({
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": code},
                ],
                "meta": {
                    "bucket": "B_code",
                    "source": row["source"],
                    "source_key": row.get("source_key"),
                    "family": row.get("family"),
                    "entry_point": entry,
                    "generation_round": row["meta"].get("generation_round", 1),
                },
            })
        else:
            for r in reasons:
                rc[r] += 1
            reject_row = {
                "id": row["id"],
                "source": row["source"],
                "source_key": row.get("source_key"),
                "family": row.get("family"),
                "prompt": row["prompt"],
                "bad_answer": raw,
                "code_extracted": code,
                "entry_point": entry,
                "tests": row["tests"],
                "repair_reasons": reasons,
                "meta": row.get("meta", {}),
            }
            rejects.append(reject_row)
            if args.mode == "raw" and args.out_repair_candidates_jsonl and repairable(reasons):
                repairs.append(reject_row)

    write_jsonl(args.out_pass_jsonl, passes)
    write_jsonl(args.out_reject_jsonl, rejects)
    if args.out_repair_candidates_jsonl:
        write_jsonl(args.out_repair_candidates_jsonl, repairs)

    print(f"Pass: {len(passes)}")
    print(f"Reject: {len(rejects)}")
    if args.out_repair_candidates_jsonl:
        print(f"Repair candidates: {len(repairs)}")
    print("[top_reasons]")
    for k, v in rc.most_common(20):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
