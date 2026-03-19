#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SYSTEM = "You are a helpful assistant."


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not path:
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```python\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None


def count_code_blocks(text: str) -> int:
    return len(re.findall(r"```", text)) // 2


def check_code_format(task_type: str, text: str) -> Tuple[bool, str]:
    if task_type in ("code_write", "code_fix", "code_script"):
        if count_code_blocks(text) != 1:
            return False, "need_exactly_one_code_block"
        code = extract_code_block(text)
        if not code or not code.strip():
            return False, "empty_code"
        return True, "ok"

    if task_type == "code_explain_then_code":
        if "Explanation:" not in text:
            return False, "missing_explanation"
        if count_code_blocks(text) != 1:
            return False, "need_exactly_one_code_block"
        code = extract_code_block(text)
        if not code or not code.strip():
            return False, "empty_code"
        return True, "ok"

    return False, "unknown_code_task"


def check_ast_parse(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "ok"
    except Exception as e:
        return False, f"ast_error:{type(e).__name__}"


def extract_final_answer(text: str) -> Optional[str]:
    m = re.search(r"Final answer:\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def check_math_format(text: str) -> Tuple[bool, str]:
    if "Brief steps:" not in text:
        return False, "missing_brief_steps"
    if "Final answer:" not in text:
        return False, "missing_final_answer"
    ans = extract_final_answer(text)
    if not ans:
        return False, "empty_final_answer"
    return True, "ok"


def try_float(s: str) -> Optional[float]:
    s = s.strip().replace("%", "")
    try:
        return float(s)
    except Exception:
        return None


def normalize_answer(s: str) -> str:
    return s.strip().lower().replace(" ", "")


# -------------------------
# Minimal math answer key
# Extend this as you add prompts.
# -------------------------
MATH_ANSWER_KEY: Dict[str, List[str]] = {
    "math_word_0001": ["18", "18.0"],
    "math_word_0002": ["60", "60.0"],
    "math_word_0003": ["180", "180.0"],
    "math_word_0004": ["11", "11.0"],
    "math_word_0005": ["4", "4.0"],
    "math_word_0006": ["34", "34.0"],
    "math_word_0007": ["126", "126.0"],
    "math_word_0008": ["15", "15.0"],
    "math_algebra_0001": ["6", "6.0"],
    "math_algebra_0002": ["9", "9.0"],
    "math_algebra_0003": ["9", "9.0"],
    "math_algebra_0004": ["24", "24.0"],
    "math_algebra_0005": ["4", "4.0"],
    "math_algebra_0006": ["13", "13.0"],
    "math_ratio_0001": ["25", "25.0", "25%"],
    "math_ratio_0002": ["55", "55.0"],
    "math_ratio_0003": ["3/4", "75", "75.0", "75%"],
    "math_ratio_0004": ["21", "21.0"],
    "math_ratio_0005": ["15", "15.0"],
    "math_stats_0001": ["5", "5.0"],
    "math_stats_0002": ["4.5"],
    "math_stats_0003": ["4", "4.0"],
    "math_stats_0004": ["5", "5.0"],
    "math_stats_0005": ["14", "14.0", "14 and 8", "mean=14,variance=8", "8 and 14"],
}


def math_answer_is_correct(prompt_id: str, answer: str) -> bool:
    allowed = MATH_ANSWER_KEY.get(prompt_id, [])
    if not allowed:
        return False
    a = normalize_answer(answer)
    allowed_norm = {normalize_answer(x) for x in allowed}
    return a in allowed_norm


def check_general(task_type: str, text: str) -> Tuple[bool, str]:
    stripped = text.strip()

    if "```" in stripped:
        return False, "no_code_blocks_allowed"

    if task_type == "general_summary":
        # loose but useful rule
        bullet_lines = [
            ln for ln in stripped.splitlines() if ln.strip().startswith("-")
        ]
        if "3 bullet" in stripped.lower():
            pass
        # we do not know prompt text here, so just accept 3 bullets or 1 paragraph
        if bullet_lines and len(bullet_lines) != 3:
            return False, "summary_wrong_bullet_count"
        return True, "ok"

    if task_type == "general_rewrite":
        bad_prefixes = [
            "here is",
            "rewritten version",
            "more polite version",
            "professional version",
        ]
        s = stripped.lower()
        if any(s.startswith(x) for x in bad_prefixes):
            return False, "rewrite_has_meta_prefix"
        return True, "ok"

    if task_type == "general_explain":
        # crude sentence count
        parts = re.split(r"[.!?]+", stripped)
        sent_count = len([x for x in parts if x.strip()])
        if sent_count > 5:
            return False, "too_many_sentences"
        return True, "ok"

    if task_type == "general_checklist":
        item_lines = [
            ln
            for ln in stripped.splitlines()
            if re.match(r"^\s*(\d+\.|-)\s+", ln.strip())
        ]
        if len(item_lines) < 3:
            return False, "too_few_items"
        return True, "ok"

    if task_type == "general_email":
        lower = stripped.lower()
        if "subject:" not in lower:
            return False, "missing_subject"
        if (
            ("best," not in lower)
            and ("sincerely," not in lower)
            and ("regards," not in lower)
        ):
            return False, "missing_closing"
        return True, "ok"

    return True, "ok"


def make_canonical(
    prompt: str, response: str, bucket: str, task_type: str, teacher: str, verifier: str
) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "meta": {
            "bucket": bucket,
            "task_type": task_type,
            "source": "stage2_distill",
            "teacher": teacher,
            "verifier": verifier,
        },
    }


def verify_code_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_type = row["task_type"]
    resp = row["response"]

    ok, why = check_code_format(task_type, resp)
    if not ok:
        return None

    code = extract_code_block(resp)
    if code is None:
        return None

    ok, why = check_ast_parse(code)
    if not ok:
        return None

    return make_canonical(
        prompt=row["prompt"],
        response=resp,
        bucket="B_code",
        task_type=task_type,
        teacher=row.get("teacher", "unknown"),
        verifier="format+ast",
    )


def verify_math_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    resp = row["response"]
    ok, why = check_math_format(resp)
    if not ok:
        return None

    ans = extract_final_answer(resp)
    if not ans:
        return None

    if not math_answer_is_correct(row["prompt_id"], ans):
        return None

    return make_canonical(
        prompt=row["prompt"],
        response=resp,
        bucket="C_basic_math",
        task_type=row["task_type"],
        teacher=row.get("teacher", "unknown"),
        verifier="format+final_answer",
    )


def verify_general_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ok, why = check_general(row["task_type"], row["response"])
    if not ok:
        return None

    return make_canonical(
        prompt=row["prompt"],
        response=row["response"].strip(),
        bucket="A_general",
        task_type=row["task_type"],
        teacher=row.get("teacher", "unknown"),
        verifier="rule_check",
    )


def pick_best_by_shortness(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(rows, key=lambda r: len(r["messages"][-1]["content"]))[0]


def split_train_val(
    rows: List[Dict[str, Any]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio)) if rows else 0
    val = rows[:n_val]
    train = rows[n_val:]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_raw", default="")
    ap.add_argument("--general_raw", default="")
    ap.add_argument("--math_raw", default="")
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    accepted: List[Dict[str, Any]] = []

    # code
    code_rows = read_jsonl(args.code_raw)
    by_prompt: Dict[str, List[Dict[str, Any]]] = {}
    for row in code_rows:
        can = verify_code_row(row)
        if can is not None:
            by_prompt.setdefault(row["prompt_id"], []).append(can)
    kept_code = [pick_best_by_shortness(v) for v in by_prompt.values()]
    accepted.extend(kept_code)
    print(f"[code] raw={len(code_rows)} kept_prompts={len(kept_code)}")

    # general
    general_rows = read_jsonl(args.general_raw)
    by_prompt = {}
    for row in general_rows:
        can = verify_general_row(row)
        if can is not None:
            by_prompt.setdefault(row["prompt_id"], []).append(can)
    kept_general = [pick_best_by_shortness(v) for v in by_prompt.values()]
    accepted.extend(kept_general)
    print(f"[general] raw={len(general_rows)} kept_prompts={len(kept_general)}")

    # math
    math_rows = read_jsonl(args.math_raw)
    by_prompt = {}
    for row in math_rows:
        can = verify_math_row(row)
        if can is not None:
            by_prompt.setdefault(row["prompt_id"], []).append(can)
    kept_math = [pick_best_by_shortness(v) for v in by_prompt.values()]
    accepted.extend(kept_math)
    print(f"[math] raw={len(math_rows)} kept_prompts={len(kept_math)}")

    train, val = split_train_val(accepted, args.val_ratio, args.seed)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_val, val)

    print(f"[all] train={len(train)} val={len(val)}")
    print(f"Wrote:\n  {args.out_train}\n  {args.out_val}")


if __name__ == "__main__":
    main()
