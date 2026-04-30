from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any, Dict, List

from general_utils import (
    body_sentence_count,
    bullet_count,
    contains_placeholder,
    detect_bullets,
    has_greeting,
    has_meta_prefix,
    has_multiple_answer_pattern,
    has_non_bullet_content,
    has_numbered_list,
    has_prompt_echo,
    has_signoff,
    line_count,
    looks_like_email,
    looks_too_technical,
    normalize_text,
    read_jsonl,
    sentence_count,
    strip_bullet_marker,
    word_count,
    write_jsonl,
)

def contains_real_code_pollution(s: str) -> bool:
    """Do not flag ordinary English 'return' as code pollution."""
    patterns = [
        r"```",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bprint\s*\(",
    ]
    return any(re.search(p, s or "", flags=re.IGNORECASE) for p in patterns)

def global_hard_fail(answer: str, prompt: str) -> List[str]:
    reasons: List[str] = []
    if not answer.strip():
        reasons.append("empty")
    if word_count(answer) < 2:
        reasons.append("too_short")
    if contains_real_code_pollution(answer):
        reasons.append("code_pollution")
    if contains_placeholder(answer):
        reasons.append("placeholder")
    if has_meta_prefix(answer):
        reasons.append("meta_prefix")
    if has_multiple_answer_pattern(answer):
        reasons.append("multiple_answers")
    if has_prompt_echo(prompt, answer):
        reasons.append("prompt_echo")
    if "as an ai" in answer.lower():
        reasons.append("meta_ai")
    return reasons

def verify_email(answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    wc = word_count(answer)
    min_words = constraints.get("min_words", 25 if constraints.get("must_be_email") else 5)
    max_words = constraints.get("max_words", 125 if constraints.get("must_be_email") else 55)
    min_sentences = constraints.get("min_sentences", 1)
    max_sentences = constraints.get("max_sentences", 7)

    if wc < min_words or wc > max_words:
        reasons.append("email_or_message_length")
    if bullet_count(answer) > 0:
        reasons.append("email_or_message_has_bullets")

    if constraints.get("must_be_email", False):
        # Greeting/signoff are useful, but not all source prompts are formal emails.
        # Keep them as repairable formatting issues rather than over-filtering everything.
        if not has_greeting(answer):
            reasons.append("missing_greeting")
        if not has_signoff(answer):
            reasons.append("missing_signoff")
        bsc = body_sentence_count(answer)
        if bsc < min_sentences or bsc > max_sentences:
            reasons.append("email_body_sentence_count")
    elif constraints.get("must_be_short_message", False):
        sc = sentence_count(answer)
        if sc < min_sentences or sc > max_sentences:
            reasons.append("short_message_sentence_count")
        if line_count(answer) > 4:
            reasons.append("short_message_too_many_lines")
    return reasons

def verify_rewrite(answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    if line_count(answer) > 3:
        reasons.append("rewrite_multi_paragraph")
    if sentence_count(answer) > constraints.get("max_sentences", 2):
        reasons.append("rewrite_too_many_sentences")
    if word_count(answer) > constraints.get("max_words", 35):
        reasons.append("rewrite_too_long")
    if looks_like_email(answer):
        reasons.append("rewrite_became_email")
    return reasons

def verify_summary(answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    exact = constraints.get("exact_bullets")
    if exact is not None:
        lines = [x for x in answer.splitlines() if x.strip()]
        bullets = detect_bullets(lines)
        if len(bullets) != exact:
            reasons.append("wrong_bullet_count")
        if has_non_bullet_content(answer, bullets):
            reasons.append("extra_non_bullet_text")
        # Do not reject mixed bullet styles here; exact count and brevity are more important.
        for b in bullets:
            if word_count(strip_bullet_marker(b)) > 20:
                reasons.append("bullet_too_long")
                break
    else:
        if sentence_count(answer) != 1:
            reasons.append("summary_not_one_sentence")
        if word_count(answer) > constraints.get("max_words", 35):
            reasons.append("summary_sentence_too_long")
    return reasons

def verify_explain(answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    sc = sentence_count(answer)
    wc = word_count(answer)
    max_s = constraints.get("max_sentences", 3)
    max_w = constraints.get("max_words", 80)

    if sc < 1 or sc > max_s:
        reasons.append("explain_sentence_count")
    if wc > max_w:
        reasons.append("explain_too_long")
    if bullet_count(answer) > 0 or has_numbered_list(answer):
        reasons.append("explain_list_format")
    if looks_too_technical(answer):
        reasons.append("too_technical")
    return reasons

def heuristic_scores(answer: str, family: str, reasons: List[str]) -> Dict[str, int]:
    penalty = len(reasons)
    wc = word_count(answer)

    if family == "rewrite_style":
        brevity = 5 if wc <= 30 else 4 if wc <= 40 else 2
    elif family == "summary_bullets":
        brevity = 5 if wc <= 70 else 4 if wc <= 90 else 2
    elif family == "explain_compare":
        brevity = 5 if wc <= 70 else 4 if wc <= 90 else 2
    elif family == "email_message":
        brevity = 5 if wc <= 100 else 4 if wc <= 125 else 2
    else:
        brevity = 4 if wc <= 90 else 2

    instr = max(1, 5 - penalty)
    cleanliness = max(1, 5 - penalty)
    naturalness = 1 if contains_placeholder(answer) or contains_real_code_pollution(answer) else 4

    return {
        "instruction_following": max(1, min(5, instr)),
        "cleanliness": max(1, min(5, cleanliness)),
        "brevity": max(1, min(5, brevity)),
        "naturalness": max(1, min(5, naturalness)),
    }

def repairable(reasons: List[str]) -> bool:
    hard = {"code_pollution", "placeholder", "prompt_echo", "multiple_answers", "meta_ai"}
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
    ap.add_argument("--min_scores", action="store_true", help="Also require scores >=4. Usually leave off for first pass.")
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    passes = []
    rejects = []
    repairs = []
    reasons_counter = Counter()
    field = answer_field(args.mode)

    for row in rows:
        answer = normalize_text(row.get(field, "") or "")
        prompt = row["prompt"]
        cons = row["constraints"]
        family = row["family"]

        reasons = global_hard_fail(answer, prompt)
        if not reasons:
            if family == "email_message":
                reasons.extend(verify_email(answer, cons))
            elif family == "rewrite_style":
                reasons.extend(verify_rewrite(answer, cons))
            elif family == "summary_bullets":
                reasons.extend(verify_summary(answer, cons))
            else:
                reasons.extend(verify_explain(answer, cons))

        scores = heuristic_scores(answer, family, reasons)
        score_ok = (
            scores["instruction_following"] >= 4 and
            scores["cleanliness"] >= 4 and
            scores["brevity"] >= 4
        )

        if not reasons and (score_ok or not args.min_scores):
            passes.append({
                "id": row["id"],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    "bucket": "A_general",
                    "family": row["family"],
                    "subfamily": row["subfamily"],
                    "source": row.get("meta", {}).get("source"),
                    "source_key": row.get("meta", {}).get("source_key"),
                    "from_template": row.get("meta", {}).get("from_template", False),
                    "generation_round": row.get("meta", {}).get("generation_round", 1),
                    "scores": scores,
                    "verifier_version": "v1_1",
                },
            })
        else:
            for r in reasons:
                reasons_counter[r] += 1
            reject_row = {
                "id": row["id"],
                "family": row["family"],
                "subfamily": row["subfamily"],
                "prompt": prompt,
                "bad_answer": answer,
                "reasons": reasons,
                "scores": scores,
                "constraints": cons,
                "meta": row.get("meta", {}),
            }
            rejects.append(reject_row)
            if args.mode == "raw" and args.out_repair_candidates_jsonl and repairable(reasons):
                repairs.append({
                    "id": row["id"],
                    "family": row["family"],
                    "subfamily": row["subfamily"],
                    "prompt": prompt,
                    "bad_answer": answer,
                    "repair_reasons": reasons,
                    "constraints": cons,
                    "meta": row.get("meta", {}),
                })

    write_jsonl(args.out_pass_jsonl, passes)
    write_jsonl(args.out_reject_jsonl, rejects)
    if args.out_repair_candidates_jsonl:
        write_jsonl(args.out_repair_candidates_jsonl, repairs)

    print(f"Pass: {len(passes)}")
    print(f"Reject: {len(rejects)}")
    if args.out_repair_candidates_jsonl:
        print(f"Repair candidates: {len(repairs)}")
    print("[top_reasons]")
    for k, v in reasons_counter.most_common(30):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
