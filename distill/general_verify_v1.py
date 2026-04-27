from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List

from general_utils import (
    body_sentence_count,
    bullet_count,
    contains_code_markers,
    contains_placeholder,
    detect_bullets,
    extract_quoted_text,
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
    mixed_bullet_styles,
    normalize_text,
    read_jsonl,
    sentence_count,
    strip_bullet_marker,
    word_count,
    write_jsonl,
)


def global_hard_fail(answer: str, prompt: str) -> List[str]:
    reasons: List[str] = []
    if not answer.strip():
        reasons.append("empty")
    if word_count(answer) < 3:
        reasons.append("too_short")
    if contains_code_markers(answer):
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


# def verify_email(answer: str, constraints: Dict[str, Any]) -> List[str]:
#     reasons = []
#     wc = word_count(answer)
#     if wc < 40 or wc > 110:
#         reasons.append("email_length")
#     if bullet_count(answer) > 0:
#         reasons.append("email_has_bullets")
#     if not has_greeting(answer):
#         reasons.append("missing_greeting")
#     if not has_signoff(answer):
#         reasons.append("missing_signoff")
#     bsc = body_sentence_count(answer)
#     if bsc < 2 or bsc > 5:
#         reasons.append("email_body_sentence_count")
#     return reasons


def verify_email(answer: str, constraints: Dict[str, Any]) -> List[str]:
    """Verify email/message outputs using the per-sample constraints.

    Cases:
    - must_be_email=True: require greeting + sign-off + short email structure.
    - must_be_short_message=True: allow a short workplace message without greeting/sign-off.
    """
    reasons = []

    wc = word_count(answer)
    min_words = constraints.get(
        "min_words", 35 if constraints.get("must_be_email") else 5
    )
    max_words = constraints.get("max_words", 110)
    min_sentences = constraints.get("min_sentences", 1)
    max_sentences = constraints.get("max_sentences", 6)

    if wc < min_words or wc > max_words:
        reasons.append("email_or_message_length")

    if bullet_count(answer) > 0:
        reasons.append("email_or_message_has_bullets")

    if constraints.get("must_be_email", False):
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
        if line_count(answer) > 3:
            reasons.append("short_message_too_many_lines")

    return reasons


def verify_rewrite(prompt: str, answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    source = extract_quoted_text(prompt) or ""
    if "\n\n" in answer.strip():
        reasons.append("rewrite_multi_paragraph")
    if sentence_count(answer) > constraints["max_sentences"]:
        reasons.append("rewrite_too_many_sentences")
    if word_count(answer) > constraints["max_words"]:
        reasons.append("rewrite_too_long")
    if source and word_count(answer) > int(word_count(source) * 1.25) + 3:
        reasons.append("rewrite_expanded_too_much")
    if looks_like_email(answer):
        reasons.append("rewrite_became_email")
    return reasons


def verify_summary(answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    if constraints.get("exact_bullets") is not None:
        lines = [x for x in answer.splitlines() if x.strip()]
        bullets = detect_bullets(lines)
        if len(bullets) != constraints["exact_bullets"]:
            reasons.append("wrong_bullet_count")
        if has_non_bullet_content(answer, bullets):
            reasons.append("extra_non_bullet_text")
        if mixed_bullet_styles(answer):
            reasons.append("mixed_bullet_styles")
        for b in bullets:
            if word_count(strip_bullet_marker(b)) > 16:
                reasons.append("bullet_too_long")
                break
    else:
        if sentence_count(answer) != 1:
            reasons.append("summary_not_one_sentence")
    return reasons


def verify_explain(answer: str, constraints: Dict[str, Any]) -> List[str]:
    reasons = []
    sc = sentence_count(answer)
    wc = word_count(answer)
    if sc < 2 or sc > constraints["max_sentences"]:
        reasons.append("explain_sentence_count")
    if wc > constraints["max_words"]:
        reasons.append("explain_too_long")
    if bullet_count(answer) > 0 or has_numbered_list(answer):
        reasons.append("explain_list_format")
    if looks_too_technical(answer):
        reasons.append("too_technical")
    return reasons


# def heuristic_scores(answer: str, family: str, reasons: List[str]) -> Dict[str, int]:
#     base = 5
#     penalty = len(reasons)
#     cleanliness = max(1, base - penalty)
#     brevity = (
#         5
#         if word_count(answer) <= 40
#         else 4
#         if word_count(answer) <= 70
#         else 3
#         if word_count(answer) <= 110
#         else 2
#     )
#     instr = max(1, base - penalty)
#     natural = 4
#     if "thank you" in answer.lower() or "best regards" in answer.lower():
#         natural = 5
#     if contains_placeholder(answer) or contains_code_markers(answer):
#         natural = 1
#     return {
#         "instruction_following": instr,
#         "cleanliness": cleanliness,
#         "brevity": brevity,
#         "naturalness": natural,
#     }


def heuristic_scores(answer: str, family: str, reasons: List[str]) -> Dict[str, int]:
    """Lightweight family-aware scores used only for ranking/filtering.

    These scores are intentionally conservative and should not replace the
    family-specific hard checks above.
    """
    penalty = len(reasons)
    wc = word_count(answer)

    # Family-aware brevity bands.
    if family == "rewrite_style":
        brevity = 5 if wc <= 25 else 4 if wc <= 35 else 2
    elif family == "summary_bullets":
        brevity = 5 if wc <= 55 else 4 if wc <= 70 else 2
    elif family == "explain_compare":
        brevity = 5 if wc <= 55 else 4 if wc <= 75 else 2
    elif family == "email_message":
        brevity = 5 if wc <= 90 else 4 if wc <= 115 else 2
    else:
        brevity = 4 if wc <= 80 else 2

    cleanliness = max(1, 5 - penalty)

    # Family-aware instruction-following proxy.
    instr = max(1, 5 - penalty)

    if family == "summary_bullets" and bullet_count(answer) > 0:
        instr = min(5, instr + 1)

    if (
        family == "rewrite_style"
        and line_count(answer) == 1
        and not looks_like_email(answer)
    ):
        instr = min(5, instr + 1)

    if family == "email_message" and (
        looks_like_email(answer) or line_count(answer) <= 3
    ):
        instr = min(5, instr + 1)

    if (
        family == "explain_compare"
        and bullet_count(answer) == 0
        and not has_numbered_list(answer)
    ):
        instr = min(5, instr + 1)

    # Family-aware naturalness proxy.
    natural = 4
    low = answer.lower()

    if family == "email_message" and any(
        x in low for x in ["thank you", "best regards", "kind regards", "best,"]
    ):
        natural = 5
    elif family == "rewrite_style" and not has_meta_prefix(answer):
        natural = 4
    elif family == "explain_compare" and not looks_too_technical(answer):
        natural = 4
    elif family == "summary_bullets":
        natural = 4

    if contains_placeholder(answer) or contains_code_markers(answer):
        natural = 1

    return {
        "instruction_following": max(1, min(5, instr)),
        "cleanliness": max(1, min(5, cleanliness)),
        "brevity": max(1, min(5, brevity)),
        "naturalness": max(1, min(5, natural)),
    }


def repairable(reasons: List[str]) -> bool:
    hard = {
        "code_pollution",
        "placeholder",
        "prompt_echo",
        "multiple_answers",
        "meta_ai",
    }
    if any(r in hard for r in reasons):
        return False
    if any(r.startswith("lost_") for r in reasons):
        return False
    return True


def answer_field(mode: str) -> str:
    return "answer_raw" if mode == "raw" else "answer_repaired"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "repair"], required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_pass_jsonl", required=True)
    ap.add_argument("--out_reject_jsonl", required=True)
    ap.add_argument("--out_repair_candidates_jsonl")
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
        reasons = global_hard_fail(answer, prompt)
        fam = row["family"]
        if not reasons:
            if fam == "email_message":
                reasons.extend(verify_email(answer, cons))
            elif fam == "rewrite_style":
                reasons.extend(verify_rewrite(prompt, answer, cons))
            elif fam == "summary_bullets":
                reasons.extend(verify_summary(answer, cons))
            else:
                reasons.extend(verify_explain(answer, cons))

        scores = heuristic_scores(answer, fam, reasons)
        if (
            not reasons
            and scores["instruction_following"] >= 4
            and scores["cleanliness"] >= 4
            and scores["brevity"] >= 4
        ):
            passes.append(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    "meta": {
                        "bucket": "A_general",
                        "family": row["family"],
                        "subfamily": row["subfamily"],
                        "source": row["meta"].get("source"),
                        "source_key": row["meta"].get("source_key"),
                        "from_template": row["meta"].get("from_template", False),
                        "generation_round": row["meta"].get("generation_round", 1),
                        "scores": scores,
                    },
                }
            )
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
            if (
                args.mode == "raw"
                and args.out_repair_candidates_jsonl
                and repairable(reasons)
            ):
                repairs.append(
                    {
                        "id": row["id"],
                        "family": row["family"],
                        "subfamily": row["subfamily"],
                        "prompt": prompt,
                        "bad_answer": answer,
                        "repair_reasons": reasons,
                        "constraints": cons,
                        "meta": row.get("meta", {}),
                    }
                )

    write_jsonl(args.out_pass_jsonl, passes)
    write_jsonl(args.out_reject_jsonl, rejects)
    if args.out_repair_candidates_jsonl:
        write_jsonl(args.out_repair_candidates_jsonl, repairs)

    print(f"Pass: {len(passes)}")
    print(f"Reject: {len(rejects)}")
    if args.out_repair_candidates_jsonl:
        print(f"Repair candidates: {len(repairs)}")
    print("[top_reasons]")
    for k, v in reasons_counter.most_common(20):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
