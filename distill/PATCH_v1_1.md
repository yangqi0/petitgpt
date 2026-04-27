# Patch v1.1 for general distill scripts

This patch addresses two valid issues:

1. `verify_email(answer, constraints)` should actually use `constraints`.
2. `heuristic_scores(answer, family, reasons)` should actually be family-aware.

It also fixes a related issue:

3. `polite_short_message` should not be forced through full email checks such as greeting/sign-off.

Copy the functions below into the corresponding files.

---

## 1. Replace `default_constraints_for_subfamily` in `distill/general_utils.py`

```python
def default_constraints_for_subfamily(subfamily: str) -> Dict[str, Any]:
    if subfamily == "polite_short_message":
        return {
            "min_words": 8,
            "max_words": 45,
            "min_sentences": 1,
            "max_sentences": 3,
            "exact_bullets": None,
            "must_be_email": False,
            "must_be_short_message": True,
            "single_answer_only": True,
            "no_code": True,
            "no_placeholders": True,
            "no_explanation": True,
            "must_keep_meaning": False,
            "must_be_everyday_language": False,
        }

    if subfamily.endswith("email") or subfamily == "request_or_confirm_email":
        return {
            "min_words": 35,
            "max_words": 110,
            "min_sentences": 2,
            "max_sentences": 6,
            "exact_bullets": None,
            "must_be_email": True,
            "must_be_short_message": False,
            "single_answer_only": True,
            "no_code": True,
            "no_placeholders": True,
            "no_explanation": True,
            "must_keep_meaning": False,
            "must_be_everyday_language": False,
        }

    if subfamily.startswith("rewrite_"):
        return {
            "max_words": 30,
            "max_sentences": 2,
            "exact_bullets": None,
            "must_be_email": False,
            "must_be_short_message": False,
            "single_answer_only": True,
            "no_code": True,
            "no_placeholders": True,
            "no_explanation": True,
            "must_keep_meaning": True,
            "must_be_everyday_language": False,
        }

    if subfamily in {"summary_3_bullets", "extract_key_points"}:
        return {
            "max_words": 60,
            "max_sentences": None,
            "exact_bullets": 3,
            "must_be_email": False,
            "must_be_short_message": False,
            "single_answer_only": True,
            "no_code": True,
            "no_placeholders": True,
            "no_explanation": True,
            "must_keep_meaning": True,
            "must_be_everyday_language": False,
        }

    if subfamily == "summary_2_takeaways":
        return {
            "max_words": 50,
            "max_sentences": None,
            "exact_bullets": 2,
            "must_be_email": False,
            "must_be_short_message": False,
            "single_answer_only": True,
            "no_code": True,
            "no_placeholders": True,
            "no_explanation": True,
            "must_keep_meaning": True,
            "must_be_everyday_language": False,
        }

    if subfamily == "summary_1_sentence":
        return {
            "max_words": 30,
            "max_sentences": 1,
            "exact_bullets": None,
            "must_be_email": False,
            "must_be_short_message": False,
            "single_answer_only": True,
            "no_code": True,
            "no_placeholders": True,
            "no_explanation": True,
            "must_keep_meaning": True,
            "must_be_everyday_language": False,
        }

    return {
        "max_words": 70,
        "max_sentences": 3,
        "exact_bullets": None,
        "must_be_email": False,
        "must_be_short_message": False,
        "single_answer_only": True,
        "no_code": True,
        "no_placeholders": True,
        "no_explanation": True,
        "must_keep_meaning": True,
        "must_be_everyday_language": True,
    }
```

---

## 2. Replace `verify_email` in `distill/general_verify_v1.py`

Make sure `sentence_count` and `line_count` are imported from `general_utils`.

```python
def verify_email(answer: str, constraints: Dict[str, Any]) -> List[str]:
    """Verify email/message outputs using the per-sample constraints.

    Cases:
    - must_be_email=True: require greeting + sign-off + short email structure.
    - must_be_short_message=True: allow a short workplace message without greeting/sign-off.
    """
    reasons = []

    wc = word_count(answer)
    min_words = constraints.get("min_words", 35 if constraints.get("must_be_email") else 5)
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
```

---

## 3. Replace `heuristic_scores` in `distill/general_verify_v1.py`

```python
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

    if family == "rewrite_style" and line_count(answer) == 1 and not looks_like_email(answer):
        instr = min(5, instr + 1)

    if family == "email_message" and (looks_like_email(answer) or line_count(answer) <= 3):
        instr = min(5, instr + 1)

    if family == "explain_compare" and bullet_count(answer) == 0 and not has_numbered_list(answer):
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
```
