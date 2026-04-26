from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from general_utils import (
    cleanup_prompt_text,
    default_constraints_for_subfamily,
    extract_quoted_text,
    read_jsonl,
    write_jsonl,
)


def classify_family(prompt: str) -> Optional[str]:
    low = prompt.lower()
    if any(
        x in low
        for x in [
            "email",
            "follow up",
            "follow-up",
            "reschedule",
            "thank you",
            "confirm receipt",
            "professional email",
            "polite email",
            "write a message",
        ]
    ):
        return "email_message"
    if any(
        x in low
        for x in [
            "rewrite",
            "rephrase",
            "more formal",
            "more concise",
            "more polite",
            "clearer",
            "shorter",
            "more direct",
        ]
    ):
        return "rewrite_style"
    if any(
        x in low
        for x in [
            "summarize",
            "summary",
            "bullet points",
            "key takeaways",
            "main points",
            "one sentence summary",
        ]
    ):
        return "summary_bullets"
    if any(
        x in low
        for x in [
            "explain",
            "what is",
            "difference between",
            "simple terms",
            "plain language",
            "everyday language",
        ]
    ):
        return "explain_compare"
    return None


def guess_email_subfamily(prompt: str) -> str:
    low = prompt.lower()
    if any(x in low for x in ["job", "application", "interview", "hiring"]):
        return "job_followup_email"
    if any(
        x in low for x in ["reschedule", "move", "postpone", "next week", "meeting"]
    ):
        return "reschedule_email"
    if "thank" in low:
        return "thank_you_email"
    if any(
        x in low
        for x in [
            "file",
            "report",
            "notes",
            "attachment",
            "slides",
            "document",
            "receipt",
        ]
    ):
        return "request_or_confirm_email"
    return "polite_short_message"


def guess_rewrite_subfamily(prompt: str) -> Optional[str]:
    low = prompt.lower()
    if "formal" in low or "professional" in low:
        return "rewrite_formal"
    if "concise" in low:
        return "rewrite_concise"
    if "polite" in low:
        return "rewrite_polite"
    if "clearer" in low or "clear" in low:
        return "rewrite_clearer"
    if "shorter" in low:
        return "rewrite_shorter"
    if "direct" in low:
        return "rewrite_more_direct"
    return None


def guess_summary_subfamily(prompt: str) -> Optional[str]:
    low = prompt.lower()
    if "exactly 3" in low and "bullet" in low:
        return "summary_3_bullets"
    if "exactly 2" in low and (
        "takeaway" in low or "main idea" in low or "key takeaway" in low
    ):
        return "summary_2_takeaways"
    if "one sentence" in low:
        return "summary_1_sentence"
    if "key points" in low or "main points" in low:
        return "extract_key_points"
    if "bullet" in low:
        return "summary_3_bullets"
    return None


def guess_explain_subfamily(prompt: str) -> str:
    low = prompt.lower()
    pairs = {
        "budget": "explain_budget_simple",
        "deadline": "explain_deadline_simple",
        "password manager": "explain_password_manager_simple",
        "agenda": "explain_meeting_agenda_simple",
        "reminder": "explain_reminder_simple",
        "receipt": "explain_receipt_simple",
        "schedule": "explain_schedule_simple",
        "feedback": "explain_feedback_simple",
        "draft": "explain_draft_simple",
        "subscription": "explain_subscription_simple",
        "task list": "explain_task_list_simple",
        "attachment": "explain_attachment_simple",
    }
    if "difference between" in low:
        if "debit card" in low and "credit card" in low:
            return "compare_debit_credit_simple"
        if "receipt" in low and "invoice" in low:
            return "compare_receipt_invoice_simple"
        if "note" in low and "report" in low:
            return "compare_note_report_simple"
        if "reminder" in low and "alarm" in low:
            return "compare_reminder_alarm_simple"
        if "agenda" in low and "notes" in low:
            return "compare_agenda_notes_simple"
        if "password" in low and "pin" in low:
            return "compare_password_pin_simple"
        if "goal" in low and "deadline" in low:
            return "compare_goal_deadline_simple"
        if "saving" in low and "spending" in low:
            return "compare_saving_spending_simple"
    for k, v in pairs.items():
        if k in low:
            return v
    return "explain_budget_simple"


def canonicalize_email(raw_prompt: str) -> Tuple[str, str]:
    sub = guess_email_subfamily(raw_prompt)
    low = raw_prompt.lower()
    if sub == "job_followup_email":
        prompt = "Write a short polite email asking for an update on your job application after an interview. Keep it professional and under 90 words. Do not use placeholders."
    elif sub == "reschedule_email":
        prompt = "Write a short professional email asking to reschedule a meeting to next week. Keep it under 85 words. Do not use placeholders."
    elif sub == "thank_you_email":
        prompt = "Write a short professional thank-you email after a meeting. Keep it under 80 words. Do not use placeholders."
    elif sub == "request_or_confirm_email":
        prompt = "Write a short polite professional email about sending, receiving, or requesting a file or document. Keep it under 80 words. Do not use placeholders."
    else:
        prompt = "Write a brief polite professional message for a workplace context. Keep it under 45 words. Do not use placeholders."
    return sub, prompt


def canonicalize_rewrite(raw_prompt: str) -> Optional[Tuple[str, str]]:
    sub = guess_rewrite_subfamily(raw_prompt)
    if not sub:
        return None
    quoted = extract_quoted_text(raw_prompt)
    if not quoted:
        return None
    mapping = {
        "rewrite_formal": "Rewrite this to be more formal",
        "rewrite_concise": "Rewrite this to be more concise",
        "rewrite_polite": "Rewrite this to sound more polite",
        "rewrite_clearer": "Rewrite this to be clearer",
        "rewrite_shorter": "Rewrite this to be shorter while keeping the meaning",
        "rewrite_more_direct": "Rewrite this to be more direct",
    }
    return sub, f'{mapping[sub]}: "{quoted}" Return only the rewritten text.'


def canonicalize_summary(raw_prompt: str) -> Optional[Tuple[str, str]]:
    sub = guess_summary_subfamily(raw_prompt)
    quoted = extract_quoted_text(raw_prompt)
    if not sub or not quoted:
        return None
    if sub == "summary_3_bullets":
        return (
            sub,
            f'Summarize this in exactly 3 bullet points: "{quoted}" Keep each bullet short.',
        )
    if sub == "summary_2_takeaways":
        return (
            sub,
            f'Give exactly 2 key takeaways from this text: "{quoted}" Keep each takeaway short.',
        )
    if sub == "summary_1_sentence":
        return sub, f'Summarize this in one sentence: "{quoted}"'
    return (
        sub,
        f'List exactly 3 key points from this text: "{quoted}" Keep each point short.',
    )


def canonicalize_explain(raw_prompt: str) -> Tuple[str, str]:
    sub = guess_explain_subfamily(raw_prompt)
    low = raw_prompt.lower()
    if sub.startswith("compare_"):
        if "difference between" in low:
            m = re.search(r"difference between\s+(.+)", raw_prompt, flags=re.IGNORECASE)
            tail = m.group(1).strip() if m else raw_prompt
            return (
                sub,
                f"Explain in simple terms the difference between {tail.rstrip('.?')}. Use at most 3 sentences.",
            )
    thing = None
    if "what is" in low:
        m = re.search(r"what is\s+(.+)", raw_prompt, flags=re.IGNORECASE)
        thing = m.group(1).strip(" .?") if m else None
    if not thing:
        # fall back to subfamily naming
        pretty = sub.replace("explain_", "").replace("_simple", "").replace("_", " ")
        thing = pretty
    return (
        sub,
        f"Explain in at most 3 sentences what {thing} is, in simple everyday language.",
    )


def canonicalize_open_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_prompt = cleanup_prompt_text(row.get("raw_prompt", "") or "")
    family = classify_family(raw_prompt)
    if not family:
        return None
    if family == "email_message":
        sub, canonical = canonicalize_email(raw_prompt)
    elif family == "rewrite_style":
        out = canonicalize_rewrite(raw_prompt)
        if not out:
            return None
        sub, canonical = out
    elif family == "summary_bullets":
        out = canonicalize_summary(raw_prompt)
        if not out:
            return None
        sub, canonical = out
    else:
        sub, canonical = canonicalize_explain(raw_prompt)
    return {
        "id": row["id"],
        "source": row["source"],
        "source_key": row.get("source_key"),
        "family": family,
        "subfamily": sub,
        "raw_prompt": raw_prompt,
        "canonical_prompt": canonical,
        "constraints": default_constraints_for_subfamily(sub),
        "meta": {
            **row.get("meta", {}),
            "source_split": row.get("source_split", ""),
            "from_template": False,
        },
    }


def normalize_template_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "source": row.get("source", "template_paraphrase"),
        "source_key": row.get("parent_seed_id", row["id"]),
        "family": row["family"],
        "subfamily": row["subfamily"],
        "raw_prompt": row["paraphrased_prompt"],
        "canonical_prompt": row["paraphrased_prompt"],
        "constraints": row["constraints"],
        "meta": {
            **row.get("meta", {}),
            "from_template": True,
            "parent_seed_id": row.get("parent_seed_id"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_open_jsonl", required=True)
    ap.add_argument("--template_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    reject = Counter()

    for row in read_jsonl(args.raw_open_jsonl):
        out = canonicalize_open_row(row)
        if out is None:
            reject["open_reject"] += 1
        else:
            rows.append(out)

    for row in read_jsonl(args.template_jsonl):
        rows.append(normalize_template_row(row))

    write_jsonl(args.out_jsonl, rows)
    print(f"Wrote {len(rows)} canonical prompts to {args.out_jsonl}")
    fam = Counter(r["family"] for r in rows)
    src = Counter(r["source"] for r in rows)
    print("[family_counts]")
    for k, v in fam.most_common():
        print(f"  {k}: {v}")
    print("[source_counts]")
    for k, v in src.most_common():
        print(f"  {k}: {v}")
    print("[rejects]")
    for k, v in reject.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
