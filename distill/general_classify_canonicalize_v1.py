from __future__ import annotations

import argparse
import json
import re
from collections import Counter
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

def normalize_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def cleanup_prompt_text(s: Any) -> str:
    s = str(s or "").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"^user:\s*", "", s, flags=re.I)
    s = re.sub(r"^instruction:\s*", "", s, flags=re.I)
    return s.strip()

def word_count(s: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", s or ""))

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
            "max_words": 35,
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
            "max_words": 65,
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
            "max_words": 55,
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
            "max_words": 35,
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
        "max_words": 75,
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

BANNED = [
    "python", "javascript", "sql", "function", "code", "algorithm", "debug", "bug",
    "story", "poem", "roleplay", "tweet", "social media", "medical advice",
    "legal advice", "investment advice",
]

def contains_banned(raw: str) -> bool:
    low = raw.lower()
    return any(x in low for x in BANNED)

def extract_quoted_text(s: str) -> Optional[str]:
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', s or "")
    texts = []
    for a, b in matches:
        t = normalize_space(a or b)
        if t:
            texts.append(t)
    return max(texts, key=len) if texts else None

def extract_after_colon_or_newline(raw: str) -> Optional[str]:
    """Extract a candidate payload text from prompts like 'Rewrite this: ...'."""
    raw = cleanup_prompt_text(raw)
    if ":" in raw:
        tail = raw.split(":", 1)[1].strip()
        if 3 <= word_count(tail) <= 80:
            return tail.strip('" ')
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if len(lines) >= 2:
        tail = lines[-1]
        if 3 <= word_count(tail) <= 80:
            return tail.strip('" ')
    return None

def extract_payload_text(row: Dict[str, Any], raw_prompt: str) -> Optional[str]:
    q = extract_quoted_text(raw_prompt)
    if q and 3 <= word_count(q) <= 100:
        return q
    tail = extract_after_colon_or_newline(raw_prompt)
    if tail:
        return tail
    ctx = cleanup_prompt_text(row.get("raw_context", ""))
    if 3 <= word_count(ctx) <= 100:
        return ctx
    return None

def classify_family(raw_prompt: str, raw_category: str = "") -> Optional[str]:
    low = (raw_prompt + " " + raw_category).lower()

    # order matters: rewrite/summary/email first, explain last because it is broad.
    rewrite_keys = [
        "rewrite", "rephrase", "paraphrase", "make this sound", "make it sound",
        "more formal", "more concise", "more polite", "clearer", "shorter",
        "more direct", "improve the wording", "professional tone", "change the tone",
        "edit this sentence", "polish this sentence", "simplify this sentence",
    ]
    if any(k in low for k in rewrite_keys):
        return "rewrite_style"

    summary_keys = [
        "summarize", "summary", "tl;dr", "tldr", "bullet points", "key takeaways",
        "main points", "main ideas", "one sentence summary", "extract key points",
        "condense this", "short summary", "brief summary",
    ]
    if any(k in low for k in summary_keys):
        return "summary_bullets"

    email_keys = [
        "email", "follow up", "follow-up", "reschedule", "thank you", "thank-you",
        "confirm receipt", "professional message", "polite message", "write a message",
        "draft a note", "write a note", "ask a colleague", "ask politely",
        "request a meeting", "schedule a meeting",
    ]
    if any(k in low for k in email_keys):
        return "email_message"

    explain_keys = [
        "explain", "what is", "what are", "difference between", "simple terms",
        "plain language", "everyday language", "describe", "define",
    ]
    if any(k in low for k in explain_keys):
        return "explain_compare"

    return None

def guess_email_subfamily(raw_prompt: str) -> str:
    low = raw_prompt.lower()
    if any(x in low for x in ["job", "application", "interview", "hiring", "recruiter"]):
        return "job_followup_email"
    if any(x in low for x in ["reschedule", "move", "postpone", "next week", "meeting"]):
        return "reschedule_email"
    if "thank" in low:
        return "thank_you_email"
    if any(x in low for x in ["file", "report", "notes", "attachment", "slides", "document", "receipt"]):
        return "request_or_confirm_email"
    return "polite_short_message"

def canonicalize_email(raw_prompt: str) -> Tuple[str, str]:
    sub = guess_email_subfamily(raw_prompt)
    prompt = cleanup_prompt_text(raw_prompt)
    # Preserve source diversity, but add our cleanliness contract.
    if "placeholder" not in prompt.lower():
        prompt += " Do not use placeholders."
    if "under" not in prompt.lower() and "word" not in prompt.lower():
        if sub == "polite_short_message":
            prompt += " Keep it under 45 words."
        else:
            prompt += " Keep it under 90 words."
    return sub, prompt

def guess_rewrite_subfamily(raw_prompt: str) -> str:
    low = raw_prompt.lower()
    if "concise" in low or "shorter" in low:
        return "rewrite_concise"
    if "polite" in low:
        return "rewrite_polite"
    if "clearer" in low or "clear" in low:
        return "rewrite_clearer"
    if "direct" in low:
        return "rewrite_more_direct"
    if "formal" in low or "professional" in low:
        return "rewrite_formal"
    return "rewrite_clearer"

def canonicalize_rewrite(row: Dict[str, Any], raw_prompt: str) -> Optional[Tuple[str, str]]:
    sub = guess_rewrite_subfamily(raw_prompt)
    text = extract_payload_text(row, raw_prompt)
    if not text:
        return None
    mapping = {
        "rewrite_formal": "Rewrite this to be more formal",
        "rewrite_concise": "Rewrite this to be more concise",
        "rewrite_polite": "Rewrite this to sound more polite",
        "rewrite_clearer": "Rewrite this to be clearer",
        "rewrite_more_direct": "Rewrite this to be more direct",
    }
    head = mapping.get(sub, "Rewrite this to be clearer")
    return sub, f'{head}: "{text}" Return only the rewritten text.'

def guess_summary_subfamily(raw_prompt: str) -> str:
    low = raw_prompt.lower()
    if "one sentence" in low:
        return "summary_1_sentence"
    if "2" in low and ("takeaway" in low or "main idea" in low or "key point" in low):
        return "summary_2_takeaways"
    if "takeaway" in low:
        return "summary_2_takeaways"
    if "key point" in low or "main point" in low or "main idea" in low:
        return "extract_key_points"
    return "summary_3_bullets"

def canonicalize_summary(row: Dict[str, Any], raw_prompt: str) -> Optional[Tuple[str, str]]:
    sub = guess_summary_subfamily(raw_prompt)
    text = extract_payload_text(row, raw_prompt)
    if not text:
        return None
    if sub == "summary_1_sentence":
        return sub, f'Summarize this in one sentence: "{text}"'
    if sub == "summary_2_takeaways":
        return sub, f'Give exactly 2 key takeaways from this text: "{text}" Keep each takeaway short.'
    if sub == "extract_key_points":
        return sub, f'List exactly 3 key points from this text: "{text}" Keep each point short.'
    return sub, f'Summarize this in exactly 3 bullet points: "{text}" Keep each bullet short.'

def guess_explain_subfamily(raw_prompt: str) -> str:
    low = raw_prompt.lower()
    if "difference between" in low:
        return "compare_simple"
    if "budget" in low:
        return "explain_budget_simple"
    if "deadline" in low:
        return "explain_deadline_simple"
    if "password" in low:
        return "explain_password_simple"
    if "receipt" in low:
        return "explain_receipt_simple"
    if "schedule" in low:
        return "explain_schedule_simple"
    return "explain_simple"

def canonicalize_explain(raw_prompt: str) -> Tuple[str, str]:
    sub = guess_explain_subfamily(raw_prompt)
    prompt = cleanup_prompt_text(raw_prompt)
    if "sentence" not in prompt.lower():
        prompt += " Use at most 3 sentences."
    if not any(x in prompt.lower() for x in ["simple", "plain", "everyday"]):
        prompt += " Use simple everyday language."
    return sub, prompt

def canonicalize_open_row(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    raw_prompt = cleanup_prompt_text(row.get("raw_prompt", ""))
    raw_category = cleanup_prompt_text(row.get("raw_category", ""))

    if not raw_prompt:
        return None, "empty_raw_prompt"
    if contains_banned(raw_prompt):
        return None, "banned_topic"

    family = classify_family(raw_prompt, raw_category)
    if not family:
        return None, "no_family"

    if family == "email_message":
        sub, canonical = canonicalize_email(raw_prompt)
    elif family == "rewrite_style":
        out = canonicalize_rewrite(row, raw_prompt)
        if not out:
            return None, "no_payload_for_rewrite"
        sub, canonical = out
    elif family == "summary_bullets":
        out = canonicalize_summary(row, raw_prompt)
        if not out:
            return None, "no_payload_for_summary"
        sub, canonical = out
    else:
        sub, canonical = canonicalize_explain(raw_prompt)

    return {
        "id": row["id"],
        "source": row.get("source", ""),
        "source_key": row.get("source_key"),
        "family": family,
        "subfamily": sub,
        "raw_prompt": raw_prompt,
        "canonical_prompt": canonical,
        "constraints": default_constraints_for_subfamily(sub),
        "meta": {
            **row.get("meta", {}),
            "from_template": False,
            "source_split": row.get("source_split", ""),
            "raw_category": raw_category,
        },
    }, ""

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

def maybe_cap_family(rows: List[Dict[str, Any]], max_explain: int) -> List[Dict[str, Any]]:
    if max_explain <= 0:
        return rows
    out = []
    explain_seen = 0
    for r in rows:
        if r["family"] == "explain_compare":
            if explain_seen >= max_explain:
                continue
            explain_seen += 1
        out.append(r)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_open_jsonl", required=True)
    ap.add_argument("--template_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--reject_jsonl", default="")
    ap.add_argument("--max_explain", type=int, default=0, help="Optional cap on explain_compare to avoid imbalance.")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    reject_counter = Counter()
    rejects: List[Dict[str, Any]] = []

    for row in read_jsonl(args.raw_open_jsonl):
        out, reason = canonicalize_open_row(row)
        if out is None:
            reject_counter[reason] += 1
            if args.reject_jsonl:
                rejects.append({**row, "reject_reason": reason})
        else:
            rows.append(out)

    for row in read_jsonl(args.template_jsonl):
        rows.append(normalize_template_row(row))

    rows = maybe_cap_family(rows, args.max_explain)

    write_jsonl(args.out_jsonl, rows)
    if args.reject_jsonl:
        write_jsonl(args.reject_jsonl, rejects)

    print(f"Wrote {len(rows)} canonical prompts to {args.out_jsonl}")
    print("[family_counts]")
    for k, v in Counter(r["family"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print("[source_counts]")
    for k, v in Counter(r["source"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print("[subfamily_counts_top30]")
    for k, v in Counter(r["subfamily"] for r in rows).most_common(30):
        print(f"  {k}: {v}")
    print("[rejects]")
    for k, v in reject_counter.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
