from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from general_utils import (
    contains_code_markers,
    contains_placeholder,
    cosine_sim,
    default_constraints_for_subfamily,
    extract_quoted_text,
    has_multiple_answer_pattern,
    has_prompt_echo,
    has_meta_prefix,
    jaccard_ngrams,
    normalize_prompt,
    openai_compatible_chat,
    print_counter,
    read_jsonl,
    semantic_vectors,
    stable_id,
    write_jsonl,
)

SYSTEM_PROMPT = """You are rewriting instruction prompts for a very small assistant.

Preserve the task intent exactly.
Preserve the required output format exactly.
Preserve any word limit, sentence limit, or bullet-count requirement exactly.
Do not add new requirements.
Do not remove important constraints.
Do not turn the task into a different task.
Do not add examples, explanations, code, or multiple options.

Return only one rewritten prompt.
"""

STYLE_INSTRUCTIONS = {
    "natural_reword": "Use natural wording changes, but keep the task identical.",
    "instruction_verb_shift": "Prefer changing the instruction verb or opening phrase, while keeping all constraints identical.",
    "light_contextualized": "Make the prompt sound slightly more natural and realistic, but do not add any new task requirements.",
}

def build_user_prompt(seed: Dict[str, Any], style: str) -> str:
    return f"""Rewrite the following instruction prompt so that it sounds natural and varied, while keeping it semantically equivalent.

Family: {seed['family']}
Subfamily: {seed['subfamily']}
Paraphrase style: {style}

Rules:
- Keep the same task.
- Keep the same output contract.
- Keep the same length or structure requirements.
- Do not add any new constraints or extra tasks.
- Do not add placeholders.
- Keep it as a single-turn user instruction.
- {STYLE_INSTRUCTIONS[style]}

Original prompt:
{seed['canonical_prompt']}
"""

def normalize_for_contract(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def fuzzy_contains_quoted_text(original_quote: str, para: str) -> bool:
    """Return True if the paraphrase appears to preserve the quoted text."""
    if not original_quote:
        return True

    q = normalize_for_contract(original_quote)
    p = normalize_for_contract(para)

    if q in p:
        return True

    q_toks = set(re.findall(r"[a-z0-9']+", q))
    p_toks = set(re.findall(r"[a-z0-9']+", p))
    if not q_toks:
        return True

    overlap = len(q_toks & p_toks) / max(1, len(q_toks))
    return overlap >= 0.75


def has_rewrite_contract(para: str) -> bool:
    """Accept several equivalent ways of saying 'return only the rewritten text'."""
    low = normalize_for_contract(para)
    patterns = [
        "return only the rewritten text",
        "return only the revised text",
        "return only the rephrased text",
        "only return the rewritten text",
        "only return the revised text",
        "respond with only the rewritten text",
        "provide only the rewritten text",
        "give only the rewritten text",
        "return just the rewritten text",
        "do not include any explanation",
        "without any explanation",
    ]
    return any(p in low for p in patterns)


def has_email_task(para: str) -> bool:
    low = normalize_for_contract(para)
    return any(
        x in low
        for x in [
            "email",
            "message",
            "note",
            "follow-up",
            "follow up",
            "reschedule",
            "thank-you",
            "thank you",
            "professional",
            "polite",
        ]
    )


def has_sentence_limit(seed_prompt: str, para: str, max_sentences: int) -> bool:
    """Accept common equivalent phrasings of sentence limits."""
    orig = normalize_for_contract(seed_prompt)
    low = normalize_for_contract(para)

    if "at most" not in orig and "use at most" not in orig and "no more than" not in orig:
        return True

    patterns = [
        f"at most {max_sentences}",
        f"no more than {max_sentences}",
        f"up to {max_sentences}",
        f"using at most {max_sentences}",
        f"in {max_sentences} sentences or fewer",
        f"in no more than {max_sentences}",
    ]
    return any(p in low for p in patterns)


def has_bullet_contract(para: str, n: int) -> bool:
    low = normalize_for_contract(para)
    return (
        f"exactly {n}" in low
        and ("bullet" in low or "takeaway" in low or "point" in low or "main idea" in low)
    )


def validate_paraphrase(seed: Dict[str, Any], para: str) -> Tuple[bool, List[str]]:
    """Validate template paraphrases without over-rejecting harmless wording changes.

    This validator is intentionally softer than the final answer verifier.
    Its job is only to ensure the prompt is still the same task family and still
    keeps the important output contract.
    """
    reasons: List[str] = []

    if not para.strip():
        reasons.append("empty")

    # Do not reject exact unchanged here. Dedup/diversity later can remove it.
    if contains_placeholder(para):
        reasons.append("placeholder")
    if contains_code_markers(para):
        reasons.append("code_pollution")
    if has_meta_prefix(para) or has_multiple_answer_pattern(para):
        reasons.append("meta_or_multi")

    cons = seed["constraints"]
    orig = seed["canonical_prompt"].lower()

    if cons.get("must_be_email") and not has_email_task(para):
        reasons.append("lost_email_task")

    if cons.get("exact_bullets") is not None:
        if not has_bullet_contract(para, int(cons["exact_bullets"])):
            reasons.append("lost_exact_bullet_constraint")

    if cons.get("max_sentences") is not None:
        if not has_sentence_limit(seed["canonical_prompt"], para, int(cons["max_sentences"])):
            reasons.append("lost_sentence_limit")

    if "return only the rewritten text" in orig and not has_rewrite_contract(para):
        reasons.append("lost_rewrite_contract")

    m = extract_quoted_text(seed["canonical_prompt"])
    if m and not fuzzy_contains_quoted_text(m, para):
        reasons.append("lost_quoted_text")

    return (len(reasons) == 0, reasons)

def exact_dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = normalize_prompt(row["paraphrased_prompt"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def lexical_dedup(rows: List[Dict[str, Any]], threshold: float = 0.92) -> List[Dict[str, Any]]:
    keep: List[Dict[str, Any]] = []
    by_sub = defaultdict(list)
    for row in rows:
        by_sub[row["subfamily"]].append(row)
    for _, group in by_sub.items():
        selected: List[Dict[str, Any]] = []
        for row in group:
            if all(jaccard_ngrams(row["paraphrased_prompt"], old["paraphrased_prompt"]) < threshold for old in selected):
                selected.append(row)
        keep.extend(selected)
    return keep

def semantic_dedup(rows: List[Dict[str, Any]], hi: float = 0.985, cluster: float = 0.965) -> List[Dict[str, Any]]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    out: List[Dict[str, Any]] = []
    for _, group in by_family.items():
        texts = [r["paraphrased_prompt"] for r in group]
        vecs = semantic_vectors(texts)
        if vecs is None:
            out.extend(group)
            continue
        parent_kept = Counter()
        used = [False] * len(group)
        for i, row in enumerate(group):
            if used[i]:
                continue
            cluster_ix = [i]
            used[i] = True
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                sim = cosine_sim(vecs, i, j)
                if sim >= hi:
                    used[j] = True
                    continue
                if sim >= cluster:
                    cluster_ix.append(j)
                    used[j] = True
            # keep at most 3 from cluster, at most 3 per parent seed overall
            kept_in_cluster = 0
            for ix in cluster_ix:
                parent = group[ix]["parent_seed_id"]
                if parent_kept[parent] >= 3:
                    continue
                out.append(group[ix])
                parent_kept[parent] += 1
                kept_in_cluster += 1
                if kept_in_cluster >= 3:
                    break
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_raw_jsonl", required=True)
    ap.add_argument("--out_dedup_jsonl", required=True)
    ap.add_argument("--out_all_jsonl", default="", help="Optional: write every generated paraphrase with validation reasons.")
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    seeds = read_jsonl(args.in_jsonl)
    raw_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    style_order = ["natural_reword", "instruction_verb_shift", "light_contextualized"]
    rejected = Counter()

    for seed in seeds:
        for idx, style in enumerate(style_order, start=1):
            try:
                para = openai_compatible_chat(
                    api_base=args.api_base,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=build_user_prompt(seed, style),
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                ).strip()
            except Exception as e:
                rejected[f"api_error:{type(e).__name__}"] += 1
                continue

            row = {
                "id": stable_id("tmplp", seed["id"], style),
                "parent_seed_id": seed["id"],
                "family": seed["family"],
                "subfamily": seed["subfamily"],
                "source": "template_paraphrase",
                "canonical_prompt": seed["canonical_prompt"],
                "paraphrased_prompt": para,
                "constraints": seed["constraints"],
                "meta": {
                    **seed.get("meta", {}),
                    "paraphrase_style": style,
                    "paraphrase_index": idx,
                },
            }
            ok, reasons = validate_paraphrase(seed, para)
            all_rows.append({**row, "accepted_by_paraphrase_validator": ok, "reject_reasons": reasons})
            if ok:
                raw_rows.append(row)
            else:
                for r in reasons:
                    rejected[r] += 1

    if args.out_all_jsonl:
        write_jsonl(args.out_all_jsonl, all_rows)
    write_jsonl(args.out_raw_jsonl, raw_rows)

    dedup_1 = exact_dedup(raw_rows)
    dedup_2 = lexical_dedup(dedup_1)
    dedup_3 = semantic_dedup(dedup_2)
    write_jsonl(args.out_dedup_jsonl, dedup_3)

    print(f"Seeds: {len(seeds)}")
    print(f"Accepted raw paraphrases: {len(raw_rows)}")
    print(f"After exact dedup: {len(dedup_1)}")
    print(f"After lexical dedup: {len(dedup_2)}")
    print(f"After semantic dedup: {len(dedup_3)}")
    print_counter("rejected", rejected)

if __name__ == "__main__":
    main()
