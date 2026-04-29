from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from general_utils import (
    contains_code_markers,
    contains_placeholder,
    has_multiple_answer_pattern,
    has_meta_prefix,
    normalize_prompt,
    openai_compatible_chat,
    read_jsonl,
    semantic_vectors,
    stable_id,
    write_jsonl,
    jaccard_ngrams,
    cosine_sim,
    extract_quoted_text,
)

SYSTEM_PROMPT = """You are rewriting instruction prompts for a small assistant.

Preserve the original task intent.
Keep the same output format and constraints.
Vary the wording naturally.
Do not add examples, explanations, code, placeholders, or multiple options.

Return only one rewritten prompt.
"""

STYLE_INSTRUCTIONS = {
    "natural_reword": "Use natural wording changes, but keep the task identical.",
    "instruction_verb_shift": "Change the instruction verb or opening phrase while keeping the task identical.",
    "light_contextualized": "Make the prompt slightly more natural and realistic, but do not add new task requirements.",
}

def normalize_for_contract(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def token_overlap(a: str, b: str) -> float:
    at = set(re.findall(r"[a-z0-9']+", normalize_for_contract(a)))
    bt = set(re.findall(r"[a-z0-9']+", normalize_for_contract(b)))
    if not at:
        return 1.0
    return len(at & bt) / max(1, len(at))

def contains_prompt_stage_code_pollution(s: str) -> bool:
    """Detect real code pollution in paraphrased prompts.

    Important: do NOT flag the word 'return' here, because many valid
    instruction prompts contain phrases such as 'Return only the rewritten text.'
    """
    low = s or ""
    patterns = [
        r"```",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bprint\s*\(",
    ]
    return any(re.search(p, low, flags=re.IGNORECASE) for p in patterns)

def build_user_prompt(seed: Dict[str, Any], style: str) -> str:
    return f"""Rewrite the following instruction prompt so that it sounds natural and varied, while keeping it semantically equivalent.

Family: {seed['family']}
Subfamily: {seed['subfamily']}
Paraphrase style: {style}

Rules:
- Keep the same task.
- Keep the same output contract.
- Keep the same length or structure requirements.
- Do not add new constraints or extra tasks.
- Do not add placeholders.
- Keep it as a single-turn user instruction.
- {STYLE_INSTRUCTIONS[style]}

Original prompt:
{seed['canonical_prompt']}
"""

def contract_warnings(seed: Dict[str, Any], para: str) -> List[str]:
    """Warnings only. These do NOT reject by default.

    The first versions rejected too aggressively for these conditions, which caused
    only a handful of paraphrases to survive. For this stage, we only need to remove
    obviously bad generations; strict filtering belongs later in answer verification.
    """
    warnings: List[str] = []
    cons = seed.get("constraints", {})
    orig = seed.get("canonical_prompt", "")
    low = normalize_for_contract(para)
    orig_low = normalize_for_contract(orig)

    if cons.get("must_be_email"):
        if not any(x in low for x in ["email", "message", "note", "follow", "reschedule", "thank", "professional", "polite"]):
            warnings.append("possibly_lost_email_task")

    if cons.get("exact_bullets") is not None:
        n = str(cons["exact_bullets"])
        if n not in low or not any(x in low for x in ["bullet", "takeaway", "point", "idea"]):
            warnings.append("possibly_lost_bullet_contract")

    if cons.get("max_sentences") is not None and any(x in orig_low for x in ["at most", "no more than", "use at most"]):
        n = str(cons["max_sentences"])
        if n not in low:
            warnings.append("possibly_lost_sentence_limit")

    if "return only the rewritten text" in orig_low:
        if not any(x in low for x in [
            "return only",
            "only return",
            "provide only",
            "respond with only",
            "give only",
            "without explanation",
            "do not include",
        ]):
            warnings.append("possibly_lost_rewrite_contract")

    quote = extract_quoted_text(orig)
    if quote and token_overlap(quote, para) < 0.55:
        warnings.append("possibly_lost_quoted_text")

    return warnings

def validate_paraphrase(seed: Dict[str, Any], para: str, strict_contract: bool = False) -> Tuple[bool, List[str], List[str]]:
    """Soft paraphrase validator.

    Reject only obvious bad generations:
    - empty
    - placeholders
    - code pollution
    - meta/multiple-answer outputs

    Contract drift is recorded as warning unless --strict_contract is used.
    """
    reasons: List[str] = []

    if not para.strip():
        reasons.append("empty")
    if contains_placeholder(para):
        reasons.append("placeholder")
    if contains_prompt_stage_code_pollution(para):
        reasons.append("code_pollution")
    if has_meta_prefix(para) or has_multiple_answer_pattern(para):
        reasons.append("meta_or_multi")

    warnings = contract_warnings(seed, para)
    if strict_contract:
        reasons.extend(warnings)

    return (len(reasons) == 0, reasons, warnings)

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

def lexical_dedup(rows: List[Dict[str, Any]], threshold: float = 0.95) -> List[Dict[str, Any]]:
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

def semantic_dedup(rows: List[Dict[str, Any]], threshold: float = 0.985) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    for _, group in by_family.items():
        texts = [r["paraphrased_prompt"] for r in group]
        vecs = semantic_vectors(texts)
        if vecs is None:
            out.extend(group)
            continue
        selected_ix: List[int] = []
        for i, row in enumerate(group):
            too_close = False
            for j in selected_ix:
                if cosine_sim(vecs, i, j) >= threshold:
                    too_close = True
                    break
            if not too_close:
                selected_ix.append(i)
                out.append(row)
    return out

def apply_dedup(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    rows = exact_dedup(rows)
    if mode == "exact":
        return rows
    rows = lexical_dedup(rows)
    if mode == "exact_lexical":
        return rows
    rows = semantic_dedup(rows)
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_raw_jsonl", required=True)
    ap.add_argument("--out_dedup_jsonl", required=True)
    ap.add_argument("--out_all_jsonl", default="")
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--strict_contract", action="store_true", help="Reject contract warnings. Usually NOT recommended.")
    ap.add_argument("--dedup_mode", choices=["exact", "exact_lexical", "exact_lexical_semantic"], default="exact")
    args = ap.parse_args()

    seeds = read_jsonl(args.in_jsonl)
    raw_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    rejected = Counter()
    warnings_counter = Counter()
    style_order = ["natural_reword", "instruction_verb_shift", "light_contextualized"]

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

            ok, reasons, warnings = validate_paraphrase(seed, para, strict_contract=args.strict_contract)
            all_rows.append({
                **row,
                "accepted_by_paraphrase_validator": ok,
                "reject_reasons": reasons,
                "contract_warnings": warnings,
            })

            for w in warnings:
                warnings_counter[w] += 1

            if ok:
                raw_rows.append(row)
            else:
                for r in reasons:
                    rejected[r] += 1

    dedup_rows = apply_dedup(raw_rows, args.dedup_mode)

    if args.out_all_jsonl:
        write_jsonl(args.out_all_jsonl, all_rows)
    write_jsonl(args.out_raw_jsonl, raw_rows)
    write_jsonl(args.out_dedup_jsonl, dedup_rows)

    print(f"Seeds: {len(seeds)}")
    print(f"Generated candidates: {len(all_rows)}")
    print(f"Accepted raw paraphrases: {len(raw_rows)}")
    print(f"After dedup ({args.dedup_mode}): {len(dedup_rows)}")

    print("\n[rejected]")
    for k, v in rejected.most_common():
        print(f"  {k}: {v}")

    print("\n[contract_warnings_not_rejected]")
    for k, v in warnings_counter.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
