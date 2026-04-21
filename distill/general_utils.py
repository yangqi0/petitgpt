from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PLACEHOLDER_PATTERNS = [
    r"\[name\]",
    r"\[your name\]",
    r"\[general\]",
    r"\[company\]",
    r"\[email\]",
    r"\[date\]",
    r"\[contact\]",
]

META_PREFIX_PATTERNS = [
    r"^here is\b",
    r"^here's\b",
    r"^below is\b",
    r"^this is\b",
    r"^rewritten\b",
    r"^drafted by\b",
]

CODE_PATTERNS = [
    r"\bimport\b",
    r"\bdef\b",
    r"\bclass\b",
    r"```",
    r"\breturn\b",
    r"\bprint\s*\(",
]

MULTI_ANSWER_PATTERNS = [
    r"\boption\s*1\b",
    r"\balternative\b",
    r"\bversion\s*1\b",
    r"\bthree versions\b",
    r"\b3 versions\b",
]

BULLET_RE = re.compile(r'^\s*(?:[-*•]|\d+\.)\s+')
QUOTE_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_text(s: str) -> str:
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize_prompt(s: str) -> str:
    s = normalize_text(s).lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^\w\s\"'-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def word_count(s: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", s or ""))

def sentence_count(s: str) -> int:
    s = normalize_text(s)
    if not s:
        return 0
    parts = re.split(r'(?<=[.!?])\s+|\n+', s)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

def line_count(s: str) -> int:
    return len([x for x in (s or "").splitlines() if x.strip()])

def detect_bullets(lines: Sequence[str]) -> List[str]:
    return [line for line in lines if BULLET_RE.match(line)]

def bullet_count(s: str) -> int:
    return len(detect_bullets([x for x in (s or "").splitlines() if x.strip()]))

def strip_bullet_marker(s: str) -> str:
    return BULLET_RE.sub("", s).strip()

def contains_placeholder(s: str) -> bool:
    s2 = (s or "").lower()
    return any(re.search(p, s2) for p in PLACEHOLDER_PATTERNS)

def contains_code_markers(s: str) -> bool:
    s2 = s or ""
    return any(re.search(p, s2, flags=re.IGNORECASE) for p in CODE_PATTERNS)

def has_meta_prefix(s: str) -> bool:
    s2 = normalize_text(s).lower()
    return any(re.search(p, s2, flags=re.IGNORECASE) for p in META_PREFIX_PATTERNS)

def has_multiple_answer_pattern(s: str) -> bool:
    s2 = s or ""
    return any(re.search(p, s2, flags=re.IGNORECASE) for p in MULTI_ANSWER_PATTERNS)

def has_prompt_echo(prompt: str, answer: str) -> bool:
    p = normalize_prompt(prompt)
    a = normalize_prompt(answer)
    if not p or not a:
        return False
    return p[:80] in a or p in a

def extract_quoted_text(prompt: str) -> Optional[str]:
    matches = QUOTE_RE.findall(prompt or "")
    texts = []
    for a, b in matches:
        txt = a or b
        txt = normalize_space(txt)
        if txt:
            texts.append(txt)
    if not texts:
        return None
    return max(texts, key=len)

def stable_id(prefix: str, *parts: str, length: int = 10) -> str:
    text = "||".join([prefix] + [str(p) for p in parts])
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:length]}"

def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> set[str]:
        toks = normalize_prompt(s).split()
        if len(toks) < n:
            return {" ".join(toks)} if toks else set()
        return {" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / max(1, len(ga | gb))

def semantic_vectors(texts: List[str]) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            return vec.fit_transform(texts)
        except Exception:
            return None

def cosine_sim(a: Any, i: int, j: int) -> float:
    try:
        import numpy as np
        vi = a[i]
        vj = a[j]
        if hasattr(vi, "toarray"):
            vi = vi.toarray()[0]
            vj = vj.toarray()[0]
        denom = (np.linalg.norm(vi) * np.linalg.norm(vj))
        if denom == 0:
            return 0.0
        return float(np.dot(vi, vj) / denom)
    except Exception:
        return 0.0

def extract_last_user_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    users = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    return normalize_text(users[-1]) if users else ""

def cleanup_prompt_text(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"^user:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^instruction:\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def default_constraints_for_subfamily(subfamily: str) -> Dict[str, Any]:
    if subfamily.endswith("email") or subfamily == "polite_short_message" or subfamily == "request_or_confirm_email":
        return {
            "max_words": 90,
            "max_sentences": 6,
            "exact_bullets": None,
            "must_be_email": True,
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
        "single_answer_only": True,
        "no_code": True,
        "no_placeholders": True,
        "no_explanation": True,
        "must_keep_meaning": True,
        "must_be_everyday_language": True,
    }

def has_greeting(answer: str) -> bool:
    first = ""
    for line in answer.splitlines():
        line = line.strip()
        if line:
            first = line.lower()
            break
    return first.startswith(("dear ", "hello ", "hi "))

def has_signoff(answer: str) -> bool:
    lines = [x.strip().lower() for x in answer.splitlines() if x.strip()]
    if not lines:
        return False
    tail = "\n".join(lines[-3:])
    return any(s in tail for s in ["best regards", "best,", "kind regards", "sincerely", "thank you,", "best wishes"])

def body_sentence_count(answer: str) -> int:
    lines = [x.strip() for x in answer.splitlines() if x.strip()]
    if len(lines) < 3:
        return sentence_count(answer)
    mid = "\n".join(lines[1:-1])
    return sentence_count(mid)

def looks_like_email(answer: str) -> bool:
    return has_greeting(answer) and has_signoff(answer)

def mixed_bullet_styles(answer: str) -> bool:
    styles = set()
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^-\s+", line):
            styles.add("-")
        elif re.match(r"^\*\s+", line):
            styles.add("*")
        elif re.match(r"^•\s+", line):
            styles.add("•")
        elif re.match(r"^\d+\.\s+", line):
            styles.add("1.")
    return len(styles) > 1

def has_non_bullet_content(answer: str, bullets: Sequence[str]) -> bool:
    bullet_set = set(bullets)
    for line in [x for x in answer.splitlines() if x.strip()]:
        if line not in bullet_set and not BULLET_RE.match(line):
            return True
    return False

def has_numbered_list(answer: str) -> bool:
    return any(re.match(r"^\s*\d+\.\s+", line) for line in answer.splitlines())

def looks_too_technical(answer: str) -> bool:
    technical = [
        "financial statement",
        "cash flow",
        "income statement",
        "liabilities",
        "equity",
        "algorithm",
        "implementation",
        "framework",
        "optimization",
    ]
    low = answer.lower()
    hits = sum(1 for t in technical if t in low)
    return hits >= 2

def semantic_diversity_scores(texts: List[str]) -> List[float]:
    if len(texts) <= 1:
        return [1.0] * len(texts)
    vecs = semantic_vectors(texts)
    if vecs is None:
        return [1.0] * len(texts)
    scores = []
    for i in range(len(texts)):
        sims = [cosine_sim(vecs, i, j) for j in range(len(texts)) if j != i]
        scores.append(1.0 - (sum(sims) / max(1, len(sims))))
    return scores

def openai_compatible_chat(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 120,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> str:
    import urllib.request
    import urllib.error

    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"]

def print_counter(title: str, counter: Counter) -> None:
    print(f"\n[{title}]")
    for k, v in counter.most_common():
        print(f"  {k}: {v}")
