from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

def normalize_prompt(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r"[^\w\s\"'`-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def dedup_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        msgs = row.get("messages", [])
        if len(msgs) < 2:
            continue
        user = normalize_prompt(msgs[0].get("content", ""))
        assistant = normalize_prompt(msgs[1].get("content", ""))
        key = (user, assistant)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def stable_group_key(row: Dict[str, Any]) -> str:
    meta = row.get("meta", {})
    source = meta.get("source", "")
    family = meta.get("family", "")
    if source in {"core_families_v1", "template_paraphrase", "template_seed"}:
        return f"{source}::{family}::{meta.get('source_key', '')}"
    return normalize_prompt(row["messages"][0]["content"])[:180]

def select_rows(rows: List[Dict[str, Any]], target: int, seed: int, prefer_round1: bool = True) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    def score(row: Dict[str, Any]) -> Tuple:
        meta = row.get("meta", {})
        scores = meta.get("scores", {}) if isinstance(meta.get("scores"), dict) else {}
        round1 = 1 if meta.get("generation_round", 1) == 1 else 0
        assistant_len = len(row.get("messages", [{}, {"content": ""}])[1].get("content", "").split())
        return (
            -round1 if prefer_round1 else 0,
            -scores.get("instruction_following", 0),
            -scores.get("cleanliness", 0),
            assistant_len,
            meta.get("source", ""),
        )
    rows.sort(key=score)
    # light group cap to reduce near-duplicate leakage from families/templates
    grouped = defaultdict(list)
    for row in rows:
        grouped[stable_group_key(row)].append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    out = []
    for group in groups:
        out.extend(group[:1])
        if len(out) >= target:
            break
    if len(out) < target:
        flat = [r for g in groups for r in g[1:]]
        rng.shuffle(flat)
        out.extend(flat[: max(0, target - len(out))])
    return out[:target]

def summarize(name: str, rows: List[Dict[str, Any]]) -> None:
    bucket = Counter()
    source = Counter()
    family = Counter()
    for row in rows:
        meta = row.get("meta", {})
        bucket[meta.get("bucket", "unknown")] += 1
        source[meta.get("source", "unknown")] += 1
        family[meta.get("family", "unknown")] += 1
    print(f"[{name}] n={len(rows)}")
    print("  bucket_counts:", dict(bucket))
    print("  source_top:", source.most_common(10))
    print("  family_top:", family.most_common(12))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_train_jsonl", required=True)
    ap.add_argument("--code_val_jsonl", required=True)
    ap.add_argument("--general_train_jsonl", required=True)
    ap.add_argument("--general_val_jsonl", required=True)
    ap.add_argument("--out_train_jsonl", required=True)
    ap.add_argument("--out_val_jsonl", required=True)
    ap.add_argument("--target_code_train", type=int, default=4800)
    ap.add_argument("--target_general_train", type=int, default=1800)
    ap.add_argument("--target_code_val", type=int, default=400)
    ap.add_argument("--target_general_val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    code_train = dedup_examples(read_jsonl(args.code_train_jsonl))
    code_val = dedup_examples(read_jsonl(args.code_val_jsonl))
    general_train = dedup_examples(read_jsonl(args.general_train_jsonl))
    general_val = dedup_examples(read_jsonl(args.general_val_jsonl))

    sel_code_train = select_rows(code_train, min(args.target_code_train, len(code_train)), args.seed)
    sel_general_train = select_rows(general_train, min(args.target_general_train, len(general_train)), args.seed + 1)
    sel_code_val = select_rows(code_val, min(args.target_code_val, len(code_val)), args.seed + 2)
    sel_general_val = select_rows(general_val, min(args.target_general_val, len(general_val)), args.seed + 3)

    train_mix = sel_code_train + sel_general_train
    val_mix = sel_code_val + sel_general_val

    rng = random.Random(args.seed)
    rng.shuffle(train_mix)
    rng.shuffle(val_mix)

    write_jsonl(args.out_train_jsonl, train_mix)
    write_jsonl(args.out_val_jsonl, val_mix)

    summarize("code_train_selected", sel_code_train)
    summarize("general_train_selected", sel_general_train)
    summarize("train_mix", train_mix)
    summarize("code_val_selected", sel_code_val)
    summarize("general_val_selected", sel_general_val)
    summarize("val_mix", val_mix)

if __name__ == "__main__":
    main()
