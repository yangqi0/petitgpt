from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from general_utils import normalize_prompt, read_jsonl, semantic_diversity_scores, stable_id, write_jsonl

TOTAL_FAMILY_QUOTAS = {
    "email_message": 1200,
    "rewrite_style": 1200,
    "summary_bullets": 900,
    "explain_compare": 700,
}
SPLIT_RATIOS = {"train": 0.85, "val": 0.075, "holdout": 0.075}

def load_and_merge(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        rows.extend(read_jsonl(p))
    return rows

def dedup_pass_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        prompt = row["messages"][0]["content"]
        answer = row["messages"][1]["content"]
        key = (normalize_prompt(prompt), normalize_prompt(answer))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def sort_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    answers = [g["messages"][1]["content"] for g in group]
    diversity = semantic_diversity_scores(answers)
    enriched = []
    for row, div in zip(group, diversity):
        scores = row["meta"].get("scores", {})
        enriched.append((
            -(1 if row["meta"].get("generation_round", 1) == 1 else 0),
            -scores.get("instruction_following", 0),
            -scores.get("cleanliness", 0),
            len(row["messages"][1]["content"].split()),
            -div,
            row,
        ))
    enriched.sort()
    return [x[-1] for x in enriched]

def quota_split(total: int) -> Dict[str, int]:
    train = int(round(total * SPLIT_RATIOS["train"]))
    val = int(round(total * SPLIT_RATIOS["val"]))
    holdout = total - train - val
    return {"train": train, "val": val, "holdout": holdout}

def grouped_key(row: Dict[str, Any]) -> str:
    if row["meta"].get("from_template"):
        return row["meta"].get("parent_seed_id") or row["meta"].get("source_key") or row["meta"]["subfamily"]
    return stable_id("grp", normalize_prompt(row["messages"][0]["content"])[:180])

def assign_splits(rows: List[Dict[str, Any]], family_quotas: Dict[str, int], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # group by family and grouped key to reduce leakage
    fam_groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        fam_groups[row["meta"]["family"]][grouped_key(row)].append(row)

    train, val, holdout = [], [], []
    for family, target in family_quotas.items():
        split_targets = quota_split(target)
        groups = list(fam_groups[family].values())
        groups.sort(key=lambda g: (-len(g), g[0]["meta"].get("from_template", False)))
        rng.shuffle(groups)
        counts = Counter()
        for group in groups:
            # pick split with most remaining capacity
            remaining = {s: split_targets[s] - counts[s] for s in ["train", "val", "holdout"]}
            split = max(remaining.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if remaining[split] <= 0:
                continue
            room = remaining[split]
            chosen = group[:room]
            counts[split] += len(chosen)
            if split == "train":
                train.extend(chosen)
            elif split == "val":
                val.extend(chosen)
            else:
                holdout.extend(chosen)
            if counts["train"] >= split_targets["train"] and counts["val"] >= split_targets["val"] and counts["holdout"] >= split_targets["holdout"]:
                break
    return train, val, holdout

def select_by_family(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["meta"]["family"]].append(row)

    selected = []
    for family, quota in TOTAL_FAMILY_QUOTAS.items():
        group = sort_group(by_family.get(family, []))
        # mild source round-robin
        by_source = defaultdict(list)
        for row in group:
            by_source[row["meta"].get("source", "unknown")].append(row)
        sources = list(by_source.keys())
        ptr = 0
        taken = 0
        while taken < quota and any(by_source.values()):
            src = sources[ptr % len(sources)]
            ptr += 1
            if not by_source[src]:
                continue
            row = by_source[src].pop(0)
            selected.append(row)
            taken += 1
        print(f"{family}: selected {taken} / target {quota}")
    return selected

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass_jsonls", nargs="+", required=True)
    ap.add_argument("--out_train_jsonl", required=True)
    ap.add_argument("--out_val_jsonl", required=True)
    ap.add_argument("--out_holdout_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = load_and_merge(args.pass_jsonls)
    rows = dedup_pass_rows(rows)
    print(f"merged+dedup rows: {len(rows)}")

    selected = select_by_family(rows)
    rng = random.Random(args.seed)
    train, val, holdout = assign_splits(selected, TOTAL_FAMILY_QUOTAS, rng)

    write_jsonl(args.out_train_jsonl, train)
    write_jsonl(args.out_val_jsonl, val)
    write_jsonl(args.out_holdout_jsonl, holdout)

    print(f"train={len(train)} val={len(val)} holdout={len(holdout)}")
    for name, group in [("train", train), ("val", val), ("holdout", holdout)]:
        fam = Counter(r["meta"]["family"] for r in group)
        src = Counter(r["meta"].get("source", "unknown") for r in group)
        print(f"[{name}_family_counts]")
        for k, v in fam.most_common():
            print(f"  {k}: {v}")
        print(f"[{name}_source_counts]")
        for k, v in src.most_common():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
