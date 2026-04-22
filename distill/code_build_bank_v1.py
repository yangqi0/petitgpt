from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from code_utils import normalize_prompt, read_jsonl, write_jsonl

DEFAULT_SOURCE_TARGETS = {
    "mbpp": 1800,
    "apps_intro": 1000,
    "core_families_v1": 1200,
}
DEFAULT_TOTAL = 5000

def dedup_pass_rows(rows):
    seen = set()
    out = []
    for row in rows:
        prompt = normalize_prompt(row["messages"][0]["content"])
        code = normalize_prompt(row["messages"][1]["content"])
        key = (prompt, code)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def select_by_source(rows, target_total):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["meta"].get("source", "unknown")].append(row)
    selected = []
    for source, quota in DEFAULT_SOURCE_TARGETS.items():
        selected.extend(by_source.get(source, [])[:quota])
    if len(selected) < target_total:
        leftovers = []
        for source, group in by_source.items():
            if source not in DEFAULT_SOURCE_TARGETS:
                leftovers.extend(group)
            else:
                leftovers.extend(group[DEFAULT_SOURCE_TARGETS[source]:])
        selected.extend(leftovers[:max(0, target_total - len(selected))])
    return selected[:target_total]

def grouped_key(row):
    src = row["meta"].get("source", "")
    if src == "core_families_v1":
        return (row["meta"].get("family") or "") + "::" + normalize_prompt(row["messages"][0]["content"])[:80]
    return normalize_prompt(row["messages"][0]["content"])[:180]

def assign_splits(rows, seed=13):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[grouped_key(row)].append(row)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    train, val, holdout = [], [], []
    total = len(rows)
    t_train = int(round(total * 0.85))
    t_val = int(round(total * 0.075))
    t_hold = total - t_train - t_val
    counts = Counter()
    for group in group_list:
        remaining = {"train": t_train - counts["train"], "val": t_val - counts["val"], "holdout": t_hold - counts["holdout"]}
        split = max(remaining.items(), key=lambda kv: (kv[1], kv[0]))[0]
        room = remaining[split]
        if room <= 0:
            continue
        chosen = group[:room]
        counts[split] += len(chosen)
        if split == "train":
            train.extend(chosen)
        elif split == "val":
            val.extend(chosen)
        else:
            holdout.extend(chosen)
    return train, val, holdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass_jsonls", nargs="+", required=True)
    ap.add_argument("--out_train_jsonl", required=True)
    ap.add_argument("--out_val_jsonl", required=True)
    ap.add_argument("--out_holdout_jsonl", required=True)
    ap.add_argument("--target_total", type=int, default=DEFAULT_TOTAL)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = []
    for p in args.pass_jsonls:
        rows.extend(read_jsonl(p))
    rows = dedup_pass_rows(rows)
    selected = select_by_source(rows, args.target_total)
    train, val, holdout = assign_splits(selected, seed=args.seed)
    write_jsonl(args.out_train_jsonl, train)
    write_jsonl(args.out_val_jsonl, val)
    write_jsonl(args.out_holdout_jsonl, holdout)

    print(f"selected={len(selected)} train={len(train)} val={len(val)} holdout={len(holdout)}")
    for name, group in [("train", train), ("val", val), ("holdout", holdout)]:
        src = Counter(r["meta"].get("source", "unknown") for r in group)
        fam = Counter((r["meta"].get("family") or "open_code") for r in group)
        print(f"[{name}_source_counts]")
        for k, v in src.most_common():
            print(f"  {k}: {v}")
        print(f"[{name}_family_counts]")
        for k, v in fam.most_common():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
