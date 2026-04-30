from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from general_utils import (
    normalize_prompt,
    read_jsonl,
    semantic_diversity_scores,
    stable_id,
    write_jsonl,
)

def load_and_merge(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        got = read_jsonl(p)
        print(f"[read] {p}: {len(got)}")
        rows.extend(got)
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

def grouped_key(row: Dict[str, Any]) -> str:
    meta = row.get("meta", {})
    if meta.get("from_template"):
        return (
            meta.get("parent_seed_id")
            or meta.get("source_key")
            or meta.get("subfamily")
            or stable_id("grp", normalize_prompt(row["messages"][0]["content"])[:180])
        )
    return stable_id("grp", normalize_prompt(row["messages"][0]["content"])[:180])

def sort_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(group) <= 1:
        return group
    answers = [g["messages"][1]["content"] for g in group]
    diversity = semantic_diversity_scores(answers)
    enriched = []
    for row, div in zip(group, diversity):
        scores = row.get("meta", {}).get("scores", {})
        enriched.append((
            -(1 if row.get("meta", {}).get("generation_round", 1) == 1 else 0),
            -scores.get("instruction_following", 0),
            -scores.get("cleanliness", 0),
            len(row["messages"][1]["content"].split()),
            -div,
            row,
        ))
    enriched.sort(key=lambda x: x[:-1])
    return [x[-1] for x in enriched]

def select_rows(rows: List[Dict[str, Any]], target_total: int, seed: int) -> List[Dict[str, Any]]:
    """Select up to target_total rows. If target_total<=0, keep all rows."""
    rows_by_family = defaultdict(list)
    for row in rows:
        rows_by_family[row["meta"]["family"]].append(row)

    # Sort within family for stability / quality.
    for fam in list(rows_by_family.keys()):
        rows_by_family[fam] = sort_group(rows_by_family[fam])

    if target_total <= 0 or target_total >= len(rows):
        selected = []
        for fam in sorted(rows_by_family.keys()):
            selected.extend(rows_by_family[fam])
        return selected

    # Proportional family quotas based on available rows, not hardcoded 4000 quotas.
    total_available = len(rows)
    quotas = {}
    remaining = target_total
    families = sorted(rows_by_family.keys())
    for fam in families[:-1]:
        q = round(target_total * len(rows_by_family[fam]) / total_available)
        q = max(0, min(q, len(rows_by_family[fam])))
        quotas[fam] = q
        remaining -= q
    last = families[-1]
    quotas[last] = max(0, min(remaining, len(rows_by_family[last])))

    # If rounding underfills, top up from families with spare capacity.
    while sum(quotas.values()) < target_total:
        changed = False
        for fam in families:
            if quotas[fam] < len(rows_by_family[fam]):
                quotas[fam] += 1
                changed = True
                if sum(quotas.values()) >= target_total:
                    break
        if not changed:
            break

    selected = []
    for fam in families:
        selected.extend(rows_by_family[fam][:quotas[fam]])
        print(f"{fam}: selected {quotas[fam]} / available {len(rows_by_family[fam])}")
    return selected

def make_groups(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for row in rows:
        fam = row["meta"]["family"]
        key = grouped_key(row)
        groups[f"{fam}::{key}"].append(row)
    return dict(groups)

def assign_splits(
    rows: List[Dict[str, Any]],
    val_ratio: float,
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group-aware split with approximate family/source balance."""
    rng = random.Random(seed)

    by_family = defaultdict(list)
    for row in rows:
        by_family[row["meta"]["family"]].append(row)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []

    for fam, fam_rows in sorted(by_family.items()):
        groups = list(make_groups(fam_rows).values())
        rng.shuffle(groups)

        n = len(fam_rows)
        val_target = int(round(n * val_ratio))
        hold_target = int(round(n * holdout_ratio))
        train_target = n - val_target - hold_target

        counts = {"train": 0, "val": 0, "holdout": 0}
        buckets = {"train": [], "val": [], "holdout": []}
        targets = {"train": train_target, "val": val_target, "holdout": hold_target}

        for g in groups:
            # Put each whole group into the split with the largest remaining need.
            candidates = sorted(
                ["train", "val", "holdout"],
                key=lambda s: (targets[s] - counts[s], s == "train"),
                reverse=True,
            )
            chosen = candidates[0]
            buckets[chosen].extend(g)
            counts[chosen] += len(g)

        train.extend(buckets["train"])
        val.extend(buckets["val"])
        holdout.extend(buckets["holdout"])

        print(f"[split:{fam}] total={n} train={len(buckets['train'])} val={len(buckets['val'])} holdout={len(buckets['holdout'])}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(holdout)
    return train, val, holdout

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass_jsonls", nargs="+", required=True)
    ap.add_argument("--out_train_jsonl", required=True)
    ap.add_argument("--out_val_jsonl", required=True)
    ap.add_argument("--out_holdout_jsonl", required=True)
    ap.add_argument("--target_total", type=int, default=0, help="0 means use all deduped pass rows.")
    ap.add_argument("--val_ratio", type=float, default=0.075)
    ap.add_argument("--holdout_ratio", type=float, default=0.075)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = load_and_merge(args.pass_jsonls)
    print(f"merged rows: {len(rows)}")
    rows = dedup_pass_rows(rows)
    print(f"merged+dedup rows: {len(rows)}")

    selected = select_rows(rows, args.target_total, args.seed)
    print(f"selected rows: {len(selected)}")

    train, val, holdout = assign_splits(
        selected,
        val_ratio=args.val_ratio,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )

    write_jsonl(args.out_train_jsonl, train)
    write_jsonl(args.out_val_jsonl, val)
    write_jsonl(args.out_holdout_jsonl, holdout)

    print(f"\n[final]")
    print(f"train={len(train)} val={len(val)} holdout={len(holdout)} total={len(train)+len(val)+len(holdout)}")

    for name, group in [("train", train), ("val", val), ("holdout", holdout)]:
        fam = Counter(r["meta"]["family"] for r in group)
        src = Counter(r["meta"].get("source", "unknown") for r in group)
        print(f"\n[{name}_family_counts]")
        for k, v in fam.most_common():
            print(f"  {k}: {v}")
        print(f"[{name}_source_counts]")
        for k, v in src.most_common():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
