#!/usr/bin/env python3

from collections import Counter, defaultdict
import json
import random

SRC = "dataset/sft/syn_mix_v4.jsonl"
OUT_TRAIN = "dataset/sft/syn_mix_v4.train.jsonl"
OUT_VAL = "dataset/sft/syn_mix_v4.val.jsonl"

SEED = 1234
N_VAL = 2000

# If you want fixed per-task counts, set this dict (must sum to N_VAL).
# Otherwise leave it as None to split proportionally to corpus distribution.
FIXED_VAL_BY_TASK = None
# Example:
# FIXED_VAL_BY_TASK = {"arithmetic": 1000, "syllogism": 500, "code": 500}


def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    rows = load_jsonl(SRC)
    assert rows, f"Empty dataset: {SRC}"

    buckets = defaultdict(list)
    for r in rows:
        task = (r.get("meta") or {}).get("task")
        if not task:
            raise ValueError("Row missing meta.task")
        buckets[task].append(r)

    task_sizes = {t: len(v) for t, v in buckets.items()}
    total = sum(task_sizes.values())

    if N_VAL >= total:
        raise ValueError(f"N_VAL={N_VAL} >= total={total}")

    # Decide val counts per task
    if FIXED_VAL_BY_TASK is not None:
        if sum(FIXED_VAL_BY_TASK.values()) != N_VAL:
            raise ValueError("FIXED_VAL_BY_TASK must sum to N_VAL")
        val_counts = dict(FIXED_VAL_BY_TASK)
    else:
        # proportional, with rounding; then fix remainder
        val_counts = {}
        remainders = []
        for t, n in task_sizes.items():
            raw = N_VAL * (n / total)
            k = int(raw)
            val_counts[t] = k
            remainders.append((raw - k, t))
        # distribute remaining
        rem = N_VAL - sum(val_counts.values())
        remainders.sort(reverse=True)
        for _, t in remainders[:rem]:
            val_counts[t] += 1

    # Sample per task
    val_rows = []
    train_rows = []
    for t, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        k = val_counts.get(t, 0)
        if k > len(items):
            raise ValueError(f"Not enough items for task={t}: need {k}, have {len(items)}")
        val_rows.extend(items[:k])
        train_rows.extend(items[k:])

    rng.shuffle(val_rows)
    rng.shuffle(train_rows)

    write_jsonl(OUT_VAL, val_rows)
    write_jsonl(OUT_TRAIN, train_rows)

    print("SRC:", SRC, "total:", len(rows))
    print(
        "VAL:",
        OUT_VAL,
        "n:",
        len(val_rows),
        "by_task:",
        dict(Counter(r["meta"]["task"] for r in val_rows)),
    )
    print(
        "TRAIN:",
        OUT_TRAIN,
        "n:",
        len(train_rows),
        "by_task:",
        dict(Counter(r["meta"]["task"] for r in train_rows)),
    )


if __name__ == "__main__":
    main()
