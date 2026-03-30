#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import random
import statistics

OUT_DIR = "dataset/stage2/prompts_v1_2"
os.makedirs(OUT_DIR, exist_ok=True)

rng = random.Random(1234)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def fmt_num(x):
    if isinstance(x, float):
        s = f"{x:.8f}".rstrip("0").rstrip(".")
        return s
    return str(x)


rows = []
answer_key = {}
mid = 1

# -------------------------------------------------
# A. word problems (main bucket)
# -------------------------------------------------

# notebook discount
for price in [6, 8, 10, 12]:
    for qty in [2, 3, 4]:
        for discount in [10, 15, 20]:
            final_price = price * qty * (1 - discount / 100)
            pid = f"math_word_{mid:04d}"
            rows.append({
                "id": pid,
                "task_type": "math_word",
                "prompt": f"A notebook costs {price} euros. You buy {qty} notebooks and get a {discount}% discount on the total. What is the final price? Show the steps briefly."
            })
            answer_key[pid] = [fmt_num(final_price)]
            mid += 1

# ladder distance
for length in [8, 10, 12, 15]:
    for climbs in [3, 4, 5]:
        total = length * climbs
        pid = f"math_word_{mid:04d}"
        rows.append({
            "id": pid,
            "task_type": "math_word",
            "prompt": f"A ladder is {length} meters long. Tom climbs it {climbs} times. How many meters does he climb in total? Show the steps briefly."
        })
        answer_key[pid] = [fmt_num(total)]
        mid += 1

# train distance
for speed in [40, 50, 60]:
    for hours in [2, 3, 4]:
        distance = speed * hours
        pid = f"math_word_{mid:04d}"
        rows.append({
            "id": pid,
            "task_type": "math_word",
            "prompt": f"A train travels {speed} kilometers per hour for {hours} hours. How far does it travel? Show the steps briefly."
        })
        answer_key[pid] = [fmt_num(distance)]
        mid += 1

# apples + bananas
for apples, aprice, bananas, bprice in [
    (4, 2, 3, 1),
    (5, 3, 2, 2),
    (3, 4, 4, 1),
    (6, 2, 5, 1),
]:
    total = apples * aprice + bananas * bprice
    pid = f"math_word_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_word",
        "prompt": f"Anna buys {apples} apples at {aprice} euros each and {bananas} bananas at {bprice} euro each. What is the total cost? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(total)]
    mid += 1

# equal sharing
for total_items, students in [(24, 6), (30, 5), (36, 4), (42, 7)]:
    each = total_items / students
    pid = f"math_word_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_word",
        "prompt": f"A box contains {total_items} pencils. If {students} students share them equally, how many pencils does each student get? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(each)]
    mid += 1

# shirts discount
for price, discount in [(40, 15), (50, 20), (60, 10), (80, 25)]:
    final_price = price * (1 - discount / 100)
    pid = f"math_word_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_word",
        "prompt": f"A shirt costs {price} euros. It is discounted by {discount}%. What is the sale price? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(final_price)]
    mid += 1

# pages read
for per_day, days in [(18, 7), (12, 5), (20, 6), (15, 8)]:
    total = per_day * days
    pid = f"math_word_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_word",
        "prompt": f"Mike reads {per_day} pages per day for {days} days. How many pages does he read in total? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(total)]
    mid += 1

# eggs / cakes
for eggs_per, cakes in [(3, 5), (2, 6), (4, 3), (5, 4)]:
    total = eggs_per * cakes
    pid = f"math_word_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_word",
        "prompt": f"A recipe needs {eggs_per} eggs for 1 cake. How many eggs are needed for {cakes} cakes? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(total)]
    mid += 1

# -------------------------------------------------
# B. algebra
# -------------------------------------------------
for x in [2, 3, 4, 5, 6, 7, 8, 9]:
    for a, b in [(2, 3), (3, 5), (4, 1), (5, 2)]:
        c = a * x + b
        pid = f"math_algebra_{mid:04d}"
        rows.append({
            "id": pid,
            "task_type": "math_algebra",
            "prompt": f"Solve for x: {a}x + {b} = {c}. Show the steps briefly."
        })
        answer_key[pid] = [fmt_num(x), f"x={fmt_num(x)}", f"x = {fmt_num(x)}"]
        mid += 1

for x in [8, 12, 16, 20, 24]:
    for d in [2, 4]:
        val = x / d
        pid = f"math_algebra_{mid:04d}"
        rows.append({
            "id": pid,
            "task_type": "math_algebra",
            "prompt": f"Solve for x: x / {d} = {fmt_num(val)}. Show the steps briefly."
        })
        answer_key[pid] = [fmt_num(x), f"x={fmt_num(x)}", f"x = {fmt_num(x)}"]
        mid += 1

# y = ax + b
for x in [2, 3, 4, 5]:
    for a, b in [(2, 3), (3, 1), (4, 2)]:
        y = a * x + b
        pid = f"math_algebra_{mid:04d}"
        rows.append({
            "id": pid,
            "task_type": "math_algebra",
            "prompt": f"If y = {a}x + {b}, what is y when x = {x}? Show the steps briefly."
        })
        answer_key[pid] = [fmt_num(y), f"y={fmt_num(y)}", f"y = {fmt_num(y)}"]
        mid += 1

# -------------------------------------------------
# C. ratio / percentage
# -------------------------------------------------
for total, absent in [(20, 5), (24, 6), (30, 6), (40, 10), (50, 5), (60, 15)]:
    pct = absent / total * 100
    pid = f"math_ratio_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_ratio",
        "prompt": f"A class has {total} students, and {absent} of them are absent. What percentage of the class is absent? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(pct), f"{fmt_num(pct)}%"]
    mid += 1

for price, inc in [(50, 10), (40, 15), (80, 5), (120, 20), (30, 25)]:
    new_price = price * (1 + inc / 100)
    pid = f"math_ratio_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_ratio",
        "prompt": f"A product costs {price} euros and its price increases by {inc}%. What is the new price? Show the steps briefly."
    })
    answer_key[pid] = [fmt_num(new_price)]
    mid += 1

for total, tea in [(40, 30), (20, 15), (24, 18), (12, 9)]:
    g = math.gcd(tea, total)
    frac_num = tea // g
    frac_den = total // g
    pct = tea / total * 100
    pid = f"math_ratio_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_ratio",
        "prompt": f"In a group of {total} people, {tea} prefer tea. What fraction and percentage prefer tea? Show the steps briefly."
    })
    answer_key[pid] = [
        f"{frac_num}/{frac_den}",
        fmt_num(pct),
        f"{fmt_num(pct)}%",
        f"{frac_num}/{frac_den} and {fmt_num(pct)}%",
        f"{fmt_num(pct)}% and {frac_num}/{frac_den}",
    ]
    mid += 1

# -------------------------------------------------
# D. minimal stats (keep small)
# -------------------------------------------------
stats_lists = [
    [2, 4, 4, 4, 5, 5, 7, 9],
    [1, 2, 3, 4, 5],
    [10, 12, 14, 16, 18],
    [3, 3, 4, 5, 7, 8],
]

for arr in stats_lists:
    pid = f"math_stats_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_stats",
        "prompt": f"Given the list {arr}, compute the mean and explain the steps briefly."
    })
    answer_key[pid] = [fmt_num(sum(arr) / len(arr))]
    mid += 1

for arr in stats_lists[:3]:
    pid = f"math_stats_{mid:04d}"
    rows.append({
        "id": pid,
        "task_type": "math_stats",
        "prompt": f"Given the list {arr}, compute the median and explain the steps briefly."
    })
    answer_key[pid] = [fmt_num(statistics.median(arr))]
    mid += 1

# keep stats small
rng.shuffle(rows)
rows = rows[:120]

kept_ids = {row["id"] for row in rows}
answer_key = {k: v for k, v in answer_key.items() if k in kept_ids}

write_jsonl(os.path.join(OUT_DIR, "basic_math_prompts_v1_2.jsonl"), rows)
write_json(os.path.join(OUT_DIR, "basic_math_answer_key_v1_2.json"), answer_key)

print("Wrote:")
print(os.path.join(OUT_DIR, "basic_math_prompts_v1_2.jsonl"), len(rows))
print(os.path.join(OUT_DIR, "basic_math_answer_key_v1_2.json"), len(answer_key))
