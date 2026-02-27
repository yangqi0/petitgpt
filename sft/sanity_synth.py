#!/usr/bin/env python3

from collections import Counter, defaultdict
import json
import re

TRAIN = "datasets/sft/synth_vNEW.train.jsonl"
VAL = "datasets/sft/synth_vNEW.val.jsonl"

EOC_LINE = "###EOC###"
EOC_BLOCK = "\n" + EOC_LINE + "\n"

_NUM_ONLY_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
_SYLL_ONLY_RE = re.compile(r"^\s*(yes|no|unknown)\s*$", re.IGNORECASE)


def load_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"{path}:{i} JSON parse error: {e}")
    return rows


def get_user_asst(row):
    msgs = row["messages"]
    user = None
    asst = None
    for m in msgs:
        if m["role"] == "user":
            user = m["content"]
        if m["role"] == "assistant":
            asst = m["content"]
    return user or "", asst or ""


def check_rows(rows, name: str, max_show: int = 5):
    task_ctr = Counter()
    kind_ctr = Counter()
    style_ctr = Counter()

    bad = defaultdict(list)

    for idx, r in enumerate(rows):
        meta = r.get("meta") or {}
        task = meta.get("task")
        task_ctr[task] += 1
        if task == "code":
            kind_ctr[meta.get("kind", "NA")] += 1
        if task == "syllogism":
            style_ctr[meta.get("style", "NA")] += 1
        if task == "arithmetic":
            style_ctr[meta.get("style", "NA")] += 1

        user, asst = get_user_asst(r)

        if task == "code":
            # Must end with exact EOC marker block
            if not asst.endswith(EOC_BLOCK):
                bad["code_missing_eoc"].append((idx, meta.get("id")))
            # Must contain return somewhere (cheap check)
            if "return" not in asst:
                bad["code_no_return"].append((idx, meta.get("id")))
            # EOC appears exactly once
            if asst.count(EOC_LINE) != 1:
                bad["code_eoc_count"].append((idx, meta.get("id")))
            # Ban obvious junk
            low = asst.lower()
            for s in ["import ", "from ", "class ", "def ", "if __name__", "yield", ";"]:
                if s in low:
                    bad["code_has_banned_substr"].append((idx, meta.get("id")))
                    break

        elif task == "arithmetic":
            # Assistant should be just a number + newline (generator might store trailing newline)
            a = asst.strip()
            if not _NUM_ONLY_RE.match(a):
                bad["arith_not_number_only"].append((idx, meta.get("id"), a[:80]))

        elif task == "syllogism":
            a = asst.strip()
            if not _SYLL_ONLY_RE.match(a):
                bad["syll_not_one_token"].append((idx, meta.get("id"), a[:80]))

        else:
            bad["unknown_task"].append((idx, meta.get("id"), task))

    print("=" * 80)
    print(f"{name}: n={len(rows)}")
    print("task:", dict(task_ctr))
    if kind_ctr:
        print("code kind:", dict(kind_ctr))
    if style_ctr:
        print("style:", dict(style_ctr))

    total_bad = sum(len(v) for v in bad.values())
    print("bad_total:", total_bad)
    for k, v in bad.items():
        if not v:
            continue
        print(f"  {k}: {len(v)}")
        for item in v[:max_show]:
            print("   ", item)


def main():
    train = load_jsonl(TRAIN)
    val = load_jsonl(VAL)
    check_rows(train, "TRAIN")
    check_rows(val, "VAL")


if __name__ == "__main__":
    main()
