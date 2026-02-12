#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, itertools
from collections import Counter

def head_class(s: str) -> str:
    t = s.lstrip()
    if not t: return "empty"
    if t.startswith("____") or t.startswith("__"): return "underscores"
    if t.startswith("||"): return "pipes||"
    if t.startswith("|"): return "pipes|"
    if t.startswith("- "): return "dash_list"
    if t.startswith("* "): return "star_list"
    if t.startswith("#"): return "heading#"
    if t.startswith(">"): return "blockquote>"
    if t[0].isdigit(): return "digit"
    if t[0].isalpha(): return "alpha"
    return "other_symbol"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--n", type=int, default=500000)
    args = ap.parse_args()

    c = Counter()
    for line in itertools.islice(open(args.jsonl,"r",encoding="utf-8"), args.n):
        try:
            obj=json.loads(line)
            t=obj.get(args.text_field,"")
        except Exception:
            continue
        if not isinstance(t,str):
            continue
        c[head_class(t)] += 1

    total = sum(c.values())
    for k,v in c.most_common():
        print(f"{k:14s}  {v:10d}  {v/total:8.3%}")
    print("TOTAL", total)

if __name__ == "__main__":
    main()
