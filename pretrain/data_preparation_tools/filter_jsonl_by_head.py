#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from tqdm import tqdm

RE_MD_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
RE_MD_TABLE = re.compile(r"^\s*\|.+\|\s*$")
RE_DASH_LIST = re.compile(r"^\s*-\s+\S")
RE_STAR_LIST = re.compile(r"^\s*\*\s+\S")
RE_BLOCKQUOTE = re.compile(r"^\s*>\s+\S")

def first_nonempty_line(t: str) -> str:
    for ln in t.split("\n"):
        if ln.strip():
            return ln.strip()
    return ""

def head_label(t: str) -> str:
    ln = first_nonempty_line(t)
    if not ln:
        return "empty"
    if RE_MD_TABLE.match(ln) or ln.startswith("|"):
        return "pipes_table"
    if RE_MD_HEADING.match(ln):
        return "md_heading"
    if RE_DASH_LIST.match(ln):
        return "dash_list"
    if RE_STAR_LIST.match(ln):
        return "star_list"
    if RE_BLOCKQUOTE.match(ln):
        return "blockquote"
    if ln.startswith("_"):
        return "underscores"
    if ln[0].isdigit():
        return "digit"
    if ln[0].isalpha():
        return "alpha"
    return "other_symbol"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--drop", default="pipes_table,md_heading,dash_list,star_list",
                    help="comma-separated labels to drop")
    ap.add_argument("--max_docs", type=int, default=0)
    args = ap.parse_args()

    drop_set = {x.strip() for x in args.drop.split(",") if x.strip()}

    inp = Path(args.in_jsonl)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    seen = kept = 0
    stats = {}

    with inp.open("r", encoding="utf-8") as fi, outp.open("w", encoding="utf-8") as fo:
        for line in tqdm(fi, desc=inp.name):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            t = ex.get("text", "")
            if not isinstance(t, str) or not t.strip():
                continue

            lab = head_label(t)
            stats[lab] = stats.get(lab, 0) + 1
            seen += 1

            if lab in drop_set:
                pass
            else:
                fo.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                kept += 1

            if args.max_docs and seen >= args.max_docs:
                break

    print("seen:", seen, "kept:", kept, "drop:", seen-kept)
    print("head_label_stats:", json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
