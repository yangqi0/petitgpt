#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from tqdm import tqdm

# 强非 Python 指纹：一旦命中就丢
NON_PY_NEEDLES = [
    "using System", "namespace ", "public class", "private ", "protected ",
    "static void main", "console.log(", "function(", "package ",
    "#include ", "#define ", "BEGIN CERTIFICATE", "<!DOCTYPE html", "<html", "<?xml",
    "SELECT ", "CREATE TABLE", "echo ", "REM ", "setlocal",
]

# 强 Python 指纹：至少命中其一才留（你可以按需要增删）
PY_SIGNAL_RE = [
    re.compile(r"^\s*def\s+\w+\s*\(", re.M),
    re.compile(r"^\s*class\s+\w+\s*[:(]", re.M),
    re.compile(r"^\s*(from|import)\s+\w+", re.M),
    re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]"),
]

def looks_like_python(t: str) -> bool:
    s = t.lstrip()
    if s.startswith("#!/usr/bin/env python") or s.startswith("#!/usr/bin/python"):
        return True
    for x in NON_PY_NEEDLES:
        if x in t:
            return False
    hits = sum(1 for r in PY_SIGNAL_RE if r.search(t) is not None)
    if hits >= 1:
        # 再做一个“反 C#/JS”保险：花括号/分号过多通常不是 Python
        if (t.count("{") + t.count("}")) > 6:
            return False
        if t.count(";") > 10:
            return False
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_docs", type=int, default=0, help="0=all")
    args = ap.parse_args()

    inp = Path(args.in_jsonl)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = drop = bad = 0
    with inp.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in tqdm(f, desc=str(inp.name)):
            try:
                obj = json.loads(line)
                t = obj.get("text", "")
                if not isinstance(t, str) or not t.strip():
                    drop += 1
                    continue
            except Exception:
                bad += 1
                continue

            if looks_like_python(t):
                g.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                kept += 1
                if args.max_docs and kept >= args.max_docs:
                    break
            else:
                drop += 1

    print({"kept": kept, "drop": drop, "bad_json": bad, "out": str(out)})

if __name__ == "__main__":
    main()
