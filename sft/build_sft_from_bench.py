#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, os, random, re
from typing import Any, Dict, List

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def mk_messages_arith(prompt: str, answer: str) -> List[Dict[str, str]]:
    # Force short answer: one number only.
    sys = (
        "You are a precise assistant.\n"
        "Rules for arithmetic questions:\n"
        "- Output ONLY the final numeric answer.\n"
        "- No explanation, no extra words, no punctuation.\n"
    )
    return [
        {"role":"system","content":sys},
        {"role":"user","content":prompt.strip()},
        {"role":"assistant","content":str(answer).strip()},
    ]

def mk_messages_syll(prompt: str, answer: str) -> List[Dict[str, str]]:
    sys = (
        "You are a logic assistant.\n"
        "Rules:\n"
        "- Output ONLY one token: yes / no / unknown.\n"
        "- Do not explain.\n"
    )
    ans = answer.strip().lower()
    assert ans in ("yes","no","unknown")
    return [
        {"role":"system","content":sys},
        {"role":"user","content":prompt.strip()},
        {"role":"assistant","content":ans},
    ]

def mk_messages_code(signature: str, tests: str, entry: str | None = None) -> List[Dict[str, str]]:
    # We supervise "function only" (not main / not extra funcs).
    # We'll ask for complete function definition (signature+body),
    # because your eval extracts body and indents it into signature anyway.
    sys = (
        "You are a coding assistant.\n"
        "Rules:\n"
        "- Output ONLY Python code.\n"
        "- Do NOT write any explanation.\n"
        "- Do NOT write extra functions.\n"
        "- Do NOT write a main block.\n"
        "- Implement exactly the requested function.\n"
    )
    user = (
        "Implement the following Python function.\n"
        "Only output the function implementation.\n\n"
        f"{signature.rstrip()}\n"
    )

    # Minimal reference solutions (tiny, but correct)
    # You can expand later with more tasks/examples.
    sig = signature.strip()
    if sig.startswith("def add"):
        assistant = "def add(a, b):\n    return a + b\n"
    elif sig.startswith("def factorial"):
        assistant = (
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    out = 1\n"
            "    for k in range(2, n + 1):\n"
            "        out *= k\n"
            "    return out\n"
        )
    elif sig.startswith("def fib"):
        assistant = (
            "def fib(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        )
    else:
        # fallback: a safe placeholder (won't be used if you keep your 3 code tasks)
        assistant = signature + "    raise NotImplementedError\n"

    return [
        {"role":"system","content":sys},
        {"role":"user","content":user},
        {"role":"assistant","content":assistant.rstrip()},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dup", type=int, default=50, help="duplicate each item to make a small-but-not-tiny SFT set")
    args = ap.parse_args()

    random.seed(args.seed)
    items = read_jsonl(args.bench)

    rows: List[Dict[str, Any]] = []
    for it in items:
        task = it["task"]
        if task == "arithmetic":
            msgs = mk_messages_arith(it["prompt"], it["answer"])
        elif task == "syllogism":
            msgs = mk_messages_syll(it["prompt"], it["answer"])
        elif task == "code":
            msgs = mk_messages_code(it["signature"], it["tests"], it.get("entry"))
        else:
            continue

        # duplicate to stabilize early SFT (cheap trick)
        for _ in range(max(1, args.dup)):
            rows.append({"messages": msgs, "meta": {"id": it.get("id"), "task": task}})

    random.shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} examples to {args.out}")

if __name__ == "__main__":
    main()
