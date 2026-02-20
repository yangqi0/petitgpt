#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_last_number(text: str) -> Optional[float]:
    nums = _NUM_RE.findall(text or "")
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None

def arithmetic_correct(gen: str, gold: str) -> bool:
    # gold is an integer string like "5"
    pred = extract_last_number(gen)
    if pred is None:
        return False
    try:
        g = float(gold)
    except Exception:
        return False
    # allow exact int match, tolerate "5.0"
    return abs(pred - g) < 1e-6

def normalize_yes_no_unknown(text: str) -> str:
    t = (text or "").strip().lower()

    # explicit "unknown"
    if "cannot conclude" in t or "can't conclude" in t or "not necessarily" in t:
        return "unknown"
    if "insufficient" in t or "cannot be determined" in t or "unknown" in t:
        return "unknown"

    # yes/no
    if re.search(r"\b(yes|true|we can conclude|therefore)\b", t):
        return "yes"
    if re.search(r"\b(no|false)\b", t):
        return "no"

    # fallback by first char
    if t[:1] == "y":
        return "yes"
    if t[:1] == "n":
        return "no"
    return "unknown"


def try_extract_code_body(generation: str, signature: str) -> str:
    g = generation or ""

    # if it repeats signature, cut after last occurrence
    idx = g.rfind(signature.strip())
    if idx != -1:
        g = g[idx + len(signature.strip()):]

    # handle fenced block
    g = g.replace("```python", "```").replace("```py", "```")
    if "```" in g:
        parts = g.split("```")
        if len(parts) >= 3:
            g = max(parts[1::2], key=len)

    lines = g.splitlines()
    kept: List[str] = []
    for line in lines:
        # stop when it starts a new top-level thing (often storytelling)
        if kept and line and (not line.startswith((" ", "\t"))):
            break
        kept.append(line)

    body = "\n".join(kept).rstrip()
    return body if body.strip() else "pass"


def run_python_tests(signature: str, body: str, tests: str, timeout_s: float = 2.0) -> Tuple[bool, str]:
    prog = []
    prog.append(signature.rstrip())

    fixed = []
    for ln in (body.splitlines() or ["pass"]):
        if ln.strip() == "":
            fixed.append("")
        elif ln.startswith((" ", "\t")):
            fixed.append(ln)
        else:
            fixed.append("    " + ln)
    prog.append("\n".join(fixed))
    prog.append("")
    prog.append(tests.rstrip())
    code = "\n".join(prog) + "\n"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tmp_eval.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            r = subprocess.run(
                [sys.executable, path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return (False, "TIMEOUT")
        if r.returncode == 0:
            return (True, "")
        return (False, (r.stderr or r.stdout or "FAILED").strip()[:2000])


def generate_one(prompt: str, args, seed: int) -> str:
    # Call your CURRENT pretrain/sample.py as a black-box CLI
    cmd = [
        sys.executable, "pretrain/sample.py",
        "--ckpt", args.ckpt,
        "--tokenizer_path", args.tokenizer_path,
        "--precision", args.precision,
        "--max_seq_len", str(args.max_seq_len),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--max_new_tokens", str(args.max_new_tokens),
        "--min_new_tokens", str(args.min_new_tokens),
        "--seed", str(seed),
        "--prompt", prompt,
        "--quiet",
        "--repetition_penalty", str(args.repetition_penalty),
        "--no_repeat_ngram_size", str(args.no_repeat_ngram_size),
        "--max_repeat_token", str(args.max_repeat_token),
    ]
    if args.greedy:
        cmd += ["--greedy", "--temperature", "0"]

    cmd += ["--repetition_penalty", str(args.repetition_penalty)]
    cmd += ["--no_repeat_ngram_size", str(args.no_repeat_ngram_size)]
    cmd += ["--max_repeat_token", str(args.max_repeat_token)]

    if args.avoid_first_whitespace:
        cmd += ["--avoid_first_whitespace", "--ban_first_steps", str(args.ban_first_steps)]

    out = subprocess.check_output(cmd, text=True)
    return out

def wrap_with_task_rules(task: str, user_prompt: str) -> str:
    if task == "arithmetic":
        sys = (
            "You are a precise assistant.\n"
            "Rules for arithmetic questions:\n"
            "- Output ONLY the final numeric answer.\n"
            "- No explanation, no extra words, no punctuation.\n"
        )
    elif task == "syllogism":
        sys = (
            "You are a logic assistant.\n"
            "Rules:\n"
            "- Output ONLY one token: yes / no / unknown.\n"
            "- Do not explain.\n"
        )
    elif task == "code":
        sys = (
            "You are a coding assistant.\n"
            "Rules:\n"
            "- Output ONLY Python code.\n"
            "- Do NOT write any explanation.\n"
            "- Do NOT write extra functions.\n"
            "- Do NOT write a main block.\n"
            "- Implement exactly the requested function.\n"
        )
    else:
        sys = ""
    if not sys:
        return user_prompt
    return sys + "\n" + user_prompt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--min_new_tokens", type=int, default=8)

    ap.add_argument("--seed_base", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=2)
    ap.add_argument("--max_repeat_token", type=int, default=2)
    ap.add_argument("--avoid_first_whitespace", action="store_true")
    ap.add_argument("--ban_first_steps", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")

    args = ap.parse_args()

    items = read_jsonl(args.bench)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    stats = {
        "total": 0,
        "arithmetic": {"n": 0, "correct": 0},
        "syllogism": {"n": 0, "correct": 0},
        "code": {"n": 0, "correct": 0},
    }
    results = []
    t0 = time.time()

    for i, it in enumerate(items):
        _id = it.get("id", f"row{i}")
        task = it["task"]
        gold = it.get("answer")

        # ---- build prompt (code tasks may not have 'prompt') ----
        prompt = it.get("prompt")
        if prompt is None and task == "code":
            sig = it["signature"]
            prompt = "Complete the following Python function:\n\n" + sig
        if prompt is None:
            raise KeyError(f"Missing prompt in item id={_id} keys={list(it.keys())}")

        # ---- generate FIRST ----
        prompt = wrap_with_task_rules(task, prompt)
        gen = generate_one(prompt, args, seed=args.seed_base + i + 1)

        ok = False
        detail = ""

        if task == "arithmetic":
            stats["arithmetic"]["n"] += 1
            pred = extract_last_number(gen)
            ok = arithmetic_correct(gen, gold)
            if ok:
                stats["arithmetic"]["correct"] += 1
            detail = f"pred={pred} gold={gold}"

        elif task == "syllogism":
            stats["syllogism"]["n"] += 1
            pred = normalize_yes_no_unknown(gen)
            ok = (pred == gold)
            if ok:
                stats["syllogism"]["correct"] += 1
            detail = f"pred={pred} gold={gold}"

        elif task == "code":
            stats["code"]["n"] += 1
            signature = it["signature"]
            tests = it["tests"]
            body = try_extract_code_body(gen, signature)
            ok, err = run_python_tests(signature, body, tests, timeout_s=2.0)
            if ok:
                stats["code"]["correct"] += 1
            detail = err

        else:
            detail = f"Unknown task={task}"

        stats["total"] += 1
        results.append({
            "id": _id,
            "task": task,
            "ok": ok,
            "prompt": prompt,
            "gold": gold,
            "gen": gen,
            "detail": detail,
        })

    def acc(d): return (d["correct"] / d["n"]) if d["n"] else 0.0

    summary = {
        "ckpt": args.ckpt,
        "bench": args.bench,
        "time_s": round(time.time() - t0, 3),
        "acc_arithmetic": acc(stats["arithmetic"]),
        "acc_syllogism": acc(stats["syllogism"]),
        "acc_code": acc(stats["code"]),
        "counts": stats,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: summary[k] for k in summary if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
