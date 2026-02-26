#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_first_number(text: str):
    nums = _NUM_RE.findall((text or "").strip())
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def arithmetic_correct(gen: str, gold: str) -> bool:
    # gold is an integer string like "5"
    pred = extract_first_number(gen)
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
        g = g[idx + len(signature.strip()) :]

    # handle fenced block
    g = g.replace("```python", "```").replace("```py", "```")
    if "```" in g:
        parts = g.split("```")
        if len(parts) >= 3:
            g = max(parts[1::2], key=len)

    lines = g.splitlines()
    kept: list[str] = []

    STOP_PREFIXES = (
        "if __name__",
        "class ",
        "def ",
        "import ",
        "from ",
        "assert ",
        "print(",
        "input(",
        "end",
    )

    for line in lines:
        line = line.rstrip()
        # Stop if model starts a new top-level program structure.
        s = line.lstrip()
        if kept and s.startswith(STOP_PREFIXES):
            break
        kept.append(line)

    body = "\n".join(kept).strip("\n")
    return body if body.strip() else "pass"


_COLON_BLOCK_RE = re.compile(
    r"^\s*(if|for|while|elif|else|try|except|finally|with|match|case)\b.*:\s*$"
)
_DEF_CLASS_RE = re.compile(r"^\s*(def|class)\b.*:\s*$")


def normalize_body_indent(body: str) -> list[str]:
    """
    Return a list of lines with *relative* indentation fixed:
    - tabs -> 4 spaces
    - strip trailing whitespace
    - repair missing indentation after ':' block starters by forcing next non-empty line to be +4
    """
    s = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", "    ")
    lines = [ln.rstrip() for ln in s.splitlines()]

    out = []
    force_next_indent = None  # (base_indent)
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue

        # count current leading spaces
        cur_indent = len(ln) - len(ln.lstrip(" "))
        txt = ln.lstrip(" ")

        # If previous line started a block, ensure this line is indented deeper.
        if force_next_indent is not None and cur_indent <= force_next_indent:
            ln = (" " * (force_next_indent + 4)) + txt
            cur_indent = force_next_indent + 4
            force_next_indent = None
        else:
            force_next_indent = None

        out.append(ln)

        # If this line starts a block, require next non-empty line to indent deeper
        if _COLON_BLOCK_RE.match(ln) or _DEF_CLASS_RE.match(ln):
            base = len(ln) - len(ln.lstrip(" "))
            force_next_indent = base

    return out


def indent_as_function_body(body: str) -> str:
    s = (body or "").replace("\t", "    ").strip("\n")
    if not s.strip():
        return "    pass"

    # 1) normalize indentation to multiples of 4 (floor), keep relative structure
    raw = []
    for ln in s.splitlines():
        ln = ln.rstrip()
        if not ln.strip():
            raw.append("")
            continue
        lead = len(ln) - len(ln.lstrip(" "))
        lead2 = (lead // 4) * 4
        raw.append((" " * lead2) + ln.lstrip(" "))

    # 2) FIX unexpected-indent noise:
    # if current indent > prev indent but previous code line is NOT a block opener (no ':'),
    # then snap current indent back to prev indent.
    fixed = []
    prev_indent = 0
    prev_ends_colon = False
    for ln in raw:
        if not ln.strip():
            fixed.append("")
            continue
        cur_indent = len(ln) - len(ln.lstrip(" "))
        txt = ln.lstrip(" ")

        if (cur_indent > prev_indent) and (not prev_ends_colon):
            cur_indent = prev_indent  # snap back

        fixed_ln = (" " * cur_indent) + txt
        fixed.append(fixed_ln)

        prev_indent = cur_indent
        prev_ends_colon = fixed_ln.rstrip().endswith(":")

    # 3) indent into function body
    return "\n".join(("    " + ln) if ln.strip() else "" for ln in fixed)


def run_python_tests(
    signature: str, body: str, tests: str, timeout_s: float = 2.0
) -> tuple[bool, str]:
    prog = []
    prog.append(signature.rstrip())
    prog.append(indent_as_function_body(body))
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
        if r.returncode != 0:
            snippet = "\n".join(code.splitlines()[:60])
            err = (r.stderr or r.stdout or "FAILED").strip()[:1200]
            return (False, err + "\n\n[CODE_SNIPPET]\n" + snippet)
        if r.returncode == 0:
            return (True, "")
        return (False, (r.stderr or r.stdout or "FAILED").strip()[:2000])


def generate_one(prompt: str, task: str, args, seed: int) -> str:
    cmd = [
        sys.executable,
        "pretrain/sample.py",
        "--ckpt",
        args.ckpt,
        "--tokenizer_path",
        args.tokenizer_path,
        "--precision",
        args.precision,
        "--max_seq_len",
        str(args.max_seq_len),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--min_new_tokens",
        str(args.min_new_tokens),
        "--seed",
        str(seed),
        "--prompt",
        prompt,
        "--quiet",
        "--repetition_penalty",
        str(args.repetition_penalty),
        "--no_repeat_ngram_size",
        str(args.no_repeat_ngram_size),
        "--max_repeat_token",
        str(args.max_repeat_token),
    ]

    # task-specific decoding + stopping
    if task == "arithmetic":
        # deterministic, short
        cmd += ["--greedy", "--temperature", "0"]
        # clamp length
        cmd[cmd.index("--max_new_tokens") + 1] = str(min(int(args.max_new_tokens), 128))
        cmd[cmd.index("--min_new_tokens") + 1] = "1"
        # turn off sampling guards that can induce weird artifacts
        cmd[cmd.index("--repetition_penalty") + 1] = "1.0"
        cmd[cmd.index("--no_repeat_ngram_size") + 1] = "0"
        cmd[cmd.index("--max_repeat_token") + 1] = "0"
        # stop when we have a number followed by a delimiter
        cmd += ["--stop_regex", r"[-+]?\d+(?:\.\d+)?(?=(?:\s|$|[^\d]))"]
        cmd += ["--stop_on_newline"]
    elif task == "syllogism":
        cmd += ["--greedy", "--temperature", "0"]
        # stop early; then we will postprocess to first token
        cmd += ["--stop_on_newline"]
    elif task == "code":
        if args.greedy:
            cmd += ["--greedy", "--temperature", "0"]
        # stop once we reach the blank line that ends the function body (your synth answers end with \n\n)
        cmd += ["--stop_regex", r"(?s)\nreturn[^\n]*\n"]
    else:
        if args.greedy:
            cmd += ["--greedy", "--temperature", "0"]

    if args.avoid_first_whitespace:
        cmd += ["--avoid_first_whitespace", "--ban_first_steps", str(args.ban_first_steps)]
    return subprocess.check_output(cmd, text=True)


_SYLL_RE = re.compile(r"\b(yes|no|unknown)\b", re.I)


def normalize_syllogism(gen: str) -> str:
    m = _SYLL_RE.search(gen or "")
    return m.group(1).lower() if m else "unknown"


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
            "- Output ONLY the function body (no def line).\n"
            "- Do NOT use 'yield'.\n"
            "- Do NOT use semicolons ';'.\n"
            "- Write exactly ONE statement per line (no two statements on one line).\n"
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
    ap.add_argument("--min_new_tokens", type=int, default=1)

    ap.add_argument("--seed_base", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=2)
    ap.add_argument("--max_repeat_token", type=int, default=2)
    ap.add_argument("--avoid_first_whitespace", action="store_true")
    ap.add_argument("--ban_first_steps", type=int, default=4)

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
        gen = generate_one(prompt, task, args, seed=args.seed_base + i + 1)

        ok = False
        detail = ""

        if task == "arithmetic":
            stats["arithmetic"]["n"] += 1
            pred = extract_first_number(gen)
            ok = arithmetic_correct(gen, gold)
            if ok:
                stats["arithmetic"]["correct"] += 1
            detail = f"pred={pred} gold={gold}"

        elif task == "syllogism":
            stats["syllogism"]["n"] += 1
            pred = normalize_syllogism(gen)
            ok = pred == gold
            if ok:
                stats["syllogism"]["correct"] += 1
            detail = f"pred={pred} gold={gold}"

        elif task == "code":
            stats["code"]["n"] += 1
            signature = it["signature"]
            tests = it["tests"]
            body = try_extract_code_body(gen, signature)
            # remove any accidental leading indentation from the model
            ok, err = run_python_tests(signature, body, tests, timeout_s=2.0)
            if ok:
                stats["code"]["correct"] += 1
            detail = err

        else:
            detail = f"Unknown task={task}"

        stats["total"] += 1
        results.append(
            {
                "id": _id,
                "task": task,
                "ok": ok,
                "prompt": prompt,
                "gold": gold,
                "gen": gen,
                "detail": detail,
            }
        )

    def acc(d):
        return (d["correct"] / d["n"]) if d["n"] else 0.0

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

    print(
        json.dumps({k: summary[k] for k in summary if k != "results"}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
