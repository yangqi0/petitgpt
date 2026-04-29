from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_input_path"] = str(path)
            row["_input_lineno"] = lineno
            rows.append(row)
    return rows

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

def normalize_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def normalize_prompt_key(s: str) -> str:
    s = normalize_space(s).lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()

def stable_id(prefix: str, *parts: str, length: int = 12) -> str:
    h = hashlib.md5("||".join([prefix] + [str(p) for p in parts]).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:length]}"

def coerce_tests(row: Dict[str, Any]) -> Any:
    for key in ["tests", "test_list", "unit_tests", "hidden_tests", "assertions"]:
        if key in row:
            return row[key]
    return None

def infer_source(row: Dict[str, Any], meta: Dict[str, Any]) -> str:
    source = normalize_space(row.get("source") or row.get("dataset") or meta.get("source") or "")
    path = str(row.get("_input_path", "")).lower()
    if source:
        return source
    if "mbpp" in path:
        return "mbpp"
    if "core_family" in path or "core" in path:
        return "core_family"
    return ""

def infer_family(row: Dict[str, Any], meta: Dict[str, Any], source: str) -> str:
    family = normalize_space(
        row.get("family")
        or row.get("task_family")
        or row.get("category")
        or meta.get("family")
        or meta.get("task_family")
        or ""
    )
    if family:
        return family
    src = source.lower()
    path = str(row.get("_input_path", "")).lower()
    if "mbpp" in src or "mbpp" in path:
        return "mbpp_simple"
    if "core_family" in src or "core_family" in path:
        return "core_family"
    return ""

def infer_subfamily(row: Dict[str, Any], meta: Dict[str, Any], family: str, source: str) -> str:
    subfamily = normalize_space(
        row.get("subfamily")
        or row.get("sub_family")
        or row.get("task")
        or row.get("task_name")
        or row.get("task_family")
        or row.get("category")
        or meta.get("subfamily")
        or meta.get("sub_family")
        or meta.get("task")
        or meta.get("task_name")
        or meta.get("task_family")
        or ""
    )
    if subfamily:
        return subfamily

    # For core-family prompts, the family itself is often already the fine label.
    if family and family not in {"core_family", "mbpp_simple"}:
        return family

    src = source.lower()
    path = str(row.get("_input_path", "")).lower()
    if "mbpp" in src or "mbpp" in path:
        return "mbpp"
    if "core_family" in src or "core_family" in path:
        # Try entry point as a useful subfamily fallback.
        ep = normalize_space(row.get("entry_point") or row.get("fn_name") or meta.get("entry_point") or "")
        return ep or "core_family"

    return ""

def normalize_row(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    prompt = row.get("canonical_prompt") or row.get("prompt") or row.get("text") or row.get("question")
    prompt = normalize_space(prompt)
    if not prompt:
        return None, "missing_canonical_prompt"

    meta_in = row.get("meta") if isinstance(row.get("meta"), dict) else {}

    source = infer_source(row, meta_in)
    if not source:
        source = args.fill_missing_source
    if not source:
        return None, "missing_source"

    family = infer_family(row, meta_in, source)
    if not family:
        family = args.fill_missing_family
    if not family:
        return None, "missing_family"

    subfamily = infer_subfamily(row, meta_in, family, source)
    if not subfamily:
        subfamily = args.fill_missing_subfamily or family
    if not subfamily:
        return None, "missing_subfamily"

    entry_point = normalize_space(
        row.get("entry_point")
        or row.get("fn_name")
        or row.get("function_name")
        or row.get("target_function")
        or meta_in.get("entry_point")
        or ""
    )

    tests = coerce_tests(row)
    constraints = row.get("constraints") if isinstance(row.get("constraints"), dict) else {}

    if args.require_entry_point and not entry_point:
        return None, "missing_entry_point"

    if args.require_tests and (tests is None or tests == [] or tests == ""):
        return None, "missing_tests"

    out: Dict[str, Any] = {
        "id": normalize_space(row.get("id") or stable_id("code", source, family, subfamily, entry_point, prompt[:200])),
        "source": source,
        "family": family,
        "subfamily": subfamily,
        "canonical_prompt": prompt,
        "entry_point": entry_point,
        "constraints": constraints,
        "meta": {
            **meta_in,
            "input_path": row.get("_input_path"),
            "input_lineno": row.get("_input_lineno"),
        },
    }

    if tests is not None:
        out["tests"] = tests

    for k in [
        "mbpp_task_id", "task_id", "difficulty", "raw_prompt",
        "reference_code", "starter_code", "code", "test_list"
    ]:
        if k in row and k not in out:
            out[k] = row[k]

    return out, None

def dedup_rows(rows: List[Dict[str, Any]], mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen = {}
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for row in rows:
        prompt_key = normalize_prompt_key(row["canonical_prompt"])
        entry = normalize_space(row.get("entry_point", "")).lower()

        if mode == "prompt":
            key = (prompt_key,)
        elif mode == "entry_prompt":
            key = (entry, prompt_key)
        elif mode == "source_entry_prompt":
            key = (row.get("source", ""), entry, prompt_key)
        else:
            raise ValueError(f"Unknown dedup mode: {mode}")

        if key in seen:
            row["_reject_reason"] = "duplicate"
            row["_duplicate_of"] = seen[key]
            dropped.append(row)
        else:
            seen[key] = row["id"]
            kept.append(row)

    return kept, dropped

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--raw_out_jsonl", default="")
    ap.add_argument("--reject_jsonl", default="")
    ap.add_argument("--dedup_mode", choices=["prompt", "entry_prompt", "source_entry_prompt"], default="entry_prompt")
    ap.add_argument("--require_entry_point", action="store_true")
    ap.add_argument("--require_tests", action="store_true")
    ap.add_argument("--fill_missing_source", default="")
    ap.add_argument("--fill_missing_family", default="")
    ap.add_argument("--fill_missing_subfamily", default="")
    args = ap.parse_args()

    raw_in = []
    for p in args.inputs:
        rows = read_jsonl(p)
        print(f"[read] {p}: {len(rows)}")
        raw_in.extend(rows)

    normalized: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reject_counter = Counter()

    for row in raw_in:
        out, reason = normalize_row(row, args)
        if reason:
            bad = dict(row)
            bad["_reject_reason"] = reason
            rejected.append(bad)
            reject_counter[reason] += 1
        else:
            normalized.append(out)

    if args.raw_out_jsonl:
        write_jsonl(args.raw_out_jsonl, normalized)

    kept, dupes = dedup_rows(normalized, args.dedup_mode)
    for d in dupes:
        reject_counter["duplicate"] += 1
    rejected.extend(dupes)

    write_jsonl(args.out_jsonl, kept)

    if args.reject_jsonl:
        write_jsonl(args.reject_jsonl, rejected)

    print("\n[summary]")
    print(f"input_rows      = {len(raw_in)}")
    print(f"normalized_rows = {len(normalized)}")
    print(f"kept_rows       = {len(kept)}")
    print(f"rejected_rows   = {len(rejected)}")
    print(f"dedup_mode      = {args.dedup_mode}")

    print("\n[reject_reasons]")
    for k, v in reject_counter.most_common():
        print(f"  {k}: {v}")

    print("\n[source_counts]")
    for k, v in Counter(r.get("source", "") for r in kept).most_common():
        print(f"  {k}: {v}")

    print("\n[family_counts]")
    for k, v in Counter(r.get("family", "") for r in kept).most_common():
        print(f"  {k}: {v}")

    print("\n[subfamily_counts_top30]")
    for k, v in Counter(r.get("subfamily", "") for r in kept).most_common(30):
        print(f"  {k}: {v}")

    missing_entry = sum(1 for r in kept if not r.get("entry_point"))
    missing_tests = sum(1 for r in kept if "tests" not in r or r.get("tests") in (None, [], ""))
    print("\n[health]")
    print(f"missing_entry_point_in_kept = {missing_entry}")
    print(f"missing_tests_in_kept       = {missing_tests}")

if __name__ == "__main__":
    main()
