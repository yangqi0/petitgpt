#!/usr/bin/env python3
"""Render or compare training curves from metrics.jsonl files.

Single run (writes <run>/curves/*.png):
    python scripts/plot_metrics.py outputs/my_run

Compare runs (overlays each metric across runs):
    python scripts/plot_metrics.py outputs/run_a outputs/run_b --out eval/compare_ab

Accepts run directories (containing metrics.jsonl) or metrics files directly.
Legacy distill logs (logs/train_loss.jsonl, no "kind" field) also work.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking import render_comparison, render_curves  # noqa: E402


def resolve_metrics_path(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        for cand in (p / "metrics.jsonl", p / "logs" / "train_loss.jsonl"):
            if cand.exists():
                return cand
        raise SystemExit(f"no metrics.jsonl found under {p}")
    if not p.exists():
        raise SystemExit(f"not found: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run dirs or metrics.jsonl files")
    ap.add_argument(
        "--out", default="", help="output dir (default: <run>/curves or ./curves_compare)"
    )
    ap.add_argument("--fields", default="", help="comma-separated fields to compare (default: all)")
    args = ap.parse_args()

    paths = [resolve_metrics_path(r) for r in args.runs]

    if len(paths) == 1:
        out = Path(args.out) if args.out else paths[0].parent / "curves"
        written = render_curves(paths[0], out)
    else:
        labels: dict[str, Path] = {}
        for p in paths:
            label = p.parent.name if p.name == "metrics.jsonl" else f"{p.parent.name}/{p.stem}"
            base, i = label, 2
            while label in labels:
                label, i = f"{base}#{i}", i + 1
            labels[label] = p
        out = Path(args.out) if args.out else Path("curves_compare")
        fields = [f for f in args.fields.split(",") if f] or None
        written = render_comparison(labels, out, fields=fields)

    for w in written:
        print(w)
    if not written:
        print("no numeric metrics found", file=sys.stderr)


if __name__ == "__main__":
    main()
