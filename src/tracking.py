"""Lightweight experiment tracking shared by all training scripts.

Metrics are appended to <out_dir>/metrics.jsonl, one JSON row per event:

    {"kind": "train", "step": 120, "loss": 3.21, "lr": 5e-5, "time": ...}

`kind` groups rows into series ("train"/"val"/"bench"/...). Curves are rendered
to <out_dir>/curves/*.png with matplotlib (Agg backend); rendering is
best-effort and never raises into a training loop. Replot or compare runs any
time with scripts/plot_metrics.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

# Row keys that are structure, not metrics.
_SKIP_FIELDS = {"kind", "step", "time", "train_jsonl", "val_jsonl", "loss_reduction", "ckpt"}

# Fields drawn first in the overview grid (everything else follows alphabetically).
_PRIORITY_FIELDS = ["loss", "val_loss", "reward", "reward_margin", "reward_acc", "kl", "lr"]


class Tracker:
    """Append metrics to <out_dir>/metrics.jsonl and render curves on demand."""

    def __init__(self, out_dir: str | Path, filename: str = "metrics.jsonl"):
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / filename
        self.curves_dir = self.out_dir / "curves"

    def log(self, kind: str, step: int, **values) -> None:
        row: dict = {"kind": str(kind), "step": int(step)}
        for k, v in values.items():
            if v is None or isinstance(v, (bool, int, str)):
                row[k] = v
            else:
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = str(v)
        row["time"] = time.time()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def render(self) -> None:
        """Best-effort curve rendering; never raises into the training loop."""
        try:
            render_curves(self.path, self.curves_dir)
        except Exception as e:
            print(f"[tracking] curve render failed: {e}")


def read_metrics(path: str | Path) -> list[dict]:
    """Read a metrics.jsonl file. Rows without 'kind' (legacy distill logs) are
    classified as 'val' if they carry val_loss, else 'train'."""
    rows: list[dict] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "step" not in row:
                continue
            row.setdefault("kind", "val" if "val_loss" in row else "train")
            rows.append(row)
    return rows


def _series(rows: list[dict]) -> dict[str, dict[str, tuple[list[int], list[float]]]]:
    """{field: {kind: (steps, values)}} over all numeric metric fields."""
    out: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
    for row in sorted(rows, key=lambda r: (int(r["step"]), float(r.get("time", 0.0)))):
        kind = str(row.get("kind", "train"))
        for k, v in row.items():
            if k in _SKIP_FIELDS or isinstance(v, (bool, str)) or v is None:
                continue
            steps, vals = out.setdefault(k, {}).setdefault(kind, ([], []))
            steps.append(int(row["step"]))
            vals.append(float(v))
    return out


def _ema(values: list[float], beta: float = 0.9) -> list[float]:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else beta * acc + (1.0 - beta) * v
        out.append(acc)
    return out


def _ordered_fields(series: dict) -> list[str]:
    rest = sorted(f for f in series if f not in _PRIORITY_FIELDS)
    return [f for f in _PRIORITY_FIELDS if f in series] + rest


def render_curves(metrics_path: str | Path, curves_dir: str | Path) -> list[Path]:
    """Render one PNG per metric field plus an overview grid. Returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_metrics(metrics_path)
    series = _series(rows)
    if not series:
        return []

    curves_dir = Path(curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fields = _ordered_fields(series)

    def draw(ax, field: str) -> None:
        for kind, (steps, vals) in sorted(series[field].items()):
            if len(vals) > 200:
                ax.plot(steps, vals, alpha=0.25, lw=0.8)
                ax.plot(steps, _ema(vals), label=kind, lw=1.4)
            else:
                marker = "o" if len(vals) <= 60 else None
                ax.plot(steps, vals, label=kind, lw=1.4, marker=marker, ms=3)
        ax.set_title(field)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if len(series[field]) > 1:
            ax.legend(fontsize=8)

    for field in fields:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        draw(ax, field)
        fig.tight_layout()
        out = curves_dir / f"{field}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)

    ncols = 3
    nrows = (len(fields) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 3.2), squeeze=False)
    for i, field in enumerate(fields):
        draw(axes[i // ncols][i % ncols], field)
    for j in range(len(fields), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(str(Path(metrics_path).parent.name))
    fig.tight_layout()
    out = curves_dir / "overview.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    written.append(out)
    return written


def render_comparison(
    runs: dict[str, str | Path], out_dir: str | Path, fields: list[str] | None = None
) -> list[Path]:
    """Overlay the same metric across runs: {label: metrics.jsonl path} -> PNGs.

    For each field, the 'val' series is preferred (less noisy); falls back to
    whichever kinds a run has, labelled "<run>/<kind>" when ambiguous.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_run = {label: _series(read_metrics(p)) for label, p in runs.items()}
    all_fields = sorted({f for s in per_run.values() for f in s})
    fields = [
        f for f in (fields or _ordered_fields({f: None for f in all_fields})) if f in all_fields
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for field in fields:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for label, series in per_run.items():
            kinds = series.get(field, {})
            if not kinds:
                continue
            chosen = {"val": kinds["val"]} if "val" in kinds else kinds
            for kind, (steps, vals) in sorted(chosen.items()):
                name = label if len(chosen) == 1 else f"{label}/{kind}"
                if len(vals) > 200:
                    ax.plot(steps, _ema(vals), label=name, lw=1.4)
                else:
                    ax.plot(steps, vals, label=name, lw=1.4)
        ax.set_title(field)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = out_dir / f"compare_{field}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
    return written
