"""Streaming, fail-before-GPU validation for post-training JSONL datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path
from typing import Any

RecordMetrics = Mapping[str, int | float]
RecordValidator = Callable[[str, dict[str, Any]], RecordMetrics | None]


def _record_error(
    errors: list[dict[str, Any]],
    *,
    max_errors: int,
    split: str,
    path: str,
    line: int,
    exc: BaseException,
) -> None:
    if len(errors) >= max_errors:
        return
    errors.append(
        {
            "split": split,
            "path": path,
            "line": line,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def run_jsonl_preflight(
    *,
    stage: str,
    datasets: Mapping[str, str | Path],
    validate_record: RecordValidator,
    report_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    max_recorded_errors: int = 50,
) -> dict[str, Any]:
    """Validate every JSONL record and persist a complete summary.

    Validation continues after bad rows so the report captures aggregate reject
    counts and representative errors. Call :func:`require_preflight_passed`
    after logging the report summary and before loading any model checkpoint.
    """
    if not stage:
        raise ValueError("stage must be non-empty")
    if not datasets:
        raise ValueError("at least one preflight dataset is required")
    if max_recorded_errors < 0:
        raise ValueError("max_recorded_errors must be non-negative")

    errors: list[dict[str, Any]] = []
    splits: dict[str, dict[str, Any]] = {}
    total_records = 0
    total_valid = 0
    total_rejected = 0
    total_errors = 0

    for split, raw_path in datasets.items():
        path = str(raw_path)
        stats: dict[str, Any] = {
            "path": path,
            "records": 0,
            "valid": 0,
            "rejected": 0,
            "metrics": {},
        }
        splits[split] = stats
        try:
            with open(path, encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stats["records"] += 1
                    total_records += 1
                    try:
                        if not line.strip():
                            raise ValueError("blank JSONL line")
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("JSONL row must be an object")
                        metrics = validate_record(split, record) or {}
                        for key, value in metrics.items():
                            if isinstance(value, bool):
                                value = int(value)
                            if not isinstance(value, (int, float)):
                                raise TypeError(
                                    f"preflight metric {key!r} must be numeric"
                                )
                            if isinstance(value, float) and not math.isfinite(value):
                                raise ValueError(
                                    f"preflight metric {key!r} must be finite"
                                )
                            stats["metrics"][key] = stats["metrics"].get(key, 0) + value
                    except (Exception, KeyboardInterrupt) as exc:
                        if isinstance(exc, KeyboardInterrupt):
                            raise
                        stats["rejected"] += 1
                        total_rejected += 1
                        total_errors += 1
                        _record_error(
                            errors,
                            max_errors=max_recorded_errors,
                            split=split,
                            path=path,
                            line=line_number,
                            exc=exc,
                        )
                    else:
                        stats["valid"] += 1
                        total_valid += 1
        except (OSError, UnicodeError) as exc:
            stats["file_error"] = str(exc)
            total_errors += 1
            _record_error(
                errors,
                max_errors=max_recorded_errors,
                split=split,
                path=path,
                line=0,
                exc=exc,
            )

        if stats["records"] == 0 and "file_error" not in stats:
            exc = ValueError("dataset is empty")
            total_errors += 1
            _record_error(
                errors,
                max_errors=max_recorded_errors,
                split=split,
                path=path,
                line=0,
                exc=exc,
            )

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "petitgpt_posttrain_preflight",
        "stage": stage,
        "status": "passed" if total_errors == 0 else "failed",
        "total_records": total_records,
        "total_valid": total_valid,
        "total_rejected": total_rejected,
        "total_errors": total_errors,
        "splits": splits,
        "errors": errors,
        "errors_truncated": total_errors > len(errors),
        "metadata": dict(metadata or {}),
    }
    _write_json_atomic(Path(report_path), report)
    return report


def require_preflight_passed(report: Mapping[str, Any]) -> None:
    """Raise after a report has been persisted when any dataset row failed."""
    if report.get("status") != "passed":
        raise ValueError(
            f"{report.get('stage', 'post-training')} preflight failed with "
            f"{report.get('total_errors', '?')} error(s) and "
            f"{report.get('total_rejected', '?')} rejected row(s)"
        )
