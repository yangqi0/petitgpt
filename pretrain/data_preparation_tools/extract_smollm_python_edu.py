#!/usr/bin/env python3

"""Extract SmolLM ``python-edu`` blobs with crash-safe local persistence.

The metadata dataset points at gzip-compressed Software Heritage blobs. This
extractor deliberately keeps blob fetching serial: deterministic checkpointing,
strict decoding, and resumability matter more here than speculative concurrency.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

_STATE_VERSION = 1


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync after an atomic rename."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(str(path.parent), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fout:
        json.dump(obj, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fout:
            fout.write(data)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp, path)
        _fsync_parent(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def decode_blob_bytes(raw: bytes, *, blob_id: str = "<unknown>") -> str:
    """Decode a source blob without silently deleting invalid bytes."""
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"{exc.reason}; blob_id={blob_id}",
        ) from exc


def blob_cache_path(cache_dir: Path, blob_id: str) -> Path:
    digest = hashlib.sha256(blob_id.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.blob"


def fetch_blob_cached(
    blob_id: str,
    fetcher: Callable[[str], bytes],
    cache_dir: Path | None,
) -> tuple[bytes, bool | None]:
    """Return raw decompressed bytes plus whether the local cache was hit."""
    if cache_dir is None:
        return fetcher(blob_id), None

    path = blob_cache_path(cache_dir, blob_id)
    try:
        return path.read_bytes(), True
    except FileNotFoundError:
        pass

    raw = fetcher(blob_id)
    _write_bytes_atomic(path, raw)
    return raw, False


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _empty_stats() -> dict[str, int]:
    return {
        "total_seen": 0,
        "kept": 0,
        "score_rejected": 0,
        "fetch_failed": 0,
        "decode_failed": 0,
        "metadata_invalid": 0,
        "empty_rejected": 0,
        "length_rejected": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }


def _state_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        Path(str(output_path) + ".partial"),
        Path(str(output_path) + ".state.json"),
    )


def _load_or_initialize_state(
    *,
    output_path: Path,
    config: Mapping[str, Any],
    resume: bool,
    overwrite: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    partial_path, state_path = _state_paths(output_path)
    fingerprint = _config_fingerprint(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    replace_existing_output = output_path.exists()
    if overwrite:
        partial_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    partial_exists = partial_path.exists()
    state_exists = state_path.exists()
    if output_path.exists() and not overwrite and not partial_exists:
        raise FileExistsError(
            f"refusing to replace completed output {output_path}; pass --overwrite"
        )
    if partial_exists != state_exists:
        raise RuntimeError(
            "incomplete resume artifacts: expected both "
            f"{partial_path} and {state_path}; use --overwrite to restart"
        )

    if partial_exists:
        if not resume:
            raise RuntimeError(
                "resume artifacts exist but resume is disabled; use --resume or --overwrite"
            )
        with open(state_path, encoding="utf-8") as fin:
            state = json.load(fin)
        if state.get("version") != _STATE_VERSION:
            raise RuntimeError(f"unsupported resume state version: {state.get('version')!r}")
        if output_path.exists() and not state.get("replace_existing_output", False):
            raise FileExistsError(
                f"completed output {output_path} appeared after extraction began; "
                "use --overwrite to restart explicitly"
            )
        if state.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                "resume configuration/dataset fingerprint changed; use --overwrite to restart"
            )
        output_bytes = int(state.get("output_bytes", -1))
        actual_bytes = partial_path.stat().st_size
        if output_bytes < 0 or output_bytes > actual_bytes:
            raise RuntimeError(
                f"invalid resume byte offset {output_bytes} for {actual_bytes}-byte partial"
            )
        with open(partial_path, "r+b") as fout:
            fout.truncate(output_bytes)
            fout.flush()
            os.fsync(fout.fileno())
        return partial_path, state_path, state

    state = {
        "version": _STATE_VERSION,
        "config": dict(config),
        "config_fingerprint": fingerprint,
        "replace_existing_output": bool(replace_existing_output),
        "next_index": 0,
        "output_bytes": 0,
        "stats": _empty_stats(),
    }
    with open(partial_path, "xb") as fout:
        fout.flush()
        os.fsync(fout.fileno())
    _write_json_atomic(state_path, state)
    return partial_path, state_path, state


def _record_for_example(
    ex: Mapping[str, Any],
    *,
    fetcher: Callable[[str], bytes],
    cache_dir: Path | None,
    min_int_score: int,
    min_chars: int,
    max_chars: int,
) -> tuple[dict[str, Any] | None, str, bool | None, str | None]:
    try:
        int_score = int(ex.get("int_score", 0))
    except (TypeError, ValueError):
        return None, "metadata_invalid", None, "invalid int_score"
    if int_score < min_int_score:
        return None, "score_rejected", None, None

    blob_id_obj = ex.get("blob_id")
    if not isinstance(blob_id_obj, str) or not blob_id_obj:
        return None, "metadata_invalid", None, "missing blob_id"
    blob_id = blob_id_obj

    try:
        raw, cache_hit = fetch_blob_cached(blob_id, fetcher, cache_dir)
    except Exception as exc:
        return None, "fetch_failed", None, repr(exc)
    try:
        text = decode_blob_bytes(raw, blob_id=blob_id)
    except UnicodeDecodeError as exc:
        return None, "decode_failed", cache_hit, str(exc)

    if not text.strip():
        return None, "empty_rejected", cache_hit, None
    if len(text) < min_chars or len(text) > max_chars:
        return None, "length_rejected", cache_hit, None

    try:
        score = float(ex.get("score", 0.0))
    except (TypeError, ValueError):
        return None, "metadata_invalid", cache_hit, "invalid score"
    record = {
        "text": text,
        "meta": {
            "source": "smollm_python_edu",
            "repo_name": ex.get("repo_name", ""),
            "path": ex.get("path", ""),
            "score": score,
            "int_score": int_score,
            "blob_id": blob_id,
        },
    }
    return record, "kept", cache_hit, None


def extract_dataset_records(
    dataset: Iterable[Mapping[str, Any]],
    *,
    fetcher: Callable[[str], bytes],
    output_path: Path,
    cache_dir: Path | None,
    config: Mapping[str, Any],
    min_int_score: int,
    min_chars: int,
    max_chars: int,
    max_samples: int | None = None,
    checkpoint_every: int = 1000,
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract records to an atomic JSONL output, resuming at exact row offsets."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("require 0 <= min_chars <= max_chars")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be >= 0")

    output_path = Path(output_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    resume_config = {
        **dict(config),
        "max_samples": max_samples,
        "min_int_score": int(min_int_score),
        "min_chars": int(min_chars),
        "max_chars": int(max_chars),
    }
    partial_path, state_path, state = _load_or_initialize_state(
        output_path=output_path,
        config=resume_config,
        resume=resume,
        overwrite=overwrite,
    )
    start_index = int(state["next_index"])
    stats = _empty_stats()
    stats.update({key: int(value) for key, value in state.get("stats", {}).items()})

    def checkpoint(fout: Any, next_index: int) -> None:
        fout.flush()
        os.fsync(fout.fileno())
        state.update(
            next_index=int(next_index),
            output_bytes=int(fout.tell()),
            stats=dict(stats),
        )
        _write_json_atomic(state_path, state)

    next_index = start_index
    with open(partial_path, "r+b") as fout:
        fout.seek(0, os.SEEK_END)
        for index, ex in enumerate(dataset):
            if max_samples is not None and index >= max_samples:
                break
            if index < start_index:
                continue

            stats["total_seen"] += 1
            record, outcome, cache_hit, error = _record_for_example(
                ex,
                fetcher=fetcher,
                cache_dir=cache_dir,
                min_int_score=min_int_score,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            stats[outcome] += 1
            if cache_hit is not None:
                stats["cache_hits" if cache_hit else "cache_misses"] += 1
            if error and stats[outcome] <= 5:
                print(
                    f"[{outcome}] index={index} blob_id={ex.get('blob_id', '')!r} "
                    f"error={error}"
                )
            if record is not None:
                line = json.dumps(record, ensure_ascii=False) + "\n"
                fout.write(line.encode("utf-8"))
                if stats["kept"] % 1000 == 0:
                    print(
                        f"[keep={stats['kept']}] total={stats['total_seen']} "
                        f"fetch_failed={stats['fetch_failed']} "
                        f"decode_failed={stats['decode_failed']}"
                    )

            next_index = index + 1
            if next_index % checkpoint_every == 0:
                checkpoint(fout, next_index)

        checkpoint(fout, next_index)

    os.replace(partial_path, output_path)
    _fsync_parent(output_path)
    state_path.unlink(missing_ok=True)
    _fsync_parent(state_path)

    metadata = {
        "status": "complete",
        "output": str(output_path),
        "config": resume_config,
        "config_fingerprint": _config_fingerprint(resume_config),
        "next_index": int(next_index),
        "stats": stats,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "strict_utf8": True,
        "atomic_output": True,
    }
    _write_json_atomic(Path(str(output_path) + ".meta.json"), metadata)
    return metadata


def make_software_heritage_fetcher() -> Callable[[str], bytes]:
    """Create an unsigned S3 fetcher returning decompressed blob bytes."""
    try:
        import boto3
        import botocore
    except ImportError as exc:
        raise RuntimeError("boto3 and botocore are required for S3 extraction") from exc

    client = boto3.client(
        "s3",
        region_name="us-west-2",
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )

    def fetch(blob_id: str) -> bytes:
        response = client.get_object(
            Bucket="softwareheritage",
            Key=f"content/{blob_id}",
        )
        body = response["Body"]
        try:
            with gzip.GzipFile(fileobj=body) as compressed:
                return compressed.read()
        finally:
            body.close()

    return fetch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num_proc", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--min_int_score", type=int, default=4)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=50000)
    ap.add_argument("--checkpoint_every", type=int, default=1000)
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Raw decompressed blob cache (default: <out>.blob_cache).",
    )
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from <out>.partial and <out>.state.json (default: enabled).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly restart partial state and atomically replace a completed output.",
    )
    ap.add_argument("--dataset_name", default="HuggingFaceTB/smollm-corpus")
    ap.add_argument("--dataset_config", default="python-edu")
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load SmolLM metadata") from exc

    load_kwargs: dict[str, Any] = {
        "split": args.split,
        "num_proc": args.num_proc,
    }
    if args.revision is not None:
        load_kwargs["revision"] = args.revision
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        **load_kwargs,
    )

    output_path = Path(args.out)
    cache_dir = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir or (str(output_path) + ".blob_cache"))
    config = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "revision": args.revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
    }
    metadata = extract_dataset_records(
        dataset,
        fetcher=make_software_heritage_fetcher(),
        output_path=output_path,
        cache_dir=cache_dir,
        config=config,
        min_int_score=args.min_int_score,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_samples=args.max_samples,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
