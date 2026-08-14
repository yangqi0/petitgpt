#!/usr/bin/env python3
"""
sanity_check_pretrain_shards.py

Quick sanity checks for token-sharded pretrain datasets produced by build_pretrain_shards.py.

What it checks:
1) Reads <out_dir>/meta.json and prints:
   - total train/val tokens, dtype, shard counts
   - per-source train token share vs configured weight (delta)
2) Samples deterministic global blocks from the virtual shard stream and reports:
   - ascii ratio
   - url density
   - html signal flag
   - placeholder flag
   - has double newline, has 4-space indent
   - (optional) EOS fraction in sampled tokens

Usage:
  python sanity_check_pretrain_shards.py \
    --out_dir datasets/pretrain_mix_7b_v2 \
    --tokenizer_path tokenizer/tokenizer.json \
    --seq_len 2048 \
    --sample_shards 6 \
    --blocks_per_shard 12 \
    --split both \
    --eos_id 3

Outputs a JSON summary to stdout and optionally to --out_json.

Notes:
- This is a *sanity* tool, not a full audit. Keep sample sizes modest.
- It reads shards via numpy memmap (fast, low RAM).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretrain.dataset_pretrain import PackedBinDataset, validate_shard_release  # noqa: E402
from src.special_tokens import (  # noqa: E402
    CANONICAL_VOCAB_SIZE,
    EOS_ID,
    assert_tokenizer_contract,
)

_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_HTML_RE = re.compile(r"(<\s*(div|p|span|script|style|table|html|body)\b|&nbsp;|</\s*\w+\s*>)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"(lorem ipsum|as an ai language model|\[deleted\]|\[removed\]|click here|subscribe|sign up)",
    re.IGNORECASE,
)


def ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    n = len(s)
    a = sum(1 for ch in s if ord(ch) < 128)
    return a / n


def url_per_kchars(s: str) -> float:
    if not s:
        return 0.0
    k = max(1e-9, len(s) / 1000.0)
    return float(len(_URL_RE.findall(s))) / k


def has_indent4(s: str) -> bool:
    return "\n    " in s or s.startswith("    ")


def has_dblnl(s: str) -> bool:
    return "\n\n" in s


def html_signal(s: str) -> bool:
    return _HTML_RE.search(s) is not None


def placeholder_signal(s: str) -> bool:
    return _PLACEHOLDER_RE.search(s) is not None


def quantiles(xs: list[float], qs=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)) -> dict[str, float]:
    if not xs:
        return {f"q{int(q * 100):02d}": 0.0 for q in qs}
    arr = np.asarray(xs, dtype=np.float64)
    out: dict[str, float] = {}
    for q in qs:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(arr, q))
    return out


def read_meta(out_dir: Path) -> dict[str, Any]:
    p = out_dir / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"meta.json not found at: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def decode_block(tok: Tokenizer, ids: np.ndarray) -> str:
    # tokenizers expects python list of ints
    return tok.decode(ids.tolist(), skip_special_tokens=False)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="dataset dir containing meta.json, train/, val/")
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--sample_shards", type=int, default=6)
    ap.add_argument("--blocks_per_shard", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--out_json", default="", help="optional path to write JSON report")

    args = ap.parse_args(argv)
    if args.seq_len <= 0 or args.sample_shards < 0 or args.blocks_per_shard < 0:
        raise ValueError("sequence length must be positive and sample counts non-negative")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.eos_id != EOS_ID:
        raise ValueError(f"production EOS ID is fixed at {EOS_ID}, got {args.eos_id}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    assert_tokenizer_contract(tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.encode_special_tokens = True
    vocab_size = tok.get_vocab_size()
    if vocab_size != CANONICAL_VOCAB_SIZE:
        raise RuntimeError(
            f"tokenizer vocab size must be {CANONICAL_VOCAB_SIZE}, got {vocab_size}"
        )
    tokenizer_sha256 = sha256_file(tokenizer_path)
    train_release = validate_shard_release(out_dir / "train")
    if train_release["tokenizer_sha256"] != tokenizer_sha256:
        raise RuntimeError("shard release tokenizer SHA-256 disagrees with --tokenizer_path")
    dtype = np.dtype(train_release["dtype"])
    meta = read_meta(out_dir)

    # ---- meta summary ----
    per_src = meta.get("per_source", {})
    accounting = meta.get("accounting")
    if not isinstance(accounting, dict) or not isinstance(per_src, dict):
        raise RuntimeError("release manifest has no current accounting/per_source objects")
    train_accounting = accounting.get("train") or {}
    val_accounting = accounting.get("val") or {}
    train_serialized = int(train_accounting.get("serialized_tokens", -1))
    val_serialized = int(val_accounting.get("serialized_tokens", -1))
    train_content = int(train_accounting.get("content_tokens", -1))
    val_content = int(val_accounting.get("content_tokens", -1))

    weight_total = sum(float(st.get("weight", 0.0)) for st in per_src.values())
    if weight_total <= 0:
        raise RuntimeError("per-source weights must have a positive sum")
    src_rows: list[dict[str, Any]] = []
    for src, st in per_src.items():
        realized = st.get("realized") or {}
        train_realized = realized.get("train") or {}
        val_realized = realized.get("val") or {}
        weight = float(st.get("weight", 0.0))
        train_tokens = int(train_realized.get("serialized_tokens", -1))
        val_tokens = int(val_realized.get("serialized_tokens", -1))
        fraction = train_tokens / train_serialized if train_serialized > 0 else 0.0
        normalized_weight = weight / weight_total
        src_rows.append({
            "source": src,
            "weight": weight,
            "normalized_weight": normalized_weight,
            "train_serialized_tokens": train_tokens,
            "val_serialized_tokens": val_tokens,
            "train_serialized_fraction": fraction,
            "train_fraction_minus_normalized_weight": fraction - normalized_weight,
        })
    src_rows.sort(key=lambda row: (-int(row["train_serialized_tokens"]), row["source"]))

    summary = {
        "out_dir": str(out_dir),
        "dtype": dtype.name,
        "vocab_size": int(vocab_size),
        "tokenizer_sha256": tokenizer_sha256,
        "release_manifest_sha256": train_release["manifest_sha256"],
        "meta_train_serialized_tokens": train_serialized,
        "meta_val_serialized_tokens": val_serialized,
        "meta_train_content_tokens": train_content,
        "meta_val_content_tokens": val_content,
        "meta_train_shards": int(meta.get("train_shards", -1)),
        "meta_val_shards": int(meta.get("val_shards", -1)),
        "sources": src_rows,
    }

    # Sample fixed global training blocks from the central virtual-stream dataset.
    requested_blocks = args.sample_shards * args.blocks_per_shard

    def analyze_split(split_name: str) -> dict[str, Any]:
        split_dir = out_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"shard split directory does not exist: {split_dir}")
        shard_artifacts = sorted(
            path.name for path in split_dir.glob("shard_*") if path.is_file()
        )
        if not shard_artifacts:
            if (
                split_name != "val"
                or int(meta.get("val_shards", -1)) != 0
                or int(meta.get("val_tokens", -1)) != 0
            ):
                raise RuntimeError(f"{split_name} split has no shards despite its manifest")
            return {
                "split": split_name,
                "release_manifest_sha256": train_release["manifest_sha256"],
                "n_shards": 0,
                "total_global_blocks": 0,
                "requested_global_blocks": requested_blocks,
                "blocks_sampled": 0,
                "sampled_global_block_ids": [],
                "cross_shard_blocks_sampled": 0,
            }

        release = validate_shard_release(split_dir)
        if release["tokenizer_sha256"] != tokenizer_sha256:
            raise RuntimeError(
                f"{split_name} release tokenizer SHA-256 disagrees with --tokenizer_path"
            )
        dataset = PackedBinDataset(
            str(split_dir),
            seq_len=args.seq_len,
            sampling_mode="deterministic",
            require_release_manifest=True,
        )
        sample_count = min(requested_blocks, len(dataset))
        split_seed = args.seed ^ (0xA5A5A5A5 if split_name == "train" else 0x5A5A5A5A)
        split_rng = random.Random(split_seed)
        block_ids = (
            sorted(split_rng.sample(range(len(dataset)), k=sample_count))
            if sample_count
            else []
        )

        ascii_rs: list[float] = []
        url_dens: list[float] = []
        lens: list[float] = []
        eos_fracs: list[float] = []
        html_flags = 0
        placeholder_flags = 0
        dblnl_flags = 0
        indent_flags = 0
        oob_tokens = 0
        total_tokens_checked = 0
        cross_shard_blocks = 0
        boundaries = dataset.shard_token_offsets[1:-1]

        for block_id in block_ids:
            input_ids, labels, _ = dataset[block_id]
            ids = np.concatenate(
                (input_ids.numpy(), labels[-1:].numpy())
            ).astype(np.int64, copy=False)
            total_tokens_checked += int(ids.size)
            oob_tokens += int(((ids < 0) | (ids >= vocab_size)).sum())
            eos_fracs.append(float((ids == EOS_ID).mean()))

            start = block_id * args.seq_len
            end = start + args.seq_len
            cross_shard_blocks += any(start < boundary <= end for boundary in boundaries)

            text = decode_block(tok, ids)
            lens.append(float(len(text)))
            ascii_rs.append(ascii_ratio(text))
            url_dens.append(url_per_kchars(text))
            html_flags += html_signal(text)
            placeholder_flags += placeholder_signal(text)
            dblnl_flags += has_dblnl(text)
            indent_flags += has_indent4(text)

        actual_blocks = len(block_ids)
        denominator = max(1, actual_blocks)
        return {
            "split": split_name,
            "release_manifest_sha256": release["manifest_sha256"],
            "n_shards": int(release["expected_shards"]),
            "total_global_blocks": len(dataset),
            "requested_global_blocks": requested_blocks,
            "blocks_sampled": actual_blocks,
            "sampled_global_block_ids": block_ids,
            "cross_shard_blocks_sampled": int(cross_shard_blocks),
            "token_ids_oob": int(oob_tokens),
            "token_ids_checked": int(total_tokens_checked),
            "ascii_ratio_quantiles": quantiles(ascii_rs),
            "len_chars_quantiles": quantiles(lens),
            "url_per_kchars_quantiles": quantiles(url_dens),
            "html_signal_frac": float(html_flags / denominator),
            "placeholder_frac": float(placeholder_flags / denominator),
            "has_dblnl_frac": float(dblnl_flags / denominator),
            "has_indent4_frac": float(indent_flags / denominator),
            "eos_frac_quantiles": quantiles(eos_fracs),
        }

    splits: list[str]
    if args.split == "both":
        splits = ["train", "val"]
    else:
        splits = [args.split]

    split_reports = [analyze_split(s) for s in splits]

    report = {
        "schema_version": 1,
        "kind": "petitgpt_pretrain_shard_sanity",
        "seed": args.seed,
        "seq_len": args.seq_len,
        "summary": summary,
        "split_reports": split_reports,
    }

    if args.out_json:
        write_json_atomic(Path(args.out_json), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
