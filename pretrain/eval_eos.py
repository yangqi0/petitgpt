#!/usr/bin/env python3
"""Deterministic EOS diagnostics for pretraining checkpoints.

This evaluator deliberately separates two questions:

* Teacher forcing measures how the model scores EOS at serialized document
  ends versus eligible, ordinary internal positions.
* Greedy generation measures whether a frozen prompt set stops too early or
  fails to stop before a fixed token budget.

The generation JSONL accepts either a raw base-model ``prompt`` or structured
``messages``. Raw strings are always encoded with special-token matching
disabled; chat control tokens are inserted by ID through ``encode_prompt``.
Supplying pre-encoded IDs is intentionally unsupported.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.dataset_pretrain import FixedSubsetSampler, PackedBinDataset  # noqa: E402
from src.special_tokens import (  # noqa: E402
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKEN_IDS,
    SYSTEM_ID,
    UNK_ID,
    USER_ID,
    assert_special_token_ids,
)

REPORT_SCHEMA_VERSION = 1
CANONICAL_VOCAB_SIZE = 32_000
CONTROL_IDS = frozenset(
    {PAD_ID, UNK_ID, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID}
)

_COUNT_KEYS = (
    "serialized_positions",
    "eligible_positions",
    "masked_positions",
    "true_document_end_positions",
    "eligible_internal_positions",
    "masked_eos_target_positions",
    "true_eos_top1_positions",
    "internal_eos_top1_positions",
)
_SUM_KEYS = (
    "supervised_nll_sum",
    "non_eos_nll_sum",
    "true_eos_nll_sum",
    "true_eos_probability_sum",
    "internal_eos_probability_sum",
)


def teacher_forced_batch_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eos_id: int = EOS_ID,
) -> dict[str, Any]:
    """Return additive EOS statistics for one batch.

    A position is eligible exactly when the data pipeline's loss mask is
    positive. A true document end is an eligible EOS target; an internal
    position is an eligible non-EOS target. This means masked BOS targets and
    repeated padding-like EOS targets cannot silently contaminate either
    denominator.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,V], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2] or loss_mask.shape != labels.shape:
        raise ValueError(
            "labels/loss_mask must match logits [B,T]: "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}, "
            f"loss_mask={tuple(loss_mask.shape)}"
        )
    if not 0 <= int(eos_id) < int(logits.shape[-1]):
        raise ValueError(f"eos_id={eos_id} is outside vocab size {logits.shape[-1]}")
    if not bool(torch.isfinite(loss_mask).all()):
        raise ValueError("loss_mask contains a non-finite value")

    eligible = loss_mask > 0
    eos_target = labels == int(eos_id)
    true_end = eligible & eos_target
    internal = eligible & ~eos_target
    masked_eos = ~eligible & eos_target

    # logsumexp(logits) - eos_logit is numerically stable -log P(EOS) and
    # avoids materializing a second [B,T,V] softmax tensor.
    logits_f32 = logits.float()
    if bool(((labels < 0) | (labels >= logits.shape[-1])).any()):
        raise ValueError("labels contains a token ID outside the logits vocabulary")
    log_partition = torch.logsumexp(logits_f32, dim=-1)
    target_nll = log_partition - logits_f32.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    eos_nll = log_partition - logits_f32[..., int(eos_id)]
    eos_probability = torch.exp(-eos_nll)
    top1_is_eos = logits_f32.argmax(dim=-1) == int(eos_id)

    return {
        "counts": {
            "serialized_positions": int(labels.numel()),
            "eligible_positions": int(eligible.sum().item()),
            "masked_positions": int((~eligible).sum().item()),
            "true_document_end_positions": int(true_end.sum().item()),
            "eligible_internal_positions": int(internal.sum().item()),
            "masked_eos_target_positions": int(masked_eos.sum().item()),
            "true_eos_top1_positions": int((top1_is_eos & true_end).sum().item()),
            "internal_eos_top1_positions": int((top1_is_eos & internal).sum().item()),
        },
        "sums": {
            "supervised_nll_sum": float(target_nll[eligible].sum().item()),
            "non_eos_nll_sum": float(target_nll[internal].sum().item()),
            "true_eos_nll_sum": float(eos_nll[true_end].sum().item()),
            "true_eos_probability_sum": float(eos_probability[true_end].sum().item()),
            "internal_eos_probability_sum": float(eos_probability[internal].sum().item()),
        },
    }


def empty_teacher_forced_statistics() -> dict[str, Any]:
    return {
        "counts": {key: 0 for key in _COUNT_KEYS},
        "sums": {key: 0.0 for key in _SUM_KEYS},
    }


def merge_teacher_forced_statistics(
    parts: Iterable[Mapping[str, Mapping[str, int | float]]],
) -> dict[str, Any]:
    """Pure additive merge used for exact multi-batch counts."""
    total = empty_teacher_forced_statistics()
    for part in parts:
        counts = part.get("counts", {})
        sums = part.get("sums", {})
        missing_counts = set(_COUNT_KEYS) - set(counts)
        missing_sums = set(_SUM_KEYS) - set(sums)
        if missing_counts or missing_sums:
            raise ValueError(
                f"incomplete statistics: missing counts={sorted(missing_counts)}, "
                f"missing sums={sorted(missing_sums)}"
            )
        for key in _COUNT_KEYS:
            total["counts"][key] += int(counts[key])
        for key in _SUM_KEYS:
            total["sums"][key] += float(sums[key])
    return total


def _safe_ratio(numerator: int | float, denominator: int) -> float | None:
    return None if int(denominator) == 0 else float(numerator) / int(denominator)


def _safe_perplexity(cross_entropy: float | None) -> float | None:
    if cross_entropy is None or cross_entropy > math.log(sys.float_info.max):
        return None
    return math.exp(cross_entropy)


def finalize_teacher_forced_statistics(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Turn additive sums into named metrics without inventing zero denominators."""
    counts = {key: int(stats["counts"][key]) for key in _COUNT_KEYS}
    sums = {key: float(stats["sums"][key]) for key in _SUM_KEYS}
    end_count = counts["true_document_end_positions"]
    internal_count = counts["eligible_internal_positions"]
    if counts["eligible_positions"] != end_count + internal_count:
        raise ValueError("eligible count does not partition into EOS ends and internal targets")
    if counts["serialized_positions"] != counts["eligible_positions"] + counts["masked_positions"]:
        raise ValueError("serialized count does not partition into eligible and masked targets")

    overall_ce = _safe_ratio(sums["supervised_nll_sum"], counts["eligible_positions"])
    non_eos_ce = _safe_ratio(sums["non_eos_nll_sum"], internal_count)
    return {
        "counts": counts,
        "metrics": {
            "overall_supervised_cross_entropy": overall_ce,
            "overall_supervised_perplexity": _safe_perplexity(overall_ce),
            "non_eos_cross_entropy": non_eos_ce,
            "non_eos_perplexity": _safe_perplexity(non_eos_ce),
            "eos_only_cross_entropy": _safe_ratio(sums["true_eos_nll_sum"], end_count),
            "eos_top1_accuracy": _safe_ratio(counts["true_eos_top1_positions"], end_count),
            "mean_p_eos_true_document_end": _safe_ratio(
                sums["true_eos_probability_sum"], end_count
            ),
            "mean_p_eos_eligible_internal": _safe_ratio(
                sums["internal_eos_probability_sum"], internal_count
            ),
            "internal_eos_top1_false_positive_rate": _safe_ratio(
                counts["internal_eos_top1_positions"], internal_count
            ),
        },
        # Keeping additive sums makes reports independently auditable and
        # allows exact aggregation across domains/checkpoints.
        "additive_sums": sums,
    }


def classify_generation(
    generated_ids: Sequence[int],
    *,
    eos_id: int = EOS_ID,
    min_tokens_before_eos: int,
) -> dict[str, Any]:
    """Classify EOS stopping for one generated sequence.

    ``min_tokens_before_eos`` is a frozen property of each diagnostic example,
    not a decoding constraint. Generation is always allowed to emit EOS on its
    first step; that is precisely how premature stopping remains observable.
    """
    minimum = int(min_tokens_before_eos)
    if minimum < 0:
        raise ValueError("min_tokens_before_eos must be >= 0")
    ids = [int(token_id) for token_id in generated_ids]
    try:
        first_eos_index: int | None = ids.index(int(eos_id))
    except ValueError:
        first_eos_index = None

    stopped = first_eos_index is not None
    tokens_before_eos = len(ids) if first_eos_index is None else first_eos_index
    tokens_after_eos = 0 if first_eos_index is None else len(ids) - first_eos_index - 1
    return {
        "stopped_by_eos": stopped,
        "failure_to_stop": not stopped,
        "premature_eos": bool(stopped and tokens_before_eos < minimum),
        "first_eos_index": first_eos_index,
        "tokens_before_eos": tokens_before_eos,
        "tokens_after_first_eos": tokens_after_eos,
        "generated_tokens_including_eos": len(ids),
        "min_tokens_before_eos": minimum,
    }


def _percentile(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(percentile)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize_generation_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pure aggregate of per-example classifications and lengths."""
    total = len(records)
    stopped = sum(bool(r["stopped_by_eos"]) for r in records)
    premature = sum(bool(r["premature_eos"]) for r in records)
    failures = sum(bool(r["failure_to_stop"]) for r in records)
    lengths = [int(r["tokens_before_eos"]) for r in records]
    stopped_lengths = [
        int(r["tokens_before_eos"]) for r in records if bool(r["stopped_by_eos"])
    ]

    def length_stats(xs: Sequence[int]) -> dict[str, float | int | None]:
        return {
            "count": len(xs),
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
            "mean": float(statistics.fmean(xs)) if xs else None,
            "p50": _percentile(xs, 0.50),
            "p90": _percentile(xs, 0.90),
            "p95": _percentile(xs, 0.95),
        }

    return {
        "counts": {
            "examples": total,
            "stopped_by_eos": stopped,
            "premature_eos": premature,
            "failure_to_stop": failures,
        },
        "rates": {
            "eos_stop_rate": _safe_ratio(stopped, total),
            "premature_eos_rate": _safe_ratio(premature, total),
            "premature_eos_rate_given_eos_stop": _safe_ratio(premature, stopped),
            "failure_to_stop_rate": _safe_ratio(failures, total),
        },
        "length_excluding_terminal_eos": length_stats(lengths),
        "stopped_length_excluding_terminal_eos": length_stats(stopped_lengths),
    }


def read_generation_cases(path: str | Path, *, max_new_tokens: int) -> list[dict[str, Any]]:
    """Read and validate the frozen generation-set JSONL schema."""
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            if "prompt_ids" in item or "input_ids" in item:
                raise ValueError(
                    f"pre-encoded prompt IDs are forbidden at {path}:{line_number}; "
                    "use prompt or messages so injection hardening cannot be bypassed"
                )
            has_prompt = isinstance(item.get("prompt"), str)
            has_messages = isinstance(item.get("messages"), list)
            if has_prompt == has_messages:
                raise ValueError(
                    f"exactly one of prompt/messages is required at {path}:{line_number}"
                )
            case_id = str(item.get("id", f"line-{line_number}"))
            if not case_id or case_id in seen_ids:
                raise ValueError(f"duplicate or empty id {case_id!r} at {path}:{line_number}")
            seen_ids.add(case_id)
            if "min_tokens_before_eos" not in item:
                raise ValueError(
                    f"min_tokens_before_eos must be frozen per example at {path}:{line_number}"
                )
            minimum = int(item["min_tokens_before_eos"])
            if minimum < 0 or minimum >= int(max_new_tokens):
                raise ValueError(
                    "min_tokens_before_eos must satisfy 0 <= minimum < max_new_tokens "
                    f"at {path}:{line_number}"
                )
            if has_messages:
                messages = item["messages"]
                if not messages or not all(isinstance(m, dict) for m in messages):
                    raise ValueError(f"messages must be a non-empty object list at {path}:{line_number}")
                valid_roles = {"system", "user", "assistant"}
                if any(str(m.get("role", "")).strip().lower() not in valid_roles for m in messages):
                    raise ValueError(f"messages contains an invalid role at {path}:{line_number}")
                if any(not isinstance(m.get("content"), str) for m in messages):
                    raise ValueError(f"every message content must be a string at {path}:{line_number}")
                if not any(str(m.get("role", "")).strip().lower() == "user" for m in messages):
                    raise ValueError(f"messages requires a user turn at {path}:{line_number}")
            cases.append(
                {
                    "id": case_id,
                    "prompt": item.get("prompt") if has_prompt else None,
                    "messages": item.get("messages") if has_messages else None,
                    "min_tokens_before_eos": minimum,
                }
            )
    if not cases:
        raise ValueError(f"no generation cases found in {path}")
    return cases


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON report without exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_exact_tokenizer_contract(tokenizer_path: str | Path) -> None:
    """Fail unless tokenizer.json has exactly the canonical seven specials."""
    assert_special_token_ids(str(tokenizer_path))
    with open(tokenizer_path, encoding="utf-8") as handle:
        obj = json.load(handle)
    vocab = (obj.get("model") or {}).get("vocab")
    if not isinstance(vocab, (dict, list)) or len(vocab) != CANONICAL_VOCAB_SIZE:
        size = len(vocab) if isinstance(vocab, (dict, list)) else None
        raise ValueError(f"tokenizer vocab size must be {CANONICAL_VOCAB_SIZE}, got {size}")
    special_entries = [
        token for token in obj.get("added_tokens") or [] if token.get("special") is True
    ]
    actual_specials = {
        str(token.get("content")): int(token.get("id"))
        for token in special_entries
    }
    if len(special_entries) != len(SPECIAL_TOKEN_IDS) or actual_specials != SPECIAL_TOKEN_IDS:
        raise ValueError(
            "tokenizer must declare exactly the canonical seven special tokens; "
            f"got {actual_specials!r}"
        )


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return device


def _resolve_precision(requested: str, device: torch.device) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    if requested == "bf16" and device.type != "cuda":
        raise ValueError("bf16 evaluation is supported on CUDA only; use fp32 on CPU")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 was requested but this CUDA device does not support it")
    return requested


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate_teacher_forced(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    parts: list[dict[str, Any]] = []
    batches = 0
    for input_storage, labels_storage, loss_mask in data_loader:
        input_ids = input_storage.to(device=device, dtype=torch.long, non_blocking=True)
        labels = labels_storage.to(device=device, dtype=torch.long, non_blocking=True)
        loss_mask = loss_mask.to(device=device, dtype=torch.float32, non_blocking=True)
        with _autocast(device, precision):
            logits = model(input_ids)
        if isinstance(logits, dict):
            logits = logits["logits"]
        parts.append(
            teacher_forced_batch_statistics(logits, labels, loss_mask, eos_id=EOS_ID)
        )
        batches += 1
    if was_training:
        model.train()
    result = finalize_teacher_forced_statistics(merge_teacher_forced_statistics(parts))
    result["batches"] = batches
    return result


def _encode_untrusted_text(tokenizer: Any, text: str, *, location: str) -> list[int]:
    ids = [int(i) for i in tokenizer.encode(str(text)).ids]
    leaked = sorted(CONTROL_IDS.intersection(ids))
    if leaked:
        raise RuntimeError(
            f"raw content at {location} encoded to control IDs {leaked}; "
            "special-token matching must remain disabled"
        )
    return ids


def _build_prompt_ids(tokenizer: Any, case: Mapping[str, Any], *, default_system: str) -> list[int]:
    if case.get("prompt") is not None:
        return [BOS_ID] + _encode_untrusted_text(
            tokenizer, str(case["prompt"]), location=f"case {case['id']} prompt"
        )

    from src.chat_template import encode_prompt

    messages = case["messages"]
    if default_system:
        _encode_untrusted_text(tokenizer, default_system, location="default_system")
    for index, message in enumerate(messages):
        _encode_untrusted_text(
            tokenizer,
            str(message.get("content", "")),
            location=f"case {case['id']} messages[{index}].content",
        )
    return encode_prompt(tokenizer, messages, default_system=default_system, mode="full_context")


@torch.inference_mode()
def run_generation_diagnostics(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    precision: str,
    max_seq_len: int,
    max_new_tokens: int,
    default_system: str,
) -> dict[str, Any]:
    """Run reproducible greedy generation with EOS as the only stop token."""
    was_training = model.training
    model.eval()
    outputs: list[dict[str, Any]] = []
    for case in cases:
        prompt_ids = _build_prompt_ids(tokenizer, case, default_system=default_system)
        if not prompt_ids:
            raise ValueError(f"case {case['id']} produced an empty prompt")
        if len(prompt_ids) + int(max_new_tokens) > int(max_seq_len):
            raise ValueError(
                f"case {case['id']} needs {len(prompt_ids) + int(max_new_tokens)} tokens "
                f"but checkpoint max_seq_len is {max_seq_len}; prompts are never silently truncated"
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with _autocast(device, precision):
            generated_tensor = model.generate(
                input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                eos_id=EOS_ID,
            )
        generated_ids = [int(i) for i in generated_tensor[0, len(prompt_ids) :].tolist()]
        classification = classify_generation(
            generated_ids,
            eos_id=EOS_ID,
            min_tokens_before_eos=int(case["min_tokens_before_eos"]),
        )
        if classification["tokens_after_first_eos"] != 0:
            raise RuntimeError(f"case {case['id']} generated tokens after EOS")
        content_ids = generated_ids[:-1] if classification["stopped_by_eos"] else generated_ids
        unexpected_control_ids = sorted(
            CONTROL_IDS.intersection(content_ids) - {EOS_ID}
        )
        record = {
            "id": case["id"],
            "input_type": "raw_prompt" if case.get("prompt") is not None else "messages",
            "prompt": case.get("prompt"),
            "messages": case.get("messages"),
            "prompt_token_count": len(prompt_ids),
            "generated_token_ids": generated_ids,
            "completion_text": (
                tokenizer.decode(content_ids, skip_special_tokens=False) if content_ids else ""
            ),
            "non_eos_control_ids_emitted": unexpected_control_ids,
            "stop_reason": "eos" if classification["stopped_by_eos"] else "max_new_tokens",
            **classification,
        }
        outputs.append(record)
    if was_training:
        model.train()
    return {
        "summary": summarize_generation_records(outputs),
        "examples": outputs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="PetitGPT checkpoint .pt")
    parser.add_argument("--tokenizer_path", required=True, help="canonical tokenizer.json")
    parser.add_argument("--out", required=True, help="atomic JSON report path")
    parser.add_argument("--val_dir", default="", help="packed deterministic validation shard dir")
    parser.add_argument(
        "--val_by_source_root",
        default="",
        help="directory whose immediate subdirectories are packed per-source validation sets",
    )
    parser.add_argument("--generation_jsonl", default="", help="frozen generation diagnostic set")
    parser.add_argument("--val_samples", type=int, default=512, help="0 means every validation block")
    parser.add_argument(
        "--val_samples_per_source", type=int, default=128, help="0 means every block per source"
    )
    parser.add_argument("--seq_len", type=int, default=0, help="0 uses checkpoint max_seq_len")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--default_system", default="You are a helpful assistant.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.val_dir and not args.val_by_source_root and not args.generation_jsonl:
        raise SystemExit("provide --val_dir, --val_by_source_root, --generation_jsonl, or a combination")
    if args.val_samples < 0 or args.val_samples_per_source < 0:
        raise SystemExit("validation sample counts must be >= 0")
    if args.batch_size <= 0 or args.num_workers < 0 or args.max_new_tokens <= 0:
        raise SystemExit("batch_size/max_new_tokens must be positive; num_workers must be >= 0")

    assert_exact_tokenizer_contract(args.tokenizer_path)
    from pretrain.sample import _load_ckpt_and_build_model
    from src.chat_template import load_chat_tokenizer

    tokenizer = load_chat_tokenizer(args.tokenizer_path)
    if int(tokenizer.get_vocab_size()) != CANONICAL_VOCAB_SIZE:
        raise ValueError(
            f"runtime tokenizer vocab size must be {CANONICAL_VOCAB_SIZE}, "
            f"got {tokenizer.get_vocab_size()}"
        )
    device = _resolve_device(args.device)
    precision = _resolve_precision(args.precision, device)
    model, checkpoint = _load_ckpt_and_build_model(args.ckpt, device=str(device))
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("loaded model has no GPT config")
    if int(cfg.vocab_size) != CANONICAL_VOCAB_SIZE:
        raise ValueError(
            f"checkpoint vocab size must be {CANONICAL_VOCAB_SIZE}, got {cfg.vocab_size}"
        )
    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(cfg.max_seq_len)
    if seq_len > int(cfg.max_seq_len):
        raise ValueError(f"seq_len={seq_len} exceeds checkpoint max_seq_len={cfg.max_seq_len}")

    raw_global_step = checkpoint.get("global_step")
    global_step = int(raw_global_step) if raw_global_step is not None else None
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "contract": {
            "vocab_size": CANONICAL_VOCAB_SIZE,
            "special_token_ids": SPECIAL_TOKEN_IDS,
            "generation_min_new_tokens": 0,
            "generation_semantic_stop_ids": [EOS_ID],
        },
        "runtime": {"device": str(device), "precision": precision},
        "checkpoint": {
            "path": str(Path(args.ckpt).resolve()),
            "sha256": _sha256(args.ckpt),
            "global_step": global_step,
            "max_seq_len": int(cfg.max_seq_len),
        },
        "tokenizer": {
            "path": str(Path(args.tokenizer_path).resolve()),
            "sha256": _sha256(args.tokenizer_path),
        },
    }

    def evaluate_directory(val_path: str | Path, sample_limit: int) -> dict[str, Any]:
        dataset = PackedBinDataset(
            str(val_path),
            seq_len=seq_len,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            mask_bos_in_loss=True,
            mask_last_label_in_loss=False,
            sampling_mode="deterministic",
            mask_repeated_eos_in_loss=True,
            require_release_manifest=True,
        )
        selected = len(dataset) if sample_limit == 0 else min(len(dataset), sample_limit)
        sampler = FixedSubsetSampler(dataset, num_samples=selected, seed=args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        teacher = evaluate_teacher_forced(
            model, loader, device=device, precision=precision
        )
        teacher["dataset"] = dataset.stats()
        teacher["selection"] = {
            "sampler": "FixedSubsetSampler",
            "seed": args.seed,
            "selected_blocks": selected,
            "available_blocks": len(dataset),
        }
        return teacher

    if args.val_dir:
        report["teacher_forced"] = evaluate_directory(args.val_dir, args.val_samples)

    if args.val_by_source_root:
        source_root = Path(args.val_by_source_root)
        if not source_root.is_dir():
            raise FileNotFoundError(f"val_by_source root does not exist: {source_root}")
        source_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
        if not source_dirs:
            raise FileNotFoundError(f"no per-source validation directories in {source_root}")
        report["teacher_forced_by_source"] = {
            source_dir.name: evaluate_directory(source_dir, args.val_samples_per_source)
            for source_dir in source_dirs
        }

    if args.generation_jsonl:
        cases = read_generation_cases(
            args.generation_jsonl, max_new_tokens=args.max_new_tokens
        )
        generation = run_generation_diagnostics(
            model,
            tokenizer,
            cases,
            device=device,
            precision=precision,
            max_seq_len=int(cfg.max_seq_len),
            max_new_tokens=args.max_new_tokens,
            default_system=args.default_system,
        )
        generation["configuration"] = {
            "input_path": str(Path(args.generation_jsonl).resolve()),
            "input_sha256": _sha256(args.generation_jsonl),
            "decoding": "greedy",
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 0,
            "semantic_stop_ids": [EOS_ID],
            "prompt_truncation": "forbidden",
        }
        report["generation"] = generation

    atomic_write_json(args.out, report)
    print(json.dumps({"out": str(Path(args.out)), "sections": sorted(set(report) - {"contract", "runtime", "checkpoint", "tokenizer", "schema_version"})}))


if __name__ == "__main__":
    main()
