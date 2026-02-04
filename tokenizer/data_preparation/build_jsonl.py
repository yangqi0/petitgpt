import argparse
import json
from pathlib import Path
import random

from datasets import load_dataset
from tqdm import tqdm


def normalize_text(s: str) -> str:
    """Normalize whitespace to make tokenizer training more stable."""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = " ".join(s.split())
    return s


def write_jsonl(ex_iter, out_path: Path, max_examples: int, seed: int, get_text_fn):
    """
    Stream examples, extract text, normalize, and write JSONL lines:
      {"text": "..."}
    """
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(ex_iter, desc=f"writing {out_path.name}"):
            if n >= max_examples:
                break
            text = get_text_fn(ex)
            if not isinstance(text, str):
                continue
            text = normalize_text(text)
            if not text:
                continue

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output directory for JSONL shards.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility).")

    # Recommended sizes for tokenizer training (adjust based on your disk/time)
    ap.add_argument(
        "--n_fineweb", type=int, default=500_000, help="Number of FineWeb samples to export."
    )
    ap.add_argument(
        "--n_ultrachat", type=int, default=200_000, help="Number of UltraChat samples to export."
    )
    ap.add_argument(
        "--n_oasst1", type=int, default=80_000, help="Number of OpenAssistant samples to export."
    )
    ap.add_argument(
        "--n_dolly", type=int, default=15_000, help="Number of Dolly samples to export."
    )

    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    # -----------------------------
    # 1) FineWeb (general English web)
    # -----------------------------
    # FineWeb provides multiple builder configs (e.g., sample-10BT, sample-100BT, ...).
    # We default to a small sample config for tokenizer training.
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    n1 = write_jsonl(
        fineweb,
        out_dir / "fineweb_sample.jsonl",
        max_examples=args.n_fineweb,
        seed=args.seed,
        get_text_fn=lambda ex: ex.get("text", ""),
    )

    # -----------------------------
    # 2) UltraChat 200k (chat-style)
    # -----------------------------
    # UltraChat provides multiple splits. For chat/SFT-style text, use "train_sft".
    ultrachat = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )

    def ultrachat_to_text(ex):
        # Typical structure is a list of messages; we concatenate them in a simple "role: content" style.
        # This is tokenizer-training-only; SFT formatting can be handled later with a chat template.
        msgs = ex.get("messages", None)
        if isinstance(msgs, list) and msgs:
            parts = []
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "")
                if isinstance(content, str) and content.strip():
                    parts.append(f"{role}: {content.strip()}")
            return "\n".join(parts)
        # Fallback
        return ex.get("text", "")

    n2 = write_jsonl(
        ultrachat,
        out_dir / "ultrachat_200k.jsonl",
        max_examples=args.n_ultrachat,
        seed=args.seed,
        get_text_fn=ultrachat_to_text,
    )

    # -----------------------------
    # 3) OpenAssistant oasst1 (human chat trees)
    # -----------------------------
    oasst = load_dataset(
        "OpenAssistant/oasst1", split="train", streaming=True
    )  # :contentReference[oaicite:9]{index=9}

    def oasst_to_text(ex):
        # Each row is a single message; we use a light format to expose tokenizer to chat tokens.
        role = ex.get("role", "")
        text = ex.get("text", "")
        if isinstance(text, str) and text.strip():
            return f"{role}: {text.strip()}"
        return ""

    n3 = write_jsonl(
        oasst,
        out_dir / "oasst1.jsonl",
        max_examples=args.n_oasst1,
        seed=args.seed,
        get_text_fn=oasst_to_text,
    )

    # -----------------------------
    # 4) Dolly 15k (instruction variety, small)
    # -----------------------------
    dolly = load_dataset(
        "databricks/databricks-dolly-15k", split="train", streaming=True
    )  # :contentReference[oaicite:10]{index=10}

    def dolly_to_text(ex):
        instr = ex.get("instruction", "")
        ctx = ex.get("context", "")
        resp = ex.get("response", "")
        parts = []
        if isinstance(instr, str) and instr.strip():
            parts.append(f"Instruction: {instr.strip()}")
        if isinstance(ctx, str) and ctx.strip():
            parts.append(f"Context: {ctx.strip()}")
        if isinstance(resp, str) and resp.strip():
            parts.append(f"Answer: {resp.strip()}")
        return "\n".join(parts)

    n4 = write_jsonl(
        dolly,
        out_dir / "dolly_15k.jsonl",
        max_examples=args.n_dolly,
        seed=args.seed,
        get_text_fn=dolly_to_text,
    )

    print("Done. Exported examples:")
    print("  FineWeb:", n1)
    print("  UltraChat:", n2)
    print("  OASST1:", n3)
    print("  Dolly:", n4)
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
