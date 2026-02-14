import argparse
from collections.abc import Iterator
import json
import os

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def iter_texts(paths: list[str], fields: list[str]) -> Iterator[str]:
    """
    Iterate over one or more JSONL files and yield text samples line by line.

    Each line in the JSONL file is expected to be a valid JSON object.
    This function extracts text from the specified `fields` and concatenates
    them with newline characters.

    Parameters
    ----------
    paths : list[str]
        List of paths to JSONL files.
    fields : list[str]
        List of field names to extract from each JSON object.
        Multiple fields will be concatenated (e.g., "prompt" + "response").

    Yields
    ------
    str
        A single text sample constructed from the specified fields.
    """
    for p in paths:
        # Open each JSONL file
        with open(p, encoding="utf-8") as f:
            # Read the file line by line (1-based line numbers for debugging)
            for lineno, line in enumerate(f, start=1):
                # Remove leading/trailing whitespace
                line = line.strip()
                if not line:
                    continue

                # Parse the JSON object from the current line
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed JSON lines (optional: log file and line number)
                    # print(f"Error parsing line {lineno} in {p}: {e}")
                    continue

                parts = []
                # Extract and clean text from the specified fields
                for k in fields:
                    v = obj.get(k, "")
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())

                # Yield the concatenated text if at least one field is non-empty
                if parts:
                    yield "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="Input JSONL files.")
    ap.add_argument(
        "--fields", nargs="+", default=["text"], help="JSON fields to concatenate as text."
    )
    ap.add_argument("--vocab_size", type=int, default=16000, help="Target vocabulary size.")
    ap.add_argument("--out_dir", type=str, default="../", help="Output directory.")
    ap.add_argument(
        "--min_freq", type=int, default=2, help="Minimum frequency for a token to be included."
    )
    ap.add_argument(
        "--strict_special_ids",
        action="store_true",
        help="If set, enforce [PAD]=0,[UNK]=1,[BOS]=2,[EOS]=3 (useful for strict reproducibility).",
    )
    args = ap.parse_args()

    # Create output directory (no error if it already exists)
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Build a Byte-Level BPE tokenizer (robust for English/symbols/code)
    # ------------------------------------------------------------
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Special tokens required by downstream training/inference
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # ------------------------------------------------------------
    # Trainer configuration
    # - initial_alphabet improves coverage for byte-level characters/symbols
    # - min_frequency filters extremely rare patterns (often noise)
    # ------------------------------------------------------------
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # Train from a streaming iterator (memory-efficient for large corpora)
    tokenizer.train_from_iterator(iter_texts(args.data, args.fields), trainer=trainer)

    # ------------------------------------------------------------
    # Sanity checks: special tokens must exist and should have unique IDs
    # ------------------------------------------------------------
    tok2id = {t: tokenizer.token_to_id(t) for t in special_tokens}
    for t, tid in tok2id.items():
        assert tid is not None, f"Missing special token in vocab: {t}"
    assert len(set(tok2id.values())) == len(tok2id), f"Special token IDs are not unique: {tok2id}"

    # Optional: enforce a strict, reproducible ID convention
    # (Only enable this if the downstream training code assumes fixed IDs.)
    if args.strict_special_ids:
        assert tok2id["[PAD]"] == 0, f"Expected [PAD]=0, got {tok2id['[PAD]']}"
        assert tok2id["[UNK]"] == 1, f"Expected [UNK]=1, got {tok2id['[UNK]']}"
        assert tok2id["[BOS]"] == 2, f"Expected [BOS]=2, got {tok2id['[BOS]']}"
        assert tok2id["[EOS]"] == 3, f"Expected [EOS]=3, got {tok2id['[EOS]']}"
    else:
        # Helpful debug output (does not fail if IDs differ)
        print("Special token IDs:", tok2id)

    # ------------------------------------------------------------
    # Post-processing: automatically add BOS/EOS for consistent train/infer behavior
    # ------------------------------------------------------------
    # tokenizer.post_processor = TemplateProcessing(
    #     single="[BOS] $A [EOS]",
    #     pair="[BOS] $A [EOS] $B:1 [EOS]:1",
    #     special_tokens=[
    #         ("[BOS]", tokenizer.token_to_id("[BOS]")),
    #         ("[EOS]", tokenizer.token_to_id("[EOS]")),
    #     ],
    # )

    # Save tokenizer.json (the core artifact for a fast tokenizer)
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # Minimal files for Transformers compatibility
    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 2048,
                "padding_side": "right",
                "truncation_side": "right",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(args.out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("trained tokenizer saved to:", args.out_dir)
    print("vocab_size =", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
