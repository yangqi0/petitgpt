"""
Sanity checks for a trained Byte-Level BPE tokenizer.

This script performs three standard validations:
1. Encode–decode consistency on a sample sentence.
2. Average compression ratio (characters per token) on the training corpus.
3. Presence of special tokens required for language model training.

The goal is to ensure that the tokenizer is:
- Functionally correct
- Reasonably efficient
- Ready to be used in downstream LLM training
"""

import json
import random

from tokenizers import Tokenizer

# ---------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------
# The tokenizer was trained on the TinyStories corpus using Byte-Level BPE.
tok = Tokenizer.from_file("../tokenizer.json")

# ---------------------------------------------------------------------
# 1a. Encode–decode consistency check
# ---------------------------------------------------------------------
# A simple qualitative test to verify that:
#   decode(encode(text)) ≈ text
# Small whitespace differences are acceptable for byte-level tokenizers.
text = "One day, a little fish named Fin was swimming near the shore."
enc = tok.encode(text)

print("Original text:")
print(text)
print("\nEncoded tokens:")
print(enc.tokens)
print("\nToken IDs:")
print(enc.ids)
print("\nDecoded text:")
print(tok.decode(enc.ids))

enc2 = tok.encode("Hello.")
print("\nBOS/EOS automatically added:", enc2.tokens)

# ---------------------------------------------------------------------
# 1b. Extra qualitative checks (chat / markdown / structured text)
# ---------------------------------------------------------------------
# These strings are common in chat-style datasets (SFT/DPO) and are good
# for verifying that the tokenizer behaves reasonably on:
# - role prefixes (User/Assistant)
# - markdown headers & bullet lists
# - code fences and JSON snippets
extra_texts = [
    "User: Summarize the following.\n\nAssistant:",
    '### Task\n- Step 1: ...\n- Step 2: ...\n\n```json\n{"a":1,"b":[2,3]}\n```',
]

for i, t in enumerate(extra_texts, start=1):
    e = tok.encode(t)
    print(f"\nExtra test #{i} - Original text:")
    print(t)
    print(f"\nExtra test #{i} - Encoded tokens:")
    print(e.tokens)
    print(f"\nExtra test #{i} - Token IDs:")
    print(e.ids)
    print(f"\nExtra test #{i} - Decoded text:")
    print(tok.decode(e.ids))


# ---------------------------------------------------------------------
# 2. Compression ratio check (chars per token) on your corpus
# ---------------------------------------------------------------------
# Note: use the same distribution you trained the tokenizer on.

jsonl_paths = [
    "../../datasets/tokenization/fineweb_sample.jsonl",
    "../../datasets/tokenization/ultrachat_200k.jsonl",
    "../../datasets/tokenization/oasst1.jsonl",
    "../../datasets/tokenization/dolly_15k.jsonl",
]

all_texts = []
for p in jsonl_paths:
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text", "")
            if isinstance(t, str) and t.strip():
                all_texts.append(t)

# Sample a subset to keep runtime bounded
samples = random.sample(all_texts, min(2000, len(all_texts)))

total_chars = 0
total_tokens = 0
for s in samples:
    total_chars += len(s)
    total_tokens += len(tok.encode(s).ids)

print("\nAverage characters per token (mixed_v2 sample):")
print(total_chars / total_tokens)


# ---------------------------------------------------------------------
# 3. Special token availability (robust to naming conventions)
# ---------------------------------------------------------------------
# Different projects use different naming conventions, e.g.:
#   [BOS]/[EOS]  vs  <bos>/<eos>
# We check both to avoid false alarms.
candidates = {
    "bos": ["<bos>", "[BOS]", "<s>", "[CLS]"],
    "eos": ["<eos>", "[EOS]", "</s>", "[SEP]"],
    "pad": ["<pad>", "[PAD]"],
    "unk": ["<unk>", "[UNK]"],
}

print("\nSpecial token IDs (first match wins):")
for key, names in candidates.items():
    found = None
    found_name = None
    for name in names:
        tid = tok.token_to_id(name)
        if tid is not None:
            found = tid
            found_name = name
            break
    print(f"{key}: {found} ({found_name})")
