# PetitGPT: End-to-End Small Language Model Training Pipeline

`PetitGPT` is a small-scale language model training project implemented in PyTorch. The goal is to develop, understand, and analyze a complete language model training and post-training pipeline under realistic compute constraints.

The project covers:

- tokenizer training and data preparation,
- pretraining a GPT-style model from scratch,
- continued pretraining,
- supervised fine-tuning,
- targeted distillation with an open-source teacher model,


The current model has approximately **137M parameters**. The entire training process can be run on a single RTX 4090 GPU. New MoE-based models are coming soon.

---

## 1. Project Scope

This project explores what can be achieved with a small GPT-style model trained from scratch and post-trained using curated instruction/code data.

The main focus areas are:

1. **End-to-end implementation**
   Building a full training pipeline rather than relying only on high-level frameworks.

2. **Small-model post-training**
   Understanding how CPT, SFT, and targeted distillation behave for a small model.

3. **Data quality and verification**
   Building filters, canonicalization scripts, AST-based code verification, unit tests, and repair loops.

4. **Failure analysis**
   Studying issues such as overfitting, prompt-like data contamination, visible teacher reasoning, EOS/boundary control, and mismatch between validation loss and sample quality.

---

## 2. Current Status

The project has completed several major stages:

- Tokenizer training,
- Pretraining on a mixed general/code/math corpus,
- Continued pretraining on a new general/code/math mix, with a particular focus on code + math,
- SFT on a general + code instruction mixture,
- Targeted distillation for simple Python function generation,
- General answer verification, AST + unit-test verification for code data,
- Multiple targeted distillation runs and checkpoint comparisons.




---

## 3. Model Overview

The main model is a GPT-style decoder-only Transformer, roughly 137M parameters.

A typical configuration used in the project is close to:

```text
n_layers   = 16
d_model    = 768
n_heads    = 12
d_ff       = 1920
seq_len    = 2048
vocab_size = 32000
RoPE       = enabled
precision  = bf16
```

The tokenizer is a 32k BPE tokenizer stored at:

```text
tokenizer/tokenizer.json
```

---

## 4. Repository Structure

The repository is organized approximately as follows:

```text
petitgpt/
├── tokenizer/
│   └── tokenizer.json
├── pretrain/
│   ├── build_pretrain_shards.py
│   ├── train_pretrain_with_bench.py
│   └── sample.py
├── sft/
│   └── train_sft.py
├── distill/
│   ├── train_distill.py
│   └── tools for generating synthetic data
├── datasets/
├── src/
│   └── model.py
├── outputs/
└── README.md
```



---

## 5. Pretraining

The pretraining stage uses a mixed corpus with general web text, educational text, Python code, Wikipedia-style text, math, and algebraic/proof-related data.

An example pretraining mix used in the project was approximately:

```text
FineWeb-Edu style text       ~56%
Python/code text             ~20%
Wikipedia-style text          ~8%
OpenWebMath-style text       ~10%
Proof/algebraic text          ~5%
Other math text               ~1%
```

A representative pretraining checkpoint later used for SFT was:

```text
outputs/pretrain_140m_v3_general_code/step_372000.pt
```

---

## 6. Supervised Fine-Tuning

The SFT stage used a mixture of general instruction data and code instruction data.

One important SFT mixture included sources such as:

```text
smol_smoltalk
codealpaca_20k
no_robots
viscode_200k
dolly_15k
alpaca_cleaned
```

A representative command was:

```bash
python sft/train_sft.py \
  --train_jsonl dataset/sft_mix_v6_general_code/train.jsonl \
  --val_jsonl dataset/sft_mix_v6_general_code/val.jsonl \
  --out_dir outputs/sft_v6_general_code \
  --tokenizer_path tokenizer/tokenizer.json \
  --init_from_pretrain outputs/pretrain_140m_v3_general_code/step_372000.pt \
  --seq_len 1024 \
  --micro_bsz 4 \
  --grad_accum 4 \
  --lr 1e-5 \
  --weight_decay 0.05 \
  --warmup_steps 400 \
  --max_steps 8000 \
  --precision bf16 \
  --eval_every 250 \
  --eval_batches 100 \
  --save_every 500 \
  --sample_every 500
```

The SFT checkpoint around step 3250-3500 was treated as the best base for targeted distillation.

---

## 7. Targeted Distillation

The targeted distillation stage was designed to improve simple Python coding behavior without fully redoing SFT.

The target simple Python families included:

```text
safe_divide
running_sum
running_max
dedup_preserve_order
count_words
lowercase_keys
flatten_once
reverse_string
is_prime
merge_counts
clamp
remove_none
```

The core code prompt bank was generated from deterministic task families and MBPP-style simple programming tasks.

A later verified code bank contained approximately:

```text
selected code examples: 719
train: 611
val: 54
holdout: 54
```

Most accepted code examples came from the curated core families, while a smaller number came from MBPP.

### 7.1 General Data Pipeline

The general distillation bank used a mixture of open-source prompts and teacher-generated answers.

The pipeline included:

1. exporting or normalizing open-source instruction prompts,
2. classifying prompts into families such as:

```text
explain_compare
rewrite_style
summary_bullets
email_message
```

3. generating answers with the teacher,
4. verifying formatting and quality,
5. repairing some failed answers,
6. building train/val/holdout splits.

A verified general bank contained approximately:

```text
train: 907 examples
val: 80 examples
holdout: 78 examples
```

A later cleaned mixture removed contaminated `template_paraphrase` examples that had been generated while teacher thinking mode was still enabled.

### 7.2 Code Verification

Code data was verified more strictly than general data.

The code verifier used:

- code block extraction,
- Python AST parsing,
- function name checks,
- banned node checks,
- recursion checks,
- line and AST-size limits,
- execution against unit tests in a restricted environment.

Typical requirements were:

```text
exactly one top-level function
correct function name
no imports/classes/exceptions/decorators
no extra top-level code
unit tests must pass
```

One bug found during verification was that normalizing generated text destroyed Python indentation before AST parsing. This caused many false `syntax_error` failures. The fix was to use a code-safe normalizer that preserves indentation.

Another issue was that the safe execution environment initially omitted some harmless Python builtins such as `isinstance`, `type`, `chr`, `ord`, and `reversed`. Adding these improved pass rate without opening unsafe operations such as `eval`, `exec`, `open`, or `__import__`.

### 7.3. Building the Targeted Distillation Mix

One targeted distillation mix used approximately:

```text
code train:    611
general train: 650-770 depending on cleaning
```

A representative mix before cleaning contained:

```text
train total: 1261
B_code: 611
A_general: 650
```

A cleaned no-template version contained approximately:

```text
train total: 1384
B_code: 611
A_general: 773
```

The general data was used mainly to reduce catastrophic narrowing toward code-only behavior, while the code data carried the targeted simple Python objective.

### 7.4. Targeted Distillation Training

A representative targeted distillation command was:

```bash
python distill/train_distill.py \
  --train_jsonl datasets/distill/targeted_distill_mix_v1/train.clean_no_template.jsonl \
  --val_jsonl datasets/distill/targeted_distill_mix_v1/val.clean_no_template.jsonl \
  --out_dir outputs/targeted_distill_v1_simple_code_clean_no_template \
  --tokenizer_path tokenizer/tokenizer.json \
  --init_from_pretrain outputs/sft_v6_general_code/step_003500.pt \
  --seq_len 1024 \
  --micro_bsz 4 \
  --grad_accum 4 \
  --lr 1e-6 \
  --weight_decay 0.01 \
  --warmup_steps 30 \
  --max_steps 800 \
  --precision bf16 \
  --eval_every 100 \
  --eval_batches 50 \
  --save_every 100 \
  --sample_every 100 \
  --sample_max_new_tokens 100 \
  --sample_temperature 0.1 \
  --loss_reduction example_mean \
  --refusal_downweight 1.0 \
  --debug_first_batch
```

Validation loss was useful but not sufficient. Several checkpoints with reasonable validation loss still showed poor generation behavior.

---


## 8. Next Steps

Future improvements could include:

- MoE-based models
- Reinforcement learning techniques for post-training

---


## 9. Disclaimer

This is a personal research-engineering project. It is not intended to compete with production LLMs. The value of the project lies in the implementation, experimentation, data pipeline, and failure analysis.
