# PetitGPT: End-to-End Small Language Model Training Pipeline

`PetitGPT` is a small-scale language model training project implemented in PyTorch. The goal is to develop, understand, and analyze a complete language model training and post-training pipeline under realistic compute constraints.

The project covers:

- tokenizer training and data preparation,
- pretraining a GPT-style model from scratch,
- continued pretraining,
- supervised fine-tuning,
- targeted distillation with an open-source teacher model,
- DPO (Direct Preference Optimization) post-training.

The current dense model has approximately **137M parameters**. The entire training process can be run on a single RTX 4090 GPU. A **Mixture-of-Experts (MoE)** variant (`src/model_moe.py`) and a **Muon** optimizer (`src/optim.py`, now the default) have also been added — see [Section 3](#3-model-overview).

---

## 1. Project Scope

This project explores what can be achieved with a small GPT-style model trained from scratch and post-trained using curated instruction/code data.

The main focus areas are:

1. **End-to-end implementation**
   Building a full training pipeline rather than relying only on high-level frameworks.

2. **Small-model post-training**
   Understanding how CPT, SFT, targeted distillation, and DPO behave for a small model.

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
- Multiple targeted distillation runs and checkpoint comparisons,
- DPO preference post-training on open preference data.

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

The architecture is a LLaMA-style modernized GPT: pre-norm with RMSNorm, RoPE, SwiGLU MLPs, fused QKV projection using `F.scaled_dot_product_attention`, tied input/output embeddings, and GPT-2 depth-scaled residual initialization. See `src/model.py`.

### 3.1 Mixture-of-Experts variant

`src/model_moe.py` provides an MoE version of the same decoder (`MoEGPT` / `MoEConfig`). Everything except the feed-forward layer is shared with the dense model (attention, RMSNorm, RoPE, initialization); each block's single SwiGLU MLP is replaced by a **top-k routed mixture of SwiGLU experts**:

- **Router** — a linear layer produces per-expert logits; a softmax over all experts selects the top-`n_experts_per_tok`, whose gate weights are (optionally) renormalized. Tokens are dispatched to their selected experts with one batched matmul per expert.
- **Load balancing** — a DeepSeek/Switch-style auxiliary loss (≈1.0 when routing is balanced) is accumulated over all MoE layers on every forward pass. It is exposed both as `model.aux_loss` and via `model(input_ids, return_aux_loss=True)`, so the default `logits = model(input_ids)` call remains a drop-in replacement for the dense model. Training code adds `cfg.moe_aux_loss_coef * aux_loss` to the cross-entropy loss.
- **Optional refinements** — always-on *shared experts* (`n_shared_experts`, DeepSeek-MoE style) and *leading dense layers* (`n_dense_layers`, keeping a plain SwiGLU FFN in the first few blocks). `num_parameters()` reports total vs. per-token-active parameter counts. MoE checkpoints embed `asdict(MoEConfig)` and stay self-describing, exactly like the dense model.

### 3.2 RoPE implementation fix

An earlier version of `src/model.py` mixed two incompatible RoPE conventions: `_rotate_half` used the interleaved (GPT-J) pairing `(2i, 2i+1)`, while the cos/sin cache used the half-split (LLaMA/GPT-NeoX) layout `cat([freqs, freqs])`. The result was not an orthogonal rotation — it did not preserve norms and, more importantly, broke the defining property that `⟨R_t q, R_s k⟩` depends only on the relative offset `s − t`. This was fixed by making `_rotate_half` half-split (matching the cache), so the implementation now agrees element-for-element with the reference LLaMA RoPE. The fix is pinned by regression tests in `tests/test_model.py` (norm preservation, relative-position invariance, reference agreement) that failed before and pass after.

### 3.3 Optimizer: Muon (default) + AdamW

All training stages build their optimizer through `src/optim.py:build_optimizer`, selected with `--optimizer {muon,adamw}` (default **muon**):

- **Muon** applies Newton–Schulz-orthogonalized momentum updates to the hidden 2D weight matrices, while embeddings, the `lm_head`, RMSNorm gains, and MoE router gates keep an AdamW update. It uses Moonlight-style RMS matching (the orthogonalized update is scaled by `0.2·sqrt(max(fan_in, fan_out))`) so the AdamW-tuned `--lr` / `--weight_decay` transfer directly with no re-tuning. Both halves live in a single optimizer instance, so the checkpoint schema is unchanged.
- **AdamW** was also corrected: weight decay is applied only to matrices/embeddings (never to 1-D norm gains and biases), with betas `(0.9, 0.95)` and the fused CUDA kernel.

Resuming a checkpoint whose optimizer state does not match the current `--optimizer` prints a warning and continues with fresh optimizer state (model weights still load).

---

## 4. Repository Structure

The repository is organized approximately as follows:

```text
petitgpt/
├── tokenizer/                 # BPE tokenizer + training/sanity-check scripts
│   └── tokenizer.json
├── configs/                   # declarative SFT mix configs (sft_mix_*.yaml)
├── pretrain/                  # shard building, pretraining, sampling, bench eval
│   ├── build_pretrain_shards.py
│   ├── train_pretrain_with_bench.py
│   ├── eval_bench_v5.py
│   └── sample.py
├── sft/                       # SFT mix preparation + training
│   ├── prepare_sft_mix_split_local.py
│   └── train_sft.py
├── distill/                   # teacher generation, verification, mix building, training
│   ├── train_distill.py
│   ├── code_verify_v1.py
│   ├── general_verify_v1.py
│   └── ...                    # teacher generation + data pipeline tools
├── dpo/                       # preference post-training
│   ├── prepare_dpo_data.py
│   └── dpo.py
├── src/
│   ├── model.py               # dense GPT model definition
│   ├── model_moe.py           # Mixture-of-Experts variant (MoEGPT / MoEConfig)
│   └── optim.py               # Muon + AdamW optimizer factory (build_optimizer)
├── tests/                     # pytest suite (CPU-only, runs in seconds)
├── .github/workflows/ci.yml   # GitHub Actions: ruff + pytest on CPU
├── eval/                      # benchmark eval results
├── datasets/                  # training data (local only, gitignored)
├── outputs/                   # checkpoints (local only, gitignored)
└── README.md
```

---

## 5. Setup

Python 3.10+ is required. Install the dependencies with:

```bash
pip install -r requirements.txt
```

Then install the CUDA build of `torch` matching your system separately, for example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Training scripts expect a CUDA GPU and can be run from anywhere as `python <stage>/<script>.py ...` — no package install is needed.

---

## 6. Pretraining

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

## 7. Supervised Fine-Tuning

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

## 8. Targeted Distillation

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

### 8.1 General Data Pipeline

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

### 8.2 Code Verification

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

### 8.3 Building the Targeted Distillation Mix

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

### 8.4 Targeted Distillation Training

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

## 9. DPO

The DPO stage applies preference post-training on top of an SFT or distillation checkpoint, using the standard DPO loss with a frozen reference model (a deep copy of the initial policy by default).

Preference data is built from open preference datasets:

```text
UltraFeedback (binarized)
Orca DPO pairs
Anthropic hh-rlhf (harmless subset)
```

`dpo/prepare_dpo_data.py` filters pairs by prompt/completion token counts (computed to exactly match training-time encoding) and writes train/val JSONL files with `messages` plus `chosen`/`rejected` completions:

```bash
python dpo/prepare_dpo_data.py \
  --tokenizer_path tokenizer/tokenizer.json \
  --out_dir datasets/dpo
```

A representative training command was:

```bash
python dpo/dpo.py \
  --train_jsonl datasets/dpo/train.jsonl \
  --val_jsonl datasets/dpo/val.jsonl \
  --out_dir outputs/dpo_run \
  --tokenizer_path tokenizer/tokenizer.json \
  --init_ckpt outputs/sft_v6_general_code/step_003500.pt \
  --beta 0.1
```

Training logs the implicit reward margin and preference accuracy alongside the loss, which helps catch runs where the loss decreases without the policy actually separating chosen from rejected completions.

---

## 10. Next Steps

Future improvements could include:

- MoE-based models,
- online reinforcement learning for post-training (e.g., PPO/GRPO-style methods).

---

## 11. Disclaimer

This is a personal research-engineering project. It is not intended to compete with production LLMs. The value of the project lies in the implementation, experimentation, data pipeline, and failure analysis.
