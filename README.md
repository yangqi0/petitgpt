# petitgpt: End-to-End Small Language Model Training Pipeline

`petitgpt` is a small-scale language model training project implemented in PyTorch. The goal is not to build a frontier-level assistant, but to develop and analyze a complete language model training and post-training pipeline under realistic compute constraints.

The project covers:

- tokenizer training and data preparation,
- pretraining a GPT-style model from scratch,
- supervised fine-tuning,
- targeted distillation with an open-source teacher model,
- teacher generation through vLLM,
- data verification and filtering,
- checkpoint selection,
- sampling and failure analysis.

The current model is approximately **137M parameters**. It remains limited as a general assistant, but the project demonstrates an end-to-end research-engineering workflow for small language model training.

---

## 1. Project Scope

This project explores what can be achieved with a small GPT-style model trained from scratch and post-trained using curated instruction/code data.

The main focus areas are:

1. **End-to-end implementation**  
   Building a full training pipeline rather than relying only on high-level frameworks.

2. **Small-model post-training**  
   Understanding how SFT and targeted distillation behave for a small 137M model.

3. **Data quality and verification**  
   Building filters, canonicalization scripts, AST-based code verification, unit tests, and repair loops.

4. **Failure analysis**  
   Studying issues such as overfitting, prompt-like data contamination, visible teacher reasoning, EOS/boundary control, and mismatch between validation loss and sample quality.

---

## 2. Current Status

The project has completed several major stages:

- pretraining from a mixed general/code/math corpus,
- SFT on a general + code instruction mixture,
- targeted distillation for simple Python function generation,
- vLLM teacher generation with Qwen-family models,
- general answer verification,
- AST + unit-test verification for code data,
- multiple targeted distillation runs and checkpoint comparisons.

The best SFT checkpoint used as the base for targeted distillation was approximately:

```text
outputs/sft_v6_general_code/step_003500.pt
```

A later targeted distillation run improved some local simple-code behavior, but generation quality remained unstable. In particular, the model often generated a reasonable function prefix but failed to stop cleanly, continuing with examples, print statements, or repeated text.

Therefore, the current project should be viewed as:

```text
A small language model training and post-training pipeline demonstration,
not a production-quality chatbot.
```

---

## 3. Model Overview

The main model is a GPT-style decoder-only Transformer, roughly 137M parameters.

A typical configuration used in the project is close to:

```text
n_layers   = 12
d_model    = 768
n_heads    = 12
d_ff       = 3072
seq_len    = 1024 or 2048
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
│   ├── general_*.py
│   ├── code_*.py
│   ├── build_targeted_distill_mix_v1.py
│   └── *_utils.py
├── datasets/
│   ├── pretrain_*
│   ├── distill/
│   └── ...
├── outputs/
│   ├── pretrain_*
│   ├── sft_*
│   └── targeted_distill_*
└── README.md
```

The exact structure may vary across experiments, but the main stages are separated into `pretrain/`, `sft/`, and `distill/`.

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

---

## 8. Teacher Generation with vLLM

The teacher model was served through vLLM using an OpenAI-compatible API endpoint, for example:

```text
http://127.0.0.1:8000/v1
```

A critical issue encountered during the project was that Qwen-family thinking models may return visible reasoning such as:

```text
Thinking Process:
1. Analyze the Request:
...
```

This contaminated generated data when not disabled.

The fix was to pass:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

in each vLLM OpenAI-compatible request, or to configure vLLM with equivalent server-level defaults.

This was an important debugging lesson: **teacher output must be inspected before it is trusted as training data**.

---

## 9. General Data Pipeline

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

---

## 10. Code Verification

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

---

## 11. Building the Targeted Distillation Mix

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

---

## 12. Targeted Distillation Training

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

## 13. Observed Failure Modes

The project produced several important engineering lessons.

### 13.1 Visible teacher reasoning contamination

When Qwen thinking mode was not disabled, generated data included visible reasoning text. Some of this entered the training pipeline and caused prompt-like or reasoning-like model outputs.

### 13.2 Prompt-like assistant contamination

Some template paraphrase data accidentally treated rewritten prompts as assistant answers. This caused outputs such as:

```text
Please rewrite the following...
Return only the rewritten text...
Format the output...
```

These examples had to be removed from the final clean mix.

### 13.3 EOS and answer-boundary instability

The model often generated a reasonable function prefix, but then continued with examples, print statements, comments, or repeated output.

Example failure pattern:

```python
def running_sum(nums):
    ...
    return result

print(running_sum(...))
# Output: ...
```

This suggests that EOS supervision, answer-boundary modeling, and decoding stop rules are important remaining bottlenecks.

### 13.4 Validation loss versus sample quality

Validation loss sometimes improved while sample quality remained poor. Manual sample inspection was essential for checkpoint selection.

### 13.5 Small-model limitations

A 137M model can learn local patterns, but robust instruction following and clean stopping behavior remain difficult, especially after small-data post-training.

---

## 14. Current Limitations

The current model remains limited.

Known limitations include:

- unstable answer boundaries,
- tendency to continue after a correct function,
- occasional wrong function signatures,
- poor general instruction following compared with modern LLMs,
- sensitivity to data contamination,
- mismatch between teacher-forcing loss and generation quality,
- limited generalization from small targeted distillation sets.

The project is therefore best understood as a **training pipeline and failure-analysis project**, not as a finished assistant model.

---

## 15. Lessons Learned

Key lessons from the project:

1. Data quality matters more than raw data quantity for small-model post-training.
2. Teacher outputs must be inspected and filtered carefully.
3. Thinking-mode teachers can silently contaminate data if visible reasoning is not disabled.
4. Code verification should use AST checks and unit tests, but must preserve indentation and allow safe builtins.
5. Validation loss alone is not enough for checkpoint selection.
6. Small models can learn local code templates but still fail at answer-boundary control.
7. Honest failure analysis is often more valuable than an over-optimistic demo.

---

## 16. Suggested Next Steps

Future improvements could include:

- verifying EOS supervision in the training script,
- adding explicit answer-boundary examples,
- improving sampling stop rules for code generation,
- rebuilding template paraphrase data after disabling teacher thinking mode,
- improving MBPP canonicalization so prompts and `entry_point` are aligned,
- adding a small code-only boundary distillation run,
- creating a lightweight benchmark for the target simple Python families,
- comparing checkpoints using executable code tests rather than only text samples.

---

## 17. Resume Summary

A concise resume description could be:

```text
Built an end-to-end GPT-style small language model training pipeline in PyTorch, covering tokenizer training, pretraining, SFT, targeted distillation, vLLM teacher generation, AST/unit-test verification for Python code data, checkpoint analysis, and systematic debugging of post-training failure modes such as data contamination and answer-boundary instability.
```

---

## 18. Disclaimer

This is a personal research-engineering project. It is not intended to compete with production LLMs. The value of the project lies in the implementation, experimentation, data pipeline, and failure analysis.
