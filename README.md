# PetitGPT

**A 124.6M-parameter language-model research project, from tokenizer training and pretraining to post-training, evaluation, and native inference.**

**Author:** Yang Qi

[Model and tokenizer](https://huggingface.co/yqi0/petitgpt) · [Tokenizer files](tokenizer/README.md) · [Technical report](docs/petitgpt-v1/TECHNICAL_REPORT.md) · [Reproducibility](TRAINING_AND_REPRODUCIBILITY.md) · [Run guide](docs/petitgpt-v1/RUN_GUIDE.md) · [Model card](docs/petitgpt-v1/MODEL_CARD.md)

PetitGPT explores what can be learned by building and evaluating a small language model under a limited training budget. The project includes a custom byte-level BPE tokenizer, approximately 13 billion pretraining tokens, controlled post-training experiments, and evaluations that distinguish reference-answer fit from complete generated-answer correctness.

The released **research-v1** checkpoint is **alpha075**.

## At a glance

| Component | Released configuration |
|---|---|
| Unique parameters | 124,635,456 |
| Transformer | 30 layers; hidden width 576; feed-forward width 1,536 |
| Attention | Grouped-query attention: 9 query heads / 3 key-value heads |
| Architecture | RoPE, RMSNorm, SwiGLU, tied input/output embeddings |
| Tokenizer | Custom 32,000-token byte-level BPE |
| Context | 2,048 tokens, including the generation budget |
| Pretraining | Approximately 13B tokens; one NVIDIA RTX 4090 |
| Released inference | Native PyTorch on CUDA; stored FP32 weights |

![PetitGPT reference validation loss across Stage A and Stage B, with a Stage B detail panel](docs/petitgpt-v1/figures/pretraining_validation.png)

*Reference validation loss during pretraining: all 10 recorded evaluations, with no smoothing; lines connect observations within each stage. The right panel enlarges Stage B using a different vertical scale. See the [data and interpretation](docs/petitgpt-v1/TECHNICAL_REPORT.md#44-observed-results).*

## What this project investigates

The central question is whether the model completes new tasks correctly while retaining earlier capabilities.

The experiments cover supervised fine-tuning, preference optimization, response distillation, a separate shared-tokenizer soft-distillation lab, LoRA adaptation, loss allocation, and parameter interpolation. Several runs improved fitting or individual tasks without delivering a balanced improvement over the selected reference. Those results are retained rather than presented as successful upgrades.

**Released weights and experimental coverage are different things.** In the released model's training path, **P2** is supervised fine-tuning on concise instructions, and **P3** is a further adaptation stage on basic instruction tasks with replay examples from earlier instruction data:

```text
Base step_049590 → P2 step750 → P3 step320
                                      ↓
alpha075 = P2 + 0.75 × (P3_step320 − P2)
```

Later DPO, response-distillation, LoRA, and unified Base-SFT updates are **not** in alpha075. The soft-logit lab used separate external models. See the [report](docs/petitgpt-v1/TECHNICAL_REPORT.md) for each experiment's initialization and evaluation scope.

## Results at a glance

The benchmark picture is mixed rather than uniformly favorable. Under the project's fixed protocol, alpha075 leads both SmolLM baselines on ARC-Easy and ARC-Challenge, trails both on PIQA and HellaSwag, and falls between SmolLM and SmolLM2 on IFEval.

**Competitive ARC results on a small pretraining budget.** PetitGPT was pretrained on approximately 13B positions using one RTX 4090. The base models behind the instruct baselines were pretrained on 600B tokens for [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M#training) and 2T tokens for [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M#training); both official model cards report 64 H100 GPUs. PetitGPT's ARC results suggest promising data efficiency under this evaluation protocol. Different tokenizers, datasets and post-training procedures make this an observational comparison; total training FLOPs, GPU-hours and costs were not compared.

All comparison results below were measured in this project on pinned model revisions; the SmolLM and SmolLM2 numbers are our own evaluations under the protocols described here, not scores copied from their official model cards.

### Zero-shot likelihood benchmarks

| Model | ARC-Easy acc / acc_norm | ARC-Challenge acc / acc_norm | PIQA acc / acc_norm | HellaSwag acc / acc_norm |
|---|---:|---:|---:|---:|
| **petitgpt-alpha075** | **57.74% / 52.36%** | **28.16% / 32.68%** | 63.49% / 62.30% | 31.28% / 35.60% |
| SmolLM-135M-Instruct | 49.24% / 43.48% | 25.43% / 27.22% | **67.08% / 67.25%** | 34.60% / 41.96% |
| SmolLM2-135M-Instruct | 54.00% / 48.82% | 25.94% / 27.73% | 66.70% / 66.76% | **35.02% / 42.90%** |

All three models are evaluated on the same rows using zero-shot completion likelihood. `acc_norm` uses character-length normalization under this project's protocol and may differ from externally reported scores. ARC-Easy and PIQA were used during project development; no contamination audit was performed.

<details>
<summary>Likelihood evaluation protocol and dataset sizes</summary>

ARC-Easy: 2,376 test rows; ARC-Challenge: 1,172 test rows; PIQA: 1,838 validation rows; HellaSwag: 10,042 validation rows. All four tasks score raw completion likelihood zero-shot on the same rows for all three models, with no chat template, no BOS/EOS insertion, FP32 parameters and forward, batch size 1, no sampling, and first-argmax tie-breaking.

`acc_norm` is the project-protocol variant: the continuation likelihood divided by the Unicode-character length of the original candidate text (for HellaSwag, the pinned task-preprocessed ending without its leading delimiter), **not** its token count, so it is not necessarily identical to an externally reported `acc_norm`. The evaluator is protocol-compatible with pinned harness code rather than a full installed-harness run.

</details>

### Instruction following: IFEval

| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose | 1,280-token cap hits |
|---|---:|---:|---:|---:|---:|
| petitgpt-alpha075 | 17.19% (93/541) | 28.54% (238/834) | 17.74% (96/541) | 29.98% (250/834) | 40/541 |
| SmolLM-135M-Instruct | 10.35% (56/541) | 21.82% (182/834) | 12.01% (65/541) | 24.10% (201/834) | 290/541 |
| **SmolLM2-135M-Instruct** | **21.63% (117/541)** | **35.85% (299/834)** | **22.55% (122/541)** | **37.29% (311/834)** | 233/541 |

IFEval scores generated responses to the same 541 prompts, using each model's own chat template and greedy decoding. Outputs are capped at 1,280 new tokens and scored as-is, including those that hit the cap. SmolLM2 leads all four instruction-following metrics, followed by alpha075 and SmolLM.

<details>
<summary>IFEval formatting, generation and scoring protocol</summary>

IFEval is a generative evaluation and is not part of the likelihood protocol above. Each model received the same official IFEval user prompt (541 prompts, 834 instructions, 25 instruction types) as a single user message, formatted by its own native chat formatter: alpha075's released formatter with no default system message, and the pinned SmolLM and SmolLM2 tokenizer chat templates, where SmolLM2's template inserts its own default system text. The formatted token inputs therefore differ across models; this is a native-chat comparison, not an identical-token-input experiment.

Generation was zero-shot and greedy with no sampling and `max_new_tokens=1280`, with no added system prompt or few-shot messages. Responses were scored by the pinned IFEval strict/loose programmatic verifier with no LLM judge; no prompts were dropped or truncated. Responses that reached the 1,280-token cap are scored as-is, and the cap-hit counts are reported without any claim about what a larger budget would change.

</details>

**Benchmark scores are not chat reliability.** In the separate, versioned full-answer review, alpha075 produced a correct Python interface on 42/46 prompts but a correct complete answer on 0/46. Ordinary QA, faithful rewriting, and context-dependent instructions also remain limited. The report's [success, failure, and scoring-boundary cases](docs/petitgpt-v1/TECHNICAL_REPORT.md#85-qualitative-cases-successes-and-failures) show what these judgments mean, with content, format, interface, and finite test evidence kept separate.

**Evaluation source:** The original ARC-Easy/PIQA FP32 V2 driver is [evaluate_v2.py](recipes/research-v1/sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime/evaluate_v2.py), with scoring and aggregation helpers in its [runtime directory](recipes/research-v1/sources/benchmark/runs/petitgpt_frozen_reference_public_benchmark_fp32_v2/runtime). The ARC-Challenge/HellaSwag and final IFEval V2 programs are in the [benchmark extension source directory](recipes/research-v1/sources/benchmark_extension/README.md). That directory's README maps all five tasks to their execution code and describes provenance and required runtime inputs; the snapshots do not include all data and evidence needed for replay.

## Run the released model

**A CUDA GPU is required by the released CLI.** Use an environment compatible with the [tested dependencies and run guide](docs/petitgpt-v1/RUN_GUIDE.md). The commands below assume the Hugging Face CLI is already available.

Download the complete native bundle at the published research-v1 revision:

```bash
hf download yqi0/petitgpt \
  --revision 7bf3df96e6880b242b2907d1e68093435feacd75 \
  --local-dir ./artifacts/petitgpt-research-v1

(cd ./artifacts/petitgpt-research-v1 && sha256sum -c SHA256SUMS)

python ./artifacts/petitgpt-research-v1/inference.py \
  --model-directory ./artifacts/petitgpt-research-v1 \
  --prompt "Say hello in one sentence." \
  --profile bf16_native \
  --max-new-tokens 32
```

Preserve the downloaded `src/` directory. The CLI returns the generated token IDs, raw text including a terminal EOS when present, and the stop reason.

Greedy decoding and two numerical profiles are supported: `bf16_native` and `fp32_math`. They are not asserted to generate identical answers. Context overflow is rejected rather than silently truncated. For multi-turn messages and the Python API, see the [run guide](docs/petitgpt-v1/RUN_GUIDE.md).

The published format is **native PyTorch**, not a Transformers `AutoModel` package. GGUF, ONNX, vLLM, llama.cpp, and CPU inference are not implemented or validated by this release.

## Intended use and limitations

PetitGPT is intended for research and experimentation. Verify generated factual claims and review generated code before use. The model has not been validated for production or high-stakes use, tool use, or broad multilingual capability; see the [model card](docs/petitgpt-v1/MODEL_CARD.md#intended-use-and-use-it-is-not-intended-for) for its intended use and limitations.

## Train and evaluate

We pretrained PetitGPT, applied supervised fine-tuning, and explored DPO and response distillation in separate experimental branches. For more details, please refer to [Technical report](docs/petitgpt-v1/TECHNICAL_REPORT.md). This repository preserves the implementations and recorded results of that work.

To train and evaluate a new model on your own prepared data using the research-v1 method, follow the [practical manual](TRAINING_AND_REPRODUCIBILITY.md):
[tokenizer contract](tokenizer/README.md) → [prepared inputs](recipes/research-v1/PREPARED_INPUTS.md)
→ [pretrain A/B](TRAINING_AND_REPRODUCIBILITY.md#pretrain-ab)
→ [P2/P3 and fixed blend](TRAINING_AND_REPRODUCIBILITY.md#posttraining)
→ [native export and likelihood evaluation](TRAINING_AND_REPRODUCIBILITY.md#export-and-evaluate).

The entry point is `recipes/research-v1/reader.py --policy new-run`. The public reader commands were added after the original runs as reproduction entry points; they call recovered stage-specific code. The manual documents the original launch scripts, required inputs, validation status, and execution limits.

For experiments beyond that workflow, the top-level [`pretrain/`](pretrain/), [`sft/`](sft/), [`dpo/`](dpo/), and [`distill/`](distill/) directories provide reusable research tools. These implementations may differ from the stage-specific versions used by the recipes.

Using different data produces new checkpoints and results, rather than reproducing the published alpha075 exactly.

## Navigate the repository

See the [repository guide](docs/REPOSITORY_GUIDE.md) for reading entry points, directory roles, and the distinction between released code and historical research artifacts.

| Location | Purpose |
|---|---|
| [`inference_native/`](inference_native/) | Released native inference code, without model weights |
| [`src/`](src/) | Shared model and training components |
| [`pretrain/`](pretrain/), [`tokenizer/`](tokenizer/) | Pretraining and tokenizer tooling |
| [`sft/`](sft/), [`distill/`](distill/README.md), [`dpo/`](dpo/) | Reusable post-training implementations; distill has a tool/status index; exact P2 uses the recovered recipe |
| [`experiments/`](experiments/README.md) | Measured research index and implementation-only GRPO |
| [`recipes/research-v1/`](recipes/research-v1/README.md), [`configs/research-v1/`](configs/research-v1/README.md) | New-run reader workflows, historical replay interfaces, recovered sources and effective settings |
| [`legacy/`](legacy/README.md) | Retained experiment summaries/configs and a short Git-history guide; obsolete snapshots removed |
| [`scripts/`](scripts/) | Reusable helpers |
| [`tests/`](tests/) | Repository checks |
| [`docs/petitgpt-v1/`](docs/petitgpt-v1/) | Current report, versioned results, model card, and run guide |
| [`HISTORICAL_README.md`](docs/petitgpt-v1/HISTORICAL_README.md) | Earlier project narrative; not the released model's specification |

The repository retains earlier experiments. Older scripts and result files must not be mistaken for the current release recipe. Private run records, complete training data, optimizer checkpoints, and some evaluation artifacts are not distributed in this public repository.

## Licensing, sources, and citation

Code, model, and tokenizer are offered under [Apache-2.0](LICENSE). Reports and documentation are licensed under [CC BY 4.0](docs/petitgpt-v1/DOCUMENTATION_LICENSE.md). Third-party terms remain applicable.

The [source notice](SOURCE_NOTICE.md) and [third-party notices](THIRD_PARTY_NOTICES.md) document provenance, differing component declarations, and historical-version limits. Disclosure is not a claim of legal clearance, and the model license does not relicense the original data.

For citation metadata, please use [`CITATION.cff`](CITATION.cff).
