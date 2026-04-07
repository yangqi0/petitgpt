# Qwen teacher pipeline on RunPod

This setup is for a **hybrid synthetic distill pipeline**:

- let a local open-weight teacher expand **general prompts**
- let the same teacher expand **code prompt wording**
- keep **hidden tests** for code so you can still do semantic verification
- answer all prompts through a local OpenAI-compatible endpoint
- build a canonical dataset for your existing SFT trainer

## Why hybrid instead of fully free-form code generation?

For general text, free-form synthetic prompt generation is fine.

For code, fully free-form question generation is risky because later verification becomes weak.
It is much safer to keep:
- known function families
- known entrypoints
- known hidden tests

and let the teacher diversify the **wording**, not the **ground truth**.

## Recommended default model on a single 24 GB GPU pod

Start with:
- `Qwen/Qwen3.5-4B`

This is a practical first default for a single RunPod 4090-style pod.
You can try larger models later if VRAM and throughput are acceptable.

## Files

- `serve_qwen35_vllm_runpod.sh`
- `bootstrap_general_prompts_with_teacher.py`
- `expand_code_prompts_with_teacher.py`
- `run_qwen35_hybrid_teacher_pipeline.sh`

## Helper files expected in the same directory

These were created earlier:
- `stage2_teacher_generate_open.py`
- `verify_code_candidates_with_tests.py`
- `build_general_code_teacher_dataset.py`

## Typical usage

Terminal A:
```bash
bash serve_qwen35_vllm_runpod.sh
```

Terminal B:
```bash
bash run_qwen35_hybrid_teacher_pipeline.sh
```

## First-run advice

Do **not** start with 5k + 5k immediately.

Start smaller:
- general prompts: 300 to 500
- code prompts: 300 to 500

Check:
- code verification pass rate
- duplicate prompt rate
- sample quality in final `train.jsonl`

Then scale up.
