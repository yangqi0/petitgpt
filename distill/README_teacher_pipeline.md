# Teacher pipeline for high-quality synthetic SFT / distill data

This small pipeline is designed for a **general + simple-code** target, which fits a small student model much better than broad synthetic pretraining.

## Files

- `make_teacher_general_prompts_v2.py`
- `make_teacher_code_prompts_v2.py`
- `stage2_teacher_generate_open.py`
- `verify_code_candidates_with_tests.py`
- `build_general_code_teacher_dataset.py`
- `run_teacher_pipeline_example.sh`

## Expected flow

1. Generate large prompt sets.
2. Sample outputs from strong open-weight teacher models through a local OpenAI-compatible endpoint.
3. Verify code candidates with:
   - exact one code block
   - AST parse
   - entrypoint check
   - minimal unit tests
4. Keep filtered general outputs and verified code outputs.
5. Build canonical `train.jsonl` / `val.jsonl` for your SFT trainer.

## Suggested teacher split

- General: `Qwen2.5-32B-Instruct`
- Code: `Qwen2.5-Coder-32B-Instruct`

## Why this is better than your current small stage-2

Your current code acceptance logic is strong on **format** but weak on **semantic correctness**. For code-heavy distillation, semantic filtering matters more than raw sample count.

## Notes

- The code verifier executes model code in a subprocess with a timeout and restricted builtins/imports.
- It is still safest to run inside a container.
- General filtering is intentionally simple. You can make it stricter later.
- For a first pass, 5k general + 5k code is a reasonable target.
