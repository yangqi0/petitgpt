# Code distill scripts v1

This package implements the code-side targeted distill pipeline:

1. generate the 12 core simple-Python family prompts
2. extract MBPP tasks
3. extract APPS introductory function-style tasks
4. teacher generate code
5. verify with AST + tests
6. repair a subset
7. build a final code bank

## Files

- distill/code_utils.py
- distill/code_gen_core_families_v1.py
- distill/code_extract_mbpp_v1.py
- distill/code_extract_apps_v1.py
- distill/code_teacher_generate_v1.py
- distill/code_verify_v1.py
- distill/code_build_bank_v1.py

## Recommended execution order

### 1) generate 12-family prompts
python distill/code_gen_core_families_v1.py \
  --out_jsonl dataset/targeted_distill_code_v1/core_family_prompts_v1.jsonl \
  --instances_per_family 80

### 2) extract MBPP
python distill/code_extract_mbpp_v1.py \
  --in_jsonl data/mbpp_train.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/mbpp_prompts_v1.jsonl

### 3) extract APPS introductory
python distill/code_extract_apps_v1.py \
  --in_jsonl data/apps_intro.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/apps_prompts_v1.jsonl

### 4) merge prompt pools
Concatenate:
- core_family_prompts_v1.jsonl
- mbpp_prompts_v1.jsonl
- apps_prompts_v1.jsonl

into:
- dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl

### 5) teacher raw generation
python distill/code_teacher_generate_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/code_teacher_raw_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.15

### 6) verify raw + prepare repair set
python distill/code_verify_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_raw_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_code_v1/code_verified_pass_round1_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_code_v1/code_verified_reject_round1_v1.jsonl \
  --out_repair_candidates_jsonl dataset/targeted_distill_code_v1/code_teacher_repair_candidates_v1.jsonl

### 7) repair
python distill/code_teacher_generate_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_repair_candidates_v1.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/code_teacher_repaired_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.25

### 8) verify repairs
python distill/code_verify_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_repaired_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_code_v1/code_verified_pass_repair_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_code_v1/code_verified_reject_repair_v1.jsonl

### 9) build bank
python distill/code_build_bank_v1.py \
  --pass_jsonls \
    dataset/targeted_distill_code_v1/code_verified_pass_round1_v1.jsonl \
    dataset/targeted_distill_code_v1/code_verified_pass_repair_v1.jsonl \
  --out_train_jsonl dataset/targeted_distill_code_v1/accepted_code_bank_v1.jsonl \
  --out_val_jsonl dataset/targeted_distill_code_v1/code_val_bank_v1.jsonl \
  --out_holdout_jsonl dataset/targeted_distill_code_v1/code_holdout_v1.jsonl
