# General distill scripts v1

These scripts implement the **general-side** targeted distill pipeline we discussed:

1. generate canonical template seeds
2. paraphrase + dedup template prompts
3. extract open-source raw prompts
4. classify + canonicalize
5. teacher generate
6. verify + repair
7. build final general bank

## Files

- `distill/general_utils.py`
- `distill/general_gen_template_seeds_v1.py`
- `distill/general_paraphrase_templates_v1.py`
- `distill/general_extract_open_v1.py`
- `distill/general_classify_canonicalize_v1.py`
- `distill/general_teacher_generate_v1.py`
- `distill/general_verify_v1.py`
- `distill/general_build_bank_v1.py`

## Expected raw inputs

These scripts assume you already have local JSONL exports for:
- no_robots
- smol_smoltalk
- alpaca_cleaned
- dolly-style

The extractor does **not** fetch from Hugging Face itself. That keeps it simple and reproducible on your own storage.

## OpenAI-compatible API

The paraphrase and teacher-generation scripts expect an OpenAI-compatible endpoint, such as local vLLM.

Example:
```bash
export OPENAI_API_KEY=EMPTY

python distill/general_paraphrase_templates_v1.py \
  --in_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl \
  --out_raw_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_raw_v1.jsonl \
  --out_dedup_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher
```

## Recommended execution order

### 1) seeds
```bash
python distill/general_gen_template_seeds_v1.py \
  --out_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl
```

### 2) paraphrase + dedup templates
```bash
python distill/general_paraphrase_templates_v1.py \
  --in_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl \
  --out_raw_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_raw_v1.jsonl \
  --out_dedup_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher
```

### 3) extract open-source prompts
```bash
python distill/general_extract_open_v1.py \
  --no_robots_jsonl data/no_robots_train_sft.jsonl \
  --smol_jsonl data/smol_smoltalk_train.jsonl \
  --alpaca_jsonl data/alpaca_cleaned_train.jsonl \
  --dolly_jsonl data/dolly_style_train.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/raw_open_prompts_v1.jsonl
```

### 4) classify + canonicalize
```bash
python distill/general_classify_canonicalize_v1.py \
  --raw_open_jsonl dataset/targeted_distill_general_v1/raw_open_prompts_v1.jsonl \
  --template_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl
```

### 5) teacher raw generation
```bash
python distill/general_teacher_generate_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/general_teacher_raw_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.3
```

### 6) verify raw + prepare repair set
```bash
python distill/general_verify_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_raw_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_general_v1/general_verified_pass_round1_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_general_v1/general_verified_reject_round1_v1.jsonl \
  --out_repair_candidates_jsonl dataset/targeted_distill_general_v1/general_teacher_repair_candidates_v1.jsonl
```

### 7) repair
```bash
python distill/general_teacher_generate_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_repair_candidates_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/general_teacher_repaired_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.4
```

### 8) verify repairs
```bash
python distill/general_verify_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_repaired_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_general_v1/general_verified_pass_repair_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_general_v1/general_verified_reject_repair_v1.jsonl
```

### 9) build bank
```bash
python distill/general_build_bank_v1.py \
  --pass_jsonls \
    dataset/targeted_distill_general_v1/general_verified_pass_round1_v1.jsonl \
    dataset/targeted_distill_general_v1/general_verified_pass_repair_v1.jsonl \
  --out_train_jsonl dataset/targeted_distill_general_v1/accepted_general_bank_v1.jsonl \
  --out_val_jsonl dataset/targeted_distill_general_v1/general_val_bank_v1.jsonl \
  --out_holdout_jsonl dataset/targeted_distill_general_v1/general_holdout_v1.jsonl
```

## Notes

- These scripts are intentionally conservative.
- They favor **clean, short, format-obedient outputs** over broad coverage.
- MBPP / APPS stay on the **code** side, not in this general pipeline.
