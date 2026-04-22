# Distill assembly scripts v1

This package connects the **general bank** and **code bank** into final targeted-distill train/val files.

## Files

- `distill/build_targeted_distill_mix_v1.py`
- `distill/build_smoke_manifests_v1.py`

## Recommended upstream inputs

### General side
- `accepted_general_bank_v1.jsonl`
- `general_val_bank_v1.jsonl`

### Code side
- `accepted_code_bank_v1.jsonl`
- `code_val_bank_v1.jsonl`

## Smoke-test manifests

Build small smoke subsets before the full teacher run:

```bash
python distill/build_smoke_manifests_v1.py \
  --general_canonical_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl \
  --code_canonical_jsonl dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl \
  --out_general_smoke_jsonl dataset/targeted_distill_mix_v1/general_smoke_v1.jsonl \
  --out_code_smoke_jsonl dataset/targeted_distill_mix_v1/code_smoke_v1.jsonl
```

## Final mixed train/val

Default targets:
- code train: 4800
- general train: 1800
- code val: 400
- general val: 100

That gives:
- train total: 6600
- val total: 500

```bash
python distill/build_targeted_distill_mix_v1.py \
  --code_train_jsonl dataset/targeted_distill_code_v1/accepted_code_bank_v1.jsonl \
  --code_val_jsonl dataset/targeted_distill_code_v1/code_val_bank_v1.jsonl \
  --general_train_jsonl dataset/targeted_distill_general_v1/accepted_general_bank_v1.jsonl \
  --general_val_jsonl dataset/targeted_distill_general_v1/general_val_bank_v1.jsonl \
  --out_train_jsonl dataset/targeted_distill_mix_v1/train.jsonl \
  --out_val_jsonl dataset/targeted_distill_mix_v1/val.jsonl
```

## Suggested training start point

Use your best v6 SFT checkpoint, which we discussed as the main candidate around `step_003500.pt`, then run a code-heavy targeted distill on the mixed train/val.
