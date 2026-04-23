# Raw data export scripts v1

This package adds the raw dataset export layer so you can create the local JSONL files required by the distill pipeline.

## Files

- `distill/export_hf_general_sources_v1.py`
- `distill/normalize_existing_smol_source_v1.py`
- `distill/export_hf_code_sources_v1.py`

## What these scripts produce

### General-side outputs
- `data/no_robots_train_sft.jsonl`
- `data/alpaca_cleaned_train.jsonl`
- `data/dolly_style_train.jsonl`

You may still provide your own smol/smoltalk export and normalize it with:
- `data/smol_smoltalk_train.jsonl`

### Code-side outputs
- `data/mbpp_train.jsonl`
- `data/apps_intro.jsonl`

## Install requirements

```bash
pip install -U datasets huggingface_hub
```

## 1) Export general raw sources

```bash
python distill/export_hf_general_sources_v1.py \
  --out_dir data/
```

Optional limits:

```bash
python distill/export_hf_general_sources_v1.py \
  --out_dir data/ \
  --limit_no_robots 3000 \
  --limit_alpaca 1400 \
  --limit_dolly 900
```

## 2) Normalize your existing smol/smoltalk export

```bash
python distill/normalize_existing_smol_source_v1.py \
  --in_jsonl data/my_smol_export.jsonl \
  --out_jsonl data/smol_smoltalk_train.jsonl
```

## 3) Export code raw sources

Default:
```bash
python distill/export_hf_code_sources_v1.py \
  --out_dir data/
```

That writes:
- `data/mbpp_train.jsonl`
- `data/apps_intro.jsonl`

If you want MBPP sanitized:

```bash
python distill/export_hf_code_sources_v1.py \
  --out_dir data/ \
  --mbpp_config sanitized
```

If you want limits:

```bash
python distill/export_hf_code_sources_v1.py \
  --out_dir data/ \
  --limit_mbpp 2000 \
  --limit_apps 1500
```
