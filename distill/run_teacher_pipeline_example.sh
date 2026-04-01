#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Example pipeline:
# 1) generate general prompts
# 2) generate code prompts
# 3) sample teacher outputs from an OpenAI-compatible local server
# 4) verify code with unit tests
# 5) build canonical train/val jsonl
#
# Edit the MODEL and BASE_URL to match your local server.
# Example local server:
#   vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --dtype auto --api-key EMPTY
# ============================================================

export PYTHONUNBUFFERED=1

PROMPT_DIR="dataset/teacher_prompts_v2"
RAW_DIR="dataset/teacher_raw_v2"
FINAL_DIR="dataset/teacher_final_v2"

mkdir -p "$PROMPT_DIR" "$RAW_DIR" "$FINAL_DIR"

GENERAL_PROMPTS="${PROMPT_DIR}/general_prompts_v2.jsonl"
CODE_PROMPTS="${PROMPT_DIR}/code_prompts_v2.jsonl"

GENERAL_RAW="${RAW_DIR}/general_raw_v2.jsonl"
CODE_RAW="${RAW_DIR}/code_raw_v2.jsonl"

CODE_ACCEPTED="${RAW_DIR}/code_verified_v2.jsonl"
CODE_REJECTED="${RAW_DIR}/code_rejected_v2.jsonl"

OUT_TRAIN="${FINAL_DIR}/train.jsonl"
OUT_VAL="${FINAL_DIR}/val.jsonl"

# ---------- Teacher endpoint ----------
GENERAL_MODEL="Qwen/Qwen2.5-32B-Instruct"
CODE_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"

# Change this to your OpenAI-compatible server URL.
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

# ---------- Prompt generation ----------
python /mnt/data/make_teacher_general_prompts_v2.py \
  --out "$GENERAL_PROMPTS" \
  --n_target 5000 \
  --seed 1234

python /mnt/data/make_teacher_code_prompts_v2.py \
  --out "$CODE_PROMPTS" \
  --n_target 5000 \
  --seed 1234

# ---------- Teacher generation ----------
python /mnt/data/stage2_teacher_generate_open.py \
  --in_jsonl "$GENERAL_PROMPTS" \
  --out_jsonl "$GENERAL_RAW" \
  --model "$GENERAL_MODEL" \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --n 1 \
  --temperature 0.2 \
  --top_p 1.0 \
  --max_new_tokens 220

python /mnt/data/stage2_teacher_generate_open.py \
  --in_jsonl "$CODE_PROMPTS" \
  --out_jsonl "$CODE_RAW" \
  --model "$CODE_MODEL" \
  --base_url "$BASE_URL" \
  --api_key "$API_KEY" \
  --n 2 \
  --temperature 0.2 \
  --top_p 1.0 \
  --max_new_tokens 220

# ---------- Code verification ----------
python /mnt/data/verify_code_candidates_with_tests.py \
  --in_jsonl "$CODE_RAW" \
  --out_jsonl "$CODE_ACCEPTED" \
  --rejected_jsonl "$CODE_REJECTED" \
  --timeout_s 3.0

# ---------- Build canonical dataset ----------
python /mnt/data/build_general_code_teacher_dataset.py \
  --general_raw "$GENERAL_RAW" \
  --code_verified "$CODE_ACCEPTED" \
  --out_train "$OUT_TRAIN" \
  --out_val "$OUT_VAL" \
  --val_ratio 0.02 \
  --seed 1234

echo "Done."
echo "Train: $OUT_TRAIN"
echo "Val:   $OUT_VAL"
