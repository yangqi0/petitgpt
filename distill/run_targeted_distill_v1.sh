#!/usr/bin/env bash
set -euo pipefail

python sft/train_distill.py \
  --train_jsonl dataset/targeted_distill_mix_v1/train.jsonl \
  --val_jsonl dataset/targeted_distill_mix_v1/val.jsonl \
  --out_dir outputs/targeted_distill_v1_from_sft_v6_3500 \
  --tokenizer_path tokenizer/tokenizer.json \
  --init_from_pretrain outputs/sft_v6_general_code/step_003500.pt \
  --seq_len 1024 \
  --micro_bsz 4 \
  --grad_accum 4 \
  --lr 2e-6 \
  --weight_decay 0.01 \
  --warmup_steps 25 \
  --max_steps 4000 \
  --precision bf16 \
  --eval_every 25 \
  --eval_batches 100 \
  --save_every 25 \
  --sample_every 25 \
  --samples_dir outputs/targeted_distill_v1_from_sft_v6_3500/samples \
  --sample_max_new_tokens 160 \
  --sample_temperature 0.2 \
  --sample_top_p 1.0 \
  --sample_top_k 0 \
  --sample_repetition_penalty 1.05 \
  --sample_repetition_window 256 \
  --sample_no_repeat_ngram 3 \
  --loss_reduction example_mean \
  --refusal_downweight 1.0 \
  --default_system "You are a helpful, concise assistant. Give direct answers. For simple coding tasks, write clear Python code." \
  --num_workers 2 \
  --seed 1234 \
  --sample_in_domain_n 10 \
  --sample_in_domain_mode full_context \
  --sample_eval_jsonl dataset/sft_canon/sample_eval.jsonl \
  --log_losses \
  --loss_log_every 50
