# targeted distill training notes

This package contains:

- `train_sft_distill_with_logging.py`
- `run_targeted_distill_v1.sh`

## What changed relative to the uploaded `train_sft.py`

The new training script keeps the same overall SFT/distill flow and adds persistent loss logging:

- train loss -> `out_dir/logs/train_loss.jsonl`
- train loss -> `out_dir/logs/train_loss.csv`
- eval loss -> `out_dir/logs/eval_loss.jsonl`
- eval loss -> `out_dir/logs/eval_loss.csv`

## New flags

- `--log_losses`
- `--loss_log_every`

## Suggested start point for this round

Use:

- init checkpoint: `outputs/sft_v6_general_code/step_003500.pt`
- mixed distill train: `dataset/targeted_distill_mix_v1/train.jsonl`
- mixed distill val: `dataset/targeted_distill_mix_v1/val.jsonl`

## Plotting later

Because the script writes both CSV and JSONL, you can later make:

- train-loss curve
- eval-loss curve
- README figures
