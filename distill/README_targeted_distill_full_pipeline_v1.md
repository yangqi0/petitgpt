# README — Targeted Distill 全流程操作指南（Runpod + vLLM + Qwen-3.5）

这份 README 把前面所有步骤整合成一份完整操作指南，目标是让你从 **Runpod 新建 Pod** 开始，一步步把这轮 targeted distill 跑通。

内容包括：

1. Runpod 上如何准备环境  
2. 如何用 vLLM 跑 `Qwen/Qwen3.5-35B-A3B-FP8` 作为 teacher  
3. 需要准备哪些文件、放到哪里  
4. 需要安装哪些 pip 包  
5. 如何导出原始数据  
6. 如何跑 general 侧脚本  
7. 如何跑 code 侧脚本  
8. 如何把 general + code 混成最终 distill train/val  
9. 如何启动 targeted distill 训练  
10. 如何记录和查看 loss 曲线  
11. `smoke` 是什么，什么时候该用，什么时候可以直接跑全量  
12. 一份建议的“从 0 到训练”的实际执行顺序

---

## 0. 默认假设

这份 README 默认你采用下面的方式：

- 你的项目根目录在：`/workspace/petitgpt`
- 你的 Pod 里会运行：
  - 数据导出脚本
  - general/code/mix 三条线脚本
  - vLLM teacher
  - targeted distill 训练
- 你希望最终训练使用：
  - `outputs/sft_v6_general_code/step_003500.pt`
  - `dataset/targeted_distill_mix_v1/train.jsonl`
  - `dataset/targeted_distill_mix_v1/val.jsonl`

如果你的目录结构不一样，请把下面命令里的路径改成你自己的。

---

## 1. 先在 Runpod 上准备 Pod

### 推荐 GPU
为了比较省心地跑 `Qwen/Qwen3.5-35B-A3B-FP8`，推荐：

- **单卡 A100 80GB**

这是这份 README 默认采用的 teacher 配置。

### 推荐模板
推荐使用 **Runpod 官方 PyTorch 模板**。  
这样通常已经带好比较合适的 CUDA / Python / PyTorch 环境。

### 推荐连接方式
优先使用：

- **SSH**
- 辅助可用 **JupyterLab**

不建议长期只用 Web Terminal。

---

## 2. Runpod 上的目录建议

请尽量把重要数据和代码放在持久化目录里，例如：

```bash
/workspace/petitgpt
/workspace/petitgpt/data
/workspace/petitgpt/dataset
/workspace/petitgpt/outputs
/workspace/petitgpt/distill
/workspace/petitgpt/sft
```

---

## 3. 你需要上传到 Runpod 的文件

你当前已经有这些包和脚本：

### 3.1 distill 脚本包
- `general_distill_scripts_v1.zip`
- `code_distill_scripts_v1.zip`
- `distill_assembly_scripts_v1.zip`
- `raw_data_export_scripts_v1.zip`

### 3.2 训练脚本
- `train_sft_distill_with_logging.py`
- `run_targeted_distill_v1.sh`

### 3.3 你项目已有的重要文件
这些通常你已经有：

```bash
tokenizer/tokenizer.json
outputs/sft_v6_general_code/step_003500.pt
dataset/sft_canon/sample_eval.jsonl
src/model.py
```

---

## 4. 如何把文件传到 Runpod

### 方法 A：JupyterLab
如果你的 Pod 带 JupyterLab：

1. 打开 JupyterLab
2. 进入 `/workspace/petitgpt`
3. 把 zip 和脚本文件上传进去

### 方法 B：SCP
如果你配置好了 SSH 和 public IP：

```bash
scp -P <SSH_PORT> general_distill_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> code_distill_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> distill_assembly_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> raw_data_export_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> train_sft_distill_with_logging.py root@<POD_IP>:/workspace/petitgpt/sft/
scp -P <SSH_PORT> run_targeted_distill_v1.sh root@<POD_IP>:/workspace/petitgpt/
```

---

## 5. 上传后先解压

```bash
cd /workspace/petitgpt

unzip -o general_distill_scripts_v1.zip -d /workspace/petitgpt/
unzip -o code_distill_scripts_v1.zip -d /workspace/petitgpt/
unzip -o distill_assembly_scripts_v1.zip -d /workspace/petitgpt/
unzip -o raw_data_export_scripts_v1.zip -d /workspace/petitgpt/

mkdir -p sft
cp train_sft_distill_with_logging.py sft/train_sft_distill_with_logging.py
chmod +x run_targeted_distill_v1.sh
```

---

## 6. Python 环境与 pip 安装

### 6.1 激活 venv

如果你已有 `.venv`：

```bash
cd /workspace/petitgpt
source .venv/bin/activate
```

如果还没有：

```bash
cd /workspace/petitgpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

---

### 6.2 安装需要的 Python 包

```bash
pip install -U \
  vllm \
  tokenizers \
  sentence-transformers \
  scikit-learn \
  datasets \
  huggingface_hub \
  pandas \
  matplotlib
```

说明：

- `vllm`：本地起 teacher
- `tokenizers`：训练和数据流程会用到
- `sentence-transformers`：semantic dedup
- `scikit-learn`：fallback 的去重/向量工具
- `datasets`：导出 Hugging Face 数据集
- `huggingface_hub`：登录、下载模型/数据
- `pandas`, `matplotlib`：画 loss 曲线

如果你的环境里没有 `torch`，再补：

```bash
pip install torch
```

---

### 6.3 推荐额外安装

```bash
apt update
apt install -y tmux unzip
```

强烈建议全程用 `tmux`：

```bash
tmux new -s teacher
```

你可以：

- 一个 pane 跑 vLLM
- 一个 pane 跑数据脚本
- 一个 pane 看日志

---

## 7. 配 Hugging Face token（推荐）

如果你要从 Hugging Face 拉 teacher 模型或数据集，建议先登录：

```bash
huggingface-cli login
```

或者设置环境变量：

```bash
export HF_TOKEN=your_hf_token_here
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
```

如果你希望每次 shell 自动带上，可以写进：

```bash
~/.bashrc
```

---

## 8. 起 vLLM teacher（Qwen-3.5）

这份 README 按下面这个模型写：

- `Qwen/Qwen3.5-35B-A3B-FP8`

### 8.1 启动命令

建议在 tmux 里运行：

```bash
export OPENAI_API_KEY=EMPTY
export VLLM_USE_V1=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --served-model-name teacher \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --dtype auto \
  --api-key EMPTY
```

### 8.2 检查 teacher 是否起来

开另一个终端检查：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

如果返回模型列表，说明 teacher 已经正常起来。

---

## 9. 先确认项目已有文件

在开始之前，先确认你已有这些文件：

```bash
ls tokenizer/tokenizer.json
ls outputs/sft_v6_general_code/step_003500.pt
ls dataset/sft_canon/sample_eval.jsonl
ls src/model.py
```

如果这些不在，先不要继续。

---

## 10. 导出原始数据

这一步的目标是生成下面这些本地 JSONL：

### general 侧
```bash
data/no_robots_train_sft.jsonl
data/smol_smoltalk_train.jsonl
data/alpaca_cleaned_train.jsonl
data/dolly_style_train.jsonl
```

### code 侧
```bash
data/mbpp_train.jsonl
data/apps_intro.jsonl
```

---

### 10.1 导出 general 侧（HF 直接导出）

```bash
python distill/export_hf_general_sources_v1.py \
  --out_dir data/
```

如果你想先小一点：

```bash
python distill/export_hf_general_sources_v1.py \
  --out_dir data/ \
  --limit_no_robots 3000 \
  --limit_alpaca 1400 \
  --limit_dolly 900
```

---

### 10.2 规范化你自己的 smol/smoltalk 导出

如果你已经有一个本地 smol/smoltalk 风格 JSONL，可以统一成后续脚本需要的格式：

```bash
python distill/normalize_existing_smol_source_v1.py \
  --in_jsonl data/my_smol_export.jsonl \
  --out_jsonl data/smol_smoltalk_train.jsonl
```

如果你还没有这份本地 export，可以先跳过，后面只用 `no_robots + alpaca + dolly + templates` 也能跑，只是 general 的 source 多样性稍弱一点。

---

### 10.3 导出 code 侧（MBPP + APPS）

```bash
python distill/export_hf_code_sources_v1.py \
  --out_dir data/
```

如果你想限制规模：

```bash
python distill/export_hf_code_sources_v1.py \
  --out_dir data/ \
  --limit_mbpp 2000 \
  --limit_apps 1500
```

如果你想用 MBPP 的 `sanitized` 版本：

```bash
python distill/export_hf_code_sources_v1.py \
  --out_dir data/ \
  --mbpp_config sanitized
```

---

## 11. 先跑一个很小的 smoke（推荐）

### 11.1 `smoke` 是什么

`smoke` 就是一个 **小规模 sanity check**。  
它不是最终训练集，而是用来先确认：

- vLLM teacher 能正常返回结果
- verifier 不会把大多数样本全拒掉
- family / source / canonicalization 没明显 bug
- API/base_url/模型名/路径/字段都通

### 11.2 为什么先跑 smoke
因为只要有一个字段名写错、路径写错、verifier 太严，你全量跑会浪费很多时间。

### 11.3 smoke 能不能做大
可以，但不建议把 smoke 当成最终训练集。  
更好的顺序是：

1. 先做小 smoke  
2. 看 reject reasons  
3. 修一下问题  
4. 再跑 full pipeline  
5. 最终训练

---

## 12. 跑 general 侧全流程

### 12.1 生成 template seeds
```bash
python distill/general_gen_template_seeds_v1.py \
  --out_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl
```

### 12.2 模板 paraphrase + dedup
```bash
python distill/general_paraphrase_templates_v1.py \
  --in_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl \
  --out_raw_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_raw_v1.jsonl \
  --out_dedup_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher
```

### 12.3 抽开源 raw prompts
```bash
python distill/general_extract_open_v1.py \
  --no_robots_jsonl data/no_robots_train_sft.jsonl \
  --smol_jsonl data/smol_smoltalk_train.jsonl \
  --alpaca_jsonl data/alpaca_cleaned_train.jsonl \
  --dolly_jsonl data/dolly_style_train.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/raw_open_prompts_v1.jsonl
```

### 12.4 classify + canonicalize
```bash
python distill/general_classify_canonicalize_v1.py \
  --raw_open_jsonl dataset/targeted_distill_general_v1/raw_open_prompts_v1.jsonl \
  --template_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl
```

### 12.5 teacher raw generation
```bash
python distill/general_teacher_generate_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/general_teacher_raw_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.3
```

### 12.6 verify + repair candidates
```bash
python distill/general_verify_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_raw_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_general_v1/general_verified_pass_round1_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_general_v1/general_verified_reject_round1_v1.jsonl \
  --out_repair_candidates_jsonl dataset/targeted_distill_general_v1/general_teacher_repair_candidates_v1.jsonl
```

### 12.7 repair
```bash
python distill/general_teacher_generate_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_repair_candidates_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/general_teacher_repaired_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.4
```

### 12.8 verify repaired
```bash
python distill/general_verify_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_repaired_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_general_v1/general_verified_pass_repair_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_general_v1/general_verified_reject_repair_v1.jsonl
```

### 12.9 build general bank
```bash
python distill/general_build_bank_v1.py \
  --pass_jsonls \
    dataset/targeted_distill_general_v1/general_verified_pass_round1_v1.jsonl \
    dataset/targeted_distill_general_v1/general_verified_pass_repair_v1.jsonl \
  --out_train_jsonl dataset/targeted_distill_general_v1/accepted_general_bank_v1.jsonl \
  --out_val_jsonl dataset/targeted_distill_general_v1/general_val_bank_v1.jsonl \
  --out_holdout_jsonl dataset/targeted_distill_general_v1/general_holdout_v1.jsonl
```

---

## 13. 跑 code 侧全流程

### 13.1 生成 12 家族 prompts
```bash
python distill/code_gen_core_families_v1.py \
  --out_jsonl dataset/targeted_distill_code_v1/core_family_prompts_v1.jsonl \
  --instances_per_family 80
```

### 13.2 抽 MBPP
```bash
python distill/code_extract_mbpp_v1.py \
  --in_jsonl data/mbpp_train.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/mbpp_prompts_v1.jsonl
```

### 13.3 抽 APPS intro
```bash
python distill/code_extract_apps_v1.py \
  --in_jsonl data/apps_intro.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/apps_prompts_v1.jsonl
```

### 13.4 合并 prompt pool
把下面三个文件合并成一个：

- `core_family_prompts_v1.jsonl`
- `mbpp_prompts_v1.jsonl`
- `apps_prompts_v1.jsonl`

生成：

- `dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl`

```bash
cat \
  dataset/targeted_distill_code_v1/core_family_prompts_v1.jsonl \
  dataset/targeted_distill_code_v1/mbpp_prompts_v1.jsonl \
  dataset/targeted_distill_code_v1/apps_prompts_v1.jsonl \
  > dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl
```

### 13.5 teacher raw generation
```bash
python distill/code_teacher_generate_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/code_teacher_raw_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.15
```

### 13.6 verify + repair candidates
```bash
python distill/code_verify_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_raw_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_code_v1/code_verified_pass_round1_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_code_v1/code_verified_reject_round1_v1.jsonl \
  --out_repair_candidates_jsonl dataset/targeted_distill_code_v1/code_teacher_repair_candidates_v1.jsonl
```

### 13.7 repair
```bash
python distill/code_teacher_generate_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_repair_candidates_v1.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/code_teacher_repaired_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.25
```

### 13.8 verify repaired
```bash
python distill/code_verify_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_repaired_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_code_v1/code_verified_pass_repair_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_code_v1/code_verified_reject_repair_v1.jsonl
```

### 13.9 build code bank
```bash
python distill/code_build_bank_v1.py \
  --pass_jsonls \
    dataset/targeted_distill_code_v1/code_verified_pass_round1_v1.jsonl \
    dataset/targeted_distill_code_v1/code_verified_pass_repair_v1.jsonl \
  --out_train_jsonl dataset/targeted_distill_code_v1/accepted_code_bank_v1.jsonl \
  --out_val_jsonl dataset/targeted_distill_code_v1/code_val_bank_v1.jsonl \
  --out_holdout_jsonl dataset/targeted_distill_code_v1/code_holdout_v1.jsonl
```

---

## 14. 组装最终 mixed distill train/val

### 14.1 可选地做 smoke manifests
```bash
python distill/build_smoke_manifests_v1.py \
  --general_canonical_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl \
  --code_canonical_jsonl dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl \
  --out_general_smoke_jsonl dataset/targeted_distill_mix_v1/general_smoke_v1.jsonl \
  --out_code_smoke_jsonl dataset/targeted_distill_mix_v1/code_smoke_v1.jsonl
```

### 14.2 build 最终 mix
```bash
python distill/build_targeted_distill_mix_v1.py \
  --code_train_jsonl dataset/targeted_distill_code_v1/accepted_code_bank_v1.jsonl \
  --code_val_jsonl dataset/targeted_distill_code_v1/code_val_bank_v1.jsonl \
  --general_train_jsonl dataset/targeted_distill_general_v1/accepted_general_bank_v1.jsonl \
  --general_val_jsonl dataset/targeted_distill_general_v1/general_val_bank_v1.jsonl \
  --out_train_jsonl dataset/targeted_distill_mix_v1/train.jsonl \
  --out_val_jsonl dataset/targeted_distill_mix_v1/val.jsonl
```

默认目标是：

- code train: 4800
- general train: 1800
- code val: 400
- general val: 100

也就是：

- train total: 6600
- val total: 500

---

## 15. 启动 targeted distill 训练

你已经有这份脚本：

- `sft/train_sft_distill_with_logging.py`

以及启动 shell：

- `run_targeted_distill_v1.sh`

### 直接运行
```bash
cd /workspace/petitgpt
source .venv/bin/activate
bash run_targeted_distill_v1.sh
```

### 这轮训练的核心配置
- `init_from_pretrain = outputs/sft_v6_general_code/step_003500.pt`
- `lr = 2e-6`
- `weight_decay = 0.01`
- `warmup_steps = 25`
- `max_steps = 400`
- `loss_reduction = example_mean`
- `eval_every = 25`
- `save_every = 25`
- `sample_every = 25`

---

## 16. loss 记录在哪里

新的训练脚本会写：

```bash
outputs/targeted_distill_v1_from_sft_v6_3500/logs/train_loss.jsonl
outputs/targeted_distill_v1_from_sft_v6_3500/logs/train_loss.csv
outputs/targeted_distill_v1_from_sft_v6_3500/logs/eval_loss.jsonl
outputs/targeted_distill_v1_from_sft_v6_3500/logs/eval_loss.csv
```

这些 CSV 可以直接用来画图和写 README。

---

## 17. 如何画 loss 曲线（最简）

```python
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("outputs/targeted_distill_v1_from_sft_v6_3500/logs/train_loss.csv")
eval_ = pd.read_csv("outputs/targeted_distill_v1_from_sft_v6_3500/logs/eval_loss.csv")

plt.figure()
plt.plot(train["step"], train["loss"], label="train")
plt.plot(eval_["step"], eval_["val_loss"], label="eval")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/targeted_distill_v1_from_sft_v6_3500/logs/loss_curve.png", dpi=160)
```

---

## 18. 一份推荐的实际执行顺序

如果你问我最稳的现实顺序，我会建议：

### 路线 A：稳健版（推荐）
1. 起 vLLM teacher  
2. 导出原始数据  
3. 跑 very small smoke  
4. 看 reject reasons  
5. 如果有必要，小修 verifier / canonicalization / path  
6. 跑 general 全量  
7. 跑 code 全量  
8. build final mix  
9. 跑 targeted distill training  
10. 看 loss 曲线和 samples，选 checkpoint  

### 路线 B：激进版
1. 起 vLLM teacher  
2. 导出原始数据  
3. 直接全量 general  
4. 直接全量 code  
5. build mix  
6. 训练  

我更推荐路线 A。

---

## 19. 开始前最后检查一遍

```bash
cd /workspace/petitgpt
source .venv/bin/activate

python -V
nvidia-smi
which python

ls tokenizer/tokenizer.json
ls outputs/sft_v6_general_code/step_003500.pt
ls src/model.py
```

起 teacher：

```bash
export OPENAI_API_KEY=EMPTY
export VLLM_USE_V1=1

vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --served-model-name teacher \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --dtype auto \
  --api-key EMPTY
```

验证 teacher：

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer EMPTY"
```

如果这里没问题，再跑 smoke 或全量。

