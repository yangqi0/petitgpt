# Targeted Distill 操作指南（Runpod + vLLM + Qwen-3.5）

这份 README 的目标是让你从 **Runpod 新建 Pod** 开始，一步步把这轮 targeted distill 跑通。

内容包括：

1. 在 Runpod 上如何准备环境  
2. 如何用 vLLM 跑 **Qwen-3.5** 作为 teacher  
3. 需要准备哪些文件、放到哪里  
4. 需要安装哪些 Python 包  
5. 如何跑 general 侧脚本  
6. 如何跑 code 侧脚本  
7. 如何把 general + code 混成最终 distill train/val  
8. 如何启动 targeted distill 训练  
9. 如何记录和查看 loss 曲线  
10. `smoke` 是什么，什么时候该用，什么时候可以直接跑全量

---

## 0. 默认假设

我这里默认你采用下面这个工作方式：

- 你有一个 Runpod Pod
- 你的项目根目录在 `/workspace/petitgpt`
- 你会把我给你的 3 个 zip 包上传到这个项目里
- 你希望在 Runpod 上本地起一个 OpenAI-compatible 的 vLLM teacher，然后用你自己的脚本生成数据
- 你希望最终训练用：
  - `outputs/sft_v6_general_code/step_003500.pt`
  - mixed targeted distill train/val

如果你的目录名不是 `/workspace/petitgpt`，请把下面命令里的路径改成你自己的目录。

---

## 1. Runpod 上如何准备 Pod

### 推荐方式
推荐用 **Runpod 官方 PyTorch 模板** 起一个 Pod。这样通常已经带了比较合适的 CUDA / Python / PyTorch 环境，也更容易通过 SSH 或 JupyterLab 连接。

### GPU 建议
如果你想跑 **Qwen-3.5-35B-A3B-FP8** 作为 teacher，我建议按：

- **单卡 A100 80GB**

来写和来跑。这是最省心的方案。

---

## 2. Runpod 上如何连接

### 2.1 推荐：SSH
如果你打算真正跑数据生成和训练，**优先用 SSH，不要长期依赖 Web Terminal**。

### 2.2 备选：JupyterLab
如果你更习惯网页文件管理和 Notebook，也可以用 JupyterLab 辅助上传文件和看结果。

---

## 3. Runpod 上的持久化目录

请尽量把项目、数据、模型权重、日志都放在：

- `/workspace`
- 或者挂载的 network volume

建议你的项目结构类似：

```bash
/workspace/petitgpt
/workspace/petitgpt/dataset
/workspace/petitgpt/outputs
/workspace/petitgpt/distill
/workspace/petitgpt/sft
/workspace/petitgpt/data
```

---

## 4. 把文件传到 Runpod

你现在手里已经有这些文件：

- `general_distill_scripts_v1.zip`
- `code_distill_scripts_v1.zip`
- `distill_assembly_scripts_v1.zip`
- `train_sft_distill_with_logging.py`
- `run_targeted_distill_v1.sh`

### 推荐上传方式 A：JupyterLab
如果你的 Pod 带 JupyterLab：

1. 打开 JupyterLab
2. 进入 `/workspace/petitgpt`
3. 用网页上传 zip 和脚本文件

### 推荐上传方式 B：SCP
如果你的 Pod 配好了 public IP + full SSH，可以直接：

```bash
scp -P <SSH_PORT> general_distill_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> code_distill_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> distill_assembly_scripts_v1.zip root@<POD_IP>:/workspace/petitgpt/
scp -P <SSH_PORT> train_sft_distill_with_logging.py root@<POD_IP>:/workspace/petitgpt/sft/
scp -P <SSH_PORT> run_targeted_distill_v1.sh root@<POD_IP>:/workspace/petitgpt/
```

---

## 5. 上传后先解压

进入项目目录：

```bash
cd /workspace/petitgpt
```

解压：

```bash
unzip -o general_distill_scripts_v1.zip -d /workspace/petitgpt/
unzip -o code_distill_scripts_v1.zip -d /workspace/petitgpt/
unzip -o distill_assembly_scripts_v1.zip -d /workspace/petitgpt/
```

然后把新的训练脚本放到 `sft/` 下：

```bash
mkdir -p sft
cp train_sft_distill_with_logging.py sft/train_sft_distill_with_logging.py
chmod +x run_targeted_distill_v1.sh
```

---

## 6. Python 环境与 pip 安装

### 6.1 如果你已经有 venv
直接激活：

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

### 6.2 建议安装的包

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

- `vllm`：起 teacher
- `tokenizers`：你原项目和训练脚本需要
- `sentence-transformers`：semantic dedup
- `scikit-learn`：fallback 的向量/去重工具
- `datasets`：从 Hugging Face 导出 MBPP/APPS/no_robots 等
- `huggingface_hub`：登录/下载模型/数据更方便
- `pandas`, `matplotlib`：后面画 loss 曲线

如果模板里没有 `torch`，再补：

```bash
pip install torch
```

### 6.3 推荐额外安装
```bash
apt update
apt install -y tmux unzip
```

---

## 7. 配 Hugging Face token（推荐）

```bash
huggingface-cli login
```

或者：

```bash
export HF_TOKEN=your_hf_token_here
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
```

---

## 8. 在 Runpod 上起 vLLM teacher（Qwen-3.5）

### 8.1 推荐模型
这份 README 按下面这个 teacher 写：

- `Qwen/Qwen3.5-35B-A3B-FP8`

### 8.2 启动命令

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

### 8.3 检查 teacher 是否起来

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer EMPTY"
```

如果返回模型信息，说明 teacher 已经起来。

---

## 9. 需要准备哪些原始数据文件

### A. 项目已有
这些通常你已经有：

```bash
tokenizer/tokenizer.json
outputs/sft_v6_general_code/step_003500.pt
dataset/sft_canon/sample_eval.jsonl
src/model.py
```

### B. 这轮额外需要的本地 JSONL

#### general 侧
```bash
data/no_robots_train_sft.jsonl
data/smol_smoltalk_train.jsonl
data/alpaca_cleaned_train.jsonl
data/dolly_style_train.jsonl
```

#### code 侧
```bash
data/mbpp_train.jsonl
data/apps_intro.jsonl
```

---

## 10. 如何准备这些 JSONL

### 10.1 推荐方式
你自己先写一个小导出脚本，用 `datasets` 库下载后导出为 JSONL。

### 10.2 general 建议字段

#### no_robots
- `prompt_id`
- `prompt`
- `messages`
- `category`

#### smol_smoltalk
- `id`
- `prompt` 或 user 内容
- `response`
- `category`

#### alpaca_cleaned
- `instruction`
- `input`
- `output`

#### dolly_style
- `instruction`
- `context`
- `response`
- `category`

### 10.3 code 建议字段

#### MBPP
- `task_id`
- `text`
- `code`
- `test_list`

#### APPS intro
- `problem_id`
- `question`
- `difficulty`
- `starter_code`
- `input_output`

其中 `input_output` 最好保留成结构化 JSON 字段，里面有：

- `fn_name`
- `inputs`
- `outputs`

---

## 11. 先跑 general 侧

### 11.1 生成 template seeds
```bash
python distill/general_gen_template_seeds_v1.py \
  --out_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl
```

### 11.2 模板 paraphrase + dedup
```bash
python distill/general_paraphrase_templates_v1.py \
  --in_jsonl dataset/targeted_distill_general_v1/general_template_seeds_v1.jsonl \
  --out_raw_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_raw_v1.jsonl \
  --out_dedup_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher
```

### 11.3 抽开源 raw prompts
```bash
python distill/general_extract_open_v1.py \
  --no_robots_jsonl data/no_robots_train_sft.jsonl \
  --smol_jsonl data/smol_smoltalk_train.jsonl \
  --alpaca_jsonl data/alpaca_cleaned_train.jsonl \
  --dolly_jsonl data/dolly_style_train.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/raw_open_prompts_v1.jsonl
```

### 11.4 classify + canonicalize
```bash
python distill/general_classify_canonicalize_v1.py \
  --raw_open_jsonl dataset/targeted_distill_general_v1/raw_open_prompts_v1.jsonl \
  --template_jsonl dataset/targeted_distill_general_v1/general_template_paraphrases_dedup_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl
```

### 11.5 teacher raw generation
```bash
python distill/general_teacher_generate_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/general_teacher_raw_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.3
```

### 11.6 verify + repair candidates
```bash
python distill/general_verify_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_raw_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_general_v1/general_verified_pass_round1_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_general_v1/general_verified_reject_round1_v1.jsonl \
  --out_repair_candidates_jsonl dataset/targeted_distill_general_v1/general_teacher_repair_candidates_v1.jsonl
```

### 11.7 repair
```bash
python distill/general_teacher_generate_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_repair_candidates_v1.jsonl \
  --out_jsonl dataset/targeted_distill_general_v1/general_teacher_repaired_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.4
```

### 11.8 verify repaired
```bash
python distill/general_verify_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_general_v1/general_teacher_repaired_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_general_v1/general_verified_pass_repair_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_general_v1/general_verified_reject_repair_v1.jsonl
```

### 11.9 build general bank
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

## 12. 再跑 code 侧

### 12.1 生成 12 家族 prompts
```bash
python distill/code_gen_core_families_v1.py \
  --out_jsonl dataset/targeted_distill_code_v1/core_family_prompts_v1.jsonl \
  --instances_per_family 80
```

### 12.2 抽 MBPP
```bash
python distill/code_extract_mbpp_v1.py \
  --in_jsonl data/mbpp_train.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/mbpp_prompts_v1.jsonl
```

### 12.3 抽 APPS intro
```bash
python distill/code_extract_apps_v1.py \
  --in_jsonl data/apps_intro.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/apps_prompts_v1.jsonl
```

### 12.4 合并 prompt pool
把下面 3 个文件合并成一个：

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

### 12.5 teacher raw generation
```bash
python distill/code_teacher_generate_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/code_teacher_raw_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.15
```

### 12.6 verify + repair candidates
```bash
python distill/code_verify_v1.py \
  --mode raw \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_raw_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_code_v1/code_verified_pass_round1_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_code_v1/code_verified_reject_round1_v1.jsonl \
  --out_repair_candidates_jsonl dataset/targeted_distill_code_v1/code_teacher_repair_candidates_v1.jsonl
```

### 12.7 repair
```bash
python distill/code_teacher_generate_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_repair_candidates_v1.jsonl \
  --out_jsonl dataset/targeted_distill_code_v1/code_teacher_repaired_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.25
```

### 12.8 verify repaired
```bash
python distill/code_verify_v1.py \
  --mode repair \
  --in_jsonl dataset/targeted_distill_code_v1/code_teacher_repaired_v1.jsonl \
  --out_pass_jsonl dataset/targeted_distill_code_v1/code_verified_pass_repair_v1.jsonl \
  --out_reject_jsonl dataset/targeted_distill_code_v1/code_verified_reject_repair_v1.jsonl
```

### 12.9 build code bank
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

## 13. 组装最终 mixed distill train/val

### 13.1 先可选地做 smoke manifests
```bash
python distill/build_smoke_manifests_v1.py \
  --general_canonical_jsonl dataset/targeted_distill_general_v1/canonical_prompts_v1.jsonl \
  --code_canonical_jsonl dataset/targeted_distill_code_v1/code_canonical_prompts_v1.jsonl \
  --out_general_smoke_jsonl dataset/targeted_distill_mix_v1/general_smoke_v1.jsonl \
  --out_code_smoke_jsonl dataset/targeted_distill_mix_v1/code_smoke_v1.jsonl
```

### 13.2 build 最终 mix
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

## 14. 启动 targeted distill 训练

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

## 15. loss 记录在哪

新的训练脚本会写：

```bash
outputs/targeted_distill_v1_from_sft_v6_3500/logs/train_loss.jsonl
outputs/targeted_distill_v1_from_sft_v6_3500/logs/train_loss.csv
outputs/targeted_distill_v1_from_sft_v6_3500/logs/eval_loss.jsonl
outputs/targeted_distill_v1_from_sft_v6_3500/logs/eval_loss.csv
```

后面你画图和写 README，可以直接用 CSV。

---

## 16. 如何画 loss 曲线（最简）

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

## 17. `smoke` 到底是什么

**`smoke` 就是规模比较小的“冒烟测试 / 流水线 sanity check”。**

它的作用不是最终训练，而是先快速确认：

- teacher 能否正常返回结果
- verifier 会不会把大多数样本全拒掉
- family / source / canonicalization 有没有明显 bug
- API/base_url/模型名/文件路径是不是都通了

---

## 18. `smoke` 可以改大吗

**可以。**

但我的建议是：

### 先用小 smoke
最开始用比较小的 smoke 很有价值，因为你能很快发现：
- vLLM 没起好
- 某个脚本字段名不对
- verifier 太严
- canonicalization 出现 family shift
- repair 不工作

这一步几百条就够。

### 然后再跑全量
一旦 smoke 通过，就不要再停留在 smoke 阶段，直接跑 full pipeline。

也就是说：

- `smoke` 不是训练集
- `smoke` 只是 check pipeline

---

## 19. 如果你想“规模大一些，使得可以最终训练一次跑通”

这个完全可以，而且这才是正常路径。

### 推荐做法
#### 第一步：小 smoke
只跑：
- general 200 左右
- code 200~300 左右

确认 teacher / verifier / build 没问题。

#### 第二步：全量 bank
跑完整的：
- general accepted 4000
- code accepted 5000 左右
- build mixed train/val
- 再训练一次

所以：

**真正用于最终训练的，不应该是 smoke，而应该是 full bank。**

---

## 20. 什么时候可以跳过 smoke

如果你已经满足下面这些条件：

1. vLLM teacher 已经稳定起过  
2. general / code 两套脚本路径都没问题  
3. 你已经检查过原始 JSONL 字段  
4. 你不担心花更多 teacher 时间  

那你可以直接跑全量。

但我还是建议你至少做一次很小的 smoke，原因很简单：  
只要某个字段名不对，你全量跑会浪费很多时间。

---

## 21. 实际上的推荐顺序

如果你问我最稳的现实顺序，我会建议：

### 路线 A：稳健版
1. 起 vLLM teacher
2. 跑 very small smoke
3. 看 reject reasons
4. 修一下问题
5. 跑 general 全量
6. 跑 code 全量
7. build final mix
8. 跑 targeted distill training

### 路线 B：激进版
1. 起 vLLM teacher
2. 直接全量 general
3. 直接全量 code
4. build mix
5. 训练

我更推荐路线 A。

---

## 22. 建议你先确认的文件列表

### 项目已有
```bash
/workspace/petitgpt/tokenizer/tokenizer.json
/workspace/petitgpt/outputs/sft_v6_general_code/step_003500.pt
/workspace/petitgpt/dataset/sft_canon/sample_eval.jsonl
/workspace/petitgpt/src/model.py
```

### 新上传
```bash
/workspace/petitgpt/general_distill_scripts_v1.zip
/workspace/petitgpt/code_distill_scripts_v1.zip
/workspace/petitgpt/distill_assembly_scripts_v1.zip
/workspace/petitgpt/sft/train_sft_distill_with_logging.py
/workspace/petitgpt/run_targeted_distill_v1.sh
```

### 你准备的本地导出 JSONL
```bash
/workspace/petitgpt/data/no_robots_train_sft.jsonl
/workspace/petitgpt/data/smol_smoltalk_train.jsonl
/workspace/petitgpt/data/alpaca_cleaned_train.jsonl
/workspace/petitgpt/data/dolly_style_train.jsonl
/workspace/petitgpt/data/mbpp_train.jsonl
/workspace/petitgpt/data/apps_intro.jsonl
```

---

## 23. 一个开始前清单

```bash
cd /workspace/petitgpt
source .venv/bin/activate

python -V
nvidia-smi
which python
```

确认这些文件存在：

```bash
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

