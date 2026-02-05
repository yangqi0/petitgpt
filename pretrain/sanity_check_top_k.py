cd /workspace/petitgpt/pretrain
python - <<'PY'
import os, sys, torch
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import GPT, GPTConfig

ckpt_path = "../checkpoints/pretrain_120m_stage2/latest.pt"  # 用你当前stage2的latest
tokenizer_path = "../tokenizer/tokenizer.json"

device = torch.device("cuda")
tok = Tokenizer.from_file(tokenizer_path)

cfg = GPTConfig(vocab_size=32000, n_layers=12, d_model=768, n_heads=12, d_ff=3072, max_seq_len=1024, dropout=0.0, tie_embeddings=True)
model = GPT(cfg).to(device)

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"]

# Normalize compiled prefix if needed
first_key = next(iter(state.keys()))
if first_key.startswith("_orig_mod."):
    state = {k[len("_orig_mod."):]: v for k, v in state.items()}

model.load_state_dict(state, strict=True)
model.eval()

prompts = [
    ("Q", "Q: Why is the sky blue?\nA:"),
    ("inst", "Instruction: Explain gradient descent in simple terms.\nResponse:"),
    ("story", "Write a short story about a small robot who learns to be kind.\n\nStory:\n"),
]

def show(prompt, add_bos):
    ids = tok.encode(prompt).ids
    if add_bos:
        ids = [2] + ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)[:, -1, :]
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    top = torch.topk(probs, k=10)
    top_ids = top.indices.tolist()
    top_ps  = top.values.tolist()

    print("\n--- add_bos =", add_bos, "len =", len(ids), "---")
    for i, (tid, p) in enumerate(zip(top_ids, top_ps), 1):
        print(i, "id", tid, "p", float(p))

for name, p in prompts:
    print("\n====================", name, "====================")
    show(p, add_bos=False)
    show(p, add_bos=True)
PY
