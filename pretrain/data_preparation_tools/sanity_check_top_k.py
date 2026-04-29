import os
import sys

from tokenizers import Tokenizer
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import GPT, GPTConfig

ckpt_path = "../checkpoints/pretrain_120m/latest.pt"
tok = Tokenizer.from_file("../tokenizer/tokenizer.json")
device = torch.device("cuda")

cfg = GPTConfig(
    vocab_size=32000,
    n_layers=12,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.0,
    tie_embeddings=True,
)
model = GPT(cfg).to(device)

ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"]
k0 = next(iter(state.keys()))
if k0.startswith("_orig_mod."):
    state = {k[len("_orig_mod.") :]: v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()

prompts = [
    "Write a short story about a small robot who learns to be kind.\n\nStory:\n",
    "Q: Why is the sky blue?\nA:",
    "Instruction: Explain gradient descent in simple terms.\nResponse:",
]

eos_id = 3
for p in prompts:
    ids = [2] + tok.encode(p).ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[:, -1, :].float().squeeze(0)  # fp32 for stable softmax
    probs = torch.softmax(logits, dim=-1)
    peos = float(probs[eos_id].item())
    top = torch.topk(probs, k=10)
    print("\nPROMPT:", p.splitlines()[0][:60], "...")
    print("P(EOS=3)=", peos)
    print("top10:", list(zip(top.indices.tolist(), [float(v) for v in top.values.tolist()])))
