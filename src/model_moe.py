"""Mixture-of-Experts variant of the PetitGPT decoder.

Same architecture as `src/model.py` (RMSNorm pre-norm, RoPE attention, SwiGLU
experts, tied embeddings, GPT-2 depth-scaled residual init) but every Block's
single dense FFN is replaced by a top-k routed MoE layer. Attention and the rest
of the stack are imported from `src.model` so the two models stay in lockstep.

MoE-specific behaviour:
- Router is a single linear layer producing per-expert logits; softmax over all
  experts, top-k selected, and (optionally) the top-k gate weights renormalized.
- A DeepSeek/Switch-style load-balancing auxiliary loss is accumulated over all
  MoE layers on every forward pass. It is exposed two ways so training code can
  fold it in without changing the common `logits = model(input_ids)` call site:
    * stashed on `model.aux_loss` after each forward, and
    * returned when called as `model(input_ids, return_aux_loss=True)`.
  Add `cfg.moe_aux_loss_coef * aux_loss` to the cross-entropy loss when training.
- Optional always-on shared experts (DeepSeek-MoE style) and optional leading
  dense layers (`n_dense_layers`) that keep a plain SwiGLU FFN.

Checkpoints embed `asdict(MoEConfig)` just like the dense model, so a checkpoint
remains self-describing and reconstructable via `MoEConfig(**cfg_dict)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import CausalSelfAttention, GPTConfig, RMSNorm, SwiGLU


@dataclass
class MoEConfig(GPTConfig):
    # Number of routed experts and how many each token is dispatched to (top-k).
    n_experts: int = 8
    n_experts_per_tok: int = 2
    # Per-expert SwiGLU hidden width. None -> reuse the dense `d_ff`.
    moe_d_ff: int | None = None
    # Always-on shared experts (DeepSeek-MoE). 0 disables. Implemented as a
    # single SwiGLU with hidden = n_shared_experts * expert_d_ff.
    n_shared_experts: int = 0
    # Leading blocks that keep a plain dense SwiGLU FFN instead of MoE.
    n_dense_layers: int = 0
    # Load-balancing auxiliary-loss weight (used by the training loop, not here).
    moe_aux_loss_coef: float = 0.01
    # Multiplicative input jitter applied to router inputs during training only.
    router_jitter: float = 0.0
    # Renormalize the selected top-k gate weights to sum to 1 per token.
    norm_topk_prob: bool = True

    def expert_d_ff(self) -> int:
        return int(self.moe_d_ff) if self.moe_d_ff is not None else int(self.d_ff)


class Expert(nn.Module):
    """A single SwiGLU FFN expert with an explicit hidden width."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        return self.drop(self.w2(x))


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of SwiGLU experts with a load-balancing aux loss."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        if cfg.n_experts_per_tok < 1 or cfg.n_experts_per_tok > cfg.n_experts:
            raise ValueError(
                f"n_experts_per_tok={cfg.n_experts_per_tok} must be in [1, n_experts={cfg.n_experts}]"
            )
        self.cfg = cfg
        self.n_experts = int(cfg.n_experts)
        self.top_k = int(cfg.n_experts_per_tok)
        self.norm_topk_prob = bool(cfg.norm_topk_prob)
        self.router_jitter = float(cfg.router_jitter)

        d_ff = cfg.expert_d_ff()
        self.gate = nn.Linear(cfg.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(cfg.d_model, d_ff, cfg.dropout) for _ in range(self.n_experts)
        ])

        if cfg.n_shared_experts > 0:
            self.shared = Expert(cfg.d_model, d_ff * int(cfg.n_shared_experts), cfg.dropout)
        else:
            self.shared = None

    def _aux_loss(self, probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """DeepSeek/Switch expert-level load-balancing loss (~1.0 when balanced).

        probs:    [N, E] full-softmax router probabilities
        topk_idx: [N, k] selected expert indices
        """
        n_tokens = probs.shape[0]
        # counts[i] = number of (token, slot) assignments routed to expert i
        onehot = F.one_hot(topk_idx, self.n_experts).sum(dim=1).float()  # [N, E]
        counts = onehot.sum(dim=0)  # [E]
        # f_i: dispatch fraction, normalized so f_i == 1 for all i under balance
        f = counts * (self.n_experts / (self.top_k * max(n_tokens, 1)))
        # P_i: mean router probability mass on expert i
        p = probs.mean(dim=0)  # [E]
        return (f * p).sum()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # [N, C]

        router_in = x_flat
        if self.training and self.router_jitter > 0.0:
            noise = torch.empty_like(router_in).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )
            router_in = router_in * noise

        logits = self.gate(router_in)  # [N, E]
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # [N, k]
        if self.norm_topk_prob:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_probs = topk_probs.to(x_flat.dtype)

        aux_loss = self._aux_loss(probs, topk_idx)

        out = torch.zeros_like(x_flat)
        # Batched dispatch: one matmul per expert over the tokens routed to it.
        for e, expert in enumerate(self.experts):
            token_ids, slot_ids = (topk_idx == e).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            gate = topk_probs[token_ids, slot_ids].unsqueeze(-1)  # [M, 1]
            contrib = gate * expert(x_flat[token_ids])  # [M, C]
            out.index_add_(0, token_ids, contrib.to(out.dtype))

        if self.shared is not None:
            out = out + self.shared(x_flat)

        return out.view(B, T, C), aux_loss


class MoEBlock(nn.Module):
    """Pre-norm transformer block; FFN is either dense SwiGLU or routed MoE."""

    def __init__(self, cfg: MoEConfig, use_moe: bool):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MoEFeedForward(cfg) if use_moe else SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.attn(self.norm1(x))
        if self.use_moe:
            ffn_out, aux_loss = self.mlp(self.norm2(x))
            x = x + ffn_out
            return x, aux_loss
        x = x + self.mlp(self.norm2(x))
        return x, None


class MoEGPT(nn.Module):
    """GPT decoder with top-k Mixture-of-Experts FFNs."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        n_dense = max(0, int(cfg.n_dense_layers))
        self.blocks = nn.ModuleList([
            MoEBlock(cfg, use_moe=(i >= n_dense)) for i in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Aux loss from the most recent forward (0 until the first call).
        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self):
        """GPT-2 depth scaling on every residual-branch output projection."""
        std = 0.02 / math.sqrt(2.0 * float(self.cfg.n_layers))
        for blk in self.blocks:
            torch.nn.init.normal_(blk.attn.proj.weight, mean=0.0, std=std)
            if blk.use_moe:
                for expert in blk.mlp.experts:
                    torch.nn.init.normal_(expert.w2.weight, mean=0.0, std=std)
                if blk.mlp.shared is not None:
                    torch.nn.init.normal_(blk.mlp.shared.w2.weight, mean=0.0, std=std)
            else:
                torch.nn.init.normal_(blk.mlp.w2.weight, mean=0.0, std=std)

    def forward(
        self, input_ids: torch.Tensor, return_aux_loss: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.cfg.max_seq_len}")
        if T < 1:
            raise ValueError("Empty sequence")

        x = self.tok_emb(input_ids)
        x = self.drop(x)

        aux_total = input_ids.new_zeros((), dtype=torch.float32)
        n_moe = 0
        for blk in self.blocks:
            x, aux = blk(x)
            if aux is not None:
                aux_total = aux_total + aux
                n_moe += 1
        if n_moe > 0:
            aux_total = aux_total / n_moe

        # Stash for training code that reads it after a plain forward call.
        self.aux_loss = aux_total.detach()

        x = self.norm_f(x)
        logits = self.lm_head(x)
        if return_aux_loss:
            return logits, aux_total
        return logits

    @torch.no_grad()
    def num_parameters(self) -> dict[str, int]:
        """Total params vs. params activated per token (top-k of the experts)."""
        total = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            total -= self.lm_head.weight.numel()  # counted once via tok_emb

        expert_params = 0
        for blk in self.blocks:
            if blk.use_moe:
                expert_params += sum(p.numel() for p in blk.mlp.experts.parameters())
        # Inactive = experts never selected for a given token.
        frac_active = self.cfg.n_experts_per_tok / max(self.cfg.n_experts, 1)
        inactive = int(expert_params * (1.0 - frac_active))
        return {"total": total, "active_per_token": total - inactive}
