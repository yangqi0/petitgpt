from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    n_layers: int = 16
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 2048  # SwiGLU recommended width for d_model=768
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True

    # RoPE (rotary positional embedding)
    rope_theta: float = 10000.0
    rope_pct: float = 1.0  # fraction of head_dim to rotate (1.0 = full head_dim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: [..., D]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Precomputes RoPE cos/sin caches up to max_seq_len."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0, pct: float = 1.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.theta = float(theta)
        self.pct = float(pct)

        rope_dim = int(self.head_dim * self.pct)
        rope_dim = rope_dim - (rope_dim % 2)
        rope_dim = max(0, min(rope_dim, self.head_dim))
        self.rope_dim = rope_dim

        if self.rope_dim > 0:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
            t = torch.arange(self.max_seq_len, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)  # [T, rope_dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [T, rope_dim]
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos = torch.empty(self.max_seq_len, 0, dtype=torch.float32)
            sin = torch.empty(self.max_seq_len, 0, dtype=torch.float32)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q,k. q,k: [B, nH, T, Hd]"""
        if seq_len > self.max_seq_len:
            raise ValueError(f"T={seq_len} exceeds max_seq_len={self.max_seq_len} for RoPE cache")
        if self.rope_dim == 0:
            return q, k

        cos = self.cos_cached[:seq_len].to(dtype=q.dtype, device=q.device)  # [T, rope_dim]
        sin = self.sin_cached[:seq_len].to(dtype=q.dtype, device=q.device)  # [T, rope_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,rope_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        q1, q2 = q[..., : self.rope_dim], q[..., self.rope_dim :]
        k1, k2 = k[..., : self.rope_dim], k[..., self.rope_dim :]

        q1 = q1 * cos + _rotate_half(q1) * sin
        k1 = k1 * cos + _rotate_half(k1) * sin

        q = torch.cat([q1, q2], dim=-1)
        k = torch.cat([k1, k2], dim=-1)
        return q, k


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return self.drop(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        # QKV fused: one matmul instead of three
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # residual branch output projection
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
            pct=cfg.rope_pct,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length T={T} exceeds max_seq_len={self.cfg.max_seq_len}")

        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)  # [B,nH,T,Hd]
        k = k.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, seq_len=T)

        dropout_p = float(self.cfg.dropout) if (self.training and self.cfg.dropout > 0) else 0.0

        if q.device.type == "cuda":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = torch.matmul(q * scale, k.transpose(-2, -1))  # [B,nH,T,T]
            mask = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if dropout_p > 0.0:
                att = F.dropout(att, p=dropout_p)
            y = torch.matmul(att, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return self.drop(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Base init everywhere...
        self.apply(self._init_weights)
        # ...then scale init ONLY on residual-branch output projections: attn.proj and mlp.w2
        self._init_residual_projections()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self):
        std = 0.02 / math.sqrt(2.0 * float(self.cfg.n_layers))
        for blk in self.blocks:
            torch.nn.init.normal_(blk.attn.proj.weight, mean=0.0, std=std)
            torch.nn.init.normal_(blk.mlp.w2.weight, mean=0.0, std=std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"T={T} exceeds max_seq_len={self.cfg.max_seq_len}")
        if T < 1:
            raise ValueError("Empty sequence")

        x = self.tok_emb(input_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)
