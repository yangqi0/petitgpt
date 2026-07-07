"""Shared optimizer construction for all training stages (pretrain/SFT/distill/DPO).

Two optimizers, one entry point (`build_optimizer`):

- "muon" (default): Muon on the hidden 2D weight matrices, AdamW on everything
  else (embeddings, lm_head, RMSNorm gains, MoE router gates). Follows the
  Moonlight variant (Liu et al., 2025): the orthogonalized update is rescaled by
  `0.2 * sqrt(max(fan_in, fan_out))` so its RMS matches a typical AdamW update,
  which means the existing AdamW-tuned `--lr` and `--weight_decay` transfer
  directly. Both halves live in ONE optimizer instance (param groups flagged
  `use_muon`), so `optimizer.state_dict()` stays a single object and the
  checkpoint schema is unchanged.
- "adamw": plain AdamW, but with best-practice grouping — weight decay only on
  ndim>=2 params (matrices + embeddings), none on norms/biases — betas
  (0.9, 0.95), and the fused CUDA kernel when available.

Every param group is stamped with `lr_ratio` (its base LR relative to the
script's `--lr`). Training loops schedule the LR as:

    lr = get_lr(step)  # scheduled value of args.lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_ratio", 1.0)

so one scalar schedule drives all groups, and resuming with a different `--lr`
behaves the same as it always did.
"""

from __future__ import annotations

import math

import torch

# Params matched by these name fragments always use the AdamW update, even when
# 2D: embeddings and the (possibly tied) output head are known to do poorly
# under Muon, and MoE router gates are kept on AdamW for routing stability.
ADAM_PARAM_NAME_KEYS = ("tok_emb", "lm_head", ".gate.")


@torch.no_grad()
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T for g = USV^T via quintic Newton-Schulz iteration.

    Coefficients (3.4445, -4.7750, 2.0315) from Keller Jordan's Muon: tuned to
    maximize convergence slope at 0 rather than to converge to exactly 1, so
    singular values land in ~[0.7, 1.2] — empirically fine for the update.
    Runs in bfloat16 on CUDA for speed (fp32 elsewhere).
    """
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16() if g.is_cuda else g.float()
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.mT
    # spectral-norm upper bound via Frobenius norm -> all singular values <= 1
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x


class Muon(torch.optim.Optimizer):
    """Muon + auxiliary AdamW in a single optimizer.

    Each param group must set `use_muon`. Muon groups use keys
    (lr, weight_decay, momentum, nesterov, ns_steps) and must contain only 2D
    params; AdamW groups use (lr, weight_decay, betas, eps). Weight decay is
    decoupled (AdamW-style) in both.
    """

    def __init__(self, param_groups: list[dict]):
        defaults = dict(
            lr=1e-3,
            weight_decay=0.0,
            use_muon=False,
            lr_ratio=1.0,
            # muon
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            # adamw
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            f"Muon groups require 2D params, got shape {tuple(p.shape)}"
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            if group["use_muon"]:
                momentum = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    # Moonlight RMS matching: scale so the update RMS ~= 0.2,
                    # in line with AdamW — lets Muon reuse AdamW lr/wd.
                    adjusted_lr = lr * 0.2 * math.sqrt(max(p.size(0), p.size(1)))
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(u.to(p.dtype), alpha=-adjusted_lr)
            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(g, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bc1 = 1.0 - beta1**t
                    bc2 = 1.0 - beta2**t
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(eps)
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-lr / bc1)

        return loss


def build_optimizer(
    model: torch.nn.Module,
    name: str = "muon",
    lr: float = 1e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    muon_lr: float = 0.0,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """Build the training optimizer with correct param grouping.

    Grouping (both modes): ndim<2 params (norm gains, biases) never get weight
    decay; embeddings/lm_head/router gates always use the AdamW update; the
    remaining 2D matrices go to Muon (name="muon") or the AdamW decay group
    (name="adamw").

    muon_lr <= 0 means "use `lr` for the Muon groups too" — sensible because the
    Moonlight scaling matches Muon's update RMS to AdamW's.
    """
    matrix_params: list[torch.nn.Parameter] = []
    adam_decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    # named_parameters() deduplicates tied weights (tok_emb/lm_head appear once)
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay_params.append(p)
        elif any(k in pname for k in ADAM_PARAM_NAME_KEYS):
            adam_decay_params.append(p)
        else:
            matrix_params.append(p)

    name = name.lower()
    if name == "adamw":
        groups = [
            {"params": matrix_params + adam_decay_params, "weight_decay": weight_decay,
             "lr_ratio": 1.0},
            {"params": no_decay_params, "weight_decay": 0.0, "lr_ratio": 1.0},
        ]
        groups = [g for g in groups if g["params"]]
        use_fused = all(p.is_cuda for g in groups for p in g["params"])
        try:
            opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=use_fused)
        except (RuntimeError, TypeError):
            opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    elif name == "muon":
        muon_lr = float(muon_lr) if muon_lr and muon_lr > 0 else float(lr)
        ratio = muon_lr / lr if lr > 0 else 1.0
        groups = [
            {"params": matrix_params, "lr": muon_lr, "weight_decay": weight_decay,
             "momentum": muon_momentum, "nesterov": True, "ns_steps": ns_steps,
             "use_muon": True, "lr_ratio": ratio},
            {"params": adam_decay_params, "lr": lr, "weight_decay": weight_decay,
             "betas": betas, "eps": eps, "use_muon": False, "lr_ratio": 1.0},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0,
             "betas": betas, "eps": eps, "use_muon": False, "lr_ratio": 1.0},
        ]
        groups = [g for g in groups if g["params"]]
        opt = Muon(groups)
    else:
        raise ValueError(f"Unknown optimizer {name!r} (expected 'muon' or 'adamw')")

    if verbose:
        n_mat = sum(p.numel() for p in matrix_params)
        n_emb = sum(p.numel() for p in adam_decay_params)
        n_nd = sum(p.numel() for p in no_decay_params)
        mat_opt = "muon" if name == "muon" else "adamw"
        print(
            f"[optim] {name}: matrices({mat_opt})={n_mat / 1e6:.2f}M "
            f"embed/head/gate(adamw)={n_emb / 1e6:.2f}M norms/bias(adamw,no-wd)={n_nd / 1e6:.2f}M"
        )
    return opt
