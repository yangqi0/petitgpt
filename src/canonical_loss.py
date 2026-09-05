"""Canonical masked, weighted cross-entropy primitives.

Extracted verbatim from the accepted production trainer preserved on the historical branch
``agent/stage-o-successor-launch-adapter-v1`` so that trainer and
the P-PILOT-CONTRACT-V2.3 pilot executor share **one** implementation rather than two that could drift. The mathematics is unchanged: this is a pure move, and
``tests/test_canonical_loss.py`` pins parity against the previous production behaviour.

The two primitives are deliberately separate because a correct *global* token-mean over many
batches must accumulate the numerator and the effective weight independently and divide once at
the end. Averaging per-batch normalized means is a different (and wrong) quantity whenever the
per-batch effective weights differ, which they do as soon as the loss mask is not uniform.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

__all__ = ["masked_weighted_ce_components", "masked_weighted_ce_loss"]


def masked_weighted_ce_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eos_id: int,
    eos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the summed weighted CE numerator and its exact target weight."""
    batch_size, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(batch_size * seq_len, vocab_size)
    targets = labels.reshape(batch_size * seq_len)
    weights = loss_mask.reshape(batch_size * seq_len)

    per_token = F.cross_entropy(flat_logits, targets, reduction="none")
    if eos_weight != 1.0:
        eos_targets = (targets == int(eos_id)).to(weights.dtype)
        weights = weights * (1.0 + eos_targets * (float(eos_weight) - 1.0))

    return (per_token * weights).sum(), weights.sum()


def masked_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    """Compute target-weighted token CE, normalized by effective mask weight."""
    numerator, target_weight = masked_weighted_ce_components(
        logits,
        labels,
        loss_mask,
        eos_id=eos_id,
        eos_weight=eos_weight,
    )
    return numerator / target_weight.clamp_min(1.0)
