"""Canonical absolute-step learning-rate schedule.

Extracted verbatim from the accepted production trainer preserved on the historical branch
``agent/stage-o-successor-launch-adapter-v1`` so that trainer and
the P-PILOT-CONTRACT-V2.3 pilot verification share one implementation. The mathematics is
unchanged -- in particular the WSD decay is **cosine**, not linear -- and
``tests/test_canonical_schedule.py`` pins parity.
"""

from __future__ import annotations

import math

__all__ = ["lr_schedule"]


def lr_schedule(
    step: int,
    warmup_steps: int,
    base_lr: float,
    *,
    schedule: str = "cosine",
    schedule_total_steps: int = 0,
    decay_start_step: int = -1,
    decay_end_step: int = -1,
    min_lr_ratio: float = 0.1,
) -> float:
    """Absolute-step LR schedule independent of the current stage stop.

    WSD means warmup -> stable -> cosine decay -> floor. Keeping the schedule
    horizon independent from ``max_steps`` lets Stage A stop and Stage B resume
    without an LR discontinuity.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        start = int(warmup_steps)
        end = int(schedule_total_steps)
    elif schedule == "wsd":
        start = int(decay_start_step)
        end = int(decay_end_step)
        if step < start:
            return base_lr
    else:
        raise ValueError(f"unknown lr schedule: {schedule}")

    if end <= start:
        raise ValueError(f"decay end must be greater than start: {end} <= {start}")
    if step <= start:
        return base_lr
    if step >= end:
        return base_lr * float(min_lr_ratio)
    t = (step - start) / float(end - start)
    t = min(max(t, 0.0), 1.0)
    min_lr = base_lr * float(min_lr_ratio)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
