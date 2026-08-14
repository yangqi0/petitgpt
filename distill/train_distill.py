#!/usr/bin/env python3

"""Targeted distillation trainer.

This pipeline distills by behavior cloning: the teacher generates verified
answer text, and the student is fine-tuned on it with the exact same
objective, chat encoding, and loss masking as SFT (sequence-level knowledge
distillation). The trainer is therefore sft/train_sft.py — this wrapper only
exists so distill runs keep their own entry point and CLI history.

Previously this file was a 1100-line verbatim copy of train_sft.py; any fix
had to be applied twice. If distillation ever diverges (e.g. logit-level KD
against a local teacher), fork the training loop back out at that point.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sft.train_sft import main  # noqa: E402

if __name__ == "__main__":
    main()
