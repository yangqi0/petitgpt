"""Chat-template tests.

The template constants and rendering logic are duplicated verbatim across
sft/train_sft.py, distill/train_distill.py, dpo/dpo.py, and grpo/grpo.py
(intentionally, per CLAUDE.md). Training/inference template drift is the classic
silent SFT bug, so these tests (a) guard the copies against drifting apart and
(b) pin down the loss-masking contract: sequences start with an unsupervised
BOS, and the supervised span is exactly the assistant *content* tokens plus one
trailing EOS per assistant turn (so the model is taught to stop).
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

import distill.train_distill as distill_mod
import dpo.dpo as dpo_mod
import grpo.grpo as grpo_mod
import sft.train_sft as sft_mod
from sft.train_sft import (
    build_example,
    clean_text_assistant,
    encode_strip_special,
    render_segments_plain,
)

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"
BOS_ID, EOS_ID, PAD_ID = 2, 3, 0


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer not found at {TOKENIZER_PATH}")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_template_constants_identical_across_scripts():
    """All duplicated copies must stay byte-for-byte identical."""
    for attr in ("SYS_PREFIX", "USER_PREFIX", "ASSIST_PREFIX", "SEP"):
        s = getattr(sft_mod, attr)
        assert getattr(dpo_mod, attr) == s, f"{attr} drifted: dpo vs sft"
        assert getattr(distill_mod, attr) == s, f"{attr} drifted: distill vs sft"
        assert getattr(grpo_mod, attr) == s, f"{attr} drifted: grpo vs sft"


def test_template_constant_values():
    assert sft_mod.SYS_PREFIX == "System: "
    assert sft_mod.USER_PREFIX == "User: "
    assert sft_mod.ASSIST_PREFIX == "Assistant: "
    assert sft_mod.SEP == "\n\n"


def test_render_segments_supervises_only_assistant_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello friend"},
    ]
    segs = render_segments_plain(messages, default_system="")
    supervised_text = "".join(t for t, sup in segs if sup)
    unsupervised_text = "".join(t for t, sup in segs if not sup)
    # only the assistant content is supervised
    assert supervised_text == clean_text_assistant("Hello friend")
    # prefixes and user/system content are not supervised
    assert "Assistant: " in unsupervised_text
    assert "User: " in unsupervised_text
    assert "Hi there" in unsupervised_text


def test_build_example_masks_prompt_and_bos(tok):
    ex = {
        "messages": [
            {"role": "user", "content": "What is two plus two"},
            {"role": "assistant", "content": "It is four"},
        ]
    }
    input_ids, labels, weight = build_example(
        ex,
        tok,
        seq_len=64,
        pad_id=PAD_ID,
        default_system="You are helpful.",
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        refusal_downweight=1.0,
        refusal_patterns=[],
        refusal_mode="contains_any",
    )
    assert input_ids.shape == labels.shape
    # sequences always start with an unsupervised BOS
    assert input_ids[0].item() == BOS_ID
    assert labels[0].item() == -100

    pairs = list(zip(input_ids.tolist(), labels.tolist(), strict=True))
    supervised_ids = [tid for tid, lab in pairs if lab != -100]
    expected = encode_strip_special(tok, clean_text_assistant("It is four"), BOS_ID, EOS_ID)
    # supervised span = assistant content + one trailing EOS (the stop signal)
    assert supervised_ids == expected + [EOS_ID]
    # where supervised, label == input (pre-shift alignment)
    for tid, lab in pairs:
        if lab != -100:
            assert lab == tid
    assert weight > 0


def test_each_assistant_turn_ends_with_supervised_eos(tok):
    ex = {
        "messages": [
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Say bye"},
            {"role": "assistant", "content": "Bye"},
        ]
    }
    input_ids, labels, _ = build_example(
        ex,
        tok,
        seq_len=64,
        pad_id=PAD_ID,
        default_system="",
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        refusal_downweight=1.0,
        refusal_patterns=[],
        refusal_mode="contains_any",
    )
    pairs = list(zip(input_ids.tolist(), labels.tolist(), strict=True))
    supervised_ids = [tid for tid, lab in pairs if lab != -100]
    a1 = encode_strip_special(tok, clean_text_assistant("Hi"), BOS_ID, EOS_ID)
    a2 = encode_strip_special(tok, clean_text_assistant("Bye"), BOS_ID, EOS_ID)
    assert supervised_ids == a1 + [EOS_ID] + a2 + [EOS_ID]


def test_empty_messages_rejected(tok):
    with pytest.raises(ValueError):
        build_example(
            {"messages": []},
            tok,
            seq_len=32,
            pad_id=PAD_ID,
            default_system="",
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            refusal_downweight=1.0,
            refusal_patterns=[],
            refusal_mode="contains_any",
        )
