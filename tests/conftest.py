"""Shared fixtures/helpers for the test suite. All tests run on CPU with tiny
models — no GPU required."""

import json

import pytest
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import torch

from src.chat_template import configure_chat_tokenizer
from src.model import GPTConfig
from src.special_tokens import SPECIAL_TOKENS

_TINY_CORPUS = [
    "Hello world, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b\n",
    "It is four. Hi there. Say hi. Bye now.",
    "You are a helpful assistant.",
    "What is two plus two? Write a poem about rain.",
    "System user assistant roles and [EOS] literal text.",
]


@pytest.fixture(scope="session")
def chat_tok() -> Tokenizer:
    """A tiny chat tokenizer trained in-process, mirroring
    tokenizer/tokenizer_training/train_tokenizer.py (byte-level BPE, no
    normalizer, the 7 special tokens at their canonical IDs). Keeps the test
    suite independent of the checked-in tokenizer artifact."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=400,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_TINY_CORPUS, trainer=trainer)
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"special {t} got id {tok.token_to_id(t)}"
    return configure_chat_tokenizer(tok)


@pytest.fixture(scope="session")
def production_chat_tok(chat_tok: Tokenizer) -> Tokenizer:
    """Canonical 32k variant used by production data-pipeline guards."""
    payload = json.loads(chat_tok.to_str())
    vocab = payload["model"]["vocab"]
    next_id = max(vocab.values()) + 1
    for token_id in range(next_id, 32_000):
        vocab[f"__unused_test_token_{token_id}__"] = token_id
    tok = Tokenizer.from_str(json.dumps(payload))
    assert tok.get_vocab_size(with_added_tokens=True) == 32_000
    return tok


@pytest.fixture(autouse=True)
def _deterministic():
    """Make every test deterministic and fast."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def tiny_cfg() -> GPTConfig:
    """A model small enough to build/forward/backward in milliseconds on CPU."""
    return GPTConfig(
        vocab_size=256,
        n_layers=3,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,  # GQA, like the canonical config
        d_ff=160,
        max_seq_len=64,
        dropout=0.0,
        tie_embeddings=True,
    )


# --------------------------------------------------------------------- Stage-M / Stage-P native
#
# Shared bounded fixtures for tests/test_stage_m_p_repair_r1.py and
# tests/test_stage_m_p_repair_r2.py. They live here rather than being imported between test
# modules, which is what pytest expects and what keeps the linter honest about redefinition.

from pathlib import Path as _Path  # noqa: E402

from pretrain.stage_m_contract_v1 import (  # noqa: E402
    M_IMPLEMENTATION_BUNDLE_FILES as _M_BUNDLE,
    P_NATIVE_IMPLEMENTATION_BUNDLE_FILES as _P_BUNDLE,
    STAGE_STREAMS as _STAGE_STREAMS,
)
import pretrain.stage_m_realize_v1 as _realize  # noqa: E402
from tests._stage_m_fixtures import (  # noqa: E402
    e2e_records as _e2e_records,
    save_tokenizer as _save_tokenizer,
    tiny_tokenizer as _tiny_tokenizer,
    write_accepted_exclusion_authorities as _write_accepted_exclusions,
    write_accepted_stage_i as _write_accepted_stage_i,
)

_REPO_ROOT = _Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tok():
    return _tiny_tokenizer()


@pytest.fixture
def accepted(tmp_path, tok):
    records = _e2e_records(tok)
    return _write_accepted_stage_i(tmp_path / "stage_i", records, records_per_shard=64), records


@pytest.fixture
def m_run(tmp_path, tok, monkeypatch, accepted):
    """A real authorized Stage-M run: plan, authorize, publish both stage releases."""
    import hashlib

    accepted_dir, _records = accepted
    monkeypatch.setattr(_realize, "assert_tokenizer_contract", lambda path: None)
    monkeypatch.setattr(_realize, "verify_environment", lambda environment: None)
    monkeypatch.setattr(_realize, "resolve_repo_root", lambda explicit=None: tmp_path.resolve())
    # Both bundles: the native chain resolves the P helper bundle from the same root.
    for relative in (*_M_BUNDLE, *_P_BUNDLE):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_REPO_ROOT / relative).read_bytes())
    tokenizer_path = _save_tokenizer(tok, tmp_path / "tok" / "tokenizer.json")
    # R3: the canonical exclusion authority is recovered from accepted G and G2, so the fixture
    # writes those two manifests and the single L1 artifact they both name.
    canonical = _write_accepted_exclusions(tmp_path)
    plan_path = tmp_path / "candidate_m_plan.json"
    assert (
        _realize.main([
            "plan",
            "--accepted-stage-i-dir",
            str(accepted_dir),
            "--tokenizer",
            str(tokenizer_path.relative_to(tmp_path)),
            "--out",
            str(plan_path),
            "--shard-tokens",
            "4096",
            "--implementation-commit",
            "0" * 40,
        ])
        == 0
    )
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    context = _realize.authorize_plan(plan_path, plan_sha, tmp_path.resolve())
    out_dir = tmp_path / "stage_m"
    _realize.realize_and_publish(context, out_dir=out_dir)
    return {
        "tmp_path": tmp_path,
        "accepted_dir": accepted_dir,
        "plan_path": plan_path,
        "plan_sha": plan_sha,
        "tokenizer_path": tokenizer_path,
        "canonical_exclusion": canonical,
        "releases": {s: out_dir / s for s in _STAGE_STREAMS},
        "context": context,
    }


@pytest.fixture
def native_e2e(m_run, monkeypatch):
    """Bounded native route: real M releases plus the frozen G/G2 authorities, no selection."""
    import hashlib
    import json as _json

    from tests.test_plan_pretrain_run import _write_full_provenance

    tmp_path = m_run["tmp_path"]
    stage_a_train = m_run["releases"]["stage_a"] / "train"
    stage_b_train = m_run["releases"]["stage_b"] / "train"
    provenance = _write_full_provenance(tmp_path, stage_a_train, stage_b_train)

    tokenizer_bytes = m_run["tokenizer_path"].read_bytes()
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    release_tokenizer = _Path(provenance["tokenizer_release_manifest"]).parent / "tokenizer.json"
    release_tokenizer.write_bytes(tokenizer_bytes)
    for path, key in (
        (_Path(provenance["tokenizer_release_manifest"]), "tokenizer_sha256"),
        (_Path(provenance["reference_val_dir"]).parent / "manifest.json", "tokenizer_sha256"),
        (stage_a_train.parent / "meta.json", "tokenizer_sha256"),
        (stage_b_train.parent / "meta.json", "tokenizer_sha256"),
    ):
        payload = _json.loads(path.read_text(encoding="utf-8"))
        payload[key] = tokenizer_sha
        path.write_text(_json.dumps(payload), encoding="utf-8")

    # _write_full_provenance rewrites each release's reference_validation_exclusion into the
    # legacy shape. Restore the canonical R3 block so the release still names the canonical L1
    # artifact; the two payloads are byte-identical, so the digests already agree.
    canonical = m_run["canonical_exclusion"]
    for meta_path in (stage_a_train.parent / "meta.json", stage_b_train.parent / "meta.json"):
        payload = _json.loads(meta_path.read_text(encoding="utf-8"))
        payload["reference_validation_exclusion"] = {
            "enabled": True,
            "manifest_count": 1,
            "union_hash_count": canonical["derived_count"],
            "enforced_at_stage": "stage_i",
            "reapplied_by_stage_m": False,
            "canonical_artifact_path": canonical["artifact_path"],
            "canonical_artifact_sha256": canonical["artifact_sha256"],
            "manifests": [
                {
                    "enabled": True,
                    "path": canonical["artifact_path"],
                    "manifest_sha256": canonical["artifact_sha256"],
                    "hash_count": canonical["derived_count"],
                }
            ],
        }
        meta_path.write_text(_json.dumps(payload), encoding="utf-8")
        check = _json.loads(meta_path.read_text(encoding="utf-8"))
        assert (
            check["reference_validation_exclusion"]["canonical_artifact_sha256"]
            == (canonical["artifact_sha256"])
        ), f"canonical exclusion block was not restored in {meta_path}"

    # The P helper bundle must also resolve from the fixture root.
    for relative in _P_BUNDLE:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes((_REPO_ROOT / relative).read_bytes())

    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda p: None)
    return {
        **m_run,
        "stage_a_dir": stage_a_train,
        "stage_b_dir": stage_b_train,
        "reference_val_dir": provenance["reference_val_dir"],
        "tokenizer_release_manifest": provenance["tokenizer_release_manifest"],
        "selection_manifest": provenance["selection_manifest"],
    }
