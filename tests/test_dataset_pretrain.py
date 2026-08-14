from __future__ import annotations

import numpy as np
import pytest
import torch

from pretrain.dataset_pretrain import (
    FixedSubsetSampler,
    PackedBinDataset,
    ResumablePermutationSampler,
)


def _write_shards(tmp_path, chunks: list[list[int]]):
    shard_dir = tmp_path / "train"
    shard_dir.mkdir(parents=True)
    for i, chunk in enumerate(chunks):
        np.asarray(chunk, dtype=np.uint16).tofile(shard_dir / f"shard_{i:05d}.bin")
    return shard_dir


def _pairs(input_ids: torch.Tensor, labels: torch.Tensor) -> list[tuple[int, int]]:
    return list(zip(input_ids.tolist(), labels.tolist(), strict=False))


def test_deterministic_stride_covers_each_usable_transition_once_across_shards(tmp_path):
    shard_dir = _write_shards(tmp_path, [[0, 1, 2], [3, 4, 5, 6, 7]])
    ds = PackedBinDataset(
        str(shard_dir),
        seq_len=3,
        bos_id=100,
        eos_id=101,
        mask_repeated_eos_in_loss=False,
    )

    assert ds.sampling_mode == "deterministic"
    assert len(ds) == 2
    assert ds.shard_lens == [3, 5]
    assert ds._lens is ds.shard_lens
    assert ds.total_raw_tokens == 8
    assert ds.total_transitions == 7
    assert ds.usable_transitions == 6
    assert ds.tail_transitions == ds.tail_tokens == 1
    assert ds.cross_shard_transitions == 1
    assert ds.covered_cross_shard_transitions == 1

    first_x, first_y, _ = ds[0]
    second_x, second_y, _ = ds[1]
    assert first_x.tolist() == [0, 1, 2]
    assert first_y.tolist() == [1, 2, 3]
    # The second window starts at 0 + T, not at 0 + (T+1).
    assert second_x.tolist() == [3, 4, 5]
    assert second_y.tolist() == [4, 5, 6]

    observed = _pairs(first_x, first_y) + _pairs(second_x, second_y)
    assert observed == [(i, i + 1) for i in range(6)]
    assert len(observed) == len(set(observed))

    stats = ds.stats()
    assert stats["shard_lens"] == [3, 5]
    assert stats["n_blocks"] == 2
    assert stats["usable_transitions"] == 6
    assert stats["tail_transitions"] == 1


def test_only_one_global_tail_is_dropped_not_one_tail_per_shard(tmp_path):
    # Neither two-token shard can hold T+1 tokens by itself. Their virtual
    # concatenation still contains a valid full block.
    shard_dir = _write_shards(tmp_path, [[10, 11], [12, 13]])
    ds = PackedBinDataset(str(shard_dir), seq_len=3, bos_id=99, eos_id=98)

    assert len(ds) == 1
    input_ids, labels, _ = ds[0]
    assert input_ids.tolist() == [10, 11, 12]
    assert labels.tolist() == [11, 12, 13]
    assert ds.tail_transitions == 0


def test_final_eos_target_is_supervised_by_default(tmp_path):
    shard_dir = _write_shards(tmp_path, [[2, 8, 9, 10, 3]])
    ds = PackedBinDataset(str(shard_dir), seq_len=4)

    _, labels, loss_mask = ds[0]
    assert labels.tolist()[-1] == 3
    assert loss_mask.tolist()[-1] == 1.0
    assert ds.mask_last_label_in_loss is False

    legacy = PackedBinDataset(
        str(shard_dir), seq_len=4, mask_last_label_in_loss=True
    )
    _, _, legacy_mask = legacy[0]
    assert legacy_mask.tolist()[-1] == 0.0


def test_uint16_uses_compact_int16_and_preserves_high_id_fallback(tmp_path):
    canonical_dir = _write_shards(tmp_path / "canonical", [[31996, 31997, 31998, 31999]])
    canonical = PackedBinDataset(
        str(canonical_dir), seq_len=3, bos_id=2, eos_id=3
    )
    input_ids, labels, _ = canonical[0]
    assert input_ids.dtype == labels.dtype == torch.int16
    assert input_ids.tolist() == [31996, 31997, 31998]
    assert labels.tolist() == [31997, 31998, 31999]
    assert canonical.stats()["torch_common_storage_dtype"] == "int16"

    high_dir = _write_shards(tmp_path / "high", [[1, 32767, 32768, 65535]])
    high = PackedBinDataset(str(high_dir), seq_len=3, bos_id=2, eos_id=3)
    high_input, high_labels, _ = high[0]
    assert high_input.dtype == high_labels.dtype == torch.int32
    assert high_input.tolist() == [1, 32767, 32768]
    assert high_labels.tolist() == [32767, 32768, 65535]


def test_deterministic_mode_never_eos_resamples(tmp_path):
    shard_dir = _write_shards(tmp_path, [[7, 3, 3, 3, 3]])
    ds = PackedBinDataset(
        str(shard_dir),
        seq_len=4,
        max_eos_frac=0.0,
        resample_tries=20,
    )
    # Any accidental use of the random rejection path fails the test.
    ds._rng = lambda: (_ for _ in ()).throw(AssertionError("random path used"))

    input_ids, labels, _ = ds[0]
    assert input_ids.tolist() == [7, 3, 3, 3]
    assert labels.tolist() == [3, 3, 3, 3]


def test_random_sampling_is_explicit_opt_in_and_samples_with_replacement(tmp_path):
    shard_dir = _write_shards(tmp_path, [list(range(80))])
    deterministic = PackedBinDataset(
        str(shard_dir), seq_len=4, bos_id=200, eos_id=201
    )
    random_ds = PackedBinDataset(
        str(shard_dir),
        seq_len=4,
        bos_id=200,
        eos_id=201,
        sampling_mode="random",
        max_eos_frac=1.0,
    )
    random_ds._rng_inst = np.random.default_rng(123)

    assert deterministic[5][0].tolist() == deterministic[5][0].tolist()
    starts = {int(random_ds[5][0][0]) for _ in range(24)}
    assert len(starts) > 1
    with pytest.raises(ValueError, match="requires deterministic"):
        ResumablePermutationSampler(random_ds)
    with pytest.raises(ValueError, match="requires deterministic"):
        FixedSubsetSampler(random_ds, num_samples=3)

    with pytest.raises(ValueError, match="sampling_mode"):
        PackedBinDataset(str(shard_dir), seq_len=4, sampling_mode="shuffle")


def test_resumable_sampler_visits_full_permutation_before_replay():
    sampler = ResumablePermutationSampler(
        range(7), seed=42, start_position=0, num_samples=14
    )
    indices = list(sampler)

    assert sorted(indices[:7]) == list(range(7))
    assert sorted(indices[7:14]) == list(range(7))
    assert len(set(indices[:7])) == 7
    assert indices == list(
        ResumablePermutationSampler(range(7), seed=42, num_samples=14)
    )


def test_resumable_sampler_reconstructs_exact_suffix_across_epoch_boundary():
    complete = list(
        ResumablePermutationSampler(range(7), seed=11, num_samples=19)
    )
    resumed = list(
        ResumablePermutationSampler(
            range(7), seed=11, start_position=5, num_samples=14
        )
    )
    assert resumed == complete[5:]


def test_resumable_sampler_committed_state_round_trip():
    sampler = ResumablePermutationSampler(
        range(9), seed=8, start_position=2, num_samples=10
    )
    sampler.commit(5)
    assert sampler.position == 7
    assert sampler.epoch == 0
    assert sampler.epoch_offset == 7

    state = sampler.state_dict()
    restored = ResumablePermutationSampler(range(9), seed=999, num_samples=1)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert len(restored) == 5
    expected = list(
        ResumablePermutationSampler(
            range(9), seed=8, start_position=7, num_samples=5
        )
    )
    assert list(restored) == expected
    with pytest.raises(ValueError, match="exceed planned end"):
        restored.commit(6)

    wrong_dataset = ResumablePermutationSampler(range(8), seed=8)
    with pytest.raises(ValueError, match="different dataset length"):
        wrong_dataset.load_state_dict(state)


def test_default_sampler_length_stops_at_current_epoch_boundary():
    sampler = ResumablePermutationSampler(range(7), seed=3, start_position=4)
    assert len(sampler) == 3
    assert len(list(sampler)) == 3


def test_fixed_subset_sampler_is_stable_unique_and_stream_wide():
    sampler = FixedSubsetSampler(range(100), num_samples=10, seed=123)
    first = list(sampler)
    second = list(sampler)

    assert first == second
    assert len(first) == len(set(first)) == 10
    assert first == sorted(first)
    assert 0 <= first[0] < 10
    assert 90 <= first[-1] < 100
    assert first != list(FixedSubsetSampler(range(100), num_samples=10, seed=124))
    assert list(FixedSubsetSampler(range(5))) == [0, 1, 2, 3, 4]
