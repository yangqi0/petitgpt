# pretrain/dataset_pretrain.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def list_shards(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    files = sorted([x for x in p.glob("*.bin") if x.is_file()])
    if not files:
        raise FileNotFoundError(f"No .bin shards found in: {dir_path}")
    return files


class PackedBinDataset(Dataset):
    """
    Robust packed dataset with EOS-collapse guards.

    Returns:
        input_ids_u16: [T] uint16
        labels_u16:    [T] uint16
        loss_mask_f32: [T] float32

    Key anti-collapse tricks:
      - Mask BOS targets (optional)
      - Mask last label in block (optional)
      - Mask *repeated EOS* targets: if labels[t]==EOS and labels[t-1]==EOS => loss_mask[t]=0
      - Reject EOS-heavy blocks by resampling a few times
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        bos_id: int = 2,
        eos_id: int = 3,
        mask_bos_in_loss: bool = True,
        mask_last_label_in_loss: bool = True,
        # NEW:
        mask_repeated_eos_in_loss: bool = True,
        max_eos_frac: float = 0.20,      # if EOS fraction in labels > this, resample
        resample_tries: int = 8,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.block = self.seq_len + 1

        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.mask_bos_in_loss = bool(mask_bos_in_loss)
        self.mask_last_label_in_loss = bool(mask_last_label_in_loss)

        self.mask_repeated_eos_in_loss = bool(mask_repeated_eos_in_loss)
        self.max_eos_frac = float(max_eos_frac)
        self.resample_tries = int(resample_tries)

        self.shards = list_shards(shard_dir)
        self._mms: List[np.memmap] = []
        self._lens: List[int] = []
        self._prefix_blocks: List[int] = [0]

        total_blocks = 0
        for f in self.shards:
            mm = np.memmap(f, dtype=np.uint16, mode="r")
            n_tokens = int(mm.shape[0])
            n_blocks = n_tokens // self.block
            self._mms.append(mm)
            self._lens.append(n_tokens)
            total_blocks += n_blocks
            self._prefix_blocks.append(total_blocks)

        if total_blocks == 0:
            raise RuntimeError(f"No full blocks found. seq_len={seq_len}, dir={shard_dir}")
        self.n_blocks = total_blocks

    def __len__(self) -> int:
        # keep a stable length; we will resample start positions internally
        return self.n_blocks

    def _pick_shard_by_index(self, i: int) -> int:
        # binary search on prefix sums
        lo, hi = 0, len(self._prefix_blocks) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._prefix_blocks[mid + 1] <= i:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _rng(self) -> np.random.Generator:
        # worker-safe RNG
        info = torch.utils.data.get_worker_info()
        if info is None:
            seed = torch.initial_seed() % (2**32)
        else:
            seed = info.seed % (2**32)
        return np.random.default_rng(seed)

    def _slice_block(self, shard_i: int, start: int) -> torch.Tensor:
        mm = self._mms[shard_i]
        toks_u16 = np.array(mm[start : start + self.block], dtype=np.uint16, copy=True)
        return torch.from_numpy(toks_u16)  # uint16 CPU

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = self._rng()

        shard_i = self._pick_shard_by_index(int(i))
        n_tokens = self._lens[shard_i]
        max_start = n_tokens - self.block
        if max_start < 0:
            raise RuntimeError("Shard too small for one block.")

        # resample if EOS-heavy / bad blocks
        last_candidate = None
        for _ in range(max(1, self.resample_tries)):
            # IMPORTANT: randomize start to avoid boundary artifacts
            start = int(rng.integers(0, max_start + 1))
            toks_u16 = self._slice_block(shard_i, start)
            last_candidate = toks_u16

            input_ids = toks_u16[:-1].contiguous()
            labels = toks_u16[1:].contiguous()

            # quick EOS-heavy check (on labels)
            eos_frac = float((labels == self.eos_id).float().mean().item())
            if eos_frac <= self.max_eos_frac:
                break

        toks_u16 = last_candidate
        input_ids = toks_u16[:-1].contiguous()
        labels = toks_u16[1:].contiguous()

        # Build loss mask
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        # 1) mask BOS targets
        if self.mask_bos_in_loss:
            loss_mask *= (labels != self.bos_id).float()

        # 2) mask repeated EOS targets (VERY IMPORTANT for EOS collapse)
        if self.mask_repeated_eos_in_loss and labels.numel() > 1:
            rep = (labels[1:] == self.eos_id) & (labels[:-1] == self.eos_id)
            loss_mask[1:] *= (~rep).float()

        # 3) mask last label to reduce boundary artifacts
        if self.mask_last_label_in_loss and loss_mask.numel() > 0:
            loss_mask[-1] = 0.0

        return input_ids, labels, loss_mask
