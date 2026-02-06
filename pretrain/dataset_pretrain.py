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
    Fast packed dataset:
      - Each shard is a uint16 token stream (continuous).
      - We sample fixed blocks of length (seq_len + 1) within each shard.
      - Returns:
          input_ids_u16: [T] uint16 (CPU)
          labels_u16:    [T] uint16 (CPU)
          loss_mask_f32: [T] float32 (CPU), 1.0 means "use this position in loss"
    Notes:
      - We intentionally build a MiniMind-style loss_mask to avoid over-training
        BOS/EOS or boundary artifacts.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        bos_id: int = 2,
        eos_id: int = 3,
        mask_bos_in_loss: bool = True,
        mask_last_label_in_loss: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.block = self.seq_len + 1

        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.mask_bos_in_loss = bool(mask_bos_in_loss)
        self.mask_last_label_in_loss = bool(mask_last_label_in_loss)

        self.shards = list_shards(shard_dir)
        self._mms: List[np.memmap] = []
        self._prefix_blocks: List[int] = [0]

        total_blocks = 0
        for f in self.shards:
            mm = np.memmap(f, dtype=np.uint16, mode="r")
            n_tokens = int(mm.shape[0])
            n_blocks = n_tokens // self.block
            self._mms.append(mm)
            total_blocks += n_blocks
            self._prefix_blocks.append(total_blocks)

        if total_blocks == 0:
            raise RuntimeError(f"No full blocks found. seq_len={seq_len}, dir={shard_dir}")
        self.n_blocks = total_blocks

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Find shard by prefix blocks (binary search).
        lo, hi = 0, len(self._prefix_blocks) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._prefix_blocks[mid + 1] <= i:
                lo = mid + 1
            else:
                hi = mid
        shard_i = lo
        local_block = i - self._prefix_blocks[shard_i]
        start = local_block * self.block
        mm = self._mms[shard_i]

        # Copy slice into writable CPU memory.
        toks_u16 = np.array(mm[start : start + self.block], dtype=np.uint16, copy=True)
        toks_u16 = torch.from_numpy(toks_u16)  # uint16 CPU

        input_ids = toks_u16[:-1].contiguous()  # [T] uint16
        labels = toks_u16[1:].contiguous()      # [T] uint16

        # Build MiniMind-style loss mask: float32 [T]
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        # 1) Do not train on BOS tokens if they appear in labels.
        if self.mask_bos_in_loss:
            loss_mask = loss_mask * (labels != self.bos_id).float()

        # 2) Optionally mask the last label in each block to reduce boundary artifacts.
        # This is a conservative stabilization trick (cheap and effective).
        if self.mask_last_label_in_loss and loss_mask.numel() > 0:
            loss_mask[-1] = 0.0

        return input_ids, labels, loss_mask
