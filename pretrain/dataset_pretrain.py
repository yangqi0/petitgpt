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
    Fast dataset:
      - Each shard is uint16 token stream.
      - Blocked sampling stays within a shard (no cross-shard stitching).
      - Returns uint16 tensors; cast to long on GPU in train loop.
    """

    def __init__(self, shard_dir: str, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.block = seq_len + 1

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

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # find shard by blocks prefix
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

        toks_u16 = np.array(mm[start : start + self.block], dtype=np.uint16, copy=True)
        toks_u16 = torch.from_numpy(toks_u16)  # uint16 CPU

        return toks_u16[:-1].contiguous(), toks_u16[1:].contiguous()
