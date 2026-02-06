import numpy as np
from pathlib import Path

bos_id = 2
eos_id = 3

f = sorted(Path("datasets/pretrain_mix/train").glob("*.bin"))[0]
x = np.memmap(f, dtype=np.uint16, mode="r")

prev = x[:-1]
nxt = x[1:]

bos_cnt = int((prev == bos_id).sum())
bos_to_eos = int(((prev == bos_id) & (nxt == eos_id)).sum())

print("bos_cnt:", bos_cnt)
print("bos->eos:", bos_to_eos, "ratio:", (bos_to_eos / max(1, bos_cnt)))
