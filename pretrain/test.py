import numpy as np
from pathlib import Path

p = Path("datasets/pretrain_mix/train")
f = sorted(p.glob("*.bin"))[0]
x = np.memmap(f, dtype=np.uint16, mode="r")
print("max_token_id:", int(x.max()))
