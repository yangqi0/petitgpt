import numpy as np
from pathlib import Path

import torch

print(torch.load("../outputs/pretrain_120m/latest.pt").keys())
