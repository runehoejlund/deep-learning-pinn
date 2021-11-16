# %%
from torch.utils.data import Dataset
import torch
import numpy as np
class DatasetFunction1D(Dataset):
    """1D function dataset."""

    def __init__(self, x):
        super(Dataset, self).__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        f = 1 + 2 * x + 3 * x**2 + 4 * x**3
        f_grad = 2 + 6 * x + 12 * x**2
        return (x, f, f_grad)