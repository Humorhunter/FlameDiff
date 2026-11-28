"""Read externally supplied high-resolution temperature frames in Kelvin."""
import numpy as np
import torch
from torch.utils.data import Dataset


class TemperatureFrames(Dataset):
    def __init__(self, path):
        self.fields = np.load(path, mmap_mode="r", allow_pickle=False)
        if (self.fields.ndim != 3 or self.fields.shape[1:] != (192, 256)
                or not len(self.fields) or self.fields.dtype.kind not in "fiu"):
            raise ValueError("Expected a nonempty Nx192x256 numeric .npy array in K.")

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, index):
        field = np.array(self.fields[index], dtype=np.float32)
        if not np.isfinite(field).all():
            raise ValueError(f"Nonfinite temperature in frame {index}")
        return torch.from_numpy(field[None])
