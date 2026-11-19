"""Deterministic single-frame data loading for spatial FlameDiff tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_DIRECTORY = "9cases_slice_5_200_192_256"
FRAME_SHAPE = (192, 256)
FRAME_COUNT = 200
SPLIT_RANGES = {
    "train": (0, 130),
    "val": (140, 160),
    "test": (170, 200),
}


class TemperatureData:
    """Discover H2 temperature arrays and expose deterministic frame splits.

    ``root`` may point either to the H2 folder containing
    ``9cases_slice_5_200_192_256`` or directly to that data directory.
    Each discovered sample is a single frame, independently of its temporal
    neighbors. Frame ranges are held out as blocks to avoid splitting adjacent
    snapshots across train, validation, and test.
    """

    def __init__(self, root: str | Path):
        root_path = Path(root).expanduser()
        if root_path.name == DATA_DIRECTORY:
            self.data_root = root_path
        else:
            self.data_root = root_path / DATA_DIRECTORY
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"Expected {DATA_DIRECTORY!r} under {root_path} or as the root itself"
            )

        self._files = self._discover_files()
        self._train_range: Optional[Tuple[float, float]] = None

    def _discover_files(self) -> List[dict]:
        files: List[dict] = []
        case_dirs = sorted(
            (path for path in self.data_root.iterdir()
             if path.is_dir() and re.fullmatch(r"T\d+", path.name)),
            key=lambda path: path.name,
        )
        if not case_dirs:
            raise ValueError(f"No T<number> case directories found in {self.data_root}")

        for case_dir in case_dirs:
            slices = sorted(
                (path for path in case_dir.iterdir()
                 if path.is_dir()
                 and re.fullmatch(r"grade_192_256_npdata_\d+", path.name)),
                key=lambda path: int(path.name.rsplit("_", 1)[1]),
            )
            if not slices:
                raise ValueError(f"No 192x256 slice directories found in {case_dir}")
            for slice_dir in slices:
                path = slice_dir / "T.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing temperature array: {path}")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (FRAME_COUNT, *FRAME_SHAPE):
                    raise ValueError(
                        f"{path} has shape {array.shape}; expected "
                        f"({FRAME_COUNT}, {FRAME_SHAPE[0]}, {FRAME_SHAPE[1]})"
                    )
                if array.dtype != np.float32:
                    raise ValueError(f"{path} has dtype {array.dtype}; expected float32")
                files.append({
                    "case": case_dir.name,
                    "slice": slice_dir.name,
                    "path": path,
                    "relative_path": path.relative_to(self.data_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                })

        return files

    def dataset(
        self,
        split: str,
        temperature_range: Optional[Tuple[float, float]] = None,
    ) -> Dataset:
        """Return a lazy dataset of ``[1, 192, 256]`` float32 frames.

        Values remain in Kelvin unless ``temperature_range=(min, max)`` is
        provided, in which case they are scaled linearly without clipping.
        """
        if split not in SPLIT_RANGES:
            raise ValueError(f"split must be one of {tuple(SPLIT_RANGES)}, got {split!r}")
        if temperature_range is not None:
            if len(temperature_range) != 2:
                raise ValueError("temperature_range must contain (min, max)")
            low, high = map(float, temperature_range)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError("temperature_range must be finite with max > min")
            temperature_range = (low, high)
        return _TemperatureFrames(self._files, split, temperature_range)

    def training_range(self) -> Tuple[float, float]:
        """Return the min/max temperature computed only from training frames."""
        if self._train_range is None:
            low, high = np.inf, -np.inf
            start, stop = SPLIT_RANGES["train"]
            for item in self._files:
                array = np.load(item["path"], mmap_mode="r", allow_pickle=False)
                frames = array[start:stop]
                if not np.isfinite(frames).all():
                    raise ValueError(f"Non-finite training temperatures in {item['relative_path']}")
                low = min(low, float(frames.min()))
                high = max(high, float(frames.max()))
            self._train_range = (low, high)
        return self._train_range

    def split_manifest(self) -> dict:
        """Return a deterministic, JSON-serializable split and file manifest."""
        return {
            "version": 1,
            "policy": "single_frame_case_slice_grouped_frame_blocks",
            "data_directory": DATA_DIRECTORY,
            "frame_shape": list(FRAME_SHAPE),
            "frame_count": FRAME_COUNT,
            "split_ranges_half_open": {
                split: list(frame_range)
                for split, frame_range in SPLIT_RANGES.items()
            },
            "excluded_ranges_half_open": [[130, 140], [160, 170]],
            "grouping": "same case and frame indices across all discovered slices",
            "groups": [
                {
                    "case": case,
                    "slices": [
                        item["relative_path"]
                        for item in self._files
                        if item["case"] == case
                    ],
                    "membership": {
                        split: list(frame_range)
                        for split, frame_range in SPLIT_RANGES.items()
                    },
                }
                for case in sorted({item["case"] for item in self._files})
            ],
            "files": [
                {
                    "case": item["case"],
                    "slice": item["slice"],
                    "path": item["relative_path"],
                    "size_bytes": item["size_bytes"],
                }
                for item in self._files
            ],
        }


class _TemperatureFrames(Dataset):
    def __init__(
        self,
        files: List[dict],
        split: str,
        temperature_range: Optional[Tuple[float, float]],
    ):
        self._files = files
        self._start, self._stop = SPLIT_RANGES[split]
        self._temperature_range = temperature_range
        self._arrays: Dict[str, np.ndarray] = {}
        self._samples = [
            (file_index, frame_index)
            for file_index in range(len(files))
            for frame_index in range(self._start, self._stop)
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_index, frame_index = self._samples[index]
        path = self._files[file_index]["path"]
        key = str(path)
        if key not in self._arrays:
            self._arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        frame = np.array(self._arrays[key][frame_index], dtype=np.float32, copy=True)
        tensor = torch.from_numpy(frame).unsqueeze(0)
        if self._temperature_range is not None:
            low, high = self._temperature_range
            tensor = (tensor - low) / (high - low)
        return tensor
