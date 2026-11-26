"""Deterministic single-frame data loading for single- and multi-field FlameDiff tasks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_DIRECTORY = "9cases_slice_5_200_192_256"
FRAME_SHAPE = (192, 256)
FRAME_COUNT = 200
SUPPORTED_FIELDS = ("T", "OH", "HO2")
SPLIT_RANGES = {
    "train": (0, 130),
    "val": (140, 160),
    "test": (170, 200),
}


def normalize_field_names(fields: Sequence[str] | None) -> Tuple[str, ...]:
    if fields is None:
        return ("T",)
    if isinstance(fields, str):
        names = tuple(part.strip() for part in fields.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in fields if str(part).strip())
    if not names:
        raise ValueError("fields must contain at least one field name")
    if len(set(names)) != len(names):
        raise ValueError(f"fields must be distinct, got {names!r}")
    unsupported = [name for name in names if name not in SUPPORTED_FIELDS]
    if unsupported:
        raise ValueError(f"Unsupported field(s) {unsupported!r}; expected subset of {SUPPORTED_FIELDS}")
    return names


class TemperatureData:
    """Discover H2 flame scalar arrays and expose deterministic frame splits.

    ``root`` may point either to the H2 folder containing
    ``9cases_slice_5_200_192_256`` or directly to that data directory.
    By default only ``("T",)`` is loaded for full backward compatibility.
    Passing ``fields=("T", "OH", "HO2")`` loads all three co-registered fields
    as a ``[3, 192, 256]`` tensor per frame.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        fields: Sequence[str] = ("T",),
        holdout_cases: Sequence[str] | None = None,
    ):
        root_path = Path(root).expanduser()
        if root_path.name == DATA_DIRECTORY:
            self.data_root = root_path
        else:
            self.data_root = root_path / DATA_DIRECTORY
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"Expected {DATA_DIRECTORY!r} under {root_path} or as the root itself"
            )

        self.fields: Tuple[str, ...] = normalize_field_names(fields)
        if holdout_cases is None:
            self.holdout_cases: Tuple[str, ...] = ()
        elif isinstance(holdout_cases, str):
            self.holdout_cases = tuple(sorted({p.strip() for p in holdout_cases.split(",") if p.strip()}))
        else:
            self.holdout_cases = tuple(sorted({str(p).strip() for p in holdout_cases if str(p).strip()}))
        self._all_files = self._discover_files()
        available_cases = {item["case"] for item in self._all_files}
        unknown_holdout = [c for c in self.holdout_cases if c not in available_cases]
        if unknown_holdout:
            raise ValueError(f"Unknown holdout_cases {unknown_holdout!r}; available: {sorted(available_cases)}")
        holdout_set = set(self.holdout_cases)
        self._files = [item for item in self._all_files if item["case"] not in holdout_set]
        self._holdout_files = [item for item in self._all_files if item["case"] in holdout_set]
        if not self._files:
            raise ValueError("holdout_cases cannot exclude all available cases")
        self._train_ranges: Optional[Dict[str, Tuple[float, float]]] = None
        self._train_stats: Optional[Dict[str, Dict[str, float]]] = None

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
                field_paths: Dict[str, Path] = {}
                field_sizes: Dict[str, int] = {}
                for field_name in self.fields:
                    path = slice_dir / f"{field_name}.npy"
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing {field_name} array: {path}")
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if array.shape != (FRAME_COUNT, *FRAME_SHAPE):
                        raise ValueError(
                            f"{path} has shape {array.shape}; expected "
                            f"({FRAME_COUNT}, {FRAME_SHAPE[0]}, {FRAME_SHAPE[1]})"
                        )
                    if array.dtype != np.float32:
                        raise ValueError(f"{path} has dtype {array.dtype}; expected float32")
                    field_paths[field_name] = path
                    field_sizes[field_name] = path.stat().st_size

                primary_path = field_paths[self.fields[0]]
                files.append({
                    "case": case_dir.name,
                    "slice": slice_dir.name,
                    "path": primary_path,
                    "relative_path": primary_path.relative_to(self.data_root).as_posix(),
                    "size_bytes": field_sizes[self.fields[0]],
                    "field_paths": field_paths,
                    "field_relative_paths": {
                        name: p.relative_to(self.data_root).as_posix()
                        for name, p in field_paths.items()
                    },
                    "field_sizes": field_sizes,
                })

        return files

    def _resolve_field_ranges(
        self,
        temperature_range: Optional[Tuple[float, float] | Mapping[str, Sequence[float]] | Sequence[Sequence[float]]],
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        if temperature_range is None:
            return None
        if isinstance(temperature_range, Mapping):
            ordered = []
            for field_name in self.fields:
                if field_name not in temperature_range:
                    raise ValueError(f"Missing normalization range for field {field_name!r}")
                bounds = temperature_range[field_name]
                if len(bounds) != 2:
                    raise ValueError(f"Range for {field_name!r} must contain (min, max)")
                low, high = map(float, bounds)
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    raise ValueError(f"Range for {field_name!r} must be finite with max > min")
                ordered.append((low, high))
            return tuple(ordered)
        if len(self.fields) == 1 and len(temperature_range) == 2 and not isinstance(temperature_range[0], (list, tuple)):
            low, high = map(float, temperature_range)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError("temperature_range must be finite with max > min")
            return ((low, high),)
        if len(temperature_range) != len(self.fields):
            raise ValueError(
                f"Expected {len(self.fields)} field ranges for {self.fields}, got {len(temperature_range)}"
            )
        ordered = []
        for field_name, bounds in zip(self.fields, temperature_range):
            if len(bounds) != 2:
                raise ValueError(f"Range for {field_name!r} must contain (min, max)")
            low, high = map(float, bounds)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(f"Range for {field_name!r} must be finite with max > min")
            ordered.append((low, high))
        return tuple(ordered)

    def dataset(
        self,
        split: str,
        temperature_range: Optional[Tuple[float, float] | Mapping[str, Sequence[float]] | Sequence[Sequence[float]]] = None,
    ) -> Dataset:
        """Return a lazy dataset of ``[C, 192, 256]`` float32 frames.

        Values remain in physical units unless ``temperature_range`` (either
        ``(min, max)`` for single-field or a mapping/sequence of per-field
        ``(min, max)`` bounds) is provided, in which case each channel is
        scaled linearly to ``[0, 1]`` without clipping.
        ``split="holdout"`` returns test-window frames from ``holdout_cases``.
        """
        resolved_ranges = self._resolve_field_ranges(temperature_range)
        if split == "holdout":
            if not self._holdout_files:
                raise ValueError("split='holdout' requires non-empty holdout_cases")
            return _TemperatureFrames(self._holdout_files, self.fields, "test", resolved_ranges)
        if split not in SPLIT_RANGES:
            raise ValueError(f"split must be one of {(*SPLIT_RANGES, 'holdout')}, got {split!r}")
        return _TemperatureFrames(self._files, self.fields, split, resolved_ranges)

    def training_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return per-field (min, max) computed strictly from training frames."""
        if self._train_ranges is None:
            start, stop = SPLIT_RANGES["train"]
            ranges: Dict[str, Tuple[float, float]] = {}
            for field_name in self.fields:
                low, high = np.inf, -np.inf
                for item in self._files:
                    array = np.load(item["field_paths"][field_name], mmap_mode="r", allow_pickle=False)
                    frames = array[start:stop]
                    if not np.isfinite(frames).all():
                        raise ValueError(
                            f"Non-finite training values in {item['field_relative_paths'][field_name]}"
                        )
                    low = min(low, float(frames.min()))
                    high = max(high, float(frames.max()))
                ranges[field_name] = (low, high)
            self._train_ranges = ranges
        return dict(self._train_ranges)

    def training_range(self) -> Tuple[float, float]:
        """Return (min, max) for single-field temperature datasets (legacy API)."""
        if self.fields != ("T",):
            raise ValueError("training_range() is only defined for single-field ('T',); use training_ranges()")
        return self.training_ranges()["T"]

    def training_channel_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-field physical and normalized-space mean/std on training frames."""
        if self._train_stats is None:
            start, stop = SPLIT_RANGES["train"]
            ranges = self.training_ranges()
            stats: Dict[str, Dict[str, float]] = {}
            for field_name in self.fields:
                low, high = ranges[field_name]
                span = high - low
                total_sum = 0.0
                total_sq = 0.0
                total_count = 0
                for item in self._files:
                    array = np.load(item["field_paths"][field_name], mmap_mode="r", allow_pickle=False)
                    frames = np.asarray(array[start:stop], dtype=np.float64)
                    total_sum += float(frames.sum())
                    total_sq += float(np.square(frames).sum())
                    total_count += int(frames.size)
                mean_val = total_sum / total_count
                var_val = max(total_sq / total_count - mean_val * mean_val, 0.0)
                std_val = float(np.sqrt(var_val))
                stats[field_name] = {
                    "min": float(low),
                    "max": float(high),
                    "span": float(span),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "norm_mean": float((mean_val - low) / span),
                    "norm_std": float(std_val / span),
                }
            self._train_stats = stats
        return {k: dict(v) for k, v in self._train_stats.items()}

    def split_manifest(self) -> dict:
        """Return a deterministic, JSON-serializable split and file manifest."""
        manifest = {
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
        if self.fields != ("T",):
            manifest["fields"] = list(self.fields)
            for file_entry, item in zip(manifest["files"], self._files):
                file_entry["field_paths"] = dict(item["field_relative_paths"])
                file_entry["field_sizes"] = dict(item["field_sizes"])
        if self.holdout_cases:
            manifest["holdout_cases"] = list(self.holdout_cases)
        return manifest


class _TemperatureFrames(Dataset):
    def __init__(
        self,
        files: List[dict],
        fields: Tuple[str, ...],
        split: str,
        field_ranges: Optional[Tuple[Tuple[float, float], ...]],
    ):
        self._files = files
        self._fields = fields
        self._start, self._stop = SPLIT_RANGES[split]
        self._field_ranges = field_ranges
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
        item = self._files[file_index]
        channels: List[np.ndarray] = []
        for c_idx, field_name in enumerate(self._fields):
            path = item["field_paths"][field_name]
            key = str(path)
            if key not in self._arrays:
                self._arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            frame = np.array(self._arrays[key][frame_index], dtype=np.float32, copy=True)
            if self._field_ranges is not None:
                low, high = self._field_ranges[c_idx]
                frame = (frame - low) / (high - low)
            channels.append(frame)
        return torch.from_numpy(np.stack(channels, axis=0))
