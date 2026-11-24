"""Case-level held-out splits for spatial FlameDiff experiments.

The default spatial split holds out frame blocks from every case.  That is a
useful within-case benchmark, but it does not measure generalization to a
condition the model has never seen.  This module provides the same small data
interface as :class:`spatial_data.TemperatureData` while holding one complete
case out for final testing.

For the current nine-condition data set, T1050 and T1080 are preselected as
separate outer folds: T1050 is an interior condition and T1080 is the upper
endpoint.  These choices should remain fixed while comparing methods.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spatial_data import (
    DATA_DIRECTORY,
    FRAME_COUNT,
    FRAME_SHAPE,
    SPLIT_RANGES,
    TemperatureData,
)


RECOMMENDED_TEST_CASES = ("T1050", "T1080")
TEST_FRAME_RANGE = (0, FRAME_COUNT)


def _validated_range(
    temperature_range: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if temperature_range is None:
        return None
    if len(temperature_range) != 2:
        raise ValueError("temperature_range must contain (min, max)")
    low, high = map(float, temperature_range)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("temperature_range must be finite with max > min")
    return low, high


class CaseHoldoutData:
    """Expose train/validation/test datasets for one complete held-out case.

    Training uses frames ``[0, 130)`` from every non-held-out case.  Validation
    uses the existing disjoint frame block ``[140, 160)`` from those same
    training cases.  Testing uses every frame ``[0, 200)`` from the held-out
    case.  Normalization limits are computed only from the training files and
    training frames, then may be applied unchanged to all three splits.

    The validation split shares cases with training, as in the existing
    benchmark, but its frame indices are disjoint.  The held-out test case is
    excluded from both training and validation, and is never used to compute
    normalization limits.
    """

    def __init__(self, root: str | Path, test_case: str):
        if not re.fullmatch(r"T\d+", test_case):
            raise ValueError(f"test_case must look like T<number>, got {test_case!r}")

        discovered = TemperatureData(root)
        self.data_root = discovered.data_root
        self._files = discovered._files
        self._cases = tuple(sorted({item["case"] for item in self._files}))
        if test_case not in self._cases:
            raise ValueError(
                f"Unknown test case {test_case!r}; available cases are {self._cases}"
            )

        self.test_case = test_case
        self.train_cases = tuple(case for case in self._cases if case != test_case)
        # Validation frames come only from the non-test cases and use a
        # disjoint temporal block, matching the established train/val policy.
        self.val_cases = self.train_cases
        self.test_cases = (test_case,)
        self._train_range: Optional[Tuple[float, float]] = None

    def _files_for(self, split: str) -> List[dict]:
        if split in {"train", "val"}:
            allowed_cases = set(self.train_cases)
        elif split == "test":
            allowed_cases = {self.test_case}
        else:
            raise ValueError("split must be one of ('train', 'val', 'test')")
        return [item for item in self._files if item["case"] in allowed_cases]

    def dataset(
        self,
        split: str,
        temperature_range: Optional[Tuple[float, float]] = None,
    ) -> Dataset:
        """Return lazy ``[1, 192, 256]`` frames for the requested fold split."""
        temperature_range = _validated_range(temperature_range)
        files = self._files_for(split)
        if split == "train":
            frame_range = SPLIT_RANGES["train"]
        elif split == "val":
            frame_range = SPLIT_RANGES["val"]
        else:
            frame_range = TEST_FRAME_RANGE
        return _CaseFrames(files, frame_range, temperature_range)

    def training_range(self) -> Tuple[float, float]:
        """Return min/max from non-held-out cases' training frames only."""
        if self._train_range is None:
            low, high = np.inf, -np.inf
            start, stop = SPLIT_RANGES["train"]
            for item in self._files_for("train"):
                array = np.load(item["path"], mmap_mode="r", allow_pickle=False)
                frames = array[start:stop]
                if not np.isfinite(frames).all():
                    raise ValueError(
                        f"Non-finite training temperatures in {item['relative_path']}"
                    )
                low = min(low, float(frames.min()))
                high = max(high, float(frames.max()))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError("Training cases do not define a finite temperature range")
            self._train_range = (low, high)
        return self._train_range

    def split_manifest(self) -> dict:
        """Return a deterministic manifest suitable for checkpoint provenance."""
        train_start, train_stop = SPLIT_RANGES["train"]
        val_start, val_stop = SPLIT_RANGES["val"]
        test_start, test_stop = TEST_FRAME_RANGE
        groups = []
        for case in self._cases:
            is_test = case == self.test_case
            groups.append(
                {
                    "case": case,
                    "role": "held_out_test" if is_test else "training_pool",
                    "slices": [
                        item["relative_path"]
                        for item in self._files
                        if item["case"] == case
                    ],
                    "membership": {
                        "train": None if is_test else [train_start, train_stop],
                        "val": None if is_test else [val_start, val_stop],
                        "test": [test_start, test_stop] if is_test else None,
                    },
                }
            )

        return {
            "version": 1,
            "policy": "single_case_holdout_within_training_pool_frame_validation",
            "data_directory": DATA_DIRECTORY,
            "frame_shape": list(FRAME_SHAPE),
            "frame_count": FRAME_COUNT,
            "test_case": self.test_case,
            "train_cases": list(self.train_cases),
            "val_cases": list(self.val_cases),
            "test_cases": list(self.test_cases),
            "split_ranges_half_open": {
                "train": [train_start, train_stop],
                "val": [val_start, val_stop],
                "test": [test_start, test_stop],
            },
            "excluded_ranges_for_training_pool_half_open": [[130, 140], [160, 200]],
            "grouping": (
                "test case is excluded from train, validation, and normalization; "
                "train and validation use disjoint frame blocks of non-test cases"
            ),
            "normalization": {
                "method": "global_min_max",
                "source_split": "train",
                "source_cases": list(self.train_cases),
                "source_frame_range_half_open": [train_start, train_stop],
            },
            "groups": groups,
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

    def case_sets(self) -> dict:
        """Return the case membership used by the three data splits."""
        return {
            "train": self.train_cases,
            "val": self.val_cases,
            "test": self.test_cases,
        }

    def audit(self) -> dict:
        """Check case/frame isolation and report the fold normalization source."""
        train_data = self.dataset("train")
        val_data = self.dataset("val")
        test_data = self.dataset("test")

        train_file_cases = {item["case"] for item in train_data._files}
        val_file_cases = {item["case"] for item in val_data._files}
        test_file_cases = {item["case"] for item in test_data._files}
        if self.test_case in train_file_cases or self.test_case in val_file_cases:
            raise AssertionError("Held-out case leaked into train or validation files")
        if test_file_cases != {self.test_case}:
            raise AssertionError("Test dataset must contain exactly the held-out case")
        if train_data.frame_range != SPLIT_RANGES["train"]:
            raise AssertionError("Unexpected training frame range")
        if val_data.frame_range != SPLIT_RANGES["val"]:
            raise AssertionError("Unexpected validation frame range")
        if test_data.frame_range != TEST_FRAME_RANGE:
            raise AssertionError("Held-out test case must use all its frames")

        manifest = self.split_manifest()
        normalization_cases = set(manifest["normalization"]["source_cases"])
        if self.test_case in normalization_cases:
            raise AssertionError("Held-out case leaked into normalization sources")

        return {
            "test_case": self.test_case,
            "train_cases": list(self.train_cases),
            "val_cases": list(self.val_cases),
            "test_cases": list(self.test_cases),
            "train_frames_per_slice": train_data.frame_range[1] - train_data.frame_range[0],
            "val_frames_per_slice": val_data.frame_range[1] - val_data.frame_range[0],
            "test_frames_per_slice": test_data.frame_range[1] - test_data.frame_range[0],
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "normalization_source_cases": sorted(normalization_cases),
            "temperature_range": list(self.training_range()),
            "held_out_case_excluded_from_train_val_normalization": True,
        }


class _CaseFrames(Dataset):
    def __init__(
        self,
        files: List[dict],
        frame_range: Tuple[int, int],
        temperature_range: Optional[Tuple[float, float]],
    ):
        self._files = files
        self.frame_range = frame_range
        self._temperature_range = temperature_range
        self._arrays: Dict[str, np.ndarray] = {}
        start, stop = frame_range
        self._samples = [
            (file_index, frame_index)
            for file_index in range(len(files))
            for frame_index in range(start, stop)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one or more complete case-held-out spatial data folds."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--test-cases",
        nargs="+",
        default=list(RECOMMENDED_TEST_CASES),
        help="held-out cases; defaults to the preselected T1050 and T1080 folds",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="optional directory to save one checkpoint-compatible manifest per fold",
    )
    args = parser.parse_args()

    if len(set(args.test_cases)) != len(args.test_cases):
        parser.error("--test-cases cannot contain duplicates")

    for test_case in args.test_cases:
        fold = CaseHoldoutData(args.data, test_case)
        result = fold.audit()
        print(json.dumps(result, indent=2))
        if args.manifest_dir:
            args.manifest_dir.mkdir(parents=True, exist_ok=True)
            output = args.manifest_dir / f"case-holdout-{test_case}.json"
            output.write_text(json.dumps(fold.split_manifest(), indent=2) + "\n")
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
