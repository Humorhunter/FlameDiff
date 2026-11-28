"""Uniformly shrink block-mean-aligned reconstruction residuals."""
import math

import numpy as np


def shrink_details(aligned, baseline, factor=0.75):
    aligned = np.asarray(aligned, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if aligned.ndim != 2 or aligned.shape != baseline.shape or not (
            np.isfinite(aligned).all() and np.isfinite(baseline).all()):
        raise ValueError("Expected two finite 2D fields of the same shape.")
    if not math.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("Expected a finite shrinkage factor in [0, 1].")
    return baseline + factor * (aligned - baseline)
