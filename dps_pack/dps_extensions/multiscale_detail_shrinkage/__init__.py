"""Shrink each of three spatial residual bands independently."""
import numpy as np
from scipy.ndimage import gaussian_filter


def shrink_details(aligned, baseline, observations, edges, project,
                   weights=(0.90, 0.75, 0.90)):
    aligned = np.asarray(aligned, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if aligned.ndim != 2 or aligned.shape != baseline.shape or not (
            np.isfinite(aligned).all() and np.isfinite(baseline).all()):
        raise ValueError("Expected two finite 2D fields of the same shape.")
    if weights.shape != (3,) or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("Expected three finite weights in [0, 1].")

    zeros = np.zeros_like(observations, dtype=np.float64)
    residual = project(aligned - baseline, zeros, *edges)
    small = gaussian_filter(residual, 1.0, mode="reflect", truncate=4.0)
    large = gaussian_filter(residual, 3.0, mode="reflect", truncate=4.0)
    bands = np.stack([project(band, zeros, *edges) for band in
                      (large, small - large, residual - small)])
    combined = baseline + np.sum(bands * weights[:, None, None], axis=0)
    return project(combined, observations, *edges)
