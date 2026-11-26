"""Separable linear interpolation between true block centers."""
import numpy as np

from ..geometry import validate_edges


def interpolate(observations, edges):
    rows, cols = validate_edges(edges)
    observations = np.asarray(observations, dtype=np.float64)
    if observations.shape != (len(rows) - 1, len(cols) - 1) or not np.isfinite(observations).all():
        raise ValueError("Expected a finite observation array matching the block edges.")
    row_centers = (rows[:-1] + rows[1:]) / 2
    col_centers = (cols[:-1] + cols[1:]) / 2
    x = np.arange(cols[-1]) + 0.5
    y = np.arange(rows[-1]) + 0.5
    along_cols = np.stack([np.interp(x, col_centers, row) for row in observations])
    return np.stack([np.interp(y, row_centers, col) for col in along_cols.T], axis=1)
