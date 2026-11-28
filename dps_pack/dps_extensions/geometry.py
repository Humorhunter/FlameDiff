"""NumPy geometry shared by the independent field postprocessors."""
import numpy as np


def partition_edges(shape, low_shape):
    if len(shape) != 2 or len(low_shape) != 2 or any(
            not 0 < low <= high for high, low in zip(shape, low_shape)):
        raise ValueError("Expected two positive HR/LR dimensions with LR <= HR.")
    return tuple(np.arange(low + 1, dtype=np.int64) * high // low
                 for high, low in zip(shape, low_shape))


def validate_edges(edges):
    axes = tuple(np.asarray(axis) for axis in edges)
    if len(axes) != 2 or any(
            axis.ndim != 1 or axis.size < 2 or axis.dtype.kind not in "iu"
            or axis[0] != 0 or np.any(axis[1:] <= axis[:-1]) for axis in axes):
        raise ValueError("Expected two strictly increasing integer edge arrays starting at zero.")
    return axes


def block_means(field, edges):
    rows, cols = validate_edges(edges)
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (rows[-1], cols[-1]) or not np.isfinite(field).all():
        raise ValueError("Expected a finite field matching the partition.")
    sums = np.add.reduceat(field, rows[:-1], axis=0)
    sums = np.add.reduceat(sums, cols[:-1], axis=1)
    return sums / np.outer(np.diff(rows), np.diff(cols))
