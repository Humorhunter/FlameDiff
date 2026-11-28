"""Project a temperature field onto measured disjoint block means."""
import numpy as np

from ..geometry import validate_edges


def project_means(prediction, observations, row_edges, col_edges):
    rows, cols = validate_edges((row_edges, col_edges))
    shape = (rows[-1], cols[-1])
    low_shape = (len(rows) - 1, len(cols) - 1)
    field = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(observations, dtype=np.float64)
    if field.shape == (1, *shape):
        field = field[0]
    if target.shape == (1, *low_shape):
        target = target[0]
    if field.shape != shape or target.shape != low_shape:
        raise ValueError(f"Expected HR {shape} and LR {low_shape}.")
    if not np.isfinite(field).all() or not np.isfinite(target).all():
        raise ValueError("Expected finite input temperatures.")
    corrected = field.copy()
    for i, (top, bottom) in enumerate(zip(rows[:-1], rows[1:])):
        for j, (left, right) in enumerate(zip(cols[:-1], cols[1:])):
            block = corrected[top:bottom, left:right]
            block += target[i, j] - block.mean()
    if not np.isfinite(corrected).all():
        raise ArithmeticError("Nonfinite projected output.")
    return corrected
