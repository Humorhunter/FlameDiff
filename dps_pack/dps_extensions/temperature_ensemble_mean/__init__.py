"""Average independently seeded fields in temperature space."""
import numpy as np


def ensemble_mean(members):
    fields = np.asarray(members, dtype=np.float64)
    if fields.ndim != 3 or not fields.size or not np.isfinite(fields).all():
        raise ValueError("Expected a nonempty KxHxW array of finite temperature fields.")
    return fields.mean(axis=0)
