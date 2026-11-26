"""Shared data contract only; no enhancement imports another enhancement."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EvaluationContext:
    truth: np.ndarray
    predictions: dict
    frame_ids: np.ndarray
    groups: list | None
    baseline: object


def relative_error(numerator, denominator):
    """Return value and validity without adding an arbitrary denominator epsilon."""
    if denominator > 0:
        return float(numerator / denominator), True
    if numerator == 0:
        return 0.0, True
    return None, False


def statistics(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'std': None}
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Enhancements must not silently discard nonfinite results.')
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def summarize(records):
    """Per-leaf statistics, with explicit valid-count information for every metric."""
    first = records[0]
    if isinstance(first, dict):
        return {k: summarize([row[k] for row in records]) for k in first}
    if isinstance(first, (str, bool)):
        return first if all(x == first for x in records) else {'values': records}
    if first is None or isinstance(first, (int, float, np.number)):
        return statistics(records)
    raise TypeError(f'Unsupported summary leaf {type(first)}')
