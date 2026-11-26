"""Model-agnostic directional Fourier evaluation for 2-D scalar fields.

The module operates on numerical arrays only.  It does not import a model,
posterior sampler, or the source database.  A batch is indexed as
``(sample, row, column)``; spatial axis 0 is the row/vertical direction and
axis 1 is the column/horizontal direction.  Frequencies are index-space
cycles per pixel, not physical wavenumbers.

Power convention
----------------
For one line of length ``N``, a periodic window ``w``, and a constant
detrend, the one-sided bin power is

    P[k] = 2 * |sum_n w[n] * x[n] * exp(-2j*pi*k*n/N)|^2
           / (N * sum_n w[n]^2)

for interior bins, with the DC and Nyquist bins not doubled.  Consequently
``sum_k P[k]`` estimates the line variance (the weighted variance when a
window is used), and power values are in squared field units per bin.  The
directional spectrum is the mean over all lines normal to the transform
axis.  The two directions are kept separate; they are not summed into an
isotropic spectrum.

Metrics use only non-DC bins.  For a direction ``d`` with spectra ``S`` and
``S_hat``::

    D_d      = sum_k |S_hat_d(k) - S_d(k)| / sum_k S_d(k)
    D_log,d  = mean_{k in A_d} |log10((S_hat_d(k)+eps_d)/(S_d(k)+eps_d))|
    C_d,B    = sum_{k in B} |S_hat_d(k) - S_d(k)| / sum_k S_d(k)
    D_d,B    = sum_{k in B} |S_hat_d(k) - S_d(k)| / sum_{k in B} S_d(k)

with ``eps_d = max(tiny, log_floor_rel * max_k S_d(k))`` and
``A_d = {k: S_d(k) > eps_d or S_hat_d(k) > eps_d}``.  ``D_log_coverage`` is
``|A_d| / (K-1)``.  DC is excluded from all of these sums; the spatial bias
is reported separately.  Band intervals are ``(lower, upper]`` and
``sum_B C_d,B = D_d``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

SCHEMA_VERSION = 'fourier-eval-v1'
FREQUENCY_UNIT = 'cycles/pixel'
SPATIAL_UNIT = 'pixel'
DEFAULT_BAND_EDGES = (0.0, 1.0 / 16.0, 1.0 / 8.0, 0.5)
DEFAULT_BAND_NAMES = ('low', 'mid', 'high')
AXES = ('axis0', 'axis1')

SPATIAL_METRICS = (
    'MAE', 'RMSE', 'bias', 'max_abs_error', 'p99_abs_error',
    'truth_min', 'truth_max', 'truth_mean',
    'pred_min', 'pred_max', 'pred_mean',
)
AXIS_METRICS = (
    'D', 'D_log', 'D_log_coverage', 'D_log_floor',
    'truth_power', 'pred_power', 'signed_rel_bias',
)
BAND_METRICS = (
    'C', 'D_band', 'signed_rel_bias',
    'truth_power', 'pred_power', 'truth_fraction', 'bin_count',
)
OVERALL_METRICS = ('D_overall', 'D_log_overall', 'signed_rel_bias_overall')


@dataclass(frozen=True)
class SpectrumConfig:
    """All choices that affect a spectrum or its metrics.

    Band intervals are ``(lower, upper]``.  The first edge must be zero and
    the last edge must be 0.5 so that every non-DC bin belongs to exactly one
    band and the band contributions sum to the direction's total deviation.
    """

    window: str = 'hann'
    detrend: str = 'constant'
    band_edges: tuple[float, ...] = DEFAULT_BAND_EDGES
    band_names: tuple[str, ...] = DEFAULT_BAND_NAMES
    log_floor_rel: float = 1e-6
    min_band_truth_fraction: float = 1e-6

    def __post_init__(self):
        window = str(self.window).lower()
        detrend = str(self.detrend).lower()
        if window not in ('hann', 'hamming', 'none'):
            raise ValueError("window must be 'hann', 'hamming', or 'none'.")
        if detrend not in ('constant', 'none'):
            raise ValueError("detrend must be 'constant' or 'none'.")
        edges = tuple(float(x) for x in self.band_edges)
        names = tuple(str(x) for x in self.band_names)
        if len(edges) < 2:
            raise ValueError('At least two band edges are required.')
        if abs(edges[0]) > 1e-12:
            raise ValueError('The first band edge must be zero.')
        if abs(edges[-1] - 0.5) > 1e-12:
            raise ValueError('The last band edge must be 0.5.')
        if any(not math.isfinite(x) for x in edges):
            raise ValueError('Band edges must be finite.')
        if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError('Band edges must be strictly increasing.')
        if len(names) != len(edges) - 1:
            raise ValueError('There must be one band name per interval.')
        if len(set(names)) != len(names) or any(not n for n in names):
            raise ValueError('Band names must be non-empty and unique.')
        if not 0 < self.log_floor_rel < 1:
            raise ValueError('log_floor_rel must be in (0,1).')
        if not 0 <= self.min_band_truth_fraction < 1:
            raise ValueError('min_band_truth_fraction must be in [0,1).')
        object.__setattr__(self, 'window', window)
        object.__setattr__(self, 'detrend', detrend)
        object.__setattr__(self, 'band_edges', edges)
        object.__setattr__(self, 'band_names', names)

    def to_dict(self):
        result = asdict(self)
        result['band_edges'] = list(self.band_edges)
        result['band_names'] = list(self.band_names)
        return result


@dataclass
class SpectrumEvaluation:
    config: SpectrumConfig
    frame_ids: np.ndarray
    groups: list[str] | None
    method_labels: tuple[str, ...]
    per_frame: list[dict]
    arrays: dict[str, np.ndarray]
    summary: dict
    summary_by_group: dict | None


def _as_batch3d(field, name):
    array = np.asarray(field)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f'{name} must have shape (N,H,W) or (H,W); got {array.shape}.')
    if array.shape[1] < 2 or array.shape[2] < 2:
        raise ValueError(f'{name} spatial dimensions must both be at least two.')
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f'{name} must be numeric.')
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains non-finite values.')
    return array


def _window(n, kind):
    if kind == 'none':
        return np.ones(n, dtype=np.float64)
    index = np.arange(n, dtype=np.float64)
    if kind == 'hann':
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * index / n)
    if kind == 'hamming':
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * index / n)
    raise ValueError(f'Unsupported window: {kind}')


def directional_psd(field, axis, config=None):
    """Return the one-sided directional PSD and frequency axis.

    ``field`` is ``(N,H,W)`` or ``(H,W)``.  ``axis`` is 0 for the vertical
    (row) direction and 1 for the horizontal (column) direction.  The result
    is ``(N,K)`` even for a single input field.
    """
    config = config or SpectrumConfig()
    if axis not in (0, 1):
        raise ValueError('axis must be 0 or 1.')
    batch = _as_batch3d(field, 'field')
    length = batch.shape[axis + 1]
    values = np.moveaxis(batch, axis + 1, -1)
    if config.detrend == 'constant':
        values = values - values.mean(axis=-1, keepdims=True)
    window = _window(length, config.window)
    transformed = np.fft.rfft(values * window, axis=-1)
    power = 2.0 * (transformed.real ** 2 + transformed.imag ** 2)
    power /= length * float(np.sum(window ** 2))
    power[..., 0] *= 0.5
    if length % 2 == 0:
        power[..., -1] *= 0.5
    power = power.mean(axis=1)
    power = np.maximum(power, 0.0)
    frequency = np.fft.rfftfreq(length, d=1.0)
    return power, frequency


def band_masks(frequency, config=None):
    """Return one boolean mask per configured band and verify the partition."""
    config = config or SpectrumConfig()
    frequency = np.asarray(frequency, dtype=np.float64)
    masks = []
    for lower, upper in zip(config.band_edges[:-1], config.band_edges[1:]):
        masks.append((frequency > lower) & (frequency <= upper))
    covered = np.zeros(frequency.shape, dtype=np.int64)
    for mask in masks:
        covered += mask.astype(np.int64)
    expected = (frequency > 0.0) & (frequency <= config.band_edges[-1])
    if not np.array_equal(covered.astype(bool), expected) or covered.max(initial=0) > 1:
        raise RuntimeError('Band masks do not form a partition of the non-DC bins.')
    return masks


def _relative_abs(pred_power, truth_power):
    numerator = float(np.abs(pred_power - truth_power).sum())
    denominator = float(truth_power.sum())
    if denominator > 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return 0.0
    return None


def _signed_bias(pred_power, truth_power):
    truth = float(truth_power.sum())
    if truth > 0.0:
        return float(pred_power.sum() - truth) / truth
    if float(pred_power.sum()) == 0.0:
        return 0.0
    return None


def _log_floor(truth_power, relative_floor):
    peak = float(truth_power.max()) if truth_power.size else 0.0
    if peak <= 0.0:
        return None
    return max(np.finfo(np.float64).tiny, relative_floor * peak)


def _log_deviation(pred_power, truth_power, floor):
    """Mean absolute log10 ratio over bins carrying signal in either field."""
    if floor is None:
        return None, 0.0
    floor = float(floor)
    active = (truth_power > floor) | (pred_power > floor)
    if not bool(active.any()):
        return 0.0, 0.0
    ratio = (pred_power[active] + floor) / (truth_power[active] + floor)
    return float(np.mean(np.abs(np.log10(ratio)))), float(active.mean())


def _spatial_metrics(truth, prediction):
    error = prediction - truth
    absolute = np.abs(error)
    return {
        'MAE': float(absolute.mean()),
        'RMSE': float(np.sqrt(np.mean(error ** 2))),
        'bias': float(error.mean()),
        'max_abs_error': float(absolute.max()),
        'p99_abs_error': float(np.percentile(absolute, 99.0)),
        'truth_min': float(truth.min()),
        'truth_max': float(truth.max()),
        'truth_mean': float(truth.mean()),
        'pred_min': float(prediction.min()),
        'pred_max': float(prediction.max()),
        'pred_mean': float(prediction.mean()),
    }


def _analyze_frame(truth, prediction, frequencies, masks, lengths, config):
    metrics = {'spatial': _spatial_metrics(truth, prediction), 'axes': {}}
    spectra = {}
    for axis in (0, 1):
        axis_name = AXES[axis]
        truth_power, _ = directional_psd(truth, axis, config)
        pred_power, _ = directional_psd(prediction, axis, config)
        error_power, _ = directional_psd(prediction - truth, axis, config)
        truth_power = truth_power[0]
        pred_power = pred_power[0]
        error_power = error_power[0]

        # All direction-level metrics exclude the DC bin.  The DC level is
        # reported through the spatial mean/bias instead; including it would
        # let a constant offset dominate a scale-content comparison.
        truth_ac = truth_power[1:]
        pred_ac = pred_power[1:]
        truth_total = float(truth_ac.sum())
        pred_total = float(pred_ac.sum())
        floor = _log_floor(truth_ac, config.log_floor_rel)
        if truth_total > 0.0:
            log_deviation, log_coverage = _log_deviation(pred_ac, truth_ac, floor)
        elif pred_total == 0.0:
            log_deviation, log_coverage = 0.0, 0.0
        else:
            log_deviation, log_coverage = None, 0.0

        bands = {}
        for band_index, name in enumerate(config.band_names):
            mask = masks[axis_name][band_index]
            bin_count = int(mask.sum())
            band_truth = float(truth_power[mask].sum())
            band_pred = float(pred_power[mask].sum())
            truth_fraction = band_truth / truth_total if truth_total > 0.0 else 0.0
            if bin_count == 0:
                band_deviation = None
                band_signed = None
                reliable = False
            else:
                band_deviation = _relative_abs(pred_power[mask], truth_power[mask])
                band_signed = _signed_bias(pred_power[mask], truth_power[mask])
                reliable = bool(
                    band_truth > 0.0
                    and band_truth >= config.min_band_truth_fraction * truth_total
                )
            if truth_total > 0.0:
                contribution = float(
                    np.abs(pred_power[mask] - truth_power[mask]).sum()
                ) / truth_total
            else:
                contribution = 0.0 if band_pred == 0.0 else None
            bands[name] = {
                'lower': float(config.band_edges[band_index]),
                'upper': float(config.band_edges[band_index + 1]),
                'bin_count': bin_count,
                'power_scope': 'non_dc',
                'truth_power': band_truth,
                'pred_power': band_pred,
                'truth_fraction': float(truth_fraction),
                'C': contribution,
                'D_band': band_deviation,
                'signed_rel_bias': band_signed,
                'reliable': reliable,
            }

        metrics['axes'][axis_name] = {
            'axis': axis,
            'length': int(lengths[axis_name]),
            'frequency_unit': FREQUENCY_UNIT,
            'nyquist': 0.5,
            'power_scope': 'non_dc',
            'D': _relative_abs(pred_ac, truth_ac),
            'D_log': log_deviation,
            'D_log_coverage': float(log_coverage),
            'D_log_floor': floor,
            'truth_power': truth_total,
            'pred_power': pred_total,
            'signed_rel_bias': _signed_bias(pred_ac, truth_ac),
            'valid': bool(truth_total > 0.0 or pred_total == 0.0),
            'bands': bands,
        }
        spectra[f'psd_truth_{axis_name}'] = truth_power
        spectra[f'psd_pred_{axis_name}'] = pred_power
        spectra[f'psd_error_{axis_name}'] = error_power
        epsilon = floor if floor is not None else np.finfo(np.float64).tiny
        # Algebraically identical, but avoids overflowing a ratio when truth
        # is exactly zero and epsilon falls back to the smallest normal float.
        spectra[f'log_ratio_{axis_name}'] = (
            np.log10(pred_power + epsilon) - np.log10(truth_power + epsilon)
        )

    axis_values = [metrics['axes'][name] for name in AXES]
    deviations = [item['D'] for item in axis_values]
    log_values = [item['D_log'] for item in axis_values]
    signed_values = [item['signed_rel_bias'] for item in axis_values]
    metrics['overall'] = {
        'D_overall': float(np.mean(deviations)) if all(v is not None for v in deviations) else None,
        'D_log_overall': float(np.mean(log_values)) if all(v is not None for v in log_values) else None,
        'signed_rel_bias_overall': float(np.mean(signed_values)) if all(v is not None for v in signed_values) else None,
        'valid': all(item['valid'] for item in axis_values),
    }
    return metrics, spectra


def _stats(values):
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {'count': 0, 'mean': None, 'std': None, 'median': None,
                'q25': None, 'q75': None, 'min': None, 'max': None}
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=1)) if array.size > 1 else 0.0,
        'median': float(np.median(array)),
        'q25': float(np.percentile(array, 25.0)),
        'q75': float(np.percentile(array, 75.0)),
        'min': float(array.min()),
        'max': float(array.max()),
    }


def _summarize(per_frame, labels, band_names):
    result = {'n_frames': len(per_frame), 'methods': {}}
    for label in labels:
        rows = [frame['methods'][label] for frame in per_frame]
        method = {
            'spatial': {
                name: _stats([row['spatial'][name] for row in rows])
                for name in SPATIAL_METRICS
            }
        }
        for axis_name in AXES:
            axis_rows = [row['axes'][axis_name] for row in rows]
            axis = {
                name: _stats([row[name] for row in axis_rows])
                for name in AXIS_METRICS
            }
            axis['valid_fraction'] = (
                float(np.mean([bool(row['valid']) for row in axis_rows]))
                if axis_rows else 0.0
            )
            axis['bands'] = {}
            for band_name in band_names:
                band_rows = [row['bands'][band_name] for row in axis_rows]
                band = {
                    name: _stats([row[name] for row in band_rows])
                    for name in BAND_METRICS
                }
                band['reliable_fraction'] = (
                    float(np.mean([bool(row['reliable']) for row in band_rows]))
                    if band_rows else 0.0
                )
                axis['bands'][band_name] = band
            method[axis_name] = axis
        overall_rows = [row['overall'] for row in rows]
        method['overall'] = {
            name: _stats([row[name] for row in overall_rows])
            for name in OVERALL_METRICS
        }
        method['overall']['valid_fraction'] = (
            float(np.mean([bool(row['valid']) for row in overall_rows]))
            if overall_rows else 0.0
        )
        result['methods'][label] = method
    return result


def evaluate_spectra(truth, predictions, config=None, frame_ids=None, groups=None):
    """Evaluate one truth batch against one or more prediction batches.

    Parameters
    ----------
    truth, predictions
        Arrays with shape ``(N,H,W)`` (or ``(H,W)`` for one frame).
    config
        ``SpectrumConfig``; defaults are Hann window, constant detrend,
        edges ``[0, 1/16, 1/8, 1/2]`` cycles/pixel.
    frame_ids
        Optional length-``N`` identifiers.  They are labels only; no temporal
        continuity is assumed.
    groups
        Optional length-``N`` group labels for separate summaries.
    """
    config = config or SpectrumConfig()
    truth_batch = _as_batch3d(truth, 'truth')
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError('predictions must be a non-empty mapping label -> array.')
    labels = tuple(str(label) for label in predictions)
    if len(set(labels)) != len(labels) or any(not label for label in labels):
        raise ValueError('Prediction labels must be non-empty and unique.')
    pred_batches = {}
    for label, value in predictions.items():
        batch = _as_batch3d(value, f'prediction[{label}]')
        if batch.shape != truth_batch.shape:
            raise ValueError(
                f'prediction[{label}] shape {batch.shape} does not match truth {truth_batch.shape}.'
            )
        pred_batches[str(label)] = batch

    n_frames = truth_batch.shape[0]
    if frame_ids is None:
        frame_ids = np.arange(n_frames, dtype=np.int64)
    frame_ids = np.asarray(frame_ids).reshape(-1)
    if frame_ids.size != n_frames:
        raise ValueError('frame_ids length must equal the number of samples.')
    if groups is not None:
        groups = [str(group) for group in groups]
        if len(groups) != n_frames:
            raise ValueError('groups length must equal the number of samples.')

    frequencies = {
        'axis0': np.fft.rfftfreq(truth_batch.shape[1], d=1.0),
        'axis1': np.fft.rfftfreq(truth_batch.shape[2], d=1.0),
    }
    lengths = {'axis0': truth_batch.shape[1], 'axis1': truth_batch.shape[2]}
    masks = {name: band_masks(frequencies[name], config) for name in AXES}

    per_frame = []
    truth_spectra = {axis_name: [] for axis_name in AXES}
    pred_spectra = {(label, axis_name): [] for label in labels for axis_name in AXES}
    error_spectra = {(label, axis_name): [] for label in labels for axis_name in AXES}
    ratios = {(label, axis_name): [] for label in labels for axis_name in AXES}

    for index in range(n_frames):
        frame_metrics = {
            'index': int(index),
            'frame': int(frame_ids[index]),
            'group': groups[index] if groups is not None else None,
            'methods': {},
        }
        frame_truth_spectra = {}
        for label in labels:
            metrics, spectra = _analyze_frame(
                truth_batch[index], pred_batches[label][index],
                frequencies, masks, lengths, config,
            )
            frame_metrics['methods'][label] = metrics
            for axis_name in AXES:
                pred_spectra[(label, axis_name)].append(spectra[f'psd_pred_{axis_name}'])
                error_spectra[(label, axis_name)].append(spectra[f'psd_error_{axis_name}'])
                ratios[(label, axis_name)].append(spectra[f'log_ratio_{axis_name}'])
                frame_truth_spectra.setdefault(
                    axis_name, spectra[f'psd_truth_{axis_name}']
                )
        for axis_name in AXES:
            truth_spectra[axis_name].append(frame_truth_spectra[axis_name])
        per_frame.append(frame_metrics)

    arrays = {
        'frame_ids': frame_ids.astype(np.int64),
        'band_edges': np.asarray(config.band_edges, dtype=np.float64),
        'band_names': np.asarray(config.band_names, dtype='U32'),
        'freq_axis0': frequencies['axis0'],
        'freq_axis1': frequencies['axis1'],
        'wavelength_axis0': np.divide(
            1.0, frequencies['axis0'],
            out=np.full_like(frequencies['axis0'], np.inf),
            where=frequencies['axis0'] > 0,
        ),
        'wavelength_axis1': np.divide(
            1.0, frequencies['axis1'],
            out=np.full_like(frequencies['axis1'], np.inf),
            where=frequencies['axis1'] > 0,
        ),
        'band_mask_axis0': np.stack(masks['axis0']),
        'band_mask_axis1': np.stack(masks['axis1']),
    }
    if groups is not None:
        arrays['groups'] = np.asarray(groups, dtype='U64')
    for axis_name in AXES:
        arrays[f'psd_truth_{axis_name}'] = np.stack(truth_spectra[axis_name])
        for label in labels:
            arrays[f'psd_pred_{label}_{axis_name}'] = np.stack(pred_spectra[(label, axis_name)])
            arrays[f'psd_error_{label}_{axis_name}'] = np.stack(error_spectra[(label, axis_name)])
            arrays[f'log_ratio_{label}_{axis_name}'] = np.stack(ratios[(label, axis_name)])

    summary = _summarize(per_frame, labels, config.band_names)
    summary_by_group = None
    if groups is not None:
        summary_by_group = {}
        for group in sorted(set(groups)):
            selected = [frame for frame in per_frame if frame['group'] == group]
            summary_by_group[group] = _summarize(selected, labels, config.band_names)

    return SpectrumEvaluation(
        config=config,
        frame_ids=frame_ids,
        groups=groups,
        method_labels=labels,
        per_frame=per_frame,
        arrays=arrays,
        summary=summary,
        summary_by_group=summary_by_group,
    )
