"""Validate inputs, compute the baseline once, then execute selected plugins."""
from dataclasses import dataclass
from importlib import import_module
from collections.abc import Mapping
import re
import numpy as np
from fourier_eval_plus.directional_spectrum import SpectrumConfig, evaluate_spectra
from .contracts import EvaluationContext, summarize

REGISTRY = {
    'paper27': 'fourier_eval_plus.methods.paper27',
    'parseval': 'fourier_eval_plus.methods.parseval',
}
SCHEMA = 'fourier-eval-plus-v2'


@dataclass
class PlusEvaluation:
    baseline: object
    enhancements: dict
    config: dict


def available_enhancements():
    return tuple(REGISTRY)


def _real_batch(value, name):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError(f'{name} must be a real numeric scalar field.')
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[0] == 0 or min(a.shape[1:]) < 2:
        raise ValueError(f'{name} must be nonempty (N,H,W), H/W >= 2.')
    a = a.astype(np.float64, copy=False)
    if not np.isfinite(a).all():
        raise ValueError(f'{name} contains NaN/Inf.')
    return a


def evaluate(truth, predictions, *, enhancements=('paper27', 'parseval'),
             spectrum_config=None, options=None, frame_ids=None, groups=None):
    """Compute only enabled methods. An empty tuple gives the original baseline alone.

    Frame IDs are unique integer labels, not an alignment mechanism. Arrays must
    already be aligned. Every plugin receives the same read-only arrays/context.
    """
    truth = _real_batch(truth, 'truth')
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError('predictions must be a nonempty label -> field mapping.')
    pred = {}
    for label, value in predictions.items():
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            raise ValueError('Method labels must use letters, numbers, underscore or hyphen.')
        a = _real_batch(value, label)
        if a.shape != truth.shape:
            raise ValueError(f'{label} shape differs from truth.')
        pred[label] = a
    ids = np.arange(len(truth)) if frame_ids is None else np.asarray(frame_ids)
    if (ids.ndim != 1 or len(ids) != len(truth) or np.iscomplexobj(ids)
            or not np.issubdtype(ids.dtype, np.number) or not np.isfinite(ids).all()
            or not np.equal(ids, np.floor(ids)).all()):
        raise ValueError('frame_ids must be a finite integer vector matching N.')
    limits = np.iinfo(np.int64)
    if any(int(x) < limits.min or int(x) > limits.max for x in ids):
        raise ValueError('frame_ids exceed int64.')
    ids = ids.astype(np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('frame_ids must be unique.')
    if groups is not None:
        groups = list(groups)
        if len(groups) != len(truth) or any(not isinstance(x, str) or len(x) > 64 for x in groups):
            raise ValueError('groups must be N strings, each at most 64 characters.')
    selected = tuple(enhancements)
    if len(set(selected)) != len(selected) or any(x not in REGISTRY for x in selected):
        raise ValueError(f'Unknown/duplicate enhancement; choose {available_enhancements()}.')
    options = dict(options or {})
    if set(options) - set(selected):
        raise ValueError('Options provided for a disabled/unknown enhancement.')
    cfg = spectrum_config or SpectrumConfig()
    reserved = {'dc', 'all', 'resolved', 'unresolved'}
    if set(cfg.band_names) & reserved:
        raise ValueError(f'Band names must not use reserved names {sorted(reserved)}.')
    # Views prevent accidental writes without changing caller-array write flags.
    truth = truth.view(); truth.flags.writeable = False
    pred = {k: v.view() for k, v in pred.items()}
    for a in pred.values():
        a.flags.writeable = False
    baseline = evaluate_spectra(truth, pred, config=cfg, frame_ids=ids, groups=groups)
    context = EvaluationContext(truth, pred, ids, groups, baseline)
    output = {}
    for key in selected:
        module = import_module(REGISTRY[key])
        result = module.evaluate(context, **options.get(key, {}))
        result['summary'] = {label: summarize([x['methods'][label] for x in result['per_frame']])
                             for label in pred}
        result['summary_by_group'] = None if groups is None else {
            group: {label: summarize([row['methods'][label] for row in result['per_frame']
                                     if row['group'] == group]) for label in pred}
            for group in sorted(set(groups))}
        output[key] = result
    return PlusEvaluation(baseline, output, {'schema': SCHEMA, 'baseline': cfg.to_dict(),
                          'enhancements': list(selected), 'options': options,
                          'array_alignment': 'caller aligned; no automatic reordering'})
