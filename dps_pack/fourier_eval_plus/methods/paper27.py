"""Cheng et al. (2025), Eq.(11), adapted from velocity energy to scalar power.

Use baseline directional PSD, but pool squared errors/targets across axes before
taking a square root, as in Eq.(11). Do not average two relative axis scores.
"""
import numpy as np
from ..contracts import relative_error


def _block(pred, truth, mask):
    count = int(mask.sum())
    difference = pred[mask] - truth[mask]
    numerator = float(np.linalg.norm(difference))
    denominator = float(np.linalg.norm(truth[mask]))
    value, valid = relative_error(numerator, denominator)
    return {'NRMSE_sp': value if count else None, 'valid': valid and count > 0,
            'bins': count, 'error_squared_sum': numerator**2,
            'truth_squared_sum': denominator**2}


def evaluate(context, stride=8):
    strides = (stride, stride) if isinstance(stride, int) and not isinstance(stride, bool) else stride
    if (not isinstance(strides, (tuple, list)) or len(strides) != 2
            or any(type(s) is not int or s < 1 for s in strides)):
        raise ValueError('paper27 stride must be an integer or two positive integers.')
    rows = []
    for i, frame in enumerate(context.frame_ids):
        methods = {}
        for label in context.predictions:
            axes = {}
            for axis in (0, 1):
                name = f'axis{axis}'
                freq = context.baseline.arrays[f'freq_{name}']
                masks = {'all': freq > 0, 'resolved': (freq > 0) & (freq <= .5/strides[axis]),
                         'unresolved': freq > .5/strides[axis]}
                masks.update({band: mask for band, mask in zip(context.baseline.config.band_names,
                              context.baseline.arrays[f'band_mask_{name}'])})
                truth = context.baseline.arrays[f'psd_truth_{name}'][i]
                pred = context.baseline.arrays[f'psd_pred_{label}_{name}'][i]
                axes[name] = {key: _block(pred, truth, mask) for key, mask in masks.items()}
            joint = {}
            for key in axes['axis0']:
                blocks = [axes[a][key] for a in axes]
                numerator = sum(b['error_squared_sum'] for b in blocks)
                denominator = sum(b['truth_squared_sum'] for b in blocks)
                value, valid = relative_error(np.sqrt(numerator), np.sqrt(denominator))
                count = sum(b['bins'] for b in blocks)
                joint[key] = {'NRMSE_sp': value if count else None, 'valid': valid and count > 0,
                              'bins': count, 'error_squared_sum': numerator,
                              'truth_squared_sum': denominator}
            methods[label] = {'axes': axes, 'joint': joint}
        rows.append({'frame': int(frame), 'group': context.groups[i] if context.groups else None,
                     'methods': methods})
    return {'schema': 'paper27-nrmse-v1', 'source': '10.1063/5.0258090 Eq.(11)',
            'config': {'stride': list(strides), 'nominal_cutoff': [.5/s for s in strides],
                       'window': context.baseline.config.window,
                       'detrend': context.baseline.config.detrend,
                       'joint_reduction': 'sqrt(sum_axis_bin_error2 / sum_axis_bin_truth2)',
                       'resolved_is_sampling_reference_not_observability': True},
            'per_frame': rows, 'arrays': {}}
