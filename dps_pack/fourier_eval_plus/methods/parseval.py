"""FFL Eq.(7)-(8) inspired, uniform complex distance with exact error accounting.

This is an evaluation adaptation, not the original dynamic-weight FFL loss.
No taper or detrend: each axis independently decomposes the FULL spatial MSE.
Single-sided orthonormal rFFT uses multiplicity 2 except DC and Nyquist.
"""
import numpy as np


def _curves(truth, prediction, axis):
    x = np.fft.rfft(np.moveaxis(truth, axis, -1), axis=-1, norm='ortho')
    y = np.fft.rfft(np.moveaxis(prediction, axis, -1), axis=-1, norm='ortho')
    a, b = np.abs(x), np.abs(y)
    ux = np.divide(x, a, out=np.zeros_like(x), where=a > 0)
    uy = np.divide(y, b, out=np.zeros_like(y), where=b > 0)
    error = np.abs(y - x)**2
    magnitude = (b - a)**2
    # Stable even at zero magnitude: no phase angles or wrapped-angle averages.
    phase = a*b*np.abs(uy - ux)**2
    n = truth.shape[axis]
    weights = np.full(x.shape[-1], 2.)
    weights[0] = 1.
    if n % 2 == 0:
        weights[-1] = 1.
    return {key: values.sum(axis=0)*weights/truth.size
            for key, values in [('error', error), ('magnitude', magnitude), ('phase', phase)]}


def _block(curves, mask, total):
    result = {key+'_mse': float(value[mask].sum()) for key, value in curves.items()}
    result['rmse'] = float(np.sqrt(result['error_mse']))
    result['fraction_of_total_mse'] = result['error_mse']/total if total > 0 else 0.
    result['bins'] = int(mask.sum())
    return result


def evaluate(context):
    rows = []; arrays = {}
    bands = context.baseline.config.band_names
    for i, frame in enumerate(context.frame_ids):
        methods = {}
        for label, prediction in context.predictions.items():
            truth = context.truth[i]; pred = prediction[i]
            direct_mse = float(np.mean((pred-truth)**2))
            axes = {}
            for axis in (0, 1):
                name = f'axis{axis}'
                curves = _curves(truth, pred, axis)
                f = context.baseline.arrays[f'freq_{name}']
                total = float(curves['error'].sum())
                if not np.isclose(total, direct_mse, rtol=2e-11, atol=1e-18):
                    raise ArithmeticError('Parseval identity failed.')
                if not np.allclose(curves['error'], curves['magnitude']+curves['phase'],
                                   rtol=2e-10, atol=max(total*1e-13, 1e-25)):
                    raise ArithmeticError('Magnitude/phase accounting identity failed.')
                masks = {'dc': f == 0}
                masks.update(dict(zip(bands, context.baseline.arrays[f'band_mask_{name}'])))
                axes[name] = {'total': _block(curves, np.ones(len(f), dtype=bool), total),
                              'bands': {key: _block(curves, mask, total) for key, mask in masks.items()},
                              'parseval_absolute_residual': abs(total-direct_mse)}
                for key, value in curves.items():
                    arrays.setdefault(f'{label}_{name}_{key}_mse', []).append(value)
            methods[label] = {'spatial_mse': direct_mse, 'spatial_rmse': float(np.sqrt(direct_mse)),
                              'axes': axes}
        rows.append({'frame': int(frame), 'group': context.groups[i] if context.groups else None,
                     'methods': methods})
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    return {'schema': 'parseval-error-v1',
            'source': 'FFL 2012.12821 Eq.(7)-(8) inspiration + explicit Parseval derivation',
            'config': {'window': 'none', 'detrend': 'none', 'fft_norm': 'ortho',
                       'single_sided_multiplicity': '2 interior, 1 DC/even Nyquist',
                       'unit': 'input_unit_squared', 'directions_are_alternative_decompositions': True,
                       'phase_term_is_amplitude_weighted_not_pure_phase_score': True},
            'per_frame': rows, 'arrays': arrays}
