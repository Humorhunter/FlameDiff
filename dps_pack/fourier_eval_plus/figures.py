"""Figures for the directional Fourier evaluation.

All figures are written to files with a non-interactive backend.  The
functions take a ``SpectrumEvaluation`` plus the original arrays; they do not
load models or recompute spectra.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .directional_spectrum import AXES

TRUTH_COLOR = '#111111'
METHOD_COLORS = ('#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e')
BAND_COLOR = '0.88'


def _method_colors(labels):
    return {
        label: METHOD_COLORS[index % len(METHOD_COLORS)]
        for index, label in enumerate(labels)
    }


def _positive_floor(values):
    positive = [float(value) for value in np.ravel(values) if value > 0 and np.isfinite(value)]
    return min(positive) * 0.3 if positive else 1e-12


def _band_spans(ax, edges, names, fmin):
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        left = max(float(lower), fmin)
        right = float(upper)
        if right <= left:
            continue
        ax.axvspan(left, right, color=BAND_COLOR, alpha=0.45, lw=0)
        middle = np.sqrt(left * right) if left > 0 else right * 0.5
        ax.text(
            middle, 0.965, str(names[index]), transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=7, color='0.45',
        )
    for edge in edges[1:-1]:
        ax.axvline(edge, color='0.55', lw=0.7, ls=':')


def _save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_directional_spectra(evaluation, out_dir, unit='K', dpi=150):
    """Median directional spectra with 25-75% bands across frames."""
    out_dir = Path(out_dir)
    colors = _method_colors(evaluation.method_labels)
    paths = []
    for axis_name in AXES:
        frequency = evaluation.arrays[f'freq_{axis_name}']
        positive = frequency > 0
        fmin = float(frequency[positive][0])
        fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
        _band_spans(ax, evaluation.config.band_edges, evaluation.config.band_names, fmin)

        spectra = [('truth', evaluation.arrays[f'psd_truth_{axis_name}'], TRUTH_COLOR)]
        for label in evaluation.method_labels:
            spectra.append((label, evaluation.arrays[f'psd_pred_{label}_{axis_name}'], colors[label]))

        all_values = np.concatenate([values[:, positive] for _, values, _ in spectra], axis=0)
        floor = _positive_floor(all_values)
        for label, values, color in spectra:
            values = values[:, positive]
            median = np.median(values, axis=0)
            q25 = np.percentile(values, 25, axis=0)
            q75 = np.percentile(values, 75, axis=0)
            median = np.maximum(median, floor)
            q25 = np.maximum(q25, floor)
            q75 = np.maximum(q75, floor)
            ax.fill_between(frequency[positive], q25, q75, color=color, alpha=0.12, lw=0)
            ax.plot(frequency[positive], median, color=color, lw=1.8,
                    label='truth' if label == 'truth' else label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(fmin, 0.5)
        ax.set_xlabel('frequency (cycles/pixel)')
        ax.set_ylabel(f'one-sided power ({unit}² per bin)')
        direction = 'axis0: variation along rows (vertical)' if axis_name == 'axis0' else 'axis1: variation along columns (horizontal)'
        ax.set_title(f'Median directional spectrum, {direction}')
        ax.grid(True, which='both', alpha=0.2, lw=0.5)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
        ax.text(0.99, 0.02, f'N = {len(evaluation.frame_ids)} frames; shaded: IQR',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='0.4')
        path = out_dir / f'spectrum_{axis_name}.png'
        _save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_log_ratios(evaluation, out_dir, dpi=150, ylim=2.0):
    """Median signed log10 power ratio, with clamped display range."""
    out_dir = Path(out_dir)
    colors = _method_colors(evaluation.method_labels)
    paths = []
    for axis_name in AXES:
        frequency = evaluation.arrays[f'freq_{axis_name}']
        positive = frequency > 0
        fmin = float(frequency[positive][0])
        fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
        _band_spans(ax, evaluation.config.band_edges, evaluation.config.band_names, fmin)
        clipped = False
        for label in evaluation.method_labels:
            ratios = evaluation.arrays[f'log_ratio_{label}_{axis_name}'][:, positive]
            median = np.median(ratios, axis=0)
            q25 = np.percentile(ratios, 25, axis=0)
            q75 = np.percentile(ratios, 75, axis=0)
            clipped = clipped or bool(np.any(median < -ylim) or np.any(median > ylim))
            ax.fill_between(frequency[positive], np.clip(q25, -ylim, ylim),
                            np.clip(q75, -ylim, ylim), color=colors[label],
                            alpha=0.12, lw=0)
            ax.plot(frequency[positive], np.clip(median, -ylim, ylim),
                    color=colors[label], lw=1.7, label=label)
        ax.axhline(0.0, color='0.25', lw=0.9)
        ax.set_xscale('log')
        ax.set_xlim(fmin, 0.5)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel('frequency (cycles/pixel)')
        ax.set_ylabel('log10(pred power / truth power)')
        direction = 'axis0: variation along rows (vertical)' if axis_name == 'axis0' else 'axis1: variation along columns (horizontal)'
        ax.set_title(f'Signed spectral ratio, {direction}')
        ax.grid(True, which='both', alpha=0.2, lw=0.5)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
        note = 'positive: excess power; negative: missing power'
        if clipped:
            note += f'; display clipped to ±{ylim:g} decades'
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha='right',
                va='bottom', fontsize=7, color='0.4')
        path = out_dir / f'log_ratio_{axis_name}.png'
        _save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_band_metrics(evaluation, out_dir, dpi=150):
    """Band contribution and within-band deviation, median and IQR across frames."""
    out_dir = Path(out_dir)
    labels = evaluation.method_labels
    colors = _method_colors(labels)
    bands = list(evaluation.config.band_names)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    metrics = (('C', 'fraction of total truth power'), ('D_band', 'within-band relative deviation'))
    for row, (metric, ylabel) in enumerate(metrics):
        for column, axis_name in enumerate(AXES):
            ax = axes[row, column]
            positions = np.arange(len(bands), dtype=np.float64)
            width = 0.8 / max(1, len(labels))
            for method_index, label in enumerate(labels):
                medians = []
                lower_errors = []
                upper_errors = []
                unreliable = []
                for band in bands:
                    stats = evaluation.summary['methods'][label][axis_name]['bands'][band]
                    median = stats[metric]['median']
                    q25 = stats[metric]['q25']
                    q75 = stats[metric]['q75']
                    median = 0.0 if median is None else median
                    q25 = median if q25 is None else min(q25, median)
                    q75 = median if q75 is None else max(q75, median)
                    medians.append(median)
                    lower_errors.append(median - q25)
                    upper_errors.append(q75 - median)
                    unreliable.append(stats['reliable_fraction'] < 0.5)
                offset = (method_index - (len(labels) - 1) / 2.0) * width
                bars = ax.bar(
                    positions + offset, medians, width * 0.92,
                    yerr=np.asarray([lower_errors, upper_errors]),
                    color=colors[label], alpha=0.88, capsize=2, label=label,
                )
                for bar, flag in zip(bars, unreliable):
                    if flag:
                        bar.set_hatch('///')
                        bar.set_edgecolor('0.25')
            ax.set_xticks(positions)
            ax.set_xticklabels(bands)
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{axis_name}: {"band contribution" if metric == "C" else "band-relative deviation"}')
            ax.grid(True, axis='y', alpha=0.2, lw=0.5)
            if row == 0 and column == 0:
                ax.legend(frameon=False, fontsize=8)
    figure.suptitle('Band metrics: median and IQR over frames; hatched = truth power too small in >50% frames',
                    fontsize=9)
    path = out_dir / 'band_metrics.png'
    _save(figure, path, dpi)
    return [path]


def plot_mae_vs_spectral(evaluation, out_dir, dpi=150):
    """Per-frame spatial MAE against overall spectral deviation."""
    out_dir = Path(out_dir)
    colors = _method_colors(evaluation.method_labels)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for label in evaluation.method_labels:
        x_values = []
        y_values = []
        frame_values = []
        for frame in evaluation.per_frame:
            method = frame['methods'][label]
            mae = method['spatial']['MAE']
            deviation = method['overall']['D_overall']
            if deviation is not None:
                x_values.append(mae)
                y_values.append(deviation)
                frame_values.append(frame['frame'])
        ax.scatter(x_values, y_values, s=24, color=colors[label], alpha=0.8, label=label)
        if label == evaluation.method_labels[0] and frame_values:
            worst = np.argsort(y_values)[-3:]
            for index in worst:
                ax.annotate(str(frame_values[index]), (x_values[index], y_values[index]),
                            textcoords='offset points', xytext=(3, 3), fontsize=7, color='0.25')
    ax.set_xlabel('spatial MAE (K)')
    ax.set_ylabel('overall spectral deviation D')
    ax.set_title('Spatial error and spectral deviation are not interchangeable')
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    path = out_dir / 'mae_vs_spectral_deviation.png'
    _save(fig, path, dpi)
    return [path]


def select_frames(evaluation):
    """Return representative frame indices: median and worst primary-method MAE."""
    primary = evaluation.method_labels[0]
    values = np.asarray([
        frame['methods'][primary]['spatial']['MAE'] for frame in evaluation.per_frame
    ])
    order = np.argsort(values)
    median_index = int(order[len(order) // 2])
    worst_index = int(order[-1])
    return {
        'median': median_index,
        'worst': worst_index,
    }


def plot_spatial_frames(evaluation, truth, predictions, out_dir, unit='K',
                        display_range=None, dpi=150,
                        error_quantiles=((95.0, 'robust'), (100.0, 'full'))):
    """Spatial truth/prediction/error panels for representative frames.

    Each frame is written twice: a shared P95 error scale for visibility and
    a shared full-range scale for the outlier view.  The metrics and saved
    arrays always use the full, unclipped errors.
    """
    out_dir = Path(out_dir)
    labels = evaluation.method_labels
    selections = select_frames(evaluation)
    truth = np.asarray(truth)
    predictions = {label: np.asarray(predictions[label]) for label in labels}
    if display_range is None:
        display_range = (float(truth.min()), float(truth.max()))
    vmin, vmax = float(display_range[0]), float(display_range[1])
    colors = _method_colors(labels)
    paths = []

    for selection, index in selections.items():
        frame_id = int(evaluation.frame_ids[index])
        errors = {
            label: predictions[label][index] - truth[index] for label in labels
        }
        absolute_errors = {label: np.abs(value) for label, value in errors.items()}
        maximum = max(float(np.abs(value).max()) for value in errors.values())
        for quantile, suffix in error_quantiles:
            error_vmax = max(
                float(np.percentile(value, quantile))
                for value in absolute_errors.values()
            )
            error_vmax = max(error_vmax, 1e-6)
            figure, axes = plt.subplots(
                len(labels), 4, figsize=(13.2, 3.2 * len(labels) + 0.8),
                constrained_layout=True, squeeze=False,
            )
            for row, label in enumerate(labels):
                prediction = predictions[label][index]
                method = evaluation.per_frame[index]['methods'][label]
                panels = (
                    (truth[index], 'inferno', vmin, vmax, f'truth, frame {frame_id}'),
                    (prediction, 'inferno', vmin, vmax, f'{label} prediction'),
                    (errors[label], 'RdBu_r', -error_vmax, error_vmax, f'{label} − truth'),
                    (absolute_errors[label], 'magma', 0.0, error_vmax, f'|{label} − truth|'),
                )
                for column, (data, cmap, low, high, title) in enumerate(panels):
                    ax = axes[row, column]
                    image = ax.imshow(data, cmap=cmap, vmin=low, vmax=high,
                                      origin='upper', aspect='auto', interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(title, fontsize=9 if row == 0 else 8)
                    if row == 0 and column > 0:
                        colorbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
                        colorbar.ax.tick_params(labelsize=7)
                axes[row, 0].set_ylabel(label, fontsize=9, color=colors[label])
                text = (
                    f'MAE {method["spatial"]["MAE"]:.3g} {unit}   '
                    f'RMSE {method["spatial"]["RMSE"]:.3g} {unit}   '
                    f'bias {method["spatial"]["bias"]:.3g} {unit}\n'
                    f'D_overall {method["overall"]["D_overall"]:.3g}   '
                    f'D_log {method["overall"]["D_log_overall"]:.3g}'
                )
                axes[row, 1].text(
                    0.01, 0.02, text, transform=axes[row, 1].transAxes,
                    ha='left', va='bottom', fontsize=7, color='white',
                    bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 2, 'edgecolor': 'none'},
                )

            if quantile >= 100.0:
                scale_note = f'shared full error scale ±{error_vmax:.3g} {unit}'
            else:
                scale_note = (
                    f'shared error scale clipped at P{quantile:g} = {error_vmax:.3g} {unit} '
                    f'(full max = {maximum:.3g} {unit})'
                )
            figure.suptitle(
                f'Frame {frame_id} ({selection}); display {vmin:.1f}–{vmax:.1f} {unit}; '
                f'{scale_note}\n'
                'Numeric arrays and metrics are not clipped; only these panels saturate colour.',
                fontsize=9,
            )
            path = out_dir / f'spatial_frame_{frame_id:03d}_{selection}_{suffix}.png'
            _save(figure, path, dpi)
            paths.append(path)
    return paths


def make_all_figures(evaluation, truth, predictions, out_dir, unit='K',
                     display_range=None, dpi=150):
    out_dir = Path(out_dir)
    paths = []
    paths.extend(plot_directional_spectra(evaluation, out_dir, unit=unit, dpi=dpi))
    paths.extend(plot_log_ratios(evaluation, out_dir, dpi=dpi))
    paths.extend(plot_band_metrics(evaluation, out_dir, dpi=dpi))
    paths.extend(plot_mae_vs_spectral(evaluation, out_dir, dpi=dpi))
    paths.extend(plot_spatial_frames(
        evaluation, truth, predictions, out_dir, unit=unit,
        display_range=display_range, dpi=dpi,
    ))
    return paths
