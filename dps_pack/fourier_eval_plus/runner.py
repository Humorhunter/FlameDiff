"""CLI orchestration, with independent metric artifacts for each enhancement."""
import argparse
import csv
import sys
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from fourier_eval_plus import SpectrumConfig
from fourier_eval_plus.field_loaders import read_field, read_vector
from .engine import evaluate, available_enhancements
from .io import dump, payload, source_record, sha, utc, load_reader_plugins


def _ids(value, n):
    ids = np.arange(n) if value is None else np.asarray(value)
    if (ids.ndim != 1 or len(ids) != n or np.iscomplexobj(ids)
            or not np.issubdtype(ids.dtype, np.number) or not np.isfinite(ids).all()
            or not np.equal(ids, np.floor(ids)).all()
            or len(set(ids.tolist())) != n):
        raise ValueError('Input frame IDs must be unique finite integers, matching N.')
    if any(int(v) < np.iinfo(np.int64).min or int(v) > np.iinfo(np.int64).max for v in ids):
        raise ValueError('Frame IDs exceed int64.')
    return ids.astype(np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--truth'); source.add_argument('--npz')
    source.add_argument('--dataset', help='Explicit fourier-dataset-v1 descriptor JSON.')
    source.add_argument('--dps-run', help='Read numeric DPS run output directly.')
    parser.add_argument('--dps-truth', choices=['embedded','database','crosscheck'], default='embedded')
    parser.add_argument('--database', help='Relocated database array; original provenance hash must match.')
    parser.add_argument('--dps-member', help='Select a pair, aggregate, or one pilot candidate inside the run.')
    parser.add_argument('--reader-plugin', action='append', default=[],
                        help='Registration hook module:function, repeatable; also required for verification.')
    parser.add_argument('--truth-key', default='truth_K')
    parser.add_argument('--method', action='append', default=[])
    parser.add_argument('--frame-ids'); parser.add_argument('--frames')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--groups-csv'); parser.add_argument('--group-column', default='role')
    parser.add_argument('--frame-column', default='frame')
    parser.add_argument('--enhancements', default='paper27,parseval')
    parser.add_argument('--stride', help='Sampling stride: integer or axis0,axis1; dataset metadata used by default.')
    parser.add_argument('--window', choices=['hann','hamming','none'], default='hann')
    parser.add_argument('--band-edges', default='0,0.0625,0.125,0.5')
    parser.add_argument('--band-names', default='low,mid,high')
    parser.add_argument('--unit'); parser.add_argument('--out', required=True)
    parser.add_argument('--no-figures', action='store_true')
    parser.add_argument('--display-range', nargs=2, type=float)
    args = parser.parse_args(argv)
    if not (args.dataset or args.dps_run) and not args.method:
        parser.error('truth/npz input requires at least one --method.')
    if (args.dataset or args.dps_run) and (args.method or args.frame_ids):
        parser.error('dataset/dps-run supplies predictions and frame IDs; do not also pass method/frame-ids.')
    if not args.dps_run and (args.database or args.dps_member or args.dps_truth != 'embedded'):
        parser.error('database/dps-member/dps-truth require dps-run.')
    out = Path(args.out).resolve()
    if out.exists():
        parser.error('Output directory must be new; existing runs are immutable.')
    out.mkdir(parents=True)
    tick = time.perf_counter(); started = utc()
    try:
        reader_plugins = load_reader_plugins(args.reader_plugin)
        dataset = None
        methods = {}
        for item in args.method:
            label, spec = item.split('=',1)
            if label in methods:
                raise ValueError('Duplicate prediction label.')
            methods[label] = f'{args.npz}#{spec}' if args.npz else spec
        truth_spec = f'{args.npz}#{args.truth_key}' if args.npz else args.truth
        ids_spec = args.frame_ids
        if args.npz and not ids_spec:
            with np.load(args.npz, allow_pickle=False) as z:
                if 'frame_indices' in z.files:
                    ids_spec = f'{args.npz}#frame_indices'
        if args.dataset or args.dps_run:
            from .datasets import load_dataset
            from .dps_inputs import load_dps_run
            dataset = (load_dataset(args.dataset) if args.dataset else load_dps_run(
                args.dps_run, truth_mode=args.dps_truth, database=args.database, member=args.dps_member))
            truth, predictions = dataset.truth, dataset.predictions
            ids = dataset.frame_ids
            if args.unit and args.unit != dataset.unit:
                raise ValueError('unit conflicts with dataset metadata; convert values explicitly in the descriptor.')
            args.unit = dataset.unit
        else:
            truth = read_field(truth_spec, 'truth')
            truth = truth[None] if truth.ndim == 2 else truth
            predictions = {k: read_field(v, k) for k,v in methods.items()}
            predictions = {k: v[None] if v.ndim == 2 else v for k,v in predictions.items()}
            ids = _ids(read_vector(ids_spec) if ids_spec else None, len(truth))
            args.unit = args.unit or 'value'
        if any(v.shape != truth.shape for v in predictions.values()):
            raise ValueError('All input batches must have identical shape and aligned frames.')
        ids = _ids(ids, len(truth))
        indices = np.arange(len(truth))
        if args.frames:
            wanted = [int(x) for x in args.frames.split(',')]
            if len(set(wanted)) != len(wanted) or not set(wanted) <= set(ids.tolist()):
                raise ValueError('Requested frames must be unique and present.')
            indices = indices[np.isin(ids, wanted)]
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ValueError('max-frames must be positive.')
            indices = indices[:args.max_frames]
        groups = [dataset.groups[i] for i in indices] if dataset is not None and dataset.groups is not None else None
        if args.groups_csv:
            mapping = {}
            with Path(args.groups_csv).open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    key = int(row[args.frame_column])
                    if key in mapping:
                        raise ValueError('Duplicate group frame.')
                    mapping[key] = row[args.group_column]
            groups = [mapping[int(ids[i])] for i in indices]
        cfg = SpectrumConfig(window=args.window,
                             band_edges=tuple(float(x) for x in args.band_edges.split(',')),
                             band_names=tuple(args.band_names.split(',')))
        selected = () if args.enhancements == 'none' else tuple(args.enhancements.split(','))
        if args.stride is not None:
            strides = tuple(int(v) for v in args.stride.split(','))
            if len(strides) not in (1, 2):
                raise ValueError('stride must be an integer or axis0,axis1.')
            stride = strides[0] if len(strides) == 1 else strides
        else:
            stride = dataset.stride if dataset is not None else 8
            if 'paper27' in selected and stride is None:
                raise ValueError('Dataset has no regular sampling stride; pass --stride explicitly or disable paper27. Do not infer stride from sparse point count.')
        options = {'paper27': {'stride': stride}} if 'paper27' in selected else {}
        manifest = {'reader_plugins': reader_plugins, 'truth': source_record(truth_spec) if dataset is None else None,
                    'predictions': {k: source_record(v) for k,v in methods.items()},
                    'frame_ids': source_record(ids_spec) if ids_spec else None,
                    'groups_csv': source_record(args.groups_csv) if args.groups_csv else None,
                    'selected_indices': indices.tolist(), 'selected_frame_ids': ids[indices].tolist(),
                    'groups': groups, 'unit': args.unit}
        if dataset is not None:
            manifest['dataset'] = dataset.resolved
        dump(out/'inputs_manifest.json', manifest)
        compute_tick = time.perf_counter()
        e = evaluate(truth[indices], {k:v[indices] for k,v in predictions.items()},
                     enhancements=selected, spectrum_config=cfg, options=options,
                     frame_ids=ids[indices], groups=groups)
        compute_seconds = time.perf_counter()-compute_tick
        dump(out/'config_resolved.json', e.config)
        dump(out/'metrics.json', payload(e,args.unit))
        np.savez_compressed(out/'baseline_spectra.npz', **e.baseline.arrays)
        for name, result in e.enhancements.items():
            folder = out/name; folder.mkdir()
            dump(folder/'metrics.json', {k:v for k,v in result.items() if k != 'arrays'})
            np.savez_compressed(folder/'spectra.npz', **result['arrays'])
        figures = []
        if not args.no_figures:
            from fourier_eval_plus.figures import make_all_figures
            figures = make_all_figures(
                e.baseline, truth[indices], {k:v[indices] for k,v in predictions.items()},
                out/'figures', unit=args.unit, display_range=args.display_range)
        from .report import write_report
        write_report(e, out/'report.md', args.unit, compute_seconds)
        # Snapshot the exact active implementation and baseline, excluding run data.
        code = {}
        repository = Path(__file__).resolve().parents[1]
        for package in ('fourier_eval_plus',):
            for p in sorted((repository/package).rglob('*.py')):
                if any(part.endswith('.egg-info') for part in p.parts):
                    continue
                if any(x in p.relative_to(repository/package).parts for x in ('runs','logs','compare','__pycache__','dist','build')):
                    continue
                relative = p.relative_to(repository)
                target = out/'source_snapshot'/relative
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,target)
                code[str(p)] = sha(p)
        dump(out/'source_manifest.json', code)
        dump(out/'execution.json', {'status':'completed','returncode':0,'started_utc':started,
             'finished_utc':utc(),'compute_seconds':compute_seconds,
             'elapsed_seconds':time.perf_counter()-tick,'command':sys.argv,
             'numpy':np.__version__,'python':sys.version,'figures':[str(p) for p in figures]})
        from .verify import verify_run
        checks = verify_run(out, include_details=True)
        details = checks.pop('checks')
        dump(out/'checks_details.json', {'schema':'fourier-check-details-v1',
             **checks, 'checks':details})
        checks['details_file'] = 'checks_details.json'
        checks['details_sha256'] = sha(out/'checks_details.json')
        dump(out/'checks.json', checks)
        if not checks['passed']:
            raise ArithmeticError('Post-run verification failed.')
        print(f'Wrote {out}; compute={compute_seconds:.3f}s; verification passed.')
        return 0
    except Exception:
        dump(out/'execution.json', {'status':'failed','returncode':1,'started_utc':started,
             'elapsed_seconds':time.perf_counter()-tick,'traceback':traceback.format_exc()})
        traceback.print_exc(); return 1
