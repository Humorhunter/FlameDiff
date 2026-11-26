"""Explicit scalar-field pairing, without model, torch, or database dependencies.

References select source rows independently from evaluation sample IDs. Resolved
descriptors contain absolute paths and input hashes for deterministic replay.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
from .field_loaders import FieldSpec, parse_spec, read_array
from .io import load, source_record, check_source

SCHEMA = 'fourier-dataset-v1'
LAYOUTS = ('HW', 'NHW', 'CHW', 'NCHW', 'HWC', 'NHWC')


@dataclass
class LoadedDataset:
    truth: np.ndarray
    predictions: dict
    frame_ids: np.ndarray
    groups: list | None
    unit: str
    stride: int | list | None
    resolved: dict


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer.')
    return int(value)


def resolve_dataset(description, *, base_dir=None):
    """Lock references/hashes. Relative paths are relative to the descriptor file."""
    origin = None
    if isinstance(description, (str, Path)):
        origin = Path(description).resolve()
        base_dir = origin.parent
        description = load(origin)
    base = Path(base_dir or '.').resolve()
    result = copy.deepcopy(description)
    if result.get('schema') != SCHEMA or not result.get('samples'):
        raise ValueError('Expected nonempty fourier-dataset-v1 samples.')
    dependencies = {r['spec']:r for r in result.get('dependencies', [])}
    for record in dependencies.values():
        if not check_source(record):
            raise ValueError(f'Dataset dependency changed or missing: {record["spec"]}')
    if origin is not None:
        record = source_record(origin)
        dependencies[record['spec']] = record
    for row in result['samples']:
        refs = [row['truth'], *row['predictions'].values()]
        if row.get('truth_reference') is not None:
            refs.append(row['truth_reference'])
        for ref in refs:
            spec = parse_spec(ref['spec'])
            path = spec.path if spec.path.is_absolute() else base/spec.path
            ref['spec'] = str(FieldSpec(path.resolve(), spec.key))
            if ref['spec'] not in dependencies:
                dependencies[ref['spec']] = source_record(ref['spec'])
            if ref.get('sha256') and dependencies[ref['spec']]['sha256'] != ref['sha256']:
                raise ValueError(f'Expected source SHA256 differs: {ref["spec"]}')
    result['dependencies'] = list(dependencies.values())
    return result


def load_dataset(description, *, base_dir=None):
    """Load a descriptor into the existing evaluate(truth, predictions) boundary."""
    resolved = resolve_dataset(description, base_dir=base_dir)
    cache = {}

    def field(ref):
        spec = ref['spec']
        if spec not in cache:
            cache[spec] = read_array(spec)
        array = cache[spec]
        layout = ref.get('layout', 'HW')
        if layout not in LAYOUTS or array.ndim != len(layout):
            raise ValueError(f'Explicit layout {layout!r} does not match {spec}: {array.shape}')
        selection = []
        for axis, size in zip(layout, array.shape):
            if axis in 'NC':
                key = 'index' if axis == 'N' else 'channel'
                index = _integer(ref.get(key), key)
                if not 0 <= index < size:
                    raise ValueError(f'{key}={index} outside source dimension {size}: {spec}')
                selection.append(index)
            else:
                selection.append(slice(None))
        if 'N' not in layout and 'index' in ref:
            raise ValueError('index supplied for a layout without N.')
        if 'C' not in layout and 'channel' in ref:
            raise ValueError('channel supplied for a layout without C.')
        values = array[tuple(selection)]
        if np.iscomplexobj(values) or min(values.shape) < 2:
            raise ValueError('Selected fields must be real HW arrays with H,W >= 2.')
        scale = float(ref.get('scale', 1)); offset = float(ref.get('offset', 0))
        if not np.isfinite([scale, offset]).all() or scale == 0:
            raise ValueError('Finite, nonzero scale and finite offset required.')
        if scale != 1 or offset != 0:
            values = values.astype(np.float64)*scale+offset
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite converted field.')
        return values

    truth=[];predictions={};ids=[];groups=[];labels=None
    for row in resolved['samples']:
        sample_id = _integer(row['id'], 'sample id')
        if not np.iinfo(np.int64).min <= sample_id <= np.iinfo(np.int64).max:
            raise ValueError('Sample id exceeds int64.')
        ids.append(sample_id);groups.append(row.get('group'))
        names = set(row['predictions'])
        if not names or (labels is not None and names != labels):
            raise ValueError('Every sample must have the same nonempty prediction labels.')
        labels = names
        target = field(row['truth'])
        if row.get('truth_reference') is not None:
            reference = field(row['truth_reference'])
            if target.shape != reference.shape or not np.array_equal(target, reference):
                raise ValueError(f'Embedded/database truth mismatch for sample {sample_id}.')
        truth.append(target)
        for name, ref in row['predictions'].items():
            value=field(ref)
            if value.shape != target.shape:
                raise ValueError(f'Prediction/truth grid mismatch for sample {sample_id}: {value.shape} vs {target.shape}')
            predictions.setdefault(name, []).append(value)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs; use distinct IDs for multiple sources or realizations.')
    if any(g is not None for g in groups) and not all(isinstance(g,str) and g for g in groups):
        raise ValueError('Supply a nonempty group string for every sample, or omit all groups.')
    if len({x.shape for x in truth}) != 1:
        raise ValueError('Mixed spatial grids must be evaluated in separate runs.')
    unit = resolved.get('unit')
    if not isinstance(unit,str) or not unit:
        raise ValueError('An explicit dataset unit is required.')
    return LoadedDataset(np.stack(truth), {k:np.stack(v) for k,v in predictions.items()},
        np.asarray(ids,dtype=np.int64), groups if groups[0] is not None else None,
        unit, resolved.get('stride'), resolved)
