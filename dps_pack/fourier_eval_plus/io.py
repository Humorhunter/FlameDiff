"""Input identity and JSON serialization. Directory hashes read actual members."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from fourier_eval_plus.field_loaders import FieldSpec, parse_spec, read_field, read_vector, directory_files


def load_reader_plugins(specs):
    """Call explicitly requested module:function registration hooks."""
    import importlib
    records = []
    for spec in specs:
        module_name, function_name = spec.rsplit(':', 1)
        module = importlib.import_module(module_name)
        getattr(module, function_name)()
        path = Path(module.__file__).resolve()
        records.append({'spec': spec, 'path': str(path), 'sha256': sha(path)})
    return records


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def source_record(spec):
    spec = spec if isinstance(spec, FieldSpec) else parse_spec(spec)
    path = spec.path.resolve()
    result = {'spec': str(FieldSpec(path, spec.key)), 'path': str(path), 'key': spec.key}
    if path.is_dir():
        files = [{'name': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in directory_files(path)]
        digest = hashlib.sha256()
        for item in files:
            digest.update(f"{item['name']}:{item['sha256']}\n".encode())
        result.update(format='directory', files=files, sha256=digest.hexdigest())
    else:
        result.update(format=path.suffix.lower(), sha256=sha(path), bytes=path.stat().st_size)
    return result


def check_source(record):
    """Re-enumerate/re-hash inputs, detecting additions, removals and modifications."""
    try:
        return source_record(record['spec']) == record
    except (OSError, ValueError):
        return False


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def payload(evaluation, unit):
    base = evaluation.baseline
    return {'schema': 'fourier-eval-plus-v2', 'unit': unit, 'config': evaluation.config,
            'baseline': {'per_frame': base.per_frame, 'summary': base.summary,
                         'summary_by_group': base.summary_by_group},
            'enhancements': {name: {k: v for k, v in result.items() if k != 'arrays'}
                             for name, result in evaluation.enhancements.items()}}


def read_inputs(manifest):
    if manifest.get('dataset') is not None:
        from .datasets import load_dataset
        data = load_dataset(manifest['dataset'])
        indices = np.asarray(manifest['selected_indices'], dtype=int)
        return data.truth[indices], {k:v[indices] for k,v in data.predictions.items()}
    truth = read_field(manifest['truth']['spec'], 'truth')
    pred = {k: read_field(v['spec'], k) for k, v in manifest['predictions'].items()}
    truth = truth[None] if truth.ndim == 2 else truth
    pred = {k: v[None] if v.ndim == 2 else v for k, v in pred.items()}
    indices = np.asarray(manifest['selected_indices'], dtype=int)
    return truth[indices], {k: v[indices] for k, v in pred.items()}
