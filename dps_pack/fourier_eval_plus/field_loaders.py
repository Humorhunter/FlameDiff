"""Format-independent array loading for the spectral evaluator.

The evaluator core never reads files.  This adapter turns a short ``spec``
string into a NumPy array and is intentionally independent of the project's
model, training, observation, and database code.

Supported specs
---------------
``path.npy``
    One 2-D field or a 3-D ``(N,H,W)`` batch.
``path.npz#key``
    One array from an NPZ archive.  ``#key`` may be omitted only when the
    archive contains exactly one array.
``path.csv`` / ``path.tsv`` / ``path.txt``
    A numeric, headerless 2-D field.  CSV uses commas, TSV uses tabs, and TXT
    uses whitespace.
``directory/``
    One 2-D ``.npy``/``.csv``/``.tsv``/``.txt`` file per frame, sorted by file
    name and stacked on a new sample axis.  Frame IDs should be supplied
    explicitly when file-name order is not the desired label order.

Additional suffixes can be registered with ``register_reader`` without
changing the evaluator or the CLI.  Colormapped PNG/JPG files are deliberately
not accepted: they do not contain a calibrated scalar field.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    path: Path
    key: str | None = None

    def __str__(self):
        return f'{self.path}#{self.key}' if self.key else str(self.path)


def parse_spec(text):
    """Parse ``path`` or ``path#key`` without resolving the path."""
    raw = str(text)
    if not raw:
        raise ValueError(f'Empty path in spec: {raw!r}')
    path = Path(raw).expanduser()
    # A real path can contain '#', including in a parent directory.
    if '#' not in raw or path.exists():
        return FieldSpec(path)
    path_text, key = raw.rsplit('#', 1)
    if not key:
        raise ValueError(f'Empty NPZ key in spec: {raw!r}')
    if not path_text:
        raise ValueError(f'Empty path in spec: {raw!r}')
    return FieldSpec(Path(path_text).expanduser(), key)


def _check_numeric(array, spec, name):
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f'{name} from {spec} is not numeric (dtype={array.dtype}).')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} from {spec} contains non-finite values.')
    return array


def _read_npz(spec):
    with np.load(spec.path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if spec.key is None:
            if len(keys) != 1:
                raise ValueError(
                    f'{spec.path} contains {len(keys)} arrays {keys}; '
                    'specify one as path.npz#key.'
                )
            key = keys[0]
        else:
            key = spec.key
        if key not in keys:
            raise ValueError(f'{spec.path} has no array {key!r}; keys={keys}.')
        return np.asarray(archive[key])


def _read_npy(spec):
    if spec.key is not None:
        raise ValueError(f'#key is only valid for NPZ, not {spec.path.suffix}.')
    return np.load(spec.path, allow_pickle=False)


def _read_text(spec, delimiter):
    if spec.key is not None:
        raise ValueError(f'#key is only valid for NPZ, not {spec.path.suffix}.')
    return np.loadtxt(spec.path, delimiter=delimiter, ndmin=2)


def _read_csv(spec):
    return _read_text(spec, ',')


def _read_tsv(spec):
    return _read_text(spec, '\t')


def _read_txt(spec):
    return _read_text(spec, None)


_READERS: dict[str, Callable[[FieldSpec], np.ndarray]] = {
    '.npz': _read_npz,
    '.npy': _read_npy,
    '.csv': _read_csv,
    '.tsv': _read_tsv,
    '.txt': _read_txt,
}
_FRAME_SUFFIXES = {'.npy', '.csv', '.tsv', '.txt'}


def register_reader(suffix, reader, *, directory_frames=False):
    """Register an additional single-file reader.

    ``reader`` receives a ``FieldSpec`` and returns a numeric 2-D or 3-D
    array. Set directory_frames=True to also accept 2-D files in frame folders.
    This is the extension point for HDF5, Zarr, or project-specific
    containers; the evaluator core does not need to change.
    """
    suffix = str(suffix).lower()
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    if not callable(reader):
        raise TypeError('reader must be callable.')
    _READERS[suffix] = reader
    if directory_frames:
        _FRAME_SUFFIXES.add(suffix)


def supported_suffixes():
    return tuple(sorted(_READERS))


def directory_files(path):
    """The same deterministic frame membership for loading and hashing."""
    return sorted(p for p in Path(path).iterdir()
                  if p.is_file() and p.suffix.lower() in _FRAME_SUFFIXES)


def _read_directory(spec):
    if spec.key is not None:
        raise ValueError('A directory spec cannot carry #key.')
    files = directory_files(spec.path)
    if not files:
        raise ValueError(
            f'{spec.path} has no frame files with suffixes {_FRAME_SUFFIXES}.'
        )
    frames = []
    for path in files:
        reader = _READERS.get(path.suffix.lower())
        if reader is None:
            raise ValueError(f'No reader for {path.suffix} ({path}).')
        frame = np.asarray(reader(FieldSpec(path, None)))
        if frame.ndim != 2:
            raise ValueError(f'Directory frames must be 2-D; {path} has shape {frame.shape}.')
        frames.append(frame)
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f'Directory frames have different shapes: {sorted(shapes)}.')
    return np.stack(frames)


def read_array(spec, name='array'):
    """Read numeric container values before explicit layout/channel selection."""
    spec = spec if isinstance(spec, FieldSpec) else parse_spec(spec)
    if not spec.path.exists():
        raise FileNotFoundError(f'{name} source does not exist: {spec.path}')
    if spec.path.is_dir():
        array = _read_directory(spec)
    else:
        reader = _READERS.get(spec.path.suffix.lower())
        if reader is None:
            raise ValueError(f'Unsupported {name} format {spec.path.suffix!r}; supported: {supported_suffixes()}')
        array = reader(spec)
    return _check_numeric(array, spec, name)


def read_field(spec, name='field'):
    """Read one 2-D field or 3-D ``(N,H,W)`` batch from a spec.

    The returned array is numeric and finite.  The evaluator performs its own
    shape and dtype validation; this function only handles container format.
    """
    spec = spec if isinstance(spec, FieldSpec) else parse_spec(spec)
    array = read_array(spec, name)
    if array.ndim not in (2, 3):
        raise ValueError(
            f'{name} from {spec} must be 2-D or 3-D; got shape {array.shape}.'
        )
    return array


def read_vector(spec, name='vector'):
    """Read a one-dimensional vector (for example frame IDs) from a spec."""
    spec = spec if isinstance(spec, FieldSpec) else parse_spec(spec)
    if spec.path.is_dir():
        raise ValueError(f'{name} must be a single file, not a directory.')
    if not spec.path.exists():
        raise FileNotFoundError(f'{name} source does not exist: {spec.path}')
    suffix = spec.path.suffix.lower()
    if suffix == '.npz':
        array = _read_npz(spec)
    elif suffix == '.npy':
        array = _read_npy(spec)
    elif suffix in ('.csv', '.tsv', '.txt'):
        delimiter = ',' if suffix == '.csv' else ('\t' if suffix == '.tsv' else None)
        array = np.loadtxt(spec.path, delimiter=delimiter, ndmin=1)
    elif suffix in _READERS:
        array = _READERS[suffix](spec)
    else:
        raise ValueError(
            f'Unsupported {name} format {suffix!r}; use NPZ, NPY, CSV, TSV, or TXT.'
        )
    array = _check_numeric(array, spec, name)
    if array.ndim > 2:
        raise ValueError(f'{name} from {spec} must be at most 2-D; got {array.shape}.')
    if array.ndim == 2:
        if 1 not in array.shape:
            raise ValueError(f'{name} from {spec} must be a vector; got {array.shape}.')
        array = array.reshape(-1)
    return array
