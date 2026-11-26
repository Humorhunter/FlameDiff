"""Adapter for source_azula DPS numeric exports, without importing its models."""
from pathlib import Path
import numpy as np
from .io import load, source_record
from .datasets import SCHEMA, resolve_dataset, load_dataset


def describe_dps_run(run, *, truth_mode='embedded', database=None, member=None):
    """Build references only; no copying arrays, generation, decoding or training.

    member explicitly selects an aggregate, single pair, or pilot candidate
    directory. Never merge multiple pilot zeta candidates implicitly.
    """
    run=Path(run).resolve()
    if not run.is_dir():
        raise ValueError('dps-run must be an existing run directory.')
    if truth_mode not in ('embedded','database','crosscheck'):
        raise ValueError('truth_mode must be embedded, database, or crosscheck.')
    dependencies=[]

    def metadata(path, default):
        if not path.is_file():return default
        dependencies.append(source_record(path))
        return load(path)

    identity=metadata(run/'inputs_manifest.json', {})
    config=metadata(run/'config_resolved.json', {})
    if config.get('unit', 'K') != 'K':
        raise ValueError('This DPS adapter expects decoded K-unit exports; use an explicit dataset descriptor for other units.')
    preview=metadata(run/'DPS_preview.json', {})
    source=identity.get('source')
    if truth_mode != 'embedded' and source is None and config.get('cache'):
        cache=Path(config['cache'])
        if not cache.is_absolute():cache=run/cache
        cache_meta=metadata(cache/'manifest.json', {})
        sources=cache_meta.get('identity',{}).get('sources',[])
        if len(sources)==1:source=sources[0]
    if truth_mode != 'embedded':
        if not source or not source.get('sha256'):
            raise ValueError('Database mode requires an unambiguous source path and SHA256 in run provenance; otherwise write an explicit dataset descriptor.')
        db=Path(database or source['path'])
        if not db.is_absolute():db=run/db
        db=db.resolve()
        db_ref={'spec':str(db),'layout':'NHW','sha256':source['sha256']}
    elif database is not None:
        raise ValueError('database override requires database or crosscheck truth mode.')

    path=Path(member) if member else run
    if not path.is_absolute():path=run/path
    path=path.resolve()
    if not path.is_relative_to(run):
        raise ValueError('member must remain inside dps-run.')
    aggregate=path/'temperature_pairs_50.npz' if path.is_dir() else path
    samples=[]
    roles={int(x['frame']):x.get('role') for x in config.get('frames',[])}

    def ref(file,key,index=None):
        result={'spec':f'{file}#{key}','layout':'HW' if index is None else 'NHW'}
        if index is not None:result['index']=index
        return result

    def sample(file, frame, index, keys, group=None):
        frame=int(frame)
        predictions={'dps':ref(file,'prediction_K',index)}
        if 'interpolation_K' in keys:predictions['interp']=ref(file,'interpolation_K',index)
        if truth_mode != 'database' and 'truth_K' not in keys:
            raise ValueError('No embedded truth_K; use database mode with recorded provenance.')
        embedded=ref(file,'truth_K',index)
        row={'id':frame,'truth':embedded,'predictions':predictions,
             'metadata':{'source_frame':frame,'export_file':str(file)}}
        if group is not None:row['group']=group
        if truth_mode != 'embedded':
            raw={**db_ref,'index':frame}
            if truth_mode=='database':row['truth']=raw
            else:row['truth_reference']=raw
        samples.append(row)

    if aggregate.is_file() and aggregate.name != 'data.npz' and aggregate.name != 'validation_pair.npz':
        with np.load(aggregate,allow_pickle=False) as z:
            if 'prediction_K' not in z or 'frame_indices' not in z:
                raise ValueError('Aggregate requires prediction_K and explicit frame_indices.')
            frames=np.asarray(z['frame_indices'])
            if (frames.ndim!=1 or not np.issubdtype(frames.dtype,np.integer)
                    or z['prediction_K'].ndim!=3 or len(frames)!=len(z['prediction_K'])):
                raise ValueError('Aggregate requires integer frame_indices aligned with NHW predictions.')
            for i,f in enumerate(frames):sample(aggregate,int(f),i,z.files,roles.get(int(f)))
    else:
        if path.is_file():files=[path]
        elif (path/'pairs').is_dir():files=sorted((path/'pairs').glob('*/data.npz'))
        elif (path/'validation_pair.npz').is_file():files=[path/'validation_pair.npz']
        elif (path/'data.npz').is_file():files=[path/'data.npz']
        else:files=sorted(path.glob('*/data.npz'))
        if not files:
            raise ValueError('No supported numeric DPS output. For pilot runs select one zeta directory with member; for new formats use a dataset descriptor.')
        for file in files:
            meta=metadata(file.parent/'metadata.json',{})
            with np.load(file,allow_pickle=False) as z:
                if 'prediction_K' not in z:raise ValueError(f'No prediction_K: {file}')
                raw_frame=z['frame_index'].item() if 'frame_index' in z else meta.get('frame',preview.get('frame'))
                if isinstance(raw_frame,bool) or not isinstance(raw_frame,(int,np.integer)):
                    raise ValueError(f'No explicit integer frame identity for {file}.')
                frame=int(raw_frame)
                if meta.get('frame', frame) != frame:
                    raise ValueError(f'Pair metadata/array frame identity differs: {file}')
                sample(file,frame,None,z.files,meta.get('role',roles.get(frame,preview.get('split'))))
    observation=config.get('observation',preview.get('observation',{}))
    stride=observation.get('stride') if observation.get('type')=='grid_subsample' else None
    descriptor={'schema':SCHEMA,'unit':'K','stride':stride,'samples':samples,
                'dependencies':dependencies,'provenance':{'adapter':'source-azula-dps-v1',
                'run':str(run),'truth_mode':truth_mode,'selection':str(path),
                'prediction_semantics':'decoded final sampler state, already in K; no clipping or normalization applied',
                'observation':observation}}
    return resolve_dataset(descriptor)


def load_dps_run(run, **options):
    return load_dataset(describe_dps_run(run, **options))


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--truth-mode',choices=['embedded','database','crosscheck'],default='embedded')
    parser.add_argument('--database');parser.add_argument('--member')
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    target=Path(args.out).resolve()
    if target.exists():parser.error('out must be a new file.')
    descriptor=describe_dps_run(args.run,truth_mode=args.truth_mode,database=args.database,member=args.member)
    loaded=load_dataset(descriptor)  # Check truth correspondence and shapes before saving.
    import json
    with target.open('x',encoding='utf-8') as f:
        json.dump(loaded.resolved,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(f'Wrote {target}; {len(loaded.frame_ids)} samples, shape={loaded.truth.shape}, unit={loaded.unit}.')
    return 0
