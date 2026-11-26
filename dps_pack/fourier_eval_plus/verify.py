"""Read-only verification with independent full-FFT formulas, never core imports.

Re-hash actual inputs; recompute PSD, complex errors, all new metrics and curves.
The implementations below intentionally use full FFT rather than core rFFT.
"""
from pathlib import Path
import numpy as np
from .io import load, check_source, sha, read_inputs, load_reader_plugins
from .contracts import summarize


def _full_transform(field, axis, window='none', detrend=False):
    values = np.moveaxis(field, axis, -1)
    if detrend:
        values = values-values.mean(axis=-1, keepdims=True)
    n = values.shape[-1]
    w = np.ones(n)
    if window == 'hann':
        w = .5-.5*np.cos(2*np.pi*np.arange(n)/n)
    elif window == 'hamming':
        w = .54-.46*np.cos(2*np.pi*np.arange(n)/n)
    return np.fft.fft(values*w, axis=-1), w


def _fold(values, n):
    result = values[..., :n//2+1].copy()
    if n % 2:
        result[..., 1:] += values[..., :n//2:-1]
    else:
        result[..., 1:-1] += values[..., :n//2:-1]
    return result


def _psd(field, axis, cfg):
    z,w = _full_transform(field,axis,cfg['window'],cfg['detrend']=='constant')
    n=field.shape[axis]
    return _fold(np.abs(z)**2/(n*np.dot(w,w)),n).mean(axis=0)


def _ratio(num, den):
    return float(num/den) if den > 0 else (0. if num == 0 else None)


def verify_run(run, *, include_details=False):
    run = Path(run).resolve(); checks=[]
    def check(name, passed):
        checks.append({'name':name,'passed':bool(passed)})
    def close(name, actual, expected):
        if actual is None or expected is None:
            check(name, actual is expected)
        elif isinstance(expected, dict):
            check(name+'.keys',set(actual)==set(expected))
            for k in expected:
                if k in actual:close(name+'.'+str(k),actual[k],expected[k])
        elif isinstance(expected, (str,bool)):
            check(name,actual==expected)
        elif isinstance(expected, list):
            check(name+'.length',len(actual)==len(expected))
            for i,(a,b) in enumerate(zip(actual,expected)):
                close(name+'.'+str(i),a,b)
        else:
            a=np.asarray(actual);b=np.asarray(expected)
            check(name,a.shape==b.shape and np.allclose(a,b,rtol=3e-10,atol=1e-11,equal_nan=False))

    def check_paper27(name, actual, numerator, denominator, count, reference_scale):
        # Full FFT and rFFT can have different roundoff in an empty band.
        # Check the independent squared sums, then check the ratio against the
        # recorded sums when dividing by a roundoff-level denominator.
        floor = (64 * np.finfo(float).eps) ** 2 * reference_scale
        ratio_num, ratio_den = numerator, denominator
        if denominator <= floor and count:
            ratio_num = actual['error_squared_sum']
            ratio_den = actual['truth_squared_sum']
            check(name + '.roundoff_band', 0 <= ratio_den <= 2 * floor and ratio_num >= 0)
        value = _ratio(np.sqrt(ratio_num), np.sqrt(ratio_den)) if count else None
        close(name, actual, {'NRMSE_sp': value, 'valid': count > 0 and value is not None,
                             'bins': count, 'error_squared_sum': numerator,
                             'truth_squared_sum': denominator})

    manifest=load(run/'inputs_manifest.json'); saved=load(run/'metrics.json')
    for plugin in manifest.get('reader_plugins', []):
        check('reader_plugin_hash:'+plugin['spec'], Path(plugin['path']).is_file() and sha(plugin['path'])==plugin['sha256'])
    load_reader_plugins([p['spec'] for p in manifest.get('reader_plugins', [])])
    if manifest.get('dataset') is not None:
        records=manifest['dataset']['dependencies']
    else:
        records=[manifest['truth'],*manifest['predictions'].values()]
    records += [manifest[k] for k in ('frame_ids','groups_csv') if manifest.get(k)]
    for i,record in enumerate(records):check(f'input_hash_{i}',check_source(record))
    for path,digest in load(run/'source_manifest.json').items():
        p=Path(path);check('code_hash:'+str(p),p.is_file() and sha(p)==digest)
        # Snapshot identity is also checked, including nested package paths.
        root=Path(__file__).resolve().parents[1]
        try: snap=run/'source_snapshot'/p.relative_to(root)
        except ValueError: snap=Path('missing_snapshot')
        check('snapshot_hash:'+str(p),snap.is_file() and sha(snap)==digest)
    try:
        truth,predictions=read_inputs(manifest)
    except (ValueError, OSError, KeyError, IndexError) as exc:
        if manifest.get('dataset') is None:
            raise
        check('dataset_replay', False)
        result={'passed':False,'checks_total':len(checks),
                'checks_passed':sum(x['passed'] for x in checks),
                'failures':[x for x in checks if not x['passed']],
                'scope':'input/source checks only; dataset replay failed, metrics not recomputed',
                'replay_error':str(exc)}
        if include_details:result['checks']=checks
        return result
    cfg=saved['config']['baseline']
    frame_ids=manifest['selected_frame_ids']
    check('frame_count',len(truth)==len(saved['baseline']['per_frame'])==len(frame_ids))
    with np.load(run/'baseline_spectra.npz',allow_pickle=False) as arrays:
        close('frame_ids',arrays['frame_ids'],frame_ids)
        for i,row in enumerate(saved['baseline']['per_frame']):
            close(f'frame_{i}',row['frame'],frame_ids[i])
            for label,pred in predictions.items():
                error=pred[i].astype(float)-truth[i].astype(float)
                expected_spatial={'MAE':float(np.abs(error).mean()),'RMSE':float(np.sqrt((error**2).mean())),
                    'bias':float(error.mean()),'max_abs_error':float(np.abs(error).max()),
                    'p99_abs_error':float(np.percentile(np.abs(error),99)),
                    'truth_min':float(truth[i].min()),'truth_max':float(truth[i].max()),'truth_mean':float(truth[i].mean(dtype=float)),
                    'pred_min':float(pred[i].min()),'pred_max':float(pred[i].max()),'pred_mean':float(pred[i].mean(dtype=float))}
                close(f'base.{i}.{label}.spatial',row['methods'][label]['spatial'],expected_spatial)
                for axis in (0,1):
                    name=f'axis{axis}';f=np.fft.rfftfreq(truth.shape[axis+1]);nonzero=f>0
                    s=_psd(truth[i].astype(float),axis,cfg);p=_psd(pred[i].astype(float),axis,cfg)
                    e=_psd(error,axis,cfg)
                    for key,v in [('psd_truth_'+name,s),('psd_pred_'+label+'_'+name,p),('psd_error_'+label+'_'+name,e)]:
                        close(f'curve.{i}.{key}',arrays[key][i],v)
                    target=row['methods'][label]['axes'][name]
                    close(f'D.{i}.{label}.{axis}',target['D'],_ratio(np.abs(p[nonzero]-s[nonzero]).sum(),s[nonzero].sum()))
                    for band,lo,hi in zip(cfg['band_names'],cfg['band_edges'][:-1],cfg['band_edges'][1:]):
                        mask=(f>lo)&(f<=hi);b=target['bands'][band]
                        close(f'C.{i}.{label}.{axis}.{band}',b['C'],_ratio(np.abs(p[mask]-s[mask]).sum(),s[nonzero].sum()))
                    floor=target['D_log_floor']
                    if floor is not None:
                        active=(s[1:]>floor)|(p[1:]>floor)
                        v=float(np.abs(np.log10((p[1:][active]+floor)/(s[1:][active]+floor))).mean()) if active.any() else 0.
                        close(f'Dlog.{i}.{label}.{axis}',target['D_log'],v)
    for method,result in saved['enhancements'].items():
        close(method+'.artifact',load(run/method/'metrics.json'),result)
        if method=='paper27':
            strides=result['config']['stride']
            for i,row in enumerate(result['per_frame']):
                for label,pred in predictions.items():
                    joint={}
                    joint_reference_scale = 0.
                    for axis in (0,1):
                        f=np.fft.rfftfreq(truth.shape[axis+1]);s=_psd(truth[i].astype(float),axis,cfg);p=_psd(pred[i].astype(float),axis,cfg)
                        reference_scale = float((s[f > 0] ** 2).sum())
                        joint_reference_scale += reference_scale
                        masks={'all':f>0,'resolved':(f>0)&(f<=.5/strides[axis]),'unresolved':f>.5/strides[axis]}
                        masks.update({b:(f>lo)&(f<=hi) for b,lo,hi in zip(cfg['band_names'],cfg['band_edges'][:-1],cfg['band_edges'][1:])})
                        for key,mask in masks.items():
                            num=float(((p[mask]-s[mask])**2).sum());den=float((s[mask]**2).sum())
                            n=int(mask.sum())
                            block = row['methods'][label]['axes'][f'axis{axis}'][key]
                            check_paper27(f'paper27.{i}.{label}.{axis}.{key}', block,
                                          num, den, n, reference_scale)
                            old=joint.setdefault(key,[0.,0.,0]);old[0]+=num;old[1]+=den;old[2]+=n
                    for key,(num,den,n) in joint.items():
                        block = row['methods'][label]['joint'][key]
                        check_paper27(f'paper27.joint.{i}.{label}.{key}', block,
                                      num, den, n, joint_reference_scale)
        elif method=='parseval':
            with np.load(run/method/'spectra.npz',allow_pickle=False) as arr:
                for i,row in enumerate(result['per_frame']):
                    for label,pred in predictions.items():
                        mse=float(((pred[i].astype(float)-truth[i])**2).mean())
                        close(f'parseval.spatial.{i}.{label}',row['methods'][label]['spatial_mse'],mse)
                        close(f'parseval.spatial_rmse.{i}.{label}',row['methods'][label]['spatial_rmse'],float(np.sqrt(mse)))
                        for axis in (0,1):
                            x,_=_full_transform(truth[i].astype(float),axis);y,_=_full_transform(pred[i].astype(float),axis)
                            n=truth.shape[axis+1];normalizer=n*truth[i].size
                            z=y*np.conj(x);a=np.abs(x);b=np.abs(y)
                            vals={'error':np.abs(y-x)**2,'magnitude':(b-a)**2,
                                  'phase':4*a*b*np.sin(np.angle(z)/2)**2}
                            curves={k:_fold(v,n).sum(axis=0)/normalizer for k,v in vals.items()}
                            f=np.fft.rfftfreq(n);masks={'dc':f==0}
                            masks.update({key:(f>lo)&(f<=hi) for key,lo,hi in zip(cfg['band_names'],cfg['band_edges'][:-1],cfg['band_edges'][1:])})
                            ax=row['methods'][label]['axes'][f'axis{axis}']
                            for k,v in curves.items():close(f'parseval.curve.{i}.{label}.{axis}.{k}',arr[f'{label}_axis{axis}_{k}_mse'][i],v)
                            for key,mask in {'total':np.ones(len(f),bool),**masks}.items():
                                target=ax['total'] if key=='total' else ax['bands'][key]
                                expected={k+'_mse':float(v[mask].sum()) for k,v in curves.items()}
                                expected.update(rmse=float(np.sqrt(expected['error_mse'])),
                                    fraction_of_total_mse=expected['error_mse']/mse if mse>0 else 0.,
                                    bins=int(mask.sum()))
                                close(f'parseval.metric.{i}.{label}.{axis}.{key}',target,expected)
                            check(f'parseval.residual.{i}.{label}.{axis}',
                                  ax['parseval_absolute_residual']<=max(mse,1.)*2e-11)
                            close(f'parseval.identity.{i}.{label}.{axis}',curves['error'].sum(),mse)
                            close(f'parseval.decomposition.{i}.{label}.{axis}',curves['magnitude']+curves['phase'],curves['error'])
        else:
            check('unknown_enhancement:'+method,False)
        for label in predictions:
            expected=summarize([row['methods'][label] for row in result['per_frame']])
            close(method+'.summary.'+label,result['summary'][label],expected)
        if manifest['groups'] is not None:
            for group in set(manifest['groups']):
                for label in predictions:
                    expected=summarize([row['methods'][label] for row in result['per_frame'] if row['group']==group])
                    close(method+'.group.'+group+'.'+label,result['summary_by_group'][group][label],expected)
    result = {'passed':all(x['passed'] for x in checks),'checks_total':len(checks),
            'checks_passed':sum(x['passed'] for x in checks),'failures':[x for x in checks if not x['passed']],
            'scope':'actual input/code hashes, independent full FFT PSD and enhancements, curves and summaries'}
    if include_details:
        result['checks'] = checks
    return result


def main(argv=None):
    """Read-only console entry point."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--details-out', help='Write named checks to a new JSON file; never overwrite.')
    args = parser.parse_args(argv)
    target = Path(args.details_out).resolve() if args.details_out else None
    if target is not None and target.exists():
        parser.error('details-out must be a new file.')
    result = verify_run(args.run, include_details=target is not None)
    details = result.pop('checks', None)
    if target is not None:
        with target.open('x', encoding='utf-8') as f:
            json.dump({'schema':'fourier-check-details-v1', 'run':str(Path(args.run).resolve()),
                       **result, 'checks':details}, f, ensure_ascii=False, indent=2)
            f.write('\n')
        result['details_file'] = str(target)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1
