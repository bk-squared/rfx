"""Replay retained estimator inputs only; run from the repository root.

Usage: OPENBLAS_NUM_THREADS=1 python <this-file> --before /path/to/old/harminv.py --out /fresh/report.json
The previous source can be obtained with git show 5ee9539b:rfx/harminv.py.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a fresh output path')
    root = Path.cwd()
    source = root / 'rfx/harminv.py'
    before, after = load(args.before, '_capacity_before'), load(source, '_capacity_after')
    records = []
    fixtures = root / 'tests/fixtures/harminv_decimation'
    notes = root / 'docs/research_notes/issue872'
    for manifest in sorted(fixtures.glob('*/capture.json')):
        for record in json.loads(manifest.read_text())['records']:
            path = manifest.parent / record['file']
            assert sha(path) == record['sha256']
            with np.load(path) as data:
                signal = data['signal']
            # Exercise auto even on cv24's explicitly undecimated controls.
            records.append((path, None, signal, record['dt'], record['f_min'], record['f_max']))
    for lane in ('cv02-meep117', 'cv02-meep134'):
        manifest = notes / lane / 'candidate-inputs/estimator-inputs.json'
        for record in json.loads(manifest.read_text())['records']:
            path = manifest.parent / record['file']
            assert sha(path) == record['sha256']
            with np.load(path) as data:
                signal = data['signal']
            records.append((path, None, signal, record['dt'], record['f_min'], record['f_max']))
    for arm in ('unfed', 'fed'):
        path = notes / 'patch-gpu-369367260362' / (arm + '-input.npz')
        with np.load(path) as data:
            for channel, signal in enumerate(data['signals']):
                records.append((path, channel, signal, float(data['dt']), 6e9, 14e9))
    stage = fixtures / 'stage1/input.npz'
    with np.load(stage) as data:
        records.append((stage, None, data['signal'], float(data['dt']),
                        float(data['fmin']), float(data['fmax'])))
    report = dict(before_sha256=sha(args.before), after_sha256=sha(source),
                  numpy=np.__version__, scipy=scipy.__version__, records=[])
    for path, channel, signal, dt, lo, hi in records:
        row = dict(file=str(path.relative_to(root)), channel=channel,
                   input_sha256=sha(path), n_samples=len(signal), dt=dt, f_min=lo, f_max=hi)
        for key, module in [('before', before), ('after', after)]:
            start = time.monotonic()
            modes = module.harminv(signal, dt, lo, hi)
            row[key] = dict(plan=module._decimation_plan(len(signal), dt, hi, 'auto'),
                            duration=module.harminv_record_duration(len(signal), dt, hi),
                            modes=[m._asdict() for m in modes], wall_s=time.monotonic()-start)
        row['exact_same_modes'] = row['before']['modes'] == row['after']['modes']
        row['same_plan'] = row['before']['plan'] == row['after']['plan']
        if path != stage:
            assert row['same_plan'] and row['exact_same_modes'], path
        else:
            continuum = 299792458 / 2 * np.sqrt(2) / .04
            for key in ('before', 'after'):
                peak = min(row[key]['modes'], key=lambda m: abs(m['freq']-continuum))['freq']
                row[key]['continuum_error_pct'] = 100 * abs(peak/continuum - 1)
            assert row['before']['continuum_error_pct'] > .03
            assert row['after']['continuum_error_pct'] <= .03
        report['records'].append(row)
        print(path.parent.name, channel, 'unchanged' if row['exact_same_modes'] else 'changed', flush=True)
    report['unchanged_records'] = sum(row['exact_same_modes'] for row in report['records'])
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
