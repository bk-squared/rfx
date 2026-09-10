"""Fresh-process G5 contraction controls; no candidate imported or measured."""
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation/research/nu_cost/g5'


def worker(label):
    import jax
    import numpy as np
    if label == 'precision_highest':
        jax.config.update('jax_default_matmul_precision', 'highest')
    m = runpy.run_path(str(ROOT / 'tests/unit/boundaries/test_cpml_localization.py'))
    arrays = {}
    for name in m['FIXTURES']:
        result = jax.jit(lambda: m['runner'](name, m['old'])[0])()
        for group in result:
            for key in group._fields:
                if key == 'step':
                    continue
                arrays[name + '/' + key] = np.asarray(getattr(group, key))
        print(label, name, 'completed', flush=True)
    np.savez_compressed(OUT / (label + '.npz'), **arrays)
    print('VERSION', jax.__version__)
    for op in (jax.lax.mul, jax.lax.add, jax.numpy.multiply, jax.lax.dot_general):
        print(op.__name__, inspect.signature(op))


def controls():
    import numpy as np
    flags = {
        'unflagged': '',
        'excess_false': '--xla_allow_excess_precision=false',
        'precision_highest': '',
        'fast_math_false': '--xla_cpu_enable_fast_math=false',
        'legacy_emitter': '--xla_cpu_use_fusion_emitters=false --xla_cpu_enable_fast_math=false',
        'no_fusion': '--xla_disable_hlo_passes=fusion --xla_cpu_enable_fast_math=false',
    }
    records = {}
    for label, flag in flags.items():
        env = dict(os.environ, XLA_FLAGS=flag, PYTHONDONTWRITEBYTECODE='1')
        if label in ('unflagged', 'no_fusion', 'legacy_emitter'):
            env['XLA_FLAGS'] += ' --xla_dump_to=' + str(OUT / ('dump_' + label)) + ' --xla_dump_hlo_as_text'
        with (OUT / (label + '.log')).open('w') as stream:
            process = subprocess.run([sys.executable, __file__, label], env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {'flags': flag, 'exit': process.returncode}
        if process.returncode == 0:
            with np.load(OUT / 'unflagged.npz') as a, np.load(OUT / (label + '.npz')) as b:
                diffs = {key: int(np.count_nonzero(a[key] != b[key])) for key in a}
                record['differing_elements'] = diffs
                record['total_differing'] = sum(diffs.values())
                record['changed_arrays'] = sum(v > 0 for v in diffs.values())
        records[label] = record
        (OUT / 'controls.json').write_text(json.dumps(records, indent=2) + '\n')
        print(label, {k:v for k,v in record.items() if k != 'differing_elements'}, flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        controls()
