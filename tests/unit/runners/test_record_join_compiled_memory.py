"""Whole-program compiled memory is context, not an eager-join acceptance bar."""
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from rfx import SnapshotSpec
from rfx.runners import _distributed_records as records
from rfx.runners._distributed_ntff import SlabNTFF
from tests._multi_device_scene import host_devices
from tests.unit.runners._record_join_frozen import assemble as frozen
from tests.unit.runners._graded_records_batching import problem
from tests.unit.runners.test_distributed_uniform_user_jit import record_scene, record_arrays
from tests.unit.runners.test_distributed_record_planes import model


def statistics(program, *args):
    compiled = program.lower(*args).compile()
    stats = compiled.memory_analysis()
    assert stats is not None, 'whole-program memory unavailable'
    result = {key: getattr(stats, key) for key in
              ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
    hlo = compiled.as_text()
    result['device_identity_lines'] = [line.strip() for line in hlo.splitlines()
                                       if 'partition-id(' in line or 'replica-id(' in line or line.startswith('HloModule')]
    return result


@pytest.mark.parametrize('kind', ['planes', 'flux', 'ntff', 'moments', 'snapshots'])
def test_compiled_memory_context(kind, monkeypatch):
    rows = {}
    for version in ('old', 'new'):
        # Distinct enclosing callables prevent an old trace from being reused.
        with monkeypatch.context() as patch:
            if version == 'old':
                for cls in (records.PlaneRecords, records.SnapshotRecords, records.MomentRecords, SlabNTFF):
                    patch.setattr(cls, 'assemble', lambda self, buffer, **kw: frozen(self, buffer))
            if kind == 'snapshots':
                sim, _, _ = model('cpml', 2)
            else:
                sim = record_scene('dft' if kind == 'planes' else kind)

            def run():
                options = dict(snapshot=SnapshotSpec(interval=1, components=('ez',))) if kind == 'snapshots' else {}
                result = sim.run(n_steps=8, devices=host_devices(2), skip_preflight=True, **options)
                return result.snapshots if kind == 'snapshots' else record_arrays(result, 'dft' if kind == 'planes' else kind)

            rows[version] = {'jit_run': statistics(jax.jit(run))}
            if kind != 'snapshots':
                solve = problem(kind, 'cpml', patch)
                def objective(alpha):
                    return sum(jnp.sum(jnp.abs(a)**2) for a in solve(alpha, True).values())
                rows[version]['grad_forward'] = statistics(jax.jit(jax.grad(objective)), jnp.float32(1.))
            else:
                rows[version]['grad_forward'] = 'not applicable: forward has no snapshot argument'
    print(json.dumps(dict(kind=kind, compiled_memory=rows)), flush=True)


@pytest.mark.parametrize('version', ['old', 'new'])
def test_enclosing_user_jit_dump_context(version, tmp_path):
    """Name the enclosing-jit identity modules separately from the rank contract."""
    dump = tmp_path / 'hlo'
    dump.mkdir()
    root = Path(__file__).resolve().parents[3]
    program = '''
import sys
import pytest
if sys.argv[1] == 'old':
    from rfx.runners import _distributed_records as records
    from rfx.runners._distributed_ntff import SlabNTFF
    from tests.unit.runners._record_join_frozen import assemble
    for cls in (records.PlaneRecords, records.SnapshotRecords, records.MomentRecords, SlabNTFF):
        cls.assemble = lambda self, buffer, **kw: assemble(self, buffer)
sys.exit(pytest.main([
    'tests/unit/runners/test_distributed_uniform_user_jit.py',
    'tests/unit/runners/test_distributed_uniform_forward_user_jit.py',
    '-p', 'no:cacheprovider', '-q']))
'''
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS='cpu',
               JAX_ENABLE_COMPILATION_CACHE='false',
               XLA_FLAGS=f'--xla_force_host_platform_device_count=2 --xla_dump_to={dump}')
    run = subprocess.run([sys.executable, '-c', program, version], cwd=root, env=env,
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr
    modules = sorted(dump.rglob('*after_optimizations.txt'))
    assert modules, 'no user-jit optimized dumps'
    identities = []
    for module in modules:
        text = module.read_text()
        operations = [op for op in ('partition-id', 'replica-id', 'partition_id', 'replica_id') if op in text]
        if operations:
            identities.append(dict(module=module.name, operations=operations))
    print(json.dumps(dict(version=version, user_jit_identity_modules=identities,
                          summary=run.stdout.splitlines()[-1])), flush=True)
    assert all('.jit_call.' in row['module'] for row in identities), identities
