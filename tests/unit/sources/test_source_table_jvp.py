"""The graded closed-box record: a field drive has no material tangent."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _record(steps):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rfx import Simulation
    from rfx.model import source_coefficients as tables
    from unittest.mock import patch
    def fwd(a, distributed, with_table=False):
        sim = Simulation(freq_max=15e9, domain=(.015, .011, .009), dx=.001,
                         boundary='pec', snap='declared',
                         dy_profile=np.array([.001] + [.0009, .0011] * 4 + [.001, .001]))
        sim.add_source((.003, .004, .003), 'ez', amplitude_kind='field')
        for pos in ((.006, .005, .004), (.011, .007, .006)):
            sim.add_probe(pos, 'ez')
        grid = sim._build_nonuniform_grid()
        built = []
        original = tables.source_table
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            built.append(result)
            return result
        with patch.object(tables, 'source_table', capture):
            result = sim.forward(
                n_steps=steps, eps_override=jnp.ones(grid.shape) * a,
                distributed=distributed,
                devices=jax.devices()[:2] if distributed else None,
                checkpoint=False, skip_preflight=True).time_series
        expected = jax.vmap(sim._ports[0].waveform)(
            jnp.arange(steps, dtype=jnp.float32) * grid.dt)
        return (result, built[-1], expected) if with_table else result
    outputs = [jax.jvp(lambda a: fwd(a, dist, True), (jnp.float32(1.1),),
                       (jnp.float32(1.),)) for dist in (False, True)]
    converted = []
    for (primal, table, expected), (tangent, dtable, _) in outputs:
        np.testing.assert_array_equal(table, expected)
        assert np.all(np.asarray(dtable) == 0.)
        converted.append((np.asarray(primal), np.asarray(tangent)))
    (a, da), (b, db) = converted
    peak = np.max(abs(a), axis=1, keepdims=True)
    ulp = float(np.max(abs(a - b) / np.spacing(peak)))
    relative = float(np.max(
        np.max(abs(da - db), axis=0) / np.max(abs(da), axis=0)))
    np.testing.assert_array_equal(a, np.asarray(fwd(jnp.float32(1.1), False)))
    assert ulp <= 9
    assert relative <= 1e-4
    return dict(steps=steps, dtype=str(a.dtype), primal_ulp=ulp, jvp_relative=relative)


@pytest.mark.parametrize('steps', [180, 720])
def test_graded_field_jvp_one_equals_two_devices(steps):
    env = dict(os.environ, JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(steps)],
                          env=env, text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.splitlines()[-1])['primal_ulp'] <= 9


if __name__ == '__main__':
    print(json.dumps(_record(int(sys.argv[1]))))
