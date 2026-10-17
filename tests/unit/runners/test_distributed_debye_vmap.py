"""PEC Debye: construct the vacuum pole fill inside the vmap trace (#1834)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._distributed_dispersive_cpml import record
from tests.unit.runners.test_distributed_nu_wrapped_step import pole_metrics, pole_scene

pytestmark = pytest.mark.distributed


def test_pec_debye_vmap_reaches_wrapped(monkeypatch):
    from rfx.runners import _distributed_nu_wrapped as runner
    from rfx.stepping.slab import _vacuum_dispersion_values
    sim = pole_scene('S1')
    grid = sim._build_nonuniform_grid()
    eps = jnp.asarray(sim._assemble_materials_nu(grid)[0].eps_r)
    original, calls = runner.build_setup, []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'build_setup', observe)
    def evaluate(devices=None):
        def forward(e):
            kw = {} if devices is None else dict(distributed=True, devices=devices)
            return sim.forward(eps_override=e, n_steps=60, checkpoint=False,
                               skip_preflight=True, **kw).time_series
        return np.asarray(jax.vmap(forward)(eps[None])[0])
    one = evaluate()
    calls.clear()
    # A warm Python cache must not conceal construction under the vmap trace.
    _vacuum_dispersion_values.cache_clear()
    multi = evaluate(jax.devices('cpu')[:2])
    assert calls, 'PEC Debye vmap did not reach the wrapped setup'
    e, h, dft = pole_metrics(one, multi, float(grid.dt))
    row = dict(entry='pec_vmap', kind='debye', devices=2, n=60,
               e_units=float(np.max(e)), h_units=float(np.max(h)), spectral=dft)
    record(row)
    assert np.max(e) <= 9, row
    assert np.max(h) <= 9, row
    assert dft <= 1e-4, row
