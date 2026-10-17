"""Item 5 gradient, traced entry paths, and lumped-port spectrum."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._distributed_dispersive_cpml import check, measure, record, scene

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize('kind', ['debye', 'lorentz', 'mixed'])
def test_eps_gradient(kind):
    sim = scene(kind, cells=False)
    eps = jnp.asarray(sim._assemble_materials_nu(sim._build_nonuniform_grid())[0].eps_r)
    def objective(e, devices=None):
        kw = {} if devices is None else dict(distributed=True, devices=devices)
        traces = sim.forward(eps_override=e, n_steps=400, checkpoint=False,
                             skip_preflight=True, **kw).time_series
        return jnp.sum(traces**2)
    one = np.asarray(jax.grad(objective)(eps), np.float64)
    for devices in (2, 3):
        multi = np.asarray(jax.grad(lambda e: objective(e, jax.devices('cpu')[:devices]))(eps), np.float64)
        assert np.isfinite(one).all() and np.isfinite(multi).all() and np.any(one)
        relative = float(np.abs(multi - one).max() / np.abs(one).max())
        row = dict(kind=kind, entry='gradient', n=400, devices=devices, relative=relative)
        record(row)
        assert relative <= 1e-4, row


