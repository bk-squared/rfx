"""Scene T lossy Lorentz absorber: law C and bar S against one device."""
import jax
import numpy as np
import pytest

from rfx.runners import _distributed_nu_wrapped as runner
from tests._distributed_dispersive_cpml import check, measure, record, scene

pytestmark = pytest.mark.distributed


def test_lossy_lorentz_law_c_and_bar_s(monkeypatch):
    sim = scene('lorentz', sigma=.5)
    grid = sim._build_nonuniform_grid()
    sigma = np.asarray(sim._assemble_materials_nu(grid)[0].sigma)
    assert np.all(sigma[:, grid.ny//2, grid.nz//2] == .5)
    one = np.asarray(sim.forward(n_steps=400, checkpoint=False, skip_preflight=True).time_series)
    original, calls = runner.build_setup, []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'build_setup', observe)
    multi = np.asarray(sim.forward(n_steps=400, checkpoint=False, skip_preflight=True,
        distributed=True, devices=jax.devices('cpu')[:2]).time_series)
    assert calls, 'wrapped setup not reached'
    row = dict(kind='lorentz_lossy', sigma=.5, entry='forward', devices=2, n=400,
               **measure(one, multi, float(grid.dt)))
    record(row)
    check(row)
