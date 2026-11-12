"""Owner-only x-face derivative, scaled above float32 underflow, old and new."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.probes.probes import flux_spectrum
from rfx.runners._distributed_records import PlaneRecords
from tests._multi_device_scene import host_devices
from tests.unit.runners._record_join_frozen import PlaneRecords as frozen_plane
from tests.unit.runners._graded_records_scene import measure, coordinate


@pytest.mark.parametrize('side', ['low', 'high'])
@pytest.mark.parametrize('scale', [pytest.param(.1, marks=pytest.mark.slow), 1.,
                                  pytest.param(10., marks=pytest.mark.slow)])
def test_x_face_gradient(side, scale, monkeypatch):
    def build(amplitude):
        profile = np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 13), 1e-3]
        sim = Simulation(freq_max=15e9, domain=(15e-3, 8e-3, 8e-3), dx=1e-3,
                         boundary='cpml', cpml_layers=2, dx_profile=profile)
        sim.add_source((5.1e-3, 3.2e-3, 4.3e-3), 'ez', amplitude_kind='field',
                       waveform=lambda t: amplitude*jnp.exp(-((t-1.2e-11)/5e-12)**2))
        grid = sim._build_nonuniform_grid()
        seam = (grid.shape[0] + 1)//2
        index = seam - 2 if side == 'low' else seam + 2
        x = coordinate(grid, 0, index)
        sim.add_flux_monitor(axis='x', coordinate=x, freqs=[3e9], name='flux')
        sim.add_probe((x, 3.2e-3, 4.3e-3), 'ez')
        return sim, grid

    sim, grid = build(1.)
    eps = jnp.full(grid.shape, 1.3, jnp.float32)
    options = dict(n_steps=64, checkpoint=False, skip_preflight=True)
    reference = sim.forward(eps_override=eps, **options)
    field_peak = float(np.max(np.abs(reference.time_series)))
    assert field_peak > 0, 'x-face calibration signal'
    sim, grid = build(1/field_peak)
    reference = sim.forward(eps_override=eps, **options)
    normalized_field = float(np.max(np.abs(reference.time_series)))
    assert .5 <= normalized_field <= 2, ('x-face field normalization', normalized_field)
    # DFT records carry seconds; normalize the objective before differentiation.
    power_scale = float(abs(np.asarray(flux_spectrum(reference.flux_monitors['flux']))).max())
    assert power_scale > 0

    def objective(e, distributed, scale):
        result = sim.forward(eps_override=e, **options,
                             **(dict(distributed=True, devices=host_devices(2), checkpoint_every=5)
                                if distributed else {}))
        return jnp.sum(flux_spectrum(result.flux_monitors['flux'])/power_scale)*scale

    expected = jax.grad(lambda e: objective(e, False, scale))(eps)
    actual = jax.grad(lambda e: objective(e, True, scale))(eps)
    with monkeypatch.context() as patch:
        patch.setattr(PlaneRecords, 'assemble', lambda self, buffer, **kw: frozen_plane(self, buffer))
        old = jax.grad(lambda e: objective(e, True, scale))(eps)
    row = dict(side=side, objective_scale=scale, field_peak=normalized_field,
               old=measure(old, expected), new=measure(actual, expected), join_change=measure(actual, old))
    print(json.dumps(row), flush=True)
    assert row['new']['peak'] > 1e-20, 'x-face gradient above floor'
    assert np.isfinite(row['new']['max_abs']), 'x-face gradient finite'
    assert row['join_change']['max_abs'] <= 9*np.spacing(np.float32(row['join_change']['peak'])), ('x-face join changed gradient', row)
    assert row['new']['relative'] <= 1e-4, ('x-face one-device gradient bar', row)
