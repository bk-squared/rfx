"""NTFF surface placement, far field and G5 derived patterns."""
import numpy as np
import pytest
import jax.numpy as jnp
from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.farfield import compute_far_field, directivity, radiation_pattern
from tests._multi_device_scene import host_devices
from tests.unit.runners._graded_records_scene import pair, measure, record, mutation_result, with_bars


DOMAIN = (15.3e-3, 8.7e-3, 10.1e-3)
SOURCE = (5.1e-3, 2.7e-3, 4.3e-3)
PROBE = (10.2e-3, 5.6e-3, 6.7e-3)
BOX = ((1.7e-3, 1.3e-3, 2.2e-3), (14.1e-3, 7.1e-3, 8.1e-3))


def add_box(sim):
    """Retain the box used by the separate gradient, pole and one-device tests."""
    sim.add_ntff_box((2e-3, 1e-3, 1e-3), (12e-3, 7e-3, 7e-3), freqs=[3e9, 7e9])


def ntff_model(boundary, n_devices):
    host_devices(n_devices)
    faces = (BoundarySpec(x=Boundary('cpml', 'cpml', 2, 4),
                          y=Boundary('cpml', 'cpml', 3, 2),
                          z=Boundary('cpml', 'cpml', 2, 3)) if boundary == 'cpml' else 'pec')
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=1e-3, boundary=faces,
                     dx_profile=np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 13), 1e-3])
    sim.add_source(SOURCE, 'ez', amplitude_kind='field',
                   waveform=lambda t: jnp.exp(-((t-1.2e-11)/5e-12)**2))
    sim.add_probe(PROBE, 'ez')
    return sim, sim._build_nonuniform_grid()


@pytest.mark.parametrize('n_devices', [2, 3])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_ntff(boundary, n_devices, entry, monkeypatch):
    sim, grid = ntff_model(boundary, n_devices)
    # Reflecting-face PEC witness checks the same record/transform arithmetic.
    if boundary == 'pec':
        monkeypatch.setattr('rfx.boundaries.features.refuse_ntff_reflecting_faces', lambda *a, **kw: None)
    sim.add_ntff_box(*BOX, freqs=[3e9, 7e9])
    one, many = pair(sim, n_devices, entry)
    # The six face sums are the physical records consumed by compute_far_field;
    # c_* leaves are internal Kahan carry. Keep full-tree PEC equality unchanged.
    names = one.ntff_data._fields if boundary == 'pec' else one.ntff_data._fields[:6]
    values = {name: measure(getattr(many.ntff_data, name), getattr(one.ntff_data, name))
              for name in names}
    theta, phi = np.linspace(0, np.pi, 9), np.linspace(0, 2 * np.pi, 11)
    ff = [compute_far_field(r.ntff_data, r.ntff_box, r.grid, theta, phi) for r in (one, many)]
    for name in ('E_theta', 'E_phi'):
        values[name] = measure(getattr(ff[1], name), getattr(ff[0], name))
    for function in (directivity, radiation_pattern):
        expected, actual = (function(f) for f in ff)
        if boundary == 'cpml':
            # dB(i) is a power ratio: convert back with 10**(dB/10).
            expected, actual = (10 ** (a / 10) for a in (expected, actual))
        values[function.__name__] = measure(actual, expected)
    # CPML wrapped step: compile-form rounding, operands byte-equal (rfx-archive DESIGN.md 4g amendment 1).
    with_bars(values, 'relative_peak' if boundary == 'cpml' else 'array_equal')
    record('ntff', dict(boundary=boundary, devices=n_devices, entry=entry, shape=grid.shape,
        domain=DOMAIN, source=SOURCE, probe=PROBE, box=BOX,
        pattern_scale='linear_power' if boundary == 'cpml' else 'dB(i)',
        face_depths=[(getattr(grid, f'pad_{a}_lo'), getattr(grid, f'pad_{a}_hi')) for a in 'xyz']), values)
    assert max(v['peak'] for v in values.values()) > 0
    assert all(v['passed'] for v in values.values()), ('NTFF bar', values)


@pytest.mark.parametrize('mutation', ['skip', 'late_phase'])
@pytest.mark.parametrize('boundary', ['pec', 'cpml'])
def test_ntff_mutations(mutation, boundary, monkeypatch):
    from rfx.runners._distributed_ntff import SlabNTFF
    original = SlabNTFF.update
    monkeypatch.setattr(SlabNTFF, 'update',
        (lambda self, buffer, state, dt, step, **kw: original(self, buffer, state, dt, step + 1, **kw))
        if mutation == 'late_phase' else (lambda self, buffer, *a, **kw: buffer))
    with pytest.raises(AssertionError, match='NTFF bar') as failure:
        test_ntff(boundary, 3, 'run', monkeypatch)
    mutation_result('ntff', f'{mutation}/{boundary}', failure.value)
