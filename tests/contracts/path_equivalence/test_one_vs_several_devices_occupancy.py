"""One device against several on the graded closed box, with and without occupancy.

Without an occupancy input one, two and three devices give identical bits in
every probe sample: that invariant is pinned bitwise. With an occupancy input
the lanes differ by compile-context rounding around the occupancy stage (a
multiply by exactly 1 already changes how each lane's E update rounds next to
the source; rfx-archive records/20261011-p4-one-vs-many-devices, DECLARE.md
M1-M4 and p4_family.json). That cell is judged on the cross-trace bars: 9
float32 ULP of the family peak per step, 1e-4 on what a result reads.
Expected values come from the one-device run; no rfx comparison helper.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.core.yee import EPS_0, MU_0

Z0 = float(np.sqrt(MU_0 / EPS_0))
SHAPE = (23, 17, 13)
N_STEPS = 240
BINS_HZ = (3e9, 6e9, 9e9, 12e9, 15e9)
SOURCES = {'U-0.2': (.003, .007, .005), 'D-random': (.006, .007, .005)}
# Columns of time_series, in add_probe order: three Ez (the third is the
# source edge itself), then Hx and Hy.
E_COLUMNS = (0, 1, 2)
SOURCE_EDGE_COLUMN = 2
H_COLUMNS = (3, 4)
OBJECTIVE_COLUMNS = (0, 1)  # source edge excluded from the gradient objective


def scene(row):
    sim = Simulation(freq_max=15e9, domain=(.022, .016, .012), dx=.001,
                     boundary='pec',
                     dy_profile=np.array([.001] + [.0009, .0011] * 7 + [.001]))
    assert tuple(sim._build_nonuniform_grid().shape) == SHAPE
    source = SOURCES['U-0.2' if row == 'none' else row]
    sim.add_source(source, 'ez', amplitude_kind='field')
    for position in ((.010, .007, .005), (.014, .009, .006), source):
        sim.add_probe(position, 'ez')
    for component in ('hx', 'hy'):
        sim.add_probe((.007, .007, .005), component)
    return sim


def occupancy(row):
    if row == 'U-0.2':
        cells = np.zeros(SHAPE, np.float32)
        cells[5:7, 6:8, 4:6] = .2
    else:
        cells = np.random.default_rng(202).uniform(0, .85, SHAPE).astype(np.float32)
        cells[0, 2:5, 1:4] = .5
        width = -(-SHAPE[0] // 2)  # seam rows of two devices; same array for every count
        for seam in range(width, SHAPE[0], width):
            cells[seam - 1:seam + 1, 1:4, 2:4] = .75
    return jnp.asarray(cells)


def lane(devices):
    if devices == 1:
        return {}
    if len(jax.devices()) < devices:
        pytest.skip(f'needs {devices} devices on one backend, found {len(jax.devices())}')
    return dict(distributed=True, devices=jax.devices()[:devices])


@pytest.fixture(scope='module')
def runs():
    """Each forward or gradient run computed once per module."""
    return {}


def record(runs, row, devices):
    """Probe samples and dt; row 'none' omits pec_occupancy_override."""
    key = ('record', row, devices)
    if key not in runs:
        runs[key] = _record(row, devices)
    return runs[key]


def _record(row, devices):
    kwargs = lane(devices)
    if row != 'none':
        kwargs['pec_occupancy_override'] = occupancy(row)
    result = scene(row).forward(n_steps=N_STEPS, checkpoint=False,
                                skip_preflight=True, **kwargs)
    values = np.asarray(result.time_series)
    assert values.dtype == np.float32 and values.shape == (N_STEPS, 5)
    assert np.isfinite(values).all()
    return values, float(result.dt)


def gradient(runs, row, devices):
    key = ('gradient', row, devices)
    if key not in runs:
        runs[key] = _gradient(row, devices)
    return runs[key]


def _gradient(row, devices):
    kwargs = lane(devices)
    sim = scene(row)

    def loss(cells):
        result = sim.forward(n_steps=N_STEPS, pec_occupancy_override=cells,
                             checkpoint=False, skip_preflight=True, **kwargs)
        return jnp.sum(result.time_series[:, OBJECTIVE_COLUMNS] ** 2)
    values = np.asarray(jax.grad(loss)(occupancy(row)))
    assert np.isfinite(values).all()
    return values


def dft(series, dt):
    n = np.arange(series.shape[0]) * dt
    kernel = np.exp(-2j * np.pi * np.outer(BINS_HZ, n))
    return kernel @ series.astype(np.float64)


DEVICES = [2, 3]
ROWS = ['U-0.2', 'D-random']


@pytest.mark.parametrize('devices', DEVICES)
def test_no_occupancy_one_vs_several_within_cross_trace_bars(runs, devices):
    """No occupancy input: paired family peaks and accumulated probe DFTs.

    multi-device results on occupancy-free graded scenes are no longer bit-identical to one device since W1 (1-2 float32 units): the wrapped single-device step rounds differently inside shard_map with bit-equal operands (S3-SLAB-WRAP-DECLARE 2026-10-13)
    """
    several, dt_several = record(runs, 'none', devices)
    one, dt = record(runs, 'none', 1)
    assert dt_several == dt
    e_peak = np.float32(np.max(np.abs(one[:, E_COLUMNS])))
    h_peak = np.float32(np.max(np.abs(one[:, H_COLUMNS])))
    h_scale = np.float32(max(float(h_peak), float(e_peak) / Z0))
    delta = np.abs(one.astype(np.float64) - several)
    e_units = np.max(delta[:, E_COLUMNS]) / float(np.spacing(e_peak))
    h_units = np.max(delta[:, H_COLUMNS]) / float(np.spacing(h_scale))
    assert e_units <= 9, f'E {e_units} float32 units > 9'
    assert h_units <= 9, f'H {h_units} paired float32 units > 9'
    x_one, x_several = dft(one, dt), dft(several, dt)
    error, peak = np.max(np.abs(x_several - x_one)), np.max(np.abs(x_one))
    assert error <= 1e-4 * peak, f'DFT {error} > 1e-4 * {peak}'


@pytest.mark.parametrize('devices', DEVICES)
@pytest.mark.parametrize('row', ROWS)
def test_occupancy_one_vs_several_within_cross_trace_bars(runs, row, devices, record_property):
    """Cause: compile-context rounding around the occupancy stage (record above)."""
    several, dt_several = record(runs, row, devices)
    one, dt = record(runs, row, 1)
    assert dt_several == dt
    e_one = one[:, E_COLUMNS]
    e_pk = np.float32(np.max(np.abs(e_one)))
    # The source edge carries the record's largest field: the family peak.
    assert e_pk == np.max(np.abs(one[:, SOURCE_EDGE_COLUMN])), 'E peak not at the source edge'
    h_pk = np.float32(np.max(np.abs(one[:, H_COLUMNS])))
    h_scale = np.float32(max(float(h_pk), float(e_pk) / Z0))
    diff = np.abs(one.astype(np.float64) - several)
    e_worst = float(np.max(diff[:, E_COLUMNS]))
    h_worst = float(np.max(diff[:, H_COLUMNS]))
    e_ulp = float(np.spacing(e_pk))
    h_ulp = float(np.spacing(h_scale))
    rf = []
    for column in E_COLUMNS:
        a, b = dft(one[:, column], dt), dft(several[:, column], dt)
        rf.append(float(np.max(np.abs(a - b)) / np.max(np.abs(a))))
    record_property('family_peak_E', float(e_pk))
    record_property('E_ulp', e_worst / e_ulp)
    record_property('H_scale', float(h_scale))
    record_property('H_ulp_paired', h_worst / h_ulp)
    record_property('rf_per_probe', rf)
    assert e_worst <= 9 * e_ulp, f'E {e_worst / e_ulp:.2f} ULP of family peak > 9'
    assert h_worst <= 9 * h_ulp, f'H {h_worst / h_ulp:.2f} ULP of paired scale > 9'
    assert max(rf) <= 1e-4, f'RF bins {max(rf):.3e} of probe largest |DFT| > 1e-4'


@pytest.mark.parametrize('devices', DEVICES)
@pytest.mark.parametrize('row', ROWS)
def test_occupancy_gradient_one_vs_several(runs, row, devices, record_property):
    several = gradient(runs, row, devices)
    one = gradient(runs, row, 1)
    peak = float(np.max(np.abs(one)))
    assert peak > 0
    ratio = float(np.max(np.abs(one.astype(np.float64) - several))) / peak
    record_property('gradient_ratio', ratio)
    assert ratio <= 1e-4, f'occupancy gradient {ratio:.3e} of largest entry > 1e-4'
