"""PI 2026-10-09: occupancy weights sources on each single-device lane."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from tests.contracts.path_equivalence.comparison import compare

WEIGHTS = (1., .5, .25, .0625)


def scene(path, w, *, constant=False):
    kwargs = {}
    if 'graded' in path:
        kwargs['dy_profile'] = (np.full(16, .001) if constant else
                                np.array([.001] + [.0009, .0011] * 7 + [.001]))
    sim = Simulation(freq_max=15e9, domain=(.020, .016, .012), dx=.001,
                     boundary='pec', **kwargs)
    grid = sim._build_nonuniform_grid() if 'graded' in path else sim._build_grid()
    sim.add_source((.006, .007, .005), 'ez', amplitude_kind='field')
    # The third probe is the source edge itself: the record's largest field.
    for position in ((.010, .007, .005), (.014, .009, .006), (.006, .007, .005)):
        sim.add_probe(position, 'ez')
    # Four incident cells supply the source edge's weight; no rfx weight helper.
    occupancy = np.zeros(grid.shape, dtype=np.float32)
    occupancy[5:7, 6:8, 4:6] = 1. - w**.25
    return sim, jnp.asarray(occupancy)


def record(path, w, *, constant=False):
    sim, occupancy = scene(path, w, constant=constant)
    kwargs = {}
    if 'distributed' in path:
        if len(jax.devices()) < 2:
            pytest.skip(f'needs two devices on one backend, found {len(jax.devices())}')
        kwargs.update(distributed=True, devices=jax.devices()[:2])
    result = sim.forward(n_steps=120, pec_occupancy_override=occupancy,
                         checkpoint=False, skip_preflight=True, **kwargs)
    values = np.asarray(result.time_series)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    return values


@pytest.fixture(scope='module')
def graded_records():
    return {w: (record('graded', w), record('distributed_graded', w))
            for w in WEIGHTS}


def difference(one, two):
    peaks = np.max(np.abs(one), axis=0)
    assert np.all(peaks > 0)
    return float(np.max(np.abs(one.astype(np.float64) - two) / peaks))


@pytest.mark.parametrize('w', WEIGHTS)
def test_g1_graded_one_vs_two_devices(graded_records, w, record_property):
    baseline = difference(*graded_records[1.])
    actual = difference(*graded_records[w])
    record_property('d(1)', baseline)
    record_property('d(w)', actual)
    for lane, values in zip(('one', 'two'), graded_records[w]):
        record_property(f'peaks_{lane}', np.max(np.abs(values), axis=0).tolist())
    assert baseline < 1e-4
    # Per-step cross-trace bar: 9 float32 ULP of the record's largest field
    # (the source edge). One device adds the stored w*s to w*E, two devices
    # still form w*(E + s) until the slab step takes the same form, so the
    # driven edge differs by one rounding per step; on x86 the w = 1 row is
    # below 9 ULP of a probe's own peak and cannot serve as the unit.
    one, two = graded_records[w]
    family_peak = np.float32(np.max(np.abs(one)))
    assert family_peak == np.max(np.abs(one[:, 2]))
    worst = float(np.max(np.abs(one.astype(np.float64) - two)))
    record_property('worst_abs', worst)
    record_property('family_peak', float(family_peak))
    assert worst <= 9 * float(np.spacing(family_peak))


@pytest.mark.parametrize('w', WEIGHTS)
def test_g1b_uniform_vs_constant_graded(w, record_property):
    one = record('uniform', w)
    two = record('graded', w, constant=True)
    measurements = []
    for probe in range(2):
        compare(one[:, probe], two[:, probe], record=f'probe_{probe + 1}',
                kind='step', measurements=measurements)
    record_property('comparisons', measurements)


@pytest.mark.parametrize('w', WEIGHTS)
def test_g1c_probe_peak_ratio(graded_records, w, record_property):
    one, two = graded_records[w]
    ratio = float(np.max(np.abs(one[:, 0])) / np.max(np.abs(two[:, 0])))
    record_property('probe_1_peak_ratio', ratio)
    assert abs(ratio - 1.) <= 1e-3


# rfx-archive records/20261008-occupancy-weight-order, main-era one-device peaks
MAIN_PEAKS = {
    'uniform': (0.033985, 0.0086755, 0.0070076, 0.0070738),
    'graded': (0.035304, 0.0087263, 0.0070309, 0.0070737),
}
TWO_DEVICE_PEAKS = (0.035304, 0.0043632, 0.0017577, 0.00044211)


@pytest.mark.parametrize('lane', ['uniform', 'graded', 'distributed_graded'])
@pytest.mark.parametrize('w', WEIGHTS)
def test_h4_reference_peak(lane, w, graded_records, record_property):
    index = WEIGHTS.index(w)
    if lane == 'uniform':
        values = record(lane, w)
    else:
        values = graded_records[w][int(lane == 'distributed_graded')]
    peak = float(np.max(np.abs(values[:, 0])))
    expected = (TWO_DEVICE_PEAKS[index] if lane == 'distributed_graded' else
                w * MAIN_PEAKS[lane][index])
    record_property('probe_1_peak', peak)
    record_property('expected_peak', expected)
    record_property('peak_over_main', peak / MAIN_PEAKS.get(lane, MAIN_PEAKS['graded'])[index])
    assert abs(peak / expected - 1.) <= 1e-3


def test_occupancy_gradient_one_device_equals_two_at_zero_occupancy():
    """At w = 1 the values agree on any form; the derivative with respect to w does not.

    One device used to add the source unweighted (w*E + s) while two devices
    formed w*(E + s): equal values at zero occupancy, different occupancy
    gradients in the cells round the source edge (5 % of the largest entry
    on this scene before the drive carried its weight). Accumulated bar.
    """
    if len(jax.devices()) < 2:
        pytest.skip(f'needs two devices on one backend, found {len(jax.devices())}')
    gradients = []
    for kwargs in ({}, dict(distributed=True, devices=jax.devices()[:2])):
        sim, occupancy = scene('graded', 1.)

        def loss(cells):
            result = sim.forward(n_steps=120, pec_occupancy_override=cells,
                                 checkpoint=False, skip_preflight=True, **kwargs)
            return jnp.sum(result.time_series[:, :2] ** 2)
        gradients.append(np.asarray(jax.grad(loss)(occupancy)))
    one, two = gradients
    peak = np.max(np.abs(one))
    assert peak > 0 and np.max(np.abs(one[5:7, 6:8, 4:6])) > .1 * peak
    assert np.max(np.abs(one - two)) <= 1e-4 * peak
