"""Normal-incidence estimate and advisory reach; no solver numbers re-pinned."""
from dataclasses import replace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.model.thin_conductors import half_sheet_film_error, warn_dc_films

CODE = 'dc_film_half_sheet_error'


def model(rs=10, *, profile=None, z=.009, sigma=None, f0=None, domain_z=None):
    extent = (.018 if profile is None else 0) if domain_z is None else domain_z
    sim = Simulation(10e9, (.018, .018, extent),
                     dx=.0015, boundary='pec', dz_profile=profile)
    sim.add_thin_conductor(Box((.003, .003, z), (.015, .015, z)),
                           sigma_bulk=1/(rs*35e-6) if sigma is None else sigma,
                           thickness=35e-6, surface_impedance_f0=f0)
    sim.add_source((.006, .006, .0045), 'ex',
                   waveform=GaussianPulse(f0=5e9), amplitude_kind='field')
    sim.add_probe((.006, .006, .012), 'ex')
    return sim


def findings(sim):
    return sim.preflight().by_code(CODE)


@pytest.mark.parametrize('rs', [1000, 377, 100, 30, 10, 3, 1])
@pytest.mark.parametrize('dx', [1.5e-3, .75e-3, .375e-3])
def test_t1_independent_matrix_cascade(rs, dx):
    theta = 2*np.pi*10e9*dx/299792458.0
    y = 376.730313668/rs
    shunt = np.array([[1, 0], [y/2, 1]], dtype=np.complex128)
    line = np.array([[np.cos(theta), 1j*np.sin(theta)],
                     [1j*np.sin(theta), np.cos(theta)]])
    a, b, c, d = (shunt @ line @ shunt).ravel()
    r, t = (a+b-c-d)/(a+b+c+d), 2/(a+b+c+d)
    single = np.array([[1, 0], [y, 1]])
    a1, b1, c1, d1 = single.ravel()
    r1, t1 = (a1+b1-c1-d1)/single.sum(), 2/single.sum()
    expected = [theta*y, 20*np.log10(abs(t)/abs(t1)),
                10*np.log10((1-abs(r)**2-abs(t)**2)/(1-abs(r1)**2-abs(t1)**2))]
    got = half_sheet_film_error(rs, dx, 10e9)
    assert all(np.asarray(v).dtype == np.float64 for v in got)
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0)


@pytest.mark.parametrize('rs,dx,t,a', [(100, .0015, -.68, .68),
                                      (30, .00075, -1.20, .96),
                                      (10, .000375, -2.06, 1.39)])
def test_t1_measured_literals(rs, dx, t, a):
    _, transmitted, absorbed = half_sheet_film_error(rs, dx, 10e9)
    assert transmitted == pytest.approx(t, abs=.15)
    assert absorbed == pytest.approx(a, abs=.05)


def test_t2_box_fires_with_exact_message():
    found = findings(model())
    assert len(found) == 1
    issue = found[0]
    assert (issue.code, issue.severity, issue.source, issue.loc) == (
        CODE, 'warning', 'warn_dc_films', 'thin_conductors[0]')
    assert str(issue) == (
        'lossy film of 10 ohm/sq is realized as two half-sheets one cell '
        '(0.0015 m) apart: at freq_max = 10 GHz, (k0 dx)(eta0/Rs) = 12, '
        'transmitted power is about -10.2 dB and absorbed power about +2.8 dB '
        'off the declared sheet at normal incidence; reflected power stays '
        'within 1 dB. Use cells of at most 0.00038 m across the film or check '
        'on two meshes (tracker 1572).')


@pytest.mark.parametrize('rs,count', [(377, 0),
    (2*np.pi*10e9*.0015/299792458.0*376.730313668/2.9, 0),
    (2*np.pi*10e9*.0015/299792458.0*376.730313668/3.1, 1)])
def test_t3_threshold(rs, count):
    assert len(findings(model(rs))) == count


@pytest.mark.parametrize('kwargs', [{'sigma': 5.8e7}, {'f0': 10e9}])
def test_t3_other_sheet_branches_silent(kwargs):
    assert findings(model(**kwargs)) == []


@pytest.mark.parametrize('z,count', [(.009, 0), (.0045, 1)])
def test_t4_occupied_graded_cell(z, count):
    profile = np.array([.0015]*6 + [.000375] + [.0015]*5)
    sim = model(profile=profile, z=z)
    found = findings(sim)
    assert len(found) == count
    if count:
        assert '(0.0015 m)' in str(found[0])
        assert '(k0 dx)(eta0/Rs) = 12,' in str(found[0])


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('pos,count', [(.009, 0), (.0045, 1)])
def test_t4_normal_axis_x_and_y(axis, pos, count):
    """The occupied width is read along the film's own normal, not along z."""
    profile = np.array([.0015]*6 + [.000375] + [.0015]*5)
    lo, hi = [.003]*3, [.015]*3
    lo[axis] = hi[axis] = pos
    kw = {('dx_profile', 'dy_profile')[axis]: profile}
    sim = Simulation(10e9, (0 if axis == 0 else .018, 0 if axis == 1 else .018, .018),
                     dx=.0015, boundary='pec', **kw)
    sim.add_thin_conductor(Box(tuple(lo), tuple(hi)), sigma_bulk=1/(10*35e-6), thickness=35e-6)
    found = findings(sim)
    assert len(found) == count
    if count:
        assert '(0.0015 m)' in str(found[0])


@pytest.mark.parametrize('sigma', [1e-14, 1e-18])
def test_near_insulating_film_adds_no_finding(sigma):
    """0/0 in the dB arithmetic must not leak a numpy warning into the report."""
    report = model(sigma=sigma).preflight()
    assert report.by_code(CODE) == []
    assert not [f for f in report if 'invalid value' in str(f)]


def test_t5_nonbox_and_existing_admission_findings():
    from tests.contracts.test_dc_film_admission import model as disc_model

    for radius in (10.5, 2.5):
        sim, _, _ = disc_model(radius=radius, centre=(24, 24), snap='declared')
        before = sim.preflight().by_code('dc_film_area')
        tc = sim._thin_conductors[0]
        sim._thin_conductors[0] = replace(tc, sigma_bulk=1/(10*tc.thickness))
        report = sim.preflight()
        assert len(report.by_code(CODE)) == 1
        assert [f.to_dict() for f in report.by_code('dc_film_area')] == [
            f.to_dict() for f in before]
        assert len(before) == int(radius == 2.5)


def capture_preflight(monkeypatch):
    reports = []
    original = Simulation._preflight_impl

    def capture(self, *args, **kwargs):
        report = original(self, *args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(Simulation, '_preflight_impl', capture)
    return reports


def test_t6_preflight_run_forward_once(monkeypatch):
    reports = capture_preflight(monkeypatch)
    for entry in ('preflight', 'run', 'forward'):
        sim = model()
        if entry == 'preflight':
            sim.preflight()
        elif entry == 'run':
            result = sim.run(n_steps=20, compute_s_params=False)
            assert np.isfinite(np.asarray(result.time_series)).all()
        else:
            result = sim.forward(n_steps=20)
            assert np.isfinite(np.asarray(result.time_series)).all()
        assert len(reports) == 1
        assert len(reports.pop().by_code(CODE)) == 1


@pytest.mark.parametrize('parameter', ['elsewhere', 'sigma_bulk', 'thickness'])
def test_t7_forward_grad(monkeypatch, parameter):
    reports = capture_preflight(monkeypatch)
    sim = model()
    tc = sim._thin_conductors[0]

    def loss(value):
        kwargs = {}
        if parameter == 'elsewhere':
            grid = sim._build_realized_grid()
            sigma = jnp.zeros(grid.shape).at[2:4, 2:4, 2:4].set(value)
            kwargs['sigma_override'] = sigma
        else:
            sim._thin_conductors[0] = replace(tc, **{parameter: value})
        return jnp.sum(sim.forward(n_steps=20, **kwargs).time_series**2)

    value = .01 if parameter == 'elsewhere' else getattr(tc, parameter)
    try:
        derivative = jax.grad(loss)(value)
    finally:
        sim._thin_conductors[0] = tc
    assert np.isfinite(float(derivative))
    assert len(reports) == 1
    assert len(reports[0].by_code(CODE)) == int(parameter == 'elsewhere')


def test_t7_traced_mesh_skips_estimate():
    def check(profile):
        sim = model(profile=profile, domain_z=.018)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warn_dc_films(sim, warnings)
        assert not [w for w in caught if getattr(w.message, 'code', None) == CODE]
        return jnp.sum(profile)

    assert np.isfinite(np.asarray(jax.jit(jax.grad(check))(jnp.full(12, .0015)))).all()


def test_two_declarations_each_emit_once():
    sim = model()
    sim._thin_conductors.append(sim._thin_conductors[0])
    found = findings(sim)
    assert len(found) == 2
    assert {f.loc for f in found} == {'thin_conductors[0]', 'thin_conductors[1]'}


@pytest.mark.parametrize('parameter', ['sigma_bulk', 'thickness'])
def test_zero_conductance_has_no_accuracy_finding(parameter):
    sim = model()
    sim._thin_conductors[0] = replace(sim._thin_conductors[0], **{parameter: 0.0})
    assert findings(sim) == []
