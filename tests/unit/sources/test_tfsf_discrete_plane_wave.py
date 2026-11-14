"""O1a declaration, independent commutator and bounded-line contracts."""
from dataclasses import replace
import itertools
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.core.yee import EPS_0, MU_0, FDTDState, init_state
from rfx.sources import tfsf_discrete_plane_wave as plane
from rfx.sources.sources import CustomWaveform
from rfx.measurement.incident import measure_normal_incident_spectrum

DIRECTIONS = ((0, 0), (10, 0), (20, 0), (30, 0), (40, 0), (50, 0),
              (60, 0), (30, 90), (45, 30), (80, 0))
TRIPLES = ((1, 0, 0), (17, 3, 0), (11, 4, 0), (26, 15, 0), (31, 26, 0),
           (26, 31, 0), (15, 26, 0), (26, 0, 15), (30, 26, 15), (3, 17, 0))
COMP = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
# Independent Yee coordinates: do not import the implementation's offsets.
OFFSETS = np.array(((.5, 0, 0), (0, .5, 0), (0, 0, .5),
                    (0, .5, .5), (.5, 0, .5), (.5, .5, 0)))
DX, DT, Z0 = .001, 1.9065748695310057e-12, 376.730313668


def scene(theta=0, phi=0, pol='s', direction='+x', *, cpml_layers=8):
    sim = Simulation(freq_max=15e9, domain=(.024,) * 3, dx=DX,
                     boundary='cpml', cpml_layers=cpml_layers)
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, margin=4, closed_box=True,
                        theta_deg=theta, phi_deg=phi, polarization=pol,
                        max_component=40, direction=direction)
    return sim


def line(theta=0, phi=0, pol='s', n_steps=300, **kwargs):
    return plane.init_plane_wave(41, DX, DT, ny=41, nz=41, cpml_layers=8,
        tfsf_margin=4, f0=10e9, bandwidth=.8, theta_deg=theta, phi_deg=phi,
        polarization=pol, max_component=40, n_steps=n_steps, **kwargs)


@pytest.mark.parametrize('pol', ('ey', 'ez'))
@pytest.mark.parametrize('direction', ('+x', '-x'))
@pytest.mark.parametrize('reference_index', (None, 20))
def test_appendix_h_incident_record_and_shared_dft(pol, direction, reference_index):
    """Appendix H: the prior scan is bit-identical; both DFT clocks meet 1e-4."""
    from rfx.stepping.incident_line import record_incident_line

    cfg, state = line(pol=pol, direction=direction)
    x = cfg.x_lo if reference_index is None else reference_index
    indices = tuple(int(np.floor(cfg.m[0] * x + cfg.projections[c])) - cfg.origin
                    for c in range(3))
    # Frozen pre-Appendix-H replay body, including its zero-state construction.
    zero = FDTDState(*(jnp.zeros_like(getattr(state, c)) for c in COMP),
                     jnp.asarray(0, jnp.int32))
    def prior_step(st, n):
        st = plane.update_line_h(cfg, st, cfg.dx_1d, DT)
        st = plane.update_line_e(cfg, st, cfg.dx_1d, DT)
        value = sum(cfg.incidence['e_hat'][c] * getattr(st, COMP[c])[indices[c]]
                    for c in range(3))
        return st, value
    _, prior = jax.lax.scan(prior_step, zero, jnp.arange(300, dtype=jnp.int32))
    record = np.asarray(record_incident_line(cfg, state, 300, DT, indices, cfg.incidence['e_hat']))
    assert record.dtype == np.float32
    np.testing.assert_array_equal(record.view(np.uint32), np.asarray(prior).view(np.uint32))
    frequencies = np.linspace(8e9, 12e9, 9)
    times = (np.arange(300) + (reference_index is not None)) * DT
    expected = np.exp(-2j * np.pi * frequencies[:, None] * times) @ record.astype(np.float64) * DT
    actual = measure_normal_incident_spectrum(
        cfg, state, 300, frequencies, DT, reference_index=reference_index)
    relative = np.abs(actual - expected) / np.max(np.abs(expected))
    assert np.all(relative <= 1e-4), relative


@pytest.mark.parametrize('angles,m', tuple(zip(DIRECTIONS, TRIPLES)))
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_d1_d2_direction_and_basis(angles, m, pol):
    info = scene(*angles, pol).tfsf_incidence()
    assert info['m'] == m
    assert info['angle_error_deg'] <= .1
    k, e, h = (info[key] for key in ('k_hat', 'e_hat', 'h_hat'))
    assert all(v.dtype == np.float64 for v in (k, e, h))
    np.testing.assert_array_equal(k, np.array(m) / np.linalg.norm(m))
    np.testing.assert_allclose(np.stack((k, e, h)) @ np.stack((k, e, h)).T,
                               np.eye(3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.cross(k, e), h, rtol=0, atol=1e-12)


def test_d1_tolerance_refuses():
    with pytest.raises(ValueError, match='tolerance|direction_tol'):
        plane.incidence(45, 30, '+x', 's', .001, 40)


@pytest.mark.parametrize('bound', (None, 0, -1, np.inf, True))
def test_d3_requires_record_bound(bound):
    with pytest.raises(ValueError, match='step count'):
        line(n_steps=bound)


@pytest.mark.parametrize('angles', DIRECTIONS)
def test_d3_clean_margin(angles):
    cfg, _ = line(*angles)
    reads = np.concatenate([t.source_index for t in cfg.tables_e + cfg.tables_h]
                           + [np.array(cfg.source_nodes)]
                           + [w.nodes for w in cfg.source_windows])
    g = 2 * max(map(abs, cfg.m)) + 2
    length = 64 * int(np.ceil(np.sqrt(sum(v * v for v in cfg.m))))
    assert cfg.absorber_nodes == length
    assert reads.min() - length >= g
    assert cfg.n_nodes - length - 1 - reads.max() >= g
    assert cfg.n_nodes == reads.max() - reads.min() + 1 + 2 * (g + length)


def dense_curl(fields, forward):
    """Float64 full-grid Yee curl, independent of line/table helpers."""
    fields = np.asarray(fields, dtype=np.float64)
    def delta(value, axis):
        other = np.roll(value, -1 if forward else 1, axis=axis)
        edge = [slice(None)] * 3
        edge[axis] = -1 if forward else 0
        other[tuple(edge)] = 0
        return other - value if forward else value - other
    x, y, z = fields
    return np.stack((delta(z, 1) - delta(y, 2), delta(x, 2) - delta(z, 0),
                     delta(y, 0) - delta(x, 1)))


def commutator_errors(angles, pol, mutate=False):
    cfg, state = line(*angles, pol)
    rng = np.random.default_rng(1662)
    arrays = [rng.normal(size=cfg.n_nodes).astype(np.float32) for _ in COMP]
    state = FDTDState(*map(jnp.asarray, arrays), state.step)
    nodes = np.moveaxis(np.indices(cfg.shape), 0, -1)
    fields = np.stack([arrays[c][np.floor((nodes + OFFSETS[c]) @ cfg.m).astype(int) - cfg.origin]
                       for c in range(6)]).astype(np.float64)
    mask = np.zeros(cfg.shape, np.float64)
    mask[cfg.x_lo:cfg.x_hi + 1, cfg.y_lo:cfg.y_hi + 1, cfg.z_lo:cfg.z_hi + 1] = 1
    expected = np.concatenate((DT / (EPS_0 * DX) * (
        mask * dense_curl(fields[3:], False) - dense_curl(mask * fields[3:], False)),
        -DT / (MU_0 * DX) * (
        mask * dense_curl(fields[:3], True) - dense_curl(mask * fields[:3], True))))
    if mutate:
        table = cfg.tables_e[0]
        indices = table.source_index.copy()
        indices[0] += 1
        cfg = replace(cfg, tables_e=(table._replace(source_index=indices),) + cfg.tables_e[1:])
    actual = plane.apply_tables(init_state(cfg.shape), cfg, state, magnetic=True)
    actual = plane.apply_tables(actual, cfg, state, magnetic=False)
    return np.array([np.max(np.abs(np.asarray(getattr(actual, c), np.float64) - expected[i]))
                     / np.spacing(np.float32(np.max(np.abs(expected[i]))))
                     for i, c in enumerate(COMP)])


@pytest.mark.parametrize('angles', DIRECTIONS)
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_v0_independent_curl(angles, pol):
    assert np.max(commutator_errors(angles, pol)) <= 9


def test_normal_reference_refuses_oblique_o1b():
    cfg, state = line(45, 30)
    with pytest.raises(NotImplementedError, match='O1b'):
        measure_normal_incident_spectrum(cfg, state, 300, [10e9], DT)


@pytest.mark.parametrize('kind', ('graded', 'subgrid',  '2d', 'order4', 'face', 'custom'))
def test_d4_declaration_refusals(kind):
    options = dict(freq_max=15e9, domain=(.024,) * 3, dx=DX,
                   boundary='cpml', cpml_layers=8)
    if kind == 'graded':
        options['dz_profile'] = [DX] * 24
    elif kind == '2d':
        options.update(mode='2d_tmz', domain=(.024, .024, DX))
    elif kind == 'order4':
        options['stencil_order'] = 4
    elif kind == 'face':
        from rfx.boundaries.spec import BoundarySpec
        options['boundary'] = BoundarySpec(x='cpml', y='cpml', z='pec')
    sim = Simulation(**options)
    if kind == 'subgrid':
        sim.add_refinement(z_range=(.008, .016), ratio=2)
    extra = dict(theta_deg=30, polarization='p', waveform=CustomWaveform(lambda t: t)) if kind == 'custom' else {}
    with pytest.raises(NotImplementedError, match='closed_box|CustomWaveform'):
        sim.add_tfsf_source(closed_box=True, margin=4, **extra)


@pytest.mark.parametrize('entry', ('run', 'forward'))
def test_d4_oblique_multidevice_refused_at_execution_declaration(entry):
    sim = scene(theta=30)
    devices = jax.devices('cpu')[:2]
    if len(devices) < 2:
        pytest.skip("requires two host devices")
    kwargs = dict(devices=devices)
    if entry == 'forward':
        kwargs['distributed'] = True
    with pytest.raises(NotImplementedError, match='oblique closed-box TFSF'):
        getattr(sim, entry)(n_steps=2, skip_preflight=True, **kwargs)


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('side', ('lo', 'hi'))
@pytest.mark.parametrize('offset', (-1, 0))
def test_d4_shell_and_stencil_vacuum(axis, side, offset):
    from rfx.geometry.csg import Box
    sim = Simulation(freq_max=15e9, domain=(.024,) * 3, dx=DX,
                     boundary='cpml', cpml_layers=8)
    sim.add_material('dielectric', eps_r=2.)
    low = np.array([.011, .011, .011])
    low[axis] = (.004 if side == 'lo' else .020) + (offset if side == 'lo' else -offset) * DX
    high = low + DX
    sim.add(Box(tuple(low), tuple(high)), material='dielectric')
    with pytest.raises(ValueError, match='vacuum.*all six faces'):
        sim.add_tfsf_source(closed_box=True, margin=4)


def resized_line(cfg, state, factor):
    """Restore the historical padded lossless line for the origin-only check."""
    reads = np.concatenate([t.source_index + cfg.origin for t in cfg.tables_e + cfg.tables_h]
                           + [np.array(cfg.source_nodes) + cfg.origin]
                           + [w.nodes + cfg.origin for w in cfg.source_windows])
    padding = int((cfg.n_steps * max(map(abs, cfg.m)) + 2) * factor)
    origin = int(reads.min()) - padding
    count = int(reads.max() - reads.min() + 1 + 2 * padding)
    shift = cfg.origin - origin
    def tables(old):
        return tuple(t._replace(source_index=t.source_index + np.int32(shift)) for t in old)
    resized = replace(cfg, origin=origin, n_nodes=count, absorber_nodes=0,
        source_nodes=tuple(n + shift for n in cfg.source_nodes),
        source_windows=tuple(w._replace(nodes=w.nodes + np.int32(shift)) for w in cfg.source_windows),
        tables_e=tables(cfg.tables_e), tables_h=tables(cfg.tables_h))
    return resized, FDTDState(*(jnp.zeros(count, a.dtype) for a in state[:6]), state.step)


def center_record(cfg, state, n_steps):
    """Six sublattices sampled nearest the same box-centre projected p."""
    center_p = np.dot(cfg.m, [(getattr(cfg, a + '_lo') + getattr(cfg, a + '_hi')) / 2 for a in 'xyz'])
    indices = np.rint(center_p - np.array(cfg.fractions)).astype(int) - cfg.origin
    def step(st, n):
        st = plane.update_line_e(cfg, plane.update_line_h(cfg, st, DX, DT), DX, DT)
        return st, jnp.stack([getattr(st, c)[i] for c, i in zip(COMP, indices)])
    _, record = jax.lax.scan(step, state, jnp.arange(n_steps, dtype=jnp.int32))
    return np.asarray(record)


@pytest.mark.parametrize('angles', DIRECTIONS)
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_d3b_bound_vs_twice_bound(angles, pol):
    cfg, state = line(*angles, pol)
    cfg, state = resized_line(cfg, state, 1)
    twice, zero = resized_line(cfg, state, 2)
    np.testing.assert_array_equal(center_record(cfg, state, 300).view(np.uint32),
                                  center_record(twice, zero, 300).view(np.uint32))


def polarization_record(theta, phi, pol, delayed=False):
    """V11: nearest sublattice samples at one projected box-centre location."""
    from scipy.optimize import brentq
    cfg, state = line(theta, phi, pol)
    if delayed:
        delays = list(cfg.delays)
        delays[1] += 1
        windows = list(cfg.source_windows)
        if windows:
            windows[1] = windows[1]._replace(delays=windows[1].delays + 1)
        cfg = replace(cfg, delays=tuple(delays), source_windows=tuple(windows))
    record = center_record(cfg, state, 300)[:, :3]
    frequencies = np.linspace(8e9, 12e9, 9)
    spectrum = np.exp(-2j * np.pi * frequencies[:, None] * ((np.arange(300) + 1) * DT)) @ record * DT
    e = cfg.incidence['e_hat']
    other = plane.incidence(theta, phi, '+x', 'p' if pol == 's' else 's', .1, 40)['e_hat']
    co, cross = spectrum @ e, spectrum @ other
    ratio = np.abs(cross) / np.abs(co)
    prediction = []
    courant = DT / (DX * np.sqrt(EPS_0 * MU_0))
    m = np.array(cfg.m)
    for frequency in frequencies:
        rhs = (np.sin(np.pi * frequency * DT) / courant) ** 2
        q = brentq(lambda q: np.sum(np.sin(q * m / 2) ** 2) - rhs,
                   0, np.pi / np.max(np.abs(m)))
        discrete_k = np.sin(q * m / 2)
        discrete_k /= np.linalg.norm(discrete_k)
        modal_e = e - discrete_k * np.dot(discrete_k, e)
        prediction.append(abs(other @ modal_e) / abs(e @ modal_e))
    return dict(theta=theta, phi=phi, pol=pol, delayed=delayed,
        frequencies=frequencies.tolist(), cross_co=ratio.tolist(),
        longitudinal_co=(np.abs(spectrum @ cfg.incidence['k_hat']) / np.abs(co)).tolist(),
        anisotropy_estimate=prediction,
        prediction_scope='estimate: principal propagating mode, co-located components, transverse discrete-k projection; finite-record launch and sampling excluded')


@pytest.mark.parametrize('angles', DIRECTIONS)
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_v11_polarization(angles, pol, record_property):
    result = polarization_record(*angles, pol)
    record_property('polarization', str(result))
    assert max(result['cross_co']) <= .01, result


def _probe_scene(theta, phi, pol, cpml_layers):
    """Read the centre, six exterior face centres and eight exterior corners."""
    sim = scene(theta, phi, pol, cpml_layers=cpml_layers)
    bounds = sim.tfsf_box_indices()
    centre = tuple(sum(bounds[a]) // 2 for a in 'xyz')
    points = [centre]
    for axis, name in enumerate('xyz'):
        for index in (bounds[name][0] - 1, bounds[name][1] + 1):
            point = list(centre)
            point[axis] = index
            points.append(tuple(point))
    points.extend(itertools.product(*[(bounds[a][0] - 1, bounds[a][1] + 1)
                                      for a in 'xyz']))
    component = 'ez' if pol == 's' else 'ey'
    positions = [tuple((i - cpml_layers) * DX for i in p) for p in points]
    for position in positions:
        sim.add_probe(position, component=component)
    result = sim.run(n_steps=300, skip_preflight=True)
    grid = result.grid
    # Quote the realized mesh, and verify that the public probes hit our nodes.
    assert grid.shape == (25 + 2 * cpml_layers,) * 3
    assert [grid.position_to_index(p) for p in positions] == points
    cfg, state = plane.init_plane_wave(
        grid.nx, float(grid.cells('x')[0]), grid.dt, ny=grid.ny, nz=grid.nz,
        cpml_layers=cpml_layers, tfsf_margin=4, f0=10e9, bandwidth=.8,
        theta_deg=theta, phi_deg=phi, polarization=pol, max_component=40,
        n_steps=300)
    # Independent Yee coordinates, not the correction table's gathers.
    offset = OFFSETS[COMP.index(component)]
    index = int(np.floor((np.asarray(centre) + offset) @ cfg.m)) - cfg.origin

    def step(st, _):
        st = plane.update_line_h(cfg, st, cfg.dx_1d, grid.dt)
        st = plane.update_line_e(cfg, st, cfg.dx_1d, grid.dt)
        electric = jnp.max(jnp.stack([jnp.max(jnp.abs(getattr(st, c))) for c in COMP[:3]]))
        magnetic = jnp.max(jnp.stack([jnp.max(jnp.abs(getattr(st, c))) for c in COMP[3:]]))
        return st, (getattr(st, component)[index], electric, magnetic)

    _, (reference, electric, magnetic) = jax.lax.scan(step, state, None, length=300)
    line_peak = np.maximum(np.asarray(electric, np.float64),
                           Z0 * np.asarray(magnetic, np.float64))
    return (np.asarray(result.time_series, np.float64),
            np.asarray(reference, np.float64), line_peak, grid)


def _assert_probe_law(trace, reference, line_peak, grid, row, record_property):
    centre = np.abs(trace[:, 0])
    peak = centre.max()
    # Design A33: the incident line drives the faces before the centre sees it.
    # Use its per-step E / Z0*H peak for the rise and accumulated leakage law.
    scale = np.maximum(line_peak, centre)
    cumulative = scale.cumsum()
    active = cumulative >= scale.max()
    sf = np.maximum.accumulate(np.max(np.abs(trace[:, 1:]), axis=1))
    # The existing 3D/stand-alone-line bar is 9 float32 peak ULP:
    # tests/unit/sources/test_tfsf_closed_box.py:28-31,143.
    centre_bar = 9 * float(np.spacing(np.float32(peak)))
    early_bar = 9 * np.spacing(scale.astype(np.float32)).astype(np.float64)
    bar = np.where(active, 1.2e-7 * cumulative, early_bar)
    error = np.abs(trace[:, 0] - reference)
    ratios = np.divide(sf, bar, out=np.full_like(sf, np.inf), where=bar > 0)
    worst = int(np.argmax(ratios))
    summary = dict(row=row, steps=300, mesh=grid.shape, dt=float(grid.dt),
                   peak=float(peak), switch_step=int(np.argmax(active) + 1),
                   sf_max=float(sf.max()), sf_law_ratio=float(ratios[active].max(initial=0)),
                   sf_early_ulp=float((sf[~active] / (early_bar[~active] / 9)).max(initial=0)),
                   centre_error=float(error.max()), centre_bar=centre_bar,
                   centre_peak_ulp=float(error.max() / (centre_bar / 9)),
                   worst_step=worst + 1, scale=float(scale[worst]),
                   sf_ratio=float(ratios[worst]), expected_sf=0.,
                   measured_sf=float(sf[worst]), sf_bar=float(bar[worst]))
    record_property('measurements', str(summary))
    print(summary, file=sys.stderr)
    assert np.isfinite(trace).all() and np.isfinite(reference).all() and np.isfinite(scale).all(), f'nonfinite field record: {summary}'
    assert peak > .1, summary
    assert np.all(sf <= bar), f'scattered-field leakage exceeds incident-scale law: {summary}'
    assert np.all(error <= centre_bar), f'centre differs from stand-alone line by >9 peak ULP: {summary}'


@pytest.mark.parametrize('theta,phi,pol', ((0, 0, 's'), (0, 0, 'p'), (30, 0, 'p'), (45, 30, 's')))
def test_v1_v2_l1(theta, phi, pol, record_property):
    """Public 3D probes cancel outside the closed box and replay its incident line."""
    trace, reference, line_peak, grid = _probe_scene(theta, phi, pol, cpml_layers=8)
    _assert_probe_law(trace, reference, line_peak, grid,
                      (theta, phi, pol, 8), record_property)


def principal_roots(m, frequencies):
    """Independent NumPy float64 principal root; no engine dispersion helper."""
    m = np.asarray(m, np.float64)
    courant = DT / (DX * np.sqrt(EPS_0 * MU_0))
    roots = []
    for frequency in frequencies:
        target = (np.sin(np.pi * frequency * DT) / courant) ** 2
        low, high = 0., np.pi / np.max(np.abs(m))
        assert np.sum(np.sin(high * m / 2) ** 2) > target
        for _ in range(100):
            mid = (low + high) / 2
            if mid in (low, high):
                break
            if np.sum(np.sin(mid * m / 2) ** 2) < target:
                low = mid
            else:
                high = mid
        roots.append((low + high) / 2)
    return np.array(roots)


def co_spectrum(cfg, state):
    """The Appendix-C p_src anchor for the native line's centre samples."""
    frequencies = np.linspace(8e9, 12e9, 9)
    trace = center_record(cfg, state, 300)[:, :3].astype(np.float64)
    spectrum = np.exp(-2j * np.pi * frequencies[:, None] * ((np.arange(300) + 1) * DT)) @ trace * DT
    p = np.dot(cfg.m, [20, 20, 20])
    sampled = np.rint(p - np.array(cfg.fractions[:3])) + cfg.fractions[:3]
    q = principal_roots(cfg.m, frequencies)
    correction = np.exp(1j * q[:, None] * (sampled - cfg.source_p))
    return (spectrum * correction) @ cfg.incidence['e_hat'], q


@pytest.mark.parametrize('angles', ((10, 0), (20, 0), (45, 30), (80, 0)))
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_v13a_clean_point_reference(angles, pol, record_property):
    """Appendix F: point references only where the alias cutoff is >=15 GHz."""
    cfg, state = line(*angles, pol)
    window, _ = co_spectrum(cfg, state)
    point, _ = co_spectrum(replace(cfg, source_windows=()), state)
    ratio = np.abs(window) / np.abs(point)
    record_property('window_over_point', ratio.tolist())
    assert np.all((ratio >= .80) & (ratio <= 1.05)), ratio


def window_width_ratio(angles, pol, zero_delays=False):
    info = plane.incidence(*angles, '+x', pol, .1, 40)
    q1, sigma = plane.launch_window_width(info['m'])
    native_windows = plane.launch_windows
    def windows(*args, **kwargs):
        result = native_windows(*args, **kwargs)
        if zero_delays:
            result = tuple(w._replace(delays=np.zeros_like(w.delays)) for w in result)
        return result
    co = []
    for multiplier in (1, 2):
        with patch.object(plane, 'launch_windows', windows), patch.object(
                plane, 'launch_window_width', return_value=(q1, multiplier * sigma)):
            cfg, state = line(*angles, pol)
        spectrum, q = co_spectrum(cfg, state)
        co.append(spectrum)
    v0 = np.linalg.norm(cfg.m) * 299792458. * DT / DX
    dq = q - 2 * np.pi * np.linspace(8e9, 12e9, 9) * DT / v0
    return np.abs(co[1]) / np.abs(co[0]), np.exp(-1.5 * (dq * sigma) ** 2)


@pytest.mark.parametrize('angles', DIRECTIONS[1:])
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_v13b_matched_window_width(angles, pol, record_property):
    """Appendix G: forward delays filter at dq=q-omega*dt/v0, not q."""
    ratio, expected = window_width_ratio(angles, pol)
    relative = np.abs(ratio / expected - 1)
    record_property('relative_error', relative.tolist())
    assert np.all(relative <= .02), (ratio, expected, relative)


@pytest.mark.parametrize('parameter,value', (('amplitude', 1.), ('f0', 10e9), ('bandwidth', .8)))
def test_appendix_i_forward_refuses_traced_source(parameter, value):
    """Source scalars must raise, rather than silently return a zero gradient."""
    sim = scene()
    sim.add_probe((.012,) * 3, component='ez')
    entry = sim._tfsf
    def objective(traced):
        sim._tfsf = replace(entry, **{parameter: traced})
        return jnp.sum(sim.forward(n_steps=2, skip_preflight=True).time_series)
    with pytest.raises(ValueError, match=(
            parameter + ': differentiating with respect to the incident source is not supported')):
        jax.grad(objective)(jnp.float32(value))


@pytest.mark.parametrize('pol', ('ey', 'ez'))
def test_appendix_i_normal_cartesian_phi_refused(pol):
    with pytest.raises(ValueError, match='nonzero phi_deg at theta_deg=0'):
        scene(0, 30, pol)


@pytest.mark.parametrize('theta', (-1, 90, 100))
def test_appendix_i_theta_half_space_refused(theta):
    with pytest.raises(ValueError, match='theta_deg must be in'):
        scene(theta, 0, 's')


def test_appendix_i_decay_requires_step_bound():
    with pytest.raises(ValueError, match='closed_box requires a finite positive integer bound on its step count'):
        scene().run(until_decay=1e-5, decay_max_steps=None, skip_preflight=True)


@pytest.mark.parametrize('angles', DIRECTIONS)
@pytest.mark.parametrize('pol', ('s', 'p'))
def test_v14_absolute_incident_amplitude(angles, pol, record_property):
    """Appendix I: absolute co spectrum, and omitted |m| normalization red."""
    normal, zero = line(pol=pol)
    reference, _ = co_spectrum(normal, zero)
    cfg, state = line(*angles, pol)
    actual, _ = co_spectrum(cfg, state)
    ratio = np.abs(actual) / np.abs(reference)
    record_property('ratio', ratio.tolist())
    assert np.all(np.abs(ratio - 1) <= .01), ratio
    norm = np.linalg.norm(cfg.m)
    if norm != 1:
        defective, _ = co_spectrum(replace(cfg, src_amp=cfg.src_amp / norm), state)
        defect_ratio = np.abs(defective) / np.abs(reference)
        record_property('omitted_factor_ratio', defect_ratio.tolist())
        assert np.all(np.abs(defect_ratio - 1) > .01), defect_ratio


@pytest.mark.parametrize('pol', ('s', 'p'))
def test_v14_max_component_pair(pol, record_property):
    """The same 30-degree request must agree for integer-search caps 40/100."""
    spectra = []
    for cap in (40, 100):
        cfg, state = plane.init_plane_wave(41, DX, DT, ny=41, nz=41, cpml_layers=8,
            tfsf_margin=4, f0=10e9, bandwidth=.8, theta_deg=30, phi_deg=0,
            polarization=pol, max_component=cap, n_steps=300)
        spectra.append(co_spectrum(cfg, state)[0])
    ratio = np.abs(spectra[1]) / np.abs(spectra[0])
    record_property('ratio_100_over_40', ratio.tolist())
    assert np.all(np.abs(ratio - 1) <= .01), ratio
