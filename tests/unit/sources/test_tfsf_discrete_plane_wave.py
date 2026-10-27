"""O1a declaration, independent commutator and bounded-line contracts."""
from dataclasses import replace
import inspect
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


def scene(theta=0, phi=0, pol='s', direction='+x'):
    sim = Simulation(freq_max=15e9, domain=(.024,) * 3, dx=DX,
                     boundary='cpml', cpml_layers=8)
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
    record = np.asarray(record_incident_line(cfg, 300, DT, indices, cfg.incidence['e_hat']))
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
def test_d3_domain_of_dependence(angles):
    cfg, _ = line(*angles)
    reads = np.concatenate([t.source_index for t in cfg.tables_e + cfg.tables_h]
                           + [np.array(cfg.source_nodes)]
                           + [w.nodes for w in cfg.source_windows])
    padding = cfg.n_steps * max(map(abs, cfg.m)) + 2
    assert reads.min() >= padding
    assert cfg.n_nodes - 1 - reads.max() >= padding


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


def test_v0_face_index_mutation_is_red():
    assert np.max(commutator_errors((45, 30), 's', mutate=True)) > 9


def test_normal_reference_refuses_oblique_o1b():
    cfg, state = line(45, 30)
    with pytest.raises(NotImplementedError, match='O1b'):
        measure_normal_incident_spectrum(cfg, state, 300, [10e9], DT)


@pytest.mark.parametrize('kind', ('graded', 'subgrid', 'adi', '2d', 'order4', 'face', 'custom'))
def test_d4_declaration_refusals(kind):
    options = dict(freq_max=15e9, domain=(.024,) * 3, dx=DX,
                   boundary='cpml', cpml_layers=8)
    if kind == 'graded':
        options['dz_profile'] = [DX] * 24
    elif kind == 'adi':
        options.update(solver='adi', boundary='pec')
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
    """Keep launch/tables fixed in physical p; change only the end distance."""
    reads = np.concatenate([t.source_index + cfg.origin for t in cfg.tables_e + cfg.tables_h]
                           + [np.array(cfg.source_nodes) + cfg.origin]
                           + [w.nodes + cfg.origin for w in cfg.source_windows])
    padding = int((cfg.n_steps * max(map(abs, cfg.m)) + 2) * factor)
    origin = int(reads.min()) - padding
    count = int(reads.max() - reads.min() + 1 + 2 * padding)
    shift = cfg.origin - origin
    def tables(old):
        return tuple(t._replace(source_index=t.source_index + np.int32(shift)) for t in old)
    resized = replace(cfg, origin=origin, n_nodes=count,
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
    twice, zero = resized_line(cfg, state, 2)
    np.testing.assert_array_equal(center_record(cfg, state, 300).view(np.uint32),
                                  center_record(twice, zero, 300).view(np.uint32))


def test_d3b_quarter_length_is_red(tmp_path, record_property):
    import json
    cfg, state = line(n_steps=1200)
    small, zero = resized_line(cfg, state, .25)
    center_p = 20
    ends = (small.origin - 1, small.origin + small.n_nodes)
    path = min(abs(cfg.source_p - end) + abs(center_p - end) for end in ends)
    return_step = int(np.ceil(path / (299792458. * DT / DX))) + 1
    assert return_step < 1200
    plan = dict(return_step=return_step, zero_extension_nodes=ends, path_p=path)
    (tmp_path / 'quarter_return_plan.json').write_text(json.dumps(plan))
    record_property('return_plan', str(plan))
    actual = center_record(small, zero, 1200)
    expected = center_record(cfg, state, 1200)
    assert np.any(actual[return_step - 1:].view(np.uint32) != expected[return_step - 1:].view(np.uint32))


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


def test_v11_one_step_delay_is_red(record_property):
    result = polarization_record(45, 30, 'p', delayed=True)
    record_property('mutation', str(result))
    assert max(result['cross_co']) > .01, result


def measure_scene(theta, phi, pol, n_steps=300, direction='+x', _line_peaks=None):
    """Step-0 protocol: capture the real setup/core, inspect all fields in 20-step scans."""
    if _line_peaks is None:
        peaks = {}
        def tap(step, electric, magnetic):
            def save(n, e, h):
                peaks[int(n)] = (float(np.max(np.abs(np.asarray(e, np.float64)))),
                                 float(np.max(np.abs(np.asarray(h, np.float64)))) * Z0)
            jax.debug.callback(save, step, electric, magnetic, ordered=True)
        source = inspect.getsource(plane.update_line_e)
        needle = '    return tree_util.tree_map(lax.stop_gradient, state._replace(**changes, step=state.step + 1))'
        assert source.count(needle) == 1
        source = source.replace(needle,
            '    _tap(state.step + 1, jnp.stack([changes[c] for c in COMPONENTS[:3]]), '
            'jnp.stack([getattr(state, c) for c in COMPONENTS[3:]]))\n' + needle)
        namespace = dict(vars(plane), _tap=tap)
        exec(compile(source, '<actual line-return tap>', 'exec'), namespace)
        with patch.object(plane, 'update_line_e', namespace['update_line_e']):
            return measure_scene(theta, phi, pol, n_steps, direction, peaks)
    import rfx.stepping.uniform_loop as engine
    sim = scene(theta, phi, pol, direction)
    grid = sim._build_grid()
    bounds = sim.tfsf_box_indices()
    lo, hi = np.array([bounds[a][0] for a in 'xyz']), np.array([bounds[a][1] for a in 'xyz'])
    center = tuple((lo + hi) // 2)
    points = (center, (int(lo[0] - 1), center[1], center[2]))
    probe = 'ez' if pol in ('ez', 's') else 'ey'
    for point in points:
        sim.add_probe(tuple((int(v) - 8) * DX for v in point), component=probe)
    import rfx.sources.tfsf as tfsf_module
    captured = {}
    setup_fn, core_fn = engine._build_step_setup, engine.make_core_step
    prepare_fn = tfsf_module.prepare_tfsf_for_scan
    def setup(*args, **kwargs):
        result = setup_fn(*args, **kwargs)
        captured['setup'] = result
        return result
    def core(*args, **kwargs):
        result = core_fn(*args, **kwargs)
        captured['core'], captured['ctx'] = result, args[0]
        return result
    def prepare(tfsf, *args):
        # The line itself; the field scan carries only its face record's step.
        captured['line'] = tfsf
        return prepare_fn(tfsf, *args)
    with patch.object(engine, '_build_step_setup', setup), patch.object(engine, 'make_core_step', core), \
            patch.object(tfsf_module, 'prepare_tfsf_for_scan', prepare):
        ordinary = sim.run(n_steps=n_steps)
    (cfg, line), initial = captured['line'], captured['setup'].carry_init
    box_shape = hi - lo + 1
    flat = np.random.default_rng(1662).choice(int(np.prod(box_shape)), 64, replace=False)
    nodes = np.array(np.unravel_index(flat, tuple(box_shape))).T + lo
    sf_mask = np.zeros(grid.shape, bool)
    sf_mask[8:-8, 8:-8, 8:-8] = True
    sf_mask[tuple(slice(a, b + 1) for a, b in zip(lo, hi))] = False
    tf_nodes = np.moveaxis(np.indices(tuple(box_shape)), 0, -1).reshape(-1, 3) + lo
    sample_indices = [np.floor((nodes + o) @ cfg.m).astype(int) - cfg.origin for o in OFFSETS]
    peak_indices = [np.unique(np.floor((tf_nodes + o) @ cfg.m).astype(int) - cfg.origin) for o in OFFSETS[:3]]
    def auxiliary(st, n):
        from rfx.stepping.tfsf import update_tfsf_1d
        st = update_tfsf_1d(cfg, st, DX, DT, n.astype(jnp.float32) * DT)
        samples = jnp.stack([getattr(st, c)[idx] for c, idx in zip(COMP, sample_indices)])
        peak = jnp.max(jnp.stack([jnp.max(jnp.abs(getattr(st, c)[idx]))
                                  for c, idx in zip(COMP[:3], peak_indices)]))
        return st, (samples, peak)
    _, (samples, incident) = jax.lax.scan(auxiliary, line, jnp.arange(n_steps))
    samples, incident = np.asarray(samples), np.asarray(incident, np.float64)
    def full_step(carry, n):
        carry, _, _ = captured['core'](carry, n, jnp.zeros(0, jnp.float32), jnp.zeros(0, jnp.float32))
        return carry, tuple(getattr(carry['fdtd'], c) for c in COMP)
    carry = initial
    sf, residual = np.zeros(n_steps), np.zeros(n_steps)
    grid_peaks = np.zeros((n_steps, 2))
    traces = np.zeros((n_steps, 2), np.float32)
    for begin in range(0, n_steps, 20):
        end = min(begin + 20, n_steps)
        carry, fields = jax.lax.scan(full_step, carry, jnp.arange(begin, end, dtype=jnp.int32))
        for c, block in enumerate(fields):
            block = np.asarray(block)
            scale = 1 if c < 3 else Z0
            grid_peaks[begin:end, c//3] = np.maximum(grid_peaks[begin:end, c//3],
                np.max(np.abs(block.astype(np.float64)), axis=(1, 2, 3)) * scale)
            sf[begin:end] = np.maximum(sf[begin:end], np.max(np.abs(block[:, sf_mask].astype(np.float64)), axis=1) * scale)
            diff = block[(slice(None),) + tuple(nodes.T)].astype(np.float64) - samples[begin:end, c]
            residual[begin:end] = np.maximum(residual[begin:end], np.max(np.abs(diff), axis=1) * scale)
            if COMP[c] == probe:
                traces[begin:end] = np.stack([block[(slice(None),) + p] for p in points], axis=1)
    jax.effects_barrier()
    tf_incident = incident.copy()
    line_peaks = np.array([_line_peaks[i] for i in range(1, n_steps + 1)])
    incident = np.max(np.concatenate([grid_peaks, line_peaks], axis=1), axis=1)
    finite = np.isfinite(incident) & np.isfinite(sf) & np.isfinite(residual)
    if not np.all(finite):
        return (dict(steps=n_steps, theta=theta, phi=phi, pol=pol, direction=direction,
            m=cfg.m, nonfinite_first_step=int(np.flatnonzero(~finite)[0] + 1),
            rSF=float('inf'), rD=float('inf'), rise_SF_ULP=float('inf'), rise_D_ULP=float('inf')),
            dict(I=incident, SF=sf, D=residual, trace=traces, ordinary=np.asarray(ordinary.time_series), nodes=nodes))
    peak = incident.max()
    ulp = float(np.spacing(np.float32(peak)))
    cumulative = incident.cumsum()
    active = cumulative >= peak
    per_step_ulp = np.spacing(incident.astype(np.float32)).astype(np.float64)
    ratios = [np.divide(np.maximum.accumulate(v), 1.2e-7 * cumulative,
                         out=np.zeros(n_steps), where=cumulative > 0) for v in (sf, residual)]
    summary = dict(steps=n_steps, theta=theta, phi=phi, pol=pol, direction=direction,
        m=cfg.m, max_I=float(peak), max_SF=float(sf.max()), max_D=float(residual.max()),
        switch_step=int(np.flatnonzero(active)[0] + 1),
        rSF=float(ratios[0][active].max()), rD=float(ratios[1][active].max()),
        final_rSF=float(ratios[0][-1]), final_rD=float(ratios[1][-1]),
        rise_SF_ULP=float((sf / per_step_ulp)[~active].max(initial=0)),
        rise_D_ULP=float((residual / per_step_ulp)[~active].max(initial=0)),
        D_ULP=float(residual.max() / ulp), leakage_dB=float(20 * np.log10(sf.max() / peak)),
        probe_unequal_bits=int(np.count_nonzero(traces.view(np.uint32) != np.asarray(ordinary.time_series).view(np.uint32))))
    return summary, dict(I=incident, I_TF=tf_incident, grid_peaks=grid_peaks,
        line_peaks=line_peaks, SF=sf, D=residual, trace=traces,
        ordinary=np.asarray(ordinary.time_series), nodes=nodes)


def assert_v1_v2(summary):
    assert all(np.isfinite(summary[key]) for key in ('rSF', 'rD', 'rise_SF_ULP', 'rise_D_ULP')), summary
    assert summary['rSF'] <= 1, summary
    assert summary['rD'] <= 1, summary
    assert summary['rise_SF_ULP'] <= 9, summary
    assert summary['rise_D_ULP'] <= 9, summary
    # Records 8/10 replace the historical normal-anchor maxima with the
    # paired line-inclusive scale; the archived-program comparison is V3.


@pytest.mark.source_scan
@pytest.mark.parametrize('theta,phi,pol', ((0, 0, 's'), (0, 0, 'p'), (30, 0, 'p'), (45, 30, 's')))
def test_v1_v2_l1(theta, phi, pol, record_property):
    summary, _ = measure_scene(theta, phi, pol)
    record_property('measurements', str(summary))
    assert_v1_v2(summary)


def measure_line_mutation(kind):
    difference = plane.line_difference
    def changed(cfg, state, target, source, axis):
        if kind == 'drop_h_half_step':
            cfg = replace(cfg, fractions=cfg.fractions[:3] + (0., 0., 0.))
        value = difference(cfg, state, target, source, axis)
        if kind == 'flip_curl_term' and (target, source, axis) == (4, 2, 0):
            value = -value
        return value
    with patch.object(plane, 'line_difference', changed):
        return measure_scene(45, 30, 'p')


@pytest.mark.source_scan
@pytest.mark.parametrize('kind', ('drop_h_half_step', 'flip_curl_term'))
def test_v1_line_mutation_is_red(kind, record_property):
    summary, _ = measure_line_mutation(kind)
    record_property('mutation', str(summary))
    assert (not np.isfinite(summary['rSF']) or summary['rSF'] > 1
            or summary['rise_SF_ULP'] > 9), summary


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


def test_v13b_zero_delay_defect_is_red(record_property):
    ratio, expected = window_width_ratio((10, 0), 's', zero_delays=True)
    record_property('defect_ratio', ratio.tolist())
    assert np.any(np.abs(ratio / expected - 1) > .02), (ratio, expected)


def table_defect_record(entry=None, epsilon=0.):
    """Native setup and native gathered values; change one coefficient only."""
    import inspect
    reads, captured = {}, {}
    native_init = plane.init_plane_wave
    def initialize(*args, **kwargs):
        cfg, state = native_init(*args, **kwargs)
        captured['cfg'] = cfg
        if entry is None:
            return cfg, state
        family, ti, ii = entry
        key = 'tables_' + family.lower()
        tables = list(getattr(cfg, key))
        coefficient = tables[ti].coefficient.copy()
        coefficient[ii] = np.float32(float(coefficient[ii]) * (1 + epsilon))
        tables[ti] = tables[ti]._replace(coefficient=coefficient)
        return replace(cfg, **{key: tuple(tables)}), state
    def tap(step, values, magnetic, ti):
        def save(n, array):
            reads[('H' if magnetic else 'E', ti, int(n))] = np.array(array)
        jax.debug.callback(save, step, values, ordered=True)
    source = inspect.getsource(plane.apply_tables)
    needle = '        values = table_operands(cfg, incident, table, ti, magnetic)'
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '\n        _tap(incident.step + 1, values, magnetic, ti)')
    namespace = dict(vars(plane), _tap=tap)
    exec(compile(source, '<actual table-read tap>', 'exec'), namespace)
    with patch.object(plane, 'init_plane_wave', initialize), patch.object(
            plane, 'apply_tables', namespace['apply_tables']):
        summary, arrays = measure_scene(45, 30, 's')
    jax.effects_barrier()
    return summary, arrays, captured['cfg'], reads


@pytest.mark.source_scan
def test_record10_scale_mutations(record_property):
    """Record 12: switch step 27; measured corner eps=3e-5 first goes red
    per-step at step 20 and on the law at step 18. At eps=1e-5 the corner's
    full-record per-step first red is step 33, law step 37. The face-centre
    eps=1e-5 is red on the law after the switch; eps=0 is green.
    """
    control, _, cfg, reads = table_defect_record()
    assert_v1_v2(control)
    candidates = []
    for family in ('E', 'H'):
        for ti, table in enumerate(getattr(cfg, 'tables_' + family.lower())):
            values = np.stack([reads[family, ti, n] for n in range(1, 301)])
            for ii in range(len(table.target_index)):
                nonzero = np.flatnonzero(values[:, ii] != 0)
                if len(nonzero):
                    step = int(nonzero[0])
                    product = abs(float(table.coefficient[ii]) * float(values[step, ii]))
                    candidates.append((step, -product, family, ti, ii))
    _, _, *corner = min(candidates)
    assert tuple(corner) == ('E', 8, 0)
    term = cfg.tables_e[corner[1]]
    assert term.target_component == 2 and term.source_component == 4
    assert np.unravel_index(term.target_index[corner[2]], cfg.shape) == (12, 12, 12)
    center = np.array([cfg.x_lo, (cfg.y_lo + cfg.y_hi)//2, (cfg.z_lo + cfg.z_hi)//2])
    candidates = []
    for ti, table in enumerate(cfg.tables_e):
        if (table.target_component, table.source_component) not in ((1, 5), (2, 4)):
            continue
        nodes = np.array(np.unravel_index(table.target_index, cfg.shape)).T
        for ii in np.flatnonzero(nodes[:, 0] == cfg.x_lo):
            candidates.append((int(np.sum((nodes[ii] - center)**2)),
                               -abs(float(table.coefficient[ii])), ti, int(ii)))
    _, _, ti, ii = min(candidates)
    assert (ti, ii) == (6, 144)
    assert cfg.tables_e[ti].target_component == 1 and cfg.tables_e[ti].source_component == 5
    assert np.unravel_index(cfg.tables_e[ti].target_index[ii], cfg.shape) == (12, 20, 20)
    corner_result, _, _, _ = table_defect_record(tuple(corner), 3e-5)
    face_result, _, _, _ = table_defect_record(('E', ti, ii), 1e-5)
    record_property('scale_mutations', dict(control=control, corner=corner_result, face=face_result))
    assert corner_result['rise_SF_ULP'] > 9, corner_result
    assert face_result['rSF'] > 1, face_result


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
