"""High-face component resolution: declaration R1–R5, decisions 1–3."""

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx._grid_metric import field_index

L = (.020, .016, .012)
C = (.0113, .0087, .0064)
DISPLACED = {'ex': (0,), 'ey': (1,), 'ez': (2,),
             'hx': (1, 2), 'hy': (0, 2), 'hz': (0, 1)}
HIGH = (19, 15, 11)
NODE_HIGH = (20, 16, 12)
LANES = ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward',
         'two_uniform_run', 'two_graded_forward', 'adi_run', 'subgrid_run')
FREQS = np.array([3e9, 5e9, 7e9])


def model(lane, axis, boundary, mirrored=False):
    kw = dict(freq_max=10e9, domain=L, dx=.001, cpml_layers=8,
              boundary={a: boundary if a == 'xyz'[axis] else 'pec' for a in 'xyz'})
    if 'graded' in lane:
        widths = np.array([1] + [.9, 1.1] * ((NODE_HIGH[axis] - 2) // 2) + [1]) * .001
        kw['d' + 'xyz'[axis] + '_profile'] = widths[::-1] if mirrored else widths
    if lane == 'adi_run':
        kw['solver'] = 'adi'
    sim = Simulation(**kw)
    if lane == 'subgrid_run':
        sim.add_refinement((.003, .009), ratio=2, xy_margin=.003)
    return sim


@pytest.mark.parametrize('lane', ('uniform_run', 'graded_run'))
@pytest.mark.parametrize('boundary', ('pec', 'pmc'))
@pytest.mark.parametrize('component', tuple(DISPLACED))
@pytest.mark.parametrize('axis', range(3))
def test_c1(lane, boundary, component, axis):
    sim = model(lane, axis, boundary)
    grid = sim._build_nonuniform_grid() if 'graded' in lane else sim._build_grid()
    for d in ((0., .4, .6, 1.) if axis in DISPLACED[component] else (0.,)):
        pos = list(C)
        pos[axis] = L[axis] - d * .001
        expected = HIGH[axis] if axis in DISPLACED[component] else NODE_HIGH[axis]
        assert field_index(grid, pos, component)[axis] == expected
    if axis in DISPLACED[component]:
        for d in (0., .4):
            pos[axis] = d * .001
            assert field_index(grid, pos, component)[axis] == 0


@pytest.mark.parametrize('lane', ('uniform_run', 'graded_run'))
@pytest.mark.parametrize('component,axis', [(c, a) for c, axes in DISPLACED.items() for a in axes])
def test_c3_absorber(lane, component, axis):
    sim = model(lane, axis, 'cpml')
    grid = sim._build_nonuniform_grid() if 'graded' in lane else sim._build_grid()
    pos = list(C)
    pos[axis] = L[axis]
    expected = ((28, 9, 6), (11, 24, 6), (11, 9, 20))[axis]
    assert field_index(grid, pos, component) == expected


@pytest.mark.parametrize('component,axis', [(c, a) for c, axes in DISPLACED.items() for a in axes])
def test_c1_periodic(component, axis):
    grid = model('uniform_run', axis, 'periodic')._build_grid()
    pos = list(C)
    pos[axis] = L[axis]
    assert field_index(grid, pos, component)[axis] == 0


@pytest.mark.parametrize('component', tuple(DISPLACED))
def test_c1_length_one(component):
    grid = Simulation(freq_max=10e9, domain=L, dx=.001, boundary='pec', mode='2d_tmz')._build_grid()
    assert field_index(grid, C, component)[2] == 0


@pytest.mark.parametrize('component,expected', (
    ('ex', (19, 16, 12)), ('ey', (20, 15, 12)), ('ez', (20, 16, 11)),
    ('hx', (20, 15, 11)), ('hy', (19, 16, 11)), ('hz', (19, 15, 12))))
def test_point_dft_consumer(component, expected):
    from rfx.probes.probes import init_dft_probe
    grid = model('uniform_run', 0, 'pec')._build_grid()
    assert init_dft_probe(grid, L, component, FREQS).index == expected


def execute(sim, lane, *, port=False):
    kwargs = dict(n_steps=200)
    if lane.startswith('two_'):
        if len(jax.devices('cpu')) < 2:
            pytest.skip('requires two CPU devices')
        kwargs['devices'] = jax.devices('cpu')[:2]
    if lane.endswith('forward'):
        kwargs['checkpoint'] = False
        if lane.startswith('two_'):
            kwargs['distributed'] = True
        if port:
            kwargs['port_s11_freqs'] = FREQS
        return sim.forward(**kwargs)
    kwargs['compute_s_params'] = port
    if port:
        kwargs.update(s_param_freqs=FREQS, s_param_n_steps=200)
    return sim.run(**kwargs)


def solve(lane, axis, boundary, row, d, side):
    mirrored = side == 'hi'
    sim = model(lane, axis, boundary, mirrored)
    wall, center = list(C), list(C)
    wall[axis] = d * .001
    if mirrored:
        wall[axis] = L[axis] - wall[axis]
        center[axis] = L[axis] - center[axis]
    ea, eb = 'e' + 'xyz'[axis], 'e' + 'xyz'[(axis + 1) % 3]
    pulse = GaussianPulse(f0=5e9, bandwidth=.8, amplitude=1.)
    if row == 'iii':
        sim.add_port(tuple(wall), ea, impedance=50., waveform=pulse)
        sim.add_probe(tuple(wall), ea)
        sim.add_probe(tuple(center), eb)
    else:
        src, comp = (wall, ea) if row == 'i' else (center, eb)
        sim.add_source(tuple(src), comp, waveform=pulse,
                       amplitude_kind='field' if lane == 'adi_run' else 'current')
        probes = [(center, eb)] if row == 'i' else [(wall, ea)] + [
            (wall, 'h' + c) for c in 'xyz' if c != 'xyz'[axis]]
        for pos, comp in probes:
            sim.add_probe(tuple(pos), comp)
    recorded_vi = row == 'iii' and lane == 'graded_run'
    if recorded_vi:
        # Graded run admits the port but exposes no lumped S request. Record
        # its four Ampere-loop legs. Only the port axis is graded, so both
        # transverse dual widths are the fixture's literal 1 mm.
        grid = sim._build_nonuniform_grid()
        cell = [11, 9, 6]
        cell[axis] = HIGH[axis] if mirrored else 0
        for comp, back_axis in ((('hz', 1), ('hy', 2)),
                                (('hx', 2), ('hz', 0)),
                                (('hy', 0), ('hx', 1)))[axis]:
            for back in (False, True):
                index = cell.copy()
                if back:
                    index[back_axis] -= 1
                sim.add_probe(tuple(grid.node_of(a, index[a]) for a in range(3)), comp)
    result = execute(sim, lane, port=row == 'iii' and not recorded_vi and lane not in ('adi_run', 'two_graded_forward'))
    data = np.array(result.time_series)
    spectrum = None
    if row == 'iii':
        if recorded_vi:
            from rfx.probes.sparam_driver import _lumped_recording_dfts
            from rfx.probes.probes import decompose_lumped_s_matrix
            v, i = _lumped_recording_dfts(data[:, [0, 2, 3, 4, 5]], FREQS, grid.dt, .001)
            v, i = v[None, ...], i[None, ...]
            # The current whole-port convention uses post-injection V/I.
            # v_ref's presence selects that convention; its values are unused.
            spectrum = np.abs(np.asarray(decompose_lumped_s_matrix(v, i, [50.], v_ref=np.zeros_like(v))))
            data = data[:, :2]
        else:
            spectrum = np.abs(np.asarray(result.s_params))
        assert np.all(np.isfinite(spectrum))
        data[:, 0] *= -.001
    return data, spectrum


def assert_identical(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), 'C2b requires bit-identical records'


def mirror(a, b, signs, record_property):
    peak_a = np.max(np.abs(a), axis=0)
    peak_b = np.max(np.abs(b), axis=0)
    assert np.all(peak_a > 0) and np.all(peak_b > 0), (peak_a, peak_b)
    residual = np.max(np.abs(b.astype(np.float64) - np.array(signs) * a.astype(np.float64)), axis=0)
    bar = 200 * np.finfo(np.float32).eps * peak_a.astype(np.float64)
    record_property('maximum_C2_fraction', float(np.max(residual / bar)))
    assert np.all(residual <= bar), dict(residual=residual.tolist(), bar=bar.tolist(), fraction=(residual / bar).tolist())


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('boundary', ('pec', 'pmc'))
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('row', ('i', 'ii', 'iii'))
def test_c2_c2b(lane, boundary, axis, row, record_property):
    # Existing admission is part of this per-path contract.
    refusal = None
    if lane == 'subgrid_run':
        refusal = 'subgridd|production|unstable'
    elif row == 'iii' and lane in ('two_graded_forward', 'adi_run'):
        refusal = 'port|impedance|ADI|adi|distributed|PMC|magnetic'
    elif boundary == 'pmc' and (lane.startswith('two_') or lane == 'adi_run'):
        refusal = 'PMC|pmc|magnetic'
    if refusal:
        for side, d in (('lo', 0.), ('hi', 0.), ('hi', .4), ('hi', 1.)):
            with pytest.raises((ValueError, NotImplementedError), match=refusal):
                solve(lane, axis, boundary, row, d, side)
        record_property('admission', 'refused')
        return
    try:
        inside, spectrum_inside = solve(lane, axis, boundary, row, 1., 'hi')
        assert np.all(np.max(np.abs(inside), axis=0) > 0)
        for d in (0., .4):
            high, spectrum_high = solve(lane, axis, boundary, row, d, 'hi')
            assert_identical(high, inside)
            if row == 'iii':
                assert_identical(spectrum_high, spectrum_inside)
            if lane != 'adi_run' and (boundary == 'pec' or axis == 2):
                low, spectrum_low = solve(lane, axis, boundary, row, d, 'lo')
                signs = (-1,) if row == 'i' else (-1, -1, -1) if row == 'ii' else (1, -1)
                mirror(low, high, signs, record_property)
                if row == 'iii':
                    residual = float(np.max(np.abs(spectrum_high - spectrum_low)))
                    record_property('maximum_S11_residual', residual)
                    assert residual <= 1e-4
        record_property('C2b', 'bit-identical; nonzero')
    finally:
        jax.clear_caches()


@pytest.mark.parametrize('lane', LANES[:-1])
@pytest.mark.parametrize('axis', range(3))
def test_c4(lane, axis, monkeypatch):
    from rfx import _realized
    from rfx.model import source_admission
    pos = list(C)
    pos[axis] = L[axis]
    expected = [(19, 9, 6), (11, 15, 6), (11, 9, 11)][axis]
    component = 'e' + 'xyz'[axis]
    seen = []
    original = source_admission.field_index

    def observe(grid, position, comp):
        result = original(grid, position, comp)
        seen.append((comp, result))
        return result

    monkeypatch.setattr(source_admission, 'field_index', observe)
    sim = model(lane, axis, 'pec')
    sim.add_source(tuple(pos), component, waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                   amplitude_kind='field' if lane == 'adi_run' else 'current')
    sim.add_probe(C, 'e' + 'xyz'[(axis + 1) % 3])
    sim.preflight()
    assert (component, expected) in seen
    assert sim.realized_geometry().ports[0].edges == (expected,)
    seen.clear()
    with _realized.capture() as captured:
        execute(sim, lane)
    assert (component, expected) in seen
    cells = [cell for record in captured.records for cell in record.get('cells', ())]
    assert (*expected, component) in cells
    tangent = model(lane, axis, 'pec')
    tangent.add_source(tuple(pos), 'e' + 'xyz'[(axis + 1) % 3], amplitude_kind='field')
    findings = tangent.preflight()
    assert any('PEC wall' in str(f) for f in findings)
    with pytest.raises(ValueError, match='Declared PEC wall'):
        execute(tangent, lane)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('boundary', ('pec', 'pmc'))
@pytest.mark.parametrize('axis', range(3))
def test_c2b_rlc(lane, boundary, axis, record_property):
    def response(d):
        sim = model(lane, axis, boundary, mirrored=True)
        point = list(C)
        point[axis] = L[axis] - d * .001
        sim.add_lumped_rlc(tuple(point), 'e' + 'xyz'[axis], R=50., L=1e-9, C=1e-13)
        sim.add_source(C, 'e' + 'xyz'[(axis + 1) % 3],
                       waveform=GaussianPulse(f0=5e9, bandwidth=.8),
                       amplitude_kind='field' if lane == 'adi_run' else 'current')
        sim.add_probe(tuple(point), 'e' + 'xyz'[axis])
        return np.asarray(execute(sim, lane).time_series)

    try:
        if lane.startswith('two_') or lane in ('adi_run', 'subgrid_run'):
            for d in (0., .4, 1.):
                with pytest.raises((ValueError, NotImplementedError), match='lumped|RLC|rlc|subgridd|PMC|magnetic'):
                    response(d)
            record_property('admission', 'refused')
            return
        inside = response(1.)
        assert np.max(np.abs(inside)) > 0
        for d in (0., .4):
            assert_identical(response(d), inside)
        record_property('C2b', 'bit-identical; nonzero')
    finally:
        jax.clear_caches()
