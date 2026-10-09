"""High-face component resolution: declaration R1–R5, decisions 1–4."""

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


@pytest.mark.parametrize('component,axis', [(c, a) for c, axes in DISPLACED.items() for a in axes])
def test_c1_periodic_last_entry(component, axis):
    grid = model('uniform_run', axis, 'periodic')._build_grid()
    pos = list(C)
    pos[axis] = L[axis] - .001
    assert field_index(grid, pos, component)[axis] == (19, 15, 11)[axis]


def subgrid_response(entity, top, d, monkeypatch):
    from rfx.subgridding import jit_runner
    sim = Simulation(freq_max=10e9, domain=L, dx=.001, boundary='pec')
    sim.add_refinement((.003, top), ratio=2, validation='off')
    point = (*C[:2], top - d * .0005)
    pulse = GaussianPulse(f0=5e9, bandwidth=.8)
    sim.add_source(point if entity == 'source' else C,
                   'ez' if entity == 'source' else 'ex', waveform=pulse)
    sim.add_probe(C if entity == 'source' else point,
                  'ex' if entity == 'source' else 'ez')
    seen = []
    original = jit_runner.run_subgridded_jit

    def capture(*args, **kwargs):
        opts = kwargs['opts']
        seen.append(tuple(opts.sources_f[0][:3]) if entity == 'source'
                    else tuple(opts.probe_indices_f[0]))
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jit_runner, 'run_subgridded_jit', capture)
        data = np.asarray(sim.run(n_steps=200).time_series)
    assert np.all(np.isfinite(data)) and np.max(np.abs(data)) > 0
    assert len(seen) == 1
    return data, seen[0]


@pytest.mark.parametrize('entity', ('source', 'probe'))
@pytest.mark.parametrize('top', (.009, .012))
def test_c6_subgrid_interface_and_wall(entity, top, monkeypatch, record_property):
    try:
        wall, index = subgrid_response(entity, top, 0., monkeypatch)
        record_property('peak', float(np.max(np.abs(wall))))
        record_property('fine_index', index)
        assert index == ((23, 17, 12) if top == .009 else (23, 17, 17))
        if top == .012:
            half, half_index = subgrid_response(entity, top, .5, monkeypatch)
            assert half_index == (23, 17, 17)
            assert_identical(wall, half)
    finally:
        jax.clear_caches()


def ringdown_wall_model(graded, full):
    kwargs = {'dz_profile': np.array([1] + [.9, 1.1] * 5 + [1]) * .001} if graded else {}
    sim = Simulation(freq_max=10e9, domain=L, dx=.001, boundary='pec', cpml_layers=0, **kwargs)
    start = 0. if full else (.0079 if graded else .008)
    sim.add_port((*C[:2], start), 'ez', impedance=50., extent=.012-start,
                 waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    return sim


@pytest.mark.parametrize('graded', (False, True))
@pytest.mark.parametrize('full', (False, True))
def test_c7_ringdown_wire_to_wall(graded, full, monkeypatch, record_property):
    from rfx.ringdown import RingdownRun, RingdownSpec
    planned = []
    original = RingdownRun._plan_probes

    def capture(self):
        keys = original(self)
        planned.extend(keys)
        return keys

    monkeypatch.setattr(RingdownRun, '_plan_probes', capture)
    sim = ringdown_wall_model(graded, full)
    steps = 4000 if graded else 3000
    try:
        result = sim.run(n_steps=steps, s_param_n_steps=steps,
                         s_param_freqs=np.linspace(3e9, 9e9, 7),
                         compute_s_params=True, ringdown=RingdownSpec(), skip_preflight=True)
        assert result.ringdown is not None
        assert result.ringdown.report.completed, result.ringdown.report.failure
        record_property('failure', result.ringdown.report.failure)
        assert planned
        for component, index in planned:
            for axis in DISPLACED[component]:
                assert index[axis] != (20, 16, 12)[axis]
    finally:
        jax.clear_caches()


def consumer_fixture(*, port=False, metal=False):
    from rfx import Box
    sim = Simulation(freq_max=10e9, domain=L, dx=.001, boundary='pec')
    point = (.020, .0087, .0064)
    if metal:
        sim.add(Box((.019, .008, .005), (.020, .010, .007)), material='pec')
    if port:
        sim.add_port(point, 'ex', impedance=50., waveform=GaussianPulse(f0=5e9, bandwidth=.8))
    grid = sim._build_grid()
    return sim, grid, point


@pytest.mark.parametrize('consumer', (
    'measurement_probe', 'measurement_port', 'settling_drive', 'settling_probe',
    'preflight_probe', 'conductor_release', 'source_admission', 'source_coefficients',
    'sample_field', 'port_voltage', 'port_current', 'init_sparam_probe',
    'port_terminal', 'material_fit', 'current_moments', 'rlc', 'source_stamp',
    'sparam_recording', 'realized_geometry', 'vmap_source', 'eager_port',
    'decay_monitor', 'design_box_port', 'design_box_source'))
def test_c8_consumers(consumer, monkeypatch):
    import importlib
    from types import SimpleNamespace
    import jax.numpy as jnp
    from rfx.model.conductors import realized_conductors
    from rfx.sources.sources import LumpedPort
    sim, grid, point = consumer_fixture(port=True)
    cell = (19, 9, 6)
    entry = sim._ports[0]
    port = LumpedPort(point, 'ex', 50., GaussianPulse(f0=5e9, bandwidth=.8))
    if consumer.startswith('measurement_'):
        from rfx.measurement.plan import build_measurement_plan
        sim.add_probe(point, 'ex')
        plan = build_measurement_plan(sim, grid, n_steps=2)
        owner = next(o for o in plan.owners if o.id == ('probe:0' if consumer.endswith('probe') else 'port:0:mode:0'))
        nodes = owner.channels[0].nodes
        assert nodes and {(n.i, n.j, n.k) for n in nodes} == {cell}
    elif consumer.startswith('settling_'):
        from rfx.probes.settling import drive_cells, source_dominated_columns
        if consumer == 'settling_drive':
            assert drive_cells(grid, [entry]) == frozenset({cell})
        else:
            assert source_dominated_columns(grid, [entry], {cell}) == frozenset({0})
    elif consumer in ('preflight_probe', 'source_admission', 'material_fit'):
        sim, grid, point = consumer_fixture(metal=True)
        if consumer == 'preflight_probe':
            sim.add_probe(point, 'ex')
            assert len(sim.preflight().by_code('port_in_pec')) == 1
        elif consumer == 'source_admission':
            from rfx.model.source_admission import _dead_sources
            sim.add_source(point, 'ex')
            messages = list(_dead_sources(sim, realized_conductors(sim, grid)))
            assert len(messages) == 1 and 'grid index (19, 9, 6)' in messages[0]
        else:
            fit = importlib.import_module('rfx.differentiable_material_fit')
            sim.add_port(point, 'ex')
            model = realized_conductors(sim, grid)
            with pytest.raises(NotImplementedError, match='PEC edge'):
                fit._require_ports_clear_of_pec(sim, grid, model.pec_cells, model.sheets, model.wires)
    elif consumer == 'conductor_release':
        from dataclasses import replace
        from rfx.model.conductors import lumped_port_stage
        root = realized_conductors(sim, grid)
        masks = tuple(jnp.ones(grid.shape, dtype=bool) for _ in range(3))
        released = lumped_port_stage(replace(root, pec_edges=masks), entry, 'port[0]')
        assert not bool(released.pec_edges[0][cell])
        assert bool(released.pec_edges[0][20, 9, 6])
        assert released.provenance[-1].edges == (cell,)
    elif consumer == 'source_coefficients':
        from rfx.model import source_coefficients
        # Exercise the pad-edge admission consumer with a PEC x face and
        # absorbers on y/z; no field solve is substituted or bypassed.
        sim = Simulation(freq_max=10e9, domain=L, dx=.001,
                         boundary={'x': 'pec', 'y': 'upml', 'z': 'upml'}, cpml_layers=2)
        sim.add_source(point, 'ex', amplitude_kind='current')
        grid = sim._build_grid()
        assert sim._boundary == 'upml'
        seen = []
        original = source_coefficients._refuse_pad_edge

        def capture(g, index, *args):
            seen.append(index)
            return original(g, index, *args)

        monkeypatch.setattr(source_coefficients, '_refuse_pad_edge', capture)
        queue = SimpleNamespace(resolve=lambda materials: [])
        from rfx.model.materials import realize_components
        materials = sim._assemble_materials(grid)[0]
        materials = materials._replace(components=realize_components(
            materials, grid, periodic=(False, False, False)))
        source_coefficients.resolve_run_sources(queue, materials, sim, grid)
        assert seen == [(19, 11, 8)]
    elif consumer in ('sample_field', 'port_voltage', 'port_current', 'init_sparam_probe'):
        from rfx.probes import probes
        from rfx.core.yee import init_state
        state = init_state(grid.shape)
        state = state._replace(ex=state.ex.at[cell].set(3.), hz=state.hz.at[cell].set(7.))
        if consumer == 'sample_field':
            assert float(probes.sample_field(state, grid, probes.FieldMonitor(point, 'ex'))) == 3.
        elif consumer == 'port_voltage':
            assert float(probes.port_voltage(state, grid, port)) == pytest.approx(-.003)
        elif consumer == 'port_current':
            assert float(probes.port_current(state, grid, port)) == pytest.approx(.007)
        else:
            assert probes.init_sparam_probe(grid, port, FREQS).port_index == cell
    elif consumer == 'port_terminal':
        from rfx.geometry.port_termination import port_terminal_points
        nodes = tuple([grid.node_of(a, i) for i in range(grid.shape[a])] for a in range(3))
        start, end = port_terminal_points('_ports', entry, grid, nodes)
        np.testing.assert_allclose(start, (.019, .009, .006), rtol=0, atol=1e-9)
        np.testing.assert_allclose(end, (.020, .009, .006), rtol=0, atol=1e-9)
    elif consumer == 'current_moments':
        from rfx.current_moments import refuse_current_the_monitor_cannot_see
        # With zero absorber layers the last stored displaced entry is
        # outside. This low-level census permits CPML declarations.
        sim = Simulation(freq_max=10e9, domain=L, dx=.001, boundary='cpml', cpml_layers=0)
        sim.add_source(point, 'ex', amplitude_kind='current')
        grid = sim._build_grid()
        monitor = SimpleNamespace(i_lo=0, i_hi=21, j_lo=0, j_hi=17, k_lo=0, k_hi=13)
        refuse_current_the_monitor_cannot_see(sim, grid, monitor)
    elif consumer == 'rlc':
        from rfx.lumped import LumpedRLCSpec, build_rlc_meta
        meta = build_rlc_meta(grid, LumpedRLCSpec(position=point, component='ex', R=50., L=1e-9),
                              sim._assemble_materials(grid)[0])
        assert (meta.i, meta.j, meta.k) == cell
    elif consumer == 'source_stamp':
        from rfx.sources.sources import setup_lumped_port
        materials = setup_lumped_port(grid, port, sim._assemble_materials(grid)[0])
        assert set(port._drive_stamps) == {cell}
        assert float(materials.sigma[cell]) > 0
    elif consumer == 'sparam_recording':
        from rfx.probes.sparam_driver import _lumped_recording_probes
        probes, _ = _lumped_recording_probes(grid, [entry])
        assert (probes[0].i, probes[0].j, probes[0].k) == cell
    elif consumer == 'realized_geometry':
        assert sim.realized_geometry().ports[0].edges == (cell,)
    elif consumer == 'vmap_source':
        sweep = importlib.import_module('rfx.vmap_sweep')
        sim._ports.clear()
        sim.add_source(point, 'ex', amplitude_kind='current')
        seen = []

        def capture(*args, **kwargs):
            seen.extend(kwargs['j_source_meta'])
            return lambda *args: None

        monkeypatch.setattr(sweep, '_build_vmap_scan_fn', capture)
        sweep._build_full_scan_fn(sim, grid, sim._assemble_materials(grid)[0], 2)
        assert seen == [(19, 9, 6, 'ex')]
    elif consumer == 'eager_port':
        import rfx.simulation as stepping
        from rfx.stepping.probe_loop import make_probe_step
        monkeypatch.setattr(stepping, 'make_core_step', lambda ctx, **kwargs: ctx)
        ctx, _ = make_probe_step(grid, sim._assemble_materials(grid)[0], [port], 0,
                                use_cpml=False, cpml_params=None, cpml_axes='',
                                debye=None, lorentz=None, pec_edge_masks=None, curl_boundary=None)
        assert tuple(int(a[0]) for a in ctx.drives.electric.nodes['ex']) == cell
    elif consumer == 'decay_monitor':
        import rfx.simulation as stepping

        class Captured(Exception):
            pass

        def capture(ctx):
            assert ctx.mon_idx == cell
            raise Captured

        monkeypatch.setattr(stepping, 'core_step_invariants', capture)
        with pytest.raises(Captured):
            stepping.run_until_decay(grid, sim._assemble_materials(grid)[0], min_steps=1,
                                     max_steps=2, monitor_position=point, monitor_component='ex')
    else:
        if consumer == 'design_box_source':
            sim._ports.clear()
            sim.add_source(point, 'ex')
        kwargs = dict(design_box=((.018, .008, .005), (.018, .009, .006)),
                      design_eps_override=jnp.ones((1, 2, 2)), design_sigma_override=None,
                      design_occupancy_override=None, eps_override=None, sigma_override=None,
                      mu_r_override=None, pec_occupancy_override=None, debye_spec=None,
                      lorentz_spec=None, kerr_chi3=None, holds_ports=True)
        if consumer == 'design_box_source':
            with pytest.raises(ValueError, match='holds a soft source'):
                sim._resolve_design_box_override(grid, **kwargs)
        else:
            spec, _ = sim._resolve_design_box_override(grid, **kwargs)
            assert spec.held_edges == ((0, 19, 9, 6),)
