"""Exact formation seams, with the realized inputs lent by each runner."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx._internal._uniform_specs import resolve_periodic
from rfx.measurement.flux_weights import build_flux_monitor_cfgs, monitor_periodic_flags
from rfx.measurement.plan import build_measurement_plan
from rfx.measurement.setup import build_planes
from rfx.model.conductors import realized_conductors
from rfx.model.graded_admission import host_cells
from rfx.runners.nonuniform import build_flux_monitor_cfgs_nu
from tests.contracts.path_equivalence.builders import build, DX, FREQS, point
from tests.contracts.measurement_oracles import jacobian_nodes
from tests.contracts.test_measurement_plan import setup, node_map
from tests._x64_compat import enable_x64

PATHS = ('run_uniform', 'fwd_uniform', 'run_nonuniform', 'fwd_nonuniform',
         'run_distributed', 'fwd_distributed', 'run_distributed_nu',
         'fwd_distributed_nu', 'run_subgridded', 'msl_uniform', 'msl_nonuniform')
MESSAGES = {
    'probe': 'probe positions', 'port': 'port geometry',
    'plane_position': 'DFT plane positions', 'plane_region': 'DFT plane regions',
    'flux_position': 'flux monitor geometry', 'flux_region': 'flux monitor geometry',
    'ntff_lo': 'NTFF box corners', 'ntff_hi': 'NTFF box corners',
    'moment_position': 'current-moment monitor geometry',
}
MESSAGES = {kind: f'measurement plan needs concrete {name} inside jax.jit: '
            'declare them outside the traced function.' for kind, name in MESSAGES.items()}
MESSAGES['pec_edges'] = (
    'port:0: wire port live cells depend on PEC geometry that is traced, '
    'so they cannot be counted. Make that geometry concrete or remove the port.')


def exact(actual, expected):
    """The seam bar includes dtype, shape and every floating point bit."""
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


def nodes(channel, expected):
    actual = [(n.component, n.i, n.j, n.k, n.weight, n.area) for n in channel.nodes]
    assert len(actual) == len(expected)
    assert [v[:4] for v in actual] == [v[:4] for v in expected]
    exact([v[4] for v in actual], [v[4] for v in expected])
    assert [v[5] is None for v in actual] == [v[5] is None for v in expected]
    exact([v[5] for v in actual if v[5] is not None],
          [v[5] for v in expected if v[5] is not None])


def run_flags(sim, grid):
    periodic = tuple(a in sim._periodic_axes for a in 'xyz') if sim._periodic_axes else None
    if sim._tfsf is not None:
        from rfx.sources.tfsf import tfsf_from_declaration
        from rfx.boundaries.tfsf import boundary_flags
        periodic = boundary_flags(tfsf_from_declaration(sim._tfsf, grid, sim._freq_max,
                                                       n_steps=12)[0], grid)[0]
    return resolve_periodic(grid, periodic)


def plan_for(sim, grid, *, path=None, **kwargs):
    nu = hasattr(grid, 'dx_arr')
    flags = kwargs.pop('periodic', tuple(a in grid.periodic_axes for a in 'xyz')
                       if nu else run_flags(sim, grid))
    conductors = kwargs.pop('conductors', None)
    if conductors is None:
        conductors = realized_conductors(sim, grid, nonuniform=nu, periodic=flags)
    with patch('rfx.measurement.plan._pec_edges', side_effect=AssertionError('reassembled PEC')):
        return build_measurement_plan(sim, grid, n_steps=12, path=path, frequencies=FREQS,
                                      conductors=conductors, periodic=flags,
                                      staggered_area=nu, **kwargs)


def flux_scene(case):
    if case == 'graded':
        sim = build(('_flux_monitors', 'flux'), 'run_nonuniform', graded=True)
    elif case == 'periodic':
        sim = build(('_periodic_axes', 'periodic'), 'run_uniform')
        sim.add_flux_monitor(axis='x', coordinate=0., freqs=FREQS)
    elif case == '2d':
        sim = Simulation(freq_max=15e9, domain=point(10, 8, 1), dx=DX,
                         mode='2d_tmz', boundary='cpml', cpml_layers=2)
        sim.add_flux_monitor(axis='z', coordinate=0., freqs=FREQS)
    else:
        sim = build(('_flux_monitors', 'flux'), 'run_uniform')
        if case == 'floquet':
            sim = Simulation(freq_max=15e9, domain=point(10, 8, 6), dx=DX,
                             boundary='cpml', cpml_layers=2)
            sim.add_flux_monitor(axis='z', coordinate=3*DX, freqs=FREQS)
            sim.add_floquet_port(3*DX, axis='z', f0=5e9)
    sim._flux_monitors[0] = replace(sim._flux_monitors[0], dft_window='hann')
    return sim, sim._build_realized_grid()


def flux_nodes(owner, cfg, grid, flags):
    exact(owner.periodic, flags)
    assert owner.dft_window == cfg.window
    exact(owner.dft_window_alpha, cfg.window_alpha)
    exact(np.asarray(owner.frequencies), np.asarray(cfg.freqs))
    components = (('ey', 'ez', 'hy', 'hz'), ('ez', 'ex', 'hz', 'hx'),
                  ('ex', 'ey', 'hx', 'hy'))[cfg.axis]
    axes = [a for a in range(3) if a != cfg.axis]
    for c, (channel, component, acc) in enumerate(zip(owner.channels, components, cfg[:4])):
        if c < 2:
            planes = [(cfg.index, 1.)]
        elif cfg.normal_h_zero:
            planes = [(cfg.index, 0.)]
        else:
            back = ((cfg.index-1) % grid.shape[cfg.axis] if cfg.normal_periodic
                    else max(cfg.index-1, 0))
            planes = [(back, cfg.normal_h_weights[0]), (cfg.index, cfg.normal_h_weights[1])]
        area = np.asarray(cfg.dA if c in (0, 3) else cfg.dA2)
        expected = []
        for plane, weight in planes:
            for u, v in np.ndindex(acc.shape[-2:]):
                index = [0, 0, 0]
                index[cfg.axis] = plane
                index[axes[0]] = (u+cfg.lo1) % grid.shape[axes[0]]
                index[axes[1]] = (v+cfg.lo2) % grid.shape[axes[1]]
                expected.append((component, *index, weight, float(area[u, v])))
        nodes(channel, expected)


@pytest.mark.parametrize('case', ['uniform', 'periodic', '2d', 'floquet', 'graded'])
def test_b2_flux(case):
    sim, grid = flux_scene(case)
    flags = tuple(a in grid.periodic_axes for a in 'xyz') if case == 'graded' else run_flags(sim, grid)
    cfg = (build_flux_monitor_cfgs_nu(sim, grid, 12) if case == 'graded' else
           build_flux_monitor_cfgs(sim, grid, 12, periodic=flags))[0]
    plan = plan_for(sim, grid, periodic=flags, dft_window=cfg.window)
    flux_nodes(next(o for o in plan.owners if o.kind == 'flux'), cfg, grid, flags)
    if case in ('2d', 'floquet'):
        # The AD carry explicitly passes this second formation today.
        carry_flags = sim._periodic_flags()
        assert carry_flags == monitor_periodic_flags(sim, grid)
        assert carry_flags == ((False, False, False) if case == '2d' else (True, True, False))
        assert flags == ((False, False, True) if case == '2d' else (True, True, False))


@pytest.mark.parametrize('case', ['2d', 'periodic'])
@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_p2_attached_flux_uses_runner_formation(case, entry):
    from rfx.model.conductors import PreparedSolve

    sim, _ = flux_scene(case)
    prepared, formations = [], []
    original_init = PreparedSolve.__init__

    def remember(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        prepared.append(self)

    def capture(simulation, grid, n_steps, **kwargs):
        configs = build_flux_monitor_cfgs(simulation, grid, n_steps, **kwargs)
        formations.append((grid, kwargs['periodic'], configs))
        return configs

    with patch.object(PreparedSolve, '__init__', remember), patch(
            'rfx.runners.uniform.build_flux_monitor_cfgs', side_effect=capture):
        options = dict(n_steps=12, skip_preflight=True)
        if entry == 'run':
            options['compute_s_params'] = False
        else:
            options['checkpoint'] = False
        result = getattr(sim, entry)(**options)
        jax.block_until_ready(result.time_series)

    assert len(prepared) == len(formations) == 1
    plan = prepared[0].measurement
    assert plan.path == ('run_uniform' if entry == 'run' else 'fwd_uniform')
    grid, flags, configs = formations[0]
    owners = [owner for owner in plan.owners if owner.kind == 'flux']
    assert len(owners) == len(configs) == 1
    for owner, config in zip(owners, configs, strict=True):
        flux_nodes(owner, config, grid, flags)


def plane_nodes(owner, cfg, grid):
    axes = [a for a in range(3) if a != cfg.axis]
    lo, hi, low, high = cfg.region or (0, grid.shape[axes[0]], 0, grid.shape[axes[1]])
    cells = [np.asarray(host_cells(grid, a)) for a in axes]
    expected = []
    for u in range(lo, hi):
        for v in range(low, high):
            index = [0, 0, 0]
            index[cfg.axis], index[axes[0]], index[axes[1]] = cfg.index, u, v
            expected.append((cfg.component, *index, 1., float(cells[0][u]*cells[1][v])))
    nodes(owner.channels[0], expected)
    exact(np.asarray(owner.frequencies), np.asarray(cfg.freqs))
    assert owner.dft_window == cfg.window


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b2_points_and_planes(graded):
    sim = build(('_dft_planes', 'dft_plane'), 'run_nonuniform' if graded else 'run_uniform', graded=graded)
    sim._dft_plane_regions = {sim._dft_planes[0].name: (2, 5, 2, 4)}
    grid = sim._build_realized_grid()
    plan = plan_for(sim, grid)
    expected_probes = setup(sim)['probes']
    for owner, entry in zip((o for o in plan.owners if o.kind == 'probe'), sim._probes):
        from rfx._internal._grid_metric import field_index
        index = field_index(grid, entry.position, entry.component)
        assert (*index, entry.component) in expected_probes
        nodes(owner.channels[0], [(entry.component, *index, 1., None)])
    for owner, cfg in zip((o for o in plan.owners if o.kind == 'dft_plane'), build_planes(sim, grid, 12)):
        plane_nodes(owner, cfg, grid)


def test_b2_windowed_raw_plane():
    sim = build(('_dft_planes', 'dft_plane'), 'run_uniform')
    grid = sim._build_realized_grid()
    plan = plan_for(sim, grid, dft_window='hann')
    for owner, cfg in zip((o for o in plan.owners if o.kind == 'dft_plane'),
                          build_planes(sim, grid, 12, dft_window='hann'), strict=True):
        plane_nodes(owner, cfg, grid)
        exact(owner.dft_window_alpha, cfg.window_alpha)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b2_wire_beside_pec(graded):
    sim = build(('_ports', 'wire_port'), 'run_nonuniform' if graded else 'run_uniform', graded=graded)
    edge = 3.9 if graded else 4
    sim.add(Box(point(edge, 2, 2), point(6, edge, 3)), material='pec')
    actual = setup(sim)
    grid = actual['grid']
    conductors = realized_conductors(sim, grid, nonuniform=graded)
    plan = plan_for(sim, grid, conductors=conductors)
    from rfx.sources.sources import _wire_port_live_cells, wire_port_from_entry
    cells, flags, _ = _wire_port_live_cells(grid, wire_port_from_entry(sim._ports[0]), conductors.pec_edges)
    live = [tuple(p) for p, flag in zip(cells, flags) if flag]
    assert len(live) == 1
    owner = next(o for o in plan.owners if o.kind == 'wire')
    from tests.contracts.test_measurement_plan import judge
    judge(plan, actual, ('_ports', 'wire_port'), graded)
    gap = next(c for c in owner.channels if c.name == 'V_port')
    assert [(n.i, n.j, n.k) for n in gap.nodes] == live


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b2_waveguide(graded):
    sim = build(('_waveguide_ports', 'waveguide_port'), 'run_nonuniform' if graded else 'run_uniform', graded=graded)
    actual = setup(sim)
    plan = plan_for(sim, actual['grid'])
    cfg = actual['waveguide_ports'][0]
    from rfx.sources.waveguide_port import modal_voltage, modal_current
    owner = next(o for o in plan.owners if o.kind == 'waveguide')
    for channel, index in zip(owner.channels, (cfg.probe_x, cfg.probe_x, cfg.ref_x, cfg.ref_x)):
        sampler = modal_voltage if channel.kind == 'E' else modal_current
        oracle = jacobian_nodes(lambda f: sampler(f, cfg, index, 0.), actual['grid'].shape)
        got = node_map(channel)
        assert got.keys() == oracle.keys()
        exact(list(got.values()), [oracle[k] for k in got])
        area = np.asarray(cfg.aperture_dA)
        repeats = 1 if channel.kind == 'E' else 2
        exact([n.area for n in channel.nodes], np.tile(np.repeat(area.ravel(), repeats), 2).astype(float))


def ntff_nodes(owner, box, grid):
    bounds = [(box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi)]
    cells = [np.asarray(host_cells(grid, a)) for a in range(3)]
    for channel in owner.channels:
        face, component = channel.name.split(':')
        axis = 'xyz'.index(face[0])
        axes = [a for a in range(3) if a != axis]
        plane = bounds[axis][int(face.endswith('hi'))]
        w = getattr(box, 'w_'+face)
        normal = [(plane-1, w), (plane, 1-w)] if component[0] == 'h' else [(plane, 1.)]
        shift = 'xyz'.index(component[1]) if component[0] == 'h' else next(a for a in axes if 'xyz'[a] != component[1])
        expected = []
        for u in range(*bounds[axes[0]]):
            for v in range(*bounds[axes[1]]):
                for p, weight in normal:
                    for delta in (0, 1):
                        index = [0, 0, 0]
                        index[axis], index[axes[0]], index[axes[1]] = p, u, v
                        index[shift] += delta
                        expected.append((component, *index, weight*.5, float(cells[axes[0]][u]*cells[axes[1]][v])))
        nodes(channel, expected)


def moment_nodes(owner, m, grid):
    from rfx.core.yee import EPS_0
    weights = [np.asarray(w) for w in (m.w_ex, m.w_ey, m.w_ez)]
    inv = [np.asarray(w) for w in (m.inv_dx_e, m.inv_dy_e, m.inv_dz_e)]
    coef = weights[0].dtype.type(EPS_0)/weights[0].dtype.type(grid.dt)
    curls = ((('hz', 1), ('hy', 2)), (('hx', 2), ('hz', 0)), (('hy', 0), ('hx', 1)))
    for channel in owner.channels:
        _, b, _, c, _, w, stage = channel.name.split(':')
        b, c, w = int(b), int(c), int(w)
        values = weights[c][w]
        seg = np.asarray(m.seg).reshape(values.shape[:2])
        expected = []
        for index in np.ndindex(values.shape):
            if seg[index[:2]] != b:
                continue
            at = tuple(a+i for a, i in zip((m.i_lo, m.j_lo, m.k_lo), index))
            value = values[index]
            if stage != 'H':
                weight = float(-value*coef)
                expected.append(('e'+'xyz'[c], *at, -weight if stage == 'E_prev' else weight, None))
            else:
                for term, (comp, axis) in enumerate(curls[c]):
                    weight = float(value*inv[axis][index[axis]]*m.curl_signs[2*c+term])
                    back = list(at)
                    back[axis] -= 1
                    expected.extend(((comp, *at, weight, None), (comp, *back, -weight, None)))
        nodes(channel, expected)


@pytest.mark.parametrize('kind', ['ntff', 'moments'])
@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b2_monitors(kind, graded):
    row = ('_ntff', 'ntff_box') if kind == 'ntff' else ('_current_moments', 'block_moments')
    sim = build(row, 'run_nonuniform' if graded else 'run_uniform', graded=graded)
    grid = sim._build_realized_grid()
    conductors = realized_conductors(sim, grid, nonuniform=graded)
    plan = plan_for(sim, grid, conductors=conductors)
    if kind == 'ntff':
        from rfx.farfield import make_ntff_box
        ntff_nodes(next(o for o in plan.owners if o.kind == 'ntff'), make_ntff_box(grid, *sim._ntff), grid)
    else:
        from rfx.current_moments import monitor_for_simulation
        m = monitor_for_simulation(sim, grid, sim._periodic_flags(), conductors=conductors)
        moment_nodes(next(o for o in plan.owners if o.kind == 'current_moment'), m, grid)


@pytest.mark.parametrize('kind', ['planes', 'flux', 'moments', 'ntff'])
def test_b2_slab_records(kind):
    from rfx.runners._distributed_records import SlabRecords
    row = {'planes': ('_dft_planes', 'dft_plane'), 'flux': ('_flux_monitors', 'flux'),
           'moments': ('_current_moments', 'block_moments'), 'ntff': ('_ntff', 'ntff_box')}[kind]
    sim = build(row, 'run_uniform')
    grid = sim._build_realized_grid()
    records = SlabRecords(sim, grid, (grid.nx+1)//2, SimpleNamespace(size=2), jnp.float32, 12)
    plan = plan_for(sim, grid, path='run_distributed', periodic=sim._periodic_flags())
    if kind == 'planes':
        for owner, cfg in zip((o for o in plan.owners if o.kind == 'dft_plane'), records.layouts[kind].templates):
            plane_nodes(owner, cfg, grid)
    elif kind == 'flux':
        flux_nodes(next(o for o in plan.owners if o.kind == 'flux'), records.layouts[kind].templates[0], grid, sim._periodic_flags())
    elif kind == 'moments':
        moment_nodes(next(o for o in plan.owners if o.kind == 'current_moment'), records.monitor, grid)
    else:
        ntff_nodes(next(o for o in plan.owners if o.kind == 'ntff'), records.ntff_box, grid)


def test_b2_uniform_slab_ports():
    from rfx.stepping.uniform_slab_ports import UniformSlabPorts
    sim = build(('_ports', 'wire_port'), 'run_uniform')
    actual = setup(sim)
    grid, specs = actual['grid'], actual['port_record_specs']
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ('x',))
    layout = SimpleNamespace(n_devices=mesh.size, nx_per_rank=(grid.nx+mesh.size-1)//mesh.size, ghost_width=1)
    boundary = SimpleNamespace(pmc_faces=sim._boundary_spec.pmc_faces(), periodic=run_flags(sim, grid))
    slab = UniformSlabPorts(grid, sim._ports, specs, boundary, layout, mesh, 12, jnp.float32, FREQS)
    plan = plan_for(sim, grid, path='run_distributed')
    owner = next(o for o in plan.owners if o.kind == 'wire')
    for channel, expected in zip(owner.channels, slab.owners[0].channels):
        nodes(channel, [(n.component, n.i, n.j, n.k, n.weight, n.area) for n in expected.nodes])


def test_b2_multidevice_graded_flux(request):
    from rfx.runners._distributed_records import SlabRecords
    sim, grid = flux_scene('graded')
    records = SlabRecords(sim, grid, (grid.nx+1)//2, SimpleNamespace(size=2), jnp.float32, 12)
    slab = records.layouts['flux'].templates[0]
    single = build_flux_monitor_cfgs_nu(sim, grid, 12)[0]
    differences = []
    for name in slab._fields:
        a, b = np.asarray(getattr(slab, name)), np.asarray(getattr(single, name))
        if a.dtype.kind not in 'biufc':
            assert a.tolist() == b.tolist()
            continue
        if a.shape != b.shape:
            differences.append(f'{name}: shapes {a.shape}/{b.shape}')
            continue
        delta = np.abs(a.astype(complex)-b.astype(complex))
        count = np.count_nonzero(delta)
        if count:
            rel = delta/np.maximum(np.abs(b), np.finfo(float).tiny)
            differences.append(f'{name}: {count}/{a.size}, max_abs={delta.max():.17g}, max_rel={rel.max():.17g}')
    print('multi-device graded flux: '+('; '.join(differences) or 'all arrays equal; 0 differing elements, max_abs=0, max_rel=0'))
    if differences:
        request.node.add_marker(pytest.mark.xfail(
            strict=True, reason='multi-device graded flux uses the uniform flux builder; separate defect'))
        assert not differences, '; '.join(differences)
    for name in slab._fields:
        exact(getattr(slab, name), getattr(single, name))
    plan = plan_for(sim, grid, path='run_distributed_nu', periodic=monitor_periodic_flags(sim, grid))
    flux_nodes(next(o for o in plan.owners if o.kind == 'flux'), slab, grid, monitor_periodic_flags(sim, grid))


@pytest.mark.parametrize('actual,expected', [(np.array([1]), np.array([2])),
    (np.array([1.]), np.array([np.nextafter(1., 2.)])),
    (np.array([0.]), np.array([-0.])), (np.ones(2), np.ones((1, 2)))])
def test_b2_comparison_is_live(actual, expected):
    with pytest.raises(AssertionError):
        exact(actual, expected)


@pytest.mark.parametrize('path', PATHS)
def test_b5_phase_leaves(path):
    nu = 'nonuniform' in path or path.endswith('_nu')
    sim = build(('_ports', 'wire_port'), 'run_nonuniform' if nu else 'run_uniform')
    grid = sim._build_realized_grid()
    @jax.jit
    def hold(freqs, dt, window):
        if hasattr(grid, '_replace'):
            g = grid._replace(dt=dt)
        else:
            g = copy(grid)
            g.dt = dt
        plan = build_measurement_plan(sim, g, n_steps=12, path=path, frequencies=freqs,
                                      pec_edges=None, dft_window=window)
        owner = next(o for o in plan.owners if o.kind == 'wire')
        assert owner.frequencies is freqs
        assert plan.time_base.dt is dt
        assert owner.dft_window is window
        assert isinstance(owner.frequencies, jax.core.Tracer)
        return owner.frequencies, plan.time_base.dt, owner.dft_window
    f, dt, w = hold(jnp.asarray(FREQS), jnp.asarray(grid.dt), jnp.ones(12))
    exact(f, FREQS)
    exact(dt, jnp.asarray(grid.dt))
    exact(w, np.ones(12, np.float32))


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('kind', ['probe', 'port', 'plane_position', 'plane_region', 'flux_position',
                                  'flux_region', 'pec_edges', 'ntff_lo', 'ntff_hi', 'moment_position'])
def test_b5_refuses_nodes(path, kind):
    nu = 'nonuniform' in path or path.endswith('_nu')
    row = {'probe': ('_probes', 'probe'), 'port': ('_ports', 'wire_port'),
           'plane_position': ('_dft_planes', 'dft_plane'), 'plane_region': ('_dft_planes', 'dft_plane'),
           'flux_position': ('_flux_monitors', 'flux'), 'flux_region': ('_flux_monitors', 'flux'),
           'pec_edges': ('_ports', 'wire_port'), 'ntff_lo': ('_ntff', 'ntff_box'),
           'ntff_hi': ('_ntff', 'ntff_box'), 'moment_position': ('_current_moments', 'block_moments')}[kind]
    sim = build(row, 'run_nonuniform' if nu else 'run_uniform')
    grid = sim._build_realized_grid()
    @jax.jit
    def traced(value):
        from copy import copy
        declaration = copy(sim)
        edges = None
        if kind in ('probe', 'port'):
            attr = '_probes' if kind == 'probe' else '_ports'
            entries = list(getattr(sim, attr))
            entries[0] = replace(entries[0], position=(value, DX, DX))
            setattr(declaration, attr, entries)
        elif kind == 'plane_region':
            declaration._dft_plane_regions = {sim._dft_planes[0].name: (value, 5, 2, 4)}
        elif kind.startswith('plane') or kind.startswith('flux'):
            attr = '_dft_planes' if kind.startswith('plane') else '_flux_monitors'
            entries = list(getattr(sim, attr))
            entries[0] = replace(entries[0], **({'size': (value, DX)} if kind == 'flux_region' else {'coordinate': value}))
            setattr(declaration, attr, entries)
        elif kind == 'pec_edges':
            edges = (jnp.full(grid.shape, value > 0),)*3
        else:
            attr = '_current_moments' if kind == 'moment_position' else '_ntff'
            spec = list(getattr(sim, attr))
            spec[int(kind == 'ntff_hi')] = (value, DX, DX)
            setattr(declaration, attr, tuple(spec))
        build_measurement_plan(declaration, grid, n_steps=12, path=path, pec_edges=edges)
        return value
    with pytest.raises(TracedStructureRefusal) as error:
        traced(jnp.asarray(DX))
    assert str(error.value) == MESSAGES[kind]


def test_unknown_path_names_token():
    sim = build(('_probes', 'probe'), 'run_uniform')
    with pytest.raises(ValueError, match='invented_path'):
        build_measurement_plan(sim, sim._build_realized_grid(), n_steps=12, path='invented_path')


@pytest.mark.parametrize('kind', ['dft_plane', 'flux', 'ntff', 'current_moment'])
def test_b5_declared_phase_leaves(kind):
    row = {'dft_plane': ('_dft_planes', 'dft_plane'), 'flux': ('_flux_monitors', 'flux'),
           'ntff': ('_ntff', 'ntff_box'), 'current_moment': ('_current_moments', 'block_moments')}[kind]
    sim = build(row, 'run_uniform')
    grid = sim._build_realized_grid()
    @jax.jit
    def hold(freqs, window):
        declaration = copy(sim)
        if kind == 'current_moment':
            spec = list(sim._current_moments)
            spec[3] = freqs
            declaration._current_moments = tuple(spec)
        elif kind == 'ntff':
            declaration._ntff = (*sim._ntff[:2], freqs)
        else:
            attr = '_dft_planes' if kind == 'dft_plane' else '_flux_monitors'
            setattr(declaration, attr, [replace(e, freqs=freqs) for e in getattr(sim, attr)])
        plan = build_measurement_plan(declaration, grid, n_steps=12, dft_window=window)
        owner = next(o for o in plan.owners if o.kind == kind)
        assert owner.frequencies is freqs
        assert owner.dft_window is window
        return owner.frequencies, owner.dft_window
    freqs, window = hold(jnp.asarray(FREQS), jnp.ones(12))
    exact(freqs, FREQS)
    exact(window, np.ones(12, np.float32))
