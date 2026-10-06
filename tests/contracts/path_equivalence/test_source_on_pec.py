"""S1 G1b: source-on-PEC is a refusal, including the S3-2 a2 geometry."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, PolylineWire, Simulation
from rfx.model.conductors import realized_conductors, solve_conductors
from rfx.model.source_admission import refuse_dead_soft_sources

DX = 1 / 1024
PATHS = ['uniform', 'graded', 'distributed', 'distributed_graded']


def point(*values):
    return tuple(v * DX for v in values)


def waveform(t):
    return jnp.cos(2e10 * t) * jnp.exp(-((t - 4e-11) / 2e-11)**2)


def model(path='uniform', kind='volume', *, adjacent=False, component='ez'):
    kwargs = {}
    if 'graded' in path:
        for axis, n in zip('xyz', (12, 8, 8)):
            profile = np.full(n, DX)
            profile[2:4] = (0.9 * DX, 1.1 * DX)
            kwargs[f'd{axis}_profile'] = profile
    sim = Simulation(freq_max=15e9, domain=point(12, 8, 8), dx=DX,
                     boundary='pec', **kwargs)
    if kind == 'volume':
        sim.add(Box(point(6, 3, 3), point(8, 5, 5)), material='pec')
    elif kind == 'sheet':
        # Off-node declaration: only the realized x=6 plane owns the source.
        sim.add(Box(point(6.2, 3, 3), point(6.2, 5, 5)), material='pec')
    elif kind == 'wire':
        sim.add(PolylineWire((point(6, 4, 3), point(6, 4, 5)), radius=0),
                material='pec')
    position = point(5 if adjacent else 6, 4, 4)
    sim.add_source(position, 'ez', waveform=waveform, amplitude_kind='field')
    # H sources are internal declarations; add_source publicly admits E only.
    sim._ports[0] = replace(sim._ports[0], component=component)
    sim.add_probe(position, component)
    return sim


def execute(sim, path, *, entry='run', skip=True):
    kwargs = dict(n_steps=3, skip_preflight=skip)
    if 'distributed' in path:
        assert len(jax.devices()) >= 2
        kwargs['devices'] = jax.devices()[:2]
        if entry == 'forward':
            kwargs['distributed'] = True
    if entry == 'run':
        kwargs['compute_s_params'] = False
    return getattr(sim, entry)(**kwargs)


@pytest.mark.parametrize('entry,path', [('run', path) for path in PATHS] +
                         [('forward', path) for path in PATHS if path != 'distributed'])
@pytest.mark.parametrize('kind', ['volume', 'sheet', 'wire'])
@pytest.mark.parametrize('skip', [False, True])
def test_dead_e_source_refused(path, kind, skip, entry):
    sim = model(path, kind)
    with pytest.raises(ValueError, match=r'Soft source _ports\[0\]') as exc:
        execute(sim, path, entry=entry, skip=skip)
    message = str(exc.value)
    for text in ('ez', str(point(6, 4, 4)), 'grid index', 'geometry[0]',
                 'assembly', 'Move the source off the conductor, or use a port'):
        assert text in message


@pytest.mark.parametrize('path', PATHS)
def test_adjacent_a1_source_runs(path):
    result = execute(model(path, adjacent=True), path)
    assert np.max(np.abs(result.time_series)) > 0


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('component', ['hx', 'hy', 'hz'])
def test_dead_h_source_refused(path, component):
    with pytest.raises(ValueError, match='all four E edges of its curl loop'):
        execute(model(path, component=component), path)


@pytest.mark.parametrize('path', PATHS)
def test_h_source_with_only_one_pec_loop_edge_passes_admission(path):
    # The z filament owns one edge of the Hx curl loop, not the other three.
    sim = model(path, 'wire', component='hx')
    nu = 'graded' in path
    grid = sim._build_nonuniform_grid() if nu else sim._build_grid()
    solve_conductors(sim, grid, nonuniform=nu)


def test_h_source_with_only_one_pec_loop_edge_runs():
    # Only uniform single-device currently carries internal H soft sources
    # and wires; other lanes retain their existing feature refusals.
    result = execute(model('uniform', 'wire', component='hx'), 'uniform')
    assert np.max(np.abs(result.time_series)) > 0


def test_probe_remains_advisory():
    sim = model(adjacent=True)
    sim.add_probe(point(6, 4, 4), 'ez')
    assert sim.preflight().by_code('port_in_pec')
    execute(sim, 'uniform', skip=False)


@pytest.mark.parametrize('path', ['uniform', 'graded'])
def test_lumped_port_releases_its_edge(path):
    sim = model(path)
    sim._ports.clear()
    sim.add_port(point(6, 4, 4), 'ez', impedance=50., waveform=waveform)
    result = execute(sim, path)
    assert np.max(np.abs(result.time_series)) > 0


@pytest.mark.parametrize('solver', ['adi', 'subgrid'])
def test_experimental_paths_refuse_before_lane_dispatch(solver):
    sim = model()
    if solver == 'adi':
        sim._solver = 'adi'
    else:
        sim.add_refinement(z_range=(0., 4 * DX), ratio=2, validation='research')
    with pytest.raises(ValueError, match='Soft source'):
        execute(sim, 'uniform', skip=False)


def test_h_loop_joint_ownership_names_each_conductor():
    sim = model(kind='wire', component='hx')
    sim.add(PolylineWire((point(6, 5, 4), point(6, 5, 5)), radius=0), material='pec')
    sim.add(PolylineWire((point(6, 4, 4), point(6, 5, 4)), radius=0), material='pec')
    sim.add(PolylineWire((point(6, 4, 5), point(6, 5, 5)), radius=0), material='pec')
    with pytest.raises(ValueError, match='Soft source') as exc:
        execute(sim, 'uniform')
    for i in range(4):
        assert f'geometry[{i}]' in str(exc.value)


def test_check_reads_object_edges_instead_of_declared_shape():
    sim = model()
    root = realized_conductors(sim, sim._build_grid())
    empty = tuple(jnp.zeros_like(edge) for edge in root.pec_edges)
    refuse_dead_soft_sources(sim, replace(root, pec_edges=empty))
    with pytest.raises(ValueError, match='Soft source'):
        refuse_dead_soft_sources(sim, root)


@pytest.mark.parametrize('path', ['uniform', 'graded'])
def test_jitted_forward_on_concrete_mesh_still_refuses(path):
    sim = model(path)
    with pytest.raises(ValueError, match='Soft source'):
        jax.jit(lambda: sim.forward(n_steps=3, skip_preflight=True).time_series)()


@pytest.mark.parametrize('kind', ['thin_conductor', 'pinned_sheet'])
def test_sheet_provenance(kind):
    sim = model()
    sim._geometry.clear()
    if kind == 'thin_conductor':
        sim.add_thin_conductor(Box(point(6, 3, 3), point(6, 5, 5)))
    else:
        sim.add_pinned_sheet(plane_index=6, i_range=(3, 5), j_range=(3, 5), normal_axis=0)
    with pytest.raises(ValueError, match=rf'{kind}\[0\]'):
        execute(sim, 'uniform')


def test_traced_classification_reports_unavailable():
    sim = model()
    root = realized_conductors(sim, sim._build_grid())

    def check(edge):
        refuse_dead_soft_sources(sim, replace(root, pec_edges=(edge,) + root.pec_edges[1:]))
        return edge

    with pytest.warns(UserWarning, match='check cannot run on a traced mesh'):
        jax.make_jaxpr(check)(root.pec_edges[0])


def test_traced_mesh_reports_unavailable_with_concrete_edges():
    sim = model('graded')
    root = realized_conductors(sim, sim._build_nonuniform_grid(), nonuniform=True)

    def check(cells):
        grid = root.grid._replace(dx_arr=cells)
        refuse_dead_soft_sources(sim, replace(root, grid=grid))
        return cells

    with pytest.warns(UserWarning, match='check cannot run on a traced mesh'):
        jax.make_jaxpr(check)(root.grid.dx_arr)


@pytest.mark.parametrize('amplitude_kind', ['field', 'current'])
def test_vmap_fast_path_refuses_dead_source(amplitude_kind, monkeypatch):
    import rfx.vmap_sweep as sweep
    sim = model()
    sim._ports[0] = replace(sim._ports[0], amplitude_kind=amplitude_kind)
    sim.add_material('design', eps_r=2.)
    sim.add(Box(point(1, 1, 1), point(2, 2, 2)), material='design')

    def no_fallback(*args, **kwargs):
        pytest.fail('this case must exercise the batched fast path')

    monkeypatch.setattr(sweep, '_sequential_fallback', no_fallback)
    with pytest.raises(ValueError, match='Soft source'):
        sweep.vmap_material_sweep(sim, 'design.eps_r', [2., 3.], n_steps=3)


@pytest.mark.parametrize('kind', ['volume', 'sheet', 'wire'])
def test_vmap_adjacent_source_matches_scalar_run(kind):
    from rfx.vmap_sweep import vmap_material_sweep
    sim = model(kind=kind, adjacent=True)
    sim.add_material('design', eps_r=2.)
    sim.add(Box(point(1, 1, 1), point(2, 2, 2)), material='design')
    swept = vmap_material_sweep(sim, 'design.eps_r', [2.], n_steps=3)
    scalar = execute(sim, 'uniform')
    np.testing.assert_array_equal(swept.time_series[0], scalar.time_series)
    assert np.max(np.abs(swept.time_series)) > 0


def test_topology_optimize_refuses_dead_source():
    pytest.importorskip('optax')
    from rfx.topology import TopologyDesignRegion, topology_optimize
    sim = model()
    region = TopologyDesignRegion(corner_lo=point(1, 1, 1),
        corner_hi=point(2, 2, 2), material_bg='air', material_fg='fr4')
    with pytest.raises(ValueError, match='Soft source'):
        topology_optimize(sim, region, lambda result: -jnp.sum(result.time_series**2),
            n_iterations=1, skip_preflight=True, verbose=False)


def test_forward_concrete_pec_override_refuses_dead_source():
    sim = model()
    sim._geometry.clear()
    grid = sim._build_grid()
    mask = jnp.zeros(grid.shape, dtype=bool).at[6:8, 3:5, 3:5].set(True)
    with pytest.raises(ValueError, match='Soft source'):
        sim.forward(n_steps=3, skip_preflight=True, pec_mask_override=mask)


def test_no_conductor_traced_mesh_does_not_warn():
    import warnings
    sim = model('graded')
    sim._geometry.clear()
    root = realized_conductors(sim, sim._build_nonuniform_grid(), nonuniform=True)
    assert root.pec_edges is None

    def check(cells):
        refuse_dead_soft_sources(sim, replace(root, grid=root.grid._replace(dx_arr=cells)))
        return cells

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        jax.make_jaxpr(check)(root.grid.dx_arr)
    assert not caught


def test_unchanged_kernel_conductors_do_not_repeat_admission(monkeypatch):
    import rfx.model.source_admission as admission
    calls = []
    original = admission.refuse_dead_soft_sources

    def check(sim, root):
        calls.append(root)
        return original(sim, root)

    monkeypatch.setattr(admission, 'refuse_dead_soft_sources', check)
    execute(model(adjacent=True), 'uniform')
    assert len(calls) == 1
