"""Declared PEC-wall source admission: owner, paths, snapping and falsifiers."""
from dataclasses import replace
import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries import pec
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import FDTDState
from rfx.model import source_admission as admission
from rfx.model.conductors import realized_conductors, solve_conductors

DX = 1 / 1024
DOMAIN = (20 * DX, 16 * DX, 12 * DX)
FACES = tuple(f'{a}_{side}' for a in 'xyz' for side in ('lo', 'hi'))
PATHS = ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward',
         'distributed_run', 'distributed_graded_run', 'distributed_graded_forward',
         'adi_run', 'adi_forward', 'subgrid_run')
WALL_CASES = (
    ('x_lo', (0, 7, 5), 'ez'), ('x_lo', (0, 7, 5), 'ey'),
    ('x_hi', (20, 7, 5), 'ez'), ('y_lo', (6, 0, 5), 'ez'),
    ('y_hi', (6, 16, 5), 'ez'), ('z_lo', (6, 7, 0), 'ex'),
    ('z_hi', (6, 7, 12), 'ey'), ('x_lo,y_lo', (0, 0, 5), 'ez'))


def model(path='uniform_run', *, boundary='pec'):
    kw = {}
    if 'graded' in path:
        kw['dy_profile'] = np.array([1] + [.9, 1.1] * 7 + [1]) * DX
    if path.startswith('adi'):
        kw['solver'] = 'adi'
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary=boundary, **kw)
    if path.startswith('subgrid'):
        sim.add_refinement((0, DOMAIN[2]), ratio=2, validation='off')
    return sim


def execute(sim, path, *, steps=1):
    kw = dict(n_steps=steps, skip_preflight=True)
    if path.startswith('distributed'):
        if len(jax.devices('cpu')) < 2:
            pytest.skip('requires two real CPU devices')
        kw['devices'] = jax.devices('cpu')[:2]
    if path.endswith('forward'):
        kw['checkpoint'] = False
        if path.startswith('distributed'):
            kw['distributed'] = True
        return sim.forward(**kw)
    return sim.run(compute_s_params=False, **kw)


def expected_masks(shape, faces):
    # Independent coordinate construction, including all rims.
    indices = np.indices(shape)
    return tuple(np.any(np.stack([
        indices['xyz'.index(face[0])] ==
        (0 if face.endswith('lo') else shape['xyz'.index(face[0])] - 1)
        for face in faces if face[0] != component] + [np.zeros(shape, bool)]), axis=0)
        for component in 'xyz')


def check_owner(faces):
    shape = (7, 5, 3)
    expected = expected_masks(shape, faces)
    actual = pec.wall_edge_masks(shape, faces)
    one = jnp.ones(shape)
    state = pec.apply_pec_faces(FDTDState(*(one for _ in range(6)), jnp.array(0)), set(faces))
    for c, (want, got) in enumerate(zip(expected, actual)):
        np.testing.assert_array_equal(got, want)
        zeroed = np.asarray(state[c]) == 0
        selection = np.ones(shape, bool)
        if c == 2 and 'z_hi' in faces:
            selection[:, :, -1] = False  # Only the extra normal ghost row.
        np.testing.assert_array_equal(got[selection], zeroed[selection])


@pytest.mark.parametrize('bits', range(64))
def test_b1_owner(bits):
    check_owner(tuple(f for n, f in enumerate(FACES) if bits & (1 << n)))


@pytest.mark.parametrize('path', ('uniform_run', 'graded_run', 'distributed_run',
                                  'distributed_graded_run'))
def test_b2_path_zero_sets(path, record_property):
    from tests.contracts.boundary_fields import observe_scans, measured_fields
    sim = model(path)
    grid = sim._build_nonuniform_grid() if 'graded' in path else sim._build_grid()
    records = []
    distributed = path.startswith('distributed')
    with observe_scans(records, grid.shape if distributed else None):
        result = execute(sim, path)
        jax.block_until_ready(result.time_series)
        jax.effects_barrier()
    values, _, _ = measured_fields(records, grid, 'distributed' if distributed else 'run')
    # Variant 11 starts with every E entry = 1, H = 0. Exactly one step.
    fields = values[11, :3]
    masks = pec.wall_edge_masks(grid.shape, FACES)
    differences = []
    equal_sets = []
    counts = {}
    for face, c in itertools.product(FACES, range(3)):
        axis = 'xyz'.index(face[0])
        if c == axis:
            continue
        plane = [slice(None)] * 3
        plane[axis] = 0 if face.endswith('lo') else -1
        plane = tuple(plane)
        actual = fields[c][plane] == 0
        want = masks[c][plane]
        equal_sets.append(np.array_equal(actual, want))
        counts[f'{face}:e{"xyz"[c]}'] = int(actual.sum())
        for index in np.argwhere(actual != want):
            differences.append([face, c, index.tolist(), bool(actual[tuple(index)])])
    record_property('b2_sets', json.dumps(dict(path=path, counts=counts, differences=differences)))
    assert all(equal_sets), f'B2 {path}: {differences}'


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('case', WALL_CASES)
@pytest.mark.parametrize('amplitude_kind', ('current', 'field'))
def test_b3_soft_wall(path, case, amplitude_kind):
    if path.startswith('adi') and amplitude_kind == 'current':
        pytest.skip('ADI does not take current-amplitude sources')
    face, position, component = case
    sim = model(path)
    pos = tuple(v * DX for v in position)
    sim.add_source(pos, component, amplitude_kind=amplitude_kind)
    sim.add_probe((4 * DX, 7 * DX, 5 * DX), component)
    sim.add_probe((13 * DX, 9 * DX, 4 * DX), component)
    with pytest.raises(ValueError, match=r'Soft source _ports\[0\]') as exc:
        execute(sim, path)
    for text in (*face.split(','), str(pos), component, 'grid index',
                 'at least one cell inside the domain, off the wall'):
        assert text in str(exc.value)


@pytest.mark.parametrize('path', ('uniform_run', 'uniform_forward', 'graded_run',
                                  'graded_forward', 'distributed_run', 'subgrid_run'))
@pytest.mark.parametrize('extent', (None, 4 * DX))
def test_b3_port_wall(path, extent):
    sim = model(path)
    sim.add_port((0, 7 * DX, 4 * DX), 'ez', impedance=50, extent=extent)
    with pytest.raises(ValueError, match=r'port _ports\[0\].*x_lo'):
        execute(sim, path)


@pytest.mark.parametrize('path', ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward'))
def test_b3_mixed_record(path):
    sim = model(path, boundary=BoundarySpec(x='pec', y='cpml', z='pmc'))
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='current')
    with pytest.raises(ValueError, match='x_lo'):
        execute(sim, path)


@pytest.mark.parametrize('path', ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward'))
@pytest.mark.parametrize('component', ('ex', 'ey'))
def test_z_lo_pec_other_faces_cpml_refuses(path, component):
    sim = model(path, boundary=BoundarySpec(x='cpml', y='cpml',
                                            z=Boundary(lo='pec', hi='cpml')))
    sim.add_source((6 * DX, 7 * DX, 0), component, amplitude_kind='current')
    with pytest.raises(ValueError, match='z_lo') as exc:
        execute(sim, path)
    assert 'shorted' in str(exc.value)


@pytest.mark.parametrize('path', ('uniform_run', 'uniform_forward', 'graded_run', 'graded_forward'))
def test_z_lo_pec_other_faces_cpml_admits_cpml_source(path):
    """The same mixed box admits its CPML-face source; evidence checks base bits."""
    sim = model(path, boundary=BoundarySpec(x='cpml', y='cpml',
                                            z=Boundary(lo='pec', hi='cpml')))
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='current')
    sim.add_probe((4 * DX, 7 * DX, 5 * DX), 'ez')
    sim.add_probe((13 * DX, 9 * DX, 4 * DX), 'ez')
    result = execute(sim, path, steps=2)
    assert np.asarray(result.time_series).shape == (2, 2)
    assert np.all(np.isfinite(result.time_series))


@pytest.mark.parametrize('mode,component', [
    ('2d_tmz', c) for c in ('ez', 'ex', 'ey', 'hx', 'hy')
] + [('2d_tez', c) for c in ('ex', 'ey', 'hz')])
@pytest.mark.parametrize('boundary', ('pec', 'cpml'))
def test_invariant_axis_sources_admitted(mode, component, boundary):
    """Equivalent invariant-axis kinds are not declared walls; base bits are captured separately."""
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, mode=mode, boundary=boundary)
    sim.add_source((6 * DX, 7 * DX, 0), component if component.startswith('e') else 'ez',
                   amplitude_kind='field')
    if component.startswith('h'):
        # add_source accepts only E; exercise the existing internal H carrier.
        sim._ports[0] = replace(sim._ports[0], component=component)
    sim.add_probe((4 * DX, 7 * DX, 0), component)
    sim.add_probe((13 * DX, 9 * DX, 0), component)
    result = execute(sim, 'uniform_run', steps=2)
    assert np.asarray(result.time_series).shape == (2, 2)
    assert np.all(np.isfinite(result.time_series))


@pytest.mark.parametrize('mode,component', (('2d_tmz', 'ez'), ('2d_tez', 'ey')))
def test_two_d_declared_in_plane_wall_still_refuses(mode, component):
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, mode=mode, boundary='pec')
    sim.add_source((0, 7 * DX, 0), component, amplitude_kind='field')
    with pytest.raises(ValueError, match='x_lo') as exc:
        execute(sim, 'uniform_run')
    assert 'z_lo' not in str(exc.value) and 'z_hi' not in str(exc.value)


@pytest.mark.parametrize('graded', (False, True))
@pytest.mark.parametrize('with_sources', (False, True))
def test_assembly_reads_boundary_only_once_when_declared(monkeypatch, graded, with_sources):
    sim = model('graded_run' if graded else 'uniform_run')
    if with_sources:
        sim.add_source((6 * DX, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
        sim.add_source((8 * DX, 9 * DX, 4 * DX), 'ey', amplitude_kind='field')
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    calls = []
    original = Simulation.boundary_model
    def spy(self):
        calls.append(self)
        return original(self)
    monkeypatch.setattr(Simulation, 'boundary_model', spy)
    solve_conductors(sim, grid, nonuniform=graded)
    assert calls == ([sim] if with_sources else [])


@pytest.mark.parametrize('path', ('uniform_run', 'graded_run'))
def test_absorber_backing_plane_source_admitted(path):
    sim = model(path, boundary='cpml')
    graded = path == 'graded_run'
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    assert grid.pad_x_lo > 0
    position = (-grid.pad_x_lo * DX, 7 * DX, 5 * DX)
    if graded:
        from rfx.nonuniform import position_to_index
        cell = position_to_index(grid, position)
    else:
        cell = grid.position_to_index(position)
    # The graded lookup clamps exterior coordinates to the physical interface.
    # Only uniform placement reaches the backing and discriminates falsifier (d).
    assert cell[0] == (grid.pad_x_lo if graded else 0)
    sim.add_source(position, 'ez', amplitude_kind='current')
    sim.add_probe((4 * DX, 7 * DX, 5 * DX), 'ez')
    sim.add_probe((13 * DX, 9 * DX, 4 * DX), 'ez')
    result = execute(sim, path, steps=2)
    assert np.asarray(result.time_series).shape == (2, 2)
    assert np.all(np.isfinite(result.time_series))


@pytest.mark.parametrize('path', ('uniform_run', 'graded_run'))
@pytest.mark.parametrize('x,face', ((-0.4 * DX, 'x_lo'), (DOMAIN[0] + 0.4 * DX, 'x_hi')))
def test_outside_coordinate_reports_domain_and_wall(path, x, face):
    sim = model(path)
    sim.add_source((x, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
    with pytest.raises(ValueError, match=face) as exc:
        execute(sim, path)
    assert f'Declared x coordinate {x} is outside domain [0, {DOMAIN[0]}]' in str(exc.value)
    assert 'at least one cell inside the domain, off the wall' in str(exc.value)
    assert 'orient it normal' not in str(exc.value)


CONTROL_PATHS = [(path, case) for path in PATHS
                 for case in ('normal', 'inside', 'cpml', 'pmc', 'periodic')
                 if not (path.startswith(('adi', 'subgrid')) and
                         case not in ('normal', 'inside'))
                 and not (path.startswith('distributed') and case in ('pmc', 'periodic'))
                 and not ('graded' in path and case == 'periodic')]


@pytest.mark.parametrize('path,case', CONTROL_PATHS)
def test_b3_control_admission(path, case):
    """Supported controls run; BEFORE/AFTER evidence checks their exact bits."""
    boundary = (BoundarySpec(x=case, y='pec', z='pec')
                if case in ('cpml', 'pmc', 'periodic') else 'pec')
    sim = model(path, boundary=boundary)
    component = 'ex' if case == 'normal' else 'ez'
    sim.add_source((DX if case == 'inside' else 0, 7 * DX, 5 * DX), component,
                   amplitude_kind='field' if path.startswith('adi') else 'current')
    sim.add_probe((4 * DX, 7 * DX, 5 * DX), component)
    sim.add_probe((13 * DX, 9 * DX, 4 * DX), component)
    result = execute(sim, path, steps=2)
    assert np.asarray(result.time_series).shape == (2, 2)
    assert np.all(np.isfinite(result.time_series))


@pytest.mark.parametrize('solver', ('yee', 'adi'))
def test_private_material_forward_refuses(solver):
    sim = model('adi_forward' if solver == 'adi' else 'uniform_forward')
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
    grid = sim._build_grid()
    root = realized_conductors(sim, grid)
    with pytest.raises(ValueError, match='x_lo'):
        sim._forward_from_materials(grid, root.materials, None, None,
                                    n_steps=1, checkpoint=False)


def test_vmap_wall_refuses():
    from rfx.vmap_sweep import vmap_material_sweep
    sim = model()
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='current')
    sim.add_probe((4 * DX, 7 * DX, 5 * DX), 'ez')
    with pytest.raises(ValueError, match='x_lo'):
        vmap_material_sweep(sim, 'eps_r', [1., 2.], n_steps=1)


@pytest.mark.parametrize('graded', (False, True))
@pytest.mark.parametrize('side,offset,expected', [
    ('lo', .49, 0), ('lo', .50, 0), ('lo', .51, 1),
    ('hi', .49, 20), ('hi', .50, 19), ('hi', .51, 19)])
def test_b4_kernel_snapping(graded, side, offset, expected):
    sim = model('graded_run' if graded else 'uniform_run')
    x = (offset if side == 'lo' else 20 - offset) * DX
    sim.add_source((x, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    if graded:
        from rfx.runners.nonuniform import pos_to_nu_index
        cell = pos_to_nu_index(grid, sim._ports[0].position)
    else:
        cell = grid.position_to_index(sim._ports[0].position)
    assert cell[0] == expected
    if expected in (0, 20):
        with pytest.raises(ValueError, match=f'x_{side}'):
            solve_conductors(sim, grid, nonuniform=graded)
    else:
        solve_conductors(sim, grid, nonuniform=graded)


@pytest.mark.parametrize('kind', ('pec', 'cpml', 'pmc', 'periodic'))
def test_record_controls_and_preflight(kind):
    sim = model(boundary=BoundarySpec(x=kind, y='pec', z='pec'))
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='current')
    report = sim.preflight()
    hits = report.by_code('source_decoupled')
    if kind == 'pec':
        assert len(hits) == 1 and hits[0].severity == 'error'
        with pytest.raises(ValueError, match='x_lo'):
            sim.run(n_steps=1, compute_s_params=False)
    else:
        assert all(h.severity == 'warning' for h in hits)
        solve_conductors(sim, sim._build_grid())


def test_geometry_release_does_not_release_wall():
    sim = model()
    sim.add(Box((6 * DX, 5 * DX, 3 * DX), (8 * DX, 9 * DX, 7 * DX)), material='pec')
    sim.add_port((6 * DX, 7 * DX, 5 * DX), 'ez', impedance=50)
    root = solve_conductors(sim, sim._build_grid())
    assert root.pec_edges[2][6, 7, 5]  # Admission has not changed the product.
    sim.add_port((0, 7 * DX, 5 * DX), 'ez', impedance=50)
    with pytest.raises(ValueError, match=r'_ports\[1\].*x_lo'):
        solve_conductors(sim, sim._build_grid())


def test_wire_with_live_edges_is_admitted():
    sim = model()
    sim.add_port((0, 7 * DX, 5 * DX), 'ex', impedance=50, extent=4 * DX)
    solve_conductors(sim, sim._build_grid())


def test_h_loop_uses_union():
    sim = model()
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
    sim._ports[0] = replace(sim._ports[0], component='hx')
    with pytest.raises(ValueError, match='all four E edges.*x_lo'):
        solve_conductors(sim, sim._build_grid())


def test_traced_position_warns_without_classification():
    sim = model()
    sim.add_source((0, 7 * DX, 5 * DX), 'ez', amplitude_kind='field')
    root = realized_conductors(sim, sim._build_grid())
    def traced(x):
        original = sim._ports[0]
        sim._ports[0] = replace(original, position=(x, 7 * DX, 5 * DX))
        try:
            admission.refuse_dead_soft_sources(sim, root)
        finally:
            sim._ports[0] = original
        return x
    with pytest.warns(UserWarning, match='cannot run on a traced mesh'):
        jax.make_jaxpr(traced)(0.)


def test_falsifier_a_conductor_only_is_red(monkeypatch):
    monkeypatch.setattr(admission, 'declared_pec_faces', lambda sim: ())
    with pytest.raises(pytest.fail.Exception, match='DID NOT RAISE'):
        test_b3_soft_wall('uniform_run', WALL_CASES[0], 'current')


def test_falsifier_b_missing_rims_is_red(monkeypatch):
    original = pec.wall_edge_masks
    def missing_rims(shape, faces):
        masks = original(shape, faces)
        for mask in masks:
            mask[:, 0, :] = False
        return masks
    monkeypatch.setattr(pec, 'wall_edge_masks', missing_rims)
    with pytest.raises(AssertionError):
        check_owner(('x_lo',))


def test_falsifier_c_boundary_string_is_red(monkeypatch):
    monkeypatch.setattr(admission, 'declared_pec_faces',
                        lambda sim: FACES if sim._boundary == 'pec' else ())
    with pytest.raises(pytest.fail.Exception, match='DID NOT RAISE'):
        test_b3_mixed_record('uniform_run')


def test_falsifier_d_absorber_backings_are_red(monkeypatch):
    from rfx.boundaries.model import electric_faces
    monkeypatch.setattr(admission, 'declared_pec_faces',
                        lambda sim: electric_faces(sim.boundary_model()))
    with pytest.raises(ValueError, match='x_lo'):
        test_absorber_backing_plane_source_admitted('uniform_run')
