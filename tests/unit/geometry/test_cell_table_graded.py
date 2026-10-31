"""Host-only graded formation, independent plane fractions, and admission fences."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rfx import Box, Sphere
from rfx.errors import RfxUnsupported
from rfx.geometry.cell_table import build_cell_table
from rfx.geometry.cell_table.body import _refuse_off_lattice_box
from rfx.geometry.conformal_cells import compute_conformal_fractions
from rfx.grid import Grid
from rfx.nonuniform import make_nonuniform_grid
from tests._cell_table_scenes import SCENES, cell_table_scene


def _graded(profiles, pad=0, periodic='', domain=None):
    # The public constructor pattern of test_rasterization_coordinate_exactness._nu_grid.
    return make_nonuniform_grid(
        tuple(float(np.sum(p)) for p in profiles[:2]), profiles[2], float(profiles[0][0]),
        cpml_layers=pad, dx_profile=profiles[0], dy_profile=profiles[1],
        periodic_axes=periodic, domain=domain)


@pytest.mark.parametrize('scene', [s for s in SCENES if s != 'lane'])
@pytest.mark.parametrize('pad', [0, 4])
def test_equal_cells_bitwise(scene, pad):
    original, shapes = cell_table_scene(scene)
    uniform = Grid(20e9, original.domain, dx=original.dx, cpml_layers=pad,
                   periodic_axes=original.periodic_axes)
    profiles = [np.full(uniform.shape[a] - getattr(uniform, f'pad_{name}_lo')
                        - getattr(uniform, f'pad_{name}_hi')
                        - (name not in uniform.periodic_axes), uniform.dx)
                for a, name in enumerate('xyz')]
    graded = _graded(profiles, pad, uniform.periodic_axes, uniform.domain)
    assert graded.shape == uniform.shape
    actual, expected = (build_cell_table(g, shapes) for g in (graded, uniform))
    for field in ('lengths', 'areas', 'edge_part_counts', 'face_part_counts'):
        for a, b in zip(getattr(actual, field), getattr(expected, field)):
            assert np.array_equal(a, b), (scene, field)
            assert a.dtype == b.dtype
    for field in ('part_owners', 'part_indices', 'part_fractions'):
        assert np.array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.cut_edges == expected.cut_edges
    assert actual.cut_faces == expected.cut_faces
    assert actual.topology_key == expected.topology_key


def _geometric_profiles():
    body = .0001 * 1.3 ** np.arange(18)
    # Equal boundary widths satisfy the public x/y constructor; the body grows by 1.3.
    profile = np.r_[np.full(3, body[-1]), body, np.full(8, body[-1])]
    return [profile * scale for scale in (1., 1.125, .875)]


def test_unequal_cells_closed_form(record_property):
    grid = _graded(_geometric_profiles())
    nodes = [np.array([grid.node_of(a, i) for i in range(grid.shape[a])])
             for a in range(3)]
    lo = tuple(n[5] + .37 * (n[6] - n[5]) for n in nodes)
    hi = tuple(n[18] + .61 * (n[19] - n[18]) for n in nodes)
    box = Box(lo, hi)
    table = build_cell_table(grid, [box])
    start = np.meshgrid(*(n[:-1] for n in nodes), indexing='ij', sparse=True)
    end = np.meshgrid(*(n[1:] for n in nodes), indexing='ij', sparse=True)
    width = [b - a for a, b in zip(start, end)]
    errors, counts = [], []
    physical = tuple(slice(1, len(n)) for n in nodes)
    for axis in range(3):
        for face in (box.corner_lo[axis], box.corner_hi[axis]):
            crossing = (start[axis] < face) & (face < end[axis])
            overlap = np.maximum(0., np.minimum(end[axis], box.corner_hi[axis])
                                 - np.maximum(start[axis], box.corner_lo[axis]))
            expected = 1. - overlap / width[axis]
            # Keep the entire owner one local cell away from every other Box face.
            # The crossing width also ensures this plane is the nearest SDF face.
            for kind, normal in [('edge', axis), *[('face', a) for a in range(3) if a != axis]]:
                mask = crossing.copy()
                for other in range(3):
                    if other == axis:
                        continue
                    stop = end[other] if kind == 'face' and other != normal else start[other]
                    margin = np.maximum(width[other], width[axis])
                    mask = mask & (start[other] - box.corner_lo[other] > margin)
                    mask = mask & (box.corner_hi[other] - stop > margin)
                values = (table.lengths if kind == 'edge' else table.areas)[normal][physical]
                mask = np.broadcast_to(mask, values.shape)
                count = int(np.count_nonzero(mask))
                assert count > 0, (axis, face, kind, normal)
                error = float(np.max(np.abs(values[mask] - np.broadcast_to(expected, values.shape)[mask])))
                counts.append(count)
                errors.append(error)
                assert error <= 1e-12, (axis, face, kind, normal, error)
    record_property('B3_owners', sum(counts))
    record_property('B3_max_abs_difference', max(errors))
    print(f'B3: owners={sum(counts)}, max |d|={max(errors):.17g}')


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('shift', [1, 3])
def test_translation_bitwise(graded, axis, shift):
    dx = 2. ** -10
    profile = dx * np.array([1.] * 6 + [.5] * 4 + [1.5] * 4 + [1.] * 10)
    profiles = [profile.copy() for _ in range(3)]
    if graded:
        grid = _graded(profiles)
        profiles[axis] = np.r_[np.full(shift, profile[0]), profile]
        moved_grid = _graded(profiles)
    else:
        grid = Grid(20e9, (24 * dx,) * 3, dx=dx, cpml_layers=0)
        domain = [24 * dx] * 3
        domain[axis] += shift * dx
        moved_grid = Grid(20e9, tuple(domain), dx=dx, cpml_layers=0)
    center = np.array([10.25, 10.5, 10.75]) * dx
    table = build_cell_table(grid, [Sphere(tuple(center), 3.25 * dx)])
    center[axis] += shift * dx
    moved = build_cell_table(moved_grid, [Sphere(tuple(center), 3.25 * dx)])
    old_slice = tuple(slice(3, n - 2) for n in grid.shape)
    new_slice = tuple(slice(s.start + (shift if a == axis else 0),
                            s.stop + (shift if a == axis else 0))
                      for a, s in enumerate(old_slice))
    for field in ('lengths', 'areas', 'edge_part_counts', 'face_part_counts'):
        for a, b in zip(getattr(table, field), getattr(moved, field)):
            assert np.array_equal(a[old_slice], b[new_slice]), field


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('signal', ['is_traced', 'cells'])
@pytest.mark.parametrize('empty', [False, True])
def test_traced_axis_refused_before_geometry(monkeypatch, axis, signal, empty):
    from rfx.geometry.cell_table import body

    def fail(*args, **kwargs):
        pytest.fail('traced grid reached node formation')

    monkeypatch.setattr(body, '_axis_node_positions', fail)
    concrete = _graded([np.full(12, .001) for _ in range(3)])
    def form(profile):
        # Trace only the selected store; constructing inside the trace traces all stores.
        grid = concrete._replace(**{('dx_arr', 'dy_arr', 'dz')[axis]: profile,
                                    ('dx_arr_f64', 'dy_arr_f64', 'dz_f64')[axis]: None})
        if signal == 'cells':
            grid = SimpleNamespace(cells=grid.cells, is_traced=lambda a: False)
        build_cell_table(grid, [] if empty else [Sphere((.006,) * 3, .002)])
        return 0.

    with pytest.raises(RfxUnsupported) as caught:
        jax.make_jaxpr(form)(concrete.cells(axis))
    assert str(caught.value) == f'cell geometry table requires concrete cells on axis {"xyz"[axis]}'


@pytest.mark.parametrize('axis', range(3))
def test_wrapper_keeps_uniform_refusal(monkeypatch, axis):
    from rfx.geometry import conformal_cells

    profiles = [np.full(12, .001) for _ in range(3)]
    profiles[axis][5] *= 1.3
    grid = _graded(profiles)
    def fail(*args, **kwargs):
        pytest.fail('wrapper called table before refusing unequal cells')
    monkeypatch.setattr(conformal_cells, 'build_cell_table', fail)
    with pytest.raises(NotImplementedError) as caught:
        compute_conformal_fractions(grid, [Sphere((.006,) * 3, .002)])
    assert type(caught.value) is NotImplementedError
    assert str(caught.value) == 'conformal PEC SC clamp requires uniform cells'


@pytest.mark.parametrize('uniform', [False, True])
def test_off_lattice_offset_uses_containing_cell(uniform):
    width = .001
    line = np.arange(8) * width if uniform else np.array([0., .001, .003, .007, .008])
    face = line[2] + .3 * (line[3] - line[2])
    box = Box((face, 0., 0.), (face, .008, .008))
    local = width if uniform else line[3] - line[2]
    expected = np.min(np.abs(line - face)) / local
    with pytest.raises(RfxUnsupported) as caught:
        _refuse_off_lattice_box(box, [(line, None, uniform)] * 3, [width] * 3,
                               (.01,) * 3, 'this zero-thickness sheet')
    assert f'x face {expected:.3g} cell off the node lattice' in str(caught.value)
    if uniform:
        old = np.min(np.abs(line - face)) / width
        assert np.float64(expected).tobytes() == np.float64(old).tobytes()


@pytest.mark.parametrize('entry', ['graded', 'waveguide_graded', 'forward'])
def test_product_refusals_unchanged(monkeypatch, entry):
    from rfx.model import tensor_operands
    from rfx.runners import nonuniform
    from tests.unit.runners.test_kottke_conformal_refusal import _scene

    def fail(*args, **kwargs):
        pytest.fail('graded conformal reached operand formation or stepping')

    monkeypatch.setattr(tensor_operands, 'form_tensor_operands', fail)
    monkeypatch.setattr(nonuniform, 'run_nonuniform', fail)
    sim = _scene('graded' if entry == 'forward' else entry, 'B')
    prefix = {
        'graded': 'Simulation.run() refuses 1 argument(s) the non-uniform mesh lane',
        'waveguide_graded': 'compute_waveguide_s_matrix() refuses 1 argument(s) the non-uniform waveguide S-matrix lane',
        'forward': 'the differentiable forward solve (forward/optimize or an S-matrix override) refuses 1 argument(s) the non-uniform forward lane',
    }[entry]
    expected = prefix + ' does not implement; running on without them would not be the run that was asked for:\n  conformal_pec=True: '
    if entry == 'graded':
        expected += ('the Dey-Mittra conformal weights are wired only into the uniform Yee lane, so PEC would be staircased. '
                     'Instead: pass conformal_pec=False and drop Boundary(conformal=True) (staircase PEC), '
                     'or run on a uniform mesh (no dx/dy/dz profile).')
    else:
        expected += ('Boundary(conformal=True) asks for Dey-Mittra conformal PEC, and this path has no conformal update, '
                     'so the walls and PEC shapes would be staircased. Instead: drop Boundary(conformal=True) (staircase PEC), or use ')
        expected += ('compute_waveguide_s_matrix() on a uniform mesh (no dx/dy/dz profile).' if entry == 'waveguide_graded'
                     else 'run() on a uniform mesh to keep conformal PEC for a forward-only result.')
    with pytest.raises(RfxUnsupported) as caught:
        if entry == 'waveguide_graded':
            sim.compute_waveguide_s_matrix(normalize=True, n_steps=1, skip_preflight=True,
                                           subpixel_smoothing=False)
        else:
            getattr(sim, 'forward' if entry == 'forward' else 'run')(
                n_steps=1, skip_preflight=True, subpixel_smoothing=False)
    assert type(caught.value) is RfxUnsupported
    assert str(caught.value) == expected


@pytest.mark.parametrize('pad', [0, 4])
@pytest.mark.parametrize('side', ['low', 'high', 'trailing'])
def test_halos_use_local_end_cell(pad, side):
    profiles = [np.full(8, .002), np.full(8, .002),
                np.array([.002] * 3 + [.001] * 3 + [.003] * 3)]
    grid = _graded(profiles, pad)
    low, high = grid.node_of(2, 0), grid.node_of(2, grid.nz - 1)
    if side == 'low':
        face, index = low - .25 * profiles[2][0], 0
        box = Box((-1., -1., face), (1., 1., 1.))
    else:
        face = high + (.25 if side == 'high' else 1.25) * profiles[2][-1]
        index = grid.nz + (side == 'trailing')
        box = Box((-1., -1., -1.), (1., 1., face))
    table = build_cell_table(grid, [box])
    for value in (table.lengths[2], table.areas[0], table.areas[1]):
        assert abs(value[4, 4, index] - .75) <= 1e-12


def test_graded_sheet_refusal_without_declared_domain():
    grid = _graded(_geometric_profiles())
    left, right = grid.node_of(0, 10), grid.node_of(0, 11)
    face = left + .3 * (right - left)
    box = Box((face, 0., 0.), (face, .01, .01))
    with pytest.raises(RfxUnsupported, match='x face 0.3 cell off the node lattice'):
        build_cell_table(grid, [box])


def test_empty_wrapper_still_returns_open_owners():
    grid = _graded(_geometric_profiles())
    for arrays in compute_conformal_fractions(grid, [], clamp=False):
        assert all(np.all(a == 1.) and a.shape == grid.shape for a in arrays)


def test_periodic_graded_halos_mirror_real_owners():
    profiles = [np.full(12, .001), np.full(12, .001),
                .001 * np.array([1., 2., 3., 2., 1.5, 2.5])]
    grid = _graded(profiles, periodic='z')
    table = build_cell_table(grid, [Sphere((.006, .006, .0005), .003)])
    for field in ('lengths', 'areas', 'edge_part_counts', 'face_part_counts'):
        for axis, values in enumerate(getattr(table, field)):
            # The high owner uses the trailing node as its far endpoint.
            for halo, physical in ((0, grid.nz), (grid.nz + 1, 1)):
                assert np.array_equal(values[:, :, halo], values[:, :, physical]), (field, axis, halo)
    assert any(np.any((values[:, :, 1] > 0.) & (values[:, :, 1] < 1.))
               for values in table.lengths)


def test_body_inside_coarse_cell_holds_no_node():
    profile = np.r_[np.full(4, .0005), np.full(3, .004), np.full(4, .0005)]
    grid = _graded([profile.copy() for _ in range(3)])
    center = sum(profile[:5]) + .5 * profile[5]
    with pytest.raises(NotImplementedError, match='the Sphere holds no grid node'):
        build_cell_table(grid, [Sphere((center,) * 3, .0009)])
