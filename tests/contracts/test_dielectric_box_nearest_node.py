"""Material Box faces round to the nearest node on every CPU assembly lane.

Expected indices are calculated from the drawing, without raster/grid helpers.
Build probes stop after the real material builder, before time stepping. The
public realized record is read separately where the lane exposes one.
Not reached: GPU resonance/settling/passivity measurements; ADI refuses material
interfaces; subgrid forward refuses refinement; subgrid run refuses Lorentz poles;
graded/distributed runs refuse Kerr material; a periodic axis refuses a varying
profile (the graded periodic fixtures grade the other two axes). The subgrid fine lane has no
public fine realized record, so its actual rasterizer output is inspected.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, LorentzPole, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.rasterize_grid import (
    GridCoords, _material_cell_mask, cell_centres_from_nodes,
    cell_sizes_from_nonuniform_grid, cell_sizes_from_uniform_grid,
    centres_from_nonuniform_grid, centres_from_uniform_grid,
    coords_from_nonuniform_grid, coords_from_uniform_grid,
)
from tests._multi_device_scene import host_devices

DX = 1e-3
FACTORS = np.array([1., .8, 1.2, .9, 1.1, .8, 1.2, .9, 1.1, 1., 1., 1.])


def _scene(offset, graded=False, periodic=False, rich=False, smoothing=False, kerr=True, poles=True, **kw):
    widths = FACTORS * DX if graded else np.full(12, DX)
    nodes = np.r_[0., np.cumsum(widths)] if graded else np.arange(13) * DX
    lo, hi = nodes[3] + offset * widths[3], nodes[8] + offset * widths[8]
    sim = Simulation(10e9, (12 * DX,) * 3, dx=DX, cpml_layers=0,
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec') if periodic else 'pec',
                     **({f'd{a}_profile': (np.full(12, DX) if periodic and a == 'x' else widths)
                         for a in 'xyz'} if graded else {}), **kw)
    props = dict(eps_r=4.)
    if rich:
        props.update(sigma=.2, mu_r=1.5)
        if poles:
            props['lorentz_poles'] = [LorentzPole(2e10, 1e9, 4e20)]
        if kerr:
            props['chi3'] = 1e-20
    sim.add_material('box', **props)
    shape = Box((lo,) * 3, (hi,) * 3)
    sim.add(shape, material='box')
    # Nearest node of the DRAWING, with no implementation helper.
    nodes = (np.arange(13) * DX if periodic else nodes, nodes, nodes)
    indices = tuple(tuple(int(np.flatnonzero(np.isclose(abs(axis_nodes - face),
                                              np.min(abs(axis_nodes - face)), rtol=0, atol=1e-15))[0]) for face in (lo, hi))
                    for axis_nodes in nodes)
    expected = np.zeros((13, 13, 13), bool)
    expected[tuple(slice(first, last) for first, last in indices)] = True
    if periodic:
        expected = expected[:-1]
    return sim, shape, expected, indices, nodes


def _assert_record(sim, indices, nodes):
    entity = sim.realized_geometry().entities[0]
    assert entity.kind == 'material'
    for axis, axis_indices, axis_nodes in zip(entity.axes, indices, nodes):
        assert axis.node_range == axis_indices
        np.testing.assert_allclose(axis.bounds_m, axis_nodes[list(axis_indices)], rtol=0, atol=1e-16)


def _assert_arrays(built, expected, rich, *, kerr=True, poles=True):
    mats, debye, lorentz = built[:3]
    np.testing.assert_array_equal(np.asarray(mats.eps_r) == 4., expected)
    if rich:
        np.testing.assert_array_equal(np.asarray(mats.sigma) > 0, expected)
        np.testing.assert_array_equal(np.asarray(mats.mu_r) == 1.5, expected)
        assert debye is None
        if poles:
            assert lorentz is not None
            np.testing.assert_array_equal(lorentz[1][0], expected)
        if kerr:
            np.testing.assert_array_equal(np.asarray(built[-1]) > 0, expected)


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('periodic', [False, True], ids=['walls', 'periodic'])
@pytest.mark.parametrize('rich', [False, True], ids=['eps', 'sigma-mu-lorentz-kerr'])
@pytest.mark.parametrize('smoothing', [False, True], ids=['unsmoothed', 'smoothed'])
def test_assembled_and_realized_faces(offset, graded, periodic, rich, smoothing):
    from rfx.model.materials import assemble_cells
    sim, shape, expected, indices, nodes = _scene(offset, graded, periodic, rich, smoothing)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    built = assemble_cells(sim, grid)
    if smoothing:
        from rfx.geometry.smoothing import compute_smoothed_eps, compute_smoothed_eps_nonuniform
        smooth = compute_smoothed_eps_nonuniform if graded else compute_smoothed_eps
        components = smooth(grid, [(shape, 4.)])
        assert all(np.isfinite(np.asarray(component)).all() for component in components)
    _assert_arrays(built, expected, rich)
    _assert_record(sim, indices, nodes)


class _Built(Exception):
    pass


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('rich', [False, True], ids=['eps', 'sigma-mu-lorentz-kerr'])
@pytest.mark.parametrize('lane', ['uniform-run', 'uniform-forward', 'graded-run',
                                  'graded-forward', 'distributed_v2-run',
                                  'distributed_v2-forward', 'distributed_nu-run',
                                  'distributed_nu-forward', 'vmap-base'])
def test_public_build_reaches_same_cells(monkeypatch, offset, lane, rich):
    import rfx.model.materials as assembly
    graded = lane.startswith(('graded', 'distributed_nu'))
    kerr = lane.startswith('uniform')
    sim, _, expected, indices, nodes = _scene(offset, graded=graded, rich=rich, kerr=kerr)
    sim.add_source((DX, DX, DX), 'ez', amplitude_kind='current')
    sim.add_probe((2 * DX,) * 3, 'ez')
    devices = host_devices(2) if lane.startswith('distributed') else None
    original = assembly.assemble_cells
    seen = []

    def capture(*args, **kwargs):
        built = original(*args, **kwargs)
        seen.append(built)
        raise _Built

    with monkeypatch.context() as patch:
        patch.setattr(assembly, 'assemble_cells', capture)
        with pytest.raises(_Built):
            if lane == 'vmap-base':
                from rfx.vmap_sweep import vmap_material_sweep
                vmap_material_sweep(sim, 'box.eps_r', [4., 5.], n_steps=1)
            elif lane.endswith('forward'):
                sim.forward(n_steps=1, checkpoint=False, skip_preflight=True,
                            **(dict(distributed=True, devices=devices) if devices else {}))
            else:
                sim.run(n_steps=1, skip_preflight=True,
                        **(dict(devices=devices) if devices else {}))
    assert len(seen) == 1
    _assert_arrays(seen[0], expected, rich, kerr=kerr)
    _assert_record(sim, indices, nodes)


@pytest.mark.parametrize('offset', [.3, .5, .7])
def test_vmap_swept_material_mask(offset):
    from rfx.vmap_sweep import _build_batched_materials
    sim, _, expected, indices, nodes = _scene(offset)
    grid = sim._build_grid()
    base = sim._assemble_materials(grid)[0]
    batched = _build_batched_materials(sim, grid, base, 'box.eps_r', jnp.array([4., 5.]))
    for value, cells in zip((4., 5.), np.asarray(batched.eps_r)):
        np.testing.assert_array_equal(cells == value, expected)
    _assert_record(sim, indices, nodes)


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('rich', [False, True], ids=['eps', 'sigma-mu'])
def test_subgrid_coarse_and_fine_build(monkeypatch, offset, rich):
    import rfx.geometry.rasterize_grid as raster
    from rfx.runners.subgridded import _run_subgridded_once
    sim, _, expected, indices, nodes = _scene(offset, rich=rich, kerr=False, poles=False)
    sim.add_refinement((2 * DX, 10 * DX), ratio=2, validation='research')
    grid = sim._build_grid()
    coarse = sim._assemble_materials(grid)
    _assert_arrays(coarse, expected, rich, kerr=False, poles=False)
    _assert_record(sim, indices, nodes)
    original = raster.rasterize_geometry
    seen = []

    def capture(*args, **kwargs):
        built = original(*args, **kwargs)
        seen.append((args[2], built))
        raise _Built

    monkeypatch.setattr(raster, 'rasterize_geometry', capture)
    with pytest.raises(_Built):
        _run_subgridded_once(sim, grid, coarse[0], coarse[3], 1)
    centres, fine = seen[0]
    expected_fine = np.zeros(centres.shape, bool)
    drawing = sim._geometry[0].shape
    slices = []
    for a, c in enumerate(centres[:3]):
        fine_nodes = np.asarray(c) - DX / 4
        lo, hi = (int(np.argmin(abs(fine_nodes - face)))
                  for face in (drawing.corner_lo[a], drawing.corner_hi[a]))
        slices.append(slice(lo, hi))
    expected_fine[tuple(slices)] = True
    _assert_arrays(fine, expected_fine, rich, kerr=False, poles=False)


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('periodic', [False, True])
def test_traced_graded_coordinates(offset, periodic):
    sim, shape, expected, _, _ = _scene(offset, graded=True, periodic=periodic)
    grid = sim._build_nonuniform_grid()
    coords = coords_from_nonuniform_grid(grid)
    widths = cell_sizes_from_nonuniform_grid(grid)

    def sample(x, y, z, dx, dy, dz):
        co = GridCoords(x, y, z, grid.shape)
        sizes = (dx, dy, dz)
        return _material_cell_mask(shape, co, cell_centres_from_nodes(co, sizes),
                                   grid=grid if periodic else None, cell_sizes=sizes)

    np.testing.assert_array_equal(jax.jit(sample)(*coords[:3], *widths), expected)


def test_traced_dielectric_bounds_stay_refused():
    from rfx.geometry.rasterize_grid import TracedShapeDimensionError
    sim, _, _, _, _ = _scene(.3)
    grid = sim._build_grid()
    coords, centres = coords_from_uniform_grid(grid), centres_from_uniform_grid(grid)
    with pytest.raises(TracedShapeDimensionError, match='traced extent'):
        jax.jit(lambda lo: _material_cell_mask(
            Box((lo,) * 3, (.0083,) * 3), coords, centres, grid=grid))(.0033)


@pytest.mark.parametrize('entry', ['run', 'forward'])
def test_adi_not_reached_refuses_material_interface(entry):
    sim, *_ = _scene(.3, solver='adi')
    with pytest.raises(NotImplementedError, match='interface'):
        getattr(sim, entry)(n_steps=1, skip_preflight=True)


def test_subgrid_forward_not_reached_refuses_refinement():
    sim, *_ = _scene(.3)
    sim.add_refinement((.002, .010), ratio=2, validation='research')
    with pytest.raises(NotImplementedError, match='refinement|subgrid'):
        sim.forward(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('operation', ['union', 'difference', 'intersection'])
def test_csg_box_leaves(offset, operation):
    from rfx.geometry import csg
    sim, first, a, _, _ = _scene(offset)
    second = Box((.005 + offset * DX,) * 3, (.010 + offset * DX,) * 3)
    b = np.zeros(a.shape, bool)
    start, stop = (int(np.floor(v / DX + .5 - 1e-10)) for v in (second.corner_lo[0], second.corner_hi[0]))
    b[start:stop, start:stop, start:stop] = True
    expected = {'union': a | b, 'difference': a & ~b, 'intersection': a & b}[operation]
    np.testing.assert_array_equal(getattr(csg, operation)(first, second, sim._build_grid()), expected)


@pytest.mark.parametrize('traced', [False, True])
def test_box_composites_use_volume_leaves(traced):
    from rfx.geometry.curved import CurvedPatch
    from rfx.geometry.via import Via
    sim, _, _, _, _ = _scene(.3)
    grid = sim._build_grid()
    coords = coords_from_uniform_grid(grid)
    via = Via(center=(.0053, .0053), drill_radius=.002,
              pad_radius=.003, layers=[(.0033, .0083)])
    expected = np.zeros(grid.shape, bool)
    expected[3:7, 3:7, 3:8] = True
    sizes = cell_sizes_from_uniform_grid(grid)

    def sample(shape):
        def evaluate(x, y, z, dx, dy, dz):
            co = GridCoords(x, y, z, grid.shape)
            widths = (dx, dy, dz)
            return _material_cell_mask(shape, co, cell_centres_from_nodes(co, widths),
                                       grid=grid, cell_sizes=widths)
        return (jax.jit(evaluate) if traced else evaluate)(*coords[:3], *sizes)

    np.testing.assert_array_equal(sample(via), expected)
    patch = CurvedPatch(center=(.0053, .0053, .0033), length=.004,
                        width=.004, radius=.01)
    # Staircase leaves have zero z thickness: they own no volume cell.
    from rfx.errors import RfxRefusal
    with pytest.raises(RfxRefusal, match='ZERO cells'):
        sample(patch)


@pytest.mark.parametrize('offset', [.3, .5, .7])
def test_dual_average_build_uses_same_primal_widths(monkeypatch, offset):
    import rfx.geometry.rasterize_grid as raster
    from rfx.runners.nonuniform import assemble_interface_eps_nu
    sim, _, expected, indices, nodes = _scene(offset, graded=True, interface_eps='dual_average')
    grid = sim._build_nonuniform_grid()
    materials = sim._assemble_materials_nu(grid)[0]
    original = raster.rasterize_geometry
    seen = []

    def capture(*args, **kwargs):
        built = original(*args, **kwargs)
        seen.append(built)
        return built

    with monkeypatch.context() as patch:
        patch.setattr(raster, 'rasterize_geometry', capture)
        components = assemble_interface_eps_nu(sim, grid, materials, high_wall=(True,) * 3)
    assert len(components) == 3 and len(seen) == 1
    _assert_arrays(seen[0], expected, False)
    _assert_record(sim, indices, nodes)


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('graded', [False, True])
def test_periodic_image_uses_centre_window(offset, graded):
    sim, _, _, _, nodes = _scene(offset, graded=graded, periodic=True)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    coords = coords_from_nonuniform_grid(grid) if graded else coords_from_uniform_grid(grid)
    widths = cell_sizes_from_nonuniform_grid(grid) if graded else cell_sizes_from_uniform_grid(grid)
    centres = cell_centres_from_nodes(coords, widths)
    # The x interval crosses the seam; the drawing is extended by one period.
    xnodes, ynodes, _ = nodes
    period = xnodes[-1]
    lo = xnodes[10] + offset * (xnodes[11] - xnodes[10])
    hi = period + xnodes[2] + offset * (xnodes[3] - xnodes[2])
    shape = Box((lo, .0033, .0033), (hi, .0083, .0083))
    extended = np.r_[xnodes[:-1], period + xnodes]
    first, last = (int(np.flatnonzero(np.isclose(abs(extended - face),
                                          np.min(abs(extended - face)), rtol=0, atol=1e-15))[0]) for face in (lo, hi))
    expected = np.zeros(grid.shape, bool)
    ylo, yhi = (int(np.argmin(abs(ynodes - face))) for face in (.0033, .0083))
    expected[np.arange(first, last) % 12, ylo:yhi, ylo:yhi] = True
    np.testing.assert_array_equal(_material_cell_mask(
        shape, coords, centres, grid=grid, cell_sizes=widths), expected)


@pytest.mark.parametrize('graded', [False, True])
def test_smoothed_composite_fallback_uses_volume_leaves(graded):
    from rfx.geometry.via import Via
    from rfx.geometry.smoothing import compute_smoothed_eps, compute_smoothed_eps_nonuniform
    sim, _, _, _, nodes = _scene(.3, graded=graded)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    via = Via(center=(.0053, .0053), drill_radius=.002,
              pad_radius=.003, layers=[(.0033, .0083)])
    expected = np.zeros(grid.shape, bool)
    xlo, xhi, zlo, zhi = (int(np.argmin(abs(nodes[0] - face)))
                          for face in (.0033, .0073, .0033, .0083))
    expected[xlo:xhi, xlo:xhi, zlo:zhi] = True
    smooth = compute_smoothed_eps_nonuniform if graded else compute_smoothed_eps
    for component in smooth(grid, [(via, 4.)]):
        np.testing.assert_array_equal(np.asarray(component) == 4., expected)


def test_subgrid_lorentz_not_reached_refuses_poles():
    sim, *_ = _scene(.3, rich=True, kerr=False)
    sim.add_refinement((.002, .010), ratio=2, validation='research')
    with pytest.raises(NotImplementedError, match='Lorentz|lorentz'):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize('distributed', [False, True])
def test_graded_kerr_not_reached_refuses_nonlinearity(distributed):
    sim, *_ = _scene(.3, graded=True, rich=True)
    devices = host_devices(2) if distributed else None
    with pytest.raises(NotImplementedError, match='Kerr|chi3|χ³'):
        sim.run(n_steps=1, skip_preflight=True, **(dict(devices=devices) if devices else {}))


def test_varying_periodic_profile_not_reached_refuses_grading():
    sim = Simulation(10e9, (12 * DX,) * 3, dx=DX, cpml_layers=0,
                     boundary=BoundarySpec(x='periodic', y='pec', z='pec'),
                     dx_profile=FACTORS * DX)
    with pytest.raises(NotImplementedError, match='constant profile'):
        sim.run(n_steps=1, skip_preflight=True)


@pytest.mark.parametrize('offset', [.3, .5, .7])
def test_csg_rasterize_uses_material_cells(offset):
    from rfx.geometry.csg import rasterize
    sim, shape, expected, indices, nodes = _scene(offset)
    eps, sigma = rasterize(sim._build_grid(), [(shape, 4., .2)])
    np.testing.assert_array_equal(np.asarray(eps) == 4., expected)
    np.testing.assert_array_equal(np.asarray(sigma) > 0, expected)
    _assert_record(sim, indices, nodes)


@pytest.mark.parametrize('offset', [.3, .5, .7])
@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('periodic', [False, True])
def test_line_stub_uses_same_material_mask(offset, graded, periodic):
    from types import SimpleNamespace
    from rfx.preflight.line_stub import _permittivity
    sim, _, expected, indices, nodes = _scene(offset, graded=graded, periodic=periodic)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    coords = coords_from_nonuniform_grid(grid) if graded else coords_from_uniform_grid(grid)
    cache = {}
    point = tuple(axis_nodes[limits[0]] for axis_nodes, limits in zip(nodes, indices))
    assert _permittivity(sim, SimpleNamespace(eps_r_sub=1.), point, grid, coords, cache) == 4.
    np.testing.assert_array_equal(next(iter(cache.values())), expected)
    _assert_record(sim, indices, nodes)


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
@pytest.mark.parametrize('pec', [False, True], ids=['dielectric', 'pec-volume'])
def test_box_on_declared_domain_face_fills_last_cell(graded, pec):
    from rfx.geometry.rasterize_grid import pec_volume_cell_mask
    from rfx.model.materials import assemble_cells
    domain = (60.4 * DX, 6 * DX, 6 * DX)
    sim = Simulation(10e9, domain, dx=DX, cpml_layers=2, boundary='cpml',
                     **({'dz_profile': np.array([.9, 1.1] * 3) * DX} if graded else {}))
    sim.add_material('volume', **({'sigma': 1e9} if pec else {'eps_r': 4.}))
    box = Box((0.,) * 3, domain)
    sim.add(box, material='volume')
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    built = assemble_cells(sim, grid, include_cpml_pad_extension=False)
    if pec:
        # Inspect the declared volume before conductor continuation into CPML.
        centres = centres_from_nonuniform_grid(grid) if graded else centres_from_uniform_grid(grid)
        sizes = cell_sizes_from_nonuniform_grid(grid) if graded else cell_sizes_from_uniform_grid(grid)
        mask = np.asarray(pec_volume_cell_mask(box, centres, sizes, grid=grid))
    else:
        mask = np.asarray(built[0].eps_r) == 4.
    # The drawing ends on the domain face: ceil(60.4) is the bounding node.
    last_node = int(np.ceil(domain[0] / DX))
    pad = 2
    assert mask[pad + last_node - 1, pad + 2, pad + 2]
    assert not mask[pad + last_node, pad + 2, pad + 2]
    entity = sim.realized_geometry().entities[0]
    if pec:
        # The solved PEC body continues through the absorber past this face.
        assert entity.axes[0].bounds_m[1] >= last_node * DX
    else:
        assert entity.axes[0].node_range == (pad, pad + last_node)
        np.testing.assert_allclose(entity.axes[0].bounds_m, (0., last_node * DX), atol=1e-16)


ZERO_LANES = ['uniform-run', 'uniform-forward', 'graded-run', 'graded-forward',
              'distributed_v2-run', 'distributed_v2-forward', 'distributed_nu-run',
              'distributed_nu-forward', 'vmap-base', 'periodic', 'graded-periodic',
              'smoothed', 'graded-smoothed', 'subgrid-coarse', 'subgrid-fine',
              'dual-average', 'csg-rasterize', 'traced', 'line-stub', 'graded-line-stub']


@pytest.mark.parametrize('lane', ZERO_LANES)
@pytest.mark.parametrize('thickness', [0., .1], ids=['zero-thickness', 'no-centre'])
def test_zero_cell_material_is_refused(monkeypatch, lane, thickness):
    from rfx.errors import RfxRefusal
    import rfx.model.materials as assembly
    graded = lane.startswith(('graded', 'distributed_nu')) or lane == 'dual-average'
    sim, *_ = _scene(.3, graded=graded, periodic='periodic' in lane,
                     rich=True, kerr=False, poles=lane != 'subgrid-coarse',
                     **({'interface_eps': 'dual_average'} if lane == 'dual-average' else {}))
    # No centre lies in this drawing on either the coarse or half-size grid.
    from dataclasses import replace
    sim._geometry[0] = replace(sim._geometry[0], shape=Box(
        (2 * DX, 2 * DX, 3.1 * DX), (6 * DX, 6 * DX, (3.1 + thickness) * DX)))
    sim.add_source((DX,) * 3, 'ez', amplitude_kind='current')
    sim.add_probe((2 * DX,) * 3, 'ez')
    if lane == 'subgrid-coarse':
        sim.add_refinement((2 * DX, 10 * DX), ratio=2, validation='research')
    original = assembly.assemble_cells

    def stop_after_build(*args, **kwargs):
        original(*args, **kwargs)
        raise _Built('empty material was admitted')

    monkeypatch.setattr(assembly, 'assemble_cells', stop_after_build)
    # Traced coordinates report a runtime callback refusal through XLA.
    error = Exception if lane == 'traced' else RfxRefusal
    with pytest.raises(error, match='ZERO cells') as caught:
        if lane == 'subgrid-fine':
            from rfx.geometry.rasterize_grid import rasterize_geometry
            lines = np.arange(24) * DX / 2 + DX / 4
            centres = GridCoords(lines, lines, lines, (24,) * 3)
            rasterize_geometry(sim._geometry, sim._resolve_material, centres,
                               centres=centres, cell_sizes=(np.full(24, DX / 2),) * 3)
        elif lane == 'csg-rasterize':
            from rfx.geometry.csg import rasterize
            rasterize(sim._build_grid(), [(sim._geometry[0].shape, 4., .2)])
        elif lane.endswith('line-stub'):
            from types import SimpleNamespace
            from rfx.preflight.line_stub import _permittivity
            grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
            coords = coords_from_nonuniform_grid(grid) if graded else coords_from_uniform_grid(grid)
            _permittivity(sim, SimpleNamespace(eps_r_sub=1.), (3 * DX,) * 3, grid, coords, {})
        elif lane == 'traced':
            grid = sim._build_grid()
            co = coords_from_uniform_grid(grid)
            sizes = cell_sizes_from_uniform_grid(grid)
            def sample(x):
                coords = GridCoords(x, co.y, co.z, co.shape)
                return _material_cell_mask(sim._geometry[0].shape, coords,
                                           cell_centres_from_nodes(coords, sizes),
                                           cell_sizes=sizes, name='box')
            jax.jit(sample)(co.x).block_until_ready()
        elif lane in ('smoothed', 'graded-smoothed'):
            sim.run(n_steps=1, skip_preflight=True, subpixel_smoothing=True)
        elif lane == 'vmap-base':
            from rfx.vmap_sweep import vmap_material_sweep
            vmap_material_sweep(sim, 'box.eps_r', [4., 5.], n_steps=1)
        elif lane.endswith(('run', 'forward')) or lane == 'subgrid-coarse':
            devices = host_devices(2) if lane.startswith('distributed') else None
            if lane.endswith('forward'):
                sim.forward(n_steps=1, checkpoint=False, skip_preflight=True,
                            **(dict(distributed=True, devices=devices) if devices else {}))
            else:
                sim.run(n_steps=1, skip_preflight=True, **(dict(devices=devices) if devices else {}))
        else:
            grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
            original(sim, grid)
    assert 'drawn thickness' in str(caught.value)
    assert 'local cell' in str(caught.value)
    assert 'Refine' in str(caught.value)


@pytest.mark.parametrize('pec', [False, True])
def test_collapsed_axis_domain_clause_never_shrinks(pec):
    from rfx.model.materials import assemble_cells
    sim = Simulation(10e9, (8 * DX, 8 * DX, DX), dx=DX, cpml_layers=0,
                     boundary='pec', mode='2d_tmz')
    sim.add_material('volume', **({'sigma': 1e9} if pec else {'eps_r': 4.}))
    sim.add(Box((2 * DX, 2 * DX, 0.), (6 * DX, 6 * DX, DX)), material='volume')
    built = assemble_cells(sim, sim._build_grid())
    mask = np.asarray(built[3]) if pec else np.asarray(built[0].eps_r) == 4.
    assert mask.sum() == 16


def test_zero_thickness_on_domain_face_is_not_inflated():
    from rfx.errors import RfxRefusal
    sim = Simulation(10e9, (8 * DX, 8 * DX, 6.4 * DX), dx=DX, cpml_layers=0,
                     boundary='pec', snap='declared')
    sim.add_material('film', eps_r=4.)
    sim.add(Box((2 * DX, 2 * DX, 6.4 * DX), (6 * DX, 6 * DX, 6.4 * DX)), material='film')
    with pytest.raises(RfxRefusal, match="Material volume 'film'.*ZERO cells"):
        sim._assemble_materials(sim._build_grid())


def test_subgrid_fine_refuses_layer_resolved_only_on_coarse_grid():
    from rfx.errors import RfxRefusal
    from rfx.runners.subgridded import _run_subgridded_once
    from dataclasses import replace
    sim, *_ = _scene(.3)
    sim._geometry[0] = replace(sim._geometry[0], shape=Box(
        (2 * DX, 2 * DX, 3.45 * DX), (6 * DX, 6 * DX, 3.55 * DX)))
    sim.add_refinement((2 * DX, 10 * DX), ratio=2, validation='research')
    grid = sim._build_grid()
    coarse = sim._assemble_materials(grid)
    assert np.any(np.asarray(coarse[0].eps_r) == 4.)
    with pytest.raises(RfxRefusal, match="Material volume 'box'.*ZERO cells"):
        _run_subgridded_once(sim, grid, coarse[0], coarse[3], 1)


def test_graded_profile_domain_placeholder_does_not_extend_box():
    from rfx.errors import RfxRefusal
    sim = Simulation(10e9, (8 * DX, 8 * DX, 0.), dx=DX, cpml_layers=0,
                     boundary='pec', dz_profile=np.array([.9, 1.1] * 4) * DX)
    sim.add_material('film', eps_r=4.)
    sim.add(Box((2 * DX, 2 * DX, -DX), (6 * DX, 6 * DX, 0.)), material='film')
    with pytest.raises(RfxRefusal, match="Material volume 'film'.*ZERO cells"):
        sim._assemble_materials_nu(sim._build_nonuniform_grid())


@pytest.mark.parametrize('graded', [False, True])
def test_one_cell_axis_domain_clause_does_not_inflate_material(graded):
    from rfx.errors import RfxRefusal
    sim = Simulation(10e9, (8 * DX, 8 * DX, .4 * DX), dx=DX,
                     cpml_layers=0, boundary='pec', snap='declared',
                     **({'dz_profile': np.array([DX])} if graded else {}))
    sim.add_material('film', eps_r=4.)
    sim.add(Box((2 * DX, 2 * DX, 0.), (6 * DX, 6 * DX, .4 * DX)), material='film')
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    from rfx.model.materials import assemble_cells
    with pytest.raises(RfxRefusal, match="Material volume 'film'.*ZERO cells"):
        assemble_cells(sim, grid)
