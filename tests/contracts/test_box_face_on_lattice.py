"""Float arithmetic at a Box face keeps its intended cells (#1138).

Build only: dielectric material assembly, PEC centre occupancy, and one
periodic image. The expected indices come from the declared cell counts,
not from the production window. No field is time-stepped.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._grid_metric import NODE_TIE_REL, half_open_volume_mask
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.rasterize_grid import (
    _box_axis_volume,
    cell_sizes_from_nonuniform_grid,
    cell_sizes_from_uniform_grid,
    centres_from_nonuniform_grid,
    centres_from_uniform_grid,
    coords_from_nonuniform_grid,
    coords_from_uniform_grid,
)
from tests._x64_compat import enable_x64


DXS = (787e-6, 25.4e-6 / 3, 1e-3)
SPELLINGS = ("m*dx", "lo+N*dx", "repeated-addition", "(dom-g)/2")
FIRST = 3
# Unequal cells both inside and outside the sampled span, and a noninteger
# total in units of dx. Graded faces are prefixes of these primal widths.
FACTORS = np.array([
    1.0, .9, 1.1, .95, 1.2, 1.0, .85, 1.1, .9, 1.2, 1.05, .8,
    1.0, 1.15, .95, 1.2, 1.05, .85, 1.07, 1.0,
])


def _declaration(dx, graded):
    widths = FACTORS * dx if graded else np.full(20, dx)
    edges = (np.r_[0., np.cumsum(widths)] if graded
             else np.arange(21, dtype=np.float64) * dx)
    domain = float(np.sum(widths)) if graded else 27.37 * dx
    return widths, edges, domain


def _faces(dx, count, spelling, graded):
    widths, edges, domain = _declaration(dx, graded)
    stop = FIRST + count
    if spelling == "m*dx":
        # On a graded axis, m is the prefix sum in units of dx.
        factors = FACTORS if graded else np.ones(20)
        lo = float(np.sum(factors[:FIRST])) * dx
        hi = float(np.sum(factors[:stop])) * dx
    elif spelling == "lo+N*dx":
        lo = float(np.sum(widths[:FIRST])) if graded else FIRST * dx
        span = float(np.sum(widths[FIRST:stop])) if graded else count * dx
        hi = lo + span
    elif spelling == "repeated-addition":
        lo = 0.
        for d in widths[:FIRST]:
            lo += float(d)
        hi = lo
        for d in widths[FIRST:stop]:
            hi += float(d)
    else:
        lo = (domain - (domain - 2 * edges[FIRST])) / 2
        hi = (domain - (domain - 2 * edges[stop])) / 2
    return float(lo), float(hi)


def _model(dx, graded):
    widths, _, domain = _declaration(dx, graded)
    sim = Simulation(
        10e9, (6.3 * dx, 7.8 * dx, domain), dx=dx, cpml_layers=5,
        boundary=BoundarySpec(
            x=Boundary("cpml", "cpml", lo_thickness=1, hi_thickness=3),
            y=Boundary("cpml", "cpml", lo_thickness=4, hi_thickness=2),
            z=Boundary("cpml", "cpml", lo_thickness=2, hi_thickness=5)),
        **({"dz_profile": widths} if graded else {}))
    sim.add_material("laminate", eps_r=3.7)
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    coords = (coords_from_nonuniform_grid(grid) if graded
              else coords_from_uniform_grid(grid))
    centres = (centres_from_nonuniform_grid(grid, coords) if graded
               else centres_from_uniform_grid(grid))
    sizes = (cell_sizes_from_nonuniform_grid(grid) if graded
             else cell_sizes_from_uniform_grid(grid))
    assert grid.pad_z_lo == 2 and grid.pad_z_hi == 5
    assert not np.isclose(domain / dx, round(domain / dx), rtol=0, atol=1e-10)
    return sim, grid, coords, centres, sizes


def _add_and_build(sim, grid, lo, hi, *, graded, material="laminate"):
    dx = grid.dx
    shape = Box((dx, 2 * dx, lo), (3 * dx, 5 * dx, hi))
    sim.add(shape, material=material)
    built = (sim._assemble_materials_nu(grid) if graded
             else sim._assemble_materials(grid))
    mask = np.asarray(built[3] if material == "pec" else built[0].eps_r > 1)
    return shape, np.any(mask, axis=(0, 1))


def _assert_span(mask, first, count):
    indices = np.flatnonzero(np.asarray(mask))
    assert indices.size == count
    assert indices[0] == first
    np.testing.assert_array_equal(indices, np.arange(first, first + count))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("dx", DXS)
@pytest.mark.parametrize("count", range(1, 13))
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_material_box_and_cell_centres_keep_first_index_and_count(
        graded, dx, count, spelling):
    sim, grid, coords, centres, sizes = _model(dx, graded)
    lo, hi = _faces(dx, count, spelling, graded)
    shape, material = _add_and_build(sim, grid, lo, hi, graded=graded)
    first = grid.pad_z_lo + FIRST
    _assert_span(material, first, count)
    node_mask = (shape.mask_on_coords(*coords[:3]) if graded else shape.mask(grid))
    _assert_span(np.any(np.asarray(node_mask), axis=(0, 1)), first, count)
    _assert_span(_box_axis_volume(centres.z, lo, hi, sizes[2]), first, count)


def test_arithmetic_matrix_contains_low_and_high_faces_above_their_nodes():
    above = [0, 0]
    off_node = [0, 0]
    for graded in (False, True):
        for dx in DXS:
            _, grid, coords, _, _ = _model(dx, graded)
            for count in range(1, 13):
                for spelling in SPELLINGS:
                    for side, (face, index) in enumerate(zip(
                            _faces(dx, count, spelling, graded), (FIRST, FIRST + count))):
                        node = coords.z[grid.pad_z_lo + index]
                        above[side] += int(face - node >= np.spacing(node))
                        off_node[int(graded)] += int(abs(face - node) >= np.spacing(node))
    # Both faces must exercise the direction the old exact window misses;
    # a matrix containing only identical or downward-rounded faces is silent.
    assert all(above), above
    assert all(off_node), off_node


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("dx", DXS)
@pytest.mark.parametrize("count", range(1, 13))
def test_centre_aligned_faces_keep_the_half_open_window(graded, dx, count):
    _, grid, _, centres, sizes = _model(dx, graded)
    first = grid.pad_z_lo + FIRST
    lo = np.nextafter(centres.z[first], np.inf)
    hi = np.nextafter(centres.z[first + count], np.inf)
    _assert_span(_box_axis_volume(centres.z, lo, hi, sizes[2]), first, count)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_pec_assembly_uses_the_centre_face_band(graded):
    sim, grid, _, centres, _ = _model(DXS[0], graded)
    first = grid.pad_z_lo + FIRST
    lo = np.nextafter(centres.z[first], np.inf)
    hi = np.nextafter(centres.z[first + 4], np.inf)
    _, cells = _add_and_build(sim, grid, lo, hi, graded=graded, material="pec")
    _assert_span(cells, first, 4)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_half_cell_face_is_not_pulled_to_a_node(graded):
    sim, grid, coords, centres, sizes = _model(DXS[0], graded)
    first = grid.pad_z_lo + FIRST
    lo = coords.z[first] + .5 * sizes[2][first]
    hi = coords.z[first + 4]
    _, material = _add_and_build(sim, grid, lo, hi, graded=graded)
    _assert_span(material, first + 1, 3)
    _assert_span(_box_axis_volume(centres.z, lo, hi, sizes[2]), first, 4)


def test_graded_centre_band_uses_primal_width_not_neighbour_spacing():
    widths = np.array([.01, .0001, .003, .02])
    nodes = np.r_[0., np.cumsum(widths[:-1])]
    centres = nodes + .5 * widths
    # Two bands above this narrow cell's centre is outside its band, even
    # though it is inside a band based on the neighbouring centre spacing.
    lo = centres[1] + 2 * NODE_TIE_REL * widths[1]
    hi = centres[3] + 2 * NODE_TIE_REL * widths[3]
    exact = (centres >= lo) & (centres < hi)
    np.testing.assert_array_equal(_box_axis_volume(centres, lo, hi, widths), exact)


@pytest.mark.parametrize("shift", [-.5, -2e-9, 2e-9, .5])
def test_samples_outside_the_band_keep_exact_membership(shift):
    dx = DXS[0]
    samples = np.arange(20, dtype=np.float64) * dx
    lo, hi = samples[FIRST] + shift * dx, samples[FIRST + 7] + shift * dx
    exact = (samples >= lo) & (samples < hi)
    np.testing.assert_array_equal(half_open_volume_mask(samples, lo, hi, dx), exact)


def test_both_samplers_accept_traced_coordinates():
    with enable_x64():
        dx = DXS[0]
        nodes = np.arange(20, dtype=np.float64) * dx
        lo, hi = np.nextafter(nodes[[FIRST, FIRST + 6]], np.inf)
        shape = Box((lo, 0., 0.), (hi, 2 * dx, 2 * dx))
        traced_nodes = jax.jit(lambda x: shape.mask_on_coords(x, x, x))(
            jnp.asarray(nodes))
        _assert_span(np.any(np.asarray(traced_nodes), axis=(1, 2)), FIRST, 6)
        centres = nodes + dx / 2
        lo, hi = np.nextafter(centres[[FIRST, FIRST + 6]], np.inf)
        traced_centres = jax.jit(_box_axis_volume)(
            jnp.asarray(centres), lo, hi, jnp.full(centres.shape, dx))
        _assert_span(traced_centres, FIRST, 6)


def test_periodic_material_image_inherits_the_node_face_band():
    dx = DXS[0]
    length = 17 * dx
    sim = Simulation(
        10e9, (length, 7.3 * dx, 5.6 * dx), dx=dx, cpml_layers=0,
        boundary=BoundarySpec(x="periodic", y=Boundary("pec", "pmc"), z="pec"))
    sim.add_material("laminate", eps_r=3.7)
    grid = sim._build_grid()
    coords = coords_from_uniform_grid(grid)
    lo = np.nextafter(coords.x[FIRST] + length, np.inf)
    hi = np.nextafter(coords.x[FIRST + 6] + length, np.inf)
    shape = Box((lo, dx, dx), (hi, 4 * dx, 3 * dx))
    sim.add(shape, material="laminate")
    materials = sim._assemble_materials(grid)[0]
    _assert_span(np.any(np.asarray(materials.eps_r) > 1, axis=(1, 2)), FIRST, 6)
    _assert_span(np.any(np.asarray(shape.mask(grid)), axis=(1, 2)), FIRST, 6)


@pytest.mark.parametrize("dx", DXS)
@pytest.mark.parametrize("delta", [1e-13, 1e-12, 1e-11])
def test_domain_and_box_share_the_band_at_a_cpml_face(dx, delta):
    """The review rig: sizing and the slab must both realize 64 cells.

    Without the sizing band, two empty hi nodes make PadFillShortfall
    raise; without the Box band, the raw slab owns a 65th interior node.
    """
    domain = 64 * dx * (1 + delta)
    sim = Simulation(
        10e9, (domain, domain, 16 * dx), dx=dx,
        boundary="cpml", cpml_layers=4)
    sim.add_material("slab", eps_r=3.7)
    sim.add(Box((0., 0., 4 * dx), (domain, domain, 8 * dx)), material="slab")
    grid = sim._build_grid()
    filled = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    raw = np.asarray(sim._assemble_materials(
        grid, include_cpml_pad_extension=False)[0].eps_r)
    for axis in (0, 1):
        lo_pad, hi_pad = grid.face_pads[2 * axis:2 * axis + 2]
        present = np.any(raw != 1.0, axis=tuple(a for a in range(3) if a != axis))
        interior = present[lo_pad:len(present) - hi_pad]
        assert interior.size == 65
        _assert_span(interior, 0, 64)
    k = grid.pad_z_lo + 5
    expected = np.asarray(3.7, dtype=filled.dtype)
    assert filled[-1, grid.pad_y_lo + 5, k] == expected
    assert filled[grid.pad_x_lo + 5, -1, k] == expected


@pytest.mark.parametrize("traced", [False, True], ids=["numpy", "traced"])
@pytest.mark.parametrize("below,above,split", [(10., 1., 4), (1., 10., 3)])
def test_stacked_graded_boxes_partition_every_node(below, above, split, traced):
    """A shared face 5 nm above a node uses that node's forward cell.

    Midpoint widths leave a gap with 10 m cells below / 1 m above, and an
    overlap with the reverse grading. Both boxes must agree on ownership.
    """
    widths = np.r_[np.full(3, below), np.full(6, above)]
    domain = float(np.sum(widths))
    sim = Simulation(
        10e6, (4., 4., domain), dx=1., dz_profile=widths,
        boundary="pec", cpml_layers=0)
    grid = sim._build_nonuniform_grid()
    coords = coords_from_nonuniform_grid(grid)
    face = float(coords.z[3]) + 5e-9
    boxes = (Box((0., 0., 0.), (4., 4., face)),
             Box((0., 0., face), (4., 4., domain)))
    masks = []
    for index, shape in enumerate(boxes):
        sim.add_material(f"layer{index}", eps_r=2. + index)
        sim.add(shape, material=f"layer{index}")
        with enable_x64():
            mask = (jax.jit(lambda z: shape.mask_on_coords(coords.x, coords.y, z))(
                jnp.asarray(coords.z)) if traced else shape.mask_on_coords(*coords[:3]))
        masks.append(np.any(np.asarray(mask), axis=(0, 1)))
    ownership = masks[0].astype(int) + masks[1].astype(int)
    np.testing.assert_array_equal(ownership[:-1], 1)
    assert ownership[-1] == 0
    _assert_span(masks[0], 0, split)
    _assert_span(masks[1], split, len(widths) - split)
    column = np.asarray(sim._assemble_materials_nu(grid)[0].eps_r)[1, 1]
    np.testing.assert_array_equal(column, [2.] * split + [3.] * (len(widths) - split) + [1.])
