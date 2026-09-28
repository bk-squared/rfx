"""A Box face drawn on a node plane realizes on that plane, whatever route
spelled it (#1138).

A 787 um laminate drawn from ``z0`` to ``z0 + h`` on a mesh of h/6 realized 7
cells (918 um) and on h/3 realized 4: the top face landed at
36.00000000000001 cells, the exact half-open ``lo <= node < hi`` test let node
36 in, and the patch sheet on that face ended up buried in the dielectric.

The contract pinned here, build-only (no solve):

* a volume whose faces are ``N`` cells apart realizes exactly ``N`` node
  planes (or ``N`` cell centres for the PEC centre sampler), for faces spelled
  ``m*dx``, as ``h`` with the cell ``h/N``, and as ``z0 + h``; on x, y and z;
  on uniform, graded and traced (float32, mesh as design variable) node lines;
* a face further than the snap tolerance from every node realizes exactly
  what the plain half-open test gives.

The expected count is always the integer the drawing states, never a value
computed by the rasterizer under test.
"""
from __future__ import annotations

import itertools

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.geometry.csg import Box
from rfx.geometry.rasterize_grid import (
    _axis_node_positions,
    _box_axis_volume,
    _uniform_axis_centres,
    _uniform_axis_nodes,
    coords_from_nonuniform_grid,
)
from rfx.nonuniform import make_nonuniform_grid

NS = range(2, 65)
DXS = (1e-3, 0.1e-3, 0.254e-3, 0.3048e-3, 12.5e-6, 1e-3 / 3)
HS = (0.787e-3, 1.524e-3, 0.508e-3, 0.8e-3, 1.6e-3, 3.175e-3, 0.127e-3)
KS = (0, 1, 3, 7, 20)
PAD = 4
_OTHER = np.array([0.5])            # one sample inside [0, 1) on the other axes


def _count(nodes, lo, hi, axis):
    """Node planes a Box ``lo -> hi`` along ``axis`` occupies on ``nodes``."""
    c_lo, c_hi = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    c_lo[axis], c_hi[axis] = lo, hi
    coords = [_OTHER, _OTHER, _OTHER]
    coords[axis] = nodes
    m = np.asarray(Box(tuple(c_lo), tuple(c_hi)).mask_on_coords(*coords))
    return int(m.sum())


def _uniform_cases():
    """(route, N, cell, lo, hi) with the faces N cells apart on node planes."""
    for n, dx, k in itertools.product(NS, DXS, KS):
        yield "m*dx", n, dx, k * dx, (k + n) * dx
    for n, h in itertools.product(NS, HS):
        yield "h, cell h/N", n, h / n, 0.0, h
        for k in KS:
            z0 = k * (h / n)
            yield "z0 + h, cell h/N", n, h / n, z0, z0 + h


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_faces_on_node_planes_realize_exactly_n_cells_on_a_uniform_axis(axis):
    wrong = []
    for route, n, dx, lo, hi in _uniform_cases():
        nodes = _uniform_axis_nodes(int(round(hi / dx)) + 30, PAD, dx)
        got = _count(nodes, lo, hi, axis)
        if got != n:
            wrong.append((route, n, dx, lo, got))
    assert not wrong, (
        f"{len(wrong)} drawings on axis {'xyz'[axis]} realized a different "
        f"number of cells than drawn (route, N, cell, lo, realized): {wrong[:10]}")


def _graded_profile(n, h, k):
    """k coarse cells (3 h/N) below the laminate, N cells of h/N in it, then
    10 cells of 2 h/N above; the laminate spans [k * 3h/N, k * 3h/N + h]."""
    dx = h / n
    return np.concatenate([np.full(PAD + k, 3 * dx), np.full(n, dx),
                           np.full(10, 2 * dx)])


def test_faces_on_node_planes_realize_exactly_n_cells_on_a_graded_axis():
    """The concrete non-uniform node line: a cumulative sum of the cells, so
    ``z0 + h`` and the node differ in the last bits on most boards."""
    wrong = []
    for n, h, k in itertools.product(NS, HS, KS):
        d = _graded_profile(n, h, k)
        nodes = _axis_node_positions(d, PAD)
        z0 = k * 3 * (h / n)
        got = _count(nodes, z0, z0 + h, 2)
        if got != n:
            wrong.append((n, h, k, got))
    assert not wrong, (
        f"{len(wrong)} graded laminates realized a different number of cells "
        f"than drawn (N, h, k, realized): {wrong[:10]}")


def test_a_graded_board_realizes_its_laminate_through_the_grid_builder():
    """Same drawing through ``make_nonuniform_grid`` and the production
    coordinate provider, on the z axis a board grades."""
    wrong = []
    for n, h in itertools.product((3, 4, 5, 6, 7, 8, 12, 16), (0.787e-3, 1.524e-3)):
        dz = np.concatenate([np.full(12, 3 * h / n), np.full(n, h / n),
                             np.full(12, 2 * h / n)])
        g = make_nonuniform_grid((1e-3, 1e-3), dz, 0.25e-3, cpml_layers=4)
        c = coords_from_nonuniform_grid(g)
        z0 = 12 * 3 * (h / n)
        m = np.asarray(Box((0.0, 0.0, z0), (1e-3, 1e-3, z0 + h))
                       .mask_on_coords(c.x, c.y, c.z))
        got = int(m[m.shape[0] // 2, m.shape[1] // 2, :].sum())
        if got != n:
            wrong.append((n, h, got))
    assert not wrong, wrong


def _traced_counter(template, boxes):
    """Counts of each Box's z column on a TRACED z profile (mesh as design
    variable), through ``coords_from_nonuniform_grid``'s traced branch."""
    ix, iy = template.nx // 2, template.ny // 2

    @jax.jit
    def counts(dz):
        c = coords_from_nonuniform_grid(template._replace(dz=dz))
        return jnp.stack([jnp.sum(b.mask_on_coords(c.x, c.y, c.z)[ix, iy, :])
                          for b in boxes])
    return counts


@pytest.mark.parametrize("h", [0.787e-3, 1.524e-3])
def test_faces_on_node_planes_realize_exactly_n_cells_on_a_traced_mesh(h):
    """The traced node line is float32, so its rounding is far above 1e-9 of
    a cell; the faces are fixed across N so one compile covers the sweep."""
    length = 3 * 64 + 20
    template = make_nonuniform_grid((1e-3, 1e-3), np.full(length, h / 8),
                                    0.25e-3, cpml_layers=4)
    pad = int(template.pad_z_lo)
    coarse = 3 * h / 8
    z0 = 12 * coarse
    # graded: 12 coarse cells, N cells of h/N, coarse cells to a fixed length
    graded = _traced_counter(template, [Box((0.0, 0.0, z0), (1e-3, 1e-3, z0 + h))])
    # uniform-valued: every cell h/N; faces at 2h and 3h are nodes for any N
    uniform = _traced_counter(template, [Box((0.0, 0.0, 0.0), (1e-3, 1e-3, h)),
                                         Box((0.0, 0.0, 2 * h), (1e-3, 1e-3, 3 * h))])
    n_total = template.nz
    wrong = []
    for n in NS:
        d = np.concatenate([np.full(pad + 12, coarse), np.full(n, h / n)])
        d = np.concatenate([d, np.full(n_total - d.size, coarse)])
        got = int(graded(jnp.asarray(d, dtype=jnp.float32))[0])
        if got != n:
            wrong.append(("graded", n, got))
        got_u = [int(v) for v in uniform(jnp.full(n_total, h / n, dtype=jnp.float32))]
        if n * 3 + pad < n_total and got_u != [n, n]:
            wrong.append(("uniform-valued", n, got_u))
    assert not wrong, (
        f"traced mesh, h = {h * 1e6:.0f} um: {len(wrong)} drawings realized a "
        f"different count (profile, N, realized): {wrong[:10]}")


@pytest.mark.parametrize("n_cells", [3, 6])
def test_the_reported_laminate_realizes_its_drawn_thickness(n_cells):
    """The reported board: 787 um of eps_r 3.38 from a ground snapped to its
    node, drawn ``z0`` to ``z0 + h``, meshed at h/3 and h/6. Counted in the
    assembled permittivity down the centre column."""
    h = 0.787e-3
    dx = h / n_cells
    z0 = round(4e-3 / dx) * dx
    sim = Simulation(freq_max=12e9, domain=(4e-3, 4e-3, 8e-3), dx=dx,
                     cpml_layers=4, boundary="cpml")
    sim.add_material("laminate", eps_r=3.38)
    sim.add(Box((0.0, 0.0, z0), (4e-3, 4e-3, z0 + h)), material="laminate")
    grid = sim._build_grid()
    materials = sim._assemble_materials(grid)[0]
    eps = np.asarray(materials.eps_r)
    column = eps[eps.shape[0] // 2, eps.shape[1] // 2, :]
    assert int(np.sum(column > 2.0)) == n_cells
    k0 = int(round(z0 / dx)) + grid.axis_pads[2]
    assert np.where(column > 2.0)[0].tolist() == list(range(k0, k0 + n_cells))


def test_pec_volume_faces_on_cell_centres_realize_exactly_n_centres():
    """The PEC volume sampler reads cell centres; a face spelled ``a + b`` on
    a centre is ON that centre."""
    wrong = []
    for n, dx, k in itertools.product(NS, DXS, KS):
        centres = _uniform_axis_centres(k + n + 30, PAD, dx)
        lo = k * dx + 0.5 * dx              # a + b, not (k + 0.5) * dx
        hi = lo + n * dx
        got = int(np.sum(_box_axis_volume(centres, lo, hi)))
        if got != n:
            wrong.append((n, dx, k, got))
    assert not wrong, (
        f"{len(wrong)} PEC volumes realized a different number of centre cells "
        f"than drawn (N, dx, k, realized): {wrong[:10]}")


def test_a_one_cell_box_with_dusty_faces_lands_on_its_lo_node():
    """The thin branch reads the same window: a one-cell box is still its
    ``lo``-face node when both faces carry rounding dust."""
    wrong = []
    for h, n in itertools.product(HS, (3, 6, 7, 11, 13, 23)):
        dx = h / n
        nodes = _uniform_axis_nodes(n + 40, PAD, dx)
        for j in range(1, n + 20):
            lo = (j - 1) * dx + dx          # j*dx through a + b
            hi = lo + dx
            occ = np.where(np.asarray(
                Box((lo, 0.0, 0.0), (hi, 1.0, 1.0))
                .mask_on_coords(nodes, _OTHER, _OTHER))[:, 0, 0])[0]
            if occ.tolist() != [PAD + j]:
                wrong.append((h, n, j, occ.tolist()))
    assert not wrong, wrong[:10]


@pytest.mark.parametrize("offset_cells", [1e-7, 1e-3, 0.25, 0.5])
def test_faces_off_the_node_realize_what_the_plain_half_open_rule_gives(offset_cells):
    """Beyond the tolerance nothing changes: the count equals the plain
    half-open ``lo <= x < hi`` count, on either side of the node."""
    dx = 0.1e-3
    nodes = _uniform_axis_nodes(80, PAD, dx)
    for n, k, s_lo, s_hi in itertools.product((2, 5, 17), (0, 3), (-1, 1), (-1, 1)):
        lo = (k + s_lo * offset_cells) * dx
        hi = (k + n + s_hi * offset_cells) * dx
        plain = int(np.sum((nodes >= lo) & (nodes < hi)))
        assert _count(nodes, lo, hi, 2) == plain, (n, k, s_lo, s_hi)


def test_traced_faces_off_the_node_realize_the_plain_rule():
    """A face 1e-2 of a cell off its node on a traced mesh is not snapped."""
    h = 0.787e-3
    template = make_nonuniform_grid((1e-3, 1e-3), np.full(80, h / 6),
                                    0.25e-3, cpml_layers=4)
    dx = h / 6
    boxes = [Box((0.0, 0.0, (3 + s_lo * 1e-2) * dx),
                 (1e-3, 1e-3, (9 + s_hi * 1e-2) * dx))
             for s_lo, s_hi in itertools.product((-1, 1), (-1, 1))]
    got = [int(v) for v in _traced_counter(template, boxes)(
        jnp.full(template.nz, dx, dtype=jnp.float32))]
    # lo below its node keeps it, above drops it; hi below drops, above keeps
    assert got == [6, 7, 5, 6], got
