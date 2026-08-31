"""Issue #802: a shape's realized cells must not depend on ``JAX_ENABLE_X64``.

PROTOTYPE regression tests (fail-before-fix on main 06cf29f0 under the default
precision; see the issue for the measurement):

``rfx/api/_compile.py::_grid_coords`` built node coordinates as
``(jnp.arange(n) - pad) * dx`` in the JAX default dtype, so in float32 the
product ``i*dx`` landed ~1e-10 m below the exact node and the half-open test
``(coords >= lo) & (coords < hi)`` flipped at node-aligned faces: hi-face node
INCLUDED, lo-face node EXCLUDED, per axis, per face, by rounding accident.

Convention pinned here (the :class:`Box` docstring's): node ``j`` at ``j*dx``
belongs iff ``lo <= j*dx < hi`` evaluated on the exact node value, with a face
declared within ``NODE_SNAP_REL`` (1e-6) of the local cell size of a node
treated as ON that node; the thin (sub-cell) branch keeps its nearest-centre
node and an exact tie goes to the lower node. Nothing here reads a
precision-dependent value: the same declaration realizes the same cells under
both settings of ``jax_enable_x64`` and on the uniform and non-uniform lanes.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.csg import Box, Cylinder
from rfx.grid import Grid

from tests._x64_compat import enable_x64

DX = 100e-6
DOMAIN = (12.5e-3, 3.4e-3, 3.9e-3)
REPRO_BOX = Box((0.0, 0.0, 2.5e-3), (12.5e-3, 3.4e-3, 2.8e-3))


def _sim():
    return Simulation(freq_max=12e9, domain=DOMAIN, dx=DX, cpml_layers=8,
                      boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"))


def _nodes(mask, pads):
    """(cells, [(lo_node, hi_node) per axis]) in interior node indices."""
    m = np.asarray(mask, dtype=bool)
    out = []
    for a in range(3):
        idx = np.where(m.any(axis=tuple(x for x in range(3) if x != a)))[0]
        out.append((int(idx.min() - pads[a]), int(idx.max() - pads[a])))
    return int(m.sum()), out


def _uniform(shape):
    g = _sim()._build_grid()
    return _nodes(shape.mask(g), g.axis_pads)


def _nu(shape):
    from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
    sim = Simulation(freq_max=12e9, domain=DOMAIN, dx=DX, cpml_layers=8,
                     boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
                     dz_profile=np.full(39, DX))
    g = sim._build_nonuniform_grid()
    c = coords_from_nonuniform_grid(g)
    return _nodes(shape.mask_on_coords(c.x, c.y, c.z),
                  (g.pad_x_lo, g.pad_y_lo, g.pad_z_lo))


# --- 1. the committed repro: exact half-open realization -----------------------

def test_repro_802_node_aligned_box_realizes_per_the_documented_convention():
    """.repro_802.py: main gives 13230 cells (x 0..125, y 0..34, z 25..27)
    under x64=0 and 12750 (x 0..124, y 0..33, z 25..27) under x64=1. The
    convention says the hi face contributes no node: 125*126... no --
    nodes 0..124 x 0..33 x 25..27 = 125*34*3 = 12750 cells."""
    cells, nodes = _uniform(REPRO_BOX)
    assert nodes == [(0, 124), (0, 33), (25, 27)], nodes
    assert cells == 125 * 34 * 3 == 12750


# --- 2. x64 toggle identity ------------------------------------------------------

_SHAPES = {
    "box_node_aligned": REPRO_BOX,
    "thin_sheet_35um": Box((1e-3, 1e-3, 2.5e-3), (2e-3, 2e-3, 2.535e-3)),
    "one_cell_box_between_node_planes": Box((0.0, 0.0, 2.5e-3),
                                            (12.5e-3, 3.4e-3, 2.6e-3)),
    "cylinder_node_aligned": Cylinder(center=(6.0e-3, 1.7e-3, 2.0e-3),
                                      radius=0.5e-3, height=1.0e-3, axis="z"),
}


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_mask_is_identical_under_both_x64_settings(name):
    shape = _SHAPES[name]
    g = _sim()._build_grid()
    m_here = np.asarray(shape.mask(g))
    with enable_x64():
        g64 = _sim()._build_grid()
        m_x64 = np.asarray(shape.mask(g64))
    assert m_here.shape == m_x64.shape
    n_diff = int(np.sum(m_here != m_x64))
    assert n_diff == 0, (
        f"{name}: {n_diff} cells differ between jax_enable_x64 settings "
        f"({_nodes(m_here, g.axis_pads)} vs {_nodes(m_x64, g.axis_pads)})")


# --- 3. a face 1e-9 m off a node is a real offset, not noise ----------------------

def test_face_1e_9_m_off_a_node_realizes_by_the_exact_inequality():
    """1e-9 m is 1e-5 of a 100 um cell -- far above float64 rounding and above
    the 1e-6*dx snap, so it must NOT be snapped, and the answer must not
    depend on the precision flag. lo = node + 1e-9: node 25 sits below lo and
    is excluded; lo = node - 1e-9: node 25 is included."""
    up = Box((0.0, 0.0, 2.5e-3 + 1e-9), (12.5e-3, 3.4e-3, 2.8e-3))
    dn = Box((0.0, 0.0, 2.5e-3 - 1e-9), (12.5e-3, 3.4e-3, 2.8e-3))
    assert _uniform(up)[1][2] == (26, 27)
    assert _uniform(dn)[1][2] == (25, 27)
    with enable_x64():
        assert _uniform(up)[1][2] == (26, 27)
        assert _uniform(dn)[1][2] == (25, 27)


# --- 4. fidelity_report reads the same realization ---------------------------------

def test_fidelity_report_realized_extents_equal_the_mask_extents():
    from rfx.fidelity import fidelity_report
    sim = _sim()
    sim.add_material("sub", eps_r=3.66)
    sim.add(REPRO_BOX, material="sub")
    report = fidelity_report(sim, print_report=False)
    item = next(r for r in report if r["entity"].startswith("geometry[0]"))
    realized = [tuple(round(v, 6) for v in ax["realized_um"]) for ax in item["axes"]]
    # occupied nodes 0..124 -> cells 0..124 -> [0, 125*dx]; z nodes 25..27 -> [2500, 2800]
    assert realized == [(0.0, 12500.0), (0.0, 3400.0), (2500.0, 2800.0)], realized
    assert item["n_cells"] == 12750
    for ax in item["axes"]:
        assert max(ax["face_residual_um"]) < 1e-6, ax


# --- 5. decimal-typed node-aligned faces on a WR-90 cell size ----------------------

@pytest.mark.parametrize("dx_str", ["100e-6", "0.762e-3", "0.3048e-3"])
def test_decimal_typed_node_aligned_faces_are_exact(dx_str):
    """A user types ``k*dx`` as a decimal; float64 ``i*dx`` lands one ulp
    below that decimal on 61/398 nodes at dx = 0.762 mm (WR-90 a/30) and
    157/398 at 0.3048 mm, so float64 coordinates ALONE still flip those
    faces. The snap tolerance makes them exact."""
    from decimal import Decimal
    dx = float(dx_str)
    n_cells = 60
    grid = Grid(freq_max=1e9, domain=(n_cells * dx, dx, dx), dx=dx, cpml_layers=0)
    pad = grid.axis_pads[0]
    bad = []
    for k in range(1, n_cells):
        face = float(Decimal(dx_str) * k)
        m_lo = np.asarray(Box((face, -1.0, -1.0), (1.0, 1.0, 1.0)).mask(grid))[:, 0, 0]
        m_hi = np.asarray(Box((-1.0, -1.0, -1.0), (face, 1.0, 1.0)).mask(grid))[:, 0, 0]
        if not m_lo[pad + k]:
            bad.append(("lo-face node excluded", k, face))
        if m_hi[pad + k]:
            bad.append(("hi-face node included", k, face))
    assert not bad, bad


# --- 6. thin-branch tie rule --------------------------------------------------------

def test_one_cell_box_between_node_planes_takes_the_lower_node():
    """A 1-cell box [25dx, 26dx] takes the thin branch (extent <= 1.01 cells)
    and its midpoint is EQUIDISTANT from nodes 25 and 26. Before #802 float32
    noise picked 26 under x64=0 and 25 under x64=1 (and prototype 85b59fe0
    picked the upper node on tests/test_fidelity_report.py's [2000, 2500] um
    sheet, moving a ground plane by a cell). The tie goes to the lower node,
    the same lo-inclusive side the volume branch keeps."""
    one_cell = _SHAPES["one_cell_box_between_node_planes"]
    assert _uniform(one_cell)[1][2] == (25, 25)
    with enable_x64():
        assert _uniform(one_cell)[1][2] == (25, 25)


# --- 7. uniform and non-uniform lanes agree ------------------------------------------

@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_nonuniform_lane_realizes_the_same_cells_as_the_uniform_lane(name):
    """The NU lane's nodes were float32 cumsums of float32 spacings, so the
    same declaration realized 13230 cells there and 12750 on the uniform
    lane under x64=1 (main). Both lanes must agree with the convention."""
    shape = _SHAPES[name]
    assert _nu(shape) == _uniform(shape)
