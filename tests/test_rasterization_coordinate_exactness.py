"""Rasterization node coordinates are exact and lane-independent (#802/#807).

The class defect: node coordinates were built three ways (csg's f32 product,
rasterize_grid's cast-once uniform builder, the NU cumsum with an f32 cast),
and under the default precision each rounded differently at the last ulp.
Node-aligned faces then flipped the documented half-open ``[lo, hi)``
inclusion per rounding accident, whole node planes of realized geometry
depended on ``jax_enable_x64``, and a one-cell sheet landed on different
planes in the uniform and non-uniform lanes.

The contract locked here:

* concrete node coordinates are HOST float64, exact, flag-independent
  (``_uniform_axis_nodes`` is THE uniform formula);
* every shape class realizes the same mask under x64=0 and x64=1;
* the uniform lane (``shape.mask(grid)``) and the NU lane
  (``shape.mask_on_coords(*coords_from_nonuniform_grid(g))``) are bitwise
  equal on a uniform-valued profile, per axis (graded one axis at a time,
  so an axis swap cannot pass);
* the traced-profile (mesh-as-design-variable) path still traces and
  differentiates;
* the shape-class census is introspected, so a NEW shape class fails this
  suite until it is added to the parametrization.

x64 is flipped per-test via the scoped ``tests/_x64_compat.enable_x64``
context — never at module level (process-global, contaminates the shard).
"""
from __future__ import annotations

import inspect

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.geometry import csg as csg_mod
from rfx.geometry import curved as curved_mod
from rfx.geometry import mesh_import as mesh_import_mod
from rfx.geometry import via as via_mod
from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere, _grid_coords
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.rasterize_grid import (
    _uniform_axis_nodes,
    coords_from_nonuniform_grid,
    coords_from_uniform_grid,
)
from rfx.geometry.via import Via
from rfx.nonuniform import make_nonuniform_grid
from tests._x64_compat import enable_x64

try:
    import trimesh
except ImportError:
    trimesh = None

DX = 100e-6
DOM = (3e-3, 3e-3, 2e-3)
CPML = 4


def _grid():
    sim = Simulation(freq_max=20e9, domain=DOM, dx=DX, cpml_layers=CPML,
                     boundary="cpml")
    return sim._build_grid()


def _nu_grid(graded_axes=("x", "y", "z")):
    """NU grid with uniform-VALUED profiles on the requested axes only."""
    n = [int(round(d / DX)) for d in DOM]
    return make_nonuniform_grid(
        (DOM[0], DOM[1]),
        np.full(n[2], DX) if "z" in graded_axes else np.full(n[2], DX),
        DX, cpml_layers=CPML,
        dx_profile=np.full(n[0], DX) if "x" in graded_axes else None,
        dy_profile=np.full(n[1], DX) if "y" in graded_axes else None)


def _mesh_shape():
    box = trimesh.creation.box(extents=(1.7e-3, 1.7e-3, 0.9e-3),
                               transform=trimesh.transformations.
                               translation_matrix((1.5e-3, 1.5e-3, 0.9e-3)))
    return mesh_import_mod.MeshShape(box)


# Census instances: at least one per shape class; Cylinder on all 3 axes.
# Node-aligned faces are the point — they are the case the f32 coordinates
# realized wrongly.
_SHAPE_CASES = {
    "box_volume_node_aligned": lambda: Box((0.5e-3, 0.5e-3, 0.4e-3),
                                           (2.5e-3, 2.5e-3, 1.2e-3)),
    "box_one_cell_sheet": lambda: Box((0.5e-3, 0.5e-3, 0.4e-3),
                                      (2.5e-3, 2.5e-3, 0.5e-3)),
    "cylinder_z": lambda: Cylinder((1.5e-3, 1.5e-3, 0.9e-3), radius=0.6e-3,
                                   height=0.8e-3, axis="z"),
    "cylinder_y": lambda: Cylinder((1.5e-3, 1.5e-3, 0.9e-3), radius=0.6e-3,
                                   height=0.8e-3, axis="y"),
    "cylinder_x": lambda: Cylinder((1.5e-3, 1.5e-3, 0.9e-3), radius=0.6e-3,
                                   height=0.8e-3, axis="x"),
    "sphere": lambda: Sphere((1.5e-3, 1.5e-3, 1.0e-3), radius=0.7e-3),
    "polyline_wire": lambda: PolylineWire(
        ((0.6e-3, 0.6e-3, 0.3e-3), (2.4e-3, 1.5e-3, 1.1e-3),
         (1.1e-3, 2.4e-3, 1.6e-3)), radius=0.23e-3),
    "via": lambda: Via(center=(1.5e-3, 1.5e-3), drill_radius=0.2e-3,
                       pad_radius=0.4e-3,
                       layers=[(0.4e-3, 0.8e-3), (0.8e-3, 1.2e-3)]),
    "curved_patch": lambda: CurvedPatch(center=(1.5e-3, 1.5e-3, 0.5e-3),
                                        length=1.6e-3, width=1.2e-3,
                                        radius=4e-3, axis="x"),
    "mesh_shape": None,  # trimesh-gated, see _shape_for()
}


def _shape_for(name):
    if name == "mesh_shape":
        if trimesh is None:
            pytest.skip("trimesh not installed (cad extra)")
        return _mesh_shape()
    return _SHAPE_CASES[name]()


def _census():
    """Every public class in the geometry modules defining mask_on_coords."""
    found = set()
    for mod in (csg_mod, via_mod, curved_mod, mesh_import_mod):
        for _nm, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__ or _nm.startswith("_"):
                continue
            if "mask_on_coords" in vars(cls) and _nm != "Shape":
                found.add(cls.__name__)
    return found


def test_shape_class_census_is_covered():
    """A new shape class must be added to the parametrized cases."""
    covered = {"Box", "Cylinder", "Sphere", "PolylineWire", "Via",
               "CurvedPatch", "MeshShape"}
    assert _census() == covered, (
        "shape-class census changed; add the new class to _SHAPE_CASES in "
        "this file so the exactness contract covers it")


def test_the_one_uniform_node_formula_is_exact_f64():
    nodes = _uniform_axis_nodes(56, 8, 100e-6)
    assert nodes.dtype == np.float64
    ref = (np.arange(56, dtype=np.float64) - 8) * 100e-6
    assert np.array_equal(nodes, ref)
    # csg and the uniform coordinate provider both delegate to it
    grid = _grid()
    gx, gy, gz = _grid_coords(grid)
    cu = coords_from_uniform_grid(grid)
    for a, b in ((gx, cu.x), (gy, cu.y), (gz, cu.z)):
        assert np.asarray(a).dtype == np.float64
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_node_aligned_box_realizes_the_documented_convention():
    """The #802 repro as a regression test (failed before the fix).

    ``Box((0,0,2.5e-3),(12.5e-3,3.4e-3,2.8e-3))`` on domain
    (12.5, 3.4, 3.9) mm, dx=100 um, cpml_layers=8: node j belongs iff
    ``lo <= j*dx < hi`` — the hi face contributes no cell. Under f32
    coordinates the x/y hi-face nodes were INCLUDED (13230 cells instead
    of 12750) at x64=0.
    """
    sim = Simulation(freq_max=20e9, domain=(12.5e-3, 3.4e-3, 3.9e-3),
                     dx=DX, cpml_layers=8, boundary="cpml")
    grid = sim._build_grid()
    box = Box((0.0, 0.0, 2.5e-3), (12.5e-3, 3.4e-3, 2.8e-3))
    m = np.asarray(box.mask(grid))
    pads = grid.axis_pads
    assert int(m.sum()) == 12750
    for axis, (lo_n, hi_n) in enumerate(((0, 124), (0, 33), (25, 27))):
        occ = np.where(m.any(axis=tuple(i for i in range(3)
                                        if i != axis)))[0]
        assert occ.min() - pads[axis] == lo_n
        assert occ.max() - pads[axis] == hi_n


@pytest.mark.parametrize("case", sorted(_SHAPE_CASES))
def test_realized_mask_is_x64_invariant(case):
    """Same declaration, same cells, regardless of jax_enable_x64."""
    shape = _shape_for(case)
    grid = _grid()
    m0 = np.asarray(shape.mask(grid))
    with enable_x64():
        m1 = np.asarray(shape.mask(grid))
    assert np.array_equal(m0, m1), (
        f"{case}: realized cells changed with jax_enable_x64 "
        f"({int(m0.sum())} vs {int(m1.sum())})")


@pytest.mark.parametrize("case", sorted(_SHAPE_CASES))
@pytest.mark.parametrize("graded", ["x", "y", "z", "xyz"])
def test_uniform_and_nu_lanes_realize_identical_masks(case, graded):
    """mask(grid) == mask_on_coords(coords_from_nonuniform_grid(g)).

    Each axis is exercised as the explicitly-profiled one separately —
    a uniform-valued profile that only ever grades z would let an axis
    swap in the coordinate builder pass unnoticed.
    """
    shape = _shape_for(case)
    grid = _grid()
    gnu = _nu_grid(tuple(graded))
    mu = np.asarray(shape.mask(grid))
    cn = coords_from_nonuniform_grid(gnu)
    mn = np.asarray(shape.mask_on_coords(cn.x, cn.y, cn.z))
    # NU allocates one extra bounding node per axis; compare shared space
    sl = tuple(slice(0, s) for s in mu.shape)
    assert np.array_equal(mu, mn[sl]), (
        f"{case} (graded={graded}): lanes realize different cells "
        f"({int(mu.sum())} vs {int(mn[sl].sum())})")
    # the stronger by-construction claim: the coordinates themselves agree
    gx, gy, gz = _grid_coords(grid)
    for a, b in ((gx, cn.x), (gy, cn.y), (gz, cn.z)):
        an, bn = np.asarray(a), np.asarray(b)
        assert np.array_equal(an, bn[:an.size])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_one_cell_sheet_lands_on_one_and_the_same_plane_in_both_lanes(axis):
    """The #807 observable: a one-cell sheet is ONE plane, the SAME plane.

    Before the fix the uniform lane put the sheet on plane 8 at x64=1 and
    plane 9 at x64=0, while the NU lane stayed on 9 — the thin-branch
    argmin at an exact half-cell tie was decided by the last ulp of two
    different f32 coordinate constructions.
    """
    lo = [0.5e-3] * 3
    hi = [2.5e-3] * 3
    lo[axis], hi[axis] = 0.4e-3, 0.5e-3
    sheet = Box(tuple(lo), tuple(hi))
    grid = _grid()
    gnu = _nu_grid()
    other = tuple(i for i in range(3) if i != axis)
    mu = np.asarray(sheet.mask(grid))
    cn = coords_from_nonuniform_grid(gnu)
    mn = np.asarray(sheet.mask_on_coords(cn.x, cn.y, cn.z))
    pu = np.where(mu.any(axis=other))[0]
    pn = np.where(mn.any(axis=other))[0]
    assert len(pu) == 1 and len(pn) == 1
    assert pu.tolist() == pn.tolist()
    with enable_x64():
        mu64 = np.asarray(sheet.mask(grid))
    assert np.array_equal(mu, mu64)
    # Tie rule: a face-registered one-cell box is an exact half-cell tie;
    # it must realize on its LO-face node — the node its own half-open
    # [lo, hi) window keeps — not on whichever side the last f64 ulp of
    # (lo+hi)*0.5 favours (cv15's stack contract and the fidelity-report
    # exact-faces fixture both encode this).
    ax_coords = np.asarray(_grid_coords(grid)[axis])
    lo_node = np.where(ax_coords == lo[axis])[0]
    assert lo_node.size == 1, "fixture: lo face must sit bitwise on a node"
    assert pu[0] == lo_node[0]


def test_traced_profile_mask_path_still_traces_and_differentiates():
    """GEO Tier-2 contract: traced coords stay traced (mesh as DoF).

    The mask itself is piecewise-constant in the profile; the gradient
    flows through the coordinate values a mask-weighted sum reads. The
    guard is that the traced branch neither hits np.asarray on a tracer
    nor breaks under jit/grad.
    """
    box = Box((0.5e-3, 0.5e-3, 0.4e-3), (2.5e-3, 2.5e-3, 1.2e-3))
    grid = _grid()
    gx, gy, _gz = (np.asarray(c) for c in _grid_coords(grid))
    nz = 20

    def f(dz_profile):
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dz_profile)])
        zc = cum[:-1] - cum[0]
        m = box.mask_on_coords(gx, gy, zc)
        return jnp.sum(jnp.where(m, zc[None, None, :], 0.0))

    prof = jnp.full(nz, DX, dtype=jnp.float32)
    val = jax.jit(f)(prof)
    g = jax.grad(f)(prof)
    assert np.isfinite(float(val))
    assert g.shape == (nz,)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.abs(g).sum()) > 0.0
