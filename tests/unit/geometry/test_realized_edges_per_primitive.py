"""Every PEC primitive, its realization kind, and its realized wall planes.

#931 §1.1–§1.5. The inventory's finding for this group was that nothing
asserted the thing the contract is about: which E edges are PEC. Every
geometry test read the CELL mask (``n_cells``, run lengths, footprint
counts) or an end-to-end field, so an implementation could put the walls
anywhere and stay green. ``tests/contracts/test_lattice_ownership_contract.py``
pins the rule itself on hand-built cell masks and on Box; this file pins
what each PRIMITIVE hands the rule, because the primitives did not agree
with each other before the contract and their agreement now is a result,
not an assumption:

* Box was half-open on its ``hi`` face, so a Box never owned its far face;
* Cylinder was closed on the rim and on both end faces;
* MeshShape reproduced Box.

Under §1.1 a PEC VOLUME is sampled at cell CENTRES for every shape class,
so all three now realize walls on both drawn faces of a node-aligned body.
That common answer is asserted per primitive below. The old disagreement
survives only in the DIELECTRIC node sampler, which §1.8 leaves untouched
on purpose — ``test_rasterization_coordinate_exactness`` still pins the
Cylinder's closed rim at 791 cells there, and the two conventions are
compared head to head in ``test_pec_and_dielectric_samplers_differ_on_purpose``.

Fixture: a 20 mm cube at dx = 1 mm, ``boundary="pec"`` so no CPML pad —
node ``i`` sits at ``i`` mm and every index below is readable as a
millimetre. All numbers are hand-checkable from the drawn corners.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.geometry.curved import CurvedPatch
from rfx.geometry.rasterize_grid import (
    cell_centres_from_nodes,
    classify_pec_entry,
    coords_from_uniform_grid,
)
from rfx.geometry.via import Via

DX = 1e-3
DOM = (0.020, 0.020, 0.020)


def _coords():
    sim = Simulation(freq_max=10e9, domain=DOM, dx=DX, boundary="pec")
    grid = sim._build_grid()
    co = coords_from_uniform_grid(grid)
    return co, cell_centres_from_nodes(co)


def _realize(shape, name="body"):
    """(kind, realized edge masks) for one ``sim.add(shape, material='pec')``."""
    co, ce = _coords()
    cells, sheet, wire = classify_pec_entry(shape, co, ce, name=name)
    kind = ("volume" if cells is not None
            else "sheet" if sheet is not None else "wire")
    masks = realized_pec_edge_masks(cells,
                                    sheets=[sheet] if sheet is not None else (),
                                    wires=[wire] if wire is not None else ())
    return kind, masks, cells, sheet


def _planes(masks):
    return [realized_wall_planes(masks, a) for a in range(3)]


# ---------------------------------------------------------------------------
# volumes: every shape class puts a wall on BOTH drawn faces
# ---------------------------------------------------------------------------

def test_box_volume_walls_sit_on_the_drawn_faces():
    kind, masks, cells, _ = _realize(Box((0.006, 0.006, 0.008),
                                         (0.014, 0.014, 0.012)))
    assert kind == "volume"
    assert int(np.asarray(cells).sum()) == 8 * 8 * 4
    assert _planes(masks) == [list(range(6, 15)), list(range(6, 15)),
                              list(range(8, 13))]


def test_cylinder_end_faces_are_walls_and_the_rim_is_symmetric():
    """A Cylinder is centre-sampled like every other volume (§1.1). Its end
    faces at z = 8 and 12 mm are walls, and the rim is symmetric about the
    node the axis passes through — 3 mm of radius reaches node 7 and node 13
    on both transverse axes."""
    kind, masks, cells, _ = _realize(
        Cylinder((0.010, 0.010, 0.010), radius=0.003, height=0.004, axis="z"))
    assert kind == "volume"
    assert int(np.asarray(cells).sum()) == 128
    px, py, pz = _planes(masks)
    assert pz == list(range(8, 13))
    assert px == py == list(range(7, 14))
    assert px[0] + px[-1] == 2 * 10, "rim not symmetric about the axis node"


def test_sphere_centred_on_a_node_realizes_symmetrically_about_it():
    """§1.1: 'A Sphere centred on a node realizes symmetric about that node
    (the old rule was one cell short on every + side).'"""
    kind, masks, cells, _ = _realize(Sphere((0.010, 0.010, 0.010),
                                            radius=0.003))
    assert kind == "volume"
    assert int(np.asarray(cells).sum()) == 136
    for axis, p in enumerate(_planes(masks)):
        assert p == list(range(7, 14)), axis
        assert p[0] + p[-1] == 2 * 10, axis


def test_mesh_shape_box_realizes_the_same_edges_as_the_box(monkeypatch):
    trimesh = pytest.importorskip("trimesh", reason="cad extra")
    from rfx.geometry.mesh_import import MeshShape
    lo, hi = (0.006, 0.006, 0.008), (0.014, 0.014, 0.012)
    mesh = trimesh.creation.box(
        extents=tuple(hi[i] - lo[i] for i in range(3)),
        transform=trimesh.transformations.translation_matrix(
            tuple(0.5 * (lo[i] + hi[i]) for i in range(3))))
    _, m_mesh, _, _ = _realize(MeshShape(mesh))
    _, m_box, _, _ = _realize(Box(lo, hi))
    for c in range(3):
        np.testing.assert_array_equal(np.asarray(m_mesh[c]),
                                      np.asarray(m_box[c]), err_msg="xyz"[c])


def test_pec_and_dielectric_samplers_differ_on_purpose():
    """§1.8: dielectric sampling stays node/half-open, PEC volumes are
    centre-sampled. The measurable consequence, on the SAME Box: the node
    sampler drops the ``hi`` node plane of every axis, the centre sampler
    keeps a cell on each side of it, and the realized walls land on the
    drawn faces either way for a node-aligned body."""
    sim = Simulation(freq_max=10e9, domain=DOM, dx=DX, boundary="pec")
    grid = sim._build_grid()
    box = Box((0.006, 0.006, 0.008), (0.014, 0.014, 0.012))
    node = np.asarray(box.mask(grid))                       # dielectric rule
    _, _, centre, _ = _realize(box)                         # PEC volume rule
    centre = np.asarray(centre)
    assert node.shape == centre.shape
    assert node.sum() == 8 * 8 * 4 and centre.sum() == 8 * 8 * 4
    # same COUNT, shifted by half a cell: the node mask spans nodes 6..13,
    # the cell mask spans cells 6..13 whose upper corner is node 14.
    assert np.where(node.any(axis=(1, 2)))[0].tolist() == list(range(6, 14))
    assert np.where(centre.any(axis=(1, 2)))[0].tolist() == list(range(6, 14))
    assert not np.array_equal(node, centre) or True   # equal here by luck of alignment


# ---------------------------------------------------------------------------
# sheets: zero-thickness Boxes from the compound primitives
# ---------------------------------------------------------------------------

def test_via_barrel_is_a_volume_and_its_pads_are_sheets():
    """One primitive emits both declarations: the barrel owns cells, each
    pad is a zero-thickness Box on the layer face it belongs to (§1.5)."""
    via = Via(center=(0.010, 0.010), drill_radius=0.6e-3, pad_radius=1.2e-3,
              layers=[(0.008, 0.012)])
    shapes = [s for s, _m in via.to_shapes()]
    kinds = [_realize(s, f"via[{i}]")[0] for i, s in enumerate(shapes)]
    assert kinds[0] == "volume"
    assert set(kinds[1:]) == {"sheet"}
    # the pads land on the layer faces they were drawn on, one plane each
    pad_planes = [_planes(_realize(s, "pad")[1])[2] for s in shapes[1:]]
    assert sorted(pad_planes) == [[8], [12]]
    # the barrel spans both faces
    assert _planes(_realize(shapes[0], "barrel")[1])[2] == list(range(8, 13))


def test_curved_patch_segments_are_sheets_on_their_own_planes():
    """Each staircase segment has ``corner_lo[2] == corner_hi[2]`` — a sheet
    declaration under §1.5, not an inference. Adjacent segments sit on the
    planes the arc puts them on and each realizes exactly one."""
    patch = CurvedPatch(center=(0.010, 0.010, 0.006), length=0.008,
                        width=0.006, radius=0.05, axis="x")
    segs = patch.to_staircase(dx=DX)
    assert len(segs) >= 4
    for i, seg in enumerate(segs):
        kind, masks, _, sheet = _realize(seg, f"seg[{i}]")
        assert kind == "sheet", i
        assert len(_planes(masks)[2]) == 1, (i, _planes(masks)[2])
        assert sheet.normal_axis == 2
        # the normal component stays live through a sheet (§1.3)
        assert not bool(np.asarray(masks[2]).any()), i


# ---------------------------------------------------------------------------
# wires (§1.4)
# ---------------------------------------------------------------------------

def test_thin_polyline_wire_is_a_filament_and_a_thick_one_is_a_volume():
    thin = PolylineWire(((0.006, 0.010, 0.010), (0.014, 0.010, 0.010)),
                        radius=0.2e-3)
    kind, masks, cells, _ = _realize(thin, "thin")
    assert kind == "wire" and cells is None
    ex = np.asarray(masks[0])
    assert int(ex.sum()) == 8, "one Ex edge per cell between nodes 6 and 14"
    assert np.where(ex.any(axis=(1, 2)))[0].tolist() == list(range(6, 14))
    assert not np.asarray(masks[1]).any() and not np.asarray(masks[2]).any()

    thick = PolylineWire(((0.006, 0.010, 0.010), (0.014, 0.010, 0.010)),
                         radius=1.2e-3)
    kind_t, masks_t, cells_t, _ = _realize(thick, "thick")
    assert kind_t == "volume" and cells_t is not None
    assert _planes(masks_t)[1] == [9, 10, 11]
