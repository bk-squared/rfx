"""Issue #589 -- REPORT-ONLY topology findings that would have named the short.

Root cause (verified on main 06cf29f0 by direct inspection of
``sim._assemble_materials``): the attempt-2 coax-MSL junction fixture
declares a full-plane PEC ground sheet and THEN a PTFE ``Cylinder`` meant
as the clearance hole. ``rfx/api/_compile.py::_assemble_materials`` is
PEC-OR-only (``pec_mask = pec_mask | mask``); a dielectric overwrites
eps/sigma but can never clear PEC, and there is no CSG subtraction shape.
The coax pin therefore passed through a SOLID ground plane and the
settled run measured S00 = (-0.9928, -0.0048) at 6 GHz -- a short. The
only structural test asserted pin-column PEC continuity, which is
trivially true through a solid ground.

A REPORT-ONLY finding (no refusal, no gate; relevance is the
user's judgment):

(i)  ``dielectric-after-conductor-no-op`` -- a ``fidelity_report`` finding
     kind on the DIELECTRIC row: cells this entity shares with a PEC-
     assembled entity declared EARLIER are a no-op. The existing
     ``claimed-by-conductor`` finding (fires for either order) stays
     byte-identical; only the ordered case adds a kind.

The fixture geometry below is a copy of the committed attempt-2 fixture
(``tests/unit/sparams/test_coax_msl_transition.py::_build_coax_msl_transition_sim_
attempt2``, constants reproduced inline so this file does not import a
file another change owns), plus the same geometry with the ground plane
built as 20 half-cell PEC Boxes generated from the integer-lattice disk
``di^2 + dj^2 <= 16`` (the attempt-3 recipe). Fail-before-fix: on the
source as of 88c49bdc, ``test_rule_i_fires_on_the_shorted_junction_copy``
FAILS (no such kind); the negative tests pass trivially there, so their
value is only in combination with the positive ones.

What the fixture declares under #931 (lattice ownership contract). Each
conductor is the kind it physically is. The ground foil and the microstrip
trace are 35 um-class copper on a laminate face, so they are SHEETS --
zero-thickness Boxes at ``N_GND * DX`` and ``N_TRACE * DX`` (contract
§1.5), realized on one node plane with the normal E edge through them
live. The coax pin is a real 3-D body and stays a PEC VOLUME,
centre-sampled (§1.1). The substrate and the PTFE clearance Cylinder are
dielectrics: node-sampled, untouched by the contract. ``_half_cell`` still
bounds the in-plane footprints, because closed footprint sampling (§1.3)
turns ``(n-0.5)dx -> (n+0.5)dx`` into exactly node ``n``; that is why the
twenty-box clearance recipe survives as twenty SHEETS on ONE plane, whose
footprints the realization unions (§1.3) -- the ledger this paragraph
replaces predicted it could not, on the reading that a one-node row needs
two zero-extent axes, and measured only z is zero-extent. Metal at the
junction plane is read as REALIZED WALL NODES through the one realization
function (§1.7), not from the primal cell mask: a sheet owns no cell, and
a volume's far face is not in the cell mask either.

``test_junction_plane_metal_under_a_sheet_ground`` below is the
declaration in isolation, on the same two oracle counts (annulus 36/36,
first lattice ring 16/16 at the junction plane).

Expected noise, not a defect: the assembly warns that the ground sheet is
buried in a dielectric over 48 of its footprint nodes. It is -- the PTFE
clearance Cylinder is drawn with a one-cell margin on each side of the
junction plane (``_margin_cylinder_z``), which is what makes it cover the
plane at all, and covering the plane is the #589 defect this file is
about. The advisory is reporting the fixture correctly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.csg import Box, Cylinder

DX = 100e-6
PIN_R = 0.2e-3
OUTER_R = 0.6e-3
EPS_COAX = 2.1
H_SUB = 300e-6
EPS_SUB = 3.66
W_TRACE = 600e-6
JUNCTION_X = 1.0e-3
LY = 3.4e-3
Y_C = LY / 2.0
N_GND, N_SUB_LO, N_SUB_HI, N_TRACE = 25, 26, 28, 29
JUNCTION_Z = N_GND * DX
CLEAR_R = PIN_R + 2 * DX
FEED_X_2 = 11.0e-3
LX_2 = 12.5e-3
LZ_2 = JUNCTION_Z + H_SUB + DX + 1.0e-3
FREQ_MAX_2 = 16.0e9

RULE_I_KIND = "dielectric-after-conductor-no-op"


def _half_cell(n_lo, n_hi):
    """Box bounds that select EXACTLY nodes [n_lo, n_hi] on an axis with
    spacing DX, under the two samplers this fixture uses: a SHEET's
    in-plane footprint (closed, #931 §1.3) and a dielectric (node,
    half-open, §1.1). Faces on cell midpoints, never on a node plane
    (#802 knife edge). NOT for a PEC volume face: a volume is sampled at
    cell CENTRES (§1.1) and ``_half_cell(n, n)`` there means cell n-1.
    """
    return (n_lo - 0.5) * DX, (n_hi + 0.5) * DX


def _margin_cylinder_z(n_lo, n_hi):
    z_lo, z_hi = n_lo * DX, n_hi * DX
    return 0.5 * (z_lo + z_hi), (z_hi - z_lo) + 2 * DX


def _ground_plane_boxes_with_clearance(lx, ly, jx, jy, k, r_cells):
    """The ground foil at node plane k as zero-thickness PEC Boxes (SHEET
    declarations, #931 §1.5) with a hole equal to the integer-lattice disk
    ``di^2 + dj^2 <= r_cells^2`` around (jx, jy).

    Two x-slabs (outer faces on the domain's own 0.0 / lx), two y-strips in
    the hole's x-band, then per row |dj| two boxes filling x beyond the
    row's half-width isqrt(r^2 - dj^2). 20 boxes for r_cells = 4, all on
    plane k; §1.3 unions their footprints before the edge rule, so the
    seams between them are metal, not slits (measured: the union is the
    full plane minus the 49-node hole, 4410 -> 4361 footprint nodes).
    The in-plane bounds stay ``_half_cell`` -- closed footprint sampling
    turns a one-cell-wide row into exactly the one node it names, so a
    row is a sheet with a one-node footprint, not a line with two
    zero-extent axes.
    """
    rows = {dj: math.isqrt(r_cells * r_cells - dj * dj)
            for dj in range(-r_cells, r_cells + 1)}
    z = k * DX
    boxes = [
        Box((0.0, 0.0, z), (_half_cell(jx - r_cells - 1, jx - r_cells - 1)[1], ly, z)),
        Box((_half_cell(jx + r_cells + 1, jx + r_cells + 1)[0], 0.0, z), (lx, ly, z)),
    ]
    xl, xh = _half_cell(jx - r_cells, jx + r_cells)
    boxes.append(Box((xl, 0.0, z),
                     (xh, _half_cell(jy - r_cells - 1, jy - r_cells - 1)[1], z)))
    boxes.append(Box((xl, _half_cell(jy + r_cells + 1, jy + r_cells + 1)[0], z),
                     (xh, ly, z)))
    for dj, w in rows.items():
        if w >= r_cells:
            continue
        yl, yh = _half_cell(jy + dj, jy + dj)
        boxes.append(Box((xl, yl, z), (_half_cell(jx - w - 1, jx - w - 1)[1], yh, z)))
        boxes.append(Box((_half_cell(jx + w + 1, jx + w + 1)[0], yl, z), (xh, yh, z)))
    return boxes


def _junction_sim(*, open_annulus: bool) -> Simulation:
    """attempt-2 geometry (open_annulus=False) or the same with the ground
    sheet built around a lattice-disk clearance hole (open_annulus=True).

    Entity order, the thing rule (i) is about: ground PEC, PTFE clearance
    Cylinder, substrate Box, trace PEC Box, pin PEC Cylinder. Under #931
    the two foils (ground, trace) are SHEET declarations and the pin is a
    VOLUME; see the module docstring.
    """
    sim = Simulation(snap="declared",
        freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    sim.add_material("sub", eps_r=EPS_SUB)
    sim.add_material("ptfe", eps_r=EPS_COAX)

    if open_annulus:
        jx, jy = int(round(JUNCTION_X / DX)), int(round(Y_C / DX))
        for b in _ground_plane_boxes_with_clearance(
                LX_2, LY, jx, jy, N_GND, int(round(CLEAR_R / DX))):
            sim.add(b, material="pec")
    else:
        z_gnd = N_GND * DX
        sim.add(Box((0.0, 0.0, z_gnd), (LX_2, LY, z_gnd)), material="pec")
    clr_c, clr_h = _margin_cylinder_z(N_GND, N_SUB_LO)
    sim.add(Cylinder(center=(JUNCTION_X, Y_C, clr_c), radius=CLEAR_R,
                     height=clr_h, axis="z"), material="ptfe")
    sub_lo, sub_hi = _half_cell(N_SUB_LO, N_SUB_HI)
    sim.add(Box((0.0, 0.0, sub_lo), (LX_2, LY, sub_hi)), material="sub")
    z_trc = N_TRACE * DX
    sim.add(Box((JUNCTION_X, Y_C - W_TRACE / 2, z_trc),
                (LX_2, Y_C + W_TRACE / 2, z_trc)), material="pec")
    pin_c, pin_h = _margin_cylinder_z(N_GND, N_TRACE)
    sim.add(Cylinder(center=(JUNCTION_X, Y_C, pin_c), radius=PIN_R,
                     height=pin_h, axis="z"), material="pec")

    sim.add_coaxial_port(
        position=(JUNCTION_X, Y_C, N_GND * DX), face="bottom",
        pin_radius=PIN_R, outer_radius=OUTER_R, impedance=50.0,
    )
    sim.add_msl_port(
        position=(FEED_X_2, Y_C, N_SUB_LO * DX), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0, eps_r_sub=EPS_SUB,
    )
    return sim


def _rows(report, material_name):
    return [it for it in report
            if it.get("entity", "").endswith(f"'{material_name}'")]


def _findings(item, kind):
    return [f for f in item.get("findings", []) if f["kind"] == kind]


def _junction_plane_metal(sim):
    """(realized wall nodes on the junction plane, r_from_axis) on the
    padded grid.

    Read through the run's own realized conductor set (#931 §1.7), which
    is the conductor view: a node carries metal iff some E
    edge tangential to the port axis and incident to it is PEC. The primal
    cell mask cannot answer this -- a sheet owns no cell, and a volume's
    far face is not a cell either (§1.9).
    """
    grid = sim._build_grid()
    realized = sim._port_realized_edges(grid)
    k = int(grid.position_to_index(sim._coaxial_ports[0].position)[2])
    plane = realized.wall_nodes_on_plane(2, k)
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * DX - JUNCTION_X
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * DX - Y_C
    return plane, np.hypot(x[:, None], y[None, :])


PIN_REACH = PIN_R + DX / math.sqrt(2.0)



def test_fixture_copies_differ_only_by_the_junction_hole():
    """The two copies' realized metal at the junction plane differs by the
    ground's hole and by nothing else.

    Counts measured on the #931 fixture (2026-09-07), with the pre-#931
    node-sampled values they replace:
      annulus nodes                        36     (geometry only, unchanged)
      shorted, metal in the annulus        36/36  (unchanged)
      open, metal in the annulus            8/36  (was 0/36)
      open, metal beyond the pin's reach     0/36 (the 8 above ARE the pin)
      metal at r <= PIN_R, either copy     13     (was 11 open / 13 shorted)
      the two wall maps' difference        28     (49-node hole - 21 pin nodes)
    Both moves have one cause: a PEC volume is centre-sampled (§1.1), so
    the pin's realized rim is symmetric -- no lattice point sits on the
    r == PIN_R knife edge any more, which is what made the open copy read
    11 and the shorted copy 13 -- and it reaches half a cell diagonal past
    200 um, to the eight nodes at 224 um. Those eight are inside the
    clearance annulus but are the PIN's own metal, not the ground's.
    Absorber continuation is outside the drawn-geometry comparison.
    """
    m_short, r = _junction_plane_metal(_junction_sim(open_annulus=False))
    m_open, _ = _junction_plane_metal(_junction_sim(open_annulus=True))
    annulus = (r > PIN_R + 1e-9) & (r <= CLEAR_R + 1e-9)
    assert int(annulus.sum()) == 36
    assert int((m_short & annulus).sum()) == 36
    assert int((m_open & annulus).sum()) == 8
    assert int((m_open & annulus & (r > PIN_REACH)).sum()) == 0
    assert sorted(set(np.round(r[m_open & annulus] * 1e6).astype(int).tolist())) \
        == [224]
    assert int((m_open & (r <= PIN_R + 1e-9)).sum()) == 13
    assert int((m_short & (r <= PIN_R + 1e-9)).sum()) == 13
    diff = m_short ^ m_open
    grid = _junction_sim(open_annulus=False)._build_grid()
    interior = np.zeros_like(diff)
    interior[grid.pad_x_lo:grid.shape[0] - grid.pad_x_hi,
             grid.pad_y_lo:grid.shape[1] - grid.pad_y_hi] = True
    diff &= interior
    assert int(diff.sum()) == 28
    assert not bool((diff & (r > CLEAR_R + 1e-9)).any())



def test_rule_i_fires_on_the_shorted_junction_copy():
    """52 metal faces have declared PTFE cells on both sides.
    The ground is a sheet: it owns no cell, so the 52 count is faces.
    """
    report = _junction_sim(open_annulus=False).fidelity_report(print_report=False)
    (ptfe,) = _rows(report, "ptfe")
    hits = _findings(ptfe, RULE_I_KIND)
    assert len(hits) == 1, [f["kind"] for f in ptfe["findings"]]
    f = hits[0]
    assert f["overlap_faces"] == 52
    assert f["overlap_cells"] == 0
    assert ptfe["n_cells"] == 156
    assert f["conductor_entities"] == [0]
    assert "geometry[0]" in f["detail"] and "52" in f["detail"]
    assert "52 faces" in f["detail"]
    assert "OR-only" in f["detail"] or "cannot carve" in f["detail"]
    assert f["remedy"]
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1


def test_rule_i_is_silent_when_the_hole_is_built_into_the_conductor():
    """Material cells inside the clearance meet no complete metal face."""
    report = _junction_sim(open_annulus=True).fidelity_report(print_report=False)
    (ptfe,) = _rows(report, "ptfe")
    assert _findings(ptfe, RULE_I_KIND) == []
    for it in report:
        assert _findings(it, RULE_I_KIND) == [], it["entity"]


@pytest.mark.parametrize("normal", range(3))
@pytest.mark.parametrize("through", [False, True])
def test_rule_i_ground_sheet_requires_dielectric_on_both_sides(normal, through):
    d = 1e-3
    def permute(values):
        values = list(values)
        values[normal], values[2] = values[2], values[normal]
        return tuple(v * d for v in values)

    sim = Simulation(freq_max=10e9, domain=permute((12, 12, 8)), dx=d,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box(permute((2, 2, 3)), permute((10, 10, 3))), material="pec")
    sim.add(Box(permute((2, 2, 2 if through else 3)), permute((10, 10, 5))),
            material="sub")
    (sub,) = _rows(sim.fidelity_report(print_report=False), "sub")
    hits = _findings(sub, RULE_I_KIND)
    if through:
        assert len(hits) == 1
        assert hits[0]["overlap_faces"] == 64
        assert hits[0]["overlap_cells"] == 0
        assert "64 faces" in hits[0]["detail"]
        assert hits[0]["conductor_entities"] == [0]
    else:
        assert hits == []


def test_rule_i_does_not_fire_on_pec_after_dielectric_the_intended_contacts():
    """The pin (PEC, declared last) passes through the PTFE Cylinder and the
    substrate and touches the trace; the trace (PEC) lies on the substrate.
    Those are PEC-AFTER-dielectric overlaps -- the conductor wins by design
    -- and get no new finding on either copy. The order-blind
    claimed-by-conductor row on the substrate/PTFE is pre-existing."""
    for open_annulus in (False, True):
        report = _junction_sim(open_annulus=open_annulus).fidelity_report(
            print_report=False)
        pec_rows = [it for it in report if it.get("entity", "").endswith("'pec'")]
        assert pec_rows, "fixture must have PEC rows"
        for it in pec_rows:
            assert _findings(it, RULE_I_KIND) == [], it["entity"]
        (sub,) = _rows(report, "sub")
        assert _findings(sub, RULE_I_KIND) == []


def test_rule_i_minimal_slab_then_cylinder_fires_and_reverse_is_silent():
    def _sim(dielectric_first: bool):
        """A one-cell PEC VOLUME, not a sheet, is built because the accumulator must hold for both kinds."""
        sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                         boundary="cpml", cpml_layers=4)
        sim.add_material("ptfe", eps_r=2.1)
        slab = Box((0.0, 0.0, 4e-3), (10e-3, 10e-3, 5e-3))
        hole = Cylinder(center=(5e-3, 5e-3, 5e-3), radius=1.5e-3, height=3e-3,
                        axis="z")
        if dielectric_first:
            sim.add(hole, material="ptfe")
            sim.add(slab, material="pec")
        else:
            sim.add(slab, material="pec")
            sim.add(hole, material="ptfe")
        return sim

    rep = _sim(dielectric_first=False).fidelity_report(print_report=False)
    (ptfe,) = _rows(rep, "ptfe")
    hits = _findings(ptfe, RULE_I_KIND)
    assert len(hits) == 1
    assert hits[0]["conductor_entities"] == [0]
    assert 0 < hits[0]["overlap_cells"] <= ptfe["n_cells"]
    assert hits[0]["overlap_faces"] == 0
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1

    rep = _sim(dielectric_first=True).fidelity_report(print_report=False)
    (ptfe,) = _rows(rep, "ptfe")
    assert _findings(ptfe, RULE_I_KIND) == []
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1


def test_rule_i_lists_every_earlier_conductor_and_sums_the_union():
    """Two earlier PEC volumes overlapping the same dielectric: the finding
    names both indices and counts the UNION of their cells (a cell claimed by
    both is one no-op cell, not two).
    The oracle must use the samplers the code uses, or it agrees for the wrong reason: a PEC VOLUME is realized from cell CENTRES and a dielectric from NODES.
    """
    sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("d", eps_r=4.0)
    sim.add(Box((0.0, 0.0, 3.5e-3), (10e-3, 10e-3, 4.5e-3)), material="pec")
    sim.add(Box((0.0, 0.0, 3.5e-3), (5e-3, 10e-3, 4.5e-3)), material="pec")
    sim.add(Box((2.5e-3, 2.5e-3, 2.5e-3), (7.5e-3, 7.5e-3, 6.5e-3)), material="d")
    rep = sim.fidelity_report(print_report=False)
    (d,) = _rows(rep, "d")
    (f,) = _findings(d, RULE_I_KIND)
    assert f["conductor_entities"] == [0, 1]
    from rfx.geometry.rasterize_grid import (
        centres_from_uniform_grid, cell_sizes_from_uniform_grid, pec_volume_cell_mask)
    grid = sim._build_grid()
    centres = centres_from_uniform_grid(grid)
    sizes = cell_sizes_from_uniform_grid(grid)
    m0 = np.asarray(pec_volume_cell_mask(sim._geometry[0].shape, centres, sizes), bool)
    m1 = np.asarray(pec_volume_cell_mask(sim._geometry[1].shape, centres, sizes), bool)
    m2 = np.asarray(sim._geometry[2].shape.mask(grid), bool)
    assert f["overlap_cells"] == int(((m0 | m1) & m2).sum())
    assert f["overlap_faces"] == 0
    n0 = np.asarray(sim._geometry[0].shape.mask(grid), bool)
    assert not bool((m0 == n0).all()), "centre and node samplers must differ here"


def test_rule_i_counts_volume_cells_and_sheet_faces_under_separate_keys():
    """An earlier PEC volume AND an earlier PEC sheet inside one dielectric:
    the volume's shared cells go to ``overlap_cells`` and the sheet's metal
    faces with dielectric on both sides go to ``overlap_faces``; neither
    count is added into the other.

    On the 1 mm grid: the volume (3,3,4)-(9,9,5) mm is centre-sampled to
    cells 3..8 x 3..8 x 4 = 36 cells; the sheet at z = 6 mm has footprint
    nodes 2..10 on x and y, so 8 x 8 = 64 complete faces; the dielectric
    (2,2,2)-(10,10,7) mm holds nodes 2..9 x 2..9 x 2..6, which covers all 36
    cells and both cells (z = 5 and 6) under every face.
    """
    d = 1e-3
    sim = Simulation(freq_max=10e9, domain=(12 * d, 12 * d, 8 * d), dx=d,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((3 * d, 3 * d, 4 * d), (9 * d, 9 * d, 5 * d)), material="pec")
    sim.add(Box((2 * d, 2 * d, 6 * d), (10 * d, 10 * d, 6 * d)), material="pec")
    sim.add(Box((2 * d, 2 * d, 2 * d), (10 * d, 10 * d, 7 * d)), material="sub")
    (sub,) = _rows(sim.fidelity_report(print_report=False), "sub")
    (f,) = _findings(sub, RULE_I_KIND)
    assert f["conductor_entities"] == [0, 1]

    from rfx.geometry.rasterize_grid import (
        centres_from_uniform_grid, cell_sizes_from_uniform_grid, pec_volume_cell_mask)
    grid = sim._build_grid()
    volume = np.asarray(pec_volume_cell_mask(
        sim._geometry[0].shape, centres_from_uniform_grid(grid),
        cell_sizes_from_uniform_grid(grid)), bool)
    dielectric = np.asarray(sim._geometry[2].shape.mask(grid), bool)
    expected_cells = int((volume & dielectric).sum())
    x = np.array([grid.node_of("x", i) for i in range(grid.nx)])
    y = np.array([grid.node_of("y", j) for j in range(grid.ny)])
    k = int(np.argmin(np.abs(np.array(
        [grid.node_of("z", n) for n in range(grid.nz)]) - 6 * d)))
    inside_x = (x > 2 * d - 1e-9) & (x < 10 * d + 1e-9)
    inside_y = (y > 2 * d - 1e-9) & (y < 10 * d + 1e-9)
    footprint = inside_x[:, None] & inside_y[None, :]
    corners = (footprint[:-1, :-1] & footprint[1:, :-1]
               & footprint[:-1, 1:] & footprint[1:, 1:])
    both_sides = dielectric[:-1, :-1, k] & dielectric[:-1, :-1, k - 1]
    expected_faces = int((corners & both_sides).sum())
    assert (expected_cells, expected_faces) == (36, 64)

    assert f["overlap_cells"] == expected_cells
    assert f["overlap_faces"] == expected_faces
    assert "64 metal faces" in f["detail"]
    assert "36 cells are already PEC" in f["detail"]


def test_rule_i_keys_on_assembled_pec_not_on_the_name_pec():
    """A named material with sigma >= the assembly's PEC threshold is
    realized as PEC by ``_assemble_materials``; the rule follows the
    assembly, exactly as ``declared-lossy-realized-pec`` does."""
    sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_material("copper", eps_r=1.0, sigma=5.8e7)
    sim.add_material("ptfe", eps_r=2.1)
    sim.add(Box((0.0, 0.0, 4.5e-3), (10e-3, 10e-3, 5.5e-3)), material="copper")
    sim.add(Cylinder(center=(5e-3, 5e-3, 5e-3), radius=1.5e-3, height=3e-3,
                     axis="z"), material="ptfe")
    rep = sim.fidelity_report(print_report=False)
    (ptfe,) = _rows(rep, "ptfe")
    (f,) = _findings(ptfe, RULE_I_KIND)
    assert f["conductor_entities"] == [0]


class _NoBoundsBox(Box):
    """A Box that exposes no bounding_box() (the ``no-analytic-bounds``
    branch of fidelity_report), while ``mask()`` still works so the
    ASSEMBLY is untouched -- only the audit's own rasterization is made to
    fail, via the monkeypatch in the tests below."""

    def bounding_box(self):
        raise AttributeError("no analytic bounds")


def _slab_then_cylinder_sim(slab_cls):
    """Minimal ordered pair: a one-cell PEC VOLUME at cell 4, then a
    dielectric Cylinder through it (the shape of the #589 no-op).

    A volume, not a sheet, and deliberately so: rule (i) has to hold for
    both realizations, and the volume half is the one whose accumulator
    never moved. The sheet half is the junction fixture above.

    Drawn ON-LATTICE (``4*dx .. 5*dx``), because ``_half_cell(4, 4)`` names
    node 4 under the two node samplers and cell 3 under the volume sampler
    (§1.1) -- the trap this file's own fixture was caught by.
    """
    sim = Simulation(freq_max=20e9, domain=(1.0e-3, 1.0e-3, 1.0e-3), dx=DX,
                     cpml_layers=4,
                     boundary=BoundarySpec(x="pec", y="pec", z="pec"))
    sim.add_material("d", eps_r=2.0)
    z_lo, z_hi = 4 * DX, 5 * DX
    sim.add(slab_cls((0.0, 0.0, z_lo), (1.0e-3, 1.0e-3, z_hi)), material="pec")
    sim.add(Cylinder(center=(0.5e-3, 0.5e-3, 0.5e-3), radius=0.25e-3,
                     height=0.6e-3, axis="z"), material="d")
    return sim


def _failing_entity_mask(monkeypatch, shape_cls, exc):
    """Make fidelity_report's OWN rasterization raise for entities whose
    shape is ``shape_cls``; everything else rasterizes as before."""
    import rfx.fidelity as fid
    real = fid._entity_mask

    def patched(entry, sim, grid, nonuniform, **kw):
        if isinstance(entry.shape, shape_cls):
            raise exc
        return real(entry, sim, grid, nonuniform, **kw)

    monkeypatch.setattr(fid, "_entity_mask", patched)


def test_rule_i_reports_a_conductor_whose_mask_fails_instead_of_dropping_it(monkeypatch):
    """Before this test: ``except Exception: pass`` around the accumulator
    silently left the conductor out of pec_before, so a real
    dielectric-after-conductor no-op went UNREPORTED with no marker. Now
    the conductor's own row carries ``rasterization-failed`` with the
    exception class, and the later dielectric row says the ordered audit
    is incomplete and names the conductor."""
    _failing_entity_mask(monkeypatch, _NoBoundsBox,
                         RuntimeError("synthetic rasterization failure"))
    sim = _slab_then_cylinder_sim(_NoBoundsBox)
    report = sim.fidelity_report(print_report=False)

    (gnd,) = [it for it in report if it["entity"].startswith("geometry[0]")]
    assert "'pec'" in gnd["entity"]
    kinds = [f["kind"] for f in gnd["findings"]]
    assert "no-analytic-bounds" in kinds
    (rf,) = _findings(gnd, "rasterization-failed")
    assert rf["exception"] == "RuntimeError"
    assert "RuntimeError" in rf["detail"]
    assert "synthetic rasterization failure" in rf["detail"]
    assert rf["remedy"]

    (d,) = _rows(report, "d")
    assert _findings(d, RULE_I_KIND) == []
    (un,) = _findings(d, "dielectric-after-conductor-unaudited")
    assert un["conductor_entities"] == [0]
    assert "geometry[0]" in un["detail"] and "RuntimeError" in un["detail"]


def test_rule_i_control_the_same_pair_with_a_working_mask_fires_normally():
    """Control for the test above: identical geometry, the slab as a plain
    Box, no injected failure -> the ordered no-op finding fires on the
    Cylinder row and no rasterization/unaudited finding exists anywhere."""
    report = _slab_then_cylinder_sim(Box).fidelity_report(print_report=False)
    (d,) = _rows(report, "d")
    (f,) = _findings(d, RULE_I_KIND)
    assert f["conductor_entities"] == [0] and f["overlap_cells"] > 0
    assert f["overlap_faces"] == 0
    for it in report:
        assert _findings(it, "rasterization-failed") == [], it["entity"]
        assert _findings(it, "dielectric-after-conductor-unaudited") == [], it["entity"]


def test_rule_i_rasterizes_each_conductor_once(monkeypatch):
    """Twelve one-cell PEC strips (cell ``s`` on x, cell 4 on z), then a
    dielectric slab over all of them: every conductor is named as a
    contributor, and the audit rasterized each entity exactly once (the
    earlier implementation re-rasterized every earlier conductor per
    overlapping dielectric).

    Drawn ON-LATTICE (``s*dx .. (s+1)*dx``) since 2026-09-07: as
    ``_half_cell`` draws the strips are centre-sampled one cell low
    (§1.1), which put strip 0 at cell -1 — outside a domain with no
    padding, where the assembly refuses it as a zero-cell volume while
    the report lists it as a refused, node-sampled row. Same counts
    either way (13 rasterizations, 12 contributors, 120 shared cells,
    measured); the fixture is now a model the solve accepts."""
    import rfx.fidelity as fid
    real = fid._entity_mask
    calls = []

    def counting(entry, sim, grid, nonuniform, **kw):
        calls.append(id(entry))
        return real(entry, sim, grid, nonuniform, **kw)

    monkeypatch.setattr(fid, "_entity_mask", counting)

    n_strips = 12
    sim = Simulation(freq_max=20e9, domain=(1.2e-3, 1.0e-3, 1.0e-3), dx=DX,
                     cpml_layers=4,
                     boundary=BoundarySpec(x="pec", y="pec", z="pec"))
    sim.add_material("d", eps_r=2.0)
    z_lo, z_hi = 4 * DX, 5 * DX
    for s in range(n_strips):
        sim.add(Box((s * DX, 0.0, z_lo), ((s + 1) * DX, 1.0e-3, z_hi)),
                material="pec")
    sim.add(Box((0.0, 0.0, _half_cell(3, 5)[0]), (1.2e-3, 1.0e-3, _half_cell(3, 5)[1])),
            material="d")
    report = sim.fidelity_report(print_report=False)

    (d,) = _rows(report, "d")
    (f,) = _findings(d, RULE_I_KIND)
    assert f["conductor_entities"] == list(range(n_strips))
    assert len(calls) == n_strips + 1
    assert len(set(calls)) == n_strips + 1



def test_junction_plane_metal_under_a_sheet_ground():
    """A full-plane ground declared as a sheet reproduces the oracle counts.

    The fixture's two load-bearing numbers are the clearance annulus (36
    nodes with ``PIN_R < r <= CLEAR_R``) and the first lattice ring (16
    nodes with ``PIN_R + dz/2 < r <= PIN_R + 3dz/2``). Under the pre-#931
    node sampler a ``_half_cell(25, 25)`` Box put metal on every node of
    plane 25 and both counts read full. Under the contract the honest
    declaration of that foil is a SHEET at ``z = N_GND * DX`` — a
    zero-thickness Box, §1.5 — whose footprint is sampled CLOSED on the
    two in-plane axes. Measured here: 36/36 and 16/16, the historic
    numbers, on the node line the port axis actually sits on.

    That symmetry is the reason the sheet is the right declaration and a
    one-cell volume is not: a volume occupying cells ``[a..b]`` realizes
    node planes ``[a..b+1]``, one node wider on the ``+`` side, so a hole
    cut on the cell lattice is never centred on the port axis.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks

    sim = Simulation(
        freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    z = N_GND * DX
    sim.add(Box((0.0, 0.0, z), (LX_2, LY, z)), material="pec")

    grid = sim._build_grid()
    sheets: list = []
    mats = sim._assemble_materials(grid, pec_sheets=sheets)
    (sheet,) = sheets
    assert mats[3] is None, "a sheet owns no cell (#931 §1.3)"
    k = int(grid.position_to_index((JUNCTION_X, Y_C, z))[2])
    assert sheet.plane == k and sheet.normal_axis == 2

    fp = np.asarray(sheet.footprint, bool)[:, :, k]
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * DX - JUNCTION_X
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * DX - Y_C
    r = np.hypot(x[:, None], y[None, :])

    annulus = (r > PIN_R + 1e-9) & (r <= CLEAR_R + 1e-9)
    assert int(annulus.sum()) == 36
    assert int((fp & annulus).sum()) == 36

    ring = (r > PIN_R + 0.5 * DX) & (r <= PIN_R + 1.5 * DX)
    assert int(ring.sum()) == 16
    assert int((fp & ring).sum()) == 16

    edges = realized_pec_edge_masks(None, sheets=sheets)
    assert bool(np.asarray(edges[0])[:, :, k].any())
    assert bool(np.asarray(edges[1])[:, :, k].any())
    assert not bool(np.asarray(edges[2]).any()), "normal Ez must stay live"
