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

Two findings, both REPORT-ONLY (no refusal, no gate; relevance is the
user's judgment):

(i)  ``dielectric-after-conductor-no-op`` -- a ``fidelity_report`` finding
     kind on the DIELECTRIC row: cells this entity shares with a PEC-
     assembled entity declared EARLIER are a no-op. The existing
     ``claimed-by-conductor`` finding (fires for either order) stays
     byte-identical; only the ordered case adds a kind.

(ii) ``coaxial_port_junction_short`` -- a preflight advisory: at each
     coaxial port's junction plane, registered PEC in the FIRST dielectric
     ring outside the pin means the pin is terminated in a short by
     registered geometry. The ring is defined ON THE LATTICE, half a cell
     outside the pin (``a + dz/2 < r <= a + 3dz/2``), so the pin's own
     knife-edge footprint cells at r == a (design review blocker 2: with a
     bare ``r > a`` and float64 node coordinates, (+2,0)/(0,+2) at exactly
     200 um were COUNTED and the advisory fired on the fixed geometry,
     2/16) are never in the ring.

The fixture geometry below is a copy of the committed attempt-2 fixture
(``tests/unit/sparams/test_coax_msl_transition.py::_build_coax_msl_transition_sim_
attempt2``, constants reproduced inline so this file does not import a
file another change owns), plus the same geometry with the ground plane
built as 20 half-cell PEC Boxes generated from the integer-lattice disk
``di^2 + dj^2 <= 16`` (the attempt-3 recipe). Fail-before-fix: on the
source as of 88c49bdc, ``test_rule_i_fires_on_the_shorted_junction_copy``
and ``test_rule_ii_fires_on_the_shorted_junction_copy`` FAIL (no such
kind / no such code); the negative tests pass trivially there, so their
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

# --- attempt-2 fixture constants, copied verbatim (tests/unit/sparams/test_coax_msl_
# transition.py, module constants + the attempt-2 block) ---------------------
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
RULE_II_CODE = "coaxial_port_junction_short"


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
    sim = Simulation(
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


def _preflight_rows(sim, code):
    return [i for i in sim.preflight() if i.code == code]


def _junction_plane_metal(sim):
    """(realized wall nodes on the junction plane, r_from_axis) on the
    padded grid.

    Read through the run's own realized conductor set (#931 §1.7), which
    is what preflight rule (ii) reads: a node carries metal iff some E
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


# The pin is a PEC VOLUME, centre-sampled: every node of an occupied cell
# carries its wall, so its own realized rim reaches half a cell diagonal
# past its radius. This is the bound preflight rule (ii) uses to separate
# the pin from OTHER registered conductor (rfx/api/_preflight.py).
PIN_REACH = PIN_R + DX / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Fixture witnesses: the two copies differ ONLY by the hole, and the hole is
# what the findings must see. Measured on 88c49bdc (design review).
# ---------------------------------------------------------------------------

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
    """
    m_short, r = _junction_plane_metal(_junction_sim(open_annulus=False))
    m_open, _ = _junction_plane_metal(_junction_sim(open_annulus=True))
    annulus = (r > PIN_R + 1e-9) & (r <= CLEAR_R + 1e-9)
    assert int(annulus.sum()) == 36
    # the shorted copy: every clearance-annulus node at the junction plane
    # carries a realized PEC wall (#589 root cause, measured 36/36)
    assert int((m_short & annulus).sum()) == 36
    # the open copy: the only metal left in the annulus is the pin's own
    # realized rim, all of it at r = 224 um <= PIN_REACH
    assert int((m_open & annulus).sum()) == 8
    assert int((m_open & annulus & (r > PIN_REACH)).sum()) == 0
    assert sorted(set(np.round(r[m_open & annulus] * 1e6).astype(int).tolist())) \
        == [224]
    # the pin is a volume in BOTH copies and realizes the same 13 nodes
    # within its own radius; the ground adds nothing there on the open copy
    assert int((m_open & (r <= PIN_R + 1e-9)).sum()) == 13
    assert int((m_short & (r <= PIN_R + 1e-9)).sum()) == 13
    # and the whole difference between the copies is the hole, minus the
    # 21 nodes the pin holds inside it
    diff = m_short ^ m_open
    assert int(diff.sum()) == 28
    assert not bool((diff & (r > CLEAR_R + 1e-9)).any())


# ---------------------------------------------------------------------------
# (i) dielectric-after-conductor-no-op (fidelity_report)
# ---------------------------------------------------------------------------

def test_rule_i_fires_on_the_shorted_junction_copy():
    """The PTFE clearance Cylinder (entity 1) is declared AFTER the ground
    sheet (entity 0): the cells it shares with the sheet at node 25 are a
    no-op under the OR-only assembly. FAILS on 88c49bdc: fidelity_report
    has no such kind.

    Pinned counts, re-derived on b5605391 (after #834's exact host-float64
    node coordinates) and identical under JAX_ENABLE_X64=0 and =1:
    overlap 48 of the Cylinder's 192 cells. The pre-#834 float32 path gave
    47 of 141 at x64=0 (and 48/192 at x64=1). Both numbers moved for the
    same reason, and it is this fixture's knife edges, not the check:
    * radius 0.4 mm = exactly 4 cells, so the lattice points (+-4, 0) and
      (0, +-4) sit ON the circle. ``r^2 <= R^2`` on exact float64 node
      coordinates resolves three of them inside ((+4,0), (-4,0), (0,-4);
      (0,+4) falls out by +3.2e-22 m^2 from the y-node product 21*1e-4 -
      1.7e-3) -> 48 cells per z plane; float32 coordinates resolved two
      ((+4,0), (0,+4)) -> 47.
    * the Cylinder's z span [24, 27] nodes (centre 25.5, height 3 dx) has
      both end faces ON node planes; the closed ``|h| <= height/2`` test
      now includes both -> 4 planes (24..27) instead of 3 (25..27):
      4 x 48 = 192, was 3 x 47 = 141.
    Re-pinning these is a fixture-realization statement, not a change to
    what rule (i) measures: the overlap is still every PTFE cell on the
    solid sheet at node 25 (48 = 49-cell lattice disk minus the one
    knife-edge cell that fell out).

    Both numbers survived the #931 redraw unchanged (measured
    2026-09-07): the ground is now a SHEET, and its realized footprint on
    plane 25 is the full plane, so the PTFE Cylinder's 48 nodes there are
    still every one of them shared. The finding names the unit -- the
    contributor reads "48 sheet footprint nodes", not "48 cells", because
    a sheet owns no cell."""
    report = _junction_sim(open_annulus=False).fidelity_report(print_report=False)
    (ptfe,) = _rows(report, "ptfe")
    hits = _findings(ptfe, RULE_I_KIND)
    assert len(hits) == 1, [f["kind"] for f in ptfe["findings"]]
    f = hits[0]
    assert f["overlap_cells"] == 48
    assert ptfe["n_cells"] == 192
    assert f["conductor_entities"] == [0]
    assert "geometry[0]" in f["detail"] and "48" in f["detail"]
    assert "sheet footprint nodes" in f["detail"]
    assert "OR-only" in f["detail"] or "cannot carve" in f["detail"]
    assert f["remedy"]
    # the pre-existing order-blind finding is untouched (it fires here too)
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1


def test_rule_i_is_silent_when_the_hole_is_built_into_the_conductor():
    """Same five entities with the ground built around the hole: the PTFE
    Cylinder's node-25 disk (48 cells on exact node coordinates, see the
    test above) is a strict subset of the 49-node lattice hole, so no
    earlier conductor shares a node with it.

    Under #931 the ground is twenty sheets on plane 25 whose unioned
    footprint is the full plane minus that hole (4361 of 4410 nodes,
    measured), and the check reads that footprint -- reading VOLUME cells
    only would make this control pass for the wrong reason, by going
    silent on the shorted copy too."""
    report = _junction_sim(open_annulus=True).fidelity_report(print_report=False)
    (ptfe,) = _rows(report, "ptfe")
    assert _findings(ptfe, RULE_I_KIND) == []
    for it in report:
        assert _findings(it, RULE_I_KIND) == [], it["entity"]


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
        sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                         boundary="cpml", cpml_layers=4)
        sim.add_material("ptfe", eps_r=2.1)
        # a one-cell PEC VOLUME drawn on-lattice (cell 4 under centre
        # sampling, §1.1), not a sheet -- the accumulator must hold for
        # both kinds
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
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1

    rep = _sim(dielectric_first=True).fidelity_report(print_report=False)
    (ptfe,) = _rows(rep, "ptfe")
    assert _findings(ptfe, RULE_I_KIND) == []
    # order-blind finding still fires in the reversed order -- unchanged
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1


def test_rule_i_lists_every_earlier_conductor_and_sums_the_union():
    """Two earlier PEC volumes overlapping the same dielectric: the finding
    names both indices and counts the UNION of their cells (a cell claimed by
    both is one no-op cell, not two)."""
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
    # The oracle must use the samplers the code uses, or it agrees for the
    # wrong reason: a PEC VOLUME is realized from cell CENTRES and a
    # dielectric from NODES (#931 §1.1). On this geometry the two PEC
    # samplers select DIFFERENT z layers (node 4 vs cell 3) that happen to
    # give the same count, so a node-sampled oracle here would not
    # discriminate.
    from rfx.geometry.rasterize_grid import (
        centres_from_uniform_grid, pec_volume_cell_mask)
    grid = sim._build_grid()
    centres = centres_from_uniform_grid(grid)
    m0 = np.asarray(pec_volume_cell_mask(sim._geometry[0].shape, centres), bool)
    m1 = np.asarray(pec_volume_cell_mask(sim._geometry[1].shape, centres), bool)
    m2 = np.asarray(sim._geometry[2].shape.mask(grid), bool)
    assert f["overlap_cells"] == int(((m0 | m1) & m2).sum())
    # and the discrimination the comment claims, measured here
    n0 = np.asarray(sim._geometry[0].shape.mask(grid), bool)
    assert not bool((m0 == n0).all()), "centre and node samplers must differ here"


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


# ---------------------------------------------------------------------------
# (ii) coaxial_port_junction_short (preflight advisory)
# ---------------------------------------------------------------------------

def test_rule_ii_fires_on_the_shorted_junction_copy():
    """Every first-ring node at the junction plane carries a realized PEC
    wall on the attempt-2 copy. FAILS on 88c49bdc: preflight has no such
    code.

    24/24, was 16/16 (measured 2026-09-07). The ring itself changed, not
    the verdict: the check's inner bound moved from ``PIN_R + dx/2`` to
    ``PIN_R + dx/sqrt(2)``, the reach of a centre-sampled pin volume's own
    rim (§1.1), and the ring is one cell wide from there -- 24 lattice
    nodes at 283, 300, 316 and 361 um instead of 16 at 283, 300 and 316.
    The first registered conductor outside the pin is still the (2,2)
    node at 282.8 um."""
    rows = _preflight_rows(_junction_sim(open_annulus=False), RULE_II_CODE)
    assert len(rows) == 1, [str(r) for r in rows]
    row = rows[0]
    assert row.severity == "warning"
    msg = str(row)
    assert "24/24" in msg
    assert "short" in msg.lower()
    assert "registered" in msg.lower()
    # the realized picture, not an inference: the first registered PEC
    # beyond the lattice-safe pin bound (r > 250 um) is the (2,2) cell at
    # 282.8 um -- the ground sheet starts where the pin ends
    assert "282.8 um" in msg


def test_rule_ii_is_silent_on_a_correctly_built_hole_with_the_pin_present():
    """Design review blocker 2: the fixed geometry keeps the pin's own
    realized footprint, and a ring defined as ``r > PIN_R`` counts it. The
    lattice ring excludes it, so the correctly built hole draws no row.
    The 0.4-0.5 mm ground lip (32/32 PEC, part of the predeclared 0.4 mm
    clearance) is outside the ring and must not trip it either.

    The blocker is sharper under #931, not softer (measured 2026-09-07).
    The pin's realized footprint went from 11 asymmetric nodes to 13
    symmetric ones (centre sampling, §1.1: no lattice point lands on the
    r == PIN_R knife edge), and its rim now reaches 224 um, so a naive
    ``PIN_R < r <= PIN_R + 1.5 dx`` band counts 8 of the pin's own nodes,
    not 2. The check's bound is ``PIN_R + dx/sqrt(2)`` = 270.7 um, the
    reach of a centre-sampled volume, and past it this geometry has 0."""
    sim = _junction_sim(open_annulus=True)
    metal, r = _junction_plane_metal(sim)
    # preconditions the test's claim depends on
    assert int((metal & (r <= PIN_R + 1e-9)).sum()) == 13
    naive = metal & (r > PIN_R + 1e-9) & (r <= PIN_R + 1.5 * DX)
    assert int(naive.sum()) == 8              # all of it the pin's own rim
    assert int((naive & (r > PIN_REACH)).sum()) == 0
    lip = (r > CLEAR_R + 1e-9) & (r <= 0.5e-3 + 1e-9)
    assert int((metal & lip).sum()) == int(lip.sum()) == 32
    assert _preflight_rows(sim, RULE_II_CODE) == []


def test_rule_ii_ring_is_lattice_based_not_r_gt_pin_radius():
    """Pin the ring's definition against the fixture's own lattice: the
    first-ring node set is exactly the 24 lattice offsets with
    hypot(di, dj) in (2.707, 3.707] cells (r = 283, 300, 316, 361 um),
    none of which the pin realizes in either copy.

    Was 16 offsets at 283/300/316 um under the ``PIN_R + dx/2`` inner
    bound. That bound is what this test's name warns against, one cell
    out: a centre-sampled pin volume realizes nodes to PIN_REACH =
    270.7 um, and the 250-270.7 um band the old bound admitted is the
    pin's own metal. Measured here, on the open copy, in the band the old
    bound would have used."""
    metal, r = _junction_plane_metal(_junction_sim(open_annulus=False))
    ring = (r > PIN_REACH) & (r <= PIN_REACH + DX)
    assert int(ring.sum()) == 24
    radii = sorted(set(np.round(r[ring] * 1e6).astype(int).tolist()))
    assert radii == [283, 300, 316, 361]
    assert int((metal & ring).sum()) == 24   # shorted copy: all PEC
    metal_open, _ = _junction_plane_metal(_junction_sim(open_annulus=True))
    assert int((metal_open & ring).sum()) == 0
    # the band the rejected ``r > PIN_R`` bound would have added is not
    # empty on this geometry -- it is the pin
    pin_band = (r > PIN_R + 1e-9) & (r <= PIN_REACH)
    assert int((metal_open & pin_band).sum()) == 8


def test_rule_ii_is_silent_without_geometry_and_without_coax_ports():
    """No registered geometry -> no registered PEC -> nothing to say; and a
    plain MSL board without a coaxial port never reaches the check."""
    sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_coaxial_port(position=(5e-3, 5e-3, 0.0), face="bottom",
                         pin_radius=0.635e-3, outer_radius=2.055e-3)
    assert _preflight_rows(sim, RULE_II_CODE) == []
    sim2 = _junction_sim(open_annulus=False)
    sim2._coaxial_ports.clear()
    assert _preflight_rows(sim2, RULE_II_CODE) == []


def test_rule_ii_message_names_the_first_registered_pec_radius_on_a_partial_hole():
    """A hole one cell too small (lattice disk r=3 -> ring nodes at 283 and
    300 um are inside the hole, 316 and 361 um nodes are not) is still a
    short by registered geometry, and the message states where the PEC
    starts.

    16/24, was 8/16 (measured 2026-09-07): the same 16 metal ring nodes,
    counted against the 24-node ring the widened inner bound defines. The
    radius the message names is unchanged at 316.2 um."""
    sim = Simulation(
        freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    jx, jy = int(round(JUNCTION_X / DX)), int(round(Y_C / DX))
    for b in _ground_plane_boxes_with_clearance(LX_2, LY, jx, jy, N_GND, 3):
        sim.add(b, material="pec")
    sim.add_coaxial_port(position=(JUNCTION_X, Y_C, N_GND * DX), face="bottom",
                         pin_radius=PIN_R, outer_radius=OUTER_R, impedance=50.0)
    rows = _preflight_rows(sim, RULE_II_CODE)
    assert len(rows) == 1
    msg = str(rows[0])
    assert "16/24" in msg, msg
    assert "316.2 um" in msg, msg


def test_rule_ii_does_not_change_the_emission_classification():
    """compute_coax_msl_transition stays DIAGNOSTIC_ONLY: the advisory
    surfaces through sim.preflight() (the driver's --preflight) and through
    run()/forward() elsewhere, not by wiring preflight into the method."""
    import ast
    import inspect
    import textwrap
    src = inspect.getsource(Simulation.compute_coax_msl_transition)
    tree = ast.parse(textwrap.dedent(src))
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"preflight", "_auto_preflight", "preflight_sparameters"}


# ---------------------------------------------------------------------------
# (i) never drops a conductor silently: a rasterization failure is a
# finding, and the audit says where it is incomplete. Rasterize-once.
# ---------------------------------------------------------------------------

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
        # ``pec_volume=`` (20a8ccfc, §1.1) is forwarded untouched: the
        # double decides WHETHER to raise, never how a row is sampled.
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
    # the no-op finding CANNOT fire (the conductor's cells are unknown) ...
    assert _findings(d, RULE_I_KIND) == []
    # ... and that gap is stated, not silent.
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


# ---------------------------------------------------------------------------
# The green half of the #931 migration: the SAME ground declared as a sheet,
# read through the one realization function. No solve, no report — just the
# geometry the two rules above will have to see once their owners land.
# ---------------------------------------------------------------------------

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
    sim.add(Box((0.0, 0.0, z), (LX_2, LY, z)), material="pec")   # sheet (§1.5)

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

    # and the realization: in-plane E zeroed on plane k, normal E live.
    edges = realized_pec_edge_masks(None, sheets=sheets)
    assert bool(np.asarray(edges[0])[:, :, k].any())
    assert bool(np.asarray(edges[1])[:, :, k].any())
    assert not bool(np.asarray(edges[2]).any()), "normal Ez must stay live"
