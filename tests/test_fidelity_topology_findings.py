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
(``tests/test_coax_msl_transition.py::_build_coax_msl_transition_sim_
attempt2``, constants reproduced inline so this file does not import a
file another change owns), plus the same geometry with the ground plane
built as 20 half-cell PEC Boxes generated from the integer-lattice disk
``di^2 + dj^2 <= 16`` (the attempt-3 recipe). Fail-before-fix: on the
source as of 88c49bdc, ``test_rule_i_fires_on_the_shorted_junction_copy``
and ``test_rule_ii_fires_on_the_shorted_junction_copy`` FAIL (no such
kind / no such code); the negative tests pass trivially there, so their
value is only in combination with the positive ones.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.csg import Box, Cylinder

# --- attempt-2 fixture constants, copied verbatim (tests/test_coax_msl_
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
    """Box bounds that rasterize to EXACTLY nodes [n_lo, n_hi] on an axis
    with spacing DX (rfx/geometry/csg.py Box docstring: corners on cell
    midpoints, never on node planes -- #802 knife edge)."""
    return (n_lo - 0.5) * DX, (n_hi + 0.5) * DX


def _margin_cylinder_z(n_lo, n_hi):
    z_lo, z_hi = n_lo * DX, n_hi * DX
    return 0.5 * (z_lo + z_hi), (z_hi - z_lo) + 2 * DX


def _ground_plane_boxes_with_clearance(lx, ly, jx, jy, k, r_cells):
    """The z=node-k ground sheet as half-cell PEC Boxes with a hole equal to
    the integer-lattice disk ``di^2 + dj^2 <= r_cells^2`` around (jx, jy).

    Two x-slabs (outer faces on the domain's own 0.0 / lx), two y-strips in
    the hole's x-band, then per row |dj| two boxes filling x beyond the
    row's half-width isqrt(r^2 - dj^2). 20 boxes for r_cells = 4.
    """
    rows = {dj: math.isqrt(r_cells * r_cells - dj * dj)
            for dj in range(-r_cells, r_cells + 1)}
    z_lo, z_hi = _half_cell(k, k)
    boxes = [
        Box((0.0, 0.0, z_lo), (_half_cell(jx - r_cells - 1, jx - r_cells - 1)[1], ly, z_hi)),
        Box((_half_cell(jx + r_cells + 1, jx + r_cells + 1)[0], 0.0, z_lo), (lx, ly, z_hi)),
    ]
    xl, xh = _half_cell(jx - r_cells, jx + r_cells)
    boxes.append(Box((xl, 0.0, z_lo),
                     (xh, _half_cell(jy - r_cells - 1, jy - r_cells - 1)[1], z_hi)))
    boxes.append(Box((xl, _half_cell(jy + r_cells + 1, jy + r_cells + 1)[0], z_lo),
                     (xh, ly, z_hi)))
    for dj, w in rows.items():
        if w >= r_cells:
            continue
        yl, yh = _half_cell(jy + dj, jy + dj)
        boxes.append(Box((xl, yl, z_lo), (_half_cell(jx - w - 1, jx - w - 1)[1], yh, z_hi)))
        boxes.append(Box((_half_cell(jx + w + 1, jx + w + 1)[0], yl, z_lo), (xh, yh, z_hi)))
    return boxes


def _junction_sim(*, open_annulus: bool) -> Simulation:
    """attempt-2 geometry (open_annulus=False) or the same with the ground
    sheet built around a lattice-disk clearance hole (open_annulus=True).

    Entity order, the thing rule (i) is about: ground PEC, PTFE clearance
    Cylinder, substrate Box, trace PEC Box, pin PEC Cylinder.
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
        gnd_lo, gnd_hi = _half_cell(N_GND, N_GND)
        sim.add(Box((0.0, 0.0, gnd_lo), (LX_2, LY, gnd_hi)), material="pec")
    clr_c, clr_h = _margin_cylinder_z(N_GND, N_SUB_LO)
    sim.add(Cylinder(center=(JUNCTION_X, Y_C, clr_c), radius=CLEAR_R,
                     height=clr_h, axis="z"), material="ptfe")
    sub_lo, sub_hi = _half_cell(N_SUB_LO, N_SUB_HI)
    sim.add(Box((0.0, 0.0, sub_lo), (LX_2, LY, sub_hi)), material="sub")
    trc_lo, trc_hi = _half_cell(N_TRACE, N_TRACE)
    sim.add(Box((JUNCTION_X, Y_C - W_TRACE / 2, trc_lo),
                (LX_2, Y_C + W_TRACE / 2, trc_hi)), material="pec")
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


def _junction_plane_pec(sim):
    """(pec_mask[:, :, k_junction], r_from_axis[um]) on the padded grid."""
    grid = sim._build_grid()
    _, _, _, pec, _, _, _ = sim._assemble_materials(grid)
    pec = np.asarray(pec, bool)
    k = int(grid.position_to_index(sim._coaxial_ports[0].position)[2])
    x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * DX - JUNCTION_X
    y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * DX - Y_C
    return pec[:, :, k], np.hypot(x[:, None], y[None, :])


# ---------------------------------------------------------------------------
# Fixture witnesses: the two copies differ ONLY by the hole, and the hole is
# what the findings must see. Measured on 88c49bdc (design review).
# ---------------------------------------------------------------------------

def test_fixture_copies_differ_only_by_the_junction_hole():
    pec_short, r = _junction_plane_pec(_junction_sim(open_annulus=False))
    pec_open, _ = _junction_plane_pec(_junction_sim(open_annulus=True))
    annulus = (r > PIN_R + 1e-9) & (r <= CLEAR_R + 1e-9)
    assert int(annulus.sum()) == 36
    # the shorted copy: every clearance-annulus cell at the junction plane is
    # registered PEC (#589 root cause, measured 36/36)
    assert int((pec_short & annulus).sum()) == 36
    # the open copy: none is (measured 0/36); the pin footprint itself is
    # still registered PEC in both (11 cells incl. the two r == PIN_R
    # knife-edge cells the review found)
    assert int((pec_open & annulus).sum()) == 0
    # the open copy's r <= PIN_R PEC is the registered pin's own asymmetric
    # 11-cell footprint (float rounding at the r == PIN_R knife edge: 13
    # lattice points lie within 200 um, 2 of them fall out); on the shorted
    # copy the solid ground makes all 13 PEC, which is exactly why the pin
    # is indistinguishable from ground there
    pin_fp = pec_open & (r <= PIN_R + 1e-9)
    assert int(pin_fp.sum()) == 11
    assert int((pec_short & (r <= PIN_R + 1e-9)).sum()) == 13


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
    knife-edge cell that fell out)."""
    report = _junction_sim(open_annulus=False).fidelity_report(print_report=False)
    (ptfe,) = _rows(report, "ptfe")
    hits = _findings(ptfe, RULE_I_KIND)
    assert len(hits) == 1, [f["kind"] for f in ptfe["findings"]]
    f = hits[0]
    assert f["overlap_cells"] == 48
    assert ptfe["n_cells"] == 192
    assert f["conductor_entities"] == [0]
    assert "geometry[0]" in f["detail"] and "48" in f["detail"]
    assert "OR-only" in f["detail"] or "cannot carve" in f["detail"]
    assert f["remedy"]
    # the pre-existing order-blind finding is untouched (it fires here too)
    assert len(_findings(ptfe, "claimed-by-conductor")) == 1


def test_rule_i_is_silent_when_the_hole_is_built_into_the_conductor():
    """Same five entities with the ground built around the hole: the PTFE
    Cylinder's node-25 disk (48 cells on exact node coordinates, see the
    test above) is a strict subset of the 49-cell lattice hole, so no
    earlier conductor shares a cell with it."""
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


def test_rule_i_minimal_sheet_then_cylinder_fires_and_reverse_is_silent():
    def _sim(dielectric_first: bool):
        sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
                         boundary="cpml", cpml_layers=4)
        sim.add_material("ptfe", eps_r=2.1)
        sheet = Box((0.0, 0.0, 4.5e-3), (10e-3, 10e-3, 5.5e-3))
        hole = Cylinder(center=(5e-3, 5e-3, 5e-3), radius=1.5e-3, height=3e-3,
                        axis="z")
        if dielectric_first:
            sim.add(hole, material="ptfe")
            sim.add(sheet, material="pec")
        else:
            sim.add(sheet, material="pec")
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
    """Two earlier PEC sheets overlapping the same dielectric: the finding
    names both indices and counts the UNION of their cells (a cell claimed by
    both sheets is one no-op cell, not two)."""
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
    grid = sim._build_grid()
    m0 = np.asarray(sim._geometry[0].shape.mask(grid), bool)
    m1 = np.asarray(sim._geometry[1].shape.mask(grid), bool)
    m2 = np.asarray(sim._geometry[2].shape.mask(grid), bool)
    assert f["overlap_cells"] == int(((m0 | m1) & m2).sum())


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
    """16 of 16 first-ring cells at the junction plane are registered PEC on
    the attempt-2 copy. FAILS on 88c49bdc: preflight has no such code."""
    rows = _preflight_rows(_junction_sim(open_annulus=False), RULE_II_CODE)
    assert len(rows) == 1, [str(r) for r in rows]
    row = rows[0]
    assert row.severity == "warning"
    msg = str(row)
    assert "16/16" in msg
    assert "short" in msg.lower()
    assert "registered" in msg.lower()
    # the realized picture, not an inference: the first registered PEC
    # beyond the lattice-safe pin bound (r > 250 um) is the (2,2) cell at
    # 282.8 um -- the ground sheet starts where the pin ends
    assert "282.8 um" in msg


def test_rule_ii_is_silent_on_a_correctly_built_hole_with_the_pin_present():
    """Design review blocker 2: the fixed geometry keeps the pin's own
    asymmetric 11-cell footprint (two cells at r == PIN_R exactly), and a
    ring defined as ``r > PIN_R`` counted them (2/16). The lattice ring
    ``PIN_R + dz/2 < r <= PIN_R + 3 dz/2`` excludes them: 0/16 -> no row.
    The 0.4-0.5 mm ground lip (32/32 PEC, part of the predeclared 0.4 mm
    clearance) is outside the ring and must not trip it either."""
    sim = _junction_sim(open_annulus=True)
    pec, r = _junction_plane_pec(sim)
    # preconditions the test's claim depends on
    assert int((pec & (r <= PIN_R + 1e-9)).sum()) == 11
    assert int((pec & (r > PIN_R + 1e-9) & (r <= PIN_R + 1.5 * DX)).sum()) == 0
    lip = (r > CLEAR_R + 1e-9) & (r <= 0.5e-3 + 1e-9)
    assert int((pec & lip).sum()) == int(lip.sum()) == 32
    assert _preflight_rows(sim, RULE_II_CODE) == []


def test_rule_ii_ring_is_lattice_based_not_r_gt_pin_radius():
    """Pin the ring's definition against the fixture's own lattice: the
    first-ring cell set is exactly the 16 lattice offsets with
    2.5 < hypot(di, dj) <= 3.5 (r = 283, 300, 316 um), none of which is a
    pin-footprint cell in either copy."""
    pec, r = _junction_plane_pec(_junction_sim(open_annulus=False))
    ring = (r > PIN_R + 0.5 * DX) & (r <= PIN_R + 1.5 * DX)
    assert int(ring.sum()) == 16
    radii = sorted(set(np.round(r[ring] * 1e6).astype(int).tolist()))
    assert radii == [283, 300, 316]
    assert int((pec & ring).sum()) == 16   # shorted copy: all PEC
    pec_open, _ = _junction_plane_pec(_junction_sim(open_annulus=True))
    assert int((pec_open & ring).sum()) == 0


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
    """A hole one cell too small (lattice disk r=3 -> ring cells at 300 um
    are inside the hole, 316 um cells are not) is still a short by
    registered geometry, and the message states where the PEC starts."""
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
    assert "8/16" in msg, msg
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


def _sheet_then_cylinder_sim(sheet_cls):
    """Minimal ordered pair: a one-node PEC sheet at node 4, then a
    dielectric Cylinder through it (the shape of the #589 no-op)."""
    sim = Simulation(freq_max=20e9, domain=(1.0e-3, 1.0e-3, 1.0e-3), dx=DX,
                     cpml_layers=4,
                     boundary=BoundarySpec(x="pec", y="pec", z="pec"))
    sim.add_material("d", eps_r=2.0)
    z_lo, z_hi = _half_cell(4, 4)
    sim.add(sheet_cls((0.0, 0.0, z_lo), (1.0e-3, 1.0e-3, z_hi)), material="pec")
    sim.add(Cylinder(center=(0.5e-3, 0.5e-3, 0.5e-3), radius=0.25e-3,
                     height=0.6e-3, axis="z"), material="d")
    return sim


def _failing_entity_mask(monkeypatch, shape_cls, exc):
    """Make fidelity_report's OWN rasterization raise for entities whose
    shape is ``shape_cls``; everything else rasterizes as before."""
    import rfx.fidelity as fid
    real = fid._entity_mask

    def patched(entry, sim, grid, nonuniform):
        if isinstance(entry.shape, shape_cls):
            raise exc
        return real(entry, sim, grid, nonuniform)

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
    sim = _sheet_then_cylinder_sim(_NoBoundsBox)
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
    """Control for the test above: identical geometry, the sheet as a plain
    Box, no injected failure -> the ordered no-op finding fires on the
    Cylinder row and no rasterization/unaudited finding exists anywhere."""
    report = _sheet_then_cylinder_sim(Box).fidelity_report(print_report=False)
    (d,) = _rows(report, "d")
    (f,) = _findings(d, RULE_I_KIND)
    assert f["conductor_entities"] == [0] and f["overlap_cells"] > 0
    for it in report:
        assert _findings(it, "rasterization-failed") == [], it["entity"]
        assert _findings(it, "dielectric-after-conductor-unaudited") == [], it["entity"]


def test_rule_i_rasterizes_each_conductor_once(monkeypatch):
    """Twelve PEC strips at one node, then a dielectric slab over all of
    them: every conductor is named as a contributor, and the audit
    rasterized each entity exactly once (the earlier implementation
    re-rasterized every earlier conductor per overlapping dielectric)."""
    import rfx.fidelity as fid
    real = fid._entity_mask
    calls = []

    def counting(entry, sim, grid, nonuniform):
        calls.append(id(entry))
        return real(entry, sim, grid, nonuniform)

    monkeypatch.setattr(fid, "_entity_mask", counting)

    n_strips = 12
    sim = Simulation(freq_max=20e9, domain=(1.2e-3, 1.0e-3, 1.0e-3), dx=DX,
                     cpml_layers=4,
                     boundary=BoundarySpec(x="pec", y="pec", z="pec"))
    sim.add_material("d", eps_r=2.0)
    z_lo, z_hi = _half_cell(4, 4)
    for s in range(n_strips):
        x_lo, x_hi = _half_cell(s, s)
        sim.add(Box((x_lo, 0.0, z_lo), (x_hi, 1.0e-3, z_hi)), material="pec")
    sim.add(Box((0.0, 0.0, _half_cell(3, 5)[0]), (1.2e-3, 1.0e-3, _half_cell(3, 5)[1])),
            material="d")
    report = sim.fidelity_report(print_report=False)

    (d,) = _rows(report, "d")
    (f,) = _findings(d, RULE_I_KIND)
    assert f["conductor_entities"] == list(range(n_strips))
    assert len(calls) == n_strips + 1
    assert len(set(calls)) == n_strips + 1
