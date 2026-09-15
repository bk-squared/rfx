"""What a SHEET owns, and the 2-D lane — lattice ownership contract §1.3 (#931).

``tests/contracts/test_lattice_ownership_contract.py`` pins the realized edge
sets: the slab battery, the closed footprint, mirror / axis-permutation
invariance, soft == hard, G4. Two clauses of §1.3 had no witness anywhere, and
both RETIRE a deleted fix, so without a test they regress the first time
somebody re-adds the convenience:

1. **A sheet owns no cell and writes no material.** #702
   (``resample_sheet_node_materials``) existed because a conductor thinner than
   a cell occupied one cell layer whose statics then read vacuum; the fix
   re-sampled ``eps_r`` / ``sigma`` for that cell at ``node + d/2``. Under the
   contract there is no "own cell", so nothing re-samples: whatever the
   dielectric boxes wrote at the sheet plane is what stands, on the uniform and
   the non-uniform lane alike. The measured #702 defect — two identical 17 µm
   buried levels reading ``eps_r`` 3.520 and 3.380 from the same geometry on
   the same mesh, because float32 rounding decided which half-cell-shifted node
   was nearest — cannot recur, because no material value depends on where the
   metal is any more.

   The fix is retired; the physics it served is not. A stack-up drawn with a
   SLOT for the foil really does leave the sheet plane in vacuum (the measured
   22.13 µm of spurious series thickness on a 31.43 µm mesh), and the contract
   makes that visible instead of absorbing it: preflight reports it
   (``sheet_slot_vacuum``) and the remedy is to extend the dielectric THROUGH
   the sheet plane. Both readings are pinned below, so "the resample is gone"
   and "the slot is still vacuum" are separate assertions rather than one
   claim.

2. **The 2-D lane** (§1.3, amended 2026-09-07). A sheet whose normal is a
   length-1 axis has no thickness direction, so its "normal" component is not a
   through-sheet edge: the closed 2-D region evaluated for that component's own
   location is the footprint itself. The design note says this is the same edge
   set the VOLUME rule gives the same drawn rectangle and calls it verified —
   that sentence had no test. It is verified here. Every ``mode="2d_tmz"``
   fixture in the suite rides the clause (``test_sheet_impedance.py``'s
   ``_tmz_sim``, ``tests/unit/misc/test_flux_monitor_finite_size.py``'s PEC
   short).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from rfx import Box, Simulation
from rfx.boundaries.pec import SheetSpec, realized_pec_edge_masks


# --------------------------------------------------------------------------
# 1. a sheet owns no cell and writes no material
# --------------------------------------------------------------------------

DX = 100e-6
T_CU = 17e-6            # a real foil: 0.17 of a cell, so it can only be a sheet
EPS_CORE = 3.38
LX = LY = 1.0e-3
LZ = 2.0e-3
Z_SHEET = 0.8e-3        # a node on BOTH lanes below

# 10 x 50 µm then 10 x 150 µm: a grading ratio unlike either transverse axis,
# so a uniform-valued profile cannot take the NU path without exercising it.
# Nodes: 0 .. 500 µm by 50, then 650, 800, ... — 800 µm is node 12.
DZ_GRADED = np.concatenate([np.full(10, 50e-6), np.full(10, 150e-6)])


def _stack(*, with_sheet, span_the_plane=True, lane="uniform"):
    """Substrate (+ optional PEC foil declared as a SHEET).

    ``span_the_plane`` picks the two stack-up conventions the contract
    distinguishes: the laminate drawn THROUGH the foil plane (the remedy), or
    drawn abutting the foil's two physical faces so the plane itself is a
    17 µm slot of vacuum (the #702 geometry).
    """
    if lane == "nu":
        sim = Simulation(freq_max=10e9, domain=(LX, LY, 0.0), dx=DX,
                         dz_profile=DZ_GRADED, boundary="pec")
    else:
        sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                         boundary="pec")
    sim.add_material("core", eps_r=EPS_CORE)
    if span_the_plane:
        sim.add(Box((0.0, 0.0, 0.4e-3), (LX, LY, 1.2e-3)), material="core")
    else:
        sim.add(Box((0.0, 0.0, 0.4e-3), (LX, LY, Z_SHEET - T_CU / 2)),
                material="core")
        sim.add(Box((0.0, 0.0, Z_SHEET + T_CU / 2), (LX, LY, 1.2e-3)),
                material="core")
    if with_sheet:
        sim.add(Box((0.2e-3, 0.2e-3, Z_SHEET), (0.8e-3, 0.8e-3, Z_SHEET)),
                material="pec")
    return sim


def _assemble(sim, lane="uniform"):
    """``(grid, materials, pec_mask, pec_sheets)`` on either lane."""
    sheets: list = []
    if lane == "nu":
        grid = sim._build_nonuniform_grid()
        mats, _, _, pec_mask = sim._assemble_materials_nu(
            grid, pec_sheets=sheets)
    else:
        grid = sim._build_grid()
        mats, _, _, pec_mask, _, _, _ = sim._assemble_materials(
            grid, pec_sheets=sheets)
    return grid, mats, pec_mask, sheets


@pytest.mark.parametrize("lane", ["uniform", "nu"])
def test_a_pec_sheet_writes_no_material_and_owns_no_cell(lane):
    """The assembled statics are BIT-IDENTICAL with and without the foil.

    That is the whole of "a sheet owns NO cell": adding a conductor to the
    model may not move one number in ``eps_r`` or ``sigma``. Re-introducing
    any convenience write at the sheet node (#702's resample, an eps stamp,
    a "mark the metal cell" helper) turns this red on both lanes at once —
    which is what the deleted two-lane parity test was for.
    """
    _g0, m0, pec0, s0 = _assemble(_stack(with_sheet=False, lane=lane), lane)
    _g1, m1, pec1, s1 = _assemble(_stack(with_sheet=True, lane=lane), lane)

    assert s0 == [] and len(s1) == 1
    assert pec0 is None or not bool(jnp.any(pec0))
    assert pec1 is None or not bool(jnp.any(pec1)), (
        "a sheet must not appear in the cell mask")
    np.testing.assert_array_equal(np.asarray(m1.eps_r), np.asarray(m0.eps_r))
    np.testing.assert_array_equal(np.asarray(m1.sigma), np.asarray(m0.sigma))


@pytest.mark.parametrize("lane", ["uniform", "nu"])
def test_the_sheet_plane_reads_the_laminate_the_boxes_drew_there(lane):
    """Stack-up drawn THROUGH the foil plane: the plane carries the laminate.

    This is the remedy the contract names for the #702 case, and the reason
    the resample is not needed once the model says what it means.
    """
    sim = _stack(with_sheet=True, span_the_plane=True, lane=lane)
    grid, mats, _pec, sheets = _assemble(sim, lane)
    k = sheets[0].plane
    eps = np.asarray(mats.eps_r)
    i, j = eps.shape[0] // 2, eps.shape[1] // 2
    assert float(eps[i, j, k]) == pytest.approx(EPS_CORE, abs=1e-5)
    assert 0 < k < grid.shape[2] - 1


def test_two_identical_slot_levels_stay_vacuum_and_agree():
    """The general form of the measured #702 defect, now impossible.

    Two foils at two node planes with the SAME local stack-up gave ``eps_r``
    3.520 at one buried level and 3.380 at the other on one board, because
    the argmin snap picked a different half-cell-shifted node at each. Under
    the contract the sheet reads nothing and writes nothing, so the two
    levels agree by construction — and the value they agree on is the vacuum
    the stack-up actually drew, not a resampled laminate. If either level
    reads ``EPS_CORE`` again, a resample has come back; if the two disagree,
    a material value is following the metal again.
    """
    z_a, z_b = 0.6e-3, 1.0e-3
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     boundary="pec")
    sim.add_material("core", eps_r=EPS_CORE)
    sim.add(Box((0.0, 0.0, 0.4e-3), (LX, LY, z_a - T_CU / 2)), material="core")
    sim.add(Box((0.0, 0.0, z_a + T_CU / 2), (LX, LY, z_b - T_CU / 2)),
            material="core")
    sim.add(Box((0.0, 0.0, z_b + T_CU / 2), (LX, LY, 1.4e-3)), material="core")
    for z in (z_a, z_b):
        sim.add(Box((0.2e-3, 0.2e-3, z), (0.8e-3, 0.8e-3, z)), material="pec")

    grid, mats, pec_mask, sheets = _assemble(sim)
    assert pec_mask is None or not bool(jnp.any(pec_mask))
    assert len(sheets) == 2
    eps = np.asarray(mats.eps_r)
    i, j = eps.shape[0] // 2, eps.shape[1] // 2
    got = [float(eps[i, j, sp.plane]) for sp in sheets]
    assert got[0] == got[1], (
        f"the two buried levels disagree: {got} — a material value is "
        "following the metal again")
    assert got[0] == pytest.approx(1.0, abs=1e-9), (
        "the slot the stack-up drew must stay vacuum; preflight's "
        "sheet_slot_vacuum is what surfaces it, not a silent resample")
    # the fixture really is the slot geometry: laminate on both sides
    for sp in sheets:
        assert float(eps[i, j, sp.plane - 1]) == pytest.approx(EPS_CORE, abs=1e-5)
        assert float(eps[i, j, sp.plane + 1]) == pytest.approx(EPS_CORE, abs=1e-5)


@pytest.mark.parametrize("lane", ["uniform", "nu"])
def test_the_realized_sheet_plane_equals_the_declared_one(lane):
    """Build-time check (no solve): realized == declared, through the shared
    helper, with the normal edge left live."""
    from tests._realized_geometry import (
        assert_sheet_planes, assert_wall_planes, node_index, realized)

    sim = _stack(with_sheet=True, lane=lane)
    rz = realized(sim)
    assert_sheet_planes(sim, 2, expected_m=(Z_SHEET,), what=f"{lane} lane")
    assert_wall_planes(sim, 2, expected_m=(Z_SHEET,), what=f"{lane} lane")
    k = node_index(rz.grid, 2, Z_SHEET)
    assert not bool(np.asarray(rz.edge_masks[2])[:, :, k].any()), (
        "the normal E through a sheet stays live (§1.3)")


# --------------------------------------------------------------------------
# 2. the 2-D lane: a sheet whose normal is a length-1 axis
# --------------------------------------------------------------------------

def _rect_footprint(shape, i0, i1, j0, j1):
    """Closed node rectangle ``i0..i1`` x ``j0..j1`` on plane k = 0."""
    fp = np.zeros(shape, dtype=bool)
    fp[i0:i1 + 1, j0:j1 + 1, 0] = True
    return fp


def test_two_d_sheet_edge_set_equals_the_same_rectangle_as_a_volume():
    """§1.3 amended: on a length-1 normal axis the sheet rule and the volume
    rule give the SAME edges for a rectangular footprint.

    The closed node footprint ``i0..i1`` is what the volume owning cells
    ``i0..i1-1`` realizes: both components tangential to a length-1 axis pick
    up the wrap neighbour, which on that axis is the cell itself. The design
    note asserts the two coincide and calls it verified; this is the
    verification.
    """
    shape = (12, 10, 1)
    i0, i1, j0, j1 = 3, 7, 2, 6
    fp = _rect_footprint(shape, i0, i1, j0, j1)
    sheet = realized_pec_edge_masks(
        None, sheets=[SheetSpec(normal_axis=2, plane=0, footprint=fp)])

    cells = np.zeros(shape, dtype=bool)
    cells[i0:i1, j0:j1, 0] = True
    volume = realized_pec_edge_masks(jnp.asarray(cells))

    for c, name in enumerate("xyz"):
        np.testing.assert_array_equal(
            np.asarray(sheet[c]), np.asarray(volume[c]),
            err_msg=f"E{name} differs between the 2-D sheet and volume rules")
    assert bool(np.asarray(sheet[2]).any()), "not the trivial agreement"


def test_two_d_sheet_keeps_its_out_of_plane_component():
    """With no thickness direction there is no through-sheet edge to keep
    live: Ez is PEC exactly on the footprint nodes, and the two in-plane
    components take the usual both-end-nodes rule (one index shorter)."""
    shape = (12, 10, 1)
    fp = _rect_footprint(shape, 3, 7, 2, 6)
    mx, my, mz = realized_pec_edge_masks(
        None, sheets=[SheetSpec(normal_axis=2, plane=0, footprint=fp)])
    np.testing.assert_array_equal(np.asarray(mz), fp)
    assert int(np.asarray(mx).sum()) == 4 * 5
    assert int(np.asarray(my).sum()) == 5 * 4
    assert int(np.asarray(mz).sum()) == 5 * 5


def test_two_d_sheet_through_the_api_is_one_plane_and_no_cell():
    """The same clause through ``sim.add`` on a ``2d_tmz`` run."""
    from tests._realized_geometry import realized

    dx = 1e-3
    sim = Simulation(freq_max=20e9, domain=(0.02, 0.02, dx), dx=dx,
                     boundary="pec", mode="2d_tmz")
    sim.add(Box((0.008, 0.008, 0.0), (0.014, 0.014, 0.0)), material="pec")
    rz = realized(sim)
    assert rz.grid.shape[2] == 1
    assert rz.pec_mask is None or not bool(jnp.any(rz.pec_mask))
    (spec,) = rz.sheets
    assert spec.normal_axis == 2 and spec.plane == 0
    np.testing.assert_array_equal(np.asarray(rz.edge_masks[2]),
                                  np.asarray(spec.footprint))


# --------------------------------------------------------------------------
# 3. the declaration survives serialization
# --------------------------------------------------------------------------

def test_a_sheet_declaration_round_trips_through_the_design_ir():
    """A serialized design must not silently reinterpret its conductors.

    Under the contract the (V)/(S) declaration IS the drawn geometry — a
    zero-extent axis on a PEC Box says "sheet" — so the IR needs no ownership
    field, but it does need the zero-extent corner to survive the round trip
    exactly. If a codec ever normalises ``corner_hi`` away from ``corner_lo``
    (a "degenerate box" repair, a tolerance snap), the rebuilt design becomes
    a one-cell VOLUME with a wall on each face and nothing says so. ``two_plane``
    is gone from the IR entirely (v2), so the declaration is the only thing
    left carrying the realization.
    """
    from rfx.interop import design_to_dict, simulation_from_design

    sim = _stack(with_sheet=True)
    doc = design_to_dict(sim)
    assert "two_plane" not in repr(doc), "two_plane must be gone from the IR"
    rebuilt = simulation_from_design(doc)

    got = []
    for s in (sim, rebuilt):
        grid = s._build_grid()
        sheets: list = []
        pec_mask = s._assemble_materials(grid, pec_sheets=sheets)[3]
        assert pec_mask is None or not bool(jnp.any(pec_mask))
        assert len(sheets) == 1
        got.append((sheets[0].normal_axis, sheets[0].plane,
                    np.asarray(sheets[0].footprint)))
    assert got[0][:2] == got[1][:2]
    np.testing.assert_array_equal(got[0][2], got[1][2])
