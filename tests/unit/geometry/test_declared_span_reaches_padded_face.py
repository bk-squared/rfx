"""A structure declared out to a padded face must be rasterized to it (#1070).

``extend_cpml_pad_materials`` fills a hi pad by replicating the interior-edge
column outward, and its #627a fallback looks exactly one column further in when
that edge is vacuum. The docstring states the bound -- the half-open
``[lo, hi)`` Box rule costs the hi face "deterministically one node ... never
more". Grid sizing could invent a SECOND empty node: ``ceil`` of a ratio one
ULP above an integer allocated a cell no declared Box reaches. The fallback
then found vacuum in both columns it inspects, gave up, and filled the whole
absorber with vacuum.

The assertion here pins the bound the fallback documents, not the arithmetic
that broke it, so any other route to an unfilled interior node at a pad seam is
caught too. Build-only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import rfx.grid as rfx_grid
from rfx import Box, Simulation
from rfx.geometry.rasterize_grid import PadFillShortfall

#: The #1070 rig: RO4003C at h = 0.787 mm, dx = h/4, cpml_layers 8. +10h is
#: the one pad value whose x ratio is BOTH one ULP above an integer AND has
#: ``232*dx`` reproducing the declared length exactly, which is what costs the
#: second node. +8h and +12h round up too but lose only one node, so #627a
#: repairs them and they are not red rigs.
H = 0.787e-3
DX = H / 4.0
EPS_R = 3.38
CPML = 8


def _rig(pad_h: int = 10, *, hi_shrink_cells: float = 0.0):
    dom_x = 38 * H + 2 * (pad_h * H)
    dom_y = 23 * H + 2 * (pad_h * H)
    dom_z = 16 * H
    sim = Simulation(freq_max=10e9, domain=(dom_x, dom_y, dom_z), dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    box_hi_x = dom_x - hi_shrink_cells * DX
    sim.add(Box((0.0, 0.0, 6 * H), (box_hi_x, dom_y, 8 * H)),
            material="ro4003c")
    return sim


def _size_by_plain_ceil(monkeypatch) -> None:
    """Put the pre-#1070 sizing back, keeping everything else."""
    import rfx.grid as grid_mod

    monkeypatch.setattr(grid_mod, "cells_spanning",
                        lambda length, dx, **kw: int(math.ceil(length / dx)))


def test_the_rig_that_found_this_is_silent_now() -> None:
    """Green-after, and the measurement behind it: the +10h x pad carries the
    substrate rather than vacuum."""
    sim = _rig(10)
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    assert list(grid.shape) == [249, 189, 81], list(grid.shape)
    k = eps.shape[2] // 2
    zs = [z for z in range(eps.shape[2])
          if abs(eps[eps.shape[0] // 2, eps.shape[1] // 2, z] - EPS_R) < 1e-12]
    k = zs[len(zs) // 2]
    assert float(eps[-1, eps.shape[1] // 2, k]) == pytest.approx(EPS_R)


def test_it_fires_when_the_grid_is_sized_the_old_way(monkeypatch) -> None:
    """Red-before, reached by restoring only the sizing.

    This is the two-node shortfall: ``ceil`` invents node 233 and node 232
    sits exactly on the box's hi face, which the half-open rule excludes.
    """
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(10)
    grid = sim._build_grid()
    assert list(grid.shape) == [250, 189, 81], (
        "the old sizing is not in place; this fixture proves nothing")
    with pytest.raises(PadFillShortfall) as excinfo:
        sim._assemble_materials(grid)
    message = str(excinfo.value)
    assert "ro4003c" in message
    assert "x-hi" in message
    assert "2 interior nodes short" in message, message


@pytest.mark.parametrize("pad_h", [6, 8, 12])
def test_the_pad_values_that_lose_only_one_node_stay_silent(
        monkeypatch, pad_h: int) -> None:
    """+8h and +12h round up as well, and #627a repairs them.

    Silent even with the old sizing, which is the measurement that says the
    extra cell alone is not the defect.
    """
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(pad_h)
    sim._assemble_materials(sim._build_grid())


@pytest.mark.parametrize("shrink", [0.3, 1.0, 3.0])
def test_a_declared_air_gap_before_the_absorber_is_left_alone(
        shrink: float) -> None:
    """The common case, and the one an unbounded backward scan would break: a
    box that does not reach the face was never declared to, so its vacuum is
    the declared structure and not a shortfall."""
    sim = _rig(10, hi_shrink_cells=shrink)
    sim._assemble_materials(sim._build_grid())


def test_an_air_gap_is_left_alone_under_the_old_sizing_too(monkeypatch) -> None:
    """Same, with the defect's arithmetic in place: the check keys on what was
    DECLARED, so it cannot be tripped by the grid being one cell too long."""
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(10, hi_shrink_cells=3.0)
    sim._assemble_materials(sim._build_grid())


def test_a_box_landing_exactly_on_the_face_is_not_a_shortfall() -> None:
    """The documented one-node case, stated as its own test.

    ``domain = n*dx`` exactly and the box reaches the face: the half-open rule
    drops that last node, #627a repairs it, and the assertion must not call
    it a defect.
    """
    dom = 64 * DX
    sim = Simulation(freq_max=10e9, domain=(dom, dom, 16 * DX), dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_material("slab", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0.0, 0.0, 4 * DX), (dom, dom, 8 * DX)), material="slab")
    grid = sim._build_grid()
    sim._assemble_materials(grid)
    eps = np.asarray(sim._assemble_materials(grid)[0].eps_r)
    k = eps.shape[2] // 2
    zs = [z for z in range(eps.shape[2])
          if abs(eps[eps.shape[0] // 2, eps.shape[1] // 2, z] - EPS_R) < 1e-12]
    k = zs[len(zs) // 2]
    assert float(eps[-1, eps.shape[1] // 2, k]) == pytest.approx(EPS_R)


def test_the_check_is_skipped_where_no_pad_will_be_filled(monkeypatch) -> None:
    """A PEC-walled domain has no pad to replicate into, so the question does
    not arise and the check must not invent it.

    Built as a PEC domain rather than by setting ``_boundary`` on a
    CPML-padded one (review of PR #1136, C). The old version mutated the flag
    after the grid was made, so the grid still carried 8-cell pads and the
    test passed for the wrong reason -- it proved the flag is read, not that a
    wall-bounded model is exempt. The old sizing is in place so the shortfall
    would exist if anything looked for it.
    """
    _size_by_plain_ceil(monkeypatch)
    dom_x = 38 * H + 2 * (10 * H)
    dom_y = 23 * H + 2 * (10 * H)
    sim = Simulation(freq_max=10e9, domain=(dom_x, dom_y, 16 * H), dx=DX,
                     boundary="pec")
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0.0, 0.0, 6 * H), (dom_x, dom_y, 8 * H)), material="ro4003c")
    grid = sim._build_grid()
    assert tuple(grid.face_pads) == (0, 0, 0, 0, 0, 0), grid.face_pads
    sim._assemble_materials(grid)


# --- PEC entries are not asked the question (review of PR #1136, C) --------

def _pec_rig(monkeypatch, *, sheet: bool):
    """The defective sizing, with the domain-spanning body declared PEC.

    The pad extension replicates eps/sigma/mu and never ``pec_mask``, so a PEC
    body cannot leave a vacuum pad behind and the check must not report one.
    """
    _size_by_plain_ceil(monkeypatch)
    dom_x = 38 * H + 2 * (10 * H)
    dom_y = 23 * H + 2 * (10 * H)
    sim = Simulation(freq_max=10e9, domain=(dom_x, dom_y, 16 * H), dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_material("copper", eps_r=1.0, sigma=5.8e7)
    z_lo = 6 * H
    z_hi = z_lo if sheet else 8 * H
    sim.add(Box((0.0, 0.0, z_lo), (dom_x, dom_y, z_hi)), material="copper")
    return sim


def test_a_pec_volume_spanning_the_domain_is_not_reported(monkeypatch) -> None:
    sim = _pec_rig(monkeypatch, sheet=False)
    grid = sim._build_grid()
    assert list(grid.shape) == [250, 189, 81], (
        "the old sizing is not in place; this fixture proves nothing")
    sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])


def test_a_pec_sheet_spanning_the_domain_is_not_reported(monkeypatch) -> None:
    """A zero-thickness Box is a SHEET under the #931 contract -- it owns no
    cell at all, so asking it about a filled pad is doubly meaningless."""
    sim = _pec_rig(monkeypatch, sheet=True)
    sim._assemble_materials(sim._build_grid(), pec_sheets=[], pec_wires=[])


def test_the_same_body_as_a_dielectric_is_still_reported(monkeypatch) -> None:
    """The control for the two above: identical geometry and sizing, declared
    as a dielectric, still raises. Without this, a check that had been
    disabled outright would pass them both."""
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(10)
    with pytest.raises(PadFillShortfall):
        sim._assemble_materials(sim._build_grid())


# --- the audit path reports instead of raising (review of PR #1136, A) -----

def test_fidelity_report_names_the_shortfall_instead_of_raising(
        monkeypatch) -> None:
    """``fidelity_report`` exists to show where the realized model differs
    from the declared one. A pad seam the structure does not reach is exactly
    that, so it must appear as a row rather than take the report down."""
    _size_by_plain_ceil(monkeypatch)
    sim = _rig(10)
    report = sim.fidelity_report(print_report=False)

    hits = [(item, finding)
            for item in report
            for finding in item.get("findings", [])
            if finding.get("kind") == "declared-span-short-of-padded-face"]
    assert len(hits) == 1, [f.get("kind") for it in report
                            for f in it.get("findings", [])]
    item, finding = hits[0]
    assert "ro4003c" in item["entity"]
    assert finding["axis"] == "x"
    assert "2 interior nodes short" in finding["detail"]
    assert "cells_spanning" in finding["remedy"]


def test_pad_fill_shortfall_is_a_value_error() -> None:
    """The degrade guards that keep an advisory path alive already catch
    ``ValueError``; a new type outside that set turned a reportable finding
    into a hard failure of the report (review of PR #1136, A)."""
    assert issubclass(PadFillShortfall, ValueError)


# --- the two pad predicates differ on purpose (review of PR #1136, B) ------

def test_the_smoothing_predicate_and_this_one_disagree_by_design() -> None:
    """Same rig, two questions, two answers -- and which fires is the point.

    ``smoothed_shape_pairs`` asks whether to CONTINUE a shape into a pad and
    scores the DECLARED corner against the REALIZED interior-edge node. On the
    defect rig that node sits beyond the declared face, so it says "does not
    reach" and continues nothing: correct for its own lane, and silent about
    the defect.

    This check asks whether a shortfall is a DEFECT and scores against the
    DECLARED face, which the box does reach by construction. So it fires where
    the other is silent. Scoring this one against the realized edge would
    compare the realized model with itself and could never fail.
    """
    import math as _math

    from rfx.geometry.smoothing import smoothed_shape_pairs

    _orig = rfx_grid.cells_spanning
    rfx_grid.cells_spanning = lambda length, dx, **kw: int(
        _math.ceil(length / dx))
    try:
        sim = _rig(10)
        grid = sim._build_grid()
        pairs, unextendable = smoothed_shape_pairs(sim, grid)
        with pytest.raises(PadFillShortfall):
            sim._assemble_materials(grid)
    finally:
        rfx_grid.cells_spanning = _orig

    assert unextendable == [], unextendable
    box = sim._geometry[0].shape
    continued = pairs[0][0]
    assert continued.corner_hi[0] == box.corner_hi[0], (
        "the smoothing lane continued the x-hi face; on this rig it is "
        "supposed to decline, because the realized interior edge lies beyond "
        "the declared corner")


def test_a_lo_face_shortfall_is_silent_and_that_is_the_documented_scope(
        monkeypatch) -> None:
    """Hi faces only (review of PR #1136, D), pinned rather than just stated.

    The rig is +6h, whose x ratio is an exact 200.0, so its hi face is clean
    under either sizing and cannot contribute a raise. The box is then drawn
    five cells inside the x-lo face and out to the x-hi face: five unfilled
    interior nodes sit in front of a padded lo face, and this check says
    nothing about them.

    The asymmetry is the rasterizer's. The half-open ``[lo, hi)`` rule costs
    the hi face a node and the lo face nothing, so a lo-face gap is declared
    rather than an artifact, and ``extend_cpml_pad_materials`` fills a lo pad
    from the boundary node itself. Replicating vacuum in front of a declared
    gap is the right answer.

    Two nodes at a HI face raise on the +10h rig, which is what says this is
    about the face rather than about the number of nodes.
    """
    _size_by_plain_ceil(monkeypatch)
    dom_x = 38 * H + 2 * (6 * H)
    dom_y = 23 * H + 2 * (6 * H)
    assert dom_x / DX == 200.0, dom_x / DX

    sim = Simulation(freq_max=10e9, domain=(dom_x, dom_y, 16 * H), dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((5.0 * DX, 0.0, 6 * H), (dom_x, dom_y, 8 * H)),
            material="ro4003c")
    grid = sim._build_grid()

    # The gap is real and it is five nodes wide, measured rather than assumed.
    raw = np.asarray(sim._assemble_materials(
        grid, include_cpml_pad_extension=False)[0].eps_r)
    k = raw.shape[2] // 2
    zs = [z for z in range(raw.shape[2])
          if abs(raw[grid.face_pads[0] + 20, raw.shape[1] // 2, z]
                 - EPS_R) < 1e-12]
    k = zs[len(zs) // 2]
    row = raw[:, raw.shape[1] // 2, k]
    lo_pad = grid.face_pads[0]
    lead = 0
    for value in row[lo_pad:]:
        if abs(float(value) - 1.0) < 1e-12:
            lead += 1
        else:
            break
    assert lead == 5, lead

    # Silent, with the full assembly and the pad extension on.
    sim._assemble_materials(grid)

    # The contrast: two nodes at a hi face on the defective rig do raise.
    with pytest.raises(PadFillShortfall):
        _rig(10)._assemble_materials(_rig(10)._build_grid())
