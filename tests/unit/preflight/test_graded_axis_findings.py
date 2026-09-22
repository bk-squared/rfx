"""Preflight checks that read a cell size now read the cell at their own site.

Each test here discriminates on the FINDING preflight emits, not on what a
helper returns. A helper-level assertion cannot see whether the check calls
it, which is the lesson #1200 left: its gate compared a helper against the
accessors the helper itself called, and reviving the defect at the call site
left every test green.

So each case below builds a mesh on which the old read and the new one
disagree, and asserts which advisory comes out.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.preflight._common import profile_boundary_cell

_DX = 1.0e-3


def _codes(report):
    return [getattr(i, "code", None) for i in report.issues]


# ---------------------------------------------------------------------------
# The gate: a dx-only graded mesh used to skip the whole family
# ---------------------------------------------------------------------------

def _dx_graded_tfsf_sim(angle_deg):
    """A mesh graded on x only, with TFSF. No z profile anywhere."""
    prof = np.concatenate([np.full(20, _DX), np.full(20, 0.5 * _DX),
                           np.full(20, _DX)])
    lx = float(np.sum(prof))
    sim = Simulation(freq_max=10e9, domain=(lx, 0.02, 0.02), dx=_DX,
                     boundary="cpml", cpml_layers=8, dx_profile=prof)
    sim.add_tfsf_source(direction="+x", f0=6e9, bandwidth=0.5,
                        angle_deg=angle_deg)
    return sim


def test_a_dx_only_graded_mesh_is_checked_for_tfsf_support():
    """The check is keyed on "an axis is graded", not on "a z profile was
    given".

    Oblique TFSF has no supported auxiliary on a graded mesh. The gate used to
    read ``if self._dz_profile is not None``, so a mesh graded only on x
    skipped this and every other check in the family and reported nothing --
    while the auxiliary line runs along x, the axis that was graded. The
    direction branch beside it cannot be reached from the public API, which
    rejects anything but +x/-x before preflight sees it, so oblique incidence
    is the branch that can be exercised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _dx_graded_tfsf_sim(30.0).preflight()
    assert "nonuniform_tfsf" in _codes(report)


def test_the_same_mesh_at_normal_incidence_is_not_flagged():
    """The gate opened a family of checks; it must not fire on the case the
    family says IS supported, or it is just noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _dx_graded_tfsf_sim(0.0).preflight()
    assert "nonuniform_tfsf" not in _codes(report)


# ---------------------------------------------------------------------------
# The z absorber thickness: which face, and which cells
# ---------------------------------------------------------------------------

def _z_graded_ends(lo_cell, hi_cell, n_end=8, n_flat=12):
    """A z profile whose two ends differ, with a ratio-capped ramp between."""
    ramp, c = [], lo_cell
    while c > hi_cell * 1.0001:
        c = max(c / 1.2, hi_cell)
        ramp.append(c)
    return np.array([lo_cell] * n_end + ramp + [hi_cell] * n_flat, float)


def _z_graded_sim(prof, cpml_layers=8, boundary="cpml"):
    return Simulation(freq_max=10e9,
                      domain=(0.02, 0.02, float(prof.sum())), dx=_DX,
                      boundary=boundary, cpml_layers=cpml_layers,
                      dz_profile=prof)


def test_the_absorber_thickness_is_measured_on_the_cells_it_sits_on():
    """The absorber on a face sits on copies of THAT face's boundary cell.

    ``_pad_profile`` fills each absorber pad with the profile's own end cell,
    so the z_hi absorber is ``n_hi`` copies of the LAST cell. The old read
    summed the first ``max(lo, hi)`` INTERIOR cells: neither face's absorber,
    and the hi face measured from the lo end.

    On this profile the hi end is a fifth of the lo end, so the two readings
    are 1.6 mm and 8.0 mm against a 2.4 mm threshold -- the advisory itself is
    the discriminator, not its wording. The old read stays silent about an
    absorber that is a fifth of the transverse one.
    """
    prof = _z_graded_ends(_DX, 0.2 * _DX)
    n_layers = 8
    new_read = n_layers * profile_boundary_cell(_DX, prof, "hi")
    old_read = float(np.sum(prof[:n_layers]))
    threshold = 0.3 * n_layers * _DX
    assert new_read < threshold < old_read

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _z_graded_sim(prof, n_layers).preflight()
    texts = [str(i) for i in report.issues
             if getattr(i, "code", None) == "nonuniform_cpml_thin"]
    assert texts, _codes(report)
    # the z figure is the hi face's absorber; ``old_read`` is above the
    # threshold asserted just above, so the old code emitted nothing at all
    assert f"CPML z-thickness is {new_read * 1e3:.1f}mm" in texts[0], texts[0]


def test_a_face_that_allocates_no_absorber_is_not_measured_as_zero():
    """A PEC z_lo has no absorber to be thin, and counting its zero would
    bring back the #647 false positive.

    The board is the mirror of the one above: the fine cells are at z_lo,
    which is PEC, and the CPML sits on the coarse z_hi end at the full 8 mm.
    Nothing about this mesh is thin, and nothing is reported.
    """
    prof = _z_graded_ends(_DX, 0.2 * _DX)[::-1].copy()
    walls = BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _z_graded_sim(prof, 8, boundary=walls).preflight()
    assert "nonuniform_cpml_thin" not in _codes(report)


# ---------------------------------------------------------------------------
# The MSL source-fringing standoff: a length, counted on the runway's cells
# ---------------------------------------------------------------------------

_NOTCH_RUNWAY = 127e-6     # the notch instrument's board
_NOTCH_H_SUB = 254e-6


def _graded_runway_board():
    """A board whose port runway is refined 2x below the boundary cell."""
    coarse = 2 * _NOTCH_RUNWAY
    ramp, c = [], coarse
    while c > _NOTCH_RUNWAY * 1.0001:
        c = max(c / 1.25, _NOTCH_RUNWAY)
        ramp.append(c)
    prof = np.array([coarse] * 14 + ramp + [_NOTCH_RUNWAY] * 60
                    + ramp[::-1] + [coarse] * 14, float)
    return prof, float(np.sum(prof[:14 + len(ramp)])), len(ramp)


def _msl_board(prof, feed, n_probe_offset):
    from rfx.geometry.csg import Box
    h, w = _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    lx, ly = float(prof.sum()), 2 * _NOTCH_H_SUB + 8 * _NOTCH_H_SUB
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, h + 1.5e-3), dx=2 * _NOTCH_RUNWAY,
        cpml_layers=8, dx_profile=prof,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - w / 2, h), (lx, y_c + w / 2, h + 2 * _NOTCH_RUNWAY)),
            material="pec")
    sim.add_msl_port(position=(feed, y_c, 0), width=w, height=h,
                     direction="+x", impedance=50.0,
                     n_probe_offset=n_probe_offset)
    return sim


def _forward_message(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc:
            sim.preflight_sparameters(calculator="forward", strict=True)
    return str(exc.value)


def test_the_standoff_floor_is_counted_on_the_runway_cell():
    """The fringing decays over about five substrate thicknesses. That is a
    LENGTH; it knows nothing about the mesh. The cell count is that length
    divided by the cells the runway actually has where the port sits.

    This port sits on a runway refined 2x below the boundary cell, so the two
    readings differ by 2x: 5 cells on the boundary scalar, 10 on the runway's
    own. An explicit ``n_probe_offset=7`` clears the first and violates the
    second, and at 7 cells of 127 um the probe is 0.889 mm from the feed
    where the physics asks for 1.270 mm. The old read called that safe.
    """
    prof, x_run, _ = _graded_runway_board()
    msg = _forward_message(_msl_board(prof, x_run + 20 * _NOTCH_RUNWAY, 7))
    assert "sits within the source fringing transient (10 cells" in msg, msg


def test_a_generous_offset_on_the_same_runway_is_not_flagged():
    """The floor must not fire on an offset that clears it, or it is noise."""
    prof, x_run, _ = _graded_runway_board()
    msg = _forward_message(_msl_board(prof, x_run + 20 * _NOTCH_RUNWAY, 14))
    assert "source fringing transient" not in msg, msg


def test_a_standoff_that_crosses_a_ramp_is_refused_not_answered():
    """A count in cells names one distance only where the cells are equal.

    Put the feed two cells into the grading ramp and the standoff span runs
    over cells of several sizes, so no single ``n_probe_offset`` names the
    1.27 mm the physics asks for. The check says that instead of answering,
    the way the S path already refuses a reference span that leaves one zone.
    """
    prof, _, _ = _graded_runway_board()
    feed = float(np.sum(prof[:16]))
    msg = _forward_message(_msl_board(prof, feed, 7))
    assert "crosses cells of more than one size on the x runway" in msg, msg
    assert "sits within the source fringing transient" not in msg, msg


def test_the_notch_instrument_board_reads_the_same_either_way():
    """The load-bearing board cannot tell the two readings apart, because its
    runway cell IS its boundary cell: 127 um everywhere, 10 cells = 1.270 mm
    either way. Its shipped ``n_probe_offset=14`` clears that floor and stays
    the user's own setting; the check is a floor, not a solve."""
    from rfx.preflight.msl import msl_source_near_field_standoff_cells as _nf

    assert _nf(_NOTCH_H_SUB, _NOTCH_RUNWAY) == 10
    assert _nf(_NOTCH_H_SUB, _NOTCH_RUNWAY) * _NOTCH_RUNWAY == pytest.approx(
        1.270e-3, rel=1e-9)
    assert _nf(_NOTCH_H_SUB, 2 * _NOTCH_RUNWAY) == 5


# ---------------------------------------------------------------------------
# The source-on-a-reflector-plane tolerance: half a cell AT THAT FACE
# ---------------------------------------------------------------------------

def _z_ends(lo_cell, hi_cell, n_end=8, n_flat=12):
    """A z profile running from ``lo_cell`` at z_lo to ``hi_cell`` at z_hi."""
    ramp, c = [], lo_cell
    down = lo_cell > hi_cell
    while (c > hi_cell * 1.0001) if down else (c < hi_cell / 1.0001):
        c = max(c / 1.2, hi_cell) if down else min(c * 1.2, hi_cell)
        ramp.append(c)
    return np.array([lo_cell] * n_end + ramp + [hi_cell] * n_flat, float)


def _pec_box_with_source_below_z_hi(prof, gap):
    lz = float(prof.sum())
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, lz), dx=_DX,
                     boundary="pec", dz_profile=prof)
    sim.add_source((0.01, 0.01, lz - gap), component="ex",
                   amplitude_kind="current")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.preflight()


def test_a_source_a_coarse_half_cell_from_the_far_wall_is_caught():
    """A tangential E source on a PEC wall is held at zero by the mirror, and
    "on the wall" means within half of the cell AT THAT WALL.

    Here z_hi's cell is 1 mm and z_lo's is 0.2 mm. A source 0.3 mm below the
    top wall is inside z_hi's half-cell, so the mirror has it. The old read
    took the profile's LEADING entry for both faces, measured this wall with
    the far wall's 0.2 mm cell, and let a decoupled source through.
    """
    report = _pec_box_with_source_below_z_hi(_z_ends(0.2 * _DX, _DX), 0.3e-3)
    assert "source_decoupled" in _codes(report), _codes(report)
    text = next(str(i) for i in report.issues
                if getattr(i, "code", None) == "source_decoupled")
    assert "z_hi" in text and "(1 mm)" in text, text


def test_a_source_outside_the_fine_wall_half_cell_is_not_flagged():
    """Mirror board: z_hi's cell is 0.2 mm, so a source 0.3 mm below the top
    wall is a cell and a half away and couples normally. The old read judged
    it against z_lo's 1 mm cell and reported a source that is not decoupled.
    """
    report = _pec_box_with_source_below_z_hi(_z_ends(_DX, 0.2 * _DX), 0.3e-3)
    assert "source_decoupled" not in _codes(report), _codes(report)
