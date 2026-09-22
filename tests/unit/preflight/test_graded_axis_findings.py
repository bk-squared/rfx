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


def _msl_board(prof, feed, n_probe_offset, direction="+x"):
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
                     direction=direction, impedance=50.0,
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


def test_the_span_inspected_runs_the_way_the_port_launches():
    """The probes sit downstream of the feed, so a -x port's standoff occupies
    the cells BELOW its feed plane and a +x port's the cells above.

    This feed sits two cells into the uniform runway with the grading ramp
    just behind it. Launching backwards, the standoff crosses the ramp and is
    refused; launching forwards from the same plane it stays on 127 um cells
    and only the floor is reported. An unsigned span inspects the +x side for
    both and misses the ramp the -x port actually runs over.
    """
    prof, x_run, _ = _graded_runway_board()
    feed = x_run + 2 * _NOTCH_RUNWAY
    back = _forward_message(_msl_board(prof, feed, 7, direction="-x"))
    fwd = _forward_message(_msl_board(prof, feed, 7, direction="+x"))
    assert "crosses cells of more than one size" in back, back
    assert "crosses cells of more than one size" not in fwd, fwd
    assert "sits within the source fringing transient (10 cells" in fwd, fwd


# ---------------------------------------------------------------------------
# One absorber, one thickness; and a proximity band the width of its own cells
# ---------------------------------------------------------------------------

def test_the_two_preflight_numbers_for_one_z_absorber_agree():
    """Preflight reports the z absorber's thickness twice, and on a graded
    mesh the two used to disagree.

    The per-face thickness that feeds the MSL and waveguide clearance
    advisories summed the LEADING profile entries for both faces, so on this
    board it called the z_hi absorber 8.000 mm where the grid allocates
    1.600 mm, a factor of 5. The mesh advisory computed its own number a
    different way. Both now come out of one function, and both match the
    cells the pad actually holds.
    """
    prof = _z_graded_ends(_DX, 0.2 * _DX)
    sim = _z_graded_sim(prof, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    thick_lo, thick_hi, _ = sim._validate_cfg_compute_cpml_thickness(8 * _DX)
    assert thick_lo[2] == pytest.approx(8 * float(prof[0]))
    assert thick_hi[2] == pytest.approx(8 * float(prof[-1]))
    assert thick_hi[2] == pytest.approx(1.600e-3)

    texts = [str(i) for i in report.issues
             if getattr(i, "code", None) == "nonuniform_cpml_thin"]
    assert texts, _codes(report)
    assert f"CPML z-thickness is {thick_hi[2] * 1e3:.1f}mm" in texts[0], texts[0]


def _source_below_z_hi(prof, gap):
    sim = _z_graded_sim(prof, 8)
    sim.add_source((0.01, 0.01, float(prof.sum()) - gap), "ez",
                   amplitude_kind="current")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.preflight()


def test_the_proximity_band_is_measured_in_the_cells_it_covers():
    """The band is two cells wide, measured inward from where the absorber
    begins, so the cells it covers are the ones at that face.

    z_hi's cells are 200 um here and the boundary scalar is 1 mm. The band is
    400 um, and a source 300 um below the top wall is inside it. The old read
    called the band 2 mm and quoted that length for a face whose cells are a
    fifth of it.
    """
    prof = _z_graded_ends(_DX, 0.2 * _DX)
    report = _source_below_z_hi(prof, 0.3e-3)
    texts = [str(i) for i in report.issues
             if getattr(i, "code", None) == "absorber_proximity"]
    assert texts, _codes(report)
    assert "(400µm)" in texts[0], texts[0]
    assert "(2mm)" not in texts[0], texts[0]


def test_a_source_outside_the_fine_face_band_is_not_flagged():
    """Same board, source 1.5 mm below the top wall: more than seven of that
    face's cells away, and silent. The 2 mm band the old read used would have
    reported it."""
    prof = _z_graded_ends(_DX, 0.2 * _DX)
    report = _source_below_z_hi(prof, 1.5e-3)
    assert "absorber_proximity" not in _codes(report)


# ---------------------------------------------------------------------------
# The MSL-side standoff: the same number as the S path, on the same cells
# ---------------------------------------------------------------------------

def _notch_like_board(feed_cells, n_probe_offset, tail=14):
    """A board whose port runway is refined 2x below the boundary cell.

    ``feed_cells`` indexes the profile, so a caller can put the feed plane
    inside the uniform runway or part way up the grading ramp.
    """
    from rfx.geometry.csg import Box
    coarse = 2 * _NOTCH_RUNWAY
    ramp, c = [], coarse
    while c > _NOTCH_RUNWAY * 1.0001:
        c = max(c / 1.25, _NOTCH_RUNWAY)
        ramp.append(c)
    prof = np.array([coarse] * 14 + ramp + [_NOTCH_RUNWAY] * 60
                    + ramp[::-1] + [coarse] * tail, float)
    h, w = _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    lx, ly = float(prof.sum()), 10 * _NOTCH_H_SUB
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, h + 1.5e-3), dx=coarse,
        cpml_layers=8, dx_profile=prof,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - w / 2, h), (lx, y_c + w / 2, h + coarse)),
            material="pec")
    sim.add_msl_port(position=(float(np.sum(prof[:feed_cells])), y_c, 0),
                     width=w, height=h, direction="+x", impedance=50.0,
                     n_probe_offset=n_probe_offset, n_probe_spacing=3,
                     n_probes=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.preflight(), len(ramp)


def test_the_msl_standoff_advisory_counts_the_runway_cell():
    """The same port, judged by the microstrip check and by the S-parameter
    routing check, must get the same answer (issue #823's invariant).

    The S path moved to the runway's own cells; this one still read the
    boundary scalar, so it asked for 5 cells where the physics asks for 10 and
    passed an offset that puts probe 0 at 889 um from the feed, well inside
    the 1.270 mm fringing transient.
    """
    report, n_ramp = _notch_like_board(14 + 20, 7)
    texts = [str(i) for i in report.issues if "OWN feed plane" in str(i)]
    assert texts, _codes(report)
    assert "standoff of 10 cells" in texts[0], texts[0]
    assert "889µm" in texts[0], texts[0]


def test_a_generous_msl_offset_on_the_same_runway_is_not_flagged():
    """The floor must not fire on an offset that clears it."""
    report, _ = _notch_like_board(14 + 20, 14)
    assert not [str(i) for i in report.issues if "OWN feed plane" in str(i)]


def test_the_msl_interval_refuses_a_standoff_that_crosses_a_ramp():
    """The advertised compliant interval is a range of cell COUNTS, and a
    count names one distance only where the cells are equal.

    With the feed two cells up the grading ramp the standoff itself runs over
    cells of several sizes, so the advisory says that instead of printing a
    range.
    """
    on_ramp, _ = _notch_like_board(16, 60, tail=4)
    text = _finding(on_ramp, "past the domain edge")
    assert "source-fringing standoff crosses cells" in text, text
    assert "compliant n_probe_offset interval" not in text, text


def test_the_interval_refuses_a_ladder_that_leaves_the_runway():
    """A port can sit in a uniform runway and still send its deepest probe
    over a ramp, and the interval endpoints count the WHOLE ladder.

    Here the feed is twenty cells inside the uniform 127 um runway, so the
    1.270 mm standoff is on one cell size and the standoff guard is happy.
    The ladder is not: at offset 60 the deepest probe runs out through the
    trailing ramp into the coarse tail, so no single offset in cells names
    one distance for it either, and the advisory names the ladder rather
    than the standoff.
    """
    report, _ = _notch_like_board(14 + 20, 60, tail=4)
    text = _finding(report, "past the domain edge")
    assert "probe ladder crosses cells of more than one size" in text, text
    assert "compliant n_probe_offset interval" not in text, text



def _stub_board(feed_cells, gap_m=3.0e-3):
    """Through-line with an open stub branched off it, on a runway refined 2x
    below the boundary cell. ``feed_cells`` indexes the profile, so the feed
    can sit inside the uniform runway or one cell up the grading ramp."""
    from rfx.geometry.csg import Box
    coarse = 2 * _NOTCH_RUNWAY
    ramp, c = [], coarse
    while c > _NOTCH_RUNWAY * 1.0001:
        c = max(c / 1.25, _NOTCH_RUNWAY)
        ramp.append(c)
    prof = np.array([coarse] * 8 + ramp + [_NOTCH_RUNWAY] * 110
                    + ramp[::-1] + [coarse] * 8, float)
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    lx, h, w = float(prof.sum()), _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    stub = 8e-3
    ly = w + 4 * (2 * h + 8 * coarse) + max(14e-3, stub + 2e-3)
    sim = Simulation(
        freq_max=9e9, domain=(lx, ly, h + 1.5e-3), dx=coarse, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dx_profile=prof)
    sim.add_material("ro", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="ro")
    y_tr = (2 * h + 8 * coarse) + w / 2
    sim.add(Box((0, y_tr - w / 2, h), (lx, y_tr + w / 2, h)), material="pec")
    feed = float(edges[feed_cells])
    x_stub = float(edges[min(feed_cells + 32, len(prof))]) + gap_m
    sim.add(Box((x_stub - w / 2, y_tr + w / 2, h),
                (x_stub + w / 2, y_tr + w / 2 + stub, h)), material="pec")
    sim.add_msl_port(position=(feed, y_tr, 0), width=w, height=h,
                     direction="+x", impedance=50.0, n_probe_offset=20,
                     n_probe_spacing=3, n_probes=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    texts = [str(i) for i in report.issues
             if "strong reflector candidate" in str(i)]
    assert texts, _codes(report)
    return texts[0]


def test_the_reflector_interval_lower_edge_is_the_runway_reading():
    """Check 4 advertises the same compliant interval check 4a does, and its
    lower edge IS the source-fringing standoff (issue #823).

    A stub branched off the line 3 mm past the deepest probe is inside the
    quarter-guide-wavelength clearance, so the advisory fires and prints the
    interval. On this runway the standoff is 10 cells of 127 um; the boundary
    scalar would have advertised 5, an offset that puts probe 0 half way into
    the fringing transient.
    """
    text = _stub_board(8 + 4 + 4)
    # both endpoints: the lower is the standoff, the upper is the walked-down
    # offset whose deepest probe still clears the reflector on the realized
    # ladder, so a cell size read anywhere else moves one or the other
    assert "compliant n_probe_offset interval \u2248 [10, 12] cells" in text, text


def test_the_reflector_interval_refuses_a_feed_on_the_ramp():
    """Same board, feed one cell up the grading ramp: the standoff runs over
    cells of several sizes, so no single offset in CELLS names the 1.27 mm the
    physics asks for, and the advisory says that instead of printing a range.
    """
    text = _stub_board(9)
    assert "does not name one distance here" in text, text
    assert "compliant n_probe_offset interval" not in text, text


# ---------------------------------------------------------------------------
# One port, one story: every site refuses a standoff that crosses a ramp
# ---------------------------------------------------------------------------

_RAMP_REFUSAL = "crosses cells of more than one size"


def _ramp_feed_board(n_probe_offset, stub_gap_m=1.0e-3):
    """The reviewer's board: 254 um substrate, a runway that ramps 254 -> 127
    um at ratio 1.25, and the feed plane at the coarse end of that ramp.

    A branch stub follows the probe ladder so the reflector advisory has
    something to report; at a large offset the ladder reaches the absorber
    instead and the stub falls outside the board.
    """
    from rfx.geometry.csg import Box
    coarse = 2 * _NOTCH_RUNWAY
    ramp, c = [], coarse
    while c > _NOTCH_RUNWAY * 1.0001:
        c = max(c / 1.25, _NOTCH_RUNWAY)
        ramp.append(c)
    prof = np.array([coarse] * 14 + ramp + [_NOTCH_RUNWAY] * 80
                    + ramp[::-1] + [coarse] * 8, float)
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    lx, h, w = float(prof.sum()), _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    stub = 8e-3
    ly = w + 4 * (2 * h + 8 * coarse) + max(14e-3, stub + 2e-3)
    sim = Simulation(
        freq_max=9e9, domain=(lx, ly, h + 1.5e-3), dx=coarse, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dx_profile=prof)
    sim.add_material("ro", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="ro")
    y_tr = (2 * h + 8 * coarse) + w / 2
    sim.add(Box((0, y_tr - w / 2, h), (lx, y_tr + w / 2, h)), material="pec")
    x_stub = float(edges[min(14 + n_probe_offset + 12, len(prof))]) + stub_gap_m
    if x_stub < lx:
        sim.add(Box((x_stub - w / 2, y_tr + w / 2, h),
                    (x_stub + w / 2, y_tr + w / 2 + stub, h)), material="pec")
    sim.add_msl_port(position=(float(edges[14]), y_tr, 0), width=w, height=h,
                     direction="+x", impedance=50.0,
                     n_probe_offset=n_probe_offset, n_probe_spacing=3,
                     n_probes=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim, sim.preflight()


def _finding(report, mark):
    hits = [str(i) for i in report.issues if mark in str(i)]
    assert hits, f"{mark!r} not reported; codes={_codes(report)}"
    return hits[0]


def test_every_site_refuses_a_standoff_that_crosses_a_ramp():
    """A feed plane sitting where the mesh changes cell size gets the same
    answer from every check that turns the source-fringing standoff into a
    cell count.

    The standoff is about five substrate thicknesses, 1.270 mm here. Reading
    the cell at the feed and multiplying gives numbers the grid does not
    realize: at offset 4 that arithmetic says probe 0 is 812.8 um from the
    feed and that 6 cells would clear the transient, while the grid puts
    probe 0 at 622.8 um and 6 cells at 876.8 um, 31 % short. So none of these
    checks prints a distance or a cell-count remedy here.
    """
    sim, report = _ramp_feed_board(4)
    standoff = _finding(report, "fringing decays over about five")
    assert _RAMP_REFUSAL in standoff
    # the distance it does quote is the one the grid realizes, read off the
    # ladder the extractor uses, not n_offset times the cell at the feed
    assert "622.8\u00b5m" in standoff, standoff
    assert "812.8\u00b5m" not in standoff, standoff
    assert _RAMP_REFUSAL in _finding(report, "strong reflector candidate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc:
            sim.preflight_sparameters(calculator="forward", strict=True)
    assert _RAMP_REFUSAL in str(exc.value), str(exc.value)

    # the fourth site needs the ladder to reach the absorber, which takes an
    # offset the stub cannot follow onto this board
    _, deep = _ramp_feed_board(82)
    assert _RAMP_REFUSAL in _finding(deep, "domain edge, just")


def test_the_standoff_floor_still_reports_where_the_cells_are_equal():
    """The refusal must not swallow the advisory it stands in for: on a feed
    inside the uniform runway the floor is still reported as a count."""
    report, _ = _notch_like_board(14 + 20, 7)
    text = _finding(report, "OWN feed plane")
    assert _RAMP_REFUSAL not in text, text
    assert "standoff of 10 cells" in text, text


# ---------------------------------------------------------------------------
# The walk-down search reads the face cells off the grid it walks
# ---------------------------------------------------------------------------

def _y_graded_board(n_probe_offset):
    """A +y microstrip on a y profile whose boundary cell is half the scalar.

    ``make_nonuniform_grid`` ties a y profile's two ends to each other but not
    to ``dx``, so 0.5 mm cells on a 1 mm scalar board is a legal mesh and the
    only thing that separates the face cell from the scalar.
    """
    from rfx.geometry.csg import Box
    dx, cy, h, w = 1.0e-3, 0.5e-3, 1.0e-3, 2.0e-3
    prof = np.full(60, cy)
    ly, lx, lz = float(prof.sum()), 12 * dx, h + 3 * dx
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dy_profile=prof)
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="sub")
    x_c = lx / 2
    sim.add(Box((x_c - w / 2, 0, h), (x_c + w / 2, ly, h + dx)), material="pec")
    sim.add_msl_port(position=(x_c, 1.0e-3, 0), width=w, height=h,
                     direction="+y", impedance=50.0,
                     n_probe_offset=n_probe_offset, n_probe_spacing=3,
                     n_probes=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim, sim.preflight()


def test_the_absorber_y_thickness_follows_the_y_face_cell():
    """The y absorber is eight copies of the y face cell, 4.0 mm, not eight
    copies of the boundary scalar. The scalar reading called it 8.0 mm, which
    is a quarter of this 30 mm board reported as absorber on each side."""
    sim, _ = _y_graded_board(45)
    thick_lo, thick_hi, _ = sim._validate_cfg_compute_cpml_thickness(8 * 1.0e-3)
    assert thick_lo[1] == pytest.approx(4.0e-3)
    assert thick_hi[1] == pytest.approx(4.0e-3)


def test_the_walk_down_endpoint_uses_the_face_cell_not_the_scalar():
    """The advertised endpoint is walked down until the real predicate clears,
    and that predicate's proximity band is two cells of the face it is
    measured from: 1.0 mm here, not the 2.0 mm the boundary scalar gives.

    A wider band rejects offsets that are in fact clear, so the endpoint the
    advisory advertises comes back short. On this port it reads 43 cells; on
    the scalar band it read 41, and a user who took that number would place
    the probes two cells inside the line for no reason.
    """
    _, report = _y_graded_board(45)
    text = _finding(report, "domain edge, just")
    assert "compliant n_probe_offset interval ≈ [10, 43] cells" in text, text


# ---------------------------------------------------------------------------
# A feed plane exactly on a node: which of the two cells it belongs to
# ---------------------------------------------------------------------------

def _two_zone_board(direction, n_probe_offset):
    """127 um in the middle, 254 um at both ends, and the feed plane exactly
    on the node where the fine zone ends.

    That node belongs to two cells. A port launching backwards measures the
    cells its probes occupy, which are the 127 um ones below the node; the
    254 um cell above it is on the other side of the port.
    """
    from rfx.geometry.csg import Box
    coarse = 2 * _NOTCH_RUNWAY
    prof = np.array([coarse] * 10 + [_NOTCH_RUNWAY] * 60 + [coarse] * 10,
                    float)
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    lx, h, w = float(prof.sum()), _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    ly = 10 * h
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, h + 1.5e-3), dx=coarse, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dx_profile=prof)
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - w / 2, h), (lx, y_c + w / 2, h + coarse)),
            material="pec")
    sim.add_msl_port(position=(float(edges[70]), y_c, 0), width=w, height=h,
                     direction=direction, impedance=50.0,
                     n_probe_offset=n_probe_offset, n_probe_spacing=3,
                     n_probes=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.preflight()


def test_a_backward_port_on_a_node_is_measured_on_the_cells_it_uses():
    """The standoff is counted on the cells the probes occupy.

    This feed sits exactly where 127 um cells give way to 254 um ones, and
    the port launches into the fine side. Read there, the 1.270 mm fringing
    transient is 10 cells and an offset of 7 does not clear it. Read on the
    254 um cell above the node -- cells this port never touches -- it is 5
    cells and the same offset looks generous.
    """
    report = _two_zone_board("-x", 7)
    text = _finding(report, "OWN feed plane")
    assert "standoff of 10 cells" in text, text
    assert "889µm" in text, text


def test_that_backward_span_is_not_read_as_leaving_its_zone():
    """The standoff ends exactly on the node the feed sits on, and every cell
    it covers is 127 um, so nothing is refused here.

    Counting the cell above the closing node made a span lying wholly inside
    one zone look mixed the moment it touched the zone's far edge, which
    refused a port that is perfectly placed.
    """
    report = _two_zone_board("-x", 7)
    assert _RAMP_REFUSAL not in _finding(report, "OWN feed plane")
