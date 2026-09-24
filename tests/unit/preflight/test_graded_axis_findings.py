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

import re
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


def test_the_same_mesh_at_normal_incidence_is_flagged_too():
    """Normal incidence along a graded x is not the supported case either.

    The runner builds the 1-D line that computes the incident wave on one
    cell size, the boundary cell, while the 3-D grid carries the wave through
    the graded x cells. The two waves part company inside the box, and the
    difference leaks out of it; the witness below measures how much.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _dx_graded_tfsf_sim(0.0).preflight()
    assert "nonuniform_tfsf" in _codes(report)


@pytest.mark.parametrize("graded", ["y", "z"])
def test_a_constant_x_with_a_graded_transverse_axis_is_not_flagged(graded):
    """The incident line runs along x, so grading y or z leaves it matching
    the 3-D grid; the witness measures the leakage there at the uniform
    level. Flagging it would be noise."""
    prof = np.concatenate([np.full(4, _DX), np.full(8, 0.5 * _DX),
                           np.full(4, _DX)])
    kw = {f"d{graded}_profile": prof}
    ext = {"y": 0.02, "z": 0.02, graded: float(np.sum(prof))}
    sim = Simulation(freq_max=10e9, domain=(0.06, ext["y"], ext["z"]),
                     dx=_DX, boundary="cpml", cpml_layers=8, **kw)
    sim.add_tfsf_source(direction="+x", f0=6e9, bandwidth=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    assert sim._uses_nonuniform_mesh
    assert "nonuniform_tfsf" not in _codes(report)


def _tfsf_leakage(**profiles):
    """Peak |Ez| in the scattered-field region beyond the far TFSF plane,
    over the peak inside the box, for a +x plane wave with NO scatterer:
    50 x 6 x 6 cells of 1 mm unless a profile says otherwise, 1500 steps.
    With nothing to scatter, anything outside the box is leakage."""
    lx = float(np.sum(profiles.get("dx_profile", np.full(50, _DX))))
    ly = float(np.sum(profiles.get("dy_profile", np.full(6, _DX))))
    lz = float(np.sum(profiles.get("dz_profile", np.full(6, _DX))))
    sim = Simulation(freq_max=10e9, domain=(lx, ly, lz), dx=_DX,
                     boundary="cpml", cpml_layers=8, **profiles)
    sim.add_tfsf_source(direction="+x", f0=6e9, bandwidth=0.5, amplitude=1.0,
                        margin=3, polarization="ez")
    sim.add_probe((lx / 2 + 0.25e-3, ly / 2, lz / 2), "ez")
    sim.add_probe((lx - 1.0e-3, ly / 2, lz / 2), "ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        codes = _codes(sim.preflight())
        ts = np.asarray(sim.run(n_steps=1500, compute_s_params=False)
                        .time_series)
    return (float(np.max(np.abs(ts[:, 1])) / np.max(np.abs(ts[:, 0]))),
            "nonuniform_tfsf" in codes)


def test_the_leakage_the_tfsf_advisory_quotes():
    """The witness for the numbers the advisory prints.

    A plane wave with nothing in its way must stay inside the TFSF box. On
    uniform 1 mm x cells the scattered-field region reads about 1e-5 of the
    total-field peak, and grading only y keeps it there. Grading x
    1 / 0.5 / 1 mm (20 cells each) puts 0.7 of the peak outside the box: the
    incident line is built on the 1 mm boundary cell, so the wave it injects
    at the far plane is not the one the graded grid delivers there. The
    advisory fires on that board and on no other.
    """
    uniform, flagged_u = _tfsf_leakage(dx_profile=np.full(50, _DX))
    y_graded, flagged_y = _tfsf_leakage(dy_profile=np.concatenate(
        [np.full(2, _DX), np.full(4, 0.5 * _DX), np.full(2, _DX)]))
    x_graded, flagged_x = _tfsf_leakage(dx_profile=np.concatenate(
        [np.full(20, _DX), np.full(20, 0.5 * _DX), np.full(20, _DX)]))
    assert uniform < 2e-5, uniform
    assert y_graded < 2e-5, y_graded
    # 0.716 with the magnetic CPML profile at the Yee half cell, the same with
    # 8, 16 or 24 absorber layers. Before #1012 this read 0.598 with 8 layers,
    # 0.677 with 16 and 0.702 with 24: the probe sits 1 mm in front of the x-hi
    # absorber, and the old absorber's reflection reached it (Addendum 11 of
    # #1012).
    assert 0.67 < x_graded < 0.77, x_graded
    assert (flagged_u, flagged_y, flagged_x) == (False, False, True)


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
    """``_notch_like_sim``'s board, preflighted: ``(report, ramp cells)``."""
    sim, n_ramp = _notch_like_sim(feed_cells, n_probe_offset, tail)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.preflight(), n_ramp


def _notch_like_sim(feed_cells, n_probe_offset, tail=14):
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
    return sim, len(ramp)


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


# ---------------------------------------------------------------------------
# The reflector interval on a uniform mesh, with the feed off its node
# ---------------------------------------------------------------------------

def _uniform_stub_finding(feed, direction, n_probe_offset):
    """Uniform 254 um cells, a 254 um substrate, and an open stub branched off
    the line 3 mm past where an offset of 20 puts the deepest probe. Returns
    the reflector finding, or None when check 4 is silent."""
    from rfx.geometry.csg import Box
    dx, h, w = 254e-6, 254e-6, 508e-6
    lx, stub = 40e-3, 8e-3
    ly = w + 4 * (2 * h + 8 * dx) + max(14e-3, stub + 2e-3)
    sim = Simulation(
        freq_max=9e9, domain=(lx, ly, h + 1.5e-3), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="ro")
    y_tr = (2 * h + 8 * dx) + w / 2
    sim.add(Box((0, y_tr - w / 2, h), (lx, y_tr + w / 2, h)), material="pec")
    sign = -1 if direction == "-x" else 1
    x_stub = feed + sign * (32 * dx + 3.0e-3)
    sim.add(Box((x_stub - w / 2, y_tr + w / 2, h),
                (x_stub + w / 2, y_tr + w / 2 + stub, h)), material="pec")
    sim.add_msl_port(position=(feed, y_tr, 0), width=w, height=h,
                     direction=direction, impedance=50.0,
                     n_probe_offset=n_probe_offset, n_probe_spacing=3,
                     n_probes=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    hits = [str(i) for i in report.issues
            if "strong reflector candidate" in str(i)]
    return hits[0] if hits else None


@pytest.mark.parametrize("feed, direction", [(36.9e-3, "-x"), (3.0e-3, "+x")])
def test_the_printed_reflector_interval_is_the_one_check_4_accepts(
        feed, direction):
    """The upper edge of the advertised interval is the largest offset whose
    deepest probe still clears the stub, and check 4 itself must agree.

    The grid stamps the source on the node nearest the declared feed and
    counts the probe ladder from there. On the -x board the feed is declared
    70 um above the node at 36.83 mm, so the node lies toward the probes;
    on the +x board it is declared 48 um below the node at 3.048 mm, the
    same way round. Measuring the ladder from the declared feed added that
    snap to the node-to-stub distance and advertised offset 16, which puts
    the deepest probe 3692 um from the stub where 3724 um is needed, and
    check 4 fired on the offset its own interval recommended.
    """
    text = _uniform_stub_finding(feed, direction, 20)
    assert text is not None
    upper = int(re.search(r"interval \u2248 \[\d+, (\d+)\] cells",
                          text).group(1))
    assert _uniform_stub_finding(feed, direction, upper) is None, upper
    assert _uniform_stub_finding(feed, direction, upper + 1) is not None, upper


# ---------------------------------------------------------------------------
# Ports declared with literal coordinates on the library's own meshes
# ---------------------------------------------------------------------------

def _ports_on(prof, dx, ports, n_probe_offset, stub_x=None, freq_max=5e9):
    """A through line on the x profile ``prof`` with one MSL port per
    ``(name, x, direction)``, each placed at the LITERAL coordinate given.
    ``stub_x`` adds an open stub centred there."""
    from rfx.geometry.csg import Box
    h, w = _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    lx, ly = float(np.sum(prof)), 24 * _NOTCH_H_SUB
    sim = Simulation(
        freq_max=freq_max, domain=(lx, ly, h + 1.5e-3), dx=dx, cpml_layers=8,
        dx_profile=prof,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - w / 2, h), (lx, y_c + w / 2, h)), material="pec")
    if stub_x is not None:
        sim.add(Box((stub_x - w / 2, y_c + w / 2, h),
                    (stub_x + w / 2, y_c + w / 2 + 4e-3, h)), material="pec")
    for name, x, direction in ports:
        sim.add_msl_port(position=(x, y_c, 0), width=w, height=h,
                         direction=direction, impedance=50.0,
                         n_probe_offset=n_probe_offset, n_probe_spacing=3,
                         n_probes=5, name=name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim, sim.preflight()


def _band_runway(a, b):
    """254 um lead-in and lead-out, a protected ~127 um runway from a to b."""
    from rfx.nonuniform import make_band_profile
    c = 2 * _NOTCH_RUNWAY
    return make_band_profile([0.0, a, b, b + 5e-3], [c, _NOTCH_RUNWAY, c],
                             max_ratio=1.3, protected=[False, True, False],
                             boundary_cell=c)


def _port_finding(report, name, mark):
    hits = [str(i) for i in report.issues
            if f"MSL port '{name}'" in str(i) and mark in str(i)]
    assert hits, f"{name}: {mark!r} not reported; codes={_codes(report)}"
    return hits[0]


def _assert_floor_on_the_runway(report, names):
    """Each port gets the standoff floor counted on the runway, 10 cells of
    ~127 um for the 1.270 mm the fringing needs, and nothing is refused."""
    for name in names:
        text = _port_finding(report, name, "OWN feed plane")
        assert "standoff of 10 cells" in text, text
    assert not [str(i) for i in report.issues if _RAMP_REFUSAL in str(i)]


def test_ports_declared_on_band_interfaces_read_the_runway():
    """``make_band_profile`` puts a node on each declared interface, to
    1e-12 m. A port declared there with the same literal lands a few 1e-18 m
    to one side of that node, and the grid stamps it ON the node.

    Here the +x port at x = 5 mm sits 8.7e-19 m below the node where the
    157 um ramp gives way to the 126 um runway. Read as "the cell containing
    the coordinate", it took the ramp cell under the node, called the feed a
    ramp feed, and refused a port whose source and probes are all on the
    runway. The -x port at 17 mm is the mirror, and it has to read the cells
    BELOW its node, the ones its probes occupy.
    """
    _, report = _ports_on(_band_runway(5e-3, 17e-3), 2 * _NOTCH_RUNWAY,
                          [("p1", 5e-3, "+x"), ("p2", 17e-3, "-x")], 7)
    _assert_floor_on_the_runway(report, ("p1", "p2"))


@pytest.mark.parametrize("p1, p2", [(3.1e-3, 15.2e-3), (5.0e-3, 17.0e-3)])
def test_ports_on_edge_aware_faces_read_the_runway(p1, p2):
    """The mesher the sheet advisory recommends, with the port planes passed
    as faces so each feed sits on a node.

    Each launch side is one cell size, to 2e-16 relative at worst, and each
    feed is within 1e-17 m of its node. Before, a feed a hair above its node
    on a -x port took the cell on the wrong side, and cells equal to the
    16th digit read as two sizes; either one refused a port that is on one
    uniform runway.
    """
    from rfx.geometry.csg import Box
    from rfx.mesh_edges import edge_aware_profiles
    h, w = _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    ly = 24 * _NOTCH_H_SUB
    trace = Box((0, ly / 2 - w / 2, h), (22e-3, ly / 2 + w / 2, h))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profs = edge_aware_profiles((22e-3, ly, h + 1.5e-3), _NOTCH_RUNWAY,
                                    sheets=[trace], faces={"x": [p1, p2]},
                                    axes="x")
    prof = np.asarray(getattr(profs["dx_profile"], "cells",
                              profs["dx_profile"]), float)
    _, report = _ports_on(prof, _NOTCH_RUNWAY,
                          [("p1", p1, "+x"), ("p2", p2, "-x")], 7)
    _assert_floor_on_the_runway(report, ("p1", "p2"))


def test_a_runway_built_from_node_coordinates_is_one_zone():
    """A profile written as ``np.diff`` of ``np.linspace`` node coordinates:
    its 90 runway cells are 127 um to 1.4e-14 relative, four distinct floats.
    The port sits ten cells into the runway, so its standoff never leaves
    it, and exact float equality called that span two sizes."""
    c = 2 * _NOTCH_RUNWAY
    nodes = np.concatenate([
        np.linspace(0, 12 * c, 13),
        np.linspace(12 * c, 12 * c + 90 * _NOTCH_RUNWAY, 91)[1:],
        np.linspace(12 * c + 90 * _NOTCH_RUNWAY,
                    24 * c + 90 * _NOTCH_RUNWAY, 13)[1:]])
    prof = np.diff(nodes)
    assert len(set(prof[12:102].tolist())) > 1
    feed = float(nodes[22]) + 0.3 * _NOTCH_RUNWAY
    _, report = _ports_on(prof, c, [("p1", feed, "+x")], 7)
    _assert_floor_on_the_runway(report, ("p1",))


def test_a_ladder_ending_on_the_runways_last_node_stays_on_the_runway():
    """The deepest probe realized exactly on the node where the runway ends.

    On this band mesh the grid's own coordinate for that node lies 1.4e-17 m
    past the profile's cumulative sum. Without the node-touch slack the
    ladder span reached into the ramp cell beyond it, and the reflector
    advisory refused to print an interval for a ladder that never leaves the
    runway. One offset more does put the deepest probe on the ramp, and that
    is still refused.
    """
    prof = _band_runway(4.3e-3, 16.9e-3)
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    i_a = int(np.argmin(np.abs(edges - 4.3e-3)))
    i_b = int(np.argmin(np.abs(edges - 16.9e-3)))
    on_runway = i_b - i_a - 12
    stub = 16.9e-3 + 1.5e-3

    _, report = _ports_on(prof, 2 * _NOTCH_RUNWAY,
                          [("p1", 4.3e-3, "+x")], on_runway, stub_x=stub,
                          freq_max=9e9)
    text = _port_finding(report, "p1", "strong reflector candidate")
    assert "probe ladder crosses cells" not in text, text

    _, report = _ports_on(prof, 2 * _NOTCH_RUNWAY,
                          [("p1", 4.3e-3, "+x")], on_runway + 1, stub_x=stub,
                          freq_max=9e9)
    text = _port_finding(report, "p1", "strong reflector candidate")
    assert "probe ladder crosses cells" in text, text


def _snap_board(offset_in_cell):
    """The notch-like runway with the feed declared part way into the last
    cell of the ramp, the last one that is not 127 um: ``offset_in_cell`` of
    it below the node where the runway starts (negative), or above the node
    where that cell starts (positive)."""
    prof, _, _ = _graded_runway_board()
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    j = max(i for i in range(prof.size // 2)
            if not np.isclose(prof[i], _NOTCH_RUNWAY))
    feed = (float(edges[j + 1]) + offset_in_cell * float(prof[j])
            if offset_in_cell < 0
            else float(edges[j]) + offset_in_cell * float(prof[j]))
    return _ports_on(prof, 2 * _NOTCH_RUNWAY, [("p1", feed, "+x")], 7)[1]


def test_a_feed_declared_off_a_node_is_judged_on_the_node_it_is_stamped_on():
    """The grid stamps the source on the node NEAREST the declared feed.

    Declared 0.3 of a ramp cell below the runway's first node, the feed is
    stamped on that node and its source and probes are all on the runway.
    Declared 0.3 of the same cell above the node before it, it is stamped on
    that earlier node, the first cell it launches into is the ramp's, and
    the refusal stands.
    """
    _assert_floor_on_the_runway(_snap_board(-0.3), ("p1",))
    assert _RAMP_REFUSAL in _port_finding(_snap_board(0.3), "p1",
                                          "fringing decays")


# ---------------------------------------------------------------------------
# An automatic offset on a runway finer than the boundary cell
# ---------------------------------------------------------------------------

def _auto_offset_board():
    """The notch runway (127 um under a 254 um boundary cell), 254 um
    substrate, f_max 20 GHz, and n_probe_offset left to add_msl_port. The
    feed is declared 0.3 of a cell past a runway node."""
    from rfx.geometry.csg import Box
    prof, x_run, _ = _graded_runway_board()
    h, w = _NOTCH_H_SUB, 2 * _NOTCH_H_SUB
    lx, ly = float(prof.sum()), 10 * _NOTCH_H_SUB
    sim = Simulation(
        freq_max=20e9, domain=(lx, ly, h + 1.5e-3), dx=2 * _NOTCH_RUNWAY,
        cpml_layers=8, dx_profile=prof,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, 0), (lx, ly, h)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - w / 2, h), (lx, y_c + w / 2, h)), material="pec")
    feed = x_run + 20.3 * _NOTCH_RUNWAY
    sim.add_msl_port(position=(feed, y_c, 0), width=w, height=h,
                     direction="+x", impedance=50.0)
    return sim


def test_an_automatic_offset_is_not_told_to_stay_automatic():
    """add_msl_port counts the 5·h_sub standoff in the boundary cell, 254 um,
    and stores 5. On this 127 um runway that is 635 um from the source,
    half the 1.270 mm the fringing needs. The advisory used to end with "or
    leave it None for the safe default", which is the setting that chose 5.
    It now names the explicit offset, and the S-parameter routing check
    says the same.

    The distance is read off the grid from the node the source is stamped
    on: 635 um. The declared feed is 38.1 um past that node, and the message
    says so rather than measuring from it.
    """
    sim = _auto_offset_board()
    assert sim._msl_ports[0].n_probe_offset == 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    text = _finding(report, "OWN feed plane")
    assert "the automatic n_probe_offset=5 puts probe 0 635\u00b5m" in text, text
    assert "38.1\u00b5m from the declared feed" in text, text
    assert "Set n_probe_offset >= 10 explicitly on this port" in text, text
    assert "leave it None" not in text, text
    msg = _forward_message(sim)
    assert "set n_probe_offset >= 10 explicitly" in msg, msg
    assert "leave it None" not in msg, msg


def test_an_explicit_offset_on_a_fine_runway_is_not_told_to_go_automatic():
    """The same arithmetic for an explicit offset of 7 on the runway: the
    automatic choice would be 5, fewer cells than the 7 already set, so
    "leave it None" is not a remedy there either. The S-parameter routing
    check builds its remedy in a separate branch, so it is asserted too."""
    sim, _ = _notch_like_sim(14 + 20, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    text = _finding(report, "OWN feed plane")
    assert "Set n_probe_offset >= 10;" in text, text
    assert "leave it None for the safe default" not in text, text
    msg = _forward_message(sim)
    assert "set n_probe_offset >= 10; leaving it None counts" in msg, msg
    assert "leave it None for the safe default" not in msg, msg
