"""Regression coverage for NTFF clearance from lateral PEC-sheet edges.

These tests exercise only the public ``Simulation`` construction and preflight
API.  They deliberately keep the source farther from the relevant NTFF face
than the PEC sheet edge so a warning cannot be attributed to the source.
"""

from __future__ import annotations

from rfx import Box, Simulation


FREQ = 10e9
DOMAIN = (0.120, 0.120, 0.120)
SHEET_LO = (0.040, 0.040, 0.050)
SHEET_HI = (0.080, 0.080, 0.052)
SOURCE = (0.060, 0.060, 0.056)


def _preflight_with_ntff(corner_lo, corner_hi):
    sim = Simulation(
        freq_max=FREQ,
        domain=DOMAIN,
        dx=2e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add(Box(SHEET_LO, SHEET_HI), material="pec")
    sim.add_source(SOURCE, "ez")
    sim.add_ntff_box(corner_lo, corner_hi, freqs=(FREQ,))
    return sim.preflight()


def _strong_ntff_clearance_issues(report):
    return [
        issue
        for issue in report
        if getattr(issue, "code", None) == "ntff_near_field"
        and "below λ/4" in issue
    ]


def test_ntff_lateral_face_warns_near_pec_sheet_edge():
    """x_hi is 4 mm from the sheet edge but 24 mm from the source."""
    report = _preflight_with_ntff(
        corner_lo=(0.020, 0.020, 0.025),
        corner_hi=(0.084, 0.100, 0.085),
    )

    issues = _strong_ntff_clearance_issues(report)
    assert any("NTFF face x_hi" in issue and "geometry 'pec'" in issue
               for issue in issues), (
        "expected x_hi clearance warning anchored on the lateral PEC edge; "
        f"got: {issues!r}"
    )


def test_ntff_box_clear_of_every_pec_edge_has_no_clearance_warning():
    """Every box face is at least 20 mm (> λ/2) from the PEC sheet."""
    report = _preflight_with_ntff(
        corner_lo=(0.020, 0.020, 0.025),
        corner_hi=(0.100, 0.100, 0.085),
    )

    issues = [
        issue
        for issue in report
        if getattr(issue, "code", None) == "ntff_near_field"
    ]
    assert issues == [], f"expected no NTFF clearance warning; got: {issues!r}"


def test_ntff_z_face_warns_near_pec_sheet_broad_side():
    """Control: z_lo is 4 mm below the sheet's broad lower side."""
    report = _preflight_with_ntff(
        corner_lo=(0.020, 0.020, 0.046),
        corner_hi=(0.100, 0.100, 0.085),
    )

    issues = _strong_ntff_clearance_issues(report)
    assert any("NTFF face z_lo" in issue and "geometry 'pec'" in issue
               for issue in issues), (
        "expected z_lo clearance warning anchored on the PEC broad side; "
        f"got: {issues!r}"
    )
