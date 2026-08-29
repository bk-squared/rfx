"""Preflight checks for MSL port geometry correctness.

These guard against the silent setup mistakes that caused multi-session
debugging on 2026-05-04 (lateral box too narrow, trace inside CPML,
substrate under-resolved). Each check:
  - fires a clear warning on bad geometry with a concrete fix message
  - stays silent on a properly-set-up MSL port

See also docs/research_notes/20260504_msl_meshconv_fixed_ly.md and
rfx/api.py:_check_msl_port_geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


# Common geometry constants (RO4350B-class)
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
LX = 14e-3


def _build_sim(*, dx: float, ly: float, port_x: float = 2e-3,
               dz_profile=None) -> Simulation:
    """``dz_profile`` (issue #752 / #766 review): when given, the domain's
    z extent is the profile's sum and the run grid is non-uniform -- the
    substrate checks must then read the substrate off THAT grid."""
    lz = H_SUB + 1.5e-3 if dz_profile is None else float(np.sum(dz_profile))
    kw = {} if dz_profile is None else dict(dz_profile=np.asarray(dz_profile))
    sim = Simulation(
        freq_max=5e9, domain=(LX, ly, lz), dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
        **kw,
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, ly, H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    sim.add(
        Box((0, y_c - W_TRACE / 2, H_SUB),
            (LX, y_c + W_TRACE / 2, H_SUB + dx)),
        material="pec",
    )
    sim.add_msl_port(position=(port_x, y_c, 0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    return sim


def _msl_warnings(sim: Simulation) -> list[str]:
    return [m for m in sim.preflight() if "MSL port" in m]


def test_clearance_warning_fires_on_narrow_ly():
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 6 * 80e-6)
    msgs = _msl_warnings(sim)
    lateral = [m for m in msgs if "lateral clearance" in m]
    assert len(lateral) >= 1, f"expected lateral-clearance warning; got: {msgs}"
    assert "508µm" in lateral[0], lateral[0]
    assert "y-extent" in lateral[0] or "sidewall" in lateral[0], lateral[0]


def test_clearance_silent_on_wide_ly():
    sim = _build_sim(dx=40e-6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    lateral = [m for m in msgs if "lateral clearance" in m]
    assert len(lateral) == 0, (
        f"expected no lateral-clearance warning at LY=W+8·h_sub, got: {lateral}"
    )


def test_substrate_resolution_warning_at_3_cells():
    """dx=100um REALIZES 3 substrate cells (254um rounds up to 300um).

    This used to build at dx=80um and expect "only 3 substrate cell(s)" --
    but at dx=80 the solve has FOUR cells of substrate (320um); "3" was the
    declared-board round(h_sub/dx), the exact confusion #752 is about
    (#766 review). Check 2 now counts the cells the run grid has, so the
    genuine 3-cell case is dx=100um, and the dx=80 case is covered by
    test_substrate_checks_count_realized_cells_at_dx_80 below."""
    sim = _build_sim(dx=100e-6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    sub = [m for m in msgs if "substrate cell" in m]
    assert len(sub) == 1, f"expected 1 substrate-cell warning, got: {sub}"
    assert "only 3 substrate cell(s)" in sub[0], sub[0]
    assert "Refine to dx" in sub[0]


def test_substrate_checks_count_realized_cells_at_dx_80():
    """At dx=80um the substrate REALIZES 4 cells = 320um (fidelity_report
    says the same), so check 2 ("< 4 cells") must be silent and only the
    mixed-cell check may speak -- carrying the realized 320um, read off the
    assembled permittivity rather than n*dx (issue #752 / #766 review)."""
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    sub = [m for m in msgs if "substrate cell(s) in z" in m]
    assert sub == [], f"check 2 must not call a 4-cell board '3 cells': {sub}"
    mixed = [m for m in msgs if "danger zone" in m]
    assert len(mixed) == 1, msgs
    assert "4 cell(s) of substrate = 320µm" in mixed[0], mixed[0]
    assert "+26% THICKER" in mixed[0], mixed[0]


def test_substrate_checks_read_the_run_grid_on_a_dz_profile():
    """The #766 BLOCK, as a regression lock: a dz_profile that resolves the
    254um substrate with four 63.5um cells realizes it EXACTLY
    (fidelity_report: 254.0 -> 254.0um). Scoring the substrate with the
    uniform grid's scalar dx=80um used to assert "only 3 substrate cells"
    and a mixed-cell danger zone at "h_sub/dx = 3.175" on this very
    simulation -- two surfaces of one codebase disagreeing about what the
    solver built. Both checks must be silent here."""
    dz = np.concatenate([np.full(4, H_SUB / 4), np.full(12, 80e-6)])
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB, dz_profile=dz)
    msgs = _msl_warnings(sim)
    noisy = [m for m in msgs if "substrate cell" in m or "danger zone" in m]
    assert noisy == [], f"exact NU substrate must not trip either check: {noisy}"


def test_substrate_checks_report_realized_thickness_on_a_dz_profile():
    """A dz_profile of uniform 80um cells puts the declared top 0.175 of a
    cell above a node: the mixed-cell check must fire, and its realized
    thickness (320um, 4 cells) must come from the run grid's permittivity,
    with the non-uniform remedy (place a node at h_sub), not a dx snap."""
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB,
                     dz_profile=np.full(15, 80e-6))
    msgs = _msl_warnings(sim)
    mixed = [m for m in msgs if "danger zone" in m]
    assert len(mixed) == 1, msgs
    assert "sits 0.175 of a cell above the nearest mesh node" in mixed[0], mixed[0]
    assert "4 cell(s) of substrate = 320µm" in mixed[0], mixed[0]
    assert "place a mesh node exactly at h_sub=254.0µm" in mixed[0], mixed[0]
    assert "set dx =" not in mixed[0], mixed[0]


def test_substrate_resolution_silent_at_6_cells():
    sim = _build_sim(dx=40e-6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    sub = [m for m in msgs if "substrate cell" in m]
    assert len(sub) == 0, f"expected no substrate-cell warning, got: {sub}"


def test_port_close_to_cpml_warning():
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB, port_x=400e-6)
    msgs = _msl_warnings(sim)
    cpml = [m for m in msgs if "x-CPML" in m]
    assert len(cpml) >= 1, f"expected x-CPML clearance warning, got: {msgs}"


def test_well_setup_msl_port_zero_warnings():
    # dx = h_sub / 6 → exactly 6 substrate cells, no mixed-cell at the
    # trace boundary (frac == 0).
    sim = _build_sim(dx=H_SUB / 6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    assert len(msgs) == 0, f"expected zero MSL warnings, got: {msgs}"


def test_mixed_cell_warning_fires_at_dx_80():
    """h_sub/dx = 3.175 (frac 0.175) — substrate-air interface bisects
    a Yee cell holding the trace.  AD-traceable
    ``pec_occupancy_override`` produces unphysical |S21|² > 1 in this
    regime (verified runs #563/#567, 2026-05-08)."""
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    mixed = [m for m in msgs if "mixed-cell danger zone" in m]
    assert len(mixed) >= 1, f"expected mixed-cell warning at dx=80, got: {msgs}"
    assert "pec_occupancy_override" in mixed[0]
    assert "snap" in mixed[0].lower() or "Snap" in mixed[0] or "h_sub/" in mixed[0]


# ---------------------------------------------------------------------------
# Issue #487: the "<5% Z0 bias at 4+ cells" promise (check 2) and the
# mixed-cell danger zone (check 2b) both got a sweep-grounded correction —
# the promise holds only on an ALIGNED mesh; misalignment is 2.56-2.94x
# worse in Z0-bias magnitude than refinement alone predicts. Numbers come
# from the committed scripts/diagnostics/msl_z0_bias_floor_sweep.py
# artifact, NOT a derived per-mesh formula (leg-1 expectation (a) broke at
# the finest aligned point, so no continuous dB advisory was added — see
# that script's docstring and _check_msl_port_geometry's class docstring).
#
# Every numeric constant quoted in both messages gets its own assertion
# below (adversarial-review finding: an unbound constant can drift silently
# — the #494->#502 coverage-hole class).
# ---------------------------------------------------------------------------
def test_substrate_resolution_warning_names_alignment_requirement():
    """Check 2's fix must fire only for an ALIGNED refinement target, and
    must cite the sweep-measured ALIGNED-class deviations (declared-board
    anchor), not just the bare '<5%'.

    Issue #752 (2026-08-27) retired the "+11% (h_sub/dx=4.233)" comparison
    this message used to add: that number was the misaligned dx=60um
    point's declared-board deviation, cited here to claim refining
    without alignment "does not reach" the aligned-class figure -- but
    the misaligned point realizes a DIFFERENT, thicker board (300um vs
    the declared 254um), so the comparison conflated board rasterization
    with extraction quality. This test pins the fix target (:.1f, not
    :.0f -- 63.5um is h_sub/4, not 64um) and the correction text, and
    positively asserts the retired figures no longer appear."""
    sim = _build_sim(dx=100e-6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    sub = [m for m in msgs if "substrate cell" in m]
    assert len(sub) == 1, f"expected 1 substrate-cell warning, got: {sub}"
    assert "an integer (aligned)" in sub[0], sub[0]
    assert "-3.8%" in sub[0], sub[0]
    assert "-1.2%" in sub[0], sub[0]
    assert "+0.7%" in sub[0], sub[0]
    assert "63.5µm" in sub[0], sub[0]  # h_sub/4 = 63.5, not the old "64µm"
    assert "vs the DECLARED-board" in sub[0], sub[0]
    assert "msl_z0_bias_floor_sweep.py" in sub[0], sub[0]
    # This mesh (dx=100) is misaligned (h_sub/dx=2.54): the realized-board
    # disclosure must be present and must state the cell count/height READ
    # OFF THE RUN GRID (3 cells = 300um, +18%), which here coincides with
    # the "only 3 substrate cell(s)" figure above because both now count
    # realized cells (#766 review).
    assert "actually realizes 3 cell(s) = 300µm" in sub[0], sub[0]
    assert "+18%" in sub[0], sub[0]
    assert "read off the assembled permittivity" in sub[0], sub[0]
    # Retired: the old "+11% (h_sub/dx=4.233)" board-mismatched comparison.
    assert "+11%" not in sub[0], sub[0]
    assert "4.233" not in sub[0], sub[0]


def test_substrate_resolution_warning_silent_wording_at_6_cells():
    """No substrate-cell warning (hence no alignment-caveat text) at 6
    TRULY aligned cells. Uses dx=h_sub/6, not the existing suite's
    dx=40um -- that reads 6.35 cells (frac 0.35, itself inside the
    [0.10, 0.40] mixed-cell danger zone), so it is not actually an
    aligned control (adversarial-review finding)."""
    sim = _build_sim(dx=H_SUB / 6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    sub = [m for m in msgs if "an integer (aligned)" in m]
    assert len(sub) == 0, f"expected no alignment-caveat text, got: {sub}"


def test_mixed_cell_warning_names_z0_bias_magnitude():
    """Issue #752 (2026-08-27) CORRECTION: check 2b used to claim Hard PEC
    is exempt from the |S21|² bug but NOT from a larger Z0 bias -- quoting
    "+20.2%/+11.0% misaligned vs -7.9%/-3.8% aligned ... 2.56-2.94x worse".
    Those four percentages are declared-board deviations measured on
    DIFFERENT realized boards (the misaligned points rasterize a 320um/
    300um substrate, not the declared 254um), so the "2.56-2.94x worse"
    framing conflated board mismatch with extractor bias. This test pins
    the retraction (those figures must NOT appear) and the replacement
    board-thickening figures (which ARE real and measured), enumerating
    both sides so a partial revert cannot silently pass."""
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    mixed = [m for m in msgs if "mixed-cell danger zone" in m]
    assert len(mixed) >= 1, f"expected mixed-cell warning at dx=80, got: {msgs}"
    # Retired extractor-bias framing -- must not reappear.
    assert "2.56-2.94x worse" not in mixed[0], mixed[0]
    assert "+20.2%" not in mixed[0], mixed[0]
    assert "-7.9%" not in mixed[0], mixed[0]
    assert "+11.0%" not in mixed[0], mixed[0]
    assert "-3.8%" not in mixed[0], mixed[0]
    # What survives: the (cited, not remeasured) |S21|^2 > 1 override risk,
    # the measured board-thickening figure, and a pointer to the sibling
    # realized-board artifact and its <=0.4% agreement.
    assert "pec_occupancy_override" in mixed[0], mixed[0]
    assert "no subpixel eps assembly" in mixed[0], mixed[0]
    assert "realizes 4 cell(s) of substrate = 320µm" in mixed[0], mixed[0]
    assert "+26%" in mixed[0], mixed[0]
    assert "msl_z0_bias_floor_sweep_realized_anchor.json" in mixed[0], mixed[0]
    assert "within 0.4%" in mixed[0], mixed[0]
    assert "msl_z0_bias_floor_sweep.py" in mixed[0], mixed[0]


def test_mixed_cell_silent_wording_at_clean_alignment():
    """No mixed-cell warning (hence no Z0-bias-magnitude text) at a clean
    alignment -- mirrors test_mixed_cell_silent_at_dx_127_clean_alignment."""
    sim = _build_sim(dx=127e-6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    mixed = [m for m in msgs if "2.6-2.9x worse" in m]
    assert len(mixed) == 0, f"expected no Z0-bias-magnitude text, got: {mixed}"


def test_mixed_cell_silent_at_dx_127_clean_alignment():
    """h_sub/dx = 2.000 exactly — substrate boundary at a cell face,
    no mixed cell."""
    sim = _build_sim(dx=127e-6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    mixed = [m for m in msgs if "mixed-cell danger zone" in m]
    assert len(mixed) == 0, (
        f"expected no mixed-cell warning at dx=127 (clean alignment), got: {mixed}"
    )


def test_mixed_cell_silent_at_dx_70_above_danger():
    """h_sub/dx = 3.629 (frac 0.629) — outside [0.10, 0.40] danger zone."""
    sim = _build_sim(dx=70e-6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    mixed = [m for m in msgs if "mixed-cell danger zone" in m]
    assert len(mixed) == 0, (
        f"expected no mixed-cell warning at dx=70 (frac 0.629), got: {mixed}"
    )


def test_strict_mode_raises_on_bad_geometry():
    """preflight(strict=True) must raise instead of warning. Strict raises
    on the first issue encountered — for the narrow-LY geometry that may be
    the trace-thickness, lateral-clearance, or substrate-cell warning. We
    just check that strict mode does raise (vs returning warnings list)."""
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 6 * 80e-6)
    with pytest.raises(ValueError):
        sim.preflight(strict=True)


# ---------------------------------------------------------------------------
# Reflector clearance check (Y2 finding 2026-05-06): the 3-probe Z₀
# extractor in compute_msl_s_matrix sits in a standing-wave region when
# a strong reflector (open λ/4 stub etc.) is too close to the V₃ probe.
# See docs/research_notes/20260506_y2_s11_notch_bias_root_cause.md.
# ---------------------------------------------------------------------------
def _build_sim_with_stub(*, dx: float, l_line_mm: float, l_stub_mm: float = 8.637,
                         freq_max: float = 9e9) -> Simulation:
    """Two-MSL-port through-line + open PEC stub branched at LX/2."""
    L_LINE = l_line_mm * 1e-3
    L_STUB = l_stub_mm * 1e-3
    PORT_MARGIN = 1e-3
    LX = L_LINE + 2 * PORT_MARGIN
    L_STUB_MAX = max(14e-3, L_STUB + 2e-3)
    LY = W_TRACE + 2 * (2 * H_SUB + 8 * dx) + L_STUB_MAX + 2 * (2 * H_SUB + 8 * dx)
    LZ = H_SUB + 1.5e-3

    sim = Simulation(
        freq_max=freq_max, domain=(LX, LY, LZ), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = (2 * H_SUB + 8 * dx) + W_TRACE / 2
    trace_y_lo = y_trace - W_TRACE / 2
    trace_y_hi = y_trace + W_TRACE / 2
    sim.add(Box((0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + dx)),
            material="pec")
    stub_xc = LX / 2
    sim.add(Box((stub_xc - W_TRACE / 2, trace_y_hi, H_SUB),
                (stub_xc + W_TRACE / 2, trace_y_hi + L_STUB, H_SUB + dx)),
            material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0),
                     width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0)
    return sim


def test_reflector_clearance_warning_fires_on_short_l_line():
    """L_LINE=9mm with stub at LX/2 — V₃ sits ~0.6–0.9mm from the stub
    PEC reflector, well under λ_g/4 ≈ 3.7mm at f_max=9GHz with
    ε_eff_proxy=5.  Expect the reflector-clearance warning to
    fire on BOTH ports (the stub is between them).

    NOTE (issue #80 Fix B): the L_LINE was 5mm prior to the
    wavelength-bound probe-placement defaults. Fix B grew the default
    3-probe span from ~0.9mm to ~3.6mm (offset 17 + 2·spacing 14 cells
    at dx=80µm, eps_r_sub≈3.66, f_max=9GHz), so at L_LINE=5mm V₃
    overshot the LX/2 stub entirely and the warning no longer fired.
    L_LINE=9mm keeps V₃ before the stub yet within λ_g/4, restoring the
    intended scenario. The λ_g/4 threshold and the fire-on-both-ports
    assertion are unchanged — only the geometry is re-tuned to the new
    defaults."""
    sim = _build_sim_with_stub(dx=80e-6, l_line_mm=9.0)
    msgs = _msl_warnings(sim)
    refl = [m for m in msgs if "reflector" in m]
    assert len(refl) == 2, (
        f"expected reflector warnings on BOTH ports at L_LINE=5mm, got: {refl}"
    )
    assert "λ_g/4" in refl[0]
    assert "L_LINE" in refl[0] or "n_probe_offset" in refl[0]


def test_reflector_clearance_silent_on_long_l_line():
    """L_LINE=30mm (cv06b geometry) with the same stub — V₃ now sits
    ~13mm from the stub, well above λ_g/4 ≈ 3.7mm.  No reflector
    warning should fire."""
    sim = _build_sim_with_stub(dx=80e-6, l_line_mm=30.0)
    msgs = _msl_warnings(sim)
    refl = [m for m in msgs if "reflector" in m]
    assert len(refl) == 0, (
        f"expected no reflector warning at L_LINE=30mm, got: {refl}"
    )


def test_reflector_clearance_silent_without_reflector():
    """Pure thru-line (no stub) — even short L_LINE should not warn,
    because the only PEC Box that intersects the line region is the
    through-trace itself, which the heuristic excludes."""
    L_LINE = 5e-3
    PORT_MARGIN = 1e-3
    dx = 80e-6
    LX = L_LINE + 2 * PORT_MARGIN
    LY = W_TRACE + 2 * (2 * H_SUB + 8 * dx) + 2e-3
    LZ = H_SUB + 1.5e-3

    sim = Simulation(
        freq_max=9e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = (2 * H_SUB + 8 * dx) + W_TRACE / 2
    sim.add(Box((0, y_trace - W_TRACE / 2, H_SUB),
                (LX, y_trace + W_TRACE / 2, H_SUB + dx)),
            material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, y_trace, 0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, y_trace, 0),
                     width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0)
    msgs = _msl_warnings(sim)
    refl = [m for m in msgs if "reflector" in m]
    assert len(refl) == 0, (
        f"expected no reflector warning on thru-only short line, got: {refl}"
    )


# ---------------------------------------------------------------------------
# Issue #510: checks 1/3 above measure clearance at x_feed only, so a probe
# SPAN could leave x_feed clear while the deepest probe (x_deep) lands
# inside/near the absorber (4a, new below) or past a second port's own feed
# plane (4b, new below) — check 4's reflector scan only sees PEC Box
# geometry, not the absorber or a source discontinuity. Reproduction
# geometry is the committed #488 diagnostic dump the issue cites: dx=80um,
# domain_x=11.36mm (= 142 cells), CPML 8 layers, msl_0 at x=2.40mm '+x',
# msl_1 at x=6.40mm '-x', n_probe_offset=31, n_probe_spacing=12, n_probes=5.
# On that geometry msl_0's probes land at 4.88/5.84/6.80/7.76/8.72mm and
# msl_1's at 3.92/2.96/2.00/1.04/0.08mm (issue body) -- msl_1's deepest
# probe (0.08mm) is within the 2-cell/0.16mm proximity margin of the x=0
# domain edge (NOT literally "inside the CPML" under the #500/#542
# exterior-padding frame this fix routes through -- the issue's own
# "x < 0.64mm" framing used the pre-#500 interior intuition), and both
# ports' probe spans cross the OTHER port's feed plane (msl_0's span
# [2.40, 8.72]mm contains msl_1's 6.40mm feed; msl_1's span [0.08, 6.40]mm
# contains msl_0's 2.40mm feed).
# ---------------------------------------------------------------------------
def _two_port_sim(*, lx: float, msl1_x: float,
                  msl0_offset: int = 31, msl1_offset: int = 31,
                  n_probe_spacing: int = 12, n_probes: int = 5) -> Simulation:
    dx = 80e-6
    ly = W_TRACE + 8 * H_SUB
    sim = Simulation(
        freq_max=10e9, domain=(lx, ly, H_SUB + 1.5e-3), dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    sim.add(
        Box((0, y_c - W_TRACE / 2, H_SUB), (lx, y_c + W_TRACE / 2, H_SUB + dx)),
        material="pec",
    )
    sim.add_msl_port(position=(2.40e-3, y_c, 0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, n_probe_offset=msl0_offset,
                     n_probe_spacing=n_probe_spacing, n_probes=n_probes,
                     name="msl_0")
    sim.add_msl_port(position=(msl1_x, y_c, 0), width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0, n_probe_offset=msl1_offset,
                     n_probe_spacing=n_probe_spacing, n_probes=n_probes,
                     name="msl_1")
    return sim


def _absorber_span_msgs(msgs: list[str]) -> list[str]:
    return [
        m for m in msgs
        if "CPML absorbing region" in m or "CPML absorber is active" in m
    ]


def _crossing_msgs(msgs: list[str]) -> list[str]:
    return [m for m in msgs if "feed plane" in m]


def test_issue510_reproduction_fires_both_new_warnings():
    """The exact #488-dump geometry from the issue body: both new checks
    must fire, naming the specific ports/probes involved."""
    sim = _two_port_sim(lx=11.36e-3, msl1_x=6.40e-3)
    msgs = _msl_warnings(sim)

    absorber = _absorber_span_msgs(msgs)
    assert absorber, f"expected an absorber-span advisory; got: {msgs}"
    # Only msl_1's deepest probe (x=0.08mm) is within the proximity
    # margin of the x=0 edge; msl_0's (x=8.72mm) is 2.64mm from the x-hi
    # edge, far outside the 0.16mm margin.
    assert any("msl_1" in m for m in absorber), absorber
    assert not any("msl_0" in m for m in absorber), absorber
    assert any("probe 4" in m and "0.08mm" in m for m in absorber), absorber

    crossing = _crossing_msgs(msgs)
    assert len(crossing) == 2, (
        f"expected a feed-crossing advisory on BOTH ports, got: {crossing}"
    )
    assert any("msl_0" in m and "msl_1" in m and "6.40mm" in m for m in crossing), crossing
    assert any("msl_1" in m and "msl_0" in m and "2.40mm" in m for m in crossing), crossing


def test_issue510_clean_geometry_neither_new_warning_fires():
    """Same probe-array parameters, but ports far enough apart that
    neither the absorber-span nor the feed-crossing check has anything to
    catch -- the half of the coverage that a warning-only test cannot
    prove (a check that always fires is not a check)."""
    sim = _two_port_sim(lx=20e-3, msl1_x=16.00e-3)
    msgs = _msl_warnings(sim)
    assert not _absorber_span_msgs(msgs), (
        f"expected no absorber-span advisory on clean geometry; got: {msgs}"
    )
    assert not _crossing_msgs(msgs), (
        f"expected no feed-crossing advisory on clean geometry; got: {msgs}"
    )


def test_issue510_absorber_span_falsifier_compliant_offset_silences_it():
    """Falsifier for check 4a: the reproduction warning itself reports a
    compliant n_probe_offset interval ([16, 30] cells on this fixture);
    moving msl_1 inside it (offset=20) must silence the advisory."""
    sim = _two_port_sim(lx=11.36e-3, msl1_x=6.40e-3, msl1_offset=20)
    msgs = _msl_warnings(sim)
    assert not _absorber_span_msgs(msgs), (
        f"expected no absorber-span advisory at compliant offset=20; got: {msgs}"
    )


def test_issue510_feed_crossing_falsifier_separated_ports_silences_it():
    """Falsifier for check 4b: moving msl_1 far enough from msl_0 (same
    probe-array parameters) that neither probe span reaches the other
    port's feed must silence the crossing advisory."""
    sim = _two_port_sim(lx=11.36e-3, msl1_x=9.60e-3)
    msgs = _msl_warnings(sim)
    assert not _crossing_msgs(msgs), (
        f"expected no feed-crossing advisory once ports are separated; got: {msgs}"
    )


# ---------------------------------------------------------------------------
# Issue #510 review round (PR #551 adversarial review, 3 BLOCKING findings):
#
# BLOCKING 1: the advertised compliant n_probe_offset interval's upper
# endpoint was computed by ALGEBRAIC INVERSION of the two absorber
# predicates (float division then int() truncation). Float64 arithmetic put
# the endpoint on the wrong side of an FP knife edge whenever the true
# boundary landed within about one ULP of an exact dx multiple -- reviewer's
# brute-force sweep found ~12k (dx, spacing, feed, domain) combinations
# where the ADVERTISED endpoint itself still tripped the very warning it
# claimed to clear. Fixed by msl_absorber_compliant_offset_max: walk
# candidate offsets DOWN from a deliberately generous starting guess and
# test the REAL predicate (via the extractor's own msl_probe_x_coords_n) at
# each one, so the result is verified compliant by construction.
#
# BLOCKING 2: x_deep was a continuous-coordinate extrapolation
# (x_feed + offset*dx), but the extractor places probes by GRID INDEX with
# rounding and clamping (rfx.sources.msl_port._msl_x_for_index /
# msl_probe_x_coords_n). Consequences: (a) up to dx/2 model error for a
# feed that is not itself grid-aligned -- the same order as the 2-cell
# absorber-proximity decision margin; (b) when the offset+spacing ladder
# runs past the grid, several probes CLAMP onto the same cell, and the
# pre-fix overlap message named a coordinate the real extractor never
# visits. Fixed by routing x_deep through the same msl_probe_x_coords_n
# call the extractor uses, and adding a distinct degenerate-ladder
# advisory when probes collapse.
#
# BLOCKING 3: the 4b lumped/wire else-branch label was untested and, when
# exercised, read "...port at x=6.40mm (component='ez')'s feed plane at
# x=6.40mm" -- the coordinate stated twice, and a possessive glued onto a
# parenthetical. Reworded to state the crossing coordinate exactly once
# with no possessive.
# ---------------------------------------------------------------------------
def _degenerate_ladder_msgs(msgs: list[str]) -> list[str]:
    return [m for m in msgs if "runs past the grid and CLAMPS" in m]


def test_issue510_absorber_offset_interval_endpoint_does_not_warn():
    """BLOCKING 1: in THIS repo's own reproduction fixture the buggy
    algebra (before the fix) evaluated ``(6.40e-3 - 0.16e-3) / 80e-6`` to
    ``77.99999999999999`` in float64 -- one below the mathematically exact
    78 -- and advertised ``[16, 29]`` where the walk-down-verified boundary
    is actually 30 (x_deep=0.16mm sits EXACTLY at the proximity margin,
    which the strict-less-than predicate treats as compliant). This
    fixture's bug happened to under-report (safe direction); extract
    whatever the CURRENTLY-advertised endpoint is from the live warning
    (not a hardcoded expectation) and confirm it, used as n_probe_offset,
    draws no absorber-span advisory -- the general property BLOCKING 1
    asked for, on real Simulation-level output."""
    import re

    sim = _two_port_sim(lx=11.36e-3, msl1_x=6.40e-3)
    msgs = _absorber_span_msgs(_msl_warnings(sim))
    assert msgs, "expected the absorber-span advisory to fire"
    m = re.search(r"interval ≈ \[(\d+), (\d+)\] cells", msgs[0])
    assert m, f"expected a compliant-interval clause in: {msgs[0]}"
    off_lo, off_hi = int(m.group(1)), int(m.group(2))
    assert off_lo == 16, f"expected the unchanged lower bound 16; got {off_lo}"
    assert off_hi == 30, (
        f"expected the walk-down-verified endpoint 30 (pre-fix buggy "
        f"algebra advertised 29 on this fixture); got {off_hi} in "
        f"{msgs[0]!r}"
    )

    sim2 = _two_port_sim(lx=11.36e-3, msl1_x=6.40e-3, msl1_offset=off_hi)
    msgs2 = _absorber_span_msgs(_msl_warnings(sim2))
    assert not msgs2, (
        f"advertised endpoint n_probe_offset={off_hi} must NOT itself "
        f"trip the absorber-span advisory; got: {msgs2}"
    )


def test_issue510_absorber_offset_max_endpoint_verified_across_geometries():
    """BLOCKING 1 continued: the reviewer's sweep found the FP-knife-edge
    failure in BOTH directions -- this repo's own fixture happened to
    under-report (safe), but other geometries over-reported (advertised an
    offset that still trips). Sweep a grid of small integer-cell
    geometries directly against msl_absorber_compliant_offset_max and
    verify the returned endpoint, plugged into the SAME
    msl_probe_x_coords_n arithmetic the real extractor uses, never trips
    either absorber predicate. The walk-down cannot fail this by
    construction -- this is a construction check, not a search for a
    counterexample."""
    import math

    from rfx.api._preflight import (
        _coord_in_absorber,
        _coord_near_absorber,
        msl_absorber_compliant_offset_max,
    )
    from rfx.grid import Grid
    from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

    dx = 80e-6
    n_pr = 5
    ct = 2 * dx
    n_checked = 0
    n_found_endpoint = 0
    for n_sp in (2, 3, 12):
        for feed_cells in range(4, 16):
            for domain_cells in range(feed_cells + 40, feed_cells + 70, 3):
                domain_x = domain_cells * dx
                feed_x = feed_cells * dx
                headroom = domain_x - feed_x
                guess_hi = int(math.ceil(headroom / dx)) - (n_pr - 1) * n_sp + 4
                grid = Grid(freq_max=10e9, domain=(domain_x, 4e-3, 2e-3),
                           dx=dx, cpml_layers=2)
                port = MSLPort(feed_x=feed_x, y_lo=1e-3, y_hi=2e-3,
                               z_lo=0.0, z_hi=1e-3, direction="+x",
                               impedance=50.0)
                off_max = msl_absorber_compliant_offset_max(
                    grid, port, n_probes=n_pr, n_spacing=n_sp, off_lo=3,
                    domain_x=domain_x, ct_lo=ct, ct_hi=ct, dx=dx,
                    guess_hi=guess_hi,
                )
                n_checked += 1
                if off_max is None:
                    continue
                n_found_endpoint += 1
                ladder = msl_probe_x_coords_n(
                    grid, port, n_probes=n_pr,
                    n_offset_cells=off_max, n_spacing_cells=n_sp,
                )
                x_deep = ladder[-1]
                assert not _coord_in_absorber(x_deep, domain_x, ct, ct), (
                    n_sp, feed_cells, domain_cells, off_max, x_deep,
                )
                assert not _coord_near_absorber(x_deep, domain_x, ct, ct, dx), (
                    n_sp, feed_cells, domain_cells, off_max, x_deep,
                )
    assert n_checked >= 300, n_checked
    assert n_found_endpoint > 0, "sweep never found a compliant endpoint to verify"


def test_issue510_absorber_span_names_real_snapped_coordinate_off_grid_feed():
    """BLOCKING 2(a): offset msl_1's feed by half a cell off-grid. The
    extractor's own probe placement (msl_probe_x_coords_n) rounds the feed
    to the nearest grid node BEFORE walking the offset ladder, so the real
    deepest-probe coordinate can differ from the naive continuous formula
    ``x_feed + offset*dx`` (which never snaps the feed at all). Confirm the
    emitted message names the coordinate msl_probe_x_coords_n actually
    returns, not the continuous extrapolation."""
    from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

    off_grid_x = 6.40e-3 + 40e-6  # half a cell off-grid at dx=80um
    sim = _two_port_sim(lx=11.36e-3, msl1_x=off_grid_x)
    msgs = _absorber_span_msgs(_msl_warnings(sim))
    assert msgs, f"expected the advisory to still fire; got: {_msl_warnings(sim)}"

    # Ground truth: the SAME production call the check itself makes.
    grid = sim._build_grid()
    y_c = (W_TRACE + 8 * H_SUB) / 2.0
    port = MSLPort(feed_x=off_grid_x, y_lo=y_c - W_TRACE / 2, y_hi=y_c + W_TRACE / 2,
                    z_lo=0.0, z_hi=H_SUB, direction="-x", impedance=50.0)
    ladder = msl_probe_x_coords_n(grid, port, n_probes=5,
                                  n_offset_cells=31, n_spacing_cells=12)
    x_deep_real = ladder[-1]
    x_deep_continuous = off_grid_x - (31 + 4 * 12) * 80e-6
    assert abs(x_deep_real - x_deep_continuous) > 1e-9, (
        "fixture did not actually exercise an off-grid feed discrepancy"
    )

    hit = msgs[0]
    assert f"x={x_deep_real * 1e3:.2f}mm" in hit, (
        f"expected the real grid-snapped coordinate "
        f"{x_deep_real * 1e3:.2f}mm in the message; got: {hit}"
    )
    assert f"x={x_deep_continuous * 1e3:.2f}mm" not in hit, (
        f"message must not name the continuous-extrapolation coordinate "
        f"{x_deep_continuous * 1e3:.2f}mm; got: {hit}"
    )


def test_issue510_degenerate_ladder_warns_on_clamped_probes():
    """BLOCKING 2(b): a probe-array ladder that runs past the grid edge
    CLAMPS -- several probes land on the SAME cell instead of spreading
    out, making the N-probe least-squares fit rank-deficient. This must
    draw its own distinct advisory (not just the absorber-overlap message,
    which fires separately on the same honest, clamped x_deep)."""
    sim = _two_port_sim(lx=11.36e-3, msl1_x=1.00e-3,
                        msl0_offset=31, msl1_offset=18, n_probe_spacing=12)
    msgs = _msl_warnings(sim)
    degenerate = _degenerate_ladder_msgs(msgs)
    assert degenerate, f"expected a degenerate-ladder advisory; got: {msgs}"
    hit = degenerate[0]
    assert "msl_1" in hit, hit
    assert "3 duplicate probe position" in hit, hit
    assert "2 of 5 probes land on distinct grid cells" in hit, hit


def test_issue510_feed_crossing_names_lumped_port_cleanly():
    """BLOCKING 3: drives an MSL probe span across a LUMPED port's feed
    (the 4b else-branch, previously untested) and asserts the cleaned-up
    wording -- the crossing coordinate stated exactly once, and no
    possessive glued onto the parenthetical component tag."""
    dx = 80e-6
    lx = 11.36e-3
    ly = W_TRACE + 8 * H_SUB
    sim = Simulation(
        freq_max=10e9, domain=(lx, ly, H_SUB + 1.5e-3), dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (lx, ly, H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    sim.add(
        Box((0, y_c - W_TRACE / 2, H_SUB), (lx, y_c + W_TRACE / 2, H_SUB + dx)),
        material="pec",
    )
    sim.add_msl_port(position=(2.40e-3, y_c, 0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, n_probe_offset=31,
                     n_probe_spacing=12, n_probes=5, name="msl_0")
    sim.add_port(position=(6.40e-3, y_c, H_SUB), component="ez", excite=False)

    msgs = _crossing_msgs(_msl_warnings(sim))
    assert msgs, f"expected a feed-crossing advisory naming the lumped port; got: {_msl_warnings(sim)}"
    hit = msgs[0]
    assert "lumped/wire port" in hit, hit
    assert "component='ez'" in hit, hit
    assert hit.count("6.40mm") == 1, (
        f"expected the crossing coordinate stated exactly once; got: {hit}"
    )
    assert "')'s" not in hit and ")'s" not in hit, (
        f"expected no possessive glued onto the parenthetical; got: {hit}"
    )
