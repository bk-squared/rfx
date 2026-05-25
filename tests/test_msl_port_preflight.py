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

import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


# Common geometry constants (RO4350B-class)
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
LX = 14e-3


def _build_sim(*, dx: float, ly: float, port_x: float = 2e-3) -> Simulation:
    sim = Simulation(
        freq_max=5e9, domain=(LX, ly, H_SUB + 1.5e-3), dx=dx,
        cpml_layers=8,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
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
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 8 * H_SUB)
    msgs = _msl_warnings(sim)
    sub = [m for m in msgs if "substrate cell" in m]
    assert len(sub) == 1, f"expected 1 substrate-cell warning, got: {sub}"
    assert "Refine to dx" in sub[0]


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
