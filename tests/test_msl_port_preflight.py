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
    sim = _build_sim(dx=40e-6, ly=W_TRACE + 8 * H_SUB, port_x=2e-3)
    msgs = _msl_warnings(sim)
    assert len(msgs) == 0, f"expected zero MSL warnings, got: {msgs}"


def test_strict_mode_raises_on_bad_geometry():
    """preflight(strict=True) must raise instead of warning. Strict raises
    on the first issue encountered — for the narrow-LY geometry that may be
    the trace-thickness, lateral-clearance, or substrate-cell warning. We
    just check that strict mode does raise (vs returning warnings list)."""
    sim = _build_sim(dx=80e-6, ly=W_TRACE + 6 * 80e-6)
    with pytest.raises(ValueError):
        sim.preflight(strict=True)
