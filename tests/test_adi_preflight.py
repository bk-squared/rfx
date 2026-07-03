"""Preflight must surface the KNOWN-INACCURATE 3D ADI (LOD) path (W3.8).

``adi_step_3d`` applies the implicit tridiagonal solve to E components whose
curl has no derivative along the sweep axis — artificial diffusion (OPT-C1):
a lossless 3D PEC cubic cavity misses the analytic eigenfrequency by ~41%
even at 2x CFL. The strict-xfail adjudication test lives in
``test_review_tier1_validation_battery.py::test_optc1_adi_3d_cavity_eigenfrequency``
(it XPASSes when the scheme is fixed, forcing removal of this warning).

This file locks the user-facing preflight warning (code ``adi_3d_accuracy``):
it must fire on ``solver='adi'`` + a 3D grid (note ``mode='3d'`` is the
constructor DEFAULT, so ``Simulation(solver='adi')`` lands on the inaccurate
path), and stay silent on the validated 2D TMz path and on the explicit Yee
solver.
"""

from __future__ import annotations

from rfx import Simulation


def _codes(issues):
    return {getattr(i, "code", "") for i in issues}


def test_adi_3d_fires_accuracy_warning():
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="adi",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" in _codes(issues), (
        f"3D ADI must warn (OPT-C1 known inaccuracy); issues: {issues!r}"
    )
    severities = {
        getattr(i, "code", ""): getattr(i, "severity", "")
        for i in issues
    }
    # WARNING severity, not error — 3D ADI must stay runnable for the
    # stability/divergence tests that characterise it.
    assert severities["adi_3d_accuracy"] == "warning"


def test_adi_default_mode_fires_accuracy_warning():
    """mode is omitted → constructor default '3d' → must still warn."""
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", solver="adi",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" in _codes(issues)


def test_adi_2d_tmz_is_silent():
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.01),
        boundary="pec", mode="2d_tmz", solver="adi",
    )
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.012, 0.01, 0.0), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" not in _codes(issues), (
        f"validated 2D TMz ADI path must NOT warn; issues: {issues!r}"
    )


def test_yee_3d_is_silent():
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="yee",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" not in _codes(issues), (
        f"explicit Yee solver must NOT get the ADI warning; issues: {issues!r}"
    )
