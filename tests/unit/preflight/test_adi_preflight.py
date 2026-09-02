"""Preflight advisory for the 3D ADI large-timestep accuracy envelope.

HISTORY (W3.8 / OPT-C1): until 2026-07-13 ``adi_step_3d`` was an LOD split
with artificial diffusion and the ``adi_3d_accuracy`` warning flagged the
whole 3D path as KNOWN-INACCURATE unconditionally. The scheme is now the
full Zheng–Chen–Zhang two-sub-step 3D ADI (issue #338 follow-up); the
adjudication test
``test_review_tier1_validation_battery.py::test_optc1_adi_3d_cavity_eigenfrequency``
passes its 2% gate at 2x CFL (measured 1.2%) and its former strict-xfail
marker is removed.

What this file now locks is the ENVELOPE advisory that replaced the old
blanket warning: dispersion error of the implicit scheme grows ~dt^2, so at
~15 cells/wavelength the <2% eigenfrequency envelope holds only up to ~2x
CFL. The ``adi_3d_accuracy`` warning must fire on ``solver='adi'`` + a 3D
grid when ``adi_cfl_factor > 2`` (note the constructor DEFAULTS are
``mode='3d'`` and ``adi_cfl_factor=5.0``, so a bare ``Simulation(
solver='adi')`` is advised), and stay silent at ``adi_cfl_factor <= 2``, on
the validated 2D TMz path, and on the explicit Yee solver.
"""

from __future__ import annotations

from rfx import Simulation


def _codes(issues):
    return {getattr(i, "code", "") for i in issues}


def test_adi_3d_large_cfl_fires_envelope_advisory():
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="adi", adi_cfl_factor=5.0,
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" in _codes(issues), (
        f"3D ADI at 5x CFL must get the dt^2-envelope advisory; "
        f"issues: {issues!r}"
    )
    severities = {
        getattr(i, "code", ""): getattr(i, "severity", "")
        for i in issues
    }
    # WARNING severity, not error — large-dt 3D ADI is a legitimate stiff-
    # mesh tool; only wavelength-scale accuracy degrades.
    assert severities["adi_3d_accuracy"] == "warning"


def test_adi_default_cfl_factor_fires_envelope_advisory():
    """Defaults are mode='3d' AND adi_cfl_factor=5.0 -> must still advise."""
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", solver="adi",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" in _codes(issues)


def test_adi_3d_within_envelope_is_silent():
    """At adi_cfl_factor <= 2 the 3D scheme meets its 2% gate — no advisory
    (measured 1.2% on the tier-1 adjudication cavity at 2x CFL)."""
    sim = Simulation(
        freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=5e-3,
        boundary="pec", mode="3d", solver="adi", adi_cfl_factor=2.0,
    )
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    issues = sim.preflight()
    assert "adi_3d_accuracy" not in _codes(issues), (
        f"3D ADI at <=2x CFL is inside the validated envelope and must NOT "
        f"warn; issues: {issues!r}"
    )


def test_adi_2d_tmz_is_silent():
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, 0.01),
        boundary="pec", mode="2d_tmz", solver="adi", adi_cfl_factor=5.0,
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
        f"explicit Yee solver must NOT get the ADI advisory; issues: {issues!r}"
    )
