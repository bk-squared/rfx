"""Stage 1 physical evidence gate for non-uniform mesh intelligence."""

from __future__ import annotations

from scripts.stage1_nu_cavity_physics_gate import run_gate


def test_stage1_nu_cavity_physics_gate_passes():
    gate = run_gate()
    assert gate.preflight_issues == ()
    assert gate.cell_savings_factor >= 40.0
    # Resolution-honest envelope: harminv (not rfft-argmax) measures ~2.54% real
    # NU-air-cavity discretization error here; the gate binds at 3.5% (issue #396,
    # matching run_gate's own assertion). The old 1% gate was FFT-bin luck.
    assert gate.resonance_error_pct <= 3.5
    assert gate.segmented_ad_gb < gate.full_ad_gb
