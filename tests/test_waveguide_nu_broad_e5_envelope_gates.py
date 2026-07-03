"""Committed gate for the WR-90 nonuniform graded-dy FLUX broad-E5 envelope.

Mirrors ``tests/test_waveguide_broad_e5_envelope_gates.py`` for the NONUNIFORM
(graded transverse ``dy``) waveguide-port flux lane. It closes the
"uncommitted ``.omx``" defect for the NU broad-E5-analytic evidence: the envelope
JSON (``tests/fixtures/waveguide_nu_broad_e5/``, regenerated on gpu-rtx4090,
VESSL 369367244527 at commit ff9bfcb, 16/16 cases <= 0.0157 vs analytic Airy) is
replayed on a CLEAN CHECKOUT, so the claim no longer rides on a gitignored
artifact (the coaxial-overclaim hole this lane was at risk of).

SCOPE / honesty -- this gate is **broad-E5-analytic (analytic-oracle, no external
solver / no AD), NOT a claims-bearing uniform-class PASS**. It locks the NU flux S-matrix against the independent analytic Airy
reference across the graded-TRANSVERSE ``dy`` mesh axis (grading ratio 1-3,
adjacent-cell ratio up to 3:1), eps_r {2, 4} (incl. the strong eps_r=4 reflector
the ``normalize=True`` path floors at ~0.077), over X-band single-mode TE10,
forward-only, single-slab. It deliberately does NOT establish (a) a broad-E4
external-solver (Meep/OpenEMS) cross-check, nor (b) AD-traceability of the NU
S-matrix. Both are now CLOSED as separate committed evidence: the external E4
cross-check lives in ``tests/fixtures/waveguide_nu_broad_e4/`` (rfx-NU-flux vs
Palace FEM across empty/PEC-short/slab, gated by
``test_waveguide_nu_broad_e4_comparison_gates.py``), and NU AD-traceability
landed in #233. Promotion of the NU lane from shadow to claims-bearing is a
separate decision; this envelope remains the analytic-oracle leg.

Layer 2 drives the producer's own ``airy()`` formula + ``MAX_TOL`` with synthetic
ideal / perturbed S-parameters to lock the gate semantics. Pure-Python (no FDTD);
both layers replay frozen numbers, so a regression in the live
``compute_waveguide_s_matrix`` does not flip them -- that live anchor is the
np=40 power/reciprocity gate in ``tests/test_waveguide_nu_nontrivial.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_wr90_nu_flux_broad_e5_envelope import (  # type: ignore  # noqa: E402
    MAX_TOL,
    airy,
)

FIXTURE = (
    REPO / "tests" / "fixtures" / "waveguide_nu_broad_e5"
    / "waveguide_wr90_nu_flux_broad_e5_envelope.json"
)

MIN_GRADING_RATIOS = 2   # mesh-refinement axis (NU uses grading ratio, not dx)
MIN_EPS_R = 2            # geometry axis
MIN_CASES = 4
_A_WG = 22.86e-3
_FC_TE10 = 299_792_458.0 / (2 * _A_WG)


def test_nu_flux_envelope_present() -> None:
    assert FIXTURE.exists(), f"missing committed NU flux envelope: {FIXTURE}"


def test_committed_nu_flux_envelope_passes_broad_e5() -> None:
    """Re-derive the broad-E5 verdict from the committed per-case numbers."""
    env = json.loads(FIXTURE.read_text())
    summ = env["envelope_summary"]

    assert env["status"] == "passed", f"status={env['status']}"
    assert env["evidence_level"].startswith("E5-broad"), env["evidence_level"]
    # Mesh axis = grading ratio (NU), geometry axis = eps_r.
    assert len(summ["grading_ratios"]) >= MIN_GRADING_RATIOS, summ["grading_ratios"]
    assert len(summ["eps_r_values"]) >= MIN_EPS_R, summ["eps_r_values"]
    assert summ["mesh_axis_kind"] == "nonuniform_dy_profile_ratio", summ["mesh_axis_kind"]
    # The graded TRANSVERSE axis is genuinely exercised (adjacent-cell ratio > 1):
    # this envelope drives the per-cell graded-dA port-plane weighting the
    # dx-graded production tests do not reach.
    assert summ["max_adjacent_ratio"] > 1.0, summ["max_adjacent_ratio"]

    cases = env["cases"]
    assert len(cases) >= MIN_CASES, cases
    per_case_max = []
    for c in cases:
        assert c["status"] == "passed", c
        assert c["max_mag_abs_diff"] <= MAX_TOL, c
        per_case_max.append(c["max_mag_abs_diff"])
    assert summ["passed_case_count"] == summ["case_count"] == len(cases)
    assert summ["max_mag_abs_diff_across_cases"] == pytest.approx(
        max(per_case_max), abs=1e-9
    )
    assert summ["max_mag_abs_diff_across_cases"] <= MAX_TOL
    # The strong eps_r=4 reflector (normalize=True 0.077-floor breaker) is covered.
    assert 4.0 in [float(x) for x in summ["eps_r_values"]], summ["eps_r_values"]


def test_primary_reference_is_independent_analytic() -> None:
    """This ENVELOPE artifact's own truth is the independent analytic Airy and
    it embeds no cross-checks. The external-solver E4 leg now lives in a
    SEPARATE committed fixture (``tests/fixtures/waveguide_nu_broad_e4/``,
    rfx-NU-flux vs Palace FEM, gated by
    ``test_waveguide_nu_broad_e4_comparison_gates.py``) — so the NU flux lane's
    last evidence rung is closed; promotion from shadow to claims-bearing is a
    separate decision."""
    env = json.loads(FIXTURE.read_text())
    assert env["primary_reference"]["label"] == "analytic_airy", env["primary_reference"]
    assert env["cross_check_references"] == [], env["cross_check_references"]


def _synthetic_airy(eps_r: float = 4.0, slab_L: float = 4.0e-3):
    f = np.linspace(8.2e9, 12.4e9, 11)
    s11, s21 = airy(f, eps_r, slab_L, _FC_TE10)
    return f, np.asarray(s11), np.asarray(s21)


def _case_max_diff(rfx_s11, rfx_s21, ref_s11, ref_s21) -> float:
    """Replicate the producer's magnitude-diff gate metric."""
    s11d = np.abs(np.abs(rfx_s11) - np.abs(ref_s11))
    s21d = np.abs(np.abs(rfx_s21) - np.abs(ref_s21))
    return max(float(s11d.max()), float(s21d.max()))


def test_gate_passes_when_rfx_equals_airy() -> None:
    """An ideal slab (rfx == analytic Airy) clears the gate with ~zero diff."""
    _, s11, s21 = _synthetic_airy()
    case_max = _case_max_diff(s11, s21, s11, s21)
    assert case_max <= MAX_TOL
    assert case_max < 1e-12


def test_gate_fails_on_magnitude_perturbation() -> None:
    """A +0.1 |S11| offset (well above MAX_TOL=0.05) must fail the gate."""
    _, s11, s21 = _synthetic_airy()
    perturbed = np.abs(s11) + 0.1
    case_max = _case_max_diff(perturbed, s21, s11, s21)
    assert case_max > MAX_TOL
    assert case_max == pytest.approx(0.1, abs=1e-9)


def test_lossless_slab_airy_is_unitary() -> None:
    """Independent witness: the analytic Airy reference itself conserves power."""
    _, s11, s21 = _synthetic_airy(eps_r=4.0)
    unit = np.abs(s11) ** 2 + np.abs(s21) ** 2
    assert np.allclose(unit, 1.0, atol=1e-6)
