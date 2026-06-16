"""Committed gate for the rectangular-waveguide broad-E5 flux envelopes.

Two layers, mirroring ``tests/test_msl_broad_e5_envelope_gates.py``:

1. **Committed-fixture re-derivation** — load every band envelope JSON under
   ``tests/fixtures/waveguide_broad_e5/`` (regenerated on gpu-rtx4090, VESSL
   369367242914, all 5 WR bands 20/20 cases pass vs analytic Airy) and
   re-assert the broad-E5 verdict from the committed per-case numbers, so the
   "broad_e5_passed" claim survives a clean checkout instead of riding on
   gitignored ``.omx`` artifacts.

2. **Gate-semantics lock** — drive the *real* ``airy_slab`` reference formula
   and the ``MAX_TOL`` tolerance from the producer with synthetic ideal /
   perturbed S-parameters, asserting the magnitude-diff gate fires exactly at
   the threshold (ideal slab passes; a +0.1 |S| perturbation fails).

These are pure-Python contracts (no FDTD).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_envelope import (  # type: ignore  # noqa: E402
    MAX_TOL,
    airy_slab,
)

FIXTURES = REPO / "tests" / "fixtures" / "waveguide_broad_e5"
EXPECTED_BANDS = {
    "wr28_kaband",
    "wr62_kuband",
    "wr15_vband",
    "wr340_sband",
    "wr10_wband",
}


BROAD_E4 = FIXTURES / "wr90_rectangular_broad_e4_comparison.json"
# Same broad-blocking tokens the auditor (check_port_external_references.py)
# rejects in an evidence_level / claim_scope.
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow", "only",
)


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("waveguide_*_broad_e5_envelope.json"))


def test_all_five_bands_present() -> None:
    """The committed fixture set must cover every promoted WR band."""
    tokens = {
        p.name.replace("waveguide_", "").replace("_broad_e5_envelope.json", "")
        for p in _fixture_files()
    }
    assert tokens == EXPECTED_BANDS, f"committed bands {tokens} != {EXPECTED_BANDS}"


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda p: p.stem)
def test_committed_band_envelope_passes_broad_e5(path: Path) -> None:
    """Re-derive the broad-E5 verdict from the committed per-case numbers."""
    env = json.loads(path.read_text())
    summ = env["envelope_summary"]

    assert env["status"] == "passed", f"{path.name} status={env['status']}"
    assert env["evidence_level"].startswith("E5-broad"), env["evidence_level"]
    # Envelope spans both the mesh and geometry axes (>=2 dx, >=2 eps_r).
    assert len(summ["dx_values_m"]) >= 2, summ["dx_values_m"]
    assert len(summ["eps_r_values"]) >= 2, summ["eps_r_values"]

    # Every case must independently clear the magnitude tolerance, and the
    # JSON-reported aggregate must match a fresh max over the per-case diffs
    # (catches a doctored summary that disagrees with its own cases).
    cases = env["cases"]
    assert len(cases) >= 4, cases
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


def _synthetic_airy(eps_r: float = 2.0, L: float = 3.0e-3, fc_v: float = 9.488e9):
    f = np.linspace(12.4e9, 18.0e9, 11)
    s11, s21 = airy_slab(f, eps_r, L, fc_v)
    return f, s11, s21


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
    perturbed = s11 * 0.0 + (np.abs(s11) + 0.1)  # magnitude bumped by 0.1
    case_max = _case_max_diff(perturbed, s21, s11, s21)
    assert case_max > MAX_TOL
    assert case_max == pytest.approx(0.1, abs=1e-9)


def test_lossless_slab_airy_is_unitary() -> None:
    """Independent witness: the analytic reference itself conserves power."""
    _, s11, s21 = _synthetic_airy()
    unit = np.abs(s11) ** 2 + np.abs(s21) ** 2
    assert np.allclose(unit, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Broad-E4 external-solver comparison (the external leg of the port-1 close)
# ---------------------------------------------------------------------------


def test_broad_e4_comparison_committed_passes() -> None:
    """rfx-vs-external-FDTD comparison across the 3 WR-90 geometries passes."""
    d = json.loads(BROAD_E4.read_text())
    assert d["status"] == "passed", d
    summ = d["summary"]
    assert summ["geometry_count"] == 3, summ
    assert summ["passed_pair_count"] == summ["pair_count"], summ
    tol = d["max_mag_abs_tol"]
    per_pair_max = [p["max_mag_abs_diff"] for p in d["pairs"]]
    for p in d["pairs"]:
        assert p["status"] == "passed", p
        assert p["max_mag_abs_diff"] <= tol, p
    assert summ["max_mag_abs_diff"] == pytest.approx(max(per_pair_max), abs=1e-9)
    # PEC-short magnitude must be essentially exact reflection (the R5 finding:
    # rfx nails |S11|=1, an invalid Meep res-3/4 ref did not -> Palace used).
    pec = [p for p in d["pairs"] if p["geometry"] == "pec_short"]
    assert pec and all(p["max_mag_abs_diff"] < 0.02 for p in pec), pec


def test_broad_e4_comparison_qualifies_for_auditor() -> None:
    """Mirror check_port_external_references.py's broad-E4 acceptance."""
    d = json.loads(BROAD_E4.read_text())
    level = d["evidence_level"].lower()
    scope = d["claim_scope"].lower()
    assert d["status"] == "passed"
    assert d["evidence_level"].startswith("E4")
    assert "broad" in scope
    assert not any(t in level or t in scope for t in BLOCKING_TOKENS), (level, scope)
