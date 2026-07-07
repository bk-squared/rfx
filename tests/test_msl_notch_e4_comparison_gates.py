"""cv06b MSL open-stub notch — rfx-vs-openEMS E4 external cross-check gates.

Locks the CHARACTERIZED external comparison committed under
``tests/fixtures/msl_notch_e4/`` (physical openEMS dx=50 µm reference + the rfx
``add_msl_port`` result at matched geometry). This is NOT a tight OpenEMS-class
validation — it is an honest characterization:

  * off-notch |S21| transmission agrees to ~0.1 (rfx and openEMS overlap away
    from the notch),
  * the notch frequency agrees to ~6% — rfx sits near the fringing-free analytic
    quarter-wave (3.69 GHz), openEMS lands lower (3.43 GHz). UPDATE 2026-07-07: a
    Palace FEM referee (conformal tets, independent method, matched geometry)
    lands at ~3.631 GHz at two mesh densities, closest to rfx (+0.1%). Our
    earlier working interpretation (open-end fringing as the driver of the
    split) is revised by that evidence; the ~6% figure remains characterized,
    not an rfx error to chase (R2-tight: one clean comparison, no chasing). See
    ``msl_stub_notch_palace_referee.json`` +
    ``test_msl_notch_palace_referee_gates.py``.

The gate fails closed if rfx drifts further from openEMS, loses passivity, or the
notch disappears — so it is a genuine regression lock on the committed evidence,
re-derived (no FDTD) by ``build_msl_notch_openems_comparison.py``.

No FDTD here: the ~65 min rfx run is committed as a fixture, mirroring the coax
broad-E4 evidence-commit pattern.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIX = _REPO_ROOT / "tests/fixtures/msl_notch_e4"
sys.path.insert(0, str(_REPO_ROOT / "scripts/diagnostics"))
from build_msl_notch_openems_comparison import build_comparison  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    return build_comparison(_FIX)


def test_both_solvers_passive(summary):
    """The committed comparison uses ONLY the physical dx=50 µm data: both rfx and
    openEMS conserve energy (|S11|^2+|S21|^2 <= ~1). The non-physical dx=80 µm
    openEMS references (energy sum up to 8.9) are deliberately excluded."""
    assert summary["rfx"]["max_energy_sum"] <= 1.05, summary["rfx"]
    assert summary["openems"]["max_energy_sum"] <= 1.05, summary["openems"]


def test_both_show_a_deep_notch(summary):
    """Both solvers produce a clear open-stub notch (the qualitative physics)."""
    assert summary["rfx"]["notch_depth_db"] < -20.0, summary["rfx"]
    assert summary["openems"]["notch_depth_db"] < -20.0, summary["openems"]


def test_notch_frequency_characterized_within_envelope(summary):
    """Notch-frequency agreement is CHARACTERIZED at ~6% (rfx high). Locked at 7%
    so a regression that pushed rfx further from the full-wave truth fails; NOT a
    tight OpenEMS-class claim."""
    rel = summary["comparison"]["notch_freq_rel_pct"]
    assert rel <= 7.0, f"notch-freq disagreement {rel}% exceeds the characterized 7% envelope"
    # rfx sits ABOVE openEMS — the sign is part of the committed characterization.
    # (The 2026-07-07 Palace FEM referee lands at ~3.631 GHz, closest to rfx;
    # see msl_stub_notch_palace_referee.json.)
    assert summary["rfx"]["notch_ghz"] > summary["openems"]["notch_ghz"], (
        "rfx notch should be higher than openEMS (staircase under-captures open-end fringing)"
    )


def test_off_notch_transmission_agrees(summary):
    """Away from the notch (2.5-6 GHz) the |S21| magnitude curves overlap well."""
    c = summary["comparison"]
    assert c["s21_mag_mean_abs_diff_2p5_6ghz"] <= 0.13, c
    assert c["s21_mag_max_abs_diff_2p5_6ghz"] <= 0.25, c


def test_committed_summary_matches_rederived(summary):
    """Fixture integrity: the committed comparison_summary.json equals what the
    producer re-derives from the two result fixtures."""
    committed = json.loads((_FIX / "comparison_summary.json").read_text())
    assert committed["comparison"] == summary["comparison"]
    assert committed["rfx"]["notch_ghz"] == summary["rfx"]["notch_ghz"]
    assert committed["openems"]["notch_ghz"] == summary["openems"]["notch_ghz"]


def test_openems_reference_is_the_physical_dx50(summary):
    """Guard the R5 finding: the reference is the physical dx=50 µm run (5.08
    substrate cells), NOT a dx=80 µm mixed-cell run (which violates passivity)."""
    oe = json.loads((_FIX / "msl_stub_notch_openems_dx50.json").read_text())
    assert oe["meta"]["dx_um"] == 50.0
    s11 = np.asarray(oe["s11_mag"], dtype=float)
    s21 = np.asarray(oe["s21_mag"], dtype=float)
    assert float((s11 ** 2 + s21 ** 2).max()) <= 1.05, "openEMS reference must be passive"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
