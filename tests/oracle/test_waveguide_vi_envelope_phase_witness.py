"""The |S21| phase residual is the discretization witness for the waveguide port (#894).

On the empty matched WR-90 guide the |S11| magnitude near cutoff carries a dx-independent
term set by whether the far-boundary round trip fits inside the DFT record, and every
convergence order fitted to it under the original record rule was a clipping artifact. The
phase of S21 against the analytic -beta*L does not pass through the absorber at all: it is
invariant to absorber thickness (3x), record length (2.5x) and precision, and it converges
at second order at every band down to f/f_c = 1.010 — where the reflection headline reads
about 1.1. That is what says the extractor itself is second order.

This test pins the FROZEN artifact ``s21_phase_residual_witness.json`` (sweep 369367258390,
rfx b59e1d99). It runs no FDTD. It exists so a future change that breaks the port's phase
convergence, or a future reader who fits an order to a near-cutoff |S11| ladder, meets a
committed table saying which observable carries the discretization and which does not.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "fixtures" / "waveguide_vi_envelope" / "s21_phase_residual_witness.json"


@pytest.fixture(scope="module")
def witness() -> dict:
    return json.loads(ART.read_text())


def test_artifact_provenance_is_recorded(witness):
    assert witness["rfx_sha"] == "b59e1d991dd62868bdf8689a1f642eeb8f7c5b89"
    assert witness["run"] == "369367258390"
    assert witness["absorber_K"] == 3.0


@pytest.mark.parametrize("band", ["R0", "R1", "R2", "R3", "R4", "R5", "R7"])
def test_phase_residual_converges_at_second_order_at_every_band(witness, band):
    """Every pairwise order lies within 0.05 of 2.0, including the two bands nearest
    cutoff (R0 at f/f_c = 1.010, R1 at 1.017) where the reflection headline does not."""
    b = witness["bands"][band]
    assert len(b["rms_deg"]) >= 3, band
    for lo, hi, order in zip(b["rungs"], b["rungs"][1:], b["pairwise_orders"]):
        assert hi == 2 * lo, (band, lo, hi)
        assert abs(order - 2.0) < 0.05, (band, lo, hi, order)


def test_phase_residual_is_absorber_invariant(witness):
    """R2 N=36 at K = 3.0 / 4.5 / 6.0 / 9.0: the residual moves by less than 0.1 %."""
    inv = witness["invariance_deg"]
    vals = [inv[f"R2_N36_K{k}"] for k in (3.0, 4.5, 6.0, 9.0)]
    assert max(vals) / min(vals) - 1 < 1e-3, vals


def test_phase_residual_is_record_invariant(witness):
    """R2 N=18 with the record extended 2.5x: less than 0.1 %."""
    inv = witness["invariance_deg"]
    assert abs(inv["R2_N18_K3p0_t10"] / inv["R2_N18_K3p0"] - 1) < 1e-3


def test_phase_residual_is_precision_invariant(witness):
    """R5 N=72 float32 vs float64: less than 0.2 %."""
    inv = witness["invariance_deg"]
    assert abs(inv["F2_R5_N72_f64"] / inv["S0a_R5_N72_K3p0"] - 1) < 2e-3


def test_the_witness_is_not_the_reflection_headline(witness):
    """The point of the artifact: at R0 the phase converges at 2.00 while the reflection
    headline in the same run converged at ~1.1. The two numbers below are the recorded
    phase orders; the headline's collapse is documented in the design note, not here,
    because it is the artifact this witness exists to distinguish itself from."""
    r0 = witness["bands"]["R0"]
    assert r0["r_lo"] == pytest.approx(1.010)
    assert all(abs(o - 2.0) < 0.01 for o in r0["pairwise_orders"]), r0["pairwise_orders"]
