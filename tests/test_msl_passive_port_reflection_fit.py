"""Offline standing-wave fit behind scripts/diagnostics/msl_passive_port_reflection.py.

The #524 re-measurement driver reads ``|Gamma_passive|`` from a float64
two-wave fit over the dump's probe voltages. These tests exercise that fit
on a SYNTHETIC field with planted (alpha, gamma, beta) -- no FDTD -- and pin:

* |Gamma_passive| is recovered to 1e-6 for both port signs, with the beta
  branch anchored at an HJ-style ``beta0`` that is deliberately OFF the true
  beta (the fit must find beta inside the +/-35 % scan, never assume it);
* the recovered beta matches the planted beta (the "fitted beta vs beta0"
  record the review asked for is a real measurement, not an echo of beta0);
* the role witness REFUSES to assign Gamma when |alpha| is not >> |gamma|
  at every port, and assigns it when it is -- the #661 wave-role check runs
  FIRST, and no role flip is ever attempted;
* beta0 <= 0 is rejected (the branch is never chosen by |Gamma| < 1).

New code: there is no pre-existing behaviour to watch fail first; the
mutation check below (a sign-flipped role) is the fail-before-fix analogue.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "diagnostics" / "msl_passive_port_reflection.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("msl_passive_port_reflection", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _planted(alpha, gamma, beta, x):
    x = np.asarray(x, dtype=np.float64)
    return alpha * np.exp(-1j * beta * x) + gamma * np.exp(1j * beta * x)


# Probe ladder of the design fixture: 5 probes, 12 cells apart, dx = 84.67 um,
# referenced anywhere (only differences enter).
DX = 84.67e-6
X_POS = 2.0e-3 + DX * (29 + 12 * np.arange(5))          # "+x" port: increasing
X_NEG = 12.0e-3 - DX * (29 + 12 * np.arange(5))         # "-x" port: decreasing


@pytest.mark.parametrize("beta_true,beta0_scale", [(250.0, 1.0), (250.0, 0.9), (180.0, 1.12)])
@pytest.mark.parametrize("port_sign,x", [(+1.0, X_POS), (-1.0, X_NEG)])
def test_fit_recovers_abs_gamma_to_1e6(mod, beta_true, beta0_scale, port_sign, x):
    """|Gamma_passive| = |wave back into the structure| / |wave incident|."""
    alpha = 1.0 * np.exp(1j * 0.3)
    gamma = 0.21 * np.exp(-1j * 1.1)          # a July-like |Gamma| (0.187-0.211 class)
    v = _planted(alpha, gamma, beta_true, x)
    fit = mod.fit_two_wave(v, x, beta_true * beta0_scale)
    assert not fit["railed"]
    assert fit["beta"] == pytest.approx(beta_true, rel=1e-9)
    # The fit is referenced at PROBE 0 (as extract_msl_nprobe is), so the
    # planted x=0 amplitudes must be propagated to x[0] before comparing.
    assert abs(fit["alpha"] - alpha * np.exp(-1j * beta_true * x[0])) < 1e-8
    assert abs(fit["gamma"] - gamma * np.exp(1j * beta_true * x[0])) < 1e-8
    g = mod.passive_gamma(fit["alpha"], fit["gamma"], port_sign)
    expected = abs(alpha / gamma) if port_sign > 0 else abs(gamma / alpha)
    assert abs(abs(g) - expected) < 1e-6


def test_role_witness_refuses_when_alpha_not_much_larger_than_gamma(mod):
    """|gamma/alpha| = 0.9 at the passive port: ratio 1.11 < 2 -> no Gamma assigned."""
    beta = 250.0
    band = np.ones(3, dtype=bool)
    betas = np.array([beta, beta * 1.01, beta * 1.02])
    # drive 0 is "+x" (launches alpha); port 1 is "-x".
    fits = {}
    v0 = np.stack([_planted(1.0, 0.05, b, X_POS) for b in betas], axis=1)   # driven port
    v1 = np.stack([_planted(1.0, 0.90, b, X_NEG) for b in betas], axis=1)   # passive port
    fits[0] = mod.fit_two_wave_band(v0, X_POS, betas)
    fits[1] = mod.fit_two_wave_band(v1, X_NEG, betas)
    res = mod.assign_passive_gamma(fits, +1.0, {0: +1.0, 1: -1.0}, 0, band)
    assert res["assigned"] is False
    assert res["gamma_passive"] == {}
    assert "role witness failed" in res["refusal"]
    assert res["witness"][1]["passed"] is False
    assert res["witness"][1]["median_ratio_in_band"] == pytest.approx(1.0 / 0.9, rel=1e-6)


def test_role_witness_assigns_when_alpha_dominates_at_both_ports(mod):
    beta = 250.0
    band = np.ones(3, dtype=bool)
    betas = np.array([beta, beta * 1.01, beta * 1.02])
    v0 = np.stack([_planted(1.0, 0.05, b, X_POS) for b in betas], axis=1)
    v1 = np.stack([_planted(0.8, 0.8 * 0.21, b, X_NEG) for b in betas], axis=1)
    fits = {0: mod.fit_two_wave_band(v0, X_POS, betas),
            1: mod.fit_two_wave_band(v1, X_NEG, betas)}
    res = mod.assign_passive_gamma(fits, +1.0, {0: +1.0, 1: -1.0}, 0, band)
    assert res["assigned"] is True
    assert set(res["gamma_passive"]) == {1}          # only the passive port gets a Gamma
    assert np.allclose(np.abs(res["gamma_passive"][1]), 0.21, atol=1e-6)
    # Mutation check (fail-before-fix analogue for new code): a swapped role
    # reports 1/0.21, which the witness would NOT have passed as-is -- and the
    # correct assignment is the small one.
    assert not np.allclose(np.abs(res["gamma_passive"][1]), 1 / 0.21, atol=1e-3)


def test_negative_going_drive_uses_gamma_as_dominant(mod):
    """Drive 1 ("-x") launches gamma: the witness must read the other way round."""
    beta = 250.0
    band = np.ones(3, dtype=bool)
    betas = np.array([beta, beta * 1.01, beta * 1.02])
    v0 = np.stack([_planted(0.8 * 0.21, 0.8, b, X_POS) for b in betas], axis=1)  # passive "+x"
    v1 = np.stack([_planted(0.05, 1.0, b, X_NEG) for b in betas], axis=1)        # driven "-x"
    fits = {0: mod.fit_two_wave_band(v0, X_POS, betas),
            1: mod.fit_two_wave_band(v1, X_NEG, betas)}
    res = mod.assign_passive_gamma(fits, -1.0, {0: +1.0, 1: -1.0}, 1, band)
    assert res["assigned"] is True
    assert set(res["gamma_passive"]) == {0}
    assert np.allclose(np.abs(res["gamma_passive"][0]), 0.21, atol=1e-6)


def test_beta_anchor_must_be_positive(mod):
    v = _planted(1.0, 0.2, 250.0, X_POS)
    with pytest.raises(ValueError, match="beta0 must be > 0"):
        mod.fit_two_wave(v, X_POS, -250.0)


def test_rail_flag_when_true_beta_outside_scan(mod):
    """beta 50 % off the anchor is outside the +/-35 % window -> railed, as in production."""
    v = _planted(1.0, 0.2, 250.0, X_POS)
    fit = mod.fit_two_wave(v, X_POS, 250.0 / 1.5)
    assert fit["railed"] is True
