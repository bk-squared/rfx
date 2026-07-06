"""Over-parameterization detection — Stage 2C fail-closed guard.

A common calibration failure mode is fitting a model with more dispersive poles
than the data supports.  This test proves the Fisher/Jacobian machinery catches
it: synthetic S11 is generated from a 1-Debye-pole truth, but the
identifiability analysis is built for a 2-Debye-pole model.  The extra pole is
redundant, so the Fisher information is rank-deficient — a near-zero eigenvalue
/ a huge condition number — and the fail-closed policy must fire
(``is_identifiable == False``) and flag the sloppy (unidentified) direction.

We evaluate the 2-Debye Jacobian at a degenerate point where both poles are
identical ``(delta_eps/2, tau)``.  The two ``delta_eps`` sensitivities are then
exactly collinear (and the two ``tau`` sensitivities degenerate), giving a
genuine rank deficiency — the textbook redundant-parameter signature.
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole
from rfx.differentiable_material_fit import _poles_to_params
from rfx.calibration_identifiability import (
    s11_residual_fn,
    s11_jacobian,
    fisher_information,
    identifiability_report,
)


FREQ_MAX = 20e9
DX = 1.0e-3
DOMAIN = (0.010, 0.004, 0.004)
NUM_PERIODS = 16.0

BASE_EPS_INF = 2.0
TRUE_DELTA_EPS = 1.5
F_RELAX = 12e9
TAU = 1.0 / (2 * np.pi * F_RELAX)

NOISE_STD = 0.01
FREQS = np.linspace(6e9, 18e9, 13)

PARAM_NAMES = ["eps_inf", "de_1", "tau_1", "de_2", "tau_2"]


def _make_sim(eps_inf, debye_poles, lorentz_poles):
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles or None)
    sim.add(Box((0.004, 0.0, 0.0), (0.002, 0.004, 0.004)), material="dut")
    sim.add_port(
        position=(0.002, 0.002, 0.002),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.9, amplitude=1.0),
    )
    return sim


def test_overparam_two_debye_is_rank_deficient():
    # 2-Debye model evaluated at the degenerate 1-pole-worth point: two
    # identical poles each carrying half the true strength.
    poles = [
        DebyePole(delta_eps=TRUE_DELTA_EPS / 2, tau=TAU),
        DebyePole(delta_eps=TRUE_DELTA_EPS / 2, tau=TAU),
    ]
    params = _poles_to_params(BASE_EPS_INF, poles, [])
    residual = s11_residual_fn(_make_sim, FREQS, n_debye=2, n_lorentz=0,
                               num_periods=NUM_PERIODS)
    J = np.asarray(s11_jacobian(residual, params))
    assert J.shape == (2 * len(FREQS), 5), J.shape

    F = fisher_information(J, NOISE_STD)
    report = identifiability_report(
        F, PARAM_NAMES, J=J, noise_std=NOISE_STD)

    eig = report.eigenvalues
    lam_min, lam_max = float(eig[0]), float(eig[-1])
    eig_ratio = lam_min / lam_max

    print("\n[overparam] 2-Debye model on 1-Debye truth (degenerate poles)")
    print(f"[overparam] eigenvalues (asc): {eig}")
    print(f"[overparam] lambda_min = {lam_min:.3e}  lambda_max = {lam_max:.3e}")
    print(f"[overparam] lambda_min/lambda_max = {eig_ratio:.3e}")
    print(f"[overparam] condition_number = {report.condition_number:.3e}")
    print(f"[overparam] numerical rank J = {report.rank} / {len(PARAM_NAMES)}")
    print(f"[overparam] is_identifiable = {report.is_identifiable}")
    print(f"[overparam] reason: {report.reason}")
    for d in report.sloppy_directions:
        print(f"[overparam] sloppy dir #{d['index']} sigma={d['singular_value']:.3e} "
              f"dominated by {d['dominant_params']}")

    # Fail-closed must fire: redundant pole -> rank-deficient / ill-conditioned.
    assert report.is_identifiable is False, (
        "fail-closed did not fire on an over-parameterized model "
        f"(cond={report.condition_number:.3e})")
    # A genuine near-zero eigenvalue relative to the largest.
    assert eig_ratio < 1e-8, f"no near-zero eigenvalue: ratio {eig_ratio:.3e}"
    # The sloppy direction(s) must be flagged and rank must be deficient.
    assert report.rank < len(PARAM_NAMES), f"rank not deficient: {report.rank}"
    assert len(report.sloppy_directions) >= 1, "no sloppy direction flagged"
    # The unidentified direction should involve the redundant Debye parameters,
    # not eps_inf.
    sloppy_params = {
        e["param"] for d in report.sloppy_directions for e in d["dominant_params"]
    }
    assert sloppy_params & {"de_1", "de_2", "tau_1", "tau_2"}, (
        f"sloppy direction not dominated by redundant pole params: {sloppy_params}")
