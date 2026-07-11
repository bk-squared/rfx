"""UQ gates for the calibration track — Stage 2 (issue #273).

Two independent gates on :func:`rfx.calibration_identifiability.calibration_uq`:

``test_noise_crlb_consistency``
    Noisy synthetic joint fit, then the Cramér-Rao report at the optimum:
    every recovered parameter must sit within ``max(5 * crlb_std_physical,
    small_floor)`` of its truth.  CRLB is a LOWER bound on the std of an
    efficient unbiased estimator, and Adam is neither efficient nor exactly
    converged, so 5x slack plus a small optimizer floor is the honest gate —
    it catches order-of-magnitude-wrong covariances (units, scaling, wrong
    Jacobian) without pretending Adam attains the bound.

``test_koh_tau_confounding_bandwidth``
    Fisher-only (no fit): on a NARROW band the reference-plane delay
    ``tau_hat`` and the material ``log_eps_inf`` are confounded (a delay and
    a dielectric both mostly rotate S11 phase over a small band) — the KOH
    "good fit, wrong parameters" failure mode.  A WIDE band breaks the
    degeneracy.  Mirrors ``test_identifiability_bandwidth.py``.

Uniform single-device, boundary="pec", complex64 — the v1 calibrator scope.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.material_fit import DebyeFitResult
from rfx.differentiable_material_fit import (
    MaterialFitResult,
    calibrate_material_s11,
    _nuisance_tau_scale,
    _poles_to_params,
)
from rfx.calibration_identifiability import (
    IdentifiabilityReport,
    calibration_param_names,
    calibration_uq,
    fisher_information,
    identifiability_report,
    s11_jacobian,
    s11_residual_fn,
)


# --------------------------------------------------------------------------
# Gate B: noisy joint fit vs Cramér-Rao bound (8x4x4 fixture, one fit)
# --------------------------------------------------------------------------

FREQ_MAX = 20e9
DX = 1.5e-3
DOMAIN = (0.012, 0.006, 0.006)   # 8 x 4 x 4 cells
NUM_PERIODS = 14.0

TRUE_EPS_INF = 4.3
WRONG_EPS_INF = 3.0

TRUE_ALPHA = 0.9
TRUE_PHI = 0.3
TRUE_TAU = 5e-12                 # s one-way (~1.13 rad round trip at 18 GHz)

SIGMA = 0.01                     # i.i.d. Gaussian noise std per Re/Im component

FREQS = np.linspace(6e9, 18e9, 9)


def _make_sim(eps_inf, debye_poles, lorentz_poles):
    """Lumped ez port on a dielectric slab inside a closed PEC box."""
    sim = Simulation(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, boundary="pec")
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles or None)
    sim.add(Box((0.004, 0.0, 0.0), (0.008, 0.006, 0.006)), material="dut")
    sim.add_port(
        position=(0.003, 0.003, 0.003),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=FREQ_MAX / 2, bandwidth=0.9, amplitude=1.0),
    )
    return sim


def _noisy_measurement():
    """True material + known nuisance + fixed-seed Gaussian noise."""
    sim = _make_sim(TRUE_EPS_INF, None, None)
    s11_true = np.asarray(sim.forward(
        port_s11_freqs=jnp.asarray(FREQS, dtype=jnp.float32),
        num_periods=NUM_PERIODS,
        checkpoint=True,
        skip_preflight=True,
    ).s_params)
    factor = TRUE_ALPHA * np.exp(1j * (TRUE_PHI - 4.0 * np.pi * FREQS * TRUE_TAU))
    s11 = factor * s11_true
    rng = np.random.default_rng(20260273)   # fixed seed: deterministic gate
    noise = rng.normal(0.0, SIGMA, FREQS.shape) + 1j * rng.normal(0.0, SIGMA, FREQS.shape)
    return (s11 + noise).astype(np.complex64)


def test_noise_crlb_consistency():
    s11_measured = _noisy_measurement()
    wrong_start = DebyeFitResult(eps_inf=WRONG_EPS_INF, poles=[], fit_error=0.0)

    result = calibrate_material_s11(
        _make_sim,
        s11_measured,
        FREQS,
        n_debye_poles=0,
        n_lorentz_poles=0,
        n_iterations=120,
        learning_rate=0.05,
        num_periods=NUM_PERIODS,
        initial_guess=wrong_start,
        fit_nuisance=True,
        weight_mag=4.0,           # anchors eps_inf/alpha against the confounded
        weight_phase=1.0,         # basin (see test_calibration_joint_nuisance)
        compute_uq=True,          # routes through calibration_uq(noise_std=SIGMA)
        noise_std=SIGMA,
        verbose=False,
    )

    report = result.uq
    assert isinstance(report, IdentifiabilityReport), type(report)
    assert report.param_names == ["log_eps_inf", "log_alpha", "phi", "tau_hat"]
    assert report.is_identifiable, report.reason

    crlb = np.asarray(report.crlb_std, dtype=float)
    assert np.all(np.isfinite(crlb)), crlb
    tau_scale = _nuisance_tau_scale(FREQS)

    # Physical-unit errors vs physical-unit CRLB stds (see calibration_uq
    # docstring for the unit conventions):
    #   log-space params -> compare in log space (relative);
    #   phi -> rad; tau -> seconds via tau_scale de-scaling.
    err = {
        "log_eps_inf": abs(np.log(result.eps_inf / TRUE_EPS_INF)),
        "log_alpha": abs(np.log(result.nuisance_alpha / TRUE_ALPHA)),
        "phi": abs(result.nuisance_phi - TRUE_PHI),
        "tau_hat": abs(result.nuisance_tau - TRUE_TAU),
    }
    crlb_physical = {
        "log_eps_inf": crlb[0],
        "log_alpha": crlb[1],
        "phi": crlb[2],
        "tau_hat": crlb[3] * tau_scale,   # scaled -> seconds
    }
    # Floors are a deterministic BACKSTOP only — they keep the gate physical
    # if a future fixture/seed change collapses the CRLB.  At this fixture the
    # 5*crlb clause is the BINDING one for all four parameters (5*crlb ~=
    # 3.9e-2 log-eps, 2.9e-2 log-alpha, 8.7e-2 phi, 6.4e-13 s tau — every
    # floor below sits under its 5*crlb).  Do NOT raise a floor above its
    # 5*crlb: that makes the CRLB clause inert and the test stops checking
    # the covariance at all (the exact defect an adversarial verify pass
    # caught on 2026-07-11).  Measured errors at the fixed seed: 3.0e-3
    # log-eps, 5.8e-3 log-alpha, 6.6e-2 phi (0.76x of gate), 3.9e-13 s tau
    # (0.61x) — phi/tau crawl along their sloppy valley (corr ~ 0.95), which
    # is the consistency this gate exists to check.
    floor = {
        "log_eps_inf": 0.01,
        "log_alpha": 0.01,
        "phi": 0.05,
        "tau_hat": 3e-13,
    }

    print(f"\n[noise-crlb] sigma = {SIGMA}, is_identifiable = {report.is_identifiable}")
    for name in report.param_names:
        bound = max(5.0 * crlb_physical[name], floor[name])
        print(f"[noise-crlb] {name:12s} |err| = {err[name]:.3e}  "
              f"5*crlb = {5.0 * crlb_physical[name]:.3e}  floor = {floor[name]:.1e}  "
              f"gate = {bound:.3e}")
    for name in report.param_names:
        bound = max(5.0 * crlb_physical[name], floor[name])
        assert err[name] < bound, (
            f"{name}: |recovered - true| = {err[name]:.3e} exceeds "
            f"max(5*crlb, floor) = {bound:.3e} "
            f"(crlb_physical = {crlb_physical[name]:.3e})"
        )


def test_sigma_hat_estimation_and_fail_closed():
    """``noise_std=None`` path: sigma_hat estimated from the residual at the
    optimum, plus the three fail-closed ValueErrors (all cheap — they fire
    before the AD Jacobian is built)."""
    s11_measured = _noisy_measurement()
    # Evaluate AT the truth point: the residual is then (minus) the injected
    # noise, so the dof-corrected sigma_hat must land near SIGMA.
    truth = MaterialFitResult(
        eps_inf=TRUE_EPS_INF,
        nuisance_alpha=TRUE_ALPHA,
        nuisance_phi=TRUE_PHI,
        nuisance_tau=TRUE_TAU,
    )
    report = calibration_uq(
        _make_sim, FREQS, truth,
        n_debye_poles=0, n_lorentz_poles=0,
        fit_nuisance=True, noise_std=None, s11_measured=s11_measured,
        num_periods=NUM_PERIODS,
    )
    # 18 residual components - 4 params -> dof 14; the chi distribution puts
    # a ~19% relative std on sigma_hat, so a factor-2 bracket genuinely
    # checks the estimate without being seed-brittle.
    assert 0.5 * SIGMA < report.noise_std < 2.0 * SIGMA, report.noise_std
    assert report.is_identifiable, report.reason

    # Fail-closed: no noise level and no measurement to estimate it from.
    with pytest.raises(ValueError, match="s11_measured"):
        calibration_uq(_make_sim, FREQS, truth, n_debye_poles=0,
                       n_lorentz_poles=0, fit_nuisance=True,
                       noise_std=None, num_periods=NUM_PERIODS)
    # Fail-closed: joint analysis of a nuisance-blind fit result.
    blind = MaterialFitResult(eps_inf=TRUE_EPS_INF)
    with pytest.raises(ValueError, match="nuisance"):
        calibration_uq(_make_sim, FREQS, blind, n_debye_poles=0,
                       n_lorentz_poles=0, fit_nuisance=True,
                       noise_std=SIGMA, num_periods=NUM_PERIODS)
    # Fail-closed: declared model order disagrees with the fitted result.
    with pytest.raises(ValueError, match="model order"):
        calibration_uq(_make_sim, FREQS, truth, n_debye_poles=1,
                       n_lorentz_poles=0, fit_nuisance=True,
                       noise_std=SIGMA, num_periods=NUM_PERIODS)


# --------------------------------------------------------------------------
# Gate C: KOH tau-vs-eps_inf confounding vs bandwidth (Fisher-only, no fit)
# --------------------------------------------------------------------------

# Thin non-resonant slab fixture from test_identifiability_bandwidth.py.
KOH_FREQ_MAX = 20e9
KOH_DX = 1.0e-3
KOH_DOMAIN = (0.010, 0.004, 0.004)   # 10 x 4 x 4 cells
KOH_NUM_PERIODS = 16.0
KOH_EPS_INF = 4.3
NOISE_STD = 0.01                      # does not affect correlations/condition

# NARROW: 3 points over +-0.8% at 12 GHz.  The band must be genuinely tight:
# at +-4% (11.5-12.5 GHz) the band curvature already re-separates the delay
# from the dielectric on this fixture (measured |corr| ~ 0.64); at +-0.8% the
# eps/tau correlation is ~0.96.
NARROW_FREQS = np.linspace(11.9e9, 12.1e9, 3)
WIDE_FREQS = np.linspace(6e9, 18e9, 13)


def _make_koh_sim(eps_inf, debye_poles, lorentz_poles):
    """Thin (2 mm) non-resonant dielectric slab, one lumped ez port, PEC box."""
    sim = Simulation(freq_max=KOH_FREQ_MAX, domain=KOH_DOMAIN, dx=KOH_DX,
                     boundary="pec")
    sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles or None)
    sim.add(Box((0.004, 0.0, 0.0), (0.002, 0.004, 0.004)), material="dut")
    sim.add_port(
        position=(0.002, 0.002, 0.002),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=KOH_FREQ_MAX / 2, bandwidth=0.9, amplitude=1.0),
    )
    return sim


def _joint_report(freqs):
    """Identifiability report for {log_eps_inf, log_alpha, phi, tau_hat}.

    Evaluated at the truth point with a NEUTRAL nuisance (alpha=1, phi=0,
    tau=0): log_alpha = 0, phi = 0, tau_hat = 0.
    """
    material = _poles_to_params(KOH_EPS_INF, [], [])
    params = jnp.concatenate([material, jnp.zeros(3, dtype=material.dtype)])
    residual = s11_residual_fn(
        _make_koh_sim, freqs, n_debye=0, n_lorentz=0,
        num_periods=KOH_NUM_PERIODS, include_nuisance=True,
    )
    J = np.asarray(s11_jacobian(residual, params))
    names = calibration_param_names(0, 0, include_nuisance=True)
    F = fisher_information(J, NOISE_STD)
    return identifiability_report(F, names, J=J, noise_std=NOISE_STD)


def test_koh_tau_confounding_bandwidth():
    narrow = _joint_report(NARROW_FREQS)
    wide = _joint_report(WIDE_FREQS)

    i_eps = narrow.param_names.index("log_eps_inf")
    i_tau = narrow.param_names.index("tau_hat")
    corr_narrow = abs(float(narrow.correlation_matrix[i_eps, i_tau]))
    corr_wide = abs(float(wide.correlation_matrix[i_eps, i_tau]))

    print("\n[koh-confounding] joint {log_eps_inf, log_alpha, phi, tau_hat}")
    print(f"[koh-confounding] narrow {NARROW_FREQS[0]/1e9:.1f}-{NARROW_FREQS[-1]/1e9:.1f} GHz "
          f"({len(NARROW_FREQS)} pts): |corr(tau,eps)| = {corr_narrow:.4f}, "
          f"cond = {narrow.condition_number:.3e}, identifiable = {narrow.is_identifiable}")
    print(f"[koh-confounding] wide   {WIDE_FREQS[0]/1e9:.1f}-{WIDE_FREQS[-1]/1e9:.1f} GHz "
          f"({len(WIDE_FREQS)} pts): |corr(tau,eps)| = {corr_wide:.4f}, "
          f"cond = {wide.condition_number:.3e}, identifiable = {wide.is_identifiable}")

    # NARROW band: delay and dielectric confounded — either the tau/eps_inf
    # correlation is near ±1 or the fail-closed verdict already fired.
    assert (corr_narrow > 0.9) or (not narrow.is_identifiable), (
        f"narrow band unexpectedly clean: |corr| = {corr_narrow:.4f}, "
        f"identifiable = {narrow.is_identifiable} ({narrow.reason})"
    )

    # WIDE band: identifiable, with the correlation bounded away from 1.
    assert wide.is_identifiable, wide.reason
    assert corr_wide < 0.9, (
        f"wide band still confounded: |corr(tau_hat, log_eps_inf)| = {corr_wide:.4f}"
    )
