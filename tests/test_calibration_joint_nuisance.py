"""Joint material + VNA-nuisance recovery — Stage 2 gate (issue #273).

Synthetic-truth self-consistency check for
``calibrate_material_s11(fit_nuisance=True)``:

    1. Reuse the tiny lumped-ez-port / dielectric-slab / PEC-box fixture from
       ``test_calibration_s11_synthetic.py`` with TRUE eps_inf = 4.3.
    2. Corrupt the true S11 with a KNOWN one-port VNA nuisance
       (alpha = 0.9, phi = 0.3 rad, tau = 5 ps one-way):
       measured = alpha * exp(j*phi) * exp(-j*4*pi*f*tau) * S11_true.
       At the 18 GHz band top the delay is ~1.13 rad of round-trip phase —
       deliberately below phase-wrap territory (see the calibrator docstring
       for the group-delay pre-estimate recommendation on real data).
    3. Joint fit from a WRONG eps_inf (3.0) and a NEUTRAL nuisance start with
       ``fit_nuisance=True`` and ``weight_phase=1.0`` (phase carries all the
       tau/phi information; the default 0.1 down-weights it 10x).
       ``weight_mag=4.0`` anchors the material against the confounded basin:
       with balanced weights the initial nuisance phase mismatch dominates
       the loss, Adam's per-parameter normalization amplifies the (initially
       weak, wrong-signed) eps_inf gradient to full-size steps, and the fit
       settles in a local minimum near eps_inf ~ 2 where (alpha, phi, tau)
       co-adapt to the wrong material — measured on this exact fixture.
       Up-weighting |S11| pins eps_inf to the magnitude curve SHAPE (alpha is
       only a uniform scale) while the phase channel sorts out phi/tau.
    4. Assert PARAMETER recovery (eps_inf, alpha, tau) — never loss-drop
       alone: a good fit with wrong parameters is exactly the failure mode
       this track exists to catch.
    5. Witness: the nuisance-BLIND fit (fit_nuisance=False) on the same
       corrupted data must end up with a wrong eps_inf or a materially worse
       loss — nuisance-blind calibration is biased.

Uniform single-device, boundary="pec", complex64 — the v1 calibrator scope.
"""
from __future__ import annotations

import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.material_fit import DebyeFitResult
from rfx.differentiable_material_fit import calibrate_material_s11


# Same tiny fixture as test_calibration_s11_synthetic.py -------------------

FREQ_MAX = 20e9
DX = 1.5e-3
DOMAIN = (0.012, 0.006, 0.006)   # 8 x 4 x 4 cells
NUM_PERIODS = 14.0

TRUE_EPS_INF = 4.3
WRONG_EPS_INF = 3.0

# Ground-truth VNA nuisance (applied to the TRUE S11 to fake a measurement).
TRUE_ALPHA = 0.9
TRUE_PHI = 0.3          # rad
TRUE_TAU = 5e-12        # s one-way; 4*pi*f*tau ~ 1.13 rad at 18 GHz (no wrap)

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


def _nuisanced_s11():
    """Synthetic 'measured' S11: true material + known VNA nuisance."""
    import jax.numpy as jnp

    sim = _make_sim(TRUE_EPS_INF, None, None)
    s11_true = np.asarray(sim.forward(
        port_s11_freqs=jnp.asarray(FREQS, dtype=jnp.float32),
        num_periods=NUM_PERIODS,
        checkpoint=True,
        skip_preflight=True,
    ).s_params)
    factor = TRUE_ALPHA * np.exp(1j * (TRUE_PHI - 4.0 * np.pi * FREQS * TRUE_TAU))
    return (factor * s11_true).astype(np.complex64)


def test_joint_recovery():
    s11_measured = _nuisanced_s11()
    assert np.all(np.isfinite(s11_measured))

    wrong_start = DebyeFitResult(eps_inf=WRONG_EPS_INF, poles=[], fit_error=0.0)

    # The concrete preflight pass (commit 1701da6) fires here and emits the
    # known advisory for this deliberately tiny fixture — "dielectric 'dut'
    # on x/y/z: 5.8 cells per λ_eff (eps_r=3.00, freq_max=20GHz, dx=1.5mm).
    # Need ≥15 cells/λ_eff" — quoted per the repo rule that preflight output
    # is part of the result. Acceptable ONLY because this is a synthetic
    # self-consistency gate (truth and fit share the same under-resolved
    # forward model); a physics-accuracy claim at this mesh would not be.
    joint = calibrate_material_s11(
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
        weight_mag=4.0,     # anchors eps_inf/alpha (see module docstring)
        weight_phase=1.0,   # explicit: phase carries the tau/phi information
        verbose=False,
    )

    err_eps = abs(joint.eps_inf - TRUE_EPS_INF) / TRUE_EPS_INF
    err_alpha = abs(joint.nuisance_alpha - TRUE_ALPHA) / TRUE_ALPHA
    err_tau = abs(joint.nuisance_tau - TRUE_TAU)
    err_phi = abs(joint.nuisance_phi - TRUE_PHI)

    print("\n[joint-nuisance] param      true         recovered")
    print(f"[joint-nuisance] eps_inf  {TRUE_EPS_INF:10.4f} {joint.eps_inf:12.4f}  ({err_eps*100:.2f} %)")
    print(f"[joint-nuisance] alpha    {TRUE_ALPHA:10.4f} {joint.nuisance_alpha:12.4f}  ({err_alpha*100:.2f} %)")
    print(f"[joint-nuisance] phi      {TRUE_PHI:10.4f} {joint.nuisance_phi:12.4f}  ({err_phi:.4f} rad)")
    print(f"[joint-nuisance] tau (s)  {TRUE_TAU:10.3e} {joint.nuisance_tau:12.3e}  ({err_tau:.3e} s)")
    print(f"[joint-nuisance] loss {joint.loss_history[0]:.4e} -> {joint.loss_history[-1]:.4e}")

    # final_s_params must be the NUISANCE-APPLIED model S11 (the quantity
    # compared against the measurement) — verify against an independent
    # recomputation: bare forward at the fitted eps_inf times the fitted
    # nuisance factor. A regression returning the BARE simulated S11 here
    # would fail this allclose (the factor differs from 1 by alpha~0.9 and
    # a ~1 rad phase ramp).
    import jax.numpy as jnp

    assert joint.final_s_params is not None
    assert joint.final_s_params.shape == FREQS.shape
    bare = np.asarray(_make_sim(joint.eps_inf, None, None).forward(
        port_s11_freqs=jnp.asarray(FREQS, dtype=jnp.float32),
        num_periods=NUM_PERIODS,
        checkpoint=True,
        skip_preflight=True,
    ).s_params)
    factor = joint.nuisance_alpha * np.exp(
        1j * (joint.nuisance_phi - 4.0 * np.pi * FREQS * joint.nuisance_tau))
    np.testing.assert_allclose(
        joint.final_s_params, factor * bare, atol=2e-4, rtol=0,
        err_msg="final_s_params is not the nuisance-applied model S11",
    )

    # Parameter recovery — the actual gate.
    assert err_eps < 0.05, f"eps_inf not recovered: {joint.eps_inf:.4f} (rel {err_eps:.3%})"
    assert err_alpha < 0.05, f"alpha not recovered: {joint.nuisance_alpha:.4f} (rel {err_alpha:.3%})"
    assert err_tau < 1.5e-12, f"tau not recovered: {joint.nuisance_tau:.3e} s (err {err_tau:.3e})"

    # ------------------------------------------------------------------
    # Witness: nuisance-BLIND fit on the same corrupted data is biased.
    # Same loss weights for a fair comparison; fewer iterations (a material-
    # only fit has no way to absorb the phase ramp, it plateaus early).
    # ------------------------------------------------------------------
    blind = calibrate_material_s11(
        _make_sim,
        s11_measured,
        FREQS,
        n_debye_poles=0,
        n_lorentz_poles=0,
        n_iterations=45,
        learning_rate=0.05,
        num_periods=NUM_PERIODS,
        initial_guess=wrong_start,
        fit_nuisance=False,
        weight_mag=4.0,     # same loss as the joint fit: comparable numbers
        weight_phase=1.0,
        verbose=False,
    )

    blind_err_eps = abs(blind.eps_inf - TRUE_EPS_INF) / TRUE_EPS_INF
    print(f"[joint-nuisance] BLIND eps_inf = {blind.eps_inf:.4f} (rel {blind_err_eps*100:.2f} %), "
          f"final loss {blind.loss_history[-1]:.4e} vs joint {joint.loss_history[-1]:.4e}")

    # The blind fit must either land on a wrong eps_inf or be stuck at a
    # materially worse loss (it cannot represent the alpha/phase corruption).
    assert (blind_err_eps > 0.05) or (
        blind.loss_history[-1] > 3.0 * joint.loss_history[-1]
    ), (
        f"nuisance-blind fit unexpectedly matched: eps_inf {blind.eps_inf:.4f} "
        f"(rel {blind_err_eps:.3%}), loss {blind.loss_history[-1]:.4e} vs "
        f"joint {joint.loss_history[-1]:.4e}"
    )
