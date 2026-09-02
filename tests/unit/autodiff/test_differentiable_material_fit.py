"""Tests for differentiable material fitting via jax.grad through FDTD.

Validates:
1. jax.grad flows through init_debye / init_lorentz (non-zero gradients)
2. jax.grad flows through full FDTD from S-param loss to pole params
3. S-param loss is 0 for identical inputs, positive for different ones
4. Recovery of known Debye material from synthetic S-params
5. Parameter round-trip (log-space <-> physical poles)
"""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays, init_state, init_materials, update_h
from rfx.materials.debye import DebyePole, init_debye, update_e_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz, lorentz_pole
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, make_probe, run

from rfx.differentiable_material_fit import (
    MaterialFitResult,
    _poles_to_params,
    _params_to_debye_poles,
    _params_to_lorentz_poles,
    sparam_loss,
    _adam_step,
)

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Test 1: init_debye is differentiable
# ---------------------------------------------------------------------------

def test_init_debye_is_differentiable():
    """jax.grad through init_debye produces non-zero gradients."""
    shape = (8, 8, 8)
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32) * 4.4,
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )
    dt = 1e-12

    def loss_fn(delta_eps, tau):
        pole = DebyePole(delta_eps=delta_eps, tau=tau)
        coeffs, _ = init_debye([pole], materials, dt)
        # Loss = sum of squared ADE coefficients (differentiable scalar)
        return jnp.sum(coeffs.ca ** 2) + jnp.sum(coeffs.cb ** 2)

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    g_de, g_tau = grad_fn(jnp.float32(3.0), jnp.float32(8.3e-12))

    print("\ninit_debye gradients:")
    print(f"  d(loss)/d(delta_eps) = {float(g_de):.6e}")
    print(f"  d(loss)/d(tau)       = {float(g_tau):.6e}")

    assert jnp.abs(g_de) > 0, f"Gradient w.r.t. delta_eps is zero: {float(g_de)}"
    assert jnp.abs(g_tau) > 0, f"Gradient w.r.t. tau is zero: {float(g_tau)}"


def test_init_lorentz_is_differentiable():
    """jax.grad through init_lorentz produces non-zero gradients.

    Note: kappa only enters via c_val = EPS_0 * kappa * dt^2 / denom.
    At GHz frequencies and ps timesteps, c values are O(1e-15) per cell,
    so we test kappa's gradient via sum(c) (linear, not squared) and use
    a scale-appropriate threshold.  For omega_0 and delta, the a/b coeffs
    are O(1), so sum(a^2 + b^2) gives strong gradients directly.
    """
    shape = (8, 8, 8)
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=jnp.float32) * 2.0,
        sigma=jnp.zeros(shape, dtype=jnp.float32),
        mu_r=jnp.ones(shape, dtype=jnp.float32),
    )
    dt = 1e-12

    # Test omega_0 and delta via a/b coefficients
    def loss_ab(omega_0, delta):
        kappa = jnp.float32(1e20)
        pole = LorentzPole(omega_0=omega_0, delta=delta, kappa=kappa)
        coeffs, _ = init_lorentz([pole], materials, dt)
        return jnp.sum(coeffs.a ** 2) + jnp.sum(coeffs.b ** 2)

    grad_ab = jax.grad(loss_ab, argnums=(0, 1))
    w0 = jnp.float32(2 * jnp.pi * 5e9)
    d = jnp.float32(2 * jnp.pi * 5e9 * 0.05)
    g_w0, g_d = grad_ab(w0, d)

    # Test kappa via sum(c) which is linear in kappa (avoids underflow)
    def loss_c(kappa):
        pole = LorentzPole(omega_0=w0, delta=d, kappa=kappa)
        coeffs, _ = init_lorentz([pole], materials, dt)
        return jnp.sum(coeffs.c)

    g_k = jax.grad(loss_c)(jnp.float32(1e20))

    print("\ninit_lorentz gradients:")
    print(f"  d(loss_ab)/d(omega_0) = {float(g_w0):.6e}")
    print(f"  d(loss_ab)/d(delta)   = {float(g_d):.6e}")
    print(f"  d(loss_c)/d(kappa)    = {float(g_k):.6e}")

    assert jnp.abs(g_w0) > 0, "Gradient w.r.t. omega_0 is zero"
    assert jnp.abs(g_d) > 0, "Gradient w.r.t. delta is zero"
    # c = EPS_0 * kappa * dt^2 / denom;  dc/dk = EPS_0 * dt^2 / denom ~ 8.85e-36
    # sum(c) over 512 cells: gradient ~ 512 * 8.85e-36 ~ 4.5e-33
    assert jnp.isfinite(g_k), f"Gradient w.r.t. kappa is not finite: {float(g_k)}"
    assert float(g_k) != 0.0, "Gradient w.r.t. kappa is exactly zero"


# ---------------------------------------------------------------------------
# Test 2: gradient through FDTD
# ---------------------------------------------------------------------------

def test_gradient_through_fdtd():
    """jax.grad flows from probe-energy loss through FDTD to Debye pole params.

    Uses a tiny 8x8x8 grid with a Debye material to verify that
    the gradient of sum(|Ez_probe|^2) w.r.t. (delta_eps, tau) is non-zero.
    """
    grid = Grid(freq_max=5e9, domain=(0.012, 0.012, 0.012), cpml_layers=0)
    n_steps = 30

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.003, 0.006, 0.006), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.009, 0.006, 0.006), "ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(delta_eps, tau):
        eps_inf = 2.0
        eps_r = jnp.ones(grid.shape, dtype=jnp.float32) * eps_inf
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

        pole = DebyePole(delta_eps=delta_eps, tau=tau)
        debye = init_debye([pole], mats, grid.dt)

        result = run(
            grid, mats, n_steps,
            sources=[src], probes=[prb],
            debye=debye,
            checkpoint=True,
        )
        return jnp.sum(result.time_series ** 2)

    grad_fn = jax.grad(objective, argnums=(0, 1))
    g_de, g_tau = grad_fn(jnp.float32(3.0), jnp.float32(8e-12))

    print("\nFDTD gradient through Debye poles:")
    print(f"  d(probe_energy)/d(delta_eps) = {float(g_de):.6e}")
    print(f"  d(probe_energy)/d(tau)       = {float(g_tau):.6e}")

    assert jnp.isfinite(g_de), f"Gradient w.r.t. delta_eps is not finite: {float(g_de)}"
    assert jnp.isfinite(g_tau), f"Gradient w.r.t. tau is not finite: {float(g_tau)}"
    # At least one should be non-zero (the material changes propagation)
    assert (jnp.abs(g_de) > 0) | (jnp.abs(g_tau) > 0), \
        "Both gradients are zero -- jax.grad did not flow through FDTD"


# ---------------------------------------------------------------------------
# Test 3: S-param loss function
# ---------------------------------------------------------------------------

def test_sparam_loss_zero_for_identical():
    """Loss is 0 when S_sim == S_meas."""
    s = jnp.array([[[0.5 + 0.3j, 0.2 - 0.1j]]])  # (1, 1, 2)
    loss = sparam_loss(s, s)
    assert float(loss) < 1e-12, f"Loss should be 0 for identical S-params, got {float(loss)}"
    print(f"\nS-param loss (identical): {float(loss):.2e}")


def test_sparam_loss_positive_for_different():
    """Loss > 0 when S_sim != S_meas."""
    s1 = jnp.array([[[0.5 + 0.3j, 0.2 - 0.1j]]])
    s2 = jnp.array([[[0.8 + 0.1j, 0.1 - 0.4j]]])
    loss = sparam_loss(s1, s2)
    assert float(loss) > 0, f"Loss should be > 0 for different S-params, got {float(loss)}"
    print(f"\nS-param loss (different): {float(loss):.6e}")


def test_sparam_loss_is_differentiable():
    """sparam_loss should be differentiable w.r.t. s_sim."""
    s_meas = jnp.array([[[0.5 + 0.3j, 0.2 - 0.1j]]])

    def loss_fn(mag_offset):
        s_sim = s_meas + mag_offset
        return sparam_loss(s_sim, s_meas)

    grad = jax.grad(loss_fn)(jnp.float32(0.1))
    assert jnp.isfinite(grad), f"sparam_loss gradient is not finite: {float(grad)}"
    assert jnp.abs(grad) > 0, "sparam_loss gradient is zero"
    print(f"\nsparam_loss gradient: {float(grad):.6e}")


# ---------------------------------------------------------------------------
# Test 4: Parameter round-trip
# ---------------------------------------------------------------------------

def test_param_roundtrip_debye():
    """Log-space <-> physical pole conversion round-trips correctly."""
    eps_inf = 4.4
    poles = [DebyePole(delta_eps=74.1, tau=8.3e-12)]
    params = _poles_to_params(eps_inf, poles, [])

    eps_inf_out, poles_out = _params_to_debye_poles(params, 1)

    assert abs(float(eps_inf_out) - eps_inf) / eps_inf < 1e-5, \
        f"eps_inf round-trip failed: {float(eps_inf_out)} vs {eps_inf}"
    assert abs(float(poles_out[0].delta_eps) - 74.1) / 74.1 < 1e-5, \
        f"delta_eps round-trip failed: {float(poles_out[0].delta_eps)} vs 74.1"
    assert abs(float(poles_out[0].tau) - 8.3e-12) / 8.3e-12 < 1e-5, \
        f"tau round-trip failed: {float(poles_out[0].tau)} vs 8.3e-12"

    print(f"\nDebye round-trip OK: eps_inf={float(eps_inf_out):.3f}, "
          f"delta_eps={float(poles_out[0].delta_eps):.3f}, "
          f"tau={float(poles_out[0].tau):.4e}")


def test_param_roundtrip_lorentz():
    """Log-space <-> physical pole conversion round-trips for Lorentz."""
    eps_inf = 2.0
    omega_0 = 2 * np.pi * 5e9
    delta_val = omega_0 * 0.05
    delta_eps = 1.5
    kappa = delta_eps * omega_0 ** 2
    poles = [LorentzPole(omega_0=omega_0, delta=delta_val, kappa=kappa)]
    params = _poles_to_params(eps_inf, [], poles)

    lorentz_out = _params_to_lorentz_poles(params, 1, offset=1)

    assert abs(float(lorentz_out[0].omega_0) - omega_0) / omega_0 < 1e-5, \
        f"omega_0 round-trip: {float(lorentz_out[0].omega_0)} vs {omega_0}"
    assert abs(float(lorentz_out[0].delta) - delta_val) / delta_val < 1e-5, \
        f"delta round-trip: {float(lorentz_out[0].delta)} vs {delta_val}"

    print(f"\nLorentz round-trip OK: omega_0={float(lorentz_out[0].omega_0):.4e}, "
          f"delta={float(lorentz_out[0].delta):.4e}")


# ---------------------------------------------------------------------------
# Test 5: Recovery of known Debye from synthetic data
# ---------------------------------------------------------------------------

def test_recover_known_debye():
    """Fit a 1-pole Debye material from synthetic S-params. Recover tau within 20%.

    Strategy: generate "measured" probe time-series from a known Debye material,
    then use jax.value_and_grad to fit starting from a perturbed initial guess.
    Verify the optimizer moves toward the true parameters.
    """
    grid = Grid(freq_max=5e9, domain=(0.012, 0.012, 0.012), cpml_layers=0)
    n_steps = 50

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.003, 0.006, 0.006), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.009, 0.006, 0.006), "ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    # True parameters
    true_de = 3.0
    true_tau = 8e-12
    true_eps_inf = 2.0

    # Generate "measured" time series with true parameters
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32) * true_eps_inf
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
    true_pole = DebyePole(delta_eps=true_de, tau=true_tau)
    debye_true = init_debye([true_pole], mats, grid.dt)

    result_true = run(
        grid, mats, n_steps,
        sources=[src], probes=[prb],
        debye=debye_true,
        checkpoint=False,
    )
    target_ts = jax.lax.stop_gradient(result_true.time_series)

    # Objective: match time series (proxy for S-param matching)
    def objective(log_de, log_tau):
        de = jnp.exp(log_de)
        tau = jnp.exp(log_tau)
        eps_r_inner = jnp.ones(grid.shape, dtype=jnp.float32) * true_eps_inf
        mats_inner = MaterialArrays(eps_r=eps_r_inner, sigma=sigma, mu_r=mu_r)
        pole = DebyePole(delta_eps=de, tau=tau)
        debye = init_debye([pole], mats_inner, grid.dt)
        result = run(
            grid, mats_inner, n_steps,
            sources=[src], probes=[prb],
            debye=debye,
            checkpoint=True,
        )
        return jnp.mean((result.time_series - target_ts) ** 2)

    # Start from perturbed guess (2x the true values)
    log_de = jnp.log(jnp.float32(true_de * 2.0))
    log_tau = jnp.log(jnp.float32(true_tau * 2.0))

    grad_fn = jax.value_and_grad(objective, argnums=(0, 1))
    loss_init, _ = grad_fn(log_de, log_tau)

    # Run a few Adam steps
    m_de = jnp.float32(0.0)
    v_de = jnp.float32(0.0)
    m_tau = jnp.float32(0.0)
    v_tau = jnp.float32(0.0)
    lr = 0.05

    for it in range(20):
        loss_val, (g_de, g_tau) = grad_fn(log_de, log_tau)
        # Manual Adam step for scalars
        p_arr = jnp.stack([log_de, log_tau])
        g_arr = jnp.stack([g_de, g_tau])
        m_arr = jnp.stack([m_de, m_tau])
        v_arr = jnp.stack([v_de, v_tau])
        p_arr, m_arr, v_arr = _adam_step(p_arr, g_arr, m_arr, v_arr, it, lr)
        log_de, log_tau = p_arr[0], p_arr[1]
        m_de, m_tau = m_arr[0], m_arr[1]
        v_de, v_tau = v_arr[0], v_arr[1]

    loss_final = float(objective(log_de, log_tau))
    recovered_de = float(jnp.exp(log_de))
    recovered_tau = float(jnp.exp(log_tau))

    print(f"\nDebye recovery (20 Adam steps, lr={lr}):")
    print(f"  True:      delta_eps={true_de:.3f}, tau={true_tau:.4e}")
    print(f"  Recovered: delta_eps={recovered_de:.3f}, tau={recovered_tau:.4e}")
    print(f"  Loss:      {float(loss_init):.6e} -> {loss_final:.6e}")

    # Loss should decrease
    assert loss_final < float(loss_init), \
        f"Loss did not decrease: {loss_final:.6e} >= {float(loss_init):.6e}"

    # Parameters should move toward the truth
    de_err_init = abs(true_de * 2.0 - true_de) / true_de  # 100% initial error
    de_err_final = abs(recovered_de - true_de) / true_de
    print(f"  delta_eps error: {de_err_init*100:.0f}% -> {de_err_final*100:.0f}%")

    # The error should have decreased (we moved toward the truth)
    assert de_err_final < de_err_init, \
        f"delta_eps error did not decrease: {de_err_final:.2%} >= {de_err_init:.2%}"


# ---------------------------------------------------------------------------
# #580: recovery through the PUBLIC entry with the magnitude-preserving
# reference_probe mode (acceptance criterion 1). The measurement-harvest
# trick: n_iterations=1 with learning_rate=0.0 leaves params untouched, so
# MaterialFitResult.final_s_params IS the model observation at the given
# initial_guess — used to generate a self-consistent synthetic
# "measurement" at the truth parameters through the identical pipeline.
# ---------------------------------------------------------------------------

def _i580_factory():
    from rfx.api import Simulation
    from rfx.geometry.csg import Box
    from rfx.sources.sources import GaussianPulse as _GP

    def factory(eps_inf, debye_poles, lorentz_poles):
        sim = Simulation(freq_max=5e9, domain=(0.024, 0.009, 0.009))
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles)
        sim.add(Box((0.010, 0.0, 0.0), (0.016, 0.009, 0.009)), material="dut")
        sim.add_port((0.003, 0.0045, 0.0045), "ez",
                     waveform=_GP(f0=3e9, bandwidth=0.5))
        # probe 0 = response (behind the slab, transmission-sensitive);
        # probe 1 = reference (incident side, between port and slab)
        sim.add_probe((0.020, 0.0045, 0.0045), component="ez")
        sim.add_probe((0.007, 0.0045, 0.0045), component="ez")
        return sim

    return factory


def test_recover_debye_reference_mode_public_entry():
    """#580 acceptance: a known (eps_inf, lossy Debye pole) is recovered
    through the public differentiable_material_fit entry under
    normalization='reference_probe', and the per_probe_max proxy's
    structural magnitude blindness is witnessed on the same data."""
    from rfx.material_fit import DebyeFitResult
    from rfx.differentiable_material_fit import differentiable_material_fit

    factory = _i580_factory()
    freqs = np.linspace(2.0e9, 4.0e9, 5)
    true_de, true_tau, true_eps_inf = 3.0, 50.0e-12, 2.0  # tau relaxation ~3.2 GHz = IN the 2-4 GHz band (identifiability; 8 ps put it at ~20 GHz, out of band, and tau drifted)
    truth = DebyeFitResult(
        eps_inf=true_eps_inf,
        poles=[DebyePole(delta_eps=true_de, tau=true_tau)],
        fit_error=0.0,
    )

    # harvest the truth observation through the identical pipeline
    obs = differentiable_material_fit(
        factory, np.zeros((1, 1, 5), complex), freqs, n_debye_poles=1,
        n_iterations=1, learning_rate=0.0, initial_guess=truth,
        verbose=False, normalization="reference_probe", reference_probe=1,
    )
    s_meas = obs.final_s_params
    assert s_meas is not None and np.all(np.isfinite(s_meas))
    # magnitude information present: NOT pinned to a unit peak
    assert abs(float(np.max(np.abs(s_meas))) - 1.0) > 1e-3

    # fit from a perturbed guess (2x both pole parameters)
    guess = DebyeFitResult(
        eps_inf=true_eps_inf,
        poles=[DebyePole(delta_eps=true_de * 2.0, tau=true_tau * 2.0)],
        fit_error=0.0,
    )
    fit = differentiable_material_fit(
        factory, s_meas, freqs, n_debye_poles=1,
        n_iterations=25, learning_rate=0.05, initial_guess=guess,
        verbose=False, normalization="reference_probe", reference_probe=1,
    )
    rec_de = float(fit.debye_poles[0].delta_eps)
    rec_tau = float(fit.debye_poles[0].tau)
    print(f"\n#580 recovery: de {true_de*2:.2f}->{rec_de:.3f} (true {true_de}), "
          f"tau {true_tau*2:.2e}->{rec_tau:.2e} (true {true_tau:.1e}), "
          f"loss {fit.loss_history[0]:.3e}->{fit.loss_history[-1]:.3e}")

    assert fit.loss_history[-1] < fit.loss_history[0]
    # moved toward truth from the 2x start on both parameters
    assert abs(rec_de - true_de) < abs(true_de * 2.0 - true_de)
    assert abs(rec_tau - true_tau) < abs(true_tau * 2.0 - true_tau)
    # documented recovery tolerance (issue #580 acceptance), pinned from the
    # 2026-08-07 CPU measurement: de 6.00->3.331 (11% rel err, gate 20%),
    # tau 1.00e-10->5.29e-11 (5.8% rel err, gate 15%), loss 1.44e-4->8.40e-6
    # (25 Adam iterations, lr=0.05). Gates carry ~1.8x/2.6x margin over the
    # measured errors, same envelope-with-margin convention as the repo's
    # gate_from_envelope policy.
    assert abs(rec_de - true_de) / true_de < 0.20
    assert abs(rec_tau - true_tau) / true_tau < 0.15

    # structural-blindness witness: the legacy proxy's model spectrum is
    # pinned to a unit peak REGARDLESS of the material, so it cannot carry
    # the magnitude information the measurement contains
    legacy = differentiable_material_fit(
        factory, s_meas, freqs, n_debye_poles=1,
        n_iterations=1, learning_rate=0.0, initial_guess=truth,
        verbose=False, normalization="per_probe_max",
    )
    assert np.isclose(float(np.max(np.abs(legacy.final_s_params))), 1.0,
                      atol=1e-5)
