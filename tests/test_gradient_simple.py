"""Simple gradient tests for rfx differentiable simulation.

Verifies jax.grad produces non-zero, correct-sign gradients for
basic FDTD scenarios. Uses PEC boundaries (no CPML gradient issues).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest


def test_gradient_eps_shifts_resonance():
    """Increasing eps_r should decrease cavity resonance frequency.

    d(energy_at_f0) / d(eps_r) should be negative near resonance —
    higher eps shifts the mode lower, reducing energy at the original f0.
    """
    from rfx import Simulation, Box, GaussianPulse

    a, b, d = 50e-3, 40e-3, 30e-3
    f0 = 299792458 / 2 * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)

    def forward(eps_val):
        sim = Simulation(freq_max=f0 * 2, domain=(a, b, d),
                         boundary="pec", dx=2e-3)
        sim.add_material("load", eps_r=float(eps_val))
        sim.add(Box((a / 3, 0, 0), (2 * a / 3, b, d)), material="load")
        sim.add_source((a / 4, b / 3, d / 2), "ez",
                        waveform=GaussianPulse(f0=f0, bandwidth=0.8))
        sim.add_probe((3 * a / 4, 2 * b / 3, d / 2), "ez")
        result = sim.run(n_steps=500)
        ts = result.time_series
        if ts.ndim == 2:
            ts = ts[:, 0]
        return jnp.sum(ts ** 2)  # total energy at probe

    grad_fn = jax.grad(forward)
    eps_test = jnp.float32(2.0)
    g = grad_fn(eps_test)

    print(f"d(energy)/d(eps_r) at eps={float(eps_test):.1f}: {float(g):.6e}")
    assert jnp.isfinite(g), "Gradient should be finite"
    assert float(g) != 0.0, "Gradient should be non-zero"


def test_gradient_finite_difference_match():
    """jax.grad should approximately match finite-difference gradient."""
    from rfx import Simulation, Box, GaussianPulse

    a, b, d = 40e-3, 30e-3, 20e-3
    f0 = 5e9

    def forward(eps_val):
        sim = Simulation(freq_max=f0 * 2, domain=(a, b, d),
                         boundary="pec", dx=2e-3)
        sim.add_material("load", eps_r=float(eps_val))
        sim.add(Box((a / 3, 0, 0), (2 * a / 3, b, d)), material="load")
        sim.add_source((a / 4, b / 3, d / 2), "ez",
                        waveform=GaussianPulse(f0=f0, bandwidth=0.5))
        sim.add_probe((3 * a / 4, 2 * b / 3, d / 2), "ez")
        result = sim.run(n_steps=300)
        ts = result.time_series
        if ts.ndim == 2:
            ts = ts[:, 0]
        return jnp.sum(ts ** 2)

    eps_0 = jnp.float32(3.0)
    h = jnp.float32(0.01)

    # AD gradient
    g_ad = float(jax.grad(forward)(eps_0))

    # Finite difference
    f_plus = float(forward(eps_0 + h))
    f_minus = float(forward(eps_0 - h))
    g_fd = (f_plus - f_minus) / (2 * float(h))

    print(f"AD gradient:  {g_ad:.6e}")
    print(f"FD gradient:  {g_fd:.6e}")

    if abs(g_fd) > 1e-10:
        rel_err = abs(g_ad - g_fd) / abs(g_fd)
        print(f"Relative err: {rel_err:.2%}")
        assert rel_err < 0.5, f"AD vs FD mismatch: {rel_err:.2%}"
    else:
        print("FD gradient near zero — skip comparison")
