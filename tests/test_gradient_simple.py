"""Simple gradient tests for rfx differentiable simulation.

Verifies jax.grad produces non-zero, correct-sign gradients for
basic FDTD scenarios. Uses sim.forward(eps_override=...) which is
designed for AD (not Simulation() constructor inside jax.grad).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.gpu


def test_gradient_eps_shifts_energy():
    """Gradient of probe energy w.r.t. eps_r should be non-zero and finite."""
    from rfx import Simulation, Box, GaussianPulse

    a, b, d = 30e-3, 20e-3, 20e-3
    f0 = 5e9

    # Build simulation ONCE (outside jax.grad)
    sim = Simulation(freq_max=f0 * 2, domain=(a, b, d),
                     boundary="pec", dx=2e-3)
    sim.add_material("load", eps_r=2.0)
    sim.add(Box((a / 3, 0, 0), (2 * a / 3, b, d)), material="load")
    sim.add_source((a / 4, b / 3, d / 2), "ez",
                    waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((3 * a / 4, 2 * b / 3, d / 2), "ez")

    # Get base eps_r array
    grid = sim._build_grid()
    base_materials, *_ = sim._assemble_materials(grid)
    base_eps = base_materials.eps_r

    def forward(eps_flat):
        """Forward: eps_r array → probe energy."""
        result = sim.forward(eps_override=eps_flat, n_steps=500)
        ts = result.time_series
        if ts.ndim == 2:
            ts = ts[:, 0]
        return jnp.sum(ts ** 2)

    grad_fn = jax.grad(forward)
    g = grad_fn(base_eps)

    max_grad = float(jnp.max(jnp.abs(g)))
    mean_grad = float(jnp.mean(jnp.abs(g)))
    print("\nGradient d(energy)/d(eps_r):")
    print(f"  Max:  {max_grad:.6e}")
    print(f"  Mean: {mean_grad:.6e}")

    assert jnp.all(jnp.isfinite(g)), "Gradient should be finite"
    assert max_grad > 0, "Gradient should be non-zero"


def test_gradient_finite_difference_match():
    """jax.grad should approximately match finite-difference gradient."""
    from rfx import Simulation, Box, GaussianPulse

    a, b, d = 30e-3, 20e-3, 20e-3
    f0 = 5e9

    sim = Simulation(freq_max=f0 * 2, domain=(a, b, d),
                     boundary="pec", dx=2e-3)
    sim.add_material("load", eps_r=3.0)
    sim.add(Box((a / 3, 0, 0), (2 * a / 3, b, d)), material="load")
    sim.add_source((a / 4, b / 3, d / 2), "ez",
                    waveform=GaussianPulse(f0=f0, bandwidth=0.5))
    sim.add_probe((3 * a / 4, 2 * b / 3, d / 2), "ez")

    grid = sim._build_grid()
    base_materials, *_ = sim._assemble_materials(grid)
    base_eps = base_materials.eps_r

    def forward(eps_flat):
        result = sim.forward(eps_override=eps_flat, n_steps=300)
        ts = result.time_series
        if ts.ndim == 2:
            ts = ts[:, 0]
        return jnp.sum(ts ** 2)

    # AD gradient
    g_ad = jax.grad(forward)(base_eps)

    # Finite difference: perturb one cell in the design region
    ci, cj, ck = grid.nx // 2, grid.ny // 2, grid.nz // 2
    h = 0.01
    eps_plus = base_eps.at[ci, cj, ck].add(h)
    eps_minus = base_eps.at[ci, cj, ck].add(-h)
    f_plus = float(forward(eps_plus))
    f_minus = float(forward(eps_minus))
    g_fd = (f_plus - f_minus) / (2 * h)
    g_ad_val = float(g_ad[ci, cj, ck])

    print(f"\nAD gradient at ({ci},{cj},{ck}):  {g_ad_val:.6e}")
    print(f"FD gradient at ({ci},{cj},{ck}):  {g_fd:.6e}")

    # Nonzero-signal precondition: a dead tape (g_ad=0) or an insensitive
    # fixture (g_fd=0) must FAIL here, not silently skip the comparison.
    # Mirrors the guard at tests/test_waveguide_flux_ad.py:82. Measured
    # values on this fixture are |g_ad|,|g_fd| ~ 1.2e-6 (baseline 2026-07-20),
    # so 1e-10 is a dead-tape floor, not a physics threshold.
    assert abs(g_fd) > 1e-10, (
        f"FD slope {g_fd:.3e} is ~zero — fixture has no eps sensitivity; "
        f"rebuild it (a dead forward tape would land here)."
    )
    assert abs(g_ad_val) > 1e-10, (
        f"AD gradient {g_ad_val:.3e} is ~zero while FD={g_fd:.3e} is not — "
        f"the reverse-mode tape is dead through the eps path."
    )

    rel_err = abs(g_ad_val - g_fd) / abs(g_fd)
    print(f"Relative error: {rel_err:.2%}")
    # Tightened 0.50 -> 0.05: executed-path AD-vs-FD agreement on this real
    # (float32, PEC-cavity, eps-perturbation) geometry measures ~0.03%
    # (baseline 2026-07-20). 5% matches the suite's float32 AD-FD envelope
    # (test_waveguide_flux_ad.py:83, test_sparam_ad_end_to_end.py:304) with
    # >100x margin over the measured value, while catching a real
    # tape/coefficient regression. The old 50% band was ~1600x above the
    # measured error and would pass a badly wrong gradient.
    assert rel_err < 0.05, f"AD vs FD mismatch: {rel_err:.2%} exceeds 5%"
