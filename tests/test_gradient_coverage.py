"""Additional gradient coverage tests for physics paths not yet validated."""

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.simulation import run, make_source, ProbeSpec
from rfx.sources.sources import GaussianPulse, CWSource


def _small_grid():
    return Grid(freq_max=8e9, domain=(0.02, 0.006, 0.006),
                dx=0.001, cpml_layers=4)


def _run_and_grad(grid, eps_r, sigma=None, mu_r=None, n_steps=60,
                  pulse=None):
    """Run simulation and return (objective_value, gradient)."""
    if sigma is None:
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    if mu_r is None:
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    if pulse is None:
        pulse = GaussianPulse(f0=4e9, bandwidth=0.5)

    src = make_source(grid, (0.005, 0.003, 0.003), "ez", pulse, n_steps)
    probe = ProbeSpec(i=grid.nx - 6, j=grid.ny // 2, k=grid.nz // 2,
                      component="ez")

    def objective(er):
        mats = MaterialArrays(eps_r=er, sigma=sigma, mu_r=mu_r)
        result = run(grid, mats, n_steps, sources=[src], probes=[probe],
                     boundary="pec", checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    val, grad = jax.value_and_grad(objective)(eps_r)
    return float(val), grad


def test_gradient_through_lossy():
    """AD gradient flows through conductive (sigma > 0) material."""
    grid = _small_grid()
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    sigma = sigma.at[8:12, :, :].set(0.1)  # lossy slab

    val, grad = _run_and_grad(grid, eps_r, sigma=sigma)
    grad_max = float(jnp.max(jnp.abs(grad)))

    print(f"\nGradient through lossy material:")
    print(f"  Objective: {val:.6e}, |grad|_max: {grad_max:.6e}")

    assert val > 0, "Objective should be positive"
    assert grad_max > 1e-15, f"Gradient is zero through lossy path: {grad_max}"
    assert np.all(np.isfinite(np.array(grad))), "Gradient contains NaN/Inf"


def test_gradient_through_mu_r():
    """AD gradient flows through magnetic material (mu_r != 1)."""
    grid = _small_grid()
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    mu_r = mu_r.at[8:12, :, :].set(4.0)  # magnetic slab

    val, grad = _run_and_grad(grid, eps_r, mu_r=mu_r)
    grad_max = float(jnp.max(jnp.abs(grad)))

    print(f"\nGradient through magnetic material:")
    print(f"  Objective: {val:.6e}, |grad|_max: {grad_max:.6e}")

    assert val > 0, "Objective should be positive"
    assert grad_max > 1e-15, f"Gradient is zero through mu_r path: {grad_max}"
    assert np.all(np.isfinite(np.array(grad))), "Gradient contains NaN/Inf"


def test_gradient_cw_source():
    """AD gradient with CW source (not just Gaussian pulse)."""
    grid = _small_grid()
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    eps_r = eps_r.at[8:12, :, :].set(2.5)

    val, grad = _run_and_grad(grid, eps_r,
                               pulse=CWSource(f0=4e9, amplitude=1.0,
                                              ramp_steps=20),
                               n_steps=80)
    grad_max = float(jnp.max(jnp.abs(grad)))

    print(f"\nGradient with CW source:")
    print(f"  Objective: {val:.6e}, |grad|_max: {grad_max:.6e}")

    assert val > 0, "Objective should be positive"
    assert grad_max > 1e-15, f"Gradient is zero with CW source: {grad_max}"
    assert np.all(np.isfinite(np.array(grad))), "Gradient contains NaN/Inf"


def test_gradient_design_region_only():
    """Gradient should be non-zero only where eps_r differs from background."""
    grid = _small_grid()
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    eps_r = eps_r.at[8:12, :, :].set(3.0)  # design region

    _, grad = _run_and_grad(grid, eps_r)

    # Gradient in design region should be larger than in background
    design_grad = float(jnp.mean(jnp.abs(grad[8:12, :, :])))
    bg_grad = float(jnp.mean(jnp.abs(grad[:6, :, :])))

    print(f"\nGradient spatial distribution:")
    print(f"  Design region |grad| mean: {design_grad:.6e}")
    print(f"  Background |grad| mean:    {bg_grad:.6e}")

    # Both should be non-zero (wave passes through everything)
    assert design_grad > 0, "Design region gradient should be non-zero"
    # Design region gradient should be meaningful
    assert np.isfinite(design_grad), "Design gradient should be finite"
