"""Tests for differentiable FDTD (Stage 3).

Validates that:
1. AD gradient w.r.t. eps_r matches finite-difference approximation
2. Checkpointed forward pass matches non-checkpointed
3. Checkpointed gradient matches non-checkpointed gradient
4. A gradient ascent step improves a probe-energy objective
"""

import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays, init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, make_probe, run


def test_gradient_matches_finite_diff():
    """AD gradient w.r.t. eps_r matches central finite difference."""
    grid = Grid(freq_max=5e9, domain=(0.015, 0.015, 0.015), cpml_layers=0)
    n_steps = 30

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.005, 0.0075, 0.0075), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.01, 0.0075, 0.0075), "ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(eps_r):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(grid, mats, n_steps, sources=[src], probes=[prb])
        return jnp.sum(result.time_series ** 2)

    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)

    # AD gradient
    ad_grad = jax.grad(objective)(eps_r)

    # Central FD at cells between source and probe
    h = 1e-4
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    test_cells = [
        (cx, cy, cz),
        (cx + 1, cy, cz),
        (cx, cy + 1, cz),
    ]

    for ci, cj, ck in test_cells:
        eps_p = eps_r.at[ci, cj, ck].add(h)
        eps_m = eps_r.at[ci, cj, ck].add(-h)
        fd = (float(objective(eps_p)) - float(objective(eps_m))) / (2 * h)
        ad = float(ad_grad[ci, cj, ck])

        if abs(fd) > 1e-12:
            rel_err = abs(ad - fd) / abs(fd)
            print(f"  Cell ({ci},{cj},{ck}): AD={ad:.6e}, FD={fd:.6e}, err={rel_err:.4e}")
            assert rel_err < 0.05, \
                f"Gradient mismatch at ({ci},{cj},{ck}): rel_err={rel_err:.4f}"

    # Gradient should be non-trivial
    assert float(jnp.max(jnp.abs(ad_grad))) > 1e-15, "Gradient is all zeros"
    print(f"\nGradient validated at {len(test_cells)} cells (< 5% relative error)")


def test_checkpoint_same_forward():
    """Checkpointed run produces the same forward result."""
    grid = Grid(freq_max=5e9, domain=(0.015, 0.015, 0.015), cpml_layers=0)
    materials = init_materials(grid.shape)
    n_steps = 30

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.005, 0.0075, 0.0075), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.01, 0.0075, 0.0075), "ez")

    r1 = run(grid, materials, n_steps, sources=[src], probes=[prb])
    r2 = run(grid, materials, n_steps, sources=[src], probes=[prb],
             checkpoint=True)

    diff = float(jnp.max(jnp.abs(r1.time_series - r2.time_series)))
    assert diff == 0.0, f"Checkpoint changed forward result: diff={diff}"
    print(f"\nCheckpoint forward matches: diff = {diff}")


def test_checkpoint_gradient_matches():
    """Checkpointed gradient matches non-checkpointed gradient."""
    grid = Grid(freq_max=5e9, domain=(0.015, 0.015, 0.015), cpml_layers=0)
    n_steps = 30

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.005, 0.0075, 0.0075), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.01, 0.0075, 0.0075), "ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)

    def obj(eps_r, ckpt):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        return jnp.sum(
            run(grid, mats, n_steps, sources=[src], probes=[prb],
                checkpoint=ckpt).time_series ** 2)

    g1 = jax.grad(lambda e: obj(e, False))(eps_r)
    g2 = jax.grad(lambda e: obj(e, True))(eps_r)

    diff = float(jnp.max(jnp.abs(g1 - g2)))
    print(f"\nCheckpoint grad diff: {diff:.2e}")
    assert diff < 1e-6, f"Checkpoint gradient differs: {diff}"


def test_gradient_descent_step():
    """A gradient ascent step on eps_r should increase probe energy.

    Start with uniform eps_r=2, maximise time-integrated |Ez|² at a
    downstream probe.  One gradient step should improve the objective.
    """
    grid = Grid(freq_max=5e9, domain=(0.015, 0.015, 0.015), cpml_layers=0)
    n_steps = 50

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.003, 0.0075, 0.0075), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.012, 0.0075, 0.0075), "ez")

    sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
    mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
    eps_r = 2.0 * jnp.ones(grid.shape, dtype=jnp.float32)

    def objective(eps_r):
        mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
        result = run(grid, mats, n_steps, sources=[src], probes=[prb],
                     checkpoint=True)
        return jnp.sum(result.time_series ** 2)

    val_before = float(objective(eps_r))
    grad = jax.grad(objective)(eps_r)

    # Gradient ascent step (maximise probe energy)
    lr = 0.1
    eps_r_new = jnp.clip(eps_r + lr * grad, 1.0, 10.0)
    val_after = float(objective(eps_r_new))

    print("\nGradient ascent step:")
    print(f"  Before: {val_before:.6e}")
    print(f"  After:  {val_after:.6e}")
    print(f"  |grad|_max: {float(jnp.max(jnp.abs(grad))):.6e}")

    assert val_after > val_before, \
        f"Gradient step didn't improve: {val_after:.6e} <= {val_before:.6e}"
