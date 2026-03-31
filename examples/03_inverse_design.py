"""Example 3: Inverse Design — Optimize a Matching Layer

Uses JAX autodiff to optimize the permittivity of a design region
to maximize transmission (minimize reflection) at a target frequency.

Expected: loss decreases over ~20 Adam iterations, S11 improves.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.simulation import run, make_source, make_probe, SourceSpec, ProbeSpec
from rfx.sources.sources import GaussianPulse

# Small domain for fast iteration
grid = Grid(freq_max=8e9, domain=(0.04, 0.01, 0.01), dx=0.001, cpml_layers=6)

# Source and probe
pulse = GaussianPulse(f0=4e9, bandwidth=0.5)
src = make_source(grid, (0.008, 0.005, 0.005), "ez", pulse, n_steps=150)
probe = ProbeSpec(
    i=grid.position_to_index((0.032, 0.005, 0.005))[0],
    j=grid.position_to_index((0.032, 0.005, 0.005))[1],
    k=grid.position_to_index((0.032, 0.005, 0.005))[2],
    component="ez",
)

# Fixed material arrays
sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

# Design region: cells 18-28 in x (middle of domain)
x_lo, x_hi = 18, 28


def objective(latent):
    """Maximize transmitted power by optimizing eps_r in design region."""
    # Sigmoid mapping: latent -> eps_r in [1, 6]
    eps_design = 1.0 + 5.0 * jax.nn.sigmoid(latent)
    eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
    eps_r = eps_r.at[x_lo:x_hi, :, :].set(
        eps_design[x_lo:x_hi, :, :]
    )
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
    result = run(grid, mats, 150, sources=[src], probes=[probe],
                 boundary="pec", checkpoint=True)
    # Negative sum of squared probe signal (maximize transmission)
    return -jnp.sum(result.time_series ** 2)


# Adam optimizer
latent = jnp.zeros(grid.shape, dtype=jnp.float32)
lr = 0.05
m = jnp.zeros_like(latent)
v = jnp.zeros_like(latent)
beta1, beta2, eps = 0.9, 0.999, 1e-8

losses = []
print("Running inverse design optimization...")
for i in range(20):
    loss, grad = jax.value_and_grad(objective)(latent)
    losses.append(float(loss))

    # Adam update
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** (i + 1))
    v_hat = v / (1 - beta2 ** (i + 1))
    latent = latent - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    if i % 5 == 0:
        print(f"  iter {i:3d}  loss = {loss:.6e}  |grad| = {jnp.max(jnp.abs(grad)):.3e}")

print(f"\nInitial loss: {losses[0]:.6e}")
print(f"Final loss:   {losses[-1]:.6e}")
print(f"Improvement:  {(1 - losses[-1]/losses[0]) * 100:.1f}%")

# Final eps_r distribution
eps_final = 1.0 + 5.0 * jax.nn.sigmoid(latent)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(losses, "b.-")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss (neg. transmitted power)")
ax1.set_title("Optimization Convergence")
ax1.grid(True, alpha=0.3)

# Show eps_r along x-axis at center y,z
eps_line = np.array(eps_final[:, grid.ny // 2, grid.nz // 2])
x_mm = np.arange(grid.nx) * grid.dx * 1e3
ax2.plot(x_mm, eps_line)
ax2.axvspan(x_lo * grid.dx * 1e3, x_hi * grid.dx * 1e3,
            alpha=0.2, color="orange", label="Design region")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("eps_r")
ax2.set_title("Optimized Permittivity")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/03_inverse_design.png", dpi=150)
print("Plot saved: examples/03_inverse_design.png")
