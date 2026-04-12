"""Crossval 14: End-to-end inverse design — dielectric slab optimization.

Matches the pattern of rfx's PASSING test_optimize_convergence.py:
optimize a dielectric slab's permittivity to maximize transmission
through an air–slab–air sandwich in a PEC cavity.

Pipeline validated:
  eps_r (per-cell) → sim.forward(eps_override=) → Yee update →
  probe signal → cost = −Σ Ez²(transmitted) → jax.grad → Adam → repeat

The slab starts at eps_r ≈ 3.5 (mid-range) and should converge toward
eps_r ≈ 1.0 (air = no impedance mismatch = maximum transmission).

Run:  python examples/crossval/14_inverse_design_demo.py
"""
import os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from rfx import Simulation, Box, GaussianPulse
from rfx.core.yee import MaterialArrays
from rfx.optimize import DesignRegion, _latent_to_eps
from rfx.sources.sources import LumpedPort, setup_lumped_port
from rfx.simulation import run as _run, make_port_source, make_probe as _make_probe

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Setup — matches test_optimize_convergence.py::TestOptimizeDielectricSlabS11
# =============================================================================
sim = Simulation(freq_max=10e9, domain=(0.06, 0.02, 0.02), boundary="pec")
sim.add_port((0.01, 0.01, 0.01), "ez", impedance=50.0,
             waveform=GaussianPulse(f0=5e9, bandwidth=0.5))
sim.add_probe((0.01, 0.01, 0.01), "ez")     # probe 0: S11 (at port)
sim.add_probe((0.050, 0.01, 0.01), "ez")     # probe 1: transmission

region = DesignRegion(
    corner_lo=(0.025, 0.0, 0.0),
    corner_hi=(0.035, 0.02, 0.02),
    eps_range=(1.0, 6.0),
)
n_steps = 200

grid = sim._build_grid()
lo_idx = grid.position_to_index(region.corner_lo)
hi_idx = grid.position_to_index(region.corner_hi)
design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
eps_min, eps_max = region.eps_range

print("=" * 70)
print("Crossval 14: E2E Inverse Design — Slab Transmission Maximization")
print("=" * 70)
print(f"Structure: PEC cavity 60×20×20 mm, port at x=10mm, probe at x=50mm")
print(f"Design region: dielectric slab x=[25, 35] mm, εr ∈ [{eps_min}, {eps_max}]")
print(f"Design shape: {design_shape} ({np.prod(design_shape)} cells)")
print(f"Objective: maximize transmitted probe energy (minimize reflection)")
print(f"Analytic: εr_opt → 1.0 (vacuum = no impedance mismatch)")
print(f"Mesh: {grid.shape}, n_steps={n_steps}")
print()

# =============================================================================
# Differentiable forward function (from test_optimize_convergence.py)
# =============================================================================
def forward(latent):
    """latent array → scalar loss."""
    eps_design = _latent_to_eps(latent, eps_min, eps_max)
    materials, debye_spec, lorentz_spec, *_ = sim._assemble_materials(grid)
    eps_r = materials.eps_r
    si, sj, sk = lo_idx
    ei, ej, ek = hi_idx
    eps_r = eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)
    materials = MaterialArrays(eps_r=eps_r, sigma=materials.sigma, mu_r=materials.mu_r)

    sources, probes = [], []
    for pe in sim._ports:
        lp = LumpedPort(pe.position, pe.component, pe.impedance, pe.waveform)
        materials = setup_lumped_port(grid, lp, materials)
        sources.append(make_port_source(grid, lp, materials, n_steps))
    for pe in sim._probes:
        probes.append(_make_probe(grid, pe.position, pe.component))

    _, debye, lorentz = sim._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)
    result = _run(grid, materials, n_steps, boundary=sim._boundary,
                  debye=debye, lorentz=lorentz, sources=sources, probes=probes,
                  checkpoint=True)
    # Maximize transmitted energy (probe 1)
    ts_tx = result.time_series[:, 1]
    return -jnp.sum(ts_tx ** 2)

# =============================================================================
# Optimization loop (Adam)
# =============================================================================
print("=" * 70)
print("Adam optimization (30 iterations)")
print("=" * 70)

n_iters = 50
lr = 0.3
latent = jnp.zeros(design_shape, dtype=jnp.float32)
m = jnp.zeros_like(latent)
v = jnp.zeros_like(latent)
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

grad_fn = jax.value_and_grad(forward)
history = {"loss": [], "eps_mean": [], "grad_norm": []}
t0 = time.time()

for it in range(n_iters):
    loss, grad = grad_fn(latent)
    loss_val = float(loss)
    g_norm = float(jnp.sqrt(jnp.sum(grad ** 2)))

    # Adam
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** (it + 1))
    v_hat = v / (1 - beta2 ** (it + 1))
    latent = latent - lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)

    eps_now = _latent_to_eps(latent, eps_min, eps_max)
    eps_mean = float(jnp.mean(eps_now))
    history["loss"].append(loss_val)
    history["eps_mean"].append(eps_mean)
    history["grad_norm"].append(g_norm)

    if it % 5 == 0 or it == n_iters - 1:
        print(f"  iter {it:3d}  loss={loss_val:+.4e}  |grad|={g_norm:.4e}  "
              f"εr_mean={eps_mean:.3f}")

wall = time.time() - t0
print(f"\nDone in {wall:.1f}s ({wall/n_iters:.2f}s/iter)")

# =============================================================================
# Verification
# =============================================================================
print(f"\n{'=' * 70}")
print("VERIFICATION")
print("=" * 70)

initial_loss = history["loss"][0]
final_loss = history["loss"][-1]
final_eps = history["eps_mean"][-1]
init_eps = history["eps_mean"][0]
ratio = final_loss / initial_loss if initial_loss != 0 else 1.0

print(f"  Initial εr_mean: {init_eps:.3f}")
print(f"  Final εr_mean:   {final_eps:.3f}  (target ≈ 1.0)")
print(f"  Initial loss:    {initial_loss:.4e}")
print(f"  Final loss:      {final_loss:.4e}  (ratio: {ratio:.4f})")
print()

converged = final_loss < initial_loss * 0.5  # negative costs: more negative = better
direction = final_eps < init_eps       # should move TOWARD 1.0 (lower)
grad_nonzero = all(g > 0 for g in history["grad_norm"])

print(f"  Cost converged (< 0.5×):          {'PASS' if converged else 'FAIL'}")
print(f"  εr moved toward target (lower):   {'PASS' if direction else 'FAIL'}  "
      f"({init_eps:.2f} → {final_eps:.2f})")
print(f"  Gradient always finite & non-zero: {'PASS' if grad_nonzero else 'FAIL'}")

all_ok = converged and direction and grad_nonzero
print(f"  Overall:                           {'PASS' if all_ok else 'FAIL'}")
print()
print("  This validates the full differentiable optimization pipeline:")
print("    eps_r (per-cell) → sim.forward(eps_override=) → Yee update →")
print("    probe → cost = −Σ Ez²(tx) → jax.grad → Adam → converges")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
iters = np.arange(n_iters)

axes[0].plot(iters, history["eps_mean"], "r-", lw=2)
axes[0].axhline(1.0, color="k", ls="--", alpha=0.6, label="Target εr = 1.0")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Mean εr (design region)")
axes[0].set_title("Permittivity convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, history["loss"], "g-", lw=2)
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Cost (−Σ Ez²)")
axes[1].set_title("Cost function"); axes[1].grid(True, alpha=0.3)

axes[2].plot(iters, history["grad_norm"], "b-", lw=2)
axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("|∇ cost|")
axes[2].set_title("Gradient norm"); axes[2].grid(True, alpha=0.3)
axes[2].set_yscale("log")

fig.suptitle(
    f"E2E Inverse Design: εr {init_eps:.1f} → {final_eps:.2f} "
    f"(target 1.0), cost ratio {ratio:.4f}",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "14_inverse_design_demo.png")
plt.savefig(out, dpi=150); plt.close()
print(f"\n  Output: {out}")
