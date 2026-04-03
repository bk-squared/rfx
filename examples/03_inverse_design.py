"""Example 3: Inverse Design — Optimize a Matching Layer

Uses the high-level Simulation API with DesignRegion and optimize()
to minimize reflection (S11) at 4 GHz by tuning permittivity of a
thin dielectric slab in a 1D transmission problem.

The optimizer uses Adam gradient descent via jax.grad through the
differentiable FDTD forward pass.

Expected: loss decreases over ~20 Adam iterations, |S11| improves.

Saves: examples/03_inverse_design.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from rfx import Simulation, Box, DesignRegion
from rfx.optimize import optimize

# ---- Simulation setup ----
# Small domain for fast gradient iterations.
f0 = 4e9
Lx, Ly, Lz = 0.04, 0.01, 0.01
dx = 0.001

sim = Simulation(
    freq_max=8e9,
    domain=(Lx, Ly, Lz),
    boundary="pec",
    dx=dx,
)

# Lumped port on the left side drives and measures reflection.
sim.add_port(
    (0.008, Ly / 2, Lz / 2),
    component="ez",
    impedance=50.0,
)

# Probe at the right side measures transmitted field.
sim.add_probe((0.032, Ly / 2, Lz / 2), component="ez")

# ---- Design region: thin slab in the middle of the domain ----
region = DesignRegion(
    corner_lo=(0.015, 0.0, 0.0),
    corner_hi=(0.025, Ly, Lz),
    eps_range=(1.0, 6.0),
)

# ---- Objective: maximize transmitted power (minimize loss) ----
def objective(result):
    """Minimize negative transmitted power (maximize transmission)."""
    return -jnp.sum(result.time_series ** 2)

# ---- Capture initial eps_r distribution for visualization ----
grid = sim._build_grid()
materials_init, _, _, _ = sim._assemble_materials(grid)
eps_init = np.asarray(materials_init.eps_r)

# ---- Run optimization ----
print("Running inverse design optimization (20 iterations) ...")
opt = optimize(sim, region, objective, n_iters=20, lr=0.05, verbose=True)

print(f"\nInitial loss : {opt.loss_history[0]:.4e}")
print(f"Final loss   : {opt.loss_history[-1]:.4e}")
improvement = (1.0 - opt.loss_history[-1] / opt.loss_history[0]) * 100
print(f"Improvement  : {improvement:.1f}%")

# ---- Build optimized eps_r array for visualization ----
# The optimized eps lives in the design region bounding box.
lo_idx = grid.position_to_index(region.corner_lo)
hi_idx = grid.position_to_index(region.corner_hi)
eps_opt = np.asarray(materials_init.eps_r.at[
    lo_idx[0]:hi_idx[0] + 1,
    lo_idx[1]:hi_idx[1] + 1,
    lo_idx[2]:hi_idx[2] + 1,
].set(np.asarray(opt.eps_design)))

# ---- 3-panel figure ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Inverse Design: Matching Layer Optimization",
             fontsize=13, fontweight="bold")

iz_ctr = grid.nz // 2
x_mm = np.arange(grid.nx) * grid.dx * 1e3

# Panel 1: Initial eps_r (x-z cross-section at y=center)
ax = axes[0]
iy_ctr = grid.ny // 2
im1 = ax.imshow(
    eps_init[:, iy_ctr, :].T, origin="lower", cmap="viridis",
    aspect="auto", vmin=1.0, vmax=6.0,
)
fig.colorbar(im1, ax=ax, label="eps_r")
# Highlight design region
dr_x0 = lo_idx[0]
dr_x1 = hi_idx[0]
dr_z0 = lo_idx[2]
dr_z1 = hi_idx[2]
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle(
    (dr_x0, dr_z0), dr_x1 - dr_x0, dr_z1 - dr_z0,
    linewidth=2, edgecolor="red", facecolor="none",
    label="Design region",
))
ax.set_xlabel("x (cells)")
ax.set_ylabel("z (cells)")
ax.set_title("Initial eps_r distribution")
ax.legend(fontsize=8)

# Panel 2: Optimized eps_r
ax = axes[1]
im2 = ax.imshow(
    eps_opt[:, iy_ctr, :].T, origin="lower", cmap="viridis",
    aspect="auto", vmin=1.0, vmax=6.0,
)
fig.colorbar(im2, ax=ax, label="eps_r")
ax.add_patch(Rectangle(
    (dr_x0, dr_z0), dr_x1 - dr_x0, dr_z1 - dr_z0,
    linewidth=2, edgecolor="red", facecolor="none",
    label="Design region",
))
ax.set_xlabel("x (cells)")
ax.set_ylabel("z (cells)")
ax.set_title("Optimized eps_r distribution")
ax.legend(fontsize=8)

# Annotate peak eps_r value in region
peak_eps = float(np.max(np.asarray(opt.eps_design)))
ax.text(
    0.05, 0.05, f"Peak eps_r = {peak_eps:.2f}",
    transform=ax.transAxes, fontsize=9, color="white",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
)

# Panel 3: Loss convergence curve
ax = axes[2]
iters = np.arange(len(opt.loss_history))
ax.plot(iters, opt.loss_history, "b.-", lw=1.5, ms=5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss (neg. transmitted power)")
ax.set_title("Convergence")
ax.grid(True, alpha=0.3)
# Annotate improvement
ax.annotate(
    f"{improvement:.1f}% improvement",
    xy=(len(opt.loss_history) - 1, opt.loss_history[-1]),
    xytext=(0.4, 0.7), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=9, color="red",
)

plt.tight_layout()
out_path = "examples/03_inverse_design.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Plot saved: {out_path}")
