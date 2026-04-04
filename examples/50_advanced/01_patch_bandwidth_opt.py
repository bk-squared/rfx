"""Example: Patch Antenna Bandwidth Optimization via Topology Optimization

Demonstrates density-based topology optimization of a 2.4 GHz microstrip
patch antenna on FR4 substrate.  The design region covers the patch shape
(binary PEC/air) and the optimizer minimizes late-time reflected energy
as a proxy for S11.

The example uses a deliberately small grid and few iterations so that it
runs on CPU in a reasonable time.  For production use, increase grid
resolution and iteration count.

Saves: examples/50_advanced/01_patch_bandwidth_opt.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import (
    Simulation, Box, GaussianPulse,
    TopologyDesignRegion, topology_optimize,
    minimize_reflected_energy,
)

OUT_DIR = "examples/50_advanced"


def main():
    # ---- Design parameters ----
    f0 = 2.4e9
    C0 = 3e8
    eps_r_sub = 4.4
    h_sub = 1.6e-3  # substrate thickness

    # Coarse Hammerstad estimate for patch size
    W_patch = C0 / (2 * f0) * np.sqrt(2 / (eps_r_sub + 1))
    eps_eff = (eps_r_sub + 1) / 2 + (eps_r_sub - 1) / 2 / np.sqrt(1 + 12 * h_sub / W_patch)
    L_patch = C0 / (2 * f0 * np.sqrt(eps_eff))

    # ---- Small domain for CPU speed ----
    dx = 2e-3  # coarse grid
    margin = 10e-3
    dom_x = L_patch + 2 * margin
    dom_y = W_patch + 2 * margin
    dom_z = h_sub + margin

    print(f"Patch design: L={L_patch*1e3:.1f} mm, W={W_patch*1e3:.1f} mm")
    print(f"Domain: {dom_x*1e3:.0f} x {dom_y*1e3:.0f} x {dom_z*1e3:.0f} mm, dx={dx*1e3:.1f} mm")

    # ---- Build base simulation ----
    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    # Substrate
    sim.add_material("fr4", eps_r=eps_r_sub, sigma=0.02)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h_sub)), material="fr4")

    # Lumped port (feed)
    feed_x = margin + L_patch / 3
    feed_y = dom_y / 2
    feed_z = h_sub / 2
    sim.add_port(
        (feed_x, feed_y, feed_z),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((feed_x, feed_y, feed_z), component="ez")

    # ---- Design region: patch area on top of substrate ----
    patch_x0 = margin
    patch_y0 = margin
    region = TopologyDesignRegion(
        corner_lo=(patch_x0, patch_y0, h_sub - dx / 2),
        corner_hi=(patch_x0 + L_patch, patch_y0 + W_patch, h_sub + dx / 2),
        material_bg="air",
        material_fg="pec",
        filter_radius=dx * 1.5,
        beta_projection=1.0,
    )

    # ---- Objective: minimize reflected energy ----
    objective = minimize_reflected_energy(port_probe_idx=0)

    # ---- Run reference simulation (uniform 0.5 density = no patch) ----
    print("\nRunning reference simulation ...")
    ref_result = sim.run(n_steps=500, compute_s_params=True)
    ref_ts = np.asarray(ref_result.time_series[:, 0])
    ref_energy = float(np.sum(ref_ts[len(ref_ts)//2:]**2))

    # ---- Run topology optimization (small iteration count for CPU) ----
    n_iter = 30
    print(f"\nRunning topology optimization ({n_iter} iterations) ...")
    topo_result = topology_optimize(
        sim, region, objective,
        n_iterations=n_iter,
        learning_rate=0.02,
        beta_schedule=[(0, 1.0), (10, 4.0), (20, 16.0)],
        verbose=True,
    )

    # ---- Results summary ----
    init_loss = topo_result.history[0]
    final_loss = topo_result.history[-1]
    improvement = (1 - final_loss / init_loss) * 100 if init_loss != 0 else 0
    binarization = float(jnp.mean(4 * topo_result.density_projected * (1 - topo_result.density_projected)))

    print(f"\n{'='*50}")
    print(f"Patch Bandwidth Optimization Results")
    print(f"{'='*50}")
    print(f"Initial loss    : {init_loss:.6e}")
    print(f"Final loss      : {final_loss:.6e}")
    print(f"Improvement     : {improvement:.1f}%")
    print(f"Binarization    : {binarization:.3f} (0=binary, 1=gray)")
    print(f"Design region   : {topo_result.density.shape}")

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Patch Antenna Bandwidth Optimization (Topology)", fontsize=14, fontweight="bold")

    # Panel 1: Initial density (uniform 0.5)
    ax = axes[0, 0]
    init_density = 0.5 * np.ones(topo_result.density.shape[:2])
    im = ax.imshow(init_density.T, origin="lower", cmap="binary_r", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Density")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Initial Design (uniform 0.5)")

    # Panel 2: Optimized density
    ax = axes[0, 1]
    opt_density = np.asarray(topo_result.density_projected)
    if opt_density.ndim == 3:
        opt_density = opt_density[:, :, opt_density.shape[2] // 2]
    im = ax.imshow(opt_density.T, origin="lower", cmap="binary_r", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Density")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Optimized Patch Shape")

    # Panel 3: Loss convergence
    ax = axes[1, 0]
    ax.plot(topo_result.history, "b.-", lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reflected energy (loss)")
    ax.set_title(f"Convergence ({improvement:.1f}% improvement)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Beta schedule
    ax = axes[1, 1]
    ax.plot(topo_result.beta_history, "r-", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Beta (projection sharpness)")
    ax.set_title("Beta Continuation Schedule")
    ax.grid(True, alpha=0.3)

    # Summary text
    summary = (
        f"f0 = {f0/1e9:.1f} GHz | FR4 (eps_r={eps_r_sub})\n"
        f"Patch: {L_patch*1e3:.1f} x {W_patch*1e3:.1f} mm\n"
        f"dx = {dx*1e3:.1f} mm | {n_iter} iters\n"
        f"Loss: {init_loss:.2e} -> {final_loss:.2e}"
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = f"{OUT_DIR}/01_patch_bandwidth_opt.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
