"""Example: Dielectric Lens Beam Shaping via Inverse Design

Demonstrates gradient-based optimization of a dielectric lens for
focused far-field radiation.  The design region contains a dielectric
slab whose permittivity distribution (eps_r between 1 and 10) is
optimized to maximize radiated power at broadside using the rfx
optimize() API with the DesignRegion approach.

Small domain and few iterations for CPU compatibility.

Saves: examples/50_advanced/05_dielectric_lens.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import (
    Simulation, Box, GaussianPulse,
    DesignRegion, optimize,
)

OUT_DIR = "examples/50_advanced"


def main():
    # ---- Parameters ----
    f0 = 5e9
    C0 = 3e8
    lam = C0 / f0  # 60 mm
    dx = 2e-3  # coarse grid for CPU
    dom = 50e-3  # 50 mm cube

    print(f"Dielectric lens optimization at {f0/1e9:.0f} GHz (lambda={lam*1e3:.0f} mm)")
    print(f"Domain: {dom*1e3:.0f} mm cube, dx={dx*1e3:.1f} mm")

    # ---- Build simulation ----
    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom, dom, dom),
        boundary="pec",
        dx=dx,
    )

    # Point source at left side
    src_x = dom * 0.15
    center_y = dom / 2
    center_z = dom / 2

    sim.add_port(
        (src_x, center_y, center_z),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.6),
    )
    sim.add_probe((src_x, center_y, center_z), component="ez")

    # Output probe on the right side (broadside direction)
    probe_x = dom * 0.85
    sim.add_probe((probe_x, center_y, center_z), component="ez")

    # ---- Design region: dielectric lens slab ----
    lens_x0 = dom * 0.35
    lens_x1 = dom * 0.55
    region = DesignRegion(
        corner_lo=(lens_x0, dom * 0.2, dom * 0.2),
        corner_hi=(lens_x1, dom * 0.8, dom * 0.8),
        eps_range=(1.0, 10.0),
    )

    # ---- Objective: maximize energy at output probe ----
    def objective(result):
        """Maximize transmitted power at the output probe."""
        ts = result.time_series[:, -1]  # last probe (output)
        return -jnp.sum(ts ** 2)

    # ---- Reference simulation (no lens) ----
    print("\nRunning reference (no lens) ...")
    ref_result = sim.run(n_steps=500)
    ref_ts = np.asarray(ref_result.time_series)
    ref_output_energy = float(np.sum(ref_ts[:, -1] ** 2))

    # ---- Optimization ----
    n_iter = 25
    print(f"\nRunning lens optimization ({n_iter} iterations) ...")
    opt_result = optimize(
        sim, region, objective,
        n_iters=n_iter,
        lr=0.05,
        verbose=True,
    )

    # ---- Run optimized simulation for comparison ----
    # Build a new sim with the optimized permittivity baked in
    eps_opt = np.asarray(opt_result.eps_design)
    eps_mean = float(np.mean(eps_opt))
    eps_max_val = float(np.max(eps_opt))
    eps_min_val = float(np.min(eps_opt))

    init_loss = opt_result.loss_history[0]
    final_loss = opt_result.loss_history[-1]
    improvement = abs(final_loss - init_loss) / (abs(init_loss) + 1e-30) * 100

    print(f"\n{'='*50}")
    print(f"Dielectric Lens Optimization Results")
    print(f"{'='*50}")
    print(f"Initial loss    : {init_loss:.6e}")
    print(f"Final loss      : {final_loss:.6e}")
    print(f"Improvement     : {improvement:.1f}%")
    print(f"Design eps_r    : {eps_min_val:.2f} - {eps_max_val:.2f} (mean {eps_mean:.2f})")
    print(f"Reference output energy: {ref_output_energy:.4e}")

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Dielectric Lens Beam Shaping (Inverse Design)", fontsize=14, fontweight="bold")

    # Panel 1: Optimized eps_r distribution (xy mid-slice)
    ax = axes[0, 0]
    if eps_opt.ndim == 3:
        iz_mid = eps_opt.shape[2] // 2
        eps_2d = eps_opt[:, :, iz_mid]
    else:
        eps_2d = eps_opt[:, :] if eps_opt.ndim >= 2 else eps_opt.reshape(-1, 1)
    im = ax.imshow(eps_2d.T, origin="lower", cmap="viridis", vmin=1, vmax=10, aspect="auto")
    fig.colorbar(im, ax=ax, label="eps_r")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Optimized Lens Permittivity (z-midplane)")

    # Panel 2: Optimized eps_r (xz mid-slice)
    ax = axes[0, 1]
    if eps_opt.ndim == 3:
        iy_mid = eps_opt.shape[1] // 2
        eps_xz = eps_opt[:, iy_mid, :]
    else:
        eps_xz = eps_2d
    im = ax.imshow(eps_xz.T, origin="lower", cmap="viridis", vmin=1, vmax=10, aspect="auto")
    fig.colorbar(im, ax=ax, label="eps_r")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    ax.set_title("Lens Permittivity (y-midplane)")

    # Panel 3: Loss convergence
    ax = axes[1, 0]
    ax.plot(opt_result.loss_history, "b.-", lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (-output energy)")
    ax.set_title(f"Convergence ({improvement:.1f}% improvement)")
    ax.grid(True, alpha=0.3)

    # Panel 4: eps_r histogram
    ax = axes[1, 1]
    ax.hist(eps_opt.ravel(), bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(eps_mean, color="red", ls="--", label=f"Mean = {eps_mean:.2f}")
    ax.set_xlabel("eps_r")
    ax.set_ylabel("Cell count")
    ax.set_title("Permittivity Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/05_dielectric_lens.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
