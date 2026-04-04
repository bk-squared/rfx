"""Example: Waveguide Filter Inverse Design via Topology Optimization

Demonstrates topology optimization of iris geometry inside a WR-90
waveguide section for bandpass filter design.  The optimizer maximizes
transmitted energy through a design region placed between input and
output ports, effectively shaping an iris coupler.

Uses maximize_transmitted_energy as the time-domain proxy for S21
maximization.  Small grid and few iterations for CPU compatibility.

Saves: examples/50_advanced/02_waveguide_filter_inverse.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import (
    Simulation, Box, GaussianPulse,
    TopologyDesignRegion, topology_optimize,
    maximize_transmitted_energy,
)

OUT_DIR = "examples/50_advanced"


def main():
    # ---- WR-90 waveguide parameters ----
    # WR-90: 22.86 mm x 10.16 mm, cutoff ~6.56 GHz for TE10
    a_wg = 22.86e-3
    b_wg = 10.16e-3
    f_center = 10e9  # center of X-band
    f_max = 15e9

    # ---- Compact domain ----
    dx = 2e-3  # coarse for CPU speed
    wg_length = 60e-3  # total waveguide length
    dom_x = wg_length
    dom_y = a_wg
    dom_z = b_wg

    print(f"WR-90 waveguide: {a_wg*1e3:.1f} x {b_wg*1e3:.1f} mm")
    print(f"Center freq: {f_center/1e9:.1f} GHz")
    print(f"Domain: {dom_x*1e3:.0f} x {dom_y*1e3:.0f} x {dom_z*1e3:.0f} mm")

    # ---- Build simulation ----
    sim = Simulation(
        freq_max=f_max,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    # Input port (left side)
    port_x = dom_x * 0.15
    sim.add_port(
        (port_x, dom_y / 2, dom_z / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f_center, bandwidth=0.6),
    )
    sim.add_probe((port_x, dom_y / 2, dom_z / 2), component="ez")

    # Output probe (right side)
    probe_x = dom_x * 0.85
    sim.add_probe((probe_x, dom_y / 2, dom_z / 2), component="ez")

    # ---- Design region: iris section in the middle ----
    iris_x0 = dom_x * 0.35
    iris_x1 = dom_x * 0.65
    region = TopologyDesignRegion(
        corner_lo=(iris_x0, 0, 0),
        corner_hi=(iris_x1, dom_y, dom_z),
        material_bg="air",
        material_fg="pec",
        filter_radius=dx * 1.2,
        beta_projection=1.0,
    )

    # ---- Objective: maximize transmitted energy at output probe ----
    objective = maximize_transmitted_energy(output_probe_idx=-1)

    # ---- Reference simulation (empty waveguide) ----
    print("\nRunning reference (empty waveguide) ...")
    ref_result = sim.run(n_steps=400, compute_s_params=True)

    # ---- Topology optimization ----
    n_iter = 25
    print(f"\nRunning topology optimization ({n_iter} iterations) ...")
    topo_result = topology_optimize(
        sim, region, objective,
        n_iterations=n_iter,
        learning_rate=0.02,
        beta_schedule=[(0, 1.0), (8, 4.0), (16, 16.0)],
        verbose=True,
    )

    # ---- Results ----
    init_loss = topo_result.history[0]
    final_loss = topo_result.history[-1]
    improvement = abs(final_loss - init_loss) / (abs(init_loss) + 1e-30) * 100

    print(f"\n{'='*50}")
    print(f"Waveguide Filter Inverse Design Results")
    print(f"{'='*50}")
    print(f"Initial loss      : {init_loss:.6e}")
    print(f"Final loss        : {final_loss:.6e}")
    print(f"Transmission gain : {improvement:.1f}%")
    print(f"Design shape      : {topo_result.density.shape}")

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Waveguide Filter Inverse Design (WR-90)", fontsize=14, fontweight="bold")

    # Panel 1: Optimized iris geometry (xy mid-slice)
    ax = axes[0, 0]
    density = np.asarray(topo_result.density_projected)
    if density.ndim == 3:
        iz_mid = density.shape[2] // 2
        density_2d = density[:, :, iz_mid]
    else:
        density_2d = density
    im = ax.imshow(density_2d.T, origin="lower", cmap="binary_r", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Material density (1=PEC)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Optimized Iris Geometry (z-midplane)")

    # Panel 2: Optimized eps distribution (xz mid-slice)
    ax = axes[0, 1]
    eps_design = np.asarray(topo_result.eps_design)
    if eps_design.ndim == 3:
        iy_mid = eps_design.shape[1] // 2
        eps_2d = eps_design[:, iy_mid, :]
    else:
        eps_2d = eps_design
    im = ax.imshow(eps_2d.T, origin="lower", cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="eps_r")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    ax.set_title("Permittivity (y-midplane)")

    # Panel 3: Loss convergence
    ax = axes[1, 0]
    ax.plot(topo_result.history, "b.-", lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (-transmitted energy)")
    ax.set_title(f"Convergence ({improvement:.1f}% transmission gain)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Reference time series
    ax = axes[1, 1]
    ref_ts = np.asarray(ref_result.time_series)
    dt = ref_result.dt
    t_ns = np.arange(ref_ts.shape[0]) * dt * 1e9
    ax.plot(t_ns, ref_ts[:, 0], label="Input port", alpha=0.8)
    if ref_ts.shape[1] > 1:
        ax.plot(t_ns, ref_ts[:, 1], label="Output probe", alpha=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez amplitude")
    ax.set_title("Reference (empty) waveguide signals")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/02_waveguide_filter_inverse.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
