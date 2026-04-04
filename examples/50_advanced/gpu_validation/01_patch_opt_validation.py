"""GPU Accuracy Validation: Topology Optimization Pipeline

Validates that the rfx topology optimization pipeline works end-to-end:
  - jax.grad flows through the full FDTD
  - Adam updates reduce the objective
  - Density filter and projection produce valid designs

Setup:
  A small PEC cavity (30 mm cube) with a broadband point source at
  the centre.  A dielectric design region (air -> FR-4, eps_r 1..4.4)
  occupies the inner volume.  The optimizer adjusts the eps_r
  distribution to maximise total probe energy (equivalently, shift
  cavity resonance toward the source centre frequency).

  Using material_fg="fr4" (not "pec") is essential because the
  topology optimizer only interpolates eps_r.  PEC's eps_r = 1.0
  (same as air), so an air-to-PEC design region gives zero gradient.

Validation criteria:
  - Optimizer completes without error
  - Loss decreases: final loss < initial loss
  - Loss history is not flat (std > 0)

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import (
    Simulation, GaussianPulse,
    TopologyDesignRegion, topology_optimize,
)
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR


def main():
    t_start = time.time()

    # ---- Parameters ----
    f0 = 5e9
    lam = C0 / f0  # ~60 mm
    dx = 1.0e-3    # 1 mm -> lambda/60
    dom = 0.030     # 30 mm cube

    print("=" * 60)
    print("GPU VALIDATION: Topology Optimization Pipeline")
    print("=" * 60)
    print(f"Frequency   : {f0/1e9:.1f} GHz (lambda = {lam*1e3:.0f} mm)")
    print(f"Resolution  : dx = {dx*1e3:.1f} mm (lambda/{lam/dx:.0f})")
    print(f"Domain      : {dom*1e3:.0f} mm PEC cube")
    print(f"Design      : air -> fr4 (eps_r 1.0 -> 4.4)")
    print()

    # ---- Build simulation ----
    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom, dom, dom),
        boundary="pec",
        dx=dx,
    )

    centre = dom / 2.0

    # Soft source (no port loading) for clean resonance excitation
    sim.add_source(
        position=(centre, centre, centre),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )

    # Probe at the same location to observe energy build-up
    sim.add_probe(position=(centre, centre, centre), component="ez")

    # ---- Design region: inner cube, air -> fr4 ----
    margin = 0.008  # 8 mm from each wall
    region = TopologyDesignRegion(
        corner_lo=(margin, margin, margin),
        corner_hi=(dom - margin, dom - margin, dom - margin),
        material_bg="air",
        material_fg="fr4",  # eps_r=4.4 -- gives real gradient
        filter_radius=2e-3,
        beta_projection=1.0,
    )

    # ---- Objective: maximise total probe energy = minimise negative energy ----
    def neg_energy(result):
        """Negative total probe energy (minimise this to maximise energy)."""
        return -jnp.sum(result.time_series ** 2)

    # ---- Run optimizer ----
    n_iter = 10
    print(f"Running topology optimization ({n_iter} iterations) ...")
    topo_result = topology_optimize(
        sim, region, neg_energy,
        n_iterations=n_iter,
        learning_rate=0.05,
        beta_schedule=[(0, 1.0), (5, 4.0)],
        verbose=True,
    )

    # ---- Evaluate ----
    init_loss = topo_result.history[0]
    final_loss = topo_result.history[-1]
    loss_improvement = (1 - final_loss / init_loss) * 100 if init_loss != 0 else 0

    loss_decreased = final_loss < init_loss
    loss_not_flat = float(np.std(topo_result.history)) > 1e-12
    pipeline_ok = loss_decreased and loss_not_flat

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Initial loss  : {init_loss:.6e}")
    print(f"Final loss    : {final_loss:.6e}")
    print(f"Improvement   : {loss_improvement:.1f}%")
    print(f"Loss decreased: {'Yes' if loss_decreased else 'No'}")
    print(f"Loss not flat : {'Yes' if loss_not_flat else 'No'}")
    print(f"Elapsed time  : {elapsed:.1f}s")

    # ---- Figures ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: Topology Optimization Pipeline",
                 fontsize=14, fontweight="bold")

    # Panel 1: Loss convergence
    ax = axes[0, 0]
    ax.plot(topo_result.history, "b.-", lw=1.5, markersize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (negative energy)")
    ax.set_title(f"Convergence ({loss_improvement:.1f}% improvement)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Beta schedule
    ax = axes[0, 1]
    ax.plot(topo_result.beta_history, "r.-", lw=1.5, markersize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Beta (projection sharpness)")
    ax.set_title("Beta Continuation Schedule")
    ax.grid(True, alpha=0.3)

    # Panel 3: Optimised density (mid-slice)
    ax = axes[1, 0]
    opt_density = np.asarray(topo_result.density_projected)
    if opt_density.ndim == 3:
        mid_z = opt_density.shape[2] // 2
        opt_density_2d = opt_density[:, :, mid_z]
    else:
        opt_density_2d = opt_density
    im = ax.imshow(opt_density_2d.T, origin="lower", cmap="binary_r",
                   vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Density (0=air, 1=fr4)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Optimised Design Region (z mid-slice)")

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    verdict = "PASS" if pipeline_ok else "FAIL"
    lines = [
        "Topology Optimization Pipeline Validation",
        "-" * 45,
        f"Domain     : {dom*1e3:.0f} mm PEC cube",
        f"dx         : {dx*1e3:.1f} mm (lambda/{lam/dx:.0f})",
        f"Materials  : air -> fr4 (eps_r 1.0 -> 4.4)",
        f"Objective  : maximise probe energy",
        f"Iterations : {n_iter}",
        "",
        f"Initial loss  : {init_loss:.6e}",
        f"Final loss    : {final_loss:.6e}",
        f"Improvement   : {loss_improvement:.1f}%",
        f"Loss decreased: {'Yes' if loss_decreased else 'No'}",
        f"Loss not flat : {'Yes' if loss_not_flat else 'No'}",
        "",
        f"Criterion: loss decreases AND history not flat",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "01_patch_opt_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # ---- Pass/Fail ----
    if pipeline_ok:
        print(f"\nPASS: Optimizer pipeline working "
              f"(loss decreased {loss_improvement:.1f}%, "
              f"init={init_loss:.4e} -> final={final_loss:.4e})")
        sys.exit(0)
    else:
        print(f"\nFAIL: Optimizer pipeline issue "
              f"(loss_decreased={loss_decreased}, "
              f"loss_not_flat={loss_not_flat}, "
              f"improvement={loss_improvement:.1f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
