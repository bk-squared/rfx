"""progressive_optimize demo — dielectric block transmission.

NOT a cross-validation. Minimal API demo that shows how
``rfx.progressive_optimize`` runs inverse design at a sequence of
increasing mesh resolutions, upsampling the latent between stages.

Moved from ``examples/crossval/`` in the 2026-04-20 audit — self-tests
belong in ``examples/inverse_design/``, not next to external-solver
cross-validations.

Setup (deliberately small for a fast demo):

  24 x 24 x 24 mm domain, CPML on all sides.
  Source: Gaussian pulse Ez on the +z side (center of the xy plane).
  Probe:  Ez on the -z side (center of the xy plane).
  Design region: 6 x 6 x 4 mm block centred in the domain.
  Objective: maximise time-integrated |E_probe|^2 (i.e. transmission).

Schedule: dx = 1.0 mm -> 0.5 mm. 5 Adam iterations per stage.

This is a **minimal API demonstration**: the objective (max transmission
through a vacuum-adjacent dielectric block) has a trivial optimum
(eps = 1) and the gradient stays small for initial eps near the sigmoid
midpoint. The point is to show how to wire up ``sim_factory``,
``ProgressiveStage``, and ``ProgressiveOptimizeResult`` — not to
showcase aggressive convergence. For a substantive inverse-design run,
use a resonance or far-field objective over many more iterations.

Usage:
    python examples/crossval/08_progressive_inverse_design.py

Produces ``examples/08_progressive_inverse_design.png`` with the loss
curve (stages separated by dashed verticals) and per-stage eps_design
slices. Total runtime: ~2-3 min on CPU, <30 s on an RTX 4090.
"""

from __future__ import annotations

import time
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.optimize import (
    DesignRegion,
    ProgressiveStage,
    progressive_optimize,
)


DOMAIN = (0.024, 0.024, 0.024)
REGION = DesignRegion(
    corner_lo=(0.009, 0.009, 0.010),
    corner_hi=(0.015, 0.015, 0.014),
    eps_range=(1.0, 8.0),
)


def sim_factory(dx: float) -> Simulation:
    """Build the transmission sim at the given cell size."""
    sim = Simulation(
        freq_max=12e9,
        domain=DOMAIN,
        dx=dx,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=6,
    )
    # Source on the +z side
    sx, sy = DOMAIN[0] / 2, DOMAIN[1] / 2
    sim.add_source((sx, sy, 0.004), "ez")
    # Probe on the -z side
    sim.add_probe((sx, sy, 0.020), "ez")
    return sim


def objective(result):
    # Maximise AVERAGE power at the probe. ``mean`` (not ``sum``) keeps
    # the loss roughly scale-invariant across stages that have different
    # n_steps — otherwise the stage boundary shows a discontinuous jump
    # that has nothing to do with the design.
    return -jnp.mean(result.time_series[:, 0] ** 2)


def main():
    schedule = [
        ProgressiveStage(dx=1.0e-3, n_iters=5, lr=0.1, num_periods=10.0),
        ProgressiveStage(dx=0.5e-3, n_iters=5, lr=0.05, num_periods=10.0),
    ]

    t0 = time.time()
    result = progressive_optimize(
        sim_factory, REGION, objective, schedule,
        verbose=True, skip_preflight=True,
    )
    dt_total = time.time() - t0
    print(f"\nTotal wall time: {dt_total:.1f} s "
          f"({len(result.loss_history)} iterations)")
    print(f"Stage boundaries: {result.stage_boundaries}")
    print(f"Final eps_design shape: {result.final_eps_design.shape}")
    print(f"Loss: {result.loss_history[0]:.3e} -> "
          f"{result.loss_history[-1]:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curve with stage boundaries
    ax = axes[0]
    ax.plot(result.loss_history, "o-", linewidth=1.3)
    for b in result.stage_boundaries[1:-1]:
        ax.axvline(b - 0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (= -integrated |E_probe|^2)")
    ax.set_title("progressive_optimize loss history")
    ax.grid(True, alpha=0.3)

    # Coarse-stage eps_design slice
    eps_coarse = np.asarray(result.stages[0].eps_design)
    mid = eps_coarse.shape[2] // 2
    axes[1].imshow(eps_coarse[:, :, mid].T, origin="lower",
                   cmap="viridis", vmin=1, vmax=8)
    axes[1].set_title(
        f"Stage 1 (dx=1.0mm): eps xy-slice {eps_coarse.shape}"
    )
    axes[1].set_xlabel("x cell"); axes[1].set_ylabel("y cell")

    # Fine-stage eps_design slice
    eps_fine = np.asarray(result.stages[-1].eps_design)
    mid = eps_fine.shape[2] // 2
    im = axes[2].imshow(eps_fine[:, :, mid].T, origin="lower",
                        cmap="viridis", vmin=1, vmax=8)
    axes[2].set_title(
        f"Stage {len(result.stages)} (dx=0.5mm): eps xy-slice {eps_fine.shape}"
    )
    axes[2].set_xlabel("x cell"); axes[2].set_ylabel("y cell")
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="eps_r")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "08_progressive_inverse_design.png")
    out = os.path.normpath(out)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
