"""progressive_optimize demo — dielectric block transmission.

NOT a cross-validation. Minimal API demo that shows how
``rfx.progressive_optimize`` runs inverse design at a sequence of
increasing mesh resolutions, upsampling the latent between stages.

Moved out of the cross-validation tree (now ``validation/crossval/``) in the
2026-04-20 audit — self-tests belong in ``examples/inverse_design/``, not next
to external-solver cross-validations.

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
showcase aggressive convergence.

Loss scale — why the objective is divided by a reference
--------------------------------------------------------
The source passes ``amplitude_kind="current"``, so its amplitude is a drive
current in amperes and the injected field carries a ``1/dV`` cell-volume
factor.  Raw ``mean|E_probe|^2`` is therefore ~1e10 here, not ~1.  Two
consequences, both measured on this pod (2026-09-05, 64-core CPU, 10
iterations):

1. The raw number startles a reader, so ``main()`` runs ONE extra forward at
   the coarse dx and divides the loss by that mean probe power.  The reported
   loss then starts at O(1) (-1.81 measured: the reference uses a 10-period
   run, the optimizer a longer one, so the ratio is not exactly 1).  Adam is scale-invariant, so this does not move
   the design — the per-stage percentages below are identical before and
   after normalizing.
2. The scale decides whether this demo optimizes AT ALL.  Adam's update is
   ``lr * m / (sqrt(v) + 1e-8)`` (``eps_adam``, ``rfx/optimize.py``).  Under
   the pre-1.7 per-path default (``amplitude_kind=None``) the loss was ~1e-8
   and its gradient ~1e-12, so ``eps_adam`` dominated the denominator and the
   latent barely moved:

   ===================  ==================  ==================  ==============
   convention           stage 1 improvement stage 2 improvement boundary jump
   ===================  ==================  ==================  ==============
   legacy (``None``)    +0.006 %            0.000 %             70x (mesh)
   ``"current"``        +3.86 %             +1.29 %             8 %
   normalized (now)     +3.86 %             +1.29 %             8 %
   ===================  ==================  ==================  ==============

   Stage 2 under the legacy default was frozen — identical to seven digits.
   The old headline "Loss: -2.594e-08 -> -3.720e-10" looked like a 70-fold
   change but was entirely the dx halving: the legacy per-path amplitude is
   resolution-DEPENDENT, so refining the mesh rescaled the loss.  With
   ``amplitude_kind="current"`` the injected power is resolution-independent
   and that artefact drops to 8 %.  The run now reports each stage separately
   for this reason. For a substantive inverse-design run,
use a resonance or far-field objective over many more iterations.

Usage:
    python examples/inverse_design/progressive_demo.py

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
    sim.add_source((sx, sy, 0.004), "ez", amplitude_kind="current")
    # Probe on the -z side
    sim.add_probe((sx, sy, 0.020), "ez")
    return sim


# Reference power, set in main() before the optimizer runs. Dividing the loss
# by it puts the reported value at O(1) and — the part that matters — puts the
# GRADIENT at O(1) too, clear of the ``eps_adam = 1e-8`` floor in the Adam
# update (``rfx/optimize.py``). See "Loss scale" in the module docstring.
LOSS_REF = 1.0


def reference_power(dx: float) -> float:
    """Mean probe power of the un-optimized design at ``dx``, as a float."""
    result = sim_factory(dx).run(
        num_periods=10.0, compute_s_params=False, skip_preflight=True
    )
    return float(np.mean(np.asarray(result.time_series)[:, 0] ** 2))


def objective(result):
    # Maximise AVERAGE power at the probe. ``mean`` (not ``sum``) keeps
    # the loss roughly scale-invariant across stages that have different
    # n_steps — otherwise the stage boundary shows a discontinuous jump
    # that has nothing to do with the design.
    return -jnp.mean(result.time_series[:, 0] ** 2) / LOSS_REF


def main():
    global LOSS_REF

    schedule = [
        ProgressiveStage(dx=1.0e-3, n_iters=5, lr=0.1, num_periods=10.0),
        ProgressiveStage(dx=0.5e-3, n_iters=5, lr=0.05, num_periods=10.0),
    ]

    # One extra forward at the coarse dx fixes the loss scale (see "Loss
    # scale" in the module docstring). One reference for BOTH stages, not one
    # per stage: a per-stage reference would divide the mesh-change jump away
    # and hide it, and that jump is a thing the reader should see.
    LOSS_REF = reference_power(schedule[0].dx)
    print(f"Loss reference (mean probe power, un-optimized, "
          f"dx={schedule[0].dx * 1e3:.1f} mm): {LOSS_REF:.4e}")

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
    # Report PER STAGE. A single first->last figure spans the dx change
    # between stages, so it mixes optimizer progress with a mesh-change jump
    # and reads as progress that did not happen.
    bounds = list(result.stage_boundaries)
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        first, last = result.loss_history[lo], result.loss_history[hi - 1]
        print(f"Stage {i} (dx={schedule[i - 1].dx * 1e3:.1f} mm) loss: "
              f"{first:.4f} -> {last:.4f} "
              f"({100 * (last - first) / abs(first):+.2f} %)")
    print(f"Mesh-change jump at the stage boundary: "
          f"{result.loss_history[bounds[1] - 1]:.4f} -> "
          f"{result.loss_history[bounds[1]]:.4f} (not optimizer progress)")
    print(f"Loss overall: {result.loss_history[0]:.4f} -> "
          f"{result.loss_history[-1]:.4f} (spans the mesh change; read the "
          f"per-stage lines above instead)")

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
