"""Example 9: Agent-Friendly RF Backend Workflow with Physics Anchor

Canonical supported workflow for using rfx as an RF research backend:
  1. define the supported-safe optimization lane
  2. run structured preflight diagnostics
  3. establish a small physics reference with forward runs
  4. launch optimize() with a built-in proxy objective in strict mode

Physics anchor
--------------
The design region sits inside an air-filled PEC guide. A uniform dielectric slab
in that region introduces an impedance mismatch, so among uniform slabs in
``eps_r ∈ [1, 4]`` the physically best transmission case is ``eps_r = 1``
(air / no mismatch). We verify this with a small forward sweep before running
optimization.

This example intentionally stays inside the currently supported-safe lane:
- PEC boundary
- explicit probes
- built-in time-domain proxy objective
- strict preflight enforcement
- forward sweep used as a physics sanity check

Saves: examples/09_agent_friendly_workflow.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import (
    Simulation,
    Box,
    DesignRegion,
    GaussianPulse,
    maximize_transmitted_energy,
    optimize,
)

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

F0 = 5e9
DOMAIN = (0.04, 0.01, 0.01)
PORT_POS = (0.008, 0.005, 0.005)
OUT_POS = (0.030, 0.005, 0.005)
REGION_LO = (0.015, 0.0, 0.0)
REGION_HI = (0.025, 0.01, 0.01)
N_STEPS = 300


def build_supported_lane_sim(*, uniform_eps_r: float | None = None) -> Simulation:
    """Construct the canonical supported-safe simulation lane.

    When ``uniform_eps_r`` is provided, a uniform dielectric slab is inserted
    into the design region for forward physics checks.
    """
    sim = Simulation(
        freq_max=10e9,
        domain=DOMAIN,
        boundary="pec",
    )
    if uniform_eps_r is not None:
        sim.add_material("design_slab", eps_r=uniform_eps_r, sigma=0.0)
        sim.add(Box(REGION_LO, REGION_HI), material="design_slab")
    sim.add_port(
        PORT_POS,
        "ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=F0, bandwidth=0.5),
    )
    sim.add_probe(PORT_POS, "ez")
    sim.add_probe(OUT_POS, "ez")
    return sim


def run_uniform_slab_energy(eps_r: float) -> tuple[float, np.ndarray]:
    """Return transmitted output energy and time traces for a uniform slab."""
    sim = build_supported_lane_sim(uniform_eps_r=eps_r)
    result = sim.run(n_steps=N_STEPS, compute_s_params=False)
    ts = np.asarray(result.time_series)
    output_energy = float(np.sum(ts[:, 1] ** 2))
    return output_energy, ts


def main() -> None:
    # -----------------------------------------------------------------------
    # Physics reference sweep
    # -----------------------------------------------------------------------
    sweep_eps = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    sweep_energy = []
    midpoint_eps = 2.5

    print("Running forward sweep for the physics anchor ...")
    for eps_r in sweep_eps:
        energy, _traces = run_uniform_slab_energy(float(eps_r))
        sweep_energy.append(energy)
        print(f"  eps_r={eps_r:.1f} -> output energy={energy:.6e}")

    mid_energy, midpoint_trace = run_uniform_slab_energy(midpoint_eps)
    air_energy = sweep_energy[0]
    best_uniform_idx = int(np.argmax(sweep_energy))
    best_uniform_eps = float(sweep_eps[best_uniform_idx])
    sweep_energy = np.asarray(sweep_energy, dtype=float)

    print(f"\nPhysics reference: best uniform slab in sweep is eps_r={best_uniform_eps:.1f}")
    print("Expected physical reference: eps_r=1.0 (matched air / no mismatch)")

    # -----------------------------------------------------------------------
    # Strict optimization in the supported-safe lane
    # -----------------------------------------------------------------------
    sim = build_supported_lane_sim()

    region = DesignRegion(
        corner_lo=REGION_LO,
        corner_hi=REGION_HI,
        eps_range=(1.0, 4.0),
    )
    obj = maximize_transmitted_energy(output_probe_idx=1)

    report = sim.preflight_optimize(region, obj, n_steps=N_STEPS)
    print("\n" + report.summary())
    if not report.strict_ok:
        raise RuntimeError("Preflight reported a strict-mode compatibility warning/error")

    print("\nRunning strict-mode optimization (20 iterations) ...")
    result = optimize(
        sim,
        region,
        obj,
        n_iters=20,
        lr=0.1,
        n_steps=N_STEPS,
        preflight_mode="strict",
        verbose=True,
    )

    loss_history = np.asarray(result.loss_history, dtype=float)
    output_energy_history = -loss_history
    optimized_energy = float(output_energy_history[-1])
    eps_design = np.asarray(result.eps_design)
    optimized_mean_eps = float(np.mean(eps_design))

    print(f"\nMidpoint uniform slab energy: {mid_energy:.6e}")
    print(f"Air reference energy:         {air_energy:.6e}")
    print(f"Optimized final energy:       {optimized_energy:.6e}")
    print(f"Optimized mean eps_r:         {optimized_mean_eps:.3f}")

    # -----------------------------------------------------------------------
    # Visualize the workflow result
    # -----------------------------------------------------------------------
    nx_d, ny_d, nz_d = eps_design.shape
    x_mm = np.linspace(region.corner_lo[0], region.corner_hi[0], nx_d) * 1e3
    z_mm = np.linspace(region.corner_lo[2], region.corner_hi[2], nz_d) * 1e3
    iy_ctr = ny_d // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Agent-Friendly RF Backend Workflow (Physics + Optimization)", fontsize=13, fontweight="bold")

    # Panel 1: physics sweep
    ax = axes[0, 0]
    ax.plot(sweep_eps, sweep_energy, "o-", lw=1.5, ms=5, label="uniform slab sweep")
    ax.axvline(1.0, color="g", ls="--", lw=1.2, label="physical optimum: air")
    ax.axvline(best_uniform_eps, color="k", ls=":", lw=1.0, label=f"sweep argmax = {best_uniform_eps:.1f}")
    ax.set_title("Step 1: physics anchor")
    ax.set_xlabel("Uniform slab eps_r")
    ax.set_ylabel("Output probe energy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: midpoint forward validation traces
    ax = axes[0, 1]
    steps = np.arange(midpoint_trace.shape[0])
    ax.plot(steps, midpoint_trace[:, 0], lw=1.0, label=f"port trace (eps={midpoint_eps:.1f})")
    ax.plot(steps, midpoint_trace[:, 1], lw=1.0, label="output trace")
    ax.set_title("Step 2: forward validation at midpoint")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Ez")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: optimized design cross-section (design region only)
    ax = axes[1, 0]
    im = ax.pcolormesh(
        x_mm,
        z_mm,
        eps_design[:, iy_ctr, :].T,
        cmap="viridis",
        shading="auto",
        vmin=1.0,
        vmax=4.0,
    )
    fig.colorbar(im, ax=ax, label="eps_r")
    ax.set_title("Step 3: optimized design region")
    ax.set_xlabel("design-region x (mm)")
    ax.set_ylabel("design-region z (mm)")
    ax.text(
        0.03,
        0.95,
        f"mean eps_r = {optimized_mean_eps:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel 4: optimization progress
    ax = axes[1, 1]
    ax.plot(np.arange(len(output_energy_history)), output_energy_history, "o-", lw=1.5, ms=4, label="optimized output energy")
    ax.axhline(mid_energy, color="r", ls="--", lw=1.2, label=f"midpoint slab = {mid_energy:.2e}")
    ax.axhline(air_energy, color="g", ls="--", lw=1.2, label=f"air reference = {air_energy:.2e}")
    ax.set_title("Step 4: strict optimization")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Output energy = -loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = "examples/09_agent_friendly_workflow.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
