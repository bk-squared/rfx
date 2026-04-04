"""GPU Accuracy Validation: Dielectric Lens Beam Focusing

Validates gradient-based optimization of a dielectric lens for beam
focusing using CPML open boundaries (essential for radiation tests).

Validation criteria:
  - Lens should improve energy at the output probe by > 2 dB
  - Compare with theoretical directivity gain: D ~ 4*pi*A/lambda^2
  - Optimized eps distribution should be physically reasonable (1 <= eps <= 10)

Reference:
  Balanis, "Advanced Engineering Electromagnetics", Ch 13 (Lens Antennas)
  Maximum directivity of an aperture: D = 4*pi*A_eff/lambda^2

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
    Simulation, Box, GaussianPulse,
    DesignRegion, optimize,
)
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# Minimum directivity improvement from lens (dB)
MIN_GAIN_DB = 2.0


def main():
    t_start = time.time()

    f0 = 5e9
    lam = C0 / f0  # 60 mm
    dx = 2.0e-3   # 2 mm resolution (~lambda/30, adequate for lens opt)
    dom_x = 0.12  # 120 mm — elongated for source → lens → probe
    dom_y = 0.06  # 60 mm
    dom_z = 0.06  # 60 mm

    # Lens aperture size (for theoretical directivity)
    lens_width = dom_y * 0.6  # 36 mm
    A_lens = lens_width ** 2  # square aperture area
    D_theory = 4 * np.pi * A_lens / lam ** 2
    D_theory_dB = 10 * np.log10(D_theory)

    print("=" * 60)
    print("GPU VALIDATION: Dielectric Lens Beam Focusing")
    print("=" * 60)
    print(f"Frequency       : {f0/1e9:.0f} GHz (lambda = {lam*1e3:.0f} mm)")
    print(f"Resolution      : dx = {dx*1e3:.1f} mm (lambda/{lam/dx:.0f})")
    print(f"Domain          : {dom_x*1e3:.0f} x {dom_y*1e3:.0f} x {dom_z*1e3:.0f} mm")
    print(f"Boundary        : CPML (open, no wall reflections)")
    print(f"Lens aperture   : {lens_width*1e3:.0f} mm x {lens_width*1e3:.0f} mm")
    print(f"Theoretical D   : {D_theory_dB:.1f} dBi (aperture limit)")
    print()

    # --- Build simulation with CPML (essential for radiation/lens tests) ---
    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, dom_z),
        boundary="cpml",
        cpml_layers=10,
        dx=dx,
    )

    # Soft source — no port impedance loading for clean radiation
    src_x = 0.030   # 30 mm from x=0 (well clear of CPML inner edge)
    center_y = dom_y / 2
    center_z = dom_z / 2

    sim.add_source(
        (src_x, center_y, center_z),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.6),
    )
    sim.add_probe((src_x, center_y, center_z), component="ez")

    # Output probe — behind the lens, well inside domain
    probe_x = 0.090  # 90 mm from x=0 (clear of CPML inner edge)
    sim.add_probe((probe_x, center_y, center_z), component="ez")

    # --- Reference: no lens ---
    print("Running reference (no lens) ...")
    ref_result = sim.run(n_steps=2000, compute_s_params=False)
    ref_ts = np.asarray(ref_result.time_series)
    ref_output_energy = float(np.sum(ref_ts[:, -1] ** 2))
    ref_energy_dB = 10 * np.log10(max(ref_output_energy, 1e-30))
    print(f"Reference output energy: {ref_output_energy:.4e} ({ref_energy_dB:.1f} dB)")

    # --- Design region: dielectric lens ---
    lens_x0 = 0.045   # 45 mm
    lens_x1 = 0.065   # 65 mm (20 mm thick lens)
    region = DesignRegion(
        corner_lo=(lens_x0, dom_y * 0.15, dom_z * 0.15),
        corner_hi=(lens_x1, dom_y * 0.85, dom_z * 0.85),
        eps_range=(1.0, 10.0),
    )

    def objective(result):
        """Maximize transmitted power at the output probe."""
        ts = result.time_series[:, -1]
        return -jnp.sum(ts ** 2)

    # --- Optimization ---
    n_iter = 40
    print(f"\nRunning lens optimization ({n_iter} iterations) ...")
    opt_result = optimize(
        sim, region, objective,
        n_iters=n_iter,
        lr=0.05,
        verbose=True,
    )

    # --- Evaluate optimized result ---
    eps_opt = np.asarray(opt_result.eps_design)
    eps_mean = float(np.mean(eps_opt))
    eps_max_val = float(np.max(eps_opt))
    eps_min_val = float(np.min(eps_opt))

    init_loss = opt_result.loss_history[0]
    final_loss = opt_result.loss_history[-1]

    # Energy at output is -loss
    opt_output_energy = -final_loss
    opt_init_energy = -init_loss  # optimizer starts from eps_r midpoint, NOT air

    # Gain MUST be measured against the true no-lens reference run, not the
    # optimizer's initial state (which starts at eps_r = midpoint of range,
    # already a dielectric slab).
    if ref_output_energy > 0 and abs(opt_output_energy) > 0:
        gain_vs_nolens_dB = 10 * np.log10(abs(opt_output_energy) / ref_output_energy)
    else:
        gain_vs_nolens_dB = 0.0

    # Also report optimizer convergence (final vs init) for diagnostics
    if abs(opt_init_energy) > 0 and abs(opt_output_energy) > 0:
        gain_vs_init_dB = 10 * np.log10(abs(opt_output_energy) / abs(opt_init_energy))
    else:
        gain_vs_init_dB = 0.0

    # --- Validation ---
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Reference energy (no lens) : {ref_output_energy:.4e}")
    print(f"Optimizer init energy      : {abs(opt_init_energy):.4e}")
    print(f"Optimized energy           : {abs(opt_output_energy):.4e}")
    print(f"Gain vs no-lens baseline   : {gain_vs_nolens_dB:.2f} dB")
    print(f"Gain vs optimizer init     : {gain_vs_init_dB:.2f} dB (diagnostic)")
    print(f"Theoretical D_max          : {D_theory_dB:.1f} dBi")
    print(f"Design eps_r range         : {eps_min_val:.2f} - {eps_max_val:.2f} (mean {eps_mean:.2f})")
    print(f"Criterion                  : gain vs no-lens > {MIN_GAIN_DB} dB")
    print(f"Elapsed time               : {elapsed:.1f}s")

    gain_achieved = gain_vs_nolens_dB  # only the true baseline matters
    criterion_gain = gain_achieved > MIN_GAIN_DB
    criterion_eps = eps_min_val >= 0.9 and eps_max_val <= 10.5

    print(f"\nCriteria:")
    print(f"  Energy gain > {MIN_GAIN_DB} dB : {'PASS' if criterion_gain else 'FAIL'} ({gain_achieved:.2f} dB)")
    print(f"  eps in [1, 10]       : {'PASS' if criterion_eps else 'FAIL'} ([{eps_min_val:.2f}, {eps_max_val:.2f}])")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: Dielectric Lens (CPML, Balanis Ch 13)", fontsize=14, fontweight="bold")

    # Panel 1: Optimized eps (xy mid-slice)
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
    ax.set_title("Optimized Lens eps_r (z-midplane)")

    # Panel 2: Optimized eps (xz mid-slice)
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
    ax.set_title("Lens eps_r (y-midplane)")

    # Panel 3: Loss convergence
    ax = axes[1, 0]
    loss_arr = np.array(opt_result.loss_history)
    ax.plot(-loss_arr, "b.-", lw=1.0)
    ax.axhline(ref_output_energy, color="r", ls="--", alpha=0.5, label="No-lens reference")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Output energy")
    ax.set_title(f"Convergence (gain = {gain_achieved:.2f} dB)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    verdict = "PASS" if (criterion_gain and criterion_eps) else "FAIL"
    lines = [
        "Dielectric Lens Validation",
        "-" * 35,
        f"f0 = {f0/1e9:.0f} GHz, lambda = {lam*1e3:.0f} mm",
        f"dx = {dx*1e3:.1f} mm, CPML boundary, {n_iter} iter",
        "",
        f"Reference energy : {ref_output_energy:.4e}",
        f"Optimized energy : {abs(opt_output_energy):.4e}",
        f"Energy gain      : {gain_achieved:.2f} dB",
        f"Theoretical D    : {D_theory_dB:.1f} dBi",
        "",
        f"eps_r range      : {eps_min_val:.2f} - {eps_max_val:.2f}",
        f"eps_r mean       : {eps_mean:.2f}",
        "",
        f"Criteria:",
        f"  Gain > {MIN_GAIN_DB} dB   : {'PASS' if criterion_gain else 'FAIL'}",
        f"  eps valid      : {'PASS' if criterion_eps else 'FAIL'}",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "05_lens_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = criterion_gain and criterion_eps
    if passed:
        print(f"\nPASS: Lens gain {gain_achieved:.2f} dB > {MIN_GAIN_DB} dB and eps in valid range")
        sys.exit(0)
    else:
        print(f"\nFAIL: Gain {gain_achieved:.2f} dB or eps range [{eps_min_val:.2f}, {eps_max_val:.2f}] invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()
