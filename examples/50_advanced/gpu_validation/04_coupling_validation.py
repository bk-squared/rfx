"""GPU Accuracy Validation: Array Mutual Coupling

Validates mutual coupling decay between two antenna elements against
analytical near-field / far-field coupling theory.

Validation criteria:
  - Coupling decay rate: fit |S12| vs distance to power law |S12| ~ d^(-n)
  - For parallel dipoles: n ~ 1 (far field) to 3 (near field)
  - Fitted exponent should be in range [1, 3]
  - Coupling should decrease monotonically with distance

Reference:
  Balanis, "Antenna Theory", 4th ed., Ch 8 (Mutual Coupling)
  Near-field coupling: ~1/d^3 (reactive), far-field: ~1/d (radiating)
  Typical exponent for microstrip elements on substrate: 1.5 - 2.5

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# Valid range for coupling decay exponent
EXPONENT_MIN = 1.0
EXPONENT_MAX = 3.0


def build_array_sim(spacing, dx=0.8e-3):
    """Build a 2-element array simulation with given element spacing."""
    f0 = 2.4e9
    eps_r_sub = 4.4
    h_sub = 1.6e-3

    dom_x = spacing + 25e-3
    dom_y = 30e-3
    dom_z = h_sub + 15e-3

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    sim.add_material("fr4", eps_r=eps_r_sub, sigma=0.02)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h_sub)), material="fr4")

    x1 = 12e-3
    y_center = dom_y / 2
    z_feed = h_sub / 2

    sim.add_port(
        (x1, y_center, z_feed),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((x1, y_center, z_feed), component="ez")

    x2 = x1 + spacing
    sim.add_probe((x2, y_center, z_feed), component="ez")

    return sim


def main():
    t_start = time.time()
    f0 = 2.4e9
    lam = C0 / f0
    dx = 0.8e-3

    print("=" * 60)
    print("GPU VALIDATION: Array Mutual Coupling")
    print("=" * 60)
    print(f"Frequency   : {f0/1e9:.1f} GHz (lambda = {lam*1e3:.0f} mm)")
    print(f"Resolution  : dx = {dx*1e3:.1f} mm")
    print()

    # --- Sweep spacing from 0.15 lambda to 1.0 lambda ---
    spacings = np.linspace(0.15 * lam, 1.0 * lam, 12)
    spacing_lam = spacings / lam

    print(f"Sweeping spacing: {spacings[0]*1e3:.0f} to {spacings[-1]*1e3:.0f} mm "
          f"({spacing_lam[0]:.2f} to {spacing_lam[-1]:.2f} lambda)")

    coupling_levels = []

    for i, d in enumerate(spacings):
        print(f"  d = {d*1e3:.0f} mm ({d/lam:.2f} lam) [{i+1}/{len(spacings)}] ...", end=" ")
        sim = build_array_sim(d, dx=dx)
        result = sim.run(n_steps=2000, compute_s_params=False)

        ts = np.asarray(result.time_series)
        if ts.ndim == 2 and ts.shape[1] >= 2:
            e1 = np.sum(ts[:, 0] ** 2)
            e2 = np.sum(ts[:, 1] ** 2)
            coupling = e2 / max(e1, 1e-30)
            coupling_db = 10 * np.log10(max(coupling, 1e-30))
        else:
            coupling_db = -100.0

        coupling_levels.append(coupling_db)
        print(f"coupling = {coupling_db:.1f} dB")

    coupling_levels = np.array(coupling_levels)

    # --- Fit power-law decay: coupling_dB = A - n * 10*log10(d/d0) ---
    # In linear scale: coupling ~ d^(-n), in dB: coupling_dB = C - n * 10*log10(d)
    log_d = np.log10(spacings)
    valid = coupling_levels > -80  # exclude very weak / noise floor

    if np.sum(valid) >= 3:
        coeffs = np.polyfit(log_d[valid], coupling_levels[valid], 1)
        fitted_exponent = -coeffs[0] / 10.0  # because dB = C - n*10*log10(d)
        fitted_offset = coeffs[1]
        coupling_fitted = np.polyval(coeffs, log_d)
    else:
        fitted_exponent = 0.0
        coupling_fitted = coupling_levels

    # --- Theoretical 1/r^2 reference ---
    r_ref = spacings / spacings[0]
    theory_coupling = coupling_levels[0] - 20 * np.log10(r_ref)

    # --- Check monotonicity ---
    diffs = np.diff(coupling_levels[valid])
    monotonic = np.all(diffs <= 0.5)  # allow small fluctuations

    # --- Validation ---
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Fitted decay exponent  : {fitted_exponent:.3f}")
    print(f"Expected range         : [{EXPONENT_MIN}, {EXPONENT_MAX}]")
    print(f"Monotonic decrease     : {'Yes' if monotonic else 'No'}")
    print(f"Coupling at 0.5 lam    : {coupling_levels[np.argmin(np.abs(spacing_lam - 0.5))]:.1f} dB")
    print(f"Coupling range         : {coupling_levels.max():.1f} to {coupling_levels.min():.1f} dB")
    print(f"Elapsed time           : {elapsed:.1f}s")

    exponent_valid = EXPONENT_MIN <= fitted_exponent <= EXPONENT_MAX
    print(f"\nCriteria:")
    print(f"  Exponent in [{EXPONENT_MIN}, {EXPONENT_MAX}]  : {'PASS' if exponent_valid else 'FAIL'} ({fitted_exponent:.3f})")
    print(f"  Monotonic decrease       : {'PASS' if monotonic else 'FAIL'}")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: Array Mutual Coupling (Balanis Ch 8)", fontsize=14, fontweight="bold")

    # Panel 1: Coupling vs spacing with fit
    ax = axes[0, 0]
    ax.plot(spacing_lam, coupling_levels, "bo-", markersize=6, lw=1.5, label="FDTD")
    ax.plot(spacing_lam, coupling_fitted, "r--", alpha=0.7,
            label=f"Fit: n = {fitted_exponent:.2f}")
    ax.plot(spacing_lam, theory_coupling, "g:", alpha=0.5, label="1/d^2 reference")
    ax.set_xlabel("Element spacing (wavelengths)")
    ax.set_ylabel("Mutual coupling (dB)")
    ax.set_title("Coupling vs Spacing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Log-log power law fit
    ax = axes[0, 1]
    ax.plot(spacings * 1e3, coupling_levels, "bo-", markersize=6, lw=1.5, label="FDTD")
    ax.plot(spacings * 1e3, coupling_fitted, "r--", lw=1.5,
            label=f"Power law fit (n={fitted_exponent:.2f})")
    ax.set_xlabel("Element spacing (mm)")
    ax.set_ylabel("Mutual coupling (dB)")
    ax.set_title("Coupling vs Physical Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Time-domain signals for close and far spacing
    ax = axes[1, 0]
    for idx_label, label_txt in [(0, f"d={spacings[0]*1e3:.0f} mm (close)"),
                                  (-1, f"d={spacings[-1]*1e3:.0f} mm (far)")]:
        sim = build_array_sim(spacings[idx_label], dx=dx)
        res = sim.run(n_steps=2000, compute_s_params=False)
        ts = np.asarray(res.time_series)
        dt = res.dt
        t_ns = np.arange(ts.shape[0]) * dt * 1e9
        ax.plot(t_ns, ts[:, 1], lw=0.8, alpha=0.8, label=label_txt)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez at element 2")
    ax.set_title("Coupled Signal Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    verdict = "PASS" if (exponent_valid and monotonic) else "FAIL"
    lines = [
        "Mutual Coupling Validation",
        "-" * 35,
        f"f0 = {f0/1e9:.1f} GHz, lambda = {lam*1e3:.0f} mm",
        f"dx = {dx*1e3:.1f} mm",
        f"Spacings: {len(spacings)} points",
        "",
        f"Fitted exponent: {fitted_exponent:.3f}",
        f"Expected range : [{EXPONENT_MIN}, {EXPONENT_MAX}]",
        f"Monotonic      : {'Yes' if monotonic else 'No'}",
        "",
        "Coupling vs spacing:",
    ]
    for d, c in zip(spacings[::3], coupling_levels[::3]):
        lines.append(f"  {d*1e3:5.0f} mm ({d/lam:.2f}l) : {c:7.1f} dB")
    lines.extend([
        "",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ])
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "04_coupling_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = exponent_valid and monotonic
    if passed:
        print(f"\nPASS: Coupling decay exponent {fitted_exponent:.3f} in [{EXPONENT_MIN}, {EXPONENT_MAX}] and monotonic")
        sys.exit(0)
    else:
        print(f"\nFAIL: Exponent {fitted_exponent:.3f} (expected [{EXPONENT_MIN}, {EXPONENT_MAX}]) or not monotonic")
        sys.exit(1)


if __name__ == "__main__":
    main()
