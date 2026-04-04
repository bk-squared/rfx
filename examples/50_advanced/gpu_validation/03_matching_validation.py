"""GPU Accuracy Validation: Broadband Matching Network

Validates lumped RLC matching network against analytical Smith chart theory.

Validation criteria:
  - At resonance, impedance should be purely real: |Im(Z)/Re(Z)| < 0.1
  - Optimal L value should match analytical L = 1/(omega^2 * C) within 30%
  - S11 minimum should be below -10 dB at the match frequency

Reference:
  Pozar, "Microwave Engineering", 4th ed., Ch 5 (Impedance Matching)
  Series LC resonance: f_res = 1 / (2*pi*sqrt(L*C))
  At resonance: Z_LC = R (purely real, imaginary parts cancel)

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse, LumpedRLCSpec, plot_smith
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR


def build_matching_sim(L_match, C_match, dx=0.8e-3):
    """Build a simulation with a lumped matching network."""
    f0 = 2.4e9
    dom = 0.025  # 25 mm cube

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom, dom, dom),
        boundary="pec",
        dx=dx,
    )

    center = dom / 2

    # Excitation port
    sim.add_port(
        (dom * 0.25, center, center),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((dom * 0.25, center, center), component="ez")

    # Matching network: series LC element at midpoint
    if L_match > 0 or C_match > 0:
        sim.add_lumped_rlc(
            (center, center, center),
            component="ez",
            R=0.0,
            L=L_match,
            C=C_match,
            topology="series",
        )

    # Load probe
    sim.add_probe((dom * 0.75, center, center), component="ez")

    return sim


def main():
    t_start = time.time()
    f0 = 2.4e9
    omega0 = 2 * np.pi * f0
    C_fixed = 1.0e-12  # 1 pF
    dx = 0.8e-3  # finer grid for accuracy

    # Analytical optimal inductance for series LC match at f0
    L_analytical = 1.0 / (omega0 ** 2 * C_fixed)
    f_res_check = 1.0 / (2 * np.pi * np.sqrt(L_analytical * C_fixed))

    print("=" * 60)
    print("GPU VALIDATION: Broadband Matching Network")
    print("=" * 60)
    print(f"Design frequency : {f0/1e9:.1f} GHz")
    print(f"C_fixed          : {C_fixed*1e12:.1f} pF")
    print(f"Analytical L_opt : {L_analytical*1e9:.3f} nH")
    print(f"Check: f_res     : {f_res_check/1e9:.4f} GHz (should = {f0/1e9:.1f} GHz)")
    print(f"Resolution       : dx = {dx*1e3:.1f} mm")
    print()

    # --- Parametric sweep of inductance ---
    L_values = np.linspace(0.5e-9, 15e-9, 15)
    print(f"Sweeping L = {L_values[0]*1e9:.1f} to {L_values[-1]*1e9:.1f} nH ({len(L_values)} points)")

    results = []
    s11_min_vals = []

    for i, L_val in enumerate(L_values):
        print(f"  L = {L_val*1e9:.2f} nH ({i+1}/{len(L_values)}) ...", end=" ")
        sim = build_matching_sim(L_val, C_fixed, dx=dx)
        result = sim.run(n_steps=1500, compute_s_params=True)
        results.append(result)

        if result.s_params is not None:
            s11_mag = np.abs(np.asarray(result.s_params)[0, 0, :])
            s11_db = 20 * np.log10(np.maximum(s11_mag, 1e-30))
            min_s11 = float(np.min(s11_db))
            s11_min_vals.append(min_s11)
            print(f"min S11 = {min_s11:.1f} dB")
        else:
            s11_min_vals.append(0.0)
            print("no S-params")

    s11_min_vals = np.array(s11_min_vals)

    # --- Find best match ---
    best_idx = np.argmin(s11_min_vals)
    best_L = L_values[best_idx]
    best_result = results[best_idx]

    # --- Extract impedance at match frequency ---
    Z_at_match = None
    f_match = None
    imag_real_ratio = None

    if best_result.s_params is not None and best_result.freqs is not None:
        freqs = np.asarray(best_result.freqs)
        s11_complex = np.asarray(best_result.s_params)[0, 0, :]

        # Find frequency of minimum |S11|
        s11_db = 20 * np.log10(np.maximum(np.abs(s11_complex), 1e-30))
        min_idx = np.argmin(s11_db)
        f_match = freqs[min_idx]

        # Convert S11 to impedance: Z = Z0 * (1 + S11) / (1 - S11)
        Z0 = 50.0
        s11_at_match = s11_complex[min_idx]
        Z_at_match = Z0 * (1 + s11_at_match) / (1 - s11_at_match)
        imag_real_ratio = abs(Z_at_match.imag / Z_at_match.real) if abs(Z_at_match.real) > 1e-10 else float('inf')

    # --- Validation ---
    elapsed = time.time() - t_start

    L_err_pct = abs(best_L - L_analytical) / L_analytical * 100

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Analytical L_opt   : {L_analytical*1e9:.3f} nH")
    print(f"FDTD best L        : {best_L*1e9:.3f} nH")
    print(f"L error            : {L_err_pct:.1f}%")
    print(f"Best min S11       : {s11_min_vals[best_idx]:.1f} dB")
    if f_match is not None:
        print(f"Match frequency    : {f_match/1e9:.4f} GHz")
    if Z_at_match is not None:
        print(f"Z at match         : {Z_at_match.real:.1f} + j{Z_at_match.imag:.1f} ohm")
        print(f"|Im(Z)/Re(Z)|      : {imag_real_ratio:.4f}")
    print(f"Elapsed time       : {elapsed:.1f}s")

    # --- Criteria ---
    criterion_imag = imag_real_ratio is not None and imag_real_ratio < 0.1
    criterion_s11 = s11_min_vals[best_idx] < -10.0
    criterion_L = L_err_pct < 30.0

    print(f"\nCriteria:")
    print(f"  |Im(Z)/Re(Z)| < 0.1 : {'PASS' if criterion_imag else 'FAIL'} ({imag_real_ratio:.4f})" if imag_real_ratio is not None else "  |Im(Z)/Re(Z)| < 0.1 : N/A")
    print(f"  min S11 < -10 dB     : {'PASS' if criterion_s11 else 'FAIL'} ({s11_min_vals[best_idx]:.1f} dB)")
    print(f"  L error < 30%        : {'PASS' if criterion_L else 'FAIL'} ({L_err_pct:.1f}%)")

    # --- Figures ---
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("GPU Validation: Broadband Matching Network (Pozar Ch 5)", fontsize=14, fontweight="bold")

    # Panel 1: Min S11 vs L
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(L_values * 1e9, s11_min_vals, "bo-", markersize=5)
    ax1.axhline(-10, color="r", ls="--", alpha=0.5, label="-10 dB")
    ax1.axvline(L_analytical * 1e9, color="g", ls=":", lw=2,
                label=f"Analytical L = {L_analytical*1e9:.2f} nH")
    ax1.plot(best_L * 1e9, s11_min_vals[best_idx], "r*", markersize=15,
             label=f"FDTD best: L = {best_L*1e9:.2f} nH")
    ax1.set_xlabel("Inductance L (nH)")
    ax1.set_ylabel("Min |S11| (dB)")
    ax1.set_title("Parametric Sweep: S11 vs Inductance")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: S11 frequency response at best L
    ax2 = fig.add_subplot(2, 2, 2)
    if best_result.s_params is not None and best_result.freqs is not None:
        f_ghz = np.asarray(best_result.freqs) / 1e9
        s11_best = np.asarray(best_result.s_params)[0, 0, :]
        s11_db_best = 20 * np.log10(np.maximum(np.abs(s11_best), 1e-30))
        ax2.plot(f_ghz, s11_db_best, "b-", lw=1.5)
        ax2.axhline(-10, color="r", ls="--", alpha=0.5, label="-10 dB")
        ax2.axvline(f0 / 1e9, color="g", ls=":", alpha=0.5, label=f"f0 = {f0/1e9:.1f} GHz")
        ax2.set_xlim(0.5, f0 * 2 / 1e9)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("|S11| (dB)")
    ax2.set_title(f"S11 at Best Match (L = {best_L*1e9:.2f} nH)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Smith chart
    ax3 = fig.add_subplot(2, 2, 3)
    if best_result.s_params is not None and best_result.freqs is not None:
        plot_smith(
            np.asarray(best_result.s_params)[0, 0, :],
            np.asarray(best_result.freqs),
            ax=ax3,
            markers=[f0],
            title=f"Smith Chart (L = {best_L*1e9:.2f} nH)",
        )

    # Panel 4: Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    all_pass = criterion_imag and criterion_s11 and criterion_L
    verdict = "PASS" if all_pass else "FAIL"
    lines = [
        "Matching Network Validation",
        "-" * 35,
        f"f0 = {f0/1e9:.1f} GHz, C = {C_fixed*1e12:.1f} pF",
        f"dx = {dx*1e3:.1f} mm",
        "",
        f"Analytical L : {L_analytical*1e9:.3f} nH",
        f"FDTD best L  : {best_L*1e9:.3f} nH  (err {L_err_pct:.1f}%)",
        f"Min S11      : {s11_min_vals[best_idx]:.1f} dB",
    ]
    if Z_at_match is not None:
        lines.extend([
            f"Z at match   : {Z_at_match.real:.1f} + j{Z_at_match.imag:.1f}",
            f"|Im/Re|      : {imag_real_ratio:.4f}",
        ])
    lines.extend([
        "",
        "Criteria:",
        f"  |Im(Z)/Re(Z)| < 0.1 : {'PASS' if criterion_imag else 'FAIL'}",
        f"  min S11 < -10 dB     : {'PASS' if criterion_s11 else 'FAIL'}",
        f"  L error < 30%        : {'PASS' if criterion_L else 'FAIL'}",
        "",
        f"Overall: {verdict}",
        f"Time: {elapsed:.1f}s",
    ])
    ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes, va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "03_matching_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = all_pass
    if passed:
        print(f"\nPASS: All matching criteria satisfied")
        sys.exit(0)
    else:
        print(f"\nFAIL: One or more matching criteria not satisfied")
        sys.exit(1)


if __name__ == "__main__":
    main()
