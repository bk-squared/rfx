"""GPU Accuracy Validation: WR-90 Iris Filter

Runs topology optimization of an iris bandpass filter inside a WR-90
waveguide at GPU-grade resolution (dx ~ 1 mm).

Validation criteria (Pozar Ch 8, coupled-cavity theory):
  - Center frequency within 5% of analytical TE10 cavity resonance
  - Report insertion loss and 3 dB bandwidth
  - Compare with coupled-cavity analytical prediction

Reference:
  Pozar, "Microwave Engineering", 4th ed., Ch 8
  WR-90: a=22.86 mm, b=10.16 mm, f_cutoff(TE10) = c/(2a) = 6.562 GHz

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
    TopologyDesignRegion, topology_optimize,
    maximize_transmitted_energy,
)
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

# Threshold for center frequency accuracy
THRESHOLD_PCT = 5.0


def main():
    t_start = time.time()

    # --- WR-90 parameters ---
    a_wg = 22.86e-3   # broad wall
    b_wg = 10.16e-3   # narrow wall
    f_cutoff_te10 = C0 / (2 * a_wg)  # ~ 6.562 GHz

    # Design: single-cavity iris filter at X-band
    f_center = 10.0e9
    f_max = 15e9
    dx = 1.0e-3  # ~1 mm resolution
    lam_g = C0 / f_center  # free-space wavelength ~ 30 mm

    # Analytical cavity resonance for a resonant section of length L_cav
    # TE10p cavity: f = c/(2) * sqrt((1/a)^2 + (p/L)^2) for p=1
    # Target f_center -> L_cav = 1 / sqrt((2*f/c)^2 - (1/a)^2)
    L_cav = 1.0 / np.sqrt((2 * f_center / C0) ** 2 - (1.0 / a_wg) ** 2)
    f_analytical = C0 / 2 * np.sqrt((1 / a_wg) ** 2 + (1 / L_cav) ** 2)

    # Waveguide wavelength
    lam_wg = C0 / np.sqrt(f_center ** 2 - f_cutoff_te10 ** 2)

    print("=" * 60)
    print("GPU VALIDATION: WR-90 Iris Filter")
    print("=" * 60)
    print(f"WR-90          : {a_wg*1e3:.2f} x {b_wg*1e3:.2f} mm")
    print(f"TE10 cutoff    : {f_cutoff_te10/1e9:.3f} GHz")
    print(f"Design center  : {f_center/1e9:.1f} GHz")
    print(f"Cavity length  : {L_cav*1e3:.2f} mm")
    print(f"Analytical f_c : {f_analytical/1e9:.4f} GHz")
    print(f"Resolution     : dx = {dx*1e3:.1f} mm (lambda_g/{lam_g*1e3/dx/1e3:.0f})")
    print()

    # --- Build simulation ---
    wg_length = 80e-3  # total waveguide length
    dom_x = wg_length
    dom_y = a_wg
    dom_z = b_wg

    sim = Simulation(
        freq_max=f_max,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    # Input port
    port_x = dom_x * 0.12
    sim.add_port(
        (port_x, dom_y / 2, dom_z / 2),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f_center, bandwidth=0.6),
    )
    sim.add_probe((port_x, dom_y / 2, dom_z / 2), component="ez")

    # Output probe
    probe_x = dom_x * 0.88
    sim.add_probe((probe_x, dom_y / 2, dom_z / 2), component="ez")

    # --- Design region: iris section around the cavity ---
    iris_x0 = dom_x * 0.30
    iris_x1 = dom_x * 0.70
    region = TopologyDesignRegion(
        corner_lo=(iris_x0, 0, 0),
        corner_hi=(iris_x1, dom_y, dom_z),
        material_bg="air",
        material_fg="pec",
        filter_radius=dx * 1.2,
        beta_projection=1.0,
    )

    objective = maximize_transmitted_energy(output_probe_idx=-1)

    # --- Reference simulation (empty waveguide) ---
    print("Running reference (empty waveguide) ...")
    ref_result = sim.run(n_steps=2000, compute_s_params=True)

    # --- Topology optimization ---
    n_iter = 50
    print(f"\nRunning topology optimization ({n_iter} iterations) ...")
    topo_result = topology_optimize(
        sim, region, objective,
        n_iterations=n_iter,
        learning_rate=0.02,
        beta_schedule=[(0, 1.0), (12, 4.0), (25, 16.0), (40, 32.0)],
        verbose=True,
    )

    # --- Run optimized simulation for S-params ---
    print("\nRunning optimized simulation for S-params ...")
    opt_result = sim.run(n_steps=2000, compute_s_params=True)

    # --- Extract metrics ---
    f_sim_center = None
    insertion_loss = None
    bw_3db = None

    if opt_result.s_params is not None and opt_result.freqs is not None:
        freqs = np.asarray(opt_result.freqs)
        s_params = np.asarray(opt_result.s_params)

        # S21 (transmission) -- use output probe energy ratio as proxy
        # In a 2-port system, check for S21
        if s_params.shape[0] >= 2:
            s21 = s_params[1, 0, :]
        else:
            # Single port: estimate from time-domain energy ratio
            ts = np.asarray(opt_result.time_series)
            if ts.ndim == 2 and ts.shape[1] >= 2:
                # FFT-based transmission estimate
                nfft = ts.shape[0] * 4
                fft_in = np.fft.rfft(ts[:, 0], n=nfft)
                fft_out = np.fft.rfft(ts[:, 1], n=nfft)
                freqs_fft = np.fft.rfftfreq(nfft, d=opt_result.dt)
                s21_raw = fft_out / np.maximum(np.abs(fft_in), 1e-30)
                # Interpolate to S-param frequency grid
                s21 = np.interp(freqs, freqs_fft, np.abs(s21_raw)) * np.exp(
                    1j * np.interp(freqs, freqs_fft, np.angle(s21_raw)))
            else:
                s21 = np.zeros_like(freqs, dtype=complex)

        s21_db = 20 * np.log10(np.maximum(np.abs(s21), 1e-30))

        # Find center frequency (peak of S21)
        valid = (freqs > f_cutoff_te10 * 1.1) & (freqs < f_max * 0.9)
        if np.any(valid):
            peak_idx = np.argmax(s21_db * valid)
            f_sim_center = freqs[peak_idx]
            insertion_loss = -s21_db[peak_idx]

            # 3 dB bandwidth
            peak_db = s21_db[peak_idx]
            above_3db = s21_db > (peak_db - 3.0)
            above_indices = np.where(above_3db & valid)[0]
            if len(above_indices) >= 2:
                bw_3db = freqs[above_indices[-1]] - freqs[above_indices[0]]

    # --- Validation ---
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")

    if f_sim_center is not None:
        freq_err = abs(f_sim_center - f_analytical) / f_analytical * 100
        print(f"Analytical f_center : {f_analytical/1e9:.4f} GHz")
        print(f"Simulated f_center  : {f_sim_center/1e9:.4f} GHz")
        print(f"Frequency error     : {freq_err:.2f}%")
    else:
        freq_err = 100.0
        print("Could not extract center frequency from S21")

    print(f"Insertion loss      : {insertion_loss:.2f} dB" if insertion_loss is not None else "Insertion loss      : N/A")
    print(f"3 dB bandwidth      : {bw_3db/1e6:.0f} MHz" if bw_3db is not None else "3 dB bandwidth      : N/A")
    print(f"Threshold           : {THRESHOLD_PCT}% frequency error")
    print(f"Elapsed time        : {elapsed:.1f}s")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: WR-90 Iris Filter (Pozar Ch 8)", fontsize=14, fontweight="bold")

    # Panel 1: S21 with analytical marker
    ax = axes[0, 0]
    if opt_result.s_params is not None:
        f_ghz = freqs / 1e9
        ax.plot(f_ghz, s21_db, "b-", lw=1.5, label="FDTD S21")
        ax.axvline(f_analytical / 1e9, color="r", ls="--", lw=1.5,
                   label=f"Analytical f_c = {f_analytical/1e9:.2f} GHz")
        if f_sim_center:
            ax.axvline(f_sim_center / 1e9, color="g", ls=":", lw=1.5,
                       label=f"FDTD f_c = {f_sim_center/1e9:.2f} GHz")
        ax.axhline(-3, color="gray", ls=":", alpha=0.5, label="-3 dB")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_title("Filter Transmission")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(6, 14)

    # Panel 2: Optimized iris geometry
    ax = axes[0, 1]
    density = np.asarray(topo_result.density_projected)
    if density.ndim == 3:
        density_2d = density[:, :, density.shape[2] // 2]
    else:
        density_2d = density
    im = ax.imshow(density_2d.T, origin="lower", cmap="binary_r", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Material density (1=PEC)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Optimized Iris Geometry (z-midplane)")

    # Panel 3: Loss convergence
    ax = axes[1, 0]
    ax.plot(topo_result.history, "b.-", lw=1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (-transmitted energy)")
    ax.set_title("Optimization Convergence")
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    verdict = "PASS" if freq_err < THRESHOLD_PCT else "FAIL"
    lines = [
        "WR-90 Iris Filter Validation",
        "-" * 35,
        f"WR-90: {a_wg*1e3:.2f} x {b_wg*1e3:.2f} mm",
        f"TE10 cutoff: {f_cutoff_te10/1e9:.3f} GHz",
        f"dx = {dx*1e3:.1f} mm",
        "",
        f"Analytical f_c : {f_analytical/1e9:.4f} GHz",
        f"FDTD f_c       : {f_sim_center/1e9:.4f} GHz" if f_sim_center else "FDTD f_c       : N/A",
        f"Freq error     : {freq_err:.2f}%",
        f"Insertion loss : {insertion_loss:.2f} dB" if insertion_loss is not None else "Insertion loss : N/A",
        f"3 dB BW        : {bw_3db/1e6:.0f} MHz" if bw_3db else "3 dB BW        : N/A",
        "",
        f"Criterion: freq error < {THRESHOLD_PCT}%",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "02_filter_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    passed = freq_err < THRESHOLD_PCT
    if passed:
        print(f"\nPASS: Center frequency error {freq_err:.2f}% < {THRESHOLD_PCT}%")
        sys.exit(0)
    else:
        print(f"\nFAIL: Center frequency error {freq_err:.2f}% >= {THRESHOLD_PCT}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
