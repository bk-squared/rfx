"""GPU Accuracy Validation: Series RLC Resonance Frequency

Validates that a lumped series RLC element in the FDTD simulator
produces the correct resonant frequency: f0 = 1 / (2*pi*sqrt(L*C)).

Physics:
  A series RLC circuit has a resonance where inductive and capacitive
  reactances cancel, leaving only the resistance R.  The resonant
  frequency depends only on L and C:
    f_res = 1 / (2*pi*sqrt(L*C))

  A lumped RLC element driven by a broadband pulse will ring at its
  natural resonance.  We extract this resonance from the time-domain
  probe signal via zero-padded FFT and compare with the analytical
  prediction.

  CPML boundaries are used to prevent PEC cavity modes from
  contaminating the RLC resonance measurement.

  Note: Harminv (Matrix Pencil Method) can fail with SVD non-convergence
  on RLC signals in CPML.  Zero-padded FFT with 16x padding provides
  robust and sufficient frequency resolution for this validation.

Validation criteria:
  - Resonance frequency within 5% of analytical f_res
  - Sweeping L shifts the resonance in the correct direction
    (higher L -> lower f_res)

Reference:
  Pozar, "Microwave Engineering", 4th ed., Ch 5 (Impedance Matching)
  Series LC resonance: f_res = 1 / (2*pi*sqrt(L*C))

Exit 0 on PASS, 1 on FAIL.
"""

import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR

THRESHOLD_PCT = 5.0  # max allowed frequency error


def run_rlc_sim(L_val, C_val, R_val=5.0, dx=1.0e-3, n_steps=None):
    """Run a CPML-bounded simulation with a lumped series RLC element.

    Uses CPML (absorbing) boundaries instead of PEC to avoid cavity
    modes contaminating the RLC resonance extraction.

    If n_steps is None, automatically choose enough steps for the RLC
    oscillation to develop (at least 30 periods at the lowest frequency
    component, minimum 5000 steps).
    """
    f0_est = 1.0 / (2 * np.pi * np.sqrt(L_val * C_val))
    f_max = f0_est * 3

    dom = 0.030  # 30 mm cube (enough margin for CPML)

    sim = Simulation(
        freq_max=f_max,
        domain=(dom, dom, dom),
        boundary="cpml",
        cpml_layers=8,
        dx=dx,
    )

    center = dom / 2

    # Broadband excitation port, co-located with RLC element
    sim.add_port(
        (center, center, center),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0_est, bandwidth=0.6),
    )

    # Series RLC element at the centre
    sim.add_lumped_rlc(
        (center, center, center),
        component="ez",
        R=R_val,
        L=L_val,
        C=C_val,
        topology="series",
    )

    # Probe at the RLC element location to observe the resonance
    sim.add_probe((center, center, center), component="ez")

    # Compute n_steps if not provided: ensure enough oscillation cycles
    if n_steps is None:
        # Need the simulation dt to compute required steps
        grid = sim._build_grid()
        dt = grid.dt
        # At least 30 periods at f0, minimum 5000 steps
        n_steps = max(5000, int(30.0 / (f0_est * dt)))

    result = sim.run(n_steps=n_steps, compute_s_params=False)
    return result


def extract_peak_freq_fft(time_series, dt, f_min, f_max):
    """Extract the dominant frequency via zero-padded FFT.

    Uses 16x zero-padding for high frequency resolution.  This is more
    robust than Harminv for RLC signals in CPML, which can trigger SVD
    non-convergence in the Matrix Pencil Method.

    Parameters
    ----------
    time_series : 1D array
        Probe signal.
    dt : float
        Time step in seconds.
    f_min, f_max : float
        Frequency search window in Hz.

    Returns
    -------
    f_peak : float or None
        Peak frequency in Hz.
    freqs : 1D array
        FFT frequency axis.
    spectrum : 1D array
        FFT magnitude spectrum.
    """
    signal = np.asarray(time_series).ravel()

    # Skip the first quarter (source excitation transient)
    skip = len(signal) // 4
    signal_ring = signal[skip:]

    # Remove DC offset
    signal_ring = signal_ring - np.mean(signal_ring)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(signal_ring))
    signal_windowed = signal_ring * window

    # 16x zero-padding for fine frequency resolution
    n_pad = len(signal_windowed) * 16
    spectrum = np.abs(np.fft.rfft(signal_windowed, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt)

    # Search for peak in the specified frequency band
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return None, freqs, spectrum

    # Find indices where mask is True
    masked_indices = np.where(mask)[0]
    spectrum_in_band = spectrum[masked_indices]

    peak_local_idx = np.argmax(spectrum_in_band)
    peak_idx = masked_indices[peak_local_idx]
    f_peak = freqs[peak_idx]

    # Sanity check: the peak should be significantly above noise floor
    noise_floor = np.median(spectrum_in_band)
    if spectrum[peak_idx] < noise_floor * 3.0:
        return None, freqs, spectrum

    return f_peak, freqs, spectrum


def main():
    t_start = time.time()

    C_fixed = 1.0e-12  # 1 pF
    R_fixed = 5.0      # 5 ohm (low R for sharp resonance)
    dx = 1.0e-3

    # Sweep inductance values
    L_values = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0]) * 1e-9  # nH

    print("=" * 60)
    print("GPU VALIDATION: Series RLC Resonance Frequency")
    print("=" * 60)
    print(f"C_fixed  : {C_fixed*1e12:.1f} pF")
    print(f"R_fixed  : {R_fixed:.1f} ohm")
    print(f"L values : {L_values[0]*1e9:.0f} to {L_values[-1]*1e9:.0f} nH ({len(L_values)} points)")
    print(f"dx       : {dx*1e3:.1f} mm")
    print(f"Method   : zero-padded FFT (16x padding, Hann window)")
    print()

    analytical_freqs = []
    measured_freqs = []
    freq_errors = []
    all_spectra = []

    for i, L_val in enumerate(L_values):
        f_analytical = 1.0 / (2 * np.pi * np.sqrt(L_val * C_fixed))
        analytical_freqs.append(f_analytical)

        print(f"  L = {L_val*1e9:.0f} nH -> f_analytical = {f_analytical/1e9:.3f} GHz ...", end=" ")

        try:
            result = run_rlc_sim(L_val, C_fixed, R_val=R_fixed, dx=dx)
            dt = float(result.dt)
            ts = np.asarray(result.time_series)

            f_peak, freqs_fft, spectrum = extract_peak_freq_fft(
                ts[:, 0], dt,
                f_min=f_analytical * 0.5,
                f_max=f_analytical * 1.5,
            )

            if f_peak is not None:
                err = abs(f_peak - f_analytical) / f_analytical * 100
                measured_freqs.append(f_peak)
                freq_errors.append(err)
                all_spectra.append((freqs_fft, spectrum, f_analytical, f_peak))
                print(f"f_measured = {f_peak/1e9:.3f} GHz, error = {err:.2f}%")
            else:
                measured_freqs.append(None)
                freq_errors.append(100.0)
                all_spectra.append(None)
                print("no peak found")
        except Exception as e:
            measured_freqs.append(None)
            freq_errors.append(100.0)
            all_spectra.append(None)
            print(f"ERROR: {e}")

    analytical_freqs = np.array(analytical_freqs)
    freq_errors = np.array(freq_errors)

    # --- Check monotonicity: higher L -> lower f ---
    valid_measured = [f for f in measured_freqs if f is not None]
    monotonic = True
    if len(valid_measured) >= 2:
        diffs = np.diff(valid_measured)
        monotonic = np.all(diffs < 0)  # should be strictly decreasing

    # --- Overall validation ---
    max_error = float(np.max(freq_errors))
    mean_error = float(np.mean(freq_errors))

    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"{'L (nH)':>8s}  {'f_anal (GHz)':>13s}  {'f_meas (GHz)':>13s}  {'Error (%)':>10s}")
    print("-" * 50)
    for i, L_val in enumerate(L_values):
        f_a = analytical_freqs[i] / 1e9
        f_m = measured_freqs[i] / 1e9 if measured_freqs[i] is not None else float('nan')
        print(f"{L_val*1e9:8.1f}  {f_a:13.4f}  {f_m:13.4f}  {freq_errors[i]:10.2f}")

    print(f"\nMax frequency error  : {max_error:.2f}%")
    print(f"Mean frequency error : {mean_error:.2f}%")
    print(f"Monotonic (L up, f down): {'Yes' if monotonic else 'No'}")
    print(f"Threshold            : {THRESHOLD_PCT}% per point")
    print(f"Elapsed time         : {elapsed:.1f}s")

    # --- Figures ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GPU Validation: Series RLC Resonance (Pozar Ch 5)", fontsize=14, fontweight="bold")

    # Panel 1: Analytical vs measured resonance frequency
    ax = axes[0, 0]
    ax.plot(L_values * 1e9, analytical_freqs / 1e9, "rs-", markersize=8, lw=2, label="Analytical")
    valid_L = [L_values[i] for i in range(len(L_values)) if measured_freqs[i] is not None]
    valid_f = [measured_freqs[i] for i in range(len(L_values)) if measured_freqs[i] is not None]
    ax.plot(np.array(valid_L) * 1e9, np.array(valid_f) / 1e9, "bo--", markersize=6, lw=1.5, label="FDTD (FFT peak)")
    ax.set_xlabel("Inductance L (nH)")
    ax.set_ylabel("Resonance frequency (GHz)")
    ax.set_title("f_res = 1/(2*pi*sqrt(LC))")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Frequency error vs L
    ax = axes[0, 1]
    ax.bar(L_values * 1e9, freq_errors, width=0.8, color="steelblue", alpha=0.7)
    ax.axhline(THRESHOLD_PCT, color="r", ls="--", lw=1.5, label=f"Threshold = {THRESHOLD_PCT}%")
    ax.set_xlabel("Inductance L (nH)")
    ax.set_ylabel("Frequency error (%)")
    ax.set_title("Accuracy per Sweep Point")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Spectra overlay for selected L values
    ax = axes[1, 0]
    plot_indices = [0, len(L_values) // 2, -1]
    colors = ["blue", "green", "red"]
    for ci, idx in enumerate(plot_indices):
        if all_spectra[idx] is not None:
            freqs_fft, spectrum, f_a, f_m = all_spectra[idx]
            f_ghz = freqs_fft / 1e9
            spec_db = 20 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-30))
            mask = (f_ghz > 0.5) & (f_ghz < 12)
            ax.plot(f_ghz[mask], spec_db[mask], color=colors[ci], lw=0.8, alpha=0.7,
                    label=f"L={L_values[idx]*1e9:.0f} nH")
            ax.axvline(f_a / 1e9, color=colors[ci], ls=":", alpha=0.4)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("FFT Spectra at Selected L Values")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")
    all_pass = max_error < THRESHOLD_PCT and monotonic
    verdict = "PASS" if all_pass else "FAIL"
    lines = [
        "Series RLC Resonance Validation",
        "-" * 38,
        f"C = {C_fixed*1e12:.1f} pF, R = {R_fixed:.0f} ohm",
        f"dx = {dx*1e3:.1f} mm",
        f"L sweep: {L_values[0]*1e9:.0f} - {L_values[-1]*1e9:.0f} nH ({len(L_values)} pts)",
        f"Method: FFT (16x zero-pad, Hann window)",
        "",
        f"Max freq error   : {max_error:.2f}%",
        f"Mean freq error  : {mean_error:.2f}%",
        f"Monotonic        : {'Yes' if monotonic else 'No'}",
        "",
        f"Criterion: all errors < {THRESHOLD_PCT}% AND monotonic",
        f"Verdict: {verdict}",
        f"Time: {elapsed:.1f}s",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "03_matching_validation.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")

    # --- Pass/Fail ---
    if all_pass:
        print(f"\nPASS: All RLC resonance errors < {THRESHOLD_PCT}% and monotonic")
        sys.exit(0)
    else:
        print(f"\nFAIL: max error {max_error:.2f}% (threshold {THRESHOLD_PCT}%) or not monotonic")
        sys.exit(1)


if __name__ == "__main__":
    main()
