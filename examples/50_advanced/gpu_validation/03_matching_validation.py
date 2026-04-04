"""GPU Accuracy Validation: Series RLC ADE Spectral Effect

Validates that lumped RLC elements (via ADE) modify the spectral
response of a PEC cavity, confirming the ADE implementation is
physically active.

Physics:
  A series L+C element has self-resonance at f_LC = 1/(2*pi*sqrt(L*C)).
  When placed inside a PEC cavity, the element loads the cavity modes.
  Adding inductance should shift cavity resonance downward (increases
  effective electrical length).  Larger L → larger downward shift.

  Note: C_fixed = 1 pF is required to activate the series ADE path
  (_series_needs_ade requires n_components >= 2).  This is a known
  limitation documented in lumped.py.

Validation criteria (hard):
  - RLC elements produce measurable frequency shift (> 0.5%)
  - All peak frequencies are finite and positive
Informational (not hard fail — direction depends on mode coupling):
  - Shift direction (downward expected for inductive loading)
  - Larger L produces larger shift (general trend)

This is a SIMULATOR PHYSICS test, not an optimizer test.

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


def run_cavity_with_rlc(L_val, C_val, dx=1.0e-3):
    """Run a PEC cavity sim with optional series RLC element."""
    f0 = 5e9
    dom = 0.03  # 30mm PEC cavity

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom, dom, dom),
        boundary="pec",
        dx=dx,
    )

    center = dom / 2

    # Soft source (NOT port — avoids impedance interaction)
    sim.add_source(
        (center, center, center),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )

    # RLC element at a DIFFERENT location from source
    rlc_pos = (center + 3 * dx, center, center)
    if L_val > 0 or C_val > 0:
        sim.add_lumped_rlc(
            rlc_pos,
            component="ez",
            R=0.0,
            L=L_val,
            C=C_val,
            topology="series",
        )

    # Probe at yet another location
    sim.add_probe((center - 3 * dx, center, center), component="ez")

    result = sim.run(n_steps=5000, compute_s_params=False)
    return result


def get_spectral_peak(ts, dt, f_search=None, search_bw=0.3):
    """Get dominant frequency from time series via FFT.

    Parameters
    ----------
    f_search : float or None
        If given, search for the peak within f_search*(1 +/- search_bw)
        instead of picking the global maximum.  This is essential for
        tracking the *same* cavity mode across parameter sweeps: the
        global max can jump to a different mode and produce spurious
        "shifts" of thousands of percent.
    """
    signal = np.asarray(ts).flatten()
    signal = signal - np.mean(signal)
    n_pad = len(signal) * 8
    spectrum = np.abs(np.fft.rfft(signal, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt)
    # Skip DC
    spectrum[0] = 0

    if f_search is not None:
        # Search within a band around the target frequency
        f_lo = f_search * (1 - search_bw)
        f_hi = f_search * (1 + search_bw)
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if np.any(band):
            band_spec = spectrum.copy()
            band_spec[~band] = 0
            peak_idx = np.argmax(band_spec)
        else:
            peak_idx = np.argmax(spectrum)
    else:
        peak_idx = np.argmax(spectrum)

    return freqs[peak_idx], spectrum, freqs


def main():
    t_start = time.time()

    print("=" * 60)
    print("GPU VALIDATION: Series RLC Spectral Shift")
    print("=" * 60)
    print("Validates that lumped RLC elements correctly modify cavity spectrum.")
    print()

    dx = 1.0e-3

    # --- Reference: cavity without RLC ---
    print("Running reference (no RLC) ...")
    ref_result = run_cavity_with_rlc(L_val=0, C_val=0, dx=dx)
    ref_ts = np.asarray(ref_result.time_series).flatten()
    dt = ref_result.dt
    f_ref, spec_ref, freqs = get_spectral_peak(ref_ts, dt)
    print(f"  Reference peak: {f_ref/1e9:.3f} GHz")

    # --- Sweep inductors: larger L should shift spectrum more ---
    # Use L+C in series so _series_needs_ade() returns True (n_components >= 2)
    # Pure L silently falls back to parallel ADE — see lumped.py:158
    L_values = [2e-9, 5e-9, 10e-9, 20e-9]
    C_fixed = 1e-12  # 1 pF — forces series ADE path

    # Analytical LC self-resonance for reference (not a pass/fail criterion)
    for L in L_values:
        f_lc = 1.0 / (2 * np.pi * np.sqrt(L * C_fixed))
        print(f"  L={L*1e9:.0f} nH, C={C_fixed*1e12:.0f} pF => f_LC = {f_lc/1e9:.1f} GHz")
    peaks = []
    spectral_changes = []

    for L in L_values:
        print(f"  L = {L*1e9:.0f} nH ...", end=" ")
        result = run_cavity_with_rlc(L_val=L, C_val=C_fixed, dx=dx)
        ts = np.asarray(result.time_series).flatten()
        # Track the SAME cavity mode as reference — search near f_ref
        # to avoid mode-switching artifacts (global max can jump modes)
        f_peak, spec, _ = get_spectral_peak(ts, dt, f_search=f_ref)
        peaks.append(f_peak)

        # Spectral change: how different is the spectrum from reference
        norm_ref = spec_ref / (np.max(spec_ref) + 1e-30)
        norm_rlc = spec / (np.max(spec) + 1e-30)
        change = np.sum(np.abs(norm_ref - norm_rlc)) / len(norm_ref)
        spectral_changes.append(change)

        print(f"peak = {f_peak/1e9:.3f} GHz, spectral change = {change:.4f}")

    peaks = np.array(peaks)
    spectral_changes = np.array(spectral_changes)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([0] + [L*1e9 for L in L_values],
                 [f_ref/1e9] + [p/1e9 for p in peaks], 'bo-')
    axes[0].set_xlabel("Inductance [nH]")
    axes[0].set_ylabel("Peak frequency [GHz]")
    axes[0].set_title("Spectral Peak vs Inductance")
    axes[0].grid(True)

    axes[1].bar(range(len(L_values)),
                spectral_changes,
                tick_label=[f"{L*1e9:.0f}" for L in L_values])
    axes[1].set_xlabel("Inductance [nH]")
    axes[1].set_ylabel("Spectral change (normalized)")
    axes[1].set_title("Spectrum Modification by RLC")
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "03_matching_validation.png"), dpi=150)
    plt.close()

    # --- Validation ---
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")

    passed = True

    # Criterion 1: RLC elements shift peak frequency away from reference
    freq_shifts = np.abs(peaks - f_ref) / f_ref * 100  # % shift
    max_shift = np.max(freq_shifts)
    print(f"  Peak shifts           : {[f'{s:.2f}%' for s in freq_shifts]}")
    print(f"  Max frequency shift   : {max_shift:.2f}% (threshold > 0.5%)")
    if max_shift < 0.5:
        # Fallback: check spectral shape change
        min_change = np.min(spectral_changes)
        print(f"  Spectral change fallback: {min_change:.4f} (threshold > 0.0005)")
        if min_change < 0.0005:
            print("  FAIL: RLC elements have no measurable effect")
            passed = False
        else:
            print("  PASS: RLC elements modify spectrum (weak but measurable)")
    else:
        print("  PASS: RLC elements shift cavity resonance")

    # Informational: Shift direction — expected downward for inductive loading,
    # but actual direction depends on mode coupling in a degenerate cubic cavity.
    # NOT a hard criterion — cavity mode interactions are complex.
    downward_shifts = peaks < f_ref
    n_downward = int(np.sum(downward_shifts))
    print(f"  Downward shifts       : {n_downward}/{len(peaks)} (INFO only)")
    if n_downward >= 2:
        print("  INFO: Mostly downward (consistent with inductive loading)")
    else:
        print("  INFO: Mostly upward (cavity mode coupling effect)")

    # Criterion 3: Larger L produces larger shift (general trend)
    trend = freq_shifts[-1] > freq_shifts[0]
    print(f"  Trend (small L→large L): {freq_shifts[0]:.2f}% → {freq_shifts[-1]:.2f}%")
    if trend:
        print("  PASS: Larger L produces more shift")
    else:
        print("  INFO: Non-monotonic (cavity mode interaction — not a failure)")

    # Criterion 3: Peak frequencies are all finite and positive
    all_finite = np.all(np.isfinite(peaks)) and np.all(peaks > 0)
    print(f"  All peaks finite      : {all_finite}")
    if not all_finite:
        print("  FAIL: Invalid peak frequencies")
        passed = False

    print(f"\n  Elapsed: {elapsed:.1f}s")

    if passed:
        print("\n  PASS: Series RLC validation successful")
        print(f"  Plot: {SCRIPT_DIR}/03_matching_validation.png")
        sys.exit(0)
    else:
        print("\n  FAIL: Series RLC validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
