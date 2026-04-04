"""Example: Broadband Matching Network with Lumped RLC

Demonstrates parametric sweep of lumped L and C values for impedance
matching a load to 50 ohm at 2.4 GHz.  Uses rfx's lumped RLC elements
and parametric_sweep API, then visualizes the S11 and Smith chart
trajectory for the best match.

Saves: examples/50_advanced/03_broadband_matching.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import (
    Simulation, Box, GaussianPulse, LumpedRLCSpec,
    plot_smith,
)

OUT_DIR = "examples/50_advanced"


def build_matching_sim(L_match, C_match):
    """Build a simulation with a lumped matching network."""
    f0 = 2.4e9
    dx = 1.5e-3
    dom = 0.03  # 30 mm cube

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
    f0 = 2.4e9

    # ---- Parametric sweep of inductance ----
    L_values = np.linspace(0.5e-9, 10e-9, 8)  # 0.5 nH to 10 nH
    C_fixed = 1e-12  # 1 pF

    print(f"Sweeping L = {L_values[0]*1e9:.1f} to {L_values[-1]*1e9:.1f} nH")
    print(f"Fixed C = {C_fixed*1e12:.1f} pF, f0 = {f0/1e9:.1f} GHz")

    results = []
    s11_min_vals = []

    for i, L_val in enumerate(L_values):
        print(f"  L = {L_val*1e9:.1f} nH ({i+1}/{len(L_values)}) ...", end=" ")
        sim = build_matching_sim(L_val, C_fixed)
        result = sim.run(n_steps=600, compute_s_params=True)
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

    # Find best match
    best_idx = np.argmin(s11_min_vals)
    best_L = L_values[best_idx]
    best_result = results[best_idx]

    print(f"\n{'='*50}")
    print(f"Broadband Matching Results")
    print(f"{'='*50}")
    print(f"Best L = {best_L*1e9:.2f} nH (C = {C_fixed*1e12:.1f} pF)")
    print(f"Best min S11 = {s11_min_vals[best_idx]:.1f} dB")
    print(f"Resonance f0 = 1/(2*pi*sqrt(LC)) = {1/(2*np.pi*np.sqrt(best_L*C_fixed))/1e9:.2f} GHz")

    # ---- 4-panel figure ----
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Broadband Matching Network (Lumped RLC)", fontsize=14, fontweight="bold")

    # Panel 1: Parametric sweep - min S11 vs L
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(L_values * 1e9, s11_min_vals, "bo-", markersize=6)
    ax1.axhline(-10, color="r", ls="--", alpha=0.5, label="-10 dB threshold")
    ax1.plot(best_L * 1e9, s11_min_vals[best_idx], "r*", markersize=15,
             label=f"Best: L={best_L*1e9:.1f} nH")
    ax1.set_xlabel("Inductance L (nH)")
    ax1.set_ylabel("Min S11 (dB)")
    ax1.set_title("Parametric Sweep: S11 vs Inductance")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: S11 frequency response for best match
    ax2 = fig.add_subplot(2, 2, 2)
    if best_result.s_params is not None and best_result.freqs is not None:
        freqs_ghz = np.asarray(best_result.freqs) / 1e9
        s11 = np.asarray(best_result.s_params)[0, 0, :]
        s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-30))
        ax2.plot(freqs_ghz, s11_db, "b-", lw=1.5)
        ax2.axhline(-10, color="r", ls="--", alpha=0.5, label="-10 dB")
        ax2.axvline(f0 / 1e9, color="g", ls=":", alpha=0.5, label=f"f0={f0/1e9:.1f} GHz")
        ax2.set_xlim(0.5, f0 * 2 / 1e9)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("|S11| (dB)")
    ax2.set_title(f"S11 — Best Match (L={best_L*1e9:.1f} nH)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Smith chart
    ax3 = fig.add_subplot(2, 2, 3)
    if best_result.s_params is not None and best_result.freqs is not None:
        s11_complex = np.asarray(best_result.s_params)[0, 0, :]
        freqs_arr = np.asarray(best_result.freqs)
        plot_smith(
            s11_complex, freqs_arr,
            ax=ax3,
            markers=[f0],
            title=f"Smith Chart (L={best_L*1e9:.1f} nH)",
        )

    # Panel 4: S11 overlay for all L values
    ax4 = fig.add_subplot(2, 2, 4)
    for i, (L_val, res) in enumerate(zip(L_values, results)):
        if res.s_params is not None and res.freqs is not None:
            freqs_ghz = np.asarray(res.freqs) / 1e9
            s11_mag = np.abs(np.asarray(res.s_params)[0, 0, :])
            s11_db = 20 * np.log10(np.maximum(s11_mag, 1e-30))
            alpha = 0.4 if i != best_idx else 1.0
            lw = 1.0 if i != best_idx else 2.5
            ax4.plot(freqs_ghz, s11_db, lw=lw, alpha=alpha,
                     label=f"L={L_val*1e9:.1f} nH" if i in (0, best_idx, len(L_values)-1) else None)
    ax4.axhline(-10, color="r", ls="--", alpha=0.5)
    ax4.set_xlabel("Frequency (GHz)")
    ax4.set_ylabel("|S11| (dB)")
    ax4.set_title("S11 Overlay (all sweep points)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, f0 * 2 / 1e9)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/03_broadband_matching.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
