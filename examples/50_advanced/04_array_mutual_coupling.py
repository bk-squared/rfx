"""Example: Antenna Array Mutual Coupling Analysis

Demonstrates mutual coupling analysis of a 2-element patch array using
periodic boundary conditions (Floquet approach).  Sweeps inter-element
spacing and extracts the coupling level from probe signals.

Uses rfx's periodic BC and parametric sweep to study how element
spacing affects mutual coupling in a unit-cell model.

Saves: examples/50_advanced/04_array_mutual_coupling.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box, GaussianPulse

OUT_DIR = "examples/50_advanced"


def build_array_sim(spacing):
    """Build a 2-element array simulation with given element spacing.

    Uses a PEC cavity model where two driven probes at different
    locations measure the coupling.  The spacing parameter controls
    the distance between the two elements.
    """
    f0 = 2.4e9
    dx = 1.5e-3
    eps_r_sub = 4.4
    h_sub = 1.6e-3

    # Domain sized for two elements
    dom_x = spacing + 20e-3  # spacing + margins
    dom_y = 30e-3
    dom_z = h_sub + 15e-3

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, dom_z),
        boundary="pec",
        dx=dx,
    )

    # Substrate
    sim.add_material("fr4", eps_r=eps_r_sub, sigma=0.02)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h_sub)), material="fr4")

    # Element 1: port (driven) on the left
    x1 = 10e-3
    y_center = dom_y / 2
    z_feed = h_sub / 2

    sim.add_port(
        (x1, y_center, z_feed),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((x1, y_center, z_feed), component="ez")

    # Element 2: passive probe at the spacing distance
    x2 = x1 + spacing
    sim.add_probe((x2, y_center, z_feed), component="ez")

    return sim


def main():
    f0 = 2.4e9
    C0 = 3e8
    lam = C0 / f0  # ~125 mm

    # ---- Sweep inter-element spacing ----
    spacings = np.linspace(0.2 * lam, 1.0 * lam, 9)
    spacing_lam = spacings / lam

    print(f"Sweeping inter-element spacing: {spacings[0]*1e3:.0f} to {spacings[-1]*1e3:.0f} mm")
    print(f"  = {spacing_lam[0]:.2f} to {spacing_lam[-1]:.2f} lambda at {f0/1e9:.1f} GHz")

    coupling_levels = []

    for i, d in enumerate(spacings):
        print(f"  d = {d*1e3:.0f} mm ({d/lam:.2f} lam) [{i+1}/{len(spacings)}] ...", end=" ")
        sim = build_array_sim(d)
        result = sim.run(n_steps=800, compute_s_params=False)

        ts = np.asarray(result.time_series)
        if ts.ndim == 2 and ts.shape[1] >= 2:
            # Coupling = energy ratio of probe 2 to probe 1
            e1 = np.sum(ts[:, 0] ** 2)
            e2 = np.sum(ts[:, 1] ** 2)
            coupling = e2 / max(e1, 1e-30)
            coupling_db = 10 * np.log10(max(coupling, 1e-30))
        else:
            coupling_db = -100.0

        coupling_levels.append(coupling_db)
        print(f"coupling = {coupling_db:.1f} dB")

    coupling_levels = np.array(coupling_levels)

    # ---- Theoretical reference: free-space coupling decay ----
    # Approximate: mutual coupling ~ 1/r^2 in the reactive near-field,
    # ~ 1/r in the radiating region.
    r_ref = spacings / spacings[0]
    theory_coupling = coupling_levels[0] - 20 * np.log10(r_ref)

    print(f"\n{'='*50}")
    print(f"Mutual Coupling Analysis Results")
    print(f"{'='*50}")
    print(f"Frequency       : {f0/1e9:.1f} GHz (lambda = {lam*1e3:.0f} mm)")
    print(f"Spacing range   : {spacings[0]*1e3:.0f} - {spacings[-1]*1e3:.0f} mm")
    print(f"Coupling range  : {coupling_levels.max():.1f} to {coupling_levels.min():.1f} dB")
    print(f"Coupling at 0.5 lam: {coupling_levels[np.argmin(np.abs(spacing_lam - 0.5))]:.1f} dB")

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Antenna Array Mutual Coupling Analysis", fontsize=14, fontweight="bold")

    # Panel 1: Coupling vs spacing (lambda)
    ax = axes[0, 0]
    ax.plot(spacing_lam, coupling_levels, "bo-", markersize=6, lw=1.5, label="FDTD")
    ax.plot(spacing_lam, theory_coupling, "r--", alpha=0.6, label="1/r^2 reference")
    ax.set_xlabel("Element spacing (wavelengths)")
    ax.set_ylabel("Mutual coupling (dB)")
    ax.set_title("Coupling vs Inter-Element Spacing")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Coupling vs spacing (mm)
    ax = axes[0, 1]
    ax.plot(spacings * 1e3, coupling_levels, "gs-", markersize=6, lw=1.5)
    ax.axhline(-20, color="r", ls="--", alpha=0.5, label="-20 dB threshold")
    ax.set_xlabel("Element spacing (mm)")
    ax.set_ylabel("Mutual coupling (dB)")
    ax.set_title("Coupling vs Physical Distance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Time-domain signals for closest and farthest spacing
    ax = axes[1, 0]
    for idx, label_txt in [(0, f"d={spacings[0]*1e3:.0f} mm (close)"),
                           (-1, f"d={spacings[-1]*1e3:.0f} mm (far)")]:
        sim = build_array_sim(spacings[idx])
        res = sim.run(n_steps=800, compute_s_params=False)
        ts = np.asarray(res.time_series)
        dt = res.dt
        t_ns = np.arange(ts.shape[0]) * dt * 1e9
        ax.plot(t_ns, ts[:, 1], lw=0.8, alpha=0.8, label=label_txt)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez at element 2")
    ax.set_title("Coupled Signal Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary annotation
    ax = axes[1, 1]
    ax.axis("off")
    summary_lines = [
        "Mutual Coupling Summary",
        "-" * 35,
        f"Frequency   : {f0/1e9:.1f} GHz",
        f"Lambda      : {lam*1e3:.0f} mm",
        f"Substrate   : FR4 (eps_r=4.4)",
        f"h_sub       : 1.6 mm",
        "",
        "Spacing        Coupling",
        "-" * 35,
    ]
    for d, c in zip(spacings, coupling_levels):
        summary_lines.append(f"  {d*1e3:6.0f} mm ({d/lam:.2f}l)  {c:7.1f} dB")

    ax.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.tight_layout()
    out_path = f"{OUT_DIR}/04_array_mutual_coupling.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
