"""Diagnostic: Non-uniform vs uniform runner comparison.

Runs the SAME patch antenna on both:
1. Uniform runner (dx=dy=dz=1mm, 200mm CPML domain)
2. Non-uniform runner (dx=dy=1mm, dz graded, 200mm CPML domain)

Compares resonance frequencies to isolate non-uniform runner bugs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# OpenEMS tutorial parameters (same as crossval 10)
# =============================================================================
f0 = 2e9
fc = 1e9
eps_r = 3.38
h = 1.524e-3   # substrate thickness
patch_W = 32e-3
patch_L = 40e-3
sub_W = 60e-3
sub_L = 60e-3
feed_x_offset = -6e-3
feed_R = 50.0
dx = 1.0e-3

f_analytical = C0 / (2 * patch_L * np.sqrt(eps_r))
box_xy = 200e-3
box_z = 150e-3

print("=" * 60)
print("Diagnostic: Non-uniform vs Uniform Runner")
print("=" * 60)
print(f"Patch: {patch_W*1e3:.0f}x{patch_L*1e3:.0f}mm, eps_r={eps_r}")
print(f"Analytical f_r: {f_analytical/1e9:.4f} GHz")
print()


def build_and_run(label, use_nonuniform=False, n_steps=None):
    """Build and run the patch antenna on the specified grid type."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    ox = box_xy / 2
    oy = box_xy / 2

    if use_nonuniform:
        # Non-uniform: fine dz in substrate, graded to air
        n_sub = 6
        dz_sub = h / n_sub
        transition = []
        dz_cur = dz_sub
        while dz_cur < dx * 0.95:
            dz_cur = min(dz_cur * 1.3, dx)
            transition.append(dz_cur)
        n_air = int(np.ceil((box_z - h - sum(transition)) / dx))
        n_air = max(n_air, 10)
        dz_profile = np.array([dz_sub] * n_sub + transition + [dx] * n_air)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(
                freq_max=(f0 + fc) * 1.5,
                domain=(box_xy, box_xy),
                dx=dx,
                dz_profile=dz_profile,
                boundary="cpml",
                cpml_layers=8,
            )

        # Use dz_sub for ground/patch thickness
        gnd_thick = dz_sub
        patch_thick = dz_sub
        port_z = dz_sub + dz_sub / 2  # center of second cell
    else:
        # Uniform: dx=dy=dz=1mm
        sim = Simulation(
            freq_max=(f0 + fc) * 1.5,
            domain=(box_xy, box_xy, box_z),
            dx=dx,
            boundary="cpml",
            cpml_layers=8,
        )
        gnd_thick = dx
        patch_thick = dx
        port_z = dx + dx / 2  # center of second cell (above ground)

    sim.add_material("substrate", eps_r=eps_r)

    # Ground plane at z=0
    sub_x0, sub_y0 = ox - sub_W / 2, oy - sub_L / 2
    sub_x1, sub_y1 = ox + sub_W / 2, oy + sub_L / 2
    sim.add(Box((sub_x0, sub_y0, 0), (sub_x1, sub_y1, gnd_thick)),
            material="pec")

    # Substrate
    sim.add(Box((sub_x0, sub_y0, 0), (sub_x1, sub_y1, h)),
            material="substrate")

    # Patch at z=h
    patch_x0 = ox - patch_W / 2
    patch_y0 = oy - patch_L / 2
    sim.add(Box((patch_x0, patch_y0, h),
                (patch_x0 + patch_W, patch_y0 + patch_L, h + patch_thick)),
            material="pec")

    # Lumped port
    port_x = ox + feed_x_offset
    port_y = oy
    sim.add_port(
        (port_x, port_y, port_z), "ez",
        impedance=feed_R,
        waveform=GaussianPulse(f0=f0, bandwidth=fc / f0),
    )

    # Probes: feed, center, and edge (where Ez is maximal for TM01)
    sim.add_probe((port_x, port_y, port_z), "ez")      # 0: feed
    sim.add_probe((ox, oy, port_z), "ez")               # 1: center
    # Edge probe: at y = oy - L/2 + dx (near radiating edge, where Ez peaks)
    sim.add_probe((ox, oy - patch_L / 2 + dx, port_z), "ez")  # 2: edge

    if use_nonuniform:
        nu_grid = sim._build_nonuniform_grid()
        print(f"Grid: {nu_grid.nx}x{nu_grid.ny}x{nu_grid.nz}, "
              f"dt={nu_grid.dt*1e12:.2f}ps")
        if n_steps is None:
            n_steps = int(np.ceil(20e-9 / nu_grid.dt))
        # Print port sigma diagnostic
        from rfx.runners.nonuniform import pos_to_nu_index
        idx = pos_to_nu_index(nu_grid, (port_x, port_y, port_z))
        k = idx[2]
        dz_k = float(nu_grid.dz[k])
        sigma_old = 1.0 / (feed_R * dz_k)
        sigma_new = dz_k / (feed_R * nu_grid.dx * nu_grid.dy)
        print(f"Port cell: z_idx={k}, dz={dz_k*1e3:.3f}mm")
        print(f"  sigma_old (1/(Z0*dz)): {sigma_old:.2f} S/m")
        print(f"  sigma_new (dz/(Z0*dx*dy)): {sigma_new:.2f} S/m")
        print(f"  Ratio: {sigma_old/sigma_new:.1f}x overload (fixed)")
    else:
        grid = sim._build_grid()
        print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, "
              f"dt={grid.dt*1e12:.2f}ps")
        if n_steps is None:
            n_steps = int(np.ceil(20e-9 / grid.dt))

    print(f"Steps: {n_steps}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=n_steps)

    # Analyze all probes
    ts_all = np.array(result.time_series)
    dt = result.dt
    probe_names = ["feed", "center", "edge"]

    results = {}
    for pidx, pname in enumerate(probe_names):
        if ts_all.ndim == 2 and ts_all.shape[1] > pidx:
            ts = ts_all[:, pidx]
        elif pidx == 0:
            ts = ts_all.ravel()
        else:
            continue

        # Late-time windowed FFT
        skip = int(2e-9 / dt)
        ts_late = ts[skip:]
        ts_w = ts_late * np.hanning(len(ts_late))
        nfft = len(ts_w) * 4
        spec = np.abs(np.fft.rfft(ts_w, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=dt) / 1e9
        band = (freqs > 1.0) & (freqs < 3.5)
        peak_idx = np.argmax(spec[band])
        f_fft = float(freqs[band][peak_idx])

        # Harminv
        modes = result.find_resonances(freq_range=(1e9, 3.5e9), probe_idx=pidx)
        f_harm = 0
        q_harm = 0
        if modes:
            best = min(modes, key=lambda m: abs(m.freq - f_analytical))
            f_harm = best.freq / 1e9
            q_harm = best.Q

        print(f"  Probe {pidx} ({pname}): FFT={f_fft:.4f}GHz, "
              f"Harminv={f_harm:.4f}GHz (Q={q_harm:.0f}), "
              f"max={np.max(np.abs(ts)):.2e}")
        results[pname] = {"fft": f_fft, "harm": f_harm, "q": q_harm,
                          "spec": spec[band], "freqs_ghz": freqs[band]}

    return results


# =============================================================================
# Run both configurations
# =============================================================================
res_uniform = build_and_run("UNIFORM (dx=dy=dz=1mm)", use_nonuniform=False)
res_nonuniform = build_and_run("NON-UNIFORM (dx=dy=1mm, dz graded)",
                                use_nonuniform=True)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'':20} {'Uniform':>15} {'Non-uniform':>15} {'Analytical':>15}")
print("-" * 65)

for probe in ["feed", "center", "edge"]:
    fu = res_uniform.get(probe, {}).get("harm", 0)
    fn = res_nonuniform.get(probe, {}).get("harm", 0)
    print(f"Harminv ({probe:6}): "
          f"{fu:>14.4f}GHz {fn:>14.4f}GHz {f_analytical/1e9:>14.4f}GHz")

print()
for probe in ["feed", "center", "edge"]:
    fu = res_uniform.get(probe, {}).get("fft", 0)
    fn = res_nonuniform.get(probe, {}).get("fft", 0)
    eu = abs(fu - f_analytical / 1e9) / (f_analytical / 1e9) * 100 if fu > 0 else 999
    en = abs(fn - f_analytical / 1e9) / (f_analytical / 1e9) * 100 if fn > 0 else 999
    print(f"FFT ({probe:6}):     "
          f"{fu:>10.4f}GHz ({eu:>5.1f}%) {fn:>10.4f}GHz ({en:>5.1f}%)")

# Plot comparison spectra
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Non-uniform vs Uniform Runner: Patch Antenna Resonance", fontsize=13)

for i, probe in enumerate(["feed", "center", "edge"]):
    ax = axes[i]
    for label, res, color in [("Uniform", res_uniform, "b"),
                               ("Non-uniform", res_nonuniform, "r")]:
        if probe in res:
            spec = res[probe]["spec"]
            freqs = res[probe]["freqs_ghz"]
            spec_db = 20 * np.log10(spec / np.max(spec) + 1e-30)
            ax.plot(freqs, spec_db, f"{color}-", lw=1.2, label=label)
    ax.axvline(f_analytical / 1e9, color="g", ls="--", alpha=0.5,
               label="Analytical")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("dB")
    ax.set_title(f"Probe: {probe}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-30, 3)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "11_nu_vs_uniform_diagnostic.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")
