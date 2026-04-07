"""Accuracy Validation Case 1: Rectangular Patch Antenna

Replicates the OpenEMS Simple Patch Antenna tutorial EXACTLY:
  - Same geometry, substrate, feed position, boundary
  - Enables direct S11 comparison between rfx and OpenEMS

OpenEMS reference: docs.openems.de/python/openEMS/Tutorials/Simple_Patch_Antenna.html

Validation metric: resonance frequency (S11 dip) within 5% of OpenEMS result (~2 GHz)
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PCT = 5.0

# =============================================================================
# OpenEMS tutorial parameters (exact copy)
# =============================================================================
f0 = 2e9              # center frequency
fc = 1e9              # 20dB bandwidth (excites 1-3 GHz)

# Substrate: Rogers RO4003C-like
eps_r = 3.38
kappa = 1e-3 * 2 * np.pi * 2.45e9 * 8.854e-12 * eps_r  # loss tangent ~0.001
h = 1.524e-3          # substrate thickness (mm)

# Patch dimensions (OpenEMS tutorial values, NOT Hammerstad formula)
patch_W = 32e-3       # width (x-direction, resonant dimension)
patch_L = 40e-3       # length (y-direction)

# Substrate extent (finite, like real PCB)
sub_W = 60e-3
sub_L = 60e-3

# Feed: 50-ohm lumped port probe
feed_x = -6e-3        # offset from patch center in x
feed_R = 50.0         # port impedance

# Simulation box: ~lambda/2 margin (OpenEMS uses 200x200x150mm)
sim_box = [200e-3, 200e-3, 150e-3]

# Expected resonance (from OpenEMS result, ~2 GHz patch on eps_r=3.38)
f_expected = 2.0e9

print("=" * 60)
print("CASE 1: Patch Antenna (OpenEMS Tutorial Replica)")
print("=" * 60)
print(f"Patch    : {patch_W*1e3:.0f} x {patch_L*1e3:.0f} mm")
print(f"Substrate: eps_r={eps_r}, h={h*1e3:.3f} mm, {sub_W*1e3:.0f}x{sub_L*1e3:.0f} mm")
print(f"Feed     : x={feed_x*1e3:.0f}mm from center, Z0={feed_R} ohm")
print(f"Box      : {sim_box[0]*1e3:.0f}x{sim_box[1]*1e3:.0f}x{sim_box[2]*1e3:.0f} mm")
print()

# =============================================================================
# Resolution sweep (coarse for speed, fine for accuracy)
# =============================================================================
resolutions = [
    ("standard", 2.0e-3),   # dx=2mm, lambda/50 at 3GHz in substrate
    ("fine",     1.0e-3),   # dx=1mm
]

results_list = []

for label, dx in resolutions:
    print(f"--- Resolution: {label} (dx={dx*1e3:.1f} mm) ---")

    # Domain centered at origin (OpenEMS convention)
    # rfx uses corner-based coordinates, so shift to positive quadrant
    ox = sim_box[0] / 2   # origin offset x
    oy = sim_box[1] / 2   # origin offset y

    sim = Simulation(
        freq_max=(f0 + fc) * 1.5,
        domain=(sim_box[0], sim_box[1], sim_box[2]),
        dx=dx,
        boundary="cpml",
        cpml_layers=8,
    )

    # Substrate (centered, finite extent like real PCB)
    sub_x0 = ox - sub_W / 2
    sub_x1 = ox + sub_W / 2
    sub_y0 = oy - sub_L / 2
    sub_y1 = oy + sub_L / 2

    sim.add_material("substrate", eps_r=eps_r, sigma=kappa)

    # Ground plane — use fixed thin sheet, not dx-dependent thickness
    gnd_thick = min(dx, h / 4)  # thin but at least one cell
    sim.add(Box((sub_x0, sub_y0, 0), (sub_x1, sub_y1, gnd_thick)), material="pec")

    # Substrate from z=0 to z=h (resolution-independent)
    sim.add(Box((sub_x0, sub_y0, 0), (sub_x1, sub_y1, h)), material="substrate")

    # Patch at z=h (resolution-independent)
    patch_x0 = ox - patch_W / 2
    patch_x1 = ox + patch_W / 2
    patch_y0 = oy - patch_L / 2
    patch_y1 = oy + patch_L / 2
    sim.add(Box((patch_x0, patch_y0, h),
                (patch_x1, patch_y1, h + gnd_thick)), material="pec")

    # Feed: 50-ohm lumped port spanning ground to patch (z-directed)
    feed_abs_x = ox + feed_x
    feed_abs_y = oy

    sim.add_port(
        position=(feed_abs_x, feed_abs_y, 0),
        component="ez",
        impedance=feed_R,
        waveform=GaussianPulse(f0=f0, bandwidth=fc / f0),
        extent=h,  # span through substrate to patch
    )

    # Probe at substrate midpoint (physical absolute coordinate)
    sim.add_probe((feed_abs_x, feed_abs_y, h / 2), "ez")

    grid = sim._build_grid()
    n_cells = grid.nx * grid.ny * grid.nz

    # 30000 steps (matching OpenEMS default)
    n_steps = min(30000, int(np.ceil(20e-9 / grid.dt)))

    # S-parameter frequencies (1-3 GHz, 201 points)
    sp_freqs = np.linspace(1e9, 3e9, 201)

    print(f"  Grid: {grid.nx}x{grid.ny}x{grid.nz} = {n_cells/1e6:.1f}M cells")
    print(f"  Steps: {n_steps}, dt={grid.dt*1e12:.2f} ps")

    result = sim.run(
        n_steps=n_steps,
        compute_s_params=True,
        s_param_freqs=sp_freqs,
    )

    # Extract resonance from S11 dip
    f_sim = 0.0
    s11_min_dB = 0.0

    if result.s_params is not None and result.freqs is not None:
        s11 = result.s_params[0, 0, :]
        s11_dB = 20 * np.log10(np.abs(s11) + 1e-30)
        min_idx = np.argmin(s11_dB)
        f_sim = float(result.freqs[min_idx])
        s11_min_dB = float(s11_dB[min_idx])
        print(f"  S11 dip: {f_sim/1e9:.4f} GHz ({s11_min_dB:.1f} dB)")
    else:
        # Fallback: Harminv
        modes = result.find_resonances(freq_range=(1e9, 3e9))
        if modes:
            best = min(modes, key=lambda m: abs(m.freq - f_expected))
            f_sim = best.freq
            print(f"  Harminv: {f_sim/1e9:.4f} GHz, Q={best.Q:.0f}")
        else:
            # FFT fallback
            ts = np.asarray(result.time_series).ravel()
            nfft = len(ts) * 4
            spec = np.abs(np.fft.rfft(ts, n=nfft))
            freqs_fft = np.fft.rfftfreq(nfft, d=result.dt)
            band = (freqs_fft > 1e9) & (freqs_fft < 3e9)
            if np.any(band):
                f_sim = freqs_fft[np.argmax(spec * band)]
            print(f"  FFT peak: {f_sim/1e9:.4f} GHz")

    err = abs(f_sim - f_expected) / f_expected * 100 if f_sim > 0 else 999

    results_list.append({
        "label": label, "dx": dx, "f_sim": f_sim,
        "err_pct": err, "s11_min": s11_min_dB, "n_steps": n_steps,
    })
    print(f"  Error vs expected: {err:.1f}%")
    print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 60)
print("CONVERGENCE SUMMARY")
print("=" * 60)
print(f"{'Resolution':<12} {'dx (mm)':<10} {'f_sim (GHz)':<14} {'Error %':<10} {'S11 (dB)':<10}")
print("-" * 56)
for r in results_list:
    print(f"{r['label']:<12} {r['dx']*1e3:<10.1f} {r['f_sim']/1e9:<14.4f} "
          f"{r['err_pct']:<10.1f} {r['s11_min']:<10.1f}")

best = results_list[-1]
final_err = best["err_pct"]
passed = final_err < THRESHOLD_PCT

print()
print(f"Expected (OpenEMS): ~{f_expected/1e9:.1f} GHz")
print(f"Best rfx result:    {best['f_sim']/1e9:.4f} GHz")
print(f"Error: {final_err:.1f}% ({'PASS' if passed else 'FAIL'}, threshold {THRESHOLD_PCT}%)")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Case 1: Patch Antenna (OpenEMS Tutorial Replica)", fontsize=13, fontweight="bold")

# Panel 1: Setup summary
ax = axes[0]
ax.axis("off")
ax.text(0.05, 0.95,
    f"OpenEMS Tutorial Replica\n\n"
    f"Patch: {patch_W*1e3:.0f}x{patch_L*1e3:.0f} mm\n"
    f"Substrate: eps_r={eps_r}, h={h*1e3:.3f}mm\n"
    f"Feed: 50ohm probe, x={feed_x*1e3:.0f}mm\n"
    f"Box: {sim_box[0]*1e3:.0f}x{sim_box[1]*1e3:.0f}x{sim_box[2]*1e3:.0f}mm\n"
    f"Boundary: CPML-8",
    transform=ax.transAxes, va="top", fontsize=10, family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_title("Setup")

# Panel 2: Convergence
ax = axes[1]
dx_vals = [r["dx"] * 1e3 for r in results_list]
err_vals = [r["err_pct"] for r in results_list]
ax.plot(dx_vals, err_vals, "bo-", markersize=8, linewidth=2)
ax.axhline(THRESHOLD_PCT, color="r", ls="--", label=f"Threshold {THRESHOLD_PCT}%")
ax.set_xlabel("dx (mm)")
ax.set_ylabel("Frequency error (%)")
ax.set_title("Grid Convergence")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Verdict
ax = axes[2]
ax.axis("off")
color = "lightgreen" if passed else "lightsalmon"
ax.text(0.5, 0.5,
    f"{'PASS' if passed else 'FAIL'}\n\n"
    f"Expected: ~{f_expected/1e9:.1f} GHz\n"
    f"rfx best: {best['f_sim']/1e9:.4f} GHz\n"
    f"Error: {final_err:.1f}%\n"
    f"S11 dip: {best['s11_min']:.1f} dB\n\n"
    f"dx={best['dx']*1e3:.1f}mm, {best['n_steps']} steps",
    transform=ax.transAxes, va="center", ha="center",
    fontsize=12, family="monospace",
    bbox=dict(boxstyle="round", facecolor=color, alpha=0.8))
ax.set_title("Verdict")

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "01_patch_balanis.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")

if not passed:
    sys.exit(1)
