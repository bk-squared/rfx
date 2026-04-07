"""Cross-validation #2: OpenEMS Simple Patch Antenna with non-uniform mesh + NTFF.

Replicates: docs.openems.de/python/openEMS/Tutorials/Simple_Patch_Antenna.html
Structure: 2 GHz rectangular patch on RO4003C, 50-ohm lumped port
Validates: S11 resonance, NTFF radiation pattern

This is the most comprehensive single benchmark — exercises:
- Non-uniform z mesh (P1-P3): fine in substrate, coarse in air
- CPML absorbing boundaries
- Lumped port excitation
- FFT resonance detection
- NTFF far-field pattern extraction
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

from rfx import Simulation, Box, compute_far_field
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# OpenEMS tutorial parameters (exact copy)
# =============================================================================
f0 = 2e9
fc = 1e9
eps_r = 3.38
h = 1.524e-3   # substrate thickness
patch_W = 32e-3
patch_L = 40e-3
sub_W = 60e-3
sub_L = 60e-3
feed_x_offset = -6e-3  # from patch center
feed_R = 50.0

# Expected resonance
f_expected = 2.0e9
f_analytical = C0 / (2 * patch_L * np.sqrt(eps_r))

# Simulation box (OpenEMS: 200x200x150mm)
box_xy = 200e-3
box_z = 150e-3

print("=" * 60)
print("Cross-Validation #2: OpenEMS Simple Patch Antenna")
print("=" * 60)
print(f"Patch: {patch_W*1e3:.0f} x {patch_L*1e3:.0f} mm on eps_r={eps_r}")
print(f"Substrate: h={h*1e3:.3f}mm, {sub_W*1e3:.0f}x{sub_L*1e3:.0f}mm")
print(f"Analytical f_r: {f_analytical/1e9:.3f} GHz")
print()

# Non-uniform z mesh: fine in substrate, manually graded to air
dx = 1.0e-3  # 1mm xy cells (same as session 1 fine resolution)
n_sub = 6
dz_sub = h / n_sub  # 0.254mm per substrate cell

# Manual grading: substrate cells → geometric transition → air cells
# Keep substrate at exact dz_sub, then grade smoothly to dx
transition = []
dz_cur = dz_sub
while dz_cur < dx * 0.95:
    dz_cur = min(dz_cur * 1.3, dx)
    transition.append(dz_cur)

n_air = int(np.ceil((box_z - h - sum(transition)) / dx))
n_air = max(n_air, 10)
dz_profile = np.array([dz_sub] * n_sub + transition + [dx] * n_air)

print(f"dx = {dx*1e3:.2f} mm, dz_sub = {dz_sub*1e3:.3f} mm")
print(f"z-cells: {n_sub} substrate + {n_air} air = {len(dz_profile)} total")

ox = box_xy / 2  # origin offset (center domain)
oy = box_xy / 2

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

sim.add_material("substrate", eps_r=eps_r)

# Ground plane at z=0
sub_x0, sub_y0 = ox - sub_W / 2, oy - sub_L / 2
sub_x1, sub_y1 = ox + sub_W / 2, oy + sub_L / 2
sim.add(Box((sub_x0, sub_y0, 0), (sub_x1, sub_y1, dz_sub)), material="pec")

# Substrate
sim.add(Box((sub_x0, sub_y0, 0), (sub_x1, sub_y1, h)), material="substrate")

# Patch at z=h
patch_x0 = ox - patch_W / 2
patch_y0 = oy - patch_L / 2
sim.add(Box((patch_x0, patch_y0, h), (patch_x0 + patch_W, patch_y0 + patch_L, h + dz_sub)),
        material="pec")

# Lumped port at feed point — z between ground and patch
# Port must be in a non-PEC cell: use center of second substrate cell
port_x = ox + feed_x_offset
port_y = oy
port_z = dz_sub + dz_sub / 2  # center of second substrate cell
sim.add_port(
    (port_x, port_y, port_z), "ez",
    impedance=feed_R,
    waveform=GaussianPulse(f0=f0, bandwidth=fc / f0),
)
sim.add_probe((port_x, port_y, port_z), "ez")
# Second probe at patch center for mode verification
sim.add_probe((ox, oy, port_z), "ez")
# Third probe at radiating edge — Ez peaks here for TM01 mode
# (center probe is at an Ez null, can't detect the fundamental)
sim.add_probe((ox, oy - patch_L / 2 + dx, port_z), "ez")

# NTFF box for radiation pattern
cpml_thick = 8 * dx
ntff_margin = cpml_thick + 5 * dx
ntff_z_hi = sum(dz_profile) - ntff_margin
sim.add_ntff_box(
    (ntff_margin, ntff_margin, ntff_margin),
    (box_xy - ntff_margin, box_xy - ntff_margin, ntff_z_hi),
    np.array([f_expected]),
)

n_steps = 12000
print(f"Steps: {n_steps}")

# Debug: check non-uniform grid z-coordinates
nu_grid = sim._build_nonuniform_grid()
dz_np = np.array(nu_grid.dz)
z_edges = np.cumsum(dz_np)
z_edges = np.insert(z_edges, 0, 0)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2
cpml_off = nu_grid.cpml_layers
z_phys = z_centers[cpml_off:] - z_centers[cpml_off]
print(f"Nu grid: nz={nu_grid.nz}, dt={nu_grid.dt*1e12:.2f}ps")
print(f"z_phys first 10: {z_phys[:10]*1e3}")
print(f"Substrate h={h*1e3:.3f}mm, port z={h/2*1e3:.3f}mm")
print()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = sim.run(n_steps=n_steps)

# Debug: check probe signal
ts = np.array(result.time_series).ravel()
print(f"Probe signal: max={np.max(np.abs(ts)):.4e}, rms={np.sqrt(np.mean(ts**2)):.4e}")

# 1. Resonance detection from edge probe (probe 2) — Ez peaks at radiating edge
#    The center probe (index 1) is at an Ez null of the TM01 mode and
#    cannot reliably detect the fundamental resonance.
ts_all = np.array(result.time_series)
dt = result.dt

EDGE_PROBE = 2  # radiating edge probe

# Try Harminv first (most accurate)
modes = result.find_resonances(freq_range=(1e9, 3.5e9), probe_idx=EDGE_PROBE)
if modes:
    best = min(modes, key=lambda m: abs(m.freq - f_analytical))
    f_sim = best.freq
    print(f"Harminv (edge probe): {f_sim/1e9:.4f} GHz, Q={best.Q:.0f}")
else:
    # Fallback: late-time windowed FFT on edge probe
    ts_edge = ts_all[:, EDGE_PROBE] if ts_all.ndim == 2 and ts_all.shape[1] > EDGE_PROBE else ts_all.ravel()
    skip = int(2e-9 / dt)
    ts_late = ts_edge[skip:]
    ts_windowed = ts_late * np.hanning(len(ts_late))
    nfft = len(ts_windowed) * 4
    spec = np.abs(np.fft.rfft(ts_windowed, n=nfft))
    freqs_fft = np.fft.rfftfreq(nfft, d=dt)
    freqs_ghz_fft = freqs_fft / 1e9
    band = (freqs_ghz_fft > 1.0) & (freqs_ghz_fft < 3.5)
    peak_idx = np.argmax(spec[band])
    f_sim = float(freqs_fft[band][peak_idx])
    print(f"FFT peak (edge probe): {f_sim/1e9:.4f} GHz")

# Also compute windowed spectrum for plotting (use edge probe)
skip = int(2e-9 / dt)
ts_edge_plot = ts_all[:, EDGE_PROBE] if ts_all.ndim == 2 and ts_all.shape[1] > EDGE_PROBE else ts_all.ravel()
ts_late_edge = ts_edge_plot[skip:]
ts_windowed_edge = ts_late_edge * np.hanning(len(ts_late_edge))
nfft = len(ts_windowed_edge) * 4
spec_plot = np.abs(np.fft.rfft(ts_windowed_edge, n=nfft))
freqs_ghz = np.fft.rfftfreq(nfft, d=dt) / 1e9
band = (freqs_ghz > 1.0) & (freqs_ghz < 3.5)
spec_band = spec_plot[band]

err_pct = abs(f_sim - f_analytical) / f_analytical * 100
print(f"Resonance: {f_sim/1e9:.4f} GHz")
print(f"Analytical: {f_analytical/1e9:.4f} GHz")
print(f"Error: {err_pct:.2f}%")

if err_pct < 5:
    print("PASS: resonance within 5%")
else:
    print(f"FAIL: {err_pct:.2f}% error")

# 2. NTFF radiation pattern
print()
try:
    grid = sim._build_grid() if not hasattr(sim, '_last_grid') else sim._last_grid
    # Use non-uniform grid for NTFF
    nu_grid = sim._build_nonuniform_grid()
    theta = np.linspace(0.05, np.pi - 0.05, 37)
    phi = np.array([0.0, np.pi / 2])
    ff = compute_far_field(result.ntff_data, result.ntff_box, nu_grid, theta, phi)
    power = np.abs(ff.E_theta) ** 2 + np.abs(ff.E_phi) ** 2
    max_power = float(np.max(power))
    print(f"Far-field max power: {max_power:.4e}")
    if max_power > 1e-20:
        print("PASS: radiation pattern extracted")
    else:
        print("WARNING: far-field power at noise floor")
except Exception as e:
    print(f"NTFF extraction: {e}")
    max_power = 0

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Patch Antenna: {patch_W*1e3:.0f}x{patch_L*1e3:.0f}mm, eps_r={eps_r}", fontsize=13)

# Resonance spectrum (late-time windowed)
ax = axes[0]
ax.plot(freqs_ghz[band], 20 * np.log10(spec_band / np.max(spec_band) + 1e-30), "b-")
ax.axvline(f_analytical / 1e9, color="r", ls="--", alpha=0.5, label="Analytical")
if f_sim > 0:
    ax.axvline(f_sim / 1e9, color="g", ls=":", label=f"rfx {err_pct:.1f}%")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("dB")
ax.set_title("Resonance Spectrum")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 3)

# E-plane pattern
ax = axes[1]
if max_power > 1e-20:
    P_E = power[:, 0]
    P_E_norm = P_E / (np.max(P_E) + 1e-30)
    ax.plot(np.degrees(theta), 10 * np.log10(P_E_norm + 1e-30), "b-", lw=1.5)
ax.set_xlabel("Theta (deg)")
ax.set_ylabel("Normalized (dB)")
ax.set_title("E-plane (phi=0)")
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 3)

# H-plane pattern
ax = axes[2]
if max_power > 1e-20:
    P_H = power[:, 1]
    P_H_norm = P_H / (np.max(P_H) + 1e-30)
    ax.plot(np.degrees(theta), 10 * np.log10(P_H_norm + 1e-30), "b-", lw=1.5)
ax.set_xlabel("Theta (deg)")
ax.set_ylabel("Normalized (dB)")
ax.set_title("H-plane (phi=90)")
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 3)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "10_openems_patch_antenna.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")
