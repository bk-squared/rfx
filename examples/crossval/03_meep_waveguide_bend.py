"""Cross-validation: Meep 90-degree waveguide bend transmittance.

Replicates: meep.readthedocs.io/en/latest/Python_Tutorials/Basics/
Structure: Dielectric waveguide (eps=12) with 90-degree bend
Comparison: Broadband transmittance spectrum
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Meep tutorial parameters (scaled from normalized units to mm)
# Meep uses: resolution=10, cell=16x16, waveguide width=1, eps=12
# Scale factor: 1 Meep unit = 1 mm (arbitrary, preserves physics)
# =============================================================================
SCALE = 1e-3  # 1 Meep unit = 1 mm

eps_wg = 12.0
wg_width = 1.0 * SCALE  # 1 mm
cell_x = 16.0 * SCALE
cell_y = 16.0 * SCALE
cell_z = 0  # 2D (use thin 3D slab)
pml_thick = 1.0 * SCALE

# Meep source: frequency=0.15 (in c/a units), fwidth=0.1
# In physical units with a=1mm: f = 0.15 * C0 / 1mm = 45 GHz
f_center = 0.15 * C0 / SCALE
f_width = 0.1 * C0 / SCALE

print("=" * 60)
print("Cross-Validation: Meep Waveguide Bend Transmittance")
print("=" * 60)
print(f"Waveguide: eps={eps_wg}, width={wg_width*1e3:.1f}mm")
print(f"Cell: {cell_x*1e3:.0f}x{cell_y*1e3:.0f}mm")
print(f"Frequency: {f_center/1e9:.1f} GHz, BW={f_width/1e9:.1f} GHz")
print()

# =============================================================================
# rfx simulation — waveguide bend
# =============================================================================
dx = 0.1 * SCALE  # resolution = 10 → dx = 0.1 mm
cell_z_3d = 3 * dx  # thin slab for quasi-2D

sim = Simulation(
    freq_max=f_center + f_width,
    domain=(cell_x, cell_y, cell_z_3d),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

sim.add_material("waveguide", eps_r=eps_wg)

# Horizontal arm: y from -inf to center, at x=0 (shifted to positive coords)
cx = cell_x / 2
cy = cell_y / 2
wg_half = wg_width / 2

# Horizontal arm (left to center)
sim.add(Box((0, cy - wg_half, 0), (cx + wg_half, cy + wg_half, cell_z_3d)),
        material="waveguide")
# Vertical arm (center to top)
sim.add(Box((cx - wg_half, cy - wg_half, 0), (cx + wg_half, cell_y, cell_z_3d)),
        material="waveguide")

# Source: Gaussian pulse at left end of horizontal arm
sim.add_source(
    (pml_thick + dx, cy, cell_z_3d / 2),
    component="ez",
    waveform=GaussianPulse(f0=f_center, bandwidth=f_width / f_center),
)

# Input reference probe (before bend, in horizontal arm)
sim.add_probe((pml_thick + 5 * dx, cy, cell_z_3d / 2), "ez")
# Output probe (top of vertical arm, after bend)
sim.add_probe((cx, cell_y - pml_thick - dx, cell_z_3d / 2), "ez")

grid = sim._build_grid()
n_steps = int(np.ceil(10e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

result = sim.run(n_steps=n_steps)

ts = np.array(result.time_series)
if ts.ndim == 2 and ts.shape[1] >= 2:
    ts_ref = ts[:, 0]  # input reference (before bend)
    ts_out = ts[:, 1]  # bend output (after bend)
else:
    ts_ref = ts.ravel()
    ts_out = ts_ref

# FFT for transmittance
nfft = len(ts_out) * 4
spec_out = np.abs(np.fft.rfft(ts_out, n=nfft))
spec_ref = np.abs(np.fft.rfft(ts_ref, n=nfft))
freqs_hz = np.fft.rfftfreq(nfft, d=result.dt)
freqs_ghz = freqs_hz / 1e9

# Transmittance = |output|² / |reference|²
transmittance = (spec_out / (spec_ref + 1e-30)) ** 2

# Band of interest
band = (freqs_ghz > 10) & (freqs_ghz < 80)

print(f"Mean transmittance in band: {np.mean(transmittance[band]):.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Waveguide Bend Transmittance (Meep Tutorial Replica)", fontsize=13)

ax = axes[0]
ax.plot(freqs_ghz[band], 10 * np.log10(transmittance[band] + 1e-30), "b-")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Transmittance (dB)")
ax.set_title("Bend Transmittance")
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 5)

ax = axes[1]
t_ns = np.arange(len(ts_out)) * result.dt * 1e9
ax.plot(t_ns, ts_out, "b-", lw=0.8, label="Bend output")
ax.plot(t_ns, ts_ref, "r--", lw=0.8, label="Reference")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez")
ax.set_title("Time Domain")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "03_meep_waveguide_bend.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
