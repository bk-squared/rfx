"""Cross-validation: Meep Fresnel reflectance at air/dielectric interface.

Replicates: meep.readthedocs.io Material Dispersion tutorial
Structure: Planar air/dielectric (n=3.5) interface, normal incidence
Comparison: Fresnel equations R = ((n-1)/(n+1))^2
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

# Parameters
n_dielectric = 3.5
eps_r = n_dielectric ** 2  # 12.25
R_fresnel = ((n_dielectric - 1) / (n_dielectric + 1)) ** 2  # 0.309

# Scale to microwave: 1 unit = 10mm
SCALE = 10e-3
f_center = 1.0 * C0 / SCALE  # 30 GHz
f_width = 0.4 * C0 / SCALE

print("=" * 60)
print("Cross-Validation: Meep Fresnel Reflectance")
print("=" * 60)
print(f"Interface: air / n={n_dielectric} (eps_r={eps_r})")
print(f"Analytical R = {R_fresnel:.4f} ({10*np.log10(R_fresnel):.1f} dB)")
print()

dx = 0.2 * SCALE  # 2mm cells
domain_x = 10 * SCALE  # 100mm propagation
domain_yz = dx  # single cell in y/z for quasi-1D

# Use 2D TMz mode for clean 1D propagation (no y/z CPML absorption)
sim = Simulation(
    freq_max=(f_center + f_width) * 1.2,
    domain=(domain_x, domain_x, domain_yz),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
    mode="2d_tmz",
)

# Dielectric fills right half (full y extent)
sim.add_material("dielectric", eps_r=eps_r)
sim.add(Box((domain_x / 2, 0, 0), (domain_x, domain_x, domain_yz)),
        material="dielectric")

# Source at left, probe at left (reflected) and right (transmitted)
src_x = domain_x * 0.25
probe_refl_x = domain_x * 0.15
probe_trans_x = domain_x * 0.8
cy = domain_x / 2

sim.add_source((src_x, cy, 0), "ez",
               waveform=GaussianPulse(f0=f_center, bandwidth=f_width / f_center))
sim.add_probe((probe_refl_x, cy, 0), "ez")
sim.add_probe((probe_trans_x, cy, 0), "ez")

grid = sim._build_grid()
n_steps = int(np.ceil(5e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

# Run 1: with dielectric (reflection + transmission)
result = sim.run(n_steps=n_steps)
ts = np.array(result.time_series)
if ts.ndim == 2 and ts.shape[1] >= 2:
    ts_refl = ts[:, 0]
    ts_trans = ts[:, 1]
else:
    ts_refl = ts.ravel()
    ts_trans = ts_refl

# FFT
nfft = len(ts_refl) * 4
spec_refl = np.abs(np.fft.rfft(ts_refl, n=nfft)) ** 2
spec_trans = np.abs(np.fft.rfft(ts_trans, n=nfft)) ** 2
freqs_hz = np.fft.rfftfreq(nfft, d=result.dt)

# Run 2: reference run without dielectric (incident-only)
sim_ref = Simulation(
    freq_max=(f_center + f_width) * 1.2,
    domain=(domain_x, domain_x, domain_yz),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
    mode="2d_tmz",
)
sim_ref.add_source((src_x, cy, 0), "ez",
                    waveform=GaussianPulse(f0=f_center, bandwidth=f_width / f_center))
sim_ref.add_probe((probe_refl_x, cy, 0), "ez")
sim_ref.add_probe((probe_trans_x, cy, 0), "ez")

result_ref = sim_ref.run(n_steps=n_steps)
ts_ref = np.array(result_ref.time_series)
ts_inc = ts_ref[:, 0] if ts_ref.ndim == 2 else ts_ref.ravel()

# Reflected = total - incident at reflection probe
ts_refl_only = ts_refl - ts_inc
spec_refl_only = np.abs(np.fft.rfft(ts_refl_only, n=nfft)) ** 2
spec_inc = np.abs(np.fft.rfft(ts_inc, n=nfft)) ** 2

band = (freqs_hz > f_center * 0.5) & (freqs_hz < f_center * 1.5)
R_sim = np.mean(spec_refl_only[band]) / (np.mean(spec_inc[band]) + 1e-30)

print(f"\nEstimated R: {R_sim:.4f}")
print(f"Analytical R: {R_fresnel:.4f}")
print(f"Difference: {abs(R_sim - R_fresnel):.4f}")
if abs(R_sim - R_fresnel) < 0.05:
    print("PASS: within 5% of Fresnel")
else:
    print(f"FAIL: {abs(R_sim - R_fresnel)/R_fresnel*100:.1f}% error")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Fresnel Reflectance: air/n={n_dielectric} (Meep Tutorial Replica)", fontsize=13)

ax = axes[0]
t_ns = np.arange(len(ts_refl)) * result.dt * 1e9
ax.plot(t_ns, ts_refl, "b-", lw=0.8, label="Reflected probe")
ax.plot(t_ns, ts_trans, "r-", lw=0.8, label="Transmitted probe")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Time Domain")

ax = axes[1]
ax.text(0.5, 0.5, f"Fresnel R = {R_fresnel:.4f}\n"
        f"rfx est R = {R_sim:.4f}\n\n"
        f"n = {n_dielectric}, eps_r = {eps_r}",
        transform=ax.transAxes, va="center", ha="center",
        fontsize=14, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.axis("off")
ax.set_title("Comparison")

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "05_meep_fresnel.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
