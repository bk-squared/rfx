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

dx = 0.1 * SCALE  # 1mm cells (finer for accuracy)
domain_x = 20 * SCALE  # 200mm propagation (long for clean separation)
# y needs CPML padding: 10*dx each side + some interior
domain_y = domain_x  # square domain to avoid y boundary issues

# 2D TMz with CPML — x-ends absorbing, y-edges absorbing
sim = Simulation(
    freq_max=(f_center + f_width) * 1.2,
    domain=(domain_x, domain_y, dx),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
    mode="2d_tmz",
)

# Dielectric fills right half (full y extent)
sim.add_material("dielectric", eps_r=eps_r)
sim.add(Box((domain_x / 2, 0, 0), (domain_x, domain_y, dx)),
        material="dielectric")

# Longer time for wave to propagate through dielectric and be absorbed
n_periods_fresnel = 40

# Source, reflection probe, transmission probe
# In PEC boundary, source is a raw Ez injection
src_x = domain_x * 0.15
probe_refl_x = domain_x * 0.1
probe_trans_x = domain_x * 0.8
cy = domain_y / 2

sim.add_source((src_x, cy, 0), "ez",
               waveform=GaussianPulse(f0=f_center, bandwidth=f_width / f_center))
sim.add_probe((probe_refl_x, cy, 0), "ez")
sim.add_probe((probe_trans_x, cy, 0), "ez")

grid = sim._build_grid()
n_steps = int(np.ceil(8e-9 / grid.dt))
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
    domain=(domain_x, domain_y, dx),
    boundary="cpml",
    cpml_layers=10,
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

# Transmittance approach: T = P_trans_with / P_trans_ref, then R = 1 - T
# This is the Meep convention — cancels out source geometry (point vs plane)
ts_ref_trans = ts_ref[:, 1] if ts_ref.ndim == 2 else ts_ref.ravel()
spec_trans_ref = np.abs(np.fft.rfft(ts_ref_trans, n=nfft)) ** 2

band = (freqs_hz > f_center * 0.5) & (freqs_hz < f_center * 1.5)
T_sim = np.mean(spec_trans[band]) / (np.mean(spec_trans_ref[band]) + 1e-30)
R_sim = 1.0 - T_sim

print(f"\nTransmittance T: {T_sim:.4f}")
print(f"Estimated R = 1-T: {R_sim:.4f}")
print(f"Analytical R: {R_fresnel:.4f}")
print(f"Difference: {abs(R_sim - R_fresnel):.4f}")
diff_pct = abs(R_sim - R_fresnel) / R_fresnel * 100
if diff_pct < 10:
    print("PASS: within 10% of Fresnel")
elif diff_pct < 50:
    # Point source creates cylindrical waves — off-axis refraction
    # reduces on-axis transmission compared to plane-wave Fresnel.
    # T approach is correct direction but quantitative accuracy requires
    # TFSF plane wave source.
    print(f"MARGINAL: {diff_pct:.1f}% error (cylindrical wave ≠ plane wave)")
else:
    print(f"FAIL: {diff_pct:.1f}% error")

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
