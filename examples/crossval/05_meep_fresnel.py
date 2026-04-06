"""Cross-validation: Fresnel reflectance at air/dielectric interface.

Uses TFSF source for clean plane-wave incidence (proven in test_physics.py).
Structure: Dielectric half-space (n=3.5, eps_r=12.25), normal incidence
Comparison: Fresnel equation R = ((n-1)/(n+1))^2 = 0.3086
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx.grid import Grid, C0
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state, init_materials,
    update_h, update_e, EPS_0,
)
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.sources.tfsf import (
    init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e,
    apply_tfsf_e, apply_tfsf_h,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parameters
n_diel = 3.5
eps_r = n_diel ** 2  # 12.25
r_analytic = abs((1.0 - n_diel) / (1.0 + n_diel))  # amplitude |r| = 0.5556
R_analytic = r_analytic ** 2  # power R = 0.3086

print("=" * 60)
print("Cross-Validation: Fresnel Reflectance (TFSF)")
print("=" * 60)
print(f"Interface: air / n={n_diel} (eps_r={eps_r:.2f})")
print(f"Analytical R = {R_analytic:.4f} ({10*np.log10(R_analytic):.1f} dB)")
print()

# Grid: 1D propagation in x, periodic y/z
grid = Grid(freq_max=10e9, domain=(0.60, 0.006, 0.006), dx=0.001, cpml_layers=10)
dt, dx = grid.dt, grid.dx
nc = grid.cpml_layers
periodic = (False, True, True)

# TFSF source: plane wave +x, f0=5 GHz
tfsf_cfg, tfsf_st = init_tfsf(
    grid.nx, dx, dt, cpml_layers=nc, tfsf_margin=5,
    f0=5e9, bandwidth=0.5, amplitude=1.0,
)

# Dielectric slab: starts at nx//4, ends 10 cells before TFSF x_hi
x_interface = grid.nx // 4
x_diel_end = tfsf_cfg.x_hi - 10

# Probe in scattered-field region (measures only reflected wave)
probe_x = tfsf_cfg.x_lo - 3
probe = (probe_x, grid.ny // 2, grid.nz // 2)
ref_1d_idx = tfsf_cfg.i0 + 5

# Time: long enough for reflection, short enough to avoid back-face echo
slab_thick = (x_diel_end - x_interface) * dx
t_backface = (2 * slab_thick) / (C0 / np.sqrt(eps_r))
t_front = (x_interface - tfsf_cfg.x_lo) * dx / C0
t_safe = t_front + t_backface
n_steps = min(int(t_safe / dt) - 50, 2000)
n_steps = max(n_steps, 800)

print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

# Materials: vacuum + dielectric slab
materials = init_materials(grid.shape)
materials = materials._replace(
    eps_r=materials.eps_r.at[x_interface:x_diel_end, :, :].set(eps_r)
)

state = init_state(grid.shape)
cp, cs = init_cpml(grid)

ts_scat = np.zeros(n_steps)
ts_inc = np.zeros(n_steps)

for step in range(n_steps):
    t = step * dt

    state = update_h(state, materials, dt, dx, periodic)
    state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
    state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
    tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

    state = update_e(state, materials, dt, dx, periodic)
    state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, dx, dt)
    state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
    tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, dx, dt, t)

    ts_scat[step] = float(state.ez[probe])
    ts_inc[step] = float(tfsf_st.e1d[ref_1d_idx])

# Spectral R in source bandwidth
freqs = np.fft.rfftfreq(n_steps, d=dt)
spec_inc = np.abs(np.fft.rfft(ts_inc))
spec_scat = np.abs(np.fft.rfft(ts_scat))

band = (freqs > 3e9) & (freqs < 7e9)
R_num = spec_scat[band] / np.maximum(spec_inc[band], 1e-30)
r_mean = float(np.mean(R_num))  # amplitude reflection coefficient
err_pct = abs(r_mean - r_analytic) / r_analytic * 100

print(f"\nNumerical |r| (mean 3-7 GHz): {r_mean:.4f}")
print(f"Analytical |r|: {r_analytic:.4f}")
print(f"Error: {err_pct:.1f}%")
if err_pct < 5:
    print("PASS: within 5% of Fresnel")
else:
    print(f"FAIL: {err_pct:.1f}% error")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Fresnel Reflectance: air / n={n_diel} (TFSF)", fontsize=13)

ax = axes[0]
t_ns = np.arange(n_steps) * dt * 1e9
ax.plot(t_ns, ts_scat, "b-", lw=0.8, label="Scattered (reflected)")
ax.plot(t_ns, ts_inc, "r--", lw=0.8, alpha=0.5, label="Incident (1D)")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Time Domain")

ax = axes[1]
f_ghz = freqs[band] / 1e9
ax.plot(f_ghz, R_num, "b-", lw=1.5, label=f"rfx |r| (mean={r_mean:.3f})")
ax.axhline(r_analytic, color="r", ls="--",
           label=f"Analytical |r|={r_analytic:.3f}")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("|R|")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Reflection Coefficient")

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "05_meep_fresnel.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
