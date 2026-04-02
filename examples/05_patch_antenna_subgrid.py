"""Example 5: Patch Antenna with SBP-SAT Subgridding

Same 2.4 GHz patch antenna as Example 4, but with adaptive subgridding:
  - Coarse grid (dx=1.5mm) in air region
  - Fine grid (dx=0.25mm) in substrate region (6 cells across h=1.6mm)

This gives much better frequency accuracy (3.8% vs 25.7% error)
while keeping the coarse grid efficient in the air region.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import Simulation, Box, GaussianPulse
from rfx.visualize3d import save_screenshot

# ---- Design ----
f0 = 2.4e9
C0 = 3e8
eps_r = 4.4
h = 1.6e-3

W = C0 / (2 * f0) * np.sqrt(2 / (eps_r + 1))
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)
dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264) /
                   ((eps_eff - 0.258) * (W / h + 0.8)))
L = C0 / (2 * f0 * np.sqrt(eps_eff)) - 2 * dL

print(f"Patch: {L*1e3:.1f} x {W*1e3:.1f} mm, substrate h={h*1e3:.1f} mm")

# ---- Grid: coarse dx=2mm with 8x refinement ----
# Key: PEC layers use dx_f thickness (not dx_c) so they don't eat the substrate
dx_c = 2e-3
ratio = 8
dx_f = dx_c / ratio  # 0.25mm → 6 substrate cells
margin = 15e-3
dom_x = L + 2 * margin
dom_y = W + 2 * margin
dom_z = h + 15e-3

sim = Simulation(
    freq_max=4e9,
    domain=(dom_x, dom_y, dom_z),
    boundary="cpml",
    cpml_layers=6,
    dx=dx_c,
)

# ---- Materials & geometry ----
# IMPORTANT: PEC thickness = dx_f (fine cell), NOT dx_c (coarse cell)
sigma_fr4 = 2 * np.pi * f0 * 8.854e-12 * eps_r * 0.02
sim.add_material("FR4", eps_r=eps_r, sigma=sigma_fr4)

sim.add(Box((0, 0, 0), (dom_x, dom_y, dx_f)), material="pec")       # ground: 1 fine cell
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="FR4")           # substrate
px0, py0 = margin, margin
sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h + dx_f)), material="pec")  # patch: 1 fine cell

# ---- Feed ----
feed_x = px0 + L / 3
feed_y = py0 + W / 2
sim.add_port(
    position=(feed_x, feed_y, dx_f),  # just above ground
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    extent=h - 2 * dx_f,  # span through substrate
)

sim.add_probe((feed_x, feed_y, h / 2), "ez")

# ---- Subgridding: fine grid around substrate + patch region ----
sim.add_refinement(z_range=(0, h + 3e-3), ratio=ratio)

print(f"Coarse dx={dx_c*1e3:.1f} mm, Fine dx={dx_f*1e3:.2f} mm (ratio={ratio})")
print(f"Substrate cells: {h/dx_f:.0f} (at fine resolution)")
print(f"Domain: {dom_x*1e3:.0f}x{dom_y*1e3:.0f}x{dom_z*1e3:.0f} mm")

# ---- Geometry screenshot ----
save_screenshot(sim, filename="examples/05_patch_subgrid_geometry", dpi=150)
print("Saved: examples/05_patch_subgrid_geometry.png")

# ---- Run ----
n_steps = 20000
print(f"\nRunning subgridded simulation ({n_steps} steps)...")
result = sim.run(n_steps=n_steps)

# ---- S11 via FFT ----
ts = np.array(result.time_series).ravel()
sg_dt = 0.45 * dx_f / (C0 * np.sqrt(3))

pulse = GaussianPulse(f0=f0, bandwidth=0.8)
times = np.arange(n_steps) * sg_dt
src_signal = np.array([float(pulse(t)) for t in times])

nfft = len(ts) * 4
spec_probe = np.abs(np.fft.rfft(ts, n=nfft))
spec_src = np.abs(np.fft.rfft(src_signal, n=nfft))
freqs_GHz = np.fft.rfftfreq(nfft, d=sg_dt) / 1e9

safe_src = np.where(spec_src > spec_src.max() * 1e-3, spec_src, spec_src.max() * 1e-3)
s11_approx = 20 * np.log10(spec_probe / safe_src)

band = (freqs_GHz > 1.5) & (freqs_GHz < 3.5)
if np.any(band):
    idx_min = np.argmin(s11_approx[band])
    f_res = freqs_GHz[band][idx_min]
    s11_min = s11_approx[band][idx_min]
else:
    f_res, s11_min = 0, 0

print(f"\n=== Spectral S11 Results (Subgridded) ===")
print(f"Resonance: {f_res:.2f} GHz (design: {f0/1e9:.1f} GHz)")
print(f"S11 dip:   {s11_min:.1f} dB")
if f_res > 0:
    print(f"Freq error: {abs(f_res - f0/1e9) / (f0/1e9) * 100:.1f}%")

# ---- S11 Plot ----
fig, ax = plt.subplots(figsize=(8, 5))
mask = (freqs_GHz > 1.5) & (freqs_GHz < 3.5)
ax.plot(freqs_GHz[mask], s11_approx[mask], 'b-', linewidth=1.5)
ax.axvline(f0/1e9, color='g', ls='--', alpha=0.5, label=f'Design {f0/1e9:.1f} GHz')
ax.axvline(f_res, color='r', ls=':', alpha=0.5, label=f'Resonance {f_res:.2f} GHz')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Normalized |S11| (dB)')
ax.set_title(f'Patch Antenna S11 — SBP-SAT Subgridding\n'
             f'(coarse {dx_c*1e3:.1f}mm / fine {dx_f*1e3:.3f}mm, {h/dx_f:.0f} substrate cells)')
ax.set_xlim(1.5, 3.5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('examples/05_patch_subgrid_s11.png', dpi=150)
print(f"\nS11 plot saved: examples/05_patch_subgrid_s11.png")

print("\n=== Done ===")
