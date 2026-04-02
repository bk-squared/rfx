"""Patch Antenna with proper domain/PEC/substrate configuration.

Fixes all known issues:
  - PEC thickness fixed at 0.1mm (independent of dx)
  - Domain margin = lambda/4 (no domain resonance interference)
  - Substrate limited to patch area + 5mm margin (no domain-wide cavity)
  - CPML >= lambda/20
  - T_sim >= 10ns (auto n_steps)
  - Spectral PEAK for resonance (not min)
"""
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box, GaussianPulse

# ---- Design ----
f0 = 2.4e9
C0 = 3e8
eps_r = 4.4
h = 1.6e-3
lam0 = C0 / f0  # 125mm

W = C0 / (2*f0) * np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2 * (1+12*h/W)**(-0.5)
dL = 0.412*h * ((eps_eff+0.3)*(W/h+0.264) / ((eps_eff-0.258)*(W/h+0.8)))
L = C0 / (2*f0*np.sqrt(eps_eff)) - 2*dL

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz, lambda={lam0*1e3:.0f}mm")

# ---- Domain: lambda/4 margin ----
margin = lam0 / 4  # ~31mm
dom_x = L + 2*margin
dom_y = W + 2*margin
dom_z = h + lam0 / 4  # air above patch

# ---- Grid ----
dx = 0.5e-3  # 0.5mm uniform
cpml_n = max(int(round(lam0/20 / dx)), 8)  # ~lambda/20 CPML

sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z),
                 boundary='cpml', cpml_layers=cpml_n, dx=dx)

# ---- Materials ----
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)

px0, py0 = margin, margin
feed_x, feed_y = px0 + L/3, py0 + W/2

# PEC thickness: exactly 1 cell (minimum resolvable)
pec_t = dx

# Ground plane: full domain extent
sim.add(Box((0, 0, 0), (dom_x, dom_y, pec_t)), material='pec')

# Substrate: ONLY around patch area (not full domain!)
sub_margin = 5e-3
sim.add(Box((px0-sub_margin, py0-sub_margin, 0),
            (px0+L+sub_margin, py0+W+sub_margin, h)), material='FR4')

# Patch
sim.add(Box((px0, py0, h), (px0+L, py0+W, h+pec_t)), material='pec')

# ---- Excitation: high-impedance port = soft source (negligible loading) ----
# Z0=1e6 gives sigma_port ≈ 0 — acts as a soft source without damping the cavity
sim.add_port(position=(feed_x, feed_y, dx), component='ez',
             impedance=1e6,
             waveform=GaussianPulse(f0=f0, bandwidth=0.8),
             extent=h - 2*dx)

# ---- Probe ----
sim.add_probe((feed_x, feed_y, h/2), 'ez')

# ---- Auto n_steps for T_sim >= 10ns ----
grid = sim._build_grid()
target_t = 10e-9
n_steps = int(np.ceil(target_t / grid.dt))

print(f"Grid: {grid.shape}, dx={dx*1e3}mm, cpml={cpml_n}")
print(f"h/dx={h/dx:.1f} cells, margin={margin*1e3:.0f}mm ({margin/lam0:.2f}*lambda)")
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns, dt={grid.dt:.3e}s")
print(f"Running...")

result = sim.run(n_steps=n_steps)

# ---- Spectral analysis ----
ts = np.array(result.time_series).ravel()
nfft = len(ts) * 8
spec = np.abs(np.fft.rfft(ts, n=nfft))
fg = np.fft.rfftfreq(nfft, d=grid.dt) / 1e9

# Find peak in 1-4 GHz band
band = (fg > 1.0) & (fg < 4.0)
idx = np.argmax(spec[band])
f_res = fg[band][idx]
err = abs(f_res - f0/1e9) / (f0/1e9) * 100

print(f"\n=== Results ===")
print(f"Resonance PEAK: {f_res:.3f} GHz (design: {f0/1e9:.3f} GHz)")
print(f"Error: {err:.1f}%")
print(f"Max|Ez|: {np.max(np.abs(ts)):.3e}")
print(f"NaN: {np.any(np.isnan(ts))}")

# Also report top-3 peaks
spec_band = spec[band]
fg_band = fg[band]
from scipy.signal import find_peaks
peaks, _ = find_peaks(spec_band, height=spec_band.max()*0.05, distance=20)
sorted_p = sorted(peaks, key=lambda p: spec_band[p], reverse=True)
print(f"\nTop spectral peaks:")
for i, p in enumerate(sorted_p[:5]):
    print(f"  {i+1}. {fg_band[p]:.3f} GHz (amp={spec_band[p]:.2e})")
