"""Example 4: Microstrip Patch Antenna at 2.4 GHz

Rectangular patch antenna on FR4 substrate with probe feed using WirePort
(multi-cell lumped port spanning ground to patch through substrate).

Expected: S11 dip near 2.4 GHz.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import Simulation, Box, GaussianPulse

# ---- Design ----
f0 = 2.4e9
C0 = 3e8
eps_r = 4.4
h = 1.6e-3  # substrate thickness

# Analytical patch dimensions
W = C0 / (2 * f0) * np.sqrt(2 / (eps_r + 1))
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)
dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264) /
                   ((eps_eff - 0.258) * (W / h + 0.8)))
L = C0 / (2 * f0 * np.sqrt(eps_eff)) - 2 * dL

print(f"Patch: {L*1e3:.1f} x {W*1e3:.1f} mm, substrate h={h*1e3:.1f} mm")

# ---- Grid: dx = 0.5mm for proper substrate resolution (3 cells) ----
dx = 0.5e-3
margin = 20e-3
dom_x = L + 2 * margin
dom_y = W + 2 * margin
dom_z = h + 15e-3  # substrate + air

sim = Simulation(
    freq_max=4e9,
    domain=(dom_x, dom_y, dom_z),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

# ---- Materials & geometry ----
sigma_fr4 = 2 * np.pi * f0 * 8.854e-12 * eps_r * 0.02  # tan_d=0.02
sim.add_material("FR4", eps_r=eps_r, sigma=sigma_fr4)

# Ground plane (z=0)
sim.add(Box((0, 0, 0), (dom_x, dom_y, dx)), material="pec")
# Substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="FR4")
# Patch (z = h)
px0, py0 = margin, margin
sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h + dx)), material="pec")

# ---- Feed: WirePort spanning ground to patch ----
feed_x = px0 + L / 3
feed_y = py0 + W / 2

sim.add_port(
    position=(feed_x, feed_y, dx),  # start just above ground
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    extent=h - 2 * dx,  # span through substrate (ground+dx to patch-dx)
)

print(f"Domain: {dom_x*1e3:.0f}x{dom_y*1e3:.0f}x{dom_z*1e3:.0f} mm, dx={dx*1e3:.1f} mm")
print(f"Feed: ({feed_x*1e3:.1f}, {feed_y*1e3:.1f}) mm, extent={h*1e3:.1f} mm")
print("Running...")

# Add a probe at the feed point to record time-domain signal
sim.add_probe((feed_x, feed_y, h / 2), "ez")

result = sim.run(n_steps=4000)

# ---- Results via FFT of time-domain probe signal ----
ts = np.array(result.time_series).ravel()
grid = sim._build_grid()
spec = np.abs(np.fft.rfft(ts, n=len(ts) * 4))
freqs_GHz = np.fft.rfftfreq(len(ts) * 4, d=grid.dt) / 1e9
# Normalize spectrum
s11_dB = 20 * np.log10(spec / max(spec.max(), 1e-30))

mask = freqs_GHz > 1.5
idx_min = np.argmin(s11_dB[mask])
f_res = freqs_GHz[mask][idx_min]
s11_min = s11_dB[mask][idx_min]

print(f"\nResonance: {f_res:.2f} GHz (design: {f0/1e9:.1f} GHz)")
print(f"S11 min:   {s11_min:.1f} dB")
if f_res > 0:
    print(f"Freq error: {abs(f_res - f0/1e9) / (f0/1e9) * 100:.1f}%")

bw = s11_dB[mask] < -10
if np.any(bw):
    f_lo = freqs_GHz[mask][bw][0]
    f_hi = freqs_GHz[mask][bw][-1]
    print(f"-10dB BW:  {f_lo:.2f}-{f_hi:.2f} GHz ({(f_hi-f_lo)/f_res*100:.1f}%)")
else:
    print("-10dB BW:  not achieved")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(freqs_GHz, s11_dB, 'b-', linewidth=1.5)
ax.axhline(-10, color='r', ls='--', alpha=0.5, label='-10 dB')
ax.axvline(f0/1e9, color='g', ls='--', alpha=0.5, label=f'Design {f0/1e9:.1f} GHz')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('|S11| (dB)')
ax.set_title(f'Patch Antenna S11 (L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, FR4)')
ax.set_xlim(1.5, 3.5)
ax.set_ylim(-25, 0)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('examples/04_patch_antenna.png', dpi=150)
print(f"Plot saved: examples/04_patch_antenna.png")
