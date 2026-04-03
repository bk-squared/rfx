"""Example 4: Microstrip Patch Antenna at 2.4 GHz

Rectangular patch antenna on FR4 substrate with probe feed using WirePort.
Demonstrates:
  - WirePort excitation through substrate
  - Time-domain spectral S11 estimation via FFT
  - Near-to-far-field transform for radiation pattern
  - 3D geometry visualization

Note: Calibrated S11 via wave decomposition requires a coaxial port model
for thin substrates (h << lambda). The FFT-based approach below shows the
resonance structure but is not a calibrated reflection coefficient.
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
h = 1.6e-3  # substrate thickness

# Analytical patch dimensions
W = C0 / (2 * f0) * np.sqrt(2 / (eps_r + 1))
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)
dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264) /
                   ((eps_eff - 0.258) * (W / h + 0.8)))
L = C0 / (2 * f0 * np.sqrt(eps_eff)) - 2 * dL

print(f"Patch: {L*1e3:.1f} x {W*1e3:.1f} mm, substrate h={h*1e3:.1f} mm")

# ---- Grid ----
dx = 0.5e-3
margin = 20e-3
dom_x = L + 2 * margin
dom_y = W + 2 * margin
dom_z = h + 15e-3

sim = Simulation(
    freq_max=4e9,
    domain=(dom_x, dom_y, dom_z),
    boundary="cpml",
    cpml_layers=8,
    dx=dx,
)

# ---- Materials & geometry ----
sigma_fr4 = 2 * np.pi * f0 * 8.854e-12 * eps_r * 0.02
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
    position=(feed_x, feed_y, 0),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    extent=h,
)

# ---- NTFF box for far-field ----
ntff_margin = 5e-3
sim.add_ntff_box(
    corner_lo=(ntff_margin, ntff_margin, ntff_margin),
    corner_hi=(dom_x - ntff_margin, dom_y - ntff_margin, dom_z - ntff_margin),
    freqs=jnp.array([f0]),
)

# Probe at the feed for time-domain signal
sim.add_probe((feed_x, feed_y, h / 2), "ez")

print(f"Domain: {dom_x*1e3:.0f}x{dom_y*1e3:.0f}x{dom_z*1e3:.0f} mm, dx={dx*1e3:.1f} mm")
print(f"Feed: ({feed_x*1e3:.1f}, {feed_y*1e3:.1f}) mm, extent={h*1e3:.1f} mm")

# ---- 3D Geometry visualization ----
print("\nRendering geometry...")
save_screenshot(sim, filename="examples/04_patch_geometry", dpi=150)
print("Saved: examples/04_patch_geometry.png")

# ---- Run simulation ----
n_steps = 4000
print(f"\nRunning simulation ({n_steps} steps)...")
from rfx.simulation import SnapshotSpec
result = sim.run(n_steps=n_steps, compute_s_params=False)

# ---- Field visualization ----
print("Saving field slice...")
from rfx.visualize import plot_field_slice
grid = sim._build_grid()
fig = plot_field_slice(result.state, grid, component="ez", axis="z",
                       index=grid.position_to_index((0, 0, h/2))[2],
                       title=f"Ez field at z=h/2 (substrate mid-plane)")
fig.savefig('examples/04_patch_field_ez.png', dpi=150)
plt.close(fig)
print("Saved: examples/04_patch_field_ez.png")

# ---- S11 via time-domain FFT ----
ts = np.array(result.time_series).ravel()
grid = sim._build_grid()

# Source waveform for normalization
pulse = GaussianPulse(f0=f0, bandwidth=0.8)
times = np.arange(n_steps) * grid.dt
src_signal = np.array([float(pulse(t)) for t in times])

# FFT
nfft = len(ts) * 4
spec_probe = np.abs(np.fft.rfft(ts, n=nfft))
spec_src = np.abs(np.fft.rfft(src_signal, n=nfft))
freqs_GHz = np.fft.rfftfreq(nfft, d=grid.dt) / 1e9

# Find resonance: peak in probe spectrum (cavity amplifies at resonance)
band = (freqs_GHz > 1.5) & (freqs_GHz < 3.5)
if np.any(band):
    idx_peak = np.argmax(spec_probe[band])
    f_res = freqs_GHz[band][idx_peak]
    peak_db = 20 * np.log10(spec_probe[band][idx_peak] / max(spec_probe[band].max(), 1e-30))
else:
    f_res = 0
    peak_db = 0

print(f"\n=== Resonance Results ===")
print(f"Resonance: {f_res:.2f} GHz (design: {f0/1e9:.1f} GHz)")
if f_res > 0:
    print(f"Freq error: {abs(f_res - f0/1e9) / (f0/1e9) * 100:.1f}%")

bw = s11_approx[band] < (s11_min + 3)
if np.any(bw):
    f_lo = freqs_GHz[band][bw][0]
    f_hi = freqs_GHz[band][bw][-1]
    print(f"-3dB BW:   {f_lo:.2f}-{f_hi:.2f} GHz ({(f_hi-f_lo)/f_res*100:.1f}%)")

# ---- S11 Plot ----
fig, ax = plt.subplots(figsize=(8, 5))
mask = (freqs_GHz > 1.5) & (freqs_GHz < 3.5)
ax.plot(freqs_GHz[mask], s11_approx[mask], 'b-', linewidth=1.5)
ax.axvline(f0/1e9, color='g', ls='--', alpha=0.5, label=f'Design {f0/1e9:.1f} GHz')
ax.axvline(f_res, color='r', ls=':', alpha=0.5, label=f'Resonance {f_res:.2f} GHz')
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Normalized |S11| (dB)')
ax.set_title(f'Patch Antenna Spectral Response (L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, FR4)')
ax.set_xlim(1.5, 3.5)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('examples/04_patch_antenna_s11.png', dpi=150)
print(f"\nS11 plot saved: examples/04_patch_antenna_s11.png")

# ---- Far-field radiation pattern ----
if result.ntff_data is not None and result.ntff_box is not None:
    from rfx.farfield import compute_far_field, radiation_pattern, directivity

    theta = np.linspace(0, np.pi, 181)
    phi = np.array([0.0, np.pi / 2])

    ff = compute_far_field(result.ntff_data, result.ntff_box, grid, theta, phi)

    D = directivity(ff)
    print(f"\n=== Far-field Results ===")
    print(f"Directivity: {D[0]:.1f} dBi at {f0/1e9:.1f} GHz")

    pat = radiation_pattern(ff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                     subplot_kw={"projection": "polar"})

    for ax, phi_idx, plane_name in [(ax1, 0, "E-plane"), (ax2, 1, "H-plane")]:
        r = pat[0, :, phi_idx]
        r = np.maximum(r, -40)
        r_shifted = r + 40
        ax.plot(theta, r_shifted, linewidth=1.5)
        ax.plot(-theta + 2 * np.pi, r_shifted, linewidth=1.5, alpha=0.5)
        ax.set_title(f"{plane_name} ({f0/1e9:.1f} GHz)\nD={D[0]:.1f} dBi")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    plt.tight_layout()
    plt.savefig('examples/04_patch_antenna_farfield.png', dpi=150)
    print(f"Far-field plot saved: examples/04_patch_antenna_farfield.png")
else:
    print("\nNTFF data not available")

print("\n=== Done ===")
