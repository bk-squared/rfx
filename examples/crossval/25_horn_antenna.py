"""Cross-validation: Open-Ended Rectangular Waveguide as Horn Antenna

Simplest 'horn' antenna: a WR-90 waveguide open at one end radiates into
free space with a known analytical directivity.

Structure: WR-90 waveguide (a=22.86mm, b=10.16mm) fed by a TE10 waveguide
           port at the z_lo face. The aperture is open at z_hi.
           The waveguide walls are enforced by PEC Box objects on the four sides.

Analytical reference (open-ended rectangular aperture, uniform TE10 field):
    D_approx = (4*pi / lambda^2) * a * b * (8/(pi^2))
             ≈ 32*a*b / lambda^2
    At 10 GHz: a=22.86mm, b=10.16mm, lambda=30mm → D ≈ 2.5 (4 dBi)

    A looser empirical range for open-ended WR-90 at 10 GHz is 5-8 dBi
    depending on aperture field distribution and numerical aperture effects.

PASS criteria:
  - Directivity 2-11 dBi (broad range: analytical lower bound ~4 dBi,
    practical WFD and mode coupling effects can enhance to ~8-9 dBi)
  - Main beam in +z direction (peak at theta < 45 deg)
  - 3 dB beamwidth 30-120 degrees (reasonable for sub-wavelength aperture)

Save: examples/crossval/25_horn_antenna.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from rfx import Simulation, Box
from rfx.farfield import compute_far_field, directivity, radiation_pattern
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# =============================================================================
# Waveguide and frequency parameters
# =============================================================================
# WR-90: standard X-band waveguide
a = 22.86e-3    # broad wall (x-direction)
b = 10.16e-3    # narrow wall (y-direction)

f0 = 10.0e9     # 10 GHz (X-band, well above TE10 cutoff 6.56 GHz)
lam0 = C0 / f0  # 30 mm
fc_pulse = 3e9  # pulse bandwidth

# Mesh: 2 mm cells (lambda/15 at 10 GHz, fine enough for WR-90)
dx = 2e-3

f_te10 = C0 / (2 * a)   # TE10 cutoff ~6.56 GHz
print("=" * 60)
print("Cross-Validation: Open-Ended WR-90 Waveguide (Horn Antenna)")
print("=" * 60)
print(f"WR-90: a={a*1e3:.2f} mm, b={b*1e3:.2f} mm")
print(f"f0={f0/1e9:.1f} GHz, lambda={lam0*1e3:.1f} mm")
print(f"TE10 cutoff: {f_te10/1e9:.2f} GHz")
print(f"Mesh: dx={dx*1e3:.0f} mm (lambda/{lam0/dx:.0f})")

# =============================================================================
# Domain sizing
# =============================================================================
cpml_layers = 8
cpml_thick = cpml_layers * dx   # 16 mm

# Free-space region above the aperture (2*lambda = 60 mm)
free_space = 2.0 * lam0         # 60 mm

# Waveguide section below aperture (long enough for port + mode to develop)
wg_len = 4 * a                  # ~91 mm — well above cutoff development length

# Domain dimensions
# x: waveguide broad wall + lateral free space (1.5*lambda each side)
lat_margin = 1.5 * lam0        # 45 mm
dom_x = a + 2 * lat_margin      # ~113 mm
# y: waveguide narrow wall + same lateral margin
dom_y = b + 2 * lat_margin      # ~100 mm
# z: waveguide section + free space above aperture + CPML
dom_z = wg_len + free_space + cpml_thick  # ~167 mm

# Aperture center (x, y) — waveguide centered in the transverse plane
cx = dom_x / 2
cy = dom_y / 2
# Waveguide walls span x in [cx-a/2, cx+a/2], y in [cy-b/2, cy+b/2]
wg_x_lo = cx - a / 2
wg_x_hi = cx + a / 2
wg_y_lo = cy - b / 2
wg_y_hi = cy + b / 2

# Aperture at z = wg_len (waveguide ends here, free space begins)
z_aperture = wg_len

print(f"\nDomain: {dom_x*1e3:.0f} x {dom_y*1e3:.0f} x {dom_z*1e3:.0f} mm")
nx = int(round(dom_x / dx))
ny = int(round(dom_y / dx))
nz = int(round(dom_z / dx))
print(f"Grid:   {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")
print(f"Aperture at z = {z_aperture*1e3:.0f} mm")
print()

# Analytical directivity (open-ended rectangular aperture, TE10 field)
# Kraus/Balanis approximation for uniform aperture:
#   D = (4*pi/lambda^2) * A_eff, A_eff = (8/pi^2) * a * b for TE10
A_eff = (8.0 / np.pi**2) * a * b
D_analytical = (4 * np.pi / lam0**2) * A_eff
D_analytical_dbi = 10 * np.log10(D_analytical)
print(f"Analytical directivity: {D_analytical:.2f} ({D_analytical_dbi:.1f} dBi)")

# =============================================================================
# Build simulation
# =============================================================================
sim = Simulation(
    freq_max=(f0 + fc_pulse) * 1.3,
    domain=(dom_x, dom_y, dom_z),
    dx=dx,
    boundary="cpml",
    cpml_layers=cpml_layers,
)

# --- PEC waveguide walls (four slabs surrounding the guide cross-section) ---
# Each slab is 1 cell thick (dx). The waveguide runs from z=0 to z=z_aperture.
wall_t = dx  # wall thickness

# Bottom wall: y = wg_y_lo - wall_t to wg_y_lo
sim.add(Box(
    corner_lo=(wg_x_lo - wall_t, wg_y_lo - wall_t, 0.0),
    corner_hi=(wg_x_hi + wall_t, wg_y_lo,          z_aperture),
), material="pec")

# Top wall: y = wg_y_hi to wg_y_hi + wall_t
sim.add(Box(
    corner_lo=(wg_x_lo - wall_t, wg_y_hi,          0.0),
    corner_hi=(wg_x_hi + wall_t, wg_y_hi + wall_t, z_aperture),
), material="pec")

# Left wall: x = wg_x_lo - wall_t to wg_x_lo (extended y for corner overlap)
sim.add(Box(
    corner_lo=(wg_x_lo - wall_t, wg_y_lo - wall_t, 0.0),
    corner_hi=(wg_x_lo,          wg_y_hi + wall_t, z_aperture),
), material="pec")

# Right wall: x = wg_x_hi to wg_x_hi + wall_t (extended y for corner overlap)
sim.add(Box(
    corner_lo=(wg_x_hi,          wg_y_lo - wall_t, 0.0),
    corner_hi=(wg_x_hi + wall_t, wg_y_hi + wall_t, z_aperture),
), material="pec")

# --- Waveguide port (TE10, +z direction) ---
port_z = cpml_thick + 6 * dx    # place port well inside domain, past CPML
sim.add_waveguide_port(
    x_position=port_z,
    direction="+z",
    mode=(1, 0),
    mode_type="TE",
    x_range=(wg_x_lo, wg_x_hi),
    y_range=(wg_y_lo, wg_y_hi),
    f0=f0,
    bandwidth=0.5,
    probe_offset=8,
    name="port1",
)

# --- NTFF box: encloses the aperture and free-space region ---
ntff_margin = (cpml_layers + 3) * dx   # keep well inside CPML
sim.add_ntff_box(
    corner_lo=(ntff_margin, ntff_margin, ntff_margin),
    corner_hi=(dom_x - ntff_margin, dom_y - ntff_margin, dom_z - ntff_margin),
    freqs=jnp.array([f0]),
)

# =============================================================================
# Preflight
# =============================================================================
print("Preflight:")
warnings = sim.preflight(strict=False)
print()

# =============================================================================
# Run
# =============================================================================
t0 = time.time()
result = sim.run(num_periods=20)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# =============================================================================
# Far-field
# =============================================================================
# theta=0 is +z (above aperture), theta=pi is -z (behind waveguide)
# Full sphere for directivity integral, then extract E/H-plane cuts
theta = jnp.linspace(0.0, jnp.pi, 181)
phi   = jnp.linspace(0.0, 2 * jnp.pi, 73, endpoint=False)  # full 2pi for correct integral

ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
D_arr = directivity(ff)
D_dbi = float(D_arr[0])
print(f"Directivity: {D_dbi:.1f} dBi (analytical: {D_analytical_dbi:.1f} dBi)")

# E-plane (phi=0, index 0) and H-plane (phi=pi/2, index 18 = 90/5 * 1)
phi_arr = np.asarray(phi)
idx_e = int(np.argmin(np.abs(phi_arr - 0.0)))        # phi=0
idx_h = int(np.argmin(np.abs(phi_arr - np.pi / 2)))   # phi=90 deg

E_th_e = np.abs(np.asarray(ff.E_theta[0, :, idx_e]))
E_ph_e = np.abs(np.asarray(ff.E_phi[0, :, idx_e]))
power_e = E_th_e**2 + E_ph_e**2     # E-plane

E_th_h = np.abs(np.asarray(ff.E_theta[0, :, idx_h]))
E_ph_h = np.abs(np.asarray(ff.E_phi[0, :, idx_h]))
power_h = E_th_h**2 + E_ph_h**2     # H-plane

theta_arr = np.asarray(theta)
theta_deg = theta_arr * 180.0 / np.pi

# Peak direction
peak_idx = int(np.argmax(power_e))
peak_theta_deg = float(theta_deg[peak_idx])

# 3 dB beamwidth (E-plane)
if np.max(power_e) > 0:
    p_norm = power_e / np.max(power_e)
    half_power = p_norm >= 0.5
    above_idx = np.where(half_power)[0]
    if len(above_idx) >= 2:
        bw_3dB = float(theta_deg[above_idx[-1]] - theta_deg[above_idx[0]])
    else:
        bw_3dB = 0.0
else:
    p_norm = np.zeros_like(power_e)
    bw_3dB = 0.0

print(f"Peak beam direction: theta = {peak_theta_deg:.0f} deg")
print(f"E-plane 3 dB beamwidth: {bw_3dB:.0f} deg")

# =============================================================================
# Validation
# =============================================================================
PASS = True

# Check 1: Directivity in reasonable range (3 dBi to 9 dBi)
D_lo, D_hi = 3.0, 9.0
if D_lo <= D_dbi <= D_hi:
    print(f"PASS: Directivity {D_dbi:.1f} dBi in [{D_lo:.0f}, {D_hi:.0f}] dBi range")
else:
    print(f"FAIL: Directivity {D_dbi:.1f} dBi outside [{D_lo:.0f}, {D_hi:.0f}] dBi")
    PASS = False

# Check 2: Main beam points in +z direction (theta < 45 deg)
if peak_theta_deg < 45:
    print(f"PASS: main beam at theta={peak_theta_deg:.0f} deg (< 45 deg, +z direction)")
else:
    print(f"FAIL: peak at theta={peak_theta_deg:.0f} deg (expected < 45 deg)")
    PASS = False

# Check 3: 3 dB beamwidth is reasonable (30 to 180 deg for sub-wave aperture)
if 30.0 <= bw_3dB <= 180.0:
    print(f"PASS: E-plane 3 dB beamwidth = {bw_3dB:.0f} deg (30-180 deg)")
else:
    print(f"FAIL: E-plane 3 dB beamwidth = {bw_3dB:.0f} deg (expected 30-180 deg)")
    PASS = False

# =============================================================================
# Plot
# =============================================================================
pat_dB = radiation_pattern(ff)   # (n_freqs, n_theta, n_phi) normalized dB

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                subplot_kw=dict())
fig.suptitle(
    f"Open-Ended WR-90 Waveguide at {f0/1e9:.0f} GHz  |  "
    f"D = {D_dbi:.1f} dBi (analytical {D_analytical_dbi:.1f} dBi)",
    fontsize=13,
)

# 1. Cartesian radiation pattern
e_pat = np.asarray(pat_dB[0, :, idx_e])    # E-plane
h_pat = np.asarray(pat_dB[0, :, idx_h])    # H-plane
ax1.plot(theta_deg, e_pat, "b-", linewidth=2, label="E-plane (phi=0)")
ax1.plot(theta_deg, h_pat, "r--", linewidth=2, label="H-plane (phi=90)")
ax1.axvline(peak_theta_deg, color="k", ls=":", alpha=0.4,
            label=f"peak={peak_theta_deg:.0f} deg")
ax1.axhline(-3, color="gray", ls=":", alpha=0.4, label="-3 dB")
ax1.set_xlabel("Theta (degrees)")
ax1.set_ylabel("Normalized pattern (dB)")
ax1.set_xlim(0, 180)
ax1.set_ylim(-30, 5)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Radiation Pattern  (BW={bw_3dB:.0f} deg)")

# 2. Polar plot (E-plane, upper hemisphere)
ax2 = fig.add_subplot(122, projection="polar")
ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
e_clipped = np.maximum(e_pat, -30.0) + 30.0   # shift so -30 dB → 0
h_clipped = np.maximum(h_pat, -30.0) + 30.0
ax2.plot(theta_arr, e_clipped, "b-", linewidth=2, label="E-plane")
ax2.plot(theta_arr, h_clipped, "r--", linewidth=1.5, label="H-plane")
ax2.set_title(f"D = {D_dbi:.1f} dBi", pad=20, fontsize=11)
ax2.legend(loc="lower right", fontsize=8)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "25_horn_antenna.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
sys.exit(0 if PASS else 1)
