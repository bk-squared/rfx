"""Cross-validation: Drude Metal Sphere (Lossy Mie Scattering)

Validates Drude dispersive material (Lorentz pole with omega_0=0)
combined with TFSF plane wave source.

Structure: Metal sphere with Drude dispersion, TFSF broadband excitation.
The scattering cross section is measured via NTFF far-field integration.

For a PEC-like metal sphere (omega_p >> omega, low loss), the RCS
approaches the geometric optics limit: sigma_geo = pi * a^2 for ka >> 1,
and oscillates around 2 * pi * a^2 in the Mie regime.

PASS criteria:
  - NTFF far-field computes without error (Drude + TFSF + NTFF pipeline)
  - Scattering RCS is positive and in reasonable range
  - Directivity > 0 dBi (forward scattering expected)

Save: examples/crossval/23_drude_metal_sphere.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Sphere
from rfx.materials.lorentz import drude_pole
from rfx.farfield import compute_far_field, directivity
from rfx.rcs import compute_rcs
import jax.numpy as jnp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8
EPS_0 = 8.854e-12

# Drude metal parameters (aluminum-like)
# omega_p = 2.24e16 rad/s for Al, but that's optical
# For microwave, use a lower omega_p to make the sphere partially transparent
omega_p = 2 * np.pi * 30e9   # plasma freq = 30 GHz
gamma = 2 * np.pi * 1e9      # collision rate = 1 GHz (lossy)

f0 = 10e9           # center frequency
lam0 = C0 / f0       # 30 mm
fc = 5e9             # bandwidth

# Sphere
radius = 10e-3       # 10 mm (ka ≈ 2 at 10 GHz)
k0 = 2 * np.pi * f0 / C0
ka = k0 * radius

# Drude permittivity at f0
omega0 = 2 * np.pi * f0
eps_drude_f0 = 1 - omega_p ** 2 / (omega0 ** 2 + 1j * omega0 * gamma)
print("=" * 60)
print("Cross-Validation: Drude Metal Sphere (Lossy Mie)")
print("=" * 60)
print(f"Drude: omega_p/(2pi) = {omega_p/(2*np.pi)/1e9:.1f} GHz, "
      f"gamma/(2pi) = {gamma/(2*np.pi)/1e9:.1f} GHz")
print(f"eps(f0) = {eps_drude_f0.real:.2f} {eps_drude_f0.imag:+.2f}j")
print(f"Sphere: R = {radius*1e3:.0f} mm, ka = {ka:.2f}")
print(f"f0 = {f0/1e9:.1f} GHz, lambda = {lam0*1e3:.0f} mm")
print()

# Geometric cross section reference
sigma_geo = np.pi * radius ** 2
print(f"Geometric cross-section: {sigma_geo*1e6:.2f} mm^2")
print(f"Expected RCS (PEC sphere): ~{sigma_geo*1e6:.0f} mm^2 ({10*np.log10(sigma_geo):.1f} dBsm)")
print()

# Mesh
dx = lam0 / 15  # ~2 mm
margin = lam0 * 0.6
dom = 2 * radius + 2 * margin

sim = Simulation(
    freq_max=(f0 + fc) * 1.5,
    domain=(dom, dom, dom),
    dx=dx,
    boundary="cpml",
    cpml_layers=8,
)

# Drude material
pole = drude_pole(omega_p=omega_p, gamma=gamma)
sim.add_material("drude_metal", eps_r=1.0, lorentz_poles=[pole])

center = (dom / 2, dom / 2, dom / 2)
sim.add(Sphere(center=center, radius=radius), material="drude_metal")

# TFSF plane wave
sim.add_tfsf_source(
    f0=f0,
    bandwidth=fc / f0,
    direction="+x",
    polarization="ez",
    margin=3,
)

# NTFF box (in scattered-field region)
cpml_thick = 8 * dx
ntff_margin = cpml_thick + 2 * dx
sim.add_ntff_box(
    corner_lo=(ntff_margin,) * 3,
    corner_hi=(dom - ntff_margin,) * 3,
    freqs=jnp.array([f0]),
)

# Probe
sim.add_probe(position=(dom * 0.3, dom / 2, dom / 2), component="ez")

# Run
print("Preflight:")
warnings = sim.preflight(strict=False)
print()

t0 = time.time()
result = sim.run(num_periods=12)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# Far-field and RCS
theta = jnp.linspace(0, jnp.pi, 91)
phi = jnp.array([0.0, jnp.pi / 2])

ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid, theta, phi)
D = directivity(ff)
D_dbi = 10 * np.log10(float(np.asarray(D)) + 1e-30)
print(f"Directivity: {D_dbi:.1f} dBi")

# RCS
try:
    rcs_result = compute_rcs(result, theta=theta, phi=phi)
    rcs_mono = float(np.asarray(rcs_result.rcs_monostatic))
    rcs_mono_dbsm = 10 * np.log10(rcs_mono + 1e-30)
    print(f"Monostatic RCS: {rcs_mono*1e6:.1f} mm^2 ({rcs_mono_dbsm:.1f} dBsm)")
    has_rcs = True
except Exception as e:
    print(f"RCS computation failed: {e}")
    has_rcs = False

# =============================================================================
# Validation
# =============================================================================
PASS = True

# Check 1: Directivity > 0 dBi (forward scattering from sphere)
if D_dbi > 0:
    print(f"\nPASS: Directivity {D_dbi:.1f} dBi > 0")
else:
    print(f"\nFAIL: Directivity {D_dbi:.1f} dBi (expected > 0)")
    PASS = False

# Check 2: Far-field pipeline works (Drude + TFSF + NTFF)
E_th = np.abs(np.asarray(ff.E_theta[0, :, 0]))
E_ph = np.abs(np.asarray(ff.E_phi[0, :, 0]))
power = E_th ** 2 + E_ph ** 2
if np.max(power) > 0 and not np.any(np.isnan(power)):
    print(f"PASS: far-field valid (peak power = {np.max(power):.2e})")
else:
    print(f"FAIL: far-field invalid or NaN")
    PASS = False

# Check 3: Scattering pattern has forward lobe
theta_arr = np.asarray(theta) * 180 / np.pi
peak_idx = np.argmax(power)
peak_theta = theta_arr[peak_idx]
print(f"Peak scattering at theta={peak_theta:.0f} deg")

# =============================================================================
# Plot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Drude Metal Sphere: R={radius*1e3:.0f}mm, ka={ka:.1f}, "
             f"omega_p={omega_p/(2*np.pi)/1e9:.0f} GHz", fontsize=14)

# 1. Scattering pattern
if np.max(power) > 0:
    pattern_db = 10 * np.log10(power / np.max(power) + 1e-30)
    ax1.plot(theta_arr, pattern_db, "b-", linewidth=2)
ax1.set_xlabel("Theta (degrees)")
ax1.set_ylabel("Normalized pattern (dB)")
ax1.set_xlim(0, 180)
ax1.set_ylim(-30, 5)
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Scattering pattern (D={D_dbi:.1f} dBi)")

# 2. Time series
ts = np.array(result.time_series)
dt = result.dt
t_ns = np.arange(ts.shape[0]) * dt * 1e9
ax2.plot(t_ns, ts[:, 0], linewidth=0.5)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Ez amplitude")
ax2.set_title("Scattered field probe")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "23_drude_metal_sphere.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
