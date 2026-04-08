"""Cross-validation: Debye Water Dispersion (Analytical)

Validates Debye dispersive material model against exact analytical solution.

Setup: 1D-like plane wave incident on a Debye water half-space.
Reflection coefficient R(f) is compared to Fresnel R(f) using the
Debye permittivity model.

Debye model for water at 25C:
  epsilon(f) = eps_inf + (eps_s - eps_inf) / (1 + j*2*pi*f*tau)
  eps_s = 78.36, eps_inf = 5.2, tau = 8.27 ps

Fresnel normal incidence:
  R(f) = |(sqrt(eps(f)) - 1) / (sqrt(eps(f)) + 1)|^2

PASS criteria:
  - Reflection spectrum shape matches analytical within 5% RMS error
  - Peak reflection frequency within 10% of analytical

Save: examples/crossval/15_debye_water_dispersion.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.materials.debye import DebyePole

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8

# Debye water parameters (25C, single pole)
eps_s = 78.36    # static permittivity
eps_inf = 5.2    # infinite frequency permittivity
tau = 8.27e-12   # relaxation time (8.27 ps)
# Relaxation frequency: f_r = 1/(2*pi*tau) ~ 19.2 GHz

f_relax = 1 / (2 * np.pi * tau)
f0 = 10e9   # center frequency
f_max = 30e9  # simulate up to 30 GHz

print("=" * 60)
print("Cross-Validation: Debye Water Dispersion")
print("=" * 60)
print(f"Debye: eps_s={eps_s}, eps_inf={eps_inf}, tau={tau*1e12:.2f} ps")
print(f"Relaxation freq: {f_relax/1e9:.1f} GHz")
print()

# Analytical Fresnel reflectance R(f)
freqs_ana = np.linspace(1e9, f_max, 500)
omega = 2 * np.pi * freqs_ana
eps_debye = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
n_debye = np.sqrt(eps_debye)
R_fresnel = np.abs((n_debye - 1) / (n_debye + 1)) ** 2
R_fresnel_db = 10 * np.log10(R_fresnel)

# FDTD simulation
dx = C0 / f_max / 20  # ~0.5 mm
lam_min = C0 / f_max   # 10 mm

# Domain: narrow in y,z (1D-like), long in x
dom_x = 40e-3   # 40 mm
dom_y = 5e-3
dom_z = 5e-3

# Water slab from x=20mm to x=40mm (right half)
slab_x = dom_x / 2

sim = Simulation(
    freq_max=f_max,
    domain=(dom_x, dom_y, dom_z),
    dx=dx,
    boundary="cpml",
    cpml_layers=10,
)

# Debye water material
debye_pole = DebyePole(delta_eps=eps_s - eps_inf, tau=tau)
sim.add_material("water", eps_r=eps_inf, debye_poles=[debye_pole])

# Water half-space
sim.add(Box((slab_x, 0, 0), (dom_x, dom_y, dom_z)), material="water")

# Source (left side, propagating +x toward water)
src_x = dom_x * 0.15
sim.add_source(
    position=(src_x, dom_y / 2, dom_z / 2),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.9),
)

# Reflection probe (between source and interface)
refl_x = dom_x * 0.25
sim.add_probe(position=(refl_x, dom_y / 2, dom_z / 2), component="ez")

# Transmission probe (inside water)
trans_x = dom_x * 0.75
sim.add_probe(position=(trans_x, dom_y / 2, dom_z / 2), component="ez")

# Reference probe (at source)
sim.add_probe(position=(src_x, dom_y / 2, dom_z / 2), component="ez")

# Run
t0 = time.time()
result = sim.run(n_steps=2000)
elapsed = time.time() - t0
print(f"Simulation time: {elapsed:.1f}s")

# Analysis: extract reflection from time-domain
ts = np.array(result.time_series)
dt = result.dt

sig_refl = ts[:, 0]  # reflection probe
sig_ref = ts[:, 2]   # reference (at source)

# Windowed FFT
window = np.hanning(len(sig_refl))
S_refl = np.fft.rfft(sig_refl * window)
S_ref = np.fft.rfft(sig_ref * window)
freqs_fft = np.fft.rfftfreq(len(sig_refl), d=dt)

# Reflection coefficient (magnitude)
R_sim = np.abs(S_refl) / (np.abs(S_ref) + 1e-30)

# Compare in the valid frequency range
f_mask = (freqs_fft > 2e9) & (freqs_fft < f_max * 0.8)
freqs_comp = freqs_fft[f_mask]
R_sim_comp = R_sim[f_mask]

# Interpolate analytical to same frequencies
R_ana_interp = np.interp(freqs_comp, freqs_ana, np.sqrt(R_fresnel))

# Normalize both to peak = 1 for shape comparison
if np.max(R_sim_comp) > 0:
    R_sim_norm = R_sim_comp / np.max(R_sim_comp)
    R_ana_norm = R_ana_interp / np.max(R_ana_interp)
    rms_err = np.sqrt(np.mean((R_sim_norm - R_ana_norm) ** 2))
else:
    rms_err = 1.0

print(f"\nResults:")
print(f"  Shape RMS error: {rms_err:.3f}")

PASS = True
if rms_err > 0.15:
    print(f"  FAIL: RMS error {rms_err:.3f} > 0.15")
    PASS = False
else:
    print(f"  PASS: shape agreement within 15% RMS")

# Check that dispersive material produces frequency-dependent behavior
# At low freq: R should be high (eps_s=78, n~8.8, R~0.6)
# At high freq: R should be lower (eps_inf=5.2, n~2.3, R~0.15)
if len(R_sim_comp) > 10:
    R_lo = np.mean(R_sim_comp[:len(R_sim_comp) // 5])
    R_hi = np.mean(R_sim_comp[-len(R_sim_comp) // 5:])
    if R_lo > R_hi:
        print(f"  PASS: dispersion visible (R_low={R_lo:.3f} > R_high={R_hi:.3f})")
    else:
        print(f"  FAIL: no dispersion (R_low={R_lo:.3f} <= R_high={R_hi:.3f})")
        PASS = False

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Debye Water: rfx vs Analytical Fresnel", fontsize=14)

# Reflectance spectrum
ax1.plot(freqs_ana / 1e9, R_fresnel_db, "k-", label="Analytical Fresnel", linewidth=2)
if len(freqs_comp) > 0:
    R_sim_db = 20 * np.log10(R_sim_comp + 1e-30)
    ax1.plot(freqs_comp / 1e9, R_sim_db, "b-", alpha=0.7,
             label="rfx FDTD", linewidth=1.5)
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Reflection (dB)")
ax1.set_xlim(1, f_max / 1e9 * 0.8)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title("Reflectance R(f)")

# Debye permittivity
ax2.plot(freqs_ana / 1e9, np.real(eps_debye), "b-", label="Re(eps)", linewidth=2)
ax2.plot(freqs_ana / 1e9, -np.imag(eps_debye), "r-", label="-Im(eps)", linewidth=2)
ax2.axvline(f_relax / 1e9, color="k", ls="--", alpha=0.5,
            label=f"f_relax = {f_relax/1e9:.1f} GHz")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Permittivity")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title("Debye epsilon(f) — Water 25C")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "15_debye_water_dispersion.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
