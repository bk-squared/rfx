"""Cross-validation 01: Waveguide Bend Transmittance (Meep Basics equivalent)

Validates the 90-degree dielectric waveguide bend transmittance against
Meep, following the Meep "Basics" tutorial geometry.

Method: single-run input/output flux normalization.
  - Input flux (x-normal) on the horizontal arm, before the bend corner
  - Output flux (y-normal) on the vertical arm, after the bend corner
  - T(f) = (output / input)_bend / (output / input)_straight
  This eliminates cross-run radiation contamination that afflicts the
  naive two-run approach.

Boundary selection:
  - Default: `upml`
  - Override with `RFX_BOUNDARY=cpml` for the material-aware CPML baseline

Run this script with JAX x64 enabled. The flux monitor accumulators need
double precision for stable SI-unit spectra:

  JAX_ENABLE_X64=1 python examples/crossval/01_meep_waveguide_bend.py

Parameters (normalized, a = 1 um):
  eps=12, w=1, fcen=0.15, fwidth=0.1, resolution=10

UPML uses D/B-equivalent material-independent PML loss (σ/ε₀ instead of
σ/(ε_r·ε₀)) with Meep-matched sigma scaling (n_layers/2 factor).
Subpixel smoothing enabled: per-component anisotropic epsilon at
dielectric boundaries (arithmetic parallel / harmonic perpendicular).
Sigma profile: R_asymptotic=1e-15, quadratic grading.

PASS criteria:
  1. Smoothed mean T in [0.3, 1.0]
  2. Straight self-T in [0.95, 1.05] (flux conservation check)
  3. |rfx − Meep| < 0.10 (single-run method comparison)

Reference: https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/

Save: examples/crossval/01_meep_waveguide_bend.png
"""

import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import uniform_filter1d
from rfx import Simulation, Box, GaussianPulse, flux_spectrum

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Parameters
# =============================================================================
a = 1.0e-6
eps_wg = 12.0
w_wg = 1.0 * a
dx = a / 10
cpml_n = 10
pml = cpml_n * dx

fcen = 0.15 * C0 / a
fwidth = 0.1 * C0 / a
n_freqs = 200
freqs = np.linspace(0.10 * C0 / a, 0.20 * C0 / a, n_freqs)
boundary = os.environ.get("RFX_BOUNDARY", "upml").strip().lower()
if boundary not in {"cpml", "upml"}:
    raise ValueError(
        f"RFX_BOUNDARY must be 'cpml' or 'upml', got {boundary!r}"
    )

sx = 16.0 * a
sy = 16.0 * a
wg_y = sy / 2
wg_x = sx / 2
src_x = pml + dx

f_cutoff = 1.0 / (2.0 * np.sqrt(eps_wg - 1.0))
n_steps = 25000

print("=" * 60)
print("Cross-Validation 01: Waveguide Bend (Meep Basics equivalent)")
print("=" * 60)
print(f"eps={eps_wg}, w={w_wg/a:.0f}a, res=10, {n_steps} steps")
print(f"Domain: {sx/a:.0f}a x {sy/a:.0f}a, boundary={boundary} ({cpml_n} layers)")
print("Method: single-run input/output flux normalization")
print()


def add_line_source(sim, x, y_center, width):
    for i in range(10):
        y = y_center - width / 2 + (i + 0.5) * width / 10
        sim.add_source(position=(x, y, 0), component="ez",
                       waveform=GaussianPulse(f0=fcen, bandwidth=fwidth / fcen,
                                              amplitude=1.0 / 10))


# =============================================================================
# Run 1: Straight waveguide (self-calibration)
# =============================================================================
print("Run 1: Straight waveguide (self-calibration)...", flush=True)
t0 = time.time()
sim_s = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                   boundary=boundary, cpml_layers=cpml_n, mode="2d_tmz")
sim_s.add_material("wg", eps_r=eps_wg)
sim_s.add(Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),
          material="wg")
add_line_source(sim_s, src_x, wg_y, w_wg)
sim_s.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")
sim_s.add_flux_monitor(axis="x", coordinate=sx - pml - 5 * dx,
                       freqs=freqs, name="output")
res_s = sim_s.run(n_steps=n_steps, subpixel_smoothing=True)
flux_in_s = np.array(flux_spectrum(res_s.flux_monitors["input"]))
flux_out_s = np.array(flux_spectrum(res_s.flux_monitors["output"]))
print(f"  {time.time()-t0:.1f}s")

# =============================================================================
# Run 2: 90-degree bend (input + output in same run)
# =============================================================================
print("Run 2: 90-degree bend...", flush=True)
t0 = time.time()
sim_b = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                   boundary=boundary, cpml_layers=cpml_n, mode="2d_tmz")
sim_b.add_material("wg", eps_r=eps_wg)
sim_b.add(Box((0, wg_y - w_wg / 2, 0),
              (wg_x + w_wg / 2, wg_y + w_wg / 2, dx)), material="wg")
sim_b.add(Box((wg_x - w_wg / 2, wg_y - w_wg / 2, 0),
              (wg_x + w_wg / 2, sy, dx)), material="wg")
add_line_source(sim_b, src_x, wg_y, w_wg)
sim_b.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")
sim_b.add_flux_monitor(axis="y", coordinate=sy - pml - 5 * dx,
                       freqs=freqs, name="output")
res_b = sim_b.run(n_steps=n_steps, subpixel_smoothing=True)
flux_in_b = np.array(flux_spectrum(res_b.flux_monitors["input"]))
flux_out_b = np.array(flux_spectrum(res_b.flux_monitors["output"]))
print(f"  {time.time()-t0:.1f}s")

# =============================================================================
# Transmittance
# =============================================================================
f_meep = freqs * a / C0
above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)

# Straight self-T (should be ~1)
safe_in_s = np.maximum(np.abs(flux_in_s), np.max(np.abs(flux_in_s)) * 1e-6)
T_self = flux_out_s / safe_in_s
T_self_smooth = uniform_filter1d(T_self, size=20)

# Bend out/in
safe_in_b = np.maximum(np.abs(flux_in_b), np.max(np.abs(flux_in_b)) * 1e-6)
T_bend_abs = flux_out_b / safe_in_b

# Normalized: (out/in)_bend / (out/in)_straight
T_norm = T_bend_abs / np.maximum(np.abs(T_self), 1e-30)
T_norm_smooth = uniform_filter1d(T_norm, size=20)

mean_self = float(np.mean(T_self_smooth[above]))
mean_T = float(np.mean(T_norm_smooth[above]))

print(f"\nStraight self-T (flux conservation): {mean_self:.4f}")
print(f"Bend T (normalized):    {mean_T:.4f}  "
      f"[{np.min(T_norm_smooth[above]):.4f}, {np.max(T_norm_smooth[above]):.4f}]")

# =============================================================================
# Meep reference (single-run method)
# =============================================================================
meep_mean = None
try:
    import meep as mp
    print("\nRunning Meep reference (single-run)...", flush=True)
    cell = mp.Vector3(sx / a + 2, sy / a + 2)
    pml_m = [mp.PML(1.0)]
    geo_m = [
        mp.Block(size=mp.Vector3(sx / (2 * a) + 0.5, 1),
                 center=mp.Vector3(-sx / (4 * a) + 0.25, 0),
                 material=mp.Medium(epsilon=12)),
        mp.Block(size=mp.Vector3(1, sy / (2 * a) + 0.5),
                 center=mp.Vector3(0, sy / (4 * a) - 0.25),
                 material=mp.Medium(epsilon=12)),
    ]
    src_m = [mp.Source(mp.GaussianSource(0.15, fwidth=0.1), component=mp.Ez,
                       center=mp.Vector3(-sx / (2 * a) + 0.1, 0),
                       size=mp.Vector3(0, 1))]
    sim_m = mp.Simulation(cell_size=cell, boundary_layers=pml_m,
                          geometry=geo_m, sources=src_m, resolution=10)
    fi_m = sim_m.add_flux(0.15, 0.1, 200,
                          mp.FluxRegion(center=mp.Vector3(-4, 0),
                                        size=mp.Vector3(0, sy / a)))
    fo_m = sim_m.add_flux(0.15, 0.1, 200,
                          mp.FluxRegion(center=mp.Vector3(0, sy / (2 * a) - 1.5),
                                        size=mp.Vector3(sx / a, 0)))
    sim_m.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(0, sy / (2 * a) - 1.5), 1e-3))
    T_meep = np.array(mp.get_fluxes(fo_m)) / np.maximum(
        np.abs(np.array(mp.get_fluxes(fi_m))), 1e-30)
    f_ref = np.array(mp.get_flux_freqs(fi_m))
    above_r = (f_ref > f_cutoff + 0.005) & (f_ref < 0.20)
    meep_mean = float(np.mean(uniform_filter1d(T_meep, size=20)[above_r]))
    print(f"  Meep T = {meep_mean:.4f}")
except ImportError:
    print("\nMeep not available — skipping reference comparison")

# =============================================================================
# Validation
# =============================================================================
PASS = True

if 0.3 <= mean_T <= 1.0:
    print(f"\nPASS: smoothed T = {mean_T:.4f} in [0.3, 1.0]")
else:
    print(f"\nFAIL: smoothed T = {mean_T:.4f} outside [0.3, 1.0]")
    PASS = False

if 0.95 <= mean_self <= 1.05:
    print(f"PASS: straight self-T = {mean_self:.4f} in [0.95, 1.05]")
else:
    print(f"FAIL: straight self-T = {mean_self:.4f} outside [0.95, 1.05]")
    PASS = False

if meep_mean is not None:
    gap = abs(mean_T - meep_mean)
    if gap < 0.10:
        print(f"PASS: |rfx - Meep| = {gap:.4f} < 0.10")
    else:
        print(f"FAIL: |rfx - Meep| = {gap:.4f} >= 0.10")
        PASS = False

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Waveguide Bend Transmittance (Meep Basics equivalent)\n"
             f"eps={eps_wg}, w=1a, res=10, boundary={boundary}",
             fontsize=13)

# Panel 1: Straight self-T
ax = axes[0]
plot_mask = (f_meep > 0.10) & (f_meep < 0.20)
ax.plot(f_meep[plot_mask], T_self[plot_mask], "b-", lw=0.5, alpha=0.3)
ax.plot(f_meep[plot_mask], T_self_smooth[plot_mask], "b-", lw=2,
        label=f"self-T (mean={mean_self:.3f})")
ax.axhline(1.0, color="k", ls="--", alpha=0.3)
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("T_self = out/in")
ax.set_ylim(0.5, 1.5)
ax.legend(fontsize=9)
ax.set_title("Straight: flux conservation")
ax.grid(True, alpha=0.3)

# Panel 2: Bend T
ax = axes[1]
ax.plot(f_meep[plot_mask], T_norm[plot_mask], "b-", lw=0.5, alpha=0.3,
        label="rfx raw")
ax.plot(f_meep[plot_mask], T_norm_smooth[plot_mask], "b-", lw=2,
        label=f"rfx (mean={mean_T:.2f})")
if meep_mean is not None:
    T_meep_smooth = uniform_filter1d(T_meep, size=20)
    m = (f_ref > 0.10) & (f_ref < 0.20)
    ax.plot(f_ref[m], T_meep[m], "r-", lw=0.5, alpha=0.3, label="Meep raw")
    ax.plot(f_ref[m], T_meep_smooth[m], "r-", lw=2,
            label=f"Meep (mean={meep_mean:.2f})")
ax.axhline(1.0, color="k", ls="--", alpha=0.3)
ax.axvline(f_cutoff, color="gray", ls=":", alpha=0.5,
           label=f"cutoff={f_cutoff:.3f}")
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("T(f)")
ax.set_ylim(-0.5, 2.0)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Bend transmittance")
ax.grid(True, alpha=0.3)

# Panel 3: Smoothed comparison
ax = axes[2]
ax.plot(f_meep[plot_mask], T_norm_smooth[plot_mask], "b-", lw=2,
        label=f"rfx ({mean_T:.2f})")
if meep_mean is not None:
    ax.plot(f_ref[m], T_meep_smooth[m], "r-", lw=2,
            label=f"Meep ({meep_mean:.2f})")
ax.axhline(1.0, color="k", ls="--", alpha=0.3)
ax.axvline(f_cutoff, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("T(f) smoothed")
ax.set_ylim(0, 1.5)
ax.legend(fontsize=10)
ax.set_title("Smoothed comparison")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "01_meep_waveguide_bend.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
