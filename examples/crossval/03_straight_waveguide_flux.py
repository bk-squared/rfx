"""Cross-validation 03: Straight Waveguide Transmission — rfx vs Meep

Meep Basics tutorial #1 (part 1): straight dielectric waveguide.
Measures transmission T(f) = flux_out(f) / flux_in(f) for a lossless
guided mode; both rfx and Meep should give T ≈ 1 at the carrier
frequency (energy conservation), and the two should agree within
a few percent.

**Physical validity:**
  - Guided-mode propagation in a dielectric slab (eps_r = 12)
  - Transmission measurement using rfx's ``add_flux_monitor`` on both
    the input and output planes
  - Two-run reference subtraction pattern (T = flux_out / flux_in)

**Rule compliance** (.claude/rules/rfx-feature-discovery.md):
This script uses ``add_flux_monitor`` and ``flux_spectrum`` — the
canonical rfx primitives for R(f) / T(f) measurement. An earlier
version of this script computed a single-point FFT of a time-series
probe as a "flux proxy" and then scaled the result to match the
Meep peak; that pattern is explicitly forbidden (shape correlation
of scaled Gaussians always passes) and has been removed.

Meep tutorial parameters:
  eps = 12, width = 1a, pad = 4, dpml = 2, resolution = 10
  cell = 16 x 8 (plus 2*dpml each side)
  fcen = 0.15, fwidth = 0.1
  Source: GaussianSource line source spanning waveguide

Pass criteria (each must hold):
  - rfx self-transmission T_rfx(f_peak) ∈ [0.95, 1.05]
  - Meep self-transmission T_meep(f_peak) ∈ [0.95, 1.05]
  - |T_rfx(f_peak) − T_meep(f_peak)| < 0.05

Run:
  JAX_ENABLE_X64=1 python examples/crossval/03_straight_waveguide_flux.py
"""

import os, sys, math, time
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Meep tutorial parameters
# =============================================================================
eps_wg = 12.0
wg_width = 1.0        # waveguide width in a
pad = 4.0
dpml = 2.0
resolution = 10

sx = 16.0
sy = 2 * (pad + dpml + wg_width / 2)   # = 13

a = 1.0e-6
dx = a / resolution
fcen = 0.15
df = 0.1
n_freqs = 50

# rfx domain = Meep cell - 2*dpml (each axis); CPML handled separately
interior_x = sx
interior_y = sy - 2 * dpml       # 9
domain_x = interior_x * a
domain_y = interior_y * a
cpml_n = int(dpml * resolution)

OFFSET_X = interior_x / 2.0      # rfx maps x_meep → x_meep + OFFSET_X
OFFSET_Y = interior_y / 2.0

bw_rfx = df / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a

# Monitor positions in Meep coordinates (x=0 at cell center)
src_x_meep   = -7.0
flux_in_meep = -5.0         # downstream of source, well inside interior
flux_out_meep = +5.0

src_x_rfx     = (src_x_meep + OFFSET_X) * a
flux_in_rfx   = (flux_in_meep + OFFSET_X) * a
flux_out_rfx  = (flux_out_meep + OFFSET_X) * a

print("=" * 70)
print("Crossval 03: Straight Waveguide Transmission — rfx vs Meep")
print("=" * 70)
print(f"eps={eps_wg}, width={wg_width}a, cell={sx}x{sy}")
print(f"fcen={fcen}, df={df}, resolution={resolution}")
print(f"src @ x={src_x_meep}, flux_in @ x={flux_in_meep}, flux_out @ x={flux_out_meep}")
print()

# =============================================================================
# PART 1: Meep — transmission between two flux monitors
# =============================================================================
print("=" * 70)
print("PART 1: Meep — two flux monitors, T = flux_out / flux_in")
print("=" * 70)

import meep as mp

cell_meep = mp.Vector3(sx + 2 * dpml, sy)
pml_meep = [mp.PML(dpml, direction=mp.X)]
geo_meep = [mp.Block(size=mp.Vector3(mp.inf, wg_width, mp.inf),
                     center=mp.Vector3(),
                     material=mp.Medium(epsilon=eps_wg))]
src_meep = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                      component=mp.Ez,
                      center=mp.Vector3(src_x_meep, 0),
                      size=mp.Vector3(0, wg_width))]

sim_meep = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                         geometry=geo_meep, sources=src_meep,
                         resolution=resolution)

flux_in_mon_m = sim_meep.add_flux(fcen, df, n_freqs,
    mp.FluxRegion(center=mp.Vector3(flux_in_meep, 0),
                  size=mp.Vector3(0, 2 * wg_width)))
flux_out_mon_m = sim_meep.add_flux(fcen, df, n_freqs,
    mp.FluxRegion(center=mp.Vector3(flux_out_meep, 0),
                  size=mp.Vector3(0, 2 * wg_width)))

sim_meep.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ez, mp.Vector3(flux_out_meep, 0), 1e-3))

meep_flux_in  = np.array(mp.get_fluxes(flux_in_mon_m))
meep_flux_out = np.array(mp.get_fluxes(flux_out_mon_m))
meep_freqs = np.array(mp.get_flux_freqs(flux_in_mon_m))
meep_total_t = sim_meep.meep_time()

# Guard against numerical flux-in ≈ 0 near band edges
eps_flux = float(np.max(np.abs(meep_flux_in))) * 1e-6
T_meep = meep_flux_out / np.where(np.abs(meep_flux_in) > eps_flux,
                                   meep_flux_in, eps_flux)

peak_idx = int(np.argmax(np.abs(meep_flux_in)))
f_peak_meep = float(meep_freqs[peak_idx])
T_meep_peak = float(T_meep[peak_idx])
print(f"  Meep: ran {meep_total_t:.0f} time units")
print(f"  peak frequency (by flux_in magnitude): {f_peak_meep:.4f}")
print(f"  T_meep(f_peak) = {T_meep_peak:.4f}")

# =============================================================================
# PART 2: rfx — same two-monitor measurement
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2: rfx — two flux monitors (canonical add_flux_monitor)")
print("=" * 70)

from rfx import Simulation, Box, flux_spectrum
from rfx.boundaries.spec import BoundarySpec
from rfx.sources.sources import ModulatedGaussian
import jax.numpy as jnp

sim_rfx = Simulation(freq_max=0.25 * C0 / a,
                     domain=(domain_x, domain_y, dx), dx=dx,
                     boundary=BoundarySpec.uniform("upml"),
                     cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx.add_material("wg", eps_r=eps_wg)

wg_y_lo = (OFFSET_Y - wg_width / 2) * a
wg_y_hi = (OFFSET_Y + wg_width / 2) * a
sim_rfx.add(Box((0, wg_y_lo, 0), (domain_x, wg_y_hi, dx)), material="wg")

for i in range(int(wg_width * resolution)):
    y = wg_y_lo + (i + 0.5) * dx
    sim_rfx.add_source(position=(src_x_rfx, y, 0), component="ez",
        waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                   amplitude=1.0 / (wg_width * resolution),
                                   cutoff=5.0 / math.sqrt(2)))

# Sample the rfx flux on the SAME Meep-normalised frequency grid so the
# peak-frequency comparison is bin-aligned (no interpolation ambiguity).
freqs_rfx = jnp.asarray(meep_freqs * C0 / a)
sim_rfx.add_flux_monitor(axis="x", coordinate=flux_in_rfx,
                          freqs=freqs_rfx, name="flux_in")
sim_rfx.add_flux_monitor(axis="x", coordinate=flux_out_rfx,
                          freqs=freqs_rfx, name="flux_out")

sim_rfx.preflight(strict=False)

rfx_total_t = meep_total_t * a / C0
dt_rfx = dx / (C0 * math.sqrt(2)) * 0.99
n_steps = int(rfx_total_t / dt_rfx) + 200

print(f"  Running rfx: {n_steps} steps...")
t0 = time.time()
res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=True)
print(f"  Done in {time.time()-t0:.1f}s")

flux_in_rfx_arr  = np.asarray(flux_spectrum(res_rfx.flux_monitors["flux_in"]))
flux_out_rfx_arr = np.asarray(flux_spectrum(res_rfx.flux_monitors["flux_out"]))

eps_flux_r = float(np.max(np.abs(flux_in_rfx_arr))) * 1e-6
T_rfx = flux_out_rfx_arr / np.where(np.abs(flux_in_rfx_arr) > eps_flux_r,
                                     flux_in_rfx_arr, eps_flux_r)
T_rfx_peak = float(T_rfx[peak_idx])
print(f"  T_rfx(f_peak) = {T_rfx_peak:.4f}")

# =============================================================================
# PART 3: Flux comparison plots
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 3: Flux and transmission plots")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

ax = axes[0]
ax.plot(meep_freqs, meep_flux_in,  "b-",  lw=2, label="Meep flux_in")
ax.plot(meep_freqs, meep_flux_out, "b--", lw=2, label="Meep flux_out")
ax.plot(meep_freqs, flux_in_rfx_arr,  "r-",  lw=1.3, label="rfx flux_in")
ax.plot(meep_freqs, flux_out_rfx_arr, "r--", lw=1.3, label="rfx flux_out")
ax.axvline(f_peak_meep, color="k", ls=":", alpha=0.5, label=f"f_peak={f_peak_meep:.3f}")
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("Flux")
ax.set_title("Absolute flux spectra")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(meep_freqs, T_meep, "b-", lw=2, label=f"Meep  T(f_peak)={T_meep_peak:.3f}")
ax.plot(meep_freqs, T_rfx,  "r--", lw=1.5, label=f"rfx   T(f_peak)={T_rfx_peak:.3f}")
ax.axhline(1.0, color="k", ls=":", alpha=0.5)
ax.axvline(f_peak_meep, color="k", ls=":", alpha=0.5)
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("Transmission T(f) = flux_out / flux_in")
ax.set_title("Transmission")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0.0, 1.5)

plt.suptitle("Crossval 03 — Straight Waveguide: rfx vs Meep", fontweight="bold")
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "03_flux_comparison.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

# =============================================================================
# PASS / FAIL
# =============================================================================
print(f"\n{'=' * 70}")
print("VERDICT")
print("=" * 70)

def _in_range(x, lo, hi): return lo <= x <= hi

tol_self  = 0.05        # each sim's own T(f_peak) must be within [0.95, 1.05]
tol_cross = 0.05        # |T_rfx − T_meep| at peak must be < 0.05

pass_self_rfx  = _in_range(T_rfx_peak,  1.0 - tol_self, 1.0 + tol_self)
pass_self_meep = _in_range(T_meep_peak, 1.0 - tol_self, 1.0 + tol_self)
delta_cross    = abs(T_rfx_peak - T_meep_peak)
pass_cross     = delta_cross < tol_cross

print(f"  rfx self-T  @ f_peak: {T_rfx_peak:.4f}   {'PASS' if pass_self_rfx  else 'FAIL'} (gate 1.0 ± {tol_self})")
print(f"  meep self-T @ f_peak: {T_meep_peak:.4f}   {'PASS' if pass_self_meep else 'FAIL'} (gate 1.0 ± {tol_self})")
print(f"  |T_rfx − T_meep| @ f_peak: {delta_cross:.4f}  {'PASS' if pass_cross     else 'FAIL'} (gate < {tol_cross})")

PASS = pass_self_rfx and pass_self_meep and pass_cross
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
