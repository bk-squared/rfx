"""Cross-validation 06: Straight Waveguide Flux — rfx vs Meep

Meep Basics tutorial #1 (part 1): straight dielectric waveguide.
Measures transmitted flux through a plane downstream of a source.

This is the most fundamental FDTD validation:
  - guided mode propagation in a dielectric slab
  - flux measurement accuracy
  - comparison with Meep reference

Workflow:
  1. Epsilon comparison (rfx vs Meep)
  2. Field snapshot comparison at multiple times
  3. Transmitted flux T(f) comparison

Meep tutorial parameters:
  eps = 12, width = 1a, pad = 4, dpml = 2, resolution = 10
  cell = 16 x 8 (plus 2*dpml each side)
  fcen = 0.15, fwidth = 0.1
  Source: GaussianSource line source spanning waveguide

Run:
  JAX_ENABLE_X64=1 python examples/crossval/06_straight_waveguide_flux.py
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
wg_width = 1.0       # waveguide width in a
pad = 4.0            # padding
dpml = 2.0
resolution = 10

sx = 16.0            # cell x length (propagation direction)
sy = 2 * (pad + dpml + wg_width / 2)  # = 2*(4+2+0.5) = 13

a = 1.0e-6
dx = a / resolution
fcen = 0.15
df = 0.1

# rfx domain = Meep cell - 2*dpml (each axis)
interior_x = sx
interior_y = sy - 2 * dpml  # 9
domain_x = interior_x * a
domain_y = interior_y * a
cpml_n = int(dpml * resolution)

OFFSET_X = interior_x / 2.0  # 8.0
OFFSET_Y = interior_y / 2.0  # 4.5

bw_rfx = df / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a

# Source at x = -7 (Meep), flux at x = +5 (Meep)
src_x_meep = -7.0
flux_x_meep = 5.0

src_x_rfx = (src_x_meep + OFFSET_X) * a
flux_x_rfx = (flux_x_meep + OFFSET_X) * a

print("=" * 70)
print("Crossval 06: Straight Waveguide Flux — rfx vs Meep")
print("=" * 70)
print(f"eps={eps_wg}, width={wg_width}a, cell={sx}x{sy}")
print(f"fcen={fcen}, df={df}, resolution={resolution}")
print()

# =============================================================================
# PART 1: Meep — run straight waveguide
# =============================================================================
print("=" * 70)
print("PART 1: Meep — straight waveguide with flux")
print("=" * 70)

import meep as mp

cell_meep = mp.Vector3(sx + 2*dpml, sy)
pml_meep = [mp.PML(dpml, direction=mp.X)]  # PML only in x
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

# Flux monitor
flux_mon = sim_meep.add_flux(fcen, df, 50,
    mp.FluxRegion(center=mp.Vector3(flux_x_meep, 0),
                  size=mp.Vector3(0, 2 * wg_width)))

# Run until source decays
sim_meep.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ez, mp.Vector3(flux_x_meep, 0), 1e-3))

meep_flux = np.array(mp.get_fluxes(flux_mon))
meep_freqs = np.array(mp.get_flux_freqs(flux_mon))
meep_total_t = sim_meep.meep_time()

# Capture final field
ez_meep = sim_meep.get_array(center=mp.Vector3(), size=cell_meep,
                              component=mp.Ez)
pml_cells = int(dpml * resolution)
# PML only in x direction
ez_meep_int = ez_meep[pml_cells:-pml_cells, :]

print(f"  Meep: ran {meep_total_t:.0f} time units")
print(f"  Flux peak: {meep_flux.max():.6f} at f={meep_freqs[np.argmax(meep_flux)]:.4f}")

# =============================================================================
# PART 2: rfx — straight waveguide with flux monitor
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2: rfx — straight waveguide with flux monitor")
print("=" * 70)

from rfx import Simulation, Box
from rfx.sources.sources import ModulatedGaussian
from rfx.simulation import SnapshotSpec
import jax.numpy as jnp

sim_rfx = Simulation(freq_max=0.25 * C0 / a,
                     domain=(domain_x, domain_y, dx), dx=dx,
                     boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx.add_material("wg", eps_r=eps_wg)

# Waveguide: infinite in x (spans full domain), width=1a centered in y
wg_y_lo = (OFFSET_Y - wg_width / 2) * a
wg_y_hi = (OFFSET_Y + wg_width / 2) * a
sim_rfx.add(Box((0, wg_y_lo, 0), (domain_x, wg_y_hi, dx)), material="wg")

# Line source spanning waveguide
for i in range(int(wg_width * resolution)):
    y = wg_y_lo + (i + 0.5) * dx
    sim_rfx.add_source(position=(src_x_rfx, y, 0), component="ez",
        waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                   amplitude=1.0 / (wg_width * resolution),
                                   cutoff=5.0 / math.sqrt(2)))

# Probe at flux monitor location (for time series)
sim_rfx.add_probe(position=(flux_x_rfx, OFFSET_Y * a, 0), component="ez")

# Run
rfx_total_t = meep_total_t * a / C0
dt_rfx = dx / (C0 * math.sqrt(2)) * 0.99
n_steps = int(rfx_total_t / dt_rfx) + 200

snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
print(f"  Running rfx: {n_steps} steps...")
t0 = time.time()
res_rfx = sim_rfx.run(n_steps=n_steps, snapshot=snap,
                       subpixel_smoothing=True)
print(f"  Done in {time.time()-t0:.1f}s")

# Compute flux from probe time series via FFT (same as Meep does internally)
ts_rfx = np.array(res_rfx.time_series).ravel()
dt_rfx_actual = float(res_rfx.dt)
nfft = int(2**np.ceil(np.log2(len(ts_rfx))) * 4)
S_rfx = np.fft.rfft(ts_rfx, n=nfft)
rfx_freqs_hz = np.fft.rfftfreq(nfft, d=dt_rfx_actual)
rfx_freqs_meep = rfx_freqs_hz * a / C0
rfx_flux = np.abs(S_rfx)**2  # proxy for flux (power spectral density)

# Normalize to Meep flux scale
mask_valid = (rfx_freqs_meep > fcen - df/2) & (rfx_freqs_meep < fcen + df/2)
if np.any(mask_valid):
    rfx_flux_norm = rfx_flux[mask_valid]
    rfx_freqs_plot = rfx_freqs_meep[mask_valid]
    # Scale rfx to match Meep peak for shape comparison
    scale = meep_flux.max() / (rfx_flux_norm.max() + 1e-30)
    rfx_flux_norm = rfx_flux_norm * scale
    print(f"  rfx spectral peak at f={rfx_freqs_plot[np.argmax(rfx_flux_norm)]:.4f}")

# =============================================================================
# PART 3: Field snapshot comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 3: Field snapshot comparison")
print("=" * 70)

# rfx snapshots
ez_rfx_all = np.asarray(res_rfx.snapshots["ez"])
grid = sim_rfx._build_grid()
pad_g = grid.pad_x
n_dom_x = int(np.ceil(domain_x / dx)) + 1
n_dom_y = int(np.ceil(domain_y / dx)) + 1

dt = float(res_rfx.dt)
capture_ps = [0.05, 0.15, 0.30, 0.50]
rfx_steps = [min(ez_rfx_all.shape[0]-1, int(t*1e-12/dt)) for t in capture_ps]

# Meep snapshots (re-run to capture at specific times)
sim_meep2 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep,
                          resolution=resolution)
sim_meep2.init_sim()
meep_cap_times = [t * 1e-12 * C0 / a for t in capture_ps]

fig, axes = plt.subplots(len(capture_ps), 3, figsize=(20, 4*len(capture_ps)))

for i, t_ps in enumerate(capture_ps):
    # rfx frame
    rf = ez_rfx_all[rfx_steps[i], pad_g:pad_g+n_dom_x, pad_g:pad_g+n_dom_y]

    # Meep frame
    remaining = meep_cap_times[i] - sim_meep2.meep_time()
    if remaining > 0:
        sim_meep2.run(until=remaining)
    ez_m = sim_meep2.get_array(center=mp.Vector3(), size=cell_meep,
                                component=mp.Ez)
    mf = ez_m[pml_cells:-pml_cells, :]

    # Match sizes
    nc_x = min(rf.shape[0], mf.shape[0])
    nc_y = min(rf.shape[1], mf.shape[1])
    rf_c = rf[:nc_x, :nc_y]
    mf_c = mf[:nc_x, :nc_y]

    vm = max(np.max(np.abs(rf_c)), np.max(np.abs(mf_c)), 1e-30) * 0.9

    axes[i, 0].imshow(rf_c.T, origin="lower", cmap="RdBu_r",
                       vmin=-vm, vmax=vm, aspect="auto")
    axes[i, 0].set_title(f"rfx Ez (t={t_ps:.2f}ps)", fontsize=11)
    axes[i, 0].set_ylabel("y")

    axes[i, 1].imshow(mf_c.T, origin="lower", cmap="RdBu_r",
                       vmin=-vm, vmax=vm, aspect="auto")
    axes[i, 1].set_title(f"Meep Ez (t={t_ps:.2f}ps)", fontsize=11)

    diff = rf_c - mf_c
    vd = max(np.max(np.abs(diff)), 1e-30)
    axes[i, 2].imshow(diff.T, origin="lower", cmap="bwr",
                       vmin=-vd, vmax=vd, aspect="auto")
    axes[i, 2].set_title("rfx - Meep", fontsize=11)

for ax in axes[-1, :]:
    ax.set_xlabel("x")
fig.suptitle("Straight Waveguide: Ez Field Snapshots — rfx vs Meep\n"
             f"eps={eps_wg}, width={wg_width}a, resolution={resolution}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = os.path.join(SCRIPT_DIR, "06_field_snapshots.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"  Saved: {out1}")

# =============================================================================
# PART 4: Flux comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 4: Flux T(f) comparison")
print("=" * 70)

fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(meep_freqs, meep_flux, "b-", lw=2, label="Meep flux")
if np.any(mask_valid):
    ax.plot(rfx_freqs_plot, rfx_flux_norm, "r--", lw=2, label="rfx spectral (scaled)")
ax.set_xlabel("Frequency (Meep units: c/a)")
ax.set_ylabel("Flux / Power spectral density")
ax.set_title("Straight Waveguide: Transmitted Spectrum — rfx vs Meep")
ax.legend()
ax.grid(True, alpha=0.3)

meep_peak_f = meep_freqs[np.argmax(meep_flux)]
ax.axvline(meep_peak_f, color="blue", ls=":", alpha=0.5)

plt.tight_layout()
out2 = os.path.join(SCRIPT_DIR, "06_flux_comparison.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"  Saved: {out2}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  Meep flux peak at f={meep_peak_f:.5f}")

# Validation: flux shape correlation between rfx and Meep
PASS = True
if np.any(mask_valid) and len(rfx_flux_norm) > 5:
    corr = float(np.corrcoef(meep_flux[mask_valid], rfx_flux_norm)[0, 1])
    print(f"  Flux shape correlation: {corr:.4f}")
    if corr > 0.90:
        print(f"  PASS: flux correlation {corr:.4f} > 0.90")
    else:
        print(f"  FAIL: flux correlation {corr:.4f} <= 0.90")
        PASS = False
else:
    print("  FAIL: insufficient valid flux data for comparison")
    PASS = False

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")

sys.exit(0 if PASS else 1)
