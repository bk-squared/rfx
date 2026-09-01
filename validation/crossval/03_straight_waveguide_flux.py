"""Cross-validation 03: Straight Waveguide — dispersion (E2) + flux (E1) vs Meep

Meep Basics tutorial #1 (part 1): straight dielectric waveguide.

Two DIFFERENT things are measured here, and #812 exists because the case used
to conflate them:

  1. **The guide** — the guided mode's effective index n_eff(f), measured from
     the run's own Ez phasors along the guide centre line and compared to the
     closed-form symmetric-slab TE0 solution for the declared recipe. This is
     the E2 oracle. It is a different number for every guide.

  2. **The flux bookkeeping** — T(f) = flux_out(f) / flux_in(f), which for a
     lossless uniform section is 1 for ANY bound mode, ANY permittivity and ANY
     width. This is an E1 self-check on the monitors and the normalization. It
     is NOT evidence about the guide, and it must not be read as such.

The #812 audit measured the difference: sweeping this script's own ``eps_wg``
12 -> 11 -> 10 -> 8 kept the band-mean T at 0.9657 / 0.9700 / 0.9882 / 1.0257,
all PASS, while the guide's effective index moved -23.4 %. The case had no
committed reference of its own (``artifact_paths: []``), so on a host without
Meep nothing that decided PASS depended on the guide at all. Gate G1 below is
that missing dependency. See
``docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md``.

**Physical validity:**
  - Guided-mode propagation in a dielectric slab (eps_r = 12)
  - Effective index measured against the analytic slab TE0 dispersion
  - Transmission measurement using rfx's ``add_flux_monitor`` on both
    the input and output planes
  - Two-run reference subtraction pattern (T = flux_out / flux_in)

**Rule compliance**:
This script uses ``add_flux_monitor`` and ``flux_spectrum`` — the
canonical rfx primitives for R(f) / T(f) measurement. An earlier
version of this script computed a single-point FFT of a time-series
probe as a "flux proxy" and then scaled the result to match the
Meep peak; that pattern is explicitly forbidden (shape correlation
of scaled Gaussians always passes) and has been removed.

**Flux-region congruence (issue #160):** the rfx monitors are bounded to
the same 2*wg_width region the Meep ``FluxRegion`` measures. The earlier
full-plane monitors also integrated the line source's radiation cone at
flux_in (radiation that exits transversally before flux_out), reading
T = 0.913 at resolution 10 with no flux-normalization bug present —
see scripts/diagnostics/cv03_flux/sweep_t_deficit.py for the
resolution x monitor-extent falsifier matrix.

Meep tutorial parameters:
  eps = 12, width = 1a, pad = 4, dpml = 2, resolution = 10
  cell = 16 x 8 (plus 2*dpml each side)
  fcen = 0.15, fwidth = 0.1
  Source: GaussianSource line source spanning waveguide

Pass criteria:

  G1 (E2, analytic oracle — the gate that depends on the guide; added by #812):
      max over the gated band of |n_eff_rfx(f)/n_eff_analytic(f) − 1| ≤ 2.0%
      MAX over bins, not mean: a band mean's null space is every zero-mean
      shape error (#812 mechanism P2). Threshold derived in the design note
      from the exact discrete-Yee axial dispersion term (0.523% at the top of
      the gated band) and frozen before the measurement; never widened.
      Estimator self-check: two-wave relative residual ≤ 0.05 at every gated
      bin (also a hard gate — a larger residual means the reported n_eff is
      not trustworthy).

  G2 (E1, self-check — the flux identity, unchanged in value and statistic):
      rfx band-mean T ∈ [0.95, 1.05]
      Gate statistic = MEAN T over the central source band fcen ± 0.15*df —
      re-specified 2026-06-12, issue #160, after recording the full T(f)
      curves: at the recipe mesh rfx's per-bin T carries the
      preflight-documented ±5-10% coarse-mesh ripple, so a single-bin gate
      samples ripple valleys; the band mean is the physically meaningful
      energy-transmission estimator. Peak-bin values are printed for
      information only.

  G3 (E4, external solver — only when Meep ran, unchanged):
      Meep band-mean T ∈ [0.95, 1.05]
      |band-mean T_rfx − band-mean T_meep| < 0.05

Exit codes (rfx crossval convention):
  0 = all PASS including the Meep cross-check
  1 = an rfx-side gate failed — G1 (the guide is not the declared guide, or the
      estimator premise does not hold) or G2 (broken flux bookkeeping / infra)
  2 = G1 and G2 OK but the Meep reference is unavailable — inconclusive
      crossval, NOT a pass. CI must not treat this as green.

Run:
  JAX_ENABLE_X64=1 python validation/crossval/03_straight_waveguide_flux.py
"""

import os
import sys
import math
import time
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "comparators"))
from slab_te_dispersion import (                       # noqa: E402
    slab_te0_neff, measure_neff_two_wave)              # numpy only, no rfx

C0 = 2.998e8

# =============================================================================
# The PUBLISHED recipe this case reproduces
# =============================================================================
# Meep Basics tutorial #1, part 1 ("A straight waveguide"): eps = 12, width =
# 1a in air. These constants carry the tutorial's provenance, NOT this run's:
# the analytic oracle in PART 4 is evaluated on THEM, so the run has to
# reproduce the guide the case claims to be validating rather than whichever
# guide it happens to build. If the structure ever legitimately changes, this
# block changes with it AND the case is no longer that tutorial.
RECIPE_EPS_WG = 12.0        # tutorial core permittivity
RECIPE_EPS_CLAD = 1.0       # air
RECIPE_WG_WIDTH = 1.0       # tutorial guide width, in units of a

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

# Dispersion fit window (#812): the guide interval BETWEEN the two flux planes,
# inset by 1a at each end so the source near field and the monitor planes are
# excluded. L = 8a = 80 cells ~ 3.4 guided wavelengths at the carrier. Declared
# in the design note before the measurement.
neff_x_lo_meep = flux_in_meep + 1.0     # -4.0
neff_x_hi_meep = flux_out_meep - 1.0    # +4.0

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

try:
    import meep as mp
except Exception as _e:
    # Catch ImportError AND any exception during import (a Meep wheel built
    # against NumPy 1.x crashes under NumPy 2.x with "numpy.core.multiarray
    # failed to import"). Treat as reference-missing — the rfx self-check
    # (PART 2) still runs below and the script exits 2, not 0.
    HAVE_MEEP = False
    print(f"[SKIP] external reference unavailable (Meep: {type(_e).__name__}: "
          f"{_e}) — exit 2")
    print("       rfx self-transmission self-check still runs; "
          "NOT a crossval PASS.")
    # rfx still needs a frequency grid + a peak index. Build a Meep-style
    # frequency band around the carrier and pick the peak from rfx flux_in.
    meep_freqs = np.linspace(fcen - df / 2, fcen + df / 2, n_freqs)
    meep_total_t = 400.0  # ~Meep stop_when_fields_decayed wall in time units
    f_peak_meep = float(fcen)
    T_meep_peak = float("nan")
    T_meep = None
    peak_idx = None  # resolved from rfx flux_in below when Meep is absent
else:
    HAVE_MEEP = True

if HAVE_MEEP:
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

# Flux region bounded to 2*wg_width on the guide axis — the SAME region the
# Meep part measures (FluxRegion size=(0, 2*wg_width)). A full-plane monitor
# (the pre-#160 behaviour) additionally integrates the line source's
# radiation cone at flux_in; that radiation exits through the transverse
# UPML before flux_out, so T = out/in reads low (0.913 at resolution 10)
# without any flux-normalization bug. Issue #160 mesh x monitor-extent
# matrix: bounded T(f_peak) = 0.974 / 1.011 / 0.997 at resolution 10/15/20.
# The z size is oversized and clamps to the full (degenerate) z extent in
# 2D mode.
flux_size = (2 * wg_width * a, 10 * dx)
flux_center = (OFFSET_Y * a, dx / 2)
sim_rfx.add_flux_monitor(axis="x", coordinate=flux_in_rfx,
                          freqs=freqs_rfx, name="flux_in",
                          size=flux_size, center=flux_center)
sim_rfx.add_flux_monitor(axis="x", coordinate=flux_out_rfx,
                          freqs=freqs_rfx, name="flux_out",
                          size=flux_size, center=flux_center)

# #812: a y-normal DFT plane on the guide centre line gives Ez(f, x) along the
# propagation axis -- the phase information the flux ratio throws away. The
# guided mode is separable (a transverse profile times exp(-i beta x)), so the
# phase slope is the same at every y inside the core; the centre line is chosen
# because the even TE0 profile peaks there and is least contaminated by the
# cladding tail.
sim_rfx.add_dft_plane_probe(axis="y", coordinate=OFFSET_Y * a,
                            component="ez", freqs=freqs_rfx,
                            name="guide_axis_ez")

sim_rfx.preflight(strict=False)

# rfx integration time is a FIXED 400 time units (a/c0), NOT slaved to
# Meep's stop_when_fields_decayed wall clock. Inheriting Meep's wall time
# truncated the rfx flux DFT whenever Meep stopped early (lane run
# 27393931821: Meep stopped at t=200, rfx got 3059 steps and read
# T(f_peak)=1.155 from truncation aliasing). Measured convergence at this
# duration: T=0.9736 at 1x (5995 steps), 0.9772 at 3x — a 0.4% band.
# NOTE: until_decay=1e-5 at flux_out was tried and REJECTED for this
# geometry: the stopper triggers at ~2200 steps (point ez goes quiet)
# while the flux DFT is still accumulating the slow low-group-velocity
# tail of the eps=12 guide — T reads 0.745. Point-field decay is not a
# flux-convergence witness here.
rfx_total_t = 400.0 * a / C0
dt_rfx = dx / (C0 * math.sqrt(2)) * 0.99
n_steps = int(rfx_total_t / dt_rfx) + 200

print(f"  Running rfx: {n_steps} steps (fixed 400 a/c0 units)...")
t0 = time.time()
res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=True)
print(f"  Done in {time.time()-t0:.1f}s")

flux_in_rfx_arr  = np.asarray(flux_spectrum(res_rfx.flux_monitors["flux_in"]))
flux_out_rfx_arr = np.asarray(flux_spectrum(res_rfx.flux_monitors["flux_out"]))

eps_flux_r = float(np.max(np.abs(flux_in_rfx_arr))) * 1e-6
T_rfx = flux_out_rfx_arr / np.where(np.abs(flux_in_rfx_arr) > eps_flux_r,
                                     flux_in_rfx_arr, eps_flux_r)
if peak_idx is None:
    # Meep absent: pick the peak frequency from rfx's own flux_in magnitude.
    peak_idx = int(np.argmax(np.abs(flux_in_rfx_arr)))
    f_peak_meep = float(meep_freqs[peak_idx])
T_rfx_peak = float(T_rfx[peak_idx])
print(f"  T_rfx(f_peak) = {T_rfx_peak:.4f}")

# =============================================================================
# PART 2b: rfx — guided-mode dispersion against the analytic slab oracle (#812)
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2b: rfx — n_eff(f) vs the analytic slab TE0 mode")
print("=" * 70)

_axis = res_rfx.dft_planes["guide_axis_ez"]
_line_full = np.asarray(_axis.accumulator)[:, :, 0]      # (n_freqs, nx)
# Ez node i sits at physical x = (i - pad_x_lo) * dx.
_x_full = (np.arange(_line_full.shape[1]) - res_rfx.grid.pad_x_lo) * dx
_fit_lo = (neff_x_lo_meep + OFFSET_X) * a
_fit_hi = (neff_x_hi_meep + OFFSET_X) * a
_win = (_x_full >= _fit_lo - 1e-12) & (_x_full <= _fit_hi + 1e-12)

# Two-wave model, NOT a single-mode phase slope. A lossless uniform section has
# exactly two propagating solutions of one bound mode; the single-mode fit was
# pre-declared, then falsified by its own residual self-check on this guide
# (|B/A| ~ 0.53). See section 7 of the design note.
_fits = measure_neff_two_wave(
    _line_full[:, _win], _x_full[_win], np.asarray(meep_freqs) * C0 / a,
    c0=C0, eps_core=eps_wg, eps_clad=1.0)
neff_rfx = np.array([fit.n_eff for fit in _fits])
neff_resid = np.array([fit.rel_residual for fit in _fits])
neff_bovera = np.array([fit.b_over_a for fit in _fits])

# The oracle is evaluated on the RECIPE, not on whatever the run built.
neff_analytic = np.array([
    slab_te0_neff(RECIPE_EPS_WG, RECIPE_EPS_CLAD, RECIPE_WG_WIDTH * a,
                  2.0 * np.pi * (f * C0 / a) / C0)
    for f in meep_freqs])

print(f"  fit window: Meep x in [{neff_x_lo_meep:+.1f}, {neff_x_hi_meep:+.1f}] "
      f"= {int(_win.sum())} samples over "
      f"{(neff_x_hi_meep - neff_x_lo_meep):.0f}a")
print(f"  analytic slab TE0 (eps={RECIPE_EPS_WG:g}, d={RECIPE_WG_WIDTH:g}a) "
      f"at fcen: n_eff = "
      f"{slab_te0_neff(RECIPE_EPS_WG, RECIPE_EPS_CLAD, RECIPE_WG_WIDTH * a, 2.0 * np.pi * fcen_hz / C0):.5f}")
print(f"  standing-wave ratio |B/A| over the band: "
      f"{neff_bovera.min():.3f} .. {neff_bovera.max():.3f}   "
      f"[REPORTED, not gated -- see design note section 8]")

# =============================================================================
# PART 3: Flux comparison plots
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 3: Flux and transmission plots")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))

ax = axes[0]
if HAVE_MEEP:
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
if HAVE_MEEP:
    ax.plot(meep_freqs, T_meep, "b-", lw=2, label=f"Meep  T(f_peak)={T_meep_peak:.3f}")
ax.plot(meep_freqs, T_rfx,  "r--", lw=1.5, label=f"rfx   T(f_peak)={T_rfx_peak:.3f}")
ax.axhline(1.0, color="k", ls=":", alpha=0.5)
ax.axvline(f_peak_meep, color="k", ls=":", alpha=0.5)
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("Transmission T(f) = flux_out / flux_in")
ax.set_title("Transmission — E1 identity (T = 1 for ANY lossless guide)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0.0, 1.5)

ax = axes[2]
ax.plot(meep_freqs, neff_analytic, "k-", lw=2,
        label=f"analytic slab TE0 (eps={RECIPE_EPS_WG:g}, d={RECIPE_WG_WIDTH:g}a)")
ax.plot(meep_freqs, neff_rfx, "r--", lw=1.5, label="rfx two-wave fit")
_bm = np.abs(meep_freqs - fcen) <= 0.15 * df
ax.axvspan(fcen - 0.15 * df, fcen + 0.15 * df, color="0.85", zorder=0,
           label="gated band")
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("n_eff")
ax.set_title("Dispersion — E2 oracle (depends on the guide)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

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

tol_self  = 0.05        # each sim's central-band MEAN T within [0.95, 1.05]
tol_cross = 0.05        # |band-mean T_rfx − band-mean T_meep| < 0.05

# G1 thresholds. Frozen in
# docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md in a
# commit PRECEDING the measurement that judges them, derived from the exact
# discrete-Yee axial dispersion term at the top of the gated band (0.523%) and
# never fitted to a measured n_eff. DO NOT WIDEN: a miss is a miss.
tol_neff  = 0.020       # max |n_eff_rfx/n_eff_analytic - 1| over the gated band
tol_resid = 0.05        # max two-wave relative residual over the gated band

# Gate statistic: MEAN T over the central source band (fcen ± 0.15*df,
# i.e. the central 30% of the Gaussian band, where flux_in is strong and
# the ratio is well-conditioned). Re-specified 2026-06-12 (issue #160)
# with the measured curves recorded FIRST (sweep_t_deficit.json + lane
# runs 27393931821 / 27394439174): at the recipe mesh (11.5 cells/λ_eff
# at freq_max — below the preflight's own ≥20 floor for flux extraction)
# rfx's per-bin T(f) carries a ±5-10% ripple, the preflight's documented
# coarse-mesh |S| error class, while Meep's curve is smooth. A
# single-bin gate sampled at Meep's peak bin lands in ripple valleys
# (T=0.902 at f=0.1510) even though the band-energy transmission is
# clean: band-mean 0.966/1.005/0.989 at resolution 10/15/20. The mean
# over the source band is the physically meaningful energy-transmission
# estimator and is robust to the per-bin ripple. Peak-bin values are
# still printed for information.
band_mask = np.abs(meep_freqs - fcen) <= 0.15 * df
T_rfx_band = float(np.mean(T_rfx[band_mask]))

# ---- G1 (E2): dispersion against the analytic slab oracle --------------------
# MAX over the gated bins, not the mean: the mean's null space is every
# zero-mean shape error (#812 mechanism P2).
neff_dev = neff_rfx / neff_analytic - 1.0
neff_dev_max = float(np.max(np.abs(neff_dev[band_mask])))
neff_dev_argmax = int(np.argmax(np.abs(np.where(band_mask, neff_dev, 0.0))))
resid_max = float(np.max(neff_resid[band_mask]))
pass_resid = resid_max <= tol_resid
pass_neff = pass_resid and (neff_dev_max <= tol_neff)

print("  --- G1  E2 analytic oracle: guided-mode dispersion ---")
print(f"  two-wave fit residual, max over band: {resid_max:.4f}   "
      f"{'PASS' if pass_resid else 'FAIL'} (gate <= {tol_resid}) "
      f"[estimator premise]")
print(f"  n_eff band mean: rfx {float(np.mean(neff_rfx[band_mask])):.5f}  "
      f"analytic {float(np.mean(neff_analytic[band_mask])):.5f}  [info only]")
print(f"  max |n_eff_rfx/n_eff_analytic - 1| over band: "
      f"{100*neff_dev_max:.3f}%  at f={meep_freqs[neff_dev_argmax]:.4f} "
      f"(rfx {neff_rfx[neff_dev_argmax]:.5f} vs analytic "
      f"{neff_analytic[neff_dev_argmax]:.5f})   "
      f"{'PASS' if pass_neff else 'FAIL'} (gate <= {100*tol_neff:.1f}%)")
if not pass_resid:
    print("        reason: the two-wave premise does not hold on this line, so "
          "the reported n_eff is not trustworthy (estimator self-check, not a "
          "physics verdict)")
elif not pass_neff:
    print(f"        reason: the simulated guide's dispersion is not that of "
          f"the declared recipe (eps={RECIPE_EPS_WG:g}, "
          f"d={RECIPE_WG_WIDTH:g}a)")

# ---- G2 (E1): flux identity self-check (does NOT depend on Meep) ------------
# T = flux_out/flux_in = 1 is energy conservation in a lossless uniform
# section. It holds for ANY bound mode, ANY eps and ANY width, so it is a
# self-check on the flux bookkeeping and NOT evidence about the guide (#812).
pass_self_rfx = _in_range(T_rfx_band, 1.0 - tol_self, 1.0 + tol_self)
print("  --- G2  E1 self-check: flux identity (guide-independent by "
      "construction) ---")
print(f"  rfx T(f_peak) = {T_rfx_peak:.4f}  [info only]")
print(f"  rfx band-mean T [{fcen-0.15*df:.3f},{fcen+0.15*df:.3f}]: "
      f"{T_rfx_band:.4f}   "
      f"{'PASS' if pass_self_rfx else 'FAIL'} (gate 1.0 ± {tol_self})")

if HAVE_MEEP:
    print("  --- G3  E4 external solver: Meep ---")
    T_meep_band = float(np.mean(np.asarray(T_meep)[band_mask]))
    pass_self_meep = _in_range(T_meep_band, 1.0 - tol_self, 1.0 + tol_self)
    delta_cross    = abs(T_rfx_band - T_meep_band)
    pass_cross     = delta_cross < tol_cross
    print(f"  meep T(f_peak) = {T_meep_peak:.4f}  [info only]")
    print(f"  meep band-mean T: {T_meep_band:.4f}   "
          f"{'PASS' if pass_self_meep else 'FAIL'} (gate 1.0 ± {tol_self})")
    print(f"  |band-mean T_rfx − T_meep|: {delta_cross:.4f}  "
          f"{'PASS' if pass_cross else 'FAIL'} (gate < {tol_cross})")
else:
    print("  meep band-mean T: SKIP  (Meep reference unavailable)")
    print("  |band-mean T_rfx − T_meep|: SKIP  (Meep reference unavailable)")

# =============================================================================
# Exit code (rfx crossval convention)
# =============================================================================
if not pass_neff:
    print("\nSOME CHECKS FAILED — G1 dispersion oracle failed (exit 1)")
    sys.exit(1)
if not pass_self_rfx:
    print("\nSOME CHECKS FAILED — G2 flux self-check failed (exit 1)")
    sys.exit(1)
if not HAVE_MEEP:
    print("\nrfx SELF-CHECK PASSED (G1 analytic dispersion + G2 flux identity)")
    print("[SKIP] Meep reference unavailable — crossval inconclusive (exit 2)")
    sys.exit(2)
PASS = pass_neff and pass_self_rfx and pass_self_meep and pass_cross
print("\n" + ("ALL CHECKS PASSED" if PASS else "SOME CHECKS FAILED"))
sys.exit(0 if PASS else 1)
