"""Standalone Meep producer: 2.4 GHz FR4 patch-antenna resonance (TM010).

THIRD cross-validation solver alongside rfx and openEMS for the canonical
2.4 GHz rectangular microstrip patch of ``examples/crossval/05_patch_antenna.py``.

Geometry matches cv05 exactly (patch W=38.0 x L=29.5 mm, FR4 eps_r=4.3,
h=1.5 mm, finite 60x55 mm ground plane, probe inset 8.0 mm from the patch
edge and centred in y). Metal is modelled as zero-thickness PEC sheets for
the patch and the finite ground plane -- same as openEMS -- so the ground is
NOT an infinite full-domain PEC wall (an infinite GP shifts the resonance
~8% high, per cv05).

Method (per the agreed plan):
  * 3D FDTD with ``mp.Mirror(mp.Y)`` symmetry. The feed is centred in y, so
    the y=0 plane is a mirror plane and the fundamental TM010 mode (Ez even
    in y) survives it -- this halves the cell count, which matters because
    this Meep build is SERIAL (non-MPI).
  * A broadband Gaussian Ez point source under the patch drives the cavity;
    after the source turns off the structure rings down and ``mp.Harminv``
    extracts the resonant frequency + Q at a patch-interior probe point.
  * Resolution / ringdown / PML margin are env-tunable so the run can be
    iterated cheaply (keep cells x steps modest for the serial build).

This is a STANDALONE producer: it imports ONLY meep/numpy (NOT rfx/jax) and
writes a JSON summary. It does not compute a calibrated S11 by default (the
coarse-mesh S11 dip is known-noisy); the Harminv resonance is the primary
deliverable -- that is the quantity rfx and openEMS cross-validate cleanly.

Run (Meep lives ONLY in the dedicated venv):
    /tmp/meepenv/bin/python scripts/research/calibration/crossval/meep_patch_s11.py

Env knobs (all optional):
    MEEP_PATCH_JSON   output path  (default scripts/research/calibration/crossval/out/meep_patch_s11.json)
    MEEP_RES          cells per mm (default 3  -> dz=0.33 mm, ~4.5 cells in FR4)
    MEEP_RUNTIME      Harminv ringdown time in Meep units after sources (default 300)
    MEEP_DPML         PML thickness in mm (default 8)
    MEEP_AIR_BELOW    air below the finite GP in mm (default 10, cf. cv05 rfx 12)
    MEEP_AIR_ABOVE    air above the patch in mm (default 15, cf. cv05 rfx 25)
    MEEP_AIR_LATERAL  lateral air margin around the GP in mm (default 10)
    MEEP_LOSSTAN      FR4 loss tangent (default 0.0 -> matches cv05 lossless FR4)
"""

import json
import math
import os
import time

import numpy as np
import meep as mp

# ---------------------------------------------------------------------------
# Physical constants and Meep unit system
# ---------------------------------------------------------------------------
C0 = 2.998e8                       # m/s (matches cv05)
UNIT_A = 1.0e-3                    # Meep length unit = 1 mm; all coords in mm


def to_meep_f(f_hz: float) -> float:
    """Real frequency [Hz] -> dimensionless Meep frequency (a = 1 mm)."""
    return f_hz * UNIT_A / C0


def from_meep_f(f_meep: float) -> float:
    """Dimensionless Meep frequency -> real frequency [Hz]."""
    return f_meep * C0 / UNIT_A


def _envf(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


# ---------------------------------------------------------------------------
# Design parameters (mm) -- identical to cv05 05_patch_antenna.py
# ---------------------------------------------------------------------------
f_design = 2.4e9                   # Hz
eps_r = 4.3                        # FR4 relative permittivity
loss_tan = _envf("MEEP_LOSSTAN", 0.0)   # cv05 uses lossless FR4 (sigma=0)

h_sub = 1.5                        # substrate thickness (mm)
W = 38.0                           # patch width  (mm, along y)
L = 29.5                           # patch length (mm, along x -> sets TM010)
gx = 60.0                          # ground plane x extent (mm)
gy = 55.0                          # ground plane y extent (mm)
probe_inset = 8.0                  # feed inset from patch edge in x (mm)

# ---------------------------------------------------------------------------
# Numerical / domain parameters (env-tunable)
# ---------------------------------------------------------------------------
resolution = _envf("MEEP_RES", 3.0)        # cells per mm
run_time = _envf("MEEP_RUNTIME", 300.0)    # Meep time units after sources
dpml = _envf("MEEP_DPML", 8.0)             # PML thickness (mm)
air_below = _envf("MEEP_AIR_BELOW", 10.0)  # below finite GP (mm)
air_above = _envf("MEEP_AIR_ABOVE", 15.0)  # above patch (mm)
air_lat = _envf("MEEP_AIR_LATERAL", 10.0)  # lateral margin around GP (mm)

# Analytic Balanis TL reference (Ch. 14) -- for mode identification / sorting.
_h_m, _W_m, _L_m = h_sub * 1e-3, W * 1e-3, L * 1e-3
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * _h_m / _W_m) ** -0.5
delta_L = 0.412 * _h_m * ((eps_eff + 0.3) * (_W_m / _h_m + 0.264)) / \
          ((eps_eff - 0.258) * (_W_m / _h_m + 0.8))
f_analytic = C0 / (2 * (_L_m + 2 * delta_L) * math.sqrt(eps_eff))

# ---------------------------------------------------------------------------
# Cell geometry. Structure is centred at (x=0, y=0); the y=0 plane is the
# Mirror(Y) symmetry plane (feed centred in y). Ground plane sits at z_gnd
# with air_below+dpml beneath it and air_above+dpml above the patch.
# ---------------------------------------------------------------------------
# Metal is 1 cell thick (grid-aligned), NOT truly zero-thickness: a size-0
# metal block on the Yee grid can fall between grid planes and conduct only
# weakly, so the patch cavity never confines and the field just radiates away
# (this is exactly the "ground-plane cell placement" corner case cv05 warns
# about for Meep). cv05's rfx build likewise uses 1-cell PEC for GP and patch.
t_metal = 1.0 / resolution               # one cell (mm)

sx = gx + 2 * air_lat + 2 * dpml
sy = gy + 2 * air_lat + 2 * dpml
sz = air_below + 2 * t_metal + h_sub + air_above + 2 * dpml
cell = mp.Vector3(sx, sy, sz)

z_gnd_lo = -sz / 2 + dpml + air_below    # bottom of ground metal
z_gnd_c = z_gnd_lo + t_metal / 2         # ground metal centre
z_sub_lo = z_gnd_lo + t_metal            # substrate bottom (on top of GP)
z_sub_mid = z_sub_lo + h_sub / 2         # substrate mid-plane (source/probe z)
z_patch_lo = z_sub_lo + h_sub            # patch metal bottom (on top of sub)
z_patch_c = z_patch_lo + t_metal / 2     # patch metal centre

feed_x = -L / 2 + probe_inset            # feed x (inset from left patch edge)
probe_x = +0.35 * L                      # interior probe toward opposite edge

# Meep loss tangent: eps(w) = eps_r*(1 + i*sigma_D/w), so tan_d = sigma_D/w
# with w = 2*pi*fcen in Meep units. fcen is defined just below.
fcen = to_meep_f(f_design)
fr4 = mp.Medium(epsilon=eps_r,
                D_conductivity=loss_tan * 2 * math.pi * fcen) \
    if loss_tan > 0 else mp.Medium(epsilon=eps_r)

geometry = [
    # FR4 substrate (finite, matches GP footprint)
    mp.Block(size=mp.Vector3(gx, gy, h_sub),
             center=mp.Vector3(0, 0, z_sub_mid),
             material=fr4),
    # Finite ground plane: 1-cell PEC sheet under the substrate
    mp.Block(size=mp.Vector3(gx, gy, t_metal),
             center=mp.Vector3(0, 0, z_gnd_c),
             material=mp.metal),
    # Patch: 1-cell PEC sheet on top of the substrate
    mp.Block(size=mp.Vector3(L, W, t_metal),
             center=mp.Vector3(0, 0, z_patch_c),
             material=mp.metal),
]

# Broadband Gaussian Ez source at the feed, inside the substrate.
#
# IMPORTANT: Meep's automatic source cutoff time is ~2*cutoff/fwidth (all in
# real units, independent of the mm length-unit choice) -- with a *narrow*
# df (e.g. 1.5 GHz) and the Meep-default cutoff=5, the driven phase alone
# eats ~6.7 ns (~1990 Meep time units @ a=1mm) before `after_sources` even
# starts, starving the ringdown budget. Use a WIDE df + reduced cutoff for a
# short, broadband excitation pulse so most of the wall-clock budget goes to
# the free-decay ringdown Harminv actually needs.
src_bw_hz = _envf("MEEP_SRC_BW_HZ", 4.0e9)
src_cutoff = _envf("MEEP_SRC_CUTOFF", 3.0)
df = to_meep_f(src_bw_hz)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df, cutoff=src_cutoff),
                     component=mp.Ez,
                     center=mp.Vector3(feed_x, 0, z_sub_mid))]

symmetries = [mp.Mirror(mp.Y)]           # feed centred in y -> y=0 mirror plane

sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    geometry=geometry,
    sources=sources,
    boundary_layers=[mp.PML(dpml)],
    symmetries=symmetries,
    default_material=mp.air,
)

n_cells = int(sx * resolution) * int(sy * resolution) * int(sz * resolution)
print("=" * 70)
print("Meep patch-antenna resonance (3rd cross-validation solver)")
print("=" * 70)
print(f"Patch W={W} x L={L} mm, FR4 eps_r={eps_r} (tan-d={loss_tan}), "
      f"h={h_sub} mm")
print(f"Ground plane {gx} x {gy} mm (1-cell PEC sheets, "
      f"t={t_metal*1e3:.0f} um)")
print(f"Feed Ez source at x={feed_x:.2f} mm (inset {probe_inset} mm), "
      f"Harminv probe at x={probe_x:.2f} mm")
print(f"Cell {sx:.1f} x {sy:.1f} x {sz:.1f} mm, resolution={resolution} /mm")
print(f"  ~{n_cells:,} cells (halved by Mirror(Y)); "
      f"dz={1/resolution*1e3:.0f} um -> {h_sub*resolution:.1f} cells in FR4")
print(f"PML={dpml} mm; air below/above/lateral = "
      f"{air_below}/{air_above}/{air_lat} mm")
print(f"Analytic (Balanis TL) target: {f_analytic/1e9:.4f} GHz")
_pre_time_est = 2 * src_cutoff / df
print(f"Source: bw={src_bw_hz/1e9:.1f} GHz, cutoff={src_cutoff} "
      f"-> driven phase ~{_pre_time_est:.0f} Meep units before ringdown")
print(f"Ringdown time (after sources): {run_time} Meep units")
print("=" * 70)

# ---------------------------------------------------------------------------
# Harminv resonance extraction
# ---------------------------------------------------------------------------
probe_pt = mp.Vector3(probe_x, 0, z_sub_mid)
harminv = mp.Harminv(mp.Ez, probe_pt, fcen, df)
# Meep's Harminv defaults to Q_thresh=50, which silently discards every mode
# below Q=50. A patch antenna radiating from a finite ground plane has a
# modest Q (tens), so relax the threshold or no modes are reported at all.
harminv.Q_thresh = _envf("MEEP_QTHRESH", 5.0)

# Also record the raw probe Ez(t) so we have a solver-independent FFT
# cross-check (and can tell "ringing cavity" from "pure radiative decay")
# even if Meep's Harminv returns nothing.
probe_ts = []
probe_tt = []
dt_sample = 1.0  # Meep time units between samples


def _record(sim):
    probe_ts.append(float(sim.get_field_point(mp.Ez, probe_pt).real))
    probe_tt.append(sim.meep_time())


t0 = time.time()
sim.run(mp.after_sources(harminv),
        mp.after_sources(mp.at_every(dt_sample, _record)),
        until_after_sources=run_time)
runtime_s = time.time() - t0
print(f"\nFDTD complete in {runtime_s:.1f}s")

# Ringing vs decay diagnostic + coarse FFT fallback (bins are ~1/T wide, so
# this only localises the band; Harminv gives the precise frequency).
ts = np.asarray(probe_ts)
tt = np.asarray(probe_tt)
fft_peak_hz = float("nan")
ring_ratio = float("nan")
if ts.size > 8:
    half = ts.size // 2
    e_early = float(np.sqrt(np.mean(ts[:half] ** 2)) + 1e-30)
    e_late = float(np.sqrt(np.mean(ts[half:] ** 2)))
    ring_ratio = e_late / e_early  # ~1 => still ringing; <<1 => decayed away
    sp = np.abs(np.fft.rfft(ts - ts.mean()))
    fq = np.fft.rfftfreq(ts.size, d=dt_sample)          # cycles / Meep-time
    fq_hz = fq * C0 / UNIT_A
    band = (fq_hz > 1.5e9) & (fq_hz < 3.5e9)
    if np.any(band):
        idx = np.where(band)[0]
        fft_peak_hz = float(fq_hz[idx[int(np.argmax(sp[idx]))]])
    print(f"Probe RMS late/early = {ring_ratio:.3f} "
          f"(~1 ringing, <<1 decayed); coarse-FFT band peak "
          f"= {fft_peak_hz/1e9:.3f} GHz")

# Collect and rank modes.
raw_modes = []
for m in harminv.modes:
    f_hz = from_meep_f(float(np.real(m.freq)))
    raw_modes.append({
        "freq_hz": f_hz,
        "Q": float(m.Q),
        "amp": float(abs(m.amp)),
        "meep_freq": float(np.real(m.freq)),
    })
raw_modes.sort(key=lambda d: d["freq_hz"])

print("\nAll Harminv modes:")
for d in raw_modes:
    print(f"  f = {d['freq_hz']/1e9:8.4f} GHz   Q = {d['Q']:9.2f}   "
          f"amp = {d['amp']:.3e}")

good = [d for d in raw_modes if d["Q"] > 2 and d["amp"] > 1e-8
        and 1.5e9 < d["freq_hz"] < 3.5e9]
# Pick the mode nearest the analytic patch resonance = the TM010 fundamental.
if good:
    good_sorted = sorted(good, key=lambda d: abs(d["freq_hz"] - f_analytic))
    best = good_sorted[0]
    f_res = best["freq_hz"]
    Q_res = best["Q"]
else:
    best = None
    f_res = float("nan")
    Q_res = float("nan")

# ---------------------------------------------------------------------------
# TM010 confirmation via the residual-field spatial pattern.
# After ringdown, the longest-lived mode dominates. Sample Ez along the x-line
# through the patch centre at y=0: TM010 (half-wave along L) has ~1 sign change
# (Ez opposite sign at the two radiating edges), whereas a feed/box mode shows
# a different count. Also confirm ~uniformity along y (TM010 is flat in y).
# ---------------------------------------------------------------------------
mode_check = {}
try:
    ez_x = sim.get_array(component=mp.Ez,
                         center=mp.Vector3(0, 0, z_sub_mid),
                         size=mp.Vector3(L * 0.9, 0, 0)).real
    # Trim tiny numerical fuzz; count sign changes on the smoothed profile.
    prof = ez_x / (np.max(np.abs(ez_x)) + 1e-30)
    sig = prof[np.abs(prof) > 0.1]
    sign_changes_x = int(np.sum(np.diff(np.sign(sig)) != 0)) if sig.size else 0

    ez_y = sim.get_array(component=mp.Ez,
                         center=mp.Vector3(feed_x, 0, z_sub_mid),
                         size=mp.Vector3(0, W * 0.9, 0)).real
    profy = ez_y / (np.max(np.abs(ez_y)) + 1e-30)
    y_uniformity = float(np.min(profy) / (np.max(profy) + 1e-30)) \
        if np.max(profy) > 0 else 0.0
    mode_check = {
        "sign_changes_along_x": sign_changes_x,
        "y_min_over_max": y_uniformity,
        "note": ("TM010 expects ~1 sign change along x (half-wave over L) "
                 "and near-uniform Ez along y"),
    }
    print(f"\nMode-shape check: {sign_changes_x} sign change(s) of Ez along x "
          f"(TM010 -> 1); y min/max = {y_uniformity:.2f} (TM010 -> ~1)")
except Exception as exc:  # field sampling is a best-effort confirmation
    mode_check = {"error": repr(exc)}
    print(f"\nMode-shape check unavailable: {exc!r}")

# ---------------------------------------------------------------------------
# Report vs the two other solvers (verified agreeing this session).
# ---------------------------------------------------------------------------
F_OPENEMS = 2.487e9
F_RFX = 2.553e9
if not math.isnan(f_res):
    d_oe = 100 * abs(f_res - F_OPENEMS) / F_OPENEMS
    d_rfx = 100 * abs(f_res - F_RFX) / F_RFX
    d_an = 100 * abs(f_res - f_analytic) / f_analytic
    in_window = 2.40e9 <= f_res <= 2.56e9
    print(f"\n  Meep TM010 resonance: {f_res/1e9:.4f} GHz  (Q={Q_res:.1f})")
    print(f"    vs openEMS 2.487 GHz : {d_oe:+.2f} %")
    print(f"    vs rfx     2.553 GHz : {d_rfx:+.2f} %")
    print(f"    vs Balanis {f_analytic/1e9:.3f} GHz : {d_an:+.2f} %")
    print(f"    target window 2.40-2.56 GHz : "
          f"{'IN WINDOW' if in_window else 'OUT OF WINDOW'}")
else:
    d_oe = d_rfx = d_an = float("nan")
    in_window = False
    print("\n  No qualifying resonance found (loosen Q/amp or lengthen run).")

# ---------------------------------------------------------------------------
# Emit JSON
# ---------------------------------------------------------------------------
out_path = os.environ.get(
    "MEEP_PATCH_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "out", "meep_patch_s11.json"),
)
payload = {
    "solver": "meep",
    "meep_version": mp.__version__,
    "resonant_freq_hz": f_res,
    "resonant_Q": Q_res,
    "harminv_modes": raw_modes,
    "selected_mode": best,
    "mode_check": mode_check,
    "coarse_fft_band_peak_hz": fft_peak_hz,
    "probe_rms_late_over_early": ring_ratio,
    "resolution_cells_per_mm": resolution,
    "cells_approx": n_cells,
    "runtime_s": runtime_s,
    "run_time_meep_units": run_time,
    "in_target_window_2p40_2p56_ghz": bool(in_window),
    "analytic_resonance_hz": f_analytic,
    "diff_vs_openems_pct": d_oe,
    "diff_vs_rfx_pct": d_rfx,
    "diff_vs_analytic_pct": d_an,
    "reference_openems_hz": F_OPENEMS,
    "reference_rfx_hz": F_RFX,
    "geometry_mm": {
        "patch_W": W, "patch_L": L, "h_sub": h_sub,
        "ground_x": gx, "ground_y": gy, "probe_inset": probe_inset,
        "feed_x": feed_x, "probe_x": probe_x,
        "cell": [sx, sy, sz], "dpml": dpml,
        "air_below": air_below, "air_above": air_above, "air_lateral": air_lat,
        "eps_r": eps_r, "loss_tangent": loss_tan,
    },
    "s11": None,  # not computed: coarse-mesh lumped-port S11 dip is known-noisy
    "notes": (
        "Third cross-validation solver (Meep) for the cv05 2.4 GHz FR4 patch. "
        "Harminv resonance is the primary deliverable; S11 intentionally omitted."
    ),
}
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
print(f"\nJSON written: {out_path}")
