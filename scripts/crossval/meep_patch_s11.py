"""Meep standalone producer: 2.4 GHz microstrip patch antenna S11 / resonance.

THIRD cross-validation solver alongside rfx and openEMS for
``examples/crossval/05_patch_antenna.py``. Geometry is copied verbatim from
cv05 (2.4 GHz rectangular patch on FR4, finite ground plane, probe/lumped
feed at an 8 mm inset). This script is a *standalone producer*: it imports
ONLY meep + numpy, never rfx/jax, and writes a single JSON so the main-env
crossval harness can pick it up.

ENVIRONMENT
-----------
Meep is importable ONLY in the isolated venv::

    /tmp/meepenv/bin/python scripts/crossval/meep_patch_s11.py

The main/system python has numpy 2.x and ``import meep`` FAILS. Do not run
this with plain ``python``.

METHOD
------
Units: 1 Meep length unit = 1 mm (``a = 1e-3 m``); Meep frequency
``f_meep = f_Hz * a / c``. Meep uses natural units with the free-space
impedance Z0 = 1, so a measured V/I ratio is multiplied by Z0 = 376.73 Ω
to recover ohms.

Stack (bottom to top), matching cv05:
  - z = 0            : finite ground-plane PEC sheet (60 x 55 mm, 2D)
  - z in [0, h]      : FR4 substrate, eps_r = 4.3 (optional loss tangent)
  - z = h            : patch PEC sheet (38 x 29.5 mm, 2D)
  - air + PML above and below (finite GP radiates below too)

Feed: a vertical Ez line-current (``size=(0,0,h)``) at the 8 mm inset,
centred in y. This is the coax/probe analogue of cv05's lumped port.

Two independent measurements:
  (a) resonance  -> ``mp.Harminv`` on Ez at a probe point inside the
      substrate near a radiating edge (filter diagonalization; clean f_res
      independent of any port calibration). This is the headline number.
  (b) S11(f)     -> input impedance Z_in(f) = V(f)/I(f) at the feed, with
      V(f) = -integral(Ez dz) across the substrate gap and I(f) = the loop
      integral of H around a small rectangle enclosing the feed post
      (Ampere's law -> total feed current). S11 = (Z-50)/(Z+50).

The y=0 plane is a mirror symmetry plane (Ez even, feed on-axis), so the
run uses ``mp.Mirror(mp.Y)`` to halve cost.

TUNABLES (env vars)
-------------------
  MEEP_PATCH_JSON      output path (default scripts/crossval/out/meep_patch_s11.json)
  MEEP_PATCH_RES       resolution = pixels per mm (default 3 -> dx = 0.333 mm)
  MEEP_PATCH_TANDELTA  FR4 loss tangent (default 0.02; set 0 to match cv05 lossless)
  MEEP_PATCH_DECAY     stop_when_fields_decayed threshold (default 1e-3)
  MEEP_PATCH_NFREQ     number of S11 frequency points (default 101)
  MEEP_PATCH_PML       PML thickness in mm (default 8)
  MEEP_PATCH_AIRLAT    lateral air gap ground-plane -> PML in mm (default 6)
  MEEP_PATCH_AIRABOVE  air above patch in mm (default 12)
  MEEP_PATCH_AIRBELOW  air below ground plane in mm (default 8)
  MEEP_PATCH_MAXTIME   hard cap on until_after_sources meep-time (default 8000)
"""

import json
import math
import os
import time

import numpy as np
import meep as mp

C0 = 2.99792458e8
Z0_FREE = 376.730313668           # free-space wave impedance (ohms)
A = 1e-3                          # 1 Meep length unit = 1 mm
MM = 1.0                          # geometry is expressed in mm below

def _f_meep(f_hz: float) -> float:
    return f_hz * A / C0

def _hz(f_meep: float) -> float:
    return f_meep * C0 / A


# ---------------------------------------------------------------------------
# Parameters (geometry copied verbatim from cv05 05_patch_antenna.py)
# ---------------------------------------------------------------------------
f_design = 2.4e9
eps_r = 4.3
tan_delta = float(os.environ.get("MEEP_PATCH_TANDELTA", "0.02"))
h_sub = 1.5                       # mm
W = 38.0                          # patch width  (y extent)
L = 29.5                          # patch length (x extent, resonant dim)
gx = 60.0                         # ground plane x
gy = 55.0                         # ground plane y
probe_inset = 8.0                 # mm from patch x-edge

resolution = float(os.environ.get("MEEP_PATCH_RES", "3"))
pml_th = float(os.environ.get("MEEP_PATCH_PML", "8"))
air_lat = float(os.environ.get("MEEP_PATCH_AIRLAT", "6"))
air_above = float(os.environ.get("MEEP_PATCH_AIRABOVE", "12"))
air_below = float(os.environ.get("MEEP_PATCH_AIRBELOW", "8"))
decay_thr = float(os.environ.get("MEEP_PATCH_DECAY", "1e-3"))
nfreq = int(os.environ.get("MEEP_PATCH_NFREQ", "101"))
max_time = float(os.environ.get("MEEP_PATCH_MAXTIME", "8000"))
json_out = os.environ.get(
    "MEEP_PATCH_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                 "meep_patch_s11.json"),
)

# ---------------------------------------------------------------------------
# Analytic reference (Balanis TL model, same as cv05)
# ---------------------------------------------------------------------------
h_m, W_m, L_m = h_sub * 1e-3, W * 1e-3, L * 1e-3
eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_m / W_m) ** (-0.5)
delta_L = 0.412 * h_m * ((eps_eff + 0.3) * (W_m / h_m + 0.264)) / \
          ((eps_eff - 0.258) * (W_m / h_m + 0.8))
f_res_analytic = C0 / (2 * (L_m + 2 * delta_L) * math.sqrt(eps_eff))

# ---------------------------------------------------------------------------
# Domain (centred at origin in x,y; z stacked). Units: mm.
# ---------------------------------------------------------------------------
sx = gx + 2 * air_lat + 2 * pml_th
sy = gy + 2 * air_lat + 2 * pml_th
z_below = air_below + pml_th               # from ground plane down to -z PML edge
z_above = air_above + pml_th               # from patch up to +z PML edge
sz = z_below + h_sub + z_above
# ground sheet at z=0, patch at z=h_sub; centre the whole z-span on 0
z0 = -(z_below) + (sz / 2) - (sz / 2)       # keep ground at z=0 (see shift below)
# We place ground plane at z=0. Meep cell is centred at origin, so shift the
# geometry so the [-z_below, h_sub+z_above] range maps into [-sz/2, sz/2].
z_shift = -(sz / 2) + z_below               # add to a "physical z (ground=0)" to get meep z
def zc(z_phys):                              # ground-plane-referenced z -> meep z
    return z_phys + z_shift

z_gnd = zc(0.0)
z_sub_c = zc(h_sub / 2.0)
z_patch = zc(h_sub)
z_mid = zc(h_sub / 2.0)

# Feed / probe locations (patch centred at x=0,y=0)
patch_x_lo = -L / 2
feed_x = patch_x_lo + probe_inset          # 8 mm inset
feed_y = 0.0                               # centred in y (symmetry plane)
# Harminv probe: inside substrate near opposite radiating edge, off the feed
probe_x = +L / 2 - 3.0
probe_y = 3.0

cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(pml_th)]

# ---------------------------------------------------------------------------
# Materials & geometry
# ---------------------------------------------------------------------------
# Meep D_conductivity for a target loss tangent at f_design:
#   tan_delta = D_conductivity / (2*pi*f_meep*eps_r)  (natural units, eps0=1)
f0_meep = _f_meep(f_design)
D_cond = tan_delta * 2 * math.pi * f0_meep * eps_r if tan_delta > 0 else 0.0
fr4 = mp.Medium(epsilon=eps_r, D_conductivity=D_cond)

geometry = [
    # FR4 substrate (0 .. h_sub)
    mp.Block(size=mp.Vector3(gx, gy, h_sub),
             center=mp.Vector3(0, 0, z_sub_c), material=fr4),
    # Ground plane: 2D PEC sheet at z=0
    mp.Block(size=mp.Vector3(gx, gy, 0),
             center=mp.Vector3(0, 0, z_gnd), material=mp.metal),
    # Patch: 2D PEC sheet at z=h_sub
    mp.Block(size=mp.Vector3(L, W, 0),
             center=mp.Vector3(0, 0, z_patch), material=mp.metal),
]

# ---------------------------------------------------------------------------
# Source: vertical Ez line current across the substrate at the feed inset
# ---------------------------------------------------------------------------
fcen = f0_meep
fmin, fmax = _f_meep(1.5e9), _f_meep(3.5e9)
fwidth = (fmax - fmin)            # broadband, covers 1.5-3.5 GHz
sources = [mp.Source(
    mp.GaussianSource(fcen, fwidth=fwidth),
    component=mp.Ez,
    center=mp.Vector3(feed_x, feed_y, z_mid),
    size=mp.Vector3(0, 0, h_sub),
)]

symmetries = [mp.Mirror(mp.Y)]   # Ez even about y=0, feed on-axis

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    symmetries=symmetries,
    force_complex_fields=False,
)

# ---------------------------------------------------------------------------
# DFT monitors for S11 = V/I at the feed
# ---------------------------------------------------------------------------
freqs = np.linspace(fmin, fmax, nfreq)
# V monitor: Ez along the vertical feed gap
dft_V = sim.add_dft_fields([mp.Ez], freqs,
                           center=mp.Vector3(feed_x, feed_y, z_mid),
                           size=mp.Vector3(0, 0, h_sub))
# I monitor: H on a small horizontal loop enclosing the feed at mid-substrate.
# Loop half-width in mm (>= ~1 cell each side).
loop_hw = max(2.0, 2.0 / resolution)
dft_H = sim.add_dft_fields([mp.Hx, mp.Hy], freqs,
                           center=mp.Vector3(feed_x, feed_y, z_mid),
                           size=mp.Vector3(2 * loop_hw, 2 * loop_hw, 0))

# ---------------------------------------------------------------------------
# Harminv resonance probe
# ---------------------------------------------------------------------------
harminv = mp.Harminv(mp.Ez, mp.Vector3(probe_x, probe_y, z_mid), fcen, fwidth)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
ncells_est = int(round(sx * resolution) * round(sy * resolution) *
                round(sz * resolution) / 2)  # /2 for y-mirror symmetry
print("=" * 70)
print("Meep patch antenna (cross-val 3rd solver)")
print("=" * 70)
print(f"  resolution = {resolution} px/mm  (dx = {1/resolution:.3f} mm, "
      f"substrate = {h_sub*resolution:.1f} cells)")
print(f"  domain     = {sx:.1f} x {sy:.1f} x {sz:.1f} mm  "
      f"(~{ncells_est/1e6:.2f} M cells with y-symmetry)")
print(f"  FR4        = eps_r {eps_r}, tan_delta {tan_delta} (D_cond={D_cond:.4g})")
print(f"  feed       = ({feed_x:.2f}, {feed_y:.2f}) mm, probe ({probe_x:.2f}, {probe_y:.2f})")
print(f"  analytic f_res (Balanis) = {f_res_analytic/1e9:.4f} GHz")
print(f"  running until fields decay < {decay_thr} (cap {max_time} meep-time)...")

t_start = time.time()
stop_cond = mp.stop_when_fields_decayed(
    50, mp.Ez, mp.Vector3(probe_x, probe_y, z_mid), decay_thr)

def _capped_stop(s):
    if s.meep_time() >= max_time:
        return True
    return stop_cond(s)

sim.run(mp.after_sources(harminv), until_after_sources=_capped_stop)
runtime_s = time.time() - t_start
print(f"  done in {runtime_s:.1f}s  (meep_time reached {sim.meep_time():.0f})")

# ---------------------------------------------------------------------------
# Harminv post-process
# ---------------------------------------------------------------------------
modes = []
for m in harminv.modes:
    f_hz = _hz(m.freq)
    modes.append({
        "freq_hz": float(f_hz),
        "Q": float(abs(m.Q)),
        "amp": float(abs(m.amp)),
        "decay": float(m.decay),
    })
# keep physical modes in band with meaningful Q
in_band = [m for m in modes
           if 1.5e9 < m["freq_hz"] < 3.5e9 and m["Q"] > 2 and m["amp"] > 1e-6]
in_band.sort(key=lambda m: abs(m["freq_hz"] - f_res_analytic))
if in_band:
    best = in_band[0]
    f_res_meep = best["freq_hz"]
    Q_meep = best["Q"]
else:
    f_res_meep = float("nan")
    Q_meep = float("nan")

print("\n  Harminv modes (in-band, Q>2):")
for m in sorted(in_band, key=lambda m: m["freq_hz"])[:8]:
    print(f"    f = {m['freq_hz']/1e9:.4f} GHz, Q = {m['Q']:.1f}, "
          f"amp = {m['amp']:.3e}")

# ---------------------------------------------------------------------------
# S11 from feed input impedance Z_in = V/I
# ---------------------------------------------------------------------------
# Voltage: V(f) = -integral(Ez dz) across the substrate gap.
ez_line = np.array([sim.get_dft_array(dft_V, mp.Ez, i) for i in range(nfreq)])
# ez_line shape: (nfreq, Nz_gap). dz in mm.
dz_mm = 1.0 / resolution
V = -np.sum(ez_line, axis=1) * dz_mm

# Current: I(f) = closed loop integral of H around the feed (Ampere).
hx = np.array([sim.get_dft_array(dft_H, mp.Hx, i) for i in range(nfreq)])
hy = np.array([sim.get_dft_array(dft_H, mp.Hy, i) for i in range(nfreq)])
# hx, hy shape: (nfreq, Nx, Ny) over the loop plane.
dl = 1.0 / resolution
def _loop_current(hx_f, hy_f):
    # rectangle boundary: bottom (+x), right (+y), top (-x), left (-y)
    # oriented counter-clockwise in xy -> encloses +z current.
    bottom = np.sum(hx_f[:, 0]) * dl          # y = y_lo, integrate Hx dx (+x)
    top = -np.sum(hx_f[:, -1]) * dl           # y = y_hi, integrate Hx dx (-x)
    right = np.sum(hy_f[-1, :]) * dl          # x = x_hi, integrate Hy dy (+y)
    left = -np.sum(hy_f[0, :]) * dl           # x = x_lo, integrate Hy dy (-y)
    return bottom + right + top + left
I = np.array([_loop_current(hx[i], hy[i]) for i in range(nfreq)], dtype=complex)

# Z in ohms (natural-unit V/I * free-space impedance). Guard divide-by-zero.
with np.errstate(divide="ignore", invalid="ignore"):
    Z = (V / I) * Z0_FREE
Z0 = 50.0
S11 = (Z - Z0) / (Z + Z0)
S11 = np.where(np.isfinite(S11), S11, 0.0)
S11_dB = 20 * np.log10(np.maximum(np.abs(S11), 1e-6))
freqs_hz = np.array([_hz(f) for f in freqs])

# S11 dip near analytic target (+-15%)
lo = int(np.searchsorted(freqs_hz, f_res_analytic * 0.85))
hi = int(np.searchsorted(freqs_hz, f_res_analytic * 1.15))
if hi > lo:
    dip_idx = lo + int(np.argmin(S11_dB[lo:hi]))
    f_s11_dip = float(freqs_hz[dip_idx])
    s11_dip_dB = float(S11_dB[dip_idx])
else:
    f_s11_dip = float("nan")
    s11_dip_dB = float("nan")

print(f"\n  S11 min (dip) near target: {f_s11_dip/1e9:.4f} GHz, "
      f"{s11_dip_dB:.2f} dB")
if not math.isnan(f_res_meep):
    err_vs_24 = 100 * abs(f_res_meep - 2.4e9) / 2.4e9
    err_vs_an = 100 * abs(f_res_meep - f_res_analytic) / f_res_analytic
    print(f"\n  Meep Harminv resonance: {f_res_meep/1e9:.4f} GHz (Q={Q_meep:.1f})")
    print(f"    vs 2.4 GHz nominal : {err_vs_24:.2f} %")
    print(f"    vs Balanis analytic: {err_vs_an:.2f} %")

# ---------------------------------------------------------------------------
# Emit JSON
# ---------------------------------------------------------------------------
payload = {
    "producer": "meep",
    "meep_version": mp.__version__,
    "resonant_freq_hz": float(f_res_meep) if not math.isnan(f_res_meep) else None,
    "resonant_Q": float(Q_meep) if not math.isnan(Q_meep) else None,
    "s11_dip_freq_hz": float(f_s11_dip) if not math.isnan(f_s11_dip) else None,
    "s11_dip_db": float(s11_dip_dB) if not math.isnan(s11_dip_dB) else None,
    "analytic_resonance_hz": float(f_res_analytic),
    "s11": {
        "freqs_hz": [float(f) for f in freqs_hz],
        "real": [float(v.real) for v in S11],
        "imag": [float(v.imag) for v in S11],
        "mag_db": [float(v) for v in S11_dB],
        "z_real_ohm": [float(z.real) for z in Z],
        "z_imag_ohm": [float(z.imag) for z in Z],
    },
    "harminv_modes": modes,
    "resolution_px_per_mm": float(resolution),
    "dx_mm": float(1.0 / resolution),
    "substrate_cells": float(h_sub * resolution),
    "runtime_s": float(runtime_s),
    "meep_time_reached": float(sim.meep_time()),
    "ncells_est": int(ncells_est),
    "geometry_mm": {
        "eps_r": eps_r, "tan_delta": tan_delta, "h_sub": h_sub,
        "patch_W": W, "patch_L": L, "ground_x": gx, "ground_y": gy,
        "probe_inset": probe_inset, "feed_xy": [feed_x, feed_y],
        "domain": [sx, sy, sz], "pml": pml_th,
        "air_lat": air_lat, "air_above": air_above, "air_below": air_below,
    },
}
os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"\n  JSON: {json_out}")
