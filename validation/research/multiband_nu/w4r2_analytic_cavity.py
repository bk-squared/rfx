"""W4R2 — supraconvergence against an ANALYTIC target (F-S4, third and
final fixture; note section W4R2, committed BEFORE any ladder run).

Both P-C-class fixtures (phase-1 W4 and the W4R port fixture) were
INCONCLUSIVE.

CORRECTION (2026-08-30, note section WP6R.12/WP6R.13): an earlier version
of this docstring stated the CAUSE as fact — "the geometry-realization
error floor of a rasterized dielectric-loaded resonator ... sits above the
ladder's fine-scale discretization errors". That reading is WITHDRAWN. It
was an interpretation, never a measurement of this lane: what this lane
measured is only that the ~20 MHz spread is present identically in
uniform-mesh arms, so it is not a grading effect. (The #786 lane reports
that fixture's f(h) is non-monotone and exonerates geometry quantization
and port loading, which would make the "floor" reading wrong outright —
but that evidence ships in PR #788 and is NOT merged, so it is cited here
as a pointer, not as this file's ground.)

What stands unchanged as the reason for THIS fixture: the PEC trace edge
violates the smoothness hypotheses of the theorem F-S4 tests in the first
place, so a P-C-class fixture is the wrong instrument for an order claim
regardless of what the spread turns out to be.

W4R2 removes EVERY instrument layer at once by testing supraconvergence
exactly where Monk & Sueli (1994) / Li & Shields (2016) state it: a
smooth-field eigenmode of an empty PEC box on a multiband tensor grid,
judged against the ANALYTIC continuum eigenfrequency.

* No geometry: `run_nonuniform` on `build_pec_fixture` (the W1 harness) —
  no Box, no rasterization, no dielectrics, no subpixel.
* No reference ladder: the target is exact
  (TE101 of a 27 x 18 x 64 mm PEC box = 6.0255352 GHz), so there is no
  reference contamination and no Richardson divisor.
* Sparse spectrum: nearest neighbour (TE102) is 1.24 GHz away.

Fixture: z multiband fine(12 mm) | coarse(14 mm) | fine(12 mm) |
coarse(14 mm) | fine(12 mm), abrupt ratio r = 1.4 (the envelope cap,
worst case); dzf = s mm, coarse = 1.4 s mm; transverse uniform
dx = 1.5 s mm; scales s in {0.25, 0.5, 1, 2} (all band lengths and both
transverse extents are exact multiples at every scale; uniform control
nz = 64/s). Ey soft source + Ey probe; observable = harminv line nearest
the analytic TE101 (guard 3 %); e(s) = |f_meas - f_TE101|.

Frozen judge (identical p-band structure to note W4R.3, never applied to
data): fit floor 0.3 MHz (extraction + f32 field-noise class, x3);
p = LS slope of log e vs log dzf over points >= floor, >= 3 points
required; fixture gate p_uc in [1.7, 2.6]; F-S4 fires iff fixture valid
AND (p_mb < 1.5 OR p_mb < p_uc - 0.4); anomaly A4 iff p_mb > p_uc + 0.4.

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w4r2_analytic_cavity
"""

from __future__ import annotations

import json
import time

import numpy as np

from rfx.harminv import harminv
from rfx.nonuniform import run_nonuniform

from .harness import build_pec_fixture

C0 = 299792458.0
A_X = 27e-3
B_Y = 18e-3
FINE_LEN = 12e-3
COARSE_LEN = 14e-3
R_CAP = 1.4
L_Z = 3 * FINE_LEN + 2 * COARSE_LEN            # 64 mm
F_TE101 = (C0 / 2) * np.sqrt((1 / A_X) ** 2 + (1 / L_Z) ** 2)
SCALES = (0.25, 0.5, 1.0, 2.0)
T_TOTAL = 15e-9
MATCH_GUARD = 0.03
E_FLOOR_HZ = 0.3e6
BAND = (5.4e9, 6.6e9)
F0_SRC = 6.0e9
SIGMA_T = 150e-12          # ~2.4 GHz bandwidth, covers TE101/TE102 only


def mb_profile(s: float) -> np.ndarray:
    dzf = s * 1e-3
    dzc = R_CAP * dzf
    nf = int(round(FINE_LEN / dzf))
    nc = int(round(COARSE_LEN / dzc))
    assert abs(nf * dzf - FINE_LEN) < 1e-12 and abs(nc * dzc - COARSE_LEN) < 1e-12
    prof = [dzf] * nf + [dzc] * nc + [dzf] * nf + [dzc] * nc + [dzf] * nf
    return np.asarray(prof, np.float64)


def uc_profile(s: float) -> np.ndarray:
    dzf = s * 1e-3
    n = int(round(L_Z / dzf))
    assert abs(n * dzf - L_Z) < 1e-12
    return np.full(n, dzf, np.float64)


def measure(s: float, multiband: bool) -> dict:
    prof = mb_profile(s) if multiband else uc_profile(s)
    dx = 1.5e-3 * s
    grid, mats = build_pec_fixture(prof, (A_X, B_Y), dx)
    dt = float(grid.dt)
    n_steps = int(round(T_TOTAL / dt))
    t = np.arange(n_steps) * dt
    t0 = 5 * SIGMA_T
    wf = (np.exp(-((t - t0) / SIGMA_T) ** 2 / 2.0)
          * np.sin(2 * np.pi * F0_SRC * (t - t0))).astype(np.float32)
    # source/probe: Ey nodes near (a/2, b/2, L/4) and (a/3-ish, b/2, L/4)
    zn = np.concatenate([[0.0], np.cumsum(prof)])
    k_src = int(np.argmin(np.abs(zn - L_Z / 4)))
    i_src = grid.nx // 2
    j_src = grid.ny // 2
    k_prb = int(np.argmin(np.abs(zn - 3 * L_Z / 8)))
    i_prb = max(1, grid.nx // 3)
    t_start = time.time()
    out = run_nonuniform(grid, mats, n_steps,
                         sources=[(i_src, j_src, k_src, "ey", wf)],
                         probes=[(i_prb, j_src, k_prb, "ey")])
    ts = np.asarray(out["time_series"][:, 0], np.float64)
    wall = time.time() - t_start
    n_skip = int((2 * t0) / dt)
    sig = ts[n_skip:] - ts[n_skip:].mean()
    modes = harminv(sig, dt, 3e9, 9e9)
    modes = sorted(modes, key=lambda m: m.freq)
    in_band = [m for m in modes if BAND[0] <= m.freq <= BAND[1]]
    if in_band:
        m_t = min(in_band, key=lambda m: abs(m.freq - F_TE101))
        f_meas = float(m_t.freq)
    else:
        f_meas = float("nan")
    valid = np.isfinite(f_meas) and abs(f_meas - F_TE101) <= MATCH_GUARD * F_TE101
    return {
        "scale": s, "multiband": multiband, "nz": len(prof),
        "cells": int(grid.nx * grid.ny * grid.nz), "n_steps": n_steps,
        "modes": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                  for m in modes[:6]],
        "f_meas": f_meas, "valid": bool(valid),
        "err_hz": abs(f_meas - F_TE101) if valid else float("nan"),
        "wallclock_s": wall,
    }


def main():
    out = {"f_te101_hz": float(F_TE101), "arms": []}
    rows = {}
    for mb in (False, True):
        for s in SCALES:
            e = measure(s, mb)
            rows[(mb, s)] = e
            out["arms"].append(e)
            print(f"{'MB' if mb else 'UC'} s={s}: f={e['f_meas']/1e9:.6f} GHz "
                  f"err={e['err_hz']/1e6:.4f} MHz valid={e['valid']} "
                  f"cells={e['cells']} wall={e['wallclock_s']:.0f}s",
                  flush=True)

    def fit_order(mb: bool):
        pts = [(1e-3 * s, rows[(mb, s)]["err_hz"]) for s in SCALES
               if rows[(mb, s)]["valid"]
               and rows[(mb, s)]["err_hz"] >= E_FLOOR_HZ]
        if len(pts) < 3:
            return None, pts
        h = np.log10([p[0] for p in pts])
        e = np.log10([p[1] for p in pts])
        return float(np.polyfit(h, e, 1)[0]), pts

    p_uc, pts_uc = fit_order(False)
    p_mb, pts_mb = fit_order(True)
    out["p_uc"] = p_uc
    out["p_mb"] = p_mb
    out["n_fit_points"] = {"uc": len(pts_uc), "mb": len(pts_mb)}

    anomaly = None
    if p_uc is None or p_mb is None:
        verdict = "INCONCLUSIVE (fewer than 3 valid fit points >= floor)"
        fired = None
    elif p_uc < 1.7 or p_uc > 2.6:
        verdict = ("FIXTURE-INVALID (p_uc %.2f outside [1.7, 2.6])" % p_uc)
        fired = None
    else:
        fired = bool(p_mb < 1.5 or p_mb < p_uc - 0.4)
        anomaly = bool(p_mb > p_uc + 0.4)
        verdict = (f"p_uc={p_uc:.2f} p_mb={p_mb:.2f} fired={fired}"
                   + (" ANOMALY(p_mb>p_uc+0.4)" if anomaly else ""))
    out["fs4_fired"] = fired
    out["anomaly_a4"] = anomaly
    out["verdict"] = verdict
    print("F-S4 (W4R2):", verdict, flush=True)

    path = ("validation/research/multiband_nu/results/"
            "w4r2_analytic_cavity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
