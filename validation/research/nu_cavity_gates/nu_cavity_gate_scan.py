"""Committed record of the scans the two NU cavity gates are derived from.

The #573 review noted that the scan numbers had become load-bearing in three
committed files with no committed artifact behind them. This script reproduces
the scans behind BOTH gate lanes (#573 committed the xy lane; #596 added the
z lane and the domain-size sweeps):

xy lane (tests/oracle/test_nonuniform_xy_cavity_accuracy.py, TM110):

  * xy resolution / grading scan -> the per-configuration envelope the xy gate
                                    uses (5 points, committed envelope 0.0282 %)
  * xy harminv window scan       -> that the committed 8000-step point is the
                                    family maximum, so the envelope covers
                                    estimator scatter by construction (4 points,
                                    committed span 0.0071 pt)
  * graded-vs-ungraded at IDENTICAL extents -> the grading term itself
                                    (0.0282% vs 0.0084%, factor 3.35x)

z lane (tests/oracle/test_nonuniform_cavity_accuracy.py, TM111 — the z-graded axis):

  * z resolution / grading scan  -> the committed envelope 0.0252 % plus its
                                    four neighbours, including the dx = 2.0 mm
                                    50x warning point (1.2567 %). NOTE: that
                                    point is reproduced the way the TEST
                                    constructs it — analytic f from the NOMINAL
                                    a, b = 40, 35 mm — so at dx = 2.0 mm it
                                    includes the b = 35 mm dimension-snapping
                                    offset on top of coarse-mesh dispersion.
                                    That is the configuration the warning
                                    fences, so the confound is part of the
                                    recorded number, not a bug in this script.
  * z harminv window scan        -> 4 points, committed span 0.0110 pt, 8000
                                    steps again the family maximum.

domain-size sweeps, BOTH lanes: cavity size swept at fixed dx = 1 mm and fixed
4:1 grading (xy committed: 0.0489/0.0390/0.0282/0.0139/0.0203; z committed:
0.0373/0.0306/0.0252/0.0264/0.0213). The #573 reviewer's exact swept sizes
were NOT recorded anywhere (checked: PR #573 body/reviews/inline comments,
research notes); this script fixes a documented re-derivation grid so the
sweep is auditable even though it cannot be bit-identical to the review's:
scale factors s in {0.6, 0.8, 1.0, 1.2, 1.4} applied to every extent, 3rd
point = the committed configuration.

  xy lane sizes: graded bands (coarse/fine) = (19/2)*s mm on x, (16/2)*s mm
                 on y, snapped to whole cells; z fixed at 10 x 1 mm.
  z  lane sizes: a, b = 40*s, 35*s mm — integer mm at every s, so no
                 dimension-snapping confound enters the sweep — and z graded
                 bands (17/2)*s mm, snapped to whole cells.

  MEASURED OUTCOME of this grid (2026-08-09, single machine): the center
  points reproduce the committed envelopes exactly (0.0282 / 0.0252 %), and
  the residual still tracks f0 (larger cavity -> generally smaller error) —
  the finding the committed sweep supports. The OFF-CENTER committed values
  do NOT reproduce on this grid (largest gap: z s=0.6 measured 0.0715 % vs
  committed 0.0373 %, ~3x the ~0.011 pt harminv-scatter class), which is the
  expected signature of a DIFFERENT (unrecorded) size grid, not of a physics
  change — every same-configuration value in this file reproduces to the last
  printed digit. The committed off-center numbers therefore remain
  reviewer-only measurements; this grid is the going-forward reference.
  Note also: at s=0.6 BOTH lanes measure above their committed gates
  (xy 0.0613 % > 0.05 %, z 0.0715 % > 0.04 %). The gates bind only the
  committed configuration, so nothing reds — but this is direct evidence for
  the in-test warning that cavity SIZE re-parameterization requires gate
  re-derivation.

Research-lane script: no gate, no fixture. Run it to re-derive, not in CI.

    python validation/research/nu_cavity_gates/nu_cavity_gate_scan.py \
        [--quick] [--only {xy,z,size} ...]

``--quick`` runs the two committed-envelope points only (unchanged from #573).
``--only`` (repeatable) restricts the full run to the named scan families.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)

from rfx import GaussianPulse, Simulation  # noqa: E402
from rfx.auto_config import smooth_grading  # noqa: E402
from rfx.grid import C0  # noqa: E402


def _graded(coarse_mm: float, fine_mm: float, dx: float, fine: float) -> list[float]:
    n_c = int(round(coarse_mm * 1e-3 / dx))
    n_f = int(round(fine_mm * 1e-3 / fine))
    return list(smooth_grading([dx] * n_c + [fine] * n_f + [dx] * n_c, max_ratio=1.3))


def _tm110_error(dx_prof, dy_prof, dz_prof, dx_boundary, steps=8000) -> float:
    a = float(np.sum(dx_prof))
    b = float(np.sum(dy_prof))
    d = float(np.sum(dz_prof))
    f = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)
    sim = Simulation(freq_max=2 * f, domain=(0, 0, 0), boundary="pec",
                     dx=dx_boundary, dx_profile=np.asarray(dx_prof),
                     dy_profile=np.asarray(dy_prof), dz_profile=np.asarray(dz_prof))
    sim.add_source((a / 3, b / 3, d / 2), "ez",
                   waveform=GaussianPulse(f0=f, bandwidth=0.8))
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), "ez")
    res = sim.run(n_steps=steps, skip_preflight=True)
    modes = sorted((m.freq for m in res.find_resonances(
        freq_range=(0.6 * f, 1.5 * f))), key=lambda v: abs(v - f))
    if not modes:
        return float("nan")
    return abs(modes[0] - f) / f * 100.0


def _tm111_error(a: float, b: float, dz_prof, dx_boundary, steps=8000) -> float:
    """z-lane leg, imitating tests/oracle/test_nonuniform_cavity_accuracy.py: uniform
    xy from ``domain=(a, b)`` + ``dx``, genuinely graded z from ``dz_profile``,
    analytic TM111 from NOMINAL a, b and the REALIZED graded z extent d —
    exactly the quantities the test feeds its closed form."""
    d = float(np.sum(dz_prof))
    f = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2 + (1 / d) ** 2)
    sim = Simulation(freq_max=2 * f, domain=(a, b), boundary="pec",
                     dx=dx_boundary, dz_profile=np.asarray(dz_prof))
    # ez source/probe at z = d/4 (NOT d/2, a TM111 node), xy at thirds — the
    # test's own placement.
    sim.add_source((a / 3, b / 3, d / 4), "ez",
                   waveform=GaussianPulse(f0=f, bandwidth=0.8))
    sim.add_probe((2 * a / 3, 2 * b / 3, d / 4), "ez")
    res = sim.run(n_steps=steps, skip_preflight=True)
    modes = sorted((m.freq for m in res.find_resonances(
        freq_range=(0.6 * f, 1.5 * f))), key=lambda v: abs(v - f))
    if not modes:
        return float("nan")
    return abs(modes[0] - f) / f * 100.0


# Committed reference values, printed alongside each measurement so a re-run is
# self-auditing. Sources: the gate blocks of the two tests (do NOT edit those
# from here — if a re-run deviates beyond the ~0.01 pt harminv-scatter class,
# that is a finding to surface, not a number to overwrite).
_XY_RES_COMMITTED = {(1.0, 4): 0.0282, (1.0, 2): 0.0224, (1.0, 1): 0.0108,
                     (0.5, 4): 0.0458, (2.0, 4): 0.0868}
_Z_RES_COMMITTED = {(1.0, 4): 0.0252, (1.0, 2): 0.0164, (1.0, 1): 0.0011,
                    (0.5, 4): 0.0032, (2.0, 4): 1.2567}
_XY_WIN_COMMITTED_SPAN = 0.0071
_Z_WIN_COMMITTED_SPAN = 0.0110
_SIZE_SCALES = (0.6, 0.8, 1.0, 1.2, 1.4)
_XY_SIZE_COMMITTED = (0.0489, 0.0390, 0.0282, 0.0139, 0.0203)
_Z_SIZE_COMMITTED = (0.0373, 0.0306, 0.0252, 0.0264, 0.0213)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="the two committed-envelope points only")
    ap.add_argument("--only", action="append", choices=("xy", "z", "size"),
                    help="restrict the full run to the named scan families "
                         "(repeatable; default = all)")
    args = ap.parse_args(argv)

    def _want(fam: str) -> bool:
        return not args.quick and (args.only is None or fam in args.only)

    dx, fine = 1e-3, 0.25e-3
    dxp = _graded(19.0, 2.0, dx, fine)
    dyp = _graded(16.0, 2.0, dx, fine)
    dzp = [dx] * 10
    a, b = float(np.sum(dxp)), float(np.sum(dyp))

    print(f"xy extents a={a * 1e3:.4f} b={b * 1e3:.4f} mm")
    e_graded = _tm110_error(dxp, dyp, dzp, dx)
    print(f"  graded 4:1 (the committed configuration) : {e_graded:.4f} %"
          "   <- xy envelope")

    nx, ny = int(round(a / dx)), int(round(b / dx))
    cx, cy = a / nx, b / ny
    e_uniform = _tm110_error(np.full(nx, cx), np.full(ny, cy), dzp, cx)
    print(f"  UNGRADED at identical extents            : {e_uniform:.4f} %"
          f"   (cells {cx * 1e3:.4f}/{cy * 1e3:.4f} mm, "
          f"{(cx / dx - 1) * 100:+.2f}%/{(cy / dx - 1) * 100:+.2f}% vs 1 mm)")
    print(f"  -> grading adds {e_graded - e_uniform:+.4f} pt, "
          f"factor {e_graded / max(e_uniform, 1e-12):.2f}x above the uniform floor")

    if args.quick:
        return 0

    if _want("xy"):
        print("\nresolution / grading scan (xy):")
        for dx_mm, ratio in ((1.0, 4), (1.0, 2), (1.0, 1), (0.5, 4), (2.0, 4)):
            d_ = dx_mm * 1e-3
            f_ = d_ / ratio if ratio > 1 else d_
            e = _tm110_error(_graded(19.0, 2.0, d_, f_), _graded(16.0, 2.0, d_, f_),
                             [d_] * 10, d_)
            print(f"  dx {dx_mm:.2f} mm, grading {ratio}:1 -> {e:.4f} % "
                  f"(committed {_XY_RES_COMMITTED[(dx_mm, ratio)]:.4f})")

        print("\nharminv window scan (xy, committed configuration; "
              f"committed span {_XY_WIN_COMMITTED_SPAN:.4f} pt, 8000-step point the max):")
        win = {s: _tm110_error(dxp, dyp, dzp, dx, s) for s in (4000, 8000, 12000, 16000)}
        for s, e in win.items():
            print(f"  n_steps {s:6d} -> {e:.4f} %")
        print(f"  -> span {max(win.values()) - min(win.values()):.4f} pt")

    # z lane: the TM111 z-graded configuration of tests/oracle/test_nonuniform_cavity_accuracy.py
    az, bz = 40e-3, 35e-3
    dzp_z = _graded(17.0, 2.0, dx, fine)

    if _want("z"):
        print(f"\nz lane (TM111): a={az * 1e3:.1f} b={bz * 1e3:.1f} mm, "
              f"graded z d={float(np.sum(dzp_z)) * 1e3:.4f} mm")
        print("resolution / grading scan (z):")
        for dx_mm, ratio in ((1.0, 4), (1.0, 2), (1.0, 1), (0.5, 4), (2.0, 4)):
            d_ = dx_mm * 1e-3
            f_ = d_ / ratio if ratio > 1 else d_
            e = _tm111_error(az, bz, _graded(17.0, 2.0, d_, f_), d_)
            note = "   <- z envelope" if (dx_mm, ratio) == (1.0, 4) else ""
            if (dx_mm, ratio) == (2.0, 4):
                note = "   <- 50x warning point (b=35 mm snaps at dx=2 mm)"
            print(f"  dx {dx_mm:.2f} mm, grading {ratio}:1 -> {e:.4f} % "
                  f"(committed {_Z_RES_COMMITTED[(dx_mm, ratio)]:.4f}){note}")

        print("\nharminv window scan (z, committed configuration; "
              f"committed span {_Z_WIN_COMMITTED_SPAN:.4f} pt, 8000-step point the max):")
        win = {s: _tm111_error(az, bz, dzp_z, dx, s) for s in (4000, 8000, 12000, 16000)}
        for s, e in win.items():
            print(f"  n_steps {s:6d} -> {e:.4f} %")
        print(f"  -> span {max(win.values()) - min(win.values()):.4f} pt")

    if _want("size"):
        print("\ndomain-size sweep (xy lane, fixed dx=1 mm / 4:1 grading; "
              "bands (19/2, 16/2) mm x s):")
        for s, committed in zip(_SIZE_SCALES, _XY_SIZE_COMMITTED):
            dxp_s = _graded(19.0 * s, 2.0 * s, dx, fine)
            dyp_s = _graded(16.0 * s, 2.0 * s, dx, fine)
            a_s, b_s = float(np.sum(dxp_s)), float(np.sum(dyp_s))
            e = _tm110_error(dxp_s, dyp_s, dzp, dx)
            note = "   <- committed configuration" if s == 1.0 else ""
            print(f"  s={s:.1f} (a={a_s * 1e3:.3f} b={b_s * 1e3:.3f} mm) -> "
                  f"{e:.4f} % (committed {committed:.4f}){note}")

        print("\ndomain-size sweep (z lane, fixed dx=1 mm / 4:1 grading; "
              "a,b = (40,35) mm x s, z bands (17/2) mm x s):")
        for s, committed in zip(_SIZE_SCALES, _Z_SIZE_COMMITTED):
            dzp_s = _graded(17.0 * s, 2.0 * s, dx, fine)
            e = _tm111_error(40e-3 * s, 35e-3 * s, dzp_s, dx)
            note = "   <- committed configuration" if s == 1.0 else ""
            print(f"  s={s:.1f} (a={40 * s:.1f} b={35 * s:.1f} "
                  f"d={float(np.sum(dzp_s)) * 1e3:.3f} mm) -> "
                  f"{e:.4f} % (committed {committed:.4f}){note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
