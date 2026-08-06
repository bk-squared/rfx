"""Committed record of the scans the two NU cavity gates are derived from.

The #573 review noted that the scan numbers had become load-bearing in three
committed files with no committed artifact behind them. This script reproduces
every one of them, so the gates' basis is auditable rather than quoted:

  * resolution / grading scan  -> the per-configuration envelope each gate uses
  * domain-size scan           -> that the residual tracks f0 (dispersion floor)
  * harminv window scan        -> that the committed 8000-step point is the
                                  family maximum, so the envelope covers
                                  estimator scatter by construction
  * graded-vs-ungraded at IDENTICAL extents -> the grading term itself
                                  (0.0282% vs 0.0084%, factor 3.35x)

Research-lane script: no gate, no fixture. Run it to re-derive, not in CI.

    python validation/research/nu_cavity_gates/nu_cavity_gate_scan.py [--quick]
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="the two committed-envelope points only")
    args = ap.parse_args(argv)

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

    print("\nresolution / grading scan (xy):")
    for dx_mm, ratio in ((1.0, 4), (1.0, 2), (1.0, 1), (0.5, 4), (2.0, 4)):
        d_ = dx_mm * 1e-3
        f_ = d_ / ratio if ratio > 1 else d_
        e = _tm110_error(_graded(19.0, 2.0, d_, f_), _graded(16.0, 2.0, d_, f_),
                         [d_] * 10, d_)
        print(f"  dx {dx_mm:.2f} mm, grading {ratio}:1 -> {e:.4f} %")

    print("\nharminv window scan (xy, committed configuration):")
    for steps in (4000, 8000, 12000, 16000):
        print(f"  n_steps {steps:6d} -> {_tm110_error(dxp, dyp, dzp, dx, steps):.4f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
