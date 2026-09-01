"""D6 (issue #786) — is the low-order error term the Meixner EDGE term?

PRE-DECLARATION AND HARNESS IN ONE FILE. ``--windows-only`` writes the
frozen windows; it is committed BEFORE any D6 fit is run, and before the
D5 numbers this fit will use have been read.

MODELS. Both are fitted to the SAME uniform-arm rungs (D0 + D5), by
nonlinear least squares, with h = dz_fine = 0.25*s mm.

  M0 (null, the model D4b already used):
        f(h) = f_inf - C h^p                      3 free parameters
  M1 (edge model, exponents FIXED BY THEORY, not fitted):
        f(h) = f_inf + A h^(4/3) - B h^2          3 free parameters
      4/3 = 2*nu with nu = pi/(2*pi - pi/2) = 2/3, the Meixner exponent
      of the 3*pi/2 conductor wedge the 1.5 mm-thick PEC trace presents
      (frozen in the base pre-declaration, D2 section).
      2 = the Yee bulk order, which D4a MEASURED as 2.0001 on the
      exact-reference twin.

WINDOWS (frozen; the 1 MHz / 10 MHz pair is the same first-principles
Cramer-Rao-derived pair as D3/D4, and 3x is a plain model-comparison
factor -- no symptom number enters):

  ATTRIBUTE the low-order term to the EDGE (candidate 2)
      iff RMS_M1 <= 1.0 MHz AND RMS_M1 <= RMS_M0 / 3
  REJECT the Meixner exponent
      iff RMS_M1 > 10.0 MHz OR RMS_M1 > 3 * RMS_M0
  INCONCLUSIVE otherwise.

Also REPORTED (diagnostics, judged by nothing): a free-exponent fit
f(h) = f_inf + A h^a - B h^b, and each model's extrapolated f_inf.

Run:  PYTHONPATH=. python -m validation.research.convergence_floor.d6_two_term_model [--windows-only]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

from validation.research.convergence_floor import fixture as fx

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
WIN = os.path.join(RES, "predeclared_windows_786_D6.json")
OUT = os.path.join(RES, "d6_two_term.json")

NU = np.pi / (2 * np.pi - np.pi / 2)     # 2/3
EDGE_ORDER = 2.0 * NU                    # 4/3
BULK_ORDER = 2.0
ATTRIBUTE_RMS_HZ = 1.0e6
REJECT_RMS_HZ = 10.0e6
RATIO = 3.0


def windows() -> dict:
    return {
        "issue": 786, "discriminator": "D6", "predeclared_utc": "2026-08-30",
        "M0": "f(h) = f_inf - C h^p (3 free parameters)",
        "M1": ("f(h) = f_inf + A h^(%.4f) - B h^%.1f (3 free parameters; "
               "exponents fixed by theory)" % (EDGE_ORDER, BULK_ORDER)),
        "edge_exponent": EDGE_ORDER,
        "edge_exponent_derivation": ("Meixner: nu = pi/(2*pi - theta) with "
                                     "theta = pi/2 (a 90 deg conductor "
                                     "corner) -> nu = 2/3, frequency error "
                                     "order 2*nu = 4/3"),
        "bulk_exponent": BULK_ORDER,
        "bulk_exponent_provenance": ("D4a measured 2.0001 on the "
                                     "exact-reference empty-box twin"),
        "attribute_to_edge": "RMS_M1 <= 1.0 MHz AND RMS_M1 <= RMS_M0 / 3",
        "reject_meixner": "RMS_M1 > 10.0 MHz OR RMS_M1 > 3 * RMS_M0",
        "inconclusive": "otherwise",
        "attribute_rms_hz": ATTRIBUTE_RMS_HZ,
        "reject_rms_hz": REJECT_RMS_HZ,
        "ratio": RATIO,
        "reported_not_judged": ["free-exponent fit f_inf + A h^a - B h^b",
                                "each model's extrapolated f_inf"],
    }


def _fit(resid, x0, bounds=None):
    kw = dict(max_nfev=200000)
    if bounds is not None:
        kw["bounds"] = bounds
    r = least_squares(resid, x0, **kw)
    return r


def main():
    fx.quiet_third_party_warnings()
    os.makedirs(RES, exist_ok=True)
    w = windows()
    if os.path.exists(WIN):
        if json.load(open(WIN)) != w:
            raise SystemExit("D6 windows on disk differ -- frozen, refusing")
    else:
        with open(WIN, "w") as fh:
            json.dump(w, fh, indent=1)
        print("wrote", WIN)
    if "--windows-only" in sys.argv:
        return

    d0 = json.load(open(os.path.join(RES, "d0_reproduction.json")))
    tbl = {r["scale"]: r["f_target"] for r in d0["rows"] if not r["multiband"]}
    p5 = os.path.join(RES, "d5_turnover.json")
    if os.path.exists(p5):
        for r in json.load(open(p5))["rows"]:
            tbl[r["scale"]] = r["f_target"]
    ss = sorted(tbl, reverse=True)
    h = np.array([fx.PC_DZF0 * s for s in ss]) * 1e3      # mm, O(0.1)
    f = np.array([tbl[s] for s in ss]) / 1e9              # GHz, O(5.5)

    def rms(r):
        return float(np.sqrt(np.mean(r ** 2)) * 1e9)      # back to Hz

    r0 = _fit(lambda p: p[0] - p[1] * h ** p[2] - f,
              [f.max() + 0.01, 1.0, 2.0],
              bounds=([0, -np.inf, 0.2], [np.inf, np.inf, 6.0]))
    r1 = _fit(lambda p: p[0] + p[1] * h ** EDGE_ORDER
              - p[2] * h ** BULK_ORDER - f,
              [f.min(), 1.0, 1.0])
    rfree = _fit(lambda p: p[0] + p[1] * h ** p[3] - p[2] * h ** p[4] - f,
                 [f.min(), 1.0, 1.0, EDGE_ORDER, BULK_ORDER],
                 bounds=([0, -np.inf, -np.inf, 0.2, 0.2],
                         [np.inf, np.inf, np.inf, 6.0, 6.0]))

    # REPORTED, JUDGED BY NOTHING: (i) a grid over exponent pairs with the
    # amplitudes solved linearly at each pair, and (ii) fixed-b=2 fits at
    # several candidate low-order exponents. Both exist to say whether the
    # exponent is IDENTIFIABLE from this ladder at all.
    grid = []
    for a in np.arange(0.30, 2.001, 0.02):
        for b in np.arange(a + 0.15, 4.001, 0.02):
            M = np.column_stack([np.ones_like(h), h ** a, h ** b])
            x, *_ = np.linalg.lstsq(M, f, rcond=None)
            grid.append((rms(f - M @ x), float(a), float(b), x.tolist()))
    grid.sort()
    fixed_b2 = {}
    for a in (0.5, 2.0 / 3.0, 1.0, EDGE_ORDER):
        M = np.column_stack([np.ones_like(h), h ** a, h ** BULK_ORDER])
        x, *_ = np.linalg.lstsq(M, f, rcond=None)
        fixed_b2["a=%.4f" % a] = {"rms_hz": rms(f - M @ x),
                                  "f_inf_hz": float(x[0] * 1e9)}

    rms0, rms1 = rms(r0.fun), rms(r1.fun)
    if rms1 <= ATTRIBUTE_RMS_HZ and rms1 <= rms0 / RATIO:
        verdict = ("ATTRIBUTED to the EDGE: the theory-fixed h^4/3 + h^2 "
                   "model fits every rung to %.3f MHz RMS, %.1fx better "
                   "than the single-power-law model (%.3f MHz)."
                   % (rms1 / 1e6, rms0 / rms1, rms0 / 1e6))
    elif rms1 > REJECT_RMS_HZ or rms1 > RATIO * rms0:
        verdict = ("MEIXNER EXPONENT REJECTED (RMS_M1 = %.3f MHz vs "
                   "RMS_M0 = %.3f MHz)" % (rms1 / 1e6, rms0 / 1e6))
    else:
        verdict = ("INCONCLUSIVE (RMS_M1 = %.3f MHz, RMS_M0 = %.3f MHz)"
                   % (rms1 / 1e6, rms0 / 1e6))

    out = {
        "issue": 786, "discriminator": "D6",
        "scales": ss, "h_mm": h.tolist(), "f_ghz": f.tolist(),
        "M0": {"f_inf_hz": float(r0.x[0] * 1e9), "C": float(r0.x[1]),
               "p": float(r0.x[2]), "rms_hz": rms0,
               "residuals_hz": (r0.fun * 1e9).tolist()},
        "M1": {"f_inf_hz": float(r1.x[0] * 1e9), "A": float(r1.x[1]),
               "B": float(r1.x[2]), "a": EDGE_ORDER, "b": BULK_ORDER,
               "rms_hz": rms1, "residuals_hz": (r1.fun * 1e9).tolist()},
        "free": {"f_inf_hz": float(rfree.x[0] * 1e9), "A": float(rfree.x[1]),
                 "B": float(rfree.x[2]), "a": float(rfree.x[3]),
                 "b": float(rfree.x[4]), "rms_hz": rms(rfree.fun)},
        "diagnostics_reported_not_judged": {
            "exponent_grid_best": [
                {"rms_hz": g[0], "a": g[1], "b": g[2],
                 "f_inf_hz": g[3][0] * 1e9, "A": g[3][1], "B": g[3][2]}
                for g in grid[:6]],
            "degenerate": bool(abs(grid[0][2] - grid[0][1]) < 0.35),
            "fixed_bulk_order_2_fits": fixed_b2,
            "f_inf_spread_over_admissible_models_hz": float(
                max([r0.x[0], r1.x[0]] + [v["f_inf_hz"] / 1e9
                                          for v in fixed_b2.values()])
                - min([r0.x[0], r1.x[0]] + [v["f_inf_hz"] / 1e9
                                            for v in fixed_b2.values()])
            ) * 1e9,
        },
        "verdict": verdict,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("M0  f_inf=%.6f GHz  p=%.3f  RMS=%.4f MHz"
          % (out["M0"]["f_inf_hz"] / 1e9, out["M0"]["p"], rms0 / 1e6))
    print("M1  f_inf=%.6f GHz  A=%.4g B=%.4g (a=4/3, b=2)  RMS=%.4f MHz"
          % (out["M1"]["f_inf_hz"] / 1e9, out["M1"]["A"], out["M1"]["B"],
             rms1 / 1e6))
    print("free f_inf=%.6f GHz  a=%.3f b=%.3f  RMS=%.4f MHz"
          % (out["free"]["f_inf_hz"] / 1e9, out["free"]["a"],
             out["free"]["b"], out["free"]["rms_hz"] / 1e6))
    print("\nD6:", verdict)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
