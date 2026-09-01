#!/usr/bin/env python3
"""cv06b BUILD-level falsifiers for the #812 P3 re-gate — GPU lane.

criterion (B) demands the new gates fail on a real BUILD carrying the defect,
not only on a replayed sweep. This runs cv06b's own solve three times:

  baseline    the shipped geometry                       -> every gate PASSES
  stub_1cell  STUB_LEN 12.0000 -> 11.9365 mm (one dx)     -> a SUB-BIN defect:
              the true notch moves +0.529 % = 0.303 bin. The old bin-argmin
              estimator reports 0.0000 % for a defect this size (measured on
              the committed dx=50um sibling sweep,
              scripts/diagnostics/cv06b_estimator_falsifiers.py case B); the
              refined estimator must report a commensurate, non-zero shift.
              G1's window (4.0 %) is an ACCURACY window against an oracle
              itself uncertain at 3.8 %, so a one-cell error is made VISIBLE
              here, not fatal -- that distinction is stated, not gated away.
  stub_narrow W_STUB 635.0 -> 317.5 um (5 cells, on-lattice). HJ(317.5um,
              254um, 3.66) ~ 68.9 ohm, so r = Z0_line/Z_stub ~ 0.67 and the
              closed form predicts a -10 dB stopband 0.673x the r=1 width.
              -> G2 must FAIL while the retained -10 dB depth gate still
              PASSES, which is the blindness #812 measured.

Each run writes its full metric dict as JSON so the verdict is re-derivable
without re-solving.

Usage (GPU):
  python scripts/diagnostics/cv06b_build_falsifiers.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CV06B = REPO / "validation/crossval/06b_msl_notch_filter_uniform.py"


def _load():
    spec = importlib.util.spec_from_file_location("_cv06b", CV06B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve(cv, label):
    sim = cv._build_sim()
    w_realized = cv._realized_trace_width(sim)
    u = w_realized / cv.H_SUB
    eps_eff = (cv.EPS_R + 1) / 2 + (cv.EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
    f_an = cv.C0 / (4 * cv.STUB_LEN * np.sqrt(eps_eff))
    print(f"\n=== {label}: STUB_LEN={cv.STUB_LEN*1e3:.4f} mm  "
          f"W_STUB={cv.W_STUB*1e6:.1f} um  W_realized={w_realized*1e6:.1f} um  "
          f"analytic {f_an/1e9:.4f} GHz ===", flush=True)
    sim.preflight(strict=False)
    t0 = time.time()
    res = sim.compute_msl_s_matrix(n_freqs=100, num_periods=20.0)
    dt = time.time() - t0
    f = np.asarray(res.freqs)
    s21 = np.abs(np.asarray(res.S[1, 0, :]))
    z0 = np.asarray(res.Z0[0, :]).real
    m = cv.evaluate(f, s21, z0, f_an)
    m["label"] = label
    m["solve_s"] = dt
    m["stub_len_m"] = float(cv.STUB_LEN)
    m["w_stub_m"] = float(cv.W_STUB)
    m["freqs_hz"] = f.tolist()
    m["s21_mag"] = s21.tolist()
    m["re_z0"] = z0.tolist()
    cv.report(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, mutate in (
        ("baseline", lambda cv: None),
        ("stub_1cell", lambda cv: setattr(cv, "STUB_LEN", cv.STUB_LEN - cv.DX)),
        ("stub_narrow", lambda cv: setattr(cv, "W_STUB", 5 * cv.DX)),
    ):
        cv = _load()            # fresh module: mutations never leak between runs
        mutate(cv)
        m = solve(cv, label)
        results[label] = m
        (out / f"cv06b_falsifier_{label}.json").write_text(json.dumps(m, indent=2))

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    b = results["baseline"]
    ok = all(b["gates"].values())
    print(f"  baseline: {'ALL GATES PASS' if ok else 'FAILED'} — "
          f"refined notch {b['f_notch_refined']/1e9:.4f} GHz "
          f"({b['sub_bin_shift']:+.3f} bin off the argmin), err "
          f"{b['err_pct']:.2f} % (< 4.0), BW ratio {b['bw_ratio']:.4f} "
          f"(0.80-1.20), witness {b['witness_bins']:.4f} bin (< 1.0)")

    c = results["stub_1cell"]
    d_ref = (c["f_notch_refined"] - b["f_notch_refined"]) / b["f_notch_refined"] * 100
    d_bin = (c["f_notch_bin"] - b["f_notch_bin"]) / b["f_notch_bin"] * 100
    # "visible" = the refined estimate responded to a defect smaller than a
    # bin. Half the true 0.529 % shift is the floor; the replayed ladder on
    # the committed sibling sweep reads +0.83 % for this shift.
    visible = abs(d_ref) >= 0.265
    print(f"  stub_1cell: true shift +0.529 % (0.303 bin). "
          f"bin argmin moved {d_bin:+.4f} %, refined moved {d_ref:+.4f} % "
          f"-> {'VISIBLE' if visible else 'NOT VISIBLE'}")

    n = results["stub_narrow"]
    g2 = n["gates"]["G2 -10 dB stopband width"]
    dep = n["gates"]["notch depth (witness only)"]
    print(f"  stub_narrow: BW ratio {n['bw_ratio']:.4f} -> G2 "
          f"{'PASS' if g2 else 'FAIL'}; depth {n['notch_depth_db']:.1f} dB -> "
          f"depth witness {'PASS' if dep else 'FAIL'}")
    good = ok and visible and (not g2) and dep
    print("\n" + ("BUILD FALSIFIERS OK" if good
                  else "BUILD FALSIFIERS NOT SATISFIED"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
