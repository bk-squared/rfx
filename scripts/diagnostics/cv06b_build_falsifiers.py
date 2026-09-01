#!/usr/bin/env python3
"""cv06b BUILD-level falsifiers for the #812 P3 re-gate — GPU lane.

criterion (B) demands the new gates fail on a real BUILD carrying the defect,
not only on a replayed sweep. This runs cv06b's own solve three times:

  baseline    the shipped geometry                       -> every gate PASSES
  stub_1cell  STUB_LEN reduced by one dx -- a SUB-BIN defect (the shift and
              its size in bins are emitted as
              summary.stub_1cell.true_shift_pct / _bins). The old bin-argmin
              estimator reports exactly 0.0000 % for a defect this size
              (tests/fixtures/cv06b_estimator_regate/
              cv06b_estimator_falsifiers.json::case_B_sub_bin_ladder.rows);
              the refined estimator must report a commensurate, non-zero
              shift. G1's window is an ACCURACY window against an oracle
              itself uncertain at the level tabulated in
              docs/design_notes/estimator_resolution_regate.md section 3 T1,
              so a one-cell error is made VISIBLE here, not fatal -- that
              distinction is stated, not gated away.
  stub_narrow W_STUB reduced to 5 cells (on-lattice), so r = Z0_line/Z_stub
              drops below 1 -- the coupling degradation whose CPU-side,
              geometry-built analogue is case C of
              scripts/diagnostics/cv06b_estimator_falsifiers.py.
              -> G2 must FAIL while the retained -10 dB depth gate still
              PASSES, which is the blindness #812 measured.

Each run writes its full metric dict as JSON, and the three are reduced to
cv06b_build_falsifiers_summary.json, so every number the verdict rests on is
re-derivable without re-solving and prose can cite it by key.

CRITERION (A) LIVES HERE TOO. The ``baseline`` leg is cv06b's own board,
own mesh and own ``evaluate()`` -- so its ``gates`` block IS the criterion-(A)
demonstration the re-gate needs, and the VESSL job additionally runs the
shipped script end to end for its exit code.

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


def _plain(o):
    """numpy scalars/arrays -> JSON-native, recursively.

    ``evaluate()``'s gate booleans are ``np.bool_`` whenever the analytic
    anchor arrives as an ``np.float64`` (it does here), and ``json.dumps``
    refuses those. Coerce rather than lose the True/False to a ``default=``
    fallback that would write 1.0/0.0.
    """
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    return o


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
        (out / f"cv06b_falsifier_{label}.json").write_text(
            json.dumps(_plain(m), indent=2) + "\n")

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
    # The true shift is DERIVED from the two builds' own analytic anchors, not
    # asserted: f_notch ~ 1/L_stub through the same eps_eff, so the anchor
    # ratio is the shift the solve should show.
    true_shift = (c["f_notch_analytic"] - b["f_notch_analytic"]) \
        / b["f_notch_analytic"] * 100
    true_bins = abs(true_shift) / 100 * b["f_notch_refined"] / b["bin_hz"]
    # "visible" = the refined estimate responded to a defect smaller than a
    # bin, by at least half the true shift.
    visible = abs(d_ref) >= 0.5 * abs(true_shift)
    print(f"  stub_1cell: true shift {true_shift:+.4f} % ({true_bins:.3f} "
          f"bin). bin argmin moved {d_bin:+.4f} %, refined moved "
          f"{d_ref:+.4f} % -> {'VISIBLE' if visible else 'NOT VISIBLE'}")

    n = results["stub_narrow"]
    g2 = n["gates"]["G2 -10 dB stopband width"]
    dep = n["gates"]["notch depth (witness only)"]
    print(f"  stub_narrow: BW ratio {n['bw_ratio']:.4f} -> G2 "
          f"{'PASS' if g2 else 'FAIL'}; depth {n['notch_depth_db']:.1f} dB -> "
          f"depth witness {'PASS' if dep else 'FAIL'}")
    good = ok and visible and (not g2) and dep

    summary = {
        "meta": {"issue": 812, "case": "cv06b",
                 "produced_by": "scripts/diagnostics/cv06b_build_falsifiers.py",
                 "board": "cv06b's own dx=63.5um, 5,729,080-cell mesh"},
        "criterion_A_baseline": {
            "gates": b["gates"], "all_pass": bool(ok),
            "err_pct": b["err_pct"], "bw_ratio": b["bw_ratio"],
            "witness_bins": b["witness_bins"],
            "notch_depth_db": b["notch_depth_db"],
            "f_notch_refined_hz": b["f_notch_refined"],
            "f_notch_bin_hz": b["f_notch_bin"],
            "f_notch_analytic_hz": b["f_notch_analytic"],
            "sub_bin_shift_bins": b["sub_bin_shift"],
            "z0_median_ohm": b["z0_median"], "solve_s": b["solve_s"]},
        "stub_1cell": {
            "stub_len_m": c["stub_len_m"], "true_shift_pct": true_shift,
            "true_shift_bins": true_bins,
            "bin_argmin_delta_pct": d_bin, "refined_delta_pct": d_ref,
            "visible": bool(visible), "gates": c["gates"],
            "solve_s": c["solve_s"]},
        "stub_narrow": {
            "w_stub_m": n["w_stub_m"], "bw_ratio": n["bw_ratio"],
            "bw_frac": n["bw_frac"], "notch_depth_db": n["notch_depth_db"],
            "err_pct": n["err_pct"], "gates": n["gates"],
            "G2_fired": bool(not g2), "depth_witness_still_passes": bool(dep),
            "solve_s": n["solve_s"]},
        "verdict": {"criterion_A": bool(ok),
                    "criterion_B_sub_bin_visible": bool(visible),
                    "criterion_B_G2_fires_on_narrow_stub": bool(not g2),
                    "criterion_B_depth_gate_still_blind": bool(dep),
                    "all_ok": bool(good)},
    }
    (out / "cv06b_build_falsifiers_summary.json").write_text(
        json.dumps(_plain(summary), indent=2) + "\n")
    print(f"\nwrote {out / 'cv06b_build_falsifiers_summary.json'}")
    print("\n" + ("BUILD FALSIFIERS OK" if good
                  else "BUILD FALSIFIERS NOT SATISFIED"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
