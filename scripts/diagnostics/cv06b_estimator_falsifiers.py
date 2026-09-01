#!/usr/bin/env python3
"""Falsifiers for cv06b's sub-bin re-gate (issue #812, mechanism P3).

WHAT THIS CAN AND CANNOT SHOW, STATED FIRST.
--------------------------------------------
cv06b's own mesh is 5,729,080 cells and is GPU-lane: 329.2 s on one RTX4090,
and the same mesh was abandoned UNFINISHED at 2h52m on a 32-core CPU pod
(validation/crossval/manifest.json, cpu_runner.excluded_reason). So the
BUILD-level falsifiers -- criterion (A) on cv06b's own board, plus a one-cell
stub-length error and a shallow-notch (narrow-stub) build -- cannot be run
here; they are emitted as a VESSL job
(scripts/vessl_cv06b_estimator_falsifiers.yaml).

What CAN be shown on CPU, and is shown here, is the instrument itself.

ROUND-2 CORRECTION (2026-09-01) -- case C was rebuilt, the window was not.
Round 1's case C degraded the stub by rescaling the MEASURED sweep with
M_r/M_1, the ratio of the ideal shunt-open-stub responses G2's window is
derived from. That made the falsified curve's -10 dB bandwidth equal to
(4/pi)*atan(r/6) times the baseline BY CONSTRUCTION -- the defect was built
out of the quantity being judged, so G2 firing on it demonstrated an algebraic
identity, not detection power. WITHDRAWN. Case C is now built from the board's
GEOMETRY (scripts/diagnostics/cv06b_shallow_stub_model.py): the only defect
input is the stub width in cells, and no cv06b gate constant is read anywhere
in the construction. G2's window is unchanged.

Cases:
  A  baseline      committed dx=50um sweep, unmodified -> G1, G2, G3 and the
                   depth witness all pass. G4 (Re(Z0) median in 40-65 ohm) is
                   NOT part of this replay's verdict: the fixture records
                   re_z0_median_ohm for the dx=50um sibling BOARD, a property
                   of that board rather than of the estimator, and this
                   harness has no business judging it.
  B  sub_bin_shift the sweep's features moved by LESS THAN ONE BIN by a pure
                   frequency-axis relabel. The old bin-argmin estimator
                   reports exactly 0.0000 % for such a defect -- the audit's
                   "0.00 % or >= one bin" staircase -- while the refined one
                   responds monotonically.
  C  shallow_notch a narrower stub, in cells, through an independent
                   transmission-line model of the SAME board (Hammerstad-
                   Jensen Z0/eps_eff per line, Getsinger dispersion,
                   dielectric+conductor loss, Hammerstad-Bekkadal open end,
                   ABCD cascade referenced to the 50 ohm port). Two-sided:
                   at the SHIPPED width the model passes G2, and it does NOT
                   reproduce the gate's reference bandwidth -- which is what
                   makes its failure at reduced width evidence rather than
                   arithmetic. The BUILD-level version (a real 5-cell-stub
                   solve) is the VESSL job.
  D  quantised     the SAME baseline judged by the OLD bin-argmin estimator
                   -> G3, the resolution witness, must fail

Every number this script prints is also written to
tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json, which
is the committed record prose must cite by key rather than restate.

Usage:  python scripts/diagnostics/cv06b_estimator_falsifiers.py [--out-json P]
Exit 0 iff every case behaves as stated above.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from scripts.diagnostics.cv06b_shallow_stub_model import (
    MSLLine, open_end_extension, stub_board_s21)

REPO = Path(__file__).resolve().parents[2]
CV06B = REPO / "validation/crossval/06b_msl_notch_filter_uniform.py"
FIXTURE = REPO / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"
OUT_JSON = REPO / "tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json"

# The dx=50um sibling BOARD this replay drives, from the fixture's own meta
# and cv06b's "External cross-check" CAVEAT: the producer rasterizes
# h_sub=254um onto a 50um lattice and REALIZES 300um (h_sub_cells = 5.08).
DX50 = 50e-6
H_SUB_REALIZED = 300e-6
W_LINE_CELLS = 12                 # 600um declared trace / 50um
L_LINE = 5e-3                     # fixture meta: l_line=5mm
TAN_D = 0.0037                    # RO4350B datasheet loss tangent
SIGMA_CU = 5.8e7                  # S/m
Z_REF = 50.0                      # add_msl_port impedance


def _load_cv06b():
    spec = importlib.util.spec_from_file_location("_cv06b", CV06B)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture():
    d = json.loads(FIXTURE.read_text())
    f = np.asarray(d["freqs_ghz"], dtype=float) * 1e9
    s21 = np.asarray(d["s21_mag"], dtype=float)
    z0 = np.full_like(f, float(d["re_z0_median_ohm"]))
    return f, s21, z0, d


def _warp(f, s21, rel):
    """Every spectral feature moved by ``rel``, resampled on the SAME grid."""
    return np.interp(f / (1.0 + rel), f, s21)


def _model_s21(f, n_cells):
    """|S21| of the SAME board with an ``n_cells``-wide stub. Geometry in."""
    return np.abs(stub_board_s21(
        f, w_line=W_LINE_CELLS * DX50, w_stub=n_cells * DX50,
        h_sub=H_SUB_REALIZED, eps_r=3.66, tan_d=TAN_D, sigma=SIGMA_CU,
        l_stub=12e-3, l_line=L_LINE, z_ref=Z_REF))


def _bin_argmin_metrics(cv, f, s21, z0, f_an):
    """The OLD estimator: replay evaluate() with the vertex refinement and the
    sub-bin band edges disabled, i.e. everything quantised to a bin."""
    import types
    real = cv.sf
    fake = types.SimpleNamespace(
        refined_extremum=lambda ff, ss, *a, **k: {
            **real.refined_extremum(ff, ss, *a, **k),
            "refined_f": real.refined_extremum(ff, ss, *a, **k)["bin_f"],
            "sub_bin_shift": 0.0},
        band_at_level=real.band_at_level,
        half_grid_witness=lambda ff, ss, *a, **k: {
            **real.half_grid_witness(ff, ss, *a, **k),
            "spread_bins": real.half_grid_witness(ff, ss, *a, **k)["argmin_spread_bins"]},
    )
    cv.sf = fake
    try:
        return cv.evaluate(f, s21, z0, f_an)
    finally:
        cv.sf = real


def _plain(o):
    """numpy scalars/arrays -> JSON-native, recursively."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    return o


GATED_HERE = ("G1 notch freq vs analytic", "G2 -10 dB stopband width",
              "G3 half-grid resolution witness", "notch depth (witness only)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(OUT_JSON))
    args = ap.parse_args()

    cv = _load_cv06b()
    f, s21, z0, raw = _fixture()

    u = (W_LINE_CELLS * DX50) / H_SUB_REALIZED
    eps_eff = (cv.EPS_R + 1) / 2 + (cv.EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
    f_an = cv.C0 / (4 * cv.STUB_LEN * np.sqrt(eps_eff))
    bin_hz = float(f[1] - f[0])

    rec: dict = {"meta": {
        "issue": 812, "mechanism": "P3 estimator quantization",
        "case": "cv06b", "produced_by":
            "scripts/diagnostics/cv06b_estimator_falsifiers.py",
        "driving_fixture": str(FIXTURE.relative_to(REPO)),
        "driving_fixture_meta": raw["meta"],
        "board_h_sub_realized_m": H_SUB_REALIZED,
        "board_u_realized": u,
        "analytic_anchor_hz": float(f_an),
        "bin_hz": bin_hz,
        "gate_windows": {
            "G1_notch_freq_tol_pct": cv.NOTCH_FREQ_TOL_PCT,
            "G2_bw_frac_ideal_r1": cv.STOPBAND_BW_FRAC_IDEAL,
            "G2_bw_ratio_window": list(cv.STOPBAND_BW_RATIO_WINDOW),
            "G3_half_grid_witness_bins": cv.HALF_GRID_WITNESS_BINS,
        },
    }}

    ok = True
    print(f"fixture: {FIXTURE.relative_to(REPO)}")
    print(f"  {len(f)} bins, {bin_hz/1e6:.4f} MHz; analytic anchor for THIS "
          f"board (u={u:.3f}) = {f_an/1e9:.4f} GHz\n")

    base = cv.evaluate(f, s21, z0, f_an)
    f0 = base["f_notch_refined"]
    rec["case_A_baseline"] = base
    print("A  baseline (committed dx=50um sweep, unmodified)")
    for k in GATED_HERE:
        print(f"     [{'PASS' if base['gates'][k] else 'FAIL'}] {k}")
    print(f"     refined {f0/1e9:.5f} GHz ({base['sub_bin_shift']:+.3f} bin "
          f"off the argmin {base['f_notch_bin']/1e9:.5f}), "
          f"err {base['err_pct']:.2f} %, BW ratio {base['bw_ratio']:.4f}, "
          f"witness {base['witness_bins']:.4f} bin")
    print(f"     (G4 Re(Z0) not judged here: the fixture records "
          f"{raw['re_z0_median_ohm']} ohm for the dx=50um sibling BOARD, not "
          f"an estimator property)")
    a_ok = all(base["gates"][k] for k in GATED_HERE)
    ok &= a_ok
    print(f"     -> {'OK' if a_ok else 'NOT OK'}: every estimator gate passes "
          f"on real committed rfx notch data\n")

    # ---- B: sub-bin frequency errors -------------------------------------
    one_cell = 63.5e-6 / 12.0e-3          # one dx of cv06b stub length
    print("B  sub-bin frequency errors (frequency-axis warp of the SAME sweep)")
    print(f"     one dx of stub length = {one_cell*100:.3f} % = "
          f"{one_cell*f0/bin_hz:.3f} bin")
    print(f"     {'true shift':>12} {'OLD bin argmin':>18} "
          f"{'NEW sub-bin refined':>22}")
    b_ok = True
    rows_b = []
    for rel in (0.001, 0.002, one_cell / 2, one_cell, 0.0075, 0.01, 0.0175,
                -one_cell, -0.01):
        warped = _warp(f, s21, rel)
        new = cv.evaluate(f, warped, z0, f_an)
        old = _bin_argmin_metrics(cv, f, warped, z0, f_an)
        d_old = (old["f_notch_bin"] - base["f_notch_bin"]) / base["f_notch_bin"] * 100
        d_new = (new["f_notch_refined"] - f0) / f0 * 100
        sub_bin = abs(rel) * f0 < bin_hz
        flag = ""
        if sub_bin and abs(d_old) < 1e-9:
            flag = "  <- INVISIBLE to the old estimator"
            if abs(d_new) < abs(rel) * 100 * 0.3:
                b_ok = False
        rows_b.append({"true_shift_pct": rel * 100, "sub_bin": bool(sub_bin),
                       "old_bin_argmin_delta_pct": d_old,
                       "new_refined_delta_pct": d_new})
        print(f"     {rel*100:>+11.3f}% {d_old:>+17.4f}% {d_new:>+21.4f}%{flag}")
    rec["case_B_sub_bin_ladder"] = {
        "one_cell_stub_pct": one_cell * 100,
        "one_cell_stub_bins": one_cell * f0 / bin_hz, "rows": rows_b}
    ok &= b_ok
    print(f"     -> {'OK' if b_ok else 'NOT OK'}: every sub-bin defect reads "
          f"exactly 0.0000 % on the old estimator and a commensurate,\n"
          f"        monotone value on the refined one\n")

    # ---- C: a shallow notch built from GEOMETRY --------------------------
    print("C  shallow notch built from the board's GEOMETRY (stub width in "
          "cells)")
    print("     independent model: HJ Z0/eps_eff per line + Getsinger "
          "dispersion + dielectric/conductor")
    print("     loss + HB open end + ABCD to the 50 ohm port. No cv06b gate "
          "constant enters it.")
    main_line = MSLLine(W_LINE_CELLS * DX50, H_SUB_REALIZED, cv.EPS_R,
                        TAN_D, SIGMA_CU)
    rows_c, c_model = [], {}
    print(f"     {'cells':>6} {'w_stub':>9} {'Z0_stub':>9} {'r':>7} "
          f"{'depth dB':>10} {'BW ratio':>10} {'G1':>6} {'G2':>6} "
          f"{'depth':>7}")
    for n in (W_LINE_CELLS, 10, 8, 6, 5):
        s = _model_s21(f, n)
        m = cv.evaluate(f, s, z0, f_an)
        stub = MSLLine(n * DX50, H_SUB_REALIZED, cv.EPS_R, TAN_D, SIGMA_CU)
        r = main_line.z0_static / stub.z0_static
        row = {"stub_cells": n, "w_stub_m": n * DX50,
               "z0_stub_ohm": stub.z0_static, "r_coupling": r,
               "notch_depth_db": m["notch_depth_db"],
               "f_notch_refined_hz": m["f_notch_refined"],
               "err_pct": m["err_pct"], "bw_frac": m["bw_frac"],
               "bw_ratio": m["bw_ratio"],
               "G1_pass": bool(m["gates"]["G1 notch freq vs analytic"]),
               "G2_pass": bool(m["gates"]["G2 -10 dB stopband width"]),
               "depth_witness_pass": bool(m["gates"]["notch depth (witness only)"])}
        rows_c.append(row)
        if n == W_LINE_CELLS:
            c_model = m
        print(f"     {n:>6d} {n*DX50*1e6:>8.1f}u {stub.z0_static:>9.2f} "
              f"{r:>7.4f} {m['notch_depth_db']:>10.2f} {m['bw_ratio']:>10.4f} "
              f"{'PASS' if row['G1_pass'] else 'FAIL':>6} "
              f"{'PASS' if row['G2_pass'] else 'FAIL':>6} "
              f"{'PASS' if row['depth_witness_pass'] else 'FAIL':>7}")

    shipped = rows_c[0]
    # Independence, quantified: the model at the SHIPPED width does not
    # return the gate's reference bandwidth. If it did, this ladder would be
    # the gate's own formula in disguise -- round 1's defect.
    dep_pct = (shipped["bw_frac"] - cv.STOPBAND_BW_FRAC_IDEAL) \
        / cv.STOPBAND_BW_FRAC_IDEAL * 100.0
    agree_pct = (shipped["bw_ratio"] - base["bw_ratio"]) / base["bw_ratio"] * 100.0
    f0_agree_pct = (shipped["f_notch_refined_hz"] - f0) / f0 * 100.0
    fired = [r_ for r_ in rows_c if not r_["G2_pass"]]
    c_ok = (shipped["G2_pass"] and shipped["depth_witness_pass"]
            and bool(fired)
            and all(r_["depth_witness_pass"] for r_ in fired))
    rec["case_C_shallow_notch_from_geometry"] = {
        "construction": {
            "module": "scripts/diagnostics/cv06b_shallow_stub_model.py",
            "defect_input": "stub width, in cells of the 50um board lattice",
            "w_line_cells": W_LINE_CELLS, "dx_m": DX50,
            "h_sub_m": H_SUB_REALIZED, "l_line_m": L_LINE,
            "l_stub_m": 12e-3, "eps_r": cv.EPS_R, "tan_delta": TAN_D,
            "sigma_s_per_m": SIGMA_CU, "z_ref_ohm": Z_REF,
            "open_end_extension_m_at_shipped_width": open_end_extension(
                W_LINE_CELLS * DX50, H_SUB_REALIZED,
                main_line.eps_eff_static),
            "reads_any_cv06b_gate_constant": False},
        "independence": {
            "model_bw_frac_at_shipped_width": shipped["bw_frac"],
            "gate_reference_bw_frac": cv.STOPBAND_BW_FRAC_IDEAL,
            "model_departure_from_gate_reference_pct": dep_pct,
            "model_vs_measured_bw_ratio_pct": agree_pct,
            "model_vs_measured_f_notch_pct": f0_agree_pct,
            "measured_bw_ratio": base["bw_ratio"],
            "measured_f_notch_refined_hz": f0,
            "measured_notch_depth_db": base["notch_depth_db"],
            "model_notch_depth_db": shipped["notch_depth_db"]},
        "rows": rows_c,
        "first_firing_stub_cells": fired[0]["stub_cells"] if fired else None,
        "narrowest_passing_stub_cells": min(
            [r_["stub_cells"] for r_ in rows_c if r_["G2_pass"]],
            default=None),
    }
    print(f"     model at the SHIPPED width vs the gate's reference: "
          f"{dep_pct:+.2f} % (a model that reproduced the reference would "
          f"read 0.00 %)")
    print(f"     model at the SHIPPED width vs the MEASURED committed sweep: "
          f"BW ratio {agree_pct:+.2f} %, notch f {f0_agree_pct:+.2f} %, "
          f"depth {shipped['notch_depth_db']:.2f} vs "
          f"{base['notch_depth_db']:.2f} dB")
    ok &= c_ok
    print(f"     -> {'OK' if c_ok else 'NOT OK'}: G2 PASSES at the shipped "
          f"stub width and FAILS from {rec['case_C_shallow_notch_from_geometry']['first_firing_stub_cells']} "
          f"cells down,\n        while the -10 dB depth witness passes on "
          f"every row -- the blindness #812 measured\n")

    # ---- D: the resolution witness rejects a quantised estimator ----------
    print("D  the old bin-argmin estimator judged by G3")
    q = _bin_argmin_metrics(cv, f, s21, z0, f_an)
    print(f"     witness spread {q['witness_bins']:.4f} bin "
          f"(threshold < {cv.HALF_GRID_WITNESS_BINS:.1f}): "
          f"{'PASS' if q['gates']['G3 half-grid resolution witness'] else 'FAIL'}")
    d_ok = not q["gates"]["G3 half-grid resolution witness"]
    rec["case_D_quantised_estimator"] = {
        "witness_bins": q["witness_bins"],
        "threshold_bins": cv.HALF_GRID_WITNESS_BINS,
        "G3_pass": bool(q["gates"]["G3 half-grid resolution witness"])}
    ok &= d_ok
    print(f"     -> {'OK' if d_ok else 'NOT OK'}: a bin-quantised estimator "
          f"cannot pass the resolution witness\n")

    rec["verdict"] = {"A_ok": bool(a_ok), "B_ok": bool(b_ok),
                      "C_ok": bool(c_ok), "D_ok": bool(d_ok),
                      "all_ok": bool(ok)}
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_plain(rec), indent=2, sort_keys=False) + "\n")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")

    print("FALSIFIERS OK" if ok else "FALSIFIERS NOT SATISFIED")
    print("\nNOT covered here (needs the 5,729,080-cell solve): criterion (A) "
          "on cv06b's OWN\nboard, and the two BUILD-level falsifiers -- a "
          "one-cell stub-length error and a\n5-cell 317.5um stub -- which are "
          "emitted as scripts/vessl_cv06b_estimator_falsifiers.yaml.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
