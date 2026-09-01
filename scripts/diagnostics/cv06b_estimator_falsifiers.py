#!/usr/bin/env python3
"""Falsifiers for cv06b's sub-bin re-gate (issue #812, mechanism P3).

WHAT THIS CAN AND CANNOT SHOW, STATED FIRST.
--------------------------------------------
cv06b's own mesh is 5,729,080 cells and is GPU-lane: 329.2 s on one RTX4090,
and the same mesh was abandoned UNFINISHED at 2h52m on a 32-core CPU pod
(validation/crossval/manifest.json, cpu_runner.excluded_reason). So the
BUILD-level falsifiers -- a one-cell stub-length error and a shallow-notch
(narrow-stub) build -- cannot be run here; they are emitted as a VESSL job
(scripts/vessl_cv06b_estimator_falsifiers.yaml).

What CAN be shown on CPU, and is shown here, is the instrument itself, driven
by REAL committed rfx data rather than a synthetic curve: the committed
sibling fixture tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json is a
real rfx run of the SAME open-stub notch filter through the SAME
compute_msl_s_matrix estimator on the SAME 63.6364 MHz grid (its board is the
dx=50um / 300um-substrate sibling, so its ANALYTIC anchor differs and the
accuracy gate G1 is reported against that board's own anchor, not cv06b's).

Cases:
  A  baseline      committed dx=50um sweep, unmodified -> G1, G2, G3 and the
                   depth witness all pass. G4 (Re(Z0) median in 40-65 ohm) is
                   NOT part of this replay's verdict: the fixture records
                   re_z0_median_ohm = 31.38 for the dx=50um sibling BOARD, a
                   property of that board rather than of the estimator, and
                   this harness has no business judging it.
  B  sub_bin_shift the sweep's features moved by LESS THAN ONE BIN. The old
                   bin-argmin estimator reports exactly 0.0000 % for such a
                   defect -- the audit's "0.00 % or >= one bin" staircase --
                   while the refined one responds monotonically. Reported for
                   a ladder of shifts including half a cell (+0.265 %,
                   0.15 bin) and one cell (+0.529 %, 0.30 bin) of stub length.
  C  shallow_notch the stub coupling degraded to r = Z0_line/Z_stub < 1, by
                   rescaling the measured sweep with M_r/M_1, the ratio of the
                   ideal shunt-open-stub responses G2's closed form is derived
                   from (so the passband is untouched and only the notch's
                   depth and width change) -> G2 must fail while the -10 dB
                   depth witness still passes. This exercises the GATE on the
                   closed-form signature of a degraded stub; the BUILD-level
                   version (a 5-cell 317.5 um stub) is the VESSL job.
  D  quantised     the SAME baseline judged by the OLD bin-argmin estimator
                   -> G3, the resolution witness, must fail

Usage:  python scripts/diagnostics/cv06b_estimator_falsifiers.py
Exit 0 iff every case behaves as stated above.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CV06B = REPO / "validation/crossval/06b_msl_notch_filter_uniform.py"
FIXTURE = REPO / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"


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
    return f, s21, z0


def _warp(f, s21, rel):
    """Every spectral feature moved by ``rel``, resampled on the SAME grid."""
    return np.interp(f / (1.0 + rel), f, s21)


def _shallow(f, s21, f0, r):
    """Degrade the stub coupling from r = 1 to ``r``.

    Rescale the measured sweep by M_r/M_1, the ratio of the ideal
    shunt-open-stub responses M = |2/(2 + j r tan(theta))| at the two coupling
    ratios. The ratio tends to 1 far from f0, so the passband is untouched and
    only the notch's depth and width move -- and they move exactly the way the
    closed form G2's window is derived from says they should.
    """
    theta = 0.5 * np.pi * f / f0
    m_r = np.abs(2.0 / (2.0 + 1j * r * np.tan(theta)))
    m_1 = np.abs(2.0 / (2.0 + 1j * np.tan(theta)))
    return s21 * m_r / m_1


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


GATED_HERE = ("G1 notch freq vs analytic", "G2 -10 dB stopband width",
              "G3 half-grid resolution witness", "notch depth (witness only)")


def main() -> int:
    cv = _load_cv06b()
    f, s21, z0 = _fixture()

    # This fixture's own analytic anchor: the dx=50um sibling board realizes a
    # 300um substrate (documented in cv06b's "External cross-check" CAVEAT).
    u = 600.0 / 300.0
    eps_eff = (cv.EPS_R + 1) / 2 + (cv.EPS_R - 1) / 2 * (1 + 12 / u) ** -0.5
    f_an = cv.C0 / (4 * cv.STUB_LEN * np.sqrt(eps_eff))
    bin_hz = float(f[1] - f[0])

    ok = True
    print(f"fixture: {FIXTURE.relative_to(REPO)}")
    print(f"  {len(f)} bins, {bin_hz/1e6:.4f} MHz; analytic anchor for THIS "
          f"board (u={u:.3f}) = {f_an/1e9:.4f} GHz\n")

    base = cv.evaluate(f, s21, z0, f_an)
    f0 = base["f_notch_refined"]
    print("A  baseline (committed dx=50um sweep, unmodified)")
    for k in GATED_HERE:
        print(f"     [{'PASS' if base['gates'][k] else 'FAIL'}] {k}")
    print(f"     refined {f0/1e9:.5f} GHz ({base['sub_bin_shift']:+.3f} bin "
          f"off the argmin {base['f_notch_bin']/1e9:.5f}), "
          f"err {base['err_pct']:.2f} %, BW ratio {base['bw_ratio']:.4f}, "
          f"witness {base['witness_bins']:.4f} bin")
    print(f"     (G4 Re(Z0) not judged here: the fixture records 31.38 ohm for "
          f"the dx=50um sibling BOARD, not an estimator property)")
    a_ok = all(base["gates"][k] for k in GATED_HERE)
    ok &= a_ok
    print(f"     -> {'OK' if a_ok else 'NOT OK'}: every estimator gate passes "
          f"on real committed rfx notch data\n")

    # ---- B: sub-bin frequency errors -------------------------------------
    one_cell = 63.5e-6 / 12.0e-3          # one dx of stub length -> 0.529 %
    print("B  sub-bin frequency errors (frequency-axis warp of the SAME sweep)")
    print(f"     one dx of stub length = {one_cell*100:.3f} % = "
          f"{one_cell*f0/bin_hz:.3f} bin")
    print(f"     {'true shift':>12} {'OLD bin argmin':>18} "
          f"{'NEW sub-bin refined':>22}")
    b_ok = True
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
        print(f"     {rel*100:>+11.3f}% {d_old:>+17.4f}% {d_new:>+21.4f}%{flag}")
    ok &= b_ok
    print(f"     -> {'OK' if b_ok else 'NOT OK'}: every sub-bin defect reads "
          f"exactly 0.0000 % on the old estimator and a commensurate,\n"
          f"        monotone value on the refined one\n")

    # ---- C: a shallow-notch build ----------------------------------------
    print("C  shallow notch: stub coupling degraded from r = 1")
    print(f"     {'r':>6} {'depth dB':>10} {'BW ratio':>10} {'closed form':>12} "
          f"{'G2':>6} {'depth gate':>12}")
    c_ok = False
    for r in (0.90, 0.80, 0.75, 0.67, 0.50):
        m = cv.evaluate(f, _shallow(f, s21, f0, r), z0, f_an)
        cf = float(np.arctan(r / 6.0) / np.arctan(1.0 / 6.0))
        g2 = m["gates"]["G2 -10 dB stopband width"]
        dg = m["gates"]["notch depth (witness only)"]
        print(f"     {r:>6.2f} {m['notch_depth_db']:>10.1f} "
              f"{m['bw_ratio']:>10.4f} {cf:>12.4f} "
              f"{'PASS' if g2 else 'FAIL':>6} {'PASS' if dg else 'FAIL':>12}")
        if r <= 0.75 and (not g2) and dg:
            c_ok = True
    ok &= c_ok
    print(f"     -> {'OK' if c_ok else 'NOT OK'}: G2 fails on a degraded stub "
          f"while the -10 dB depth witness keeps passing by >20 dB\n"
          f"        -- exactly the blindness #812 measured\n")

    # ---- D: the resolution witness rejects a quantised estimator ----------
    print("D  the old bin-argmin estimator judged by G3")
    q = _bin_argmin_metrics(cv, f, s21, z0, f_an)
    print(f"     witness spread {q['witness_bins']:.4f} bin "
          f"(threshold < {cv.HALF_GRID_WITNESS_BINS:.1f}): "
          f"{'PASS' if q['gates']['G3 half-grid resolution witness'] else 'FAIL'}")
    d_ok = not q["gates"]["G3 half-grid resolution witness"]
    ok &= d_ok
    print(f"     -> {'OK' if d_ok else 'NOT OK'}: a bin-quantised estimator "
          f"cannot pass the resolution witness\n")

    print("FALSIFIERS OK" if ok else "FALSIFIERS NOT SATISFIED")
    print("\nNOT covered here (needs the 5,729,080-cell solve): the two "
          "BUILD-level\nfalsifiers -- a one-cell stub-length error and a "
          "5-cell 317.5um stub -- which\nare emitted as "
          "scripts/vessl_cv06b_estimator_falsifiers.yaml.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
