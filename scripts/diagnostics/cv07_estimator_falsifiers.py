#!/usr/bin/env python3
"""Falsifiers for cv07's sub-bin re-gate (issue #812, mechanism P3).

Criterion (B) of the #812 re-gate contract: a new gate is only worth anything
if it FAILS on the specific defect the audit measured the old gate blind to,
and fails for the right reason. This script builds those defect legs
deterministically from the committed ones, runs `07_sheen_lpf.py compare`
against each in an isolated tree (the committed legs are never touched), and
reports the gate-by-gate verdict.

Two defects:

  erased_zero
      The LOWER doublet member is filled in: 20log10|S21| between the 6.3992
      GHz shoulder bin and the 7.8739 GHz deep zero is replaced by a straight
      line in dB vs f through those two bins, which are themselves untouched.
      Nothing else in the leg changes. The audit measured 17/17 PASS on this.
      EXPECTED NOW: C5 (zero count 1 != 2) and C4 (lower zero) fail.

  corner_m20
      A -20 % corner-frequency error, ISOLATED. A monotone piecewise-linear
      warp of the frequency axis moves the -3 dB corner 5.5036 -> 4.4029 GHz
      while leaving f <= 3.0 GHz (the gated passband) and f >= 6.3992 GHz
      (both transmission zeros) exactly where they were.
      EXPECTED NOW: C6 fails and NOTHING ELSE does -- which is the proof that
      the corner was an ungated quantity and now is not.

      NOT the audit's own leg. The audit reported a -20 % corner defect that
      "passes all 17 script gates" with "two spurious zeros appearing"; a
      naive global f -> f/0.8 compression of the committed leg does NOT
      reproduce that (it drags the argmin to 6.3992 GHz, which fails today's
      C1 at 18.7 %), so the audit's construction is not recoverable from what
      it published. The warp above is this lane's own construction and is
      documented as such: it reproduces the audit's stated PROPERTY (a -20 %
      corner that every pre-existing gate is blind to) rather than its
      unpublished leg.

Every number this script prints, and every window derivation the design note
cites for cv07, is also written to
tests/fixtures/cv07_estimator_regate/cv07_estimator_falsifiers.json -- the
committed record prose must reference by key rather than restate (#812
round-2 numeric-provenance discipline).

Usage:  python scripts/diagnostics/cv07_estimator_falsifiers.py [--keep]
                                                               [--out-json P]
Exit 0 iff every falsifier failed exactly the gates it was built to fail.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CV07 = REPO / "validation/crossval/07_sheen_lpf.py"
RES = REPO / "validation/crossval/_07_sheen_results"
COMPARATORS = REPO / "validation/crossval/comparators/spectral_features.py"
REFEREE = REPO / "tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json"
SPECTRAL = REPO / "validation/crossval/comparators/spectral_features.py"
OUT_JSON = REPO / "tests/fixtures/cv07_estimator_regate/cv07_estimator_falsifiers.json"

# Geometry the T4/T6 windows are derived from, read from the case rather than
# retyped: one dx cell on the wide patch's transverse extent.
DX_M = 200e-6
PATCH_TRV_M = 20.320e-3
UPPER_WIN_GHZ = (7.5, 8.6)

ANCHOR_LO_GHZ = 6.3992     # shoulder bin, untouched
ANCHOR_HI_GHZ = 7.8739     # deep-zero bin, untouched
FC_COMMITTED_GHZ = 5.5036
FC_DEFECT_GHZ = FC_COMMITTED_GHZ * 0.80


def erased_zero(leg: dict) -> dict:
    d = copy.deepcopy(leg)
    f = np.asarray(d["freqs_hz"], dtype=float) / 1e9
    s = np.asarray(d["s21_mag"], dtype=float)
    y = 20.0 * np.log10(s)
    ia = int(np.argmin(np.abs(f - ANCHOR_LO_GHZ)))
    ib = int(np.argmin(np.abs(f - ANCHOR_HI_GHZ)))
    for k in range(ia + 1, ib):
        t = (f[k] - f[ia]) / (f[ib] - f[ia])
        y[k] = y[ia] + t * (y[ib] - y[ia])
    s2 = 10.0 ** (y / 20.0)
    d["s21_mag"] = s2.tolist()
    d["energy_sum"] = (np.asarray(d["s11_mag"]) ** 2 + s2 ** 2).tolist()
    return d


def corner_m20(leg: dict) -> dict:
    """S_defect(f) = S_committed(g(f)), g monotone piecewise linear with
    g(f)=f outside [3.0, ANCHOR_LO] GHz and g(FC_DEFECT) = FC_COMMITTED."""
    d = copy.deepcopy(leg)
    f = np.asarray(d["freqs_hz"], dtype=float) / 1e9
    knots_x = [f[0], 3.0, FC_DEFECT_GHZ, ANCHOR_LO_GHZ, f[-1]]
    knots_y = [f[0], 3.0, FC_COMMITTED_GHZ, ANCHOR_LO_GHZ, f[-1]]
    assert all(b > a for a, b in zip(knots_x, knots_x[1:])), knots_x
    assert all(b > a for a, b in zip(knots_y, knots_y[1:])), knots_y
    g = np.interp(f, knots_x, knots_y)
    for key in ("s11_mag", "s21_mag", "re_z0", "passivity_correction"):
        if key in d:
            d[key] = np.interp(g, f, np.asarray(d[key], dtype=float)).tolist()
    d["energy_sum"] = (np.asarray(d["s11_mag"]) ** 2
                       + np.asarray(d["s21_mag"]) ** 2).tolist()
    return d


DEFECTS = {
    "erased_zero": (erased_zero, {"C5 rfx transmission-zero count",
                                  "C4 rfx lower zero, sub-bin refined"}),
    "corner_m20": (corner_m20, {"C6 rfx -3 dB corner frequency"}),
}

GATE_RE = re.compile(r"^\s*\[(PASS|FAIL)\]\s+(.*?):", re.M)


def run_compare(leg: dict, workdir: Path) -> tuple[int, dict[str, str], str]:
    cv = workdir / "validation/crossval"
    (cv / "comparators").mkdir(parents=True, exist_ok=True)
    (cv / "_07_sheen_results").mkdir(parents=True, exist_ok=True)
    (workdir / "tests/fixtures/sheen_lpf_e4").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CV07, cv / CV07.name)
    shutil.copy2(COMPARATORS, cv / "comparators" / COMPARATORS.name)
    shutil.copy2(REFEREE, workdir / "tests/fixtures/sheen_lpf_e4" / REFEREE.name)
    shutil.copy2(RES / "openems.json", cv / "_07_sheen_results/openems.json")
    (cv / "_07_sheen_results/rfx.json").write_text(json.dumps(leg, indent=2))
    proc = subprocess.run([sys.executable, str(cv / CV07.name), "compare"],
                          capture_output=True, text=True)
    verdicts = {name: st for st, name in GATE_RE.findall(proc.stdout)}
    return proc.returncode, verdicts, proc.stdout


def _window_derivations(leg: dict) -> dict:
    """Re-derive every quantity the design note's cv07 windows rest on.

    Nothing here judges anything; it exists so the note can cite a key instead
    of a digit, and so a changed leg or estimator moves the cited number.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sf", SPECTRAL)
    sf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sf)

    f = np.asarray(leg["freqs_hz"], dtype=float) / 1e9
    s = np.asarray(leg["s21_mag"], dtype=float)
    bin_ghz = float(f[1] - f[0])
    base = sf.refined_extremum(f, s, *UPPER_WIN_GHZ)
    warp = {}
    for rel in (0.01, -0.01):
        w = np.interp(f / (1.0 + rel), f, s)
        r = sf.refined_extremum(f, w, UPPER_WIN_GHZ[0] * (1 + rel),
                                UPPER_WIN_GHZ[1] * (1 + rel))
        warp[f"{rel*100:+.3f}pct"] = (
            (r["refined_f"] - base["refined_f"]) / base["refined_f"] * 100.0)
    one_cell_pct = DX_M / PATCH_TRV_M * 100.0
    return {
        "sweep_bin_mhz": bin_ghz * 1e3,
        "sweep_bin_pct_at_bin_argmin": bin_ghz / base["bin_f"] * 100.0,
        "sweep_bin_pct_at_refined_zero": bin_ghz / base["refined_f"] * 100.0,
        "upper_zero_bin_argmin_ghz": base["bin_f"],
        "upper_zero_refined_ghz": base["refined_f"],
        "one_cell_transverse_pct": one_cell_pct,
        "estimator_response_to_frequency_warp_pct": warp,
        # fc ~ 1/sqrt(LC), C ~ patch area ~ transverse extent
        "one_cell_corner_shift_pct": 0.5 * one_cell_pct,
        "prominence_0p5_db_in_amplitude_pct": (10 ** (0.5 / 20) - 1) * 100.0,
        "prominence_0p5_db_in_power_pct": (10 ** (0.5 / 10) - 1) * 100.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true",
                    help="keep the isolated work trees for inspection")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    args = ap.parse_args()

    committed = json.loads((RES / "rfx.json").read_text())
    ok = True
    rec = {"meta": {"issue": 812, "mechanism": "P3 estimator quantization",
                    "case": "cv07",
                    "produced_by":
                        "scripts/diagnostics/cv07_estimator_falsifiers.py",
                    "driving_leg": str(RES.relative_to(REPO) / "rfx.json")},
           "window_derivations": _window_derivations(committed),
           "defects": {}}

    tmp = Path(tempfile.mkdtemp(prefix="cv07_falsifiers_"))
    try:
        rc0, v0, _ = run_compare(committed, tmp / "baseline")
        n = len(v0)
        fails0 = sorted(k for k, s in v0.items() if s == "FAIL")
        print(f"[baseline]   exit {rc0}   {n} gates   "
              f"{'ALL PASS' if not fails0 else 'FAIL: ' + str(fails0)}")
        ok &= (rc0 == 0 and not fails0)
        rec["baseline"] = {"exit": rc0, "n_gates": n,
                           "failed": fails0, "all_pass": not fails0,
                           "verdicts": v0}

        for name, (build, expect) in DEFECTS.items():
            rc, v, out = run_compare(build(committed), tmp / name)
            fails = {k for k, s in v.items() if s == "FAIL"}
            print(f"\n[{name}]  exit {rc}   {len(v)} gates")
            for k in sorted(fails):
                line = next(ln for ln in out.splitlines()
                            if ln.strip().startswith("[FAIL]") and k in ln)
                print("   " + line.strip())
            missing = expect - fails
            extra = fails - expect
            good = rc == 1 and not missing and not extra
            ok &= good
            if missing:
                print(f"   !! expected-but-passed: {sorted(missing)}")
            if extra:
                print(f"   !! unexpected failures: {sorted(extra)}")
            print(f"   -> {'OK' if good else 'NOT OK'}: "
                  f"failed exactly the {len(expect)} gate(s) it was built to "
                  f"fail; the other {len(v) - len(fails)} still pass")
            rec["defects"][name] = {
                "exit": rc, "n_gates": len(v),
                "expected_failures": sorted(expect),
                "observed_failures": sorted(fails),
                "unexpected_failures": sorted(extra),
                "expected_but_passed": sorted(missing),
                "n_still_passing": len(v) - len(fails),
                "as_designed": bool(good),
                "failure_lines": [ln.strip() for ln in out.splitlines()
                                  if ln.strip().startswith("[FAIL]")]}
    finally:
        if args.keep:
            print(f"\nwork trees kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    rec["verdict"] = {"all_ok": bool(ok)}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nwrote {out_path}")

    print("\n" + ("FALSIFIERS OK — every new gate fails on the defect it was "
                  "added for, and only on that defect."
                  if ok else "FALSIFIERS NOT SATISFIED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
