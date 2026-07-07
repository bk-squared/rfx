#!/usr/bin/env python3
"""Build the cv06b MSL open-stub notch rfx-vs-openEMS E4 comparison summary.

Re-derives the comparison metrics from the two committed result fixtures
(``tests/fixtures/msl_notch_e4/msl_stub_notch_{rfx,openems}_dx50.json``) so the
evidence survives a clean checkout without re-running the ~65 min rfx FDTD. This
is a CHARACTERIZED external cross-check, NOT a tight OpenEMS-class validation:
the notch frequency agrees to ~6% (rfx sits near the fringing-free analytic
3.69 GHz; openEMS lands lower). UPDATE 2026-07-07: a Palace FEM referee (conformal
tets, independent method, matched geometry) lands at ~3.631 GHz at two mesh
densities and SIDES WITH rfx (+0.1%) — openEMS's dx=50 µm notch is the OUTLIER
(~5.9% away), so the split is an openEMS staircase offset, not the "openEMS
captures more open-end fringing" story once assumed. The off-notch |S21|
transmission agrees to ~0.1. See ``msl_stub_notch_palace_referee.json``.

Why dx=50 µm (not cv06b's shipped dx=80 µm): at dx=80 µm the substrate is only
3.175 cells (the "mixed-cell danger zone" rfx preflight warns about), where the
openEMS MSL-port extraction is NON-PHYSICAL (|S11|^2+|S21|^2 up to 8.9, passivity
grossly violated). dx=50 µm gives 5.08 substrate cells where BOTH solvers are
passive, so it is the only valid matched-mesh comparison.

Usage::

    python scripts/diagnostics/build_msl_notch_openems_comparison.py \
        --output-dir tests/fixtures/msl_notch_e4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests/fixtures/msl_notch_e4"
_BAND_LO_GHZ = 2.5
_BAND_HI_GHZ = 6.0


def _energy_sum(rec: dict[str, Any]) -> float:
    s11 = np.asarray(rec["s11_mag"], dtype=float)
    s21 = np.asarray(rec["s21_mag"], dtype=float)
    return float((s11 ** 2 + s21 ** 2).max())


def build_comparison(fixtures_dir: Path) -> dict[str, Any]:
    rfx = json.loads((fixtures_dir / "msl_stub_notch_rfx_dx50.json").read_text())
    oe = json.loads((fixtures_dir / "msl_stub_notch_openems_dx50.json").read_text())

    f_rfx = np.asarray(rfx["freqs_ghz"], dtype=float)
    s21_rfx = np.asarray(rfx["s21_mag"], dtype=float)
    f_oe = np.asarray(oe["freqs_ghz"], dtype=float)
    s21_oe = np.asarray(oe["s21_mag"], dtype=float)

    # Off-notch |S21| magnitude agreement on the openEMS grid.
    rfx_on_oe = np.interp(f_oe, f_rfx, s21_rfx)
    band = (f_oe >= _BAND_LO_GHZ) & (f_oe <= _BAND_HI_GHZ)
    d = np.abs(rfx_on_oe[band] - s21_oe[band])

    fn_rfx = float(rfx["notch"]["f_ghz"])
    fn_oe = float(oe["notch"]["f_ghz"])

    summary = {
        "claim": (
            "cv06b MSL open-stub notch — rfx add_msl_port vs openEMS, matched "
            "geometry dx=50 µm (converged, both passive)"
        ),
        "comparison_kind": "characterized_external_cross_check",
        "rfx": {
            "notch_ghz": round(fn_rfx, 4),
            "notch_depth_db": round(float(rfx["notch"]["depth_db"]), 2),
            "max_energy_sum": round(_energy_sum(rfx), 4),
            "re_z0_median_ohm": rfx.get("re_z0_median_ohm"),
        },
        "openems": {
            "notch_ghz": round(fn_oe, 4),
            "notch_depth_db": round(float(oe["notch"]["depth_db"]), 2),
            "max_energy_sum": round(_energy_sum(oe), 4),
        },
        "comparison": {
            "notch_freq_abs_diff_ghz": round(abs(fn_rfx - fn_oe), 4),
            "notch_freq_rel_pct": round(abs(fn_rfx - fn_oe) / fn_oe * 100.0, 2),
            "s21_mag_max_abs_diff_2p5_6ghz": round(float(d.max()), 4),
            "s21_mag_mean_abs_diff_2p5_6ghz": round(float(d.mean()), 4),
        },
        "verdict": (
            "CHARACTERIZED (not tight OpenEMS-class): off-notch |S21| agrees to "
            "~0.1; notch freq ~6% high (rfx near fringing-free analytic 3.69 GHz, "
            "openEMS captures more open-end fringing → lower). cv06b's shipped "
            "dx=80 µm is non-physical for openEMS (3.175 mixed-cell substrate)."
        ),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=str(_FIXTURES))
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    summary = build_comparison(out_dir)
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    c = summary["comparison"]
    print(
        "notch_rel_pct={} s21_mean_abs_diff={} s21_max_abs_diff={}".format(
            c["notch_freq_rel_pct"],
            c["s21_mag_mean_abs_diff_2p5_6ghz"],
            c["s21_mag_max_abs_diff_2p5_6ghz"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
