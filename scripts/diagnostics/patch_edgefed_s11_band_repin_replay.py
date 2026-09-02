"""Falsifier F1 replay for the #782 S11-gate re-pin — no FDTD.

Loads the two saved arms from ``docs/design_notes/patch_edgefed_s11_band_repin_results.json``
(written by ``patch_edgefed_s11_band_repin.py``) and evaluates the COMMITTED gate's own
``_gate_readings`` + assertion conditions (imported from
``tests/locks/test_patch_edgefed_s11_passivity.py``, not re-implemented) on each arm:

  * main arm    -> every gate condition must PASS;
  * retired arm -> the in-band-crossing witness (2b) and/or the antiresonance
                   Re(Zin) floor (2c) must FAIL — the gate discriminates the
                   bit-exact pre-#702 physics.

Exit 0 = F1 satisfied, 1 = not.
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import numpy as np  # noqa: E402

from tests.locks.test_patch_edgefed_s11_passivity import (  # noqa: E402
    PASSIVE_TOL, RES_BAND_GHZ, RES_BAND_RE_ZIN_MIN_OHM, RES_BAND_S11_MIN,
    _gate_readings,
)


def evaluate(arm: dict) -> dict:
    fr = np.asarray(arm["freqs_ghz"], dtype=float)
    s = np.asarray(arm["s11_re"], dtype=float) + 1j * np.asarray(arm["s11_im"], dtype=float)
    z0 = np.asarray(arm["z0_re"], dtype=float) + 1j * np.asarray(arm["z0_im"], dtype=float)
    g = _gate_readings(fr, s, z0)
    return dict(
        passivity=g["max_s11"] <= PASSIVE_TOL,
        band_floor=g["band_min_s11"] > RES_BAND_S11_MIN,
        band_crossing=bool(g["band_crossings_ghz"]),
        band_re_zin=g["band_max_re_zin"] > RES_BAND_RE_ZIN_MIN_OHM,
        dip_above_band=g["f_dip_ghz"] > RES_BAND_GHZ[1],
        readings=dict(
            max_s11=round(g["max_s11"], 4),
            band_min_s11=round(g["band_min_s11"], 4),
            band_max_re_zin=round(g["band_max_re_zin"], 1),
            crossings_ghz=[round(c, 4) for c in g["crossings_ghz"]],
            band_crossings_ghz=[round(c, 4) for c in g["band_crossings_ghz"]],
            f_dip_ghz=round(g["f_dip_ghz"], 4),
        ),
    )


def main() -> int:
    path = os.path.join(_REPO, "docs", "design_notes",
                        "patch_edgefed_s11_band_repin_results.json")
    with open(path) as f:
        results = json.load(f)
    print(f"[F1] evidence: {path}\n[F1] measured on tree {results['git_head']}\n"
          f"[F1] gate constants: band {RES_BAND_GHZ} GHz, floor {RES_BAND_S11_MIN}, "
          f"Re(Zin) > {RES_BAND_RE_ZIN_MIN_OHM} ohm, passivity {PASSIVE_TOL}")

    verdicts = {}
    for tag in ("main", "retired"):
        v = evaluate(results[tag])
        verdicts[tag] = v
        print(f"\n[F1] arm {tag} (bypass_resample={results[tag]['bypass_resample']}):")
        for k in ("passivity", "band_floor", "band_crossing", "band_re_zin",
                  "dip_above_band"):
            print(f"    {k:15s} {'PASS' if v[k] else 'FAIL'}")
        print(f"    readings: {v['readings']}")

    main_ok = all(v for k, v in verdicts["main"].items() if k != "readings")
    retired_red = (not verdicts["retired"]["band_crossing"]
                   or not verdicts["retired"]["band_re_zin"])
    print(f"\n[F1] main arm all-PASS: {main_ok}")
    print(f"[F1] retired arm goes RED on the discriminating assertions: {retired_red}")
    ok = main_ok and retired_red
    print(f"[F1] VERDICT: {'SATISFIED' if ok else 'NOT SATISFIED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
