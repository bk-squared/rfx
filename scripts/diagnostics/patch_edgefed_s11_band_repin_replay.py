"""Falsifier F1 replay for the #782 S11-gate re-pin — RETIRED BY #931, no FDTD.

DISCHARGED BY #931 (2026-09-07). This script refuses to run against the
committed gate constants, and the refusal is the point: a dated record is not
a gate, and the live discrimination evidence is the re-pin runs (VESSL
369367259225 Board H / 369367259226 Board S), not this file.

WHAT IT WAS. It loaded the two saved arms from
``docs/design_notes/patch_edgefed_s11_band_repin_results.json`` (written by
``patch_edgefed_s11_band_repin.py``) and evaluates the COMMITTED gate's own
``_gate_readings`` + assertion conditions (imported from
``tests/locks/test_patch_edgefed_s11_passivity.py``, not re-implemented) on each
arm:

  * main arm    -> every gate condition must PASS;
  * retired arm -> the in-band-crossing witness (2b) and/or the antiresonance
                   Re(Zin) floor (2c) must FAIL — the gate discriminates the
                   bit-exact pre-#702 physics.

Both arms were measured on a board that RESERVED a cell for each foil, which
rfx then re-sampled to laminate (#702), so its electrical cavity was 983.75 um
against a declared 787. The lattice ownership contract deletes that re-sample,
the gate's board is redrawn with each foil ON the laminate face it bounds, and
``RES_BAND_GHZ`` moved (8.4, 9.2) -> (7.4, 8.2) with the antiresonance it
brackets: crossing 8.8189 -> 7.7620 GHz (VESSL 369367259226, confirmed
369367259239).

So evaluating the saved arms against the imported constants now compares two
different boards, and it would report "F1 NOT SATISFIED" — a sentence about
arithmetic, not about physics. Freezing the old constants inside this script
instead would keep it green and keep it meaningless: the mechanism F1
discriminated (the #702 own-cell re-sample) does not exist to be discriminated
any more. The discharge is recorded in
``docs/design_notes/issue782_retired_resonance_predeclaration.md`` Section 4.

The evidence JSONs stay committed as dated evidence for the pre-#931 board, and
this script stays as the reader that knows how to open them: pass
``--historical-band`` to replay F1 against the constants it was written for,
which is the only reading of it that means anything.

Exit 0 = F1 satisfied (historical replay), 1 = not, 2 = refused.
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


def evaluate(arm: dict, band: tuple) -> dict:
    fr = np.asarray(arm["freqs_ghz"], dtype=float)
    s = np.asarray(arm["s11_re"], dtype=float) + 1j * np.asarray(arm["s11_im"], dtype=float)
    z0 = np.asarray(arm["z0_re"], dtype=float) + 1j * np.asarray(arm["z0_im"], dtype=float)
    g = _gate_readings(fr, s, z0)
    # _gate_readings brackets with the CURRENT RES_BAND_GHZ; the band-scoped
    # readings are re-taken here against the band this replay was scored with,
    # so nothing silently mixes the two boards' constants.
    zin = z0 * (1.0 + s) / (1.0 - s)
    in_band = (fr >= band[0]) & (fr <= band[1])
    g = dict(g)
    g["band_min_s11"] = float(np.min(np.abs(s)[in_band]))
    g["band_max_re_zin"] = float(np.max(zin.real[in_band]))
    g["band_crossings_ghz"] = [c for c in g["crossings_ghz"]
                               if band[0] <= c <= band[1]]
    return dict(
        passivity=g["max_s11"] <= PASSIVE_TOL,
        band_floor=g["band_min_s11"] > RES_BAND_S11_MIN,
        band_crossing=bool(g["band_crossings_ghz"]),
        band_re_zin=g["band_max_re_zin"] > RES_BAND_RE_ZIN_MIN_OHM,
        dip_above_band=g["f_dip_ghz"] > band[1],
        readings=dict(
            max_s11=round(g["max_s11"], 4),
            band_min_s11=round(g["band_min_s11"], 4),
            band_max_re_zin=round(g["band_max_re_zin"], 1),
            crossings_ghz=[round(c, 4) for c in g["crossings_ghz"]],
            band_crossings_ghz=[round(c, 4) for c in g["band_crossings_ghz"]],
            f_dip_ghz=round(g["f_dip_ghz"], 4),
        ),
    )


# The band F1 was scored against, frozen here as HISTORY: it is the value
# RES_BAND_GHZ held from the #782 re-pin until #931 redrew the board. It is not
# a gate constant and nothing imports it — it exists so the historical replay
# reads the saved arms with the constants they were measured under.
HISTORICAL_RES_BAND_GHZ = (8.4, 9.2)


def main() -> int:
    historical = "--historical-band" in sys.argv[1:]
    if not historical:
        print("[F1] REFUSED. This replay is DISCHARGED by #931 — see the module "
              "docstring. The saved arms are the pre-#931 board (reserved foil "
              "cells, 983.75 um electrical cavity); the imported gate constants "
              f"are the redrawn board's (RES_BAND_GHZ = {RES_BAND_GHZ}, "
              f"was {HISTORICAL_RES_BAND_GHZ}). Scoring one against the other "
              "compares two boards. Re-run with --historical-band to replay F1 "
              "as it was scored, or read the discharge in "
              "docs/design_notes/issue782_retired_resonance_predeclaration.md "
              "Section 4.")
        return 2

    band = HISTORICAL_RES_BAND_GHZ
    path = os.path.join(_REPO, "docs", "design_notes",
                        "patch_edgefed_s11_band_repin_results.json")
    with open(path) as f:
        results = json.load(f)
    print(f"[F1] HISTORICAL replay (band {band} GHz, the #782 value)")
    print(f"[F1] evidence: {path}\n[F1] measured on tree {results['git_head']}\n"
          f"[F1] gate constants: band {band} GHz, floor {RES_BAND_S11_MIN}, "
          f"Re(Zin) > {RES_BAND_RE_ZIN_MIN_OHM} ohm, passivity {PASSIVE_TOL}")

    verdicts = {}
    for tag in ("main", "retired"):
        v = evaluate(results[tag], band)
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
