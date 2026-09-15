#!/usr/bin/env python3
"""Re-derive the canonical-patch far-field envelope constants (#931, group X-A).

``tests/crossval/test_patch_canonical_farfield_e4.py`` gates the rfx side of the
openEMS Simple_Patch tutorial geometry against the committed openEMS reference
fixture.  Every constant it gates with -- ``D_ABS_TOL_DB``, ``F_RES_REL_LO`` /
``F_RES_REL_HI``, and the mode-pair band -- was measured on the PRE-#931 board,
whose ground was a one-cell PEC Box with its own vacuum cell sitting inside the
modelled cavity (the #693 "+15%p" term).  Under the lattice ownership contract
the foils are declared SHEETS on the laminate's own node planes and that cell is
not in the mesh, so the constants cannot be translated: the case is re-solved and
the envelope re-derived from the new population.  ``_ENVELOPES_REDERIVED_FOR_931``
holds the three slow gates skipped until that happens.

This script IS that re-derivation.  It does not re-implement the measurement: it
imports the test module and calls the module-scoped ``rfx_run`` fixture's own
undecorated function, so the build, the settling witness, the far-field
reduction, the radiating-bin selection and the mode list are byte-for-byte the
ones the gates read.  It writes a JSON record with the measured values, the
committed reference they are compared against, and the arithmetic for each
proposed constant -- so the constant that lands in the test file is a number this
run produced, with its derivation on the record beside it.

The proposed constants follow the file's own existing rules, which are stated in
its comments and reproduced here verbatim in ``rule``:

* ``D_ABS_TOL_DB``: measured |D_rfx - D_openEMS| carried to a round number with
  margin.  The pre-#931 pin was measured 0.60 dB -> locked 1.0 dB, i.e.
  round-UP(measured x 1.5) to one decimal.  Same arithmetic here.  The
  constant is never lowered below the pre-#931 1.0 dB by this script: a
  narrower gate is a separate, argued decision, not a by-product of a rerun.
* ``F_RES_REL_LO`` / ``F_RES_REL_HI``: the measured relative offset bracketed to
  the enclosing whole percent on each side, then widened by one further percent
  of margin -- the shape of the committed [+6%, +16%] around a measured +11.3%
  (floor(11.3)-4 ... ceil(11.3)+4 is not it; the committed band is measured
  +-5 percentage points, so the rule reproduced here is measured +- 5 pp,
  rounded outward to whole percent).
* mode-pair ratio band: the committed [1.15, 1.30] around a measured 1.217, i.e.
  measured +- 0.07 rounded outward to two decimals.

Nothing here edits the test file.  It prints and writes; a human (or the ingest
commit) reads the record and moves the constants, quoting this run.

Usage::

    JAX_PLATFORMS=cpu python scripts/diagnostics/measure_patch_canonical_farfield_e4.py \
        --out canonical_farfield_e4_measured.json

Cost: one FDTD run of the lean frame (dx = 2 mm, num_periods = 110), ~12 min CPU.
That is far past the shared pod's budget, so it runs on VESSL
(``scripts/vessl_931/cv05_farfield_envelope.yaml``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TEST = REPO / "tests/crossval/test_patch_canonical_farfield_e4.py"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_test_module():
    """Import the gate file itself, so the measurement IS the gate's measurement."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("_e4_gates", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _round_up(x: float, decimals: int) -> float:
    scale = 10.0 ** decimals
    return math.ceil(x * scale - 1e-12) / scale


def _floor_pct(x: float) -> float:
    return math.floor(x * 100.0 + 1e-9) / 100.0


def _ceil_pct(x: float) -> float:
    return math.ceil(x * 100.0 - 1e-9) / 100.0


# The gate file's fast, no-FDTD tests. They are the precondition for the
# measurement: an envelope derived from a board that does not realize the
# planes it declares, or checked against a reference fixture whose provenance
# block has been stripped, is not evidence. Called directly rather than through
# pytest — the solver image has no pytest, and a measurement that refuses on a
# missing test runner instead of on physics is a wasted cluster run.
_PRECHECKS = (
    "test_reference_fixture_pins_the_recorded_openems_numbers",
    "test_reference_fixture_provenance_and_reproduce_gate_are_recorded",
    "test_reference_fixture_geometry_matches_this_test",
    "test_the_board_realizes_the_planes_it_declares",
)


def _run_prechecks(mod) -> list[str]:
    """Return the names that failed, with their reason; empty means proceed."""
    failed = []
    for name in _PRECHECKS:
        fn = getattr(mod, name, None)
        if fn is None:
            failed.append(f"{name}: MISSING from the gate file")
            continue
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 — reported, not raised
            first = str(exc).strip().splitlines()
            failed.append(f"{name}: {first[0] if first else type(exc).__name__}")
        else:
            print(f"  precheck OK  {name}")
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    mod = _load_test_module()
    ref = json.loads(mod._FIXTURE.read_text(encoding="utf-8"))

    print("=== fast gates (no FDTD) — the board and the reference fixture ===")
    bad = _run_prechecks(mod)
    if bad:
        print("REFUSING TO MEASURE — the gate file's own fast tests fail:")
        for line in bad:
            print(f"  ! {line}")
        return 3

    print("=== canonical patch, sheet-declared board — one FDTD run ===")
    run = mod.rfx_run.__wrapped__()          # the gate fixture's own function

    k = int(run["k_star"])
    if k < 0:
        print("NO BROADSIDE BIN — the radiating mode was not identified; "
              "no envelope can be derived from this run.")
        return 2

    d_rfx = float(run["d_dbi"][k])
    d_ref = float(ref["directivity_dbi"])
    d_diff = abs(d_rfx - d_ref)

    f_rfx = float(run["f_radiating_hz"])
    f_ref = float(ref["f_res_ghz"]) * 1e9
    f_rel = (f_rfx - f_ref) / f_ref

    pair = sorted(m.freq for m in run["modes"] if 1.8e9 < m.freq < 3.1e9)
    ratio = (pair[-1] / pair[0]) if len(pair) >= 2 else float("nan")

    # ---- the proposed constants, each with its arithmetic on the record ----
    d_tol_raw = _round_up(d_diff * 1.5, 1)
    d_tol = max(d_tol_raw, mod.D_ABS_TOL_DB)   # never narrower than the old pin
    lo = _floor_pct(f_rel - 0.05)
    hi = _ceil_pct(f_rel + 0.05)
    ratio_lo = (math.floor((ratio - 0.07) * 100) / 100) if ratio == ratio else None
    ratio_hi = (math.ceil((ratio + 0.07) * 100) / 100) if ratio == ratio else None

    record = {
        "_what": (
            "Re-derivation of tests/crossval/test_patch_canonical_farfield_e4.py's "
            "gated envelope constants on the #931 sheet-declared canonical patch. "
            "Measured by calling that file's own rfx_run fixture function, so the "
            "build and the reduction are the gate's, not a copy."
        ),
        "provenance": {
            "repo_commit": _git("rev-parse", "HEAD"),
            "repo_dirty": bool(_git("status", "--porcelain")),
            "recorded_utc": _dt.datetime.now(_dt.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "producer": "scripts/diagnostics/measure_patch_canonical_farfield_e4.py",
            "gate_file": "tests/crossval/test_patch_canonical_farfield_e4.py",
            "frame": {"dx_m": mod.DX, "n_sub": mod.N_SUB,
                      "num_periods": mod.NUM_PERIODS,
                      "ntff_freqs_hz": [float(f) for f in mod.NTFF_FREQS]},
        },
        "reference": {
            "source": "tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json",
            "f_res_hz": f_ref,
            "directivity_dbi": d_ref,
        },
        "measured": {
            "settling_end_db": float(run["end_db"]),
            "settling_bar_db": float(mod.SETTLING_BAR_DB),
            "settling_clears_bar": bool(run["end_db"] < mod.SETTLING_BAR_DB),
            "wall_s": float(run["wall_s"]),
            "k_star": k,
            "f_radiating_hz": f_rfx,
            "q_radiating": float(run["q_radiating"]),
            "directivity_dbi": d_rfx,
            "d_abs_diff_db": d_diff,
            "f_rel_vs_reference": f_rel,
            "mode_pair_ghz": [f / 1e9 for f in pair],
            "mode_pair_ratio": ratio,
            "beam_peak_theta_deg": [float(t) for t in run["peak_theta_deg"]],
            "cuts_deg": run["cuts"],
            "p_rel_db": [float(v) for v in run["p_rel_db"]],
            "d_dbi_per_bin": [float(v) for v in run["d_dbi"]],
            "modes": [{"freq_hz": float(m.freq), "Q": float(m.Q),
                       "amplitude": float(m.amplitude)} for m in run["modes"]],
            "preflight_advisories": run["advisories"],
        },
        "proposed_constants": {
            "D_ABS_TOL_DB": {
                "old": mod.D_ABS_TOL_DB,
                "new": d_tol,
                "rule": "round-UP(measured x 1.5, 1 decimal), floored at the "
                        "pre-#931 pin so a rerun can never narrow the gate",
                "arithmetic": f"|{d_rfx:.4f} - {d_ref:.4f}| = {d_diff:.4f} dB; "
                              f"ceil({d_diff:.4f} x 1.5, .1) = {d_tol_raw:.1f}; "
                              f"max({d_tol_raw:.1f}, {mod.D_ABS_TOL_DB}) = {d_tol:.1f}",
            },
            "F_RES_REL_LO": {
                "old": mod.F_RES_REL_LO,
                "new": lo,
                "rule": "measured - 5 percentage points, rounded outward to whole percent",
                "arithmetic": f"measured {f_rel * 100:+.2f}%; "
                              f"floor({f_rel * 100:+.2f} - 5) = {lo * 100:+.0f}%",
            },
            "F_RES_REL_HI": {
                "old": mod.F_RES_REL_HI,
                "new": hi,
                "rule": "measured + 5 percentage points, rounded outward to whole percent",
                "arithmetic": f"measured {f_rel * 100:+.2f}%; "
                              f"ceil({f_rel * 100:+.2f} + 5) = {hi * 100:+.0f}%",
            },
            "mode_pair_ratio_band": {
                "old": [1.15, 1.30],
                "new": [ratio_lo, ratio_hi],
                "rule": "measured +- 0.07, rounded outward to two decimals",
                "arithmetic": f"measured {ratio:.4f}; "
                              f"[{ratio_lo}, {ratio_hi}]" if ratio == ratio
                              else "mode pair NOT resolved",
            },
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")

    print()
    print(f"settling      {run['end_db']:.1f} dB (bar {mod.SETTLING_BAR_DB}) "
          f"-> {'CLEARS' if run['end_db'] < mod.SETTLING_BAR_DB else 'UNDER-SETTLED'}")
    print(f"radiating bin k={k} at {mod.NTFF_FREQS[k] / 1e9:.1f} GHz")
    print(f"D             {d_rfx:.2f} dBi vs openEMS {d_ref:.2f} -> |diff| {d_diff:.2f} dB")
    print(f"f_radiating   {f_rfx / 1e9:.4f} GHz vs openEMS {f_ref / 1e9:.4f} "
          f"-> {f_rel * 100:+.2f}%")
    print(f"mode pair     {[round(f / 1e9, 4) for f in pair]} GHz, ratio {ratio:.4f}")
    print()
    for name, blk in record["proposed_constants"].items():
        print(f"{name}: {blk['old']} -> {blk['new']}   [{blk['arithmetic']}]")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
