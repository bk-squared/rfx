#!/usr/bin/env python3
"""Where the coaxial open end goes non-passive: the rung and board for an always-on check.

An open-circuited lossless coaxial line reflects what it receives and no more,
|Gamma| <= 1, and its reflection does not change when the record is lengthened
once the line has rung down. The one-port lane as shipped before issue 1218's
fix absorbed on z only and left the line in a closed PEC can; its open end read
|Gamma| above 1 and moved with the record at 9 annulus cells (1.018 at 12 line
traversals, 1.046 at 24). The fix absorbs on all three axes. An always-on test
of the fix needs the cheapest mesh and board on which the shipped lane FAILS
the check the fix has to pass:

    max |Gamma| <= 1.02 at 12 and at 24 traversals, and
    max over bins of ||Gamma_24| - |Gamma_12|| < 0.0259 (the battery's doubling bound).

This driver solves the lane's open termination, on whichever lane the checkout
carries, at one annulus-cell rung and one board length, at 12 and 24 record
units, and writes the per-bin S with the check's numbers beside it. Run it at a
commit whose ``rfx/`` still carries the shipped lane and at the fix's commit;
the record names which (``absorbing_axes``, read from the lane's own default).
Board, cross-section, drive and record unit are the coax chain battery's
(``coax_chain_battery_measure.py``, imported); only the z length varies.

    PYTHONPATH=. python scripts/diagnostics/coax_open_check_config.py \\
        --rung 4 --z-mm 40 --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/coax_open_check_config.py \\
        --summarize <dir with every record> [--run-index <json>]
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import coax_chain_battery_measure as battery  # noqa: E402  (puts the repo on sys.path)

import jax  # noqa: E402

from rfx.api import Simulation  # noqa: E402

SCHEMA = "rfx.coax_open_check_config"
DRIVER = "scripts/diagnostics/coax_open_check_config.py"
RECORD_UNITS = (12.0, 24.0)
MAX_ABS_GAMMA = 1.02
DOUBLING_SHIFT_MAX = (10 ** (battery.BAR["magnitude_db"] / 20.0) - 1.0) / 10.0   # 0.025893
LATERAL_M = battery.DOMAIN_ONEPORT[0]


def record_name(rung: int, z_mm: float) -> str:
    return f"open_check_rung{rung}_z{z_mm:g}mm.json"


def lane_absorbing_axes() -> str:
    return str(inspect.signature(Simulation.compute_coaxial_line_reflection)
               .parameters["cpml_axes"].default)


def summary(res_record: dict) -> dict:
    g = battery._S(res_record["S11"])
    f = np.asarray(res_record["freqs_hz"], dtype=float)
    mag = np.abs(g)
    k = int(np.argmax(mag))
    j = int(np.argmin(mag))
    return {"max_abs_gamma": float(mag[k]), "argmax_hz": float(f[k]),
            "min_abs_gamma": float(mag[j]), "argmin_hz": float(f[j]),
            "n_bins_above_1": int(np.sum(mag > 1.0)), "all_finite": bool(np.all(np.isfinite(g))),
            "status": res_record["status"]}


def stage(args, out: Path) -> None:
    rung, z_mm = int(args.rung), float(args.z_mm)
    domain = (LATERAL_M, LATERAL_M, z_mm * 1e-3)
    rec = battery._base(args, "open_check_config", "open", rung)
    rec.update({"schema": SCHEMA, "driver": DRIVER, "reuses": battery.DRIVER,
                "board_m": list(domain), "absorbing_axes": lane_absorbing_axes(),
                "device_kind": [d.device_kind for d in jax.devices()], "records": {}})
    for units in RECORD_UNITS:
        sim = battery.build_sim(rung, "open", domain=domain)
        geo = battery.assert_realized(sim, rung, "open")
        grid = sim._build_grid()
        n_steps = battery.record_steps(grid, units)
        est = battery.cost_estimate(grid, n_steps, battery.N_FREQS, 2 * battery.PROBE_COUNT, 1)
        battery._log(f"open check rung={rung} z={z_mm:g} mm units={units:g} n_steps={n_steps} "
                     f"lane absorbs on '{rec['absorbing_axes']}'")
        with battery._Captured() as cap:
            res = battery.solve_one_port(sim, "open", n_steps=n_steps)
        battery._log_one_port(f"open check r{rung} z{z_mm:g} u{units:g}", res)
        result = battery._one_port_record(res)
        rec["records"][f"u{units:g}"] = {
            "record_units": units, "n_steps": n_steps, "realized": geo, "cost": est,
            "cross_check": battery.cross_check_result_against_layout(geo, res, "one_port"),
            "result": result, "summary": summary(result),
            "warnings": cap.warnings, "wall_s": cap.wall,
        }
        battery._write(out, rec)
    a = battery._S(rec["records"]["u12"]["result"]["S11"])
    b = battery._S(rec["records"]["u24"]["result"]["S11"])
    shift = float(np.max(np.abs(np.abs(b) - np.abs(a))))
    worst = max(rec["records"][u]["summary"]["max_abs_gamma"] for u in ("u12", "u24"))
    rec["check"] = {
        "max_abs_gamma_12": rec["records"]["u12"]["summary"]["max_abs_gamma"],
        "max_abs_gamma_24": rec["records"]["u24"]["summary"]["max_abs_gamma"],
        "shift_12_24_per_bin": shift,
        "max_abs_gamma_bound": MAX_ABS_GAMMA, "shift_bound": DOUBLING_SHIFT_MAX,
        "passes": bool(worst <= MAX_ABS_GAMMA and shift < DOUBLING_SHIFT_MAX),
        "cell_steps": int(sum(r["cost"]["cell_steps"] for r in rec["records"].values())),
        "wall_s": float(sum(r["wall_s"] for r in rec["records"].values())),
    }
    rec["peak_memory"] = battery.peak_memory()
    battery._write(out, rec)
    battery._log(f"check r{rung} z{z_mm:g}: {json.dumps(rec['check'])}")


def _rows(directory: Path) -> list[dict]:
    index_path = directory / "run_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    rows = []
    for p in sorted(directory.glob("open_check_rung*_z*mm.json")):
        rec = json.loads(p.read_text())
        if "check" not in rec:
            continue
        rows.append({
            "file": p.name,
            "provenance": battery.fixture_provenance(rec["provenance"], index, p.name),
            "absorbing_axes": rec["absorbing_axes"],
            "rung_annulus_cells": rec["rung_annulus_cells"],
            "board_m": rec["board_m"], "device_kind": rec["device_kind"],
            **rec["check"],
            "records": {u: {"record_units": r["record_units"], "n_steps": r["n_steps"],
                            "S11": r["result"]["S11"], "summary": r["summary"],
                            "status": r["result"]["status"], "wall_s": r["wall_s"]}
                        for u, r in rec["records"].items()},
        })
    return rows


def summarize(args) -> int:
    """Print one row per record, or — with ``--new`` and ``--artifact-out`` —
    write the committed record of the sweep: the shipped lane's records (the
    ``--summarize`` directory) beside the fixed lane's (``--new``), board by
    board. Arithmetic only."""
    old = _rows(Path(args.summarize))
    if not args.new:
        print(json.dumps([{k: v for k, v in r.items() if k != "records"} for r in old],
                         indent=1))
        return 0
    new = _rows(Path(args.new))
    boards = sorted({(r["rung_annulus_cells"], r["board_m"][2]) for r in old + new})
    table = []
    for rung, z in boards:
        pick = {lane: next((r for r in rows if r["rung_annulus_cells"] == rung
                            and r["board_m"][2] == z), None)
                for lane, rows in (("shipped", old), ("fixed", new))}
        table.append({"rung_annulus_cells": rung, "board_z_m": z,
                      **{f"{lane}_{k}": (None if r is None else r[k])
                         for lane, r in pick.items()
                         for k in ("absorbing_axes", "max_abs_gamma_12", "max_abs_gamma_24",
                                   "shift_12_24_per_bin", "passes", "cell_steps")}})
    art = {
        "schema": SCHEMA, "driver": DRIVER, "reuses": battery.DRIVER,
        "what": ("The coax one-port lane's open termination at 4 and 6 annulus cells on "
                 "the battery's 8 x 8 mm cross-section, 40 mm and 25 mm long, at 12 and 24 "
                 "line traversals: the shipped lane (absorbers on z only) beside the fixed "
                 "one (all three axes), with the check an always-on test applies. No verdict."),
        "check": {"max_abs_gamma_bound": MAX_ABS_GAMMA, "shift_bound": DOUBLING_SHIFT_MAX,
                  "shift_definition": ("max over bins of | |Gamma| at 24 traversals - "
                                       "|Gamma| at 12 |"),
                  "passes": "max |Gamma| <= bound at both records and shift < bound"},
        "freqs_hz": battery.FREQS.astype(float).tolist(),
        "table": table, "shipped": old, "fixed": new,
    }
    out = Path(args.artifact_out)
    out.write_text(json.dumps(art, indent=1) + "\n")
    for row in table:
        print(json.dumps(row))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", type=int, choices=battery.RUNGS)
    ap.add_argument("--z-mm", type=float, default=battery.DOMAIN_ONEPORT[2] * 1e3)
    ap.add_argument("--out")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--summarize", default=None,
                    help="a directory of records (with its run_index.json)")
    ap.add_argument("--new", default=None,
                    help="summarize only: the fixed lane's directory, beside --summarize's")
    ap.add_argument("--artifact-out", default=None)
    args = ap.parse_args()
    if args.summarize:
        return summarize(args)
    if args.rung is None or not args.out:
        ap.error("--rung and --out are required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stage(args, out / record_name(args.rung, args.z_mm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
