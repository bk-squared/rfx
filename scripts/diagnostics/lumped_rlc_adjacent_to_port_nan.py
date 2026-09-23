#!/usr/bin/env python3
"""A series R+C lumped element one cell from a port drives the fields to NaN.

``add_lumped_rlc(..., topology="series")`` placed on a cell that carries no
port conductance grows without bound below a resistance threshold, and the run
ends in non-finite fields. Nothing warns; the returned arrays are NaN.

This is NOT caused by the port-extraction change of 2026-09-21 — it reproduces
identically on the commit before it — and it is not fixed here. The script
exists so the defect has a reproduction attached to it.

What it shows
-------------
The smallest case is a 5-cell PEC box at dx = 1 mm with one lumped port and a
series R = 50 ohm, C = 0.20 pF element one cell along x from the port cell.
The fields grow past 1e35 and the first non-finite value appears a couple of
hundred steps in. The element does NOT need a port at all: an ``add_source``
in its place diverges the same way, so the port is not part of the mechanism.

The script then prints two envelopes, because they are what make the defect
actionable:

* **resistance** — the same fixture at R = 50 … 300 ohm. It diverges well
  below the threshold, sits marginal around it, and decays cleanly above it.
* **cell size** — R = 50 and R = 200 at dx = 0.5, 1.0 and 2.0 mm. The
  behaviour does not move with the mesh over that 4x range, so whatever sets
  the threshold is not a CFL-style condition on dt and dx.

Co-locating the element WITH the port cell is stable: the port conductance
folded into that cell is what holds it. That is also why no existing test
caught this — every fixture in the repo that combines ``add_lumped_rlc`` with
``add_port`` puts them on the same cell.

Using it
--------
No arguments. Takes about a minute.

    python scripts/diagnostics/lumped_rlc_adjacent_to_port_nan.py

Writes ``lumped_rlc_adjacent_to_port_nan.json`` beside this file.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfx import GaussianPulse, Simulation  # noqa: E402

OUT_JSON = Path(__file__).with_suffix(".json")

COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")
C_FARAD = 0.20e-12
N_CELLS = 5
N_STEPS = 1200


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            text=True, stderr=subprocess.PIPE).strip()
        return out or None
    except Exception:  # noqa: BLE001 — provenance is recorded, never fatal
        return None


def run_case(dx: float, r_ohm: float, *, drive: str = "port",
             offset_cells: int = 1, n_steps: int = N_STEPS) -> dict:
    """One solve. Returns where the fields first go non-finite, and in what."""
    port_pos = (2 * dx, 2 * dx, 2 * dx)
    rlc_pos = (port_pos[0] + offset_cells * dx, port_pos[1], port_pos[2])

    sim = Simulation(freq_max=10e9, domain=(N_CELLS * dx,) * 3, dx=dx,
                     boundary="pec")
    if drive == "port":
        sim.add_port(position=port_pos, component="ez", impedance=50.0,
                     waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    else:
        sim.add_source(position=port_pos, component="ez",
                       waveform=GaussianPulse(f0=5e9, bandwidth=0.9),
                       amplitude_kind="field")
    sim.add_lumped_rlc(position=rlc_pos, component="ez",
                       R=r_ohm, C=C_FARAD, topology="series")
    sim.add_vector_probe(rlc_pos)

    ts = np.asarray(sim.run(n_steps=n_steps, skip_preflight=True).time_series)
    bad = ~np.isfinite(ts)
    first_step = int(np.argmax(bad.any(axis=1))) if bad.any() else None
    first_components = (
        [c for k, c in enumerate(COMPONENTS) if bad[first_step, k]]
        if first_step is not None else []
    )
    finite = np.where(np.isfinite(ts), ts, 0.0)
    return {
        "dx_m": dx,
        "r_ohm": r_ohm,
        "c_farad": C_FARAD,
        "topology": "series",
        "drive": drive,
        "offset_cells": offset_cells,
        "n_steps": n_steps,
        "first_non_finite_step": first_step,
        "first_non_finite_components": first_components,
        "peak_abs_field_before": float(np.abs(finite).max()),
        "last_abs_field": float(np.abs(finite[-1]).max()),
    }


def _line(tag: str, case: dict) -> str:
    step = case["first_non_finite_step"]
    comps = ",".join(case["first_non_finite_components"]) or "-"
    return (f"  {tag:34s} first non-finite step "
            f"{'none' if step is None else str(step):>6s}  in {comps:14s} "
            f"peak |field| {case['peak_abs_field_before']:.3e}  "
            f"last {case['last_abs_field']:.3e}")


def main() -> int:
    record: dict = {
        "schema": "rfx.lumped_rlc_adjacent_to_port_nan",
        "schema_version": 1,
        "what": ("a series R+C lumped element one cell from a port drives the "
                 "fields to non-finite values, with no warning"),
        "script": "scripts/diagnostics/lumped_rlc_adjacent_to_port_nan.py",
        "commit": git_sha(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "fixture": {
            "box_cells": N_CELLS, "boundary": "pec",
            "port_node_cells": 2, "c_farad": C_FARAD, "topology": "series",
        },
    }

    print("Smallest reproduction: 5-cell PEC box, dx = 1 mm, lumped port at "
          "cell 2,\nseries R = 50 ohm C = 0.20 pF one cell along x.\n")
    smallest = run_case(1e-3, 50.0)
    record["smallest"] = smallest
    print(_line("port drive", smallest))

    no_port = run_case(1e-3, 50.0, drive="source")
    record["without_a_port"] = no_port
    print(_line("plain source instead of a port", no_port))

    co_located = run_case(1e-3, 50.0, offset_cells=0)
    record["co_located_with_the_port"] = co_located
    print(_line("element ON the port cell", co_located))

    print("\nResistance envelope, dx = 1 mm:")
    record["resistance_envelope"] = []
    for r in (50.0, 100.0, 120.0, 150.0, 180.0, 200.0, 300.0):
        case = run_case(1e-3, r)
        record["resistance_envelope"].append(case)
        print(_line(f"R = {r:.0f} ohm", case))

    print("\nCell-size envelope:")
    record["cell_size_envelope"] = []
    for dx in (0.5e-3, 1e-3, 2e-3):
        for r in (50.0, 200.0):
            case = run_case(dx, r)
            record["cell_size_envelope"].append(case)
            print(_line(f"dx = {dx * 1e3:.1f} mm, R = {r:.0f} ohm", case))

    OUT_JSON.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
