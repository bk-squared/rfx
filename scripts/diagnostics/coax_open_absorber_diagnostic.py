#!/usr/bin/env python3
"""The coaxial open end moved away from the board's boundaries: one pre-declared test.

A coaxial line whose conductors simply stop is an open circuit, and on a
lossless line its reflection can only be as large as unity: whatever the end
fringes or radiates leaves |S11| below 1. The coaxial chain battery measured
the one-port lane's open end above that (max |S11| 1.0118 / 1.0068 / 1.0185 at
4 / 6 / 9 annulus cells), and at 9 cells doubling the record moved max|S| by
0.0526, twice the bound; the PEC short on the same board reads 1.0012-1.0015.
On that board the line ends ``dut_offset_cells = 4`` cells above the -z
absorber, and the shell's outer surface is 0.945 mm from the lateral pads.

This script runs the arms that move the open end away from the -z absorber
(A1), enlarge the board laterally (A2), or both (A3), each at the battery's
record and at the doubled one, and records the numbers. It writes no verdict:
the assembled artifact carries each measurement beside the pre-declared
threshold and nothing else.

What the lateral arms change, read from the code before any run: this lane
absorbs on z only. ``compute_coaxial_line_reflection`` accepts no
``cpml_axes`` but ``"z"`` and hands it to the runner, so the 16-cell x and y
pads the grid still allocates are vacuum with no absorber, and the grid's
outer faces are PEC (``apply_pec`` on every non-periodic axis). A wider board
therefore moves those PEC walls away from the shell; it does not move an
absorber. Every record states the pads, the absorbing axes and both distances
(to the pad's inner face and to the PEC wall).

Physics is not forked. The board, the line, the stamps, the drive, the probes,
the record length, the provenance and the writer are the battery driver's own
(``coax_chain_battery_measure.py``, imported); the only thing this script
passes that the battery does not is the lane's ``dut_offset_cells`` and a
wider ``domain``.

Usage (from a clean checkout; the ``rfx`` import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/coax_open_absorber_diagnostic.py \\
        --arm A3 --rung 9 --record-units 24 --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/coax_open_absorber_diagnostic.py \\
        --layout-only
    PYTHONPATH=. python scripts/diagnostics/coax_open_absorber_diagnostic.py \\
        --assemble --out <dir with every arm JSON> --run-index <json> \\
        --artifact-out tests/fixtures/coax_chain_battery/open_absorber_diagnostic.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import coax_chain_battery_measure as battery  # noqa: E402  (puts the repo on sys.path)

from rfx.api import Simulation  # noqa: E402
from rfx.sources.coaxial_port import stamp_coaxial_line  # noqa: E402
from rfx.sparams.coax import _coax_pec_edge_masks  # noqa: E402

SCHEMA = "rfx.coax_open_absorber_diagnostic"
SCHEMA_VERSION = 1
DRIVER = "scripts/diagnostics/coax_open_absorber_diagnostic.py"
ARTIFACT = "tests/fixtures/coax_chain_battery/open_absorber_diagnostic.json"
BATTERY_ARTIFACT = battery.ARTIFACT
PREDECLARATION = ("docs/research_notes/coax_open_absorber_predeclaration.md "
                  "(local research note, not committed; the arms and thresholds it "
                  "declares are restated in this artifact's 'predeclared' block)")

DUT = "open"
BOARD_Z_M = battery.DOMAIN_ONEPORT[2]           # 40 mm in every arm
LATERAL_BATTERY_M = battery.DOMAIN_ONEPORT[0]   # 8 mm, the battery's board
LATERAL_WIDE_M = 0.016

# offset: "battery" is the lane default the battery ran with (4 cells at every
# rung); "two_b" is twice the port's outer radius, rounded to the nearest whole
# cell at the rung. Both are handed to the lane as ``dut_offset_cells``.
ARMS = {
    "A0": {"offset": "battery", "lateral_m": LATERAL_BATTERY_M,
           "what": "the battery's board: open end 4 cells above the -z absorber, 8 x 8 mm"},
    "A1": {"offset": "two_b", "lateral_m": LATERAL_BATTERY_M,
           "what": "open end 2b above the -z absorber, 8 x 8 mm"},
    "A2": {"offset": "battery", "lateral_m": LATERAL_WIDE_M,
           "what": "open end 4 cells above the -z absorber, 16 x 16 mm"},
    "A3": {"offset": "two_b", "lateral_m": LATERAL_WIDE_M,
           "what": "open end 2b above the -z absorber, 16 x 16 mm"},
}

# (arm, rung, record units) — the pre-declared matrix, nothing more.
CAMPAIGN = (
    ("A0", 9, 12.0), ("A0", 9, 24.0), ("A0", 9, 48.0),
    ("A1", 9, 12.0), ("A1", 9, 24.0),
    ("A2", 9, 12.0), ("A2", 9, 24.0),
    ("A3", 9, 12.0), ("A3", 9, 24.0), ("A3", 9, 48.0),
    ("A3", 4, 12.0), ("A3", 4, 24.0),
    ("A3", 6, 12.0), ("A3", 6, 24.0),
)

# The thresholds the pre-declaration names, from the battery's own bar.
POWER_MAX = battery.BAR["column_power_max"]                          # 1.02
DOUBLING_SHIFT_MAX = (10 ** (battery.BAR["magnitude_db"] / 20.0) - 1.0) / 10.0   # 0.025893


def lane_default(name: str):
    """A keyword default of the lane under test, read from its signature."""
    return inspect.signature(Simulation.compute_coaxial_line_reflection).parameters[name].default


def record_name(arm: str, rung: int, units: float) -> str:
    return f"open_absorber_{arm}_rung{rung}_u{units:g}.json"


def domain_of(arm: str) -> tuple[float, float, float]:
    lat = float(ARMS[arm]["lateral_m"])
    return (lat, lat, float(BOARD_Z_M))


def offset_of(arm: str, rung: int) -> dict:
    """The DUT offset the arm hands the lane, in cells, and the metres it names."""
    dx = battery.dx_of(rung)
    _, b = battery.port_radii()
    kind = ARMS[arm]["offset"]
    if kind == "battery":
        cells = int(battery.DUT_OFFSET_CELLS)
        target_m = None
    elif kind == "two_b":
        target_m = 2.0 * b
        cells = int(round(target_m / dx))
    else:
        raise ValueError(f"unknown offset kind {kind!r}")
    return {"kind": kind, "cells": cells, "metres": cells * dx, "target_m": target_m,
            "dx_m": dx}


# ---------------------------------------------------------------------------
# where the open end and the shell sit relative to the boundaries — measured
# from the lane's own stamp and its own cell-to-edge rule, before any step
# ---------------------------------------------------------------------------

def _touched_nodes(masks, axis: int) -> tuple[int, int]:
    """The lowest and highest node index along ``axis`` that a shorted E edge
    touches. An edge of component c runs from node i to i + 1 along c and sits
    on node i along the other two axes."""
    idx: list[int] = []
    for comp, m in enumerate(masks):
        other = tuple(t for t in range(3) if t != axis)
        hit = np.nonzero(np.asarray(m, dtype=bool).any(axis=other))[0]
        if hit.size == 0:
            continue
        idx += [int(hit.min()), int(hit.max())]
        if comp == axis:
            idx.append(int(hit.max()) + 1)
    if not idx:
        raise RuntimeError("no shorted E edge anywhere: the line was not realized")
    return min(idx), max(idx)


def boundary_geometry(sim, geo: dict) -> dict:
    """The realized open end against the -z absorber, and the realized shell
    against the lateral pads and the PEC walls, in cells and metres.

    Node ``i`` of an axis sits at ``(i - pad_lo) * dx`` (the battery driver's
    and the stamper's convention). The -z CPML slab covers nodes
    ``0 .. pad_z_lo - 1``, so node ``pad_z_lo`` is the first node outside it; the
    +z slab covers the last ``pad_z_hi`` nodes. The x and y pads have the same
    extent but, in this lane, no absorber (see ``absorbing_axes``).
    """
    grid = sim._build_grid()
    dx = float(grid.dx)
    nx, ny, nz = (int(s) for s in grid.shape)
    pad = {f"{ax}_{side}": int(getattr(grid, f"pad_{ax}_{side}"))
           for ax in "xyz" for side in ("lo", "hi")}
    port = sim._coaxial_ports[0]
    a, b = float(port.pin_radius), float(port.outer_radius)
    cx, cy = float(port.position[0]), float(port.position[1])
    lay = geo["layout"]

    materials, _, _ = sim._build_materials(grid)
    _, _, pec_cells = stamp_coaxial_line(
        grid, materials, center_xy=(cx, cy), z_lo_index=lay["z_dut"],
        z_hi_index=lay["z_hi_coax"], pin_radius=a, outer_radius=b)
    masks = _coax_pec_edge_masks(pec_cells)

    ilo, ihi = _touched_nodes(masks, 0)
    jlo, jhi = _touched_nodes(masks, 1)
    klo, khi = _touched_nodes(masks, 2)

    def xnode(i):
        return (i - pad["x_lo"]) * dx

    def ynode(j):
        return (j - pad["y_lo"]) * dx

    extent = {"x_lo": cx - xnode(ilo), "x_hi": xnode(ihi) - cx,
              "y_lo": cy - ynode(jlo), "y_hi": ynode(jhi) - cy}
    pad_face = {"x_lo": 0.0, "x_hi": (nx - 1 - pad["x_hi"] - pad["x_lo"]) * dx,
                "y_lo": 0.0, "y_hi": (ny - 1 - pad["y_hi"] - pad["y_lo"]) * dx}
    pec_wall = {"x_lo": -pad["x_lo"] * dx, "x_hi": (nx - 1 - pad["x_lo"]) * dx,
                "y_lo": -pad["y_lo"] * dx, "y_hi": (ny - 1 - pad["y_lo"]) * dx}
    centre = {"x_lo": cx, "x_hi": cx, "y_lo": cy, "y_hi": cy}
    sign = {"x_lo": -1.0, "x_hi": 1.0, "y_lo": -1.0, "y_hi": 1.0}
    axis_to_pad = {s: sign[s] * (pad_face[s] - centre[s]) for s in extent}
    axis_to_wall = {s: sign[s] * (pec_wall[s] - centre[s]) for s in extent}
    shell_to_pad = {s: axis_to_pad[s] - extent[s] for s in extent}
    shell_to_wall = {s: axis_to_wall[s] - extent[s] for s in extent}
    shell_outer_declared = float(geo["shell_outer_radius_m"])

    z_dut = int(lay["z_dut"])
    first_hi_pad_node = nz - pad["z_hi"]
    return {
        "grid_shape": [nx, ny, nz],
        "dx_m": dx,
        "pads_cells": pad,
        "grid_cpml_axes": str(getattr(grid, "cpml_axes", "")),
        "absorbing_axes": str(lane_default("cpml_axes")),
        "absorbing_axes_source": (
            "compute_coaxial_line_reflection's cpml_axes, the only value it accepts; "
            "the runner applies the CPML correction on these axes only, and the x and y "
            "pads are vacuum bounded by PEC grid faces"),
        "outer_faces": "PEC on every non-periodic axis (the runner's default pec_axes)",
        "declared": {
            "dut_offset_cells": z_dut - pad["z_lo"],
            "dut_offset_m": (z_dut - pad["z_lo"]) * dx,
            "shell_outer_radius_m": shell_outer_declared,
            "shell_to_pad_face_m": {s: axis_to_pad[s] - shell_outer_declared for s in extent},
        },
        "open_end": {
            "reference_plane_node": z_dut,
            "reference_plane_above_z_pad_cells": z_dut - pad["z_lo"],
            "reference_plane_above_z_pad_m": (z_dut - pad["z_lo"]) * dx,
            "lowest_conductor_node": klo,
            "open_end_above_z_pad_cells": klo - pad["z_lo"],
            "open_end_above_z_pad_m": (klo - pad["z_lo"]) * dx,
            "reference_plane_above_open_end_cells": z_dut - klo,
            "highest_conductor_node": khi,
            "top_minus_first_hi_pad_node_cells": khi - first_hi_pad_node,
        },
        "lateral": {
            "conductor_extent_from_axis_m": extent,
            "conductor_max_extent_m": max(extent.values()),
            "axis_to_pad_face_m": axis_to_pad,
            "axis_to_pec_wall_m": axis_to_wall,
            "shell_to_pad_face_m": shell_to_pad,
            "shell_to_pec_wall_m": shell_to_wall,
            "min_shell_to_pad_face_m": min(shell_to_pad.values()),
            "min_shell_to_pec_wall_m": min(shell_to_wall.values()),
            "node_extents": {"x": [ilo, ihi], "y": [jlo, jhi]},
        },
    }


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------

def result_summary(result: dict) -> dict:
    """Per-bin |S11|, its extremes and the phase at the lane's reference plane,
    read from the record the battery's own ``_one_port_record`` wrote."""
    g = battery._S(result["S11"])
    f = np.asarray(result["freqs_hz"], dtype=float)
    mag = np.abs(g)
    k_max, k_min = int(np.argmax(mag)), int(np.argmin(mag))
    phase = np.angle(g)
    return {
        "abs_s11": mag.astype(float).tolist(),
        "max_abs_s11": float(mag[k_max]),
        "argmax_bin": k_max, "argmax_hz": float(f[k_max]),
        "max_abs_s11_sq": float(mag[k_max] ** 2),
        "min_abs_s11": float(mag[k_min]),
        "argmin_bin": k_min, "argmin_hz": float(f[k_min]),
        "phase_rad_at_reference_plane": phase.astype(float).tolist(),
        "phase_deg_at_reference_plane": np.degrees(phase).astype(float).tolist(),
        "max_recurrence_residual": float(np.max(result["recurrence_residual"])),
        "max_fit_residual": float(np.max(result["fit_residual"])),
    }


def stage_arm(args, out: Path) -> None:
    arm, rung, units = args.arm, int(args.rung), float(args.record_units)
    domain = domain_of(arm)
    off = offset_of(arm, rung)
    sim = battery.build_sim(rung, DUT, drive=battery.DEFAULT_DRIVE, domain=domain)
    geo = battery.assert_realized(sim, rung, DUT, dut_offset_cells=off["cells"])
    bgeo = boundary_geometry(sim, geo)
    pf = battery.preflight_record(sim)
    grid = sim._build_grid()
    n_steps = battery.record_steps(grid, units)
    est = battery.cost_estimate(grid, n_steps, battery.N_FREQS, 2 * battery.PROBE_COUNT, 1)

    rec = battery._base(args, "open_absorber", DUT, rung)
    rec.update({
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "driver": DRIVER, "reuses": battery.DRIVER, "predeclaration": PREDECLARATION,
        "arm": arm, "arm_definition": ARMS[arm], "offset": off,
        "lane": "one_port", "drive": battery.DEFAULT_DRIVE,
        "record_units": units, "n_steps": n_steps,
        "declared": battery.declared(rung, DUT, domain=domain),
        "realized": geo, "boundary_geometry": bgeo, "preflight": pf, "cost": est,
        "result": None,
    })
    # persist the realized geometry before the solve: a job that dies in the
    # scan still leaves what it measured
    battery._write(out, rec)

    battery._log(f"open-absorber arm={arm} rung={rung} units={units:g} n_steps={n_steps} "
                 f"dut_offset_cells={off['cells']} lateral={domain[0]*1e3:g} mm")
    with battery._Captured() as cap:
        res = battery.solve_one_port(sim, DUT, n_steps=n_steps,
                                     dut_offset_cells=off["cells"])
    battery._log_one_port(f"open-absorber {arm} rung {rung} u{units:g}", res)
    rec["result"] = battery._one_port_record(res)
    rec["summary"] = result_summary(rec["result"])
    rec["cross_check"] = battery.cross_check_result_against_layout(geo, res, "one_port")
    rec["warnings"] = cap.warnings
    rec["wall_s"] = cap.wall
    rec["peak_memory"] = battery.peak_memory()
    battery._write(out, rec)


def stage_layout_only() -> int:
    """Every arm of the campaign built and asserted, nothing solved."""
    seen = set()
    for arm, rung, units in CAMPAIGN:
        if (arm, rung) in seen:
            continue
        seen.add((arm, rung))
        off = offset_of(arm, rung)
        domain = domain_of(arm)
        sim = battery.build_sim(rung, DUT, domain=domain)
        geo = battery.assert_realized(sim, rung, DUT, dut_offset_cells=off["cells"])
        bgeo = boundary_geometry(sim, geo)
        grid = sim._build_grid()
        oe, lat = bgeo["open_end"], bgeo["lateral"]
        target = "-" if off["target_m"] is None else f"{off['target_m'] * 1e3:.4f}"
        print(f"--- {arm} rung {rung}: dx {geo['dx_m']*1e6:.3f} um, board "
              f"{domain[0]*1e3:g} x {domain[1]*1e3:g} x {domain[2]*1e3:g} mm, grid "
              f"{bgeo['grid_shape']}, absorbing axes '{bgeo['absorbing_axes']}'")
        print(f"    dut_offset {off['cells']} cells = {off['metres']*1e3:.4f} mm "
              f"(target {target} mm); "
              f"reference plane node {oe['reference_plane_node']}, lowest conductor node "
              f"{oe['lowest_conductor_node']} = {oe['open_end_above_z_pad_cells']} cells = "
              f"{oe['open_end_above_z_pad_m']*1e3:.4f} mm above the -z pad; top node "
              f"{oe['highest_conductor_node']} ({oe['top_minus_first_hi_pad_node_cells']:+d} "
              f"vs the first +z pad node)")
        print(f"    shell reaches {lat['conductor_max_extent_m']*1e3:.4f} mm; shell to pad face "
              f"{lat['min_shell_to_pad_face_m']*1e3:.4f} mm, to PEC wall "
              f"{lat['min_shell_to_pec_wall_m']*1e3:.4f} mm; probes {geo['layout']['probes']}")
        for u in sorted({u for a_, r_, u in CAMPAIGN if a_ == arm and r_ == rung}):
            n = battery.record_steps(grid, u)
            battery.cost_estimate(grid, n, battery.N_FREQS, 2 * battery.PROBE_COUNT, 1)
    return 0


# ---------------------------------------------------------------------------
# assemble — arithmetic only
# ---------------------------------------------------------------------------

def _shift(rec_a: dict, rec_b: dict) -> float:
    """The battery's record-length metric: max over bins of ||S_a| - |S_b||."""
    ga, gb = battery._S(rec_a["result"]["S11"]), battery._S(rec_b["result"]["S11"])
    return float(np.max(np.abs(np.abs(ga) - np.abs(gb))))


def stage_assemble(args, out: Path, artifact_out: Path) -> None:
    commits = battery._refuse_a_set_spanning_commits(out)
    index = json.loads(Path(args.run_index).read_text()) if args.run_index else None
    records: dict[str, dict] = {}
    missing = []
    for arm, rung, units in CAMPAIGN:
        p = out / record_name(arm, rung, units)
        if not p.exists():
            missing.append(p.name)
            continue
        rec = json.loads(p.read_text())
        if rec.get("result") is None:
            missing.append(p.name + " (geometry only: the solve did not finish)")
            continue
        records[f"{arm}_rung{rung}_u{units:g}"] = rec

    battery_fix = json.loads((battery.REPO / BATTERY_ARTIFACT).read_text())
    art = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "what": ("The one-port lane's open end at three board layouts besides the "
                 "battery's own: the end moved 2b above the -z absorber, the board "
                 "widened to 16 x 16 mm, and both. Measurements beside the pre-declared "
                 "thresholds; no verdict."),
        "driver": DRIVER, "reuses": battery.DRIVER, "artifact": ARTIFACT,
        "battery_artifact": BATTERY_ARTIFACT, "predeclaration": PREDECLARATION,
        "assembled_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "assembler_commit": battery.git_sha(),
        "measured_at_commit": next(iter(commits)),
        "missing_records": missing,
        "arms": ARMS,
        "predeclared": {
            "campaign": [{"arm": a, "rung_annulus_cells": r, "record_units": u}
                         for a, r, u in CAMPAIGN],
            "column_power_max": POWER_MAX,
            "doubling_shift_max": DOUBLING_SHIFT_MAX,
            "doubling_shift_definition": ("max over bins of | |S11| at 24 units - |S11| at "
                                          "12 units |, the battery's record-length metric"),
            "predictions": [
                "A3 has max |S11|^2 <= 1.02 and a 12 -> 24 unit shift of max|S| below "
                "0.0259, at 4, 6 and 9 annulus cells",
                "at least one of A1 / A2 has the same at 9 annulus cells",
            ],
            "falsified_if": ("A3 at 9 annulus cells exceeds 1.02 in power, or its 12 -> 24 "
                             "unit shift is not below 0.0259"),
        },
        "battery_reference": {
            k: {"max_abs_s11": battery_fix["solves"][k]["max_abs_s11"],
                "n_steps": battery_fix["solves"][k]["n_steps"],
                "compute_run_id": battery_fix["solves"][k]["provenance"].get("compute_run_id"),
                "jax_default_backend": battery_fix["solves"][k]["provenance"].get(
                    "jax_default_backend")}
            for k in ("open_rung4", "open_rung6", "open_rung9", "short_rung9")
        },
        "battery_reference_doubling": {
            k: battery_fix["record_length_invariance"]["open"][k]
            for k in ("record_units", "n_steps", "max_abs_shift", "max_column_power")
        },
        "freqs_hz": battery.FREQS.astype(float).tolist(),
        "records": {},
        "table": [],
        "doubling": {},
        "predictions": [],
    }

    for key, rec in records.items():
        prov = battery.fixture_provenance(rec["provenance"], index,
                                          record_name(rec["arm"], rec["rung_annulus_cells"],
                                                      rec["record_units"]))
        art["records"][key] = {
            "arm": rec["arm"], "rung_annulus_cells": rec["rung_annulus_cells"],
            "record_units": rec["record_units"], "n_steps": rec["n_steps"],
            "offset": rec["offset"], "declared": rec["declared"],
            "realized": rec["realized"], "boundary_geometry": rec["boundary_geometry"],
            "cross_check": rec["cross_check"], "preflight_text": rec["preflight"]["text"],
            "warnings": rec["warnings"], "wall_s": rec["wall_s"],
            "peak_memory": rec["peak_memory"], "provenance": prov,
            "result": rec["result"], "summary": rec["summary"],
        }

    def rec_of(arm, rung, units):
        return records.get(f"{arm}_rung{rung}_u{units:g}")

    for arm, rung, units in CAMPAIGN:
        rec = rec_of(arm, rung, units)
        if rec is None:
            continue
        s, bg = rec["summary"], rec["boundary_geometry"]
        half = rec_of(arm, rung, units / 2.0)
        art["table"].append({
            "arm": arm, "rung_annulus_cells": rung, "record_units": units,
            "n_steps": rec["n_steps"], "dx_m": rec["realized"]["dx_m"],
            "board_m": list(domain_of(arm)),
            "dut_offset_cells": bg["open_end"]["reference_plane_above_z_pad_cells"],
            "dut_offset_m": bg["open_end"]["reference_plane_above_z_pad_m"],
            "open_end_above_z_pad_cells": bg["open_end"]["open_end_above_z_pad_cells"],
            "open_end_above_z_pad_m": bg["open_end"]["open_end_above_z_pad_m"],
            "shell_to_pad_face_m": bg["lateral"]["min_shell_to_pad_face_m"],
            "shell_to_pec_wall_m": bg["lateral"]["min_shell_to_pec_wall_m"],
            "max_abs_s11": s["max_abs_s11"], "argmax_bin": s["argmax_bin"],
            "argmax_hz": s["argmax_hz"], "max_abs_s11_sq": s["max_abs_s11_sq"],
            "min_abs_s11": s["min_abs_s11"], "argmin_hz": s["argmin_hz"],
            "max_recurrence_residual": s["max_recurrence_residual"],
            "max_fit_residual": s["max_fit_residual"],
            "status": rec["result"]["status"],
            "shift_from_half_record": None if half is None else _shift(half, rec),
            "power_within_bar": bool(s["max_abs_s11_sq"] <= POWER_MAX),
            "wall_s": rec["wall_s"],
        })

    for arm, rung in sorted({(a, r) for a, r, _ in CAMPAIGN}):
        us = sorted(u for a, r, u in CAMPAIGN if a == arm and r == rung)
        pairs = {}
        for i, u0 in enumerate(us):
            for u1 in us[i + 1:]:
                r0, r1 = rec_of(arm, rung, u0), rec_of(arm, rung, u1)
                if r0 is not None and r1 is not None:
                    pairs[f"{u0:g}->{u1:g}"] = _shift(r0, r1)
        art["doubling"][f"{arm}_rung{rung}"] = {
            "shifts": pairs, "bound": DOUBLING_SHIFT_MAX,
            "shift_12_24_within_bound": (None if "12->24" not in pairs
                                         else bool(pairs["12->24"] < DOUBLING_SHIFT_MAX)),
        }

    def power_12(arm, rung):
        r = rec_of(arm, rung, 12.0)
        return None if r is None else r["summary"]["max_abs_s11_sq"]

    def shift_12_24(arm, rung):
        return art["doubling"].get(f"{arm}_rung{rung}", {}).get("shifts", {}).get("12->24")

    def holds(arm, rung):
        p, s = power_12(arm, rung), shift_12_24(arm, rung)
        if p is None or s is None:
            return None
        return bool(p <= POWER_MAX and s < DOUBLING_SHIFT_MAX)

    def all_of(values):
        """Three-valued AND: False if any is False, None if any is unknown."""
        values = list(values)
        if any(v is False for v in values):
            return False
        return None if any(v is None for v in values) else True

    def any_of(values):
        """Three-valued OR: True if any is True, None if any is unknown."""
        values = list(values)
        if any(v is True for v in values):
            return True
        return None if any(v is None for v in values) else False

    a3 = {r: {"max_abs_s11_sq_12u": power_12("A3", r), "shift_12_24": shift_12_24("A3", r),
              "holds": holds("A3", r)} for r in (4, 6, 9)}
    a12 = {a: {"max_abs_s11_sq_12u": power_12(a, 9), "shift_12_24": shift_12_24(a, 9),
               "holds": holds(a, 9)} for a in ("A1", "A2")}
    art["predictions"] = [
        {"id": "A3_all_rungs", "reads": "the 12-unit record's max |S11|^2 and the 12 -> 24 shift",
         "per_rung": a3, "holds": all_of(v["holds"] for v in a3.values())},
        {"id": "A1_or_A2_rung9", "reads": "the same two numbers at 9 annulus cells",
         "per_arm": a12, "holds": any_of(v["holds"] for v in a12.values())},
        {"id": "falsifier_A3_rung9",
         "reads": "A3 at 9 annulus cells: power above 1.02, or the 12 -> 24 shift not below 0.0259",
         "max_abs_s11_sq_12u": power_12("A3", 9), "shift_12_24": shift_12_24("A3", 9),
         "fired": (None if a3[9]["holds"] is None else (not a3[9]["holds"]))},
    ]

    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact_out.with_suffix(artifact_out.suffix + ".tmp")
    tmp.write_text(json.dumps(art, indent=1, sort_keys=False) + "\n")
    tmp.replace(artifact_out)
    battery._log(f"wrote {artifact_out} ({len(records)} records, missing {missing})")
    for row in art["table"]:
        sh = row["shift_from_half_record"]
        sh_txt = "-" if sh is None else f"{sh:.5f}"
        print(f"{row['arm']} r{row['rung_annulus_cells']} u{row['record_units']:g} "
              f"off {row['dut_offset_cells']} cells ({row['dut_offset_m']*1e3:.3f} mm) "
              f"end {row['open_end_above_z_pad_m']*1e3:.3f} mm "
              f"pad {row['shell_to_pad_face_m']*1e3:.3f} wall {row['shell_to_pec_wall_m']*1e3:.3f} mm | "
              f"max|S11| {row['max_abs_s11']:.5f} @ {row['argmax_hz']/1e9:.2f} GHz "
              f"^2 {row['max_abs_s11_sq']:.5f} min {row['min_abs_s11']:.5f} "
              f"shift {sh_txt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--rung", type=int, choices=battery.RUNGS, default=battery.CLAIMS_RUNG)
    ap.add_argument("--record-units", type=float, default=battery.DEFAULT_RECORD_UNITS)
    ap.add_argument("--out", help="directory for the arm JSONs")
    ap.add_argument("--run-id", default=None, help="the compute run id, recorded as-is")
    ap.add_argument("--layout-only", action="store_true",
                    help="build and assert every arm of the campaign, solve nothing")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--run-index", default=None,
                    help="assemble only: a JSON mapping each arm file to its compute "
                         "run id and run directory name")
    ap.add_argument("--artifact-out", default=str(battery.REPO / ARTIFACT))
    args = ap.parse_args()

    if args.layout_only:
        return stage_layout_only()
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        stage_assemble(args, out, Path(args.artifact_out))
        return 0
    if args.arm is None:
        ap.error("one of --arm, --assemble or --layout-only is required")
    if (args.arm, args.rung, float(args.record_units)) not in CAMPAIGN:
        ap.error(f"({args.arm}, rung {args.rung}, {args.record_units:g} units) is not in "
                 "the pre-declared campaign")
    stage_arm(args, out / record_name(args.arm, args.rung, float(args.record_units)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
