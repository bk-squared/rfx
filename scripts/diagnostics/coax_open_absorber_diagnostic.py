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
        --artifact-out <path outside the repository>

Measurement records stay out of this repository (PI, 2026-09-24). The record of
2026-09-23 is ``rfx/records/20260924-coax-closed-can/open_absorber_diagnostic.json``
in ``bk-squared/rfx-archive``.
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
BATTERY_ARTIFACT = battery.ARTIFACT
PREDECLARATION = ("the leader's pre-declaration, a local note that is not committed; the "
                  "arms and thresholds it declares are restated in this artifact's "
                  "'predeclared' block")

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
           "what": ("control: open end 4 cells above the -z absorber, 16 x 16 mm. This "
                    "lane absorbs on z only, so the wider board moves the PEC walls of "
                    "the closed region around the shell from about 3.5 mm to about "
                    "7.4 mm from the shell's outer surface and moves no absorber")},
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


def line_witness(rec: dict) -> dict:
    """What the lane's own matrix-pencil fit says about the line and about the
    two waves at the probes. Derived from the record; nothing is re-solved.

    ``gamma = alpha + j beta`` is fitted from the modal voltage's shape along z;
    ``measured_beta`` (the battery's) turns beta into the permittivity a uniform
    TEM line would need. The lane reports Gamma at its reference plane by
    carrying the two fitted waves from the probe centroid down to that plane
    (``coaxial_line_reflection_from_plane_voltages``), so
    ``|Gamma_ref| = |B/A| exp(2 alpha d)`` with ``d`` the centroid-to-plane
    distance, and ``|B/A| = |S11| exp(-2 alpha d)`` is the ratio of the two
    fitted waves at the centroid itself. Both are DERIVED numbers.
    """
    res = rec["result"]
    f = np.asarray(res["freqs_hz"], dtype=float)
    g = battery._S(res["gamma"])
    s = battery._S(res["S11"])
    lay = rec["realized"]["layout"]
    dx = float(rec["realized"]["dx_m"])
    d = (float(np.mean(lay["probes"])) - float(lay["z_dut"])) * dx
    alpha = np.real(g)
    gain = np.exp(2.0 * alpha * d)
    b_over_a = np.abs(s) / gain
    mb = battery.measured_beta(res["gamma"], f, float(battery.PTFE_EPS_R))
    return {
        "derived": True,
        "formula": ("extrapolation_gain = exp(2 alpha d); abs_b_over_a = |S11| / "
                    "extrapolation_gain; d = (mean(probe nodes) - z_dut) * dx"),
        "centroid_to_reference_plane_m": d,
        "alpha_np_per_m": alpha.astype(float).tolist(),
        "max_alpha_np_per_m": float(alpha.max()),
        "extrapolation_gain": gain.astype(float).tolist(),
        "max_extrapolation_gain": float(gain.max()),
        "abs_b_over_a": b_over_a.astype(float).tolist(),
        "max_abs_b_over_a": float(b_over_a.max()),
        "eps_eff_fitted_mean": mb["mean_eps_eff_fitted"],
        "eps_eff_analytic": mb["eps_eff_analytic"],
        "beta_ratio_min": mb["min_beta_ratio"],
        "beta_ratio_max": mb["max_beta_ratio"],
    }


def z_ends(rec: dict) -> dict:
    """Both realized conductor ends against both z absorbers, from the stored
    ``boundary_geometry`` (nothing re-solved).

    Node ``k`` sits at ``z = (k - pad_z_lo) dx``. The -z CPML slab covers nodes
    ``0 .. pad_z_lo - 1`` and the +z slab the last ``pad_z_hi`` nodes. A pad
    face is the stamper's "pad inner edge": the last node outside each slab,
    ``pad_z_lo`` below and ``nz - 1 - pad_z_hi`` above. A negative distance
    means the conductor end lies inside that slab.
    """
    bg = rec["boundary_geometry"]
    dx = float(bg["dx_m"])
    nz = int(bg["grid_shape"][2])
    lo_pad, hi_pad = int(bg["pads_cells"]["z_lo"]), int(bg["pads_cells"]["z_hi"])
    lo = int(bg["open_end"]["lowest_conductor_node"])
    hi = int(bg["open_end"]["highest_conductor_node"])
    lo_face, hi_face = lo_pad, nz - 1 - hi_pad
    return {
        "lower_end_node": lo, "lower_end_z_m": (lo - lo_pad) * dx,
        "upper_end_node": hi, "upper_end_z_m": (hi - lo_pad) * dx,
        "minus_z_pad_face_node": lo_face, "minus_z_pad_face_z_m": 0.0,
        "plus_z_pad_face_node": hi_face, "plus_z_pad_face_z_m": (hi_face - lo_pad) * dx,
        "minus_z_slab_nodes": [0, lo_pad - 1],
        "plus_z_slab_nodes": [nz - hi_pad, nz - 1],
        "lower_end_to_minus_z_pad_face_cells": lo - lo_face,
        "lower_end_to_minus_z_pad_face_m": (lo - lo_face) * dx,
        "upper_end_to_plus_z_pad_face_cells": hi_face - hi,
        "upper_end_to_plus_z_pad_face_m": (hi_face - hi) * dx,
        "upper_end_is_first_plus_z_slab_node": bool(hi == nz - hi_pad),
        "reference_plane_node": int(bg["open_end"]["reference_plane_node"]),
        "reference_plane_minus_lower_end_cells": int(bg["open_end"]["reference_plane_node"]) - lo,
    }


# Where the lane's own text says the conductors stop, quoted for the facts
# block. Line numbers are those of the commit the records were measured at.
LANE_UPPER_END_COMMENTS = (
    ("rfx/sparams/coax.py:453-454",
     "The conductors deliberately stop ~2 cells short of the +z PML — running "
     "PEC into CPML is numerically unstable."),
    ("rfx/sparams/coax.py:645-646",
     "# Axial layout: DUT just above the -z PML; coax runs up to ~2 cells short "
     "# of the +z PML; matched feed one cell below the coax top; source below it."),
    ("rfx/sources/coaxial_port.py:2023-2026",
     "The conductors must NOT extend into a CPML region: running PEC into the PML "
     "is numerically unstable (verified: max|E| diverges to NaN). Stop the line at "
     "least ~2 cells short of the absorbing boundary and terminate the feed with "
     ":func:`stamp_coaxial_annular_resistor`."),
)

# The leader's revision of the hypothesis and of A2's role. It arrived after
# every job of the campaign had finished, so the arms below ran under the
# original declaration; this is kept verbatim beside the arms as run.
REVISION_AFTER_THE_RUN = {
    "received": ("2026-09-23, after all 14 jobs had finished at 4e008494 and the "
                 "results had been reported; the campaign ran under the original "
                 "declaration"),
    "hypothesis": ("the open end's fringing field reaches the -z absorber, which sits 3 "
                   "cells below the REALIZED open end at every rung (about 1.07 / 0.71 / "
                   "0.47 mm at 4 / 6 / 9 annulus cells); an absorber inside a near field "
                   "returns more than it receives and keeps ringing. Predictions and the "
                   "falsifier stay as declared, now judged on A1 (offset 2b from the "
                   "realized end, i.e. set the offset so the realized end is 2b = 4.11 mm "
                   "above the pad face) and A3."),
    "a2_role": ("A2 is now a control, not a test of H: widening the lateral domain to "
                "16 mm moves the PEC walls of the closed region around the shell from "
                "~3.5 mm to ~7.5 mm from the shell's outer surface; it moves no absorber."),
}


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
        "driver": DRIVER, "reuses": battery.DRIVER, "artifact": str(artifact_out),
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
            "line_witness": line_witness(rec),
            "z_ends": z_ends(rec),
        }

    def rec_of(arm, rung, units):
        return records.get(f"{arm}_rung{rung}_u{units:g}")

    # A0 is the battery's own open at 9 cells, run again on this campaign's
    # backend: the difference is what the backend and this driver's plumbing
    # change, before any arm moves the geometry.
    bat9 = battery_fix["solves"]["open_rung9"]
    battery_s = {12.0: battery._S(bat9["S11"]),
                 24.0: battery._S(battery_fix["record_length_invariance"]["open"]["S_doubled"])}
    art["a0_against_battery"] = {}
    for units, ref in battery_s.items():
        r = rec_of("A0", 9, units)
        if r is None:
            continue
        s = battery._S(r["result"]["S11"])
        art["a0_against_battery"][f"u{units:g}"] = {
            "max_abs_complex_diff": float(np.max(np.abs(s - ref))),
            "max_abs_magnitude_diff": float(np.max(np.abs(np.abs(s) - np.abs(ref)))),
            "battery_max_abs_s11": float(np.max(np.abs(ref))),
            "this_max_abs_s11": float(np.max(np.abs(s))),
            "battery_backend": bat9["provenance"].get("jax_default_backend"),
            "this_backend": r["provenance"].get("jax_default_backend"),
        }

    for arm, rung, units in CAMPAIGN:
        rec = rec_of(arm, rung, units)
        if rec is None:
            continue
        s, bg = rec["summary"], rec["boundary_geometry"]
        lw = art["records"][f"{arm}_rung{rung}_u{units:g}"]["line_witness"]
        ze = art["records"][f"{arm}_rung{rung}_u{units:g}"]["z_ends"]
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
            "lower_end_node": ze["lower_end_node"],
            "lower_end_to_minus_z_pad_face_m": ze["lower_end_to_minus_z_pad_face_m"],
            "upper_end_node": ze["upper_end_node"],
            "upper_end_to_plus_z_pad_face_cells": ze["upper_end_to_plus_z_pad_face_cells"],
            "upper_end_is_first_plus_z_slab_node": ze["upper_end_is_first_plus_z_slab_node"],
            "max_abs_s11": s["max_abs_s11"], "argmax_bin": s["argmax_bin"],
            "argmax_hz": s["argmax_hz"], "max_abs_s11_sq": s["max_abs_s11_sq"],
            "min_abs_s11": s["min_abs_s11"], "argmin_hz": s["argmin_hz"],
            "max_recurrence_residual": s["max_recurrence_residual"],
            "max_fit_residual": s["max_fit_residual"],
            "max_abs_b_over_a_derived": lw["max_abs_b_over_a"],
            "max_alpha_np_per_m": lw["max_alpha_np_per_m"],
            "eps_eff_fitted_mean": lw["eps_eff_fitted_mean"],
            "n_bins": len(s["abs_s11"]),
            "n_bins_abs_s11_above_1": int(np.sum(np.asarray(s["abs_s11"]) > 1.0)),
            "n_bins_power_above_bar": int(np.sum(np.asarray(s["abs_s11"]) ** 2 > POWER_MAX)),
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

    # --- facts about the realized geometry, one entry per board and rung ----
    geometry = {}
    for arm, rung, units in CAMPAIGN:
        r = art["records"].get(f"{arm}_rung{rung}_u{units:g}")
        if r is None or f"{arm}_rung{rung}" in geometry:
            continue
        ze, bg = r["z_ends"], r["boundary_geometry"]
        geometry[f"{arm}_rung{rung}"] = {
            "absorbing_axes": bg["absorbing_axes"],
            "dut_offset_cells_as_run": bg["open_end"]["reference_plane_above_z_pad_cells"],
            "reference_plane_node": ze["reference_plane_node"],
            "lower_end_node": ze["lower_end_node"],
            "lower_end_to_minus_z_pad_face_cells": ze["lower_end_to_minus_z_pad_face_cells"],
            "lower_end_to_minus_z_pad_face_m": ze["lower_end_to_minus_z_pad_face_m"],
            "upper_end_node": ze["upper_end_node"],
            "first_plus_z_slab_node": ze["plus_z_slab_nodes"][0],
            "upper_end_to_plus_z_pad_face_cells": ze["upper_end_to_plus_z_pad_face_cells"],
            "upper_end_to_plus_z_pad_face_m": ze["upper_end_to_plus_z_pad_face_m"],
            "upper_end_is_first_plus_z_slab_node": ze["upper_end_is_first_plus_z_slab_node"],
            "min_shell_to_pad_face_m": bg["lateral"]["min_shell_to_pad_face_m"],
            "min_shell_to_pec_wall_m": bg["lateral"]["min_shell_to_pec_wall_m"],
        }
    art["facts"] = {
        "what": "realized geometry read from the stored records, no interpretation",
        "absorbing_axes": sorted({g["absorbing_axes"] for g in geometry.values()}),
        "lateral_pads": ("vacuum with the PEC grid faces behind them: the lane's run "
                         "applies no CPML correction on x or y"),
        "upper_end_lane_text": [{"where": w, "text": t} for w, t in LANE_UPPER_END_COMMENTS],
        "per_geometry": geometry,
    }

    # --- the leader's revision, beside the arms as they actually ran --------
    two_b = 2.0 * battery.port_radii()[1]
    as_run = {}
    for key, g in geometry.items():
        arm, rung = key.split("_rung")[0], int(key.split("_rung")[1])
        if ARMS[arm]["offset"] != "two_b":
            continue
        dx = battery.dx_of(rung)
        target = int(round(two_b / dx))
        as_run[key] = {
            "realized_end_cells_as_run": g["lower_end_to_minus_z_pad_face_cells"],
            "realized_end_m_as_run": g["lower_end_to_minus_z_pad_face_m"],
            "dut_offset_cells_as_run": g["dut_offset_cells_as_run"],
            "realized_end_cells_revision": target,
            "realized_end_m_revision": target * dx,
            "dut_offset_cells_revision": target + (g["reference_plane_node"]
                                                   - g["lower_end_node"]),
        }
    art["revision_after_the_run"] = dict(REVISION_AFTER_THE_RUN)
    art["revision_after_the_run"].update({
        "arms_as_run_against_the_revision": as_run,
        "predictions_on_the_arms_as_run": [
            {"id": "A3_all_rungs", "per_rung": a3,
             "holds": all_of(v["holds"] for v in a3.values())},
            {"id": "A1_rung9", "max_abs_s11_sq_12u": power_12("A1", 9),
             "shift_12_24": shift_12_24("A1", 9), "holds": holds("A1", 9)},
            {"id": "falsifier_A3_rung9", "max_abs_s11_sq_12u": power_12("A3", 9),
             "shift_12_24": shift_12_24("A3", 9),
             "fired": (None if a3[9]["holds"] is None else (not a3[9]["holds"]))},
        ],
        "read_with": ("these judge A1 and A3 as run, whose realized open ends sit the "
                      "number of cells given in arms_as_run_against_the_revision short "
                      "of the revision's 2b; A2 is the control"),
    })

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
    ap.add_argument("--artifact-out", default=None,
                    help="assemble only: where the record is written, outside the repository")
    args = ap.parse_args()

    if args.layout_only:
        return stage_layout_only()
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        stage_assemble(args, out,
                       battery.record_path_outside_the_repository(ap, args.artifact_out))
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
