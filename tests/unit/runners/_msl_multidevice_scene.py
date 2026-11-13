"""J0 one-device scene observations, never an acceptance bar.

Builders use public declarations. Reporters inspect the production realization;
Slab is used only for hypothetical ownership, never to run multiple devices.
The reference is contracts/test_msl_one_reference_plane.py::_thru; explicit
profiles follow contracts/test_grouped_drives.py::port_scene.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.pec import realized_pec_edge_masks
from rfx.measurement.sparam_advisories import settling_verdict
from rfx.sources.msl_port import (
    compute_msl_mode_profile,
    msl_cell,
    msl_cross_section_span,
    msl_port_from_entry,
    msl_probe_x_coords_n,
    setup_msl_port,
)
from rfx.stepping.slab import Slab

FREQS = np.linspace(2e9, 18e9, 4)
PERIODS = (3.0, 4.5, 6.0)
SCENES = ("T1", "T1_graded", "T2", "Y1", "Y1_graded", "T1L")


@dataclass
class Scene:
    name: str
    sim: Simulation


def build_scene(name="T1", *, mode="laplace"):
    """Declare one of the six J0 scenes, with p1 driven and p2 passive."""
    if name.removesuffix("_graded") not in ("T1", "T2", "Y1", "T1L"):
        raise ValueError(f"Unknown scene {name!r}")
    is_y = name.startswith("Y1")
    domain = (0.008, 0.012, 0.0032) if is_y else (0.012, 0.008, 0.0032)
    options = {}
    if name.endswith("graded"):
        options["dz_profile"] = np.r_[
            np.full(4, 0.0002), np.full(4, 0.00018), np.full(8, 0.00021)
        ]
    sim = Simulation(
        snap="declared",
        freq_max=20e9,
        domain=domain,
        dx=0.0002,
        boundary="cpml",
        cpml_layers=8,
        **options,
    )
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (*domain[:2], 0.0008)), material="sub")
    sim.add(Box((0, 0, 0), (*domain[:2], 0)), material="pec")
    centre, width = (0.0039, 0.0018) if is_y else (0.004, 0.0012)
    if is_y:
        sim.add(
            Box(
                (centre - width / 2, 0.002, 0.0008), (centre + width / 2, 0.010, 0.0008)
            ),
            material="pec",
        )
    else:
        sim.add(
            Box(
                (0.002, centre - width / 2, 0.0008), (0.010, centre + width / 2, 0.0008)
            ),
            material="pec",
        )
    for index, (feed, sign) in enumerate(((0.002, "+"), (0.010, "-"))):
        sim.add_msl_port(
            position=(centre, feed, 0) if is_y else (feed, centre, 0),
            width=width,
            height=0.0008,
            direction=sign + ("y" if is_y else "x"),
            impedance=50.0,
            eps_r_sub=2.2,
            name=f"p{index + 1}",
            mode=mode,
            excite=index == 0,
            n_probe_offset=16 if name.removesuffix("_graded") == "T2" and index == 0 else 12,
            n_probe_spacing=3,
            n_probes=5,
        )
    if name.removesuffix("_graded") == "T1L":
        sim.add_port(
            position=(0.008, centre, 0.0004),
            component="ez",
            impedance=50.0,
            excite=False,
        )
    return Scene(name, sim)


def _ports(scene, grid):
    for entry in scene.sim._resolve_msl_probe_entries(grid):
        port = msl_port_from_entry(entry)
        span = msl_cross_section_span(grid, port)
        coords = msl_probe_x_coords_n(
            grid, port, entry.n_probes, entry.n_probe_offset, entry.n_probe_spacing
        )
        yield entry, port, span, coords


def _cells_report(cells, slab):
    cells = [tuple(map(int, c)) for c in cells]
    xs = sorted({c[0] for c in cells})
    return {
        "cells": cells,
        "x_indices": xs,
        "x_owners": [slab.owner(i) for i in xs],
        "counts_per_rank": [
            sum(slab.owner(c[0]) == r for c in cells) for r in range(slab.n_devices)
        ],
    }


def report_realized(scene, n_devices):
    """Inspect feed, ladder, sigma and release targets without time stepping.

    Release targets are the Ez cells passed to clear_conductor_edges; published
    release provenance is the wider Laplace footprint. Neither count claims
    that every targeted edge was PEC before clearing.
    """
    grid = scene.sim._build_realized_grid()
    slab = Slab(grid.nx, n_devices)
    sheets, wires = [], []
    assemble = (
        scene.sim._assemble_materials_nu
        if scene.name.endswith("graded")
        else scene.sim._assemble_materials
    )
    assembled = assemble(grid, pec_sheets=sheets, pec_wires=wires)
    edges = realized_pec_edge_masks(
        assembled[3], sheets=sheets, wires=wires, periodic=scene.sim._periodic_flags()
    )
    ez_pec = np.asarray(edges[2]).copy()
    result = {
        "scene": scene.name,
        "n_devices": n_devices,
        "grid_shape": grid.shape,
        "nx_per_rank": slab.nx_per_rank,
        "pad_x": slab.pad_x,
        "seams": [r * slab.nx_per_rank for r in range(1, n_devices)],
        "ports": [],
        "ownership_flags": [],
    }
    for entry, port, span, coords in _ports(scene, grid):
        profile = (
            compute_msl_mode_profile(grid, port, entry.eps_r_sub)
            if entry.mode == "laplace"
            else None
        )
        setup_msl_port(grid, port, assembled[0], mode_profile=profile)
        ladder = []
        for coordinate in coords:
            ip = grid.index_of(span["prop_axis"], coordinate)
            cells = [
                msl_cell(port.direction, ip, w, k)
                for w in span["width_nodes"]
                for k in range(span["n_lo"], span["n_hi"])
            ]
            ladder.append(
                dict(
                    coordinate_m=coordinate, prop_index=ip, **_cells_report(cells, slab)
                )
            )
        feed = _cells_report(span["cells"], slab)
        row = {
            "name": entry.name,
            "direction": entry.direction,
            "feed_prop_index": span["i_feed"],
            "feed": feed,
            "ladder": ladder,
            "sigma": _cells_report(port._drive_stamps, slab),
            "ez_clear_targets": _cells_report(span["cells"], slab),
            "published_release_edges": _cells_report(
                profile["cell_indices"] if profile is not None else span["cells"], slab
            ),
        }
        actual = [c for c in span["cells"] if ez_pec[tuple(c)]]
        row["released_pec_edges"] = _cells_report(actual, slab)
        for cell in span["cells"]:
            ez_pec[tuple(cell)] = False
        result["ports"].append(row)
    if scene.name.startswith("Y1") and n_devices == 2:
        counts = result["ports"][0]["feed"]["counts_per_rank"]
        fraction = counts[0] / sum(counts)
        result["y1_rank0_feed_fraction"] = fraction
        if not 0.55 <= fraction <= 0.65:
            result["ownership_flags"].append("Y1 feed not about 60/40 (55-65% rank0)")
    if not scene.name.startswith("Y1"):
        owners = [p["feed"]["x_owners"][0] for p in result["ports"]]
        if owners[0] == owners[1]:
            result["ownership_flags"].append("T feeds share a rank")
    if scene.name == "T2" and n_devices == 2:
        owners = {r for p in result["ports"][0]["ladder"] for r in p["x_owners"]}
        if len(owners) < 2:
            result["ownership_flags"].append("T2 p1 ladder does not cross seam")
    result["lumped_ports"] = []
    for entry in scene.sim._ports:
        cell = tuple(grid.index_of(axis, coordinate)
                     for axis, coordinate in zip("xyz", entry.position))
        result["lumped_ports"].append(_cells_report([cell], slab))
        if (
            n_devices == 2
            and slab.owner(int(cell[0])) == result["ports"][0]["feed"]["x_owners"][0]
        ):
            result["ownership_flags"].append("T1L lumped port shares p1 owner")
    return result


def _observers(scene):
    sim = scene.sim
    grid = sim._build_realized_grid()
    points = []
    for entry, port, span, coords in _ports(scene, grid):
        centre = msl_cell(
            port.direction,
            span["i_feed"],
            span["w_centre"],
            (span["n_lo"] + span["n_hi"]) // 2,
        )
        points.append((f"{entry.name}_feed_centre", centre))
        sim.add_dft_plane_probe(
            axis=span["prop_axis"],
            coordinate=grid.node_of(span["prop_axis"], span["i_feed"]),
            component="ez",
            freqs=FREQS,
            name=f"{entry.name}_feed",
        )
        if scene.name == "T2":
            for q, coord in enumerate(coords):
                sim.add_dft_plane_probe(
                    axis=span["prop_axis"],
                    coordinate=coord,
                    component="ez",
                    freqs=FREQS,
                    name=f"{entry.name}_ladder{q}",
                )
    points.append(("far_port_centre", points[-1][1]))
    # For y propagation, the seam cuts the trace width; sample at mid-line y.
    _, port, span, _ = next(_ports(scene, grid))
    mid_prop = grid.index_of(span["prop_axis"], 0.006)
    mid = list(
        msl_cell(
            port.direction,
            mid_prop,
            span["w_centre"],
            (span["n_lo"] + span["n_hi"]) // 2,
        )
    )
    for n in (2, 3):
        slab = Slab(grid.nx, n)
        for rank in range(1, n):
            for side in (-1, 1):
                cell = mid.copy()
                cell[0] = rank * slab.nx_per_rank + side
                points.append((f"n{n}_seam{rank}_{side:+d}", tuple(cell)))
    for _, cell in points:
        sim.add_probe(
            position=tuple(grid.node_of(a, i) for a, i in zip("xyz", cell)),
            component="ez",
        )
    return points


def _error(exc):
    return {"error_type": type(exc).__name__, "message": str(exc)}


def report_one_device(scene):
    """Run declared observers, then independent S drives at three lengths.

    A family refusal is recorded once with all affected period requests, so
    the same known refusal is not executed twice. Unexpected failures propagate.
    """
    from rfx.errors import RfxRefusal, RfxUnsupported

    started = time.monotonic()
    points = _observers(scene)
    result = {"scene": scene.name, "n_devices": 1, "n_steps": 400, "freqs_hz": FREQS}
    try:
        run = scene.sim.run(n_steps=400, compute_s_params=False)
    except (RfxRefusal, RfxUnsupported) as exc:
        result["run_refusal"] = _error(exc)
        result["s_not_attempted"] = "Same scene refusal; stop rather than repeat"
        return result
    values = np.asarray(run.time_series)
    result["probe_peaks"] = [
        {"name": name, "cell": cell, "peak": float(np.max(np.abs(values[:, i])))}
        for i, (name, cell) in enumerate(points)
    ]
    result["plane_accumulator_peaks"] = {
        name: np.max(np.abs(np.asarray(plane.accumulator)), axis=(1, 2))
        for name, plane in run.dft_planes.items()
    }
    result["s_runs"] = []
    for periods in PERIODS:
        try:
            s = scene.sim.compute_msl_s_matrix(freqs=FREQS, num_periods=periods)
        except (RfxRefusal, RfxUnsupported) as exc:
            result["s_runs"].append(dict(num_periods=periods, **_error(exc)))
            result["s_remaining_not_attempted"] = [p for p in PERIODS if p > periods]
            break
        witnesses = s.settling_witness
        row = {
            "num_periods": periods,
            "abs_s11": np.abs(np.asarray(s.S)[0, 0]),
            "abs_s21": np.abs(np.asarray(s.S)[1, 0]),
            "settling": [],
        }
        for i, db in enumerate(s.settling_db):
            witness = witnesses[i] if witnesses is not None else None
            row["settling"].append(
                {
                    "driven_port": f"p{i + 1}",
                    "db": db,
                    "verdict": settling_verdict(db),
                    "witness": witness,
                    "tail_level_exposed": witness is not None
                    and "share_per_bin" in witness,
                    "tail_level_kind": "pole-tail share_per_bin; not end/peak amplitude",
                }
            )
        result["s_runs"].append(row)
    result["wall_seconds"] = time.monotonic() - started
    return result


def _json(value):
    """Strict JSON, retaining nonfinite values as explicit strings."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def j0_main():
    """One labrun job, JSON lines on stdout, no acceptance decision."""
    import jax

    def emit(kind, data):
        print(json.dumps(_json(dict(kind=kind, **data)), allow_nan=False), flush=True)

    emit(
        "metadata",
        {
            "purpose": "J0 only; never a bar",
            "conclusion": "리더가 채움",
            "devices": [str(d) for d in jax.devices()],
            "tail_note": "Settling exposes pole-tail shares, not raw tail/peak L.",
        },
    )
    if len(jax.devices()) != 1:
        raise RuntimeError("J0 requires exactly one visible device")
    started = time.monotonic()
    for name in SCENES:
        emit("start_scene", {"scene": name})
        scene = build_scene(name)
        for n in (2, 3):
            emit("realized", report_realized(scene, n))
        emit("one_device", report_one_device(scene))
        jax.clear_caches()
    emit(
        "complete",
        {"wall_seconds": time.monotonic() - started, "conclusion": "리더가 채움"},
    )
