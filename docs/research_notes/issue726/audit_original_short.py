"""Reproduce the baseline #726 short fixture's geometry without field solves."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np

from rfx import Simulation
from rfx.api._preflight import msl_source_near_field_standoff_cells
from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.sources.msl_port import (
    msl_cross_section_span, msl_port_from_entry, msl_probe_x_coords_n,
)

BASE = "0662c3b96d82ca65d7bd041b94c4ab984230e00d"
REPO = Path(__file__).resolve().parents[3]
REL = "scripts/diagnostics/msl_probe_clearance_shorted_line.py"


def forbidden(*args, **kwargs):
    raise AssertionError("field solve forbidden during geometry audit")


def main():
    Simulation.run = forbidden
    Simulation.forward = forbidden
    Simulation.compute_msl_s_matrix = forbidden
    source = subprocess.check_output(["git", "show", f"{BASE}:{REL}"], cwd=REPO)
    mod = ModuleType("historical_short")
    mod.__file__ = str(REPO / REL)
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    rows = []
    for label, feed_x in (("clean", .002), ("near", .0184)):
        sim, nominal = mod.build(.020, feed_x, .020)
        grid = sim._build_grid()
        entry = sim._msl_ports[0]
        port = msl_port_from_entry(entry)
        span = msl_cross_section_span(grid, port)
        sheets, wires = [], []
        assembled = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
        edges = realized_pec_edge_masks(assembled[3], sheets=sheets, wires=wires,
                                       periodic=sim._periodic_flags())
        xs = np.asarray(coords_from_uniform_grid(grid).x)
        ladder = msl_probe_x_coords_n(grid, port, entry.n_probes,
                                     entry.n_probe_offset, entry.n_probe_spacing)
        indices = [int(np.argmin(abs(xs - x))) for x in ladder]
        jc = span["w_centre"]
        km = (span["n_lo"] + span["n_hi"]) // 2
        walls = realized_wall_planes(edges, 0, ij=(jc, km),
                                     periodic=sim._periodic_flags())
        rows.append(dict(
            arm=label, actual_feed_x_m=float(xs[span["i_feed"]]),
            probe_x_m=list(ladder), short_wall_x_m=[float(xs[i]) for i in walls],
            actual_deepest_gap_m=float(xs[walls[0]] - ladder[-1]),
            script_claimed_gap_m=.020 - nominal,
            source_offsets_cells=[i - span["i_feed"] for i in indices],
            required_source_standoff_cells=msl_source_near_field_standoff_cells(
                entry.height, grid.dx),
            pec_substrate_edges_per_probe=[int(np.asarray(edges[2])[
                i, jc, span["n_lo"]:span["n_hi"]].sum()) for i in indices],
        ))
    output = dict(base_sha=BASE, source_sha256=hashlib.sha256(source).hexdigest(),
                  no_field_solve=True, arms=rows)
    path = Path(__file__).with_name("original-short-build.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
