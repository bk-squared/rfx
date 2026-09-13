"""Read-only #752 geometry receipt; run from the selected checkout, no FDTD.

Example: cd /root/rfx-752 && taskset -c 4-5 env JAX_PLATFORMS=cpu
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python
/root/rfx-codex/.git/issue752/review/geometry_review.py > receipt.json
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from tests._realized_geometry import realized
from rfx.sources.msl_port import (
    msl_port_from_entry, msl_cross_section_span, validate_msl_port_geometry,
)
from rfx.geometry.rasterize_grid import coords_from_uniform_grid
from rfx.fidelity import fidelity_report


def main():
    path = Path("scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor.py")
    spec = importlib.util.spec_from_file_location("anchor_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for label, dx in module.DX_GRID:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = module._build_sim_no_solve(dx)
            rz = realized(sim)
            coords = coords_from_uniform_grid(rz.grid)
            report = fidelity_report(sim, print_report=False)
        substrate = next(a for a in report[1]["axes"] if a["axis"] == "z")
        trace = next(a for a in report[2]["axes"] if a["axis"] == "y")
        ports = []
        for pe in sim._msl_ports:
            port = msl_port_from_entry(pe)
            span = msl_cross_section_span(rz.grid, port)
            walls = rz.wall_planes(2, ij=(span["i_feed"], span["w_centre"]))
            try:
                validate_msl_port_geometry(
                    rz.grid, port, pec_edge_masks=rz.edge_masks,
                    pec_faces=sim._boundary_spec.pec_faces(),
                    periodic=sim._periodic_flags(), name=pe.name,
                )
                status = "valid"
            except Exception as exc:
                status = f"{type(exc).__name__}: {exc}"
            z, y = np.asarray(coords.z), np.asarray(coords.y)
            occupied = np.flatnonzero(np.asarray(rz.edge_masks[0])[
                span["i_feed"], :, span["n_hi"]])
            ports.append(dict(
                direction=pe.direction,
                normal_nodes=[span["n_lo"], span["n_hi"]],
                source_gap_um=float((z[span["n_hi"]] - z[span["n_lo"]]) * 1e6),
                trace_wall_um=(z[walls] * 1e6).tolist(),
                longitudinal_edge_width_nodes=[int(occupied.min()), int(occupied.max())],
                longitudinal_edge_span_um=float((y[occupied.max()] - y[occupied.min()]) * 1e6),
                source_width_nodecount=span["w_hi"] - span["w_lo"] + 1,
                source_width_node_extent_um=float((y[span["w_hi"]] - y[span["w_lo"]]) * 1e6),
                validation=status,
            ))
        rows.append(dict(
            label=label, dx_um=dx * 1e6,
            dielectric_bbox_h_um=substrate["realized_extent_um"],
            pec_body_w_um=trace["realized_extent_um"], ports=ports,
        ))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
