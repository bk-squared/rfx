"""#831 / #1043 — what a pad replication on the SMOOTHED array actually writes.

No FDTD.  Companion to ``probe_pad_replication.py``, written after that probe
ran, because its instrument counter reported "159/201 both ways" and hid the
one thing that matters about the fix's shape.

THE FINDING.  ``extend_cpml_pad_materials`` copies the **interior-edge slice**
outward.  On the array ``_assemble_materials`` builds, that edge column is the
material (``eps_r = 12``) and replicating it is right.  On the array
``compute_smoothed_eps`` builds, the edge column is the **Kottke half-cell**
(``eps_r = 6.5`` for a guide that ends flush with the domain edge), so the naive
reuse fills the whole pad with 6.5 -- a uniform medium that is neither the guide
nor vacuum.

So a real fix cannot just pipe the smoothing output through
``extend_cpml_pad_materials``.  It has to either smooth the ALREADY-EXTENDED
array, or source the replication one column inward, the way the #627a hi-face
fallback already does for its own case.

This script prints and records both, side by side, so #1043 carries the shape
rather than a description of it.

Run:

    PYTHONPATH=<repo> python3 \
      scripts/diagnostics/cv03_seam_facet/pad_replication_shape.py \
      --output scripts/diagnostics/_artifacts/cv03_seam_facet/pad_replication_shape.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _cv01_module():
    path = REPO / "scripts" / "diagnostics" / "cv01_cpml_flux_selfcheck.py"
    spec = importlib.util.spec_from_file_location("cv01_selfcheck", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cv01_selfcheck"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(cv01):
    """Build cv01's committed sim and return (built_row, smoothed_row, grid)."""
    import rfx
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.geometry.smoothing import compute_smoothed_eps

    sim = Simulation(freq_max=0.25 * cv01.C0 / cv01.a,
                     domain=(cv01.sx, cv01.sy, cv01.dx), dx=cv01.dx,
                     boundary=BoundarySpec.uniform("cpml"),
                     cpml_layers=20, mode="2d_tmz")
    sim.add_material("wg", eps_r=cv01.eps_wg)
    sim.add(rfx.Box((0, cv01.wg_y - cv01.w_wg / 2, 0),
                    (cv01.sx, cv01.wg_y + cv01.w_wg / 2, cv01.dx)),
            material="wg")
    grid = sim._build_grid()
    mats, *_ = sim._assemble_materials(grid)
    pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
             for e in sim._geometry]
    _, _, az = compute_smoothed_eps(grid, pairs, background_eps=1.0)
    jy = int(round(cv01.wg_y / cv01.dx)) + int(grid.pad_y_lo)
    return (np.asarray(mats.eps_r)[:, jy, 0].astype(float),
            np.asarray(az)[:, jy, 0].astype(float), grid)


def _naive_extend(row, plx, phx):
    """What extend_cpml_pad_materials writes: replicate the interior edge."""
    out = row.copy()
    out[:plx] = row[plx]
    out[len(row) - phx:] = row[len(row) - phx - 1]
    return out


def _sourced_one_inward(row, plx, phx):
    """Source one column inward, the shape #627a already uses for its case."""
    out = row.copy()
    out[:plx] = row[plx + 1]
    out[len(row) - phx:] = row[len(row) - phx - 2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cv01 = _cv01_module()
    built, smoothed, grid = _row(cv01)
    plx, phx = int(grid.pad_x_lo), int(grid.pad_x_hi)
    nx = len(smoothed)

    variants = {
        "committed (no replication)": smoothed,
        "naive extend_cpml_pad_materials on the smoothed array":
            _naive_extend(smoothed, plx, phx),
        "sourced one column inward": _sourced_one_inward(smoothed, plx, phx),
        "reference: _assemble_materials (the interior route)": built,
    }

    out = {
        "schema": "issue831-pad-replication-shape-v1",
        "issue": 831, "landing_issue": 1043, "runs_fdtd": False,
        "provenance": {"commit": _git("rev-parse", "HEAD"),
                       "branch": _git("rev-parse", "--abbrev-ref", "HEAD")},
        "rig": "cv01 committed build, cpml, 20 layers, guide centre row",
        "nx": nx, "pad_x_lo": plx, "pad_x_hi": phx,
        "eps_wg": float(cv01.eps_wg),
        "variants": {},
    }
    for name, row in variants.items():
        out["variants"][name] = {
            "pad_lo_value": float(row[0]),
            "pad_hi_value": float(row[-1]),
            "interior_edge_lo": float(row[plx]),
            "interior_edge_hi": float(row[nx - phx - 1]),
            "n_cells_at_eps_wg": int(np.sum(np.isclose(row, cv01.eps_wg,
                                                       rtol=1e-3))),
            "pad_is_the_material": bool(
                np.isclose(row[0], cv01.eps_wg, rtol=1e-3)),
            "row_around_lo_seam": [float(v) for v in row[plx - 2:plx + 3]],
        }
        v = out["variants"][name]
        print(f"{name:58s} pad={v['pad_lo_value']:6.3f} "
              f"seam={v['row_around_lo_seam']} "
              f"n(eps=12)={v['n_cells_at_eps_wg']}")

    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
