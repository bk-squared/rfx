#!/usr/bin/env python3
"""Issue #1066: who opts into smoothing on the waveguide lane, and what the pad holds.

Two measurements, both cheap and neither solving:

A1, the CALLER CENSUS. ``subpixel_smoothing`` defaults to False on
``compute_waveguide_s_matrix`` (``rfx/sparams/waveguide.py``), so the block that
builds ``shape_eps_pairs`` is only entered when a caller opts in. Counting those
callers by grep is wrong and was wrong: the calls span lines, and a line-oriented
grep found one where there are four. This walks the AST instead.

A3, the PAD. For a dielectric that runs into a port-side CPML face, it builds the
pairs BOTH ways on one sim and one grid -- the lane's own list comprehension and
``rfx.geometry.smoothing.smoothed_shape_pairs``, which the three runner sites use
since #1043 stage B -- smooths each, and prints the permittivity down the x-hi
pad. Build-only: nothing time-steps, so this is a statement about what the solver
would be handed, not about any S-parameter.

Run::

    python scripts/diagnostics/waveguide_lane_pad_continuation_census.py
    python scripts/diagnostics/waveguide_lane_pad_continuation_census.py --census
    python scripts/diagnostics/waveguide_lane_pad_continuation_census.py --pad
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: The lane's public entry points. ``compute_waveguide_s_matrix`` is the one
#: that takes ``subpixel_smoothing``; the others are listed so a future rename
#: or split shows up here as a count change rather than as silence.
LANE_ENTRY_POINTS = frozenset({
    "compute_waveguide_s_matrix",
    "compute_waveguide_s_params",
    "compute_waveguide_s_matrix_flux",
})

_SKIP_PREFIXES = (".venv", "build/", "dist/")


def _callee(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def census() -> list[tuple[str, int, str, str, bool]]:
    """Every call into the lane that passes ``subpixel_smoothing``."""
    hits: list[tuple[str, int, str, str, bool]] = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(_SKIP_PREFIXES):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _callee(node) not in LANE_ENTRY_POINTS:
                continue
            for kw in node.keywords:
                if kw.arg != "subpixel_smoothing":
                    continue
                if isinstance(kw.value, ast.Constant):
                    shown, truthy = repr(kw.value.value), bool(kw.value.value)
                else:
                    # A name or expression: it CAN be truthy, so it counts.
                    shown, truthy = ast.unparse(kw.value), True
                hits.append((rel, node.lineno, _callee(node), shown, truthy))
    return hits


def pad_comparison() -> dict:
    """Lane pairs vs runner pairs, smoothed, on a pad-reaching dielectric."""
    import contextlib
    import io

    import numpy as np
    import jax.numpy as jnp

    import rfx
    from rfx import Box, Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.smoothing import compute_smoothed_eps, smoothed_shape_pairs

    a_wg, b_wg, dx = 22.86e-3, 10.16e-3, 1.27e-3
    lx = 60 * dx
    freqs = jnp.asarray([8.5e9, 10.0e9, 11.5e9])

    sim = Simulation(
        freq_max=12e9, domain=(lx, a_wg, b_wg), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
    )
    sim.add_material("slab", eps_r=4.0)
    # Runs INTO the x-hi face, i.e. into the port-side absorber.
    sim.add(Box((lx / 2.0, 0.0, 0.0), (lx, a_wg, b_wg)), material="slab")
    for pos, direction, nm in ((6 * dx, "+x", "left"),
                               (lx - 6 * dx, "-x", "right")):
        sim.add_waveguide_port(pos, direction=direction, freqs=freqs,
                               f0=1e10, name=nm)

    with contextlib.redirect_stdout(io.StringIO()):
        grid = sim._build_grid()
    lane_pairs = [(e.shape, sim._resolve_material(e.material_name).eps_r)
                  for e in sim._geometry]
    runner_pairs, unextendable = smoothed_shape_pairs(sim, grid)
    lane = np.asarray(compute_smoothed_eps(grid, lane_pairs,
                                           background_eps=1.0)[0])
    runner = np.asarray(compute_smoothed_eps(grid, runner_pairs,
                                             background_eps=1.0)[0])
    pad = int(grid.face_pads[1])
    jm, km = lane.shape[1] // 2, lane.shape[2] // 2
    diff = np.abs(lane - runner)
    return {
        "rfx_file": rfx.__file__,
        "grid_shape": tuple(int(v) for v in lane.shape),
        "boundary": getattr(sim, "_boundary", None),
        "cpml_layers": int(getattr(sim, "_cpml_layers", 0)),
        "x_hi_pad_cells": pad,
        "unextendable": list(unextendable),
        "column": [(int(i), float(lane[i, jm, km]), float(runner[i, jm, km]))
                   for i in range(lane.shape[0] - pad - 2, lane.shape[0])],
        "max_abs_diff": float(diff.max()),
        "cells_differing": int((diff > 1e-12).sum()),
        "cells_total": int(diff.size),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", action="store_true", help="A1 only")
    parser.add_argument("--pad", action="store_true", help="A3 only")
    args = parser.parse_args(argv)
    both = not (args.census or args.pad)

    if both or args.census:
        hits = census()
        print("A1 -- callers passing subpixel_smoothing into the lane")
        print(f"{'file':60s} {'line':>5}  {'value':16s} truthy")
        print("-" * 96)
        for rel, line, _callee_name, shown, truthy in hits:
            print(f"{rel:60s} {line:5d}  {shown:16s} {truthy}")
        print(f"\n{len(hits)} call site(s); "
              f"{sum(1 for h in hits if h[4])} truthy.\n")

    if both or args.pad:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        result = pad_comparison()
        print("A3 -- eps_xx down the x-hi pad, mid y/z (build-only, no solve)")
        print(f"rfx.__file__ = {result['rfx_file']}")
        print(f"grid {result['grid_shape']}  boundary={result['boundary']!r}  "
              f"cpml_layers={result['cpml_layers']}  "
              f"x-hi pad = {result['x_hi_pad_cells']} cells  "
              f"unextendable={result['unextendable']}")
        print(f"{'i':>5} {'lane':>9} {'runner':>9}  differ")
        for i, lane_v, runner_v in result["column"]:
            mark = "YES" if abs(lane_v - runner_v) > 1e-12 else ""
            print(f"{i:5d} {lane_v:9.4f} {runner_v:9.4f}  {mark}")
        print(f"\nmax |lane - runner| = {result['max_abs_diff']:.6f}; "
              f"{result['cells_differing']} of {result['cells_total']} cells "
              f"differ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
