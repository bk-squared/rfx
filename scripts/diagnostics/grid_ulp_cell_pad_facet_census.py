#!/usr/bin/env python3
"""Issue #1070: the one-ULP extra cell, and the vacuum it leaves in a pad.

Rebuilds the issue's rig from its recipe -- the #801 isolated-patch fixture,
RO4003C at h = 0.787 mm, substrate Box across the whole declared domain,
dx = h/4, cpml_layers = 8, boundary "cpml", lateral domain ``38*h + 2*(pad*h)``
on x and ``23*h + 2*(pad*h)`` on y -- and reports, per pad value:

* ``domain/dx`` as a float, and whether ``ceil`` bought a cell;
* how many interior nodes at each hi face the rasterizer leaves EMPTY before
  the CPML pad extension runs;
* what the #627a fallback sees in the two columns it inspects;
* whether the hi pad ends up vacuum.

Build-only: nothing time-steps. Run it on a tree before and after the fix.

    python scripts/diagnostics/grid_ulp_cell_pad_facet_census.py
    python scripts/diagnostics/grid_ulp_cell_pad_facet_census.py --falsifier
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

H = 0.787e-3          # RO4003C substrate height, the #801 patch fixture
EPS_R = 3.38
DX = H / 4.0
CPML_LAYERS = 8
PADS = (6, 8, 10, 12)


def _build(pad_h: int, shrink_ulp: int = 0):
    import numpy as np

    from rfx import Box, Simulation

    dom_x = 38 * H + 2 * (pad_h * H)
    dom_y = 23 * H + 2 * (pad_h * H)
    dom_z = 16 * H
    for _ in range(shrink_ulp):
        dom_x = math.nextafter(dom_x, 0.0)
        dom_y = math.nextafter(dom_y, 0.0)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(dom_x, dom_y, dom_z), dx=DX,
                         boundary="cpml", cpml_layers=CPML_LAYERS)
        sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
        sim.add(Box((0.0, 0.0, 6 * H), (dom_x, dom_y, 8 * H)),
                material="ro4003c")
        grid = sim._build_grid()
        filled = np.asarray(sim._assemble_materials(grid)[0].eps_r)
        raw = np.asarray(sim._assemble_materials(
            grid, include_cpml_pad_extension=False)[0].eps_r)
    return grid, filled, raw, (dom_x, dom_y, dom_z)


def _substrate_plane(grid, raw) -> int:
    for k in range(raw.shape[2]):
        if abs(raw[grid.face_pads[0] + 2, grid.face_pads[2] + 2, k]
               - EPS_R) < 1e-12:
            return k
    raise SystemExit("no substrate plane found -- the rig did not build")


def _axis_view(arr, axis: int, k: int):
    return arr[:, arr.shape[1] // 2, k] if axis == 0 else arr[arr.shape[0] // 2, :, k]


def _empty_tail(row, lo_pad: int, n_interior: int) -> int:
    out = 0
    for value in row[lo_pad:lo_pad + n_interior][::-1]:
        if abs(float(value) - 1.0) < 1e-12:
            out += 1
        else:
            break
    return out


def census() -> None:
    import rfx

    print(f"rfx.__file__ = {rfx.__file__}")
    print(f"h = {H!r}  dx = {DX!r}  eps_r = {EPS_R}  "
          f"cpml_layers = {CPML_LAYERS}\n")
    print(f"{'pad':>5} {'axis':>5} {'domain/dx':>22} {'ceil+1?':>8} "
          f"{'grid':>18} {'empty tail':>11} {'hi pad':>10}")
    for pad in PADS:
        grid, filled, raw, dom = _build(pad)
        k = _substrate_plane(grid, raw)
        for axis, name in ((0, "x"), (1, "y")):
            ratio = dom[axis] / DX
            lo_pad = grid.face_pads[2 * axis]
            hi_pad = grid.face_pads[2 * axis + 1]
            n_int = filled.shape[axis] - lo_pad - hi_pad
            tail = _empty_tail(_axis_view(raw, axis, k), lo_pad, n_int)
            edge = float(_axis_view(filled, axis, k)[-1])
            print(f"{pad:>4}h {name:>5} {ratio!r:>22} "
                  f"{str(math.ceil(ratio) != ratio):>8} "
                  f"{str(list(grid.shape)):>18} {tail:>11} "
                  f"{('VACUUM' if abs(edge - 1.0) < 1e-12 else f'{edge:.2f}'):>10}")

    print("\nThe two columns the #627a fallback inspects "
          "(substrate mid-plane, pre-extension):")
    for pad in PADS:
        grid, filled, raw, dom = _build(pad)
        k = _substrate_plane(grid, raw)
        for axis, name in ((0, "x"), (1, "y")):
            lo_pad = grid.face_pads[2 * axis]
            hi_pad = grid.face_pads[2 * axis + 1]
            row = _axis_view(raw, axis, k)
            hi = filled.shape[axis] - hi_pad
            cols = [float(row[hi - 1 - d]) for d in range(3)]
            edge = float(_axis_view(filled, axis, k)[-1])
            verdict = ("PAD IS VACUUM" if abs(edge - 1.0) < 1e-12
                       else "pad carries the substrate")
            print(f"  pad {pad:>2}h {name}: declared {dom[axis] / DX!r:<22} "
                  f"| edge {cols[0]:.2f}, one in {cols[1]:.2f}, "
                  f"two in {cols[2]:.2f} -> {verdict}")


def falsifier() -> None:
    print("Nudging the declared domain length down by one ULP:")
    for pad in PADS:
        for label, n in (("as declared", 0), ("-1 ULP", 1)):
            grid, filled, raw, dom = _build(pad, shrink_ulp=n)
            k = _substrate_plane(grid, raw)
            faces = []
            for axis, name in ((0, "x"), (1, "y")):
                if abs(float(_axis_view(filled, axis, k)[-1]) - 1.0) < 1e-12:
                    faces.append(f"{name}-hi")
            print(f"  pad {pad:>2}h  {label:<12} grid {list(grid.shape)}  "
                  f"vacuum facet on {faces if faces else 'no face'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--falsifier", action="store_true",
                        help="the -1 ULP comparison only")
    args = parser.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if not args.falsifier:
        census()
        print()
    falsifier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
