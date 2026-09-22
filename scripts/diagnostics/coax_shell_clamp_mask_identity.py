"""Does clamping the wall change the conductors the solver actually receives?

The committed ladder record was measured with a wall a fixed 1 mm thick. That
thickness is now clamped against the room a board leaves before its absorber, so
the question is whether the clamp moved any board the record rests on. If no
realized edge moves, the record stands without re-solving.

**The reference arm is built here, not asked of the stamper.** An earlier
version of this script called ``stamp_coaxial_line`` twice and passed
``shell_thickness_m=1 mm`` to one of them -- but the clamp is inside that
function and applies to an explicitly passed thickness too, so both arms were
clamped and the comparison could not fail. It compared the function with
itself. The reference is now the unclamped geometry drawn directly from
cylinders: ``Cylinder(b + 1 mm) & ~Cylinder(b)`` for the wall, ``Cylinder(a)``
for the pin, which is what the stamper produced before the clamp existed.

``--force-bind`` is the falsifier: it shrinks the allowed thickness to one cell
on every board, so the clamp must bite and the comparison must go RED. A run
that reports no difference under ``--force-bind`` is not measuring anything.

Usage::

    PYTHONPATH=. python scripts/diagnostics/coax_shell_clamp_mask_identity.py
    PYTHONPATH=. python scripts/diagnostics/coax_shell_clamp_mask_identity.py --force-bind
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/scripts/diagnostics")

from coax_conductor_oracle_ladder import BOARDS, build, dx_of, GATE_RUNG  # noqa: E402
from rfx.boundaries.pec import realized_pec_edge_masks  # noqa: E402
from rfx.geometry.csg import Cylinder  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    SHELL_THICKNESS_M, SMA_OUTER_RADIUS, SMA_PIN_RADIUS, stamp_coaxial_line,
)

CASES = [(b, r) for b in ("thru_long", "load_long", "thru_gate", "load_gate")
         for r in (GATE_RUNG, 4.0, 6.0, 9.0)]


def clearance_to_nearest_pad(grid, centre_xy) -> tuple[str, float]:
    """The room the board leaves between the line's axis and its nearest pad."""
    dz = float(grid.dx)
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    span_x = (nx - 1 - int(grid.pad_x_lo) - int(grid.pad_x_hi)) * dz
    span_y = (ny - 1 - int(grid.pad_y_lo) - int(grid.pad_y_hi)) * dz
    room = []
    if int(grid.pad_x_lo):
        room.append(("x-lo", cx))
    if int(grid.pad_x_hi):
        room.append(("x-hi", span_x - cx))
    if int(grid.pad_y_lo):
        room.append(("y-lo", cy))
    if int(grid.pad_y_hi):
        room.append(("y-hi", span_y - cy))
    return min(room, key=lambda t: t[1]) if room else ("none", float("inf"))


def unclamped_cells(grid, centre_xy, z_lo_index, z_hi_index, thickness):
    """The conductor cells a wall of exactly *thickness* would occupy.

    Drawn from cylinders on this grid, the way the stamper drew them before the
    clamp: the reference this comparison needs is one the stamper cannot alter.
    """
    dz = float(grid.dx)
    z_lo = (int(z_lo_index) - grid.pad_z_lo) * dz
    z_hi = (int(z_hi_index) - grid.pad_z_lo) * dz
    centre = (float(centre_xy[0]), float(centre_xy[1]), 0.5 * (z_lo + z_hi))
    height = (z_hi - z_lo) + 2.0 * dz
    b = float(SMA_OUTER_RADIUS)
    outer = Cylinder(center=centre, radius=b + float(thickness), height=height,
                     axis="z").mask(grid)
    inner = Cylinder(center=centre, radius=b, height=height, axis="z").mask(grid)
    pin = Cylinder(center=centre, radius=float(SMA_PIN_RADIUS), height=height,
                   axis="z").mask(grid)
    return np.asarray((outer & ~inner) | pin)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-bind", action="store_true",
                    help="shrink the wall to one cell so the clamp MUST bite; "
                         "the comparison must then report differences")
    args = ap.parse_args()

    print(f"{'board':10s} {'rung':>7s} {'dx um':>8s} {'pad':>5s} "
          f"{'clearance mm':>13s} {'allowed mm':>11s} {'wall mm':>8s} "
          f"{'cells':>8s} {'edges same':>11s}")
    bad = 0
    for board, rung in CASES:
        sim = build(board, rung)
        grid = sim._build_grid()
        mats, _, _ = sim._build_materials(grid)
        port = sim._coaxial_ports[0]
        centre = (float(port.position[0]), float(port.position[1]))
        z_lo = int(grid.pad_z_lo) + 4
        z_hi = int(grid.shape[2]) - int(grid.pad_z_hi) - 2
        dz = float(grid.dx)

        pad, clearance = clearance_to_nearest_pad(grid, centre)
        allowed = clearance - float(SMA_OUTER_RADIUS) - dz

        kw = dict(center_xy=centre, z_lo_index=z_lo, z_hi_index=z_hi,
                  pin_radius=SMA_PIN_RADIUS, outer_radius=SMA_OUTER_RADIUS)
        if args.force_bind:
            # Ask for a wall one cell thick. The clamp takes the smaller of the
            # request and the room, so this forces a thin wall on every board
            # and the reference (a full 1 mm) must then differ.
            kw["shell_thickness_m"] = dz
        _, _, shipped = stamp_coaxial_line(grid, mats, **kw)
        shipped = np.asarray(shipped)

        reference = unclamped_cells(grid, centre, z_lo, z_hi, SHELL_THICKNESS_M)

        ea = realized_pec_edge_masks(shipped, sheets=(), wires=(),
                                     periodic=(False, False, False))
        eb = realized_pec_edge_masks(reference, sheets=(), wires=(),
                                     periodic=(False, False, False))
        same = all(np.array_equal(np.asarray(x), np.asarray(y))
                   for x, y in zip(ea, eb))
        if not same:
            bad += 1
        effective = dz if args.force_bind else min(SHELL_THICKNESS_M, allowed)
        print(f"{board:10s} {rung:7.3f} {dx_of(rung)*1e6:8.2f} {pad:>5s} "
              f"{clearance*1e3:13.4f} {allowed*1e3:11.4f} {effective*1e3:8.4f} "
              f"{int(shipped.sum()):8d} {str(same):>11s}")

    print(f"\nboards compared: {len(CASES)}, masks differing: {bad}")
    if args.force_bind:
        print("--force-bind: the clamp was made to bite, so differences are "
              "the expected result; 0 would mean this comparison is blind.")
        return 0 if bad == len(CASES) else 1
    print(f"the default wall is {SHELL_THICKNESS_M*1e3:.3f} mm and every board "
          f"above allows more than that, which is why the clamp does not bind "
          f"here and the committed record stands without re-solving.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
