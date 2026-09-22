"""Are the realized conductor edges the same before and after the clamp?

No solve. Builds every board the committed record was measured on, stamps the
line each way -- the unclamped 1 mm wall the record was produced with, and the
clamped thickness this change introduces -- and compares the PEC edge masks the
runner would receive, bit for bit. If they are identical the record stays valid
without re-solving.
"""
import pathlib
import sys

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/scripts/diagnostics")

from coax_conductor_oracle_ladder import BOARDS, build, dx_of, GATE_RUNG  # noqa: E402
from rfx.boundaries.pec import realized_pec_edge_masks  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    SHELL_THICKNESS_M, SMA_OUTER_RADIUS, SMA_PIN_RADIUS, stamp_coaxial_line,
)

CASES = [(b, r) for b in ("thru_long", "load_long", "thru_gate", "load_gate")
         for r in (GATE_RUNG, 4.0, 6.0, 9.0)]

print(f"{'board':10s} {'rung':>7s} {'dx um':>8s} {'clamped thk':>12s} "
      f"{'cells':>9s} {'edges same':>11s}")
bad = 0
for board, rung in CASES:
    sim = build(board, rung)
    grid = sim._build_grid()
    mats, _, _ = sim._build_materials(grid)
    kw = dict(center_xy=(sim._coaxial_ports[0].position[0],
                         sim._coaxial_ports[0].position[1]),
              z_lo_index=int(grid.pad_z_lo) + 4,
              z_hi_index=int(grid.shape[2]) - int(grid.pad_z_hi) - 2,
              pin_radius=SMA_PIN_RADIUS, outer_radius=SMA_OUTER_RADIUS)
    # after: the shipped default, now clamped inside the function
    _, _, cells_after = stamp_coaxial_line(grid, mats, **kw)
    # before: the unclamped 1 mm the record was produced with. The clamp only
    # ever REDUCES, so asking for exactly 1 mm reproduces the old geometry
    # wherever the board had room for it.
    _, _, cells_before = stamp_coaxial_line(grid, mats,
                                            shell_thickness_m=SHELL_THICKNESS_M, **kw)
    ea = realized_pec_edge_masks(np.asarray(cells_after), sheets=(), wires=(),
                                 periodic=(False, False, False))
    eb = realized_pec_edge_masks(np.asarray(cells_before), sheets=(), wires=(),
                                 periodic=(False, False, False))
    same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(ea, eb))
    if not same:
        bad += 1
    n = int(np.asarray(cells_after).sum())
    print(f"{board:10s} {rung:7.3f} {dx_of(rung)*1e6:8.2f} "
          f"{'(unchanged)' if same else '(DIFFERS)':>12s} {n:9d} {str(same):>11s}")
print(f"\nboards compared: {len(CASES)}, masks differing: {bad}")
sys.exit(1 if bad else 0)
