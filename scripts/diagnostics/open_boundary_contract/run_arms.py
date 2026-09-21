"""#801: score the thin-absorber ring-down arms on ONE source tree.

Usage:  python run_arms.py <tree_root> <out_json>

Environment (both optional):
  ARMS=name1,name2   run only those arms (same definitions as below)
  CEIL_SIZING=1      put the pre-#1136 grid sizing back (plain ceil of domain/dx) and
                     nothing else, so an arm runs on today's code with the old realized
                     grid. If assembly then refuses, the refusal is recorded as the result.

The builder and both metrics are the committed oracle's own
(tests/oracle/test_lossless_open_domain_ringdown_does_not_grow.py), imported
from <tree_root>, so nothing here re-implements the score.

Physics: a lossless patch on a grounded substrate inside absorbing boundaries,
struck once. The field can only ring down. An arm "grows" when the late-time
envelope climbs instead.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

tree_root, out_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree_root)

import jax  # noqa: E402
import rfx  # noqa: E402
from tests.oracle import test_lossless_open_domain_ringdown_does_not_grow as oracle  # noqa: E402

# (name, cells per substrate thickness n, lateral pad in h, absorber layers)
ARMS = [
    ("n3_pad10_cpml6", 3, 10, 6),   # recorded 2026-09-15: GROWS
    ("n2_pad10_cpml4", 2, 10, 4),   # recorded 2026-09-15: GROWS
    ("n3_pad10_cpml8", 3, 10, 8),   # recorded 2026-09-15: settles
    ("n2_pad0_cpml4", 2, 0, 4),     # recorded 2026-09-15: settles (no lateral pad)
    ("n4_pad10_cpml8", 4, 10, 8),   # the committed gate point: settles
]

only = [s for s in os.environ.get("ARMS", "").split(",") if s]
if only:
    unknown = sorted(set(only) - {a[0] for a in ARMS})
    if unknown:
        raise SystemExit(f"unknown arm name(s): {unknown}")
    ARMS = [a for a in ARMS if a[0] in only]

ceil_sizing = os.environ.get("CEIL_SIZING", "") == "1"
if ceil_sizing:
    import rfx.grid as grid_mod
    if not hasattr(grid_mod, "cells_spanning"):
        raise SystemExit("CEIL_SIZING=1 asked for, but this tree has no rfx.grid.cells_spanning")
    grid_mod.cells_spanning = lambda length, dx, **_kw: int(math.ceil(length / dx))

print("rfx from", rfx.__file__, "| devices", jax.devices(), "| x64", jax.config.jax_enable_x64,
      "| ceil_sizing", ceil_sizing, flush=True)
rows = []
for name, n, pad_h, cpml in ARMS:
    sim = oracle._build(n=n, pad_h=pad_h, cpml=cpml)
    grid = sim._build_grid()
    row = dict(arm=name, n=n, pad_h=pad_h, cpml_layers=cpml,
               grid=[int(grid.nx), int(grid.ny), int(grid.nz)])
    try:
        result = sim.run(num_periods=oracle.NUM_PERIODS, skip_preflight=True)
    except Exception as exc:  # recorded, not hidden: a refusal is a result here
        row.update(refused=f"{type(exc).__name__}: {str(exc)[:600]}")
        rows.append(row)
        print(json.dumps(row), flush=True)
        continue
    series = np.asarray(result.time_series)
    rates = oracle._late_time_log_rate_per_step(series)
    row.update(steps=int(series.shape[0]), rates_per_step=[float(r) for r in rates],
               worst_rate_per_step=float(max(rates)),
               settling_db=float(oracle._settling_db(series)))
    rows.append(row)
    print(json.dumps(row), flush=True)

with open(out_json, "w") as fh:
    json.dump(dict(tree_root=tree_root, rfx_file=rfx.__file__,
                   x64=bool(jax.config.jax_enable_x64), ceil_sizing=ceil_sizing,
                   num_periods=float(oracle.NUM_PERIODS), arms=rows), fh, indent=1)
print("wrote", out_json, flush=True)
