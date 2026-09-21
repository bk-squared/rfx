#!/usr/bin/env python3
"""The RT5880 patch crossval case solved on edge-aware in-plane mesh lines,
held against the committed openEMS leg.

Runs the case's OWN builder and runner (``validation/crossval/
15_patch_antenna_rt5880.py``: same geometry, feed, waveform, S11 extraction)
with one change -- ``Simulation`` additionally receives the
``dx_profile`` / ``dy_profile`` from :func:`rfx.mesh_edges.edge_aware_profiles`
at the case's own cell size. z stays the case's uniform cells. The feed and
patch-centre planes are passed as faces so they stay on nodes.

The committed ``rfx.json`` ran 45 source periods, so ``--uniform`` re-runs the
uniform grid at this script's 70 into ``rfx_uniform_70_periods.json``: the two
arms then share the record length in seconds. The runner counts its steps
with the UNIFORM grid's time step; the edge-aware grid's step is shorter, so
its period count is scaled up by the ratio to keep the same seconds.

Writes ``_15_patch_results/rfx_edge_aware_mesh.json`` (a named arm; the case's
``rfx.json`` is not touched) and prints the S11 dips side by side.

    python scripts/diagnostics/patch_vs_openems_edge_mesh.py [--uniform]
"""
from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from rfx import Box  # noqa: E402
from rfx.mesh_edges import edge_aware_profiles  # noqa: E402

CASE = _REPO / "validation" / "crossval" / "15_patch_antenna_rt5880.py"
OUT_NAME = "rfx_edge_aware_mesh.json"
UNIFORM_NAME = "rfx_uniform_70_periods.json"
PERIODS = 70.0


def main() -> int:
    spec = importlib.util.spec_from_file_location("cv_patch", CASE)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    if "--uniform" in sys.argv:
        case.run_rfx(PERIODS, 181, False, out_name=UNIFORM_NAME)
        return 0
    cx, cy = case.DOM_X / 2, case.DOM_Y / 2
    z = 0.0
    ground = Box((cx - case.GP_X / 2, cy - case.GP_Y / 2, z),
                 (cx + case.GP_X / 2, cy + case.GP_Y / 2, z))
    patch = Box((cx - case.L_PATCH / 2, cy - case.W_PATCH / 2, z),
                (cx + case.L_PATCH / 2, cy + case.W_PATCH / 2, z))
    prof = edge_aware_profiles(
        (case.DOM_X, case.DOM_Y, case.DOM_Z), case.DX, sheets=[ground, patch],
        faces={"x": [cx + case.FEED_OFFSET_X, cx], "y": [cy]}, axes="xy")
    print({k: (len(v), float(v.min()), float(v.max())) for k, v in prof.items()})
    # The case imports ``Simulation`` from ``rfx`` inside its builder, and its
    # runner reads the z stack, the feed and dt from a UNIFORM grid
    # (``sim._build_grid()``), which a simulation with in-plane profiles
    # refuses. z is uniform and identical in both, so those checks are given
    # the case's own uniform twin; the SOLVE runs on the profiled simulation.
    import rfx
    uniform_cls = rfx.Simulation
    build = case.build_rfx_sim
    twins = {}

    def build_with_profiles(**kw):
        rfx.Simulation = uniform_cls
        twin = build(**kw)[0]
        rfx.Simulation = functools.partial(uniform_cls, **prof)
        try:
            sim, patch_shape, geom = build(**kw)
        finally:
            rfx.Simulation = uniform_cls
        twins[id(sim)] = twin
        sim._build_grid = twin._build_grid
        return sim, patch_shape, geom

    case.build_rfx_sim = build_with_profiles
    for name in ("assert_realized_stack", "assert_galvanic_feed"):
        orig = getattr(case, name)
        setattr(case, name, (lambda o: lambda sim, *a, **k: o(
            twins.get(id(sim), sim), *a, **k))(orig))
    probe_sim = build_with_profiles()[0]
    dt_uniform = float(twins[id(probe_sim)]._build_grid().dt)
    dt_profiled = float(probe_sim._build_nonuniform_grid().dt)
    periods = PERIODS * dt_uniform / dt_profiled
    print(f"dt uniform {dt_uniform:.6e} s, profiled {dt_profiled:.6e} s; "
          f"{periods:.3f} runner periods = {PERIODS:g} periods of record")
    case.run_rfx(periods, 181, False, out_name=OUT_NAME)
    res = Path(case.RES_DIR)
    legs = {"openEMS (committed)": json.loads((res / "openems.json").read_text()),
            "rfx uniform (committed, 45 periods)": json.loads((res / "rfx.json").read_text()),
            "rfx edge-aware": json.loads((res / OUT_NAME).read_text())}
    if (res / UNIFORM_NAME).exists():
        legs["rfx uniform (70 periods)"] = json.loads((res / UNIFORM_NAME).read_text())
    for name, leg in legs.items():
        print(f"{name:36s} S11 dip {leg['f_dip_hz'] / 1e9:.4f} GHz "
              f"({leg['s11_dip_db']:.1f} dB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
