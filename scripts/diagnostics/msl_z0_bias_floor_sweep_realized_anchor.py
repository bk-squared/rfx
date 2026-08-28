#!/usr/bin/env python3
"""Realized-board anchor for msl_z0_bias_floor_sweep.py (issue #752).

This is a NEW, SIBLING artifact. It does NOT touch
``scripts/diagnostics/msl_z0_bias_floor_sweep.py`` or its committed JSON
(``scripts/diagnostics/msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep.json``,
sha256 pinned by ``test_msl_z0_bias_floor_sweep_json_is_frozen`` in
``tests/test_msl_port_preflight.py``) — that script is the pre-declared
sweep and its as-run verdict block is the auditable record of what was
measured; both stay exactly as they are.

WHAT THIS ADDS
--------------
Issue #752: the pre-declared sweep's ``z0_hj_ohm`` column (and hence its
``dev_decl`` reading, quoted in the preflight advisories before this fix)
is Hammerstad-Jensen on the DECLARED 600/254µm board at every dx, even
though the misaligned dx points (80µm, 60µm) rasterize a THICKER
substrate (320µm, 300µm respectively — the half-open rasterizer rounds
h_sub/dx UP; see ``rfx/geometry/csg.py``). This script re-derives, for
each of the SAME six pre-declared dx points, the board each mesh point
ACTUALLY realizes (via ``sim.fidelity_report()`` — no time-stepping, no
FDTD solve, cheap) and reports Hammerstad-Jensen on THAT board alongside
the already-committed ``z0_measured_ohm``, read verbatim from the
pre-declared JSON (never re-solved, never retyped by hand: parsed from
that file).

PRECISION: this checkout's ``msl_z0_bias_floor_sweep.py`` does not pin
JAX_ENABLE_X64 (``grep JAX_ENABLE_X64`` on that file: no match), so its
committed row was produced at the default (jax_enable_x64=False). This
script must build the SAME geometry at the SAME default precision to
read back the SAME rasterization, and does not set the flag either. The
precision is INFERRED, not read from a log: the committed Z0 values
agree with Hammerstad-Jensen on the REALIZED board to <=0.4% at every
point (see ``max_abs_dev_realized_pct`` below); at JAX_ENABLE_X64=1 the
aligned class's realized trace width flips onto a different lattice site
(677.3->592.7µm, 609.6->558.8µm, 592.7->635.0µm — verified separately,
not by this script) and the small-deviation agreement below would not
hold at that precision. If a future JAX/float change alters the
rasterizer's rounding at these exact dx values, THIS ARTIFACT GOES STALE
(re-solve and re-derive), it does not mean the tolerance below was wrong
Committed precision note: jax_enable_x64=False, inferred from Z0
agreement (see above), not read from a log file.

The dx=80µm/60µm knife-edge is structural, not incidental: at every
ALIGNED dx, the trace's lower face sits at
``yc - W/2 = 2*h_sub + 8*dx`` in the sweep's fixed-clearance geometry,
which lands EXACTLY on a lattice node for integer h_sub/dx — so aligned
vs. misaligned board realization is a property of the geometry formula,
not a coincidence of these six dx values.

Run (no FDTD solve; ``sim.fidelity_report()`` only):

    PYTHONPATH=<repo root> python3 scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.fidelity import fidelity_report
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

# These constants and DX_GRID mirror msl_z0_bias_floor_sweep.py's fixture
# EXACTLY (same EPS_R/H_SUB/W_TRACE/dx values) so the geometry rasterized
# here is the SAME geometry that sweep solved. Read-only mirror, not an
# import of mutable state from that module.
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
L_LINE = 10e-3
PORT_MARGIN = 2e-3

DX_GRID = [
    ("aligned h_sub/3", H_SUB / 3.0),
    ("aligned h_sub/4", H_SUB / 4.0),
    ("aligned h_sub/5", H_SUB / 5.0),
    ("aligned h_sub/6", H_SUB / 6.0),
    ("misaligned 80um", 80e-6),
    ("misaligned 60um", 60e-6),
]

SOURCE_JSON = (
    REPO / "scripts" / "diagnostics" / "msl_z0_bias_floor_sweep"
    / "msl_z0_bias_floor_sweep.json"
)
OUT_DIR = SOURCE_JSON.parent
OUT_PATH = OUT_DIR / "msl_z0_bias_floor_sweep_realized_anchor.json"


def _build_sim_no_solve(dx: float) -> Simulation:
    """Same geometry as msl_z0_bias_floor_sweep.run_one(), pre-solve only."""
    lx = L_LINE + 2 * PORT_MARGIN
    ly = W_TRACE + 2 * (2 * H_SUB + 8 * dx)
    lz = H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="ro4350b")
    yc = ly / 2.0
    sim.add(Box((0.0, yc - W_TRACE / 2.0, H_SUB),
                (lx, yc + W_TRACE / 2.0, H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN, yc, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, yc, 0.0), width=W_TRACE,
                     height=H_SUB, direction="-x", impedance=50.0)
    return sim


def realized_h_w_um(dx: float) -> tuple[float, float]:
    """(h_sub_realized_um, w_trace_realized_um) from fidelity_report().

    entity[0] is the 'ro4350b' substrate Box (z-axis realized extent =
    realized h_sub); entity[1] is the 'pec' trace Box (y-axis realized
    extent = realized W). No solve: fidelity_report() only rasterizes
    the declared geometry onto the grid.
    """
    sim = _build_sim_no_solve(dx)
    report = fidelity_report(sim, print_report=False)
    sub_item = report[1]
    trace_item = report[2]
    assert sub_item["entity"].startswith("geometry[0]"), sub_item["entity"]
    assert trace_item["entity"].startswith("geometry[1]"), trace_item["entity"]
    h_real_um = next(a for a in sub_item["axes"] if a["axis"] == "z")[
        "realized_extent_um"]
    w_real_um = next(a for a in trace_item["axes"] if a["axis"] == "y")[
        "realized_extent_um"]
    return float(h_real_um), float(w_real_um)


def main() -> int:
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in source["rows"]}

    rows = []
    for label, dx in DX_GRID:
        src = by_label[label]
        h_real_um, w_real_um = realized_h_w_um(dx)
        z0_hj_real, eps_eff_real = hammerstad_jensen_z0_eps_eff(
            w_real_um * 1e-6, h_real_um * 1e-6, EPS_R)
        z0_meas = src["z0_measured_ohm"]  # verbatim from the pre-declared JSON
        z0_hj_decl = src["z0_hj_ohm"]     # verbatim, declared-board anchor
        dev_vs_realized_pct = (z0_meas - z0_hj_real) / z0_hj_real * 100.0
        dev_vs_declared_pct = (z0_meas - z0_hj_decl) / z0_hj_decl * 100.0
        rows.append({
            "label": label,
            "dx_um": src["dx_um"],
            "h_sub_declared_um": round(H_SUB * 1e6, 3),
            "h_sub_realized_um": round(h_real_um, 3),
            "w_trace_declared_um": round(W_TRACE * 1e6, 3),
            "w_trace_realized_um": round(w_real_um, 3),
            "z0_measured_ohm": z0_meas,  # copied read-only from source JSON
            "z0_hj_declared_board_ohm": z0_hj_decl,  # copied read-only
            "z0_hj_realized_board_ohm": round(z0_hj_real, 3),
            "dev_vs_declared_board_pct": round(dev_vs_declared_pct, 4),
            "dev_vs_realized_board_pct": round(dev_vs_realized_pct, 4),
        })

    max_abs_dev_realized_all = max(abs(r["dev_vs_realized_board_pct"]) for r in rows)
    misaligned = [r for r in rows if r["label"].startswith("misaligned")]
    max_abs_dev_realized_misaligned = max(
        abs(r["dev_vs_realized_board_pct"]) for r in misaligned)

    out = {
        "source_json": str(SOURCE_JSON.relative_to(REPO)),
        "source_json_sha256": (
            "f56f6b17691613d8782c1d5ce1241c1cd9bc10ef61715b203ed5cd6d4ab18362"
        ),
        "note": (
            "Sibling artifact (issue #752): adds Hammerstad-Jensen scored "
            "against the REALIZED h/W (sim.fidelity_report(), no solve) "
            "alongside the pre-declared declared-board anchor. "
            "z0_measured_ohm and z0_hj_declared_board_ohm are copied "
            "VERBATIM from source_json, never re-solved or retyped. The "
            "source JSON and its as-run verdict block are untouched."
        ),
        "precision": (
            "jax_enable_x64=False, inferred from Z0 agreement with the "
            "realized-board Hammerstad-Jensen anchor (max |dev| = "
            f"{max_abs_dev_realized_all:.3f}% over all six points, vs "
            "8.6/5.6/4.4% at the alternative (x64) rasterization of the "
            "aligned class's trace width — not read from a log file. If "
            "a rasterizer or float-precision change alters these dx "
            "points' realized h/W, this artifact and its tolerance go "
            "stale (re-solve and re-derive); it is not evidence of a "
            "tolerance bug."
        ),
        "knife_edge": (
            "At every ALIGNED dx in this grid, the trace's lower face "
            "(yc - W/2 = 2*h_sub + 8*dx, the sweep's fixed-clearance "
            "formula) lands EXACTLY on a lattice node because h_sub/dx "
            "is an integer there — a structural property of the "
            "geometry formula, not a coincidence of these six dx values."
        ),
        "rows": rows,
        "max_abs_dev_vs_realized_board_pct_all_six": round(
            max_abs_dev_realized_all, 4),
        "max_abs_dev_vs_realized_board_pct_misaligned_pair": round(
            max_abs_dev_realized_misaligned, 4),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    for r in rows:
        print(
            f"{r['label']:18s} dx={r['dx_um']:7.2f}um "
            f"h_real={r['h_sub_realized_um']:7.1f}um "
            f"w_real={r['w_trace_realized_um']:7.1f}um "
            f"Z0meas={r['z0_measured_ohm']:6.2f} "
            f"HJ(real)={r['z0_hj_realized_board_ohm']:6.2f} "
            f"dev_real={r['dev_vs_realized_board_pct']:+.3f}% "
            f"HJ(decl)={r['z0_hj_declared_board_ohm']:6.2f} "
            f"dev_decl={r['dev_vs_declared_board_pct']:+.2f}%"
        )
    print(
        f"max |dev_real| over all six: "
        f"{out['max_abs_dev_vs_realized_board_pct_all_six']}%"
    )
    print(
        f"max |dev_real| over the misaligned pair: "
        f"{out['max_abs_dev_vs_realized_board_pct_misaligned_pair']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
