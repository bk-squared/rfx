#!/usr/bin/env python3
"""Historical realized-board anchor for msl_z0_bias_floor_sweep.py (#752).

CURRENT STATUS: generation is retired. The archived Z0 values and their
recorded geometry belong to an earlier solver/rasterizer. Combining them
with today's geometry creates an invalid comparison. The command refuses
generation; --show-archive verifies and prints the frozen historical data
without a solve, model re-evaluation, or file writes. No current Z0 accuracy
bound follows from this archive. The build helpers below remain available
only for geometry-drift checks; material extents are not conductor gaps.

The original explanation below describes the historical operation.

This is a NEW, SIBLING artifact. It does NOT touch
``scripts/diagnostics/msl_z0_bias_floor_sweep.py`` or its committed JSON
(``scripts/diagnostics/msl_z0_bias_floor_sweep/msl_z0_bias_floor_sweep.json``,
sha256 pinned by ``test_msl_z0_bias_floor_sweep_json_is_frozen`` in
``tests/unit/ports/test_msl_port_preflight.py``) — that script is the pre-declared
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

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.fidelity import fidelity_report

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
ANCHOR_JSON = SOURCE_JSON.parent / "msl_z0_bias_floor_sweep_realized_anchor.json"
_SOURCE_SHA256 = "f56f6b17691613d8782c1d5ce1241c1cd9bc10ef61715b203ed5cd6d4ab18362"
_ANCHOR_SHA256 = "fbf7985d970a3976652f53a9a0fe1b65e7e7dd9e733041c04b94430b3db6863c"


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
    """Current material/PEC-body extents for historical geometry-drift checks.

    entity[0] is the 'ro4350b' substrate Box (z-axis realized extent =
    realized h_sub); entity[1] is the 'pec' trace Box (y-axis realized
    extent = realized W). These are geometry extents, NOT an RF gap or an
    electrical-width calibration: the dielectric column can extend into a
    PEC volume. For example the 80um case has a 320um dielectric extent
    but a 240um ground-to-trace gap under #931. No solve is performed.
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the frozen historical #752 anchor.")
    parser.add_argument("--show-archive", action="store_true",
                        help="verify and print the historical record; never generate a current comparison")
    args = parser.parse_args(argv)
    if not args.show_archive:
        parser.error(
            "Historical anchor generation is retired: the frozen Z0 measurements "
            "cannot be paired with current geometry or used as a current accuracy "
            "bound. Use --show-archive to inspect the existing record. A new "
            "comparison requires geometry and field measurements from the same run.")
    source_bytes = SOURCE_JSON.read_bytes()
    anchor_bytes = ANCHOR_JSON.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != _SOURCE_SHA256:
        parser.error("The frozen source record changed; restore its recorded bytes before inspection.")
    if hashlib.sha256(anchor_bytes).hexdigest() != _ANCHOR_SHA256:
        parser.error("The frozen anchor record changed; restore its recorded bytes before inspection.")
    print(json.dumps({
        "record_kind": "historical_realized_anchor",
        "current_solver_validation": False,
        "current_geometry_or_model_recomputed": False,
        "source_json_sha256": _SOURCE_SHA256,
        "anchor_json_sha256": _ANCHOR_SHA256,
        "provenance_note": (
            "Geometry and inferred precision below are inherited historical "
            "metadata, not newly verified provenance of the original field run. "
            "No current error bound is claimed."),
        "historical_record": json.loads(anchor_bytes),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
