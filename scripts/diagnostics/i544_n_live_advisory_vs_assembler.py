#!/usr/bin/env python3
"""Issue #544: preflight advisory says n_live/n=3/4, the assembler uses 4.

======================================================================
PRE-DECLARED (RF-intensifier discipline, R2 tightened -- one attempt)
======================================================================

Question: which count is geometrically correct for the #488 lane's
committed lumped/wire<->MSL fixture (dx=80um) -- does the boundary-adjacent
extent cell of the wire port conduct (live) or not?

Method: dump, for the EXACT committed fixture
(`tests/unit/sparams/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl`, reused
verbatim, imitating i517_mixed_solve_vs_ratio_measurement.py's precedent):

  (a) the ADVISORY's counting path as it existed BEFORE this issue's fix
      -- a standalone bounding-box-vs-cell-CENTER geometric approximation
      (frozen here verbatim as `_old_buggy_dead_cell_classification`; the
      live code in `rfx/api/_preflight.py::_validate_cfg_port_inside_pec`
      no longer contains this path -- see the PR diff for the fix),
  (b) the ASSEMBLER's actual per-cell classification -- the exact call
      `rfx/api/_sparams.py`'s `compute_mixed_s_matrix` makes,
      `_wire_port_live_cells(grid, wp, pec_mask)` (`rfx/sources/sources.py`),
  (c) the GROUND TRUTH -- the rasterized PEC occupancy the FDTD solver
      itself will see: `Simulation._assemble_materials(grid)`'s `pec_mask`,
      dumped at every node index the wire port's extent covers.

Verdict rule (pre-declared): the count that matches (c) is correct. PR
#543's measured passive-port Z_in = Z0/4 = 12.5 ohm flat is the INDEPENDENT
physics witness -- if (c) contradicted that arithmetic, this script STOPS
and reports rather than reconciling by hand. (c) is read directly from the
same `pec_mask` array the solver's Yee update masks against
(`rfx/api/_compile.py::_assemble_materials`), not re-derived.

======================================================================
R1 memory citations
======================================================================
  - `rfx-known-issues.md` "#318 RESOLVED" entry: dead cells are excluded
    from the port's sigma distribution/drive/normalization by design --
    this script does not question THAT mechanism, only which cells the
    ADVISORY (vs the assembler) classifies as dead on this one fixture.
  - `i517_mixed_solve_vs_ratio.json` (PR #543, committed): `n_live_lw =
    [4]`, `z0_lw = [50.0]` -- the exact assembler value this script
    independently reproduces below via `_wire_port_live_cells`, read from
    the committed artifact rather than re-run (no new FDTD needed: this
    is a static geometry/rasterization question, zero solve).
  - `feedback_gate_can_bind_artifact.md`: no gate is strengthened here;
    this is a diagnosis-then-fix, with the ground-truth dump as the
    falsifier for the fix, matching the "mandate: falsifier re-run" spirit
    for a *changed* advisory (it changes from firing to silent on this
    fixture).

======================================================================
R3 pre-commit self-audit
======================================================================
  1. Contradicted by known memory? No -- see R1; this SHARPENS the #318/
     #319 dead-cell-accounting class (docs/agent-memory/rfx-known-issues.md
     "#318 RESOLVED" entry), it does not contradict it.
  2. R2 trip? No new FDTD run, no new mechanism-hypothesis attempt -- one
     geometry-only instrumented comparison, pre-declared above, first
     attempt on this question.
  3. Falsifier: the ground-truth `pec_mask` dump IS the falsifier -- if it
     had shown the boundary cell (index 3, z=280um center) genuinely PEC,
     the advisory would have been right and the assembler wrong (and
     PR #543's Z0/4 physics would need re-explaining, not this advisory).

======================================================================
Fixture (pre-declared, the #488 lane's own committed lumped/wire<->MSL
fixture, `tests/unit/sparams/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl`,
reused verbatim) -- IDENTICAL to `i517_mixed_solve_vs_ratio_measurement.py`.
======================================================================

  RO4350B-like substrate eps_r=3.66, h_sub=254um, trace width=600um,
  dx=80um, domain 8mm x 3mm x 754um, PEC z_lo / CPML elsewhere,
  cpml_layers=8. Vertical wire feed (`add_port`, component="ez",
  impedance=50, extent=h_sub) at x=2mm; MSL port (`add_msl_port`,
  direction="-x") at x=5.5mm. wire_mode=True. This script needs no FDTD
  run at all -- (a)/(b)/(c) are all static, geometry-only quantities
  (grid + `_assemble_materials`), so it is a ZERO-solve diagnostic.

======================================================================
RESULT (2026-08-04)
======================================================================

Ground truth (c): the PEC trace box spans z in [254um, 334um], exactly
ONE cell thick (== dx), so `Box.mask` takes the THIN-SHEET branch
(`rfx/geometry/csg.py::Box.mask_on_coords`): the single z-NODE nearest the
box's midpoint 294um. Node coordinates are z_k = k*dx = 0, 80, 160, 240,
320, ... um (no half-cell offset -- `Grid.position_to_index` and
`_grid_coords` both use `round(pos/dx)`, not `pos/dx + 0.5`). Nearest node
to 294um is z_4=320um (|294-320|=26um vs |294-240|=54um) -- so pec_mask is
True ONLY at k=4. The wire port's 4 rasterized cells are k=0..3 (z=0 to
254um -> `round(254e-6/80e-6)=3`) -- k=4 is OUTSIDE the port's extent
entirely. GROUND TRUTH: all 4 port cells are LIVE. n_live/n = 4/4.

(b) assembler: `_wire_port_live_cells(grid, wp, pec_mask)` returns
`live_flags=[True,True,True,True]`, `n_live=4` -- MATCHES ground truth
exactly, confirming the committed `i517` artifact's `n_live_lw=[4]`.

(a) old advisory: `_wire_port_cell_centers` computed cell CENTERS by
adding a +0.5*dx Yee-component offset (z=40,120,200,280um) then tested
each center against the PEC box's bounding box with a CLOSED interval
`c1<=center<=c2`. Cell index 3's center (280um) falls inside [254,334]um
-> flagged DEAD. This is WRONG: it uses a reference point (component
center, closed interval) that does not match how `Box.mask` actually
rasterizes (node coordinate, half-open interval, thin-sheet nearest-node
rule) -- the true PEC node (k=4, z=320um) is a full cell away from what
the approximation flagged (k=3, z=280um center). n=4, dead=[3], n_live/n
= 3/4 -- the exact text from the committed `i517` JSON's
`preflight_text` ("... n_live/n = 3/4 ... terminates at 50 ohm across its
3 live cells ...").

VERDICT: (b) matches (c); (a) does not. Per the pre-declared rule, the
ASSEMBLER's count (4) is geometrically correct; the ADVISORY (3) was
wrong. Z0/4 = 12.5 ohm stays fully consistent with ground truth -- no
contradiction, no STOP. Fixed in `rfx/api/_preflight.py` by making the
advisory call `_wire_port_live_cells` against the SAME assembled
`pec_mask`, i.e. sharing (b)'s primitive directly rather than
re-approximating it -- the two paths cannot drift again by construction.
Post-fix, `sim.preflight()` on this exact fixture no longer emits
`wire_port_dead_extent_cells` for the lw port (dumped below, "AFTER").
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Stale-editable-install guard (workspace memory
# feedback_stale_editable_install_shadow.md): running this file as
# ``python scripts/diagnostics/i544_....py`` puts the script's own
# directory on sys.path[0], not the repo root, so a plain ``import rfx``
# can silently resolve to a DIFFERENT installed checkout instead of this
# worktree. Force the worktree root first (same precedent as
# ``build_waveguide_wr90_nu_flux_broad_e4_comparison.py``).
sys.path.insert(0, str(REPO))

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import (
    GaussianPulse, WirePort, _wire_port_cells, _wire_port_live_cells,
)

OUT_DIR = REPO / "scripts" / "diagnostics" / "i544_n_live_advisory_vs_assembler"
I517_JSON = (REPO / "scripts" / "diagnostics" / "i517_mixed_solve_vs_ratio"
             / "i517_mixed_solve_vs_ratio.json")

# ---------------------------------------------------------------------------
# Fixture -- verbatim from tests/unit/sparams/test_mixed_port_sparam.py (_base_sim,
# _add_feed, _add_msl), the #488 lane's own committed lumped/wire<->MSL
# fixture. Not re-derived; imitates the lane's own test invocation (same
# precedent as i517_mixed_solve_vs_ratio_measurement.py).
# ---------------------------------------------------------------------------
_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6


def _base_sim(**kw):
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        **kw,
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    return sim, y_c


def _add_msl(sim, y_c, x=5.5e-3, direction="-x", **kw):
    sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction=direction, impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5), **kw)


def _add_feed(sim, y_c, x=2e-3):
    sim.add_port(position=(x, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB)


# ---------------------------------------------------------------------------
# (a) The OLD (pre-#544) advisory counting path, frozen verbatim for the
# historical record. This is a STANDALONE reimplementation -- the live
# `rfx/api/_preflight.py::_validate_cfg_port_inside_pec` no longer contains
# this logic (see the PR diff). It is reproduced here ONLY so this script
# can show numerically what the advisory used to compute, without needing
# to check out a pre-fix commit.
# ---------------------------------------------------------------------------
def _old_buggy_dead_cell_classification(sim, grid, wp, extent):
    """Pre-#544 geometric approximation: cell CENTER (+0.5*dx Yee offset
    along the component axis) vs PEC bounding box, closed interval.
    """
    axis = {"ex": 0, "ey": 1, "ez": 2}[wp.component]
    d = (grid.dx, getattr(grid, "dy", grid.dx), getattr(grid, "dz", grid.dx))
    pad = (getattr(grid, "pad_x_lo", 0), getattr(grid, "pad_y_lo", 0),
           getattr(grid, "pad_z_lo", 0))
    cells = _wire_port_cells(grid, wp)
    centers = []
    for cell in cells:
        pos = [(cell[ax] - pad[ax]) * d[ax] for ax in range(3)]
        pos[axis] += 0.5 * d[axis]
        centers.append(tuple(pos))

    pec_bboxes = []
    for entry in sim._geometry:
        if entry.material_name != "pec":
            continue
        if not hasattr(entry.shape, "bounding_box"):
            continue
        c1, c2 = entry.shape.bounding_box()
        pec_bboxes.append((entry.material_name, c1, c2))

    dead_indices = []
    for idx, center in enumerate(centers):
        for name, c1, c2 in pec_bboxes:
            if all(c1[ax] <= center[ax] <= c2[ax] for ax in range(3)):
                dead_indices.append(idx)
                break
    n = len(centers)
    n_live = n - len(dead_indices)
    return {
        "cells": cells, "centers": centers, "dead_indices": dead_indices,
        "n": n, "n_live": n_live,
    }


def main() -> int:
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    grid = sim._build_grid()
    wp = WirePort(start=(2e-3, y_c, 0.0), end=(2e-3, y_c, _H_SUB),
                  component="ez", impedance=50.0)
    cells = _wire_port_cells(grid, wp)
    i0, j0 = cells[0][0], cells[0][1]

    materials, debye, lorentz, pec_mask, pec_shapes, boundary_pec, kerr = \
        sim._assemble_materials(grid)
    pec_mask_np = np.asarray(pec_mask)

    print("=" * 78)
    print("(c) GROUND TRUTH -- rasterized pec_mask at the wire port's (i,j) column")
    print("=" * 78)
    k_span = list(range(0, 9))
    col = pec_mask_np[i0, j0, k_span].tolist()
    print(f"  wire port cells (i,j,k): {cells}")
    print(f"  pec_mask[i={i0}, j={j0}, k=0..8]: {col}")
    print(f"  node z (um) for k=0..8: {[round(k * grid.dx * 1e6, 3) for k in k_span]}")
    dead_k_ground_truth = [k for k in k_span if col[k]]
    print(f"  PEC node index/indices (ground truth): {dead_k_ground_truth}")

    # (b) assembler's actual call (rfx/api/_sparams.py:3426)
    cells_b, live_flags_b, n_live_b = _wire_port_live_cells(grid, wp, pec_mask)
    print()
    print("=" * 78)
    print("(b) ASSEMBLER -- _wire_port_live_cells(grid, wp, pec_mask)")
    print("=" * 78)
    print(f"  live_flags: {live_flags_b}  n_live: {n_live_b}/{len(cells_b)}")

    # (a) old buggy advisory path, frozen verbatim
    old = _old_buggy_dead_cell_classification(sim, grid, wp, _H_SUB)
    print()
    print("=" * 78)
    print("(a) OLD ADVISORY (pre-#544, frozen verbatim reimplementation)")
    print("=" * 78)
    print(f"  centers (m): {old['centers']}")
    print(f"  dead_indices: {old['dead_indices']}  "
         f"n_live: {old['n_live']}/{old['n']}")

    ground_truth_n_live = sum(1 for c in cells if not bool(
        pec_mask_np[c[0], c[1], c[2]]))
    verdict_b_matches_c = (n_live_b == ground_truth_n_live)
    verdict_a_matches_c = (old["n_live"] == ground_truth_n_live)

    print()
    print("=" * 78)
    print("VERDICT (pre-declared rule: the count matching (c) is correct)")
    print("=" * 78)
    print(f"  ground truth n_live (directly from pec_mask): {ground_truth_n_live}")
    print(f"  (b) assembler matches ground truth: {verdict_b_matches_c}")
    print(f"  (a) old advisory matches ground truth: {verdict_a_matches_c}")

    # Cross-check against the committed i517 artifact (no new FDTD run --
    # static comparison against an already-committed measurement).
    i517_n_live = None
    i517_preflight_line = None
    if I517_JSON.exists():
        i517 = json.loads(I517_JSON.read_text())
        i517_n_live = i517["n_live_lw"]
        for line in i517["preflight_text"].splitlines():
            if "Wire port" in line and "rasterizes" in line:
                i517_preflight_line = line.strip()
                break
    print()
    print("=" * 78)
    print("CROSS-CHECK vs committed i517 artifact (PR #543, no new FDTD)")
    print("=" * 78)
    print(f"  i517 n_live_lw: {i517_n_live}  (this script's (b): [{n_live_b}])")
    print(f"  i517 preflight text (pre-#544-fix, quoted verbatim): "
         f"{i517_preflight_line}")
    i517_consistent = (i517_n_live == [n_live_b])

    # AFTER: run the CURRENT (fixed) preflight on this exact fixture and
    # show the advisory now agrees with ground truth (silent for this
    # port -- no dead cells).
    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        issues_after = sim.preflight()
    preflight_text_after = stdout_buf.getvalue()
    codes_after = [getattr(s, "code", None) for s in issues_after]
    dead_extent_fires_after = "wire_port_dead_extent_cells" in codes_after
    midpoint_fires_after = "wire_port_midpoint_in_pec" in codes_after
    print()
    print("=" * 78)
    print("AFTER THE FIX -- sim.preflight() on the identical fixture")
    print("=" * 78)
    print(f"  wire_port_dead_extent_cells fires: {dead_extent_fires_after}")
    print(f"  wire_port_midpoint_in_pec fires: {midpoint_fires_after}")
    print("  (expected: both False -- ground truth is fully live, 4/4)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "i544_n_live_advisory_vs_assembler.json"
    out_path.write_text(json.dumps({
        "fixture": {
            "eps_r": _EPS_R, "h_sub_m": _H_SUB, "w_trace_m": _W_TRACE,
            "dx_m": _DX, "feed_x_m": 2e-3, "msl_x_m": 5.5e-3,
        },
        "wire_port_cells_ijk": cells,
        "ground_truth": {
            "pec_mask_column_k0_8": col,
            "node_z_um_k0_8": [round(k * grid.dx * 1e6, 3) for k in k_span],
            "pec_node_indices": dead_k_ground_truth,
            "n_live": ground_truth_n_live,
            "n": len(cells),
        },
        "b_assembler": {
            "live_flags": live_flags_b, "n_live": n_live_b, "n": len(cells_b),
            "matches_ground_truth": verdict_b_matches_c,
        },
        "a_old_advisory": {
            "centers_m": old["centers"], "dead_indices": old["dead_indices"],
            "n_live": old["n_live"], "n": old["n"],
            "matches_ground_truth": verdict_a_matches_c,
        },
        "i517_cross_check": {
            "i517_n_live_lw": i517_n_live,
            "i517_preflight_line": i517_preflight_line,
            "consistent_with_this_scripts_b": i517_consistent,
        },
        "after_fix": {
            "wire_port_dead_extent_cells_fires": dead_extent_fires_after,
            "wire_port_midpoint_in_pec_fires": midpoint_fires_after,
            "preflight_text": preflight_text_after,
        },
        "verdict": {
            "ground_truth_n_live": ground_truth_n_live,
            "assembler_correct": verdict_b_matches_c,
            "old_advisory_correct": verdict_a_matches_c,
            "z0_over_4_consistent_with_ground_truth": (
                ground_truth_n_live == 4
            ),
            "conclusion": (
                "assembler (n_live=4) matches ground truth pec_mask; "
                "old advisory (n_live=3) did not -- the advisory was "
                "wrong, fixed by sharing _wire_port_live_cells against "
                "the same assembled pec_mask."
            ),
        },
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)

    assert verdict_b_matches_c, "assembler must match ground truth"
    assert not verdict_a_matches_c, (
        "expected the OLD advisory to disagree with ground truth on this "
        "fixture (that disagreement is issue #544) -- if this assertion "
        "fails, the frozen reimplementation no longer reproduces the bug "
        "and the historical record above needs revisiting"
    )
    assert i517_consistent, (
        "this script's (b) must match the already-committed i517 artifact"
    )
    assert not dead_extent_fires_after and not midpoint_fires_after, (
        "post-fix preflight must be silent on this fixture (ground truth "
        "is fully live)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
