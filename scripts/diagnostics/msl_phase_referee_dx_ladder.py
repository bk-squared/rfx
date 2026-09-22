#!/usr/bin/env python3
"""The MSL phase-referee thru fixture re-solved on a dx ladder (issue #830, H-b).

Same PHYSICAL problem at dx = 50, 25 and 12.5 um.  Everything the committed
builder ``scripts/diagnostics/build_msl_thru_phase_dx50um_reference.py``
expresses in CELLS is pinned here in METRES at its dx = 50 um value, so the
only intended change across the ladder is the mesh -- and the strip, which is
one cell thick by construction and therefore thins with dx.

WHAT IS HELD CONSTANT, IN METRES (all equal to the dx = 50 um builder's value)

  quantity                     metres        dx=50um     dx=25um     dx=12.5um
  ---------------------------  ------------  ----------  ----------  ---------
  trace width W                600.0 um      declared, dx-independent
  substrate thickness h        254.0 um      declared, dx-independent
  line length L_LINE           10.000 mm     declared, dx-independent
  port margin                  2.000 mm      declared, dx-independent
  domain LX = L+2*margin       14.000 mm     declared, dx-independent
  domain LY = W+2(2h+8*DXREF)  2416.0 um     the 8*DX term is PINNED at
                                             8 * 50 um = 400 um; it does NOT
                                             follow dx (that is the point)
  domain LZ = h + 1.5 mm       1754.0 um     declared, dx-independent
  CPML thickness               400.0 um      8 layers   16 layers   32 layers
  probe-0 offset from feed     2500.0 um     50 cells   100 cells   200 cells
  probe-to-probe spacing       550.0 um      11 cells   22 cells    44 cells
  feed plane, port 0           2.000 mm      exact cell multiple at all dx
  feed plane, port 1           12.000 mm     exact cell multiple at all dx
  probe-0 plane, port 0        4.500 mm      = 2.0 mm + 2.5 mm, exact
  probe-0 plane, port 1        9.500 mm      = 12.0 mm - 2.5 mm, exact
  record length (seconds)      num_periods=12 at f0 = f_max/2; dt scales with
                               dx so n_steps scales as 1/dx and the physical
                               record length is the same
  frequency grid               30 bins, linspace(f_max/10, f_max), f_max=5 GHz
  x64                          jax_enable_x64 True, as the committed builder

THE ONE QUANTITY THAT INTENTIONALLY CHANGES
  strip thickness t = dx (one cell).  The strip is drawn exactly as the builder
  draws it: Box from z = H_SUB (254 um, declared) to z = H_SUB + dx.  On every
  rung the PEC volume's cell-centre rasterization realizes a single cell whose
  wall planes are z = 250 um and z = 250 + dx um, so 250 um of dielectric sits
  under the strip at all three dx and the closed form to compare against is the
  same one (W = 600 um, h = 250 um, eps_r = 3.66) on every rung.

WHAT COULD NOT BE HELD CONSTANT (read this before reading E(25)/E(50))
  The domain is declared in metres, but the rasterizer rounds each axis UP to a
  whole cell, so the REALIZED clear extents differ slightly between rungs.  The
  script measures and records them (``realized_domain_clear_m``); they are not
  asserted, because no single declared length can make them equal at three dx.
  The lateral clearance from the strip edge to the domain wall is ~950 um, i.e.
  nearly four substrate thicknesses, so a few tens of um of difference there is
  far outside the fringing field -- but it is a difference, and it is reported
  rather than assumed negligible.
  The CPML is equal in THICKNESS, not identical: the same 400 um is graded over
  8, 16 or 32 layers, so its discrete reflection is not byte-identical either.

HOW BETA IS REPORTED
  Twice.  ``beta_production`` is what ``compute_msl_s_matrix`` returns.  The
  production estimator scans 41 nodes over beta0*[0.65, 1.35] and refines the
  best node with one 3-point parabola, which carries a sawtooth bias of up to
  +-0.165 % of beta0 (measured on synthetic two-wave voltages, issue #830 H-a).
  ``beta_refit`` is a float64 Levenberg-Marquardt fit of the SAME two-wave model
  to the SAME stored probe-plane phasors, started off the production anchor --
  the same independent witness the H-a bench used, which recovered a known beta
  to 1e-15.  Every derived quantity is computed for both.

WITNESSES
  ``settling_db`` is what this lane exposes for ring-down: per driven run, the
  worst over all port probe planes of 10*log10(mean Ez^2 over the last 10 % of
  the record / peak Ez^2).  It is a PROBE-PLANE Ez ratio, not a total-domain
  field-energy ratio; ``compute_msl_s_matrix`` exposes no domain-energy witness
  and none is invented here.  Also recorded: ``beta_railed``, ``reliable``,
  ``cond_a``, ``passivity_correction``, ``assembly``, ``probe_clearance``,
  ``reference_impedances``.  The per-bin FIT RESIDUAL is NOT on the result
  object and NOT in the dump, so it is recomputed here from the dumped phasors
  by calling the production extractor on them.
  Preflight stdout is teed to the log and saved verbatim; every warning raised
  during the solve is recorded and re-printed verbatim, never suppressed.

PERSISTENCE ORDER
  ``result_core.json`` is written immediately after the solve, before any
  optional stage.  Two earlier multi-hour runs were lost to an optional stage
  that ran before the write.  ``result.json`` (core + derived) and the figure
  come afterwards and cannot cost the measurement.

Usage::

    python scripts/diagnostics/msl_phase_referee_dx_ladder.py \\
        --dx-um 50 --out-dir /path/to/out
    python scripts/diagnostics/msl_phase_referee_dx_ladder.py \\
        --dx-um 25 --out-dir /tmp/smoke --smoke
    python scripts/diagnostics/msl_phase_referee_dx_ladder.py \\
        --dx-um 12.5 --out-dir /tmp/geom --geometry-only
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
import warnings
from decimal import Decimal
from pathlib import Path

import jax

# Script-level, matching the committed builder (NOT test-level -- see
# tests/contracts/test_no_module_level_x64.py).
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import rfx  # noqa: E402
from rfx.api import Simulation  # noqa: E402
from rfx.api._sparams import _resolve_msl_auto_offsets  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff  # noqa: E402
from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json"
REFEREE_CASE = REPO_ROOT / "validation/crossval/20_msl_phase_referee.py"

# --- declared geometry, identical to the committed builder ------------------
EPS_R = 3.66            # RO4350B
H_SUB = 254e-6          # declared substrate thickness
W_TRACE = 600e-6        # declared trace width
L_LINE = 10e-3          # thru-line length
PORT_MARGIN = 2e-3      # feed -> domain edge clearance
F_MAX = 5e9
N_FREQS = 30
NUM_PERIODS = 12

# --- the reference mesh the ladder is pinned to -----------------------------
DX_REF = 50e-6
CPML_LAYERS_REF = 8
N_PROBE_OFFSET_REF = 50     # cells at DX_REF
N_PROBE_SPACING_REF = 11    # cells at DX_REF
N_PROBES = 5

# --- the same lengths, in metres (what is actually held constant) -----------
CPML_THICKNESS_M = CPML_LAYERS_REF * DX_REF        # 400.0 um
PROBE_OFFSET_M = N_PROBE_OFFSET_REF * DX_REF       # 2500.0 um
PROBE_SPACING_M = N_PROBE_SPACING_REF * DX_REF     # 550.0 um

LX = L_LINE + 2 * PORT_MARGIN                      # 14.000 mm
SIDE_ALLOWANCE_M = 2 * H_SUB + 8 * DX_REF          # 908.0 um per side, 8*DX PINNED
AIR_HEIGHT_M = 1.5e-3                              # air above the board
LY = W_TRACE + 2 * SIDE_ALLOWANCE_M                # 2416.0 um at box_scale 1
LZ = H_SUB + AIR_HEIGHT_M                          # 1754.0 um at box_scale 1


def domain_ly_lz(box_scale: float) -> tuple[float, float]:
    """Lateral and vertical domain extents at a given box scale.

    ``box_scale = 1`` reproduces the committed builder's LY and LZ exactly.
    ``box_scale = 2`` doubles the side allowance and the air height and nothing
    else -- hypothesis H-d, lateral/top truncation: if the residual is partly
    the side walls and the top boundary clipping the fringing field, it moves
    when the box grows at FIXED dx; if it is the mesh or the extractor, it does
    not.  The strip, the feed planes, the probe planes, the line length, the
    port margin, the CPML thickness, the frequencies and the record length are
    untouched, so the two runs differ in the box and in nothing else.
    """
    return (W_TRACE + 2 * box_scale * SIDE_ALLOWANCE_M,
            H_SUB + box_scale * AIR_HEIGHT_M)


# The probe planes, in metres, that every rung and every box scale must land
# on. Probe n sits offset + n*spacing from the feed plane, along the port's own
# direction. Asserted rather than described: it is the single invariant that
# says the ladder compares the same measurement at three mesh sizes, and it is
# also what answers "does the port/probe placement depend on LY?" -- if it did,
# the box-scale run would move these and the assert would refuse it.
CANONICAL_PROBE_X_M = {
    "msl_0": tuple(PORT_MARGIN + (PROBE_OFFSET_M + n * PROBE_SPACING_M)
                   for n in range(N_PROBES)),
    "msl_1": tuple(PORT_MARGIN + L_LINE - (PROBE_OFFSET_M + n * PROBE_SPACING_M)
                   for n in range(N_PROBES)),
}

# --- comparison constants ---------------------------------------------------
# Exact SI definition of the speed of light. The crossval case
# validation/crossval/20_msl_phase_referee.py uses _C0 = 2.998e8 instead, which
# is 2.5e-5 low and shifts a beta deviation by +0.0025 % -- small next to the
# ~1.3 % being attributed, but it is a different number and this script does
# not silently inherit it.
C0 = 299792458.0
# The board rfx actually builds: 250 um of dielectric under the strip, at every
# rung (asserted below from the realized wall planes).
H_CLOSED_FORM = 250e-6
GATE_F_LO_HZ = 3.0e9
GATE_F_HI_HZ = 4.5e9

_TAGS = {50.0: "dx50", 25.0: "dx25", 12.5: "dx12p5"}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def _cells(length_m: float, dx: float, what: str) -> int:
    """``length_m`` in cells, refusing a length that is not a cell multiple.

    A silently rounded cell count is how a "held constant" length stops being
    held constant, so this raises instead of rounding away the remainder.
    """
    n = length_m / dx
    n_int = int(round(n))
    if abs(n - n_int) > 1e-9:
        raise SystemExit(
            f"{what} = {length_m * 1e6:.6f} um is not a whole number of "
            f"{dx * 1e6:g} um cells ({n:.9f} cells). The ladder cannot hold "
            f"it constant at this dx."
        )
    return n_int


def build_sim(dx: float, box_scale: float = 1.0) -> tuple[Simulation, dict]:
    """The committed builder's geometry, with every cell-counted length scaled
    so its METRES value is the dx = 50 um one."""
    cpml_layers = _cells(CPML_THICKNESS_M, dx, "CPML thickness")
    n_offset = _cells(PROBE_OFFSET_M, dx, "probe-0 offset")
    n_spacing = _cells(PROBE_SPACING_M, dx, "probe spacing")
    ly, lz = domain_ly_lz(box_scale)

    sim = Simulation(
        freq_max=F_MAX,
        domain=(LX, ly, lz),
        dx=dx,
        cpml_layers=cpml_layers,
        boundary=BoundarySpec(
            x="cpml", y="cpml",
            z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, ly, H_SUB)), material="ro4350b")
    y_centre = ly / 2.0
    trace_y_lo = y_centre - W_TRACE / 2.0
    trace_y_hi = y_centre + W_TRACE / 2.0
    # The strip: one cell thick, drawn exactly as the committed builder draws
    # it (H_SUB .. H_SUB + dx). This is the one quantity that changes.
    sim.add(
        Box((0.0, trace_y_lo, H_SUB), (LX, trace_y_hi, H_SUB + dx)),
        material="pec",
    )
    # EXPLICIT offsets/spacings. The committed builder leaves both AUTO and the
    # driver-time solve lands on 50/11 at dx = 50 um; passing the scaled cell
    # counts explicitly reproduces that layout in metres at every rung, and an
    # explicit value is never re-solved (rfx/sparams/_common.py
    # _resolve_msl_auto_offsets: "Explicit offsets are never touched").
    for feed_x, direction in ((PORT_MARGIN, "+x"),
                              (PORT_MARGIN + L_LINE, "-x")):
        sim.add_msl_port(
            position=(feed_x, y_centre, 0.0),
            width=W_TRACE, height=H_SUB, direction=direction, impedance=50.0,
            n_probe_offset=n_offset, n_probe_spacing=n_spacing,
            n_probes=N_PROBES,
        )
    scaling = {
        "dx_m": dx,
        "cpml_layers": cpml_layers,
        "cpml_thickness_m": cpml_layers * dx,
        "n_probe_offset_cells": n_offset,
        "probe_offset_m": n_offset * dx,
        "n_probe_spacing_cells": n_spacing,
        "probe_spacing_m": n_spacing * dx,
        "n_probes": N_PROBES,
        "box_scale": float(box_scale),
        "declared_domain_m": [LX, ly, lz],
        "side_allowance_per_side_m": box_scale * SIDE_ALLOWANCE_M,
        "air_height_m": box_scale * AIR_HEIGHT_M,
        "y_centre_m": y_centre,
        "trace_y_declared_m": [trace_y_lo, trace_y_hi],
        "strip_box_z_declared_m": [H_SUB, H_SUB + dx],
    }
    return sim, scaling


# ---------------------------------------------------------------------------
# Geometry read-outs (imitating the committed builder's helpers)
# ---------------------------------------------------------------------------
def reference_plane_geometry(sim: Simulation) -> dict:
    """Probe-0 plane per port plus the grid, as the committed builder records
    it -- the fields the dx = 50 identity assert compares against."""
    grid = sim._build_grid()
    entries = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), grid)
    out: dict = {}
    for pe in entries:
        x_feed, y_centre, z_lo = pe.position
        mp = MSLPort(
            feed_x=float(x_feed), y_lo=float(y_centre - pe.width / 2),
            y_hi=float(y_centre + pe.width / 2), z_lo=float(z_lo),
            z_hi=float(z_lo + pe.height), direction=pe.direction,
            impedance=pe.impedance, excitation=pe.waveform,
        )
        xs = msl_probe_x_coords_n(
            grid, mp, n_probes=pe.n_probes,
            n_offset_cells=pe.n_probe_offset,
            n_spacing_cells=pe.n_probe_spacing,
        )
        out[pe.name] = {
            "feed_x_m": float(x_feed),
            "direction": pe.direction,
            "n_probe_offset": int(pe.n_probe_offset),
            "n_probe_spacing": int(pe.n_probe_spacing),
            "n_probes": int(pe.n_probes),
            "probe0_x_m": float(xs[0]),
            "probe_x_m": [float(v) for v in xs],
        }
    interior = {
        ax: int(getattr(grid, f"n{ax}")
                - getattr(grid, f"pad_{ax}_lo")
                - getattr(grid, f"pad_{ax}_hi"))
        for ax in ("x", "y", "z")
    }
    out["_grid"] = {
        "shape": [int(n) for n in grid.shape],
        "dx": float(grid.dx),
        "dt_s": float(grid.dt),
        "pad_x_lo": int(grid.pad_x_lo), "pad_x_hi": int(grid.pad_x_hi),
        "pad_y_lo": int(grid.pad_y_lo), "pad_y_hi": int(grid.pad_y_hi),
        "pad_z_lo": int(grid.pad_z_lo), "pad_z_hi": int(grid.pad_z_hi),
        "interior_cells": interior,
        "rasterized_clear_m": {
            ax: interior[ax] * float(grid.dx) for ax in ("x", "y", "z")
        },
        "n_cells_total": int(np.prod(np.asarray(grid.shape, dtype=np.int64))),
    }
    return out


def realized_trace_geometry(sim: Simulation) -> dict:
    """The strip's realized wall planes and y bounds, from the shared realized-
    edge owner (``rfx.boundaries.pec``) and NOT from ``fidelity_report`` -- the
    committed builder's own reason: for a PEC VOLUME the fidelity report's node
    sampler and the solver's cell-centre sampler disagree by one cell."""
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    grid = sim._build_grid()
    sheets: list = []
    wires: list = []
    out = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    pec_mask = out[3]
    edges = realized_pec_edge_masks(
        pec_mask, sheets, wires, periodic=tuple(sim._periodic_flags()))
    coords = coords_from_uniform_grid(grid)
    xs = np.asarray(coords.x, dtype=float)
    ys = np.asarray(coords.y, dtype=float)
    zs = np.asarray(coords.z, dtype=float)
    lx_sim, ly_sim, _ = (float(v) for v in sim._domain)
    i = int(np.argmin(np.abs(xs - lx_sim / 2.0)))
    j = int(np.argmin(np.abs(ys - ly_sim / 2.0)))
    planes = realized_wall_planes(
        edges, 2, ij=(i, j), periodic=tuple(sim._periodic_flags()))
    mx = np.asarray(edges[0], dtype=bool)
    rows = np.flatnonzero(mx[:, :, planes[0]].any(axis=0))
    y_lo, y_hi = float(ys[rows.min()]), float(ys[rows.max()])
    return {
        "trace_wall_planes_realized": [int(k) for k in planes],
        "trace_wall_planes_realized_z_m": [float(zs[k]) for k in planes],
        "trace_realization_kind": ("sheet" if sheets else "volume"),
        "t_metal_realized_m": (len(planes) - 1) * float(grid.dx),
        "trace_y_lo_realized_m": y_lo,
        "trace_y_hi_realized_m": y_hi,
        "w_trace_realized_m": y_hi - y_lo,
        "trace_y_centre_realized_m": 0.5 * (y_lo + y_hi),
    }


def realized_board_geometry(sim: Simulation, dx: float) -> dict:
    """Realized substrate thickness, live from ``sim.fidelity_report()``.

    The dielectric IS node half-open sampled, so fidelity agrees with the
    solver here (unlike the PEC volume rows above).  Returns the whole report
    as well: "assert realized, not declared" needs the rows kept, not only the
    one number the assert reads.
    """
    report = sim.fidelity_report(print_report=False)
    sub = next(item for item in report
               if item["entity"] == "geometry[0] 'ro4350b'")
    h_ax = next(ax for ax in sub["axes"] if ax["axis"] == "z")
    h_sub_realized_m = float(h_ax["realized_extent_um"]) * 1e-6
    return {
        "h_sub_realized_m": h_sub_realized_m,
        "n_z_sub_realized": int(round(h_sub_realized_m / dx)),
        "fidelity_rows": report,
    }


# ---------------------------------------------------------------------------
# Asserts -- realized, not declared
# ---------------------------------------------------------------------------
def assert_realized_board(realized: dict, dx: float) -> list[str]:
    """Abort unless the rasterizer built the board this ladder assumes."""
    lines = []
    problems = []

    def check(name: str, got: float, want: float, tol: float) -> None:
        ok = abs(got - want) <= tol
        lines.append(
            f"  {name:34s} realized {got * 1e6:12.6f} um   expected "
            f"{want * 1e6:12.6f} um   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(
                f"{name}: realized {got * 1e6:.6f} um, expected "
                f"{want * 1e6:.6f} um (tol {tol * 1e6:g} um)")

    lines.append("REALIZED BOARD (assert realized, not declared)")
    check("trace width W", realized["w_trace_realized_m"], W_TRACE, 1e-12)
    planes = realized["trace_wall_planes_realized_z_m"]
    if len(planes) != 2:
        problems.append(
            f"trace wall planes: realized {len(planes)} plane(s) "
            f"{planes}, expected exactly 2 (a one-cell-thick strip realizes "
            f"a wall at each face)")
        lines.append(f"  trace wall planes                  realized {planes} "
                     f"-- MISMATCH, expected 2 planes")
    else:
        check("strip lower wall plane z", planes[0], H_CLOSED_FORM, 1e-12)
        check("strip upper wall plane z", planes[1], H_CLOSED_FORM + dx, 1e-12)
    check("realized metal thickness t", realized["t_metal_realized_m"], dx, 1e-12)
    check("realized substrate top", realized["h_sub_realized_m"],
          H_CLOSED_FORM + dx, 1e-12)
    lines.append(
        f"  dielectric under the strip         "
        f"{planes[0] * 1e6:12.6f} um  (the closed form's h)")
    lines.append(
        f"  trace realization kind             "
        f"{realized['trace_realization_kind']}")
    lines.append(
        f"  realized trace y bounds            "
        f"[{realized['trace_y_lo_realized_m'] * 1e6:.4f}, "
        f"{realized['trace_y_hi_realized_m'] * 1e6:.4f}] um, centre "
        f"{realized['trace_y_centre_realized_m'] * 1e6:.4f} um  (RECORDED, "
        f"not asserted)")
    text = "\n".join(lines)
    print(text)
    if problems:
        raise SystemExit(
            "ABORT: the rasterizer did not build the board this ladder "
            "assumes, so nothing below would be a measurement of the declared "
            "problem.\n  - " + "\n  - ".join(problems))
    return lines


def assert_probe_planes(ref_geom: dict) -> list[str]:
    """The five probe planes per port must sit at the SAME metres coordinates
    on every rung and at every box scale.

    This is the invariant the whole ladder rests on -- it is what makes three
    different meshes a measurement of one arrangement rather than three
    arrangements.  It also answers, by test rather than by reading the code,
    whether the port and probe placement depends on the lateral domain size:
    ``y_centre`` is ``LY/2``, so a box-scaled run moves the port's y
    coordinate, and if anything downstream let that leak into the propagation
    axis these coordinates would move and this assert would refuse the run.
    """
    lines = ["PROBE PLANES (must be identical in metres at every dx and box "
             "scale)"]
    problems = []
    for port, want in CANONICAL_PROBE_X_M.items():
        got = ref_geom[port]["probe_x_m"]
        ok = len(got) == len(want) and all(
            abs(g - w) <= 1e-12 for g, w in zip(got, want))
        lines.append(
            f"  {port}  " + " ".join(f"{v * 1e3:7.4f}" for v in got)
            + f" mm   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(
                f"{port}: probe planes {[round(v * 1e3, 6) for v in got]} mm, "
                f"expected {[round(v * 1e3, 6) for v in want]} mm")
    text = "\n".join(lines)
    print(text)
    if problems:
        raise SystemExit(
            "ABORT: the probe planes are not where the ladder pins them, so "
            "this run would not be comparable with the other rungs.\n  - "
            + "\n  - ".join(problems))
    return lines


def assert_dx50_identity(ref_geom: dict) -> list[str]:
    """At dx = 50 um this must build what the committed builder built."""
    fx = json.loads(FIXTURE.read_text())
    want = fx["reference_plane_geometry"]
    lines = ["DX = 50 um IDENTITY ASSERT (against the committed fixture "
             f"{FIXTURE.relative_to(REPO_ROOT)})"]
    problems = []

    def check(name: str, got, expected, tol: float | None = None) -> None:
        if tol is None:
            ok = got == expected
        else:
            ok = abs(float(got) - float(expected)) <= tol
        lines.append(f"  {name:34s} got {got!s:24s} fixture {expected!s:24s} "
                     f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{name}: got {got!r}, fixture has {expected!r}")

    check("grid shape", ref_geom["_grid"]["shape"], want["_grid"]["shape"])
    check("grid interior cells", ref_geom["_grid"]["interior_cells"],
          want["_grid"]["interior_cells"])
    for port in ("msl_0", "msl_1"):
        check(f"{port} probe0_x_m", ref_geom[port]["probe0_x_m"],
              want[port]["probe0_x_m"], tol=1e-12)
        check(f"{port} n_probe_offset", ref_geom[port]["n_probe_offset"],
              want[port]["n_probe_offset"])
        check(f"{port} n_probe_spacing", ref_geom[port]["n_probe_spacing"],
              want[port]["n_probe_spacing"])
        check(f"{port} n_probes", ref_geom[port]["n_probes"],
              want[port]["n_probes"])
        check(f"{port} feed_x_m", ref_geom[port]["feed_x_m"],
              want[port]["feed_x_m"], tol=1e-12)
    text = "\n".join(lines)
    print(text)
    if problems:
        raise SystemExit(
            "ABORT: at dx = 50 um this script does not build what the "
            "committed builder built, so the ladder's bottom rung is not the "
            "fixture's board.\n  - " + "\n  - ".join(problems))
    return lines


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
def load_referee_case():
    """The crossval case module, loaded by path (its filename starts with a
    digit, so it is not importable by name).  Same idiom as
    ``tests/crossval/test_msl_phase_referee_header.py``."""
    spec = importlib.util.spec_from_file_location(
        "_msl_phase_referee_case", REFEREE_CASE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the crossval case at {REFEREE_CASE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def closed_form_beta(freqs_hz: np.ndarray) -> tuple[np.ndarray, float, float]:
    """beta = 2*pi*f*sqrt(eps_eff)/c0 on the board rfx builds.

    eps_eff from the repo's own ``hammerstad_jensen_z0_eps_eff`` -- not
    retyped -- at W = 600 um, h = 250 um, eps_r = 3.66, and c0 exact.
    """
    z0_cf, eps_eff_cf = hammerstad_jensen_z0_eps_eff(
        W_TRACE, H_CLOSED_FORM, EPS_R)
    beta = 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) * float(
        np.sqrt(eps_eff_cf)) / C0
    return beta, float(eps_eff_cf), float(z0_cf)


def refit_beta_float64(v: np.ndarray, x: np.ndarray,
                       beta_start: np.ndarray) -> tuple[np.ndarray, dict]:
    """float64 Levenberg-Marquardt on V_n = a e^{-j beta x} + g e^{+j beta x}.

    Imitates ``beta_from_least_squares`` in
    ``scripts/diagnostics/msl_beta_fit_synthetic_accuracy.py`` (branch
    fix/msl-beta-fit-accuracy-and-envelope-args), the witness that recovered a
    known beta to 1e-15 on synthetic data.  The start is deliberately the
    closed-form anchor, not the production estimate, so a shared local minimum
    cannot be mistaken for agreement.

    Returns the per-bin beta AND the per-bin fit quality.  A refitted beta with
    no convergence flag beside it is a surface metric: on a bin whose phasors
    are noise the solver still returns a number, and nothing downstream would
    say so.  ``cost`` is the scale-normalized least-squares cost, ``gamma_mag``
    the fitted backward/forward amplitude ratio (a healthy thru is small), and
    ``status`` / ``success`` come straight from SciPy.
    """
    from scipy.optimize import least_squares

    xv = np.asarray(x, dtype=np.float64)
    out = np.empty(v.shape[0], dtype=np.float64)
    quality: dict = {"cost": [], "success": [], "status": [],
                     "gamma_over_alpha_mag": [], "v_scale": []}
    for k in range(v.shape[0]):
        vv = np.asarray(v[k], dtype=np.complex128)
        scale = np.max(np.abs(vv))
        quality["v_scale"].append(float(scale) if np.isfinite(scale) else None)
        if not np.isfinite(scale) or scale == 0.0:
            out[k] = np.nan
            quality["cost"].append(None)
            quality["success"].append(False)
            quality["status"].append(-99)
            quality["gamma_over_alpha_mag"].append(None)
            continue
        vv = vv / scale

        def residual(p, vv=vv):
            a = p[0] + 1j * p[1]
            g = p[2] + 1j * p[3]
            model = a * np.exp(-1j * p[4] * xv) + g * np.exp(+1j * p[4] * xv)
            d = vv - model
            return np.concatenate([d.real, d.imag])

        sol = least_squares(
            residual, x0=[1.0, 0.0, 0.0, 0.0, float(beta_start[k])],
            xtol=1e-15, ftol=1e-15, gtol=1e-15)
        out[k] = float(sol.x[4])
        alpha = complex(sol.x[0], sol.x[1])
        gamma = complex(sol.x[2], sol.x[3])
        quality["cost"].append(float(sol.cost))
        quality["success"].append(bool(sol.success))
        quality["status"].append(int(sol.status))
        quality["gamma_over_alpha_mag"].append(
            float(abs(gamma) / abs(alpha)) if abs(alpha) > 0 else None)
    return out, quality


def production_beta0(freqs_hz: np.ndarray) -> np.ndarray:
    """``beta0`` exactly as ``rfx/sparams/msl.py`` anchors it for this port:
    the DECLARED board h = 254 um, not the realized 250 um."""
    from rfx.core.yee import EPS_0 as _EPS_0, MU_0 as _MU_0
    c0_rfx = 1.0 / float(np.sqrt(_MU_0 * _EPS_0))
    _z0, eps_eff = hammerstad_jensen_z0_eps_eff(W_TRACE, H_SUB, EPS_R)
    return 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64) * float(
        np.sqrt(eps_eff)) / c0_rfx


def derive(core: dict, dump_path: Path, dx: float) -> dict:
    """E(dx) = signed beta deviation - one-cell-thickness prediction.

    Per bin and band-mean over the gated bins, for the production beta and for
    the float64 refit of the same stored phasors.
    """
    case = load_referee_case()
    freqs = np.asarray(core["freqs_hz"], dtype=np.float64)
    beta_prod = np.asarray(core["beta_production_real"], dtype=np.float64)
    beta_cf, eps_eff_cf, z0_cf = closed_form_beta(freqs)
    gate = (freqs >= GATE_F_LO_HZ) & (freqs <= GATE_F_HI_HZ)

    # The one-cell-thickness prediction, from the crossval case's own term,
    # with t = dx passed by KEYWORD (the call site in that file passes t_m and
    # dx_m in swapped positional order -- inert at dx = 50 where both are
    # 50 um, wrong at any second dx, which is exactly this ladder).
    bg_eps_frac = case._bahl_garg_thickness_eps_eff_dev_frac(
        eps_r=EPS_R, w_m=W_TRACE, h_m=H_CLOSED_FORM, t_m=dx)
    bg_beta_frac = case._beta_frac_from_eps_eff_frac(bg_eps_frac)

    out: dict = {
        "closed_form": {
            "w_m": W_TRACE, "h_m": H_CLOSED_FORM, "eps_r": EPS_R,
            "eps_eff": eps_eff_cf, "z0_ohm": z0_cf,
            "c0_m_s": C0,
            "c0_note": ("exact SI c. The crossval case uses _C0 = 2.998e8, "
                        "which shifts a beta deviation by +0.0025 %."),
            "source": "rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff",
            "beta_rad_per_m": beta_cf.tolist(),
        },
        "thickness_prediction": {
            "t_m": dx,
            "h_m": H_CLOSED_FORM,
            "eps_eff_dev_frac": float(bg_eps_frac),
            "beta_dev_frac": float(bg_beta_frac),
            "source": ("validation/crossval/20_msl_phase_referee.py "
                       "_bahl_garg_thickness_eps_eff_dev_frac, t_m passed by "
                       "keyword"),
            "alternatives": {
                f"h={h * 1e6:g}um": float(case._beta_frac_from_eps_eff_frac(
                    case._bahl_garg_thickness_eps_eff_dev_frac(
                        eps_r=EPS_R, w_m=W_TRACE, h_m=h, t_m=dx)))
                for h in (H_CLOSED_FORM, H_SUB, 300e-6)
            },
        },
        "gate_band_hz": [GATE_F_LO_HZ, GATE_F_HI_HZ],
        "gate_bin_indices": [int(k) for k in np.flatnonzero(gate)],
        "gate_bin_freqs_hz": freqs[gate].tolist(),
        "arms": {},
    }

    arms = {"production": beta_prod}

    # The refit arm, from the dumped phasors. Absent (with a reason) rather
    # than invented if the dump is missing.
    refit_note = None
    if dump_path.exists():
        with np.load(dump_path, allow_pickle=True) as npz:
            raw_v = np.asarray(npz["raw_v"])
            dump_freqs = np.asarray(npz["freqs_hz"], dtype=np.float64)
        # beta is reported from the driven == 0 run at port 0.
        v_port0 = np.asarray(raw_v[0, 0, :N_PROBES, :]).T  # (n_freqs, n_probes)
        x_probe = np.asarray(
            core["reference_plane_geometry"]["msl_0"]["probe_x_m"],
            dtype=np.float64)
        if not np.allclose(dump_freqs, freqs):
            refit_note = "dump frequency grid differs from the result grid"
        else:
            beta0 = production_beta0(freqs)
            arms["refit_float64_lsq"], out["refit_fit_quality"] = \
                refit_beta_float64(v_port0, x_probe, beta0)
            # The per-bin residual the result object does not expose: the
            # production extractor re-run on the dumped phasors.
            try:
                import jax.numpy as jnp
                from rfx.probes.msl_wave_decomp import extract_msl_nprobe
                res = extract_msl_nprobe(
                    jnp.asarray(v_port0), jnp.asarray(x_probe),
                    jnp.asarray(np.ones(len(freqs), dtype=np.complex128)),
                    jnp.asarray(beta0),
                )
                out["production_fit_residual_replayed"] = np.asarray(
                    jax.device_get(res["residual"]), dtype=float).tolist()
                out["production_beta_replayed_real"] = np.asarray(
                    jax.device_get(jnp.real(res["beta"])), dtype=float).tolist()
                out["production_beta_railed_replayed"] = np.asarray(
                    jax.device_get(res["beta_railed"])).astype(bool).tolist()
            except Exception as exc:  # pragma: no cover - diagnostic only
                out["production_fit_residual_replayed"] = None
                out["production_fit_residual_replay_error"] = repr(exc)
            out["refit_inputs"] = {
                "probe_x_m": x_probe.tolist(),
                "beta0_anchor_rad_per_m": beta0.tolist(),
                "beta0_anchor_board_h_m": H_SUB,
                "raw_v_slice": "raw_v[driven=0, port=0, :n_probes, :]",
            }
    else:
        refit_note = f"raw dump not written at {dump_path}"
    if refit_note:
        out["refit_absent_reason"] = refit_note

    for name, beta in arms.items():
        beta = np.asarray(beta, dtype=np.float64)
        dev = beta / beta_cf - 1.0
        E = dev - bg_beta_frac
        out["arms"][name] = {
            "beta_rad_per_m": beta.tolist(),
            "signed_beta_dev_frac": dev.tolist(),
            "E_frac": E.tolist(),
            "gated": {
                "signed_beta_dev_frac_mean": float(np.mean(dev[gate])),
                "signed_beta_dev_frac_min": float(np.min(dev[gate])),
                "signed_beta_dev_frac_max": float(np.max(dev[gate])),
                "E_frac_mean": float(np.mean(E[gate])),
                "E_frac_min": float(np.min(E[gate])),
                "E_frac_max": float(np.max(E[gate])),
            },
        }
    return out


# ---------------------------------------------------------------------------
# stdout tee
# ---------------------------------------------------------------------------
class _Tee(io.TextIOBase):
    """Write to the real stdout AND to a buffer, so preflight output lands in
    the job log and in a saved file without being suppressed in either."""

    def __init__(self, real):
        self._real = real
        self.buf = io.StringIO()

    def write(self, s):  # type: ignore[override]
        self._real.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):  # type: ignore[override]
        self._real.flush()


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dx-um", type=float, required=True,
                   choices=[50.0, 25.0, 12.5],
                   help="mesh rung, in micrometres")
    p.add_argument("--out-dir", required=True,
                   help="directory for result_core.json, result.json, the raw "
                        "phasor dump, the preflight capture and the figure")
    p.add_argument("--smoke", action="store_true",
                   help="stop after --smoke-steps timesteps and still write "
                        "every file, marking the JSON smoke: true. Proves the "
                        "geometry build, the realized-board asserts and the "
                        "output path; the numbers are NOT physics.")
    p.add_argument("--smoke-steps", type=int, default=200)
    p.add_argument("--geometry-only", action="store_true",
                   help="build the grid, run the asserts, write "
                        "geometry.json, and stop before time stepping")
    p.add_argument("--box-scale", type=float, default=1.0,
                   help="multiply the lateral side allowance and the air "
                        "height by this factor and nothing else "
                        "(hypothesis H-d, lateral/top truncation). Default "
                        "1 reproduces the committed builder's box; at "
                        "default the dx = 50 identity assert still runs "
                        "and must still pass.")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    # NOT ``args.dx_um * 1e-6``. That product lands one ULP BELOW the literal
    # ``50e-6``, which makes ``LX / dx`` read 280.00000000000006 instead of
    # 280.0, the rasterizer round one extra cell up, and the dx = 50 um grid
    # come out (298, 66, 45) against the committed fixture's (297, 66, 45) --
    # i.e. a different board at the rung that is supposed to reproduce the
    # fixture exactly. The dx = 50 identity assert caught it. Going through
    # Decimal reproduces the source literal bit for bit at every rung.
    dx = float(Decimal(str(args.dx_um)) * Decimal("1e-6"))
    tag = _TAGS[args.dx_um]
    if args.box_scale != 1.0:
        tag = f"{tag}-box{args.box_scale:g}".replace(".", "p")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rfx_file = str(Path(rfx.__file__).resolve())
    print(f"rfx.__file__ = {rfx_file}")
    print(f"repo root    = {REPO_ROOT}")
    if not rfx_file.startswith(str(REPO_ROOT) + "/"):
        raise SystemExit(
            f"ABORT: rfx imported from {rfx_file}, which is NOT this checkout "
            f"({REPO_ROOT}). An installed rfx shadows the worktree and the run "
            f"would not measure the pinned commit.")
    print(f"jax {jax.__version__} | backend {jax.default_backend()} | "
          f"x64 {jax.config.read('jax_enable_x64')}")
    print(f"dx = {args.dx_um} um   tag = {tag}")

    tee = _Tee(sys.stdout)
    real_stdout = sys.stdout
    t_build = time.time()
    sys.stdout = tee
    try:
        sim, scaling = build_sim(dx, box_scale=args.box_scale)
        ref_geom = reference_plane_geometry(sim)
        realized = realized_board_geometry(sim, dx)
        realized.update(realized_trace_geometry(sim))
        print()
        print(f"HELD CONSTANT IN METRES AT dx = {args.dx_um} um")
        print(f"  CPML thickness      {CPML_THICKNESS_M * 1e6:9.3f} um = "
              f"{scaling['cpml_layers']:3d} layers x {args.dx_um} um")
        print(f"  probe-0 offset      {PROBE_OFFSET_M * 1e6:9.3f} um = "
              f"{scaling['n_probe_offset_cells']:3d} cells x {args.dx_um} um")
        print(f"  probe spacing       {PROBE_SPACING_M * 1e6:9.3f} um = "
              f"{scaling['n_probe_spacing_cells']:3d} cells x {args.dx_um} um")
        _ly, _lz = domain_ly_lz(args.box_scale)
        print(f"  domain (declared)   LX {LX * 1e3:.4f} mm  LY "
              f"{_ly * 1e6:.3f} um  LZ {_lz * 1e6:.3f} um"
              f"   [box_scale {args.box_scale:g}]")
        print(f"  grid shape          {ref_geom['_grid']['shape']}  "
              f"({ref_geom['_grid']['n_cells_total']:,} cells)  dt "
              f"{ref_geom['_grid']['dt_s']:.6e} s")
        print(f"  realized clear      "
              f"{ {k: round(v * 1e6, 4) for k, v in ref_geom['_grid']['rasterized_clear_m'].items()} } um"
              f"   (RECORDED, not asserted -- see the module docstring)")
        print()
        assert_lines = assert_realized_board(realized, dx)
        print()
        assert_lines += assert_probe_planes(ref_geom)
        print()
        if args.dx_um == 50.0 and args.box_scale == 1.0:
            assert_lines += assert_dx50_identity(ref_geom)
        elif args.dx_um == 50.0:
            print(f"DX = 50 um IDENTITY ASSERT: skipped at box_scale "
                  f"{args.box_scale:g} (a scaled box changes the grid's y "
                  f"and z extent on purpose; the probe-plane and "
                  f"realized-board asserts above still bind)")
        else:
            print(f"DX = 50 um IDENTITY ASSERT: skipped at dx = {args.dx_um} um "
                  f"(it compares against the dx = 50 um fixture)")
        build_elapsed = time.time() - t_build

        geometry_record = {
            "dx_m": dx, "dx_um": args.dx_um, "tag": tag,
            "box_scale": float(args.box_scale),
            "rfx_file": rfx_file,
            "scaling": scaling,
            "reference_plane_geometry": ref_geom,
            "realized": {k: v for k, v in realized.items()
                         if k != "fidelity_rows"},
            "fidelity_rows": realized["fidelity_rows"],
            "realized_domain_clear_m": ref_geom["_grid"]["rasterized_clear_m"],
            "build_elapsed_s": round(build_elapsed, 2),
        }
        (out_dir / f"geometry_{tag}.json").write_text(
            json.dumps(geometry_record, indent=2, sort_keys=True, default=str)
            + "\n")
        print(f"\nwrote {out_dir / f'geometry_{tag}.json'} "
              f"(build {build_elapsed:.1f} s)")

        if args.geometry_only:
            return 0  # the finally below still saves the preflight capture

        # --- solve ----------------------------------------------------------
        dump_path = out_dir / f"raw_nprobe_{tag}.npz"
        print(f"\n=== solve: compute_msl_s_matrix(n_freqs={N_FREQS}, "
              f"num_periods={NUM_PERIODS}"
              + (f", n_steps={args.smoke_steps} [SMOKE])" if args.smoke
                 else ")") + " ===")
        # the dump records S_raw / passivity_correction beside S, which exist
        # only on the projected path; the committed summaries were taken so
        kwargs: dict = {"n_freqs": N_FREQS, "num_periods": NUM_PERIODS,
                        "raw_3probe_dump_path": str(dump_path),
                        "enforce_passivity": True}
        if args.smoke:
            kwargs["n_steps"] = int(args.smoke_steps)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sim.compute_msl_s_matrix(**kwargs)
        elapsed = time.time() - t0
        # Nothing is suppressed: every warning caught above is re-emitted here
        # verbatim, into the same teed stream as the preflight banner.
        print(f"\n=== {len(caught)} warning(s) raised during the solve "
              f"(verbatim) ===")
        warning_records = []
        for w in caught:
            text = (f"[{w.category.__name__}] {w.filename}:{w.lineno}\n"
                    f"{w.message}")
            print(text)
            print("-" * 72)
            warning_records.append({
                "category": w.category.__name__,
                "filename": str(w.filename), "lineno": int(w.lineno),
                "message": str(w.message),
            })
    finally:
        sys.stdout = real_stdout
        # In the finally, not after it: a solver that raises still leaves its
        # preflight banner on disk, which is the part a failed multi-hour run
        # most needs to be readable afterwards.
        (out_dir / f"preflight_{tag}.txt").write_text(tee.buf.getvalue())

    # --- PERSIST FIRST. Optional stages come after this write. --------------
    def _np(x):
        return None if x is None else np.asarray(jax.device_get(x))

    S = _np(result.S)
    beta = _np(result.beta)
    Z0 = _np(result.Z0)
    freqs = _np(result.freqs)
    S_raw = _np(result.S_raw)
    core = {
        "smoke": bool(args.smoke),
        "smoke_steps": int(args.smoke_steps) if args.smoke else None,
        "dx_m": dx, "dx_um": args.dx_um, "tag": tag,
        "box_scale": float(args.box_scale),
        "elapsed_s": round(elapsed, 2),
        "rfx_file": rfx_file,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "scaling": scaling,
        "reference_plane_geometry": ref_geom,
        "realized": {k: v for k, v in realized.items() if k != "fidelity_rows"},
        "fidelity_rows": realized["fidelity_rows"],
        "realized_domain_clear_m": ref_geom["_grid"]["rasterized_clear_m"],
        "assert_output": assert_lines,
        "freqs_hz": freqs.tolist(),
        "beta_production_real": np.real(beta).tolist(),
        "beta_production_imag": np.imag(beta).tolist(),
        "s11": [[float(c.real), float(c.imag)] for c in S[0, 0, :]],
        "s21": [[float(c.real), float(c.imag)] for c in S[1, 0, :]],
        "z0_port0": [[float(c.real), float(c.imag)] for c in Z0[0, :]],
        "z0_port1": [[float(c.real), float(c.imag)] for c in Z0[1, :]],
        "witnesses": {
            "settling_db": None if result.settling_db is None
            else _np(result.settling_db).tolist(),
            "settling_db_definition": (
                "per driven run, the WORST over all port probe planes of "
                "10*log10(mean Ez^2 over the last 10 % of the record / peak "
                "Ez^2). This is a PROBE-PLANE Ez ring-down ratio. "
                "compute_msl_s_matrix exposes no total-domain field-energy "
                "witness and none is fabricated here; above -40 dB the record "
                "was truncated before the structure rang down."),
            "beta_railed": None if result.beta_railed is None
            else _np(result.beta_railed).astype(bool).tolist(),
            "reliable": None if result.reliable is None
            else _np(result.reliable).astype(bool).tolist(),
            "cond_a": None if getattr(result, "cond_a", None) is None
            else _np(result.cond_a).tolist(),
            "passivity_correction": None
            if result.passivity_correction is None
            else _np(result.passivity_correction).tolist(),
            "assembly": result.assembly,
            "reference_impedances_ohm": None
            if result.reference_impedances is None
            else _np(result.reference_impedances).tolist(),
            "probe_clearance": [str(pc) for pc in (result.probe_clearance or ())],
            "s_raw_present": S_raw is not None,
            "per_bin_fit_residual": (
                "NOT exposed by MSLSMatrixResult and NOT in the npz dump. "
                "extract_msl_nprobe returns it; compute_msl_s_matrix does not "
                "carry it out. Recomputed in result.json "
                "(production_fit_residual_replayed) by re-running the "
                "production extractor on the dumped phasors."),
            "warnings": warning_records,
        },
        "raw_dump_path": str(dump_path),
        "raw_dump_note": (
            "rfx.msl_nprobe_dump schema v4 via the supported "
            "raw_3probe_dump_path keyword. raw_v has shape (n_driven, "
            "n_ports, n_probes_max, n_freqs) -- the probe-plane voltage "
            "phasors, so beta can be re-fitted offline with another "
            "estimator."),
    }
    if S_raw is not None:
        core["s11_raw"] = [[float(c.real), float(c.imag)] for c in S_raw[0, 0, :]]
        core["s21_raw"] = [[float(c.real), float(c.imag)] for c in S_raw[1, 0, :]]
    core_path = out_dir / f"result_core_{tag}.json"
    core_path.write_text(
        json.dumps(core, indent=2, sort_keys=True, default=str) + "\n")
    print(f"PERSISTED {core_path}  (elapsed {elapsed:.1f} s) -- optional "
          f"stages start now and cannot cost this file")

    # --- optional stages ----------------------------------------------------
    full = dict(core)
    try:
        full["derived"] = derive(core, dump_path, dx)
        d = full["derived"]
        for name, arm in d["arms"].items():
            g = arm["gated"]
            print(f"  [{name:18s}] gated signed beta dev "
                  f"{g['signed_beta_dev_frac_min'] * 100:+.4f} .. "
                  f"{g['signed_beta_dev_frac_max'] * 100:+.4f} % "
                  f"(mean {g['signed_beta_dev_frac_mean'] * 100:+.4f} %)   "
                  f"E mean {g['E_frac_mean'] * 100:+.4f} pp")
        print(f"  thickness prediction (t = dx = {args.dx_um} um, h = 250 um): "
              f"{d['thickness_prediction']['beta_dev_frac'] * 100:+.4f} %")
    except Exception as exc:
        full["derived"] = None
        full["derived_error"] = repr(exc)
        print(f"  derived stage FAILED (core is already persisted): {exc!r}")

    result_path = out_dir / f"result_{tag}.json"
    result_path.write_text(
        json.dumps(full, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {result_path}")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            d = full.get("derived")
            fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
            f_ghz = np.asarray(core["freqs_hz"]) / 1e9
            axes[0].plot(f_ghz, core["beta_production_real"], "o-",
                         label="beta production")
            if d:
                axes[0].plot(f_ghz, d["closed_form"]["beta_rad_per_m"], "k--",
                             label="closed form, h = 250 um")
                if "refit_float64_lsq" in d["arms"]:
                    axes[0].plot(
                        f_ghz, d["arms"]["refit_float64_lsq"]["beta_rad_per_m"],
                        "s--", ms=3, label="beta float64 refit")
            axes[0].set_ylabel("beta [rad/m]")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)
            if d:
                for name, arm in d["arms"].items():
                    axes[1].plot(f_ghz,
                                 100 * np.asarray(arm["signed_beta_dev_frac"]),
                                 "o-", ms=3, label=f"signed dev, {name}")
                axes[1].axhline(
                    100 * d["thickness_prediction"]["beta_dev_frac"],
                    color="k", ls=":",
                    label=f"one-cell thickness prediction (t = {args.dx_um} um)")
                axes[1].axvspan(GATE_F_LO_HZ / 1e9, GATE_F_HI_HZ / 1e9,
                                color="0.85", zorder=0)
            axes[1].set_ylabel("signed beta deviation [%]")
            axes[1].set_xlabel("frequency [GHz]")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fig_path = out_dir / f"beta_ladder_{tag}.png"
            fig.savefig(fig_path, dpi=150)
            print(f"wrote {fig_path}")
        except Exception as exc:  # pragma: no cover - optional
            print(f"  figure stage FAILED (results already persisted): {exc!r}")

    import resource
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"peak RSS {peak_gib:.2f} GiB")
    summary = {
        "tag": tag, "dx_um": args.dx_um, "box_scale": float(args.box_scale),
        "smoke": bool(args.smoke),
        "elapsed_s": round(elapsed, 2),
        "rc": 0,
        "grid_shape": ref_geom["_grid"]["shape"],
        "n_cells_total": ref_geom["_grid"]["n_cells_total"],
        "peak_rss_gib": round(peak_gib, 3),
        "settling_db": core["witnesses"]["settling_db"],
        "n_warnings": len(warning_records),
    }
    if full.get("derived"):
        summary["gated"] = {
            k: v["gated"] for k, v in full["derived"]["arms"].items()}
        summary["thickness_prediction_beta_frac"] = \
            full["derived"]["thickness_prediction"]["beta_dev_frac"]
    (out_dir / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_dir / f'summary_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
