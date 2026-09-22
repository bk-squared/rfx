"""The MSL open-stub notch filter — cross-validated against openEMS's own tutorial.

A 50 Ω microstrip line 600 µm wide on 254 µm of lossless εr = 3.66 carries a
12 mm open-circuit stub of the same width, branching off at the middle of the
line.  The stub is a quarter wavelength near 3.6 GHz, where its open end
transforms into a short across the line, so |S21| has a deep transmission
minimum whose FREQUENCY is set by the stub's electrical length and whose SHAPE
(the −10 dB bandwidth) is set by the stub-to-line impedance ratio.  The
T-junction has no closed form, so rfx's |S21| curve is compared with frozen
external results instead of with an analytic expression.

The three boards
----------------
Substrate, trace and stub are the same in all three; only the length of the
line either side of the stub differs.

* the openEMS tutorial record — 50 mm of line each side of the stub centre,
  the tutorial's own arms;
* the Palace FEM record — 2.5 mm each side (5 mm port plane to port plane);
* rfx here — ``ARM_LENGTH_M`` = 10 mm each side, in rfx's own box.

rfx does not reproduce the tutorial's 100 mm-long box because at h/6 it would
cost the weekly lane about an hour.  A lossless line's |S21| and |S11|
MAGNITUDES do not depend on how long the line is beyond the ripple the port
mismatch leaves, so the magnitudes stay comparable; the PHASES do not, and the
record says so itself (``meta.comparability``: its phases are referenced at the
tutorial's measurement planes on 50 mm arms).  This case compares magnitudes
only, and measures the length premise once on rfx itself — the arm-length
witness at the coarsest rung rebuilds the same filter on 15.08 mm arms and
reports how far the two curves move apart.

References (``reference/``, provenance in ``reference/PROVENANCE.md``):

* ``openems_tutorial.json`` — openEMS 0.37.0, FDTD, the tool's own
  ``MSL_NotchFilter`` tutorial reproduced and then re-run at finer cell sizes.
  1601 points, 0.001–7 GHz, LINEAR magnitude and degrees.  Three rungs:
  ``stage_a`` (= ``stage_b_coarse``, the tutorial's own mesh, 447.7 µm, 4 cells
  across the substrate), ``stage_b_mid`` (316.6 µm, 6 cells) and
  ``stage_b_fine`` (223.9 µm, 8 cells).  **JUDGED against ``stage_b_fine``**;
  ``stage_a`` is printed as the reference's own mesh statement.
* ``palace_fem.json`` — Palace, frequency-domain FEM on conformal tetrahedra,
  order 2, lossless, first-order absorbing far box, on the 5 mm board.  Two
  meshes: ``coarse`` (lc 0.12 mm, 101 points, 2–7 GHz) and ``mid``
  (lc 0.085 mm, 33 points, 3.2–4.0 GHz).  Arrays are LINEAR magnitude.
  REPORTED, never judged: the record conserves 22–86 % of the power it
  receives, so its magnitudes cannot be held to a 2 dB bar; its notch
  frequency is printed beside its ``max_energy_sum``.

The ladder is dx = h_sub/n so the substrate top always lands on a node plane
(issue #723): 127 µm (n = 2), 63.5 µm (n = 4), 42.33 µm (n = 6).  The
thresholds are the v2 accuracy bar (1 % in frequency, 2 dB in magnitude) and
the three rules the PI approved with it on 2026-09-22: the mesh statement
(the last two rungs within 1 %), the deep-null exclusion (a bin where either
curve is below −20 dB is judged by position only) and the ring-down witness
(−40 dB, the repo's rule).  Every one is a named constant below with its
source.  No threshold is derived from a run, and this case commits no record
of a run.

How to run it
-------------
The ladder is marked ``gpu`` and ``slow``, so the PR lane never collects it::

    pytest tests/crossval/msl_notch_filter -m gpu

Set ``RFX_CROSSVAL_FIG_DIR`` to keep the |S21| figure somewhere durable;
without it the figure goes under pytest's ``tmp_path``.  Set
``RFX_MSL_NOTCH_RUNGS`` to a comma-separated list of cell sizes in metres
(e.g. ``RFX_MSL_NOTCH_RUNGS=127e-6``) to run part of the ladder — the mesh
statement and the comparison then apply to the rungs given, which is a
diagnostic, not the case's verdict.  Unset, the full ladder runs.

The two fast tests carry no mark: they build the structure and read the
reference files, and never step the FDTD.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

from tests._realized_geometry import _node_line, realized

# ---------------------------------------------------------------- geometry
# Every length in metres, every frequency in hertz.  Substrate, trace and stub
# are the numbers both references were made with (their `meta` blocks carry the
# same values); the box and the arms are rfx's own — see the module docstring.
EPS_R_SUBSTRATE = 3.66
SUBSTRATE_THICKNESS_M = 254e-6
TRACE_WIDTH_M = 600e-6
STUB_LENGTH_M = 12e-3
PORT_IMPEDANCE_OHM = 50.0
FREQ_MAX_HZ = 7e9
CPML_LAYERS = 8

# dx = h_sub / n keeps the substrate top on a node plane (issue #723).
LADDER_M = (
    SUBSTRATE_THICKNESS_M / 2,   # 127 µm
    SUBSTRATE_THICKNESS_M / 4,   # 63.5 µm
    SUBSTRATE_THICKNESS_M / 6,   # 42.33 µm
)
DX_COARSEST_M = LADDER_M[0]

# --- along the line ------------------------------------------------------
# ARM_LENGTH_M is the port plane to the stub centre, each side.  It is set by
# what the MSL port needs downstream of the feed: the deepest probe of the
# three-probe extractor sits a quarter guided wavelength back at the lowest
# frequency it serves, 4.79 mm on this substrate, and it must be clear of the
# stub.  PORT_MARGIN_M is the port plane to the x face, set by the upstream
# probe offset max(λ/4π, 5·h) = 1.8 mm.  Both from the preflight text of run
# 369367263038 at h/6.
ARM_LENGTH_M = 10e-3
PORT_MARGIN_M = 2e-3
# The same filter on longer arms; only the arm-length witness builds it.  The
# arm grows by 5.08 mm rather than 5 mm because lengthening the arm moves the
# stub with it: 5.08 mm is a whole number of coarsest cells (40 × 127 µm, and
# so also 80 × 63.5 µm and 120 × 42.33 µm), which leaves the stub at the same
# place inside a cell and the longer board realizes the SAME filter.  A 5 mm
# move puts the stub's 600 µm footprint on five node columns instead of four
# at the coarsest rung, and the witness would then be measuring a stub that
# got wider, not an arm that got longer.
WITNESS_ARM_LENGTH_M = 15.08e-3
assert abs((WITNESS_ARM_LENGTH_M - ARM_LENGTH_M) / DX_COARSEST_M
           - round((WITNESS_ARM_LENGTH_M - ARM_LENGTH_M) / DX_COARSEST_M)) < 1e-9


def _x_geometry(arm_length_m: float) -> tuple[float, float]:
    """``(domain x extent, stub centre x)`` for arms of ``arm_length_m``.

    The two port planes sit at ``PORT_MARGIN_M`` and
    ``PORT_MARGIN_M + 2·arm_length_m``; the stub centre is halfway between.
    """
    return 2.0 * (arm_length_m + PORT_MARGIN_M), arm_length_m + PORT_MARGIN_M


DOMAIN_X_M, STUB_CENTRE_X_M = _x_geometry(ARM_LENGTH_M)   # 24.0 mm, 12.0 mm

# --- across the line -----------------------------------------------------
# The metal must stay clear of the CPML by the port's lateral requirement,
# 2·h_sub plus eight cells.  "Eight cells" is a physical length only once a
# cell size is named, so it is evaluated at the COARSEST rung and HELD for
# every rung: the board a finer rung solves is then the same board, not a
# smaller one.
LATERAL_CLEARANCE_MIN_M = 2.0 * SUBSTRATE_THICKNESS_M + 8.0 * DX_COARSEST_M  # 1.524 mm
# That minimum is exactly twelve coarsest cells, and every finer rung's cell
# size divides it too, so putting the line's near edge there puts it exactly on
# a node line at all three rungs while the stub's edge (11.7 mm) falls inside a
# cell.  A 600 µm footprint then realizes five node rows for the line and four
# node columns for the stub at 127 µm — a 635 µm line beside a 508 µm stub,
# which is not the board either reference solved, and `assert_realized` refuses
# it (measured: rows 5/10/15 against columns 4/9/14 down the ladder).  The
# clearance is a MINIMUM, so it is rounded up to the next 50 µm; that puts the
# line's edge inside a cell at every rung and both footprints realize the same
# width.
LATERAL_CLEARANCE_M = 1.55e-3
assert LATERAL_CLEARANCE_M >= LATERAL_CLEARANCE_MIN_M
TRACE_CENTRE_Y_M = LATERAL_CLEARANCE_M + TRACE_WIDTH_M / 2.0             # 1.850 mm
DOMAIN_Y_M = (TRACE_WIDTH_M + 2.0 * LATERAL_CLEARANCE_M
              + STUB_LENGTH_M + 2.0 * LATERAL_CLEARANCE_M)               # 18.800 mm
DOMAIN_Z_M = SUBSTRATE_THICKNESS_M + 1.5e-3                              # 1.754 mm

# Record length in periods of freq_max — the value the retired script ran
# with, kept because the ring-down witness (SETTLING_DB) confirms it: −109 to
# −111 dB at every rung of a four-rung run h/2 … h/8 (VESSL run 369367263038;
# the case's own ladder below stops at h/6).
NUM_PERIODS = 20.0
N_FREQS = 100
# The band both references cover; the notch is searched inside it, never over
# rfx's wider sweep.
REFERENCE_BAND_HZ = (2.0e9, 7.0e9)

# ------------------------------------------------------------ thresholds
# The v2 accuracy bar (rfx CLAUDE.md, PI 2026-09-20):
FREQ_BAR = 0.01          # resonances, cutoffs, notches, band edges: within 1 %
MAG_BAR_DB = 2.0         # power and magnitude: within 2 dB
# Rules approved by the PI with this case's plan (2026-09-22):
LADDER_AGREEMENT = 0.01  # mesh statement: the last two rungs differ by < 1 %
DEEP_NULL_DB = -20.0     # a bin where either curve is below this is judged by
#                          position only, never by magnitude
# Witnesses on rfx's own record, not comparisons (rfx CLAUDE.md, Validation
# rules): a record that has not rung down to −40 dB is truncation-suspect, and
# a passive structure cannot scatter more power than it receives, so a raw
# extraction more than one percent above the passive bound is a measurement
# artefact — the rung stops instead of being compared.  Measured on the four
# rungs of run 369367263038: settling −109 … −111 dB, excess 0.002 … 0.004.
SETTLING_DB = -40.0
PASSIVITY_EXCESS_BAR = 0.01

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_PALACE_JSON = _REFERENCE_DIR / "palace_fem.json"
_OPENEMS_JSON = _REFERENCE_DIR / "openems_tutorial.json"
# The rung of the openEMS record this case is judged against, and the rung that
# is its own mesh statement.
OPENEMS_JUDGED_STAGE = "stage_b_fine"
OPENEMS_COARSE_STAGE = "stage_b_coarse"


# ------------------------------------------------------------------ build
def build(dx: float, arm_length_m: float = ARM_LENGTH_M) -> Simulation:
    """The reference structure on a uniform ``dx`` mesh.

    Substrate is a lossless dielectric slab filling the board; the trace and
    the stub are zero-thickness PEC SHEETS on the substrate-top node plane
    (both z corners equal — a Box drawn ``h -> h + dx`` would be a one-cell
    VOLUME with a wall on each face, #931).  The ground is the z_lo PEC face.

    ``arm_length_m`` lengthens the line either side of the stub and the box
    with it; nothing else moves.  Only the arm-length witness passes anything
    but the default.
    """
    domain_x, stub_centre_x = _x_geometry(arm_length_m)
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=(domain_x, DOMAIN_Y_M, DOMAIN_Z_M),
        dx=dx,
        cpml_layers=CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=EPS_R_SUBSTRATE)
    sim.add(Box((0.0, 0.0, 0.0), (domain_x, DOMAIN_Y_M, SUBSTRATE_THICKNESS_M)),
            material="substrate")

    y_lo = TRACE_CENTRE_Y_M - TRACE_WIDTH_M / 2.0
    y_hi = TRACE_CENTRE_Y_M + TRACE_WIDTH_M / 2.0
    sim.add(Box((0.0, y_lo, SUBSTRATE_THICKNESS_M),
                (domain_x, y_hi, SUBSTRATE_THICKNESS_M)), material="pec")
    sim.add(Box((stub_centre_x - TRACE_WIDTH_M / 2.0, y_hi, SUBSTRATE_THICKNESS_M),
                (stub_centre_x + TRACE_WIDTH_M / 2.0, y_hi + STUB_LENGTH_M,
                 SUBSTRATE_THICKNESS_M)), material="pec")

    sim.add_msl_port(position=(PORT_MARGIN_M, TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="+x", impedance=PORT_IMPEDANCE_OHM)
    sim.add_msl_port(position=(PORT_MARGIN_M + 2.0 * arm_length_m,
                               TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="-x", impedance=PORT_IMPEDANCE_OHM)
    return sim


def _build_trace_as_volume(dx: float) -> Simulation:
    """Same board, but the main line drawn as a one-cell thick VOLUME.

    Only the mutation test uses this: it is the defect ``assert_realized``
    exists to refuse (two wall planes instead of one sheet plane).
    """
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=(DOMAIN_X_M, DOMAIN_Y_M, DOMAIN_Z_M),
        dx=dx,
        cpml_layers=CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=EPS_R_SUBSTRATE)
    sim.add(Box((0.0, 0.0, 0.0), (DOMAIN_X_M, DOMAIN_Y_M, SUBSTRATE_THICKNESS_M)),
            material="substrate")
    y_lo = TRACE_CENTRE_Y_M - TRACE_WIDTH_M / 2.0
    y_hi = TRACE_CENTRE_Y_M + TRACE_WIDTH_M / 2.0
    sim.add(Box((0.0, y_lo, SUBSTRATE_THICKNESS_M),
                (DOMAIN_X_M, y_hi, SUBSTRATE_THICKNESS_M + dx)), material="pec")
    sim.add(Box((STUB_CENTRE_X_M - TRACE_WIDTH_M / 2.0, y_hi, SUBSTRATE_THICKNESS_M),
                (STUB_CENTRE_X_M + TRACE_WIDTH_M / 2.0, y_hi + STUB_LENGTH_M,
                 SUBSTRATE_THICKNESS_M)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN_M, TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="+x", impedance=PORT_IMPEDANCE_OHM)
    sim.add_msl_port(position=(PORT_MARGIN_M + 2.0 * ARM_LENGTH_M,
                               TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="-x", impedance=PORT_IMPEDANCE_OHM)
    return sim


def _build_short_stub(dx: float, cells_short: int, lift_cells: int = 0) -> Simulation:
    """The board with the stub drawn ``cells_short`` cells shorter than 12 mm
    and, with ``lift_cells``, its near end drawn that many cells clear of the
    main line (the open end then stays where :func:`build` puts it when
    ``cells_short == lift_cells``: a floating strip, not a stub).

    Everything else is what :func:`build` declares.
    """
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=(DOMAIN_X_M, DOMAIN_Y_M, DOMAIN_Z_M),
        dx=dx,
        cpml_layers=CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=EPS_R_SUBSTRATE)
    sim.add(Box((0.0, 0.0, 0.0), (DOMAIN_X_M, DOMAIN_Y_M, SUBSTRATE_THICKNESS_M)),
            material="substrate")
    y_lo = TRACE_CENTRE_Y_M - TRACE_WIDTH_M / 2.0
    y_hi = TRACE_CENTRE_Y_M + TRACE_WIDTH_M / 2.0
    sim.add(Box((0.0, y_lo, SUBSTRATE_THICKNESS_M),
                (DOMAIN_X_M, y_hi, SUBSTRATE_THICKNESS_M)), material="pec")
    sim.add(Box((STUB_CENTRE_X_M - TRACE_WIDTH_M / 2.0, y_hi + lift_cells * dx,
                 SUBSTRATE_THICKNESS_M),
                (STUB_CENTRE_X_M + TRACE_WIDTH_M / 2.0,
                 y_hi + STUB_LENGTH_M - cells_short * dx,
                 SUBSTRATE_THICKNESS_M)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN_M, TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="+x", impedance=PORT_IMPEDANCE_OHM)
    sim.add_msl_port(position=(PORT_MARGIN_M + 2.0 * ARM_LENGTH_M,
                               TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="-x", impedance=PORT_IMPEDANCE_OHM)
    return sim


# -------------------------------------------------------- realized metal
def realized_geometry(sim: Simulation) -> dict:
    """What the lattice built, read from the product's own realization owner.

    ``tests/_realized_geometry.realized`` assembles the simulation without a
    time step and hands back the realized PEC edge masks, so every number here
    is the solver's, not a rule re-derived in the test.

    A zero-thickness sheet IS its set of tangential edges, and *n* footprint
    node rows carry *n* − 1 edges.  Both spans are therefore reported twice:
    ``trace_width_node_span_m`` = (n − 1)·dx is the geometric extent between
    the outermost node rows, ``trace_width_strip_m`` = n·dx is the strip width
    a quasi-TEM formula takes.  The stub length is measured from the main
    line's far node row to the open end.
    """
    from rfx.boundaries.pec import realized_wall_planes

    rz = realized(sim)
    grid = rz.grid
    mx, my, _mz = (np.asarray(e) for e in rz.edge_masks)
    xs, ys, zs = (_node_line(grid, a) for a in (0, 1, 2))
    dx = float(xs[1] - xs[0])

    planes = [int(p) for p in realized_wall_planes(rz.edge_masks, 2)]
    n_volume_cells = 0 if rz.pec_mask is None else int(np.asarray(rz.pec_mask).sum())

    out: dict = {
        "sheet_planes": planes,
        "n_sheet_planes": len(planes),
        "n_declared_sheets": len(rz.sheets),
        "n_volume_cells": n_volume_cells,
        "dx_m": dx,
        "grid_shape": tuple(int(n) for n in (xs.size, ys.size, zs.size)),
        "n_cells": int(xs.size * ys.size * zs.size),
        "n_ports": len(getattr(sim, "_msl_ports", []) or []),
    }
    if len(planes) != 1:
        return out

    k = planes[0]
    out["sheet_plane_k"] = k
    out["sheet_plane_z_m"] = float(zs[k])

    ex_per_row = mx[:, :, k].sum(axis=0)
    if not ex_per_row.any():
        return out
    rows = np.flatnonzero(ex_per_row == ex_per_row.max())
    j0, j1 = int(rows.min()), int(rows.max())
    out.update(
        trace_rows=(j0, j1),
        n_trace_rows=int(rows.size),
        trace_y_m=(float(ys[j0]), float(ys[j1])),
        trace_width_node_span_m=float(ys[j1] - ys[j0]),
        trace_width_strip_m=float(rows.size) * dx,
    )

    stub_cols = np.flatnonzero(my[:, j1:, k].any(axis=1))
    if stub_cols.size == 0:
        return out
    i0, i1 = int(stub_cols.min()), int(stub_cols.max())
    # ``Ey[i, j, k]`` spans node row j -> j+1 on column i.  Counted from the
    # main line's far row upward, the stub's transverse edges must START at
    # that row (the T is joined — measured: the lattice does realize the edge
    # j1 -> j1+1 when the stub Box begins at the trace's edge) and run without
    # a gap to the open end.  A stub drawn clear of the line realizes its
    # first edge higher up, and a stub that also keeps its open end in place
    # has the same length by the old measure — so the joint is checked, not
    # only the length.
    above = np.flatnonzero(my[i0:i1 + 1, j1:, k].any(axis=0)) + j1
    first_j, last_j = int(above.min()), int(above.max())
    open_j = last_j + 1
    contiguous = (last_j - first_j + 1) == int(above.size)
    out.update(
        stub_cols=(i0, i1),
        n_stub_cols=int(stub_cols.size),
        stub_x_m=(float(xs[i0]), float(xs[i1])),
        stub_width_node_span_m=float(xs[i1] - xs[i0]),
        stub_width_strip_m=float(stub_cols.size) * dx,
        stub_first_edge_row=first_j,
        stub_gap_rows=first_j - j1,
        stub_contiguous=contiguous,
        stub_attached=(first_j == j1) and contiguous,
        stub_open_row=open_j,
        stub_open_y_m=float(ys[open_j]),
        stub_length_m=float(ys[open_j] - ys[j1]),
    )
    return out


def assert_realized(sim: Simulation, dx: float) -> dict:
    """Refuse to solve unless the lattice built the declared board.

    Runs before every rung, with no time step.  Raises with the measured
    numbers; the caller prints the dict either way.
    """
    g = realized_geometry(sim)

    if g["n_sheet_planes"] != 1:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the conductor realizes {g['n_sheet_planes']} "
            f"tangential wall planes along z ({g['sheet_planes']}), not one "
            "sheet plane. The trace and the stub are declared as zero-thickness "
            "SHEETS on the substrate top; two planes is what a one-cell VOLUME "
            "realizes (#931 §1.3).")
    if g["n_volume_cells"] != 0:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_volume_cells']} PEC VOLUME cell(s) were "
            "realized; this board declares sheets only.")
    dz = abs(g["sheet_plane_z_m"] - SUBSTRATE_THICKNESS_M)
    if dz >= 1e-12:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the sheet plane is at z={g['sheet_plane_z_m']*1e6:.3f}µm, "
            f"{dz*1e6:.3f}µm off the declared substrate top "
            f"{SUBSTRATE_THICKNESS_M*1e6:.3f}µm.")
    if g["n_trace_rows"] != g["n_stub_cols"]:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the main line realizes {g['n_trace_rows']} node "
            f"rows and the stub {g['n_stub_cols']} node columns. Both are "
            f"declared {TRACE_WIDTH_M*1e6:.0f}µm wide, so a difference means one "
            "of the two footprints snapped to a different node count.")
    if abs(g["trace_width_strip_m"] - TRACE_WIDTH_M) > dx:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: realized strip width "
            f"{g['trace_width_strip_m']*1e6:.1f}µm is more than one cell "
            f"({dx*1e6:.2f}µm) from the declared {TRACE_WIDTH_M*1e6:.0f}µm "
            f"(node span {g['trace_width_node_span_m']*1e6:.1f}µm over "
            f"{g['n_trace_rows']} rows).")
    if not g["stub_attached"]:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the stub is not joined to the main line: its "
            f"first transverse PEC edge is at node row {g['stub_first_edge_row']}, "
            f"{g['stub_gap_rows']} row(s) above the line's far row "
            f"{g['trace_rows'][1]}"
            + ("" if g["stub_contiguous"] else ", and its edges are not contiguous")
            + ". A floating strip beside a through line is not a shunt stub.")
    if abs(g["stub_length_m"] - STUB_LENGTH_M) > dx:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: realized stub length "
            f"{g['stub_length_m']*1e6:.1f}µm is more than one cell "
            f"({dx*1e6:.2f}µm) from the declared {STUB_LENGTH_M*1e6:.0f}µm "
            f"(open end at y={g['stub_open_y_m']*1e6:.1f}µm, main line far row "
            f"at y={g['trace_y_m'][1]*1e6:.1f}µm).")
    if g["n_ports"] != 2:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_ports']} MSL port(s) registered, expected 2.")
    return g


def _print_realized(g: dict, dx: float) -> None:
    print(f"  realized at dx = {dx*1e6:.3f} µm:")
    print(f"    grid {g['grid_shape']}  cells {g['n_cells']}")
    print(f"    sheet plane k={g.get('sheet_plane_k')} "
          f"z={g.get('sheet_plane_z_m', float('nan'))*1e6:.3f} µm; "
          f"declared sheets {g['n_declared_sheets']}; PEC volume cells "
          f"{g['n_volume_cells']}")
    print(f"    trace rows {g.get('trace_rows')} n={g.get('n_trace_rows')} "
          f"y {g.get('trace_y_m')} node span "
          f"{g.get('trace_width_node_span_m', float('nan'))*1e6:.1f} µm, strip "
          f"{g.get('trace_width_strip_m', float('nan'))*1e6:.1f} µm "
          f"(declared {TRACE_WIDTH_M*1e6:.0f} µm)")
    print(f"    stub cols {g.get('stub_cols')} n={g.get('n_stub_cols')} "
          f"x {g.get('stub_x_m')} node span "
          f"{g.get('stub_width_node_span_m', float('nan'))*1e6:.1f} µm, strip "
          f"{g.get('stub_width_strip_m', float('nan'))*1e6:.1f} µm")
    print(f"    stub open end y={g.get('stub_open_y_m', float('nan'))*1e3:.4f} mm, "
          f"length {g.get('stub_length_m', float('nan'))*1e3:.4f} mm "
          f"(declared {STUB_LENGTH_M*1e3:.1f} mm)")
    print(f"    MSL ports {g['n_ports']}")


# ------------------------------------------------------------------- solve
def run_rung(dx: float, arm_length_m: float = ARM_LENGTH_M) -> dict:
    """Preflight, solve, and hand back the curve with its two witnesses."""
    sim = build(dx, arm_length_m)
    g = assert_realized(sim, dx)
    print(f"  arms {arm_length_m*1e3:.2f} mm each side, box x "
          f"{_x_geometry(arm_length_m)[0]*1e3:.3f} mm")
    _print_realized(g, dx)

    print(f"  preflight at dx = {dx*1e6:.3f} µm:")
    report = sim.preflight()
    print(f"    {len(report)} finding(s)")
    for i, line in enumerate(report):
        print(f"    [{i}] {line}")

    t0 = time.perf_counter()
    res = sim.compute_msl_s_matrix(n_freqs=N_FREQS, num_periods=NUM_PERIODS,
                                   enforce_passivity=False)
    wall_s = time.perf_counter() - t0

    freqs = np.asarray(res.freqs, dtype=float)
    s = np.asarray(res.S)
    settling = (np.asarray(res.settling_db, dtype=float)
                if res.settling_db is not None else None)
    excess = (np.asarray(res.sigma_max_excess, dtype=float)
              if res.sigma_max_excess is not None else None)

    out = dict(
        dx_m=dx, arm_length_m=arm_length_m,
        freqs_hz=freqs, s11=s[0, 0, :], s21=s[1, 0, :],
        z0=np.asarray(res.Z0), settling_db=settling,
        sigma_max_excess=excess, wall_s=wall_s,
        n_cells=g["n_cells"], grid_shape=g["grid_shape"], realized=g,
    )

    print(f"  wall time {wall_s:.1f} s, {g['n_cells']} cells, "
          f"num_periods={NUM_PERIODS}, n_freqs={N_FREQS}")
    print(f"  settling_db per driven run: "
          f"{None if settling is None else np.round(settling, 3).tolist()}")
    worst_excess = None if excess is None else float(np.max(excess))
    print(f"  max sigma excess over the band: {worst_excess}")

    if settling is None:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the result carries no settling witness; the "
            "ring-down cannot be judged and no S value may be quoted.")
    if float(np.max(settling)) > SETTLING_DB:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: worst ring-down settling "
            f"{float(np.max(settling)):.2f} dB is above {SETTLING_DB:.0f} dB "
            f"(per run {np.round(settling, 3).tolist()}). The record ended "
            "before the stub rang down, so this rung's S is truncation-suspect. "
            "Raise NUM_PERIODS; do not compare it.")
    if worst_excess is not None and worst_excess > PASSIVITY_EXCESS_BAR:
        k = int(np.argmax(excess))
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the raw extraction exceeds the passive bound by "
            f"{worst_excess:.4f} at {freqs[k]/1e9:.4f} GHz (bar "
            f"{PASSIVITY_EXCESS_BAR}). A passive board cannot scatter more power "
            "than it receives, so this is a measurement artefact; do not compare "
            "this rung.")
    return out


# --------------------------------------------------------------- estimator
def _load_refined_extremum():
    """The repository's one notch estimator, ``refined_extremum`` in
    ``validation/crossval/comparators/spectral_features.py`` (the module that
    exists so the cases and the referee producers cannot drift apart).  Loaded
    by path because ``validation/`` is not a package; it imports numpy only."""
    import importlib.util

    path = _REPO_ROOT / "validation" / "crossval" / "comparators" / "spectral_features.py"
    spec = importlib.util.spec_from_file_location("_msl_notch_spectral_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.refined_extremum


_refined_extremum = _load_refined_extremum()


def notch_frequency(freqs, s21_mag, lo=None, hi=None) -> dict:
    """Notch frequency, depth and −10 dB bandwidth of a |S21| minimum inside
    ``[lo, hi]`` (same units as ``freqs``; ``None`` means the whole array).

    The frequency is the shared estimator's: the deepest bin in the band and
    the vertex of the parabola through it and its two neighbours, fitted in
    log|S| (the same vertex as a fit in dB).  The Palace record's
    ``parabolic_f_ghz`` was computed from its arrays with that rule by the
    producer the case removal retired, so the second fast test below is a
    regression pin of the shared estimator on a frozen number, not an
    independent check of it.

    The bandwidth is the width between the two linear-interpolated crossings
    of −10 dB either side of the minimum bin; ``nan`` when the curve does not
    come back up to −10 dB inside the array.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s21_mag, dtype=float)
    y = 20.0 * np.log10(np.maximum(s, 1e-300))
    ext = _refined_extremum(f, s, lo, hi, transform="log")
    i = int(ext["index"])
    h = float(ext["bin_width"])
    shift = float(ext["sub_bin_shift"])

    def _cross(lo_idx: int, hi_idx: int, step: int) -> float:
        j = lo_idx
        while 0 <= j + step < f.size and j != hi_idx:
            a, b = y[j], y[j + step]
            if (a + 10.0) * (b + 10.0) <= 0.0 and a != b:
                t = (-10.0 - a) / (b - a)
                return float(f[j] + t * (f[j + step] - f[j]))
            j += step
        return float("nan")

    f_lo = _cross(i, 0, -1)
    f_hi = _cross(i, f.size - 1, +1)
    return {
        "index": i,
        "bin_f": float(ext["bin_f"]),
        "f": float(ext["refined_f"]),
        "sub_bin_shift": shift,
        "depth_db": float(y[i]),
        "bw_10db": float(f_hi - f_lo),
        "f_lo_10db": f_lo,
        "f_hi_10db": f_hi,
        "bin_width": h,
    }


# ------------------------------------------------------------- references
def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _db(mag) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-300))


def _interp_db(freqs_hz, s21_db, ref_f_ghz) -> np.ndarray:
    """rfx's |S21| in dB sampled at the reference frequencies (dB is what the
    2 dB bar is stated in, so the interpolation happens in dB)."""
    return np.interp(np.asarray(ref_f_ghz, dtype=float) * 1e9,
                     np.asarray(freqs_hz, dtype=float),
                     np.asarray(s21_db, dtype=float),
                     left=np.nan, right=np.nan)


def _compare(label: str, rung: dict, ref_f_ghz, ref_s21_mag) -> dict:
    """ΔdB of rfx − reference at every reference frequency inside the band
    both curves cover where BOTH are above the deep-null level, plus the
    notch-frequency distance.

    The deep-null exclusion is two-sided on purpose: a bin where one curve is
    in its null and the other is not is the two notches sitting at different
    frequencies, which the frequency bar already judges; counting it again
    as a magnitude error would score one offset twice."""
    ref_f_hz = np.asarray(ref_f_ghz, dtype=float) * 1e9
    ref_db = _db(ref_s21_mag)
    ours = _interp_db(rung["freqs_hz"], _db(np.abs(rung["s21"])), ref_f_ghz)
    lo, hi = REFERENCE_BAND_HZ
    in_band = (ref_f_hz >= lo) & (ref_f_hz <= hi)
    keep = (in_band & (ref_db >= DEEP_NULL_DB) & (ours >= DEEP_NULL_DB)
            & np.isfinite(ours))
    delta = ours - ref_db
    ref_notch = notch_frequency(ref_f_hz, ref_s21_mag, lo, hi)
    our_notch = notch_frequency(rung["freqs_hz"], np.abs(rung["s21"]), lo, hi)
    max_abs = float(np.max(np.abs(delta[keep]))) if keep.any() else float("nan")
    worst = int(np.argmax(np.where(keep, np.abs(delta), -np.inf))) if keep.any() else -1
    out = dict(
        label=label, ref_f_ghz=np.asarray(ref_f_ghz, float), ref_db=ref_db,
        ours_db=ours, delta_db=delta, compared=keep, in_band=in_band,
        n_in_band=int(in_band.sum()),
        n_compared=int(keep.sum()), max_abs_delta_db=max_abs, worst_index=worst,
        ref_notch_ghz=ref_notch["f"] / 1e9, our_notch_ghz=our_notch["f"] / 1e9,
        ref_notch_depth_db=ref_notch["depth_db"],
        our_notch_depth_db=our_notch["depth_db"],
        notch_pct=100.0 * abs(our_notch["f"] - ref_notch["f"]) / ref_notch["f"],
    )
    return out


_MAX_TABLE_ROWS = 80


def _print_comparison(c: dict) -> None:
    lo, hi = REFERENCE_BAND_HZ
    print(f"  --- {c['label']} ---")
    print(f"    notch: rfx {c['our_notch_ghz']:.4f} GHz ({c['our_notch_depth_db']:.2f} dB) "
          f"vs reference {c['ref_notch_ghz']:.4f} GHz "
          f"({c['ref_notch_depth_db']:.2f} dB) -> {c['notch_pct']:.3f} %")
    print(f"    |S21| dB compared at {c['n_compared']} of the {c['n_in_band']} "
          f"reference frequencies inside {lo/1e9:.0f}–{hi/1e9:.0f} GHz "
          f"({c['ref_f_ghz'].size} in the record; both curves above "
          f"{DEEP_NULL_DB:.0f} dB); max |ΔdB| = {c['max_abs_delta_db']:.3f} dB")
    idx = np.flatnonzero(c["in_band"])
    stride = max(1, int(np.ceil(idx.size / _MAX_TABLE_ROWS)))
    shown = set(idx[::stride].tolist())
    if c["worst_index"] >= 0:
        shown.add(c["worst_index"])
    if stride > 1:
        print(f"    (in-band table printed every {stride} reference bins, plus "
              "the worst compared bin marked *)")
    print("    f_GHz, ref_dB, rfx_dB, delta_dB, compared")
    for i in sorted(shown):
        mark = "*" if i == c["worst_index"] else " "
        print(f"     {mark}{c['ref_f_ghz'][i]:8.4f} {c['ref_db'][i]:10.3f} "
              f"{c['ours_db'][i]:10.3f} {c['delta_db'][i]:9.3f}  "
              f"{bool(c['compared'][i])}")


# ------------------------------------------------------------------ ladder
def _rungs() -> tuple[float, ...]:
    raw = os.environ.get("RFX_MSL_NOTCH_RUNGS", "").strip()
    if not raw:
        return LADDER_M
    return tuple(float(v) for v in raw.split(",") if v.strip())


ARM_WITNESS_BAR_DB = 0.5   # see `_arm_length_witness`; applies to |S21|
# The witness runs on the first two rungs of the ladder and is JUDGED on the
# second (h/4); the coarsest rung is reported as the trend. Measured
# 2026-09-22 at h/2: the realized strip is 508 µm, a 57 Ω line (fitted Z0)
# against the 50 Ω port reference, and that mismatch makes the interference
# between the port reflections and the stub's depend on the arm length —
# |S21| moved 1.2 dB and |S11| up to 9 dB near its minima between 10 mm and
# 15.08 mm arms. At h/4 the strip is 571.5 µm (about 51.5 Ω) and the effect
# shrinks with the mismatch; at the judged rung (h/6, 592.7 µm) it is smaller
# still. |S11| in dB is not judged: near its minima a small mismatch moves it
# by many dB.
ARM_WITNESS_RUNGS = 2


def _arm_length_witness(base: dict, dx: float) -> dict:
    """Build the same filter on longer arms, solve it at the same cell size,
    and report how far its |S21| and |S11| move.

    This is a witness on rfx's own record, not a comparison with a reference.
    The openEMS record is taken on 50 mm arms and rfx runs on 10 mm arms, and
    the magnitudes are only comparable because a lossless line's transmission
    and reflection magnitudes do not depend on the line's length.  Here that
    premise is measured on the structure itself.
    """
    print(f"\n  --- arm-length witness at dx = {dx*1e6:.3f} µm ---")
    print(f"  the same filter on {WITNESS_ARM_LENGTH_M*1e3:.2f} mm arms "
          f"(box x {_x_geometry(WITNESS_ARM_LENGTH_M)[0]*1e3:.2f} mm instead of "
          f"{_x_geometry(ARM_LENGTH_M)[0]*1e3:.2f} mm; nothing else changes)")
    other = run_rung(dx, arm_length_m=WITNESS_ARM_LENGTH_M)

    f = np.asarray(base["freqs_hz"], float)
    assert np.allclose(f, np.asarray(other["freqs_hz"], float)), (
        "the two arm lengths were sampled on different frequency grids; the "
        "difference below would mix a frequency shift into a magnitude one.")
    lo, hi = REFERENCE_BAND_HZ
    in_band = (f >= lo) & (f <= hi)

    out: dict = {"dx_m": dx, "freqs_hz": f,
                 "arm_length_m": WITNESS_ARM_LENGTH_M,
                 "n_cells": other["n_cells"], "wall_s": other["wall_s"]}
    n_witness = notch_frequency(f, np.abs(other["s21"]), lo, hi)
    print(f"  notch on the long arms {n_witness['f']/1e9:.5f} GHz, depth "
          f"{n_witness['depth_db']:.2f} dB (short arms "
          f"{base['notch']['f']/1e9:.5f} GHz, {base['notch']['depth_db']:.2f} dB)")
    out["notch_ghz"] = n_witness["f"] / 1e9
    for label, rec in (("short", base), ("long", other)):
        z0 = np.asarray(rec["z0"], dtype=complex).ravel()
        out[f"z0_median_{label}_ohm"] = float(np.median(z0.real))
    print(f"    fitted line Z0 (median over the band): short arms "
          f"{out['z0_median_short_ohm']:.1f} Ω, long arms "
          f"{out['z0_median_long_ohm']:.1f} Ω, port reference "
          f"{PORT_IMPEDANCE_OHM:.0f} Ω")

    for key in ("s21", "s11"):
        a = _db(np.abs(base[key]))
        b = _db(np.abs(other[key]))
        keep = in_band & (a >= DEEP_NULL_DB) & (b >= DEEP_NULL_DB)
        d = b - a
        worst = int(np.argmax(np.where(keep, np.abs(d), -np.inf)))
        out[f"max_abs_delta_{key}_db"] = float(np.max(np.abs(d[keep])))
        out[f"n_compared_{key}"] = int(keep.sum())
        out[f"worst_f_{key}_ghz"] = float(f[worst] / 1e9)
        print(f"    |{key.upper()}|: max |Δ| = {out[f'max_abs_delta_{key}_db']:.3f} dB "
              f"over {out[f'n_compared_{key}']} of the "
              f"{int(in_band.sum())} rfx bins in {lo/1e9:.0f}–{hi/1e9:.0f} GHz "
              f"where both curves are above {DEEP_NULL_DB:.0f} dB; worst at "
              f"{out[f'worst_f_{key}_ghz']:.4f} GHz")
    print(f"    f_GHz, {WITNESS_ARM_LENGTH_M*1e3:.2f}mm |S21| dB, "
          f"{ARM_LENGTH_M*1e3:.0f}mm |S21| dB, Δ, "
          f"{WITNESS_ARM_LENGTH_M*1e3:.2f}mm |S11| dB, "
          f"{ARM_LENGTH_M*1e3:.0f}mm |S11| dB, Δ")
    a21, b21 = _db(np.abs(base["s21"])), _db(np.abs(other["s21"]))
    a11, b11 = _db(np.abs(base["s11"])), _db(np.abs(other["s11"]))
    for i in np.flatnonzero(in_band):
        print(f"      {f[i]/1e9:8.5f} {b21[i]:10.3f} {a21[i]:10.3f} "
              f"{b21[i]-a21[i]:8.3f} {b11[i]:10.3f} {a11[i]:10.3f} "
              f"{b11[i]-a11[i]:8.3f}")

    return out


def _assert_arm_length_witness(out: dict) -> None:
    """The witness's verdict on |S21|, held back until the block is printed."""
    assert out["max_abs_delta_s21_db"] <= ARM_WITNESS_BAR_DB, (
        f"at dx = {out['dx_m']*1e6:.2f} µm, lengthening each arm from "
        f"{ARM_LENGTH_M*1e3:.0f} mm to {WITNESS_ARM_LENGTH_M*1e3:.2f} mm moved "
        f"|S21| by {out['max_abs_delta_s21_db']:.3f} dB at "
        f"{out['worst_f_s21_ghz']:.4f} GHz (bar {ARM_WITNESS_BAR_DB} dB; fitted "
        f"line Z0 {out['z0_median_short_ohm']:.1f} Ω against the "
        f"{PORT_IMPEDANCE_OHM:.0f} Ω reference). A lossless line's transmission "
        "magnitude does not depend on its length; the residual is the port "
        "mismatch interference, which shrinks as the realized strip approaches "
        "600 µm. A larger move means this board's magnitudes still depend on "
        "its arms and may not be compared with a record taken on other arms.")


@pytest.mark.gpu
@pytest.mark.slow
def test_msl_notch_filter_matches_the_openems_tutorial(tmp_path):
    """The mesh ladder, the convergence statement, then the comparison."""
    rungs = _rungs()
    print(f"\nThe MSL notch filter — ladder {[f'{d*1e6:.2f}µm' for d in rungs]}")
    print(f"  board: arms {ARM_LENGTH_M*1e3:.1f} mm each side of the stub "
          f"centre, port margin {PORT_MARGIN_M*1e3:.1f} mm, box "
          f"{DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × {DOMAIN_Z_M*1e3:.3f} mm, "
          f"trace centre y {TRACE_CENTRE_Y_M*1e3:.3f} mm, stub centre x "
          f"{STUB_CENTRE_X_M*1e3:.3f} mm, lateral clearance "
          f"{LATERAL_CLEARANCE_M*1e3:.3f} mm (the port's minimum 2·h + "
          f"8·{DX_COARSEST_M*1e6:.1f} µm = {LATERAL_CLEARANCE_MIN_M*1e3:.3f} mm, "
          "rounded up to the next 50 µm so the line's edge does not land on a "
          "node line)")

    results = []
    arm_witnesses: list = []
    for dx in rungs:
        print(f"\n=== rung dx = {dx*1e6:.3f} µm ===")
        r = run_rung(dx)
        s21_db = _db(np.abs(r["s21"]))
        n = notch_frequency(r["freqs_hz"], np.abs(r["s21"]), *REFERENCE_BAND_HZ)
        r["notch"] = n
        r["s21_db"] = s21_db
        print(f"  notch {n['f']/1e9:.5f} GHz (bin {n['bin_f']/1e9:.5f}, "
              f"sub-bin shift {n['sub_bin_shift']:+.3f}), depth "
              f"{n['depth_db']:.2f} dB, −10 dB bandwidth "
              f"{n['bw_10db']/1e6:.1f} MHz "
              f"({n['f_lo_10db']/1e9:.4f} – {n['f_hi_10db']/1e9:.4f} GHz)")
        print("  |S21| on the rfx frequency grid — f_GHz, |S21| dB, |S11| dB:")
        s11_db = _db(np.abs(r["s11"]))
        for f, a, b in zip(r["freqs_hz"], s21_db, s11_db):
            print(f"    {f/1e9:8.5f} {a:10.3f} {b:10.3f}")
        results.append(r)
        # The premise the comparison rests on — the arms this board carries do
        # not set its magnitudes — measured on the first two rungs: the
        # coarsest is the trend, the second is judged (see ARM_WITNESS_RUNGS).
        if len(results) <= ARM_WITNESS_RUNGS:
            arm_witnesses.append(_arm_length_witness(r, dx))

    palace = _load(_PALACE_JSON)
    openems = _load(_OPENEMS_JSON)
    judged_stage = openems[OPENEMS_JUDGED_STAGE]
    coarse_stage = openems[OPENEMS_COARSE_STAGE]

    # ---- (b) mesh statement: the notch must settle along the ladder --------
    notches = [r["notch"]["f"] for r in results]
    print("\n  mesh statement — notch frequency along the rfx ladder:")
    for r in results:
        print(f"    dx {r['dx_m']*1e6:8.3f} µm  {r['notch']['f']/1e9:.5f} GHz  "
              f"{r['n_cells']} cells  {r['wall_s']:.1f} s")
    if len(notches) >= 2:
        last_two_pct = 100.0 * abs(notches[-1] - notches[-2]) / notches[-2]
        print(f"    last two rungs differ by {last_two_pct:.3f} % "
              f"(bar {LADDER_AGREEMENT*100:.0f} %)")
    else:
        last_two_pct = float("nan")

    # The reference's own mesh statement, read off its record: the tutorial's
    # own mesh, then the same structure at finer cell sizes.
    ref_coarse_f = coarse_stage["notch"]["refined_f_ghz"]
    ref_fine_f = judged_stage["notch"]["refined_f_ghz"]
    print("\n  the openEMS record's own mesh statement:")
    for name in ("stage_a", "stage_b_mid", "stage_b_fine"):
        st = openems[name]
        ms = openems["meta"]["stages"][name]
        print(f"    {name:14s} {ms['resolution_um']:7.1f} µm  "
              f"{ms['mesh_realized']['substrate_z_cells_realized']} substrate "
              f"cells  {ms['mesh_realized']['n_cells']:9d} cells  "
              f"{st['notch']['refined_f_ghz']:.5f} GHz  "
              f"{st['notch']['depth_db']:7.2f} dB  "
              f"max energy sum {st['max_energy_sum_band']:.4f}")
    print(f"    coarse ({OPENEMS_COARSE_STAGE}) -> fine ({OPENEMS_JUDGED_STAGE}): "
          f"{100.0*(ref_fine_f-ref_coarse_f)/ref_coarse_f:+.3f} %")

    # A partial ladder shows no convergence trend, so it states no verdict:
    # the distances below are still printed — they are the diagnostic the
    # env override exists for — and the test skips at the end instead of
    # passing or failing.
    partial = len(rungs) < len(LADDER_M)
    monotone = (all(b > a for a, b in zip(notches, notches[1:]))
                or all(b < a for a, b in zip(notches, notches[1:])))

    # ---- (c) the comparison: finest rung against the openEMS fine rung -----
    # Every distance and the figure are printed BEFORE any verdict, so a run
    # that fails the mesh statement still leaves the whole curve in the log.
    finest = results[-1]
    print("\n  openEMS tutorial (judged)")
    fine = _compare(
        f"openEMS tutorial, {OPENEMS_JUDGED_STAGE} "
        f"({openems['meta']['stages'][OPENEMS_JUDGED_STAGE]['resolution_um']:.1f} µm)"
        " — JUDGED",
        finest, judged_stage["freqs_ghz"], judged_stage["s21_mag"])
    _print_comparison(fine)
    ref_coarse = _compare(
        f"openEMS tutorial, {OPENEMS_COARSE_STAGE} = stage_a, the tutorial's own "
        f"mesh ({openems['meta']['stages'][OPENEMS_COARSE_STAGE]['resolution_um']:.1f} µm)"
        " — the reference's mesh statement, reported",
        finest, coarse_stage["freqs_ghz"], coarse_stage["s21_mag"])
    _print_comparison(ref_coarse)

    # ---- (d) Palace: reported, never judged -------------------------------
    print("\n  Palace FEM (reported, not judged — it does not conserve power)")
    for mesh, lc in (("mid", 0.085), ("coarse", 0.12)):
        rec = palace[mesh]
        print(f"    Palace {mesh} (lc {lc} mm): max energy sum "
              f"{rec['max_energy_sum']:.6f} over its own {len(rec['freqs_ghz'])} "
              f"points; notch {rec['notch']['parabolic_f_ghz']:.5f} GHz, depth "
              f"{rec['notch']['depth_db']:.2f} dB")
    pal_mid = _compare("Palace FEM, mid mesh (lc 0.085 mm) — REPORTED, NOT JUDGED",
                       finest, palace["mid"]["freqs_ghz"],
                       palace["mid"]["s21_mag"])
    _print_comparison(pal_mid)
    pal_coarse = _compare("Palace FEM, coarse mesh (lc 0.12 mm) — REPORTED, NOT JUDGED",
                          finest, palace["coarse"]["freqs_ghz"],
                          palace["coarse"]["s21_mag"])
    _print_comparison(pal_coarse)

    # ---- (e) the figure ---------------------------------------------------
    fig_path = _write_figure(results, palace, openems, tmp_path)
    print(f"\n  figure: {fig_path}")
    for w in arm_witnesses:
        judged = w is arm_witnesses[-1] and len(arm_witnesses) >= ARM_WITNESS_RUNGS
        print(f"  arm-length witness at {w['dx_m']*1e6:.3f} µm: "
              f"|S21| {w['max_abs_delta_s21_db']:.3f} dB, |S11| "
              f"{w['max_abs_delta_s11_db']:.3f} dB (reported), fitted Z0 "
              f"{w['z0_median_short_ohm']:.1f} Ω — "
              + ("JUDGED against " if judged else "trend, not judged; bar ")
              + f"{ARM_WITNESS_BAR_DB} dB on |S21|")

    # ---- the premise before the comparison ---------------------------------
    # Judged on the second witnessed rung whether or not the rest of the
    # ladder ran: if the magnitudes on this board depend on its arm length,
    # comparing them with a record taken on other arms says nothing. A ladder
    # restricted to one rung leaves the witness reported, not judged.
    if len(arm_witnesses) >= ARM_WITNESS_RUNGS:
        _assert_arm_length_witness(arm_witnesses[-1])

    if partial:
        pytest.skip(
            "RFX_MSL_NOTCH_RUNGS restricted the ladder to "
            f"{[f'{d*1e6:.2f}µm' for d in rungs]}; the notch frequencies are "
            f"{[f'{f/1e9:.5f} GHz' for f in notches]}. A partial ladder shows "
            "no convergence trend, so the distances above are a diagnostic "
            "and this run states no verdict.")

    # ---- (b) the mesh statement comes first: without it nothing is judged --
    assert monotone, (
        "the notch frequency does not move monotonically along the ladder: "
        f"{[f'{f/1e9:.5f} GHz' for f in notches]} at "
        f"{[f'{d*1e6:.2f}µm' for d in rungs]}. Without a monotone trend the "
        "mesh is not shown to converge and the comparison says nothing.")
    assert last_two_pct < LADDER_AGREEMENT * 100.0, (
        "the two finest rungs put the notch "
        f"{last_two_pct:.3f} % apart (bar {LADDER_AGREEMENT*100:.0f} %): "
        f"{[f'{f/1e9:.5f} GHz' for f in notches]}. The mesh has not "
        "converged, so the comparison above states no verdict.")
    assert fine["notch_pct"] < FREQ_BAR * 100.0, (
        f"the notch sits {fine['notch_pct']:.3f} % from the openEMS tutorial "
        f"record's {fine['ref_notch_ghz']:.4f} GHz (bar {FREQ_BAR*100:.0f} %); "
        f"rfx reads {fine['our_notch_ghz']:.4f} GHz on the "
        f"{finest['dx_m']*1e6:.2f} µm mesh.")
    assert fine["max_abs_delta_db"] <= MAG_BAR_DB, (
        f"|S21| differs from the openEMS tutorial record by up to "
        f"{fine['max_abs_delta_db']:.3f} dB (bar {MAG_BAR_DB:.0f} dB) over the "
        f"{fine['n_compared']} reference frequencies where both curves are above "
        f"{DEEP_NULL_DB:.0f} dB.")


def _write_figure(results, palace, openems, tmp_path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.environ.get("RFX_CROSSVAL_FIG_DIR")
    directory = Path(out_dir) if out_dir else Path(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "msl_notch_filter.png"

    stages = openems["meta"]["stages"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for r in results:
        ax.plot(np.asarray(r["freqs_hz"]) / 1e9, r["s21_db"],
                label=f"rfx dx {r['dx_m']*1e6:.2f} µm "
                      f"({r['arm_length_m']*1e3:.0f} mm arms)")
    # Coarse first, JUDGED on top: the two rungs agree to within a line width
    # over most of the band, and the one being judged against must be the one
    # the eye reads there.
    ax.plot(openems[OPENEMS_COARSE_STAGE]["freqs_ghz"],
            _db(openems[OPENEMS_COARSE_STAGE]["s21_mag"]), "-", lw=2.4,
            alpha=0.45, color="tab:green",
            label=f"openEMS tutorial {OPENEMS_COARSE_STAGE} = stage_a "
                  f"({stages[OPENEMS_COARSE_STAGE]['resolution_um']:.0f} µm, "
                  "the tutorial's own mesh)")
    ax.plot(openems[OPENEMS_JUDGED_STAGE]["freqs_ghz"],
            _db(openems[OPENEMS_JUDGED_STAGE]["s21_mag"]), "-", lw=1.0,
            color="k",
            label=f"openEMS tutorial {OPENEMS_JUDGED_STAGE} "
                  f"({stages[OPENEMS_JUDGED_STAGE]['resolution_um']:.0f} µm) — JUDGED")
    ax.plot(palace["mid"]["freqs_ghz"], _db(palace["mid"]["s21_mag"]),
            "--", label="Palace FEM mid (lc 0.085 mm) — reported")
    ax.plot(palace["coarse"]["freqs_ghz"], _db(palace["coarse"]["s21_mag"]),
            "--", label="Palace FEM coarse (lc 0.12 mm) — reported")
    ax.set_xlim(*(f / 1e9 for f in REFERENCE_BAND_HZ))
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# -------------------------------------------------------------- fast tests
def test_the_lattice_builds_the_declared_board():
    """Build-time only: the board realizes as declared, and the three defects
    ``assert_realized`` exists for are refused."""
    print(f"\n  declared board: arms {ARM_LENGTH_M*1e3:.1f} mm each side, port "
          f"margin {PORT_MARGIN_M*1e3:.1f} mm, box {DOMAIN_X_M*1e3:.3f} × "
          f"{DOMAIN_Y_M*1e3:.3f} × {DOMAIN_Z_M*1e3:.3f} mm, trace centre y "
          f"{TRACE_CENTRE_Y_M*1e3:.3f} mm, stub centre x {STUB_CENTRE_X_M*1e3:.3f} mm, "
          f"lateral clearance {LATERAL_CLEARANCE_M*1e3:.3f} mm")
    for dx in (SUBSTRATE_THICKNESS_M / 2, SUBSTRATE_THICKNESS_M / 4):
        g = assert_realized(build(dx), dx)
        _print_realized(g, dx)

    # The arm-length witness builds a longer board; it must realize too, or the
    # witness would fail for the wrong reason.
    dx = SUBSTRATE_THICKNESS_M / 2
    g = assert_realized(build(dx, WITNESS_ARM_LENGTH_M), dx)
    print(f"  witness board, arms {WITNESS_ARM_LENGTH_M*1e3:.2f} mm, box x "
          f"{_x_geometry(WITNESS_ARM_LENGTH_M)[0]*1e3:.2f} mm:")
    _print_realized(g, dx)

    # One cell short is inside the one-cell tolerance BY DESIGN: a 12 mm stub
    # cannot land on a node at every cell size, so the check must not refuse a
    # rounding it forces itself.
    g = assert_realized(_build_short_stub(dx, 1), dx)
    print(f"  stub drawn one cell short: realized length "
          f"{g['stub_length_m']*1e3:.4f} mm — accepted")

    with pytest.raises(AssertionError) as two_cells:
        assert_realized(_build_short_stub(dx, 2), dx)
    print(f"  stub drawn two cells short: {two_cells.value}")
    assert "stub length" in str(two_cells.value)

    with pytest.raises(AssertionError) as volume:
        assert_realized(_build_trace_as_volume(dx), dx)
    print(f"  trace drawn as a one-cell volume: {volume.value}")
    assert "sheet" in str(volume.value)

    # The stub lifted three cells clear of the line AND shortened by the same
    # three cells: the open end does not move, so the length measured from the
    # line's far row is the shipped one and only the joint can tell the two
    # boards apart (a review of this case found the length check alone let
    # this board through).
    with pytest.raises(AssertionError) as floating:
        assert_realized(_build_short_stub(dx, 3, lift_cells=3), dx)
    print(f"  stub lifted three cells, open end unmoved: {floating.value}")
    assert "not joined" in str(floating.value)


def test_the_reference_files_are_what_the_provenance_says():
    """The two frozen references load, carry the bands, point counts and mesh
    rungs PROVENANCE.md states, conserve power on the band they serve, and the
    shared notch estimator reproduces the notch frequency each record froze.
    Those numbers were computed from the records' own arrays with the same
    log-parabolic vertex rule by their producers, so this is a regression pin
    of the estimator on frozen values (to 1 kHz), not an independent check of
    it and not of rfx."""
    palace = _load(_PALACE_JSON)
    openems = _load(_OPENEMS_JSON)

    # --- the openEMS tutorial record ------------------------------------
    assert openems["meta"]["tool"] == "openEMS"
    solved = ("stage_a", "stage_b_mid", "stage_b_fine")
    for name in solved + (OPENEMS_COARSE_STAGE,):
        rec = openems[name]
        assert len(rec["freqs_ghz"]) == 1601
        for key in ("s11_mag", "s11_deg", "s21_mag", "s21_deg", "energy_sum"):
            assert len(rec[key]) == 1601, (name, key)
        assert rec["freqs_ghz"][0] == pytest.approx(0.001)
        assert rec["freqs_ghz"][-1] == pytest.approx(7.0)
        assert tuple(rec["witness_band_ghz"]) == (2.0, 7.0)

    # ``stage_b_coarse`` is not a second solve: the record says it carries
    # stage_a's arrays because the rung IS the tutorial's own mesh.
    assert "reused_from" in openems["meta"]["stages"][OPENEMS_COARSE_STAGE]
    for key in ("freqs_ghz", "s11_mag", "s21_mag"):
        assert openems[OPENEMS_COARSE_STAGE][key] == openems["stage_a"][key]

    # A passive board cannot scatter more power than it receives.  The witness
    # the record defines is over the band it serves: below 2 GHz both port
    # voltages sit at the numerical floor and their ratio is not an
    # S-parameter the structure produced (``meta.passivity_witness``).
    tol = openems["meta"]["passivity_tol"]
    assert tol == 1.05
    for name in solved:
        rec = openems[name]
        band = np.asarray(rec["s11_mag"], float) ** 2 + np.asarray(rec["s21_mag"], float) ** 2
        f = np.asarray(rec["freqs_ghz"], float)
        in_band = (f >= 2.0) & (f <= 7.0)
        print(f"  openEMS {name}: max energy sum over 2–7 GHz "
              f"{rec['max_energy_sum_band']:.6f} (recomputed "
              f"{float(band[in_band].max()):.6f}), over the whole 0.001–7 GHz "
              f"grid {rec['max_energy_sum_full']:.6f}")
        assert rec["max_energy_sum_band"] == pytest.approx(
            float(band[in_band].max()), abs=1e-9)
        assert rec["max_energy_sum_band"] <= tol

    # The Stage A gate is the tutorial reproduction: the notch the maker
    # measured, reproduced here from the record's own arrays.
    for name in solved:
        rec = openems[name]
        got = notch_frequency(np.asarray(rec["freqs_ghz"], float) * 1e9,
                              rec["s21_mag"], *REFERENCE_BAND_HZ)
        print(f"  openEMS {name}: estimator {got['f']/1e9:.6f} GHz, record "
              f"{rec['notch']['refined_f_ghz']:.6f} GHz, depth "
              f"{got['depth_db']:.4f} dB vs {rec['notch']['depth_db']:.4f} dB")
        assert got["f"] == pytest.approx(rec["notch"]["refined_f_ghz"] * 1e9,
                                         abs=1e3)
        assert got["depth_db"] == pytest.approx(rec["notch"]["depth_db"], abs=1e-6)

    gate = openems["meta"]["stage_a_gate"]
    assert gate["passed"] is True
    assert openems["stage_a"]["notch"]["refined_f_ghz"] * 1e9 == pytest.approx(
        gate["measured_f_notch_hz"], abs=1e3)

    # --- the Palace FEM record -------------------------------------------
    assert palace["meta"]["solver"] == "palace"
    assert len(palace["coarse"]["freqs_ghz"]) == 101 == len(palace["coarse"]["s21_mag"])
    assert palace["coarse"]["freqs_ghz"][0] == pytest.approx(2.0)
    assert palace["coarse"]["freqs_ghz"][-1] == pytest.approx(7.0)
    assert len(palace["mid"]["freqs_ghz"]) == 33 == len(palace["mid"]["s21_mag"])
    assert palace["mid"]["freqs_ghz"][0] == pytest.approx(3.2)
    assert palace["mid"]["freqs_ghz"][-1] == pytest.approx(4.0)

    for mesh in ("coarse", "mid"):
        rec = palace[mesh]
        got = notch_frequency(np.asarray(rec["freqs_ghz"], float) * 1e9,
                              rec["s21_mag"])
        print(f"  Palace {mesh}: estimator {got['f']/1e9:.6f} GHz, record "
              f"{rec['notch']['parabolic_f_ghz']:.6f} GHz, depth "
              f"{got['depth_db']:.4f} dB vs {rec['notch']['depth_db']:.4f} dB")
        assert got["f"] / 1e9 == pytest.approx(
            rec["notch"]["parabolic_f_ghz"], abs=1e-6)
        assert got["depth_db"] == pytest.approx(rec["notch"]["depth_db"], abs=1e-6)
