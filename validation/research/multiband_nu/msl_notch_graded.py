"""The MSL open-stub notch filter on a three-axis band-graded mesh.

A 50 ohm microstrip line 600 um wide on 254 um of lossless eps_r 3.66 carries a
12 mm open stub at its middle.  The stub is a quarter wave near 3.67 GHz and
shorts the line there, so |S21| has a deep minimum whose frequency is set by the
stub's electrical length.  On a UNIFORM staircase mesh rfx reads that minimum
about 2 % high at any affordable cell size, because the 600 um metal realizes 4,
9 and 14 node rows down the h/2, h/4, h/6 ladder.  This module solves the SAME
board on a mesh that is fine only across the metal edges and through the
substrate, and coarse everywhere else.

The board, the estimator, the comparison and every threshold come from the case
module ``tests.crossval.msl_notch_filter.test_msl_notch_filter``; nothing is
re-typed here.  What this module adds is the mesh, the refusals that must pass
before a solve, and a recorder.

The declaration this instrument implements is
``docs/design_notes/20260922_msl_notch_graded_mesh_predeclaration.md``.  Its
section 5 windows are frozen; this module computes their arithmetic and records
HELD / FIRED.  It writes no sentence that interprets a measurement.

Mesh
----
Coarse cell ``C_COARSE_M`` = 127 um (the case's h/2) outside the bands.  Fine
bands, uniform inside and containing the whole of the declared margin:

* x: the stub's two x edges +- ``BAND_MARGIN_M``;
* y: the line's two y edges +- ``BAND_MARGIN_M``, and the stub's open end
  +- ``BAND_MARGIN_M``;
* z: 0 -> 2h at ``FZ = h / n_z`` (the substrate plus as much air again).

Every in-plane profile starts and ends with ``RUNWAY_CELLS`` = 9 cells of
exactly ``C_COARSE_M``, so no grading transition stands in the absorber's own
runway.  Ratio cap 1.3 in plane, 1.4 along z.

Two node placements at a metal edge, each an arm:

* ``"on-node"`` -- a node ON each drawn edge, ``F = W / n``;
* ``"offset"`` -- a node ``EDGE_OFFSET`` (0.35) of a fine cell INSIDE the metal
  at each free in-plane edge, the sheet drawn at its true 600 um, so its SOLVED
  edge lands on the drawn edge.  ``F = W / (n + 2*EDGE_OFFSET)``.

Two board numbers differ from the case's, and both are recorded per arm
-----------------------------------------------------------------------
1. ``LATERAL_CLEARANCE_M`` = 17 * C = 2.159 mm, where the case draws the line
   1.55 mm from the y_lo face.  Arithmetic: the declared y band starts 500 um
   below the line's lower edge, at 1.05 mm, and nine cells of 127 um need
   1.143 mm -- the runway does not fit under the band on the case's board, and
   ``make_nonuniform_grid`` additionally requires both ends of a y profile to
   carry the SAME cell size, so a fine y_lo runway is not available either.
   17 * C leaves room for the runway plus the widest transition any rung needs
   (C_off: 9*C + 336 um).  Everything else about the board -- trace width, stub
   length, arm length, port margin, the 4.65 mm above the stub's open end, the
   substrate and the box height -- is the case's, unchanged.
2. ``DOMAIN_Y`` follows: 19.409 mm instead of 18.800 mm.

The z column above 2h holds ``RUNWAY_CELLS`` cells of ONE solved size
``cz <= C``, not of C: 1.246 mm of air cannot hold nine 127 um cells plus a
ratio-1.4 transition down to FZ.  ``cz`` is solved per rung and recorded.  This
is what ``rfx.nonuniform.make_band_profile`` does with a free segment.

Running it
----------
One arm per invocation::

    python validation/research/multiband_nu/msl_notch_graded.py \
        --arm A_off --run-id <vessl run id>

``--dry-run`` builds, runs R1-R3 and preflight, and stops before the solve.
``--n-steps-cap N`` shortens the record to N steps; it is a smoke option and
refuses to write the results file.  ``--merge a.json b.json ...`` folds
per-run files into one, refusing a duplicate arm key.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rfx import Box, Simulation                                    # noqa: E402
from rfx.boundaries.pec import realized_wall_planes                # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec             # noqa: E402
from rfx.mesh_edges import EDGE_OFFSET                             # noqa: E402
from rfx.nonuniform import node_positions_from_profile             # noqa: E402
from rfx.sources.msl_port import (                                 # noqa: E402
    msl_port_from_entry, msl_probe_x_coords_n,
)

from tests._realized_geometry import _node_line, realized          # noqa: E402
from tests.crossval.msl_notch_filter import (                      # noqa: E402
    test_msl_notch_filter as case,
)

__all__ = [
    "ARMS", "RUNGS", "LADDER", "Arm",
    "profiles", "build_graded", "realized_graded",
    "assert_profiles_graded", "assert_preflight_graded",
    "assert_realized_graded", "assert_witnesses",
    "run_arm", "write_result", "merge_files",
    "notch_of", "richardson_limit",
    "w1_mesh_statement", "w2_comparison", "w3_cross_ladder",
    "w4_arm_length_witness", "verdicts",
]

# --------------------------------------------------------------- mesh constants
#: Coarse cell outside every band -- the case's coarsest uniform rung.
C_COARSE_M = case.DX_COARSEST_M
#: In-plane and z adjacent-cell ratio caps (the two validated caps; rfx's
#: preflight reads the same two numbers from its own module constants).
CAP_XY = 1.3
CAP_Z = 1.4
#: Uniform coarse cells against every in-plane absorber face.  Eight is what
#: the absorber's runway check asks for; the ninth keeps the transition's first
#: cell out of it.
RUNWAY_CELLS = 9
#: Declared half-width of the fine band beyond each metal edge.
BAND_MARGIN_M = 500e-6
#: Node tolerance for "this coordinate is a node" (R1) and for every realized
#: length this module judges (R3).
NODE_TOL_M = 1e-9

_H = case.SUBSTRATE_THICKNESS_M
_W = case.TRACE_WIDTH_M

#: Line's lower edge to the y_lo face.  See the module docstring: the case's
#: 1.55 mm cannot hold nine cells of C under the declared band.
LATERAL_CLEARANCE_M = 17.0 * C_COARSE_M
#: Stub's open end to the y_hi face -- the case's own number, unchanged.
TOP_CLEARANCE_M = case.DOMAIN_Y_M - (
    case.TRACE_CENTRE_Y_M + _W / 2.0 + case.STUB_LENGTH_M)

#: Probe placement, set explicitly on both ports because the propagation axis
#: is graded and the auto-solve skips a graded axis.  These are the values the
#: uniform case resolves at C (127 um cells: 1.778 mm offset, 0.762 mm spacing).
N_PROBE_OFFSET = 14
N_PROBE_SPACING = 6
N_PROBES = 5

#: The uniform ladder's extrapolated limit, the W3 comparison target.  A window
#: value frozen by the pre-declaration (its section 1 table, from the ledger
#: entry of 2026-09-22 and PR #1177), not a number this instrument measured.
UNIFORM_LADDER_LIMIT_GHZ = 3.67
#: The uniform h/6 rung's cost, the cost-ratio denominator.  Same source.
UNIFORM_FINEST_CELLS = 13_800_000
UNIFORM_FINEST_WALL_S = 661.0


@dataclass(frozen=True)
class Arm:
    """One solve: a rung (mesh density) and a node placement."""

    rung: str
    placement: str
    arm_length_m: float


#: rung -> (substrate cells n_z, cells across the 600 um metal n).
RUNGS: dict[str, tuple[int, int]] = {"A": (6, 12), "B": (8, 16), "C": (12, 24)}

ARMS: dict[str, Arm] = {
    "A_on": Arm("A", "on-node", case.ARM_LENGTH_M),
    "A_off": Arm("A", "offset", case.ARM_LENGTH_M),
    "A_off_longarms": Arm("A", "offset", case.WITNESS_ARM_LENGTH_M),
    "B_off": Arm("B", "offset", case.ARM_LENGTH_M),
    "C_off": Arm("C", "offset", case.ARM_LENGTH_M),
}
#: The offset ladder, coarse to fine -- W1 and W3 read it in this order.
LADDER = ("A_off", "B_off", "C_off")

RESULTS_DIR = _REPO_ROOT / "validation" / "research" / "multiband_nu" / "results"
DEFAULT_OUT = RESULTS_DIR / "msl_notch_graded.json"


# ------------------------------------------------------------------- profiles
def _geom_sum(fine: float, r: float, k: int) -> float:
    """Sum of ``k`` geometric cells ``fine*r, fine*r^2, ... fine*r^k``."""
    return float(fine * np.sum(r ** np.arange(1, k + 1)))


def _min_ramp_cells(fine: float, coarse: float, cap: float) -> int:
    """Fewest geometric cells that bridge ``fine`` to ``coarse`` under ``cap``."""
    return max(0, int(np.ceil(np.log(coarse / fine) / np.log(cap) - 1e-12)) - 1)


def _solve_run(fine: float, coarse: float, length: float, cap: float,
               *, mirrored: bool = False, k_extra: int = 12,
               p_max: int = 8000) -> np.ndarray:
    """A coarse run that meets a fine band, summing to ``length`` exactly.

    The run is ``p`` cells of exactly ``coarse`` plus a geometric ramp of ``k``
    cells down to (not including) ``fine``; with ``mirrored`` the ramp is
    repeated on the far side and the coarse cells sit between them.  Returned
    coarse-first.  The fewest ramp cells win, then the fewest coarse cells, so
    the answer is a function of the inputs alone.
    """
    ratio = coarse / fine
    k_min = _min_ramp_cells(fine, coarse, cap)
    sides = 2 if mirrored else 1
    for k in range(max(k_min, 1), max(k_min, 1) + k_extra):
        lo_r = (ratio / cap) ** (1.0 / k)
        hi_r = min(cap, ratio ** (1.0 / k))
        if lo_r > hi_r * (1.0 + 1e-12):
            continue
        lo_s = sides * _geom_sum(fine, lo_r, k)
        hi_s = sides * _geom_sum(fine, hi_r, k)
        for p in range(0, p_max):
            rest = length - p * coarse
            if rest < lo_s - 1e-15:
                break
            if rest > hi_s + 1e-15:
                continue
            a, b = lo_r, hi_r
            for _ in range(200):
                mid = 0.5 * (a + b)
                if sides * _geom_sum(fine, mid, k) < rest:
                    a = mid
                else:
                    b = mid
            r = 0.5 * (a + b)
            cells = fine * r ** np.arange(1, k + 1)
            # Close the sum on the ramp's coarsest cell so the profile lands on
            # the declared length to the last bit, not to the bisection's.
            cells[-1] += (rest / sides) - float(cells.sum())
            if mirrored:
                # Fine band on both sides: rise, hold, fall.
                return np.concatenate([cells, np.full(p, coarse), cells[::-1]])
            return np.concatenate([np.full(p, coarse), cells[::-1]])
    raise ValueError(
        f"no ratio<={cap} run from {fine * 1e6:.4f} um to {coarse * 1e6:.4f} um "
        f"fits in {length * 1e6:.4f} um"
        + (" (two ramps)" if mirrored else ""))


def _lo_face_run(length: float, fine: float, cap: float) -> np.ndarray:
    """Cells from a lo absorber face up to (not including) a fine band."""
    return np.concatenate([
        np.full(RUNWAY_CELLS, C_COARSE_M),
        _solve_run(fine, C_COARSE_M, length - RUNWAY_CELLS * C_COARSE_M, cap),
    ])


def _hi_face_run(length: float, fine: float, cap: float) -> np.ndarray:
    """Cells from a fine band up to a hi absorber face."""
    return np.concatenate([
        _solve_run(fine, C_COARSE_M, length - RUNWAY_CELLS * C_COARSE_M,
                   cap)[::-1],
        np.full(RUNWAY_CELLS, C_COARSE_M),
    ])


def _mid_run(length: float, fine: float, cap: float) -> np.ndarray:
    """Coarse run between two fine bands of the same cell size."""
    return _solve_run(fine, C_COARSE_M, length, cap, mirrored=True)


def _z_profile(n_z: int) -> tuple[np.ndarray, float]:
    """``2*n_z`` cells of ``FZ`` over 0 -> 2h, then a ramp and a uniform tail.

    The tail cell ``cz`` is SOLVED, not set to C: the air column above 2h is
    1.246 mm and cannot hold nine 127 um cells plus a ratio-1.4 transition down
    to FZ at any rung.  ``cz <= C`` and the tail is at least ``RUNWAY_CELLS``
    cells long, so the z_hi absorber still stands on a uniform runway.
    """
    fz = _H / n_z
    length = case.DOMAIN_Z_M - 2.0 * _H
    k = max(1, _min_ramp_cells(fz, C_COARSE_M, CAP_Z))

    def total(cz: float, p: int) -> float:
        r = (cz / fz) ** (1.0 / (k + 1))
        return _geom_sum(fz, r, k) + p * cz

    for p in range(RUNWAY_CELLS, RUNWAY_CELLS + 80):
        if total(C_COARSE_M, p) < length:
            continue
        lo, hi = fz, C_COARSE_M
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if total(mid, p) < length:
                lo = mid
            else:
                hi = mid
        cz = 0.5 * (lo + hi)
        r = (cz / fz) ** (1.0 / (k + 1))
        ramp = fz * r ** np.arange(1, k + 1)
        tail = np.full(p, cz)
        ramp[-1] += length - float(ramp.sum()) - float(tail.sum())
        return np.concatenate([np.full(2 * n_z, fz), ramp, tail]), float(cz)
    raise ValueError(f"z tail: no compliant profile for n_z={n_z}")


def fine_cell(rung: str, placement: str) -> float:
    """The in-plane fine cell.  ``n`` cells span the metal on the on-node arm,
    ``n + 2*EDGE_OFFSET`` on the offset arm (the metal then reaches
    ``EDGE_OFFSET`` of a cell past its outermost node at each free edge)."""
    _n_z, n = RUNGS[rung]
    if placement == "on-node":
        return _W / n
    if placement == "offset":
        return _W / (n + 2.0 * EDGE_OFFSET)
    raise ValueError(f"placement must be 'on-node' or 'offset', got {placement!r}")


def board(rung: str, placement: str, arm_length_m: float) -> dict:
    """Every DECLARED coordinate of the board this arm draws.

    Declared, not realized: these are the numbers the note states, computed
    from the case's constants.  :func:`profiles` reads the matching node off
    the profile's own cumulative sum and R1 compares the two.
    """
    f = fine_cell(rung, placement)
    off = 0.0 if placement == "on-node" else EDGE_OFFSET * f
    margin_cells = int(np.ceil(BAND_MARGIN_M / f - 1e-9))
    y_lo = LATERAL_CLEARANCE_M
    y_hi = y_lo + _W
    y_open = y_hi + case.STUB_LENGTH_M
    stub_x = arm_length_m + case.PORT_MARGIN_M
    return dict(
        fine_cell_m=f, substrate_cell_m=_H / RUNGS[rung][0],
        n_across_metal=RUNGS[rung][1], n_substrate_cells=RUNGS[rung][0],
        edge_offset_m=off, margin_cells=margin_cells,
        trace_y_lo_m=y_lo, trace_y_hi_m=y_hi, trace_centre_y_m=y_lo + _W / 2.0,
        stub_open_y_m=y_open, stub_centre_x_m=stub_x,
        arm_length_m=arm_length_m,
        domain_y_m=y_open + TOP_CLEARANCE_M,
        port_x_m=(case.PORT_MARGIN_M, case.PORT_MARGIN_M + 2.0 * arm_length_m),
        # The nodes the metal's free in-plane edges must own, and the substrate
        # top, as the board DECLARES them.
        declared=dict(
            trace_y_lo_node=y_lo + off, trace_y_hi_node=y_hi - off,
            stub_open_node=y_open - off,
            stub_x_lo_node=stub_x - _W / 2.0 + off,
            stub_x_hi_node=stub_x + _W / 2.0 - off,
            trace_y_centre_node=y_lo + _W / 2.0,
            stub_x_centre_node=stub_x,
            substrate_top=_H, z_band_top=2.0 * _H,
        ),
    )


def profiles(rung: str, placement: str,
             arm_length_m: float = case.ARM_LENGTH_M
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The three cell-size profiles of one arm, and the board they are cut to.

    Returns ``(dx_profile, dy_profile, dz_profile, board)``.  A band is a whole
    number of ``F`` cells laid down from the node the metal must own, so every
    declared metal-edge node, the substrate top and every band edge falls on a
    node of the profile's cumulative sum.  The board dict carries both the
    DECLARED coordinate and the node the profile actually produced; R1 compares
    them and :func:`build_graded` draws to the NODE, because a face drawn to a
    coordinate 1e-19 m off its node is on the wrong side of the rasterizer's
    inclusion test (measured on rung B: one extra 31.75 um cell of substrate
    above the trace plane).
    """
    b = board(rung, placement, arm_length_m)
    f, m = b["fine_cell_m"], b["margin_cells"]
    n = b["n_across_metal"]
    n_z = b["n_substrate_cells"]
    d = b["declared"]

    half_band = (n / 2.0 + m) * f
    x_lo_run = _lo_face_run(b["stub_centre_x_m"] - half_band, f, CAP_XY)
    x_band = np.full(n + 2 * m, f)
    x_hi_run = _hi_face_run(b["stub_centre_x_m"] - half_band, f, CAP_XY)
    x = np.concatenate([x_lo_run, x_band, x_hi_run])

    b1_lo = d["trace_y_lo_node"] - m * f
    b1_hi = d["trace_y_hi_node"] + m * f
    b2_lo = d["stub_open_node"] - m * f
    b2_hi = d["stub_open_node"] + m * f
    y_lo_run = _lo_face_run(b1_lo, f, CAP_XY)
    y_band1 = np.full(n + 2 * m, f)
    y_mid = _mid_run(b2_lo - b1_hi, f, CAP_XY)
    y_band2 = np.full(2 * m, f)
    y_hi_run = _hi_face_run(b["domain_y_m"] - b2_hi, f, CAP_XY)
    y = np.concatenate([y_lo_run, y_band1, y_mid, y_band2, y_hi_run])

    z, cz = _z_profile(n_z)

    nx, ny, nz = (node_positions_from_profile(p) for p in (x, y, z))
    ix = int(x_lo_run.size)
    iy1 = int(y_lo_run.size)
    iy2 = iy1 + int(y_band1.size) + int(y_mid.size)
    axis_of = dict(
        stub_x_lo_node=(nx, ix + m),
        stub_x_hi_node=(nx, ix + m + n),
        stub_x_centre_node=(nx, ix + m + n // 2),
        trace_y_lo_node=(ny, iy1 + m),
        trace_y_hi_node=(ny, iy1 + m + n),
        trace_y_centre_node=(ny, iy1 + m + n // 2),
        stub_open_node=(ny, iy2 + m),
        substrate_top=(nz, n_z),
        z_band_top=(nz, 2 * n_z),
    )
    index = {k: int(i) for k, (_a, i) in axis_of.items()}
    node = {k: float(a[i]) for k, (a, i) in axis_of.items()}

    b = dict(
        b, node=node, node_index=index,
        band_x_m=(float(nx[ix]), float(nx[ix + x_band.size])),
        band_y_trace_m=(float(ny[iy1]), float(ny[iy1 + y_band1.size])),
        band_y_stub_end_m=(float(ny[iy2]), float(ny[iy2 + y_band2.size])),
        band_z_m=(0.0, float(nz[2 * n_z])),
        z_tail_cell_m=cz, coarse_cell_m=C_COARSE_M,
        declared_band_x_m=(b["stub_centre_x_m"] - half_band,
                           b["stub_centre_x_m"] + half_band),
        declared_band_y_trace_m=(b1_lo, b1_hi),
        declared_band_y_stub_end_m=(b2_lo, b2_hi),
        segment_cells=dict(x_lo=int(x_lo_run.size), x_band=int(x_band.size),
                           x_hi=int(x_hi_run.size), y_lo=int(y_lo_run.size),
                           y_band1=int(y_band1.size), y_mid=int(y_mid.size),
                           y_band2=int(y_band2.size),
                           y_hi=int(y_hi_run.size), z=int(z.size)),
    )
    return x, y, z, b


# ---------------------------------------------------------------------- build
def build_graded(rung: str, placement: str,
                 arm_length_m: float = case.ARM_LENGTH_M,
                 *, _mutate_draw_offset: float | None = None,
                 _mutate_stub_short_m: float = 0.0
                 ) -> tuple[Simulation, tuple[np.ndarray, ...], dict]:
    """The notch filter on this arm's graded mesh.

    Substrate is a lossless slab filling the board; the line and the stub are
    zero-thickness PEC SHEETS on the substrate-top node plane; the ground is
    the z_lo PEC face -- the case's declaration, on a different mesh.

    Every face that must land on a node is drawn to the NODE the profile
    produced, never to the arithmetic that asked for it.  On rung B the
    substrate top's node sits 5e-20 m below 254 um (float dust in a cumulative
    sum of eight equal cells), and a box drawn to 254 um exactly then swallows
    the 31.75 um cell above the trace plane: the board solved carries 285.75 um
    of dielectric and the trace sheet is buried half a cell inside it.

    The two ``_mutate_*`` arguments exist for the build test's deliberate
    defects and are never used by :func:`run_arm`.
    """
    x, y, z, b = profiles(rung, placement, arm_length_m)
    domain = tuple(float(node_positions_from_profile(p)[-1]) for p in (x, y, z))
    nd = b["node"]

    draw_off = b["edge_offset_m"] if _mutate_draw_offset is None \
        else float(_mutate_draw_offset)
    z_sheet = nd["substrate_top"]
    y_lo_draw = nd["trace_y_lo_node"] - draw_off
    y_hi_draw = nd["trace_y_hi_node"] + draw_off
    x_lo_draw = nd["stub_x_lo_node"] - draw_off
    x_hi_draw = nd["stub_x_hi_node"] + draw_off
    y_open_draw = nd["stub_open_node"] + draw_off - _mutate_stub_short_m

    sim = Simulation(
        freq_max=case.FREQ_MAX_HZ, domain=domain, dx=C_COARSE_M,
        cpml_layers=case.CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        dx_profile=x, dy_profile=y, dz_profile=z)
    sim.add_material("substrate", eps_r=case.EPS_R_SUBSTRATE)
    sim.add(Box((0.0, 0.0, 0.0), (domain[0], domain[1], z_sheet)),
            material="substrate")
    sim.add(Box((0.0, y_lo_draw, z_sheet), (domain[0], y_hi_draw, z_sheet)),
            material="pec")
    sim.add(Box((x_lo_draw, y_hi_draw, z_sheet),
                (x_hi_draw, y_open_draw, z_sheet)), material="pec")
    for x_feed, direction in zip(b["port_x_m"], ("+x", "-x")):
        sim.add_msl_port(position=(x_feed, nd["trace_y_centre_node"], 0.0),
                         width=_W, height=z_sheet, direction=direction,
                         impedance=case.PORT_IMPEDANCE_OHM,
                         n_probe_offset=N_PROBE_OFFSET,
                         n_probe_spacing=N_PROBE_SPACING, n_probes=N_PROBES)
    b = dict(b, domain_m=domain,
             drawn_trace_y_m=(y_lo_draw, y_hi_draw),
             drawn_stub_x_m=(x_lo_draw, x_hi_draw),
             drawn_stub_open_y_m=y_open_draw,
             drawn_sheet_z_m=z_sheet,
             drawn_edge_offset_m=draw_off)
    return sim, (x, y, z), b


# ------------------------------------------------------------------ refusal R1
def profile_summary(p: np.ndarray, cap: float) -> dict:
    r = np.asarray(p[1:], dtype=float) / np.asarray(p[:-1], dtype=float)
    return dict(
        n_cells=int(p.size), min_m=float(p.min()), max_m=float(p.max()),
        worst_ratio=float(max(r.max(), (1.0 / r).max())), cap=float(cap),
        sum_m=float(np.sum(p)),
        first_cell_m=float(p[0]), last_cell_m=float(p[-1]))


def assert_profiles_graded(prof: tuple[np.ndarray, ...], b: dict) -> dict:
    """R1: nodes, ratios and runways -- before anything is built.

    Every intended node (the metal's free-edge nodes, the substrate top, the
    band edges) is a node of the profile's own cumulative sum to
    ``NODE_TOL_M``; every adjacent ratio obeys its axis cap; the first and last
    ``RUNWAY_CELLS`` cells of x and of y are ``C_COARSE_M`` to the bit, and the
    last ``RUNWAY_CELLS`` cells of z are one solved size to the bit.
    """
    x, y, z = prof
    out = {"x": profile_summary(x, CAP_XY), "y": profile_summary(y, CAP_XY),
           "z": profile_summary(z, CAP_Z)}
    for name, p, cap in (("x", x, CAP_XY), ("y", y, CAP_XY), ("z", z, CAP_Z)):
        if out[name]["worst_ratio"] > cap * (1.0 + 1e-9):
            raise AssertionError(
                f"R1: d{name}_profile worst adjacent cell ratio "
                f"{out[name]['worst_ratio']:.6f} exceeds the {cap} cap.")

    for name, p in (("x", x), ("y", y)):
        head, tail = p[:RUNWAY_CELLS], p[-RUNWAY_CELLS:]
        if not (np.all(head == C_COARSE_M) and np.all(tail == C_COARSE_M)):
            raise AssertionError(
                f"R1: the first and last {RUNWAY_CELLS} cells of d{name}_profile "
                f"must equal C = {C_COARSE_M * 1e6:.3f} um bit-exactly; got "
                f"{head[0] * 1e6:.6f} / {tail[-1] * 1e6:.6f} um. A grading "
                "transition standing in the absorber's runway changes the "
                "discrete medium in the boundary-normal direction exactly where "
                "the absorber starts.")
    z_tail = z[-RUNWAY_CELLS:]
    if not np.all(z_tail == z_tail[0]):
        raise AssertionError(
            f"R1: the last {RUNWAY_CELLS} cells of dz_profile are not one size "
            f"({np.unique(z_tail).size} distinct values).")
    if z_tail[0] > C_COARSE_M * (1.0 + 1e-12):
        raise AssertionError(
            f"R1: the z tail cell {float(z_tail[0]) * 1e6:.4f} um is coarser "
            f"than C = {C_COARSE_M * 1e6:.3f} um.")

    misses = {}
    for label, declared in b["declared"].items():
        got = b["node"][label]
        miss = abs(got - declared)
        misses[label] = float(miss)
        if miss > NODE_TOL_M:
            raise AssertionError(
                f"R1: the board declares {label} at {declared * 1e6:.6f} um and "
                f"the profile puts the node at {got * 1e6:.6f} um, "
                f"{miss * 1e9:.4f} nm away (tolerance "
                f"{NODE_TOL_M * 1e9:.1f} nm).")
    for label, declared, got in (
            ("x band lo", b["declared_band_x_m"][0], b["band_x_m"][0]),
            ("x band hi", b["declared_band_x_m"][1], b["band_x_m"][1]),
            ("trace band lo", b["declared_band_y_trace_m"][0],
             b["band_y_trace_m"][0]),
            ("trace band hi", b["declared_band_y_trace_m"][1],
             b["band_y_trace_m"][1]),
            ("stub-end band lo", b["declared_band_y_stub_end_m"][0],
             b["band_y_stub_end_m"][0]),
            ("stub-end band hi", b["declared_band_y_stub_end_m"][1],
             b["band_y_stub_end_m"][1])):
        miss = abs(got - declared)
        misses[label] = float(miss)
        if miss > NODE_TOL_M:
            raise AssertionError(
                f"R1: {label} is declared at {declared * 1e6:.6f} um and the "
                f"profile puts it at {got * 1e6:.6f} um, {miss * 1e9:.4f} nm "
                f"away.")
    out["intended_node_miss_m"] = misses
    return out


# ------------------------------------------------------------------ refusal R2
#: The two preflight codes R2 refuses.  A graded transition inside the
#: absorber's runway, and geometry standing inside the absorber.
R2_FORBIDDEN = (
    # nu_grading_reaches_absorber -- a transition inside the absorber's runway.
    "is not uniform within the",
    # geometry_in_absorber -- a body standing inside the absorbing medium.
    "extends into CPML region",
    # A sheet with the same dielectric on both sides of its plane: the board
    # solved is not the board drawn.  Not in the note's R2 list because the
    # note's feasibility build never produced it; rung B does, when the
    # substrate box is drawn to 254 um instead of to its node.
    "buried half a cell inside the dielectric",
)


def assert_preflight_graded(sim: Simulation) -> list[str]:
    """R2: preflight draws no absorber-runway and no geometry-in-absorber line.

    Every line is returned verbatim for the record, refusal or not.
    """
    report = [str(line) for line in sim.preflight()]
    for line in report:
        for needle in R2_FORBIDDEN:
            if needle in line:
                raise AssertionError(
                    "R2: preflight reports a finding this mesh is declared not "
                    f"to produce:\n  {line}")
    return report


# ------------------------------------------------------------------ refusal R3
def realized_graded(sim: Simulation) -> dict:
    """What the lattice built, read off the graded grid's own node lines.

    The same statement as the case's ``realized_geometry`` -- one sheet plane,
    the line's node rows, the stub's node columns, the joint and the open end --
    except that every length is a difference of NODE COORDINATES.  ``xs[1] -
    xs[0]`` is a cell size only on a uniform mesh; here it is the first cell of
    the x_lo runway and has nothing to do with the metal.
    """
    rz = realized(sim)
    mx, my, _mz = (np.asarray(e) for e in rz.edge_masks)
    xs, ys, zs = (_node_line(rz.grid, a) for a in (0, 1, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eps = np.asarray(
            sim._assemble_materials_nu(rz.grid, pec_sheets=[],
                                       pec_wires=[])[0].eps_r)

    planes = [int(p) for p in realized_wall_planes(rz.edge_masks, 2)]
    n_volume = 0 if rz.pec_mask is None else int(np.asarray(rz.pec_mask).sum())
    out: dict = {
        "sheet_planes": planes, "n_sheet_planes": len(planes),
        "n_declared_sheets": len(rz.sheets), "n_volume_cells": n_volume,
        "grid_shape": tuple(int(v) for v in (xs.size, ys.size, zs.size)),
        "n_cells": int(xs.size * ys.size * zs.size),
        "n_ports": len(getattr(sim, "_msl_ports", []) or []),
        "dt_s": float(rz.grid.dt),
    }
    if len(planes) != 1:
        return out
    k = planes[0]
    out["sheet_plane_k"] = k
    out["sheet_plane_z_m"] = float(zs[k])
    # The two cells the sheet plane separates, at the board's centre column.
    # A dielectric on both sides means the substrate reached past the trace.
    ic, jc = eps.shape[0] // 2, eps.shape[1] // 2
    out["eps_below_sheet"] = float(eps[ic, jc, k - 1])
    out["eps_above_sheet"] = float(eps[ic, jc, k])

    ex_per_row = mx[:, :, k].sum(axis=0)
    if not ex_per_row.any():
        return out
    rows = np.flatnonzero(ex_per_row == ex_per_row.max())
    j0, j1 = int(rows.min()), int(rows.max())
    out.update(trace_rows=(j0, j1), n_trace_rows=int(rows.size),
               trace_y_m=(float(ys[j0]), float(ys[j1])),
               trace_width_node_span_m=float(ys[j1] - ys[j0]))

    stub_cols = np.flatnonzero(my[:, j1:, k].any(axis=1))
    if stub_cols.size == 0:
        return out
    i0, i1 = int(stub_cols.min()), int(stub_cols.max())
    above = np.flatnonzero(my[i0:i1 + 1, j1:, k].any(axis=0)) + j1
    first_j, last_j = int(above.min()), int(above.max())
    open_j = last_j + 1
    out.update(
        stub_cols=(i0, i1), n_stub_cols=int(stub_cols.size),
        stub_x_m=(float(xs[i0]), float(xs[i1])),
        stub_width_node_span_m=float(xs[i1] - xs[i0]),
        stub_first_edge_row=first_j, stub_gap_rows=first_j - j1,
        stub_contiguous=bool((last_j - first_j + 1) == int(above.size)),
        stub_attached=bool(first_j == j1
                           and (last_j - first_j + 1) == int(above.size)),
        stub_open_row=open_j, stub_open_y_m=float(ys[open_j]),
        stub_length_m=float(ys[open_j] - ys[j1]))
    return out


def _check_realized_fields(g: dict, b: dict) -> None:
    """The R3 statements themselves, on an already-measured dict.

    Split out so the build test can hand it a doctored measurement: a check
    that never raises on a wrong number is a check that is not there.
    """
    f = b["fine_cell_m"]
    n = b["n_across_metal"]
    # What the ARM declares, not what the build happened to draw: an arm whose
    # sheets were drawn to their nodes has to be refused, and it would agree
    # with itself if the check read the drawn offset back.
    draw = b["edge_offset_m"]

    if g["n_sheet_planes"] != 1:
        raise AssertionError(
            f"R3: the conductor realizes {g['n_sheet_planes']} tangential wall "
            f"planes along z ({g['sheet_planes']}), not one sheet plane.")
    if g["n_volume_cells"] != 0:
        raise AssertionError(
            f"R3: {g['n_volume_cells']} PEC VOLUME cell(s) realized; this board "
            "declares sheets only.")
    dz = abs(g["sheet_plane_z_m"] - _H)
    if dz > NODE_TOL_M:
        raise AssertionError(
            f"R3: the sheet plane is at z={g['sheet_plane_z_m'] * 1e6:.6f} um, "
            f"{dz * 1e9:.4f} nm off the substrate top "
            f"{_H * 1e6:.3f} um.")
    if g.get("eps_above_sheet") is not None and g["eps_above_sheet"] != 1.0:
        raise AssertionError(
            f"R3: the cells immediately above the sheet plane carry eps_r "
            f"{g['eps_above_sheet']:.4f}, not air. The substrate reaches past "
            "the trace plane, so the board solved is thicker than the board "
            "drawn.")
    if g["n_ports"] != 2:
        raise AssertionError(f"R3: {g['n_ports']} MSL port(s), expected 2.")
    if g["n_trace_rows"] != n + 1 or g["n_stub_cols"] != n + 1:
        raise AssertionError(
            f"R3: the line realizes {g['n_trace_rows']} node rows and the stub "
            f"{g['n_stub_cols']} node columns; this rung lays {n} fine cells "
            f"across the 600 um metal, so both must be {n + 1}.")
    for what, span in (("line", g["trace_width_node_span_m"]),
                       ("stub", g["stub_width_node_span_m"])):
        if abs(span - n * f) > NODE_TOL_M:
            raise AssertionError(
                f"R3: the {what}'s realized node span "
                f"{span * 1e6:.6f} um is not {n} fine cells "
                f"({n * f * 1e6:.6f} um).")
    if not g["stub_attached"]:
        raise AssertionError(
            f"R3: the stub is not joined to the line: its first transverse PEC "
            f"edge is at node row {g['stub_first_edge_row']}, "
            f"{g['stub_gap_rows']} row(s) above the line's far row "
            f"{g['trace_rows'][1]}"
            + ("" if g["stub_contiguous"] else ", and its edges are not "
               "contiguous")
            + ". A floating strip beside a through line is not a shunt stub.")
    if abs(g["stub_length_m"] - case.STUB_LENGTH_M) > NODE_TOL_M:
        raise AssertionError(
            f"R3: realized stub length {g['stub_length_m'] * 1e6:.4f} um is "
            f"{abs(g['stub_length_m'] - case.STUB_LENGTH_M) * 1e6:.4f} um from "
            f"the declared {case.STUB_LENGTH_M * 1e6:.1f} um. On this mesh the "
            "open end sits on a node, so the length is exact or the board is "
            "not the declared one.")

    # The placement itself: the outermost realized node of every free in-plane
    # edge stands `draw` inside the DRAWN edge.  Stated against the drawn
    # coordinates, so an arm whose sheets were drawn to the nodes instead of to
    # their true size is refused here and not somewhere downstream.
    edges = (
        ("line y_lo", g["trace_y_m"][0], b["drawn_trace_y_m"][0], +1.0),
        ("line y_hi", g["trace_y_m"][1], b["drawn_trace_y_m"][1], -1.0),
        ("stub x_lo", g["stub_x_m"][0], b["drawn_stub_x_m"][0], +1.0),
        ("stub x_hi", g["stub_x_m"][1], b["drawn_stub_x_m"][1], -1.0),
        ("stub open end", g["stub_open_y_m"], b["drawn_stub_open_y_m"], -1.0),
    )
    for label, node, drawn, inward in edges:
        got = inward * (node - drawn)
        if abs(got - draw) > NODE_TOL_M:
            raise AssertionError(
                f"R3: the outermost realized node of the {label} edge stands "
                f"{got * 1e6:+.6f} um inside the drawn edge at "
                f"{drawn * 1e6:.4f} um; this arm declares "
                f"{draw * 1e6:+.6f} um ({draw / f if f else 0:.3f} of a "
                f"{f * 1e6:.4f} um fine cell). A PEC sheet is solved about "
                f"{EDGE_OFFSET} of a cell beyond its last node, so where that "
                "node sits IS the arm.")


def assert_realized_graded(sim: Simulation, b: dict) -> dict:
    """R3: refuse to solve unless the lattice built the declared board."""
    g = realized_graded(sim)
    _check_realized_fields(g, b)
    return g


# ------------------------------------------------------------------ refusal R4
def assert_witnesses(settling_db, sigma_max_excess, freqs_hz) -> dict:
    """R4: the case's two witnesses on rfx's own record, after the solve."""
    if settling_db is None:
        raise AssertionError(
            "R4: the result carries no settling witness; the ring-down cannot "
            "be judged and no S value may be quoted.")
    settling = np.asarray(settling_db, dtype=float)
    worst_settling = float(np.max(settling))
    excess = (None if sigma_max_excess is None
              else np.asarray(sigma_max_excess, dtype=float))
    worst_excess = None if excess is None else float(np.max(excess))
    out = dict(settling_db=settling.tolist(), worst_settling_db=worst_settling,
               worst_sigma_excess=worst_excess,
               settling_bar_db=case.SETTLING_DB,
               passivity_excess_bar=case.PASSIVITY_EXCESS_BAR)
    if worst_settling > case.SETTLING_DB:
        raise AssertionError(
            f"R4: worst ring-down settling {worst_settling:.2f} dB is above "
            f"{case.SETTLING_DB:.0f} dB (per driven run "
            f"{np.round(settling, 3).tolist()}). The record ended before the "
            "stub rang down; this arm's S is truncation-suspect.")
    if worst_excess is not None and worst_excess > case.PASSIVITY_EXCESS_BAR:
        k = int(np.argmax(excess))
        f = np.asarray(freqs_hz, dtype=float)
        raise AssertionError(
            f"R4: the raw extraction exceeds the passive bound by "
            f"{worst_excess:.4f} at {f[k] / 1e9:.4f} GHz (bar "
            f"{case.PASSIVITY_EXCESS_BAR}). A passive board cannot scatter more "
            "power than it receives.")
    return out


# ------------------------------------------------------------------- the probes
def probe_planes(sim: Simulation) -> list[list[float]]:
    """Where each port's probe planes land, read from the library's own placer."""
    rz = realized(sim)
    out = []
    for pe in sim._msl_ports:
        mp = msl_port_from_entry(pe)
        xs = msl_probe_x_coords_n(
            rz.grid, mp, n_probes=int(pe.n_probes),
            n_offset_cells=int(pe.n_probe_offset),
            n_spacing_cells=int(pe.n_probe_spacing))
        out.append([float(v) for v in np.asarray(xs).ravel()])
    return out


def assert_probes_in_uniform_region(planes: list[list[float]], b: dict) -> None:
    """Every probe plane sits in the uniform-C part of the propagation axis.

    The extractor reads beta and Z0 from the spacing between these planes; a
    plane inside the graded band would be read with the wrong spacing.
    """
    lo, hi = b["band_x_m"]
    for i, xs in enumerate(planes):
        for x in xs:
            if lo <= x <= hi:
                raise AssertionError(
                    f"probe plane {x * 1e3:.4f} mm of port {i} lies inside the "
                    f"x fine band [{lo * 1e3:.4f}, {hi * 1e3:.4f}] mm.")


# ------------------------------------------------------------------- windows
def notch_of(record: dict) -> dict:
    """The case's own notch estimator on a recorded arm's |S21|."""
    f = np.asarray(record["freqs_hz"], dtype=float)
    s21 = np.asarray(record["s21_re"], dtype=float) + \
        1j * np.asarray(record["s21_im"], dtype=float)
    return case.notch_frequency(f, np.abs(s21), *case.REFERENCE_BAND_HZ)


def _as_rung(record: dict) -> dict:
    """A recorded arm in the shape the case's ``_compare`` reads."""
    return dict(
        freqs_hz=np.asarray(record["freqs_hz"], dtype=float),
        s21=np.asarray(record["s21_re"], float) + 1j * np.asarray(
            record["s21_im"], float),
        s11=np.asarray(record["s11_re"], float) + 1j * np.asarray(
            record["s11_im"], float))


def richardson_limit(h: list[float], f: list[float]) -> dict:
    """Observed order and the limit of ``f = f_inf + A h^p`` on three rungs.

    The order is fitted on all three; the limit is taken from the two finest at
    that order, which is what the pre-declaration asks for.
    """
    h1, h2, h3 = (float(v) for v in h)
    f1, f2, f3 = (float(v) for v in f)
    num, den = f1 - f2, f2 - f3
    if den == 0.0 or num / den <= 0.0:
        return dict(order=float("nan"), limit=float("nan"),
                    reason="the three notches do not fall monotonically")

    def g(p: float) -> float:
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - num / den

    lo, hi = 1e-3, 8.0
    if g(lo) * g(hi) > 0.0:
        return dict(order=float("nan"), limit=float("nan"),
                    reason="no order in (0, 8] reproduces the three notches")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    p = 0.5 * (lo + hi)
    a = (f2 - f3) / (h2 ** p - h3 ** p)
    return dict(order=float(p), limit=float(f3 - a * h3 ** p), reason="")


def w1_mesh_statement(arms: dict) -> dict:
    """W1: the offset ladder's last two rungs within 1 %, three rungs monotone."""
    notches = [notch_of(arms[k])["f"] for k in LADDER]
    last_two_pct = 100.0 * abs(notches[2] - notches[1]) / notches[1]
    monotone = (all(b > a for a, b in zip(notches, notches[1:]))
                or all(b < a for a, b in zip(notches, notches[1:])))
    held = bool(monotone and last_two_pct <= case.LADDER_AGREEMENT * 100.0)
    return dict(window="W1", arms=list(LADDER),
                notches_ghz=[v / 1e9 for v in notches],
                last_two_pct=float(last_two_pct),
                bar_pct=case.LADDER_AGREEMENT * 100.0,
                monotone=bool(monotone),
                verdict="HELD" if held else "FIRED")


def w2_comparison(arms: dict) -> dict:
    """W2: the finest offset rung against the openEMS tutorial's judged stage."""
    openems = case._load(case._OPENEMS_JSON)
    stage = openems[case.OPENEMS_JUDGED_STAGE]
    c = case._compare(f"graded {LADDER[-1]} vs openEMS "
                      f"{case.OPENEMS_JUDGED_STAGE}",
                      _as_rung(arms[LADDER[-1]]),
                      stage["freqs_ghz"], stage["s21_mag"])
    held = bool(c["notch_pct"] <= case.FREQ_BAR * 100.0
                and c["max_abs_delta_db"] <= case.MAG_BAR_DB)
    return dict(window="W2", arm=LADDER[-1],
                our_notch_ghz=float(c["our_notch_ghz"]),
                ref_notch_ghz=float(c["ref_notch_ghz"]),
                notch_pct=float(c["notch_pct"]),
                notch_bar_pct=case.FREQ_BAR * 100.0,
                max_abs_delta_db=float(c["max_abs_delta_db"]),
                mag_bar_db=case.MAG_BAR_DB,
                n_compared=int(c["n_compared"]), n_in_band=int(c["n_in_band"]),
                our_notch_depth_db=float(c["our_notch_depth_db"]),
                ref_notch_depth_db=float(c["ref_notch_depth_db"]),
                verdict="HELD" if held else "FIRED")


def w3_cross_ladder(arms: dict) -> dict:
    """W3: the graded ladder's Richardson limit against the uniform ladder's."""
    h = [arms[k]["substrate_cell_m"] for k in LADDER]
    f = [notch_of(arms[k])["f"] for k in LADDER]
    r = richardson_limit(h, f)
    limit_ghz = r["limit"] / 1e9
    pct = (float("nan") if not np.isfinite(limit_ghz)
           else 100.0 * abs(limit_ghz - UNIFORM_LADDER_LIMIT_GHZ)
           / UNIFORM_LADDER_LIMIT_GHZ)
    held = bool(np.isfinite(pct) and pct <= case.FREQ_BAR * 100.0)
    return dict(window="W3", arms=list(LADDER),
                substrate_cells_m=[float(v) for v in h],
                notches_ghz=[v / 1e9 for v in f],
                fitted_order=r["order"], limit_ghz=limit_ghz,
                uniform_limit_ghz=UNIFORM_LADDER_LIMIT_GHZ,
                distance_pct=pct, bar_pct=case.FREQ_BAR * 100.0,
                note=r["reason"], verdict="HELD" if held else "FIRED")


def w4_arm_length_witness(arms: dict) -> dict:
    """W4: |S21| between 10 mm and 15.08 mm arms at the coarsest offset rung.

    The same arithmetic the case's own arm-length witness runs, on two records
    instead of two solves in one process: the deep-null exclusion is two-sided
    and the band is the reference band.
    """
    short, long = arms["A_off"], arms["A_off_longarms"]
    f = np.asarray(short["freqs_hz"], dtype=float)
    f_long = np.asarray(long["freqs_hz"], dtype=float)
    if not np.allclose(f, f_long):
        raise AssertionError(
            "W4: the two arm lengths were sampled on different frequency "
            "grids; their difference would mix a frequency shift into a "
            "magnitude one.")
    lo, hi = case.REFERENCE_BAND_HZ
    in_band = (f >= lo) & (f <= hi)
    out = dict(window="W4", arms=["A_off", "A_off_longarms"],
               arm_length_m=[short["arm_length_m"], long["arm_length_m"]],
               bar_db=case.ARM_WITNESS_BAR_DB,
               n_in_band=int(in_band.sum()))
    for key in ("s21", "s11"):
        a = case._db(np.abs(np.asarray(short[f"{key}_re"], float)
                            + 1j * np.asarray(short[f"{key}_im"], float)))
        b = case._db(np.abs(np.asarray(long[f"{key}_re"], float)
                            + 1j * np.asarray(long[f"{key}_im"], float)))
        keep = in_band & (a >= case.DEEP_NULL_DB) & (b >= case.DEEP_NULL_DB)
        d = b - a
        worst = int(np.argmax(np.where(keep, np.abs(d), -np.inf)))
        out[f"max_abs_delta_{key}_db"] = float(np.max(np.abs(d[keep])))
        out[f"n_compared_{key}"] = int(keep.sum())
        out[f"worst_f_{key}_ghz"] = float(f[worst] / 1e9)
    out["verdict"] = ("HELD" if out["max_abs_delta_s21_db"]
                      <= case.ARM_WITNESS_BAR_DB else "FIRED")
    return out


def verdicts(arms: dict) -> dict:
    """Every window whose arms are present, with its arithmetic."""
    out: dict = {}
    if all(k in arms for k in LADDER):
        out["W1"] = w1_mesh_statement(arms)
        out["W2"] = w2_comparison(arms)
        out["W3"] = w3_cross_ladder(arms)
    if "A_off" in arms and "A_off_longarms" in arms:
        out["W4"] = w4_arm_length_witness(arms)
    if "A_on" in arms and "A_off" in arms:
        on = notch_of(arms["A_on"])["f"] / 1e9
        off = notch_of(arms["A_off"])["f"] / 1e9
        out["edge_offset_worth"] = dict(
            window=None, a_on_ghz=on, a_off_ghz=off, delta_ghz=on - off,
            delta_pct=100.0 * (on - off) / off)
    out["cost"] = {
        k: dict(n_cells=arms[k]["n_cells"], wall_s=arms[k]["wall_s"],
                cells_ratio=arms[k]["n_cells"] / UNIFORM_FINEST_CELLS,
                wall_ratio=arms[k]["wall_s"] / UNIFORM_FINEST_WALL_S)
        for k in sorted(arms)}
    return out


# ------------------------------------------------------------------- recording
def markdown_tables(arms: dict) -> str:
    """The note's Results section, as tables only.

    Every number is read from the record or computed from it by the functions
    above; none is typed.  The interpreting sentences are not this function's
    to write, so it ends with the line that says whose they are.
    """
    v = verdicts(arms)
    order = [k for k in ("A_on", "A_off", "B_off", "C_off", "A_off_longarms")
             if k in arms]
    out: list[str] = []
    w = out.append

    w("## Results (facts)")
    w("")
    w("Appended after the runs; section 5 above is unchanged.  Every number "
      "below is read from")
    w("`validation/research/multiband_nu/results/msl_notch_graded.json` by "
      "`markdown_tables()` in")
    w("the instrument, and re-derived from the same file by "
      "`tests/unit/nonuniform/test_msl_notch_graded_replay.py`.")
    w("")

    w("### R.1 The mesh each arm solved")
    w("")
    w("| arm | rung | placement | F (um) | FZ (um) | z tail cell (um) | "
      "band margin (fine cells) | interior cells | grid | dt (fs) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        a = arms[k]
        w(f"| {k} | {a['rung']} | {a['placement']} | "
          f"{a['fine_cell_m'] * 1e6:.4f} | {a['substrate_cell_m'] * 1e6:.4f} | "
          f"{a['z_tail_cell_m'] * 1e6:.4f} | {a['margin_cells']} | "
          f"{a['n_interior_cells']:,} | "
          f"{'x'.join(str(n) for n in a['interior_shape'])} | "
          f"{a['dt_s'] * 1e15:.4f} |")
    w("")

    w("### R.2 What the lattice realized")
    w("")
    w("| arm | sheet plane z (um) | PEC volume cells | line node rows | "
      "stub node cols | metal node span (um) | stub length (um) | "
      "node inside the drawn edge (um) |")
    w("|---|---|---|---|---|---|---|---|")
    for k in order:
        a = arms[k]
        g = a["realized"]
        w(f"| {k} | {g['sheet_plane_z_m'] * 1e6:.4f} | {g['n_volume_cells']} | "
          f"{g['n_trace_rows']} | {g['n_stub_cols']} | "
          f"{g['trace_width_node_span_m'] * 1e6:.4f} | "
          f"{g['stub_length_m'] * 1e6:.4f} | "
          f"{a['edge_offset_m'] * 1e6:.4f} |")
    w("")

    w("### R.3 What each arm measured")
    w("")
    w("| arm | notch (GHz) | depth (dB) | -10 dB BW (MHz) | fitted Z0 (ohm) | "
      "worst settling (dB) | worst passivity excess | wall (s) | "
      "cells / uniform h6 | wall / uniform h6 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        a = arms[k]
        n = notch_of(a)
        c = v["cost"][k]
        excess = a["witnesses"]["worst_sigma_excess"]
        w(f"| {k} | {n['f'] / 1e9:.5f} | {n['depth_db']:.2f} | "
          f"{n['bw_10db'] / 1e6:.1f} | {a['z0_median_ohm']:.2f} | "
          f"{a['witnesses']['worst_settling_db']:.2f} | "
          f"{'n/a' if excess is None else f'{excess:.5f}'} | "
          f"{a['wall_s']:.1f} | {c['cells_ratio']:.4f} | "
          f"{c['wall_ratio']:.4f} |")
    w("")
    w(f"Bars, for reading the two witness columns: ring-down "
      f"{case.SETTLING_DB:.0f} dB, passivity excess "
      f"{case.PASSIVITY_EXCESS_BAR}.  The uniform h/6 rung this cost is "
      f"divided by is {UNIFORM_FINEST_CELLS:,} cells and "
      f"{UNIFORM_FINEST_WALL_S:.0f} s (pre-declaration section 1).  Both cell "
      f"counts in that ratio are GRID cells, absorber pad included, which is "
      f"what the uniform ladder's own record counted; table R.1's interior "
      f"count is the smaller number the pre-declaration's section 3 quotes.")
    w("")

    w("### R.4 The frozen windows")
    w("")
    if "W1" in v:
        r = v["W1"]
        w(f"**W1 mesh statement ({' -> '.join(r['arms'])}).**")
        w("")
        w("| notch A_off (GHz) | notch B_off (GHz) | notch C_off (GHz) | "
          "|f_C - f_B| / f_B (%) | bar (%) | monotone | verdict |")
        w("|---|---|---|---|---|---|---|")
        w(f"| {r['notches_ghz'][0]:.5f} | {r['notches_ghz'][1]:.5f} | "
          f"{r['notches_ghz'][2]:.5f} | {r['last_two_pct']:.4f} | "
          f"{r['bar_pct']:.0f} | {r['monotone']} | {r['verdict']} |")
        w("")
    if "W2" in v:
        r = v["W2"]
        w(f"**W2 comparison ({r['arm']} against the openEMS tutorial's "
          f"{case.OPENEMS_JUDGED_STAGE}).**")
        w("")
        w("| rfx notch (GHz) | reference notch (GHz) | distance (%) | "
          "bar (%) | max abs dB | bar (dB) | bins compared | verdict |")
        w("|---|---|---|---|---|---|---|---|")
        w(f"| {r['our_notch_ghz']:.5f} | {r['ref_notch_ghz']:.5f} | "
          f"{r['notch_pct']:.4f} | {r['notch_bar_pct']:.0f} | "
          f"{r['max_abs_delta_db']:.4f} | {r['mag_bar_db']:.0f} | "
          f"{r['n_compared']} of {r['n_in_band']} | {r['verdict']} |")
        w("")
    if "W3" in v:
        r = v["W3"]
        w("**W3 cross-ladder consistency.**")
        w("")
        w("| fitted order in the substrate cell | graded limit (GHz) | "
          "uniform ladder limit (GHz) | distance (%) | bar (%) | verdict |")
        w("|---|---|---|---|---|---|")
        w(f"| {r['fitted_order']:.4f} | {r['limit_ghz']:.5f} | "
          f"{r['uniform_limit_ghz']:.5f} | {r['distance_pct']:.4f} | "
          f"{r['bar_pct']:.0f} | {r['verdict']} |")
        if r["note"]:
            w("")
            w(f"Fit note: {r['note']}.")
        w("")
    if "W4" in v:
        r = v["W4"]
        w(f"**W4 arm-length witness ({r['arm_length_m'][0] * 1e3:.2f} mm "
          f"against {r['arm_length_m'][1] * 1e3:.2f} mm arms).**")
        w("")
        w("| max abs delta |S21| (dB) | bar (dB) | worst at (GHz) | "
          "bins compared | max abs delta |S11| (dB), reported | verdict |")
        w("|---|---|---|---|---|---|")
        w(f"| {r['max_abs_delta_s21_db']:.4f} | {r['bar_db']} | "
          f"{r['worst_f_s21_ghz']:.4f} | "
          f"{r['n_compared_s21']} of {r['n_in_band']} | "
          f"{r['max_abs_delta_s11_db']:.4f} | {r['verdict']} |")
        w("")
    if "edge_offset_worth" in v:
        r = v["edge_offset_worth"]
        w("**Reported, no window: what the 0.35-cell edge offset is worth.**")
        w("")
        w("| A_on notch (GHz) | A_off notch (GHz) | difference (GHz) | "
          "difference (%) |")
        w("|---|---|---|---|")
        w(f"| {r['a_on_ghz']:.5f} | {r['a_off_ghz']:.5f} | "
          f"{r['delta_ghz']:+.5f} | {r['delta_pct']:+.4f} |")
        w("")

    w("### R.5 Provenance")
    w("")
    w("| arm | VESSL run | commit | dirty | jax | backend | started (UTC) |")
    w("|---|---|---|---|---|---|---|")
    for k in order:
        p = arms[k]["provenance"]
        w(f"| {k} | {p['run_id']} | {p['git_sha'][:12]} | {p['git_dirty']} | "
          f"{p['jax_version']} | {p['jax_backend']} | {p['started_utc']} |")
    w("")
    w("Conclusions: leader fills.")
    w("")
    return "\n".join(out)


def _git(args: list[str]) -> str | None:
    """A git answer, or ``None`` when git cannot answer here."""
    try:
        return subprocess.run(["git", "-C", str(_REPO_ROOT)] + args,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def provenance(run_id: str | None, commit: str | None = None) -> dict:
    """Who produced this record, refusing to guess any of it.

    The commit comes from ``--commit`` (or ``RFX_NOTCH_SHA``) when the tree is
    not a git worktree, which is the normal case on the GPU: the job exports
    the pinned tree with ``git archive`` into scratch, so ``git rev-parse``
    inside it has nothing to read.  A record that cannot name its commit is
    refused here, before the solve, rather than written with a placeholder --
    a placeholder looks like provenance and links to nothing.
    """
    import jax
    import rfx

    inside = _git(["rev-parse", "--is-inside-work-tree"]) == "true"
    if commit:
        sha = str(commit).strip()
        source = ("supplied by the submitter; the job verified it with "
                  "git rev-parse --verify in the source repository")
        dirty = (_git(["status", "--porcelain"]) or "") != "" if inside else None
        describe = _git(["describe", "--always", "--dirty"]) if inside else None
    elif inside:
        sha = _git(["rev-parse", "HEAD"]) or ""
        source = "git rev-parse HEAD in this worktree"
        dirty = (_git(["status", "--porcelain"]) or "") != ""
        describe = _git(["describe", "--always", "--dirty"])
    else:
        raise SystemExit(
            f"refusing to record an arm with no commit: {_REPO_ROOT} is not a "
            "git worktree, so the instrument cannot read one. Pass --commit "
            "<sha> (or set RFX_NOTCH_SHA); the GPU job takes it from the "
            "commit.txt it wrote with git rev-parse --verify.")
    if not _SHA_RE.match(sha):
        raise SystemExit(
            f"refusing to record an arm with commit {sha!r}: a commit is forty "
            "hexadecimal characters, and anything else is a placeholder.")
    return dict(
        rfx_file=rfx.__file__,
        git_sha=sha,
        git_sha_source=source,
        git_dirty=dirty,
        git_describe=describe,
        argv=list(sys.argv),
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        jax_version=jax.__version__,
        jax_backend=jax.default_backend(),
        jax_devices=[str(d) for d in jax.devices()],
        python=platform.python_version(),
        hostname=platform.node(),
        run_id=run_id,
    )


def run_arm(arm_key: str, *, run_id: str | None = None,
            commit: str | None = None, dry_run: bool = False,
            n_steps_cap: int | None = None) -> dict:
    """Build, refuse, solve and record one arm.  Everything is printed too."""
    arm = ARMS[arm_key]
    print(f"\n=== arm {arm_key}: rung {arm.rung}, {arm.placement}, arms "
          f"{arm.arm_length_m * 1e3:.2f} mm ===")
    sim, prof, b = build_graded(arm.rung, arm.placement, arm.arm_length_m)
    x, y, z = prof

    print(f"  fine cell F {b['fine_cell_m'] * 1e6:.6f} um "
          f"({b['n_across_metal']} across the 600 um metal), substrate cell "
          f"{b['substrate_cell_m'] * 1e6:.6f} um "
          f"({b['n_substrate_cells']} cells), coarse cell "
          f"{C_COARSE_M * 1e6:.3f} um, z tail cell "
          f"{b['z_tail_cell_m'] * 1e6:.4f} um")
    print(f"  edge offset drawn {b['edge_offset_m'] * 1e6:.6f} um, band margin "
          f"{b['margin_cells']} fine cells "
          f"({b['margin_cells'] * b['fine_cell_m'] * 1e6:.3f} um, declared "
          f"{BAND_MARGIN_M * 1e6:.0f})")
    print(f"  domain {tuple(round(v * 1e3, 9) for v in b['domain_m'])} mm; "
          f"lateral clearance {LATERAL_CLEARANCE_M * 1e3:.4f} mm "
          f"(the case draws {case.LATERAL_CLEARANCE_M * 1e3:.3f} mm)")

    prof_report = assert_profiles_graded(prof, b)
    for axis in ("x", "y", "z"):
        s = prof_report[axis]
        print(f"  d{axis}_profile: {s['n_cells']} cells "
              f"{s['min_m'] * 1e6:.4f}-{s['max_m'] * 1e6:.4f} um, worst ratio "
              f"{s['worst_ratio']:.6f} (cap {s['cap']}), sum "
              f"{s['sum_m'] * 1e3:.9f} mm")
    print("  R1 declared coordinate vs the profile's node (nm apart): "
          + ", ".join(f"{k} {v * 1e9:.4f}"
                      for k, v in prof_report["intended_node_miss_m"].items()))

    g = assert_realized_graded(sim, b)
    print(f"  R3 realized: grid {g['grid_shape']}, {g['n_cells']} cells with "
          f"the absorber pad ({x.size} x {y.size} x {z.size} = "
          f"{x.size * y.size * z.size} interior), dt {g['dt_s']:.6e} s")
    print(f"    sheet plane k={g['sheet_plane_k']} at "
          f"z={g['sheet_plane_z_m'] * 1e6:.4f} um; PEC volume cells "
          f"{g['n_volume_cells']}; declared sheets {g['n_declared_sheets']}")
    print(f"    line rows {g['trace_rows']} n={g['n_trace_rows']}, node span "
          f"{g['trace_width_node_span_m'] * 1e6:.4f} um, y "
          f"{g['trace_y_m'][0] * 1e6:.4f}..{g['trace_y_m'][1] * 1e6:.4f}")
    print(f"    stub cols {g['stub_cols']} n={g['n_stub_cols']}, node span "
          f"{g['stub_width_node_span_m'] * 1e6:.4f} um, x "
          f"{g['stub_x_m'][0] * 1e6:.4f}..{g['stub_x_m'][1] * 1e6:.4f}")
    print(f"    stub attached {g['stub_attached']}, open end "
          f"{g['stub_open_y_m'] * 1e3:.6f} mm, length "
          f"{g['stub_length_m'] * 1e6:.4f} um")

    planes = probe_planes(sim)
    assert_probes_in_uniform_region(planes, b)
    for i, xs in enumerate(planes):
        print(f"    port {i} probe planes (mm): "
              + " ".join(f"{v * 1e3:.4f}" for v in xs))
    print(f"    x fine band {b['band_x_m'][0] * 1e3:.4f}"
          f"..{b['band_x_m'][1] * 1e3:.4f} mm")

    report = assert_preflight_graded(sim)
    print(f"  R2 preflight: {len(report)} finding(s)")
    for i, line in enumerate(report):
        print(f"    [{i}] {line}")

    record: dict = dict(
        arm=arm_key, rung=arm.rung, placement=arm.placement,
        arm_length_m=arm.arm_length_m,
        fine_cell_m=b["fine_cell_m"], substrate_cell_m=b["substrate_cell_m"],
        coarse_cell_m=C_COARSE_M, z_tail_cell_m=b["z_tail_cell_m"],
        n_across_metal=b["n_across_metal"],
        n_substrate_cells=b["n_substrate_cells"],
        margin_cells=b["margin_cells"], edge_offset_m=b["edge_offset_m"],
        domain_m=list(b["domain_m"]),
        lateral_clearance_m=LATERAL_CLEARANCE_M,
        band_x_m=list(b["band_x_m"]), band_y_trace_m=list(b["band_y_trace_m"]),
        band_y_stub_end_m=list(b["band_y_stub_end_m"]),
        band_z_m=list(b["band_z_m"]),
        declared=dict(b["declared"]), node=dict(b["node"]),
        node_index=dict(b["node_index"]),
        segment_cells=dict(b["segment_cells"]),
        drawn_trace_y_m=list(b["drawn_trace_y_m"]),
        drawn_stub_x_m=list(b["drawn_stub_x_m"]),
        drawn_stub_open_y_m=b["drawn_stub_open_y_m"],
        drawn_sheet_z_m=b["drawn_sheet_z_m"],
        profiles={"x": x.tolist(), "y": y.tolist(), "z": z.tolist()},
        profile_summary={a: prof_report[a] for a in ("x", "y", "z")},
        intended_node_miss_m=prof_report["intended_node_miss_m"],
        realized={k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in g.items()},
        preflight=report,
        probe_planes_m=planes,
        n_cells=g["n_cells"], grid_shape=list(g["grid_shape"]),
        # The padded grid the solver steps, and the interior the board owns
        # (the note's section 3 counts the interior).
        n_interior_cells=int(x.size * y.size * z.size),
        interior_shape=[int(x.size), int(y.size), int(z.size)],
        dt_s=g["dt_s"],
        n_freqs=case.N_FREQS, num_periods=case.NUM_PERIODS,
        provenance=provenance(run_id, commit),
    )

    if dry_run:
        print("  --dry-run: build, R1, R2, R3 and the probe check only; "
              "no time step, nothing written.")
        record["dry_run"] = True
        return record

    num_periods = case.NUM_PERIODS
    if n_steps_cap is not None:
        num_periods = float(n_steps_cap) * g["dt_s"] * case.FREQ_MAX_HZ
        print(f"  SMOKE: --n-steps-cap {n_steps_cap} shortens the record to "
              f"num_periods={num_periods:.6g} (the case runs "
              f"{case.NUM_PERIODS}). Not a recordable arm.")
    record["num_periods"] = num_periods
    record["n_time_steps_derived"] = int(
        np.ceil(num_periods / case.FREQ_MAX_HZ / g["dt_s"]))
    record["smoke"] = n_steps_cap is not None

    t0 = time.perf_counter()
    res = sim.compute_msl_s_matrix(n_freqs=case.N_FREQS,
                                   num_periods=num_periods,
                                   enforce_passivity=False)
    wall_s = time.perf_counter() - t0

    freqs = np.asarray(res.freqs, dtype=float)
    s = np.asarray(res.S)
    z0 = np.asarray(res.Z0, dtype=complex).ravel()
    record.update(
        wall_s=float(wall_s), freqs_hz=freqs.tolist(),
        s11_re=np.real(s[0, 0, :]).tolist(), s11_im=np.imag(s[0, 0, :]).tolist(),
        s21_re=np.real(s[1, 0, :]).tolist(), s21_im=np.imag(s[1, 0, :]).tolist(),
        s12_re=np.real(s[0, 1, :]).tolist(), s12_im=np.imag(s[0, 1, :]).tolist(),
        s22_re=np.real(s[1, 1, :]).tolist(), s22_im=np.imag(s[1, 1, :]).tolist(),
        z0_re=np.real(z0).tolist(), z0_im=np.imag(z0).tolist(),
        z0_median_ohm=float(np.median(np.real(z0))),
        sigma_max_excess=(None if res.sigma_max_excess is None
                          else np.asarray(res.sigma_max_excess,
                                          dtype=float).tolist()),
    )
    print(f"  wall time {wall_s:.1f} s, {g['n_cells']} cells, "
          f"{record['n_time_steps_derived']} time steps (derived from dt), "
          f"n_freqs={case.N_FREQS}, num_periods={num_periods}")

    witness = assert_witnesses(res.settling_db, res.sigma_max_excess, freqs)
    record["witnesses"] = witness
    print(f"  R4 settling per driven run {witness['settling_db']} dB "
          f"(bar {case.SETTLING_DB:.0f}); worst passivity excess "
          f"{witness['worst_sigma_excess']} (bar {case.PASSIVITY_EXCESS_BAR})")
    print(f"  fitted line Z0 median {record['z0_median_ohm']:.3f} ohm "
          f"(port reference {case.PORT_IMPEDANCE_OHM:.0f})")

    n = notch_of(record)
    record["notch"] = {k: float(v) for k, v in n.items()}
    print(f"  notch {n['f'] / 1e9:.6f} GHz (bin {n['bin_f'] / 1e9:.6f}, sub-bin "
          f"shift {n['sub_bin_shift']:+.4f}), depth {n['depth_db']:.3f} dB, "
          f"-10 dB bandwidth {n['bw_10db'] / 1e6:.2f} MHz")
    print("  |S21| and |S11| on the rfx frequency grid -- f_GHz, S21 dB, S11 dB:")
    s21_db = case._db(np.abs(s[1, 0, :]))
    s11_db = case._db(np.abs(s[0, 0, :]))
    for fv, a, c in zip(freqs, s21_db, s11_db):
        print(f"    {fv / 1e9:9.6f} {a:10.4f} {c:10.4f}")
    return record


def figure(record: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    openems = case._load(case._OPENEMS_JSON)
    stage = openems[case.OPENEMS_JUDGED_STAGE]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"msl_notch_graded_{record['arm']}.png"
    f = np.asarray(record["freqs_hz"], dtype=float) / 1e9
    s21 = np.asarray(record["s21_re"], float) + 1j * np.asarray(
        record["s21_im"], float)
    s11 = np.asarray(record["s11_re"], float) + 1j * np.asarray(
        record["s11_im"], float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(f, case._db(np.abs(s21)), label=(
        f"rfx graded {record['arm']} -- F {record['fine_cell_m'] * 1e6:.2f} um, "
        f"FZ {record['substrate_cell_m'] * 1e6:.2f} um, "
        f"{record['n_cells'] / 1e6:.2f} M cells"))
    ax.plot(f, case._db(np.abs(s11)), lw=0.8, alpha=0.6, label="rfx |S11|")
    ax.plot(stage["freqs_ghz"], case._db(stage["s21_mag"]), "k-", lw=1.0,
            label=f"openEMS tutorial {case.OPENEMS_JUDGED_STAGE} -- JUDGED")
    ax.set_xlim(*(v / 1e9 for v in case.REFERENCE_BAND_HZ))
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("dB")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _empty_file() -> dict:
    return {
        "schema": "msl_notch_graded/1",
        "note": ("docs/design_notes/"
                 "20260922_msl_notch_graded_mesh_predeclaration.md"),
        "windows": {
            "W1_ladder_agreement_pct": case.LADDER_AGREEMENT * 100.0,
            "W2_freq_bar_pct": case.FREQ_BAR * 100.0,
            "W2_mag_bar_db": case.MAG_BAR_DB,
            "W2_reference": f"{case.OPENEMS_JUDGED_STAGE} of "
                            "reference/openems_tutorial.json",
            "W3_uniform_limit_ghz": UNIFORM_LADDER_LIMIT_GHZ,
            "W4_arm_witness_bar_db": case.ARM_WITNESS_BAR_DB,
            "deep_null_db": case.DEEP_NULL_DB,
            "settling_db": case.SETTLING_DB,
            "passivity_excess_bar": case.PASSIVITY_EXCESS_BAR,
            "reference_band_hz": list(case.REFERENCE_BAND_HZ),
        },
        "mesh": {
            "coarse_cell_m": C_COARSE_M, "cap_xy": CAP_XY, "cap_z": CAP_Z,
            "runway_cells": RUNWAY_CELLS, "band_margin_m": BAND_MARGIN_M,
            "edge_offset_cells": EDGE_OFFSET,
            "lateral_clearance_m": LATERAL_CLEARANCE_M,
            "case_lateral_clearance_m": case.LATERAL_CLEARANCE_M,
            "n_probe_offset": N_PROBE_OFFSET,
            "n_probe_spacing": N_PROBE_SPACING, "n_probes": N_PROBES,
        },
        "arms": {},
    }


def write_result(record: dict, out: Path) -> Path:
    """Fold one arm into the results file, refusing an arm that is already in it."""
    if record.get("smoke"):
        raise SystemExit(
            "refusing to write a --n-steps-cap record: the cap is a smoke "
            "option and its S is not the declared record length.")
    data = _empty_file()
    if out.exists():
        with out.open() as fh:
            data = json.load(fh)
    if record["arm"] in data.get("arms", {}):
        raise SystemExit(
            f"refusing to overwrite arm {record['arm']!r}, already present in "
            f"{out}. One attempt per arm (R2 of the note): record the failure, "
            "do not re-run with a changed mesh.")
    data.setdefault("arms", {})[record["arm"]] = record
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return out


def merge_files(paths: list[Path], out: Path) -> Path:
    data = _empty_file()
    if out.exists():
        with out.open() as fh:
            data = json.load(fh)
    for p in paths:
        with Path(p).open() as fh:
            other = json.load(fh)
        for key, rec in other.get("arms", {}).items():
            if key in data["arms"]:
                raise SystemExit(f"refusing to overwrite arm {key!r} from {p}.")
            data["arms"][key] = rec
            print(f"  merged {key} from {p}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"  wrote {out} with arms {sorted(data['arms'])}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--run-id", default=os.environ.get("RFX_RUN_ID"))
    ap.add_argument("--commit", default=os.environ.get("RFX_NOTCH_SHA"),
                    help="the commit this tree was exported from; required "
                         "when the tree is not a git worktree")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="build, R1-R3 and preflight only; no time step")
    ap.add_argument("--n-steps-cap", type=int, default=None,
                    help="smoke only: shorten the record to this many steps; "
                         "the result may not be written")
    ap.add_argument("--merge", nargs="+", type=Path, default=None,
                    help="fold these per-run files into --out")
    ap.add_argument("--verdicts", action="store_true",
                    help="print every window's arithmetic from --out")
    ap.add_argument("--tables", action="store_true",
                    help="print the note's Results section from --out")
    args = ap.parse_args(argv)

    if args.merge:
        merge_files(args.merge, args.out)
        return 0
    if args.verdicts or args.tables:
        with args.out.open() as fh:
            data = json.load(fh)
        if args.tables:
            print(markdown_tables(data["arms"]))
        else:
            print(json.dumps(verdicts(data["arms"]), indent=1, sort_keys=True))
        return 0
    if not args.arm:
        ap.error("--arm is required (or --merge / --verdicts)")

    if not args.dry_run and args.n_steps_cap is None and args.out.exists():
        with args.out.open() as fh:
            if args.arm in json.load(fh).get("arms", {}):
                raise SystemExit(
                    f"arm {args.arm!r} is already in {args.out}; one attempt "
                    "per arm.")

    record = run_arm(args.arm, run_id=args.run_id, commit=args.commit,
                     dry_run=args.dry_run, n_steps_cap=args.n_steps_cap)
    if args.dry_run:
        return 0
    if args.n_steps_cap is not None:
        print("  smoke run complete; nothing written.")
        return 0
    path = write_result(record, args.out)
    print(f"  wrote {path}")
    print(f"  figure: {figure(record, args.fig_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
