"""The MSL open-stub notch filter — cross-validated against two external solvers.

A 50 Ω microstrip line 600 µm wide on 254 µm of lossless εr = 3.66 carries a
12 mm open-circuit stub of the same width, branching off at the middle of the
5 mm port-to-port line.  The stub is a quarter wavelength near 3.6 GHz, where
its open end transforms into a short across the line, so |S21| has a deep
transmission minimum whose FREQUENCY is set by the stub's electrical length and
whose SHAPE (the −10 dB bandwidth) is set by the stub-to-line impedance ratio.
The T-junction has no closed form, so rfx's |S21| curve is compared with two
frozen external results instead of with an analytic expression.

References (``reference/``, provenance in ``reference/PROVENANCE.md``):

* ``palace_fem.json`` — Palace, frequency-domain FEM on conformal tetrahedra,
  order 2, the same geometry, lossless, first-order absorbing far box.  Two
  meshes: ``coarse`` (lc 0.12 mm, 101 points, 2–7 GHz) and ``mid``
  (lc 0.085 mm, 33 points, 3.2–4.0 GHz).  Arrays are LINEAR magnitude.
  The finest rung is JUDGED against ``mid``.
* ``openems_dx50um.json`` — openEMS, FDTD, dx 50 µm, 50 points 2–7 GHz.
  REPORTED, never judged: no reproduction of an openEMS tutorial result is
  recorded for it.

The ladder is dx = h_sub/n so the substrate top always lands on a node plane
(issue #723): 127 µm (n = 2), 63.5 µm (n = 4), 42.33 µm (n = 6).  The bar is
the v2 accuracy bar — a converged mesh first, then the trend over frequency,
2 dB in magnitude and 1 % in frequency — and nothing else.  No threshold here
is derived from a run, and this case commits no record of a run.

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
# Every length in metres, every frequency in hertz.  These ARE the numbers the
# two references were made with (their `meta` blocks carry the same values).
EPS_R_SUBSTRATE = 3.66
SUBSTRATE_THICKNESS_M = 254e-6
TRACE_WIDTH_M = 600e-6
STUB_LENGTH_M = 12e-3
LINE_LENGTH_M = 5e-3          # port plane to port plane
PORT_MARGIN_M = 1e-3          # feed plane to the x domain face
DOMAIN_X_M = 7.0e-3
DOMAIN_Y_M = 16.232e-3
DOMAIN_Z_M = 1.754e-3
TRACE_CENTRE_Y_M = 1.208e-3
STUB_CENTRE_X_M = 3.5e-3
PORT_IMPEDANCE_OHM = 50.0
FREQ_MAX_HZ = 7e9
CPML_LAYERS = 8

# dx = h_sub / n keeps the substrate top on a node plane (issue #723).
LADDER_M = (
    SUBSTRATE_THICKNESS_M / 2,   # 127 µm
    SUBSTRATE_THICKNESS_M / 4,   # 63.5 µm
    SUBSTRATE_THICKNESS_M / 6,   # 42.33 µm
)

# Record length in periods of freq_max.  Set from the ring-down witness, not
# from a comparison: see SETTLING_DB below.
NUM_PERIODS = 20.0
N_FREQS = 100

# ------------------------------------------------------------- the v2 bar
FREQ_BAR = 0.01          # resonances, cutoffs, notches, band edges: within 1 %
MAG_BAR_DB = 2.0         # power and magnitude: within 2 dB
DEEP_NULL_DB = -20.0     # below this the reference's own null is not compared
LADDER_AGREEMENT = 0.01  # last two rungs must differ by < 1 %
SETTLING_DB = -40.0      # ring-down witness; above it the record was truncated
# Witness, not a comparison: a passive structure cannot scatter more power than
# it receives, so a raw extraction above the passive bound is a measurement
# artefact and the rung stops instead of being compared.
PASSIVITY_EXCESS_BAR = 0.02

_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_PALACE_JSON = _REFERENCE_DIR / "palace_fem.json"
_OPENEMS_JSON = _REFERENCE_DIR / "openems_dx50um.json"


# ------------------------------------------------------------------ build
def build(dx: float) -> Simulation:
    """The reference structure on a uniform ``dx`` mesh.

    Substrate is a lossless dielectric slab filling the board; the trace and
    the stub are zero-thickness PEC SHEETS on the substrate-top node plane
    (both z corners equal — a Box drawn ``h -> h + dx`` would be a one-cell
    VOLUME with a wall on each face, #931).  The ground is the z_lo PEC face.
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
    sim.add(Box((STUB_CENTRE_X_M - TRACE_WIDTH_M / 2.0, y_hi, SUBSTRATE_THICKNESS_M),
                (STUB_CENTRE_X_M + TRACE_WIDTH_M / 2.0, y_hi + STUB_LENGTH_M,
                 SUBSTRATE_THICKNESS_M)), material="pec")

    sim.add_msl_port(position=(PORT_MARGIN_M, TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="+x", impedance=PORT_IMPEDANCE_OHM)
    sim.add_msl_port(position=(PORT_MARGIN_M + LINE_LENGTH_M, TRACE_CENTRE_Y_M, 0.0),
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
    sim.add_msl_port(position=(PORT_MARGIN_M + LINE_LENGTH_M, TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="-x", impedance=PORT_IMPEDANCE_OHM)
    return sim


def _build_short_stub(dx: float, cells_short: int) -> Simulation:
    """The board with the stub drawn ``cells_short`` cells shorter than 12 mm.

    Everything but the stub Box's far face is what :func:`build` declares.
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
    sim.add(Box((STUB_CENTRE_X_M - TRACE_WIDTH_M / 2.0, y_hi, SUBSTRATE_THICKNESS_M),
                (STUB_CENTRE_X_M + TRACE_WIDTH_M / 2.0,
                 y_hi + STUB_LENGTH_M - cells_short * dx,
                 SUBSTRATE_THICKNESS_M)), material="pec")
    sim.add_msl_port(position=(PORT_MARGIN_M, TRACE_CENTRE_Y_M, 0.0),
                     width=TRACE_WIDTH_M, height=SUBSTRATE_THICKNESS_M,
                     direction="+x", impedance=PORT_IMPEDANCE_OHM)
    sim.add_msl_port(position=(PORT_MARGIN_M + LINE_LENGTH_M, TRACE_CENTRE_Y_M, 0.0),
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
    stub_edges = np.flatnonzero(my[i0:i1 + 1, :, k].any(axis=0))
    open_j = int(stub_edges.max()) + 1
    out.update(
        stub_cols=(i0, i1),
        n_stub_cols=int(stub_cols.size),
        stub_x_m=(float(xs[i0]), float(xs[i1])),
        stub_width_node_span_m=float(xs[i1] - xs[i0]),
        stub_width_strip_m=float(stub_cols.size) * dx,
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
def run_rung(dx: float) -> dict:
    """Preflight, solve, and hand back the curve with its two witnesses."""
    sim = build(dx)
    g = assert_realized(sim, dx)
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
        dx_m=dx, freqs_hz=freqs, s11=s[0, 0, :], s21=s[1, 0, :],
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
def notch_frequency(freqs, s21_mag) -> dict:
    """Notch frequency, depth and −10 dB bandwidth of a |S21| minimum.

    The frequency is the vertex of the parabola through the minimum bin and
    its two neighbours, fitted in dB.  This is the estimator the Palace record
    used for its ``parabolic_f_ghz`` (a parabola in dB and one in ln|S| have
    the same vertex, dB being 20/ln10 times ln|S|), and the second fast test
    below pins that it reproduces the recorded value.

    The bandwidth is the width between the two linear-interpolated crossings
    of −10 dB either side of the minimum bin; ``nan`` when the curve does not
    come back up to −10 dB inside the band.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s21_mag, dtype=float)
    y = 20.0 * np.log10(np.maximum(s, 1e-300))
    i = int(np.argmin(s))
    h = float(f[i + 1] - f[i]) if i + 1 < f.size else float(f[i] - f[i - 1])
    if 0 < i < f.size - 1:
        denom = float(y[i - 1] - 2.0 * y[i] + y[i + 1])
        shift = 0.0 if denom <= 0.0 else float(
            np.clip(0.5 * float(y[i - 1] - y[i + 1]) / denom, -1.0, 1.0))
    else:
        shift = 0.0

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
        "bin_f": float(f[i]),
        "f": float(f[i]) + shift * h,
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
    """ΔdB of rfx − reference at every reference frequency the reference does
    not put in its own deep null, plus the notch-frequency distance."""
    ref_db = _db(ref_s21_mag)
    ours = _interp_db(rung["freqs_hz"], _db(np.abs(rung["s21"])), ref_f_ghz)
    keep = (ref_db >= DEEP_NULL_DB) & np.isfinite(ours)
    delta = ours - ref_db
    ref_notch = notch_frequency(np.asarray(ref_f_ghz, float) * 1e9, ref_s21_mag)
    our_notch = notch_frequency(rung["freqs_hz"], np.abs(rung["s21"]))
    max_abs = float(np.max(np.abs(delta[keep]))) if keep.any() else float("nan")
    out = dict(
        label=label, ref_f_ghz=np.asarray(ref_f_ghz, float), ref_db=ref_db,
        ours_db=ours, delta_db=delta, compared=keep,
        n_compared=int(keep.sum()), max_abs_delta_db=max_abs,
        ref_notch_ghz=ref_notch["f"] / 1e9, our_notch_ghz=our_notch["f"] / 1e9,
        notch_pct=100.0 * abs(our_notch["f"] - ref_notch["f"]) / ref_notch["f"],
    )
    return out


def _print_comparison(c: dict) -> None:
    print(f"  --- {c['label']} ---")
    print(f"    notch: rfx {c['our_notch_ghz']:.4f} GHz vs reference "
          f"{c['ref_notch_ghz']:.4f} GHz -> {c['notch_pct']:.3f} %")
    print(f"    |S21| dB compared at {c['n_compared']} of "
          f"{c['ref_f_ghz'].size} reference frequencies "
          f"(reference above {DEEP_NULL_DB:.0f} dB); "
          f"max |ΔdB| = {c['max_abs_delta_db']:.3f} dB")
    print("    f_GHz, ref_dB, rfx_dB, delta_dB, compared")
    for f, r, o, d, k in zip(c["ref_f_ghz"], c["ref_db"], c["ours_db"],
                             c["delta_db"], c["compared"]):
        print(f"      {f:8.4f} {r:10.3f} {o:10.3f} {d:9.3f}  {bool(k)}")


# ------------------------------------------------------------------ ladder
def _rungs() -> tuple[float, ...]:
    raw = os.environ.get("RFX_MSL_NOTCH_RUNGS", "").strip()
    if not raw:
        return LADDER_M
    return tuple(float(v) for v in raw.split(",") if v.strip())


@pytest.mark.gpu
@pytest.mark.slow
def test_msl_notch_filter_matches_the_fem_reference(tmp_path):
    """The mesh ladder, the convergence statement, then the comparison."""
    rungs = _rungs()
    print(f"\nThe MSL notch filter — ladder {[f'{d*1e6:.2f}µm' for d in rungs]}")

    results = []
    for dx in rungs:
        print(f"\n=== rung dx = {dx*1e6:.3f} µm ===")
        r = run_rung(dx)
        s21_db = _db(np.abs(r["s21"]))
        n = notch_frequency(r["freqs_hz"], np.abs(r["s21"]))
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

    palace = _load(_PALACE_JSON)
    openems = _load(_OPENEMS_JSON)

    # ---- (b) mesh statement: the notch must settle along the ladder --------
    notches = [r["notch"]["f"] for r in results]
    print("\n  mesh statement — notch frequency along the ladder:")
    for r in results:
        print(f"    dx {r['dx_m']*1e6:8.3f} µm  {r['notch']['f']/1e9:.5f} GHz  "
              f"{r['n_cells']} cells  {r['wall_s']:.1f} s")
    if len(notches) >= 2:
        last_two_pct = 100.0 * abs(notches[-1] - notches[-2]) / notches[-2]
        print(f"    last two rungs differ by {last_two_pct:.3f} % "
              f"(bar {LADDER_AGREEMENT*100:.0f} %)")
    else:
        last_two_pct = float("nan")

    # A partial ladder shows no convergence trend, so it states no verdict:
    # the distances below are still printed — they are the diagnostic the
    # env override exists for — and the test skips at the end instead of
    # passing or failing.
    partial = len(rungs) < len(LADDER_M)
    monotone = (all(b > a for a, b in zip(notches, notches[1:]))
                or all(b < a for a, b in zip(notches, notches[1:])))

    # ---- (c) the comparison: finest rung against the FEM mid mesh ----------
    # Every distance and the figure are printed BEFORE any verdict, so a run
    # that fails the mesh statement still leaves the whole curve in the log.
    finest = results[-1]
    print("\n  Palace FEM (judged)")
    mid = _compare("Palace FEM, mid mesh (lc 0.085 mm) — JUDGED",
                   finest, palace["mid"]["freqs_ghz"], palace["mid"]["s21_mag"])
    _print_comparison(mid)
    coarse = _compare("Palace FEM, coarse mesh (lc 0.12 mm) — reported",
                      finest, palace["coarse"]["freqs_ghz"],
                      palace["coarse"]["s21_mag"])
    _print_comparison(coarse)

    # ---- (d) openEMS: reported, never judged ------------------------------
    print("\n  openEMS dx 50 µm (reported, not judged)")
    oe = _compare("openEMS, dx 50 µm — REPORTED, NOT JUDGED",
                  finest, openems["freqs_ghz"], openems["s21_mag"])
    _print_comparison(oe)

    # ---- (e) the figure ---------------------------------------------------
    fig_path = _write_figure(results, palace, openems, tmp_path)
    print(f"\n  figure: {fig_path}")

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
    assert mid["notch_pct"] < FREQ_BAR * 100.0, (
        f"the notch sits {mid['notch_pct']:.3f} % from the FEM reference's "
        f"{mid['ref_notch_ghz']:.4f} GHz (bar {FREQ_BAR*100:.0f} %); rfx reads "
        f"{mid['our_notch_ghz']:.4f} GHz on the {finest['dx_m']*1e6:.2f} µm mesh.")
    assert mid["max_abs_delta_db"] <= MAG_BAR_DB, (
        f"|S21| differs from the FEM reference by up to "
        f"{mid['max_abs_delta_db']:.3f} dB (bar {MAG_BAR_DB:.0f} dB) over the "
        f"{mid['n_compared']} reference frequencies above {DEEP_NULL_DB:.0f} dB.")


def _write_figure(results, palace, openems, tmp_path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.environ.get("RFX_CROSSVAL_FIG_DIR")
    directory = Path(out_dir) if out_dir else Path(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "msl_notch_filter.png"

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for r in results:
        ax.plot(np.asarray(r["freqs_hz"]) / 1e9, r["s21_db"],
                label=f"rfx dx {r['dx_m']*1e6:.2f} µm")
    ax.plot(palace["coarse"]["freqs_ghz"], _db(palace["coarse"]["s21_mag"]),
            "--", label="Palace FEM coarse (lc 0.12 mm)")
    ax.plot(palace["mid"]["freqs_ghz"], _db(palace["mid"]["s21_mag"]),
            "--", label="Palace FEM mid (lc 0.085 mm)")
    ax.plot(openems["freqs_ghz"], _db(openems["s21_mag"]),
            ":", label="openEMS dx 50 µm")
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
    for dx in (SUBSTRATE_THICKNESS_M / 2, SUBSTRATE_THICKNESS_M / 4):
        g = assert_realized(build(dx), dx)
        _print_realized(g, dx)

    dx = SUBSTRATE_THICKNESS_M / 2

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


def test_the_reference_files_are_what_the_provenance_says():
    """The two frozen references load, carry the bands and point counts
    PROVENANCE.md states, and the notch estimator reproduces the Palace
    record's own ``parabolic_f_ghz`` (this pins the estimator, not rfx)."""
    palace = _load(_PALACE_JSON)
    openems = _load(_OPENEMS_JSON)

    assert openems["meta"]["solver"] == "openEMS"
    assert len(openems["freqs_ghz"]) == 50 == len(openems["s21_mag"])
    assert openems["freqs_ghz"][0] == pytest.approx(2.0)
    assert openems["freqs_ghz"][-1] == pytest.approx(7.0)

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
            rec["notch"]["parabolic_f_ghz"], abs=1e-3)
        assert got["depth_db"] == pytest.approx(rec["notch"]["depth_db"], abs=1e-6)
