"""The RT/Duroid 5880 probe-fed patch antenna — cross-validated against openEMS.

A 39.7 × 49.25 mm rectangular patch sits on 3.175 mm of RT/Duroid 5880 (εr 2.2,
tanδ 0.001) over a 55.6 × 65.1 mm ground, fed by a 50 Ω probe from the ground
to the patch 8.73125 mm off centre along the resonant length.  The patch and
the ground form an open cavity whose TM010 mode resonates in the record's
window (``RESONANCE_BAND_HZ``): the cavity's length plus the fringing field at
its two radiating edges sets the frequency, and radiation through those edges
loads it, so the probe sees a matched |S11| dip there.  The fringing extension has no closed form accurate
to 1 %, so rfx's |S11| curve is compared with frozen openEMS results instead of
with an analytic expression.

The boards
----------
The metal, the substrate and the probe are the same in both solvers; the box
around them is not.

* the openEMS record — the ground centred at the origin, absorbing PML_8 on all
  six faces with their inner faces at x ±88, y ±93 and z −40 / +90 mm
  (``meta.stages.*.geometry_realized.pml``);
* rfx here — the patch centred at (``PATCH_CENTRE_X_M``, ``PATCH_CENTRE_Y_M``),
  the declared domain 24·h × 28·h in plane and centred on it (``X_CLEARANCE_M``
  and ``Y_CLEARANCE_M`` clear of the ground's edges), 9.525 mm below the ground
  and 15.875 mm above the patch, with a 19.05 mm deep CPML OUTSIDE that domain on
  every face (rfx grid convention): 24, 48 and 72 cells on the three rungs,
  the same absorber in metres.

rfx does not run its ladder in the record's 176 × 186 × 130 mm box: at h/12
it would hold about 230 M cells before the absorber.  What the box does to
the resonance was measured on the retired 40 × 50 mm board at h/4 (the
comment above ``AIR_H``): it moved with the absorber's depth and with the air
added, and had not levelled.  The reference-box run solves the coarsest rung
once more with the declared domain's faces on the record's own
absorber inner faces (x ±88, y ±93, z −40 / +90 mm about the ground's centre,
read from the record and snapped to the nearest h/4 node) and the ladder's
absorber depth outside them, and reports how far the resonance and |S11| move
against the ladder box; nothing is judged on it.

Reference (``reference/``, provenance in ``reference/PROVENANCE.md``):

* ``openems_patch.json`` — openEMS 0.37.0, FDTD, VESSL run 369367264675 (the board on the rfx lattice).
  openEMS's own ``Simple_Patch_Antenna`` tutorial is reproduced first as the
  reproduce gate (``stage_a``; a DIFFERENT patch on a different substrate,
  never compared with this board): 2.43303 GHz against the 2.430 GHz its
  documentation plots.  Then this board on three meshes, the substrate carrying
  4, 6 and 8 cells: ``stage_b_coarse``, ``stage_b_mid``, ``stage_b_fine``.
  901 points, 1.6–3.4 GHz, complex S11 and the port's own Zin on every bin.
  **JUDGED against ``stage_b_fine``**; the coarse and mid rungs are the
  record's own mesh statement.  The mesh refinement near the patch edges is not
  a uniform √2 ladder (the record's metal-edge cells go 0.60 → 0.42 → 0.30 mm
  but the cells either side of them do not scale alike), so the record's own
  steps are printed and never extrapolated.

The ladder
----------
dx = h_sub/n puts both substrate faces — the ground sheet and the patch sheet —
on node planes on every rung (issue #723).  In plane, a PEC sheet realizes the
nodes inside its drawn footprint and the substrate, a volume, whole cells, so
the two realize different edges, and an edge that falls between nodes
realizes a different length on each h/n.  The board therefore puts every edge
just outside a node common to h/4, h/8 and h/12 (PI 2026-09-25; the patch
centre is a node): the patch covers node columns 39.6875 mm and rows
49.2125 mm apart and the ground 55.5625 × 65.0875 mm on all three rungs, and
the substrate one cell more than the ground's node span along x and along y.
``assert_realized`` holds each rung to exactly that, and the probe to its
node, and refuses a rung that realizes anything else, the way the Sheen
low-pass filter's test refuses a rung whose two feeds realize different
widths.  The rungs that pass are h/4, h/8 and h/12 (793.75, 396.875 and
264.583 µm).

The thresholds are the v2 accuracy bar (1 % in frequency, 2 dB in magnitude)
and the three rules the PI approved with the MSL notch filter's plan on
2026-09-22: the mesh statement (the last two rungs within 1 %), the deep-null
exclusion (a bin where either curve is below −20 dB is judged by position only)
and the ring-down witness (−40 dB, the repo's rule); and what the PI
approved on 2026-09-24: a ladder step within 0.1 % is flat, not a direction,
and this case is judged by decision (가).  The shared constants are in
``tests/crossval/_v2_judging.py``, the rest below, each with its source.  No
threshold is derived from a run, and this case commits no record of a run.

What is judged (decision (가), PI 2026-09-24): the TM010 resonance frequency
f0 within 1 % (``FREQ_BAR``), and the input RESISTANCE there,
|R_rfx − R_ref| / R_ref within 2 % (``RESISTANCE_BAR``).  f0 is where Re(Zin)
peaks (the lane leader's decision of 2026-09-24, after #715's request for a
port-independent resonant frequency): near the mode the port sees
Zin ≈ jX_probe + R / (1 + jQ(f/f0 − f0/f)), so the Re(Zin) maximum sits at f0
at height R whatever the probe's series reactance, while the |S11| minimum
moves with it.  Both curves go through one estimator, ``refined_remax`` in
``tests/crossval/_v2_judging.py``: the largest Re(Zin) bin in the record's
window, the vertex of the parabola through it and its neighbours, R the
parabola's value there, X Im(Zin) interpolated at the vertex; the mesh
statement runs on the same f0 along the ladder.  REPORTED, not judged: the
|S11| minimum's frequency and depth, X(f0), the −10 dB band edges, and |S11|
aligned and as solved.  The reason, which the test prints: neither solver
declares a probe radius, so each solver's feed cell sets the probe's series
reactance; the input reactance at resonance differs, and it sets the dip's
depth and width.

How to run it
-------------
The ladder is marked ``gpu`` and ``slow``, so the PR lane never collects it::

    pytest tests/crossval/rt5880_patch -m gpu

Set ``RFX_CROSSVAL_FIG_DIR`` to keep the |S11| figure somewhere durable;
without it the figure goes under pytest's ``tmp_path``.  Set
``RFX_RT5880_PATCH_RUNGS`` to a comma-separated list of cell sizes in metres
(e.g. ``RFX_RT5880_PATCH_RUNGS=793.75e-6``) to run part of the ladder — the
mesh statement and the comparison then apply to the rungs given, which is a
diagnostic, not the case's verdict.  Unset, the full ladder runs.

The fast tests carry no mark: they build the structure (in the ladder's box
at h/4 and h/8, and every rung from h/4 to h/12 in a build-only box just
around the board; the ladder's own h/12 board is checked by
``assert_realized`` inside the ladder, before it is solved) and read the
reference file, and never step the FDTD.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from rfx import Box, GaussianPulse, PolylineWire, Simulation
from rfx.boundaries.spec import BoundarySpec

from tests._realized_geometry import _node_line, realized
from tests.crossval._v2_judging import (
    CONVERGED,
    DEEP_NULL_DB,
    FLAT_STEP,
    FREQ_BAR,
    LADDER_AGREEMENT,
    MAG_BAR_DB,
    RESISTANCE_BAR,
    aligned_magnitude,
    input_resistance,
    mesh_statement,
    refined_remax,
    remax_half_grid_witness,
)

# ---------------------------------------------------------------- geometry
# Every length in metres, every frequency in hertz.  The board is the retired
# script's (``validation/crossval/15_patch_antenna_rt5880.py``, removed with
# this test's arrival; its constants are the reference maker's own
# ``RETIRED_CONSTANTS`` block, which the maker's self-check pins).
EPS_R = 2.2                  # RT/Duroid 5880
TAN_DELTA = 1.0e-3
H_SUB = 3.175e-3             # 1/8 inch
EPS0 = 8.8541878128e-12
# The loss tangent as a conductivity at 2.4 GHz — the retired builder's and the
# openEMS record's own ``kappa`` (``stand_in.materials.sub.kappa``).
SIGMA_SUB = 2.0 * math.pi * 2.4e9 * EPS0 * EPS_R * TAN_DELTA
# PI 2026-09-25: the board every rung can represent.  Each edge sits just
# outside a node common to h/4, h/8 and h/12 (the patch centre is a node):
# half-extents 25, 31, 35 and 41 cells of h/4 plus 6.25, 18.75, 18.75 and
# 6.25 µm.  The retired script's board was 40 × 50 mm on a 56 × 66 mm ground;
# the openEMS maker declares the same move (its delta 10).  Its 40 mm length
# covers the same node columns as 39.7 mm on all three rungs, so
# ``check_realized`` cannot tell the two apart; the board-vs-record test does.
L_PATCH = 39.7e-3            # resonant length, along x
W_PATCH = 49.25e-3           # radiating-edge width, along y
GP_X = 55.6e-3               # ground plane; the substrate covers the same footprint
GP_Y = 65.1e-3
# What every rung must realize (``check_realized``): the patch and ground node
# spans between the common nodes, and the substrate, a volume, one cell more
# than the ground's node span: its cell count times dx is that span plus dx,
# along x and along y.
_Q = H_SUB / 4
PATCH_NODE_SPAN_M = (2 * 25 * _Q, 2 * 31 * _Q)      # 39.6875 × 49.2125 mm
GROUND_NODE_SPAN_M = (2 * 35 * _Q, 2 * 41 * _Q)     # 55.5625 × 65.0875 mm
SPAN_REALIZED_TOL_M = 1e-9
# The probe, off centre along the resonant length.  PI 2026-09-24: moved from
# the retired script's −9.0 mm to −8.73125 mm = −11, −22 and −33 cells of h/4,
# h/8 and h/12 from the patch centre, a lattice node on every rung, so the
# ladder solves one board; the openEMS maker declares the same move (its
# delta 9).  ``check_realized`` holds the realized probe to it within 1 nm.
FEED_OFFSET_X = -8.73125e-3
# The realized probe (the node its column sits on, relative to the realized
# patch centre) must be the declared one to this, on every rung.
PROBE_REALIZED_TOL_M = 1e-9
PORT_IMPEDANCE_OHM = 50.0
# The retired builder's excitation and band.
F_EXCITE_HZ = 2.4e9
EXCITE_BANDWIDTH = 1.0
FREQ_MAX_HZ = 4.0e9

# --- the box --------------------------------------------------------------
# The patch centre is a whole number of substrate thicknesses from the domain
# origin on both axes, so it is a NODE on every h/n rung: the patch and the
# ground then realize symmetric about it, and the probe's realized offset from
# the patch centre is its own node minus that node.  The domain is twice that,
# so the box is symmetric too.  Every face of the box is a whole number of h
# from the metal, so the box is also the same on every rung.
#
# The air between the metal and the absorber, and the absorber's own
# thickness, are both held in PHYSICAL length on every rung: ``AIR_H`` more
# substrate thicknesses of air than the smallest box below on every face, and
# a CPML ``CPML_THICKNESS_H`` substrate thicknesses deep, i.e.
# CPML_THICKNESS_H·n cells at h/n.  A CPML of a fixed CELL count (the sibling
# cases' eight) gets physically thinner as the mesh refines, and on this board
# the absorber moves the resonance (the table below), so a fixed cell count
# would put an absorber change into the mesh trend.
#
# MEASURED on the retired 40 × 50 mm board at h/4 (CPU, 2026-09-24; the table
# is in this file at 0a92af49): with a 16-cell absorber the resonance kept
# falling as air was added, up to 9·h, and had not levelled there; with 24
# cells it fell too, and at every air gap a deeper absorber lowered it.  The
# ladder's box is ``AIR_H`` = 0 with a 6·h (19.05 mm) absorber: at h/12 it is
# 52.7 M cells, where +6·h of air at the same depth would be 143 M (an RTX 4090
# ran out of memory on 134 M, docs/agent/gpu-throughput.mdx).  The
# reference-box run below (``reference_box_frame``) solves the coarsest rung
# once more in the openEMS record's own box on every ladder run, and reports it.
AIR_H = 0
CPML_THICKNESS_H = 6


class Frame(NamedTuple):
    """Where the board sits in the box, for ``air_h`` h of extra air."""
    air_h: int
    cx: float           # patch (and ground) centre, x
    cy: float           # patch (and ground) centre, y
    z_ground: float     # the substrate floor, the ground sheet's plane
    z_patch: float      # the substrate top, the patch sheet's plane
    domain: tuple       # the declared domain, CPML outside it
    label: str = ""     # what the box is, for the banner (empty: the ladder's)
    cpml_cells: int = 0  # the absorber in cells; 0 is the ladder's CPML_THICKNESS_H·h


def frame(air_h: int = AIR_H) -> Frame:
    cx = (12 + air_h) * H_SUB
    cy = (14 + air_h) * H_SUB
    z_ground = (3 + air_h) * H_SUB
    return Frame(air_h, cx, cy, z_ground, z_ground + H_SUB,
                 (2 * cx, 2 * cy, (9 + 2 * air_h) * H_SUB))


def reference_box_frame(dx: float, rec: dict) -> Frame:
    """The openEMS record's own box around this board, at ``dx``.

    The record lays its absorber's inner faces at fixed positions about the
    ground's centre on every rung (``meta.stages.<rung>.geometry_realized.pml.
    faces``: x ±88, y ±93, z −40 / +90 mm).  Here the declared domain's faces
    go there, each snapped to the nearest node of ``dx`` counted from the
    ground's centre and from the ground plane, so that centre and both substrate
    faces stay on nodes and the board realizes exactly as on the ladder's rung;
    the CPML is the ladder's, ``CPML_THICKNESS_H``·h outside the domain.
    """
    stages = rec["meta"]["stages"]
    faces = [stages[n]["geometry_realized"]["pml"]["faces"]
             for n in OPENEMS_MESH_STATEMENT_STAGES + (OPENEMS_JUDGED_STAGE,)]
    keys = ("inner_face_lo_mm", "inner_face_hi_mm")
    for f in faces[1:]:
        assert all(f[a][k] == faces[0][a][k] for a in "xyz" for k in keys), (
            "the record's rungs do not share one absorber box")
    f = faces[0]

    def cells(mm: float) -> int:
        return int(round(abs(mm) * 1e-3 / dx))

    nx_lo, nx_hi = cells(f["x"]["inner_face_lo_mm"]), cells(f["x"]["inner_face_hi_mm"])
    ny_lo, ny_hi = cells(f["y"]["inner_face_lo_mm"]), cells(f["y"]["inner_face_hi_mm"])
    nz_lo, nz_hi = cells(f["z"]["inner_face_lo_mm"]), cells(f["z"]["inner_face_hi_mm"])
    z_ground = nz_lo * dx
    return Frame(-1, nx_lo * dx, ny_lo * dx, z_ground, z_ground + H_SUB,
                 ((nx_lo + nx_hi) * dx, (ny_lo + ny_hi) * dx, (nz_lo + nz_hi) * dx),
                 label=(f"the openEMS record's box: faces x −{nx_lo*dx*1e3:.3f} / "
                        f"+{nx_hi*dx*1e3:.3f}, y −{ny_lo*dx*1e3:.3f} / "
                        f"+{ny_hi*dx*1e3:.3f}, z −{nz_lo*dx*1e3:.3f} / "
                        f"+{nz_hi*dx*1e3:.3f} mm about the ground's centre (record: "
                        f"x {f['x']['inner_face_lo_mm']:+.0f} / "
                        f"{f['x']['inner_face_hi_mm']:+.0f}, y "
                        f"{f['y']['inner_face_lo_mm']:+.0f} / "
                        f"{f['y']['inner_face_hi_mm']:+.0f}, z "
                        f"{f['z']['inner_face_lo_mm']:+.0f} / "
                        f"{f['z']['inner_face_hi_mm']:+.0f} mm)"))


FRAME = frame(AIR_H)
PATCH_CENTRE_X_M = FRAME.cx
PATCH_CENTRE_Y_M = FRAME.cy
GROUND_Z_M = FRAME.z_ground
PATCH_Z_M = FRAME.z_patch
DOMAIN_X_M, DOMAIN_Y_M, DOMAIN_Z_M = FRAME.domain
# Clearances this box leaves, for the banner.
X_CLEARANCE_M = PATCH_CENTRE_X_M - GP_X / 2
Y_CLEARANCE_M = PATCH_CENTRE_Y_M - GP_Y / 2
BELOW_GROUND_M = GROUND_Z_M
ABOVE_PATCH_M = DOMAIN_Z_M - PATCH_Z_M

# The board in a box barely larger than its ground, with an eight-cell absorber
# outside it.  Nothing is solved in it: the fast tests build every candidate
# rung in it, h/12 included, whose build in the ladder's box does not fit the
# PR lane (``FAST_LADDER_M``).  Its patch centre and ground plane are whole
# numbers of h from the domain origin, as in the ladder's box, so every h/n
# puts the same nodes under the board; the fast test checks that the two boxes
# realize the same board on the rungs it builds in both.
BUILD_CHECK_FRAME = Frame(-2, 10 * H_SUB, 12 * H_SUB, H_SUB, 2 * H_SUB,
                          (20 * H_SUB, 24 * H_SUB, 3 * H_SUB),
                          label="a build-only box just around the board",
                          cpml_cells=8)


def cpml_layers(dx: float) -> int:
    """CPML cells at ``dx`` for an absorber ``CPML_THICKNESS_H``·h deep."""
    n = CPML_THICKNESS_H * H_SUB / dx
    assert abs(n - round(n)) < 1e-9, (dx, n)
    return int(round(n))


# --- the ladder -----------------------------------------------------------
# dx = h_sub/n keeps both substrate faces on node planes (issue #723).
#
# THE RUNG CRITERION IS ONE BOARD.  Each edge is declared just outside a node
# common to h/4, h/8 and h/12 and the probe on one (the constants above), and
# ``check_realized`` refuses a rung whose patch or ground node spans, substrate
# cell counts or probe node differ from those.  Every h/n from h/4 to h/12 is
# built in the fast test (``CANDIDATE_RUNGS_N``); only h/4, h/8 and h/12 pass.
LADDER_M = (
    H_SUB / 4,    # 793.750 µm,  4 substrate cells
    H_SUB / 8,    # 396.875 µm,  8 substrate cells
    H_SUB / 12,   # 264.583 µm, 12 substrate cells
)
DX_COARSEST_M = LADDER_M[0]

# --- the sweep ------------------------------------------------------------
# rfx samples the record's OWN frequency grid, linspace(1.6, 3.4 GHz, 901), so
# the judged comparison needs no interpolation.  DFT bins cost no time steps.
# The bin is 2 MHz; 1 % of any frequency above 2 GHz is ten bins or more.
N_FREQS = 901
SWEEP_HZ = (1.6e9, 3.4e9)
FREQS_HZ = np.linspace(SWEEP_HZ[0], SWEEP_HZ[1], N_FREQS)

# The resonance is searched inside the record's own window
# (``meta.resonance_band_ghz``): 0.80–1.20 × the Balanis transmission-line
# estimate of this patch's TM010 (``meta.tl_model_tm010_hz``), which is the
# retired script's own search window.  Two estimators run in it: the judged
# resonance f0 is the Re(Zin) peak (``refined_remax`` in
# tests/crossval/_v2_judging.py), and the reported |S11| minimum is the
# repository's shared ``refined_extremum`` on log|S11|, the record's own.
F_TM010_TL_MODEL_HZ = 2433373588.639094
RESONANCE_BAND_HZ = (0.80 * F_TM010_TL_MODEL_HZ, 1.20 * F_TM010_TL_MODEL_HZ)
MINUS10_DB = -10.0

# --- the record length ----------------------------------------------------
# ``num_periods`` is in periods of ``freq_max`` (0.25 ns here).  MEASURED on
# the retired 40 × 50 mm board at h/4 (CPU, 2026-09-24; the table is in this
# file at 0a92af49): 60 met the −40 dB rule with margin and moved the
# resonance and its depth by nothing that matters against the doubled record.
# The ring-down is set by the patch's radiation Q, a physical time, so the same
# record length in nanoseconds serves every rung, and every run checks its own
# settling against ``SETTLING_DB`` before it is compared.
NUM_PERIODS = 60.0

# --- the ring-down witness probe -------------------------------------------
# Ez inside the substrate, halfway up, under the patch near its +x radiating
# edge where the TM010 field is strong, and 24 mm from the probe so it never
# shares a cell with the drive (a source-dominated record measures the pulse
# turning off, not the ring-down, #1090).  A node on every rung: 15 mm and
# 5 mm snap to the nearest node, and h/2 is a whole number of cells on each.
WITNESS_PROBE_OFFSET_M = (15.0e-3, 5.0e-3, H_SUB / 2)

# ------------------------------------------------------------ thresholds
# The v2 accuracy bar (FREQ_BAR 1 %, MAG_BAR_DB 2 dB; PI 2026-09-20), the
# mesh statement (LADDER_AGREEMENT 1 %, FLAT_STEP 0.1 %), the deep-null level
# (DEEP_NULL_DB −20 dB) and the input-resistance bar (RESISTANCE_BAR 2 %,
# decision (가)) are imported from tests/crossval/_v2_judging.py, with the
# rules that use them (PI 2026-09-22 and 2026-09-24).  MAG_BAR_DB judges
# nothing here since decision (가): the magnitude is reported.
# Witnesses on rfx's own record, not comparisons (rfx CLAUDE.md, Validation
# rules): a record that has not rung down to −40 dB is truncation-suspect, and
# a passive one-port cannot reflect more power than it receives, so |S11| more
# than one percent above one is a measurement artefact — the rung stops instead
# of being compared.
SETTLING_DB = -40.0
PASSIVITY_EXCESS_BAR = 0.01

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_OPENEMS_JSON = _REFERENCE_DIR / "openems_patch.json"
# The rung of the openEMS record this case is judged against, and the two that
# are its own mesh statement.
OPENEMS_JUDGED_STAGE = "stage_b_fine"
OPENEMS_MESH_STATEMENT_STAGES = ("stage_b_coarse", "stage_b_mid")


# ------------------------------------------------------------------ build
def _sim(dx: float, fr: Frame = FRAME) -> Simulation:
    """The box, the boundary and the lossy substrate material — shared by
    every builder."""
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=fr.domain,
        dx=dx,
        cpml_layers=fr.cpml_cells or cpml_layers(dx),
        boundary=BoundarySpec.uniform("cpml"),
    )
    sim.add_material("rt5880", eps_r=EPS_R, sigma=SIGMA_SUB)
    return sim


def _ground_box(fr: Frame = FRAME, z: float | None = None) -> Box:
    z = fr.z_ground if z is None else z
    return Box((fr.cx - GP_X / 2, fr.cy - GP_Y / 2, z),
               (fr.cx + GP_X / 2, fr.cy + GP_Y / 2, z))


def _substrate_box(fr: Frame = FRAME, footprint: tuple | None = None) -> Box:
    fx, fy = (GP_X, GP_Y) if footprint is None else footprint
    return Box((fr.cx - fx / 2, fr.cy - fy / 2, fr.z_ground),
               (fr.cx + fx / 2, fr.cy + fy / 2, fr.z_patch))


def _patch_box(fr: Frame = FRAME, *, x_hi_trim: float = 0.0,
               extra_thickness: float = 0.0) -> Box:
    return Box((fr.cx - L_PATCH / 2, fr.cy - W_PATCH / 2, fr.z_patch),
               (fr.cx + L_PATCH / 2 - x_hi_trim, fr.cy + W_PATCH / 2,
                fr.z_patch + extra_thickness))


def _add_probe_and_witness(sim: Simulation, fr: Frame = FRAME, *,
                           extent: float = H_SUB, feed_offset: float | None = None) -> None:
    """The 50 Ω probe from the ground sheet to the patch sheet, and the
    ring-down witness probe.  ``feed_offset`` (default ``FEED_OFFSET_X``) is
    only moved by the mutation test."""
    feed_offset = FEED_OFFSET_X if feed_offset is None else feed_offset
    sim.add_port(position=(fr.cx + feed_offset, fr.cy, fr.z_ground), component="ez",
                 impedance=PORT_IMPEDANCE_OHM, extent=extent,
                 waveform=GaussianPulse(f0=F_EXCITE_HZ, bandwidth=EXCITE_BANDWIDTH))
    ox, oy, oz = WITNESS_PROBE_OFFSET_M
    sim.add_probe(position=(fr.cx + ox, fr.cy + oy, fr.z_ground + oz), component="ez")


def build(dx: float, fr: Frame = FRAME) -> Simulation:
    """The reference structure on a uniform ``dx`` mesh.

    The ground and the patch are zero-thickness PEC SHEETS on the substrate's
    two faces (both z corners equal — a Box drawn ``h -> h + dx`` would be a
    one-cell VOLUME with a wall on each face, #931 §1.3); the openEMS record
    draws both with zero thickness too.  The substrate is the lossy laminate
    over the ground's footprint.  The probe is a wire port from the ground
    sheet's plane to the patch sheet's, the span openEMS's lumped port has.
    ``fr`` places the board in the box; only the reference-box run passes
    anything but the default.
    """
    sim = _sim(dx, fr)
    sim.add(_ground_box(fr), material="pec")
    sim.add(_substrate_box(fr), material="rt5880")
    sim.add(_patch_box(fr), material="pec")
    _add_probe_and_witness(sim, fr)
    return sim


def _build_patch_as_volume(dx: float) -> Simulation:
    """Same board, but the patch drawn as a one-cell thick VOLUME.

    Only the mutation test builds it: it is the defect ``assert_realized``
    exists to refuse (a wall on each face of the slab, one plane more than the
    zero-thickness patch the record solves).
    """
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(), material="rt5880")
    sim.add(_patch_box(extra_thickness=dx), material="pec")
    _add_probe_and_witness(sim)
    return sim


def _build_ground_one_plane_low(dx: float) -> Simulation:
    """The ground sheet drawn one node plane below the substrate floor.

    The pre-#931 build of this board put its ground wall exactly there, and the
    cavity it solved was a vacuum cell thicker than the laminate: the resonance
    read 6 % high (the retired script's #740 history).  Only the mutation test
    builds it.
    """
    sim = _sim(dx)
    sim.add(_ground_box(z=FRAME.z_ground - dx), material="pec")
    sim.add(_substrate_box(), material="rt5880")
    sim.add(_patch_box(), material="pec")
    _add_probe_and_witness(sim)
    return sim


def _build_feed_one_cell_short(dx: float) -> Simulation:
    """The probe drawn one cell short of the patch.

    A probe that stops a cell below the patch couples to it through a gap
    capacitance instead of touching it: the retired script's floating post
    (#920), whose |S11| dip filled in to −0.32 dB.  Only the mutation test
    builds it.
    """
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(), material="rt5880")
    sim.add(_patch_box(), material="pec")
    _add_probe_and_witness(sim, extent=H_SUB - dx)
    return sim


def _build_patch_one_cell_short(dx: float) -> Simulation:
    """The patch drawn one cell shorter along its resonant length (its +x
    radiating edge pulled in by dx).  Only the mutation test builds it."""
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(), material="rt5880")
    sim.add(_patch_box(x_hi_trim=dx), material="pec")
    _add_probe_and_witness(sim)
    return sim


def _build_substrate_on_the_knife_edge(dx: float) -> Simulation:
    """The substrate drawn exactly on the ground's outermost nodes (the common
    nodes) instead of just outside them.  Only the mutation test builds it:
    a volume whose face sits on a node takes or drops the cell beyond it by
    rounding, so the laminate no longer overhangs the ground by one cell."""
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(footprint=GROUND_NODE_SPAN_M), material="rt5880")
    sim.add(_patch_box(), material="pec")
    _add_probe_and_witness(sim)
    return sim


def _build_ground_and_substrate_shifted_one_cell(dx: float) -> Simulation:
    """The ground and the substrate drawn one cell to +x of the patch.  Only
    the mutation test builds it: every span holds, but the board is not
    centred under the patch."""
    sim = _sim(dx)
    shifted = FRAME._replace(cx=FRAME.cx + dx)
    sim.add(_ground_box(shifted), material="pec")
    sim.add(_substrate_box(shifted), material="rt5880")
    sim.add(_patch_box(), material="pec")
    _add_probe_and_witness(sim)
    return sim


def _build_substrate_shifted_one_cell(dx: float) -> Simulation:
    """The substrate alone drawn one cell to +y of the ground and the patch.
    Only the mutation test builds it: its cell count holds, its position does
    not."""
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(FRAME._replace(cy=FRAME.cy + dx)), material="rt5880")
    sim.add(_patch_box(), material="pec")
    _add_probe_and_witness(sim)
    return sim


def _build_probe_at_the_retired_offset(dx: float) -> Simulation:
    """The probe drawn at the retired script's −9.0 mm.  Only the mutation test
    builds it: at h/8 that is another node (−23 cells, −9.12813 mm) than the
    declared −8.73125 mm (−22); at h/4 both snap to the same node."""
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(), material="rt5880")
    sim.add(_patch_box(), material="pec")
    _add_probe_and_witness(sim, feed_offset=-9.0e-3)
    return sim


def _build_probe_shorted_by_a_wire(dx: float) -> Simulation:
    """A PEC wire drawn along the probe, from the ground sheet to the patch
    sheet: every edge the port drives is then a PEC edge, and the port drives a
    short instead of a gap.  Only the mutation test builds it."""
    sim = _sim(dx)
    sim.add(_ground_box(), material="pec")
    sim.add(_substrate_box(), material="rt5880")
    sim.add(_patch_box(), material="pec")
    px, py = FRAME.cx + FEED_OFFSET_X, FRAME.cy
    sim.add(PolylineWire(points=((px, py, FRAME.z_ground), (px, py, FRAME.z_patch)),
                         radius=1e-6), material="pec")
    _add_probe_and_witness(sim)
    return sim


# -------------------------------------------------------- realized board
def _span(fp: np.ndarray, xs: np.ndarray, ys: np.ndarray, dx: float) -> dict:
    idx = np.where(fp)
    i0, i1 = int(idx[0].min()), int(idx[0].max())
    j0, j1 = int(idx[1].min()), int(idx[1].max())
    return {
        "cols": (i0, i1), "rows": (j0, j1),
        "n_cols": i1 - i0 + 1, "n_rows": j1 - j0 + 1,
        "x_m": (float(xs[i0]), float(xs[i1])),
        "y_m": (float(ys[j0]), float(ys[j1])),
        "x_node_span_m": float(xs[i1] - xs[i0]),
        "y_node_span_m": float(ys[j1] - ys[j0]),
        "x_strip_m": (i1 - i0 + 1) * dx,
        "y_strip_m": (j1 - j0 + 1) * dx,
        "centre_m": (0.5 * float(xs[i0] + xs[i1]), 0.5 * float(ys[j0] + ys[j1])),
    }


def realized_geometry(sim: Simulation) -> dict:
    """What the lattice built, read from the product's own realization owner.

    ``tests/_realized_geometry.realized`` assembles the simulation without a
    time step and hands back the realized PEC edge masks and the declared sheet
    footprints; the wall planes come from ``rfx.boundaries.pec.
    realized_wall_planes``, the probe's driven edges from the port runner's own
    ``_wire_port_cells`` and the substrate from the assembled material arrays.
    Every number here is the solver's, not a rule re-derived in the test.

    A zero-thickness sheet IS its set of tangential edges, and *n* footprint
    node columns carry *n* − 1 edges.  Every span is therefore reported twice:
    ``*_node_span_m`` = (n − 1)·dx is the geometric extent between the
    outermost nodes, ``*_strip_m`` = n·dx the width a quasi-TEM formula takes.
    """
    from rfx.boundaries.pec import edge_is_pec, realized_wall_planes
    from rfx.sources.sources import WirePort, _wire_port_cells

    rz = realized(sim)
    grid = rz.grid
    xs, ys, zs = (_node_line(grid, a) for a in (0, 1, 2))
    dx = float(xs[1] - xs[0])
    periodic = sim._periodic_flags()

    planes = [int(p) for p in realized_wall_planes(rz.edge_masks, 2)]
    n_volume_cells = 0 if rz.pec_mask is None else int(np.asarray(rz.pec_mask).sum())
    z_sheets = [sp for sp in rz.sheets if int(sp.normal_axis) == 2]

    out: dict = {
        "wall_planes": planes,
        "wall_planes_z_m": [float(zs[k]) for k in planes],
        "n_wall_planes": len(planes),
        "n_declared_sheets": len(rz.sheets),
        "n_z_sheets": len(z_sheets),
        "n_volume_cells": n_volume_cells,
        "dx_m": dx,
        "grid_shape": tuple(int(n) for n in (xs.size, ys.size, zs.size)),
        "n_cells": int(xs.size * ys.size * zs.size),
        "n_ports": len(getattr(sim, "_ports", []) or []),
    }

    # The substrate, from the assembled material: the cells along z that carry
    # the laminate's permittivity at the probe's own column, and every distinct
    # permittivity the assembly wrote (a sheet owns no cell and writes none).
    # The arrays are float32: 2.2 is stored as 2.2000000477, so values are
    # compared to 1e-6, far below any permittivity step and far above that.
    mats = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    eps = np.asarray(mats.eps_r, dtype=float)
    out["distinct_eps_r"] = sorted(float(v) for v in np.unique(np.round(eps, 6)))

    # The probe, from the registered port and the runner's own edge span.
    ports = list(getattr(sim, "_ports", []) or [])
    if len(ports) == 1 and ports[0].extent is not None:
        pe = ports[0]
        end = (pe.position[0], pe.position[1], pe.position[2] + pe.extent)
        cells = _wire_port_cells(grid, WirePort(start=tuple(pe.position), end=end,
                                                component=pe.component,
                                                impedance=pe.impedance))
        i, j = int(cells[0][0]), int(cells[0][1])
        k0, k1 = int(cells[0][2]), int(cells[-1][2]) + 1
        column_planes = [int(p) for p in realized_wall_planes(
            rz.edge_masks, 2, ij=(i, j), periodic=periodic)]
        live = [not edge_is_pec(rz.edge_masks, pe.component, *c) for c in cells]
        col_eps = eps[i, j, :]
        sub = np.flatnonzero(np.abs(col_eps - EPS_R) < 1e-6)
        out.update(
            probe_component=pe.component,
            probe_impedance_ohm=float(pe.impedance),
            probe_column=(i, j),
            probe_x_m=float(xs[i]), probe_y_m=float(ys[j]),
            probe_nodes=(k0, k1),
            probe_z_m=(float(zs[k0]), float(zs[k1])),
            probe_n_edges=len(cells),
            probe_n_live_edges=int(sum(live)),
            probe_column_wall_planes=column_planes,
            substrate_cells=int(sub.size),
            substrate_cells_k=(int(sub.min()), int(sub.max())) if sub.size else None,
            substrate_contiguous=bool(sub.size) and int(sub.max() - sub.min() + 1) == int(sub.size),
        )

    if len(z_sheets) != 2:
        return out
    # The ground is the larger of the two sheets.
    spans = [(_span(np.asarray(sp.footprint), xs, ys, dx), int(sp.plane)) for sp in z_sheets]
    spans.sort(key=lambda t: t[0]["n_cols"] * t[0]["n_rows"], reverse=True)
    (ground, k_ground), (patch, k_patch) = spans
    out.update(
        ground=ground, patch=patch,
        ground_plane_k=k_ground, patch_plane_k=k_patch,
        ground_plane_z_m=float(zs[k_ground]), patch_plane_z_m=float(zs[k_patch]),
        cells_between_sheets=k_patch - k_ground,
    )
    if "probe_x_m" in out:
        out["probe_offset_from_patch_centre_m"] = (
            out["probe_x_m"] - patch["centre_m"][0],
            out["probe_y_m"] - patch["centre_m"][1])
    # rfx's own model of where a PEC sheet's edge is SOLVED
    # (``rfx.mesh_edges.EDGE_OFFSET``: 0.35 cell beyond its last node, the
    # model the preflight's ``sheet_effective_size`` advisory prints).
    # Reported, never asserted.
    from rfx.mesh_edges import EDGE_OFFSET
    out["edge_offset_cells"] = float(EDGE_OFFSET)
    out["patch_x_solved_m"] = patch["x_node_span_m"] + 2 * EDGE_OFFSET * dx
    out["patch_y_solved_m"] = patch["y_node_span_m"] + 2 * EDGE_OFFSET * dx
    # The substrate in plane, from the same assembled material: the cells that
    # carry the laminate's permittivity along x and along y through the
    # patch's centre node, halfway between the two sheets.
    i_c = int(np.argmin(np.abs(xs - patch["centre_m"][0])))
    j_c = int(np.argmin(np.abs(ys - patch["centre_m"][1])))
    k_c = (k_ground + k_patch) // 2
    on = np.abs(eps - EPS_R) < 1e-6
    out["substrate_x_cells"] = int(on[:, j_c, k_c].sum())
    out["substrate_y_cells"] = int(on[i_c, :, k_c].sum())
    out["substrate_x_m"] = out["substrate_x_cells"] * dx
    out["substrate_y_m"] = out["substrate_y_cells"] * dx
    # Where those cells sit: the first and last laminate node along each line,
    # counted from the patch centre node (a centred substrate reads (-N, +N)).
    xi = np.flatnonzero(on[:, j_c, k_c])
    yj = np.flatnonzero(on[i_c, :, k_c])
    out["substrate_x_nodes_from_centre"] = (
        (int(xi.min()) - i_c, int(xi.max()) - i_c) if xi.size else None)
    out["substrate_y_nodes_from_centre"] = (
        (int(yj.min()) - j_c, int(yj.max()) - j_c) if yj.size else None)
    return out


def assert_realized(sim: Simulation, dx: float, fr: Frame = FRAME) -> dict:
    """Refuse to solve unless the lattice built the declared board.

    Runs before every rung, with no time step.  Raises with the measured
    numbers; the caller prints the dict either way.
    """
    return check_realized(realized_geometry(sim), dx, fr)


def check_realized(g: dict, dx: float, fr: Frame = FRAME) -> dict:
    """The checks ``assert_realized`` runs, on an already measured board."""
    n_sub = int(round(H_SUB / dx))

    # --- the two sheets, each on exactly one node plane ---------------------
    if g["n_z_sheets"] != 2 or g["n_wall_planes"] != 2:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the conductors realize {g['n_z_sheets']} "
            f"z-normal sheet(s) and tangential wall planes along z at "
            f"{g['wall_planes']} (z = "
            f"{[round(z*1e3, 5) for z in g['wall_planes_z_m']]} mm), not two sheets "
            "on two planes. The ground and the patch are declared as "
            "zero-thickness SHEETS on the substrate's two faces; a third wall "
            "plane is what a one-cell VOLUME realizes (#931 §1.3), a wall the "
            "openEMS record's zero-thickness metal has no counterpart for.")
    if g["n_volume_cells"] != 0:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_volume_cells']} PEC VOLUME cell(s) were "
            "realized; this board declares sheets only.")
    for name, got, declared in (("ground", g["ground_plane_z_m"], fr.z_ground),
                                ("patch", g["patch_plane_z_m"], fr.z_patch)):
        if abs(got - declared) >= 1e-12:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the {name} sheet's wall plane is at "
                f"z={got*1e3:.5f} mm, {abs(got-declared)*1e6:.3f} µm off the "
                f"declared substrate {'floor' if name == 'ground' else 'top'} "
                f"{declared*1e3:.5f} mm. A ground a plane below the laminate "
                "leaves a vacuum layer inside the cavity (the retired script's "
                "#740 build read 6 % high for exactly this).")

    # --- the substrate, n cells between the two sheets -------------------------
    if g["cells_between_sheets"] != n_sub:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['cells_between_sheets']} cells lie between the "
            f"ground plane k={g['ground_plane_k']} and the patch plane "
            f"k={g['patch_plane_k']}; h/dx is {n_sub}.")
    if g.get("substrate_cells") != n_sub or not g.get("substrate_contiguous"):
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the laminate is {g.get('substrate_cells')} "
            f"cell(s) thick at the probe column (cells k={g.get('substrate_cells_k')}, "
            f"contiguous={g.get('substrate_contiguous')}); h/dx is {n_sub}.")
    if g["distinct_eps_r"] != sorted({1.0, EPS_R}):
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the assembly wrote permittivities "
            f"{g['distinct_eps_r']}, not only vacuum and the laminate's {EPS_R}. "
            "A sheet owns no cell and writes no material (#931 §1.3).")

    # --- the metal, within a cell of the drawing ---------------------------------
    for name, got, declared in (
        ("patch x (resonant length)", g["patch"]["x_strip_m"], L_PATCH),
        ("patch y (radiating width)", g["patch"]["y_strip_m"], W_PATCH),
        ("ground x", g["ground"]["x_strip_m"], GP_X),
        ("ground y", g["ground"]["y_strip_m"], GP_Y),
    ):
        if abs(got - declared) > dx:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the {name} realizes a {got*1e3:.4f} mm strip, "
                f"more than one cell from the declared {declared*1e3:.3f} mm.")

    # --- one board on every rung: the sheets and the substrate -------------------
    # PI 2026-09-25: the patch and ground node spans are the common nodes, and
    # the substrate's cells span the ground's node span plus one cell, on every
    # rung.  A rung that realizes anything else solves a different antenna, and
    # a ladder through it would put that change into the mesh trend.
    held = (("the patch's node span along x (the resonant length)",
             g["patch"]["x_node_span_m"], PATCH_NODE_SPAN_M[0]),
            ("the patch's node span along y (the radiating width)",
             g["patch"]["y_node_span_m"], PATCH_NODE_SPAN_M[1]),
            ("the ground's node span along x", g["ground"]["x_node_span_m"],
             GROUND_NODE_SPAN_M[0]),
            ("the ground's node span along y", g["ground"]["y_node_span_m"],
             GROUND_NODE_SPAN_M[1]),
            ("the substrate's cells along x times dx", g["substrate_x_m"],
             GROUND_NODE_SPAN_M[0] + dx),
            ("the substrate's cells along y times dx", g["substrate_y_m"],
             GROUND_NODE_SPAN_M[1] + dx))
    for name, got, want in held:
        if abs(got - want) > SPAN_REALIZED_TOL_M:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: {name} is {got*1e3:.6f} mm; the board holds "
                f"{want*1e3:.6f} mm on every rung ({100*(got-want)/want:+.4f} %). "
                "Every edge is declared just outside a node common to h/4, h/8 and "
                "h/12; a rung that realizes another span solves a different "
                "antenna, and a ladder through it would put that change into the "
                "mesh trend.")
    cx, cy = g["patch"]["centre_m"]
    if abs(cx - fr.cx) > 1e-9 or abs(cy - fr.cy) > 1e-9:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the realized patch is centred at "
            f"({cx*1e3:.4f}, {cy*1e3:.4f}) mm, not at the declared "
            f"({fr.cx*1e3:.4f}, {fr.cy*1e3:.4f}) mm; the "
            "probe's offset from the patch centre is then not the declared one.")
    # Sizes alone do not make one board: the ground and the laminate must also
    # sit under the patch.  One cell of laminate at a board edge moved R(f0) by
    # 1.37 % at h/4 (VESSL 369367264633), so a shifted ground or substrate is a
    # different antenna even when every span above holds.
    gx, gy = g["ground"]["centre_m"]
    if abs(gx - cx) > 1e-9 or abs(gy - cy) > 1e-9:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the realized ground is centred at "
            f"({gx*1e3:.4f}, {gy*1e3:.4f}) mm and the patch at "
            f"({cx*1e3:.4f}, {cy*1e3:.4f}) mm; the board holds the ground centred "
            "under the patch.")
    for axis in ("x", "y"):
        rng = g[f"substrate_{axis}_nodes_from_centre"]
        n = g[f"substrate_{axis}_cells"]
        if rng is None or rng[0] != -rng[1] or rng[1] - rng[0] + 1 != n:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the substrate's cells along {axis} run from "
                f"node {None if rng is None else rng[0]} to "
                f"{None if rng is None else rng[1]} about the patch centre node "
                f"({n} laminate cells on that line); the board holds a contiguous "
                "laminate centred under the patch.")

    # --- the probe, from the ground sheet to the patch sheet -------------------------
    if g["n_ports"] != 1 or g.get("probe_component") != "ez":
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_ports']} port(s) registered; this board "
            "has one z-directed wire port.")
    k0, k1 = g["probe_nodes"]
    contacts = (k0 == g["ground_plane_k"] and k0 in g["probe_column_wall_planes"],
                k1 == g["patch_plane_k"] and k1 in g["probe_column_wall_planes"])
    if not all(contacts):
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the probe's driven edges run from node k={k0} "
            f"(z={g['probe_z_m'][0]*1e3:.5f} mm) to k={k1} "
            f"(z={g['probe_z_m'][1]*1e3:.5f} mm) at column {g['probe_column']}, "
            f"where the realized wall planes are {g['probe_column_wall_planes']}; "
            f"the ground sheet is at k={g['ground_plane_k']} and the patch sheet "
            f"at k={g['patch_plane_k']}. The probe does not reach "
            + ("the ground sheet" if not contacts[0] else "the patch sheet")
            + ": an end that stops short couples through a gap capacitance "
              "instead of touching the metal (the retired script's floating "
              "post, #920), and openEMS's lumped port spans ground to patch.")
    if g["probe_n_live_edges"] == 0:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: every one of the probe's {g['probe_n_edges']} "
            "driven edges is a PEC edge; contact with two wall planes inside one "
            "metal body is not a driven gap.")
    ox, oy = g["probe_offset_from_patch_centre_m"]
    if abs(ox - FEED_OFFSET_X) > PROBE_REALIZED_TOL_M or abs(oy) > PROBE_REALIZED_TOL_M:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the probe realizes ({ox*1e3:.6f}, {oy*1e3:.6f}) mm "
            f"from the patch centre; declared ({FEED_OFFSET_X*1e3:.6f}, 0) mm "
            f"(tolerance {PROBE_REALIZED_TOL_M*1e9:.0f} nm). The probe is declared "
            "on a node of every rung so that the ladder solves one antenna: the "
            "input resistance at resonance follows the probe's distance from the "
            "patch centre, and a probe that snaps to a different node per rung "
            "puts that change into the mesh trend.")
    return g


def _print_realized(g: dict, dx: float) -> None:
    print(f"  realized at dx = {dx*1e6:.3f} µm:")
    print(f"    grid {g['grid_shape']}  cells {g['n_cells']}  "
          f"substrate {g.get('substrate_cells')} cells (h/dx = {H_SUB/dx:.3f}); "
          f"permittivities written {g['distinct_eps_r']}")
    print(f"    wall planes along z {g['wall_planes']} at z = "
          f"{[round(z*1e3, 5) for z in g['wall_planes_z_m']]} mm; z-normal sheets "
          f"{g['n_z_sheets']}; PEC volume cells {g['n_volume_cells']}")
    for name, key, dec_x, dec_y in (("patch ", "patch", L_PATCH, W_PATCH),
                                    ("ground", "ground", GP_X, GP_Y)):
        s = g.get(key)
        if s is None:
            continue
        print(f"    {name}: plane k={g[f'{key}_plane_k']} "
              f"z={g[f'{key}_plane_z_m']*1e3:.5f} mm; cols {s['cols']} n={s['n_cols']}, "
              f"rows {s['rows']} n={s['n_rows']}")
        print(f"            x node span {s['x_node_span_m']*1e3:9.4f} mm, strip "
              f"{s['x_strip_m']*1e3:9.4f} mm (declared {dec_x*1e3:.3f}); y node span "
              f"{s['y_node_span_m']*1e3:9.4f} mm, strip {s['y_strip_m']*1e3:9.4f} mm "
              f"(declared {dec_y*1e3:.3f}); centre ({s['centre_m'][0]*1e3:.4f}, "
              f"{s['centre_m'][1]*1e3:.4f}) mm")
    if "substrate_x_cells" in g:
        print(f"    substrate through the patch centre: {g['substrate_x_cells']} × "
              f"{g['substrate_y_cells']} cells = {g['substrate_x_m']*1e3:.6f} × "
              f"{g['substrate_y_m']*1e3:.6f} mm")
    if "patch_x_solved_m" in g:
        print(f"    patch as rfx's EDGE_OFFSET model solves it "
              f"({g['edge_offset_cells']:.2f} cell past each edge node; reported): "
              f"{g['patch_x_solved_m']*1e3:.4f} × {g['patch_y_solved_m']*1e3:.4f} mm "
              f"({100*(g['patch_x_solved_m']-L_PATCH)/L_PATCH:+.3f} % × "
              f"{100*(g['patch_y_solved_m']-W_PATCH)/W_PATCH:+.3f} %)")
    if "probe_nodes" in g:
        off = g.get("probe_offset_from_patch_centre_m", (float("nan"),) * 2)
        print(f"    probe: column {g['probe_column']}, nodes {g['probe_nodes']} "
              f"(z {g['probe_z_m'][0]*1e3:.5f} -> {g['probe_z_m'][1]*1e3:.5f} mm), "
              f"{g['probe_n_live_edges']}/{g['probe_n_edges']} live edges, wall planes "
              f"on its column {g['probe_column_wall_planes']}; offset from the patch "
              f"centre ({off[0]*1e3:+.4f}, {off[1]*1e3:+.4f}) mm (declared "
              f"{FEED_OFFSET_X*1e3:+.5f}, 0)")


# ------------------------------------------------------------------- solve
def run_rung(dx: float, fr: Frame = FRAME) -> dict:
    """Preflight, solve, and hand back |S11| with its two witnesses.

    The S11 extraction is ``Simulation.run(compute_s_params=True)`` on the one
    wire port ``add_port(component="ez", extent=H_SUB)`` — the retired script's
    feed.  A wire port is a lumped 50 Ω port whose gap spans the probe from the
    ground sheet to the patch sheet, which is what openEMS's ``AddLumpedPort``
    spans in the record; for a single wire port ``run()`` reads the whole-gap
    voltage and the port current from the same record
    (``rfx/runners/uniform.py``, the single-wire path through
    ``driven_port_reflection``), the V/I definition the record's ``CalcPort``
    uses.  ``compute_s_matrix()`` has no lumped/wire lane by design
    (``rfx/sparams/dispatch.py`` raises and points at ``run()``).
    """
    import jax.numpy as jnp

    sim = build(dx, fr)
    g = assert_realized(sim, dx, fr)
    print(f"  box {fr.domain[0]*1e3:.3f} × {fr.domain[1]*1e3:.3f} × "
          f"{fr.domain[2]*1e3:.3f} mm (declared domain, "
          + (fr.label if fr.label else f"{fr.air_h}·h of air added")
          + f"; {cpml_layers(dx)} CPML cells = {cpml_layers(dx)*dx*1e3:.3f} mm "
          "outside it on every face)")
    _print_realized(g, dx)

    print(f"  preflight at dx = {dx*1e6:.3f} µm:")
    report = sim.preflight()
    print(f"    {len(report)} finding(s)")
    for i, line in enumerate(report):
        print(f"    [{i}] {line}")

    grid = sim._build_grid()
    n_steps = grid.num_timesteps(num_periods=NUM_PERIODS)
    t0 = time.perf_counter()
    res = sim.run(num_periods=NUM_PERIODS, compute_s_params=True,
                  s_param_freqs=jnp.asarray(FREQS_HZ))
    wall_s = time.perf_counter() - t0

    s = np.asarray(res.s_params)
    s11 = np.asarray(s[0, 0, :], dtype=complex)
    freqs = np.asarray(FREQS_HZ, dtype=float)
    settling = res.settling_db
    witness = res.settling_witness or {}
    mag = np.abs(s11)

    out = dict(dx_m=dx, freqs_hz=freqs, s11=s11, settling_db=settling,
               settling_witness=witness, wall_s=wall_s, n_steps=int(n_steps),
               dt_s=float(grid.dt), n_cells=g["n_cells"], grid_shape=g["grid_shape"],
               realized=g, n_preflight=len(report))

    print(f"  wall time {wall_s:.1f} s, {g['n_cells']} cells, {n_steps} steps "
          f"(dt {grid.dt*1e12:.4f} ps, record {n_steps*grid.dt*1e9:.3f} ns), "
          f"num_periods={NUM_PERIODS}, n_freqs={N_FREQS}")
    print(f"  ring-down settling: {settling} dB ({witness.get('status')}, "
          f"worst record {witness.get('worst_record')}, source-dominated "
          f"{witness.get('source_dominated')})")
    print(f"  max |S11| over {SWEEP_HZ[0]/1e9:.1f}–{SWEEP_HZ[1]/1e9:.1f} GHz: "
          f"{float(mag.max()):.5f} at {freqs[int(np.argmax(mag))]/1e9:.4f} GHz")

    if settling is None:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the result carries no settling witness "
            f"({witness.get('reason')}); the ring-down cannot be judged and no S "
            "value may be quoted.")
    if witness.get("source_dominated"):
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the settling record is source-dominated "
            f"({witness.get('qualifier')}); it measures the drive turning off, not "
            "the ring-down.")
    if float(settling) > SETTLING_DB:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: ring-down settling {float(settling):.2f} dB is "
            f"above {SETTLING_DB:.0f} dB. The record ended before the patch rang "
            "down, so this rung's S11 is truncation-suspect. Raise NUM_PERIODS; do "
            "not compare it.")
    excess = float(mag.max()) - 1.0
    if excess > PASSIVITY_EXCESS_BAR:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: |S11| reaches {float(mag.max()):.5f} at "
            f"{freqs[int(np.argmax(mag))]/1e9:.4f} GHz, {excess:.4f} above one (bar "
            f"{PASSIVITY_EXCESS_BAR}). A passive one-port cannot reflect more power "
            "than it receives, so this is a measurement artefact; do not compare "
            "this rung.")
    return out


# --------------------------------------------------------------- estimator
def _load_spectral_features():
    """The repository's one estimator module,
    ``validation/crossval/comparators/spectral_features.py`` — the module that
    exists so the cases and the reference producers cannot drift apart.  Loaded
    by path because ``validation/`` is not a package; it imports numpy only."""
    import importlib.util

    path = _REPO_ROOT / "validation" / "crossval" / "comparators" / "spectral_features.py"
    spec = importlib.util.spec_from_file_location("_rt5880_spectral_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sf = _load_spectral_features()


def zin_from_s11(s11) -> np.ndarray:
    """The port's input impedance from its reflection against 50 Ω."""
    s = np.asarray(s11, dtype=complex)
    return PORT_IMPEDANCE_OHM * (1.0 + s) / (1.0 - s)


def resonance(freqs_hz, s11_mag, zin) -> dict:
    """The |S11| minimum inside ``RESONANCE_BAND_HZ`` and what goes with it —
    the openEMS record's own rules (its maker's ``_one_port_features``).

    * frequency and depth: the shared ``refined_extremum`` on log|S11|, the
      deepest bin in the window and the vertex of the parabola through it and
      its two neighbours;
    * −10 dB band: the shared ``band_at_level``, both edges linearly
      interpolated in dB between the bracketing bins;
    * Zin at the resonance: Re and Im linearly interpolated between the
      bracketing bins at the refined frequency.
    """
    f = np.asarray(freqs_hz, dtype=float)
    mag = np.asarray(s11_mag, dtype=float)
    z = np.asarray(zin, dtype=complex)
    ext = _sf.refined_extremum(f, mag, RESONANCE_BAND_HZ[0], RESONANCE_BAND_HZ[1],
                               transform="log")
    band = _sf.band_at_level(f, mag, MINUS10_DB, ext["index"])
    fr = float(ext["refined_f"])
    out = {
        "index": int(ext["index"]),
        "bin_f": float(ext["bin_f"]),
        "f": fr,
        "sub_bin_shift": float(ext["sub_bin_shift"]),
        "depth_db": float(ext["depth_db"]),
        "bin_width": float(ext["bin_width"]),
        "zin": complex(float(np.interp(fr, f, z.real)), float(np.interp(fr, f, z.imag))),
    }
    if band is None:
        out.update(f_lo_10db=float("nan"), f_hi_10db=float("nan"), bw_10db=float("nan"))
    else:
        out.update(f_lo_10db=band[0], f_hi_10db=band[1], bw_10db=band[1] - band[0])
    return out


# ------------------------------------------------------------- references
def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _db(mag) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-300))


def _stage_curve(stage: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(freqs_hz, |S11|, Zin)`` of one openEMS rung, from its own arrays."""
    f = np.asarray(stage["freqs_ghz"], dtype=float) * 1e9
    mag = np.asarray(stage["s11_mag"], dtype=float)
    zin = (np.asarray(stage["zin_re_ohm"], dtype=float)
           + 1j * np.asarray(stage["zin_im_ohm"], dtype=float))
    return f, mag, zin


def _compare(label: str, rung: dict, stage: dict) -> dict:
    """The two judged quantities (decision (가)) and the reported ones.

    Judged, by ``input_resistance`` in ``tests/crossval/_v2_judging.py``: f0
    (the Re(Zin) peak, ``refined_remax``) and R at it, each located on its own
    curve by the same estimator over ``RESONANCE_BAND_HZ``.

    Reported: the |S11| minimum's frequency and depth and the −10 dB band
    (``resonance()``, the record's own dip rules), X(f0), and ΔdB of rfx −
    reference at every bin of the record where BOTH curves are above the
    deep-null level, taken twice by ``aligned_magnitude``: on the two axes as
    solved (the two grids are the same 901 bins, so nothing is interpolated)
    and with rfx's axis scaled by f0_ref / f0_rfx.  Rule R-b (PI 2026-09-24)
    judges the magnitude after aligning the JUDGED feature; the judged
    feature is f0 now, so the alignment is on f0, but R-b's 2 dB judgement
    does not apply to this case: decision (가) reports the magnitude and
    judges nothing on it.

    The deep-null exclusion is two-sided on purpose: a bin where one curve is
    in its dip and the other is not is the two dips sitting at different
    frequencies; counting it as a magnitude error would score one offset
    twice."""
    ref_f, ref_mag, ref_zin = _stage_curve(stage)
    ours_f = np.asarray(rung["freqs_hz"], dtype=float)
    assert ours_f.shape == ref_f.shape and np.allclose(ours_f, ref_f, rtol=0, atol=1.0), (
        "rfx was not sampled on the record's own frequency grid")
    ref_db = _db(ref_mag)
    ref_res = resonance(ref_f, ref_mag, ref_zin)
    our_res = rung["resonance"]
    zr = input_resistance(ref_f, ref_zin, ours_f, zin_from_s11(rung["s11"]),
                          RESONANCE_BAND_HZ)
    m = aligned_magnitude(ref_f, ref_db, ours_f, _db(np.abs(rung["s11"])),
                          zr["ref"]["f0_hz"], zr["ours"]["f0_hz"], SWEEP_HZ,
                          floor_db=DEEP_NULL_DB)
    solved, aligned = m["unaligned"], m["aligned"]
    return dict(
        resistance=zr, f0_pct=zr["f0_pct"],
        label=label, f_hz=ref_f, ref_db=ref_db, ours_db=solved["ours_db"],
        delta_db=solved["delta_db"], compared=solved["compared"],
        n_bins=int(ref_f.size), n_compared=solved["n_compared"],
        max_abs_delta_db=solved["max_abs_delta_db"],
        worst_index=solved["worst_index"],
        scale=m["scale"],
        aligned_ours_db=aligned["ours_db"], aligned_delta_db=aligned["delta_db"],
        aligned_compared=aligned["compared"],
        aligned_n_compared=aligned["n_compared"],
        aligned_max_abs_delta_db=aligned["max_abs_delta_db"],
        aligned_worst_index=aligned["worst_index"],
        aligned_worst_f_ghz=aligned["f_at_max_hz"] / 1e9,
        ref_res=ref_res, our_res=our_res,
        res_pct=100.0 * (our_res["f"] - ref_res["f"]) / ref_res["f"],
    )


_MAX_TABLE_ROWS = 90


# Why the magnitude features are reported and not judged (decision (가),
# PI 2026-09-24); printed beside them with the two reactances filled in.
REPORTED_REASON = (
    "neither solver declares a probe radius, so each solver's feed cell sets "
    "the probe's series reactance; the input reactance at resonance differs "
    "(rfx {x_ours:+.2f} Ω vs openEMS {x_ref:+.2f} Ω), which sets the dip depth "
    "and width")


def _print_comparison(c: dict) -> None:
    r, o, z = c["ref_res"], c["our_res"], c["resistance"]
    zo, zf = z["ours"], z["ref"]
    print(f"  --- {c['label']} ---")
    print(f"    resonance f0 = the Re(Zin) peak (judged, bar {FREQ_BAR*100:.0f} %): "
          f"rfx {zo['f0_hz']/1e9:.6f} GHz vs reference {zf['f0_hz']/1e9:.6f} GHz -> "
          f"{z['f0_pct']:+.3f} % — both by refined_remax (largest Re(Zin) bin in "
          f"{RESONANCE_BAND_HZ[0]/1e9:.4f}–{RESONANCE_BAND_HZ[1]/1e9:.4f} GHz, "
          f"parabola vertex; sub-bin shifts {zo['sub_bin_shift']:+.3f} / "
          f"{zf['sub_bin_shift']:+.3f})")
    print(f"    input resistance R(f0) (judged, bar {z['bar']*100:g} %): rfx "
          f"{z['r_ours_ohm']:.3f} Ω vs reference {z['r_ref_ohm']:.3f} Ω -> "
          f"{100.0*(z['r_ours_ohm']-z['r_ref_ohm'])/z['r_ref_ohm']:+.3f} % "
          f"(|Δ| {z['rel']*100:.3f} %)")
    print("    REPORTED, not judged — "
          + REPORTED_REASON.format(x_ours=z["x_ours_ohm"], x_ref=z["x_ref_ohm"]) + ":")
    print(f"    X(f0) (reported): rfx {z['x_ours_ohm']:+.2f} Ω vs {z['x_ref_ohm']:+.2f} Ω")
    print(f"    |S11| minimum (reported): rfx {o['f']/1e9:.5f} GHz vs {r['f']/1e9:.5f} "
          f"GHz -> {c['res_pct']:+.3f} %; depth rfx {o['depth_db']:.2f} dB vs "
          f"{r['depth_db']:.2f} dB")
    print(f"    −10 dB band edges (reported): rfx {o['f_lo_10db']/1e9:.4f} – "
          f"{o['f_hi_10db']/1e9:.4f} GHz ({o['bw_10db']/1e6:.1f} MHz) vs "
          f"{r['f_lo_10db']/1e9:.4f} – {r['f_hi_10db']/1e9:.4f} GHz "
          f"({r['bw_10db']/1e6:.1f} MHz)")
    print(f"    |S11| as solved (reported): dB compared at {c['n_compared']} of the record's {c['n_bins']} "
          f"bins, {SWEEP_HZ[0]/1e9:.1f}–{SWEEP_HZ[1]/1e9:.1f} GHz (both curves above "
          f"{DEEP_NULL_DB:.0f} dB); max |ΔdB| = {c['max_abs_delta_db']:.3f} dB"
          + (f" at {c['f_hz'][c['worst_index']]/1e9:.4f} GHz"
             if c["worst_index"] >= 0 else ""))
    print(f"    |S11| aligned (reported; rfx frequency axis × {c['scale']:.6f}, so "
          f"the two f0 coincide): dB compared at {c['aligned_n_compared']} of the "
          f"{c['n_bins']}; max |ΔdB| = {c['aligned_max_abs_delta_db']:.3f} dB at "
          f"{c['aligned_worst_f_ghz']:.4f} GHz")
    idx = np.arange(c["n_bins"])
    stride = max(1, int(np.ceil(idx.size / _MAX_TABLE_ROWS)))
    shown = set(idx[::stride].tolist())
    for worst in (c["worst_index"], c["aligned_worst_index"]):
        if worst >= 0:
            shown.add(worst)
    shown.update((r["index"], o["index"], zf["index"], zo["index"]))
    print(f"    (table printed every {stride} bins, plus the two |S11| minimum bins, "
          "the two Re(Zin) peak bins and "
          "the worst compared bin marked * as solved and ^ aligned, # both)")
    print("    f_GHz, ref_dB, rfx_dB, delta_dB, compared, "
          "rfx_aligned_dB, delta_aligned_dB, compared_aligned")
    for i in sorted(shown):
        worst_solved = i == c["worst_index"]
        worst_aligned = i == c["aligned_worst_index"]
        mark = ("#" if worst_solved and worst_aligned else
                "*" if worst_solved else "^" if worst_aligned else " ")
        print(f"     {mark}{c['f_hz'][i]/1e9:8.4f} {c['ref_db'][i]:10.3f} "
              f"{c['ours_db'][i]:10.3f} {c['delta_db'][i]:9.3f}  "
              f"{bool(c['compared'][i])!s:5} "
              f"{c['aligned_ours_db'][i]:10.3f} {c['aligned_delta_db'][i]:9.3f}  "
              f"{bool(c['aligned_compared'][i])}")


# ------------------------------------------------------------------ ladder
def _rungs() -> tuple[float, ...]:
    raw = os.environ.get("RFX_RT5880_PATCH_RUNGS", "").strip()
    if not raw:
        return LADDER_M
    return tuple(float(v) for v in raw.split(",") if v.strip())


def _reference_box_run(base: dict, dx: float, rec: dict) -> dict:
    """Solve the same board at ``dx`` in the openEMS record's own box
    (``reference_box_frame``) and report how far its resonance and |S11| move
    against the ladder box's run at the same ``dx``.

    A witness on rfx's own record, not a comparison with the reference: rfx's
    ladder solves this board in a smaller box than the openEMS record does
    (see the box constants), and this measures, on the ladder's own code, what
    solving it in the record's box instead does to the observable.  REPORTED,
    not judged — no bar has been set for it.
    """
    fr = reference_box_frame(dx, rec)
    print(f"\n  --- the openEMS record's box at dx = {dx*1e6:.3f} µm ---")
    print(f"  the same board in {fr.label}; box {fr.domain[0]*1e3:.3f} × "
          f"{fr.domain[1]*1e3:.3f} × {fr.domain[2]*1e3:.3f} mm instead of the "
          f"ladder's {DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × "
          f"{DOMAIN_Z_M*1e3:.3f} mm; the absorber depth unchanged")
    other = run_rung(dx, fr)
    other_res = resonance(other["freqs_hz"], np.abs(other["s11"]), zin_from_s11(other["s11"]))
    other_rm = refined_remax(other["freqs_hz"], zin_from_s11(other["s11"]),
                             *RESONANCE_BAND_HZ)
    base_rm = base["remax"]
    f = np.asarray(base["freqs_hz"], dtype=float)
    a = _db(np.abs(base["s11"]))
    b = _db(np.abs(other["s11"]))
    keep = (a >= DEEP_NULL_DB) & (b >= DEEP_NULL_DB)
    d = b - a
    worst = int(np.argmax(np.where(keep, np.abs(d), -np.inf)))
    out = {
        "dx_m": dx, "frame": fr, "s11": other["s11"],
        "f0_hz": other_rm["f0_hz"], "r_ohm": other_rm["r_ohm"], "x_ohm": other_rm["x_ohm"],
        "f0_shift_pct": 100.0 * (other_rm["f0_hz"] - base_rm["f0_hz"]) / base_rm["f0_hz"],
        "r_shift_pct": 100.0 * (other_rm["r_ohm"] - base_rm["r_ohm"]) / base_rm["r_ohm"],
        "resonance_hz": other_res["f"], "depth_db": other_res["depth_db"],
        "zin": other_res["zin"], "bw_10db_hz": other_res["bw_10db"],
        "resonance_shift_pct": 100.0 * (other_res["f"] - base["resonance"]["f"])
                               / base["resonance"]["f"],
        "max_abs_delta_db": float(np.max(np.abs(d[keep]))), "n_compared": int(keep.sum()),
        "worst_f_hz": float(f[worst]), "n_cells": other["n_cells"],
        "wall_s": other["wall_s"], "settling_db": other["settling_db"],
    }
    print(f"  f0 (the Re(Zin) peak) {other_rm['f0_hz']/1e9:.6f} GHz, R "
          f"{other_rm['r_ohm']:.3f} Ω, X {other_rm['x_ohm']:+.3f} Ω in the record's box "
          f"vs {base_rm['f0_hz']/1e9:.6f} GHz, {base_rm['r_ohm']:.3f} Ω, "
          f"{base_rm['x_ohm']:+.3f} Ω in the ladder's: f0 {out['f0_shift_pct']:+.3f} %, "
          f"R {out['r_shift_pct']:+.3f} %")
    print(f"  |S11| minimum {other_res['f']/1e9:.6f} GHz ({other_res['depth_db']:.2f} dB, "
          f"−10 dB band {other_res['bw_10db']/1e6:.2f} MHz, Zin "
          f"{other_res['zin'].real:.2f}{other_res['zin'].imag:+.2f}j Ω) in the "
          f"record's box vs {base['resonance']['f']/1e9:.6f} GHz "
          f"({base['resonance']['depth_db']:.2f} dB) in the ladder's: "
          f"{out['resonance_shift_pct']:+.3f} %")
    print(f"  |S11|: max |Δ| = {out['max_abs_delta_db']:.3f} dB over "
          f"{out['n_compared']} of {f.size} bins where both curves are above "
          f"{DEEP_NULL_DB:.0f} dB; worst at {out['worst_f_hz']/1e9:.4f} GHz")
    print("    f_GHz, record's box |S11| dB, ladder box |S11| dB, Δ")
    for i in range(0, f.size, 10):
        print(f"      {f[i]/1e9:8.4f} {b[i]:10.3f} {a[i]:10.3f} {d[i]:8.3f}")
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_rt5880_patch_matches_the_openems_reference(tmp_path):
    """The mesh ladder, the convergence statement, then the comparison."""
    rungs = _rungs()
    print(f"\nThe RT/Duroid 5880 probe-fed patch — ladder "
          f"{[f'{d*1e6:.3f}µm' for d in rungs]}")
    print(f"  board: {L_PATCH*1e3:.1f} × {W_PATCH*1e3:.1f} mm patch on "
          f"{H_SUB*1e3:.3f} mm εr {EPS_R} tanδ {TAN_DELTA} (σ {SIGMA_SUB:.5e} S/m) over a "
          f"{GP_X*1e3:.0f} × {GP_Y*1e3:.0f} mm ground; 50 Ω probe "
          f"{FEED_OFFSET_X*1e3:+.5f} mm from the patch centre along x; box "
          f"{DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × {DOMAIN_Z_M*1e3:.3f} mm "
          f"(ground edges {X_CLEARANCE_M*1e3:.3f} / {Y_CLEARANCE_M*1e3:.3f} mm clear "
          f"in x / y, {BELOW_GROUND_M*1e3:.3f} mm below the ground, "
          f"{ABOVE_PATCH_M*1e3:.3f} mm above the patch), CPML "
          f"{CPML_THICKNESS_H*H_SUB*1e3:.3f} mm deep outside it")

    openems = _load(_OPENEMS_JSON)
    judged_stage = openems[OPENEMS_JUDGED_STAGE]
    stages = openems["meta"]["stages"]

    results = []
    for dx in rungs:
        print(f"\n=== rung dx = {dx*1e6:.3f} µm ===")
        r = run_rung(dx)
        zin = zin_from_s11(r["s11"])
        r["resonance"] = res = resonance(r["freqs_hz"], np.abs(r["s11"]), zin)
        r["remax"] = rm = refined_remax(r["freqs_hz"], zin, *RESONANCE_BAND_HZ)
        r["remax_half_grid"] = hg = remax_half_grid_witness(r["freqs_hz"], zin,
                                                            *RESONANCE_BAND_HZ)
        r["s11_db"] = _db(np.abs(r["s11"]))
        print(f"  resonance f0 (the Re(Zin) peak) {rm['f0_hz']/1e9:.6f} GHz (bin "
              f"{rm['bin_f_hz']/1e9:.5f}, sub-bin shift {rm['sub_bin_shift']:+.3f}), "
              f"R {rm['r_ohm']:.3f} Ω, X {rm['x_ohm']:+.3f} Ω, flags {rm['flags']}; "
              f"half-grid estimates {[round(v/1e9, 6) for v in hg['f0_even_odd_hz']]} GHz "
              f"({hg['spread_bins']:.3f} bin apart, reported)")
        print(f"  |S11| minimum (reported) {res['f']/1e9:.6f} GHz (bin "
              f"{res['bin_f']/1e9:.5f}, sub-bin shift {res['sub_bin_shift']:+.3f}), "
              f"depth {res['depth_db']:.3f} dB, −10 dB band {res['bw_10db']/1e6:.2f} MHz "
              f"({res['f_lo_10db']/1e9:.5f} – {res['f_hi_10db']/1e9:.5f} GHz), Zin "
              f"there {res['zin'].real:.2f}{res['zin'].imag:+.2f}j Ω")
        print("  |S11| on the record's grid — f_GHz, |S11| dB, Re Zin, Im Zin:")
        for f, a, z in zip(r["freqs_hz"], r["s11_db"], zin):
            print(f"    {f/1e9:8.4f} {a:10.4f} {z.real:10.3f} {z.imag:10.3f}")
        results.append(r)
        # The board in the openEMS record's own box, on the coarsest rung:
        # what the ladder's smaller box does to the observable, reported.
        if len(results) == 1:
            ref_box = _reference_box_run(r, dx, openems)

    # ---- (a) mesh statement: f0 (the Re(Zin) peak) must settle along the ladder
    resonances = [r["remax"]["f0_hz"] for r in results]
    print("\n  mesh statement — f0 (the Re(Zin) peak) along the rfx ladder; the "
          "|S11| minimum beside it, reported:")
    for r in results:
        print(f"    dx {r['dx_m']*1e6:8.3f} µm  f0 {r['remax']['f0_hz']/1e9:.6f} GHz  "
              f"R {r['remax']['r_ohm']:7.3f} Ω  X {r['remax']['x_ohm']:+7.3f} Ω  "
              f"|S11| min {r['resonance']['f']/1e9:.6f} GHz "
              f"{r['resonance']['depth_db']:7.2f} dB  {r['n_cells']} cells  "
              f"{r['n_steps']} steps  {r['wall_s']:.1f} s")
    mesh_stmt = mesh_statement(resonances, [f"{r['dx_m']*1e6:.3f}µm" for r in results])
    if len(resonances) >= 2:
        for (a, b), (ra, rb) in zip(zip(resonances, resonances[1:]),
                                    zip(results, results[1:])):
            print(f"    {ra['dx_m']*1e6:.3f} -> {rb['dx_m']*1e6:.3f} µm: "
                  f"{100.0*(b-a)/a:+.3f} %")
        print(f"    last two rungs differ by {mesh_stmt['last_two_pct']:.3f} % "
              f"(bar {LADDER_AGREEMENT*100:.0f} %)")
        print("    steps " + ", ".join(
            f"{p:+.3f} %" + (" (flat)" if fl else "")
            for p, fl in zip(mesh_stmt["steps_pct"], mesh_stmt["flat"]))
            + f" (flat when within {FLAT_STEP*100:.1f} %); monotone "
            f"{mesh_stmt['monotone']} -> {mesh_stmt['verdict']}")

    # R(f0) along the same ladder, its own mesh statement: the R-a structure
    # scaled to R's bar -- a step within RESISTANCE_BAR / 10 is flat, the last
    # two rungs within RESISTANCE_BAR -- as the frequency's is scaled to
    # FREQ_BAR.  What the frequency's own parameters would say about R is
    # printed beside it and not used.
    r_values = [r["remax"]["r_ohm"] for r in results]
    rung_labels = [f"{r['dx_m']*1e6:.3f}µm" for r in results]
    r_stmt = mesh_statement(r_values, rung_labels, flat_step=RESISTANCE_BAR / 10.0,
                            agreement=RESISTANCE_BAR, unit=("Ω", 1.0, 3))
    r_stmt_freq_bars = mesh_statement(r_values, rung_labels, unit=("Ω", 1.0, 3))
    print("\n  mesh statement — R(f0) along the rfx ladder (R's bars: flat within "
          f"{RESISTANCE_BAR/10*100:.1f} %, last two within {RESISTANCE_BAR*100:.0f} %):")
    print(f"    {r_stmt['verdict']} — {r_stmt['reason']}")
    print(f"    with the frequency's bars ({FLAT_STEP*100:.1f} % / "
          f"{LADDER_AGREEMENT*100:.0f} %), reported only: {r_stmt_freq_bars['verdict']} "
          f"— {r_stmt_freq_bars['reason']}")

    # The reference's own mesh statement, read off its record.  Its edge
    # refinement is not a uniform √2 ladder, so its steps are printed, never
    # extrapolated.
    print("\n  the openEMS record's own mesh statement:")
    ref_res = []
    for name in OPENEMS_MESH_STATEMENT_STAGES + (OPENEMS_JUDGED_STAGE,):
        st = openems[name]
        ms = stages[name]
        f, mag, zin = _stage_curve(st)
        rr = resonance(f, mag, zin)
        rx = refined_remax(f, zin, *RESONANCE_BAND_HZ)
        ref_res.append(rx["f0_hz"])
        print(f"    {name:15s} factor {ms['resolution_factor']:.4f}  "
              f"{ms['mesh_realized']['substrate_z_cells_realized']} substrate cells  "
              f"{ms['mesh_realized']['n_cells']:9d} cells  f0 {rx['f0_hz']/1e9:.6f} GHz "
              f"R {rx['r_ohm']:.3f} Ω X {rx['x_ohm']:+.3f} Ω;  |S11| minimum "
              f"{rr['f']/1e9:.6f} GHz  {rr['depth_db']:7.2f} dB  −10 dB band "
              f"{rr['bw_10db']/1e6:.1f} MHz  Zin {rr['zin'].real:.2f}"
              f"{rr['zin'].imag:+.2f}j Ω  end energy "
              f"{ms['solver_run']['final_energy_db']:.2f} dB")
    print(f"    f0 coarse -> mid {100.0*(ref_res[1]-ref_res[0])/ref_res[0]:+.3f} %, "
          f"mid -> fine {100.0*(ref_res[2]-ref_res[1])/ref_res[1]:+.3f} %")

    # A partial ladder shows no convergence trend, so it states no verdict: the
    # distances below are still printed — they are the diagnostic the env
    # override exists for — and the test skips at the end instead of passing or
    # failing.  Any rung set that is not the ladder itself, in order, is partial.
    partial = tuple(rungs) != LADDER_M

    # ---- (b) the comparison: finest rung against the openEMS fine rung ------
    # Every distance and the figure are printed BEFORE any verdict, so a run
    # that fails the mesh statement still leaves the whole curve in the log.
    finest = results[-1]
    print("\n  openEMS (judged)")
    fine = _compare(
        f"openEMS {OPENEMS_JUDGED_STAGE} ({stages[OPENEMS_JUDGED_STAGE]['mesh_realized']['substrate_z_cells_realized']}"
        " substrate cells) — JUDGED", finest, judged_stage)
    _print_comparison(fine)
    for name in OPENEMS_MESH_STATEMENT_STAGES:
        _print_comparison(_compare(
            f"openEMS {name} ({stages[name]['mesh_realized']['substrate_z_cells_realized']}"
            " substrate cells) — the reference's own mesh statement, reported",
            finest, openems[name]))

    # ---- (c) the figure ----------------------------------------------------
    fig_path = _write_figure(results, openems, tmp_path, ref_box)
    print(f"\n  figure: {fig_path}")
    print(f"  the openEMS record's box at {ref_box['dx_m']*1e6:.3f} µm: f0 "
          f"{ref_box['f0_hz']/1e9:.6f} GHz, {ref_box['f0_shift_pct']:+.3f} %, R "
          f"{ref_box['r_ohm']:.3f} Ω ({ref_box['r_shift_pct']:+.3f} %); |S11| minimum "
          f"{ref_box['resonance_hz']/1e9:.6f} GHz, "
          f"{ref_box['resonance_shift_pct']:+.3f} % against the ladder box at the "
          f"same cell; |S11| up to {ref_box['max_abs_delta_db']:.3f} dB apart "
          f"({ref_box['n_cells']} cells, {ref_box['wall_s']:.1f} s) — reported, "
          "not judged")

    if partial:
        pytest.skip(
            "RFX_RT5880_PATCH_RUNGS restricted the ladder to "
            f"{[f'{d*1e6:.3f}µm' for d in rungs]}; f0 (the Re(Zin) peak) reads "
            f"{[f'{f/1e9:.6f} GHz' for f in resonances]}. A partial ladder shows no "
            "convergence trend, so the distances above are a diagnostic and this "
            "run states no verdict.")

    # ---- (a) the mesh statements come first: without them nothing is judged -
    # f0 not shown to converge — a step larger than FLAT_STEP against the trend
    # (PI 2026-09-24), or the last two rungs LADDER_AGREEMENT or more apart
    # (PI 2026-09-22) — states the numbers and xfails; it is not a failure.
    # f0 converged but R not: f0 is judged, then R's statement xfails. Both
    # converged: both are judged. Every witness above has already hard-failed
    # a bad rung.
    if mesh_stmt["verdict"] != CONVERGED:
        # PI decisions 2026-09-22 and 2026-09-24, carried over from the MSL
        # notch filter case: the v2 bar is never loosened, and a case that
        # cannot meet it says so with its numbers. The xfail is IMPERATIVE, so
        # every witness above — the ring-down, the passivity excess, the
        # realized board — still FAILS the test; only the two judged bars
        # below (resonance, input resistance) are held back. When the mesh converges this branch is no
        # longer taken and the comparison judges for real.
        pytest.xfail(
            "known limitation (PI 2026-09-22, 2026-09-24): the uniform mesh is "
            f"not shown to converge — {mesh_stmt['reason']}. The finest rung's f0 "
            f"sits {fine['f0_pct']:+.3f} % from the openEMS record's "
            f"{fine['resistance']['ref']['f0_hz']/1e9:.6f} GHz and its input "
            f"resistance there {fine['resistance']['rel']*100:.3f} % from the record's "
            f"({fine['resistance']['r_ours_ohm']:.3f} vs "
            f"{fine['resistance']['r_ref_ohm']:.3f} Ω). The comparison above is "
            "reported, not judged.")
    zr = fine["resistance"]
    assert abs(fine["f0_pct"]) < FREQ_BAR * 100.0, (
        f"the resonance f0 (the Re(Zin) peak) sits {fine['f0_pct']:+.3f} % from the "
        f"openEMS record's {zr['ref']['f0_hz']/1e9:.6f} GHz (bar "
        f"{FREQ_BAR*100:.0f} %); rfx reads {zr['ours']['f0_hz']/1e9:.6f} GHz on the "
        f"{finest['dx_m']*1e6:.3f} µm mesh.")
    if r_stmt["verdict"] != CONVERGED:
        # R's ladder, not f0's, is what is not shown to converge: f0 has just
        # been judged; R is held back, imperatively, with its numbers.
        pytest.xfail(
            "known limitation (PI 2026-09-24): f0 converges and is judged "
            f"({fine['f0_pct']:+.3f} %, bar {FREQ_BAR*100:.0f} %), but the input "
            f"resistance R(f0) is not shown to converge along the ladder — "
            f"{r_stmt['reason']}. The finest rung's R is {zr['r_ours_ohm']:.3f} Ω "
            f"against the record's {zr['r_ref_ohm']:.3f} Ω ({zr['rel']*100:.3f} %), "
            "reported, not judged.")
    # The input resistance at f0, each curve read at its own Re(Zin) peak
    # (decision (가), PI 2026-09-24).
    assert zr["passed"], (
        f"the input resistance at f0 differs by {zr['rel']*100:.3f} % "
        f"(bar {zr['bar']*100:g} %): rfx R {zr['r_ours_ohm']:.3f} Ω at "
        f"{zr['ours']['f0_hz']/1e9:.6f} GHz vs the openEMS record's "
        f"{zr['r_ref_ohm']:.3f} Ω at {zr['ref']['f0_hz']/1e9:.6f} GHz, each at its "
        "own Re(Zin) peak.")
    reason = REPORTED_REASON.format(x_ours=zr["x_ours_ohm"], x_ref=zr["x_ref_ohm"])
    print(f"\n  judged: f0 (the Re(Zin) peak) {fine['f0_pct']:+.3f} % from the openEMS "
          f"record (bar {FREQ_BAR*100:.0f} %); R(f0) {zr['r_ours_ohm']:.3f} vs "
          f"{zr['r_ref_ohm']:.3f} Ω, {zr['rel']*100:.3f} % "
          f"(bar {RESISTANCE_BAR*100:.0f} %); both ladders converged")
    print(f"  reported, not judged — {reason}: X(f0) {zr['x_ours_ohm']:+.2f} vs "
          f"{zr['x_ref_ohm']:+.2f} Ω; |S11| minimum {fine['res_pct']:+.3f} % "
          f"({fine['our_res']['f']/1e9:.5f} vs {fine['ref_res']['f']/1e9:.5f} GHz), "
          f"depth {fine['our_res']['depth_db']:.2f} vs "
          f"{fine['ref_res']['depth_db']:.2f} dB; −10 dB band "
          f"{fine['our_res']['f_lo_10db']/1e9:.4f}–{fine['our_res']['f_hi_10db']/1e9:.4f}"
          f" vs {fine['ref_res']['f_lo_10db']/1e9:.4f}–"
          f"{fine['ref_res']['f_hi_10db']/1e9:.4f} GHz; |S11| aligned max |ΔdB| "
          f"{fine['aligned_max_abs_delta_db']:.3f} dB, as solved "
          f"{fine['max_abs_delta_db']:.3f} dB")


def _write_figure(results, openems, tmp_path, ref_box=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.environ.get("RFX_CROSSVAL_FIG_DIR")
    directory = Path(out_dir) if out_dir else Path(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rt5880_patch.png"

    stages = openems["meta"]["stages"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for r in results:
        ax.plot(np.asarray(r["freqs_hz"]) / 1e9, r["s11_db"], "--", lw=1.3, zorder=3,
                label=f"rfx dx {r['dx_m']*1e6:.2f} µm")
    if ref_box is not None:
        ax.plot(np.asarray(results[0]["freqs_hz"]) / 1e9, _db(np.abs(ref_box["s11"])), ":",
                lw=1.5, zorder=3, color="tab:purple",
                label=f"rfx dx {ref_box['dx_m']*1e6:.2f} µm in the openEMS record's box "
                      "(reported)")
    # The mesh-statement rungs first, JUDGED on top, so the curve being judged
    # against is the one the eye reads where they overlap.
    for name, alpha in zip(OPENEMS_MESH_STATEMENT_STAGES, (0.35, 0.55)):
        ax.plot(openems[name]["freqs_ghz"], _db(openems[name]["s11_mag"]), "-",
                lw=2.2, alpha=alpha, color="tab:green",
                label=f"openEMS {name} "
                      f"({stages[name]['mesh_realized']['substrate_z_cells_realized']} "
                      "substrate cells)")
    ax.plot(openems[OPENEMS_JUDGED_STAGE]["freqs_ghz"],
            _db(openems[OPENEMS_JUDGED_STAGE]["s11_mag"]), "-", lw=1.0, color="k",
            label=f"openEMS {OPENEMS_JUDGED_STAGE} "
                  f"({stages[OPENEMS_JUDGED_STAGE]['mesh_realized']['substrate_z_cells_realized']}"
                  " substrate cells) — JUDGED")
    ax.axhline(DEEP_NULL_DB, color="0.6", lw=0.6, ls=":")
    ax.set_xlim(SWEEP_HZ[0] / 1e9, SWEEP_HZ[1] / 1e9)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# -------------------------------------------------------------- fast tests
# The candidate rungs: every h/n from h/4 to h/12.  Each is built in
# ``BUILD_CHECK_FRAME`` and must be refused unless it is a rung of the ladder.
CANDIDATE_RUNGS_N = tuple(range(4, 13))
# The ladder rungs the fast test also builds in the ladder's own box: h/4 and
# h/8.  There the h/12 board is 52.7 M cells, too large a build for the PR
# lane's runner; the ladder's own ``assert_realized`` checks it before it is
# solved.
FAST_LADDER_M = LADDER_M[:2]
_SAME_BOARD_KEYS = (("patch", "x_node_span_m"), ("patch", "y_node_span_m"),
                    ("ground", "x_node_span_m"), ("ground", "y_node_span_m"),
                    ("substrate_x_cells",), ("substrate_y_cells",),
                    ("probe_offset_from_patch_centre_m",))


def _board_of(g: dict) -> tuple:
    out = []
    for key in _SAME_BOARD_KEYS:
        v = g
        for k in key:
            v = v[k]
        out.extend(v if isinstance(v, tuple) else (v,))
    return tuple(float(v) for v in out)


def test_the_lattice_builds_the_declared_board_on_every_rung():
    """Build-time only: every rung of the ladder realizes the declared board,
    and every other h/n from h/4 to h/12 is refused for realizing another one.
    All candidates are built in the build-only box; h/4 and h/8 are also built
    in the ladder's box, and the two boxes must realize the same board there."""
    fr = BUILD_CHECK_FRAME
    print(f"\n  declared board: patch {L_PATCH*1e3:.2f} × {W_PATCH*1e3:.2f} mm centred "
          f"at ({PATCH_CENTRE_X_M*1e3:.3f}, {PATCH_CENTRE_Y_M*1e3:.3f}) mm, ground "
          f"{GP_X*1e3:.2f} × {GP_Y*1e3:.2f} mm on z = {GROUND_Z_M*1e3:.3f} mm, patch on "
          f"z = {PATCH_Z_M*1e3:.3f} mm, probe {FEED_OFFSET_X*1e3:+.5f} mm along x; box "
          f"{DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × {DOMAIN_Z_M*1e3:.3f} mm")
    accepted, built = [], {}
    for n in CANDIDATE_RUNGS_N:
        dx = H_SUB / n
        g = realized_geometry(build(dx, fr))
        built[n] = g
        try:
            check_realized(g, dx, fr)
        except AssertionError as exc:
            verdict = f"refused — {str(exc)[:110]}..."
        else:
            verdict = "accepted"
            accepted.append(n)
        print(f"  h/{n:<2d} {dx*1e6:8.3f} µm: patch node span "
              f"{g['patch']['x_node_span_m']*1e3:.4f} × "
              f"{g['patch']['y_node_span_m']*1e3:.4f} mm, ground "
              f"{g['ground']['x_node_span_m']*1e3:.4f} × "
              f"{g['ground']['y_node_span_m']*1e3:.4f} mm, substrate "
              f"{g['substrate_x_cells']} × {g['substrate_y_cells']} cells, probe "
              f"{g['probe_offset_from_patch_centre_m'][0]*1e3:+.4f} mm — {verdict}")
    ladder_n = tuple(int(round(H_SUB / d)) for d in LADDER_M)
    assert tuple(accepted) == ladder_n, (
        f"the rungs assert_realized accepts among h/{CANDIDATE_RUNGS_N[0]} … "
        f"h/{CANDIDATE_RUNGS_N[-1]} are {accepted}, not the ladder's {ladder_n}")

    for dx in FAST_LADDER_M:
        g = assert_realized(build(dx), dx)
        _print_realized(g, dx)
        n = int(round(H_SUB / dx))
        assert _board_of(g) == pytest.approx(_board_of(built[n]), abs=1e-12), (
            f"h/{n}: the ladder's box and the build-only box realize different "
            f"boards: {_board_of(g)} against {_board_of(built[n])}")


def test_a_board_off_the_common_nodes_is_refused(monkeypatch):
    """The case declared with the retired 50 mm width (``W_PATCH`` = 50 mm),
    built on the three rungs of the ladder in the build-only box.  Its radiating
    edges fall between the common nodes, so a finer rung covers other node rows
    than a coarser one, and the ladder refuses it before any rung is solved."""
    import sys
    monkeypatch.setattr(sys.modules[__name__], "W_PATCH", 50.0e-3)
    fr = BUILD_CHECK_FRAME
    verdicts = {}
    for dx in LADDER_M:
        n = int(round(H_SUB / dx))
        g = realized_geometry(build(dx, fr))
        try:
            check_realized(g, dx, fr)
        except AssertionError as exc:
            verdicts[n] = str(exc)
        else:
            verdicts[n] = "accepted"
        print(f"\n  W = 50 mm at h/{n}: patch node span "
              f"{g['patch']['x_node_span_m']*1e3:.4f} × "
              f"{g['patch']['y_node_span_m']*1e3:.4f} mm — {verdicts[n]}")
    refused = [n for n, v in verdicts.items() if v != "accepted"]
    assert refused, "a 50 mm wide patch passed every rung of the ladder"
    assert all("the patch's node span along y" in verdicts[n] for n in refused), verdicts


def test_assert_realized_catches_each_mutation():
    """Each defect ``assert_realized`` exists for, drawn on the coarsest rung
    through the builder's own helpers, is refused with the check that names it.
    A defect that survives the coarsest mesh survives every finer one."""
    dx = DX_COARSEST_M
    cases = (
        ("patch drawn as a one-cell volume", _build_patch_as_volume,
         "tangential wall planes along z"),
        ("ground drawn one node plane below the substrate floor",
         _build_ground_one_plane_low, "ground sheet's wall plane"),
        ("probe drawn one cell short of the patch", _build_feed_one_cell_short,
         "does not reach the patch sheet"),
        ("patch drawn one cell short along its resonant length",
         _build_patch_one_cell_short, "the board holds"),
        ("substrate drawn exactly on the ground's outermost nodes (the knife edge)",
         _build_substrate_on_the_knife_edge, "the substrate's cells along"),
        ("a PEC wire drawn along the probe, ground to patch",
         _build_probe_shorted_by_a_wire, "is a PEC edge"),
        ("ground and substrate drawn one cell to +x of the patch",
         _build_ground_and_substrate_shifted_one_cell, "the ground centred under the patch"),
        ("substrate alone drawn one cell to +y",
         _build_substrate_shifted_one_cell, "a contiguous laminate centred under the patch"),
    )
    for label, builder, expected in cases:
        with pytest.raises(AssertionError) as caught:
            assert_realized(builder(dx), dx)
        print(f"\n  {label}: {caught.value}")
        assert expected in str(caught.value), (label, str(caught.value))


def _record_board(rec: dict, stage: str) -> dict:
    """The board one Stage B rung of the openEMS record solved, in metres,
    siemens per metre and ohms, read from the record's own fields and from
    nothing in this file: the substrate from ``plan_estimate.stand_in``
    (``materials.sub`` epsilon and kappa, and the excitation's f0, at which the
    maker turned tanδ into kappa), the port from ``stand_in.ports``, the patch
    edges from ``geometry_realized.patch_edges``, the probe from
    ``geometry_realized.feed``, and the ground edges and the two metal planes
    from ``line_check``.  The record's frame has the ground centred at the
    origin; offsets are taken from the record's own patch centre."""
    st = rec["meta"]["stages"][stage]
    stand_in = st["plan_estimate"]["stand_in"]
    mm = stand_in["unit"]
    sub = stand_in["materials"]["sub"]
    (port,) = stand_in["ports"]
    edges = st["geometry_realized"]["patch_edges"]
    rows = {r["what"]: r for r in st["line_check"]["rows"]}
    f0 = float(stand_in["excite"][0])
    x_lo, x_hi = (edges[k]["declared_mm"] * mm for k in ("patch_x_lo", "patch_x_hi"))
    y_lo, y_hi = (edges[k]["declared_mm"] * mm for k in ("patch_y_lo", "patch_y_hi"))
    feed_x, feed_y = (v * mm for v in st["geometry_realized"]["feed"]["declared_xy_mm"])
    z_ground = rows["ground plane, z = 0"]["declared_mm"] * mm
    z_patch = rows["patch plane, z = h"]["declared_mm"] * mm
    return {
        "eps_r": float(sub["epsilon"]),
        "kappa": float(sub["kappa"]),
        "f0_hz": f0,
        "tan_delta": float(sub["kappa"]) / (2.0 * math.pi * f0 * EPS0 * float(sub["epsilon"])),
        "h": z_patch - z_ground,
        "l_patch": x_hi - x_lo,
        "w_patch": y_hi - y_lo,
        "gp_x": (rows["ground x+ edge"]["declared_mm"]
                 - rows["ground x- edge"]["declared_mm"]) * mm,
        "gp_y": (rows["ground y+ edge"]["declared_mm"]
                 - rows["ground y- edge"]["declared_mm"]) * mm,
        "feed_offset": (feed_x - 0.5 * (x_lo + x_hi), feed_y - 0.5 * (y_lo + y_hi)),
        "port_r": float(port["R"]),
        "port_direction": port["direction"],
        "port_xy": (port["start"][0] * mm, port["start"][1] * mm),
        "port_z": (port["start"][2] * mm, port["stop"][2] * mm),
        "z_planes": (z_ground, z_patch),
        "absorber_faces_mm": {a: (st["geometry_realized"]["pml"]["faces"][a]["inner_face_lo_mm"],
                                  st["geometry_realized"]["pml"]["faces"][a]["inner_face_hi_mm"])
                              for a in "xyz"},
    }


def test_a_probe_off_its_declared_node_is_refused():
    """The declared probe is a node of every rung, and the realized probe is
    held to it within 1 nm.  Drawn at the retired −9.0 mm on the h/8 rung the
    probe snaps to −9.12813 mm, and the rung is refused; every ladder rung
    realizes the declared −8.73125 mm exactly (checked by
    ``test_the_lattice_builds_the_declared_board_on_every_rung`` on h/4 and h/8
    and by the ladder itself on h/12)."""
    dx = LADDER_M[1]
    for k, d in enumerate(LADDER_M):
        cells = FEED_OFFSET_X / d
        assert abs(cells - round(cells)) < 1e-9, (d, cells)
        assert round(cells) == (-11, -22, -33)[k]
    with pytest.raises(AssertionError) as caught:
        assert_realized(_build_probe_at_the_retired_offset(dx), dx)
    print(f"\n  probe drawn at -9.0 mm at h/8: {caught.value}")
    assert "the probe realizes (-9.128125" in str(caught.value)


def test_the_board_is_the_openems_record_s_board():
    """The constants rfx builds this board from, held against the board the
    openEMS record solved, read from the record's own fields on every Stage B
    rung (``_record_board``): εr, tanδ (and the conductivity it becomes at the
    record's f0), the substrate thickness, the patch and the ground, the
    probe's offset from the patch centre and its span, and the port
    resistance.  Then the h/4 board rfx actually builds (``realized_geometry``)
    against the record directly, and the reference-box frame's faces against
    the record's absorber faces.  Without this test a constant could drift from
    the record and every other test here would still pass: they compare the
    realized board with this file's own constants."""
    rec = _load(_OPENEMS_JSON)
    for stage in OPENEMS_MESH_STATEMENT_STAGES + (OPENEMS_JUDGED_STAGE,):
        b = _record_board(rec, stage)
        print(f"\n  {stage}: εr {b['eps_r']}, kappa {b['kappa']:.6e} S/m at f0 "
              f"{b['f0_hz']/1e9:.2f} GHz (tanδ {b['tan_delta']:.6g}), h "
              f"{b['h']*1e3:.3f} mm, patch {b['l_patch']*1e3:.3f} × "
              f"{b['w_patch']*1e3:.3f} mm, ground {b['gp_x']*1e3:.3f} × "
              f"{b['gp_y']*1e3:.3f} mm, probe offset ({b['feed_offset'][0]*1e3:+.3f}, "
              f"{b['feed_offset'][1]*1e3:+.3f}) mm spanning z {b['port_z'][0]*1e3:.3f} "
              f"-> {b['port_z'][1]*1e3:.3f} mm, port {b['port_r']:.1f} Ω")
        assert EPS_R == b["eps_r"], (stage, EPS_R, b["eps_r"])
        assert TAN_DELTA == pytest.approx(b["tan_delta"], rel=1e-9), (stage, TAN_DELTA)
        assert SIGMA_SUB == pytest.approx(b["kappa"], rel=1e-12), (stage, SIGMA_SUB)
        assert H_SUB == pytest.approx(b["h"], abs=1e-12), (stage, H_SUB)
        assert L_PATCH == pytest.approx(b["l_patch"], abs=1e-12), (stage, L_PATCH)
        assert W_PATCH == pytest.approx(b["w_patch"], abs=1e-12), (stage, W_PATCH)
        assert GP_X == pytest.approx(b["gp_x"], abs=1e-12), (stage, GP_X)
        assert GP_Y == pytest.approx(b["gp_y"], abs=1e-12), (stage, GP_Y)
        assert FEED_OFFSET_X == pytest.approx(b["feed_offset"][0], abs=1e-12), (
            stage, FEED_OFFSET_X, b["feed_offset"])
        assert b["feed_offset"][1] == pytest.approx(0.0, abs=1e-12), stage
        assert b["port_xy"] == pytest.approx((b["feed_offset"][0], b["feed_offset"][1]),
                                             abs=1e-12), stage
        assert b["port_direction"] == "z"
        assert b["port_z"] == pytest.approx(b["z_planes"], abs=1e-12), (
            f"{stage}: the record's port spans z {b['port_z']}, its metal planes "
            f"are {b['z_planes']}")
        assert PORT_IMPEDANCE_OHM == b["port_r"], (stage, PORT_IMPEDANCE_OHM)

    # The board rfx builds, against the record rather than against the
    # constants: h/4, the coarsest rung, is enough — every rung realizes the
    # same board (``assert_realized``).
    b = _record_board(rec, OPENEMS_JUDGED_STAGE)
    dx = DX_COARSEST_M
    g = realized_geometry(build(dx))
    ox, oy = g["probe_offset_from_patch_centre_m"]
    print(f"  rfx at h/4: patch strip {g['patch']['x_strip_m']*1e3:.4f} × "
          f"{g['patch']['y_strip_m']*1e3:.4f} mm, ground strip "
          f"{g['ground']['x_strip_m']*1e3:.4f} × {g['ground']['y_strip_m']*1e3:.4f} mm, "
          f"probe ({ox*1e3:+.4f}, {oy*1e3:+.4f}) mm, {g['substrate_cells']} substrate "
          f"cells, port {g['probe_impedance_ohm']:.1f} Ω, permittivities "
          f"{g['distinct_eps_r']}")
    assert g["probe_impedance_ohm"] == b["port_r"]
    assert g["distinct_eps_r"] == sorted({1.0, round(b["eps_r"], 6)})
    assert g["substrate_cells"] * dx == pytest.approx(b["h"], abs=1e-12)
    for got, want in ((g["patch"]["x_strip_m"], b["l_patch"]),
                      (g["patch"]["y_strip_m"], b["w_patch"]),
                      (g["ground"]["x_strip_m"], b["gp_x"]),
                      (g["ground"]["y_strip_m"], b["gp_y"])):
        assert abs(got - want) <= dx, (got, want)
    assert abs(ox - b["feed_offset"][0]) <= 0.5 * dx + 1e-12, (ox, b["feed_offset"])
    assert abs(oy - b["feed_offset"][1]) <= 1e-12, (oy, b["feed_offset"])

    # The reference-box run's domain faces sit on the record's absorber inner
    # faces, each to within half a cell.
    fr = reference_box_frame(dx, rec)
    faces = b["absorber_faces_mm"]
    got_faces = {
        "x": (-fr.cx, fr.domain[0] - fr.cx),
        "y": (-fr.cy, fr.domain[1] - fr.cy),
        "z": (-fr.z_ground, fr.domain[2] - fr.z_ground),
    }
    print(f"  reference-box frame at h/4: {fr.label}")
    for axis in "xyz":
        for got, want in zip(got_faces[axis], faces[axis]):
            assert abs(got - want * 1e-3) <= 0.5 * dx + 1e-12, (axis, got, want)


def test_the_reference_file_is_what_the_provenance_says():
    """The frozen openEMS record loads and carries the tool, the run, the
    reproduce gate, the three rungs, the band and the grid PROVENANCE.md states;
    every rung is passive over its band; and the shared estimators reproduce the
    resonance, depth, −10 dB band and Zin the record froze.  Those numbers were
    computed from the record's own arrays with the same rules by its maker, so
    this is a regression pin of the estimators on frozen values (to 1 kHz), not
    an independent check of them and not of rfx."""
    rec = _load(_OPENEMS_JSON)
    meta = rec["meta"]

    # --- tool, run, image -------------------------------------------------
    assert meta["tool"] == "openEMS"
    assert meta["openems"]["version"] == "0.37.0"
    assert meta["ci_runs_this"] is False
    assert rec["run_id"] == "369367264675"
    assert meta["rfx_commit"] == "be0a84a9629914d8d0a3d3ece7a9f2f0b67f607b"
    assert meta["rfx_openems_commit"] == "5b423bdfe0c84064cf9028167bb759007c33b182"
    assert meta["rfx_openems_image"] == (
        "ghcr.io/bk-squared/rfx-openems@sha256:"
        "ea8df42dbf1bdcbc93479cfe0618c33695dc5266363b4873d392d10f813264ff")
    assert meta["record_complete"] is True
    assert meta["stages_completed"] == ["stage_a", "stage_b_coarse", "stage_b_mid",
                                        "stage_b_fine"]
    assert meta["produced_by"] == "tests/crossval/rt5880_patch/reference/make_openems_reference.py"

    # --- the reproduce gate: openEMS's own Simple_Patch_Antenna tutorial ----
    gate = meta["stage_a_gate"]
    assert gate["passed"] is True
    assert gate["documented_f_hz"] == pytest.approx(2.430e9)
    assert gate["measured_f_hz"] == pytest.approx(2.43303e9, abs=1e4)
    assert gate["window_rel"] == 0.01
    assert "not a measurement of this board" in meta["stage_a_is"]
    assert meta["stage_a_tutorial"]["sha256"] == (
        "33017faa27dc22134fe56964b5cb56623c4043916df01921f27ee54ad1b6a79a")

    # --- the board and the band this case reads it on -----------------------
    assert meta["tl_model_tm010_hz"] == pytest.approx(F_TM010_TL_MODEL_HZ, rel=1e-15)
    assert tuple(meta["resonance_band_ghz"]) == pytest.approx(
        (RESONANCE_BAND_HZ[0] / 1e9, RESONANCE_BAND_HZ[1] / 1e9), rel=1e-12)
    assert meta["boundary"] == ["PML_8"] * 6

    # --- the three rungs ----------------------------------------------------
    rungs = ("stage_b_coarse", "stage_b_mid", "stage_b_fine")
    stages = meta["stages"]
    assert [stages[n]["mesh_realized"]["substrate_z_cells_realized"] for n in rungs] == [4, 6, 8]
    factors = [stages[n]["resolution_factor"] for n in rungs]
    assert factors == pytest.approx([1.0, 2.0 ** -0.5, 0.5], rel=1e-12)
    for n in rungs:
        assert stages[n]["end_criteria_reached"] is True
        assert stages[n]["mesh_realized"]["substrate_top_on_a_line"] is True
        assert stages[n]["geometry_realized"]["feed"]["dx_mm"] == 0.0
        assert stages[n]["geometry_realized"]["stack"]["substrate_z_cells"] == (
            stages[n]["mesh_realized"]["substrate_z_cells_realized"])

    # Frozen values PROVENANCE.md tabulates, pinned so a re-derived or swapped
    # record is caught rather than sliding under a round bound.
    frozen = {
        "stage_b_coarse": (2.3502478870603607, -28.710297676700577, 70.51488407755357),
        "stage_b_mid": (2.3565745658798134, -24.418950147988795, 68.74919698934035),
        "stage_b_fine": (2.359061319234132, -20.58337699975389, 65.17136229823394),
    }
    for n in rungs:
        st = rec[n]
        f, mag, zin = _stage_curve(st)
        assert f.size == N_FREQS
        for key in ("s11_re", "s11_im", "s11_mag", "s11_deg", "zin_re_ohm",
                    "zin_im_ohm", "s11_power"):
            assert len(st[key]) == N_FREQS, (n, key)
        assert f == pytest.approx(FREQS_HZ, rel=1e-12)
        assert tuple(st["witness_band_ghz"]) == (1.6, 3.4)
        # A passive one-port cannot reflect more power than it receives.
        assert st["passivity_tol"] == 1.05
        assert st["max_s11_power_band"] == pytest.approx(float(np.max(mag ** 2)), abs=1e-9)
        assert st["max_s11_power_band"] <= st["passivity_tol"]

        got = resonance(f, mag, zin)
        want = st["resonance"]
        print(f"  openEMS {n}: estimator {got['f']/1e9:.6f} GHz vs record "
              f"{want['refined_f_ghz']:.6f} GHz, depth {got['depth_db']:.4f} vs "
              f"{want['depth_db']:.4f} dB, −10 dB band {got['bw_10db']/1e6:.3f} vs "
              f"{want['band_minus10db']['width_mhz']:.3f} MHz, Zin "
              f"{got['zin'].real:.3f}{got['zin'].imag:+.3f}j vs "
              f"{want['zin_at_refined_ohm']['re']:.3f}"
              f"{want['zin_at_refined_ohm']['im']:+.3f}j Ω")
        assert got["f"] == pytest.approx(want["refined_f_ghz"] * 1e9, abs=1e3)
        assert got["depth_db"] == pytest.approx(want["depth_db"], abs=1e-9)
        assert got["bw_10db"] == pytest.approx(want["band_minus10db"]["width_mhz"] * 1e6,
                                               abs=1e3)
        assert got["zin"].real == pytest.approx(want["zin_at_refined_ohm"]["re"], abs=1e-6)
        assert got["zin"].imag == pytest.approx(want["zin_at_refined_ohm"]["im"], abs=1e-6)
        # f0 and R by the judged estimator, the Re(Zin) peak, on the record's
        # own arrays.
        rx = refined_remax(f, zin, *RESONANCE_BAND_HZ)
        print(f"  openEMS {n}: f0 (the Re(Zin) peak) {rx['f0_hz']/1e9:.6f} GHz, R "
              f"{rx['r_ohm']:.3f} Ω, X {rx['x_ohm']:+.3f} Ω, flags {rx['flags']}")
        assert rx["flags"] == []
        if n == OPENEMS_JUDGED_STAGE:
            # Frozen values the case's two judged quantities are held against
            # (the lane leader's check of 2026-09-25 read the same numbers).
            assert rx["f0_hz"] == pytest.approx(2.319490e9, abs=1e3)
            assert rx["r_ohm"] == pytest.approx(73.457, abs=1e-3)
            assert rx["x_ohm"] == pytest.approx(42.95, abs=1e-2)
        f_frozen, depth_frozen, bw_frozen = frozen[n]
        assert want["refined_f_ghz"] == pytest.approx(f_frozen, abs=1e-12)
        assert want["depth_db"] == pytest.approx(depth_frozen, abs=1e-9)
        assert want["band_minus10db"]["width_mhz"] == pytest.approx(bw_frozen, abs=1e-9)
