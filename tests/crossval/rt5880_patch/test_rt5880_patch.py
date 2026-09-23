"""The RT/Duroid 5880 probe-fed patch antenna — cross-validated against openEMS.

A 40 × 50 mm rectangular patch sits on 3.175 mm of RT/Duroid 5880 (εr 2.2,
tanδ 0.001) over a 56 × 66 mm ground, fed by a 50 Ω probe from the ground to
the patch 9 mm off centre along the 40 mm resonant length.  The patch and the
ground form an open cavity whose TM010 mode resonates near 2.34 GHz: the
cavity's length plus the fringing field at its two radiating edges sets the
frequency, and radiation through those edges loads it, so the probe sees a
matched |S11| dip there.  The fringing extension has no closed form accurate
to 1 %, so rfx's |S11| curve is compared with frozen openEMS results instead of
with an analytic expression.

The boards
----------
The metal, the substrate and the probe are the same in both solvers; the box
around them is not.

* the openEMS record — the ground centred at the origin, absorbing PML_8 on all
  six faces with their inner faces 60 mm clear of the ground's edges, 40 mm
  below it and 86.8 mm above the patch (``meta.stages.*.geometry_realized.pml``);
* rfx here — the patch centred at (``PATCH_CENTRE_X_M``, ``PATCH_CENTRE_Y_M``),
  the declared domain 10.1 mm clear of the ground's x edges, 11.45 mm clear of
  its y edges, 9.525 mm below the ground and 15.875 mm above the patch, with a
  19.05 mm deep CPML OUTSIDE that domain on every face (rfx grid convention):
  24, 48 and 72 cells on the three rungs, the same absorber in metres.

rfx does not reproduce the record's 176 × 186 × 130 mm box: at h/12 it would
hold about 230 M cells before the absorber.  What the box does to this board
was measured at h/4 (the table above ``AIR_H``): across the absorber depths
and air gaps tried the resonance spans 2.3316–2.3485 GHz (0.72 %), and with a
16-cell absorber it had not levelled at 28.6 mm of added air.  The case states that table, and the box witness
solves the coarsest rung once more with 9.525 mm more air on every face and
reports how far the curve moves; nothing is judged on it.

Reference (``reference/``, provenance in ``reference/PROVENANCE.md``):

* ``openems_patch.json`` — openEMS 0.37.0, FDTD, VESSL run 369367264105.
  openEMS's own ``Simple_Patch_Antenna`` tutorial is reproduced first as the
  reproduce gate (``stage_a``; a DIFFERENT patch on a different substrate,
  never compared with this board): 2.43242 GHz against the 2.430 GHz its
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
on node planes on every rung (issue #723).  The in-plane edges cannot land on
nodes: 20 mm is 6.2992 cells of h, never a whole number of any h/n.  A PEC
sheet covers the nodes inside its drawn footprint, so the RESONANT LENGTH the
lattice builds is the span of the patch's node columns, and on most h/n rungs
that span jumps around: 39.1583 mm at h/6, 39.9143 mm at h/7, 39.6875 mm at
h/8 (measured, ``LADDER_M`` below carries the table).  A ladder that mixed them
would put a length change of up to 2 % — about 1.3 % in the resonance — into a
trend that is supposed to be the mesh's.  ``assert_realized`` therefore holds
the resonant length the coarsest rung realizes and refuses a rung that realizes
another, the way the Sheen low-pass filter's test refuses a rung whose two
feeds realize different widths.  The rungs that pass are h/4, h/8 and h/12
(793.75, 396.875 and 264.583 µm): every one of them covers the patch with
node columns 39.6875 mm apart.

The thresholds are the v2 accuracy bar (1 % in frequency, 2 dB in magnitude)
and the three rules the PI approved with the MSL notch filter's plan on
2026-09-22: the mesh statement (the last two rungs within 1 %), the deep-null
exclusion (a bin where either curve is below −20 dB is judged by position only)
and the ring-down witness (−40 dB, the repo's rule).  Every one is a named
constant below with its source.  No threshold is derived from a run, and this
case commits no record of a run.

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

The three fast tests carry no mark: they build the structure and read the
reference file, and never step the FDTD.
"""

from __future__ import annotations

import functools
import json
import math
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec

from tests._realized_geometry import _node_line, realized

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
L_PATCH = 40.0e-3            # resonant length, along x
W_PATCH = 50.0e-3            # radiating-edge width, along y
GP_X = 56.0e-3               # ground plane
GP_Y = 66.0e-3
FEED_OFFSET_X = -9.0e-3      # probe, 9 mm off centre along the resonant length
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
# MEASURED at h/4 on CPU (local, 2026-09-24; this file's board, NUM_PERIODS =
# 60), the resonance and depth against the air added beyond the smallest box
# below on every face (that box: 10.1 / 11.45 mm clear of the ground's x / y
# edges, 9.525 mm below it, 15.875 mm above the patch) and the CPML depth;
# the last column is max |ΔdB| of |S11| against the +6h / 24-cell run over the
# bins where both curves are above −20 dB (every worst bin sits on the dip's
# shoulders, 2.32–2.35 GHz):
#
#     air added      CPML              resonance      depth       max |ΔdB|
#     0              8 (6.35 mm)       2.348539 GHz   −17.90 dB   7.309 dB
#     0             16 (12.70 mm)      2.339521 GHz   −20.95 dB   3.984 dB
#     0             24 (19.05 mm)      2.337070 GHz   −22.11 dB   2.684 dB
#     3h (9.53 mm)   8                 2.345119 GHz   −28.56 dB   5.942 dB
#     3h            16                 2.338055 GHz   −25.72 dB   2.771 dB
#     3h            24                 2.335882 GHz   −25.06 dB   (not kept)
#     6h (19.05 mm)  8                 2.338409 GHz   −36.41 dB   3.306 dB
#     6h            16                 2.334838 GHz   −29.36 dB   1.179 dB
#     6h            24                 2.333734 GHz   −26.52 dB   —
#     9h (28.58 mm) 16                 2.331641 GHz   −28.38 dB   1.632 dB
#
# (The 3h / 24-cell row is this file's own box witness run locally; its curve
# was not kept.)  At 16 cells the resonance keeps falling as air is added,
# −0.06, −0.14 and −0.14 % per 3·h step up to 9·h, and has not levelled there;
# at 24 cells −0.05 and −0.09 % per step up to 6·h; at every air gap a deeper
# absorber lowers it too.  The ladder's box is ``AIR_H`` = 0 with
# a 6·h (19.05 mm) absorber: at h/12 it is 52.7 M cells, where +6·h of air at
# the same depth would be 143 M (an RTX 4090 ran out of memory on 134 M,
# docs/agent/gpu-throughput.mdx).  At h/4 this box reads 2.337070 GHz, +0.143 %
# from the +6h / 24-cell box and +0.233 % from the +9h / 16-cell one.  The box
# witness below measures the next step of air on every ladder run.
AIR_H = 0
CPML_THICKNESS_H = 6
# The box witness: the same board with BOX_WITNESS_EXTRA_AIR_H more
# substrate thicknesses of air on every face, solved at the coarsest rung
# only, and REPORTED (see ``_box_witness``).
BOX_WITNESS_EXTRA_AIR_H = 3


class Frame(NamedTuple):
    """Where the board sits in the box, for ``air_h`` h of extra air."""
    air_h: int
    cx: float           # patch (and ground) centre, x
    cy: float           # patch (and ground) centre, y
    z_ground: float     # the substrate floor, the ground sheet's plane
    z_patch: float      # the substrate top, the patch sheet's plane
    domain: tuple       # the declared domain, CPML outside it


def frame(air_h: int = AIR_H) -> Frame:
    cx = (12 + air_h) * H_SUB
    cy = (14 + air_h) * H_SUB
    z_ground = (3 + air_h) * H_SUB
    return Frame(air_h, cx, cy, z_ground, z_ground + H_SUB,
                 (2 * cx, 2 * cy, (9 + 2 * air_h) * H_SUB))


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


def cpml_layers(dx: float) -> int:
    """CPML cells at ``dx`` for an absorber ``CPML_THICKNESS_H``·h deep."""
    n = CPML_THICKNESS_H * H_SUB / dx
    assert abs(n - round(n)) < 1e-9, (dx, n)
    return int(round(n))


# --- the ladder -----------------------------------------------------------
# dx = h_sub/n keeps both substrate faces on node planes (issue #723).
#
# MEASURED at build time on this board (``realized_geometry`` below, the
# product's own sheet footprints), the node span of the patch's node columns
# along the resonant length, and whether ``assert_realized`` accepts the rung:
#
#     n   dx (µm)    resonant length (mm)        n   dx (µm)    resonant length (mm)
#     4   793.750    39.6875   accepted          9   352.778    39.5111   refused
#     5   635.000    39.3700   refused          10   317.500    39.3700   refused
#     6   529.167    39.1583   refused          11   288.636    39.8318   refused
#     7   453.571    39.9143   refused          12   264.583    39.6875   accepted
#     8   396.875    39.6875   accepted
#
# WHY: 20 mm is 6.2992·n cells of h/n, and a sheet covers the nodes inside
# its footprint, so each half of the patch realizes floor(6.2992·n) cells.
# For n = 4, 8, 12 that is 25, 50 and 75 cells — the same 19.84375 mm each
# side — and for every other n up to 12 it is a different length.  (n = 7, 14,
# 21 share a second family, 39.9143 mm; its second rung alone, h/14, would cost
# more cell-steps in this box than this whole ladder.)
#
# THE RUNG CRITERION IS THE RESONANT LENGTH, AND ONLY THAT.  The width, the
# ground and the probe offset realize within a cell of their declared values on
# every rung but not identically — measured, patch width node span 49.2125 /
# 49.2125 / 49.7417 mm, ground node spans 55.5625 × 65.0875 / 55.5625 × 65.8812
# / 55.5625 × 65.6167 mm, probe 8.73125 / 9.12813 / 8.99583 mm from the patch
# centre (declared 9 mm) at h/4, h/8, h/12.  They are printed per rung and not
# judged.
LADDER_M = (
    H_SUB / 4,    # 793.750 µm,  4 substrate cells
    H_SUB / 8,    # 396.875 µm,  8 substrate cells
    H_SUB / 12,   # 264.583 µm, 12 substrate cells
)
DX_COARSEST_M = LADDER_M[0]

# --- the sweep ------------------------------------------------------------
# rfx samples the record's OWN frequency grid, linspace(1.6, 3.4 GHz, 901), so
# the judged comparison needs no interpolation.  DFT bins cost no time steps.
# The bin is 2 MHz, under a tenth of the 1 % frequency bar at the resonance
# (23 MHz at 2.34 GHz).
N_FREQS = 901
SWEEP_HZ = (1.6e9, 3.4e9)
FREQS_HZ = np.linspace(SWEEP_HZ[0], SWEEP_HZ[1], N_FREQS)

# The resonance is searched inside the record's own window
# (``meta.resonance_band_ghz``): 0.80–1.20 × the Balanis transmission-line
# estimate of this patch's TM010 (``meta.tl_model_tm010_hz``), which is the
# retired script's own search window.  Estimator: the repository's shared
# ``refined_extremum`` on log|S11|, the record's own.
F_TM010_TL_MODEL_HZ = 2415595433.5060616
RESONANCE_BAND_HZ = (0.80 * F_TM010_TL_MODEL_HZ, 1.20 * F_TM010_TL_MODEL_HZ)
MINUS10_DB = -10.0

# --- the record length ----------------------------------------------------
# ``num_periods`` is in periods of ``freq_max`` (0.25 ns here).  MEASURED on
# this board at h/4 on CPU (local, 2026-09-24; in the smallest box with an
# 8-cell absorber, the first row of the box table above), ring-down settling of
# the witness probe, the resonance and its depth:
#
#     num_periods = 30    −44.57 dB    2.348553 GHz   −17.381 dB
#     num_periods = 40    −57.65 dB    2.348549 GHz   −17.800 dB
#     num_periods = 60    −83.35 dB    2.348539 GHz   −17.896 dB
#     num_periods = 120  −117.58 dB    2.348540 GHz   −17.899 dB
#
# 60 meets the −40 dB rule with 43 dB to spare and puts the resonance within
# 1 kHz and the depth within 0.003 dB of the doubled record; in the ladder's box
# (19.05 mm absorber) the same 60 periods ring down to −80.62 dB.  The
# ring-down is set by the patch's radiation Q, a physical time, so the same
# record length in nanoseconds serves every rung.
NUM_PERIODS = 60.0

# --- the ring-down witness probe -------------------------------------------
# Ez inside the substrate, halfway up, under the patch near its +x radiating
# edge where the TM010 field is strong, and 24 mm from the probe so it never
# shares a cell with the drive (a source-dominated record measures the pulse
# turning off, not the ring-down, #1090).  A node on every rung: 15 mm and
# 5 mm snap to the nearest node, and h/2 is a whole number of cells on each.
WITNESS_PROBE_OFFSET_M = (15.0e-3, 5.0e-3, H_SUB / 2)

# ------------------------------------------------------------ thresholds
# The v2 accuracy bar (rfx CLAUDE.md, PI 2026-09-20):
FREQ_BAR = 0.01          # resonances, cutoffs, notches, band edges: within 1 %
MAG_BAR_DB = 2.0         # power and magnitude: within 2 dB
# Rules approved by the PI with the MSL notch filter's plan (2026-09-22):
LADDER_AGREEMENT = 0.01  # mesh statement: the last two rungs differ by < 1 %
DEEP_NULL_DB = -20.0     # a bin where either curve is below this is judged by
#                          position only, never by magnitude
# Witnesses on rfx's own record, not comparisons (rfx CLAUDE.md, Validation
# rules): a record that has not rung down to −40 dB is truncation-suspect, and
# a passive one-port cannot reflect more power than it receives, so |S11| more
# than one percent above one is a measurement artefact — the rung stops instead
# of being compared.  Measured on this board at h/4 in the ladder's box
# (NUM_PERIODS = 60): settling −80.62 dB, max |S11| 0.99276 over 1.6–3.4 GHz.
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
        cpml_layers=cpml_layers(dx),
        boundary=BoundarySpec.uniform("cpml"),
    )
    sim.add_material("rt5880", eps_r=EPS_R, sigma=SIGMA_SUB)
    return sim


def _ground_box(fr: Frame = FRAME, z: float | None = None) -> Box:
    z = fr.z_ground if z is None else z
    return Box((fr.cx - GP_X / 2, fr.cy - GP_Y / 2, z),
               (fr.cx + GP_X / 2, fr.cy + GP_Y / 2, z))


def _substrate_box(fr: Frame = FRAME) -> Box:
    return Box((fr.cx - GP_X / 2, fr.cy - GP_Y / 2, fr.z_ground),
               (fr.cx + GP_X / 2, fr.cy + GP_Y / 2, fr.z_patch))


def _patch_box(fr: Frame = FRAME, *, x_hi_trim: float = 0.0,
               extra_thickness: float = 0.0) -> Box:
    return Box((fr.cx - L_PATCH / 2, fr.cy - W_PATCH / 2, fr.z_patch),
               (fr.cx + L_PATCH / 2 - x_hi_trim, fr.cy + W_PATCH / 2,
                fr.z_patch + extra_thickness))


def _add_probe_and_witness(sim: Simulation, fr: Frame = FRAME, *,
                           extent: float = H_SUB) -> None:
    """The 50 Ω probe from the ground sheet to the patch sheet, and the
    ring-down witness probe."""
    sim.add_port(position=(fr.cx + FEED_OFFSET_X, fr.cy, fr.z_ground), component="ez",
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
    ``fr`` places the board in the box; only the box witness passes anything
    but the default.
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
    # The ground is the larger of the two sheets (56 × 66 mm against 40 × 50).
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
    return out


@functools.lru_cache(maxsize=1)
def _ladder_resonant_length_m() -> float:
    """The patch's resonant length as the coarsest rung realizes it — the
    node span every rung of the ladder must realize (see ``LADDER_M``)."""
    g = realized_geometry(build(DX_COARSEST_M))
    return float(g["patch"]["x_node_span_m"])


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

    # --- the resonant length, the same on every rung -------------------------------
    held = _ladder_resonant_length_m()
    got = g["patch"]["x_node_span_m"]
    if abs(got - held) > 1e-9:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the patch's node columns span {got*1e3:.4f} mm "
            f"along its resonant length; the ladder holds {held*1e3:.4f} mm, what "
            f"the coarsest rung ({DX_COARSEST_M*1e6:.2f} µm) realizes. The TM010 "
            "resonance scales with that length, so this rung solves a different "
            f"antenna ({100*(got-held)/held:+.3f} % in length) and a ladder "
            "through it would put a length change into the mesh trend.")
    cx, cy = g["patch"]["centre_m"]
    if abs(cx - fr.cx) > 1e-9 or abs(cy - fr.cy) > 1e-9:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the realized patch is centred at "
            f"({cx*1e3:.4f}, {cy*1e3:.4f}) mm, not at the declared "
            f"({fr.cx*1e3:.4f}, {fr.cy*1e3:.4f}) mm; the "
            "probe's offset from the patch centre is then not the declared one.")

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
    if abs(ox - FEED_OFFSET_X) > 0.5 * dx + 1e-12 or abs(oy) > 1e-12:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the probe realizes ({ox*1e3:.4f}, {oy*1e3:.4f}) mm "
            f"from the patch centre; declared ({FEED_OFFSET_X*1e3:.1f}, 0) mm, "
            "and the nearest node is never more than half a cell away.")
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
              f"{FEED_OFFSET_X*1e3:+.1f}, 0)")


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
          f"{fr.domain[2]*1e3:.3f} mm (declared domain, {fr.air_h}·h of air added; "
          f"{cpml_layers(dx)} CPML cells = {cpml_layers(dx)*dx*1e3:.3f} mm outside it "
          "on every face)")
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
    """ΔdB of rfx − reference at every bin of the record where BOTH curves are
    above the deep-null level, plus the resonance distance and the reported
    features.

    The two grids are the same 901 bins, so nothing is interpolated.  The
    deep-null exclusion is two-sided on purpose: a bin where one curve is in
    its dip and the other is not is the two dips sitting at different
    frequencies, which the frequency bar already judges; counting it again as a
    magnitude error would score one offset twice."""
    ref_f, ref_mag, ref_zin = _stage_curve(stage)
    ours_f = np.asarray(rung["freqs_hz"], dtype=float)
    assert ours_f.shape == ref_f.shape and np.allclose(ours_f, ref_f, rtol=0, atol=1.0), (
        "rfx was not sampled on the record's own frequency grid")
    ref_db = _db(ref_mag)
    ours_db = _db(np.abs(rung["s11"]))
    keep = (ref_db >= DEEP_NULL_DB) & (ours_db >= DEEP_NULL_DB)
    delta = ours_db - ref_db
    ref_res = resonance(ref_f, ref_mag, ref_zin)
    our_res = rung["resonance"]
    max_abs = float(np.max(np.abs(delta[keep]))) if keep.any() else float("nan")
    worst = int(np.argmax(np.where(keep, np.abs(delta), -np.inf))) if keep.any() else -1
    return dict(
        label=label, f_hz=ref_f, ref_db=ref_db, ours_db=ours_db, delta_db=delta,
        compared=keep, n_bins=int(ref_f.size), n_compared=int(keep.sum()),
        max_abs_delta_db=max_abs, worst_index=worst,
        ref_res=ref_res, our_res=our_res,
        res_pct=100.0 * (our_res["f"] - ref_res["f"]) / ref_res["f"],
    )


_MAX_TABLE_ROWS = 90


def _print_comparison(c: dict) -> None:
    r, o = c["ref_res"], c["our_res"]
    print(f"  --- {c['label']} ---")
    print(f"    resonance: rfx {o['f']/1e9:.5f} GHz vs reference {r['f']/1e9:.5f} GHz "
          f"-> {c['res_pct']:+.3f} % (|Δ| {abs(c['res_pct']):.3f} %)")
    print(f"    depth (reported): rfx {o['depth_db']:.2f} dB vs {r['depth_db']:.2f} dB")
    print(f"    −10 dB band (reported): rfx {o['bw_10db']/1e6:.1f} MHz "
          f"({o['f_lo_10db']/1e9:.4f}–{o['f_hi_10db']/1e9:.4f} GHz) vs "
          f"{r['bw_10db']/1e6:.1f} MHz ({r['f_lo_10db']/1e9:.4f}–"
          f"{r['f_hi_10db']/1e9:.4f} GHz)")
    print(f"    Zin at the resonance (reported): rfx {o['zin'].real:.2f}"
          f"{o['zin'].imag:+.2f}j Ω vs {r['zin'].real:.2f}{r['zin'].imag:+.2f}j Ω")
    print(f"    |S11| dB compared at {c['n_compared']} of the record's {c['n_bins']} "
          f"bins, {SWEEP_HZ[0]/1e9:.1f}–{SWEEP_HZ[1]/1e9:.1f} GHz (both curves above "
          f"{DEEP_NULL_DB:.0f} dB); max |ΔdB| = {c['max_abs_delta_db']:.3f} dB"
          + (f" at {c['f_hz'][c['worst_index']]/1e9:.4f} GHz"
             if c["worst_index"] >= 0 else ""))
    idx = np.arange(c["n_bins"])
    stride = max(1, int(np.ceil(idx.size / _MAX_TABLE_ROWS)))
    shown = set(idx[::stride].tolist())
    if c["worst_index"] >= 0:
        shown.add(c["worst_index"])
    shown.add(r["index"])
    shown.add(o["index"])
    print(f"    (table printed every {stride} bins, plus the two resonance bins and "
          "the worst compared bin marked *)")
    print("    f_GHz, ref_dB, rfx_dB, delta_dB, compared")
    for i in sorted(shown):
        mark = "*" if i == c["worst_index"] else " "
        print(f"     {mark}{c['f_hz'][i]/1e9:8.4f} {c['ref_db'][i]:10.3f} "
              f"{c['ours_db'][i]:10.3f} {c['delta_db'][i]:9.3f}  "
              f"{bool(c['compared'][i])}")


# ------------------------------------------------------------------ ladder
def _rungs() -> tuple[float, ...]:
    raw = os.environ.get("RFX_RT5880_PATCH_RUNGS", "").strip()
    if not raw:
        return LADDER_M
    return tuple(float(v) for v in raw.split(",") if v.strip())


def _box_witness(base: dict, dx: float) -> dict:
    """Solve the same board with ``BOX_WITNESS_EXTRA_AIR_H`` more substrate
    thicknesses of air on every face, at the same cell size and with the same
    absorber depth, and report how far its resonance and |S11| move.

    A witness on rfx's own record, not a comparison with the reference: rfx
    solves this board in a smaller box than the openEMS record does (see the
    box constants), and this measures what the box does to the observable on
    the ladder's own code.  REPORTED, not judged — no bar has been set for it.
    """
    fr = frame(AIR_H + BOX_WITNESS_EXTRA_AIR_H)
    print(f"\n  --- box witness at dx = {dx*1e6:.3f} µm ---")
    print(f"  the same board with {BOX_WITNESS_EXTRA_AIR_H}·h = "
          f"{BOX_WITNESS_EXTRA_AIR_H*H_SUB*1e3:.3f} mm more air on every face (box "
          f"{fr.domain[0]*1e3:.3f} × {fr.domain[1]*1e3:.3f} × {fr.domain[2]*1e3:.3f} mm "
          f"instead of {DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × "
          f"{DOMAIN_Z_M*1e3:.3f} mm; the absorber depth unchanged)")
    other = run_rung(dx, fr)
    other_res = resonance(other["freqs_hz"], np.abs(other["s11"]), zin_from_s11(other["s11"]))
    f = np.asarray(base["freqs_hz"], dtype=float)
    a = _db(np.abs(base["s11"]))
    b = _db(np.abs(other["s11"]))
    keep = (a >= DEEP_NULL_DB) & (b >= DEEP_NULL_DB)
    d = b - a
    worst = int(np.argmax(np.where(keep, np.abs(d), -np.inf)))
    out = {
        "dx_m": dx, "extra_air_m": BOX_WITNESS_EXTRA_AIR_H * H_SUB,
        "resonance_hz": other_res["f"], "depth_db": other_res["depth_db"],
        "resonance_shift_pct": 100.0 * (other_res["f"] - base["resonance"]["f"])
                               / base["resonance"]["f"],
        "max_abs_delta_db": float(np.max(np.abs(d[keep]))), "n_compared": int(keep.sum()),
        "worst_f_hz": float(f[worst]), "n_cells": other["n_cells"],
        "wall_s": other["wall_s"],
    }
    print(f"  resonance {other_res['f']/1e9:.6f} GHz ({other_res['depth_db']:.2f} dB) "
          f"in the larger box vs {base['resonance']['f']/1e9:.6f} GHz "
          f"({base['resonance']['depth_db']:.2f} dB): {out['resonance_shift_pct']:+.3f} %")
    print(f"  |S11|: max |Δ| = {out['max_abs_delta_db']:.3f} dB over "
          f"{out['n_compared']} of {f.size} bins where both curves are above "
          f"{DEEP_NULL_DB:.0f} dB; worst at {out['worst_f_hz']/1e9:.4f} GHz")
    print("    f_GHz, larger box |S11| dB, ladder box |S11| dB, Δ")
    for i in range(0, f.size, 10):
        print(f"      {f[i]/1e9:8.4f} {b[i]:10.3f} {a[i]:10.3f} {d[i]:8.3f}")
    return out


def moves_monotonically(values) -> bool:
    """The mesh statement's monotone check: every step along the ladder moves
    the same way.

    The ONE place the three cross-validation tests' monotone rule lives in this
    file — the PI is deciding a small tolerance for it, and the leader applies
    that decision to the MSL notch filter, the Sheen low-pass filter and this
    case in one commit.  Until then it is the notch test's rule: strictly
    increasing or strictly decreasing, no tolerance.
    """
    v = [float(x) for x in values]
    return (all(b > a for a, b in zip(v, v[1:]))
            or all(b < a for a, b in zip(v, v[1:])))


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
          f"{FEED_OFFSET_X*1e3:+.1f} mm from the patch centre along x; box "
          f"{DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × {DOMAIN_Z_M*1e3:.3f} mm "
          f"(ground edges {X_CLEARANCE_M*1e3:.3f} / {Y_CLEARANCE_M*1e3:.3f} mm clear "
          f"in x / y, {BELOW_GROUND_M*1e3:.3f} mm below the ground, "
          f"{ABOVE_PATCH_M*1e3:.3f} mm above the patch), CPML "
          f"{CPML_THICKNESS_H*H_SUB*1e3:.3f} mm deep outside it")

    results = []
    for dx in rungs:
        print(f"\n=== rung dx = {dx*1e6:.3f} µm ===")
        r = run_rung(dx)
        r["resonance"] = res = resonance(r["freqs_hz"], np.abs(r["s11"]),
                                         zin_from_s11(r["s11"]))
        r["s11_db"] = _db(np.abs(r["s11"]))
        print(f"  resonance {res['f']/1e9:.6f} GHz (bin {res['bin_f']/1e9:.5f}, "
              f"sub-bin shift {res['sub_bin_shift']:+.3f}), depth "
              f"{res['depth_db']:.3f} dB, −10 dB band {res['bw_10db']/1e6:.2f} MHz "
              f"({res['f_lo_10db']/1e9:.5f} – {res['f_hi_10db']/1e9:.5f} GHz), Zin "
              f"{res['zin'].real:.2f}{res['zin'].imag:+.2f}j Ω")
        print("  |S11| on the record's grid — f_GHz, |S11| dB, Re Zin, Im Zin:")
        zin = zin_from_s11(r["s11"])
        for f, a, z in zip(r["freqs_hz"], r["s11_db"], zin):
            print(f"    {f/1e9:8.4f} {a:10.4f} {z.real:10.3f} {z.imag:10.3f}")
        results.append(r)
        # The premise the comparison rests on — this box does not set the
        # observable — measured on the coarsest rung, and reported.
        if len(results) == 1:
            box_witness = _box_witness(r, dx)

    openems = _load(_OPENEMS_JSON)
    judged_stage = openems[OPENEMS_JUDGED_STAGE]
    stages = openems["meta"]["stages"]

    # ---- (a) mesh statement: the resonance must settle along the ladder ------
    resonances = [r["resonance"]["f"] for r in results]
    print("\n  mesh statement — resonance along the rfx ladder:")
    for r in results:
        print(f"    dx {r['dx_m']*1e6:8.3f} µm  {r['resonance']['f']/1e9:.6f} GHz  "
              f"{r['resonance']['depth_db']:7.2f} dB  {r['n_cells']} cells  "
              f"{r['n_steps']} steps  {r['wall_s']:.1f} s")
    if len(resonances) >= 2:
        for (a, b), (ra, rb) in zip(zip(resonances, resonances[1:]),
                                    zip(results, results[1:])):
            print(f"    {ra['dx_m']*1e6:.3f} -> {rb['dx_m']*1e6:.3f} µm: "
                  f"{100.0*(b-a)/a:+.3f} %")
        last_two_pct = 100.0 * abs(resonances[-1] - resonances[-2]) / resonances[-2]
        print(f"    last two rungs differ by {last_two_pct:.3f} % "
              f"(bar {LADDER_AGREEMENT*100:.0f} %)")
    else:
        last_two_pct = float("nan")

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
        ref_res.append(rr["f"])
        print(f"    {name:15s} factor {ms['resolution_factor']:.4f}  "
              f"{ms['mesh_realized']['substrate_z_cells_realized']} substrate cells  "
              f"{ms['mesh_realized']['n_cells']:9d} cells  resonance "
              f"{rr['f']/1e9:.6f} GHz  {rr['depth_db']:7.2f} dB  −10 dB band "
              f"{rr['bw_10db']/1e6:.1f} MHz  Zin {rr['zin'].real:.2f}"
              f"{rr['zin'].imag:+.2f}j Ω  end energy "
              f"{ms['solver_run']['final_energy_db']:.2f} dB")
    print(f"    coarse -> mid {100.0*(ref_res[1]-ref_res[0])/ref_res[0]:+.3f} %, "
          f"mid -> fine {100.0*(ref_res[2]-ref_res[1])/ref_res[1]:+.3f} %")

    # A partial ladder shows no convergence trend, so it states no verdict: the
    # distances below are still printed — they are the diagnostic the env
    # override exists for — and the test skips at the end instead of passing or
    # failing.  Any rung set that is not the ladder itself, in order, is partial.
    partial = tuple(rungs) != LADDER_M
    monotone = moves_monotonically(resonances)

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
    fig_path = _write_figure(results, openems, tmp_path)
    print(f"\n  figure: {fig_path}")
    print(f"  box witness at {box_witness['dx_m']*1e6:.3f} µm "
          f"(+{box_witness['extra_air_m']*1e3:.3f} mm of air on every face): resonance "
          f"{box_witness['resonance_shift_pct']:+.3f} %, |S11| up to "
          f"{box_witness['max_abs_delta_db']:.3f} dB — reported, not judged")

    if partial:
        pytest.skip(
            "RFX_RT5880_PATCH_RUNGS restricted the ladder to "
            f"{[f'{d*1e6:.3f}µm' for d in rungs]}; the resonances are "
            f"{[f'{f/1e9:.6f} GHz' for f in resonances]}. A partial ladder shows no "
            "convergence trend, so the distances above are a diagnostic and this "
            "run states no verdict.")

    # ---- (a) the mesh statement comes first: without it nothing is judged ---
    assert monotone, (
        "the resonance does not move monotonically along the ladder: "
        f"{[f'{f/1e9:.6f} GHz' for f in resonances]} at "
        f"{[f'{d*1e6:.3f}µm' for d in rungs]}. Without a monotone trend the mesh is "
        "not shown to converge and the comparison says nothing.")
    if not last_two_pct < LADDER_AGREEMENT * 100.0:
        # PI decision 2026-09-22, carried over from the MSL notch filter case:
        # the v2 bar is never loosened, and a case that cannot meet it says so
        # with its numbers. The xfail is IMPERATIVE, so every witness above —
        # the ring-down, the passivity excess, the realized board — and a
        # monotonicity failure still FAIL the test; only the two comparison
        # bars below are held back. When the mesh converges this branch is no
        # longer taken and the comparison judges for real.
        pytest.xfail(
            "known limitation (PI 2026-09-22): the uniform mesh has not "
            f"converged — the two finest rungs put the resonance "
            f"{last_two_pct:.3f} % apart (bar {LADDER_AGREEMENT*100:.0f} %): "
            f"{[f'{f/1e9:.6f} GHz' for f in resonances]}; the finest rung sits "
            f"{fine['res_pct']:+.3f} % from the openEMS record's "
            f"{fine['ref_res']['f']/1e9:.5f} GHz and its |S11| differs by up to "
            f"{fine['max_abs_delta_db']:.3f} dB. The comparison above is "
            "reported, not judged.")
    assert abs(fine["res_pct"]) < FREQ_BAR * 100.0, (
        f"the resonance sits {fine['res_pct']:+.3f} % from the openEMS record's "
        f"{fine['ref_res']['f']/1e9:.5f} GHz (bar {FREQ_BAR*100:.0f} %); rfx reads "
        f"{fine['our_res']['f']/1e9:.5f} GHz on the {finest['dx_m']*1e6:.3f} µm mesh.")
    assert fine["max_abs_delta_db"] <= MAG_BAR_DB, (
        f"|S11| differs from the openEMS record by up to "
        f"{fine['max_abs_delta_db']:.3f} dB (bar {MAG_BAR_DB:.0f} dB) over the "
        f"{fine['n_compared']} bins where both curves are above {DEEP_NULL_DB:.0f} dB.")


def _write_figure(results, openems, tmp_path) -> Path:
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
# The candidate rungs the ladder was chosen from, with what ``assert_realized``
# does with each (the table above ``LADDER_M``).
CANDIDATE_RUNGS_N = (4, 5, 6, 7, 8)


def test_the_lattice_builds_the_declared_board_on_every_rung():
    """Build-time only: every ladder rung realizes the declared board, and every
    other h/n from h/4 to h/8 is refused for realizing another resonant length
    (h/9 … h/11, measured in the table above ``LADDER_M``, are not rebuilt
    here: each is 40–50 M cells in this box)."""
    print(f"\n  declared board: patch {L_PATCH*1e3:.1f} × {W_PATCH*1e3:.1f} mm centred "
          f"at ({PATCH_CENTRE_X_M*1e3:.3f}, {PATCH_CENTRE_Y_M*1e3:.3f}) mm, ground "
          f"{GP_X*1e3:.0f} × {GP_Y*1e3:.0f} mm on z = {GROUND_Z_M*1e3:.3f} mm, patch on "
          f"z = {PATCH_Z_M*1e3:.3f} mm, probe {FEED_OFFSET_X*1e3:+.1f} mm along x; box "
          f"{DOMAIN_X_M*1e3:.3f} × {DOMAIN_Y_M*1e3:.3f} × {DOMAIN_Z_M*1e3:.3f} mm")
    for dx in LADDER_M:
        g = assert_realized(build(dx), dx)
        _print_realized(g, dx)
        assert g["patch"]["x_node_span_m"] == pytest.approx(_ladder_resonant_length_m(),
                                                            abs=1e-9)

    accepted = []
    for n in CANDIDATE_RUNGS_N:
        dx = H_SUB / n
        g = realized_geometry(build(dx))
        try:
            check_realized(g, dx)
        except AssertionError as exc:
            verdict = f"refused — {str(exc)[:110]}..."
        else:
            verdict = "accepted"
            accepted.append(n)
        print(f"  h/{n:<2d} {dx*1e6:8.3f} µm: resonant length (node span) "
              f"{g['patch']['x_node_span_m']*1e3:.4f} mm, width "
              f"{g['patch']['y_node_span_m']*1e3:.4f} mm, probe "
              f"{g['probe_offset_from_patch_centre_m'][0]*1e3:+.4f} mm — {verdict}")
    ladder_n = tuple(int(round(H_SUB / d)) for d in LADDER_M)
    assert tuple(accepted) == tuple(n for n in CANDIDATE_RUNGS_N if n in ladder_n), (
        f"the rungs assert_realized accepts among h/{CANDIDATE_RUNGS_N[0]} … "
        f"h/{CANDIDATE_RUNGS_N[-1]} are {accepted}, not the ladder's {ladder_n}")


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
         _build_patch_one_cell_short, "the ladder holds"),
    )
    for label, builder, expected in cases:
        with pytest.raises(AssertionError) as caught:
            assert_realized(builder(dx), dx)
        print(f"\n  {label}: {caught.value}")
        assert expected in str(caught.value), (label, str(caught.value))


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
    assert rec["run_id"] == "369367264105"
    assert meta["rfx_commit"] == "ebf340500c6c7bf9a94a589e1fe39539a805ed6e"
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
    assert gate["measured_f_hz"] == pytest.approx(2.43242e9, abs=1e4)
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
        "stage_b_coarse": (2.334990563684854, -28.757887079746723, 69.89792186376764),
        "stage_b_mid": (2.339617424180239, -26.702811267531928, 69.17622746111319),
        "stage_b_fine": (2.3441803781571533, -22.095807992279795, 66.25704805809995),
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
        f_frozen, depth_frozen, bw_frozen = frozen[n]
        assert want["refined_f_ghz"] == pytest.approx(f_frozen, abs=1e-12)
        assert want["depth_db"] == pytest.approx(depth_frozen, abs=1e-9)
        assert want["band_minus10db"]["width_mhz"] == pytest.approx(bw_frozen, abs=1e-9)
