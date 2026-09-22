"""The Sheen stepped-impedance low-pass filter — cross-validated against openEMS.

Two 2.413 mm wide 50 Ω microstrip feeds on 0.794 mm of lossless RT/Duroid
(εr = 2.2) are joined by one wide 20.320 × 2.540 mm section.  The wide
section is a shunt capacitance between two line steps: below about 5 GHz it
passes, and above it the two steps reflect, so |S21| falls away into a deep
stopband.  The stopband is a DOUBLE transmission zero near 7 and 8 GHz, not a
single null — the reference maker's header records that measurement, and it is
why the estimator below is windowed and the whole curve is printed.  The
structure has no closed form, so rfx's curve is compared with frozen external
results instead of with an analytic expression.

The two feeds are not collinear: the input feed's centre sits at y 9.8565 mm
and the output feed's at y 16.4635 mm, 6.607 mm apart across the wide section.
The board, the box and the ports are the ones the retired script
``validation/crossval/07_sheen_lpf.py`` declared (its ``build_rfx_sim``); this
file is where they live now.

The boards
----------
rfx and the openEMS record solve the SAME box — 27.472 × 26.320 × 3.794 mm,
the same substrate, the same trace, the same port planes — so there is no
arm-length premise to measure here and no arm-length witness (the MSL notch
filter case needs one because rfx runs that board on shorter arms; this one
does not).  The Palace FEM record is on the same frame again but with lumped
ports, and it conserves 0.1–76 % of the power it receives, so it is REPORTED
and never judged.

References (``reference/``, provenance in ``reference/PROVENANCE.md``):

* ``openems_sheen.json`` — openEMS 0.37.0, FDTD.  openEMS's own
  ``MSL_NotchFilter`` tutorial is reproduced first in every job as the
  reproduce gate (``stage_a``; a DIFFERENT structure, never compared with this
  board), then this board on three meshes: ``stage_b_coarse`` (198.5 µm, 4
  substrate cells), ``stage_b_mid`` (140.4 µm, 6) and ``stage_b_fine``
  (99.25 µm, 10 realized).  801 points, 0.5–20 GHz, LINEAR magnitude and
  degrees.  **JUDGED against ``stage_b_fine``**; the coarse and mid rungs are
  the record's own mesh statement.
* ``palace_fem.json`` — Palace, frequency-domain FEM on conformal tetrahedra,
  order 2, lossless, first-order absorbing far box, two lumped ports.  Two
  meshes: ``coarse`` (lc 0.25 mm, 81 points, 4–12 GHz) and ``mid``
  (lc 0.18 mm, 51 points, 6–9 GHz).  Arrays are LINEAR magnitude.  REPORTED,
  never judged.

The record is TRUNCATED by design: every rung stops at a declared record
length of about 10 ns rather than at a decay level, because the box energy on
this board sits flat at about −31 dB from N to 2N steps (a trapped mode that no
end criterion closes).  What the record offers instead is its own N/2N witness:
the same rung solved twice, at N and 2N timesteps, |ΔS21| ≤ 1.1e-3 dB and
|ΔS11| ≤ 1.4e-3 dB over 2–12 GHz.  This case PRINTS those three numbers from
the record and asserts only that they are under
``RECORD_LENGTH_WITNESS_BAR_DB`` — a sanity bound on the premise, one twentieth
of the 2 dB magnitude bar, not a gate on the physics.

The ladder is dx = h_sub/n so the substrate top always lands on a node plane
(issue #723): 264.67 µm (n = 3), 158.8 µm (n = 5), 113.43 µm (n = 7),
66.17 µm (n = 12), i.e. 3, 5, 7 and 12 cells across the substrate.  Those four
are not every h/n: the two feed centres sit 6.607 mm apart and 6.607 mm / dx is
never a whole number of cells, so the two feeds land at different sub-cell
offsets and on some rungs round to DIFFERENT node-row counts — 12 vs 13 at
n = 4, 24 vs 25 at n = 8.  ``assert_realized`` refuses such a board, because
the two feeds are one declared object and a mesh that realizes them a row
apart makes a symmetric board asymmetric.  ``LADDER_M`` below carries the
measured n = 2 … 12 table.  The coarsest rung, h/3, is below the four normal
intervals the MSL port's own preflight asks for; it is here as the end a trend
is measured from, and nothing is judged on it.  The judged rung is h/12.

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

    pytest tests/crossval/sheen_lpf -m gpu

Set ``RFX_CROSSVAL_FIG_DIR`` to keep the |S21| figure somewhere durable;
without it the figure goes under pytest's ``tmp_path``.  Set
``RFX_SHEEN_RUNGS`` to a comma-separated list of cell sizes in metres
(e.g. ``RFX_SHEEN_RUNGS=397e-6``) to run part of the ladder — the mesh
statement and the comparison then apply to the rungs given, which is a
diagnostic, not the case's verdict.  Unset, the full ladder runs.

The two fast tests carry no mark: they build the structure and read the
reference files, and never step the FDTD.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

from tests._realized_geometry import _node_line, realized

# ---------------------------------------------------------------- geometry
# Every length in metres, every frequency in hertz.  These are the retired
# script's own constants, which are the reference maker's own constants
# (``tests/crossval/sheen_lpf/reference/make_openems_reference.py`` carries the
# same block, copied from the same place): the Elsherbeni-Demir Sec. 6.2
# reproduction of Sheen 1990, as transcribed in roseengineering/rffdtd's
# ``examples/lowpass.py``.
#
# Native Sheen frame: x_S transverse (the patch's length), y_S propagation
# (the feeds), z stack.  rfx/openEMS x is the paper's y and rfx/openEMS y is
# its x (the maker's delta 5), so everything below is in the rfx frame with
# x = 0 at the input board edge.
EPS_R = 2.2
H_SUB = 0.794e-3
W_FEED = 2.413e-3           # 50 Ω feed width
BOARD_YS = 19.472e-3        # Sheen y extent (propagation)

IN_FEED_XS_C = 0.5 * (6.650e-3 + 9.063e-3)      # 7.8565 mm
OUT_FEED_XS_C = 0.5 * (13.257e-3 + 15.670e-3)   # 14.4635 mm
PATCH_XS_LO, PATCH_XS_HI = 1.000e-3, 21.320e-3  # transverse span of the patch
PATCH_YS_LO, PATCH_YS_HI = 8.466e-3, 11.006e-3  # propagation span of the patch
IN_FEED_LEN = PATCH_YS_LO                        # 8.466 mm (edge -> patch)
OUT_FEED_LEN = BOARD_YS - PATCH_YS_HI            # 8.466 mm (patch -> edge)

EXTEND_FEED = 4.0e-3        # extra matched 50 Ω feed per side (de-embed room)
Y_CLEAR = 3.0e-3            # transverse clearance, patch edge to boundary
Z_AIR = 3.0e-3              # air above the trace
FREQ_MAX_HZ = 20.0e9        # the Sheen analysis band top
FREQ_MIN_HZ = 0.5e9
CPML_LAYERS = 8
PORT_IMPEDANCE_OHM = 50.0

PATCH_X0 = EXTEND_FEED + IN_FEED_LEN                # 12.466 mm
PATCH_LEN_PROP = PATCH_YS_HI - PATCH_YS_LO          # 2.540 mm
PATCH_X1 = PATCH_X0 + PATCH_LEN_PROP                # 15.006 mm
LX = PATCH_X1 + OUT_FEED_LEN + EXTEND_FEED          # 27.472 mm

PATCH_TRV_LEN = PATCH_XS_HI - PATCH_XS_LO           # 20.320 mm
Y_SHIFT = Y_CLEAR - PATCH_XS_LO                     # patch_lo -> Y_CLEAR


def _yS(x_sheen: float) -> float:
    """Sheen-frame x (transverse) to this domain's y."""
    return x_sheen + Y_SHIFT


IN_FEED_YC = _yS(IN_FEED_XS_C)                      # 9.8565 mm
OUT_FEED_YC = _yS(OUT_FEED_XS_C)                    # 16.4635 mm
PATCH_Y_LO, PATCH_Y_HI = _yS(PATCH_XS_LO), _yS(PATCH_XS_HI)   # 3.0, 23.32 mm
LY = PATCH_Y_HI + Y_CLEAR                           # 26.320 mm
LZ = H_SUB + Z_AIR                                  # 3.794 mm

PORT_MARGIN = 2.5e-3        # port plane distance from the x boundary (> 2 h)

# --- the probe offset -----------------------------------------------------
# ``n_probe_offset`` is a CELL count, so it has to be re-derived per rung or
# the probes move physically as the mesh refines.  Here the PHYSICAL distance
# is held and the cell count follows the rung.
#
# 5.0 mm, leader decision 2026-09-23.  The retired script ran 6 mm (30 cells at
# its dx = 200 µm) and recorded why it is not smaller: 20 cells left the probes
# inside the feed's near field and dragged the argmin onto a spurious dip at
# 7.22 GHz.  But 6 mm puts the deepest probe 953 µm (p1) and 874 µm (p2) from
# the wide section's first step, against the λ_g/4 = 1676 µm the port preflight
# asks for at 20 GHz, and the preflight warns on BOTH ports at h/2.
#
# What 5.0 mm does, measured, ON EACH RUNG — the realized offset and every
# remaining probe-clearance finding are printed per rung, and a rung whose
# window excludes 5.0 mm is reported rather than tuned:
#
#   * on the LADDER (h/3, h/5, h/7, h/12 = 19, 31, 44, 76 cells = 5.03, 4.92,
#     4.99, 5.03 mm) the preflight leaves ZERO probe-clearance findings;
#   * at h/2 — which is NOT a ladder rung — 5.0 mm rounds to 13 cells =
#     5.161 mm and ONE finding is left, on p2 only: its deepest probe sits
#     1668 µm from the wide section against the recommended 1676 µm, 8 µm
#     short.  The preflight then prints a window for p2 only, [10, 12] cells =
#     3.970-4.764 mm, and 13 cells is OUTSIDE it.  No window is printed for p1.
PROBE_OFFSET_M = 5.0e-3


def probe_offset_cells(dx: float) -> int:
    """The cell count that puts the probes ``PROBE_OFFSET_M`` upstream."""
    return max(1, int(round(PROBE_OFFSET_M / dx)))


# --- the ladder -----------------------------------------------------------
# dx = h_sub/n keeps the substrate top on a node plane (issue #723).
#
# MEASURED at build time on this board, node rows realized by the input and
# the output feed (both declared 2.413 mm wide), n = 2 … 12:
#
#     n      dx (µm)   in / out rows        n      dx (µm)   in / out rows
#     2      397.00      6 /  6             8       99.25    24 / 25
#     3      264.67      9 /  9             9       88.22    27 / 28
#     4      198.50     12 / 13            10       79.40    31 / 30
#     5      158.80     15 / 15            11       72.18    34 / 33
#     6      132.33     18 / 18            12       66.17    37 / 37
#     7      113.43     21 / 21
#
# WHY the row counts differ on some rungs, and not on others: the two feed
# centres sit 6.607 mm apart across the board, and 6.607 mm / dx is never a
# whole number of cells on an h/n mesh, so the two footprints land at DIFFERENT
# sub-cell offsets.  Each rounds to the row count its own offset gives, and the
# two agree only where the rounding happens to fall the same way — the table
# above, not a property of the mesh family.  ``assert_realized`` refuses a
# board where they differ: the two feeds are the same declared object, and a
# mesh that realizes them a row apart makes a symmetric board asymmetric and
# puts the two ports on different lines.  The retired script recorded the same
# effect at n = 4 in its own header ("12 and 13 node rows wide (2183.5 vs
# 2382.0 µm; HJ Z0 54.22 vs 51.19 ohm)") and named n = 5 and n = 6 as the
# aligned meshes that restore it.
#
# The ladder is therefore the four rungs that realize both feeds equal AND
# still refine geometrically enough to state a trend (leader decision,
# 2026-09-23).  n = 4 and n = 8 are not on it because the check above refuses
# them; n = 2 is not, because two cells across the substrate is half what the
# MSL port's own preflight asks for.
#
# THE RUNG CRITERION IS TRANSVERSE ROW EQUALITY, AND ONLY THAT.  Along x the
# two feeds are not congruent on the two coarsest rungs, and rfx's own
# preflight names it — "1 congruent-conductor group(s) realize to UNEQUAL PEC
# edge sets on this lattice … 807 vs 790 edges … slide the lattice origin by
# 80 µm".  Measured node COLUMNS and x strips, input / output feed (both
# declared 12466.0 µm long):
#
#     h/3   48 / 47 columns   12704.0 / 12439.3 µm
#     h/5   79 / 78 columns   12545.2 / 12386.4 µm
#     h/7  110 / 110 columns  12477.1 / 12477.1 µm
#     h/12 189 / 189 columns  12505.5 / 12505.5 µm
#
# So the judged rung (h/12) and its neighbour (h/7) are congruent on BOTH axes,
# and the two coarsest rungs are asymmetric by one column along the direction
# of propagation.  Every deviation above is inside one cell, which is what
# ``assert_realized`` holds the feed length to; the column counts and x strips
# are printed per rung beside the rows.  Nothing here is judged.
#
# h/3 IS below the documented MSL minimum too — the port preflight asks for at
# least 4 normal intervals between ground and trace, and 3 is 3.  It is here as
# the COARSEST RUNG OF A LADDER, the end a trend is measured from; nothing is
# judged on it.  The judged rung is h/12.
LADDER_M = (
    H_SUB / 3,    # 264.67 µm,  3 substrate cells, both feeds  9 rows
    H_SUB / 5,    # 158.80 µm,  5 substrate cells, both feeds 15 rows
    H_SUB / 7,    # 113.43 µm,  7 substrate cells, both feeds 21 rows
    H_SUB / 12,   #  66.17 µm, 12 substrate cells, both feeds 37 rows
)

# --- the sweep ------------------------------------------------------------
# rfx samples the record's OWN frequency grid, linspace(0.5, 20 GHz, 801), so
# the judged comparison needs no interpolation at all.  DFT bins cost no time
# steps.  The bin is 24.375 MHz, under a third of the 1 % frequency bar at the
# stopband null (80 MHz at 8 GHz), so the three-point parabola's grid-phase
# spread stays well inside the bar.
N_FREQS = 801
SWEEP_HZ = (FREQ_MIN_HZ, FREQ_MAX_HZ)

# --- the record length ----------------------------------------------------
# ``num_periods`` is in periods of ``freq_max`` (50 ps here).  MEASURED on this
# board at h/2 on CPU AT THE 5.0 mm PROBE OFFSET THIS CASE HOLDS (13 cells at
# that rung), ring-down settling per driven run and the stopband null:
#
#     num_periods = 20   −23.34 / −23.28 dB    null 7.7725 GHz
#     num_periods = 60   −62.53 / −63.19 dB    null 8.5607 GHz
#     num_periods = 120  −109.68 / −110.57 dB  null 8.5592 GHz
#
# 20 (the MSL notch filter case's value) does not ring this board down — the
# retired script recorded the same thing at its own 6 mm offset, −24.7 /
# −24.3 dB — so it is raised to the script's own 60, which meets the −40 dB
# rule with 22 dB to spare and puts the null within 1.5 MHz of the doubled
# record.  h/2 is not a ladder rung; it is the cheapest board on which this
# question can be asked.
NUM_PERIODS = 60.0

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
# a passive structure cannot scatter more power than it receives.  MEASURED at
# h/2 with NUM_PERIODS = 60 and the 5.0 mm probe offset: settling −62.53 /
# −63.19 dB, excess 0.0039 inside the judged band.  The excess is read over the
# JUDGED BAND, for the reason the record gives for its own passivity witness
# (``meta.passivity_witness``): outside it the record accounts for less than
# half the power it receives, and rfx's own raw extraction on the same run runs
# to 0.787 excess (worst sigma_max 1.787) at 18.64 GHz — which is not a
# statement about the band either curve is read on.
SETTLING_DB = -40.0
PASSIVITY_EXCESS_BAR = 0.01
# The openEMS record is truncated by design and carries its own N/2N witness
# instead of a decay level.  This case reports the three rungs' maxima and
# bounds them at one twentieth of the 2 dB magnitude bar — a sanity bound on
# the premise the comparison rests on, not a gate on the physics.  Recorded
# maxima: |ΔS21| ≤ 1.1e-3 dB, |ΔS11| ≤ 1.4e-3 dB.
RECORD_LENGTH_WITNESS_BAR_DB = 0.1

# --- the bands ------------------------------------------------------------
# The band both the record and this case are read on: the record's own
# ``meta.witness_band_ghz``.  Its energy sum runs 0.78–1.00 across 2–12 GHz and
# falls to 0.41 by 20 GHz, so 12 GHz is where the record stops being one a
# magnitude can be held against.
REFERENCE_BAND_HZ = (2.0e9, 12.0e9)
# The estimator windows, the record's own (``meta.null_estimator`` and
# ``meta.cutoff_estimator``).
NULL_BAND_HZ = (5.0e9, 10.0e9)
PASSBAND_HZ = (1.0e9, 4.0e9)
CUTOFF_SEARCH_FROM_HZ = 2.0e9

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_PALACE_JSON = _REFERENCE_DIR / "palace_fem.json"
_OPENEMS_JSON = _REFERENCE_DIR / "openems_sheen.json"
# The rung of the openEMS record this case is judged against, and the two that
# are its own mesh statement.
OPENEMS_JUDGED_STAGE = "stage_b_fine"
OPENEMS_MESH_STATEMENT_STAGES = ("stage_b_coarse", "stage_b_mid")


# ------------------------------------------------------------------ build
def _sim(dx: float) -> Simulation:
    """The box, the boundary and the substrate — shared by every builder."""
    sim = Simulation(
        freq_max=FREQ_MAX_HZ,
        domain=(LX, LY, LZ),
        dx=dx,
        cpml_layers=CPML_LAYERS,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("duroid", eps_r=EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (LX, LY, H_SUB)), material="duroid")
    return sim


def _add_ports(sim: Simulation, dx: float) -> None:
    """The two MSL ports, on the two x faces' feed lines.

    ``width`` and ``height`` stay DECLARED (2.413 mm, 0.794 mm).  That is the
    retired script's decision with its measurement behind it: ``width`` sets
    both the port's cross-section rows and the Hammerstad-Jensen reference
    impedance the wave split uses, and the two want different numbers.  The
    realized-vs-declared row gap is printed by ``_print_realized`` rather than
    asserted; closing it is core work (#729 class), not a case constant.
    """
    n = probe_offset_cells(dx)
    sim.add_msl_port(position=(PORT_MARGIN, IN_FEED_YC, 0.0),
                     width=W_FEED, height=H_SUB, direction="+x",
                     impedance=PORT_IMPEDANCE_OHM, eps_r_sub=EPS_R, name="p1",
                     n_probe_offset=n)
    sim.add_msl_port(position=(LX - PORT_MARGIN, OUT_FEED_YC, 0.0),
                     width=W_FEED, height=H_SUB, direction="-x",
                     impedance=PORT_IMPEDANCE_OHM, eps_r_sub=EPS_R, name="p2",
                     n_probe_offset=n)


def build(dx: float) -> Simulation:
    """The reference structure on a uniform ``dx`` mesh.

    The substrate is a lossless slab filling the board footprint; the input
    feed, the wide section and the output feed are three zero-thickness PEC
    SHEETS on the substrate-top node plane (both z corners equal — a Box drawn
    ``h -> h + dx`` would be a one-cell VOLUME with a wall on each face, #931
    §1.3, and the openEMS record draws the identical metal with zero thickness).
    The ground is the z_lo PEC face.
    """
    sim = _sim(dx)
    tz = H_SUB
    sim.add(Box((0.0, IN_FEED_YC - W_FEED / 2, tz),
                (PATCH_X0, IN_FEED_YC + W_FEED / 2, tz)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz),
                (PATCH_X1, PATCH_Y_HI, tz)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz),
                (LX, OUT_FEED_YC + W_FEED / 2, tz)), material="pec")
    _add_ports(sim, dx)
    return sim


def _build_input_feed_as_volume(dx: float) -> Simulation:
    """Same board, but the input feed drawn as a one-cell thick VOLUME.

    Only the mutation test builds it: it is the defect ``assert_realized``
    exists to refuse (two wall planes instead of one sheet plane).
    """
    sim = _sim(dx)
    tz = H_SUB
    sim.add(Box((0.0, IN_FEED_YC - W_FEED / 2, tz),
                (PATCH_X0, IN_FEED_YC + W_FEED / 2, tz + dx)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz),
                (PATCH_X1, PATCH_Y_HI, tz)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz),
                (LX, OUT_FEED_YC + W_FEED / 2, tz)), material="pec")
    _add_ports(sim, dx)
    return sim


def _build_input_feed_lifted(dx: float, cells: int) -> Simulation:
    """The input feed pulled ``cells`` cells back from the wide section.

    Both ends are moved, so the DECLARED length is the shipped one (the far
    end then runs past the x = 0 board edge and the realized footprint is
    clipped there instead).  Nothing in ``assert_realized`` measures a feed's
    length, so the joint is the only check that can tell this board from the
    shipped one — which is the point of the mutation.

    A 50 Ω line that does not reach the low-impedance section is not this
    filter: it is an open-ended stub beside it.
    """
    sim = _sim(dx)
    tz = H_SUB
    sim.add(Box((-cells * dx, IN_FEED_YC - W_FEED / 2, tz),
                (PATCH_X0 - cells * dx, IN_FEED_YC + W_FEED / 2, tz)),
            material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz),
                (PATCH_X1, PATCH_Y_HI, tz)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz),
                (LX, OUT_FEED_YC + W_FEED / 2, tz)), material="pec")
    _add_ports(sim, dx)
    return sim


def _build_input_feed_short(dx: float, cells: int) -> Simulation:
    """The input feed started ``cells`` cells inside the board, its far end at
    the wide section where :func:`build` puts it.

    The joint is intact and the width is right; only the length is wrong, and
    the port then drives a line that is not there for the first ``cells``·dx
    of its run.
    """
    sim = _sim(dx)
    tz = H_SUB
    sim.add(Box((cells * dx, IN_FEED_YC - W_FEED / 2, tz),
                (PATCH_X0, IN_FEED_YC + W_FEED / 2, tz)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz),
                (PATCH_X1, PATCH_Y_HI, tz)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz),
                (LX, OUT_FEED_YC + W_FEED / 2, tz)), material="pec")
    _add_ports(sim, dx)
    return sim


def _build_patch_short(dx: float, cells: int) -> Simulation:
    """The wide section drawn ``cells`` cells short along x (propagation).

    Its length is what sets the shunt capacitance, so a board whose realized
    section is not the declared 2.540 mm is a different filter.
    """
    sim = _sim(dx)
    tz = H_SUB
    sim.add(Box((0.0, IN_FEED_YC - W_FEED / 2, tz),
                (PATCH_X0, IN_FEED_YC + W_FEED / 2, tz)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz),
                (PATCH_X1 - cells * dx, PATCH_Y_HI, tz)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz),
                (LX, OUT_FEED_YC + W_FEED / 2, tz)), material="pec")
    _add_ports(sim, dx)
    return sim


def _build_input_feed_narrow(dx: float, cells: int = 1) -> Simulation:
    """The input feed drawn ``cells`` cells narrower than the output feed.

    Two nominally identical 50 Ω feeds that realize different row counts are a
    symmetric board made asymmetric; the two ports then sit on different lines.
    """
    sim = _sim(dx)
    tz = H_SUB
    half = W_FEED / 2 - cells * dx
    sim.add(Box((0.0, IN_FEED_YC - half, tz),
                (PATCH_X0, IN_FEED_YC + half, tz)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz),
                (PATCH_X1, PATCH_Y_HI, tz)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz),
                (LX, OUT_FEED_YC + W_FEED / 2, tz)), material="pec")
    _add_ports(sim, dx)
    return sim


# -------------------------------------------------------- realized metal
_SHEET_NAMES = ("input feed", "wide section", "output feed")


def realized_geometry(sim: Simulation) -> dict:
    """What the lattice built, read from the product's own realization owner.

    ``tests/_realized_geometry.realized`` assembles the simulation without a
    time step and hands back the realized PEC edge masks and the declared sheet
    footprints, so every number here is the solver's, not a rule re-derived in
    the test.

    A zero-thickness sheet IS its set of tangential edges, and *n* footprint
    node rows carry *n* − 1 edges.  Every span is therefore reported twice:
    ``*_node_span_m`` = (n − 1)·dx is the geometric extent between the
    outermost nodes, ``*_strip_m`` = n·dx is the strip width a quasi-TEM
    formula takes and the one the declared dimensions are compared against.

    The joints are read from the Ex edge mask along each feed's centre row:
    the feed's own columns and the wide section's columns must carry an
    unbroken run of tangential edges.  A feed drawn clear of the section
    realizes a gap there while its length, measured end to end, is unchanged.
    """
    from rfx.boundaries.pec import realized_wall_planes

    rz = realized(sim)
    grid = rz.grid
    mx, _my, _mz = (np.asarray(e) for e in rz.edge_masks)
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
        "n_probe_offset": probe_offset_cells(dx),
        "probe_offset_realized_m": probe_offset_cells(dx) * dx,
    }
    if len(planes) != 1 or len(rz.sheets) != 3:
        return out

    k = planes[0]
    out["sheet_plane_k"] = k
    out["sheet_plane_z_m"] = float(zs[k])

    spans = []
    for sp in rz.sheets:
        fp = np.asarray(sp.footprint)
        idx = np.where(fp)
        i0, i1 = int(idx[0].min()), int(idx[0].max())
        j0, j1 = int(idx[1].min()), int(idx[1].max())
        spans.append({
            "cols": (i0, i1), "rows": (j0, j1),
            "n_cols": i1 - i0 + 1, "n_rows": j1 - j0 + 1,
            "x_m": (float(xs[i0]), float(xs[i1])),
            "y_m": (float(ys[j0]), float(ys[j1])),
            "x_node_span_m": float(xs[i1] - xs[i0]),
            "y_node_span_m": float(ys[j1] - ys[j0]),
            "x_strip_m": (i1 - i0 + 1) * dx,
            "y_strip_m": (j1 - j0 + 1) * dx,
        })
    in_feed, patch, out_feed = spans
    out.update(in_feed=in_feed, patch=patch, out_feed=out_feed)

    # The joints, from the realized Ex edges on each feed's centre row.
    # ``Ex[i, j, k]`` spans node column i -> i+1 on row j, so a run of columns
    # lo..hi-1 is what joins node lo to node hi without a break.
    pi0, pi1 = patch["cols"]
    for name, feed, (lo, hi) in (
        ("in", in_feed, (in_feed["cols"][0], pi1)),
        ("out", out_feed, (pi0, out_feed["cols"][1])),
    ):
        j0, j1 = feed["rows"]
        jc = (j0 + j1) // 2
        run = mx[lo:hi, jc, k]
        out[f"{name}_feed_joint_row"] = jc
        out[f"{name}_feed_joint_cols"] = (lo, hi)
        out[f"{name}_feed_joint_edges"] = (int(run.sum()), int(run.size))
        out[f"{name}_feed_joined"] = bool(run.size) and bool(run.all())
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
            "sheet plane. The two feeds and the wide section are declared as "
            "zero-thickness SHEETS on the substrate top; two planes is what a "
            "one-cell VOLUME realizes (#931 §1.3).")
    if g["n_volume_cells"] != 0:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_volume_cells']} PEC VOLUME cell(s) were "
            "realized; this board declares sheets only.")
    if g["n_declared_sheets"] != 3:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_declared_sheets']} PEC sheet(s) were "
            "classified, expected 3 (input feed, wide section, output feed).")
    dz = abs(g["sheet_plane_z_m"] - H_SUB)
    if dz >= 1e-12:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the sheet plane is at "
            f"z={g['sheet_plane_z_m']*1e6:.3f}µm, {dz*1e6:.3f}µm off the "
            f"declared substrate top {H_SUB*1e6:.3f}µm.")
    if g["in_feed"]["n_rows"] != g["out_feed"]["n_rows"]:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the two 50 Ω feeds realize "
            f"{g['in_feed']['n_rows']} and {g['out_feed']['n_rows']} node rows "
            f"({g['in_feed']['y_strip_m']*1e6:.1f} vs "
            f"{g['out_feed']['y_strip_m']*1e6:.1f} µm strip). They are the same "
            "declared object at different sub-cell offsets; a mesh that "
            "realizes them a row apart makes a symmetric board asymmetric and "
            "puts the two ports on different lines.")
    for name, feed in (("input", g["in_feed"]), ("output", g["out_feed"])):
        if abs(feed["y_strip_m"] - W_FEED) > dx:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the {name} feed's realized strip width "
                f"{feed['y_strip_m']*1e6:.1f}µm is more than one cell "
                f"({dx*1e6:.2f}µm) from the declared {W_FEED*1e6:.0f}µm "
                f"(node span {feed['y_node_span_m']*1e6:.1f}µm over "
                f"{feed['n_rows']} rows).")
    # And each feed's LENGTH along x. Without this a feed that starts well
    # inside the board — past its own port plane — realizes the right width,
    # joins the wide section and passes everything above, while the port drives
    # a line that is not there for the first several millimetres.
    for name, feed, declared in (
        ("input", g["in_feed"], PATCH_X0),
        ("output", g["out_feed"], LX - PATCH_X1),
    ):
        if abs(feed["x_strip_m"] - declared) > dx:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the {name} feed's realized length "
                f"{feed['x_strip_m']*1e6:.1f}µm is more than one cell "
                f"({dx*1e6:.2f}µm) from the declared {declared*1e6:.1f}µm "
                f"(node span {feed['x_node_span_m']*1e6:.1f}µm over "
                f"{feed['n_cols']} columns, x "
                f"{feed['x_m'][0]*1e3:.4f}–{feed['x_m'][1]*1e3:.4f} mm). The "
                "feed length is the matched 50 Ω line between the port plane "
                "and the first step; a short one leaves the port driving a "
                "line that is not there.")
    for name, got, declared in (
        ("x (propagation)", g["patch"]["x_strip_m"], PATCH_LEN_PROP),
        ("y (transverse)", g["patch"]["y_strip_m"], PATCH_TRV_LEN),
    ):
        if abs(got - declared) > dx:
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the wide section's realized {name} strip "
                f"{got*1e6:.1f}µm is more than one cell ({dx*1e6:.2f}µm) from "
                f"the declared {declared*1e6:.1f}µm. That dimension sets the "
                "shunt capacitance, so this is a different filter.")
    for name, key in (("input", "in"), ("output", "out")):
        if not g[f"{key}_feed_joined"]:
            on, total = g[f"{key}_feed_joint_edges"]
            raise AssertionError(
                f"dx={dx*1e6:.2f}µm: the {name} feed is not joined to the wide "
                f"section: along its centre node row {g[f'{key}_feed_joint_row']} "
                f"only {on} of the {total} tangential PEC edges between node "
                f"columns {g[f'{key}_feed_joint_cols']} are realized. A 50 Ω "
                "line that does not reach the low-impedance section is an "
                "open-ended stub beside it, not this filter.")
    if g["n_ports"] != 2:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: {g['n_ports']} MSL port(s) registered, expected 2.")
    return g


def _print_realized(g: dict, dx: float) -> None:
    print(f"  realized at dx = {dx*1e6:.3f} µm:")
    print(f"    grid {g['grid_shape']}  cells {g['n_cells']}  "
          f"substrate {int(round(H_SUB/dx))} cells")
    print(f"    sheet plane k={g.get('sheet_plane_k')} "
          f"z={g.get('sheet_plane_z_m', float('nan'))*1e6:.3f} µm "
          f"(declared {H_SUB*1e6:.1f} µm); declared sheets "
          f"{g['n_declared_sheets']}; PEC volume cells {g['n_volume_cells']}")
    for name, key, dec_x, dec_y in (
        ("input feed  ", "in_feed", PATCH_X0, W_FEED),
        ("wide section", "patch", PATCH_LEN_PROP, PATCH_TRV_LEN),
        ("output feed ", "out_feed", LX - PATCH_X1, W_FEED),
    ):
        s = g.get(key)
        if s is None:
            continue
        print(f"    {name}: cols {s['cols']} n={s['n_cols']} rows {s['rows']} "
              f"n={s['n_rows']}")
        print(f"                  x strip {s['x_strip_m']*1e6:9.1f} µm "
              f"(node span {s['x_node_span_m']*1e6:9.1f}, declared "
              f"{dec_x*1e6:9.1f}); y strip {s['y_strip_m']*1e6:9.1f} µm "
              f"(node span {s['y_node_span_m']*1e6:9.1f}, declared "
              f"{dec_y*1e6:9.1f})")
    i, o = g.get("in_feed"), g.get("out_feed")
    if i and o:
        # The two feeds are one declared object. Row equality is what the rung
        # criterion holds; the column counts are printed beside it because on
        # some rungs they differ by one and rfx's own preflight names that as a
        # congruent-conductor group realizing unequal PEC edge sets.
        print(f"    the two 50 Ω feeds: rows {i['n_rows']} / {o['n_rows']} "
              f"({'equal' if i['n_rows'] == o['n_rows'] else 'DIFFER'}); "
              f"columns {i['n_cols']} / {o['n_cols']} "
              f"({'equal' if i['n_cols'] == o['n_cols'] else 'differ'}); "
              f"x strips {i['x_strip_m']*1e6:.1f} / {o['x_strip_m']*1e6:.1f} µm "
              f"(declared {PATCH_X0*1e6:.1f} / {(LX-PATCH_X1)*1e6:.1f})")
    for name, key in (("input", "in"), ("output", "out")):
        if f"{key}_feed_joined" in g:
            on, total = g[f"{key}_feed_joint_edges"]
            print(f"    {name} feed joint: row {g[f'{key}_feed_joint_row']}, "
                  f"columns {g[f'{key}_feed_joint_cols']}, {on}/{total} "
                  f"tangential edges -> joined={g[f'{key}_feed_joined']}")
    print(f"    MSL ports {g['n_ports']}, n_probe_offset "
          f"{g['n_probe_offset']} cells = "
          f"{g['probe_offset_realized_m']*1e3:.3f} mm upstream "
          f"(declared {PROBE_OFFSET_M*1e3:.1f} mm)")


def _print_port_rows(sim: Simulation) -> None:
    """The declared-vs-realized port aperture, reported and never asserted.

    ``add_msl_port`` is handed the DECLARED feed width while the sheet
    realizes another; the port rounds each face to the nearest node while a
    sheet footprint is closed [lo, hi], so the port can integrate a row that
    carries no metal. Printing it keeps the open gap (#729 class) readable.
    """
    from rfx.sources.msl_port import msl_cross_section_span, msl_port_from_entry

    grid = sim._build_grid()
    for pe in getattr(sim, "_msl_ports", []) or []:
        span = msl_cross_section_span(grid, msl_port_from_entry(pe))
        lo, hi = int(span["w_lo"]), int(span["w_hi"])
        print(f"    port {getattr(pe, 'name', '?')}: cross-section node rows "
              f"{(lo, hi)} ({hi - lo + 1} rows)")


# ------------------------------------------------------------------- solve
def run_rung(dx: float) -> dict:
    """Preflight, solve, and hand back the curve with its two witnesses."""
    sim = build(dx)
    g = assert_realized(sim, dx)
    print(f"  box {LX*1e3:.3f} × {LY*1e3:.3f} × {LZ*1e3:.3f} mm, the openEMS "
          "record's own box")
    _print_realized(g, dx)
    _print_port_rows(sim)

    print(f"  preflight at dx = {dx*1e6:.3f} µm:")
    report = sim.preflight()
    print(f"    {len(report)} finding(s)")
    clearance = [line for line in report if "probe" in line and "reflector" in line]
    for i, line in enumerate(report):
        print(f"    [{i}] {line}")
    print(f"    port probe-clearance findings: {len(clearance)} "
          f"({'clear' if not clearance else 'NOT clear'})")
    # Name what is left and, when the preflight prints its own admissible
    # window, quote it in cells AND in millimetres beside the offset this case
    # holds. A rung whose window excludes PROBE_OFFSET_M is reported, not tuned.
    for line in clearance:
        port = re.search(r"MSL port '([^']+)'", line)
        dist = re.search(r"sits (\d+)µm from", line)
        want = re.search(r"recommended ≥ (\d+)µm", line)
        print(f"      NOT clear: port {port.group(1) if port else '?'} — deepest "
              f"probe {dist.group(1) if dist else '?'} µm from the wide section, "
              f"recommended ≥ {want.group(1) if want else '?'} µm")
    for line in report:
        win = re.search(r"compliant n_probe_offset interval ≈ \[(\d+), (\d+)\] cells", line)
        port = re.search(r"MSL port '([^']+)'", line)
        if win:
            lo_c, hi_c = int(win.group(1)), int(win.group(2))
            print(f"      preflight's own compliant n_probe_offset window for "
                  f"port {port.group(1) if port else '?'}: [{lo_c}, {hi_c}] cells "
                  f"= {lo_c*dx*1e3:.3f}–{hi_c*dx*1e3:.3f} mm; this case holds "
                  f"{PROBE_OFFSET_M*1e3:.1f} mm "
                  f"({g['n_probe_offset']} cells, "
                  f"{'inside' if lo_c <= g['n_probe_offset'] <= hi_c else 'OUTSIDE'})")

    freqs_in = np.linspace(SWEEP_HZ[0], SWEEP_HZ[1], N_FREQS)
    t0 = time.perf_counter()
    res = sim.compute_msl_s_matrix(freqs=freqs_in, num_periods=NUM_PERIODS,
                                   enforce_passivity=False)
    wall_s = time.perf_counter() - t0

    freqs = np.asarray(res.freqs, dtype=float)
    s = np.asarray(res.S)
    settling = (np.asarray(res.settling_db, dtype=float)
                if res.settling_db is not None else None)
    excess = (np.asarray(res.sigma_max_excess, dtype=float)
              if res.sigma_max_excess is not None else None)

    lo, hi = REFERENCE_BAND_HZ
    in_band = (freqs >= lo) & (freqs <= hi)
    out = dict(
        dx_m=dx, freqs_hz=freqs, s11=s[0, 0, :], s21=s[1, 0, :],
        z0=np.asarray(res.Z0), settling_db=settling,
        sigma_max_excess=excess, wall_s=wall_s,
        n_cells=g["n_cells"], grid_shape=g["grid_shape"], realized=g,
        n_preflight=len(report), n_probe_clearance_findings=len(clearance),
    )

    print(f"  wall time {wall_s:.1f} s, {g['n_cells']} cells, "
          f"num_periods={NUM_PERIODS}, n_freqs={N_FREQS}")
    print(f"  settling_db per driven run: "
          f"{None if settling is None else np.round(settling, 3).tolist()}")
    band_excess = None if excess is None else float(np.max(excess[in_band]))
    full_excess = None if excess is None else float(np.max(excess))
    out["max_excess_in_band"] = band_excess
    out["max_excess_full_sweep"] = full_excess
    esum = np.abs(out["s11"]) ** 2 + np.abs(out["s21"]) ** 2
    out["energy_sum"] = esum
    out["energy_sum_band"] = (float(esum[in_band].min()), float(esum[in_band].max()))
    print(f"  max sigma excess: {band_excess} inside "
          f"{lo/1e9:.0f}–{hi/1e9:.0f} GHz, {full_excess} over the whole "
          f"{SWEEP_HZ[0]/1e9:.1f}–{SWEEP_HZ[1]/1e9:.0f} GHz sweep")
    print(f"  |S11|²+|S21|² over {lo/1e9:.0f}–{hi/1e9:.0f} GHz: "
          f"{out['energy_sum_band'][0]:.4f} – {out['energy_sum_band'][1]:.4f}")
    print(f"  fitted line Re(Z0), median over the band: "
          f"{float(np.median(np.asarray(out['z0'], dtype=complex).ravel().real)):.1f} Ω "
          f"(port reference {PORT_IMPEDANCE_OHM:.0f} Ω)")

    if settling is None:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the result carries no settling witness; the "
            "ring-down cannot be judged and no S value may be quoted.")
    if float(np.max(settling)) > SETTLING_DB:
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: worst ring-down settling "
            f"{float(np.max(settling)):.2f} dB is above {SETTLING_DB:.0f} dB "
            f"(per run {np.round(settling, 3).tolist()}). The record ended "
            "before the board rang down, so this rung's S is "
            "truncation-suspect. Raise NUM_PERIODS; do not compare it.")
    if band_excess is not None and band_excess > PASSIVITY_EXCESS_BAR:
        k = int(np.argmax(np.where(in_band, excess, -np.inf)))
        raise AssertionError(
            f"dx={dx*1e6:.2f}µm: the raw extraction exceeds the passive bound "
            f"by {band_excess:.4f} at {freqs[k]/1e9:.4f} GHz inside "
            f"{lo/1e9:.0f}–{hi/1e9:.0f} GHz (bar {PASSIVITY_EXCESS_BAR}). A "
            "passive board cannot scatter more power than it receives, so this "
            "is a measurement artefact; do not compare this rung.")
    return out


# --------------------------------------------------------------- estimators
def _load_spectral_features():
    """The repository's one estimator module,
    ``validation/crossval/comparators/spectral_features.py`` — the module that
    exists so the cases and the reference producers cannot drift apart.  Loaded
    by path because ``validation/`` is not a package; it imports numpy only."""
    import importlib.util

    path = _REPO_ROOT / "validation" / "crossval" / "comparators" / "spectral_features.py"
    spec = importlib.util.spec_from_file_location("_sheen_spectral_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sf = _load_spectral_features()


def stopband_null(freqs, s21_mag, lo=NULL_BAND_HZ[0], hi=NULL_BAND_HZ[1]) -> dict:
    """The deepest |S21| minimum inside ``[lo, hi]`` (same units as ``freqs``).

    The shared estimator's rule: the deepest bin in the band, then the vertex
    of the parabola through it and its two neighbours fitted in log|S| (the
    same vertex as a fit in dB) — the rule the openEMS record's own ``null``
    block was computed with.

    This board's stopband is a DOUBLE transmission zero near 7 and 8 GHz, so
    the number below names ONE member of a pair whose members can swap places
    between meshes and solvers.  The whole curve is printed beside it.
    """
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s21_mag, dtype=float)
    ext = _sf.refined_extremum(f, s, lo, hi, transform="log")
    return {
        "index": int(ext["index"]),
        "bin_f": float(ext["bin_f"]),
        "f": float(ext["refined_f"]),
        "sub_bin_shift": float(ext["sub_bin_shift"]),
        "depth_db": float(ext["depth_db"]),
        "bin_width": float(ext["bin_width"]),
    }


def passband_cutoff_3db(freqs, s21_mag) -> dict:
    """The passband mean |S21| (1–4 GHz) and the first falling crossing of that
    mean divided by √2 above 2 GHz — the record's own ``passband`` and
    ``cutoff_3db`` rule, linearly interpolated between the bracketing bins."""
    f = np.asarray(freqs, dtype=float)
    s = np.asarray(s21_mag, dtype=float)
    pb = (f >= PASSBAND_HZ[0]) & (f <= PASSBAND_HZ[1])
    mean_lin = float(np.mean(s[pb]))
    level = mean_lin / np.sqrt(2.0)
    fc = _sf.level_crossing(f, s, level, f_min=CUTOFF_SEARCH_FROM_HZ)
    return {
        "n_bins": int(np.count_nonzero(pb)),
        "mean_linear": mean_lin,
        "mean_db": float(20.0 * np.log10(mean_lin + 1e-30)),
        "level_linear": level,
        "f_3db": None if fc is None else float(fc),
    }


# ------------------------------------------------------------- references
def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _db(mag) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(mag, dtype=float), 1e-300))


def _interp_db(freqs_hz, s21_db, ref_f_ghz) -> np.ndarray:
    """rfx's |S21| in dB sampled at the reference frequencies (dB is what the
    2 dB bar is stated in, so the interpolation happens in dB).  On the openEMS
    record the two grids are identical and this is a copy."""
    return np.interp(np.asarray(ref_f_ghz, dtype=float) * 1e9,
                     np.asarray(freqs_hz, dtype=float),
                     np.asarray(s21_db, dtype=float),
                     left=np.nan, right=np.nan)


def _compare(label: str, rung: dict, ref_f_ghz, ref_s21_mag) -> dict:
    """ΔdB of rfx − reference at every reference frequency inside the band both
    curves cover where BOTH are above the deep-null level, plus the
    null-frequency distance.

    The deep-null exclusion is two-sided on purpose: a bin where one curve is
    in its null and the other is not is the two nulls sitting at different
    frequencies, which the frequency bar already judges; counting it again as a
    magnitude error would score one offset twice."""
    ref_f_hz = np.asarray(ref_f_ghz, dtype=float) * 1e9
    ref_db = _db(ref_s21_mag)
    ours = _interp_db(rung["freqs_hz"], _db(np.abs(rung["s21"])), ref_f_ghz)
    lo, hi = REFERENCE_BAND_HZ
    in_band = (ref_f_hz >= lo) & (ref_f_hz <= hi)
    keep = (in_band & (ref_db >= DEEP_NULL_DB) & (ours >= DEEP_NULL_DB)
            & np.isfinite(ours))
    delta = ours - ref_db
    ref_null = stopband_null(ref_f_hz, ref_s21_mag)
    our_null = stopband_null(rung["freqs_hz"], np.abs(rung["s21"]))
    ref_cut = passband_cutoff_3db(ref_f_hz, ref_s21_mag)
    our_cut = passband_cutoff_3db(rung["freqs_hz"], np.abs(rung["s21"]))
    max_abs = float(np.max(np.abs(delta[keep]))) if keep.any() else float("nan")
    worst = int(np.argmax(np.where(keep, np.abs(delta), -np.inf))) if keep.any() else -1
    return dict(
        label=label, ref_f_ghz=np.asarray(ref_f_ghz, float), ref_db=ref_db,
        ours_db=ours, delta_db=delta, compared=keep, in_band=in_band,
        n_in_band=int(in_band.sum()), n_compared=int(keep.sum()),
        max_abs_delta_db=max_abs, worst_index=worst,
        ref_null_ghz=ref_null["f"] / 1e9, our_null_ghz=our_null["f"] / 1e9,
        ref_null_depth_db=ref_null["depth_db"],
        our_null_depth_db=our_null["depth_db"],
        null_pct=100.0 * abs(our_null["f"] - ref_null["f"]) / ref_null["f"],
        ref_cutoff=ref_cut, our_cutoff=our_cut,
    )


_MAX_TABLE_ROWS = 80


def _print_comparison(c: dict) -> None:
    lo, hi = REFERENCE_BAND_HZ
    print(f"  --- {c['label']} ---")
    print(f"    stopband null: rfx {c['our_null_ghz']:.4f} GHz "
          f"({c['our_null_depth_db']:.2f} dB) vs reference "
          f"{c['ref_null_ghz']:.4f} GHz ({c['ref_null_depth_db']:.2f} dB) -> "
          f"{c['null_pct']:.3f} %")
    rc, oc = c["ref_cutoff"], c["our_cutoff"]
    if rc["f_3db"] and oc["f_3db"]:
        print(f"    −3 dB corner (reported): rfx {oc['f_3db']/1e9:.4f} GHz vs "
              f"reference {rc['f_3db']/1e9:.4f} GHz -> "
              f"{100.0*abs(oc['f_3db']-rc['f_3db'])/rc['f_3db']:.3f} %; "
              f"passband mean {oc['mean_db']:.3f} vs {rc['mean_db']:.3f} dB")
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
    raw = os.environ.get("RFX_SHEEN_RUNGS", "").strip()
    if not raw:
        return LADDER_M
    return tuple(float(v) for v in raw.split(",") if v.strip())


def _print_record_length_witness(openems: dict) -> float:
    """The record's own N/2N witness, printed per rung and bounded once.

    The record is truncated by design (the box energy sits flat at about
    −31 dB from N to 2N steps), so what says the declared length is long
    enough is not a decay level but how far the S-parameters move when the
    record is doubled.  Reported here from the record; the return value is the
    worst of the six numbers, which the caller holds against
    ``RECORD_LENGTH_WITNESS_BAR_DB``.
    """
    w = openems["meta"]["record_length_witness"]
    lengths = openems["meta"]["record_length_s_per_rung"]
    print("\n  the openEMS record's own record-length witness (N vs 2N steps, "
          f"{w['coarse']['band_ghz'][0]:.0f}–{w['coarse']['band_ghz'][1]:.0f} GHz, "
          f"both curves above {w['coarse']['floor_db']:.0f} dB):")
    worst = 0.0
    for rung in ("coarse", "mid", "fine"):
        r = w[rung]
        worst = max(worst, abs(r["max_abs_delta_s21_db"]),
                    abs(r["max_abs_delta_s11_db"]))
        print(f"    {rung:6s} {r['n_steps']:7d} -> {r['n2_steps']:7d} steps "
              f"({lengths[rung]*1e9:.3f} ns): |ΔS21| ≤ "
              f"{r['max_abs_delta_s21_db']:.2e} dB at "
              f"{r['f_ghz_at_max_abs_delta_s21']:.4f} GHz over "
              f"{r['s21_bins_compared']} bins, |ΔS11| ≤ "
              f"{r['max_abs_delta_s11_db']:.2e} dB at "
              f"{r['f_ghz_at_max_abs_delta_s11']:.4f} GHz over "
              f"{r['s11_bins_compared']} bins")
    spread = openems["meta"]["record_length_s_spread_pct"]
    print(f"    the three declared lengths agree within {spread:.3f} %; worst "
          f"|Δ| over all six numbers {worst:.2e} dB "
          f"(bar {RECORD_LENGTH_WITNESS_BAR_DB} dB — a sanity bound on the "
          f"premise, one twentieth of the {MAG_BAR_DB:.0f} dB magnitude bar)")
    return worst


@pytest.mark.gpu
@pytest.mark.slow
def test_sheen_lpf_matches_the_openems_tutorial_reference(tmp_path):
    """The mesh ladder, the convergence statement, then the comparison."""
    rungs = _rungs()
    print(f"\nThe Sheen low-pass filter — ladder "
          f"{[f'{d*1e6:.2f}µm' for d in rungs]}")
    print(f"  board: two {W_FEED*1e3:.3f} mm 50 Ω feeds at y "
          f"{IN_FEED_YC*1e3:.4f} and {OUT_FEED_YC*1e3:.4f} mm, joined by a "
          f"{PATCH_TRV_LEN*1e3:.3f} × {PATCH_LEN_PROP*1e3:.3f} mm wide section "
          f"at x {PATCH_X0*1e3:.3f}–{PATCH_X1*1e3:.3f} mm; substrate εr "
          f"{EPS_R}, h {H_SUB*1e3:.3f} mm; box {LX*1e3:.3f} × {LY*1e3:.3f} × "
          f"{LZ*1e3:.3f} mm; ports on both x faces, {PORT_MARGIN*1e3:.1f} mm in")

    results = []
    for dx in rungs:
        print(f"\n=== rung dx = {dx*1e6:.3f} µm ===")
        r = run_rung(dx)
        s21_db = _db(np.abs(r["s21"]))
        n = stopband_null(r["freqs_hz"], np.abs(r["s21"]))
        cut = passband_cutoff_3db(r["freqs_hz"], np.abs(r["s21"]))
        r["null"] = n
        r["cutoff"] = cut
        r["s21_db"] = s21_db
        print(f"  stopband null {n['f']/1e9:.5f} GHz (bin "
              f"{n['bin_f']/1e9:.5f}, sub-bin shift {n['sub_bin_shift']:+.3f}), "
              f"depth {n['depth_db']:.2f} dB — the deepest minimum in "
              f"{NULL_BAND_HZ[0]/1e9:.0f}–{NULL_BAND_HZ[1]/1e9:.0f} GHz; this "
              "board's stopband is a double zero near 7 and 8 GHz, so read the "
              "curve for both")
        print(f"  passband mean {cut['mean_db']:.3f} dB over "
              f"{PASSBAND_HZ[0]/1e9:.0f}–{PASSBAND_HZ[1]/1e9:.0f} GHz "
              f"({cut['n_bins']} bins); −3 dB corner "
              + ("none in the sweep" if cut["f_3db"] is None
                 else f"{cut['f_3db']/1e9:.5f} GHz") + " (reported)")
        print("  |S21| on the rfx frequency grid — f_GHz, |S21| dB, |S11| dB, "
              "|S11|²+|S21|²:")
        s11_db = _db(np.abs(r["s11"]))
        for f, a, b, e in zip(r["freqs_hz"], s21_db, s11_db, r["energy_sum"]):
            print(f"    {f/1e9:8.5f} {a:10.3f} {b:10.3f} {e:10.4f}")
        results.append(r)

    palace = _load(_PALACE_JSON)
    openems = _load(_OPENEMS_JSON)
    judged_stage = openems[OPENEMS_JUDGED_STAGE]

    # ---- (a) the reference's own premise ----------------------------------
    worst_record_length = _print_record_length_witness(openems)

    # ---- (b) mesh statement: the null must settle along the ladder --------
    nulls = [r["null"]["f"] for r in results]
    print("\n  mesh statement — stopband null along the rfx ladder:")
    for r in results:
        print(f"    dx {r['dx_m']*1e6:8.3f} µm  {r['null']['f']/1e9:.5f} GHz  "
              f"{r['null']['depth_db']:7.2f} dB  {r['n_cells']} cells  "
              f"{r['wall_s']:.1f} s")
    if len(nulls) >= 2:
        last_two_pct = 100.0 * abs(nulls[-1] - nulls[-2]) / nulls[-2]
        print(f"    last two rungs differ by {last_two_pct:.3f} % "
              f"(bar {LADDER_AGREEMENT*100:.0f} %)")
    else:
        last_two_pct = float("nan")

    # The reference's own mesh statement, read off its record.
    stages = openems["meta"]["stages"]
    print("\n  the openEMS record's own mesh statement:")
    for name in OPENEMS_MESH_STATEMENT_STAGES + (OPENEMS_JUDGED_STAGE,):
        st = openems[name]
        ms = stages[name]
        print(f"    {name:16s} {ms['resolution_um']:7.2f} µm  "
              f"{ms['mesh_realized']['substrate_z_cells_realized']:2d} substrate "
              f"cells  {ms['mesh_realized']['n_cells']:9d} cells  null "
              f"{st['null']['refined_f_ghz']:.5f} GHz "
              f"{st['null']['depth_db']:7.2f} dB  passband "
              f"{st['passband']['mean_db']:7.3f} dB  −3 dB "
              f"{st['cutoff_3db']['f_ghz']:.4f} GHz  Re(Z0) "
              f"{st['re_z0_median_ohm']:.2f} Ω  energy sum "
              f"{st['min_energy_sum_band']:.4f}–{st['max_energy_sum_band']:.4f}")
    ref_nulls = [openems[n]["null"]["refined_f_ghz"]
                 for n in OPENEMS_MESH_STATEMENT_STAGES + (OPENEMS_JUDGED_STAGE,)]
    print(f"    coarse -> mid {100.0*(ref_nulls[1]-ref_nulls[0])/ref_nulls[0]:+.3f} %, "
          f"mid -> fine {100.0*(ref_nulls[2]-ref_nulls[1])/ref_nulls[1]:+.3f} %")

    # A partial ladder shows no convergence trend, so it states no verdict:
    # the distances below are still printed — they are the diagnostic the env
    # override exists for — and the test skips at the end instead of passing
    # or failing.
    # Any rung set that is not the ladder itself, in order: a SUBSET is
    # partial, and so is a same-length set of other cell sizes.
    partial = tuple(rungs) != LADDER_M
    monotone = (all(b > a for a, b in zip(nulls, nulls[1:]))
                or all(b < a for a, b in zip(nulls, nulls[1:])))

    # ---- (c) the comparison: finest rung against the openEMS fine rung -----
    # Every distance and the figure are printed BEFORE any verdict, so a run
    # that fails the mesh statement still leaves the whole curve in the log.
    finest = results[-1]
    print("\n  openEMS (judged)")
    fine = _compare(
        f"openEMS {OPENEMS_JUDGED_STAGE} "
        f"({stages[OPENEMS_JUDGED_STAGE]['resolution_um']:.2f} µm, "
        f"{stages[OPENEMS_JUDGED_STAGE]['mesh_realized']['substrate_z_cells_realized']}"
        " substrate cells) — JUDGED",
        finest, judged_stage["freqs_ghz"], judged_stage["s21_mag"])
    _print_comparison(fine)
    for name in OPENEMS_MESH_STATEMENT_STAGES:
        c = _compare(
            f"openEMS {name} ({stages[name]['resolution_um']:.2f} µm) — the "
            "reference's own mesh statement, reported",
            finest, openems[name]["freqs_ghz"], openems[name]["s21_mag"])
        _print_comparison(c)

    # ---- (d) Palace: reported, never judged -------------------------------
    print("\n  Palace FEM (reported, not judged — it does not conserve power)")
    for mesh in ("mid", "coarse"):
        rec = palace[mesh]
        m = palace["meta"]["mesh"][mesh]
        e = (np.asarray(rec["s11_mag"], float) ** 2
             + np.asarray(rec["s21_mag"], float) ** 2)
        print(f"    Palace {mesh} (lc {m['lc_mm']} mm, {m['tets']} tets): "
              f"|S11|²+|S21|² runs {float(e.min()):.6f}–{float(e.max()):.6f} "
              f"over its own {len(rec['freqs_ghz'])} points "
              f"({rec['freqs_ghz'][0]:.2f}–{rec['freqs_ghz'][-1]:.2f} GHz); "
              f"its recorded doublet is {rec['doublet']['lower']['parabolic_f_ghz']:.5f} "
              f"and {rec['doublet']['upper']['parabolic_f_ghz']:.5f} GHz")
        _print_comparison(_compare(
            f"Palace FEM, {mesh} mesh (lc {m['lc_mm']} mm) — REPORTED, NOT JUDGED",
            finest, rec["freqs_ghz"], rec["s21_mag"]))

    # ---- (e) the figure ---------------------------------------------------
    fig_path = _write_figure(results, palace, openems, tmp_path)
    print(f"\n  figure: {fig_path}")

    # ---- the premise before the comparison ---------------------------------
    assert worst_record_length <= RECORD_LENGTH_WITNESS_BAR_DB, (
        f"the openEMS record's own N/2N witness moves by "
        f"{worst_record_length:.3e} dB, above the "
        f"{RECORD_LENGTH_WITNESS_BAR_DB} dB sanity bound. The record is "
        "truncated at a declared length, so if doubling it moves the "
        "S-parameters this much the record is not long enough to be compared "
        "against.")

    if partial:
        pytest.skip(
            "RFX_SHEEN_RUNGS restricted the ladder to "
            f"{[f'{d*1e6:.2f}µm' for d in rungs]}; the stopband nulls are "
            f"{[f'{f/1e9:.5f} GHz' for f in nulls]}. A partial ladder shows no "
            "convergence trend, so the distances above are a diagnostic and "
            "this run states no verdict.")

    # ---- (b) the mesh statement comes first: without it nothing is judged --
    assert monotone, (
        "the stopband null does not move monotonically along the ladder: "
        f"{[f'{f/1e9:.5f} GHz' for f in nulls]} at "
        f"{[f'{d*1e6:.2f}µm' for d in rungs]}. Without a monotone trend the "
        "mesh is not shown to converge and the comparison says nothing. "
        "Beware: this board's stopband is a double zero, so a jump between "
        "rungs can also be the estimator naming the other member of the pair "
        "— read the printed curves before reading this as non-convergence.")
    if not last_two_pct < LADDER_AGREEMENT * 100.0:
        # PI decision 2026-09-22, carried over from the MSL notch filter case:
        # the v2 bar is never loosened, and a case that cannot meet it says so
        # with its numbers. The xfail is IMPERATIVE, so every witness above —
        # the ring-down, the passivity excess, the record-length premise — and
        # a monotonicity failure still FAIL the test; only the two comparison
        # bars below are held back. When the mesh converges (a graded mesh,
        # #810's question) this branch is no longer taken and the comparison
        # judges for real.
        pytest.xfail(
            "known limitation (PI 2026-09-22): the uniform mesh has not "
            f"converged — the two finest rungs put the stopband null "
            f"{last_two_pct:.3f} % apart (bar {LADDER_AGREEMENT*100:.0f} %): "
            f"{[f'{f/1e9:.5f} GHz' for f in nulls]}; the finest rung sits "
            f"{fine['null_pct']:.3f} % from the openEMS record's "
            f"{fine['ref_null_ghz']:.4f} GHz and its |S21| differs by up to "
            f"{fine['max_abs_delta_db']:.3f} dB. The comparison above is "
            "reported, not judged.")
    assert fine["null_pct"] < FREQ_BAR * 100.0, (
        f"the stopband null sits {fine['null_pct']:.3f} % from the openEMS "
        f"record's {fine['ref_null_ghz']:.4f} GHz (bar {FREQ_BAR*100:.0f} %); "
        f"rfx reads {fine['our_null_ghz']:.4f} GHz on the "
        f"{finest['dx_m']*1e6:.2f} µm mesh.")
    assert fine["max_abs_delta_db"] <= MAG_BAR_DB, (
        f"|S21| differs from the openEMS record by up to "
        f"{fine['max_abs_delta_db']:.3f} dB (bar {MAG_BAR_DB:.0f} dB) over the "
        f"{fine['n_compared']} reference frequencies where both curves are "
        f"above {DEEP_NULL_DB:.0f} dB.")


def _write_figure(results, palace, openems, tmp_path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.environ.get("RFX_CROSSVAL_FIG_DIR")
    directory = Path(out_dir) if out_dir else Path(tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sheen_lpf.png"

    stages = openems["meta"]["stages"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for r in results:
        ax.plot(np.asarray(r["freqs_hz"]) / 1e9, r["s21_db"],
                label=f"rfx dx {r['dx_m']*1e6:.2f} µm")
    # The mesh-statement rungs first, JUDGED on top: they agree to within a
    # line width over most of the band, and the one being judged against must
    # be the one the eye reads there.
    for name, alpha in zip(OPENEMS_MESH_STATEMENT_STAGES, (0.35, 0.5)):
        ax.plot(openems[name]["freqs_ghz"], _db(openems[name]["s21_mag"]),
                "-", lw=2.2, alpha=alpha, color="tab:green",
                label=f"openEMS {name} ({stages[name]['resolution_um']:.0f} µm)")
    ax.plot(openems[OPENEMS_JUDGED_STAGE]["freqs_ghz"],
            _db(openems[OPENEMS_JUDGED_STAGE]["s21_mag"]), "-", lw=1.0,
            color="k",
            label=f"openEMS {OPENEMS_JUDGED_STAGE} "
                  f"({stages[OPENEMS_JUDGED_STAGE]['resolution_um']:.0f} µm) — JUDGED")
    for mesh in ("mid", "coarse"):
        ax.plot(palace[mesh]["freqs_ghz"], _db(palace[mesh]["s21_mag"]), "--",
                label=f"Palace FEM {mesh} "
                      f"(lc {palace['meta']['mesh'][mesh]['lc_mm']} mm) — reported")
    ax.set_xlim(*(f / 1e9 for f in REFERENCE_BAND_HZ))
    ax.set_ylim(-70, 5)
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
    """Build-time only: the board realizes as declared, and the four defects
    ``assert_realized`` exists for are refused."""
    print(f"\n  declared board: feeds {W_FEED*1e6:.0f} µm wide at y "
          f"{IN_FEED_YC*1e3:.4f} / {OUT_FEED_YC*1e3:.4f} mm, wide section "
          f"{PATCH_TRV_LEN*1e3:.3f} × {PATCH_LEN_PROP*1e3:.3f} mm at x "
          f"{PATCH_X0*1e3:.3f}–{PATCH_X1*1e3:.3f} mm, box {LX*1e3:.3f} × "
          f"{LY*1e3:.3f} × {LZ*1e3:.3f} mm")
    # The two coarsest ladder rungs, and the mutations on the coarsest: a
    # defect that survives the coarsest mesh survives every finer one.
    dx = LADDER_M[0]
    for d in LADDER_M[:2]:
        g = assert_realized(build(d), d)
        _print_realized(g, d)

    with pytest.raises(AssertionError) as volume:
        assert_realized(_build_input_feed_as_volume(dx), dx)
    print(f"\n  input feed drawn as a one-cell volume: {volume.value}")
    assert "sheet plane" in str(volume.value)

    with pytest.raises(AssertionError) as lifted:
        assert_realized(_build_input_feed_lifted(dx, 3), dx)
    print(f"\n  input feed pulled 3 cells back, length kept: {lifted.value}")
    assert "not joined" in str(lifted.value)

    with pytest.raises(AssertionError) as short:
        assert_realized(_build_patch_short(dx, 1), dx)
    print(f"\n  wide section drawn one cell short in x: {short.value}")
    assert "wide section's realized x" in str(short.value)

    # The width, the joint and the wide section are all right here; only the
    # length of the matched line between the port plane and the first step is
    # wrong. 30 cells is 7.94 mm at this rung, so the feed starts well past its
    # own port plane at 2.5 mm.
    with pytest.raises(AssertionError) as feed_len:
        assert_realized(_build_input_feed_short(dx, 30), dx)
    print(f"\n  input feed started 30 cells inside the board edge, joint "
          f"intact: {feed_len.value}")
    assert "feed's realized length" in str(feed_len.value)

    with pytest.raises(AssertionError) as narrow:
        assert_realized(_build_input_feed_narrow(dx, 1), dx)
    print(f"\n  input feed drawn one cell narrower: {narrow.value}")
    assert "node rows" in str(narrow.value)


def test_the_reference_files_are_what_the_provenance_says():
    """The two frozen references load, carry the bands, point counts and mesh
    rungs PROVENANCE.md states, and the shared estimators reproduce the
    features each record froze.  Those numbers were computed from the records'
    own arrays with the same rules by their producers, so this is a regression
    pin of the estimators on frozen values (to 1 kHz), not an independent check
    of them and not of rfx."""
    palace = _load(_PALACE_JSON)
    openems = _load(_OPENEMS_JSON)

    # --- the openEMS record ------------------------------------------------
    assert openems["meta"]["tool"] == "openEMS"
    assert openems["meta"]["openems"]["version"] == "0.37.0"
    assert openems["meta"]["ci_runs_this"] is False
    rungs = ("stage_b_coarse", "stage_b_mid", "stage_b_fine")
    for name in rungs:
        rec = openems[name]
        assert len(rec["freqs_ghz"]) == 801
        for key in ("s11_mag", "s11_deg", "s21_mag", "s21_deg", "energy_sum",
                    "re_z0"):
            assert len(rec[key]) == 801, (name, key)
        assert rec["freqs_ghz"][0] == pytest.approx(0.5)
        assert rec["freqs_ghz"][-1] == pytest.approx(20.0)
        assert tuple(rec["witness_band_ghz"]) == (2.0, 12.0)
        assert tuple(rec["null"]["band_ghz"]) == (5.0, 10.0)
        assert tuple(rec["passband"]["band_ghz"]) == (1.0, 4.0)

    # The band this case reads the record on is the record's own.
    assert tuple(openems["meta"]["witness_band_ghz"]) == (
        REFERENCE_BAND_HZ[0] / 1e9, REFERENCE_BAND_HZ[1] / 1e9)
    # The sentence the magnitudes-only comparison rests on.
    comparability = openems["meta"]["comparability"].lower()
    assert "phase" in comparability and "magnitude" in comparability

    # The mesh rungs PROVENANCE.md tabulates, read from the record itself: the
    # substrate carries 4, 6 and 10 cells.  The fine rung DECLARED 8; CSXCAD's
    # own smoothing added two lines, and the record reports what it realized.
    stages = openems["meta"]["stages"]
    assert [stages[n]["mesh_realized"]["substrate_z_cells_realized"]
            for n in rungs] == [4, 6, 10]
    assert [stages[n]["substrate_z_cells_declared"] for n in rungs] == [4, 6, 8]
    for n in rungs:
        assert stages[n]["mesh_realized"]["substrate_top_on_a_line"] is True
    res = [stages[n]["resolution_um"] for n in rungs]
    assert res[2] == pytest.approx(res[0] / 2.0, rel=1e-6)
    assert res[1] == pytest.approx(res[0] / 2.0 ** 0.5, rel=1e-6)

    # The record is a MERGE of three cluster runs; there is no single run id,
    # and each part's is in ``meta.merged_from``.
    assert openems["run_id"] is None
    parts = openems["meta"]["merged_from"]
    assert [p["run_id"] for p in parts] == ["369367263406", "369367263407",
                                            "369367263408"]
    assert {p["maker_commit"] for p in parts} == {
        "4610452442dc560602e09987d4ec2e9cf5e5e443"}
    assert openems["meta"]["maker_commits_agree"] is True

    # Stage A is the reproduce gate — openEMS's own MSL_NotchFilter tutorial,
    # a DIFFERENT structure — and the record says so itself.
    gate = openems["meta"]["stage_a_gate"]
    assert gate["passed"] is True
    assert gate["measured_f_notch_hz"] == pytest.approx(3.672242e9, abs=1e3)
    assert gate["measured_depth_db"] < -20.0
    assert "not comparable" in openems["meta"]["stage_a_is"].lower()
    repro = openems["meta"]["stage_a_reproducibility"]
    assert repro["max_abs_delta_s21_mag"] <= repro["magnitude_tol"]
    assert repro["max_abs_delta_s11_mag"] <= repro["magnitude_tol"]
    assert repro["notch_spread_pct"] <= repro["notch_tol_pct"]

    # Every rung is truncated at a declared length, and the three lengths
    # agree; the N/2N witness is what says the length is enough.
    lengths = openems["meta"]["record_length_s_per_rung"]
    assert openems["meta"]["record_length_witness_ran"] is True
    for rung, name in zip(("coarse", "mid", "fine"), rungs):
        assert openems[name]["truncated"] is True
        assert lengths[rung] == pytest.approx(1.0e-8, rel=0.01)
        w = openems["meta"]["record_length_witness"][rung]
        assert tuple(w["band_ghz"]) == (2.0, 12.0)
        assert w["n2_steps"] == 2 * w["n_steps"]
        assert abs(w["max_abs_delta_s21_db"]) <= RECORD_LENGTH_WITNESS_BAR_DB
        assert abs(w["max_abs_delta_s11_db"]) <= RECORD_LENGTH_WITNESS_BAR_DB
    assert openems["meta"]["record_length_s_spread_pct"] < 1.0

    # A passive board cannot scatter more power than it receives.  The witness
    # the record defines is over the band it serves; the DEFICIT below unity is
    # recorded, not judged (``meta.passivity_witness``).
    tol = openems["meta"]["passivity_tol"]
    assert tol == 1.05
    for name in rungs:
        rec = openems[name]
        band = (np.asarray(rec["s11_mag"], float) ** 2
                + np.asarray(rec["s21_mag"], float) ** 2)
        f = np.asarray(rec["freqs_ghz"], float)
        in_band = (f >= 2.0) & (f <= 12.0)
        print(f"  openEMS {name}: energy sum over 2–12 GHz "
              f"{rec['min_energy_sum_band']:.6f}–{rec['max_energy_sum_band']:.6f} "
              f"(recomputed {float(band[in_band].min()):.6f}–"
              f"{float(band[in_band].max()):.6f}), whole 0.5–20 GHz grid min "
              f"{rec['min_energy_sum_full']:.6f}")
        assert rec["max_energy_sum_band"] == pytest.approx(
            float(band[in_band].max()), abs=1e-9)
        assert rec["min_energy_sum_band"] == pytest.approx(
            float(band[in_band].min()), abs=1e-9)
        assert rec["max_energy_sum_band"] <= tol

    # The features the record froze, reproduced from its own arrays with the
    # repository's shared estimators.
    for name in rungs:
        rec = openems[name]
        f_hz = np.asarray(rec["freqs_ghz"], float) * 1e9
        got = stopband_null(f_hz, rec["s21_mag"])
        cut = passband_cutoff_3db(f_hz, rec["s21_mag"])
        print(f"  openEMS {name}: null estimator {got['f']/1e9:.6f} GHz vs "
              f"record {rec['null']['refined_f_ghz']:.6f} GHz, depth "
              f"{got['depth_db']:.4f} vs {rec['null']['depth_db']:.4f} dB; "
              f"passband {cut['mean_db']:.6f} vs "
              f"{rec['passband']['mean_db']:.6f} dB; −3 dB "
              f"{cut['f_3db']/1e9:.6f} vs {rec['cutoff_3db']['f_ghz']:.6f} GHz; "
              f"Re(Z0) median {rec['re_z0_median_ohm']:.3f} Ω")
        assert got["f"] == pytest.approx(rec["null"]["refined_f_ghz"] * 1e9,
                                         abs=1e3)
        assert got["depth_db"] == pytest.approx(rec["null"]["depth_db"], abs=1e-6)
        assert cut["mean_linear"] == pytest.approx(
            rec["passband"]["mean_s21_linear"], rel=1e-12)
        assert cut["f_3db"] == pytest.approx(rec["cutoff_3db"]["f_ghz"] * 1e9,
                                             abs=1e3)
        assert rec["re_z0_median_ohm"] == pytest.approx(
            float(np.median(np.asarray(rec["re_z0"], float))), abs=1e-9)

    # --- the Palace FEM record --------------------------------------------
    assert palace["meta"]["solver"] == "palace"
    assert palace["meta"]["order"] == 2
    assert len(palace["coarse"]["freqs_ghz"]) == 81 == len(palace["coarse"]["s21_mag"])
    assert palace["coarse"]["freqs_ghz"][0] == pytest.approx(4.0)
    assert palace["coarse"]["freqs_ghz"][-1] == pytest.approx(12.0)
    assert len(palace["mid"]["freqs_ghz"]) == 51 == len(palace["mid"]["s21_mag"])
    assert palace["mid"]["freqs_ghz"][0] == pytest.approx(6.0)
    assert palace["mid"]["freqs_ghz"][-1] == pytest.approx(9.0)
    assert palace["meta"]["mesh"]["coarse"]["lc_mm"] == 0.25
    assert palace["meta"]["mesh"]["mid"]["lc_mm"] == 0.18

    # REPORTED, never judged: the record does not conserve power, so its
    # magnitudes cannot be held to the 2 dB bar.
    for mesh in ("coarse", "mid"):
        rec = palace[mesh]
        e = (np.asarray(rec["s11_mag"], float) ** 2
             + np.asarray(rec["s21_mag"], float) ** 2)
        print(f"  Palace {mesh}: |S11|²+|S21|² {float(e.min()):.6f}–"
              f"{float(e.max()):.6f} over its own {e.size} points; recorded "
              f"max {rec['max_energy_sum']:.6f}; doublet "
              f"{rec['doublet']['lower']['parabolic_f_ghz']:.5f} / "
              f"{rec['doublet']['upper']['parabolic_f_ghz']:.5f} GHz")
        assert rec["max_energy_sum"] == pytest.approx(float(e.max()), abs=1e-9)
        # The record's own frozen maxima, the numbers PROVENANCE.md tabulates:
        # 0.755176 on coarse and 0.660265 on mid. Pinned as values, so a
        # re-derived or swapped record that conserves more power is caught
        # rather than sliding under a round bound.
        assert rec["max_energy_sum"] == pytest.approx(
            {"coarse": 0.755176, "mid": 0.660265}[mesh], abs=1e-6)
        assert rec["max_energy_sum"] < 1.0
        got = stopband_null(np.asarray(rec["freqs_ghz"], float) * 1e9,
                            rec["s21_mag"])
        assert got["f"] / 1e9 == pytest.approx(rec["null"]["parabolic_f_ghz"],
                                               abs=1e-6)
        assert got["depth_db"] == pytest.approx(rec["null"]["depth_db"], abs=1e-6)
