"""Validation study D2: the 3-D FDFD spiral inductor (``rfx.fdfd.spiral``)
against the independent Greenhouse referee
(``validation/crossval/comparators/spiral_greenhouse.py``) under a
PHYSICS-CONSISTENT protocol, plus the shape derivatives three ways.

This replaces the refuted first study, which mixed three model differences
(PEC surface current vs the referee's DC uniform current, a closed PEC box
whose five extra walls carried 30 % of L, and the corner convention) into
one 5 % gate and then tried to repair them on the referee's side with a
box-image lattice that did not reproduce the FDFD wall by wall. Here each
difference is removed on the side where removing it is cheap:

A. metal. The FDFD is run with a VOLUMETRIC conductor: every conductor cell
   of every fixture becomes ``eps_r = 1 - j sigma / (omega eps0)`` and NO
   interior PEC mask is applied (``solve_spiral(..., sigma_volumetric=...)``,
   a hook added for this study). With ``sigma = 3e6`` S/m at 100 MHz the
   skin depth is 29.1 um -- 14.6 x the 2 um metal thickness and 2.9 x the
   10 um width -- so the current density is uniform over the cross-section
   and L includes the DC internal inductance: exactly the referee's model.
   The plateau is measured, not assumed (``sigma_plateau`` in the JSON):
   L_dut = 251.3 / 261.5 / 262.4 / 262.5 pH at sigma = 3e5 / 1e6 / 3e6 /
   1e7 S/m on the W/1 grid, i.e. flat to 0.04 % over the last decade while
   3e5 is still 4.2 % low (there the metal resistance is large enough for
   the shunt capacitance to eat the reactance: Re Z11 = 20.7 Ohm at 3e5 vs
   2.1 Ohm at 3e6). The dielectrics are set to VACUUM (``eps_si = eps_ox =
   1``) because the referee is magnetoquasistatic and has none, and because
   they are what makes that RC contamination large: with the real stack the
   same sigma = 3e5 point sits 18 % low instead of 4.2 %.
   The PEC run is kept at every level and reported (V1b).

B. walls. The referee models ONLY the PEC ground plane, by images (exact for
   an infinite plane). The FDFD box must therefore not feel its other five
   walls. Two mechanisms were measured:

   * The ``Yee3DSpec`` PML on the five non-ground faces (``pml_cells``,
     ``pml_kappa_max``, hooks added for this study) DOES NOT WORK at
     100 MHz and is not used. Measured on the W/1 and W/2 grids:
     ``pml_kappa_max`` = 20 and 40 give bit-identical L (the stretch
     ``s = kappa - j sigma_w / (omega eps0)`` has |s| ~ 6e5 at 100 MHz, so
     the real part is irrelevant), and L moves NON-monotonically by 5-15 %
     with the PML depth (271.5 -> 312.1 pH for 2 -> 3 cells at W/2) and
     downward when the walls are moved out. A PML absorbs propagating
     waves; a quasi-static near field sees an arbitrary complex material.
   * The documented fallback -- plain geometric grading of the cells toward
     the walls -- works. ``pad_cells = 4`` cells of ratio 1.5 appended
     outside the meshed box on the four side walls and the lid
     (``spiral.pad_lines``, a hook added for this study) put the side walls
     184 um and the lid 162 um away from a 104 um device for 8 in-plane and
     4 vertical cells. L_dut converges monotonically: 179.9 (closed box) /
     261.1 / 262.4 / 262.7 / 262.7 pH for pad = 0 / 3 / 4 / 6 / 8 at
     W/1, so the V5 gate (add 4 cells -> < 1 %) passes with +0.114 % and the
     residual wall systematic at pad = 4 is +0.114 %. The ground plane
     (z = 0) is never padded and never PML: it is the physical wall both
     tools model, and it extends under the padding, i.e. it is effectively
     infinite. Tripling the meshed air gap above the metal on top of the
     padding moves L by +0.504 % -- that is a LID + vertical-cell effect,
     see F, not a wall effect (the wall gate above moves the lid from
     162 to 779 um for +0.114 %).

C. corners. ``rect_spiral_segments`` lets each bar END at the corner point,
   so at every right angle one quadrant of the W x W corner square is
   covered by two bars and the opposite quadrant by none (the sum of the bar
   areas is right, the mass is misplaced). The area-exact convention used
   here translates every bar by +W/2 along its own direction, keeping the
   two free ends of the chain fixed: the total centreline length is
   unchanged, the bars become exactly disjoint and their union IS the mitred
   ``gds.rect_spiral`` strip polygon. Gate: sum of the bar rectangle areas =
   ``gds.polygon_area`` of the strip to 1e-12 relative AND union = sum to
   1e-12 (the sum alone is convention-blind -- it is equal for both
   conventions, measured -- so the non-overlap half of the gate is what
   pins the convention). The corner systematic is +1.21 % (345.13 ->
   349.37 pH); the area-exact value is used for every gate.

D. ground and reference plane. Both tools use the same ground height
   ``h = 26 um`` (the ``rfx.fdfd.spiral`` stack: 20 um silicon + 1 um oxide
   + 2 um M1 + 2 um via + 1 um to the M2 centre line) and the same
   underpass depth (4 um below the M2 centre). The FDFD's de-embedded
   reference plane is the lead-column footprint, one strip width of y
   starting at the port line (``build_spiral``: the short standard's bar
   covers exactly those cells, independently of ``base_dx``), so the
   referee's lead is shortened to its CENTRE, ``lead_length = lead - W/2``.
   On the strip alone that choice is worth +-1.46 % (354.46 / 344.27 pH at
   the port line / far edge) -- but see E: on the quantity the fixture
   really measures it collapses to +0.091 %. The plane TRACKS the width
   (``referee_segments(shift=None)``), which is 2.1 % of ``dL/dwidth``.

E. what the fixture measures is the strip MINUS the short standard's
   bridge. Open/short de-embedding subtracts ``Z_short'`` from ``Z_dut'``,
   and ``build_spiral``'s short standard is not an empty gap: it runs a bar
   out of the outer lead column on M2 and a bar out of the inner one on M1
   to a common grounded post between them. Driven differentially the post
   carries no current (the two arms' post currents cancel -- that is why
   its impedance drops out of ``Z11 - Z12 - Z21 + Z22``), so the short's
   differential path is ``column + bridge + column`` while the DUT's is
   ``column + strip + column``; the columns are z-directed and the strip
   and the bridge are x/y-directed, so every column-to-conductor mutual
   vanishes (perpendicular Neumann integrand) and the columns cancel term
   by term:

       L(de-embedded) = L(strip) - L(bridge) = 349.37 - 16.32 = 333.06 pH.

   Two independent checks, because this is the correction that moves V1:

   * INVARIANCE (referee side, ``referee_deembedded_block``). The reference
     plane inside the W x W footprint cuts the strip in y and the bridge in
     x. Sweeping it from the port line to the far edge moves the strip by
     +2.92 % and the DIFFERENCE by +0.091 % -- a 32x collapse. The
     leftover convention freedom, where the post splits the bridge between
     M2 and M1, is +0.205 % over the whole range.
   * DIRECT (FDFD side, ``thru_probe``, gate V6). Make the DUT identical to
     the bridge: ``dut_kind="bar"`` with ``bar_length`` = the column
     separation, 40 um, so the DUT occupies exactly the M2 cells the
     short's bars and post occupy. Its raw L is 24.37 pH and the referee
     for that bar is 16.90 pH, but it DE-EMBEDS TO +1.62 pH, 9.6 % of
     the bar: the fixture's zero is a strip between the column tops, not an
     empty gap.

   The probe also measures the one thing the referee cannot model, and it is
   not negligible: the short's POST is a 26 um block that touches the ground
   plane, so over its width the bridge's return path collapses and the
   fixture subtracts LESS than a whole bridge. That residual is exactly
   ``L_thru = L(bar) - L(bridge as built)``, which makes

       L_dut - L_thru = L(strip) - L(whole bridge)

   the quantity the referee's ``deembedded`` block computes. It is also the
   comparison with no residual bias: writing the FDFD's inductance deficit
   as a factor r on any piece of this conductor,
   ``L_dut - L_thru = r L(strip) - r L(bar) = r x`` the de-embedded referee,
   so the ratio IS r, with the bar's own discretisation error cancelling.
   The raw ``L_dut`` instead sits exactly ``L_thru`` above ``r x`` the
   referee (+1.6 pH = +0.5 % at W/1, and more at the finer levels),
   i.e. it flatters the agreement by the post short. So V1 subtracts the
   probe level by level (-21.7 / -13.4 / -11.6 % instead of
   the raw -21.2 / -12.0 / -9.8 %) and gates the corrected series. The
   correction is positive, i.e. it makes the gap LARGER, and it GROWS with
   refinement (9.6 / 27.1 / 35.8 % of the bar at W/1 / W/2 /
   W/3) because the post is ``outer.i0 - inner.i1 - 2`` cells wide and so
   covers more of the 40 um gap as ``base_dx`` shrinks: part of the raw
   L_dut's apparent convergence is the short standard changing, not the
   strip converging. The residual approximation is that the probe's bridge
   is all-M2 while the spiral's is mixed M2/M1, and that the probe's mesh is
   not bit-identical to the spiral's at the same ``base_dx``.

   The strip alone -- the referee the first version of this study used --
   is 4.9 % larger than the de-embedded one, so every gap against it is
   correspondingly deeper (-12.2 % to -9.6 % on the uncorrected
   extrapolation); that comparison is reported as
   ``rel_range_vs_strip_only`` but is NOT the gate.

F. the second grid direction. The three levels refine IN PLANE only:
   ``base_dz = 10 um`` and ``metal_cells = 2`` are held fixed, so the
   vertical error is common to all three and cancels out of the in-plane
   Richardson -- which therefore CANNOT see it. Measured separately at W/1
   (``systematics.vertical_grid``, ``vertical_budget``): halving and
   quartering ``base_dz`` (nz = 15 -> 20 -> 29, only 2 cells across the
   26 um ground gap at the base) raises L_dut by +1.91 % and +2.50 %, observed
   order 1.70, extrapolating to [+2.70, +3.09] %. Same sign as the V1 gap
   and the second-largest identified term; it is folded into V1c, not into
   V1, and it is NOT in the list of small fixture systematics.

Geometry (identical in both tools)
----------------------------------
2-turn square spiral, ``r_out = 52 um``, ``W = S = 10 um``, lead ``1.5 W``,
M2 strip 2 um thick, M1 underpass 4 um below its centre line, PEC ground
26 um below it, 100 MHz. ``r_out = 52 um`` is the smallest 2-turn spiral of
this pitch that leaves the >= 3 cells between the lead columns that the
short standard needs at ``base_dx = W``. The referee has no dielectrics, no
vias and no lead columns; the FDFD's open/short DE-EMBEDDED ``L_diff`` is
what is compared.

Levels and cost
---------------
``base_dx = W/1, W/2, W/3`` (1, 2, 3 cells across the strip width);
``base_dz = 10 um`` and ``metal_cells = 2`` at EVERY level (see F).
N = 23062 / 56408 / 108898 unknowns, one three-fixture solve 20 / 100 / 347 s
(SuperLU/COLAMD, x64 CPU), the FD4 gradient stencil 178 / 1154 / 5034 s.
``W/1.5`` is not a level: the mandatory grid lines of this geometry are
10 um apart, so ``base_dx`` in (5, 10] um gives the same 2-cell grid as W/2
(measured: (32,34,15) vs (33,35,15)). ``W/4`` would be 162k unknowns, above
the ~105k single-solve guideline, and is NOT run. The whole study is
140 min wall clock (two other agents sharing the machine).

Run::

    .venv/bin/python validation/fdfd/spiral_convergence.py [--levels 1 2 3]
        [--fd2-from 3] [--no-systematics]

writes ``spiral_convergence.json`` and ``spiral_convergence.png`` next to
this file; the JSON is rewritten after every level and every systematics
block, so a run that is interrupted still leaves everything it had already
measured on disk (there is no resume: a restart recomputes).

Gates (the numbers are in the JSON; the test file asserts what the W/1 grid
supports)
--------
V1  FAILS. Uniform-current FDFD ``L_dut`` = 262.44 / 293.00 / 300.56 pH at
    W/1 / W/2 / W/3; minus the thru probe's post residual (E) the matched
    series is 260.82 / 288.42 / 294.51 pH against the de-embedded referee
    333.06 pH: -21.7 %, -13.4 %, -11.6 % (uncorrected -21.2 / -12.0 / -9.8 %).
    Richardson over the three corrected levels gives an observed order
    1.70 and extrapolates to 306.70 (p = 1, finest pair) / 299.39 (p = 2) /
    300.63 pH (observed order), i.e. the range [299.39, 306.70] pH sits
    -10.1 % to -7.9 % from the referee, outside +-5 %. (Uncorrected
    -7.9 % to -5.2 %; against the strip alone, which is NOT what the
    fixture measures, -12.2 % to -9.6 %.)
V1c the same comparison with the vertical grid extrapolated too (F):
    [307.47, 316.19] pH = -7.7 % to -5.1 %, still outside +-5 %. That
    residual is about twice the p = 1 / p = 2 spread of the in-plane
    extrapolation itself (2.4 %), so the two measured corrections do NOT
    close the gap: something of order 5 % is left over.
    The mechanism the rest is attributed to is the FDFD's DISCRETE
    SELF-INDUCTANCE of a conductor resolved by 1-3 cells across its
    cross-section, measured fixture-free by the lead-length differential:
    lengthening the lead by 20 um (which adds 40 um of conductor: the lead
    AND the underpass, which ends on the port line) adds conductor to both
    tools and the FDFD/referee ratio of that increment is 0.763 / 0.876 / 0.901 at
    W/1 / W/2 / W/3, against total ratios 0.783 / 0.866 / 0.884 -- the
    inductance per unit length of the strip is low by about as much as L
    is, so the deficit is distributed along the conductor rather than a
    constant fixture offset (the V3 gradient ratios say the same). A
    strip's self-inductance goes as ``ln(2 l / GMD)`` with
    ``GMD = 0.2235 (W + t)`` = 2.68 um here; the Yee grid replaces that GMD
    by one set by the cell size. The remaining fixture systematics are
    small: de-embedding invariance (``port_gap_cells`` 1 -> 2) -0.035 %,
    wall residual +0.114 %, lid + air cells +0.504 %, ``metal_cells`` 2 -> 4
    +0.281 %, frequency flatness 2e-07 between 50 and 200 MHz, reference
    plane +0.091 %, corner convention +1.21 %.
V1b the PEC fixture carries no DC internal inductance, so the
    uniform-current L must exceed it by the conductor's internal
    inductance -- a FINITE limit, not something that shrinks with
    refinement (the criterion the first version used, which had no
    physical basis). Measured deficit 17.56 / 26.64 / 27.20 pH against the
    round-wire DC value ``mu0 l / (8 pi)`` = 32.10 pH for the same
    642 um of conductor, which bounds a flat strip's from above:
    0.547 / 0.830 / 0.847 of it, level-to-level change 9.08 -> 0.56 pH.
    PASSES.
V2  ``jax.grad`` of the volumetric FDFD ``L_dut`` vs FD4 with 1 % steps
    (which move L by ~1e-2 relative, far above the ~1e-9 LU noise floor)
    <= 1e-4 relative, all three parameters, every level: worst 5.72e-07. PASSES.
V3  ``jax.grad`` (FDFD) vs FD4 of the DE-EMBEDDED referee (E -- the bridge
    depends on the spacing and the width, so this is not the same as the
    strip's gradient: -1.08863e-5 vs -9.8337e-6 H/m for ``dL/dspacing``):
    same sign for all three parameters and within 15 % at the finest level
    run. Ratios at W/3 0.918 / 0.936 / 0.880. PASSES.
V4  the per-level change of ``dL/dwidth`` decreases with refinement:
    7.69e-07 then 9.08e-07. FAILS.
V5  wall independence: ``pad_cells`` 4 -> 8 moves ``L_dut`` by < 1 % at the
    coarsest level (measured +0.114 %). PASSES.
V6  the thru probe of E: a DUT identical to the short's bridge de-embeds to
    9.6 / 27.1 / 35.8 % of the referee value of that bar
    instead of 100 %, i.e. the de-embedding removes most of it at every
    level. Tolerance 50 %, worst 35.8 %; the tolerance is a
    sanity bound on "most of it", not a convergence claim -- the residual
    is the post short and it GROWS with refinement (E). PASSES.

Scope fence. One geometry, one frequency; in-plane refinement only (the
vertical direction is measured at the coarsest level and extrapolated
separately in V1c, not refined jointly); the referee is magnetoquasistatic
with uniform current density, no substrate eddy currents and no via, and it
cannot represent the short standard's grounded post -- which the thru
probe measures on the FDFD side instead, and V1 subtracts (up to 1.75 % of
the referee at W/3). Nothing here validates losses, the frequency
dependence or the PEC metal model -- the PEC numbers are reported for
contrast only. Sizes are CPU-SuperLU sizes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
REFEREE_PATH = REPO / "validation" / "crossval" / "comparators" / "spiral_greenhouse.py"
JSON_PATH = HERE / "spiral_convergence.json"
PNG_PATH = HERE / "spiral_convergence.png"

# ---- the fixture (both tools) ---------------------------------------------
FREQ = 1e8
N_TURNS = 2
R_OUT, SPACING, WIDTH = 52e-6, 10e-6, 10e-6
LEAD = 1.5 * WIDTH
MARGIN = 10e-6
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2, T_OX_HIGH, T_AIR = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6, 3e-6, 10e-6
BASE_DZ = 10e-6            # vertical grid, the SAME at every level
METAL_CELLS = 2            # >= 2 cells across every metal slab
PAD_CELLS, PAD_RATIO = 4, 1.5
SIGMA_VOL = 3e6            # uniform-current plateau (measured, see the module doc)
GROUND_H = T_SI + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2          # 26 um, ground -> M2 centre
DZ_UNDER = 0.5 * T_M1 + T_VIA + 0.5 * T_M2                      # 4 um, M2 centre -> M1 centre
REF_SHIFT = 0.5 * WIDTH    # referee lead ends at the lead-column CENTRE
FD_STEP_REL = 0.01         # 1 % steps (V2 measures the agreement they give)
PARAMS = ("r_out", "spacing", "width")
THETA0 = (R_OUT, SPACING, WIDTH)
LEAD_PROBE = 35e-6         # lead-length differential probe (fixture-free)
BRIDGE_SPLIT = 0.5         # M2 fraction of the short standard's bridge (grid detail)
THRU_TOL = 0.50            # V6: the de-embedding must remove more than half the bridge
                           # at every level (measured 9.6 / 27.1 / 35.8 %)
MU0 = 4e-7 * np.pi


# ----------------------------------------------------------------------------
# 1. referee (host numpy; loaded from its file -- the comparators directory
#    is not a package -- and never imported by rfx)

def load_referee():
    spec = importlib.util.spec_from_file_location("spiral_greenhouse", REFEREE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def area_exact_chain(sg, segs: Sequence[Any]) -> list:
    """The area-exact corner convention (C in the module doc): translate
    every bar by ``+width/2`` along its own direction, except the START of
    the first bar and the END of the last one, which stay put.

    At an interior vertex the incoming bar then reaches W/2 PAST the corner
    (it owns the whole corner square) and the outgoing bar starts W/2 after
    it, so consecutive rectangles touch without overlapping; collinear
    neighbours (the lead and side 0) simply stay abutting. The total
    centreline length is unchanged."""
    out, n = [], len(segs)
    for i, s in enumerate(segs):
        a = np.asarray(s.start, dtype=np.float64)
        b = np.asarray(s.end, dtype=np.float64)
        d = b - a
        ell = float(np.linalg.norm(d))
        if ell <= 0:
            raise ValueError("degenerate segment")
        u = d / ell
        sa = a if i == 0 else a + 0.5 * s.width * u
        sb = b if i == n - 1 else b + 0.5 * s.width * u
        out.append(sg.Segment(tuple(sa), tuple(sb), s.width, s.thickness, s.current))
    return out


def referee_segments(sg, theta: Sequence[float], convention: str = "area_exact",
                     lead: float = LEAD, shift: float | None = None) -> list:
    """Referee segment list for ``theta = (r_out, spacing, width)``: the
    strip (lead shortened to the de-embedded reference plane) plus the
    underpass. ``convention`` is ``"greenhouse"`` (the delivered
    centreline-to-corner-point convention, (i) in the module doc) or
    ``"area_exact"`` ((ii), the one the gates use).

    ``shift = None`` (the default) puts the reference plane at the CENTRE of
    the lead-column footprint, ``0.5 * width`` of the theta being evaluated
    -- the plane TRACKS the width, because the FDFD footprint is the
    ``width x width`` square under the column. That matters only for the
    derivative: holding the shift at ``0.5 * WIDTH`` while differentiating
    w.r.t. the width misses ``0.5 dL/d(lead_length)`` = 2.1 % of
    ``dL/dwidth`` (measured: -2.4491e-5 vs -2.5001e-5 H/m). Pass a float to
    pin the plane (the reference-plane band uses 0, ``width``)."""
    r_out, spacing, width = (float(v) for v in theta)
    shift = 0.5 * width if shift is None else float(shift)
    segs = sg.rect_spiral_segments(N_TURNS, r_out, width, spacing, T_M2,
                                   lead_length=lead - shift, underpass=(DZ_UNDER, T_M1))
    if convention == "greenhouse":
        return segs
    if convention != "area_exact":
        raise ValueError(f"unknown corner convention {convention!r}")
    # the underpass is a lone bar on the other layer: no corner, keep it
    return area_exact_chain(sg, segs[:-1]) + [segs[-1]]


def referee_value(sg, theta: Sequence[float], convention: str = "area_exact",
                  lead: float = LEAD, shift: float | None = None,
                  ground: float | None = GROUND_H) -> Any:
    return sg.greenhouse_terms(referee_segments(sg, theta, convention, lead, shift), ground_height=ground)


def _bar_rect(seg) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` of a segment's plan-view rectangle."""
    a = np.asarray(seg.start, dtype=np.float64)
    b = np.asarray(seg.end, dtype=np.float64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ax = 0 if abs(b[0] - a[0]) > 0 else 1
    pe = 1 - ax
    r = [0.0, 0.0, 0.0, 0.0]
    r[ax], r[ax + 2] = lo[ax], hi[ax]
    r[pe], r[pe + 2] = lo[pe] - 0.5 * seg.width, hi[pe] + 0.5 * seg.width
    return (r[0], r[1], r[2], r[3])


def area_gate(sg, theta: Sequence[float], lead: float = LEAD,
               shift: float | None = None) -> dict[str, Any]:
    """Gate C: for both corner conventions, the sum of the bar rectangle
    areas and the area of their UNION against ``gds.polygon_area`` of the
    ``rect_spiral`` strip polygon (built with the same shortened lead) and
    of the underpass polygon. The area-exact convention must match both to
    1e-12 relative."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    from rfx.fdfd import gds

    r_out, spacing, width = (float(v) for v in theta)
    shift = 0.5 * width if shift is None else float(shift)
    sp = gds.rect_spiral(N_TURNS, r_out, width, spacing, lead - shift)
    a_strip = gds.polygon_area(sp.polygons[(134, 0)][0])
    a_under = gds.polygon_area(sp.polygons[(126, 0)][0])
    out: dict[str, Any] = {"strip_polygon_area": a_strip, "underpass_polygon_area": a_under,
                           "strip_centreline_length": sp.length,
                           "underpass_centreline_length": sp.underpass_length}
    for conv in ("greenhouse", "area_exact"):
        segs = referee_segments(sg, theta, conv, lead, shift)
        rects = [box(*_bar_rect(s)) for s in segs[:-1]]
        total = float(sum(r.area for r in rects))
        union = float(unary_union(rects).area)
        under = float(box(*_bar_rect(segs[-1])).area)
        out[conv] = {
            "n_bars": len(rects), "sum_of_areas": total, "union_area": union,
            "sum_over_polygon_minus_1": total / a_strip - 1.0,
            "union_over_polygon_minus_1": union / a_strip - 1.0,
            "union_over_sum_minus_1": union / total - 1.0,
            "underpass_area_over_polygon_minus_1": under / a_under - 1.0,
            "total_centreline_length": float(sum(
                float(np.linalg.norm(np.asarray(s.end, float) - np.asarray(s.start, float))) for s in segs[:-1])),
        }
    return out


def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def fd2(f: Callable[[float], float], x0: float, d: float) -> float:
    return (f(x0 + d) - f(x0 - d)) / (2 * d)


# ---- E. the quantity the FIXTURE measures: strip MINUS the short's bridge --

def terminal_points(theta: Sequence[float], shift: float | None = None) -> tuple[float, float, float]:
    """``(x_outer, x_inner, y_bridge)`` of the two de-embedding reference
    planes, in the referee's coordinates.

    The referee's ``rect_spiral_centreline`` puts the outer terminal on
    ``x = a0 = r_out - width/2`` and the inner one on
    ``x = a_in = a0 - (width + spacing) * n_turns``, both on the port line
    ``y = -a0 - lead``; the lead-column footprints are the ``W x W`` squares
    whose LOW-y edge is that line. A reference plane ``shift`` into the
    footprint therefore cuts the strip (which leaves the footprint in +y) at
    ``y = -a0 - lead + shift`` and the short's bridge (which leaves it in
    -x from the outer column and +x from the inner one) at
    ``x = a0 + width/2 - shift`` and ``x = a_in - width/2 + shift``. Same
    plane, two directions -- that pairing is what makes the difference below
    reference-plane invariant."""
    r_out, spacing, width = (float(v) for v in theta)
    shift = 0.5 * width if shift is None else float(shift)
    a0 = r_out - 0.5 * width
    a_in = a0 - (width + spacing) * N_TURNS
    return (a0 + 0.5 * width - shift, a_in - 0.5 * width + shift,
            -a0 - LEAD + 0.5 * width)


def bridge_segments(sg, theta: Sequence[float], shift: float | None = None,
                    split: float = BRIDGE_SPLIT) -> list:
    """The conductor the OPEN/SHORT de-embedding subtracts from the DUT:
    the short standard's bridge between the two reference planes.

    ``build_spiral``'s short standard runs a bar out of the outer column on
    M2 and a bar out of the inner column on M1 to a common grounded post
    between them. Driven differentially the post carries NO current (the
    two arms' post currents cancel -- that is why the post's impedance
    drops out of ``Z11 - Z12 - Z21 + Z22``), so the short's differential
    path is exactly ``column + bridge + column`` while the DUT's is
    ``column + strip + column``. The columns are z-directed and the strip
    and bridge are x/y-directed, so every column-to-strip and
    column-to-bridge mutual is zero (perpendicular Neumann integrand) and
    the columns cancel term by term:

        L(de-embedded) = L(strip) - L(bridge).

    ``split`` is the fraction of the bridge that sits on M2 (the post's
    position between the columns, a grid detail: 0 = all M1, 1 = all M2;
    the spread of the de-embedded referee over the whole range is 0.205 %,
    measured, and the bridge itself moves 16.90 -> 16.32 -> 16.42 pH for
    split = 0 / 0.5 / 1)."""
    xo, xi, y_b = terminal_points(theta, shift)
    width = float(theta[2])
    xs = xi + float(split) * (xo - xi)
    segs = []
    if xs < xo:
        segs.append(sg.Segment((xo, y_b, 0.0), (xs, y_b, 0.0), width, T_M2))
    if xs > xi:
        segs.append(sg.Segment((xs, y_b, -DZ_UNDER), (xi, y_b, -DZ_UNDER), width, T_M1))
    if not segs:
        raise ValueError("degenerate bridge")
    return segs


def bridge_value(sg, theta: Sequence[float], shift: float | None = None,
                 split: float = BRIDGE_SPLIT, ground: float | None = GROUND_H) -> Any:
    return sg.greenhouse_terms(bridge_segments(sg, theta, shift, split), ground_height=ground)


def referee_deembedded(sg, theta: Sequence[float], shift: float | None = None,
                       split: float = BRIDGE_SPLIT, convention: str = "area_exact") -> float:
    """``L(strip) - L(bridge)``: the referee quantity that the FDFD's
    de-embedded ``L_diff`` actually measures (see :func:`bridge_segments`)."""
    return float(referee_value(sg, theta, convention, shift=shift).total
                 - bridge_value(sg, theta, shift, split).total)


def referee_deembedded_block(sg, theta: Sequence[float] = THETA0) -> dict[str, Any]:
    """Everything about the de-embedded referee that a reader needs to
    judge it: the value, the bridge, the reference-plane band of the
    DIFFERENCE (which is what collapses), and the M2/M1 split sensitivity."""
    width = float(theta[2])
    planes = {"port_line": 0.0, "column_centre": 0.5 * width, "far_edge": width}
    out: dict[str, Any] = {
        "total": referee_deembedded(sg, theta),
        "strip": float(referee_value(sg, theta, "area_exact").total),
        "bridge": float(bridge_value(sg, theta).total),
        "bridge_split_on_m2": BRIDGE_SPLIT,
        "bridge_worst_error": float(bridge_value(sg, theta).worst_error),
        "per_plane": {}, "per_split": {},
    }
    for name, sh in planes.items():
        out["per_plane"][name] = {
            "strip": float(referee_value(sg, theta, "area_exact", shift=sh).total),
            "bridge": float(bridge_value(sg, theta, sh).total),
            "deembedded": referee_deembedded(sg, theta, shift=sh)}
    vals = [v["deembedded"] for v in out["per_plane"].values()]
    ref = out["per_plane"]["column_centre"]["deembedded"]
    out["reference_plane_spread"] = (max(vals) - min(vals)) / ref
    strips = [v["strip"] for v in out["per_plane"].values()]
    out["reference_plane_spread_strip_only"] = (max(strips) - min(strips)) / \
        out["per_plane"]["column_centre"]["strip"]
    for sp_ in (0.0, 0.25, 0.5, 0.75, 1.0):
        out["per_split"][f"{sp_:g}"] = referee_deembedded(sg, theta, split=sp_)
    sv = list(out["per_split"].values())
    out["split_spread"] = (max(sv) - min(sv)) / ref
    return out


def bar_referee(sg, l_bar: float, width: float = WIDTH, theta: Sequence[float] = THETA0) -> float:
    """Referee value of ONE straight M2 bar of length ``l_bar`` centred
    where the spiral's bridge is (the thru probe's DUT), ground image
    included."""
    _, _, y_b = terminal_points(theta)
    seg = sg.Segment((0.5 * l_bar, y_b, 0.0), (-0.5 * l_bar, y_b, 0.0), width, T_M2)
    return float(sg.greenhouse_terms([seg], ground_height=GROUND_H).total)


def referee_gradient(sg, theta: Sequence[float] = THETA0, convention: str = "area_exact",
                     deembedded: bool = True) -> list[float]:
    """``dL/dtheta`` of the referee by FD4 with 1 % steps (host numpy, no
    solver noise: the only error is the FD truncation). ``deembedded``
    differentiates ``L(strip) - L(bridge)`` -- the quantity the FDFD's
    ``L_dut`` is -- which is what V3 compares against; ``False`` gives the
    strip alone (reported for contrast: the bridge is independent of
    ``r_out`` but not of ``spacing`` or ``width``, so the two agree on
    ``dL/dr_out`` and differ by 11 % / 1.9 % on the other two, measured)."""
    base = np.asarray(theta, dtype=np.float64)
    out = []
    for k in range(3):
        def f(v: float, k: int = k) -> float:
            t = base.copy()
            t[k] = v
            return (referee_deembedded(sg, t, convention=convention) if deembedded
                    else float(referee_value(sg, t, convention).total))
        out.append(float(fd4(f, float(base[k]), FD_STEP_REL * float(base[k]))))
    return out


# ----------------------------------------------------------------------------
# 2. the FDFD side

def stack(base_dz: float = BASE_DZ, metal_cells: int = METAL_CELLS, t_air: float = T_AIR):
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=T_SI, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=t_air, base_dz=base_dz,
                         metal_cells=metal_cells, eps_si=1.0, eps_ox=1.0)


def build_level(div: float, pad: int = PAD_CELLS, pad_z: int = 0, lead: float = LEAD,
                base_dz: float = BASE_DZ, metal_cells: int = METAL_CELLS,
                port_gap_cells: int = 1, margin: float = MARGIN, t_air: float = T_AIR):
    """The static FDFD model at ``base_dx = WIDTH / div``."""
    from rfx.fdfd import spiral as sm
    spec = sm.SpiralSpec(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH, lead=lead,
                         base_dx=WIDTH / div, margin=margin,
                         stack=stack(base_dz, metal_cells, t_air), port_gap_cells=port_gap_cells,
                         pad_cells=pad, pad_cells_z=pad_z, pad_ratio=PAD_RATIO)
    return sm.build_spiral(spec)


def l_dut(model, theta, sigma=SIGMA_VOL):
    """De-embedded ``L_diff`` (H) of the volumetric-metal fixture -- the
    quantity the referee is compared against. ``sigma=None`` gives the PEC
    fixture."""
    from rfx.fdfd import spiral as sm
    return sm.solve_spiral(model, FREQ, theta=theta, sigma_volumetric=sigma).L_diff


def model_record(model) -> dict[str, Any]:
    return {"shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
            "wall_x": [float(model.x_nom[0]), float(model.x_nom[-1])],
            "wall_y": [float(model.y_nom[0]), float(model.y_nom[-1])],
            "lid_z": float(model.z[-1]), "n_z_lines": int(len(model.z)),
            "build_seconds": float(model.build_seconds)}


def run_level(div: float, order: int = 4, verbose: bool = True) -> dict[str, Any]:
    """One level: ``L_dut`` and ``jax.grad`` of the volumetric fixture, the
    FD check of that gradient, and the PEC fixture for V1b.

    Timings in the record: ``value_and_grad_seconds`` is a cold
    three-fixture solve plus the reverse pass (which reuses the same LU
    factorisations), ``pec_seconds`` a cold three-fixture solve of the PEC
    fixture and ``fd_seconds`` the whole FD stencil (4 cold solves per
    parameter for FD4, 2 for FD2). ``solve_seconds`` is NOT a solve time:
    that call re-uses the LU the gradient already factorised (LRU cache in
    ``rfx.fdfd.linear_solve``) and only re-reads L_raw, Q and the raw
    S-matrix from it."""
    import jax
    import jax.numpy as jnp

    from rfx.fdfd import spiral as sm

    model = build_level(div)
    rec: dict[str, Any] = {"base_dx": WIDTH / div, "div": div, "grid": model_record(model)}
    if verbose:
        print(f"  W/{div:g}: {model.shape} N={model.n_unknowns}", flush=True)

    def scalar(t) -> Any:
        return jnp.real(l_dut(model, t))

    t0 = time.time()
    val, grad = jax.value_and_grad(scalar)(jnp.asarray(THETA0, dtype=jnp.float64))
    jax.block_until_ready(grad)
    rec["value_and_grad_seconds"] = time.time() - t0
    rec["L_dut"] = float(val)
    rec["grad"] = [float(v) for v in grad]

    t0 = time.time()
    res = sm.solve_spiral(model, FREQ, sigma_volumetric=SIGMA_VOL)
    rec["L_raw"] = float(res.L_raw)
    rec["Q_diff"] = float(res.Q_diff)
    rec["z_dut"] = [[float(np.real(res.z_dut[i, j])), float(np.imag(res.z_dut[i, j]))]
                    for i in range(2) for j in range(2)]
    s = np.asarray(res.s_raw)
    rec["s_raw_max_singular_value"] = float(np.linalg.svd(s, compute_uv=False).max())
    rec["s_raw_reciprocity"] = float(abs(s[0, 1] - s[1, 0]))
    rec["solve_seconds"] = time.time() - t0

    t0 = time.time()
    rec["L_pec"] = float(sm.solve_spiral(model, FREQ, sigma_volumetric=None).L_diff)
    rec["pec_seconds"] = time.time() - t0
    rec["L_pec_over_L_uniform_minus_1"] = rec["L_pec"] / rec["L_dut"] - 1.0

    t0 = time.time()
    fd = []
    stencil = fd4 if order == 4 else fd2
    for k in range(3):
        def f(v: float, k: int = k) -> float:
            t = list(THETA0)
            t[k] = v
            return float(scalar(jnp.asarray(t, dtype=jnp.float64)))
        fd.append(float(stencil(f, THETA0[k], FD_STEP_REL * THETA0[k])))
    rec["fd_order"] = order
    rec["fd"] = fd
    rec["fd_seconds"] = time.time() - t0
    rec["grad_vs_fd_rel"] = [abs(g - f_) / abs(f_) for g, f_ in zip(rec["grad"], fd)]
    if verbose:
        print(f"     L_dut={rec['L_dut'] * 1e12:.3f} pH  L_pec={rec['L_pec'] * 1e12:.3f} pH  "
              f"grad vs FD{order} worst {max(rec['grad_vs_fd_rel']):.2e}  "
              f"({rec['value_and_grad_seconds']:.0f}+{rec['fd_seconds']:.0f} s)", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 3. the thru probe: what the de-embedding's ZERO is

def build_bar_level(div: float, l_bar: float, pad: int = PAD_CELLS):
    """The SAME fixture with the spiral replaced by one straight M2 bar of
    length ``l_bar`` between the two lead columns (``dut_kind="bar"``)."""
    from rfx.fdfd import spiral as sm
    spec = sm.SpiralSpec(n_turns=N_TURNS, r_out=R_OUT, spacing=SPACING, width=WIDTH, lead=LEAD,
                         base_dx=WIDTH / div, margin=MARGIN, stack=stack(),
                         port_gap_cells=1, pad_cells=pad, pad_ratio=PAD_RATIO,
                         dut_kind="bar", bar_length=l_bar)
    return sm.build_spiral(spec)


def thru_probe(sg, divs: Sequence[float]) -> dict[str, Any]:
    """Gate V6, the DIRECT measurement of what open/short de-embedding
    subtracts.

    The DUT is made IDENTICAL to the short standard's bridge: one straight
    M2 bar spanning the two lead-column footprints, i.e. exactly the cells
    the short's bars and post occupy on M2. If the de-embedded ``L_diff``
    were "the inductance of the conductor between the reference planes"
    (the reading the study's first version assumed), this fixture would
    return the referee value of that bar; if instead the de-embedding
    subtracts the bridge, it must return ~0. One three-fixture solve per
    level on a fixture much smaller than the spiral's."""
    l_bar = float(terminal_points(THETA0)[0] - terminal_points(THETA0)[1])
    ref = bar_referee(sg, l_bar)
    out: dict[str, Any] = {"bar_length": l_bar, "referee_bar": ref, "levels": {}}
    from rfx.fdfd import spiral as sm
    for div in divs:
        m = build_bar_level(div, l_bar)
        t0 = time.time()
        r = sm.solve_spiral(m, FREQ, sigma_volumetric=SIGMA_VOL)
        rec = {"base_dx": WIDTH / div, "n_unknowns": int(m.n_unknowns),
               "L_raw": float(r.L_raw), "L_deembedded": float(r.L_diff),
               "over_referee_bar": float(r.L_diff) / ref,
               "over_L_raw": float(r.L_diff) / float(r.L_raw),
               "seconds": time.time() - t0}
        out["levels"][f"{div:g}"] = rec
        print(f"    W/{div:g}: N={rec['n_unknowns']} L_raw={rec['L_raw'] * 1e12:.3f} "
              f"L_deembedded={rec['L_deembedded'] * 1e12:+.3f} pH "
              f"= {100 * rec['over_referee_bar']:+.1f} % of the referee bar "
              f"({ref * 1e12:.3f} pH)", flush=True)
    worst = max(abs(v["over_referee_bar"]) for v in out["levels"].values())
    out["worst_over_referee_bar"] = worst
    return out


# ----------------------------------------------------------------------------
# 4. systematics of the FDFD fixture (all on the coarsest grid unless said)

def wall_gate(div: float, pads: Sequence[int] = (0, 3, 4, 6, 8)) -> dict[str, Any]:
    """Gate B/V5: ``L_dut`` against the number of graded padding cells; the
    gate is the relative change from ``PAD_CELLS`` to ``PAD_CELLS + 4``."""
    out: dict[str, Any] = {"div": div, "pad_cells": list(pads), "L_dut": [], "wall_x": [], "lid_z": [],
                           "n_unknowns": [], "seconds": []}
    for p in pads:
        m = build_level(div, pad=p)
        t0 = time.time()
        out["L_dut"].append(float(l_dut(m, THETA0)))
        out["seconds"].append(time.time() - t0)
        out["wall_x"].append(float(m.x_nom[-1]))
        out["lid_z"].append(float(m.z[-1]))
        out["n_unknowns"].append(int(m.n_unknowns))
        print(f"    pad={p}: N={m.n_unknowns} wall={out['wall_x'][-1] * 1e6:.0f} um "
              f"lid={out['lid_z'][-1] * 1e6:.0f} um L={out['L_dut'][-1] * 1e12:.3f} pH", flush=True)
    idx = list(pads).index(PAD_CELLS)
    out["L_at_pad_cells"] = out["L_dut"][idx]
    if PAD_CELLS + 4 in pads:
        j = list(pads).index(PAD_CELLS + 4)
        out["rel_change_plus_4_cells"] = out["L_dut"][j] / out["L_dut"][idx] - 1.0
    return out


def sigma_plateau(div: float, sigmas: Sequence[float] = (3e5, 1e6, 3e6, 1e7)) -> dict[str, Any]:
    """Gate A: the DC (uniform-current) plateau of ``L_dut`` in the metal
    conductivity, with the skin depth of each point."""
    from rfx.fdfd import spiral as sm
    m = build_level(div)
    out: dict[str, Any] = {"div": div, "sigma": list(map(float, sigmas)), "L_dut": [], "Q_diff": [],
                           "re_z11": [], "skin_depth": [], "skin_over_width": [],
                           "skin_over_thickness": []}
    for sg_ in sigmas:
        r = sm.solve_spiral(m, FREQ, sigma_volumetric=sg_)
        d = float(sm.skin_depth(FREQ, sg_))
        out["L_dut"].append(float(r.L_diff))
        out["Q_diff"].append(float(r.Q_diff))
        out["re_z11"].append(float(np.real(r.z_dut[0, 0])))
        out["skin_depth"].append(d)
        out["skin_over_width"].append(d / WIDTH)
        out["skin_over_thickness"].append(d / T_M2)
        print(f"    sigma={sg_:.0e} delta={d * 1e6:.1f} um: L={out['L_dut'][-1] * 1e12:.3f} pH "
              f"Q={out['Q_diff'][-1]:.4g}", flush=True)
    i = out["sigma"].index(SIGMA_VOL)
    out["rel_change_to_last"] = out["L_dut"][-1] / out["L_dut"][i] - 1.0
    return out


def lead_differential(sg, divs: Sequence[float], base_values: dict[str, float]) -> dict[str, Any]:
    """The FIXTURE-FREE probe of the FDFD's inductance per unit length of
    strip, at every level: lengthen ``lead`` from 1.5 W to 3.5 W, which adds
    exactly ``2 x 20 um`` of conductor (the lead AND the underpass, which
    ends on the port line) to BOTH tools, and compare the two increments.
    Every fixture systematic -- the ports, the lead columns, the
    de-embedding, the reference plane, the walls -- is common to the two
    solves and cancels in the difference, so the ratio measures only how
    well the grid represents the inductance of a piece of strip. One extra
    solve per level (the short-lead value is the level's own ``L_dut``)."""
    d_ref = float(referee_value(sg, THETA0, lead=LEAD_PROBE).total
                  - referee_value(sg, THETA0).total)
    out: dict[str, Any] = {"lead_from": LEAD, "lead_to": LEAD_PROBE,
                           "added_conductor_length": 2.0 * (LEAD_PROBE - LEAD),
                           "dL_referee": d_ref, "levels": {}}
    for div in divs:
        key = f"{div:g}"
        base = base_values[key]
        m = build_level(div, lead=LEAD_PROBE)
        t0 = time.time()
        long_lead = float(l_dut(m, THETA0))
        out["levels"][key] = {"base_dx": WIDTH / div, "n_unknowns": int(m.n_unknowns),
                              "L_short_lead": base, "L_long_lead": long_lead,
                              "dL_fdfd": long_lead - base,
                              "ratio_fdfd_over_referee": (long_lead - base) / d_ref,
                              "seconds": time.time() - t0}
        r = out["levels"][key]
        print(f"    W/{div:g}: dL_fdfd={r['dL_fdfd'] * 1e12:+.3f} dL_ref={d_ref * 1e12:+.3f} pH "
              f"ratio={r['ratio_fdfd_over_referee']:.3f}", flush=True)
    return out


def fixture_systematics(sg, div: float) -> dict[str, Any]:
    """Everything on the FDFD side that could fake a model difference, each
    measured on the coarsest grid: de-embedding invariance, frequency
    flatness, the vertical grid, the metal cross-section cells, the lead
    length differential against the referee (fixture-free)."""
    from rfx.fdfd import spiral as sm
    out: dict[str, Any] = {"div": div}
    m = build_level(div)
    base = float(l_dut(m, THETA0))
    out["L_dut"] = base

    out["freq_flatness"] = {}
    for f in (0.5 * FREQ, FREQ, 2.0 * FREQ):
        v = float(sm.solve_spiral(m, f, sigma_volumetric=SIGMA_VOL).L_diff)
        out["freq_flatness"][f"{f:.3e}"] = v
    vals = list(out["freq_flatness"].values())
    out["freq_flatness_rel_spread"] = (max(vals) - min(vals)) / base

    m2 = build_level(div, port_gap_cells=2)
    out["port_gap_cells_2"] = {"n_unknowns": int(m2.n_unknowns), "L_dut": float(l_dut(m2, THETA0))}
    out["port_gap_cells_2"]["rel"] = out["port_gap_cells_2"]["L_dut"] / base - 1.0

    # the wall sensitivity per FACE, on top of the padding: the four side
    # walls alone (a 3x margin pushes them 20 um further out) and the lid
    # alone (a 3x air gap does the same vertically)
    mm = build_level(div, margin=3 * MARGIN)
    out["margin_3x"] = {"n_unknowns": int(mm.n_unknowns), "L_dut": float(l_dut(mm, THETA0)),
                        "wall_x": float(mm.x_nom[-1])}
    out["margin_3x"]["rel"] = out["margin_3x"]["L_dut"] / base - 1.0
    ma = build_level(div, t_air=3 * T_AIR)
    out["t_air_3x"] = {"n_unknowns": int(ma.n_unknowns), "L_dut": float(l_dut(ma, THETA0)),
                       "lid_z": float(ma.z[-1])}
    out["t_air_3x"]["rel"] = out["t_air_3x"]["L_dut"] / base - 1.0
    print(f"    side walls 3x margin: {100 * out['margin_3x']['rel']:+.3f} %   "
          f"lid 3x air: {100 * out['t_air_3x']['rel']:+.3f} %", flush=True)

    out["vertical_grid"] = []
    for bdz, padz in ((BASE_DZ, 0), (0.5 * BASE_DZ, 6), (0.25 * BASE_DZ, 8)):
        mz = build_level(div, base_dz=bdz, pad_z=padz)
        v = float(l_dut(mz, THETA0))
        out["vertical_grid"].append({"base_dz": bdz, "pad_cells_z": padz, "nz": int(mz.shape[2]),
                                     "n_unknowns": int(mz.n_unknowns), "lid_z": float(mz.z[-1]),
                                     "L_dut": v, "rel": v / base - 1.0})
        print(f"    base_dz={bdz * 1e6:.2f} um: nz={mz.shape[2]} L={v * 1e12:.3f} pH "
              f"({100 * (v / base - 1):+.2f} %)", flush=True)

    out["metal_cells"] = []
    for mc in (METAL_CELLS, 4):
        mc_model = build_level(div, metal_cells=mc)
        v = float(l_dut(mc_model, THETA0))
        out["metal_cells"].append({"metal_cells": mc, "nz": int(mc_model.shape[2]),
                                   "n_unknowns": int(mc_model.n_unknowns), "L_dut": v,
                                   "rel": v / base - 1.0})
        print(f"    metal_cells={mc}: nz={mc_model.shape[2]} L={v * 1e12:.3f} pH "
              f"({100 * (v / base - 1):+.2f} %)", flush=True)

    ml = build_level(div, lead=LEAD_PROBE)
    v = float(l_dut(ml, THETA0))
    d_ref = float(referee_value(sg, THETA0, lead=LEAD_PROBE).total - referee_value(sg, THETA0).total)
    out["lead_differential"] = {
        "lead_from": LEAD, "lead_to": LEAD_PROBE,
        "added_conductor_length": 2.0 * (LEAD_PROBE - LEAD),
        "L_dut_long_lead": v, "dL_fdfd": v - base, "dL_referee": d_ref,
        "ratio_fdfd_over_referee": (v - base) / d_ref}
    print(f"    lead {LEAD * 1e6:.0f}->{LEAD_PROBE * 1e6:.0f} um: dL_fdfd={(v - base) * 1e12:+.3f} "
          f"dL_ref={d_ref * 1e12:+.3f} pH  ratio={(v - base) / d_ref:.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------
# 5. Richardson extrapolation

def richardson(h: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    """Extrapolations of ``L(h) = L_inf - C h^p`` to ``h -> 0``: ``p = 1``
    and ``p = 2`` from the finest pair, and the OBSERVED order from the
    finest three levels (solved for ``p`` from the two differences, then
    ``L_inf``). Returns every estimate and their range."""
    h = np.asarray(h, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    o = np.argsort(-h)               # coarse -> fine
    h, v = h[o], v[o]
    est: dict[str, float] = {}
    if len(h) < 2:
        return {"h": h.tolist(), "values": v.tolist(), "observed_order": None,
                "estimates": {}, "range": [float(v[-1]), float(v[-1])]}
    for p in (1.0, 2.0):
        h1, h2 = h[-2], h[-1]
        est[f"p{int(p)}_finest_pair"] = float(v[-1] + (v[-1] - v[-2]) * h2 ** p / (h1 ** p - h2 ** p))
    observed = None
    if len(h) >= 3:
        h1, h2, h3 = h[-3], h[-2], h[-1]
        d1, d2 = v[-2] - v[-3], v[-1] - v[-2]
        if d1 * d2 > 0 and abs(d2) < abs(d1):
            def resid(p: float) -> float:
                return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - d1 / d2
            lo, hi = 0.2, 6.0
            if resid(lo) * resid(hi) < 0:
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if resid(lo) * resid(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                observed = 0.5 * (lo + hi)
                est["observed_order"] = float(v[-1] + d2 * h3 ** observed / (h2 ** observed - h3 ** observed))
    vals = list(est.values())
    return {"h": h.tolist(), "values": v.tolist(), "observed_order": observed,
            "estimates": est, "range": [float(min(vals)), float(max(vals))]}


def vertical_budget(study: dict[str, Any]) -> dict[str, Any]:
    """The SECOND convergence direction, extrapolated. The levels refine in
    plane only (``base_dz`` and ``metal_cells`` are held fixed so the
    vertical error is common to all three and cancels out of the in-plane
    Richardson); this turns the study's own ``base_dz`` sequence at the
    coarsest level into the correction that in-plane extrapolation cannot
    see. Reported, not hidden: it has the SIGN of the V1 gap and it is the
    second-largest identified term."""
    vg = study.get("systematics", {}).get("vertical_grid")
    if not vg or len(vg) < 3:
        return {}
    h = [float(v["base_dz"]) for v in vg]
    rel = [float(v["rel"]) for v in vg]
    r = richardson(h, [1.0 + v for v in rel])
    out: dict[str, Any] = {
        "base_dz": h, "rel": rel, "lid_z": [float(v["lid_z"]) for v in vg],
        "nz": [int(v["nz"]) for v in vg],
        "observed_order": r["observed_order"], "estimates": r["estimates"],
        "range": [r["range"][0] - 1.0, r["range"][1] - 1.0],
        "note": "measured at the coarsest in-plane level; assumed separable",
    }
    return out


# ----------------------------------------------------------------------------
# 6. gates

def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    ref = study["referee"]["area_exact"]["total"]
    levels = [study["levels"][k] for k in sorted(study["levels"], key=float)]
    g: dict[str, Any] = {}

    dee = study["referee"]["deembedded"]["total"]
    # The referee's bridge is a plain bar; the FIXTURE's bridge is shorted to
    # the ground plane over the width of the short standard's post, so the
    # fixture subtracts LESS than a whole bridge. The thru probe measures
    # that at every level on the same grid and the same column separation
    # (L_thru = L(bar) - L(bridge as built), the two being the same M2 cells),
    # so ``L_dut - L_thru = L(strip) - L(whole bridge)`` -- the quantity the
    # referee's ``deembedded`` block computes. The correction is positive, so
    # it makes the gap LARGER; the uncorrected numbers are reported too.
    bias = {k: v["L_deembedded"] for k, v in (study.get("thru_probe") or {}).get("levels", {}).items()}
    rich_raw = study["richardson"]
    rich = study.get("richardson_post_corrected") or rich_raw
    rel = [v / dee - 1.0 for v in rich["range"]]
    g["V1"] = {
        "referee_deembedded": dee,
        "referee_strip_only": ref,
        "referee_bridge": study["referee"]["deembedded"]["bridge"],
        "quantity": ("L(strip) - L(short-standard bridge), the FDFD side corrected for the "
                     "post short by the thru probe; see bridge_segments and thru_probe"),
        "post_corrected": bool(bias and len(bias) == len(levels)),
        "post_bias": bias,
        "per_level_gap": {f"W/{lv['div']:g}": (lv["L_dut"] - bias.get(f"{lv['div']:g}", 0.0)) / dee - 1.0
                          for lv in levels},
        "per_level_gap_uncorrected": {f"W/{lv['div']:g}": lv["L_dut"] / dee - 1.0 for lv in levels},
        "per_level_gap_vs_strip_only": {f"W/{lv['div']:g}": lv["L_dut"] / ref - 1.0 for lv in levels},
        "L_extrapolated_estimates": rich["estimates"],
        "L_extrapolated_range": rich["range"],
        "L_extrapolated_range_uncorrected": rich_raw["range"],
        "observed_order": rich["observed_order"],
        "observed_order_uncorrected": rich_raw["observed_order"],
        "rel_range": rel,
        "rel_range_uncorrected": [v / dee - 1.0 for v in rich_raw["range"]],
        "rel_range_vs_strip_only": [v / ref - 1.0 for v in rich_raw["range"]],
        "tolerance": 0.05,
        "passed": bool(max(abs(rel[0]), abs(rel[1])) <= 0.05),
    }
    # V1c: the same comparison with the SECOND (vertical) grid direction
    # extrapolated too -- the in-plane levels share one vertical grid, so the
    # in-plane Richardson above cannot see that error at all.
    vb = study.get("vertical_budget") or {}
    if vb.get("range"):
        corr = vb["range"]
        lo = min(v * (1.0 + c) for v in rich["range"] for c in corr)
        hi = max(v * (1.0 + c) for v in rich["range"] for c in corr)
        relc = [lo / dee - 1.0, hi / dee - 1.0]
        g["V1c"] = {"vertical_correction_range": corr,
                    "vertical_observed_order": vb["observed_order"],
                    "L_range_corrected": [lo, hi], "rel_range": relc, "tolerance": 0.05,
                    "passed": bool(max(abs(relc[0]), abs(relc[1])) <= 0.05)}
    # V1b: the PEC fixture carries no DC internal inductance, so the
    # uniform-current value must exceed it by the conductor's internal
    # inductance -- a FINITE limit, not something that shrinks. The scale
    # referee is the round-wire DC value mu0 l / (8 pi) for the same
    # conductor length, which bounds a flat strip's from above.
    length = (study["area_gate"]["strip_centreline_length"]
              + study["area_gate"]["underpass_centreline_length"])
    l_int_round = MU0 * length / (8.0 * np.pi)
    dl = [lv["L_dut"] - lv["L_pec"] for lv in levels]
    frac = [v / l_int_round for v in dl]
    steps = [abs(b - a) for a, b in zip(dl, dl[1:])]
    g["V1b"] = {
        "conductor_length": length, "internal_inductance_round_wire": l_int_round,
        "per_level_deficit": {f"W/{lv['div']:g}": d for lv, d in zip(levels, dl)},
        "per_level_over_round_wire": {f"W/{lv['div']:g}": v for lv, v in zip(levels, frac)},
        "per_level_rel": {f"W/{lv['div']:g}": lv["L_pec_over_L_uniform_minus_1"] for lv in levels},
        "level_to_level_change": steps,
        "all_positive_and_below_round_wire": bool(all(0.0 < v < 1.0 for v in frac)),
        "finest_within_25_percent": bool(abs(frac[-1] - 1.0) <= 0.25),
        "converging": bool(len(steps) >= 2 and abs(steps[-1]) <= 0.25 * abs(steps[0])),
        "passed": bool(all(0.0 < v < 1.0 for v in frac) and abs(frac[-1] - 1.0) <= 0.25
                       and len(steps) >= 2 and abs(steps[-1]) <= 0.25 * abs(steps[0])),
    }
    worst = max(max(lv["grad_vs_fd_rel"]) for lv in levels)
    g["V2"] = {"per_level": {f"W/{lv['div']:g}": lv["grad_vs_fd_rel"] for lv in levels},
               "fd_order": {f"W/{lv['div']:g}": lv["fd_order"] for lv in levels},
               "worst": worst, "tolerance": 1e-4, "passed": bool(worst <= 1e-4)}
    gref = study["referee"]["gradient_deembedded_fd4"]
    per = {}
    for lv in levels:
        per[f"W/{lv['div']:g}"] = {
            "ratio": [a / b for a, b in zip(lv["grad"], gref)],
            "same_sign": bool(all(a * b > 0 for a, b in zip(lv["grad"], gref))),
        }
    fin = per[f"W/{levels[-1]['div']:g}"]
    g["V3"] = {"referee_gradient": gref,
               "referee_gradient_strip_only": study["referee"]["gradient_area_exact_fd4"],
               "per_level": per,
               "finest_within_15_percent": bool(max(abs(r - 1.0) for r in fin["ratio"]) <= 0.15),
               "all_same_sign": bool(all(v["same_sign"] for v in per.values())),
               "passed": bool(all(v["same_sign"] for v in per.values())
                              and max(abs(r - 1.0) for r in fin["ratio"]) <= 0.15)}
    dw = [lv["grad"][2] for lv in levels]
    steps = [abs(b - a) for a, b in zip(dw, dw[1:])]
    g["V4"] = {"dL_dwidth": dw, "level_to_level_change": steps,
               "passed": bool(all(b < a for a, b in zip(steps, steps[1:]))) if len(steps) >= 2 else None}
    w = study.get("wall_gate")
    if w and "rel_change_plus_4_cells" in w:
        g["V5"] = {"rel_change_plus_4_cells": w["rel_change_plus_4_cells"], "tolerance": 0.01,
                   "mechanism": "graded wall padding (the PML is unusable at 100 MHz, see the module doc)",
                   "passed": bool(abs(w["rel_change_plus_4_cells"]) <= 0.01)}
    t = study.get("thru_probe")
    if t:
        # the short's post spans everything between the two column footprints
        # except one cell on each side, so its width is analytic:
        # (bar_length - width) - 2 base_dx. The residual of the thru probe is
        # what that post shorts out, and it must track it.
        keys = sorted(t["levels"], key=float)
        post = {k: (t["bar_length"] - WIDTH) - 2.0 * t["levels"][k]["base_dx"] for k in keys}
        res = [t["levels"][k]["over_referee_bar"] for k in keys]
        pw = [post[k] for k in keys]
        g["V6"] = {
            "what": ("the DUT made identical to the short standard's bridge must de-embed to a "
                     "small part of that bridge, and the residual must track the post's width"),
            "referee_bar": t["referee_bar"],
            "per_level": {k: t["levels"][k]["over_referee_bar"] for k in keys},
            "post_width": post,
            "post_width_over_bar_length": {k: v / t["bar_length"] for k, v in post.items()},
            "residual_tracks_post_width": bool(
                all(b > a for a, b in zip(res, res[1:])) and all(b > a for a, b in zip(pw, pw[1:]))),
            "worst_over_referee_bar": t["worst_over_referee_bar"],
            "tolerance": THRU_TOL,
            "passed": bool(t["worst_over_referee_bar"] <= THRU_TOL
                           and all(b > a for a, b in zip(res, res[1:]))
                           and all(b > a for a, b in zip(pw, pw[1:])))}
    a = study["area_gate"]["area_exact"]
    g["C_area"] = {"sum_over_polygon_minus_1": a["sum_over_polygon_minus_1"],
                   "union_over_sum_minus_1": a["union_over_sum_minus_1"],
                   "tolerance": 1e-12,
                   "passed": bool(abs(a["sum_over_polygon_minus_1"]) <= 1e-12
                                  and abs(a["union_over_sum_minus_1"]) <= 1e-12)}
    return g


# ----------------------------------------------------------------------------
# 7. figure

def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [study["levels"][k] for k in sorted(study["levels"], key=float)]
    h = np.array([lv["base_dx"] for lv in levels]) * 1e6
    ldut = np.array([lv["L_dut"] for lv in levels]) * 1e12
    lpec = np.array([lv["L_pec"] for lv in levels]) * 1e12
    ref = study["referee"]
    ld = study.get("lead_differential")
    ncol = 3 if ld else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.9 * ncol, 4.8))
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(h, ldut, "o-", color="#7fb3d5", label="FDFD L_dut as de-embedded")
    ax1.plot(h, lpec, "s--", color="#9ecae1", label="FDFD L_dut, PEC metal")
    tp = study.get("thru_probe")
    rich = study.get("richardson_post_corrected") or study["richardson"]
    if tp and study.get("richardson_post_corrected"):
        lcor = np.array([lv["L_dut"] - tp["levels"][f"{lv['div']:g}"]["L_deembedded"]
                         for lv in levels]) * 1e12
        ax1.plot(h, lcor, "o-", color="#1f77b4", linewidth=1.8,
                 label="FDFD, post short removed (thru probe)")
    for j, (name, val) in enumerate(sorted(rich["estimates"].items(), key=lambda kv: -kv[1])):
        ax1.plot([0.0], [val * 1e12], marker="*", markersize=11, linestyle="none", color="#1f77b4")
        ax1.annotate(f"{name} {val * 1e12:.1f}", (0.0, val * 1e12), textcoords="offset points",
                     xytext=(7, 6 - 11 * j), fontsize=7, color="#1f77b4")
    lo, hi = rich["range"]
    ax1.fill_between([0.0, h.max() * 1.05], lo * 1e12, hi * 1e12, color="#1f77b4", alpha=0.12,
                     label="Richardson range (p=1, p=2, observed order)")
    dee = ref["deembedded"]
    ax1.axhline(dee["total"] * 1e12, color="#d62728", linestyle="-", linewidth=1.6,
                label=f"referee de-embedded, strip - bridge {dee['total'] * 1e12:.1f} pH")
    ax1.axhline(ref["area_exact"]["total"] * 1e12, color="#d62728", linestyle=":", linewidth=1.1,
                label=f"referee strip alone {ref['area_exact']['total'] * 1e12:.1f} pH "
                      f"(what the fixture does NOT measure)")
    rp = ref["reference_plane"]
    ax1.fill_between([0.0, h.max() * 1.05], rp["far_edge"] * 1e12, rp["port_line"] * 1e12,
                     color="#d62728", alpha=0.08, label="strip-only reference-plane band (+-1.5 %)")
    pp = [v["deembedded"] * 1e12 for v in dee["per_plane"].values()]
    ax1.fill_between([0.0, h.max() * 1.05], min(pp), max(pp), color="#d62728", alpha=0.35,
                     label="de-embedded reference-plane band (+-0.05 %)")
    vb = study.get("vertical_budget") or {}
    if vb.get("range"):
        lo2 = min(v * (1.0 + c) for v in rich["range"] for c in vb["range"]) * 1e12
        hi2 = max(v * (1.0 + c) for v in rich["range"] for c in vb["range"]) * 1e12
        ax1.fill_between([0.0, h.max() * 1.05], lo2, hi2, color="#2ca02c", alpha=0.15,
                         label="+ vertical-grid extrapolation (V1c)")
    ax1.set_xlabel("base_dx [um]  (W/1, W/2, W/3)")
    ax1.set_ylabel("L [pH]")
    ax1.set_xlim(0.0, h.max() * 1.05)
    ax1.set_title(f"de-embedded L vs in-plane refinement ({FREQ / 1e6:.0f} MHz)")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(alpha=0.3)

    labels = [f"W/{lv['div']:g}" for lv in levels]
    gref = np.array(ref["gradient_deembedded_fd4"])
    grad = np.array([lv["grad"] for lv in levels])
    fdv = np.array([lv["fd"] for lv in levels])
    for k, (name, c) in enumerate(zip(PARAMS, ("#1f77b4", "#2ca02c", "#d62728"))):
        ax2.plot(labels, grad[:, k] / gref[k], "o-", color=c, label=f"dL/d{name}  FDFD / referee")
        ax2.plot(labels, fdv[:, k] / gref[k], "x", color=c, markersize=9,
                 label=f"dL/d{name}  FD / referee")
    ax2.axhline(1.0, color="k", linewidth=1)
    ax2.axhspan(0.85, 1.15, color="k", alpha=0.08, label="V3 +-15 %")
    ax2.set_ylabel("gradient ratio FDFD / referee")
    ax2.set_title("shape derivatives vs the de-embedded referee (jax.grad, FD, FD4)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    if ld:
        ax3 = axes[2]
        keys = sorted(ld["levels"], key=float)
        hs = np.array([ld["levels"][k]["base_dx"] for k in keys]) * 1e6
        ratio = np.array([ld["levels"][k]["ratio_fdfd_over_referee"] for k in keys])
        ax3.plot(hs, 100.0 * (1.0 - ratio), "o-", color="#8c564b",
                 label="deficit of dL/d(lead): 1 - FDFD/referee")
        bias = np.array([(study.get("thru_probe") or {"levels": {}})["levels"]
                         .get(f"{lv['div']:g}", {}).get("L_deembedded", 0.0) for lv in levels])
        gap = (np.array([lv["L_dut"] for lv in levels]) - bias) / ref["deembedded"]["total"] - 1.0
        ax3.plot(h, -100.0 * gap, "s--", color="#1f77b4",
                 label="deficit of L_dut (post short removed) vs the referee")
        ax3.plot(hs, 100.0 * (1.0 - ratio[-1]) * hs / hs[-1], ":", color="#8c564b", linewidth=1,
                 label="O(h) through the finest point")
        ax3.set_xlabel("base_dx [um]")
        ax3.set_ylabel("deficit [%]")
        ax3.set_xlim(0.0, h.max() * 1.05)
        ax3.set_ylim(0.0, None)
        ax3.set_title("V1 diagnosis: the deficit is distributed along the strip")
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.3)

    fig.suptitle("D2: FDFD spiral vs Greenhouse referee -- uniform current, walls removed, "
                 "de-embedding-consistent referee", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. main

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", type=float, nargs="+", default=[1.0, 2.0, 3.0],
                    help="base_dx = WIDTH / level")
    ap.add_argument("--fd2-from", type=float, default=None,
                    help="use FD2 instead of FD4 for the gradient check from this level on")
    ap.add_argument("--no-systematics", action="store_true")
    ap.add_argument("--no-wall-gate", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="reuse the per-level records already in the JSON (no resume otherwise)")
    ap.add_argument("--recompute", nargs="*", default=(),
                    help="measurement blocks NOT to reuse with --from-json")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    sg = load_referee()

    t_start = time.time()
    previous: dict[str, Any] = json.loads(JSON_PATH.read_text()) if (
        args.from_json and JSON_PATH.exists()) else {}
    study: dict[str, Any] = {
        "fixture": {
            "freq": FREQ, "n_turns": N_TURNS, "r_out": R_OUT, "spacing": SPACING, "width": WIDTH,
            "lead": LEAD, "margin": MARGIN, "base_dz": BASE_DZ, "metal_cells": METAL_CELLS,
            "pad_cells": PAD_CELLS, "pad_ratio": PAD_RATIO, "sigma_volumetric": SIGMA_VOL,
            "ground_height": GROUND_H, "underpass_depth": DZ_UNDER,
            "reference_plane_shift": REF_SHIFT, "eps_dielectrics": 1.0,
            "stack": {"t_si": T_SI, "t_ox_low": T_OX_LOW, "t_m1": T_M1, "t_via": T_VIA,
                      "t_m2": T_M2, "t_ox_high": T_OX_HIGH, "t_air": T_AIR},
            "fd_step_rel": FD_STEP_REL,
        },
        "referee": {},
        "levels": dict(previous.get("levels", {})),
    }

    # --- referee -----------------------------------------------------------
    print("referee (host numpy):", flush=True)
    for conv in ("greenhouse", "area_exact"):
        g = referee_value(sg, THETA0, conv)
        study["referee"][conv] = {"total": g.total, "self_sum": g.self_sum, "mutual_sum": g.mutual_sum,
                                  "image_sum": g.image_sum, "worst_error": g.worst_error}
        print(f"  {conv:12s} total={g.total * 1e12:.3f} pH  self={g.self_sum * 1e12:.2f} "
              f"mutual={g.mutual_sum * 1e12:.2f} image={g.image_sum * 1e12:.2f} "
              f"worst_err={g.worst_error:.2g}", flush=True)
    study["referee"]["corner_systematic"] = (study["referee"]["area_exact"]["total"]
                                             / study["referee"]["greenhouse"]["total"] - 1.0)
    study["referee"]["free_space_area_exact"] = referee_value(sg, THETA0, ground=None).total
    study["referee"]["reference_plane"] = {
        "column_centre": referee_value(sg, THETA0, shift=0.5 * WIDTH).total,
        "port_line": referee_value(sg, THETA0, shift=0.0).total,
        "far_edge": referee_value(sg, THETA0, shift=WIDTH).total,
    }
    rp = study["referee"]["reference_plane"]
    study["referee"]["reference_plane_sensitivity"] = [
        rp["port_line"] / rp["column_centre"] - 1.0, rp["far_edge"] / rp["column_centre"] - 1.0]
    study["referee"]["deembedded"] = referee_deembedded_block(sg, THETA0)
    d = study["referee"]["deembedded"]
    print(f"  de-embedded   strip={d['strip'] * 1e12:.3f} - bridge={d['bridge'] * 1e12:.3f} "
          f"= {d['total'] * 1e12:.3f} pH   (reference-plane spread "
          f"{100 * d['reference_plane_spread']:+.3f} % vs "
          f"{100 * d['reference_plane_spread_strip_only']:+.3f} % on the strip alone; "
          f"M2/M1 split spread {100 * d['split_spread']:.3f} %)", flush=True)
    study["referee"]["gradient_deembedded_fd4"] = referee_gradient(sg, THETA0, "area_exact", True)
    study["referee"]["gradient_area_exact_fd4"] = referee_gradient(sg, THETA0, "area_exact", False)
    study["referee"]["gradient_greenhouse_fd4"] = referee_gradient(sg, THETA0, "greenhouse", False)
    study["area_gate"] = area_gate(sg, THETA0)
    t = study.get("thru_probe")
    if t:
        # the short's post spans everything between the two column footprints
        # except one cell on each side, so its width is analytic:
        # (bar_length - width) - 2 base_dx. The residual of the thru probe is
        # what that post shorts out, and it must track it.
        keys = sorted(t["levels"], key=float)
        post = {k: (t["bar_length"] - WIDTH) - 2.0 * t["levels"][k]["base_dx"] for k in keys}
        res = [t["levels"][k]["over_referee_bar"] for k in keys]
        pw = [post[k] for k in keys]
        g["V6"] = {
            "what": ("the DUT made identical to the short standard's bridge must de-embed to a "
                     "small part of that bridge, and the residual must track the post's width"),
            "referee_bar": t["referee_bar"],
            "per_level": {k: t["levels"][k]["over_referee_bar"] for k in keys},
            "post_width": post,
            "post_width_over_bar_length": {k: v / t["bar_length"] for k, v in post.items()},
            "residual_tracks_post_width": bool(
                all(b > a for a, b in zip(res, res[1:])) and all(b > a for a, b in zip(pw, pw[1:]))),
            "worst_over_referee_bar": t["worst_over_referee_bar"],
            "tolerance": THRU_TOL,
            "passed": bool(t["worst_over_referee_bar"] <= THRU_TOL
                           and all(b > a for a, b in zip(res, res[1:]))
                           and all(b > a for a, b in zip(pw, pw[1:])))}
    a = study["area_gate"]["area_exact"]
    print(f"  area gate (area-exact): sum/polygon-1={a['sum_over_polygon_minus_1']:+.2e} "
          f"union/sum-1={a['union_over_sum_minus_1']:+.2e}  "
          f"(Greenhouse convention union/sum-1="
          f"{study['area_gate']['greenhouse']['union_over_sum_minus_1']:+.3e})", flush=True)
    print(f"  reference plane band: {rp['far_edge'] * 1e12:.2f} .. {rp['port_line'] * 1e12:.2f} pH "
          f"({100 * study['referee']['reference_plane_sensitivity'][0]:+.2f} % / "
          f"{100 * study['referee']['reference_plane_sensitivity'][1]:+.2f} %)", flush=True)

    def dump() -> None:
        study["seconds"] = time.time() - t_start
        # --from-json recomputes the derived blocks in seconds; the wall time
        # of the run that actually did the solves is what the doc quotes
        prev_s = previous.get("seconds_measuring") or previous.get("seconds")
        study["seconds_measuring"] = (float(prev_s) if args.from_json and prev_s
                                      else study["seconds"])
        JSON_PATH.write_text(json.dumps(study, indent=1, sort_keys=False))

    dump()

    # --- FDFD levels -------------------------------------------------------
    print("FDFD levels (volumetric metal, graded wall padding):", flush=True)
    for div in args.levels:
        if f"{div:g}" in study["levels"]:
            lv = study["levels"][f"{div:g}"]
            print(f"  W/{div:g}: reused from the JSON (L_dut={lv['L_dut'] * 1e12:.3f} pH)", flush=True)
            continue
        order = 2 if (args.fd2_from is not None and div >= args.fd2_from) else 4
        study["levels"][f"{div:g}"] = run_level(div, order=order)
        dump()

    # --- systematics and gates --------------------------------------------
    coarse = min(args.levels)

    def block(name: str, fn: Callable[[], Any]) -> None:
        """Compute a measurement block, or reuse the one in the JSON with
        ``--from-json`` (the blocks are independent of each other)."""
        if args.from_json and name in previous and name not in args.recompute:
            study[name] = previous[name]
            print(f"  {name}: reused from the JSON", flush=True)
        else:
            study[name] = fn()
        dump()

    if not args.no_wall_gate:
        print("wall gate (V5):", flush=True)
        block("wall_gate", lambda: wall_gate(coarse))
    if not args.no_systematics:
        print("thru probe (V6: what the de-embedding's zero is):", flush=True)
        block("thru_probe", lambda: thru_probe(sg, args.levels))
        print("sigma plateau (A):", flush=True)
        block("sigma_plateau", lambda: sigma_plateau(coarse))
        print("fixture systematics:", flush=True)
        block("systematics", lambda: fixture_systematics(sg, coarse))
        print("lead-length differential per level (the V1 diagnosis):", flush=True)
        block("lead_differential", lambda: lead_differential(
            sg, args.levels, {k: v["L_dut"] for k, v in study["levels"].items()}))

    levels = [study["levels"][k] for k in sorted(study["levels"], key=float)]
    study["richardson"] = richardson([lv["base_dx"] for lv in levels], [lv["L_dut"] for lv in levels])
    bias = {k: v["L_deembedded"] for k, v in (study.get("thru_probe") or {}).get("levels", {}).items()}
    if len(bias) == len(levels):
        study["richardson_post_corrected"] = richardson(
            [lv["base_dx"] for lv in levels],
            [lv["L_dut"] - bias[f"{lv['div']:g}"] for lv in levels])
    study["vertical_budget"] = vertical_budget(study)
    study["gates"] = evaluate_gates(study)
    dump()

    figure(study, PNG_PATH)
    print(f"\nwrote {JSON_PATH} and {PNG_PATH} ({study['seconds'] / 60:.1f} min)", flush=True)
    for name, gate in study["gates"].items():
        print(f"  {name}: passed={gate.get('passed')}", flush=True)
    v1 = study["gates"]["V1"]
    print(f"  V1 detail: referee {v1['referee_deembedded'] * 1e12:.2f} pH "
          f"(= strip {v1['referee_strip_only'] * 1e12:.2f} - bridge "
          f"{v1['referee_bridge'] * 1e12:.2f}), extrapolated "
          f"[{v1['L_extrapolated_range'][0] * 1e12:.2f}, {v1['L_extrapolated_range'][1] * 1e12:.2f}] pH "
          f"= [{100 * v1['rel_range'][0]:+.2f}, {100 * v1['rel_range'][1]:+.2f}] %, "
          f"observed order {v1['observed_order']}", flush=True)
    if "V1c" in study["gates"]:
        v1c = study["gates"]["V1c"]
        print(f"  V1c detail: + vertical extrapolation "
              f"[{100 * v1c['vertical_correction_range'][0]:+.2f}, "
              f"{100 * v1c['vertical_correction_range'][1]:+.2f}] % -> "
              f"[{100 * v1c['rel_range'][0]:+.2f}, {100 * v1c['rel_range'][1]:+.2f}] %", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
