"""Validation study D2c: does a RIGHT-ANGLE CORNER explain the residual of
``validation/fdfd/spiral_convergence.py`` that the straight bar left over?

Answer, up front: NO. One right-angle corner's discrepancy against the
Greenhouse referee, ``[excess]_FDFD - [excess]_referee``, is
+0.532 +- 0.415 pH at one cell across the strip width and +0.131 +- 1.025 pH
at two cells (wall-extrapolated, three-point ladders): POSITIVE, i.e. the
FDFD's corner excess is slightly LESS negative than the referee's. The D2
spiral has EIGHT right-angle corners (counted from the polygon, below), so
in the one sign convention this study uses throughout -- FDFD minus
referee, in pH -- the corners contribute +1.051 +- 8.201 pH, the interval
[-7.151, +9.252] pH, to the spiral's residual. That residual is NEGATIVE
(the FDFD spiral is LOW): [-15.181, -8.249] pH at the continuum limit of
the level-invariant ladder of the same spiral (``invariant_ladder.json``,
K5's primary comparison), [-15.474, -8.441] pH after that study's measured
corrections, and [-19.191, -8.409] pH for this study's original, now
superseded, remainder. The corner interval misses all three, by 1.098,
1.291 and 1.259 pH, and its central value has the wrong sign: a positive
corner term pushes the FDFD spiral UP, so it would shrink the deficit, not
explain it. K5 FAILS. What the data DO show positively is that the FDFD
gets the corner's PHYSICS right while getting the magnitude wrong: it
reproduces the referee's corner excess to 6.8 % (one cell) and 1.7 % (two
cells) at levels where its absolute de-embedded L is 25.6 % and 13.4 % low.

Where this study sits
---------------------
The spiral study (D2) ends 5.1-7.7 % below its Greenhouse referee after
both of its measured corrections (``gates.V1.corrections.both`` =
``gates.V1c.with_post_short``), and D2b (``straight_bar_convergence.py``)
showed that most of that is NOT spiral-specific: a corner-free, via-free
straight bar on the SAME fixture has the same distributed per-unit-length
deficit (0.83 / 0.93 / 0.98 of the exact uniform-current inductance at
1 / 2 / 3 cells across the width, wall-extrapolated). D2b's own conclusion
names the next decomposition: "separating a corner contribution from the
via would need a corner-only DUT on the same fixture". This is that DUT.

Since then the level-invariant ladder (``invariant_ladder.py``, study P)
has re-measured the same spiral on a fixture whose walls, short standard
and port gap are physical coordinates, refined jointly in x, y and z. Its
gate P6 shows that D2's plateau was mainly a vertical grid that was never
refined, and its continuum limit sits -4.558 % to -2.477 % from the same
333.055 pH referee (``residual.limit_rel_range``), -4.646 % to -2.535 %
after its measured corrections (``residual.after_corrections_rel_range``).
At a continuum limit the discretisation deficits are gone, so the D2b
per-unit-length term this study first subtracted from D2's residual no
longer applies: K5's primary comparison is against that limit, read from
``invariant_ladder.json`` at run time, and the original D2-minus-D2b
remainder is kept as a secondary, reported result.

The DUT (``rfx.fdfd.spiral``, ``dut_kind="bend"``, an additive hook)
-------------------------------------------------------------------
ONE L-shaped strip of width ``W`` and thickness 2 um on the top metal, with
a single right angle: a leg of length ``l1`` in ``+y`` -- the direction in
which the DUT current leaves the outer lead column, i.e. the spiral's LEAD
direction -- then the corner, then a leg of length ``l2`` in ``-x``.
``l1`` and ``l2`` are centre-to-centre lengths: the outer lead-column
footprint centre is ``(l2/2, -l1/2)``, the corner ``(l2/2, l1/2)`` and the
inner footprint centre ``(-l2/2, l1/2)``, so the de-embedded centreline is
exactly ``l1 + l2`` long and the drawn strip (``gds.offset_polyline`` of the
three-point centreline, mitred, plan area exactly ``W x`` centreline
length: measured 2.1e-9 m^2 = ``(l1 + l2 + W) W`` to -4.4e-16 relative) is
``l1 + l2 + W``, overhanging each footprint centre by ``W/2`` exactly as
the straight bar does. ``theta = (l1, l2, W)``; the four x breakpoints are
``+-(l2 +- W)/2`` and the four y breakpoints ``+-(l1 +- W)/2``, so both
``W x W`` footprints are grid-exact at every ``base_dx``.

The two lead columns therefore differ in BOTH x and y: unlike the bar's
they are not collinear, and unlike the spiral's they do not share a port
line. Everything else -- grid, graded wall padding, lead columns, lumped
ports, the three fixtures, the open/short de-embedding, the traced
body-fitted metric -- is the spiral fixture unchanged.

The short standard, exactly (the referee must subtract exactly this)
--------------------------------------------------------------------
``build_spiral`` places the short standard from the two column footprints,
and the bend reuses that placement unchanged: a bar out of each column on
that terminal's own metal level (M2 for both here), covering the cells where
the DUT's current leaves the lead, running to a COMMON PEC post that goes
from the ground up through the metal levels. In cell indices, with ``outer``
the ``+x`` column and ``inner`` the other one:

* the post occupies ``i in [inner.i1 + 1, outer.i0 - 1)`` (one isolating
  cell on each side -- a post touching a column would make that port's own
  Ez edges PEC), ``j`` over the UNION of the two columns' terminal rows,
  ``z`` from the ground up through M2;
* the outer arm is ``i in [post_i0, outer.i1)`` on the OUTER column's
  terminal row, the inner arm ``i in [inner.i0, post_i1)`` on the INNER
  column's row. Both overlap the post's x range on their own row, which is
  what connects them to it.

For the spiral and the straight bar the two terminal rows coincide and this
is bit-identical to the single-row code it generalises (verified in this
session on six grids -- the D2 spiral at W/1 and W/3, the D2b bar at W/1
(W = 10 um) and W/3 (W = 20 um), a ``lead_stub_cells = 1`` spiral and a
3-turn spiral: cells, PEC edge masks, grid lines, columns, ports and
``n_unknowns`` unchanged bit for bit).

Driven differentially the post carries no current (the two arms' post
currents cancel -- that is why its impedance drops out of
``Z11 - Z12 - Z21 + Z22``), so the short's differential path is
``column + two arms + column`` and the DUT's is ``column + strip + column``;
the columns are z-directed and the arms and the strip are x/y-directed, so
every column-to-conductor mutual vanishes and the columns cancel term by
term:

    L(de-embedded) = L(strip between the planes) - L(the two arms).

Each arm runs from its reference plane (the footprint CENTRE, D2's
convention) to the post's near edge, i.e. ``W/2 + one cell``: 15.000 /
10.000 / 8.333 um at W/1 / W/2 / W/3, read off the built grid rather than
assumed. BOTH arms are x-directed. For the bend that is perpendicular to
the DUT current at the outer column (as for the spiral) and collinear with
it at the inner column (as for the bar) -- the one asymmetry of this
fixture. It costs almost nothing on the quantity that is gated: taking the
arms at ``W/2`` or at ``W/2 + 2 base_dx`` instead, or subtracting no bridge
at all, moves the referee's corner excess by 0.19 % of itself
(``arm_sensitivity``: -7.7798 / -7.7846 / -7.7934 pH for the three arm
lengths at W/2, -7.7783 pH strip-only) while it moves each fixture's own L
by 8 %. That insensitivity is the reason the corner excess, not the
absolute L, is this study's measurement.

The referee and the corner excess
---------------------------------
Greenhouse (``validation/crossval/comparators/spiral_greenhouse.py``:
Hoer-Love partial self and mutual inductances of rectangular bars) in the
AREA-EXACT corner convention of D2 -- each bar translated by ``+W/2`` along
its own direction except the two free ends, so the corner square is covered
exactly once and the two rectangles are disjoint -- plus the PEC-ground
image, minus the short's two arms valued the same way. Gate ``K_area``:
the area-exact rectangles sum to ``(l1 + l2) W`` and their UNION equals
that sum, both to 2.2e-16 relative; the delivered Greenhouse convention
misses it by -1.25 % on the union (the corner square counted twice in one
quadrant and not at all in the opposite one) while giving the same sum, so
the non-overlap half of the gate is what pins the convention. On the total
the two conventions differ by only +0.0055 % here (101.8310 vs 101.8254 pH)
because the bend's two bars have no mutual term to move.

The comparison quantity is the CORNER EXCESS

    excess = L(bend, l1 + l2) - L(bar, l1 + l2)

at equal total centreline length, on both sides. It is NEGATIVE, and the
referee says why (``excess.referee_mechanism``, strip only): the bend's two
perpendicular bars have EXACTLY zero mutual inductance while one straight
bar of the same length carries the collinear mutual of its two halves
inside its self term, which is worth -26.989 pH; the ground images lose the
same coupling and give +19.211 pH of it back; net -7.778 pH. So at equal
length a bend is the lower inductance by 7.1 % of the bar. That sign, in
both tools at every level, is gate K3.

The per-level discrepancy of the excess, ``[excess]_FDFD - [excess]_referee``,
is the discrepancy of ONE right-angle corner.

Protocol (identical to D2 and D2b)
----------------------------------
100 MHz, ``sigma_volumetric = 3e6`` S/m (skin depth 29.1 um = 2.9 W and
14.5 x the 2 um thickness, so the current is uniform over the cross-section
and L carries the DC internal inductance, as the referee does), VACUUM
dielectrics, the same small stack (20 um Si + 1 um oxide + M1 + via + M2;
ground to M2 centre 26 um), ``base_dz = 10 um``, ``metal_cells = 2``,
``margin = 10 um``, ``port_gap_cells = 1``, ``pad_ratio = 1.5``,
``W = 10 um``, ``t = 2 um``, ``l1 = l2 = 100 um`` and therefore a 200 um
straight bar. Levels ``base_dx = W/1, W/2, W/3``; ``pad_cells`` 4 / 6 / 8
three-point wall ladders -- the scheme D2b validated against direct
``pad_cells = 12`` solves (+0.026 % on a +0.9 % correction, +0.107 % on a
+2.9 % one, against +0.807 % for the 4/6 pair alone) -- wherever both legs
fit under the single-solve cap.

Cost, and why W/3 has no wall ladder (``N_MAX = 110000``)
---------------------------------------------------------
The bend's meshed box is SQUARE (130 x 130 um), so its LU fill is far worse
than the elongated bar and spiral boxes at the same unknown count. Measured
here, one three-fixture solve (SuperLU/COLAMD, x64 CPU, two or three other
solver jobs on the machine, so every time is an UPPER BOUND):

  bend  W/1 pad 4/6/8   N = 22044 / 34892 / 51900     18 / 47 / 98 s
  bend  W/2 pad 4/6/8   N = 56455 / 79209 / 107371   115 / 239 / 466 s
  bend  W/3 pad 4       N = 106752                   324 s
  bar   W/1 pad 4/6/8   N = 17344 / 29592 / 46000      8 / 24 / 68 s
  bar   W/2 pad 4/6/8   N = 37655 / 58009 / 83771     28 / 77 / 184 s
  bar   W/3 pad 4/6     N = 64452 / 93740             65 / 142 s

Peak RSS scales with the fill, not with N: 8.4 GB at N = 56455 and 18.8 GB
at N = 107371 with ``linear_solve.factor_cache_size(1)`` (which this study
sets around every forward solve -- the default of 4 kept factors measured
10.8 GB at N = 56455 for nothing, since a forward-only solve never reuses
one). ``MMD_AT_PLUS_A`` was tried on the bend at W/2 and abandoned: it had
not finished one three-fixture solve in 600 s against COLAMD's 115 s,
exactly the 3-D behaviour ``linear_solve``'s docstring reports.

Three grids are therefore above the cap and are recorded as skips in
``study["skipped"]``: bend W/3 pad 6 (N = 141440) and pad 8 (N = 182784),
bar W/3 pad 8 (N = 129684). W/3 consequently has a ONE-point ladder on the
bend leg and a two-point ladder on the bar leg, so by the three-point rule
(K1) it is EXCLUDED from every gate and from the attribution; its raw
``pad_cells = 4`` numbers are reported. Total measuring cost of everything
on disk: 34.3 min (``seconds_measuring``); a ``--from-json`` reassembly,
which builds and solves nothing, re-derives everything else and writes the
JSON (and the PNG unless ``--no-figure``) in about a second
(``seconds``, which is the wall clock of the LAST run and so moves).

Run::

    .venv/bin/python validation/fdfd/corner_convergence.py
        [--levels 1 2 3] [--kinds bend bar] [--pads 4 6 8]
        [--n-max 110000] [--no-gradient] [--fresh] [--from-json] [--no-figure]

writes ``corner_convergence.json`` and ``corner_convergence.png`` next to
this file. Every measured point is written to the JSON as soon as it is
measured and every later run REUSES what is already there (pass ``--fresh``
to start over), so the study is assembled in chunks -- as it was: seven
invocations, none longer than 600 s -- and an interrupted run loses nothing.
Everything derived (referee, ladders, excess, attribution, gates, figure) is
re-derived from scratch on every run, so a reused point can never carry an
older derivation forward. ``--from-json`` is the strict form of that
reassembly: it reuses every measured point, every recorded skip and the
gradient from the JSON, builds and solves NOTHING, and stops with an error
if a requested point is neither measured nor a recorded skip. Every run
also reads ``invariant_ladder.json`` for K5 (``INVARIANT_LADDER_PATH``).

Results (every number is in the JSON)
-------------------------------------
Referee, de-embedded (strip minus the two arms), per level::

  level   arms      bend strip  bend bridge  bend L    bar strip  bar bridge  bar L     excess
  W/1     15.000 um   101.831      8.936     92.895     109.609     8.921    100.689   -7.7934
  W/2     10.000 um   101.831      4.894     96.937     109.609     4.888    104.721   -7.7846
  W/3      8.333 um   101.831      3.698     98.133     109.609     3.693    105.916   -7.7826

FDFD, de-embedded ``L_diff``, in pH::

  level  bend pad4   bar pad4   bend wall-extr.    bar wall-extr.     gap bend   gap bar
  W/1     68.5481    75.2438    69.0924 (+0.794%)  76.3535 (+1.475%)  -25.62 %   -24.17 %
  W/2     82.4735    88.3117    83.9277 (+1.763%)  91.5809 (+3.702%)  -13.42 %   -12.55 %
  W/3     85.6468    90.2074    no 3-point ladder  no 3-point ladder  -12.72 %   -14.83 %
                                                                      (pad 4)    (pad 4)

The wall bias is NOT the same for the two shapes -- the L-bend's return
path is shorter than the open bar's, so its ``pad_cells = 4`` bias is about
half: +0.794 % vs +1.475 % at W/1 and +1.763 % vs +3.702 % at W/2 (and the
bar's two-point W/3 ladder reads +7.940 %). That asymmetry is exactly why
the excess must be wall-corrected leg by leg and why the raw
``pad_cells = 4`` excess is useless: it drifts the WRONG way with
refinement (below).

Corner excess and the discrepancy of ONE corner, in pH::

  level  excess referee  excess FDFD (wall-corr.)  discrepancy         excess ratio
  W/1       -7.7934        -7.2611                 +0.5323 +- 0.4151    0.9317
  W/2       -7.7846        -7.6533                 +0.1313 +- 1.0251    0.9831
  W/3       -7.7826        -4.5607 (pad 4 only)    +3.2219 (EXCLUDED)   0.5860

The uncertainty is the SUM of the two legs' own 2-point/3-point wall
extrapolation spreads (0.201 % and 0.362 % of L at W/1, 0.390 % and 0.762 %
at W/2) -- systematics of the same extrapolation, so they are added, not
combined in quadrature. At W/2 the uncertainty is 7.8 x the discrepancy: the
measurement says the corner discrepancy is BELOW about 1 pH there, not that
it is 0.13 pH.

Gates
-----
K1  STRUCTURAL, PASSES. Only levels with a full ``pad_cells`` 4/6/8 ladder
    on BOTH legs enter a gate: W/1 and W/2 (ladder points [3, 3] each).
    W/3 has [0, 2] -- three of its six grids are above ``N_MAX = 110000``
    (above) -- so it is kept out of K3, K4 and K5. Its bend leg has no
    ladder at all, so it has no wall-corrected excess; only its raw
    ``pad_cells = 4`` numbers are recorded (``excess_fdfd_pad4``,
    ``discrepancy_pad4``; a level with a two-point ladder on BOTH legs
    would carry ``*_two_point_EXCLUDED`` keys instead).
    ``wall_ladder_points_per_level`` and ``study["skipped"]`` make the rule
    auditable from the JSON alone.
K2  PASSES. ``jax.value_and_grad`` of the de-embedded ``L_diff`` of the
    BEND in ``theta = (l1, l2, W)`` against FD4 with 1 % steps (which move
    L by ~1e-2 relative, far above the ~1e-9 LU noise floor) at
    ``base_dx = W`` (N = 22044): AD gives ``dL/dl1`` = +4.514579e-07 and
    ``dL/dW`` = -2.2004126e-06 H/m, FD4 gives +4.514578e-07 and
    -2.2004127e-06, agreeing to 1.59e-07 and 4.28e-08 relative (gate
    1e-4); 23 s for the gradient and 134 s for the eight FD solves.
    ``dL/dl2`` = +3.543002e-07 H/m is NOT equal to ``dL/dl1`` -- this
    fixture is not symmetric in ``l1 <-> l2`` (the short standard's two
    arms are both x-directed, so stretching ``l2`` also stretches the cell
    that sets their length, while stretching ``l1`` only moves them apart
    in y) -- and this study never checked it against FD: the JSON has no
    FD4 for ``l2`` (``gradient.fd_params`` is ``[l1, width]``). An
    independent verifier later measured FD4 with a 0.5 % step at
    3.542998e-07 H/m against AD 3.543002e-07, 9.69e-07 relative. That
    number was supplied by the verifier; it is NOT in the JSON and is not
    part of K2.
K3  PASSES. On the ADMITTED levels -- the set K4 and K5 use -- the
    wall-corrected corner excess is negative in BOTH tools: -7.2611 vs
    -7.7934 pH at W/1 and -7.6533 vs -7.7846 pH at W/2, i.e. the FDFD
    reproduces the sign and the mechanism: perpendicular halves do not
    couple, collinear ones do. The RATIO to the referee is 0.9317 at one
    cell across W and 0.9831 at two -- the corner excess is reproduced to
    6.8 % and 1.7 % at levels where the absolute L is 25.6 % and 13.4 %
    low, so whatever the FDFD is missing is not the corner. W/3 has no
    three-point ladder and does not enter the gate: its raw pad-4 excess,
    -4.5607 pH against -7.7826 pH (the same sign, ratio 0.5860), is
    reported under ``excluded_informational`` only.
K4  PASSES on the admitted series: ``|discrepancy|`` falls 0.5323 ->
    0.1313 pH from one to two cells across W. It FAILS on the raw
    ``pad_cells = 4`` series, which GROWS 1.0977 -> 1.9464 -> 3.2219 pH
    (``raw_pad4_decreases`` is false) -- that growth is the wall bias
    creeping in at different rates on the two shapes, D2b's finding
    applied to a shape pair.
K5  FAILS. The D2 spiral has 8 right-angle corners, counted from
    ``gds.rect_spiral``'s own centreline (``corner_count``): the 8 sides
    of the 2-turn strip give 7 interior joints, all right angles; the last
    side meets the inner extension at one more; the lead is collinear with
    side 0 by construction (0); the strip-to-underpass transition is a via
    with a 180-degree in-plane reversal on the lower metal, not a right
    angle (0). ONE sign convention throughout: FDFD minus referee, in pH,
    so a negative range is the FDFD LOW. Each corner adds its discrepancy
    to the spiral's FDFD-minus-referee, so the corners add
    8 x (+0.1313 +- 1.0251) pH = +1.051 +- 8.201 pH, the interval
    [-7.151, +9.252] pH (``finest_admitted``, two cells across W), and K5
    passes only if that interval intersects the spiral's residual.
    PRIMARY (the gate): the continuum-limit residual of the level-invariant
    ladder, ``residual.limit_rel_range`` x ``referee.total`` of
    ``validation/fdfd/invariant_ladder.json`` (read at run time; its
    referee is D2's 333.055 pH, checked to 1e-12), [-15.181, -8.249] pH.
    The corner interval's lower end, -7.151 pH, lies 1.098 pH above its
    upper end. Reported beside it: the same limit after that study's
    measured corrections (``residual.after_corrections_rel_range``),
    [-15.474, -8.441] pH, missed by 1.291 pH. SECONDARY (superseded,
    reported, never gated): the original construction -- D2's residual
    after both of its corrections, [-25.586, -16.869] pH, minus D2b's
    per-unit-length deficit at 3-4 cells across W (ratios 0.9746-0.9808),
    [-8.460, -6.395] pH -- leaves [-19.191, -8.409] pH, missed by
    1.259 pH. It no longer applies: the invariant ladder showed D2's
    plateau to be mainly an unrefined vertical grid, and at a continuum
    limit the discretisation deficit it subtracted is gone. Against all
    three the corner term's central value, +1.051 pH, has the OPPOSITE
    sign: a positive corner term pushes the FDFD spiral up and shrinks its
    deficit instead of explaining it. (The first version of this study set
    the signed corner interval against the POSITIVE magnitude of the
    deficit and reported a marginal pass; that was a sign error.)
    8 x the raw pad-4 W/3 discrepancy, +25.775 pH, is recorded under
    ``finest_raw_pad4_EXCLUDED``: further still from every residual, and
    an artifact of the uncorrected walls (K4).
K_area PASSES (above).

Conclusion: what the data support
---------------------------------
1. The spiral's residual is NOT in its right-angle corners. At two cells
   across the width the FDFD reproduces the referee's corner excess to
   1.7 % (-7.653 vs -7.785 pH) while its absolute L is 13.4 % low, and the
   discrepancy of one corner is +0.131 +- 1.025 pH. With consistent signs
   (FDFD minus referee) the 8 corners contribute [-7.151, +9.252] pH,
   while the spiral's residual is [-15.181, -8.249] pH at the invariant
   ladder's continuum limit ([-15.474, -8.441] pH after its corrections,
   [-19.191, -8.409] pH for the superseded remainder). The corners cannot
   account for it against any of the three -- they miss by about
   1.1-1.3 pH -- and their central value has the wrong sign.
2. What the corner DOES cost is large and both tools agree on it: at equal
   total length a bend has 7.1 % less inductance than a straight bar
   (-7.78 pH of 109.61 pH), because its two halves are perpendicular and
   do not couple. The referee's decomposition is -26.99 pH of self plus
   collinear mutual, +19.21 pH given back by the ground images.
3. What the residual IS, this study does not say; it removes the corners
   from the candidates. The level-invariant ladder has since shown that
   D2's plateau was mainly its never-refined vertical grid (its gate P6),
   and its continuum limit still sits -4.558 % to -2.477 % from the
   referee, which that study reports as unexplained (with the caveat that
   the residual is the size of its own extrapolation spread). The
   candidates this study did not isolate remain: the underpass and the via
   (neither the bend nor the bar has one) and the turn-to-turn mutual
   coupling of a closed loop.
4. Method findings, for whoever runs the next decomposition: (a) the wall
   bias of ``pad_cells = 4`` depends on the SHAPE, not just on the level
   (half as large for the L-bend as for the open bar at the same
   ``base_dx``), so a difference of two shapes must be wall-corrected leg
   by leg -- the raw difference here moves the wrong way with refinement;
   (b) a square meshed box is much more expensive than an elongated one at
   the same N (18.8 GB and 466 s for the 42 x 42 x 19 bend at
   N = 107371, against 184 s for the 62 x 22 x 19 bar at N = 83771 on the
   same machine), which is what put W/3 out of reach; (c) the corner excess is almost independent of the
   bridge/reference-plane convention (0.19 % over the whole arm band
   against 8 % on each L), which is what makes it a usable observable on a
   fixture whose absolute plane is worth 11 % (D2b).

Scope fence. ONE corner, one leg length, one width, one thickness, one
frequency, one stack, one ground height; in-plane refinement only, and only
to two cells across the width with a validated wall ladder. The referee is
magnetoquasistatic with uniform current density and an infinite PEC ground
and cannot represent the short standard's grounded post -- for this fixture
the post is large (it fills the L's interior in the SHORT standard only),
which is why the bridge is valued as the two un-shorted arms and why the
arm-length band is reported; D2's thru probe measured that residual on the
spiral at 0.5-1.8 % of its referee and no equivalent probe was run here.
Nothing here validates losses, the frequency dependence or the PEC metal
model. The D2 and D2b numbers used in K5's secondary comparison are QUOTED
from their JSONs (``spiral_reference``) and the invariant ladder's residual
is READ from ``invariant_ladder.json`` at run time (``attribution.primary``
names the keys); only the corner COUNT is measured here. The corner term set
against that continuum-limit residual is the finest ADMITTED level's (two
cells across W), not a continuum limit of its own: two admitted levels
cannot be extrapolated. Sizes are CPU-SuperLU sizes and every timing is an
upper bound under contention.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time
from typing import Any, Callable, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
REFEREE_PATH = REPO / "validation" / "crossval" / "comparators" / "spiral_greenhouse.py"
JSON_PATH = HERE / "corner_convergence.json"
PNG_PATH = HERE / "corner_convergence.png"

# ---- the fixture (both tools), copied from D2 / D2b ------------------------
FREQ = 1e8
WIDTH = 10e-6
L1 = 100e-6                # bend leg along the lead direction (+y)
L2 = 100e-6                # bend leg perpendicular to it (-x)
MARGIN = 10e-6
T_SI, T_OX_LOW, T_M1, T_VIA, T_M2, T_OX_HIGH, T_AIR = 20e-6, 1e-6, 2e-6, 2e-6, 2e-6, 3e-6, 10e-6
BASE_DZ = 10e-6            # vertical grid, the SAME at every level
METAL_CELLS = 2            # >= 2 cells across the metal thickness
PAD_CELLS, PAD_RATIO = 4, 1.5
WALL_PADS = (4, 6, 8)
SIGMA_VOL = 3e6            # uniform-current plateau (measured in D2 and D2b)
GROUND_H = T_SI + T_OX_LOW + T_M1 + T_VIA + 0.5 * T_M2          # 26 um, ground -> M2 centre
FD_STEP_REL = 0.01         # 1 % steps (K2 measures the agreement they give)
LEVELS = (1.0, 2.0, 3.0)
KINDS = ("bend", "bar")
# single-solve cap. The bend's box is SQUARE (130 x 130 um meshed), so its
# LU fill is far worse than the elongated bar and spiral boxes at the same N:
# measured on this machine, one three-fixture solve peaks at 8.4 GB / 115 s
# at N = 56455 and 18.8 GB / 466 s at N = 107371 (the times are the JSON's
# ``levels.bend.2.{4,8}.seconds``), against 505 s for D2b's N = 104888
# elongated bar. 110k is therefore the cap on N here as it was
# there, and it is what excludes the bend's W/3 wall ladder (below).
N_MAX = 110_000
# the spiral D2 fixture, written out here (never imported: that study's
# driver is owned elsewhere and must not be able to move this block)
SPIRAL_FIXTURE = dict(n_turns=2, r_out=52e-6, spacing=10e-6, width=10e-6, lead=15e-6)
# D2's published residual, QUOTED from its JSON for the attribution
SPIRAL_REFEREE = 3.3305510896906523e-10        # gates.V1.referee_deembedded
SPIRAL_BOTH_REL = (-0.07682185547554532, -0.050649408342909985)   # gates.V1c.with_post_short
# D2b's wall-extrapolated per-unit-length ratio at 3 cells across W
BAR_PUL_RATIO_3CELLS = (0.9746, 0.9808)        # 4 cells and 3 cells (W = 20 um)
# the level-invariant ladder of the SAME spiral (study P, invariant_ladder.py):
# K5's primary comparison is its continuum-limit residual, READ from this JSON
# at run time (never written here), under these dotted keys
INVARIANT_LADDER_PATH = HERE / "invariant_ladder.json"
INVARIANT_LADDER_KEYS = {
    "referee": "referee.total",
    "continuum_limit_rel_range": "residual.limit_rel_range",
    "after_corrections_rel_range": "residual.after_corrections_rel_range",
}


# ----------------------------------------------------------------------------
# 1. referee (host numpy, loaded from its file; never imported by rfx)

def load_referee():
    spec = importlib.util.spec_from_file_location("spiral_greenhouse", REFEREE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bend_segments(sg, l1: float = L1, l2: float = L2, width: float = WIDTH,
                  thickness: float = T_M2, convention: str = "area_exact") -> list:
    """The bend's conductor BETWEEN THE TWO REFERENCE PLANES (the lead-column
    footprint centres), as two Greenhouse bars.

    ``"greenhouse"`` is the delivered convention -- each bar ends at the
    corner POINT, so one quadrant of the ``W x W`` corner square is covered
    twice and the opposite one not at all. ``"area_exact"`` is the spiral
    study's convention: translate every bar by ``+W/2`` along its own
    direction except the start of the first and the end of the last, which
    stay on their reference planes. The incoming leg then owns the whole
    corner square, the outgoing one starts just after it, the two rectangles
    are exactly disjoint, their union is the drawn L polygon clipped to the
    planes and the total centreline length is unchanged at ``l1 + l2``."""
    hw = 0.5 * width
    p_o = (0.5 * l2, -0.5 * l1, 0.0)
    corner = (0.5 * l2, 0.5 * l1, 0.0)
    p_i = (-0.5 * l2, 0.5 * l1, 0.0)
    if convention == "greenhouse":
        return [sg.Segment(p_o, corner, width, thickness),
                sg.Segment(corner, p_i, width, thickness)]
    if convention != "area_exact":
        raise ValueError(f"unknown corner convention {convention!r}")
    return [sg.Segment(p_o, (corner[0], corner[1] + hw, 0.0), width, thickness),
            sg.Segment((corner[0] - hw, corner[1], 0.0), p_i, width, thickness)]


def bar_segments(sg, ell: float, width: float = WIDTH, thickness: float = T_M2) -> list:
    """The straight bar's conductor between its two reference planes: ONE bar
    of length ``ell`` (centre to centre of the footprints). No corner
    convention exists and the plan area is ``ell x width`` exactly."""
    return [sg.Segment((0.5 * ell, 0.0, 0.0), (-0.5 * ell, 0.0, 0.0), width, thickness)]


def bend_bridge_segments(sg, arm_o: float, arm_i: float, l1: float = L1, l2: float = L2,
                         width: float = WIDTH, thickness: float = T_M2) -> list:
    """The bend's short standard between the same two planes: the two
    x-directed arms, from each footprint centre to the post's near edge
    (``arm_o`` and ``arm_i``, read off the built grid). Both carry the
    differential current in ``-x``, so as a series chain they are PARALLEL
    (positive mutual), ``l1`` apart in y; the post between them carries no
    differential current and is not part of the path."""
    return [sg.Segment((0.5 * l2, -0.5 * l1, 0.0), (0.5 * l2 - arm_o, -0.5 * l1, 0.0),
                       width, thickness),
            sg.Segment((-0.5 * l2 + arm_i, 0.5 * l1, 0.0), (-0.5 * l2, 0.5 * l1, 0.0),
                       width, thickness)]


def bar_bridge_segments(sg, arm_o: float, arm_i: float, ell: float, width: float = WIDTH,
                        thickness: float = T_M2) -> list:
    """The straight bar's short standard between its planes: the same two
    arms, here COLLINEAR (both on ``y = 0``, both directed ``-x``)."""
    return [sg.Segment((0.5 * ell, 0.0, 0.0), (0.5 * ell - arm_o, 0.0, 0.0), width, thickness),
            sg.Segment((-0.5 * ell + arm_i, 0.0, 0.0), (-0.5 * ell, 0.0, 0.0), width, thickness)]


def referee_point(sg, kind: str, arm_o: float, arm_i: float, l1: float = L1, l2: float = L2,
                  width: float = WIDTH, convention: str = "area_exact") -> dict[str, Any]:
    """The referee's de-embedded value of one fixture: ``L(conductor between
    the planes) - L(the short's two arms)``, both with the PEC-ground image
    (``ground_height = GROUND_H``)."""
    if kind == "bend":
        strip = sg.greenhouse_terms(bend_segments(sg, l1, l2, width, convention=convention),
                                    ground_height=GROUND_H)
        bridge = sg.greenhouse_terms(bend_bridge_segments(sg, arm_o, arm_i, l1, l2, width),
                                     ground_height=GROUND_H)
    elif kind == "bar":
        ell = l1 + l2
        strip = sg.greenhouse_terms(bar_segments(sg, ell, width), ground_height=GROUND_H)
        bridge = sg.greenhouse_terms(bar_bridge_segments(sg, arm_o, arm_i, ell, width),
                                     ground_height=GROUND_H)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return {"kind": kind, "arm_outer": arm_o, "arm_inner": arm_i,
            "L_strip": strip.total, "L_strip_self": strip.self_sum,
            "L_strip_mutual": strip.mutual_sum, "L_strip_image": strip.image_sum,
            "L_bridge": bridge.total, "L_deembedded": strip.total - bridge.total,
            "worst_error": max(strip.worst_error, bridge.worst_error)}


def area_gate(sg, l1: float = L1, l2: float = L2, width: float = WIDTH) -> dict[str, Any]:
    """Gate on the corner convention (D2's gate C, for the bend): for both
    conventions, the sum of the two bar rectangle areas and the area of
    their UNION, against the exact ``(l1 + l2) * width`` the de-embedded
    conductor must have. The area-exact convention must match BOTH to
    1e-12 relative -- the sum alone is convention-blind, so the non-overlap
    half is what pins the convention. The drawn ``gds`` polygon is checked
    too: ``(l1 + l2 + width) * width`` (it overhangs both planes by
    ``width/2``)."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    from rfx.fdfd import gds
    from rfx.fdfd import spiral as sm

    sp = sm.bend_geometry(l1, l2, width)
    drawn = abs(gds.polygon_area(sp.polygons[sm.KEY_M2][0]))
    feet = [abs(gds.polygon_area(p)) for p in sp.polygons[sm.KEY_M2][1:]]
    want = (l1 + l2) * width
    out: dict[str, Any] = {
        "l1": l1, "l2": l2, "width": width,
        "drawn_polygon_area": drawn,
        "drawn_over_expected_minus_1": drawn / ((l1 + l2 + width) * width) - 1.0,
        "footprint_over_expected_minus_1": [a / (width * width) - 1.0 for a in feet],
        "deembedded_area_expected": want,
        "drawn_centreline_length": sp.length,
    }
    for conv in ("greenhouse", "area_exact"):
        rects = []
        for s in bend_segments(sg, l1, l2, width, convention=conv):
            a = np.asarray(s.start, float)[:2]
            b = np.asarray(s.end, float)[:2]
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            ax = 0 if b[0] != a[0] else 1
            pe = 1 - ax
            r = [0.0, 0.0, 0.0, 0.0]
            r[ax], r[ax + 2] = lo[ax], hi[ax]
            r[pe], r[pe + 2] = lo[pe] - 0.5 * s.width, hi[pe] + 0.5 * s.width
            rects.append(box(*r))
        total = float(sum(r.area for r in rects))
        union = float(unary_union(rects).area)
        out[conv] = {"sum_of_areas": total, "union_area": union,
                     "sum_over_expected_minus_1": total / want - 1.0,
                     "union_over_expected_minus_1": union / want - 1.0,
                     "union_over_sum_minus_1": union / total - 1.0}
    out["passed"] = bool(
        abs(out["area_exact"]["sum_over_expected_minus_1"]) <= 1e-12
        and abs(out["area_exact"]["union_over_expected_minus_1"]) <= 1e-12
        and abs(out["area_exact"]["union_over_sum_minus_1"]) <= 1e-12
        and abs(out["drawn_over_expected_minus_1"]) <= 1e-12
        and max(abs(v) for v in out["footprint_over_expected_minus_1"]) <= 1e-12)
    return out


# ----------------------------------------------------------------------------
# 2. the FDFD side

def stack(base_dz: float = BASE_DZ, metal_cells: int = METAL_CELLS, t_air: float = T_AIR):
    from rfx.fdfd import spiral as sm
    return sm.SmallStack(t_si=T_SI, t_ox_low=T_OX_LOW, t_m1=T_M1, t_via=T_VIA, t_m2=T_M2,
                         t_ox_high=T_OX_HIGH, t_air=t_air, base_dz=base_dz,
                         metal_cells=metal_cells, eps_si=1.0, eps_ox=1.0)


def build(kind: str, div: float, pad: int = PAD_CELLS, width: float = WIDTH,
          l1: float = L1, l2: float = L2, margin: float = MARGIN):
    """The static model of either fixture at ``base_dx = width / div``: the
    L-shaped bend (``dut_kind="bend"``) or the straight bar of the same total
    centreline length ``l1 + l2`` (``dut_kind="bar"``)."""
    from rfx.fdfd import spiral as sm
    common = dict(width=width, base_dx=width / div, margin=margin, stack=stack(),
                  port_gap_cells=1, pad_cells=pad, pad_ratio=PAD_RATIO)
    if kind == "bend":
        spec = sm.SpiralSpec(dut_kind="bend", bend_l1=l1, bend_l2=l2, **common)
    elif kind == "bar":
        spec = sm.SpiralSpec(dut_kind="bar", bar_length=l1 + l2, **common)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return sm.build_spiral(spec)


def arm_lengths(model) -> tuple[float, float]:
    """The short standard's two arm lengths READ OFF THE BUILT GRID: from
    each lead-column footprint centre to the post's near edge, i.e.
    ``width/2`` plus the one isolating cell (``post_i0 = inner.i1 + 1``,
    ``post_i1 = outer.i0 - 1``). Both arms are x-directed, so both lengths
    are x distances."""
    outer, inner = model.columns
    x = model.x_nom
    xc_o = 0.5 * (x[outer.i0] + x[outer.i1])
    xc_i = 0.5 * (x[inner.i0] + x[inner.i1])
    return float(xc_o - x[outer.i0 - 1]), float(x[inner.i1 + 1] - xc_i)


def wall_distance(model, kind: str, l1: float = L1, l2: float = L2,
                  width: float = WIDTH) -> float:
    """Smallest distance from the conductor to a PEC wall (the four side
    walls and the lid; the ground is physical and stays where it is)."""
    hx = 0.5 * ((l2 if kind == "bend" else l1 + l2) + width)
    hy = 0.5 * ((l1 if kind == "bend" else 0.0) + width)
    return float(min(model.x_nom[-1] - hx, -model.x_nom[0] - hx,
                     model.y_nom[-1] - hy, -model.y_nom[0] - hy,
                     model.z[-1] - (GROUND_H + 0.5 * T_M2)))


def l_dut(model, theta=None, sigma: float | None = SIGMA_VOL):
    """De-embedded ``L_diff`` (H) of the volumetric-metal fixture -- the
    quantity the referee is compared against."""
    from rfx.fdfd import spiral as sm
    return sm.solve_spiral(model, FREQ, theta=theta, sigma_volumetric=sigma).L_diff


def run_point(kind: str, div: float, pad: int, n_max: int = N_MAX,
              verbose: bool = True) -> dict[str, Any] | None:
    """One (fixture, level, pad_cells) point. Returns ``None`` -- a recorded
    SKIP, not a failure -- when the grid would exceed ``n_max`` unknowns."""
    import jax

    from rfx.fdfd import linear_solve as ls

    model = build(kind, div, pad)
    if model.n_unknowns > n_max:
        if verbose:
            print(f"    {kind} W/{div:g} pad={pad}: N={model.n_unknowns} > {n_max}, SKIPPED",
                  flush=True)
        return None
    arm_o, arm_i = arm_lengths(model)
    rec: dict[str, Any] = {
        "kind": kind, "div": div, "base_dx": WIDTH / div, "pad_cells": pad,
        "shape": list(model.shape), "n_unknowns": int(model.n_unknowns),
        "wall_distance": wall_distance(model, kind),
        "wall_x": [float(model.x_nom[0]), float(model.x_nom[-1])],
        "wall_y": [float(model.y_nom[0]), float(model.y_nom[-1])],
        "lid_z": float(model.z[-1]), "nz": int(model.shape[2]),
        "arm_outer": arm_o, "arm_inner": arm_i,
        "build_seconds": float(model.build_seconds),
    }
    # one factor live at a time: the bend's square box has a huge LU and the
    # default cache of 4 factors multiplies the peak (measured 10.8 -> 8.4 GB
    # at N = 56455). Forward-only solves never re-use a factor anyway.
    old = ls.factor_cache_size()
    ls.factor_cache_size(1)
    try:
        t0 = time.time()
        val = l_dut(model)
        jax.block_until_ready(val)
        rec["L_dut"] = float(np.real(val))
        rec["seconds"] = time.time() - t0
    finally:
        ls.clear_factor_cache()
        ls.factor_cache_size(old)
    if verbose:
        print(f"    {kind} W/{div:g} pad={pad}: N={rec['n_unknowns']} {tuple(model.shape)} "
              f"wall={rec['wall_distance'] * 1e6:.0f} um arms="
              f"{arm_o * 1e6:.3f}/{arm_i * 1e6:.3f} um  L={rec['L_dut'] * 1e12:.4f} pH "
              f"({rec['seconds']:.0f} s)", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 3. numerics (the same formulas D2 and D2b use, copied so this file does not
#    depend on another study module)

def fd4(f: Callable[[float], float], x0: float, d: float) -> float:
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def richardson(h: Sequence[float], values: Sequence[float]) -> dict[str, Any]:
    """Extrapolations of ``L(h) = L_inf - C h^p`` to ``h -> 0``: ``p = 1`` and
    ``p = 2`` from the finest pair, and the OBSERVED order from the finest
    three levels."""
    h = np.asarray(h, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    o = np.argsort(-h)
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
                est["observed_order"] = float(
                    v[-1] + d2 * h3 ** observed / (h2 ** observed - h3 ** observed))
    vals = list(est.values())
    return {"h": h.tolist(), "values": v.tolist(), "observed_order": observed,
            "estimates": est, "range": [float(min(vals)), float(max(vals))]}


def ladder_entry(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Everything derived from one ``pad_cells`` ladder, as a pure function of
    its rows (so a ladder reused from the JSON gets exactly the same derived
    fields as a freshly measured one).

    ``L_wall_extrapolated`` is the Richardson in ``1 / wall_distance`` over
    ALL the rows, ``L_wall_extrapolated_2pt`` the same over the two COARSEST
    rows (pad 4 / 6); for a three-point ladder their difference is the error
    bar on the correction at that level. ``three_point`` is what admits the
    level into every gate: D2b validated the 3-point ladder against direct
    ``pad_cells = 12`` solves (+0.026 % on a 0.9 % correction, +0.107 % on a
    2.9 % one) and measured the 4/6 pair alone to be 7.5 x worse."""
    rows = sorted(rows, key=lambda r: r["pad_cells"])
    rich = richardson([1.0 / r["wall_distance"] for r in rows], [r["L_dut"] for r in rows])
    rich2 = richardson([1.0 / r["wall_distance"] for r in rows[:2]],
                       [r["L_dut"] for r in rows[:2]])
    entry: dict[str, Any] = {
        "pad_cells": [int(r["pad_cells"]) for r in rows],
        "wall_distance": [r["wall_distance"] for r in rows],
        "L_dut": [r["L_dut"] for r in rows],
        "n_unknowns": [int(r["n_unknowns"]) for r in rows],
        "seconds": [r["seconds"] for r in rows],
        "L_pad4": rows[0]["L_dut"], "n_pad_points": len(rows),
        "three_point": bool(len(rows) >= 3),
        "wall_richardson": rich,
        "L_wall_extrapolated": float(np.mean(rich["range"])),
        "L_wall_extrapolated_2pt": float(np.mean(rich2["range"])),
    }
    entry["rel_pad4"] = entry["L_wall_extrapolated"] / entry["L_pad4"] - 1.0
    entry["extrapolation_uncertainty"] = abs(
        entry["L_wall_extrapolated_2pt"] / entry["L_wall_extrapolated"] - 1.0)
    entry["uncertainty_abs"] = abs(entry["L_wall_extrapolated_2pt"] - entry["L_wall_extrapolated"])
    return entry


# ----------------------------------------------------------------------------
# 4. the corner excess

def excess_block(sg, levels: dict[str, Any], ladders: dict[str, Any]) -> dict[str, Any]:
    """``[L_bend - L_bar]`` at equal total centreline length, per level, on
    both sides, and the discrepancy of ONE corner.

    The FDFD excess is formed from the two WALL-CORRECTED values where both
    legs have a full three-point ``pad_cells`` 4/6/8 ladder (``admitted``),
    and from the raw ``pad_cells = 4`` values in every case
    (``excess_fdfd_pad4``, reported, never gated). The uncertainty on an
    admitted point is the SUM of the two legs' own 2-point/3-point
    extrapolation spreads -- these are systematics of the same
    extrapolation, not independent random errors, so they are added, not
    combined in quadrature."""
    out: dict[str, Any] = {"levels": {}}
    for dk in sorted(set(levels["bend"]) & set(levels["bar"]), key=float):
        bend, bar = levels["bend"][dk], levels["bar"][dk]
        if "4" not in bend or "4" not in bar:
            continue
        b4, r4 = bend["4"], bar["4"]
        ref_b = referee_point(sg, "bend", b4["arm_outer"], b4["arm_inner"])
        ref_r = referee_point(sg, "bar", r4["arm_outer"], r4["arm_inner"])
        row: dict[str, Any] = {
            "div": b4["div"], "base_dx": b4["base_dx"],
            "cells_across_width": b4["div"],
            "n_unknowns": {"bend": b4["n_unknowns"], "bar": r4["n_unknowns"]},
            "seconds_pad4": {"bend": b4["seconds"], "bar": r4["seconds"]},
            "referee": {"bend": ref_b, "bar": ref_r},
            "L_referee": {"bend": ref_b["L_deembedded"], "bar": ref_r["L_deembedded"]},
            "excess_referee": ref_b["L_deembedded"] - ref_r["L_deembedded"],
            "L_pad4": {"bend": b4["L_dut"], "bar": r4["L_dut"]},
            "excess_fdfd_pad4": b4["L_dut"] - r4["L_dut"],
            "gap_pad4": {"bend": b4["L_dut"] / ref_b["L_deembedded"] - 1.0,
                         "bar": r4["L_dut"] / ref_r["L_deembedded"] - 1.0},
        }
        row["discrepancy_pad4"] = row["excess_fdfd_pad4"] - row["excess_referee"]
        lb, lr = ladders.get("bend", {}).get(dk), ladders.get("bar", {}).get(dk)
        pts = [(lb or {}).get("n_pad_points", 0), (lr or {}).get("n_pad_points", 0)]
        row["wall_ladder_points"] = pts
        if lb and lr:
            row["L_wall_corrected"] = {"bend": lb["L_wall_extrapolated"],
                                       "bar": lr["L_wall_extrapolated"]}
            row["wall_correction"] = {"bend": lb["rel_pad4"], "bar": lr["rel_pad4"]}
            row["gap_wall_corrected"] = {
                "bend": lb["L_wall_extrapolated"] / ref_b["L_deembedded"] - 1.0,
                "bar": lr["L_wall_extrapolated"] / ref_r["L_deembedded"] - 1.0}
            exc = lb["L_wall_extrapolated"] - lr["L_wall_extrapolated"]
            disc = exc - row["excess_referee"]
            unc = lb["uncertainty_abs"] + lr["uncertainty_abs"]
            if min(pts) >= 3:
                row["excess_fdfd"] = exc
                row["discrepancy"] = disc
                row["uncertainty"] = unc
                row["admitted"] = True
            else:
                # a short ladder on at least one leg: recorded, labelled and
                # kept out of every gate and out of the conclusion
                row["excess_fdfd_two_point_EXCLUDED"] = exc
                row["discrepancy_two_point_EXCLUDED"] = disc
                row["uncertainty_two_point_EXCLUDED"] = unc
                row["admitted"] = False
        else:
            row["admitted"] = False
        out["levels"][dk] = row
        tail = (f"  corner discrepancy {row['discrepancy'] * 1e12:+.4f} "
                f"+- {row['uncertainty'] * 1e12:.4f} pH" if row.get("admitted") else
                f"  NO 3-POINT LADDER (pad points {pts}), pad-4 discrepancy "
                f"{row['discrepancy_pad4'] * 1e12:+.4f} pH EXCLUDED")
        print(f"    W/{row['div']:g}: excess_FDFD(pad4)={row['excess_fdfd_pad4'] * 1e12:+.4f} "
              f"excess_referee={row['excess_referee'] * 1e12:+.4f} pH" + tail, flush=True)
    # WHY the excess is negative, in the referee's own terms (strip only, so
    # level-independent): at equal total length the bend's two perpendicular
    # bars have ZERO mutual, while one straight bar of that length carries the
    # collinear mutual of its two halves inside its self term -- and the
    # ground images lose exactly the same coupling, which gives part of it
    # back. The two differences are recorded so the sign is traceable.
    b = sg.greenhouse_terms(bend_segments(sg), ground_height=GROUND_H)
    r_ = sg.greenhouse_terms(bar_segments(sg, L1 + L2), ground_height=GROUND_H)
    out["referee_mechanism"] = {
        "bend": {"total": b.total, "self": b.self_sum, "mutual": b.mutual_sum,
                 "image": b.image_sum},
        "bar": {"total": r_.total, "self": r_.self_sum, "mutual": r_.mutual_sum,
                "image": r_.image_sum},
        "self_plus_mutual_difference": (b.self_sum + b.mutual_sum) - (r_.self_sum + r_.mutual_sum),
        "image_difference": b.image_sum - r_.image_sum,
        "excess_strip_only": b.total - r_.total,
        "note": "bend mutual is EXACTLY zero (perpendicular bars); the bar's single "
                "self term contains the collinear mutual of its two halves",
    }
    adm = [r for r in out["levels"].values() if r.get("admitted")]
    out["admitted_levels"] = [f"W/{r['div']:g}" for r in adm]
    if adm:
        fin = max(adm, key=lambda r: r["div"])
        out["finest_admitted"] = {"div": fin["div"], "discrepancy": fin["discrepancy"],
                                  "uncertainty": fin["uncertainty"],
                                  "excess_fdfd": fin["excess_fdfd"],
                                  "excess_referee": fin["excess_referee"]}
    return out


def arm_sensitivity(sg, div: float) -> dict[str, Any]:
    """How much of the corner excess is the bridge convention worth?

    The primary convention values the short standard's arms from each
    reference plane to the post's NEAR EDGE, ``W/2 + one cell``. The band
    reported here is the excess with the arms taken at ``W/2`` (post edge at
    the footprint's DUT-side edge, the ``h -> 0`` geometric limit) and at
    ``W/2 + 2 base_dx`` (one cell further out on each side), plus the
    STRIP-ONLY excess (no bridge subtracted at all)."""
    dx = WIDTH / div
    out: dict[str, Any] = {"div": div, "base_dx": dx, "per_arm": {}}
    for name, arm in (("W/2", 0.5 * WIDTH), ("W/2+dx", 0.5 * WIDTH + dx),
                      ("W/2+2dx", 0.5 * WIDTH + 2.0 * dx)):
        b = referee_point(sg, "bend", arm, arm)
        r = referee_point(sg, "bar", arm, arm)
        out["per_arm"][name] = {"arm": arm, "bend": b["L_deembedded"], "bar": r["L_deembedded"],
                                "excess": b["L_deembedded"] - r["L_deembedded"]}
    sb = sg.greenhouse_terms(bend_segments(sg), ground_height=GROUND_H).total
    sr = sg.greenhouse_terms(bar_segments(sg, L1 + L2), ground_height=GROUND_H).total
    out["strip_only"] = {"bend": sb, "bar": sr, "excess": sb - sr}
    vals = [v["excess"] for v in out["per_arm"].values()] + [out["strip_only"]["excess"]]
    ref = out["per_arm"]["W/2+dx"]["excess"]
    out["excess_spread_abs"] = float(max(vals) - min(vals))
    out["excess_spread_rel"] = float((max(vals) - min(vals)) / abs(ref))
    return out


# ----------------------------------------------------------------------------
# 5. how many right-angle corners does the D2 spiral have?

def spiral_corner_count(fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Count the right-angle corners of the D2 small spiral EXPLICITLY, from
    the polygon generator's own centreline -- not from ``4 * n_turns``.

    ``gds.rect_spiral`` emits a strip centreline of ``lead``, ``4 n_turns``
    sides and a final inner extension, plus a straight underpass on the
    lower metal joined through the via. Every joint of the strip chain is
    classified by the angle between consecutive unit directions:
    ``right_angle`` (dot = 0), ``collinear`` (dot = +1) or ``reversal``
    (dot = -1). The lead is collinear with side 0 by construction, so it
    contributes NO corner; the strip-to-underpass transition is a vertical
    via with a 180-degree in-plane reversal on another layer, not a right
    angle, and is reported separately."""
    from rfx.fdfd import gds
    fx = dict(SPIRAL_FIXTURE if fixture is None else fixture)
    sp = gds.rect_spiral(fx["n_turns"], fx["r_out"], fx["width"], fx["spacing"], fx["lead"])
    c = np.asarray(sp.centreline, dtype=np.float64)
    d = np.diff(c, axis=0)
    u = d / np.linalg.norm(d, axis=1)[:, None]
    joints = []
    for k in range(len(u) - 1):
        dot = float(np.dot(u[k], u[k + 1]))
        cross = float(u[k, 0] * u[k + 1, 1] - u[k, 1] * u[k + 1, 0])
        kind = ("right_angle" if abs(dot) < 1e-12 else
                "collinear" if dot > 0 else "reversal")
        joints.append({"joint": k, "at": [float(c[k + 1, 0]), float(c[k + 1, 1])],
                       "dir_in": [float(u[k, 0])], "dot": dot,
                       "sense": "ccw" if cross > 0 else "cw", "kind": kind})
    n_right = sum(1 for j in joints if j["kind"] == "right_angle")
    out: dict[str, Any] = {
        "fixture": fx, "n_centreline_points": int(len(c)),
        "n_strip_segments": int(len(u)),
        "segment_lengths_um": [float(v) * 1e6 for v in sp.segment_lengths],
        "strip_centreline_length": float(sp.length),
        "underpass_length": float(sp.underpass_length),
        "joints": joints,
        "n_right_angle": int(n_right),
        "n_collinear": int(sum(1 for j in joints if j["kind"] == "collinear")),
        "n_reversal": int(sum(1 for j in joints if j["kind"] == "reversal")),
        "turns_times_four": int(4 * fx["n_turns"]),
        "underpass_transition": {
            "kind": "via, 180-degree in-plane reversal onto the lower metal",
            "right_angles": 0},
        "accounting": (
            "4 * n_turns = {t} sides give {t} - 1 = {i} interior joints between "
            "sides, all right angles; + 1 right angle where the last side meets the "
            "inner extension; + 0 for the lead/side-0 joint (collinear by "
            "construction) and + 0 for the strip-to-underpass via (a 180-degree "
            "reversal, not a right angle) = {n}".format(
                t=4 * fx["n_turns"], i=4 * fx["n_turns"] - 1, n=n_right)),
    }
    out["n_corners"] = int(n_right)
    return out


def load_invariant_ladder(path: pathlib.Path = INVARIANT_LADDER_PATH) -> dict[str, Any]:
    """The level-invariant ladder's residual of the SAME spiral, as K5 uses
    it, read from ``path`` (``validation/fdfd/invariant_ladder.json``) under
    ``INVARIANT_LADDER_KEYS``: ``referee.total`` (the 333.055 pH referee),
    ``residual.limit_rel_range`` (the continuum limit -- that study's
    Richardson range -- relative to the referee) and
    ``residual.after_corrections_rel_range`` (the same after its measured
    signed corrections). Both ranges are returned relative AND in H, as
    FDFD - referee. The referee must be D2's ``SPIRAL_REFEREE`` to 1e-12
    relative: the comparison assumes one spiral and one referee."""
    path = pathlib.Path(path)
    doc = json.loads(path.read_text())

    def dig(dotted: str) -> Any:
        node = doc
        for k in dotted.split("."):
            node = node[k]
        return node

    ref = float(dig(INVARIANT_LADDER_KEYS["referee"]))
    if abs(ref / SPIRAL_REFEREE - 1.0) > 1e-12:
        raise ValueError(f"{path}: referee {ref!r} is not D2's {SPIRAL_REFEREE!r}")
    lim = sorted(float(v) for v in dig(INVARIANT_LADDER_KEYS["continuum_limit_rel_range"]))
    aft = sorted(float(v) for v in dig(INVARIANT_LADDER_KEYS["after_corrections_rel_range"]))
    try:
        source = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        source = str(path)
    return {
        "what": "the continuum-limit residual of the level-invariant ladder of the same "
                "spiral (study P), FDFD - referee; at a continuum limit the discretisation "
                "deficits are gone, so nothing is subtracted from it",
        "source": source, "keys": dict(INVARIANT_LADDER_KEYS), "referee": ref,
        "continuum_limit_rel_range": lim,
        "continuum_limit_range": [v * ref for v in lim],
        "after_corrections_rel_range": aft,
        "after_corrections_range": [v * ref for v in aft],
    }


def _versus(total: float, unc: float, target: Sequence[float]) -> dict[str, Any]:
    """The corners' SIGNED interval ``[total - unc, total + unc]`` against one
    SIGNED residual interval, both FDFD - referee (H). ``miss`` is the gap
    between the two intervals, 0 when they intersect."""
    c_lo, c_hi = total - unc, total + unc
    t_lo, t_hi = float(min(target)), float(max(target))
    lo_i, hi_i = max(c_lo, t_lo), min(c_hi, t_hi)
    hit = bool(hi_i >= lo_i)
    return {"target_range": [t_lo, t_hi], "intersects": hit,
            "overlap_range": [lo_i, hi_i] if hit else None,
            "miss": 0.0 if hit else float(max(c_lo - t_hi, t_lo - c_hi)),
            "central_value_inside": bool(t_lo <= total <= t_hi),
            "central_value_same_sign": bool(np.sign(total) == np.sign(t_lo + t_hi))}


def attribution(excess: dict[str, Any], corners: dict[str, Any],
                invariant: dict[str, Any] | None = None) -> dict[str, Any]:
    """K5: can ``n_corners x`` the per-corner discrepancy account for the
    spiral's residual against its referee?

    ONE sign convention throughout: FDFD minus referee, in H (pH in the
    prose), so a negative range is the FDFD LOW. Each corner adds its
    discrepancy ``[excess]_FDFD - [excess]_referee`` to the spiral's
    FDFD - referee, so the corner term is ``n_corners x`` the per-corner
    discrepancy with ``n_corners x`` its uncertainty, and K5 passes only if
    that SIGNED interval intersects the SIGNED residual interval.

    PRIMARY (the gate): the continuum-limit residual of the level-invariant
    ladder of the same spiral (``invariant``; read by
    ``load_invariant_ladder`` from ``validation/fdfd/invariant_ladder.json``
    when ``None``): ``residual.limit_rel_range`` x ``referee.total``,
    -4.558 % to -2.477 % of 333.055 pH = [-15.181, -8.249] pH, and, reported
    beside it, ``residual.after_corrections_rel_range`` x ``referee.total``
    = [-15.474, -8.441] pH.

    SECONDARY (superseded, reported, never gated): the original remainder.
    D2's residual after both of its measured corrections
    (``gates.V1.corrections.both`` = ``gates.V1c.with_post_short`` of
    ``spiral_convergence.json``, quoted, never remeasured here) is -7.682 %
    to -5.065 % of the 333.055 pH referee, [-25.586, -16.869] pH; D2b's
    per-unit-length ratio at 3-4 cells across W (0.9746-0.9808, quoted)
    puts the strip's discretisation deficit at [-8.460, -6.395] pH (1.9 to
    2.5 % of L); the interval difference, [-19.191, -8.409] pH, was the
    remainder. At a continuum limit that deficit is gone, and the invariant
    ladder showed D2's plateau to be mainly an unrefined vertical grid, so
    the construction no longer applies."""
    n = int(corners["n_corners"])
    inv = load_invariant_ladder() if invariant is None else invariant
    lo_rel, hi_rel = min(SPIRAL_BOTH_REL), max(SPIRAL_BOTH_REL)
    residual_d2 = [lo_rel * SPIRAL_REFEREE, hi_rel * SPIRAL_REFEREE]      # FDFD - referee (H)
    pul = [(min(BAR_PUL_RATIO_3CELLS) - 1.0) * SPIRAL_REFEREE,
           (max(BAR_PUL_RATIO_3CELLS) - 1.0) * SPIRAL_REFEREE]            # FDFD - exact (H)
    remainder = [residual_d2[0] - pul[1], residual_d2[1] - pul[0]]         # interval difference
    targets = {"vs_continuum_limit": inv["continuum_limit_range"],
               "vs_continuum_limit_after_corrections": inv["after_corrections_range"],
               "vs_old_remainder": remainder}
    out: dict[str, Any] = {
        "sign_convention": "FDFD - referee (H) throughout: a negative range is the FDFD LOW. "
                           "The corner term is n_corners x the per-corner discrepancy "
                           "[excess]_FDFD - [excess]_referee, +- n_corners x its "
                           "uncertainty; K5 passes only if that signed interval intersects "
                           "the signed continuum-limit residual (primary)",
        "n_corners": n,
        "corner_count_accounting": corners["accounting"],
        "spiral_referee_deembedded": SPIRAL_REFEREE,
        "primary": inv,
        "secondary_old_remainder": {
            "what": "SUPERSEDED, reported, never gated: D2's residual after both of its "
                    "corrections minus D2b's per-unit-length discretisation deficit at 3-4 "
                    "cells across W. At a continuum limit that deficit is gone, and the "
                    "invariant ladder showed D2's plateau to be mainly an unrefined "
                    "vertical grid, so this construction no longer applies",
            "spiral_both_corrections_rel": list(SPIRAL_BOTH_REL),
            "spiral_residual_range": residual_d2,
            "bar_per_unit_length_ratio_3cells": list(BAR_PUL_RATIO_3CELLS),
            "bar_explained_range": pul,
            "remainder_range": remainder,
            "remainder_width": remainder[1] - remainder[0],
        },
    }
    for label in ("finest_admitted", "finest_raw_pad4"):
        if label == "finest_admitted":
            row = excess.get("finest_admitted")
            if not row:
                continue
            d, u, div = row["discrepancy"], row["uncertainty"], row["div"]
        else:
            rows = list(excess["levels"].values())
            if not rows:
                continue
            fin = max(rows, key=lambda r: r["div"])
            d, u, div = fin["discrepancy_pad4"], 0.0, fin["div"]
        total, unc = n * d, n * u
        blk: dict[str, Any] = {
            "div": div, "cells_across_width": div,
            "per_corner_discrepancy": d, "per_corner_uncertainty": u,
            "per_corner_uncertainty_dominated": bool(u > abs(d)),
            "corners_total": total, "corners_total_uncertainty": unc,
            "corners_interval": [total - unc, total + unc],
            "fraction_of_spiral_referee": total / SPIRAL_REFEREE,
        }
        for key, target in targets.items():
            blk[key] = _versus(total, unc, target)
        out[label] = blk
    fa = out.get("finest_admitted")
    out["passed"] = bool(fa is not None and fa["vs_continuum_limit"]["intersects"])
    if fa is None:
        out["statement"] = "no level has a three-point wall ladder on both legs"
        return out
    pk = 1e12

    def rng(v: Sequence[float]) -> str:
        return f"[{v[0] * pk:+.3f}, {v[1] * pk:+.3f}] pH"

    def verdict(c: dict[str, Any]) -> str:
        if c["intersects"]:
            return f"the corner interval reaches it (overlap {rng(c['overlap_range'])})"
        return f"the corner interval misses it by {c['miss'] * pk:.3f} pH"

    p = fa["vs_continuum_limit"]
    a = fa["vs_continuum_limit_after_corrections"]
    o = fa["vs_old_remainder"]
    d, u = fa["per_corner_discrepancy"], fa["per_corner_uncertainty"]
    if not out["passed"]:
        tail = "K5 FAILS: the corners cannot account for the spiral's residual"
        if not p["central_value_same_sign"]:
            tail += ("; their central value has the opposite sign (a positive corner term "
                     "raises the FDFD spiral and shrinks its deficit)")
    elif p["central_value_inside"]:
        tail = "K5 passes: the corners account for it"
    else:
        tail = "K5 passes, but only through the uncertainty"
    out["statement"] = (
        f"FDFD - referee throughout. {n} right-angle corners x "
        f"({d * pk:+.4f} +- {u * pk:.4f}) pH per corner at {fa['cells_across_width']:g} "
        f"cells across W = {fa['corners_total'] * pk:+.3f} +- "
        f"{fa['corners_total_uncertainty'] * pk:.3f} pH = {rng(fa['corners_interval'])}. "
        f"PRIMARY, the spiral's continuum-limit residual ({inv['source']}: "
        f"{inv['keys']['continuum_limit_rel_range']} x {inv['keys']['referee']}) = "
        f"{rng(p['target_range'])}: {verdict(p)}. Reported: {rng(a['target_range'])} after "
        f"that study's measured corrections ({inv['keys']['after_corrections_rel_range']}): "
        f"{verdict(a)}; SECONDARY (superseded), D2's residual minus D2b's per-unit-length "
        f"deficit = {rng(o['target_range'])}: {verdict(o)}. {tail}")
    return out


# ----------------------------------------------------------------------------
# 6. the gradient check

def gradient_check(div: float = 1.0) -> dict[str, Any]:
    """K2: ``jax.value_and_grad`` of the de-embedded ``L_diff`` of the BEND
    in ``theta = (l1, l2, W)`` against a 4th-order central finite difference
    with 1 % steps (which move L by ~1e-2 relative, far above the ~1e-9 LU
    noise floor), at ``base_dx = W`` -- the two derivatives the task asks
    for, ``dL/dl1`` and ``dL/dW``, are entries 0 and 2."""
    import jax
    import jax.numpy as jnp

    model = build("bend", div, PAD_CELLS)
    theta0 = (L1, L2, WIDTH)

    def scalar(t) -> Any:
        return jnp.real(l_dut(model, t))

    t0 = time.time()
    val, grad = jax.value_and_grad(scalar)(jnp.asarray(theta0, dtype=jnp.float64))
    jax.block_until_ready(grad)
    rec: dict[str, Any] = {"l1": L1, "l2": L2, "width": WIDTH, "div": div,
                           "n_unknowns": int(model.n_unknowns),
                           "params": ["l1", "l2", "width"],
                           "L_dut": float(val), "grad": [float(v) for v in grad],
                           "value_and_grad_seconds": time.time() - t0,
                           "fd_step_rel": FD_STEP_REL, "fd_params": ["l1", "width"]}
    t0 = time.time()
    fd: list[float] = []
    for k in (0, 2):
        def f(v: float, k: int = k) -> float:
            t = list(theta0)
            t[k] = v
            return float(scalar(jnp.asarray(t, dtype=jnp.float64)))
        fd.append(float(fd4(f, theta0[k], FD_STEP_REL * theta0[k])))
    rec["fd4"] = fd
    rec["fd_seconds"] = time.time() - t0
    rec["grad_checked"] = [rec["grad"][0], rec["grad"][2]]
    rec["grad_vs_fd_rel"] = [abs(g - f_) / abs(f_) for g, f_ in zip(rec["grad_checked"], fd)]
    print(f"    dL/dl1={rec['grad'][0]:.6e} (FD4 {fd[0]:.6e}), "
          f"dL/dW={rec['grad'][2]:.6e} (FD4 {fd[1]:.6e}); worst rel "
          f"{max(rec['grad_vs_fd_rel']):.2e}  ({rec['value_and_grad_seconds']:.0f}+"
          f"{rec['fd_seconds']:.0f} s)", flush=True)
    return rec


# ----------------------------------------------------------------------------
# 7. gates

def evaluate_gates(study: dict[str, Any]) -> dict[str, Any]:
    g: dict[str, Any] = {}
    exc = study["excess"]
    rows = [exc["levels"][k] for k in sorted(exc["levels"], key=float)]
    adm = [r for r in rows if r.get("admitted")]

    # --- K1: three-point wall ladders only (structural) --------------------
    ladder_pts = {f"W/{r['div']:g}": r["wall_ladder_points"] for r in rows}
    gated_keys = sorted({k for k in exc["admitted_levels"]})
    g["K1"] = {
        "what": "every level that enters a gate has a full pad_cells 4/6/8 ladder on BOTH "
                "legs; levels with a shorter ladder are recorded, labelled "
                "*_two_point_EXCLUDED and kept out of every gate",
        "wall_ladder_points_per_level": ladder_pts,
        "admitted_levels": gated_keys,
        "excluded_levels": [f"W/{r['div']:g}" for r in rows if not r.get("admitted")],
        "n_max": study["fixture"]["n_max"],
        "why_excluded": study.get("skipped", {}),
        "passed": bool(adm) and all(min(r["wall_ladder_points"]) >= 3 for r in adm),
    }

    # --- K2: AD vs FD4 ------------------------------------------------------
    gr = study.get("gradient")
    if gr:
        worst = max(gr["grad_vs_fd_rel"])
        g["K2"] = {"params": gr["fd_params"], "grad": gr["grad_checked"], "fd4": gr["fd4"],
                   "grad_vs_fd_rel": gr["grad_vs_fd_rel"], "fd_step_rel": gr["fd_step_rel"],
                   "div": gr["div"], "n_unknowns": gr["n_unknowns"],
                   "worst": worst, "tolerance": 1e-4, "passed": bool(worst <= 1e-4)}

    # --- K3: the sign of the corner excess agrees at every ADMITTED level ---
    def sign_row(f_: float, r: dict[str, Any], basis: str) -> dict[str, Any]:
        return {"excess_fdfd": f_, "excess_referee": r["excess_referee"], "basis": basis,
                "sign_fdfd": int(np.sign(f_)), "sign_referee": int(np.sign(r["excess_referee"])),
                "agrees": bool(np.sign(f_) == np.sign(r["excess_referee"])),
                "ratio": f_ / r["excess_referee"]}

    # the gate: the admitted levels, the same set K4 and K5 use
    signs = {f"W/{r['div']:g}": sign_row(r["excess_fdfd"], r, "wall-corrected, three-point "
                                         "ladders on both legs") for r in adm}
    # reported, never gated: a level without a three-point ladder on both legs
    excluded = {}
    for r in rows:
        if r.get("admitted"):
            continue
        if "excess_fdfd_two_point_EXCLUDED" in r:
            f_, basis = r["excess_fdfd_two_point_EXCLUDED"], "two-point wall ladder (EXCLUDED)"
        else:
            f_, basis = (r["excess_fdfd_pad4"],
                         "raw pad_cells = 4: at least one leg has no wall ladder (EXCLUDED)")
        excluded[f"W/{r['div']:g}"] = sign_row(f_, r, basis)
    g["K3"] = {"what": "at equal total centreline length a bend has LESS mutual coupling "
                       "between its halves than a straight bar, so the excess must be "
                       "NEGATIVE in both tools at every ADMITTED level (the set K4 and K5 "
                       "use); a level without a three-point ladder on both legs is reported "
                       "under excluded_informational and never gated",
               "admitted_levels": list(signs),
               "per_level": signs,
               "excluded_informational": excluded,
               "passed": bool(signs) and all(v["agrees"] for v in signs.values())}

    # --- K4: the per-corner discrepancy decreases with refinement ----------
    seq = [(r["div"], r["discrepancy"], r["uncertainty"]) for r in adm]
    seq.sort()
    mono = all(abs(b[1]) <= abs(a[1]) + 1e-30 for a, b in zip(seq, seq[1:]))
    raw = [(r["div"], r["discrepancy_pad4"]) for r in rows]
    raw.sort()
    g["K4"] = {"what": "|discrepancy of the corner excess| against refinement, on the "
                       "admitted (3-point-ladder) levels",
               "admitted": [{"div": d, "discrepancy": v, "uncertainty": u} for d, v, u in seq],
               "raw_pad4": [{"div": d, "discrepancy": v} for d, v in raw],
               "raw_pad4_decreases": bool(
                   all(abs(b[1]) <= abs(a[1]) + 1e-30 for a, b in zip(raw, raw[1:]))),
               "n_admitted": len(seq),
               "passed": bool(len(seq) >= 2 and mono)}

    # --- K5: the attribution ------------------------------------------------
    att = study["attribution"]
    g["K5"] = {"what": "the SIGNED corner interval, n_corners x (per-corner discrepancy +- "
                       "its uncertainty), FDFD - referee, must intersect the SIGNED "
                       "continuum-limit residual of the invariant ladder (primary); the same "
                       "limit after its corrections and the superseded D2-minus-D2b "
                       "remainder are reported beside it",
               **{k: att[k] for k in ("sign_convention", "n_corners", "corner_count_accounting",
                                      "primary", "secondary_old_remainder", "statement",
                                      "passed")}}
    if "finest_admitted" in att:
        g["K5"]["finest_admitted"] = att["finest_admitted"]
    if "finest_raw_pad4" in att:
        g["K5"]["finest_raw_pad4_EXCLUDED"] = att["finest_raw_pad4"]

    # --- the corner-convention area gate (reported, must pass) -------------
    g["K_area"] = {"what": "the area-exact corner convention: each corner square covered "
                           "exactly once", "checks": study["area_gate"],
                   "passed": study["area_gate"]["passed"]}
    return g


# ----------------------------------------------------------------------------
# 8. figure

def figure(study: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_BEND, C_BAR, C_D = "#1f77b4", "#d62728", "#7f3fbf"
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    rows = [study["excess"]["levels"][k] for k in sorted(study["excess"]["levels"], key=float)]

    # (a) the two de-embedded L against the grid, with their referees
    for kind, c in (("bend", C_BEND), ("bar", C_BAR)):
        h = [r["base_dx"] * 1e6 for r in rows]
        ax1.plot(h, [r["L_pad4"][kind] * 1e12 for r in rows], "o-", color=c,
                 label=f"{kind} FDFD, pad_cells = 4")
        wc = [(r["base_dx"] * 1e6, r["L_wall_corrected"][kind] * 1e12) for r in rows
              if r.get("admitted")]
        if wc:
            ax1.plot([a for a, _ in wc], [b for _, b in wc], "s--", color=c, alpha=0.8,
                     label=f"{kind} wall-extrapolated (3-point)")
        ax1.plot(h, [r["L_referee"][kind] * 1e12 for r in rows], ":", color=c, linewidth=1.4,
                 label=f"{kind} referee (bridge-corrected)")
    ax1.set_xlabel("base_dx [um]")
    ax1.set_ylabel("de-embedded L [pH]")
    ax1.set_xlim(0.0, None)
    ax1.set_title("(a) the L-bend and the straight bar of the SAME total length\n"
                  f"l1 = l2 = {L1 * 1e6:g} um, bar = {(L1 + L2) * 1e6:g} um, W = "
                  f"{WIDTH * 1e6:g} um, t = {T_M2 * 1e6:g} um")
    ax1.legend(fontsize=6)
    ax1.grid(alpha=0.3)

    # (b) the corner excess, both tools
    h = [r["base_dx"] * 1e6 for r in rows]
    ax2.plot(h, [r["excess_fdfd_pad4"] * 1e12 for r in rows], "o-", color=C_BEND,
             label="FDFD, pad_cells = 4")
    wc = [(r["base_dx"] * 1e6, r["excess_fdfd"] * 1e12, r["uncertainty"] * 1e12)
          for r in rows if r.get("admitted")]
    if wc:
        ax2.errorbar([a for a, _, _ in wc], [b for _, b, _ in wc],
                     yerr=[e for _, _, e in wc], fmt="s--", color=C_BEND, capsize=3,
                     label="FDFD wall-corrected (3-point ladder)")
    ex = [(r["base_dx"] * 1e6, r["excess_fdfd_two_point_EXCLUDED"] * 1e12) for r in rows
          if "excess_fdfd_two_point_EXCLUDED" in r]
    if ex:
        ax2.plot([a for a, _ in ex], [b for _, b in ex], "x", color=C_BEND, markersize=9,
                 markeredgewidth=2, label="short ladder (EXCLUDED)")
    ax2.plot(h, [r["excess_referee"] * 1e12 for r in rows], "-", color=C_BAR, linewidth=1.6,
             label="Greenhouse referee (area-exact, bridge-corrected)")
    ax2.axhline(0.0, color="k", linewidth=1)
    ax2.set_xlabel("base_dx [um]")
    ax2.set_ylabel("L_bend - L_bar [pH]")
    ax2.set_xlim(0.0, None)
    ax2.set_title("(b) K3: the corner excess at equal length is NEGATIVE in both tools\n"
                  "(perpendicular halves do not couple; collinear ones do)")
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.3)

    # (c) the per-corner discrepancy
    ax3.plot(h, [r["discrepancy_pad4"] * 1e12 for r in rows], "o-", color=C_D,
             label="pad_cells = 4 (reported, not gated)")
    wc = [(r["base_dx"] * 1e6, r["discrepancy"] * 1e12, r["uncertainty"] * 1e12)
          for r in rows if r.get("admitted")]
    if wc:
        ax3.errorbar([a for a, _, _ in wc], [b for _, b, _ in wc],
                     yerr=[e for _, _, e in wc], fmt="s--", color=C_D, capsize=3,
                     label="wall-corrected (3-point ladder): K4")
    ex = [(r["base_dx"] * 1e6, r["discrepancy_two_point_EXCLUDED"] * 1e12) for r in rows
          if "discrepancy_two_point_EXCLUDED" in r]
    if ex:
        ax3.plot([a for a, _ in ex], [b for _, b in ex], "x", color=C_D, markersize=9,
                 markeredgewidth=2, label="short ladder (EXCLUDED)")
    ax3.axhline(0.0, color="k", linewidth=1)
    ax3.set_xlabel("base_dx [um]")
    ax3.set_ylabel("[L_bend - L_bar]_FDFD - referee [pH]")
    ax3.set_xlim(0.0, None)
    ax3.set_title("(c) K4: the discrepancy of ONE right-angle corner")
    ax3.legend(fontsize=6)
    ax3.grid(alpha=0.3)

    # (d) the attribution, in ONE sign convention: FDFD - referee
    att = study["attribution"]
    targets = ((att["primary"]["continuum_limit_range"], "#555555",
                "invariant ladder, continuum limit (PRIMARY)"),
               (att["primary"]["after_corrections_range"], "#aaaaaa",
                "invariant ladder, after its corrections"),
               (att["secondary_old_remainder"]["remainder_range"], "#ff7f0e",
                "D2 minus D2b per-unit-length (superseded)"))
    for y, (span, c, name) in enumerate(targets):
        a0, a1 = span[0] * 1e12, span[1] * 1e12
        ax4.barh(y, a1 - a0, left=a0, height=0.5, color=c,
                 label=f"{name}: [{a0:.2f}, {a1:.2f}] pH")
    fa = att.get("finest_admitted")
    if fa:
        t, u = fa["corners_total"] * 1e12, fa["corners_total_uncertainty"] * 1e12
        ax4.errorbar([t], [3], xerr=[[u], [u]], fmt="D", color=C_D, capsize=4, markersize=8,
                     label=f"{att['n_corners']} corners x this study "
                           f"({t:+.2f} +- {u:.2f} pH, {fa['cells_across_width']:g} cells/W): "
                           f"misses the primary by "
                           f"{fa['vs_continuum_limit']['miss'] * 1e12:.2f} pH")
    fr = att.get("finest_raw_pad4")
    if fr:
        ax4.plot([fr["corners_total"] * 1e12], [4], "x", color=C_D, markersize=10,
                 markeredgewidth=2,
                 label=f"{att['n_corners']} corners x pad-4 at "
                       f"{fr['cells_across_width']:g} cells/W (EXCLUDED)")
    ax4.axvline(0.0, color="k", linewidth=1)
    ax4.use_sticky_edges = False              # keep a margin left of the widest bar
    ax4.autoscale_view()
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_yticklabels(["residual (limit)", "residual (corrected)", "old remainder",
                         "corners (admitted)", "corners (pad-4, excluded)"], fontsize=7)
    ax4.set_ylim(-0.7, 7.2)
    ax4.set_xlabel("FDFD - referee [pH]  (negative: the FDFD is LOW)")
    ax4.set_title(f"(d) K5 {'passes' if att['passed'] else 'FAILS'}: "
                  "the corners against the spiral's residual\n"
                  f"(referee {SPIRAL_REFEREE * 1e12:.1f} pH, "
                  f"{att['n_corners']} right-angle corners)")
    ax4.legend(fontsize=6, loc="upper left")
    ax4.grid(alpha=0.3, axis="x")

    fig.suptitle("D2c: one right-angle corner on the spiral fixture vs the Greenhouse "
                 f"referee ({FREQ / 1e6:.0f} MHz, uniform-current metal)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 9. main

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS),
                    help="base_dx = W / level")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--pads", type=int, nargs="+", default=list(WALL_PADS))
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--no-gradient", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the JSON on disk instead of reusing its measured points")
    ap.add_argument("--from-json", action="store_true",
                    help="REASSEMBLE only: reuse every measured point, recorded skip and the "
                         "gradient from the JSON on disk and re-derive everything else; "
                         "builds and solves nothing, and stops if a requested point is "
                         "neither measured nor a recorded skip")
    ap.add_argument("--no-figure", action="store_true", help="do not write the PNG")
    args = ap.parse_args(argv)
    if args.from_json and (args.fresh or not JSON_PATH.exists()):
        ap.error(f"--from-json needs {JSON_PATH.name} on disk and excludes --fresh")

    import jax
    jax.config.update("jax_enable_x64", True)
    sg = load_referee()

    t_start = time.time()
    previous: dict[str, Any] = {} if args.fresh or not JSON_PATH.exists() else \
        json.loads(JSON_PATH.read_text())

    study: dict[str, Any] = {
        "fixture": {
            "freq": FREQ, "width": WIDTH, "l1": L1, "l2": L2, "bar_length": L1 + L2,
            "levels": list(args.levels), "pads": list(args.pads), "margin": MARGIN,
            "base_dz": BASE_DZ, "metal_cells": METAL_CELLS, "pad_cells": PAD_CELLS,
            "pad_ratio": PAD_RATIO, "sigma_volumetric": SIGMA_VOL, "ground_height": GROUND_H,
            "port_gap_cells": 1, "eps_dielectrics": 1.0, "n_max": args.n_max,
            "fd_step_rel": FD_STEP_REL,
            "stack": {"t_si": T_SI, "t_ox_low": T_OX_LOW, "t_m1": T_M1, "t_via": T_VIA,
                      "t_m2": T_M2, "t_ox_high": T_OX_HIGH, "t_air": T_AIR},
        },
        "spiral_reference": {
            "note": "D2 (validation/fdfd/spiral_convergence.json) and D2b "
                    "(straight_bar_convergence.json) numbers are QUOTED for K5's "
                    "secondary, superseded comparison; the invariant ladder's residual is "
                    "READ from validation/fdfd/invariant_ladder.json at run time for K5's "
                    "primary one (attribution.primary). None of them is measured here. "
                    "The corner COUNT is measured here, from gds.rect_spiral's own "
                    "centreline.",
            "referee_deembedded": SPIRAL_REFEREE,
            "both_corrections_rel": list(SPIRAL_BOTH_REL),
            "bar_per_unit_length_ratio_3cells": list(BAR_PUL_RATIO_3CELLS),
            "invariant_ladder_json": "validation/fdfd/invariant_ladder.json",
            "invariant_ladder_keys": dict(INVARIANT_LADDER_KEYS),
        },
        "levels": {k: dict(previous.get("levels", {}).get(k, {})) for k in KINDS},
        "skipped": dict(previous.get("skipped", {})),
    }

    def dump() -> None:
        study["seconds"] = time.time() - t_start
        JSON_PATH.write_text(json.dumps(study, indent=1, sort_keys=False))

    # --- the corner count and the area gate (host numpy, instant) ----------
    study["corner_count"] = spiral_corner_count()
    print(f"the D2 spiral's right-angle corners: {study['corner_count']['n_corners']} "
          f"({study['corner_count']['accounting']})", flush=True)
    study["area_gate"] = area_gate(sg)
    print(f"area gate (area-exact corner convention): union/expected-1="
          f"{study['area_gate']['area_exact']['union_over_expected_minus_1']:+.2e}, "
          f"union/sum-1={study['area_gate']['area_exact']['union_over_sum_minus_1']:+.2e}, "
          f"passed={study['area_gate']['passed']}", flush=True)
    dump()

    # --- the solves ---------------------------------------------------------
    if args.from_json:
        missing = [f"{kind}_W{div:g}_pad{pad}" for div in args.levels for kind in args.kinds
                   for pad in args.pads
                   if f"{pad}" not in study["levels"].get(kind, {}).get(f"{div:g}", {})
                   and f"{kind}_W{div:g}_pad{pad}" not in study["skipped"]]
        if missing:
            raise SystemExit(f"--from-json: neither measured nor a recorded skip: {missing}")
        print("FDFD points: every requested point is in the JSON or a recorded skip "
              "(--from-json: nothing built, nothing solved)", flush=True)
    else:
        print("FDFD points (volumetric uniform-current metal, graded wall padding):",
              flush=True)
    for div in ([] if args.from_json else args.levels):
        dk = f"{div:g}"
        for kind in args.kinds:
            study["levels"].setdefault(kind, {}).setdefault(dk, {})
            for pad in args.pads:
                pk = f"{pad}"
                if pk in study["levels"][kind][dk]:
                    print(f"    {kind} W/{div:g} pad={pad}: reused from the JSON "
                          f"(L={study['levels'][kind][dk][pk]['L_dut'] * 1e12:.4f} pH)",
                          flush=True)
                    continue
                rec = run_point(kind, div, pad, n_max=args.n_max)
                if rec is None:
                    m = build(kind, div, pad)
                    study["skipped"][f"{kind}_W{div:g}_pad{pad}"] = {
                        "n_unknowns": int(m.n_unknowns), "shape": list(m.shape),
                        "n_max": args.n_max,
                        "why": "above the single-solve cap (see N_MAX in the module doc)"}
                    dump()
                    continue
                study["levels"][kind][dk][pk] = rec
                dump()

    # --- the wall ladders ---------------------------------------------------
    study["wall_ladder"] = {}
    for kind in KINDS:
        study["wall_ladder"][kind] = {}
        for dk, pads in study["levels"].get(kind, {}).items():
            rows = [pads[p] for p in sorted(pads, key=int)]
            if len(rows) >= 2:
                study["wall_ladder"][kind][dk] = ladder_entry(rows)
                e = study["wall_ladder"][kind][dk]
                print(f"    {kind} W/{float(dk):g}: {e['n_pad_points']}-point ladder -> "
                      f"{e['L_wall_extrapolated'] * 1e12:.4f} pH "
                      f"({100 * e['rel_pad4']:+.3f} % on pad=4, 2-point basis spread "
                      f"{100 * e['extrapolation_uncertainty']:+.3f} %)", flush=True)
    dump()

    # --- the corner excess --------------------------------------------------
    print("corner excess [L_bend - L_bar] at equal total centreline length:", flush=True)
    study["excess"] = excess_block(sg, study["levels"], study["wall_ladder"])
    study["arm_sensitivity"] = {f"{d:g}": arm_sensitivity(sg, d) for d in args.levels}
    dump()

    # --- the attribution ----------------------------------------------------
    study["attribution"] = attribution(study["excess"], study["corner_count"])
    print("attribution: " + study["attribution"]["statement"], flush=True)
    dump()

    # --- the gradient -------------------------------------------------------
    if not (args.no_gradient or args.from_json):
        print("gradient check K2 (bend, W/1):", flush=True)
        if "gradient" in previous and not args.fresh:
            study["gradient"] = previous["gradient"]
            print("  reused from the JSON", flush=True)
        else:
            study["gradient"] = gradient_check(1.0)
        dump()
    elif "gradient" in previous and not args.fresh:
        study["gradient"] = previous["gradient"]

    # the measuring cost of everything on disk, however many runs it took
    solve_s = sum(r["seconds"] + r["build_seconds"]
                  for kind in KINDS for pads in study["levels"].get(kind, {}).values()
                  for r in pads.values())
    grad_s = (study["gradient"]["value_and_grad_seconds"] + study["gradient"]["fd_seconds"]
              if "gradient" in study else 0.0)
    study["seconds_measuring"] = solve_s + grad_s
    study["gates"] = evaluate_gates(study)
    dump()
    if not args.no_figure:
        figure(study, PNG_PATH)
    print(f"\nwrote {JSON_PATH}" + ("" if args.no_figure else f" and {PNG_PATH}") +
          f" ({study['seconds']:.1f} s)", flush=True)
    for name, gate in study["gates"].items():
        print(f"  {name}: passed={gate.get('passed')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
